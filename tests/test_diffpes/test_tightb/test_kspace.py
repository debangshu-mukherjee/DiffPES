r"""Validate differentiable k-space conversion and fixed-shape mesh builders.

The tests cover the graphene reciprocal-lattice truth, Cartesian round trips,
first-zone geometry, path distance, ARPES raster rotation, gradients, and JIT
trace counts.
"""

import json
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.typing import Any, Callable, Tuple
from hypothesis import given, settings, strategies
from jaxtyping import Array, Bool, Complex128, Float64, Int64, jaxtyped

from diffpes.simul import (
    detector_angles_to_kpar,
    kz_from_inner_potential_at_fermi,
)
from diffpes.tightb import (
    build_arpes_kmesh,
    build_bz_mesh,
    build_kmesh_hv_at_fermi,
    build_kpath,
    first_bz_mask,
    kpath_arc_length,
    kpoints_cart_to_frac,
    kpoints_frac_to_cart,
)
from diffpes.types import CrystalGeometry, KGrid, KPath, make_crystal_geometry
from tests._assertions import assert_rejects
from tests._gradients import assert_grad_matches_fd, gradient_gate


@jaxtyped(typechecker=beartype)
def _make_geometry(lattice: Float64[Array, "3 3"]) -> CrystalGeometry:
    """PRIVATE: Create a one-site geometry for a specified lattice.

    Parameters
    ----------
    lattice : Float64[Array, "3 3"]
        Real-space lattice vectors as rows in Angstrom.

    Returns
    -------
    geometry : CrystalGeometry
        Right-handed one-site crystal geometry.

    Notes
    -----
    The helper places one synthetic site at the origin. The site does not
    affect any k-space operation in this test module.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        jnp.zeros((1, 3)),
        ("X",),
    )
    return geometry


class TestKpointsFracToCart:
    """Validate :func:`~diffpes.tightb.kpoints_frac_to_cart`.

    The cases use the analytic graphene reciprocal basis and its lattice-scale
    derivative.

    :see: :func:`~diffpes.tightb.kpoints_frac_to_cart`
    """

    def test_matches_the_graphene_reciprocal_closed_form(self) -> None:
        r"""Match the magnitude and direction of the graphene K point.

        The fractional point ``(1/3, 1/3, 0)`` must have magnitude
        :math:`4\pi/(3a)` and azimuth 60 degrees within ``1e-12``.

        Notes
        -----
        The test constructs the hexagonal real-space basis and compares the
        reciprocal rows with their independent closed forms.
        """
        lattice_constant: float = 2.46
        layer_height: float = 15.0
        lattice: Float64[Array, "3 3"] = jnp.array(
            [
                [lattice_constant, 0.0, 0.0],
                [
                    -0.5 * lattice_constant,
                    0.5 * jnp.sqrt(3.0) * lattice_constant,
                    0.0,
                ],
                [0.0, 0.0, layer_height],
            ]
        )
        geometry: CrystalGeometry = _make_geometry(lattice)
        expected_reciprocal: Float64[Array, "3 3"] = jnp.array(
            [
                [
                    2.0 * jnp.pi / lattice_constant,
                    2.0 * jnp.pi / (jnp.sqrt(3.0) * lattice_constant),
                    0.0,
                ],
                [
                    0.0,
                    4.0 * jnp.pi / (jnp.sqrt(3.0) * lattice_constant),
                    0.0,
                ],
                [0.0, 0.0, 2.0 * jnp.pi / layer_height],
            ]
        )
        cartesian: Float64[Array, "1 3"] = kpoints_frac_to_cart(
            jnp.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]]), geometry
        )
        magnitude: Float64[Array, ""] = jnp.linalg.norm(cartesian[0])
        azimuth: Float64[Array, ""] = jnp.arctan2(
            cartesian[0, 1], cartesian[0, 0]
        )

        chex.assert_trees_all_close(
            geometry.reciprocal,
            expected_reciprocal,
            atol=1e-12,
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            magnitude,
            4.0 * jnp.pi / (3.0 * lattice_constant),
            atol=1e-12,
            rtol=1e-12,
        )
        chex.assert_trees_all_close(
            azimuth, jnp.pi / 3.0, atol=1e-12, rtol=1e-12
        )

    @given(
        strategies.floats(1.0, 6.0),
        strategies.floats(1.0, 6.0),
        strategies.floats(1.0, 6.0),
        strategies.floats(-0.4, 0.4),
        strategies.floats(-0.4, 0.4),
        strategies.floats(-0.4, 0.4),
    )
    @settings(max_examples=24, deadline=None)
    def test_random_reciprocal_rows_satisfy_the_direct_identity(
        self,
        a: float,
        b: float,
        c: float,
        xy: float,
        xz: float,
        yz: float,
    ) -> None:
        r"""Verify :math:`A B^T=2\pi I` for random lattice rows.

        The direct real--reciprocal identity must hold within ``1e-10`` for
        every generated right-handed, non-degenerate lattice.

        Notes
        -----
        Hypothesis generates positive triangular cells with bounded shear.
        The test multiplies the real-space rows by the independently stored
        reciprocal rows rather than relying on a coordinate round trip.
        """
        lattice: Float64[Array, "3 3"] = jnp.array(
            [[a, xy, xz], [0.0, b, yz], [0.0, 0.0, c]]
        )
        geometry: CrystalGeometry = _make_geometry(lattice)
        direct_identity: Float64[Array, "3 3"] = (
            lattice @ geometry.reciprocal.T
        )
        expected: Float64[Array, "3 3"] = 2.0 * jnp.pi * jnp.eye(3)
        chex.assert_trees_all_close(
            direct_identity,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_matches_the_analytic_graphene_scale_derivative(self) -> None:
        r"""Match :math:`d|K|/da=-4\pi/(3a^2)` for graphene.

        The automatic derivative must match the independent closed form
        within a relative tolerance of ``1e-10``.

        Notes
        -----
        The test differentiates lattice construction and reciprocal conversion
        together at ``a=2.46`` Angstrom.
        """

        def k_magnitude(
            lattice_constant: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Compute the graphene K-point magnitude for one lattice scale."""
            lattice: Float64[Array, "3 3"] = jnp.array(
                [
                    [lattice_constant, 0.0, 0.0],
                    [
                        -0.5 * lattice_constant,
                        0.5 * jnp.sqrt(3.0) * lattice_constant,
                        0.0,
                    ],
                    [0.0, 0.0, 15.0],
                ]
            )
            geometry: CrystalGeometry = _make_geometry(lattice)
            cartesian: Float64[Array, "1 3"] = kpoints_frac_to_cart(
                jnp.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]]), geometry
            )
            magnitude: Float64[Array, ""] = jnp.linalg.norm(cartesian[0])
            return magnitude

        lattice_constant: Float64[Array, ""] = jnp.asarray(2.46)
        actual: Float64[Array, ""] = jax.grad(k_magnitude)(lattice_constant)
        expected: Float64[Array, ""] = (
            -4.0 * jnp.pi / (3.0 * lattice_constant * lattice_constant)
        )
        chex.assert_trees_all_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_preserves_all_generic_lattice_sensitivities(self) -> None:
        """Preserve finite nonzero gradients through all nine lattice entries.

        Automatic derivatives must match central differences for a generic
        non-orthogonal cell.

        Notes
        -----
        A weighted Cartesian reduction makes every reciprocal-lattice entry
        relevant. The shared gate checks reverse mode and every coordinate.
        """
        lattice: Float64[Array, "3 3"] = jnp.array(
            [[2.4, 0.2, 0.1], [0.1, 2.8, 0.3], [0.2, 0.1, 3.2]]
        )
        fractional: Float64[Array, "2 3"] = jnp.array(
            [[0.2, -0.3, 0.4], [0.7, 0.1, -0.2]]
        )
        weights: Float64[Array, "2 3"] = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]]
        )

        def loss(candidate: Float64[Array, "3 3"]) -> Float64[Array, ""]:
            """Reduce converted points with generic Cartesian weights."""
            geometry: CrystalGeometry = _make_geometry(candidate)
            cartesian: Float64[Array, "2 3"] = kpoints_frac_to_cart(
                fractional, geometry
            )
            result: Float64[Array, ""] = jnp.sum(cartesian * weights)
            return result

        gradient_gate(
            loss,
            lattice,
            modes=("rev",),
            elementwise=True,
        )


class TestKpointsCartToFrac:
    """Validate :func:`~diffpes.tightb.kpoints_cart_to_frac`.

    The case checks the exact inverse contract across random non-orthogonal
    lattices.

    :see: :func:`~diffpes.tightb.kpoints_cart_to_frac`
    """

    @given(
        strategies.floats(1.0, 6.0),
        strategies.floats(1.0, 6.0),
        strategies.floats(1.0, 6.0),
        strategies.floats(-0.4, 0.4),
        strategies.floats(-0.4, 0.4),
        strategies.floats(-0.4, 0.4),
    )
    @settings(max_examples=16, deadline=None)
    def test_round_trips_random_nonorthogonal_lattices(
        self,
        a: float,
        b: float,
        c: float,
        xy: float,
        xz: float,
        yz: float,
    ) -> None:
        """Recover fractional points after Cartesian conversion.

        The round trip must agree within ``1e-12`` for each generated
        right-handed lattice.

        Notes
        -----
        Hypothesis supplies positive diagonal values and bounded shear values.
        The resulting upper-triangular lattice cannot be singular.
        """
        lattice: Float64[Array, "3 3"] = jnp.array(
            [[a, xy, xz], [0.0, b, yz], [0.0, 0.0, c]]
        )
        geometry: CrystalGeometry = _make_geometry(lattice)
        fractional: Float64[Array, "3 3"] = jnp.array(
            [[0.2, -0.3, 0.4], [0.7, 0.1, -0.2], [-0.5, 0.6, 0.9]]
        )
        cartesian: Float64[Array, "3 3"] = kpoints_frac_to_cart(
            fractional, geometry
        )
        restored: Float64[Array, "3 3"] = kpoints_cart_to_frac(
            cartesian, geometry
        )
        chex.assert_trees_all_close(
            restored, fractional, rtol=1e-12, atol=1e-12
        )


class TestBuildKpath:
    """Validate :func:`~diffpes.tightb.build_kpath`.

    The cases cover endpoint duplication, label indices, and absolute anchor
    conversion.

    :see: :func:`~diffpes.tightb.build_kpath`
    """

    def test_repeats_junctions_and_places_each_label(self) -> None:
        """Repeat shared anchors and place labels at Chinook-compatible indices.

        Two three-point segments must produce six points with the middle
        anchor at indices two and three.

        Notes
        -----
        The test compares the generated fractional path with a direct linear
        interpolation table.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        anchors: Float64[Array, "3 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]
        )
        kpath: KPath = build_kpath(anchors, geometry, 3, ("G", "X", "M"))
        expected: Float64[Array, "6 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.25, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )

        chex.assert_trees_all_close(kpath.kpoints, expected)
        chex.assert_equal(kpath.label_indices, (0, 3, 5))
        chex.assert_equal(kpath.n_per_segment, 3)

    def test_converts_absolute_anchors_once(self) -> None:
        """Convert Cartesian anchors to the fractional path convention.

        A cubic cell with unit real-space length maps ``pi`` to one-half on
        the matching fractional axis.

        Notes
        -----
        The test supplies two Cartesian anchors and compares both generated
        endpoints with the analytic fractional coordinates.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        anchors: Float64[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [jnp.pi, 0.0, 0.0]]
        )
        kpath: KPath = build_kpath(
            anchors,
            geometry,
            2,
            ("G", "X"),
            anchor_units="absolute",
        )
        expected: Float64[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        )
        chex.assert_trees_all_close(kpath.kpoints, expected, atol=1e-15)

    def test_rejects_each_invalid_path_setting(self) -> None:
        """Reject invalid anchor counts, segment sizes, labels, and units.

        Static validation must reject every setting before interpolation.

        Notes
        -----
        The shared helper checks the four static branches and one traced
        finite-value branch in eager and compiled execution.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        anchors: Float64[Array, "2 3"] = jnp.zeros((2, 3))
        assert_rejects(
            build_kpath,
            anchors[:1],
            geometry,
            2,
            ("G",),
            match="at least two anchors",
        )
        assert_rejects(
            build_kpath,
            anchors,
            geometry,
            1,
            ("G", "X"),
            match="n_per_segment",
        )
        assert_rejects(
            build_kpath,
            anchors,
            geometry,
            2,
            ("G",),
            match="one entry",
        )
        assert_rejects(
            build_kpath,
            anchors,
            geometry,
            2,
            ("G", "X"),
            anchor_units="scaled",
            match="anchor_units",
        )
        assert_rejects(
            build_kpath,
            anchors.at[1, 0].set(jnp.nan),
            geometry,
            2,
            ("G", "X"),
            match="anchors must be finite",
        )


class TestKpathArcLength:
    """Validate :func:`~diffpes.tightb.kpath_arc_length`.

    The cases cover a closed-form cubic path and lattice gradients across a
    repeated junction.

    :see: :func:`~diffpes.tightb.kpath_arc_length`
    """

    def test_matches_a_cubic_two_segment_path(self) -> None:
        """Match cumulative distance for a cubic path with one repeated point.

        The path must have total length ``pi`` in 1/Angstrom for a two
        Angstrom cubic lattice.

        Notes
        -----
        The test builds two perpendicular half-reciprocal segments and
        compares all six cumulative positions with a closed form.
        """
        geometry: CrystalGeometry = _make_geometry(2.0 * jnp.eye(3))
        kpath: KPath = build_kpath(
            jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
            geometry,
            3,
            ("G", "X", "M"),
        )
        actual: Float64[Array, "6"] = kpath_arc_length(kpath, geometry)
        expected: Float64[Array, "6"] = (
            jnp.array([0.0, 0.25, 0.5, 0.5, 0.75, 1.0]) * jnp.pi
        )
        chex.assert_trees_all_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_matches_finite_differences_through_a_repeated_junction(
        self,
    ) -> None:
        """Match lattice gradients with a repeated path junction.

        The zero-length junction must not produce a NaN or an incorrect
        derivative.

        Notes
        -----
        The shared gradient harness compares reverse mode with central finite
        differences for a weighted cumulative path distance.
        """
        lattice: Float64[Array, "3 3"] = jnp.array(
            [[2.3, 0.1, 0.0], [0.2, 2.7, 0.1], [0.1, 0.2, 3.1]]
        )
        points: Float64[Array, "6 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.4, 0.2, 0.0],
                [0.4, 0.4, 0.0],
            ]
        )
        kpath: KPath = KPath(
            kpoints=points,
            kz=None,
            labels=("G", "X", "M"),
            label_indices=(0, 3, 5),
            n_per_segment=3,
        )
        weights: Float64[Array, "6"] = jnp.arange(1.0, 7.0)

        def loss(candidate: Float64[Array, "3 3"]) -> Float64[Array, ""]:
            """Reduce cumulative distance for one candidate lattice."""
            geometry: CrystalGeometry = _make_geometry(candidate)
            distances: Float64[Array, "6"] = kpath_arc_length(kpath, geometry)
            result: Float64[Array, ""] = jnp.sum(weights * distances)
            return result

        assert_grad_matches_fd(loss, lattice, modes=("rev",))


class TestFirstBzMask:
    """Validate :func:`~diffpes.tightb.first_bz_mask`.

    The case uses the exact first-zone cube for a simple cubic lattice.

    :see: :func:`~diffpes.tightb.first_bz_mask`
    """

    def test_matches_exact_cubic_membership(self) -> None:
        """Include exactly the cubic points with each magnitude at most pi.

        Boundary points must remain inside, while points beyond one face must
        remain outside.

        Notes
        -----
        The test uses a one Angstrom cubic cell and compares the Boolean mask
        with a hand-classified point table.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        points: Float64[Array, "7 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [jnp.pi, 0.0, 0.0],
                [-jnp.pi, jnp.pi, -jnp.pi],
                [0.5 * jnp.pi, -0.5 * jnp.pi, 0.25 * jnp.pi],
                [jnp.pi + 1e-8, 0.0, 0.0],
                [0.0, -jnp.pi - 1e-8, 0.0],
                [0.0, 0.0, jnp.pi + 1e-8],
            ]
        )
        actual: Bool[Array, "7"] = first_bz_mask(points, geometry)
        expected: Bool[Array, "7"] = jnp.array(
            [True, True, True, True, False, False, False]
        )
        chex.assert_trees_all_equal(actual, expected)

    def test_matches_the_pinned_chinook_mesh_reduction(self) -> None:
        """Match the pinned Chinook reduction for a skew lattice.

        The mask must select the same indices and Cartesian points within the
        artifact tolerance.

        Notes
        -----
        The test validates the gate and source metadata. It reconstructs the
        geometry and compares its reciprocal basis and retained mesh points.
        """
        tests_root: Path = Path(__file__).resolve().parents[2]
        artifact_path: Path = (
            tests_root / "data" / "kspace" / "mesh_reduce_reference.json"
        )
        reference: dict[str, Any] = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
        metadata: dict[str, Any] = reference["metadata"]
        expected_metadata: dict[str, str] = {
            "chinook_commit": "24913de8cc5b8c162f7c1b4acc64bd1b54dd548b",
            "diffpes_commit": "afe36cfbb703510f01de6da376b35627eaac8d4d",
            "environment_sha256": (
                "6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531"
            ),
            "generator": "gen_chinook_kspace_reference.py",
            "numpy_version": "1.26.4",
            "python": "3.11.13",
        }

        chex.assert_equal(
            reference["requirement"], "kmesh-reduction-reference"
        )
        key: str
        expected_value: str
        for key, expected_value in expected_metadata.items():
            chex.assert_equal(metadata[key], expected_value)

        lattice: Float64[Array, "3 3"] = jnp.asarray(
            reference["lattice_angstrom"], dtype=jnp.float64
        )
        geometry: CrystalGeometry = _make_geometry(lattice)
        expected_reciprocal: Float64[Array, "3 3"] = jnp.asarray(
            reference["reciprocal_inv_ang"], dtype=jnp.float64
        )
        mesh: Float64[Array, "n_k 3"] = jnp.asarray(
            reference["mesh_cartesian_inv_ang"], dtype=jnp.float64
        )
        expected_indices: Int64[Array, " n_kept"] = jnp.asarray(
            reference["kept_indices"], dtype=jnp.int32
        )
        expected_points: Float64[Array, "n_kept 3"] = jnp.asarray(
            reference["reduced_cartesian_inv_ang"], dtype=jnp.float64
        )
        tolerance: float = float(reference["rtol"])
        mask: Bool[Array, " n_k"] = first_bz_mask(mesh, geometry)
        actual_count: int = int(jnp.sum(mask))
        expected_count: int = expected_indices.shape[0]

        chex.assert_equal(actual_count, expected_count)
        actual_indices: Int64[Array, " n_kept"] = jnp.flatnonzero(
            mask, size=expected_count
        )
        actual_points: Float64[Array, "n_kept 3"] = mesh[actual_indices]
        chex.assert_trees_all_close(
            geometry.reciprocal,
            expected_reciprocal,
            rtol=tolerance,
            atol=tolerance,
        )
        chex.assert_trees_all_equal(actual_indices, expected_indices)
        chex.assert_trees_all_close(
            actual_points,
            expected_points,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_handles_an_unreduced_basis_beyond_the_nearest_shell(self) -> None:
        """Reject a point whose closest competitor has coefficient minus two.

        The nearest 3 by 3 by 3 coefficient shell gives a false positive for
        this well-conditioned, right-handed lattice. The default static shell
        must include the missed reciprocal vector and return exact membership.

        Notes
        -----
        Use a fixed regression from a larger-shell independent search. Check
        the public result and the insufficient-radius diagnostic under JIT.
        """
        lattice: Float64[Array, "3 3"] = jnp.asarray(
            [
                [
                    -0.24707167010074266,
                    -1.0469393233990996,
                    0.3113395661549326,
                ],
                [1.5570143899623088, -0.0946897458348115, 1.7247869905052686],
                [0.21831194558249203, 1.0328348308246038, 0.9538407178405433],
            ]
        )
        geometry: CrystalGeometry = _make_geometry(lattice)
        fractional: Float64[Array, "1 3"] = jnp.asarray(
            [[-0.5248987892602519, -0.9982714929009722, 0.03732934725712278]]
        )
        point: Float64[Array, "1 3"] = kpoints_frac_to_cart(
            fractional,
            geometry,
        )

        actual: Bool[Array, " 1"] = first_bz_mask(point, geometry)

        chex.assert_trees_all_equal(actual, jnp.asarray([False]))
        assert_rejects(
            first_bz_mask,
            point,
            geometry,
            shell_radius=1,
            match="shell_radius is not provably sufficient",
        )


class TestBuildBzMesh:
    """Validate :func:`~diffpes.tightb.build_bz_mesh`.

    The cases cover fixed output shapes and the reciprocal-cell volume
    identity.

    :see: :func:`~diffpes.tightb.build_bz_mesh`
    """

    def test_preserves_all_mesh_points_and_estimates_the_zone_volume(
        self,
    ) -> None:
        """Keep the full mesh and estimate the cubic first-zone volume.

        The weighted volume error must decrease at the expected order of the
        axis spacing.

        Notes
        -----
        The test samples a 33-point fractional axis. It multiplies the mask
        fraction by the volume of the sampled reciprocal cube.
        """
        n_per_axis: int = 33
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        kgrid: KGrid
        mask: Bool[Array, " n_k"]
        kgrid, mask = build_bz_mesh(geometry, n_per_axis)
        reciprocal_volume: Float64[Array, ""] = jnp.abs(
            jnp.linalg.det(geometry.reciprocal)
        )
        estimated_volume: Float64[Array, ""] = (
            jnp.mean(mask.astype(jnp.float64)) * 8.0 * reciprocal_volume
        )
        relative_error: Float64[Array, ""] = jnp.abs(
            estimated_volume / reciprocal_volume - 1.0
        )

        chex.assert_shape(kgrid.kpoints, (n_per_axis**3, 3))
        chex.assert_shape(mask, (n_per_axis**3,))
        chex.assert_equal(kgrid.mesh_shape, (n_per_axis**2, n_per_axis))
        chex.assert_trees_all_close(
            relative_error, 0.0, atol=4.0 / n_per_axis, rtol=0.0
        )

    def test_rejects_an_axis_with_fewer_than_two_samples(self) -> None:
        """Reject a reciprocal axis that cannot span the sampled cube.

        The static sample count must contain at least both cube endpoints.

        Notes
        -----
        The shared helper passes ``n_per_axis=1`` through eager and compiled
        execution and matches the sample-count diagnostic.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        assert_rejects(
            build_bz_mesh,
            geometry,
            1,
            match="n_per_axis",
        )

    def test_rejects_a_basis_without_proven_cube_coverage(self) -> None:
        """Reject an unreduced basis whose first zone can leave the cube.

        The fixed fractional cube is valid only when reciprocal-basis
        inequalities bound every first-zone coordinate by one.

        Notes
        -----
        Use the unreduced regression lattice. Require the coverage diagnostic
        in eager and compiled execution instead of accepting a partial zone.
        """
        lattice: Float64[Array, "3 3"] = jnp.asarray(
            [
                [
                    -0.24707167010074266,
                    -1.0469393233990996,
                    0.3113395661549326,
                ],
                [1.5570143899623088, -0.0946897458348115, 1.7247869905052686],
                [0.21831194558249203, 1.0328348308246038, 0.9538407178405433],
            ]
        )
        geometry: CrystalGeometry = _make_geometry(lattice)

        assert_rejects(
            build_bz_mesh,
            geometry,
            5,
            match="does not prove cube coverage",
        )


class TestBuildArpesKmesh:
    """Validate :func:`~diffpes.tightb.build_arpes_kmesh`.

    The cases cover sample rotation, fractional conversion, azimuth
    derivatives, and fixed-shape JIT reuse.

    :see: :func:`~diffpes.tightb.build_arpes_kmesh`
    """

    def test_composes_detector_lab_momentum_with_sample_azimuth(self) -> None:
        """Match the laboratory-to-sample composition at nonzero azimuth.

        Detector kinematics supplies laboratory parallel components. The
        raster builder must apply the inverse sample orientation exactly once
        before fractional conversion.

        Notes
        -----
        Compose both slit maps with a generic sample azimuth. Convert the
        one-point rasters back to Cartesian coordinates and compare with
        ``S.T @ [kpar_lab, kz]``.
        """
        geometry: CrystalGeometry = _make_geometry(
            jnp.array([[2.4, 0.2, 0.0], [0.1, 2.7, 0.0], [0.0, 0.0, 4.1]])
        )
        tx: Float64[Array, ""] = jnp.array(0.23)
        ty: Float64[Array, ""] = jnp.array(-0.17)
        energy: Float64[Array, ""] = jnp.array(35.0)
        kz: Float64[Array, ""] = jnp.array(1.4)
        azimuth: Float64[Array, ""] = jnp.array(0.41)
        cosine: Float64[Array, ""] = jnp.cos(azimuth)
        sine: Float64[Array, ""] = jnp.sin(azimuth)
        sample_orientation: Float64[Array, "3 3"] = jnp.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        slit: str
        for slit in ("H", "V"):
            k_parallel_lab: Float64[Array, "2"] = detector_angles_to_kpar(
                tx,
                ty,
                energy,
                slit,
            )
            lab_momentum: Float64[Array, "3"] = jnp.concatenate(
                (k_parallel_lab, kz[None])
            )
            expected: Float64[Array, "1 3"] = (
                sample_orientation.T @ lab_momentum
            )[None, :]
            kgrid: KGrid = build_arpes_kmesh(
                k_parallel_lab[0:1],
                k_parallel_lab[1:2],
                kz,
                azimuth,
                geometry,
            )
            actual: Float64[Array, "1 3"] = kpoints_frac_to_cart(
                kgrid.kpoints,
                geometry,
            )
            chex.assert_trees_all_close(
                actual,
                expected,
                rtol=1e-13,
                atol=1e-13,
            )

    def test_rotates_the_laboratory_raster_into_the_sample(self) -> None:
        """Rotate a positive laboratory x point to negative sample y.

        A positive 90-degree sample azimuth must apply a negative 90-degree
        coordinate rotation.

        Notes
        -----
        The test converts the generated fractional grid back to Cartesian
        coordinates and compares one row with the rotated closed form.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        kgrid: KGrid = build_arpes_kmesh(
            jnp.array([1.0]),
            jnp.array([0.0]),
            0.5,
            0.5 * jnp.pi,
            geometry,
        )
        cartesian: Float64[Array, "1 3"] = kpoints_frac_to_cart(
            kgrid.kpoints, geometry
        )
        expected: Float64[Array, "1 3"] = jnp.array([[0.0, -1.0, 0.5]])
        chex.assert_trees_all_close(
            cartesian, expected, atol=1e-12, rtol=1e-12
        )
        chex.assert_equal(kgrid.mesh_shape, (1, 1))

    def test_matches_finite_differences_for_sample_azimuth(self) -> None:
        """Match the nonzero azimuth derivative with central differences.

        The registered raster must retain sensitivity to sample alignment.

        Notes
        -----
        The shared gradient gate differentiates a weighted fractional raster
        at a generic nonzero azimuth.
        """
        geometry: CrystalGeometry = _make_geometry(
            jnp.array([[2.4, 0.1, 0.0], [0.2, 2.7, 0.0], [0.0, 0.0, 4.0]])
        )
        weights: Float64[Array, "6 3"] = jnp.arange(1.0, 19.0).reshape((6, 3))

        def loss(azimuth: Float64[Array, ""]) -> Float64[Array, ""]:
            """Reduce one ARPES raster at the candidate sample azimuth."""
            kgrid: KGrid = build_arpes_kmesh(
                jnp.array([-0.4, 0.1, 0.7]),
                jnp.array([-0.2, 0.3]),
                0.8,
                azimuth,
                geometry,
            )
            result: Float64[Array, ""] = jnp.sum(kgrid.kpoints * weights)
            return result

        gradient_gate(loss, jnp.asarray(0.23), modes=("rev",))

    def test_reuses_one_trace_for_traced_mesh_values(self) -> None:
        """Reuse one trace across fixed-shape kz, azimuth, and energy sweeps.

        Changes to traced values must not cause a second compilation when all
        axis lengths stay fixed.

        Notes
        -----
        A Python counter records traces around both ARPES mesh builders. Two
        calls change every traced scalar and the photon-energy values.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        trace_count: list[int] = [0]

        def counted(
            kz: Float64[Array, ""],
            azimuth: Float64[Array, ""],
            photon_energies: Float64[Array, "2"],
        ) -> Array:
            """Build both fixed-shape rasters and return their coordinates."""
            trace_count[0] += 1
            arpes_grid: KGrid = build_arpes_kmesh(
                jnp.array([-0.2, 0.2]),
                jnp.array([-0.1, 0.1]),
                kz,
                azimuth,
                geometry,
            )
            photon_grid: KGrid = build_kmesh_hv_at_fermi(
                jnp.array([-0.2, 0.2]),
                photon_energies,
                4.5,
                12.0,
                azimuth,
                jnp.array([1.0, 0.0]),
                geometry,
            )
            result: Array = jnp.concatenate(
                (arpes_grid.kpoints.ravel(), photon_grid.kpoints.ravel())
            )
            return result

        compiled: Callable[..., Any] = eqx.filter_jit(counted)
        compiled(
            jnp.asarray(0.5), jnp.asarray(0.1), jnp.array([30.0, 50.0])
        ).block_until_ready()
        compiled(
            jnp.asarray(0.8), jnp.asarray(0.3), jnp.array([35.0, 60.0])
        ).block_until_ready()
        chex.assert_equal(trace_count[0], 1)


class TestBuildKmeshHvAtFermi:
    """Validate :func:`~diffpes.tightb.build_kmesh_hv_at_fermi`.

    The case checks direct free-electron composition and photon-axis metadata.

    :see: :func:`~diffpes.tightb.build_kmesh_hv_at_fermi`
    """

    def test_composes_the_free_electron_kz_rows(self) -> None:
        """Match every out-of-plane row with the kinematics primitive.

        The Cartesian third coordinate must equal the real propagating
        momentum for each photon energy and parallel momentum.

        Notes
        -----
        The test computes the expected rows by direct vectorization of
        :func:`diffpes.simul.kz_from_inner_potential_at_fermi`.
        """
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        parallel_axis: Float64[Array, "3"] = jnp.array([-0.5, 0.0, 0.5])
        photon_energies: Float64[Array, "2"] = jnp.array([30.0, 50.0])
        kgrid: KGrid = build_kmesh_hv_at_fermi(
            parallel_axis,
            photon_energies,
            4.5,
            12.0,
            0.0,
            jnp.array([1.0, 0.0]),
            geometry,
        )
        cartesian: Float64[Array, "2 3 3"] = kpoints_frac_to_cart(
            kgrid.kpoints, geometry
        ).reshape((2, 3, 3))

        def expected_row(
            photon_energy: Float64[Array, ""],
        ) -> Tuple[Complex128[Array, "3"], Bool[Array, "3"]]:
            """Compute one direct free-electron row."""
            kz_values: Complex128[Array, "3"]
            propagating: Bool[Array, "3"]
            kz_values, propagating = kz_from_inner_potential_at_fermi(
                photon_energy, 4.5, 12.0, jnp.abs(parallel_axis)
            )
            result: Tuple[Complex128[Array, "3"], Bool[Array, "3"]] = (
                kz_values,
                propagating,
            )
            return result

        expected_complex: Complex128[Array, "2 3"]
        propagating: Bool[Array, "2 3"]
        expected_complex, propagating = jax.vmap(expected_row)(photon_energies)
        expected_kz: Float64[Array, "2 3"] = jnp.real(expected_complex)
        chex.assert_trees_all_equal(propagating, jnp.ones((2, 3), dtype=bool))
        expected_x: Float64[Array, "2 3"] = jnp.broadcast_to(
            parallel_axis, (2, 3)
        )

        chex.assert_trees_all_close(cartesian[..., 0], expected_x, rtol=1e-12)
        chex.assert_trees_all_close(cartesian[..., 1], 0.0, atol=1e-15)
        chex.assert_trees_all_close(cartesian[..., 2], expected_kz, rtol=1e-12)
        chex.assert_trees_all_close(
            kgrid.photon_energy_axis_ev, photon_energies
        )
        chex.assert_equal(kgrid.mesh_shape, (2, 3))

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(700)
    def test_large_composed_raster_has_static_shape_and_linear_memory(
        self,
    ) -> None:
        """Build the full 64 by 256-squared photon-energy raster.

        The case retains static carrier metadata under JIT compilation. It
        allocates the expected 96 MiB result without point-quadratic storage.

        Notes
        -----
        ``build_kmesh_hv_at_fermi`` internally vmaps
        :func:`diffpes.simul.kz_from_inner_potential_at_fermi` over all 64
        photon energies. The RSS fixture caps retained growth at 700 MiB while
        this test checks the exact dense-result byte count.
        """
        n_photon: int = 64
        n_parallel: int = 256 * 256
        geometry: CrystalGeometry = _make_geometry(jnp.eye(3))
        parallel_axis: Float64[Array, " 65536"] = jnp.linspace(
            -1.5,
            1.5,
            n_parallel,
        )
        photon_energies: Float64[Array, " 64"] = jnp.linspace(
            20.0,
            150.0,
            n_photon,
        )

        def composed_builder(
            axis: Float64[Array, " 65536"],
            energies: Float64[Array, " 64"],
        ) -> KGrid:
            """Build one full photon-energy raster through the kz primitive."""
            kgrid: KGrid = build_kmesh_hv_at_fermi(
                axis,
                energies,
                4.5,
                12.0,
                0.23,
                jnp.array([1.0, 0.0]),
                geometry,
            )
            return kgrid

        kgrid: KGrid = jax.jit(composed_builder)(
            parallel_axis,
            photon_energies,
        )
        kgrid.kpoints.block_until_ready()
        chex.assert_shape(kgrid.kpoints, (n_photon * n_parallel, 3))
        chex.assert_shape(kgrid.photon_energy_axis_ev, (n_photon,))
        chex.assert_equal(kgrid.mesh_shape, (n_photon, n_parallel))
        chex.assert_tree_all_finite(
            (kgrid.kpoints, kgrid.photon_energy_axis_ev)
        )
        chex.assert_equal(
            kgrid.kpoints.nbytes,
            n_photon * n_parallel * 3 * jnp.dtype(jnp.float64).itemsize,
        )
