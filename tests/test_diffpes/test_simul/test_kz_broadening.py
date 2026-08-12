"""Validate the kz broadening module.

The cases use analytic values, invariants, and finite differences.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    broaden_kz,
    kz_broadening,
    kz_fractional_nodes,
    kz_wrapped_lorentzian_bin_weights,
)
from diffpes.types import (
    CrystalGeometry,
    SurfaceCell,
)
from tests._assertions import assert_rejects
from tests._gradients import assert_gradients_match_finite_differences

from ._effects_helpers import (
    _surface_kz_fixture,
    _wrapped_cauchy_fourier_bin_masses,
)


class TestKzFractionalNodes:
    """Verify :func:`diffpes.simul.kz_fractional_nodes`.

    The class owns the static midpoint grid and one-node rejection.
    """

    def test_returns_exact_uniform_centres_and_jits(self) -> None:
        """Build the registered half-open midpoint grid under eager and JIT.

        The four-bin fixture has exact binary-representable centres.

        Notes
        -----
        Marking ``n_kz`` static preserves one compiled shape per node count.
        """
        desired: Float64[Array, "4"] = jnp.array(
            [-0.375, -0.125, 0.125, 0.375]
        )
        eager: Float64[Array, "..."] = kz_fractional_nodes(4)
        compiled: Float64[Array, "..."] = jax.jit(
            kz_fractional_nodes, static_argnums=0
        )(4)

        chex.assert_trees_all_equal(eager, desired)
        chex.assert_trees_all_equal(compiled, desired)
        chex.assert_trees_all_equal(jnp.diff(eager), jnp.full(3, 0.25))

    @pytest.mark.parametrize("invalid_count", [0, 1, True])
    def test_rejects_nonquadrature_counts(self, invalid_count: int) -> None:
        """Reject empty, one-node, and boolean finite-width grids.

        A one-node midpoint erases every mean-free-path dependence.

        Notes
        -----
        ``bulk_direct`` is a separate no-node route and cannot use this helper.
        """
        with pytest.raises(ValueError, match="static integer of at least two"):
            kz_fractional_nodes(invalid_count)


class TestKzWrappedLorentzianBinWeights:
    """Verify :func:`diffpes.simul.kz_wrapped_lorentzian_bin_weights`.

    The class owns analytic wrapped bin mass, units, and validation.
    """

    def test_matches_fourier_bin_masses_across_period_seam(self) -> None:
        """Match an independent Fourier integral on unequal fractional bins.

        The centre at ``0.487`` forces probability across the period seam.

        Notes
        -----
        The omitted Fourier tail is below ``1e-22`` for this fixture, well
        inside the registered ``1e-12`` reference-remainder ceiling.
        """
        edges_np: Float64[NDArray, " Np1"] = (
            -0.5 + np.linspace(0.0, 1.0, 18) ** 1.3
        )
        center: float = 0.487
        mean_free_path: float = 7.5
        period: float = 2.2
        gamma_frac: float = 0.5 / (mean_free_path * period)
        desired: Float64[NDArray, " N"] = _wrapped_cauchy_fourier_bin_masses(
            edges_np, center, gamma_frac
        )
        actual: Float64[Array, "..."] = kz_wrapped_lorentzian_bin_weights(
            jnp.asarray(edges_np),
            jnp.asarray(center),
            mean_free_path,
            period,
        )
        decay: float = np.exp(-2.0 * np.pi * gamma_frac)
        remainder_bound: float = (
            2.0 * decay**257 / (np.pi * 257.0 * (1.0 - decay))
        )

        np.testing.assert_allclose(actual, desired, rtol=1.0e-13, atol=5e-15)
        assert bool(jnp.all(actual > 0.0))
        np.testing.assert_allclose(np.sum(actual), 1.0, rtol=1.0e-13, atol=0.0)
        assert remainder_bound <= 1.0e-12

    def test_private_streamed_bin_equals_every_public_vector_mass(
        self,
    ) -> None:
        """Match the scalar-bin streaming seam to batched public weights.

        Three centres include both sides of the branch cut and the origin.

        Notes
        -----
        The private helper vmaps only over bins and leaves the centre batch
        intact. A driver can therefore scan without a complete K-by-E-by-node
        carrier.
        """
        edges: Float64[Array, "33"] = jnp.linspace(-0.5, 0.5, 33)
        centres: Float64[Array, "3"] = jnp.array([-0.49, 0.0, 0.49])
        public: Float64[Array, "..."] = kz_wrapped_lorentzian_bin_weights(
            edges, centres, 10.0, 1.8
        )
        node_first: Float64[Array, "..."] = jax.vmap(
            lambda lower, upper: (
                kz_broadening._kz_wrapped_lorentzian_bin_weight(  # noqa: SLF001
                    lower,
                    upper,
                    centres,
                    10.0,
                    1.8,
                )
            )
        )(edges[:-1], edges[1:])
        streamed: Float64[Array, "..."] = jnp.moveaxis(node_first, 0, -1)
        compiled: Float64[Array, "..."] = jax.jit(
            kz_wrapped_lorentzian_bin_weights
        )(edges, centres, 10.0, 1.8)

        chex.assert_trees_all_equal(streamed, public)
        chex.assert_trees_all_close(compiled, public, rtol=1.0e-13, atol=0.0)

    def test_uses_fractional_width_and_preserves_physical_units(
        self,
    ) -> None:
        """Keep weights invariant at fixed ``lambda * G_perp``.

        Omitting division by the physical period changes the planted result.

        Notes
        -----
        This executable counterexample prevents mixing fractional bin edges
        with the inverse-angstrom Lorentzian HWHM.
        """
        edges: Float64[Array, "65"] = jnp.linspace(-0.5, 0.5, 65)
        center: Float64[Array, ""] = jnp.asarray(0.173)
        first: Float64[Array, "..."] = kz_wrapped_lorentzian_bin_weights(
            edges, center, 5.0, 2.0
        )
        rescaled: Float64[Array, "..."] = kz_wrapped_lorentzian_bin_weights(
            edges, center, 10.0, 1.0
        )
        planted_wrong_units: Float64[Array, "..."] = (
            kz_wrapped_lorentzian_bin_weights(edges, center, 5.0, 1.0)
        )

        chex.assert_trees_all_equal(first, rescaled)
        assert float(jnp.max(jnp.abs(first - planted_wrong_units))) > 1.0e-2

    @pytest.mark.parametrize(
        ("edges", "center", "mean_free_path", "period", "message"),
        [
            (
                jnp.array([-0.4, 0.0, 0.5]),
                jnp.asarray(0.0),
                10.0,
                2.0,
                "span",
            ),
            (
                jnp.array([-0.5, 0.0, 0.5]),
                jnp.asarray(0.5),
                10.0,
                2.0,
                "folded kz centres",
            ),
            (
                jnp.array([-0.5, 0.0, 0.5]),
                jnp.asarray(0.0),
                0.0,
                2.0,
                "mean_free_path_ang",
            ),
            (
                jnp.array([-0.5, 0.0, 0.5]),
                jnp.asarray(0.0),
                10.0,
                np.inf,
                "period_inv_ang",
            ),
        ],
    )
    def test_rejects_invalid_physical_domains_eager_and_jit(
        self,
        edges: Float64[Array, "..."],
        center: Float64[Array, "..."],
        mean_free_path: float,
        period: float,
        message: str,
    ) -> None:
        """Reject malformed edges, centres, and physical scales.

        The test exercises each invalid value through eager and compiled calls.

        Notes
        -----
        The finite-width path admits neither an infinite-lambda endpoint nor
        an unfolded centre at the excluded positive boundary.
        """
        assert_rejects(
            kz_wrapped_lorentzian_bin_weights,
            edges,
            center,
            mean_free_path,
            period,
            match=message,
        )


class TestBroadenKz:
    """Verify :func:`diffpes.simul.broaden_kz`.

    The class owns wrapped quadrature averaging and the local lambda
    derivative.
    """

    def test_preserves_constant_and_matches_wrapped_voigt_fourier(
        self,
    ) -> None:
        """Preserve unit density and match the wrapped-Voigt Fourier truth.

        The refined midpoint grid leaves less than ``1e-8`` relative error.

        Notes
        -----
        A wrapped Gaussian supplies the input. Multiplying its Fourier
        coefficients by the wrapped-Cauchy coefficients gives an independent
        analytic wrapped-Voigt value at the requested centre.
        """
        n_kz: int = 16_384
        nodes_np: Float64[NDArray, " N"] = np.asarray(
            kz_fractional_nodes(n_kz)
        )
        edges: Float64[Array, "..."] = jnp.linspace(-0.5, 0.5, n_kz + 1)
        center: float = 0.18
        gaussian_center: float = -0.23
        sigma_frac: float = 0.07
        mean_free_path: float = 8.0
        period: float = 1.6
        gamma_frac: float = 0.5 / (mean_free_path * period)
        harmonics: Float64[NDArray, " M"] = np.arange(1, 65, dtype=np.float64)
        gaussian_coefficients: Float64[NDArray, " M"] = np.exp(
            -0.5 * np.square(2.0 * np.pi * sigma_frac * harmonics)
        )
        wrapped_gaussian: Float64[NDArray, " N"] = 1.0 + 2.0 * np.sum(
            gaussian_coefficients[:, None]
            * np.cos(
                2.0
                * np.pi
                * harmonics[:, None]
                * (nodes_np[None, :] - gaussian_center)
            ),
            axis=0,
        )
        weights: Float64[Array, "..."] = kz_wrapped_lorentzian_bin_weights(
            edges,
            jnp.asarray(center),
            mean_free_path,
            period,
        )
        actual: Float64[Array, "..."] = broaden_kz(
            jnp.asarray(wrapped_gaussian), weights
        )
        desired: float = 1.0 + 2.0 * np.sum(
            np.exp(-2.0 * np.pi * gamma_frac * harmonics)
            * gaussian_coefficients
            * np.cos(2.0 * np.pi * harmonics * (center - gaussian_center))
        )
        constant: Float64[Array, "..."] = broaden_kz(jnp.ones(n_kz), weights)

        np.testing.assert_allclose(actual, desired, rtol=1.0e-8, atol=0.0)
        np.testing.assert_allclose(constant, 1.0, rtol=1.0e-13, atol=0.0)

    @pytest.mark.parametrize("mean_free_path", [5.0, 10.0, 50.0])
    def test_mean_free_path_gradient_matches_fd_and_is_nonzero(
        self, mean_free_path: float
    ) -> None:
        """Match forward/reverse lambda derivatives at all tested lengths.

        The asymmetric periodic intensity keeps every gradient nonzero.

        Notes
        -----
        The shared smooth f64 ladder supplies directional and elementwise
        central-finite-difference comparisons.
        """
        n_kz: int = 96
        nodes: Float64[Array, "..."] = kz_fractional_nodes(n_kz)
        edges: Float64[Array, "..."] = jnp.linspace(-0.5, 0.5, n_kz + 1)
        intensity: Float64[Array, "..."] = (
            1.2
            + 0.31 * jnp.cos(2.0 * jnp.pi * nodes)
            + 0.17 * jnp.sin(4.0 * jnp.pi * nodes)
        )

        def loss(candidate: Float64[Array, "..."]) -> Float64[Array, "..."]:
            candidate_weights: Float64[Array, "..."] = (
                kz_wrapped_lorentzian_bin_weights(
                    edges,
                    jnp.asarray(0.173),
                    candidate,
                    jnp.asarray(1.8),
                )
            )
            returned: Float64[Array, "..."] = broaden_kz(
                intensity, candidate_weights
            )
            return returned

        assert_gradients_match_finite_differences(
            loss,
            jnp.asarray(mean_free_path),
            regime="smooth",
            scale_floor=1.0,
        )

    def test_jit_and_vmap_match_direct_lambda_centre_sweeps(self) -> None:
        """Compile and batch the full weight-plus-reduction success path.

        Three centre/lambda pairs share one static node schedule.

        Notes
        -----
        Direct scalar evaluations provide the independent batched comparison.
        """
        n_kz: int = 64
        nodes: Float64[Array, "..."] = kz_fractional_nodes(n_kz)
        edges: Float64[Array, "..."] = jnp.linspace(-0.5, 0.5, n_kz + 1)
        intensity: Float64[Array, "..."] = 1.1 + 0.2 * jnp.cos(
            2.0 * jnp.pi * (nodes - 0.07)
        )
        centres: Float64[Array, "3"] = jnp.array([-0.31, 0.04, 0.39])
        lengths: Float64[Array, "3"] = jnp.array([5.0, 10.0, 50.0])

        def evaluate(
            center: Float64[Array, "..."], length: Float64[Array, "..."]
        ) -> Float64[Array, "..."]:
            candidate_weights: Float64[Array, "..."] = (
                kz_wrapped_lorentzian_bin_weights(
                    edges, center, length, jnp.asarray(1.8)
                )
            )
            returned: Float64[Array, "..."] = broaden_kz(
                intensity, candidate_weights
            )
            return returned

        direct: Float64[Array, "..."] = jnp.stack(
            [
                evaluate(center, length)
                for center, length in zip(centres, lengths, strict=True)
            ]
        )
        batched: Float64[Array, "..."] = jax.jit(jax.vmap(evaluate))(
            centres, lengths
        )

        chex.assert_trees_all_close(batched, direct, rtol=1.0e-13, atol=0.0)

    def test_joint_refinement_approaches_off_grid_direct_value(
        self,
    ) -> None:
        """Verify convergence toward a periodic direct value during refinement.

        Node counts grow by eight while fractional HWHM shrinks by four.

        Notes
        -----
        Thus ``delta_u / gamma_u`` halves on every step. A fixed-grid
        infinite-lambda limit is deliberately neither formed nor claimed.
        """
        center: float = 0.173
        period: float = 2.0
        counts: Tuple[int, ...] = (64, 512, 4096)
        lengths: Tuple[float, ...] = (3.125, 12.5, 50.0)
        direct: float = (
            1.0
            + 0.3 * np.cos(2.0 * np.pi * center)
            + 0.1 * np.sin(4.0 * np.pi * center)
        )
        errors: List[float] = []
        ratios: List[float] = []
        count: int
        length: float
        for count, length in zip(counts, lengths, strict=True):
            nodes: Float64[Array, "..."] = kz_fractional_nodes(count)
            edges: Float64[Array, "..."] = jnp.linspace(-0.5, 0.5, count + 1)
            intensity: Float64[Array, "..."] = (
                1.0
                + 0.3 * jnp.cos(2.0 * jnp.pi * nodes)
                + 0.1 * jnp.sin(4.0 * jnp.pi * nodes)
            )
            candidate_weights: Float64[Array, "..."] = (
                kz_wrapped_lorentzian_bin_weights(
                    edges, jnp.asarray(center), length, period
                )
            )
            broadened: Float64[Array, "..."] = broaden_kz(
                intensity, candidate_weights
            )
            errors.append(abs(float(broadened) - direct))
            gamma_frac: float = 0.5 / (length * period)
            ratios.append((1.0 / count) / gamma_frac)

        assert errors[2] < errors[1] < errors[0]
        assert ratios[2] < ratios[1] < ratios[0]

    def test_rejects_one_node_shapes_and_nonphysical_values(self) -> None:
        """Reject the one-node counterexample and malformed weighted inputs.

        Shape checks remain static while value checks run eagerly and in JIT.

        Notes
        -----
        A zero weight is invalid because finite wrapped-Cauchy bins have
        strictly positive mass over the complete primitive period.
        """
        with pytest.raises(ValueError, match="at least two nodes"):
            broaden_kz(jnp.ones(1), jnp.ones(1))
        with pytest.raises(ValueError, match="remaining static shapes"):
            broaden_kz(jnp.ones((3, 2)), jnp.ones((3,)) / 3.0)
        assert_rejects(
            broaden_kz,
            jnp.array([1.0, -0.1]),
            jnp.array([0.5, 0.5]),
            match="finite and nonnegative",
        )
        assert_rejects(
            broaden_kz,
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 0.0]),
            match="finite, positive, and sum to one",
        )


class TestSurfaceKzFrame:
    """Verify the private primitive surface reciprocal-frame seam.

    The class owns unit-cell advance and reciprocal identities. It rejects
    stale
    data before any bulk model evaluation.
    """

    def test_cubic_frame_is_exact_and_jittable(self) -> None:
        """Recover the cubic direct, reciprocal, normal, and period values.

        Identity coefficients and rotation make the external truth explicit.

        Notes
        -----
        The compiled carrier replay exercises traced cross-carrier validation.
        """
        geometry: CrystalGeometry
        cell: SurfaceCell
        geometry, cell = _surface_kz_fixture(oblique=False)
        direct: Float64[Array, "..."]
        reciprocal: Float64[Array, "..."]
        normal: Float64[Array, "..."]
        period: Float64[Array, "..."]
        direct, reciprocal, normal, period = kz_broadening._surface_kz_frame(  # noqa: SLF001
            cell, geometry
        )
        compiled: Tuple[
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
            Float64[Array, "..."],
        ] = jax.jit(
            kz_broadening._surface_kz_frame  # noqa: SLF001
        )(cell, geometry)

        chex.assert_trees_all_close(direct, jnp.eye(3))
        chex.assert_trees_all_close(reciprocal, 2.0 * jnp.pi * jnp.eye(3))
        chex.assert_trees_all_close(normal, jnp.array([0.0, 0.0, 1.0]))
        chex.assert_trees_all_close(period, 2.0 * jnp.pi)
        chex.assert_trees_all_close(
            compiled, (direct, reciprocal, normal, period)
        )

    def test_rejects_stale_doubled_stacking_before_mapping(self) -> None:
        """Reject numerical ``g=2`` data carrying stale unit-advance metadata.

        The carrier factory alone accepts the internally shaped planted cell.

        Notes
        -----
        Reconstruction from bulk lattice and exact coefficients exposes the
        doubled vector before its false half-period reaches a dispersion.
        """
        geometry: CrystalGeometry
        stale_cell: SurfaceCell
        geometry, stale_cell = _surface_kz_fixture(
            oblique=False, doubled_stacking=True
        )
        assert_rejects(
            kz_broadening._surface_kz_frame,  # noqa: SLF001
            stale_cell,
            geometry,
            match="coefficient @ bulk lattice @ rotation",
        )


class TestMapSurfaceFractionalToBulk:
    """Verify the private arbitrary surface-to-bulk momentum map.

    The class owns bulk-direct centres, oblique coupling, and periodicity.
    """

    def test_maps_arbitrary_k_by_energy_centres_and_jits(self) -> None:
        """Verify off-grid ``(K,E)`` centres with exact third coordinates.

        The centres exercise the generic bulk-direct surface used by drivers.

        Notes
        -----
        JIT output includes all reciprocal and cross-carrier checks.
        """
        geometry: CrystalGeometry
        cell: SurfaceCell
        geometry, cell = _surface_kz_fixture(oblique=True)
        k_parallel: Float64[Array, "2 3"] = jnp.array(
            [[0.21, -0.17, 0.0], [-0.31, 0.23, 0.0]]
        )
        centres: Float64[Array, "2 3"] = jnp.array(
            [[-0.37, 0.04, 0.29], [-0.42, -0.11, 0.33]]
        )
        surface: Float64[Array, "..."]
        bulk_fractional: Float64[Array, "..."]
        surface, bulk_fractional = (
            kz_broadening._map_surface_fractional_to_bulk(  # noqa: SLF001
                k_parallel, centres, cell, geometry
            )
        )
        compiled: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
            jax.jit(
                kz_broadening._map_surface_fractional_to_bulk  # noqa: SLF001
            )(k_parallel, centres, cell, geometry)
        )
        direct: Float64[Array, "..."] = kz_broadening._surface_kz_frame(  # noqa: SLF001
            cell, geometry
        )[0]
        recovered: Float64[Array, "..."] = surface @ direct.T / (2.0 * jnp.pi)

        chex.assert_trees_all_close(
            recovered[..., 2], centres, rtol=1.0e-12, atol=1.0e-14
        )
        chex.assert_trees_all_close(compiled, (surface, bulk_fractional))


class TestMapSurfaceKzNodesToBulkFractional:
    """Verify the private registered-node surface-to-bulk map.

    The class owns lateral stacking coupling, folding, and periodicity.
    """

    def test_oblique_map_round_trips_and_preserves_periodicity(
        self,
    ) -> None:
        """Verify an oblique cell and reciprocal-translation periodicity.

        The third surface coordinate equals every registered node exactly.

        Notes
        -----
        Shifting physical momentum by the in-plane projection of the first
        surface reciprocal row changes bulk fractional momentum by an integer,
        preserving dispersion. The generic mapper supplies the reciprocal
        row's compensating normal component through ``u_parallel``.
        """
        geometry: CrystalGeometry
        cell: SurfaceCell
        geometry, cell = _surface_kz_fixture(oblique=True)
        nodes: Float64[Array, "..."] = kz_fractional_nodes(8)
        k_parallel: Float64[Array, "2 3"] = jnp.array(
            [[0.21, -0.17, 0.0], [-0.31, 0.23, 0.0]]
        )
        direct: Float64[Array, "..."]
        reciprocal: Float64[Array, "..."]
        normal: Float64[Array, "..."]
        direct, reciprocal, normal, _ = kz_broadening._surface_kz_frame(  # noqa: SLF001
            cell, geometry
        )
        in_plane_reciprocal_shift: Float64[Array, "..."] = (
            reciprocal[0] - jnp.dot(reciprocal[0], normal) * normal
        )
        surface: Float64[Array, "..."]
        bulk_fractional: Float64[Array, "..."]
        surface, bulk_fractional = (
            kz_broadening._map_surface_kz_nodes_to_bulk_fractional(  # noqa: SLF001
                k_parallel, nodes, cell, geometry
            )
        )
        shifted_surface: Float64[Array, "..."]
        shifted_bulk_fractional: Float64[Array, "..."]
        shifted_surface, shifted_bulk_fractional = jax.jit(
            kz_broadening._map_surface_kz_nodes_to_bulk_fractional  # noqa: SLF001
        )(k_parallel + in_plane_reciprocal_shift, nodes, cell, geometry)
        surface_fractional: Float64[Array, "..."] = (
            surface @ direct.T / (2.0 * jnp.pi)
        )
        shift_difference: Float64[Array, "..."] = (
            shifted_bulk_fractional - bulk_fractional
        )

        chex.assert_trees_all_close(
            surface_fractional[..., 2],
            jnp.broadcast_to(nodes, surface_fractional[..., 2].shape),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            shift_difference,
            jnp.broadcast_to(
                jnp.array([1.0, 0.0, 0.0]), shift_difference.shape
            ),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        intensity: Float64[Array, "..."] = 1.3 + 0.2 * jnp.sum(
            jnp.cos(2.0 * jnp.pi * bulk_fractional), axis=-1
        )
        shifted_intensity: Float64[Array, "..."] = 1.3 + 0.2 * jnp.sum(
            jnp.cos(2.0 * jnp.pi * shifted_bulk_fractional), axis=-1
        )
        chex.assert_trees_all_close(
            shifted_intensity, intensity, rtol=1.0e-12, atol=1.0e-14
        )
        assert bool(jnp.all(jnp.isfinite(shifted_surface)))

    def test_planted_scalar_append_loses_oblique_lateral_coupling(
        self,
    ) -> None:
        """Make the forbidden scalar-fractional append disagree visibly.

        The compliant physical map uses ``u_parallel`` from the full v3 row.

        Notes
        -----
        Appending the node to Cartesian in-plane components changes the bulk
        point for the oblique fixture.
        """
        geometry: CrystalGeometry
        cell: SurfaceCell
        geometry, cell = _surface_kz_fixture(oblique=True)
        nodes: Float64[Array, "..."] = kz_fractional_nodes(4)
        k_parallel: Float64[Array, "3"] = jnp.array([0.27, -0.19, 0.0])
        actual: Float64[Array, "..."]
        _, actual = kz_broadening._map_surface_kz_nodes_to_bulk_fractional(  # noqa: SLF001
            k_parallel, nodes, cell, geometry
        )
        planted_wrong: Float64[Array, "..."] = jnp.stack(
            (
                jnp.full_like(nodes, k_parallel[0]),
                jnp.full_like(nodes, k_parallel[1]),
                nodes,
            ),
            axis=-1,
        )

        assert float(jnp.max(jnp.abs(actual - planted_wrong))) > 1.0e-2

    def test_rejects_nonplane_momentum_and_unregistered_nodes(self) -> None:
        """Reject a normal momentum component and a shifted node schedule.

        Both counterexamples violate the private physical mapping boundary.

        Notes
        -----
        The shared rejection helper exercises traced checks in eager and JIT.
        """
        geometry: CrystalGeometry
        cell: SurfaceCell
        geometry, cell = _surface_kz_fixture(oblique=True)
        nodes: Float64[Array, "..."] = kz_fractional_nodes(4)
        assert_rejects(
            kz_broadening._map_surface_kz_nodes_to_bulk_fractional,  # noqa: SLF001
            jnp.array([0.2, 0.1, 0.03]),
            nodes,
            cell,
            geometry,
            match="surface plane",
        )
        assert_rejects(
            kz_broadening._map_surface_kz_nodes_to_bulk_fractional,  # noqa: SLF001
            jnp.array([0.2, 0.1, 0.0]),
            nodes + 0.01,
            cell,
            geometry,
            match="registered uniform fractional centres",
        )
