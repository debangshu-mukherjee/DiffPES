"""Validate Slater--Koster blocks, neighbors, models, and gradients.

The analytic gate covers every s/p/d sigma, pi, and delta channel on fifty
generic directions. Additional cases pin parity, swapped-shell Hermiticity,
exact integer connectivity, hand-counted honeycomb/fcc shells, and the
transverse derivative at both bond poles.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float
from numpy.typing import NDArray

from diffpes.tightb import (
    bloch_hamiltonian,
    build_sk_model,
    neighbor_shells,
    sk_block,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_slater_koster_params,
)
from tests._gradients import assert_grad_matches_fd, assert_nonzero_grad

_TABLE_DIRECTIONS: int = 50
_ALL_SK_KEYS: tuple[str, ...] = (
    "X-X:ss_sigma",
    "X-X:sp_sigma",
    "X-X:sd_sigma",
    "X-X:pp_sigma",
    "X-X:pp_pi",
    "X-X:pd_sigma",
    "X-X:pd_pi",
    "X-X:dd_sigma",
    "X-X:dd_pi",
    "X-X:dd_delta",
)


def _reference_d_tensors() -> Float[Array, "5 3 3"]:
    """Return the normalized Table-I d-orbital Cartesian tensors."""
    inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six: float = 1.0 / np.sqrt(6.0)
    tensors: Float[Array, "5 3 3"] = jnp.asarray(
        (
            (
                (0.0, inverse_sqrt_two, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, inverse_sqrt_two, 0.0),
            ),
            (
                (-inverse_sqrt_six, 0.0, 0.0),
                (0.0, -inverse_sqrt_six, 0.0),
                (0.0, 0.0, 2.0 * inverse_sqrt_six),
            ),
            (
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, 0.0, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
            ),
            (
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, -inverse_sqrt_two, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
        dtype=jnp.float64,
    )
    return tensors


def _table_i_blocks(
    direction: Float[Array, " 3"],
    values: Float[Array, " 10"],
) -> dict[tuple[int, int], Float[Array, "m1 m2"]]:
    """Evaluate the direction-cosine polynomials for all ten channels."""
    p_axes: Float[Array, "3 3"] = jnp.asarray(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        dtype=jnp.float64,
    )
    tensors: Float[Array, "5 3 3"] = _reference_d_tensors()
    p_direction: Float[Array, " 3"] = p_axes @ direction
    tensor_direction: Float[Array, " 5"] = jnp.einsum(
        "aij,i,j->a",
        tensors,
        direction,
        direction,
    )
    d_sigma: Float[Array, " 5"] = jnp.sqrt(3.0 / 2.0) * tensor_direction
    tensor_vectors: Float[Array, "5 3"] = jnp.einsum(
        "aij,j->ai",
        tensors,
        direction,
    )
    p_dot_tensor: Float[Array, "3 5"] = p_axes @ tensor_vectors.T
    pd_pi_coefficients: Float[Array, "3 5"] = jnp.sqrt(2.0) * (
        p_dot_tensor - p_direction[:, None] * tensor_direction[None, :]
    )
    d_sigma_projector: Float[Array, "5 5"] = jnp.outer(
        d_sigma,
        d_sigma,
    )
    d_pi_projector: Float[Array, "5 5"] = 2.0 * (
        tensor_vectors @ tensor_vectors.T
        - jnp.outer(tensor_direction, tensor_direction)
    )
    d_identity: Float[Array, "5 5"] = jnp.eye(5, dtype=jnp.float64)

    blocks: dict[tuple[int, int], Float[Array, "m1 m2"]] = {
        (0, 0): values[0:1, None],
        (0, 1): values[1] * p_direction[None, :],
        (0, 2): values[2] * d_sigma[None, :],
        (1, 1): (
            values[4] * jnp.eye(3, dtype=jnp.float64)
            + (values[3] - values[4]) * jnp.outer(p_direction, p_direction)
        ),
        (1, 2): (
            values[5] * jnp.outer(p_direction, d_sigma)
            + values[6] * pd_pi_coefficients
        ),
        (2, 2): (
            values[9] * d_identity
            + (values[7] - values[9]) * d_sigma_projector
            + (values[8] - values[9]) * d_pi_projector
        ),
    }
    return blocks


def _graphene_geometry() -> CrystalGeometry:
    """Construct the two-atom honeycomb geometry used by the shell gate."""
    lattice_constant: float = 2.46
    lattice: Float[Array, "3 3"] = jnp.asarray(
        (
            (lattice_constant, 0.0, 0.0),
            (
                lattice_constant / 2.0,
                lattice_constant * np.sqrt(3.0) / 2.0,
                0.0,
            ),
            (0.0, 0.0, 10.0),
        ),
        dtype=jnp.float64,
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        jnp.asarray(
            ((0.0, 0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0, 0.0)),
            dtype=jnp.float64,
        ),
        ("C", "C"),
    )
    return geometry


def _compact_spd_basis() -> OrbitalBasis:
    """Construct generic s, px, and dxy orbitals on each of two atoms."""
    basis: OrbitalBasis = make_orbital_basis(
        (0, 0, 0, 1, 1, 1),
        (1, 2, 3, 1, 2, 3),
        (0, 1, 2, 0, 1, 2),
        (0, 1, -2, 0, 1, -2),
    )
    return basis


class TestSkBlock:
    """Validate :func:`diffpes.tightb.sk_block`."""

    def test_all_table_i_channels_on_fifty_random_directions(self) -> None:
        """Match direction-cosine polynomials for all ten s/p/d channels.

        The deterministic directions are generic and the parameter values are
        distinct, preventing accidental sigma/pi/delta interchange.

        Notes
        -----
        Compare every entry of all six canonical shell-pair blocks at the
        Plan-04 G3 tolerance.
        """
        generator: np.random.Generator = np.random.default_rng(93281)
        raw: Float[NDArray, "n_direction 3"] = generator.normal(
            size=(_TABLE_DIRECTIONS, 3)
        )
        directions: Float[NDArray, "n_direction 3"] = raw / np.linalg.norm(
            raw,
            axis=1,
            keepdims=True,
        )
        values: Float[Array, " 10"] = jnp.asarray(
            (0.37, -1.1, 0.83, 2.3, -0.61, 1.7, -0.42, 3.1, -0.91, 0.28),
            dtype=jnp.float64,
        )
        channel_vectors: dict[tuple[int, int], Float[Array, " n_m"]] = {
            (0, 0): values[0:1],
            (0, 1): values[1:2],
            (0, 2): values[2:3],
            (1, 1): values[3:5],
            (1, 2): values[5:7],
            (2, 2): values[7:10],
        }

        direction: Float[NDArray, " 3"]
        for direction in directions:
            bond: Float[Array, " 3"] = jnp.asarray(
                direction,
                dtype=jnp.float64,
            )
            references: dict[tuple[int, int], Float[Array, "m1 m2"]] = (
                _table_i_blocks(bond, values)
            )
            angular_pair: tuple[int, int]
            integrals: Float[Array, " n_m"]
            for angular_pair, integrals in channel_vectors.items():
                actual: Float[Array, "m1 m2"] = sk_block(
                    angular_pair[0],
                    angular_pair[1],
                    integrals,
                    bond,
                )
                np.testing.assert_allclose(
                    actual,
                    references[angular_pair],
                    rtol=1e-12,
                    atol=2e-14,
                )

    @pytest.mark.parametrize(
        ("l1", "l2"),
        tuple((l1, l2) for l1 in range(3) for l2 in range(3)),
    )
    def test_parity_and_swapped_shell_hermiticity(
        self,
        l1: int,
        l2: int,
    ) -> None:
        """Preserve fixed-shell parity and reverse-bond Hermiticity.

        The nine shell-order pairs include separately dimensioned rectangular
        blocks.

        Notes
        -----
        Apply the radial-integral reversal convention automatically for a
        swapped shell order.
        """
        bond: Float[Array, " 3"] = jnp.asarray(
            (0.31, -0.47, 0.73),
            dtype=jnp.float64,
        )
        integrals: Float[Array, " n_m"] = jnp.arange(
            1,
            min(l1, l2) + 2,
            dtype=jnp.float64,
        )
        block: Float[Array, "m1 m2"] = sk_block(
            l1,
            l2,
            integrals,
            bond,
        )
        reversed_bond: Float[Array, "m1 m2"] = sk_block(
            l1,
            l2,
            integrals,
            -bond,
        )
        swapped: Float[Array, "m2 m1"] = sk_block(
            l2,
            l1,
            integrals,
            -bond,
        )

        np.testing.assert_allclose(
            reversed_bond,
            (-1) ** (l1 + l2) * block,
            rtol=1e-13,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            swapped.T,
            block,
            rtol=1e-13,
            atol=2e-14,
        )

    @pytest.mark.parametrize("pole", (1.0, -1.0))
    def test_transverse_gradient_is_analytic_at_bond_poles(
        self,
        pole: float,
    ) -> None:
        """Retain the nonzero s--px derivative at positive and negative z.

        The s--px Table-I element is ``x / norm(bond) * V_sp_sigma``.

        Notes
        -----
        Compare reverse-mode AD with its exact transverse pole derivative.
        """
        integral: float = 1.7
        bond: Float[Array, " 3"] = jnp.asarray(
            (0.0, 0.0, 2.0 * pole),
            dtype=jnp.float64,
        )

        def s_px(candidate: Float[Array, " 3"]) -> Float[Array, ""]:
            """Return the s--px matrix element."""
            value: Float[Array, ""] = sk_block(
                0,
                1,
                jnp.asarray((integral,), dtype=jnp.float64),
                candidate,
            )[0, 2]
            return value

        gradient: Float[Array, " 3"] = jax.grad(s_px)(bond)

        np.testing.assert_allclose(
            gradient,
            jnp.asarray((integral / 2.0, 0.0, 0.0)),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_jit_shape_dtype_and_rejections(self) -> None:
        """Compile a d--p block and reject malformed physical inputs.

        The successful path fixes the rectangular shell dimensions and
        double-precision output contract.

        Notes
        -----
        Pin the rectangular real float64 output and zero-bond diagnostic.
        """
        compiled: Callable[[int, int, Array, Array], Array] = jax.jit(
            sk_block,
            static_argnums=(0, 1),
        )
        block: Float[Array, "5 3"] = compiled(
            2,
            1,
            jnp.asarray((0.8, -0.3), dtype=jnp.float64),
            jnp.asarray((0.2, 0.4, 0.7), dtype=jnp.float64),
        )

        assert block.shape == (5, 3)
        assert block.dtype == jnp.float64
        with pytest.raises(Exception, match="bond nonzero"):
            sk_block(
                1,
                1,
                jnp.asarray((1.0, -0.2), dtype=jnp.float64),
                jnp.zeros((3,), dtype=jnp.float64),
            )
        with pytest.raises(ValueError, match="v_llm length"):
            sk_block(
                2,
                2,
                jnp.ones((2,), dtype=jnp.float64),
                jnp.ones((3,), dtype=jnp.float64),
            )


class TestNeighborShells:
    """Validate :func:`diffpes.tightb.neighbor_shells`."""

    def test_honeycomb_has_three_unique_nearest_neighbor_bonds(self) -> None:
        """Verify the three undirected A--B bonds of a honeycomb cell.

        The records retain distinct exact cells for the translated nearest
        neighbors.

        Notes
        -----
        Also derive every fractional displacement from its exact integer cell.
        """
        geometry: CrystalGeometry = _graphene_geometry()
        atom_pairs: tuple[tuple[int, int], ...]
        cells: tuple[tuple[int, int, int], ...]
        displacements: Float[Array, "3 3"]
        distances: Float[Array, " 3"]
        atom_pairs, cells, displacements, distances = neighbor_shells(
            geometry,
            1.5,
        )

        assert atom_pairs == ((0, 1), (0, 1), (0, 1))
        assert cells == ((-1, 0, 0), (0, -1, 0), (0, 0, 0))
        assert all(
            type(component) is int for cell in cells for component in cell
        )
        expected: Float[Array, "3 3"] = (
            jnp.asarray(cells, dtype=jnp.float64)
            + geometry.positions[1]
            - geometry.positions[0]
        )
        np.testing.assert_allclose(displacements, expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            distances,
            jnp.full((3,), 2.46 / np.sqrt(3.0)),
            rtol=1e-13,
            atol=0.0,
        )

    def test_fcc_primitive_cell_has_six_unique_first_shell_bonds(self) -> None:
        """Verify half of the twelve directed fcc nearest neighbors.

        Canonical ordering keeps each undirected primitive-cell bond once.

        Notes
        -----
        The builder later emits one reverse record for each of these six
        canonical undirected bonds.
        """
        lattice_constant: float = 3.6
        lattice: Float[Array, "3 3"] = (
            lattice_constant
            / 2.0
            * jnp.asarray(
                ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
                dtype=jnp.float64,
            )
        )
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice,
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("Cu",),
        )
        atom_pairs: tuple[tuple[int, int], ...]
        cells: tuple[tuple[int, int, int], ...]
        distances: Float[Array, " 6"]
        atom_pairs, cells, _, distances = neighbor_shells(
            geometry,
            float(lattice_constant / np.sqrt(2.0) + 1e-8),
        )

        assert len(atom_pairs) == 6
        assert len(cells) == 6
        np.testing.assert_allclose(
            distances,
            jnp.full((6,), lattice_constant / np.sqrt(2.0)),
            rtol=1e-13,
            atol=0.0,
        )

    def test_small_cell_search_extends_beyond_radius_two(self) -> None:
        """Find every cutoff bond in a lattice shorter than the cutoff.

        A radius-two search omits the third translated copy at 1.2 Angstrom.
        The singular-value certificate must extend the enumeration.

        Notes
        -----
        Pin exact translation metadata as well as the Cartesian distances.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((0.4, 10.0, 10.0), dtype=jnp.float64)),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        atom_pairs: tuple[tuple[int, int], ...]
        cells: tuple[tuple[int, int, int], ...]
        distances: Float[Array, " 3"]
        atom_pairs, cells, _, distances = neighbor_shells(geometry, 1.25)

        assert atom_pairs == ((0, 0), (0, 0), (0, 0))
        assert cells == ((-3, 0, 0), (-2, 0, 0), (-1, 0, 0))
        np.testing.assert_allclose(
            distances,
            jnp.asarray((1.2, 0.8, 0.4), dtype=jnp.float64),
            rtol=0.0,
            atol=1e-14,
        )

    def test_skew_lattice_search_extends_beyond_radius_two(self) -> None:
        """Retain a distant integer cell made short by lattice skew.

        The ``(-3, 3, 0)`` translation nearly cancels the first two lattice
        rows and lies inside the cutoff despite both indices exceeding two.

        Notes
        -----
        This is a counterexample to any componentwise radius chosen without
        the lattice's smallest singular value.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (0.99, 0.1, 0.0),
                    (0.0, 0.0, 10.0),
                ),
                dtype=jnp.float64,
            ),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        cells: tuple[tuple[int, int, int], ...]
        distances: Float[Array, " n_bond"]
        _, cells, _, distances = neighbor_shells(geometry, 0.31)

        target_index: int = cells.index((-3, 3, 0))
        np.testing.assert_allclose(
            distances[target_index],
            np.sqrt(0.03**2 + 0.3**2),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_undersized_explicit_radius_is_rejected(self) -> None:
        """Reject a caller override that cannot certify completeness.

        The small-cell geometry needs translations beyond the legacy default.

        Notes
        -----
        Require the diagnostic to expose the computed minimum radius.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((0.4, 10.0, 10.0), dtype=jnp.float64)),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )

        with pytest.raises(
            ValueError,
            match=r"supercell_radius=2 is incomplete; certified minimum is 4",
        ):
            neighbor_shells(geometry, 1.25, supercell_radius=2)


class TestBuildSkModel:
    """Validate :func:`diffpes.tightb.build_sk_model`."""

    def test_distance_shell_keys_select_distinct_integrals(self) -> None:
        """Verify independent first- and second-neighbor chain hoppings.

        Distinct distance-shell keys must select their matching amplitudes.

        Notes
        -----
        Pin the one-based ``@N`` key grammar and setup-time distance binning.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((1.0, 10.0, 10.0), dtype=jnp.float64)),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0,),
            (1,),
            (0,),
            (0,),
            labels=("X_s",),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-1.0, -0.3), dtype=jnp.float64),
            ("X-X@1:ss_sigma", "X-X@2:ss_sigma"),
        )
        model: TBModel = build_sk_model(
            geometry,
            basis,
            params,
            jnp.zeros((1,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            (-1,),
            2.1,
        )
        by_cell: dict[tuple[int, int, int], complex] = dict(
            zip(
                model.hopping_cells,
                np.asarray(model.hopping_amplitudes),
                strict=True,
            )
        )

        assert by_cell[(-1, 0, 0)] == -1.0
        assert by_cell[(1, 0, 0)] == -1.0
        assert by_cell[(-2, 0, 0)] == -0.3
        assert by_cell[(2, 0, 0)] == -0.3

    def test_graphene_model_is_closed_and_uses_exact_cells(self) -> None:
        """Build the three-bond pz graphene model from one pi integral.

        The builder must emit both orientations for every nearest-neighbor
        bond.

        Notes
        -----
        Require the six directed records and their exact conjugate closure.
        """
        geometry: CrystalGeometry = _graphene_geometry()
        basis: OrbitalBasis = make_orbital_basis(
            (0, 1),
            (2, 2),
            (1, 1),
            (0, 0),
            labels=("A_pz", "B_pz"),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-2.7,), dtype=jnp.float64),
            ("C-C:pp_pi",),
        )
        model: TBModel = build_sk_model(
            geometry,
            basis,
            params,
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )

        assert model.hopping_pairs == (
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
        )
        assert model.hopping_cells == (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        np.testing.assert_allclose(
            model.hopping_amplitudes,
            -2.7,
            rtol=0.0,
            atol=1e-14,
        )

    def test_jit_rejects_uncertified_traced_geometry(self) -> None:
        """Reject neighbor discovery when compilation hides its geometry.

        The compiled lattice lacks a concrete primal, so the host cannot
        freeze a singular-value topology certificate.

        Notes
        -----
        Static-geometry compiled rebuilds remain supported because host setup
        selects their topology before tracing.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.eye(3, dtype=jnp.float64),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0,),
            (1,),
            (0,),
            (0,),
            labels=("X_s",),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-1.0,), dtype=jnp.float64),
            ("X-X:ss_sigma",),
        )

        def hopping(lattice: Float[Array, "3 3"]) -> Float[Array, " n_hop"]:
            """Build hopping amplitudes from a traced lattice."""
            candidate: CrystalGeometry = eqx.tree_at(
                lambda item: item.lattice,
                geometry,
                lattice,
            )
            model: TBModel = build_sk_model(
                candidate,
                basis,
                params,
                jnp.zeros((1,), dtype=jnp.float64),
                jnp.zeros((0,), dtype=jnp.float64),
                (-1,),
                1.1,
            )
            return jnp.real(model.hopping_amplitudes)

        with pytest.raises(
            ValueError,
            match="cannot certify neighbor topology from fully traced geometry",
        ):
            jax.jit(hopping)(geometry.lattice)

    @pytest.mark.parametrize("pole", (1.0, -1.0))
    def test_position_gradient_flows_through_frozen_topology(
        self,
        pole: float,
    ) -> None:
        """Differentiate an assembled s--px hopping at both bond poles.

        The setup selects the neighbor tuple from the concrete AD primal while
        the bond direction remains traced.

        Notes
        -----
        Compare the fractional-position derivative with the analytic Cartesian
        derivative times the lattice-vector length.
        """
        lattice: Float[Array, "3 3"] = jnp.diag(
            jnp.asarray((10.0, 10.0, 10.0), dtype=jnp.float64)
        )
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice,
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.2 * pole)),
                dtype=jnp.float64,
            ),
            ("A", "B"),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0, 1),
            (1, 2),
            (0, 1),
            (0, 1),
            labels=("A_s", "B_px"),
        )
        integral: float = 1.4
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((integral,), dtype=jnp.float64),
            ("A-B:sp_sigma",),
        )

        def hopping(
            positions: Float[Array, "2 3"],
        ) -> Float[Array, ""]:
            """Return the forward s--px hopping on frozen topology."""
            candidate: CrystalGeometry = eqx.tree_at(
                lambda item: item.positions,
                geometry,
                positions,
            )
            model: TBModel = build_sk_model(
                candidate,
                basis,
                params,
                jnp.zeros((2,), dtype=jnp.float64),
                jnp.zeros((0,), dtype=jnp.float64),
                (-1, -1),
                3.0,
            )
            value: Float[Array, ""] = jnp.real(model.hopping_amplitudes[0])
            return value

        gradient: Float[Array, "2 3"] = jax.grad(hopping)(geometry.positions)

        np.testing.assert_allclose(
            gradient[:, 0],
            jnp.asarray((-5.0 * integral, 5.0 * integral)),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            gradient[:, 1:],
            0.0,
            rtol=0.0,
            atol=1e-13,
        )

    @pytest.mark.rss_limit_mb(900)
    def test_every_integral_has_fd_correct_band_spectral_gradient(
        self,
    ) -> None:
        """Differentiate a band spectral loss with respect to all ten values.

        Generic s, px, and dxy orbitals on two atoms exercise every
        fundamental integral without a needlessly large eigensystem. Squared
        eigenvalues form a gauge-invariant band loss with nonzero sensitivity
        to every channel.

        Notes
        -----
        Apply the program-wide f64 finite-difference harness in forward and
        reverse mode, then enforce the zero-gradient tripwire.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((5.0, 6.0, 7.0), dtype=jnp.float64)),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.21, 0.17, 0.13)),
                dtype=jnp.float64,
            ),
            ("X", "X"),
        )
        basis: OrbitalBasis = _compact_spd_basis()
        initial: Float[Array, " 10"] = jnp.asarray(
            (-0.8, 1.1, -0.7, 1.5, -0.4, 0.9, -0.3, 1.2, -0.6, 0.2),
            dtype=jnp.float64,
        )
        onsite: Float[Array, " 6"] = jnp.linspace(
            -0.35,
            0.42,
            6,
            dtype=jnp.float64,
        )
        kpoint: Float[Array, " 3"] = jnp.asarray(
            (0.137, -0.219, 0.083),
            dtype=jnp.float64,
        )

        def spectral_loss(values: Float[Array, " 10"]) -> Float[Array, ""]:
            """Return the sum of squared tight-binding band energies."""
            params: SlaterKosterParams = make_slater_koster_params(
                values,
                _ALL_SK_KEYS,
            )
            model: TBModel = build_sk_model(
                geometry,
                basis,
                params,
                onsite,
                jnp.zeros((0,), dtype=jnp.float64),
                (-1,) * 6,
                2.0,
            )
            eigenvalues: Float[Array, " 6"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            loss: Float[Array, ""] = jnp.sum(eigenvalues**2)
            return loss

        assert_grad_matches_fd(spectral_loss, initial)
        assert_nonzero_grad(spectral_loss, initial, elementwise=True)
