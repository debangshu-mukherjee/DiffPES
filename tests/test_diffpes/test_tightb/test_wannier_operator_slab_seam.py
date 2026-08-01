"""Certify Plan-05 exact gathers and the Wannier-operator slab seam.

The tests exercise Plan-05 numerical and structural contracts.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any
from jaxtyping import Int
from numpy.typing import NDArray

from diffpes.tightb import (
    diagonalize_tb,
    layer_resolved_group_traces,
)
from diffpes.tightb.slab import (
    _propagate_hoppings,
    gen_slab,
    gen_slab_with_operators,
    rotate_tb_model,
)
from diffpes.types import (
    TBModel,
    WannierOperatorData,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
    make_wannier_operator_data,
)


def _exact_inverse(
    rows: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ],
) -> Int[NDArray, "3 3"]:
    """Return a unimodular integer inverse from its exact adjugate."""
    matrix: Int[NDArray, "3 3"] = np.asarray(rows, dtype=np.int64)
    determinant: int = int(
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )
    cofactors: Int[NDArray, "3 3"] = np.asarray(
        (
            (
                matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1],
                -(matrix[1, 0] * matrix[2, 2]) + matrix[1, 2] * matrix[2, 0],
                matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0],
            ),
            (
                -(matrix[0, 1] * matrix[2, 2]) + matrix[0, 2] * matrix[2, 1],
                matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0],
                -(matrix[0, 0] * matrix[2, 1]) + matrix[0, 1] * matrix[2, 0],
            ),
            (
                matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1],
                -(matrix[0, 0] * matrix[1, 2]) + matrix[0, 2] * matrix[1, 0],
                matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0],
            ),
        ),
        dtype=np.int64,
    )
    assert abs(determinant) == 1
    return cofactors.T // determinant


def _long_range_model(maximum_range: int) -> TBModel:
    """Build a one-orbital model with registered normal ranges 1..R."""
    forward_cells: tuple[tuple[int, int, int], ...] = tuple(
        (distance % 2, -(distance // 2), distance)
        for distance in range(1, maximum_range + 1)
    )
    reverse_cells: tuple[tuple[int, int, int], ...] = tuple(
        tuple(-component for component in cell) for cell in forward_cells
    )
    forward: jax.Array = jnp.asarray(
        [
            -0.13 * distance + 0.017j * distance
            for distance in range(1, maximum_range + 1)
        ],
        dtype=jnp.complex128,
    )
    return make_tb_model(
        hopping_amplitudes=jnp.concatenate((forward, jnp.conj(forward))),
        onsite_energies=jnp.asarray((0.23,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        ),
        basis=make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
            labels=("s",),
        ),
        hopping_pairs=((0, 0),) * (2 * maximum_range),
        hopping_cells=forward_cells + reverse_cells,
        shell_index=(-1,),
    )


def _p_shell_operator_fixture() -> tuple[TBModel, WannierOperatorData]:
    """Build a complete p shell with generic complex position blocks."""
    basis: Any
    geometry: Any
    geometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
        labels=("py", "pz", "px"),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.asarray((0.1, -0.2, 0.4), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(-1, -1, -1),
    )
    centres: jax.Array = jnp.asarray(
        (
            (0.12, -0.11, -0.21),
            (0.24, -0.16, -0.27),
            (0.31, -0.23, -0.34),
        ),
        dtype=jnp.float64,
    )
    cells: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (1, 2, -1),
        (-1, -2, 1),
    )
    zero_seed: jax.Array = jnp.asarray(
        np.arange(27).reshape(3, 3, 3) / 37.0
        + 1j * np.arange(27, 54).reshape(3, 3, 3) / 53.0,
        dtype=jnp.complex128,
    )
    zero: jax.Array = 0.5 * (
        zero_seed + jnp.swapaxes(jnp.conj(zero_seed), 0, 1)
    )
    diagonal: jax.Array = jnp.arange(3)
    zero = zero.at[diagonal, diagonal].set(centres)
    forward: jax.Array = jnp.asarray(
        np.arange(54, 81).reshape(3, 3, 3) / 29.0
        + 1j * np.arange(81, 108).reshape(3, 3, 3) / 31.0,
        dtype=jnp.complex128,
    )
    reverse: jax.Array = jnp.swapaxes(jnp.conj(forward), 0, 1)
    operator_data: WannierOperatorData = make_wannier_operator_data(
        position_matrices=jnp.stack((zero, forward, reverse)),
        centres_cart=centres,
        cells=cells,
        degeneracies=(3, 5, 7),
        spin_layout="block_down_up",
        source_format="tb",
    )
    return model, operator_data


def _graphene_model() -> TBModel:
    """Return nearest-neighbour graphene for the N=30 edge-state gate."""
    basis: Any
    geometry: Any
    root_three: float = math.sqrt(3.0)
    geometry = make_crystal_geometry(
        lattice=jnp.asarray(
            (
                (root_three / 2.0, 0.0, 1.5),
                (-root_three / 2.0, 0.0, 1.5),
                (0.0, -10.0, 0.0),
            ),
            dtype=jnp.float64,
        ),
        positions=jnp.asarray(
            ((0.0, 0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0, 0.0)),
            dtype=jnp.float64,
        ),
        species=("C", "C"),
    )
    basis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(2, 2),
        l=(0, 0),
        m=(0, 0),
        labels=("pz-A", "pz-B"),
    )
    forward_cells: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
    )
    reverse_cells: tuple[tuple[int, int, int], ...] = tuple(
        tuple(-component for component in cell) for cell in forward_cells
    )
    return make_tb_model(
        hopping_amplitudes=jnp.full((6,), -1.0, dtype=jnp.complex128),
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1),) * 3 + ((1, 0),) * 3,
        hopping_cells=forward_cells + reverse_cells,
        shell_index=(-1, -1),
    )


class TestExactLongRangeGather:
    """Certify the complete G10 Hamiltonian and hr-sidecar bookkeeping."""

    @pytest.mark.parametrize(
        ("n_layers", "maximum_range"),
        ((6, 2), (13, 4), (20, 5)),
    )
    def test_exact_amplitudes_cells_and_gather_bounds(
        self,
        n_layers: int,
        maximum_range: int,
    ) -> None:
        """Match an independent integer enumeration for 6--20 layers.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        amplitudes: Any
        bulk_cell: Any
        cells: Any
        frame: Any
        gather: Any
        hopping: Any
        pairs: Any
        slab: Any
        source_layer: Any
        spec: Any
        bulk: TBModel = _long_range_model(maximum_range)
        slab, spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=float(n_layers - 1),
            vacuum_ang=7.0,
        )
        rotated: TBModel = rotate_tb_model(
            bulk,
            spec.surface_cell.rotation,
        )
        amplitudes, pairs, cells, gather = _propagate_hoppings(rotated, spec)
        frame = (
            *spec.surface_cell.in_plane_coeffs,
            spec.surface_cell.stacking_coeffs,
        )
        inverse: Int[NDArray, "3 3"] = _exact_inverse(frame)
        expected_pairs: list[tuple[int, int]] = []
        expected_cells: list[tuple[int, int, int]] = []
        expected_gather: list[int] = []
        for source_layer in range(n_layers):
            for hopping, bulk_cell in enumerate(rotated.hopping_cells):
                transformed: Int[NDArray, " 3"] = (
                    np.asarray(bulk_cell, dtype=np.int64) @ inverse
                )
                target_layer: int = source_layer + int(transformed[2])
                if not 0 <= target_layer < n_layers:
                    continue
                expected_pairs.append((source_layer, target_layer))
                expected_cells.append(
                    (int(transformed[0]), int(transformed[1]), 0)
                )
                expected_gather.append(hopping)

        assert spec.n_layers == n_layers
        assert pairs == tuple(expected_pairs)
        assert cells == tuple(expected_cells)
        assert gather == tuple(expected_gather)
        assert all(0 <= index < len(rotated.hopping_cells) for index in gather)
        assert jnp.array_equal(
            amplitudes,
            rotated.hopping_amplitudes[jnp.asarray(gather)],
        )
        assert slab.hopping_cells == cells
        assert jnp.array_equal(slab.hopping_amplitudes, amplitudes)
        assert (
            max(abs(target - source) for source, target in pairs)
            == maximum_range
        )

    def test_hr_cells_are_exact_and_serialized_degeneracies_are_consumed(
        self,
    ) -> None:
        """Verify remapping replaces already-applied WS weights.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        propagated: Any
        spec: Any
        maximum_range: int = 4
        bulk: TBModel = _long_range_model(maximum_range)
        operator_data: WannierOperatorData = make_wannier_operator_data(
            position_matrices=None,
            centres_cart=jnp.zeros((1, 3), dtype=jnp.float64),
            cells=bulk.hopping_cells,
            degeneracies=tuple(range(2, 2 + len(bulk.hopping_cells))),
            spin_layout="block_down_up",
            source_format="hr",
        )
        _, spec, propagated = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(0, 0, 1),
            thickness_ang=5.0,
            vacuum_ang=7.0,
        )
        inverse: Int[NDArray, "3 3"] = _exact_inverse(
            (
                *spec.surface_cell.in_plane_coeffs,
                spec.surface_cell.stacking_coeffs,
            )
        )
        expected_cells: tuple[tuple[int, int, int], ...] = tuple(
            sorted(
                {
                    (
                        int(transformed[0]),
                        int(transformed[1]),
                        0,
                    )
                    for cell in bulk.hopping_cells
                    for transformed in (
                        np.asarray(cell, dtype=np.int64) @ inverse,
                    )
                }
            )
        )

        assert propagated.position_matrices is None
        assert propagated.cells == expected_cells
        assert propagated.degeneracies == (1,) * len(expected_cells)

    def test_rejects_unpaired_operator_cell_grid(self) -> None:
        """Reject a sidecar that omits a Hamiltonian translation cell.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bulk: TBModel = _long_range_model(2)
        operator_data: WannierOperatorData = make_wannier_operator_data(
            position_matrices=None,
            centres_cart=jnp.zeros((1, 3), dtype=jnp.float64),
            cells=bulk.hopping_cells[:-1],
            degeneracies=(1,) * (len(bulk.hopping_cells) - 1),
            spin_layout="block_down_up",
            source_format="hr",
        )

        with pytest.raises(ValueError, match="must cover"):
            gen_slab_with_operators(
                bulk,
                operator_data,
                miller=(0, 0, 1),
                thickness_ang=5.0,
                vacuum_ang=7.0,
            )


class TestCompleteShellOperatorPropagation:
    """Certify Wigner, Cartesian, origin, cell, centre, and depth laws."""

    def test_generic_complex_p_shell_matches_analytic_transformation(
        self,
    ) -> None:
        """Match D O D-dagger and Cartesian rotation for every slab block.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bulk: Any
        bulk_cell: Any
        cell_index: Any
        operator_data: Any
        propagated: Any
        slab: Any
        source: Any
        source_layer: Any
        spec: Any
        target: Any
        bulk, operator_data = _p_shell_operator_fixture()
        slab, spec, propagated = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(1, 0, 0),
            thickness_ang=5.0,
            vacuum_ang=7.0,
        )
        assert operator_data.position_matrices is not None
        assert propagated.position_matrices is not None
        rotation: jax.Array = spec.surface_cell.rotation
        permutation: jax.Array = jnp.asarray(
            ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            dtype=jnp.float64,
        )
        representation: jax.Array = permutation @ rotation @ permutation.T
        orbital_rotated: jax.Array = jnp.einsum(
            "ai,rijc,bj->rabc",
            representation,
            operator_data.position_matrices,
            representation.conj(),
        )
        analytically_rotated: jax.Array = jnp.einsum(
            "rijc,ac->rija",
            orbital_rotated,
            rotation,
        )
        inverse: Int[NDArray, "3 3"] = _exact_inverse(
            (
                *spec.surface_cell.in_plane_coeffs,
                spec.surface_cell.stacking_coeffs,
            )
        )
        cell_lookup: dict[tuple[int, int, int], int] = {
            cell: index for index, cell in enumerate(propagated.cells)
        }
        expected: jax.Array = jnp.zeros_like(propagated.position_matrices)
        for source_layer in range(spec.n_layers):
            for cell_index, bulk_cell in enumerate(operator_data.cells):
                transformed: Int[NDArray, " 3"] = (
                    np.asarray(bulk_cell, dtype=np.int64) @ inverse
                )
                target_layer: int = source_layer + int(transformed[2])
                if not 0 <= target_layer < spec.n_layers:
                    continue
                slab_cell: tuple[int, int, int] = (
                    int(transformed[0]),
                    int(transformed[1]),
                    0,
                )
                for source in range(3):
                    for target in range(3):
                        expected = expected.at[
                            cell_lookup[slab_cell],
                            3 * source_layer + source,
                            3 * target_layer + target,
                        ].add(analytically_rotated[cell_index, source, target])
        zero_index: int = cell_lookup[(0, 0, 0)]
        slab_diagonal: jax.Array = jnp.arange(3 * spec.n_layers)
        expected = expected.at[
            zero_index,
            slab_diagonal,
            slab_diagonal,
        ].set(propagated.centres_cart)

        assert spec.n_layers == 6
        assert propagated.cells == tuple(sorted(cell_lookup))
        assert jnp.allclose(
            propagated.position_matrices,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        assert slab.orbital_positions is not None
        model_centres: jax.Array = (
            slab.orbital_positions @ slab.geometry.lattice
        )
        assert jnp.allclose(
            model_centres,
            propagated.centres_cart,
            rtol=1e-12,
            atol=1e-12,
        )
        assert slab.depths is not None
        assert jnp.allclose(
            slab.depths,
            jnp.max(propagated.centres_cart[:, 2])
            - propagated.centres_cart[:, 2],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_origin_shift_changes_only_zero_cell_identity_block(
        self,
    ) -> None:
        """Apply r-prime = r + delta and recover the covariant slab law.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        base: Any
        base_model: Any
        base_spec: Any
        bulk: Any
        operator_data: Any
        shifted: Any
        shifted_model: Any
        shifted_spec: Any
        bulk, operator_data = _p_shell_operator_fixture()
        assert operator_data.position_matrices is not None
        delta: jax.Array = jnp.asarray(
            (0.037, -0.043, 0.071),
            dtype=jnp.float64,
        )
        zero_index: int = operator_data.cells.index((0, 0, 0))
        diagonal: jax.Array = jnp.arange(3)
        shifted_matrices: jax.Array = operator_data.position_matrices.at[
            zero_index,
            diagonal,
            diagonal,
        ].add(delta)
        shifted_data: WannierOperatorData = make_wannier_operator_data(
            position_matrices=shifted_matrices,
            centres_cart=operator_data.centres_cart + delta,
            cells=operator_data.cells,
            degeneracies=operator_data.degeneracies,
            spin_layout=operator_data.spin_layout,
            source_format=operator_data.source_format,
        )
        base_model, base_spec, base = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(1, 0, 0),
            thickness_ang=5.0,
            vacuum_ang=7.0,
        )
        shifted_model, shifted_spec, shifted = gen_slab_with_operators(
            bulk,
            shifted_data,
            miller=(1, 0, 0),
            thickness_ang=5.0,
            vacuum_ang=7.0,
        )
        assert base.position_matrices is not None
        assert shifted.position_matrices is not None
        rotated_delta: jax.Array = delta @ base_spec.surface_cell.rotation.T
        expected_difference: jax.Array = jnp.zeros_like(base.position_matrices)
        slab_zero: int = base.cells.index((0, 0, 0))
        slab_diagonal: jax.Array = jnp.arange(base.centres_cart.shape[0])
        expected_difference = expected_difference.at[
            slab_zero,
            slab_diagonal,
            slab_diagonal,
        ].set(rotated_delta)

        assert shifted_spec == base_spec
        assert shifted.cells == base.cells
        assert jnp.allclose(
            shifted.centres_cart - base.centres_cart,
            rotated_delta,
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.allclose(
            shifted.position_matrices - base.position_matrices,
            expected_difference,
            rtol=1e-12,
            atol=1e-12,
        )
        assert base_model.depths is not None
        assert shifted_model.depths is not None
        assert jnp.allclose(
            shifted_model.depths,
            base_model.depths,
            rtol=1e-12,
            atol=1e-12,
        )


class TestZigzagEdgeSurfaceLocalization:
    """Validate the Plan-05 G9 C component on the N=30 zero-mode group."""

    def test_zero_mode_group_is_exactly_edge_localized(self) -> None:
        """Use a complete degenerate trace and both-edge probability.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: Any
        slab: Any
        spec: Any
        n_chains: int = 30
        slab, spec = gen_slab(
            _graphene_model(),
            miller=(1, 0, 0),
            thickness_ang=(n_chains - 1) * 1.5,
            vacuum_ang=8.0,
        )
        bands = diagonalize_tb(
            slab,
            jnp.asarray(((0.5, 0.0, 0.0),), dtype=jnp.float64),
        )
        zero_group: tuple[int, int] = (n_chains - 1, n_chains)
        surface_trace: jax.Array = layer_resolved_group_traces(
            bands,
            (zero_group,),
            0.1,
        )[0, 0]
        assert slab.depths is not None
        zero_vectors: jax.Array = bands.eigenvectors[0, list(zero_group)]
        probabilities: jax.Array = jnp.abs(zero_vectors) ** 2
        edge_mask: jax.Array = (slab.depths == 0.0) | (
            slab.depths == jnp.max(slab.depths)
        )

        assert spec.n_layers == n_chains
        assert jnp.allclose(
            bands.eigenvalues[0, list(zero_group)],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        assert surface_trace == pytest.approx(1.0, abs=1e-12)
        assert jnp.sum(probabilities[:, edge_mask]) == pytest.approx(
            2.0,
            abs=1e-12,
        )
