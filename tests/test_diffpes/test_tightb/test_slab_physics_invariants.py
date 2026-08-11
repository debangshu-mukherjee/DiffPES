"""Validate the classical physics invariants for tight-binding slabs.

The tests exercise slab numerical and structural contracts.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, List, Tuple
from jaxtyping import Array, Complex128, Float64, Int64
from numpy.typing import NDArray

from diffpes.tightb import (
    bloch_hamiltonian,
    find_surface_cell,
    gen_slab,
    validate_open_surface_adjacency,
)
from diffpes.types import (
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_surface_cell,
    make_tb_model,
)


def _graphene_model(hopping: float = -1.0) -> TBModel:
    """PRIVATE: Return nearest-neighbour graphene in a three-dimensional
    carrier.

    Parameters
    ----------
    hopping : float
        Nearest-neighbor hopping amplitude in eV.

    Returns
    -------
    model : TBModel
        Two-atom honeycomb model with three forward bonds and their
        reverse partners, all at the same hopping, and zero onsite
        energies.

    Notes
    -----
    The lattice places the honeycomb plane in x--z with bond length one
    and puts the 10 Angstrom vacuum axis along y. A (100) cut of this
    orientation produces the zigzag nanoribbon for the Nakada edge-state
    check.
    """
    basis: Any
    geometry: Any
    root_three: float = math.sqrt(3.0)
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        (
            (root_three / 2.0, 0.0, 1.5),
            (-root_three / 2.0, 0.0, 1.5),
            (0.0, -10.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    geometry = make_crystal_geometry(
        lattice=lattice,
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
    forward_cells: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
    )
    reverse_cells: Tuple[Tuple[int, int, int], ...] = tuple(
        (-cell[0], -cell[1], -cell[2]) for cell in forward_cells
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.full(
            (6,),
            hopping,
            dtype=jnp.complex128,
        ),
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1),) * 3 + ((1, 0),) * 3,
        hopping_cells=forward_cells + reverse_cells,
        shell_index=(-1, -1),
    )
    return model


def _dense_block_records(
    cell: Tuple[int, int, int],
    block: Complex128[NDArray, "n_orb n_orb"],
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int, int]],
    List[complex],
]:
    """PRIVATE: Flatten one dense real-space block into model records.

    Parameters
    ----------
    cell : Tuple[int, int, int]
        Integer lattice cell shared by every record of the block.
    block : Complex128[NDArray, "n_orb n_orb"]
        Dense hopping block in eV.

    Returns
    -------
    records : Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]],
        List[complex]]
        Parallel lists of orbital pairs, repeated cells, and complex
        amplitudes, one entry per block element in row-major order.

    Notes
    -----
    The builder consumes flat parallel record lists, so this expansion
    lets tests state models as dense matrices per cell.
    """
    column: Any
    row: Any
    pairs: List[Tuple[int, int]] = []
    cells: List[Tuple[int, int, int]] = []
    amplitudes: List[complex] = []
    for row in range(block.shape[0]):
        for column in range(block.shape[1]):
            pairs.append((row, column))
            cells.append(cell)
            amplitudes.append(complex(block[row, column]))
    records: Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int, int]],
        List[complex],
    ] = (pairs, cells, amplitudes)
    return records


def _inversion_bulk_model() -> Tuple[TBModel, Int64[NDArray, " n_orb"]]:
    """PRIVATE: Build a generic-complex model with a nontrivial inversion
    action.

    Returns
    -------
    model_and_permutation : Tuple[TBModel, Int64[NDArray, " n_orb"]]
        A four-atom model and the orbital permutation ``(2, 3, 0, 1)``
        that represents the inversion.

    Notes
    -----
    Starts from generic complex trial matrices and symmetrizes them
    with the permutation representation ``P``. The x-direction block
    becomes ``(T + P T^dagger P) / 2`` and the onsite block becomes
    ``(T + P T P) / 2``. The model is therefore exactly inversion
    symmetric but otherwise generic. Atom positions come in two pairs
    that map onto each other through the inversion centre. The z
    hoppings are a scalar ``-0.4`` eV identity. The off-diagonal
    onsite entries enter as home-cell hopping records.
    """
    basis: Any
    block: Any
    block_amplitudes: Any
    block_cells: Any
    block_pairs: Any
    cell: Any
    column: Any
    geometry: Any
    row: Any
    local_permutation: Int64[NDArray, " 4"] = np.asarray((2, 3, 0, 1))
    inversion: Float64[NDArray, "4 4"] = np.eye(4)[local_permutation]
    trial: Complex128[NDArray, "4 4"] = np.asarray(
        (
            (0.2 + 0.3j, 0.1 - 0.4j, 0.2 + 0.1j, -0.3j),
            (0.4 + 0.2j, -0.1 + 0.5j, 0.7 - 0.2j, 0.2 + 0.6j),
            (-0.2 + 0.8j, 0.3 - 0.2j, 0.1 - 0.6j, 0.4 + 0.1j),
            (0.5 - 0.2j, 0.6 + 0.7j, -0.1 + 0.2j, -0.2 - 0.3j),
        ),
        dtype=np.complex128,
    )
    x_block: Complex128[NDArray, "4 4"] = (
        trial + inversion @ trial.conj().T @ inversion
    ) / 2.0
    onsite_trial: Complex128[NDArray, "4 4"] = np.asarray(
        (
            (0.2, 0.1 + 0.2j, 0.3, 0.4j),
            (0.1 - 0.2j, -0.4, 0.2 + 0.1j, 0.5),
            (0.3, 0.2 - 0.1j, 0.7, -0.1 + 0.3j),
            (-0.4j, 0.5, -0.1 - 0.3j, 0.1),
        ),
        dtype=np.complex128,
    )
    onsite_block: Complex128[NDArray, "4 4"] = (
        onsite_trial + inversion @ onsite_trial @ inversion
    ) / 2.0
    pairs: List[Tuple[int, int]] = []
    cells: List[Tuple[int, int, int]] = []
    amplitudes: List[complex] = []
    for cell, block in (
        ((1, 0, 0), x_block),
        ((-1, 0, 0), x_block.conj().T),
        ((0, 0, 1), -0.4 * np.eye(4)),
        ((0, 0, -1), -0.4 * np.eye(4)),
    ):
        block_pairs, block_cells, block_amplitudes = _dense_block_records(
            cell,
            block,
        )
        pairs.extend(block_pairs)
        cells.extend(block_cells)
        amplitudes.extend(block_amplitudes)
    onsite_energies: Float64[NDArray, " 4"] = np.real(np.diag(onsite_block))
    for row in range(4):
        for column in range(4):
            if row != column:
                pairs.append((row, column))
                cells.append((0, 0, 0))
                amplitudes.append(complex(onsite_block[row, column]))
    geometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.asarray(
            (
                (0.20, 0.10, 0.0),
                (0.35, 0.30, 0.0),
                (0.80, 0.90, 0.0),
                (0.65, 0.70, 0.0),
            ),
            dtype=jnp.float64,
        ),
        species=("X",) * 4,
    )
    basis = make_orbital_basis(
        atom_indices=(0, 1, 2, 3),
        n=(1,) * 4,
        l=(0,) * 4,
        m=(0,) * 4,
        labels=("a", "b", "c", "d"),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray(amplitudes, dtype=jnp.complex128),
        onsite_energies=jnp.asarray(onsite_energies, dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(pairs),
        hopping_cells=tuple(cells),
        shell_index=(-1,) * 4,
    )
    result: Tuple[TBModel, Int64[NDArray, " n_orb"]] = (
        model,
        local_permutation,
    )
    return result


def _assert_inversion_witness(
    model: TBModel,
    permutation: Int64[NDArray, " n_orb"],
    centre: Float64[NDArray, " 3"],
) -> None:
    """PRIVATE: Validate positions, depths, signatures, and Hamiltonian
    covariance.

    Parameters
    ----------
    model : TBModel
        Slab model that claims the inversion symmetry.
    permutation : Int64[NDArray, " n_orb"]
        Orbital permutation that represents the inversion.
    centre : Float64[NDArray, " 3"]
        Fractional inversion centre.

    Notes
    -----
    Asserts four independent witnesses. The permutation is a bijection
    whose atom images match ``2*centre - position`` modulo in-plane
    lattice vectors. Quantum-number signatures and species stay fixed
    under the permutation. Permuted depths in Angstrom mirror onto
    ``max(depth) - depth``. At two generic k-points the permuted Bloch
    Hamiltonian equals the Hamiltonian at ``-k`` within 1e-12.
    Together these witnesses certify that the slab inherits the bulk
    inversion.
    """
    kpoint: Any
    assert model.depths is not None
    np.testing.assert_array_equal(
        np.sort(permutation),
        np.arange(permutation.size),
    )
    positions: Float64[NDArray, "n_atom 3"] = np.asarray(
        model.geometry.positions
    )
    image_residual: Float64[NDArray, "n_atom 3"] = positions[permutation] - (
        2.0 * centre - positions
    )
    image_residual[:, :2] -= np.rint(image_residual[:, :2])
    np.testing.assert_allclose(image_residual, 0.0, atol=1e-12, rtol=0.0)

    signatures: Tuple[Tuple[int, int, int], ...] = tuple(
        zip(model.basis.n, model.basis.l, model.basis.m, strict=True)
    )
    assert tuple(signatures[index] for index in permutation) == signatures
    orbital_species: Tuple[str, ...] = tuple(
        model.geometry.species[atom] for atom in model.basis.atom_indices
    )
    assert tuple(orbital_species[index] for index in permutation) == (
        orbital_species
    )
    depths: Float64[NDArray, " n_orb"] = np.asarray(model.depths)
    np.testing.assert_allclose(
        depths[permutation],
        np.max(depths) - depths,
        atol=1e-12,
        rtol=1e-12,
    )
    for kpoint in (
        jnp.asarray((0.17, -0.23, 0.0), dtype=jnp.float64),
        jnp.asarray((0.31, 0.14, 0.0), dtype=jnp.float64),
    ):
        hamiltonian: Complex128[NDArray, "n_orb n_orb"] = np.asarray(
            bloch_hamiltonian(model, kpoint)
        )
        inverted: Complex128[NDArray, "n_orb n_orb"] = np.asarray(
            bloch_hamiltonian(model, -kpoint)
        )
        np.testing.assert_allclose(
            hamiltonian[np.ix_(permutation, permutation)],
            inverted,
            atol=1e-12,
            rtol=1e-12,
        )


def _oblique_long_range_model() -> TBModel:
    """PRIVATE: Return an oblique one-orbital model with a long in-plane
    bond.

    Returns
    -------
    model : TBModel
        One-site fully oblique model whose only conjugate hopping pair
        of ``-0.7`` eV spans the distant cell ``(7, -5, 1)``.

    Notes
    -----
    The large integer cell offset makes the bond reach across many
    surface cells after extrusion. This stresses the in-plane
    reindexing and normal-range bookkeeping of the slab generator on a
    lattice without any orthogonal axis.
    """
    basis: Any
    geometry: Any
    geometry = make_crystal_geometry(
        lattice=jnp.asarray(
            ((2.0, 0.0, 0.2), (0.4, 1.7, 0.1), (0.3, 0.2, 1.1)),
            dtype=jnp.float64,
        ),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray((-0.7, -0.7), dtype=jnp.complex128),
        onsite_energies=jnp.asarray((0.2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((7, -5, 1), (-7, 5, -1)),
        shell_index=(-1,),
    )
    return model


class TestGrapheneAnalyticInvariants:
    """Verify the full graphene-zigzag-dispersion Nakada nanoribbon check.

    The cases compare zigzag zero modes and armchair levels with analytic
    nanoribbon results.
    """

    def test_zigzag_zero_mode_and_finite_width_bound(self) -> None:
        """Resolve the N=30 flat edge band including the exact pi limit.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        ka: Any
        slab: Any
        spec: Any
        n_chains: int = 30
        slab, spec = gen_slab(
            _graphene_model(),
            miller=(1, 0, 0),
            thickness_ang=(n_chains - 1) * 1.5,
            vacuum_ang=8.0,
        )

        assert spec.n_layers == n_chains
        for ka in (0.70 * math.pi, 0.75 * math.pi, 0.90 * math.pi):
            kpoint: Float64[Array, " 3"] = jnp.asarray(
                (ka / (2.0 * math.pi), 0.0, 0.0),
                dtype=jnp.float64,
            )
            energies: Float64[Array, " nband"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(slab, kpoint)
            )
            edge_energy: float = float(jnp.min(jnp.abs(energies)))
            penetration: float = abs(2.0 * math.cos(ka / 2.0))
            assert edge_energy <= penetration**n_chains + 1e-12

        pi_energies: Float64[Array, " nband"] = jnp.linalg.eigvalsh(
            bloch_hamiltonian(
                slab,
                jnp.asarray((0.5, 0.0, 0.0), dtype=jnp.float64),
            )
        )
        assert float(jnp.min(jnp.abs(pi_energies))) <= 1e-12

    @pytest.mark.parametrize("n_lines", [5, 6, 7])
    def test_armchair_k_zero_levels(self, n_lines: int) -> None:
        """Match Eq. (9) and its N=3m+2 metallicity criterion.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        spec: Any
        spacing: float = math.sqrt(3.0) / 2.0
        slab, spec = gen_slab(
            _graphene_model(),
            miller=(1, -1, 0),
            thickness_ang=(n_lines - 1) * spacing,
            vacuum_ang=8.0,
        )
        actual: Float64[Array, " nband"] = jnp.sort(
            jnp.linalg.eigvalsh(
                bloch_hamiltonian(
                    slab,
                    jnp.zeros((3,), dtype=jnp.float64),
                )
            )
        )
        mode: Int64[Array, " n_line"] = jnp.arange(1, n_lines + 1)
        positive: Float64[Array, " n_line"] = jnp.sort(
            jnp.abs(1.0 + 2.0 * jnp.cos(mode * jnp.pi / (n_lines + 1)))
        )
        expected: Float64[Array, " nband"] = jnp.sort(
            jnp.concatenate((-positive, positive))
        )

        assert spec.n_layers == n_lines
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        is_metallic: bool = float(jnp.min(jnp.abs(actual))) <= 1e-12
        assert is_metallic is (n_lines % 3 == 2)


class TestDepthAndInversionInvariants:
    """Verify slab-depth-translation-invariance and slab-inversion-symmetry.

    The cases check primitive fcc depths and an explicit inversion bijection
    with planted defects.
    """

    def test_fcc_111_primitive_depths(self) -> None:
        """Match FCC(111) d-spacing and reject a doubled Miller normal.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        basis: Any
        geometry: Any
        slab: Any
        spec: Any
        lattice_constant: float = 4.2
        lattice: Float64[Array, "3 3"] = jnp.asarray(
            (
                (0.0, lattice_constant / 2.0, lattice_constant / 2.0),
                (lattice_constant / 2.0, 0.0, lattice_constant / 2.0),
                (lattice_constant / 2.0, lattice_constant / 2.0, 0.0),
            ),
            dtype=jnp.float64,
        )
        geometry = make_crystal_geometry(
            lattice=lattice,
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        )
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        bulk: TBModel = make_tb_model(
            hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
            onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
            soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
            hopping_pairs=(),
            hopping_cells=(),
            shell_index=(-1,),
        )
        spacing: float = lattice_constant / math.sqrt(3.0)
        n_planes: int = 7
        slab, spec = gen_slab(
            bulk,
            miller=(1, 1, 1),
            thickness_ang=(n_planes - 1) * spacing,
            vacuum_ang=6.0,
        )

        assert spec.n_layers == n_planes
        assert (
            sum(
                miller * coefficient
                for miller, coefficient in zip(
                    spec.surface_cell.miller,
                    spec.surface_cell.stacking_coeffs,
                    strict=True,
                )
            )
            == 1
        )
        assert spec.surface_cell.interlayer_spacing_ang == pytest.approx(
            spacing,
            rel=1e-12,
        )
        assert slab.depths is not None
        assert jnp.allclose(
            jnp.sort(slab.depths),
            jnp.arange(n_planes, dtype=jnp.float64) * spacing,
            rtol=1e-12,
            atol=1e-12,
        )
        with pytest.raises(ValueError, match="must equal one"):
            make_surface_cell(
                in_plane_vectors=spec.surface_cell.in_plane_vectors,
                stacking_vector=2.0 * spec.surface_cell.stacking_vector,
                rotation=spec.surface_cell.rotation,
                interlayer_spacing_ang=2.0 * spacing,
                miller=spec.surface_cell.miller,
                in_plane_coeffs=spec.surface_cell.in_plane_coeffs,
                stacking_coeffs=tuple(
                    2 * coefficient
                    for coefficient in spec.surface_cell.stacking_coeffs
                ),
            )
        with pytest.raises(ValueError, match="gcd-reduced"):
            find_surface_cell(geometry, (2, 2, 2))

    def test_explicit_inversion_bijection_and_planted_failures(self) -> None:
        """Verify covariance and reject incomplete inversion proxies.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bulk: Any
        local_permutation: Any
        noncentrosymmetric_geometry: Any
        slab: Any
        spec: Any
        bulk, local_permutation = _inversion_bulk_model()
        slab, spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=3.0,
            vacuum_ang=5.0,
        )
        n_local: int = local_permutation.size
        permutation: Int64[NDArray, " n_orb"] = np.asarray(
            [
                (spec.n_layers - 1 - layer) * n_local
                + local_permutation[orbital]
                for layer in range(spec.n_layers)
                for orbital in range(n_local)
            ]
        )
        positions: Float64[NDArray, "n_atom 3"] = np.asarray(
            slab.geometry.positions
        )
        centre: Float64[NDArray, " 3"] = np.asarray(
            (
                0.5,
                0.5,
                (np.min(positions[:, 2]) + np.max(positions[:, 2])) / 2.0,
            )
        )
        _assert_inversion_witness(slab, permutation, centre)

        reversed_order: Int64[NDArray, " n_orb"] = np.arange(permutation.size)[
            ::-1
        ]
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            _assert_inversion_witness(slab, reversed_order, centre)

        displaced_positions: Float64[Array, "natom 3"] = (
            slab.geometry.positions.at[0, 0].add(0.031)
        )
        noncentrosymmetric_geometry = make_crystal_geometry(
            lattice=slab.geometry.lattice,
            positions=displaced_positions,
            species=slab.geometry.species,
        )
        isospectral_noncentrosymmetric: TBModel = make_tb_model(
            hopping_amplitudes=slab.hopping_amplitudes,
            onsite_energies=slab.onsite_energies,
            soc_lambdas=slab.soc_lambdas,
            geometry=noncentrosymmetric_geometry,
            basis=slab.basis,
            hopping_pairs=slab.hopping_pairs,
            hopping_cells=slab.hopping_cells,
            shell_index=slab.shell_index,
            spinor=slab.spinor,
            depths=slab.depths,
        )
        reference_spectrum: Float64[Array, " nband"] = jnp.linalg.eigvalsh(
            bloch_hamiltonian(
                slab,
                jnp.zeros((3,), dtype=jnp.float64),
            )
        )
        displaced_spectrum: Float64[Array, " nband"] = jnp.linalg.eigvalsh(
            bloch_hamiltonian(
                isospectral_noncentrosymmetric,
                jnp.zeros((3,), dtype=jnp.float64),
            )
        )
        assert jnp.allclose(
            reference_spectrum,
            displaced_spectrum,
            rtol=0.0,
            atol=1e-12,
        )
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            _assert_inversion_witness(
                isospectral_noncentrosymmetric,
                permutation,
                centre,
            )


class TestOpenNormalAdversaries:
    """Verify open-surface adjacency without wraparound.

    The cases check vacuum plateaus and reject paths that mix periodic images
    with ordinary bonds.
    """

    def test_positive_vacuum_spectral_plateau(self) -> None:
        """Keep a certified slab spectrum invariant under added vacuum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        compact: Any
        compact_spec: Any
        padded: Any
        padded_spec: Any
        bulk: TBModel = _oblique_long_range_model()
        spacing: float = float(
            find_surface_cell(bulk.geometry, (0, 0, 1)).interlayer_spacing_ang
        )
        thickness: float = 7.0 * spacing
        compact, compact_spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=thickness,
            vacuum_ang=3.0,
        )
        padded, padded_spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=thickness,
            vacuum_ang=100.0,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.137, -0.219, 0.0),
            dtype=jnp.float64,
        )

        validate_open_surface_adjacency(compact)
        validate_open_surface_adjacency(padded)
        assert compact_spec.n_layers == padded_spec.n_layers
        assert all(cell[2] == 0 for cell in compact.hopping_cells)
        assert any(
            abs(cell[0]) >= 5 or abs(cell[1]) >= 5
            for cell in compact.hopping_cells
        )
        assert jnp.allclose(
            jnp.linalg.eigvalsh(bloch_hamiltonian(compact, kpoint)),
            jnp.linalg.eigvalsh(bloch_hamiltonian(padded, kpoint)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_mixed_image_and_ordinary_path_is_rejected(self) -> None:
        """Reject an image edge hidden inside a two-step surface path.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        basis: Any
        geometry: Any
        geometry = make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.asarray(
                ((0.0, 0.0, 0.9), (0.0, 0.0, 0.5), (0.0, 0.0, 0.1)),
                dtype=jnp.float64,
            ),
            species=("X",) * 3,
        )
        basis = make_orbital_basis(
            atom_indices=(0, 1, 2),
            n=(1,) * 3,
            l=(0,) * 3,
            m=(0,) * 3,
        )
        mixed_path: TBModel = make_tb_model(
            hopping_amplitudes=jnp.ones((4,), dtype=jnp.complex128),
            onsite_energies=jnp.zeros((3,), dtype=jnp.float64),
            soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
            hopping_pairs=((0, 1), (1, 0), (1, 2), (2, 1)),
            hopping_cells=((0, 0, 1), (0, 0, -1), (0, 0, 0), (0, 0, 0)),
            shell_index=(-1,) * 3,
        )

        assert (0, 2) not in mixed_path.hopping_pairs
        assert (2, 0) not in mixed_path.hopping_pairs
        with pytest.raises(
            ValueError,
            match=r"normal-image hopping:.*component_path=\(0, 2\)",
        ):
            validate_open_surface_adjacency(mixed_path)
