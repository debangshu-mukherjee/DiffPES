"""Validate explicit hopping-list and normative Wannier90 TB ingestion.

The tests cover exact cells, degeneracy normalization, format dispatch,
operator blocks, spin permutations, analytic bands, and malformed inputs.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy import ndarray as NDArray  # noqa: N812

import diffpes.inout.tb_files as tb_files
from diffpes.inout.tb_files import (
    read_hopping_list,
    read_wannier90_hr,
    read_wannier90_tb,
)
from diffpes.tightb.hamiltonian import bloch_hamiltonian
from diffpes.types.geometry import CrystalGeometry, make_crystal_geometry
from diffpes.types.radial_params import OrbitalBasis, make_orbital_basis
from diffpes.types.tb_model import TBModel
from diffpes.types.wannier import WannierOperatorData


def _write_degeneracies(
    lines: list[str],
    degeneracies: tuple[int, ...],
) -> None:
    """Append normative groups of at most fifteen degeneracies."""
    for start in range(0, len(degeneracies), 15):
        lines.append(
            " ".join(str(value) for value in degeneracies[start : start + 15])
        )


def _pair_order(n_orbitals: int) -> tuple[tuple[int, int], ...]:
    """Return a deliberately non-writer-loop matrix-index order."""
    pairs: list[tuple[int, int]] = [
        (first, second)
        for first in range(n_orbitals)
        for second in range(n_orbitals)
    ]
    return tuple(reversed(pairs))


def _write_hr_fixture(
    path: Path,
    cells: tuple[tuple[int, int, int], ...],
    degeneracies: tuple[int, ...],
    matrices: NDArray,
) -> None:
    """Write independently assembled normative ``hr.dat`` text."""
    n_orbitals: int = matrices.shape[1]
    lines: list[str] = [
        "hand-built neutral hr fixture",
        str(n_orbitals),
        str(len(cells)),
    ]
    _write_degeneracies(lines, degeneracies)
    order: tuple[tuple[int, int], ...] = _pair_order(n_orbitals)
    for cell_index, cell in enumerate(cells):
        raw_matrix: NDArray = matrices[cell_index] * degeneracies[cell_index]
        for first, second in order:
            value: complex = complex(raw_matrix[first, second])
            lines.append(
                f"{cell[0]} {cell[1]} {cell[2]} "
                f"{first + 1} {second + 1} "
                f"{value.real:.17g} {value.imag:.17g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tb_fixture(
    path: Path,
    lattice: NDArray,
    cells: tuple[tuple[int, int, int], ...],
    degeneracies: tuple[int, ...],
    hamiltonians: NDArray,
    positions: NDArray,
) -> None:
    """Write independently assembled normative ``tb.dat`` text."""
    n_orbitals: int = hamiltonians.shape[1]
    lines: list[str] = ["hand-built neutral tb fixture"]
    lines.extend(
        " ".join(f"{component:.17g}" for component in row) for row in lattice
    )
    lines.extend((str(n_orbitals), str(len(cells))))
    _write_degeneracies(lines, degeneracies)
    order: tuple[tuple[int, int], ...] = _pair_order(n_orbitals)
    for cell_index, cell in enumerate(cells):
        lines.append("")
        lines.append(f"{cell[0]} {cell[1]} {cell[2]}")
        raw_matrix: NDArray = (
            hamiltonians[cell_index] * degeneracies[cell_index]
        )
        for first, second in order:
            value: complex = complex(raw_matrix[first, second])
            lines.append(
                f"{first + 1} {second + 1} {value.real:.17g} {value.imag:.17g}"
            )
    for cell_index in reversed(range(len(cells))):
        cell: tuple[int, int, int] = cells[cell_index]
        lines.append("")
        lines.append(f"{cell[0]} {cell[1]} {cell[2]}")
        raw_matrix = positions[cell_index] * degeneracies[cell_index]
        for first, second in order:
            components: NDArray = raw_matrix[first, second]
            fields: list[str] = [f"{first + 1}", f"{second + 1}"]
            for component in components:
                value = complex(component)
                fields.extend((f"{value.real:.17g}", f"{value.imag:.17g}"))
            lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _two_orbital_context() -> tuple[CrystalGeometry, OrbitalBasis]:
    """Build a two-site context for Cartesian hopping-list tests."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        species=("X", "Y"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        labels=("X_s", "Y_s"),
    )
    return geometry, basis


def _chain_context() -> tuple[CrystalGeometry, OrbitalBasis]:
    """Build the one-orbital context used by the analytic hr gate."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    return geometry, basis


def _spin_basis() -> OrbitalBasis:
    """Build the native down-block then up-block two-orbital basis."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1, 0, 1),
        n=(1, 1, 1, 1),
        l=(0, 0, 0, 0),
        m=(0, 0, 0, 0),
        spin=(-1, -1, 1, 1),
        labels=("X_s_dn", "Y_s_dn", "X_s_up", "Y_s_up"),
    )
    return basis


def _serialize_interleaved(native: NDArray) -> NDArray:
    """Convert native block-down/up matrices into interleaved up/down order."""
    serialized_to_native: tuple[int, ...] = (2, 0, 3, 1)
    serialized: NDArray = np.take(native, serialized_to_native, axis=1)
    serialized = np.take(serialized, serialized_to_native, axis=2)
    return serialized


class TestReadHoppingList:
    """Validate :func:`diffpes.inout.read_hopping_list`."""

    def test_recovers_exact_cells_and_complex_closed_hoppings(
        self,
        tmp_path: Path,
    ) -> None:
        """Convert Cartesian bonds only after validating integer recovery.

        Complex reverse records and real onsite entries must retain their
        exact metadata.

        Notes
        -----
        Parse a literal file and compare cells, pairs, amplitudes, and onsite
        values.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _two_orbital_context()
        path: Path = tmp_path / "neutral_hoppings.txt"
        path.write_text(
            "0,0,0,0,0,0.2\n"
            "1,1,0,0,0,-0.1\n"
            "0,1,0.25,0,0,0.3,0.2\n"
            "1,0,-0.25,0,0,0.3,-0.2\n"
            "0,1,1.25,0,0,-0.4,0.1\n"
            "1,0,-1.25,0,0,-0.4,-0.1\n",
            encoding="utf-8",
        )

        model: TBModel = read_hopping_list(str(path), geometry, basis)

        chex.assert_trees_all_close(
            model.onsite_energies,
            jnp.asarray([0.2, -0.1]),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            model.hopping_amplitudes,
            jnp.asarray([0.3 + 0.2j, 0.3 - 0.2j, -0.4 + 0.1j, -0.4 - 0.1j]),
            rtol=0.0,
            atol=0.0,
        )
        assert model.hopping_pairs == ((0, 1), (1, 0), (0, 1), (1, 0))
        assert model.hopping_cells == (
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
        )
        fractional_k: jax.Array = jnp.asarray(0.17)
        actual: jax.Array = bloch_hamiltonian(
            model,
            jnp.asarray([fractional_k, 0.0, 0.0]),
        )
        expected_01: jax.Array = (0.3 + 0.2j) * jnp.exp(
            2j * jnp.pi * fractional_k * 0.25
        ) + (-0.4 + 0.1j) * jnp.exp(2j * jnp.pi * fractional_k * 1.25)
        expected: jax.Array = jnp.asarray(
            [[0.2, expected_01], [jnp.conj(expected_01), -0.1]],
            dtype=jnp.complex128,
        )
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_names_noninteger_cell_recovery_row(self, tmp_path: Path) -> None:
        """Reject a Cartesian bond outside the named ``1e-10`` tolerance.

        The diagnostic must identify the physical source row and candidate.

        Notes
        -----
        Perturb one literal bond component beyond the recovery threshold.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _two_orbital_context()
        path: Path = tmp_path / "noninteger.txt"
        path.write_text(
            "0,1,0.2500000002,0,0,0.3\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"line 1:.*noninteger cell"):
            read_hopping_list(str(path), geometry, basis)

    @pytest.mark.parametrize(
        ("contents", "match"),
        (
            ("", "contains no records"),
            ("0 0 0 0 0 1\n", "comma-separated"),
            ("2,0,0,0,0,1\n", "orbital indices"),
            (
                "0,1,0.25,0,0,0.3\n",
                r"row 1: missing Hermitian reverse",
            ),
            (
                "0,1,0.25,0,0,0.3\n1,0,-0.25,0,0,0.4\n",
                r"rows 1 and 2: reverse hopping amplitudes differ",
            ),
        ),
    )
    def test_rejects_malformed_hopping_lists(
        self,
        tmp_path: Path,
        contents: str,
        match: str,
    ) -> None:
        """Reject empty, misdelimited, indexed, open, and inconsistent rows.

        Independent malformed clauses must reach their specific parser
        diagnostics.

        Notes
        -----
        Parameterize literal file contents and match each expected message.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _two_orbital_context()
        path: Path = tmp_path / "malformed.txt"
        path.write_text(contents, encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            read_hopping_list(str(path), geometry, basis)


class TestReadWannier90Hr:
    """Validate :func:`diffpes.inout.read_wannier90_hr`."""

    def test_applies_degeneracies_and_matches_chain_value_and_derivative(
        self,
        tmp_path: Path,
    ) -> None:
        r"""Recover :math:`E(k)=e_0+2t\cos(2\pi k)` and its derivative.

        A weighted three-cell file provides a complete analytic chain
        reference.

        Notes
        -----
        Parse the literal file, evaluate Bloch energies, and differentiate
        with respect to k.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _chain_context()
        cells: tuple[tuple[int, int, int], ...] = (
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        )
        degeneracies: tuple[int, ...] = (2, 1, 2)
        hopping: float = -0.6
        onsite: float = 0.4
        matrices: NDArray = np.asarray(
            [[[hopping]], [[onsite]], [[hopping]]],
            dtype=np.complex128,
        )
        path: Path = tmp_path / "chain_hr.dat"
        _write_hr_fixture(path, cells, degeneracies, matrices)
        centres: jax.Array = jnp.asarray([[0.37, -0.2, 0.1]])

        model: TBModel
        data: WannierOperatorData
        model, data = read_wannier90_hr(
            str(path),
            geometry,
            basis,
            centres,
        )

        assert model.hopping_cells == ((-1, 0, 0), (1, 0, 0))
        chex.assert_trees_all_close(
            model.hopping_amplitudes,
            hopping,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            model.onsite_energies,
            onsite,
            rtol=0.0,
            atol=0.0,
        )
        assert data.position_matrices is None
        assert data.cells == cells
        assert data.degeneracies == degeneracies
        chex.assert_trees_all_close(data.centres_cart, centres)

        fractional_k: jax.Array = jnp.asarray(0.23)

        def energy(kx: jax.Array) -> jax.Array:
            """Return the parsed scalar chain Hamiltonian."""
            value: jax.Array = bloch_hamiltonian(
                model,
                jnp.stack((kx, jnp.asarray(0.0), jnp.asarray(0.0))),
            )[0, 0]
            return jnp.real(value)

        expected: jax.Array = onsite + 2.0 * hopping * jnp.cos(
            2.0 * jnp.pi * fractional_k
        )
        expected_derivative: jax.Array = (
            -4.0 * jnp.pi * hopping * jnp.sin(2.0 * jnp.pi * fractional_k)
        )
        chex.assert_trees_all_close(
            energy(fractional_k),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )
        chex.assert_trees_all_close(
            jax.grad(energy)(fractional_k),
            expected_derivative,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rejects_bad_weights_indices_centres_and_closure(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject independent malformed clauses with source-row diagnostics.

        Header, weight, indexing, centre, and Hermiticity errors must remain
        distinguishable.

        Notes
        -----
        Parameterize normative-looking files and match each parser message.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _chain_context()
        cells: tuple[tuple[int, int, int], ...] = (
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        )
        matrices: NDArray = np.asarray(
            [[[-0.5]], [[0.2]], [[-0.5]]],
            dtype=np.complex128,
        )

        bad_weight: Path = tmp_path / "bad_weight_hr.dat"
        _write_hr_fixture(bad_weight, cells, (2, 1, 2), matrices)
        lines: list[str] = bad_weight.read_text(encoding="utf-8").splitlines()
        lines[3] = "2 0 2"
        bad_weight.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="degeneracies must be positive"):
            read_wannier90_hr(
                str(bad_weight),
                geometry,
                basis,
                jnp.zeros((1, 3)),
            )

        bad_index: Path = tmp_path / "bad_index_hr.dat"
        _write_hr_fixture(bad_index, cells, (2, 1, 2), matrices)
        lines = bad_index.read_text(encoding="utf-8").splitlines()
        fields: list[str] = lines[4].split()
        fields[3] = "2"
        lines[4] = " ".join(fields)
        bad_index.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"line 5: matrix indices"):
            read_wannier90_hr(
                str(bad_index),
                geometry,
                basis,
                jnp.zeros((1, 3)),
            )

        bad_closure: Path = tmp_path / "bad_closure_hr.dat"
        _write_hr_fixture(bad_closure, cells, (2, 1, 2), matrices)
        lines = bad_closure.read_text(encoding="utf-8").splitlines()
        fields = lines[4].split()
        fields[5] = "-0.8"
        lines[4] = " ".join(fields)
        bad_closure.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(
            ValueError,
            match=r"rows 5 and 7: reverse hopping amplitudes differ",
        ):
            read_wannier90_hr(
                str(bad_closure),
                geometry,
                basis,
                jnp.zeros((1, 3)),
            )

        valid: Path = tmp_path / "bad_centres_hr.dat"
        _write_hr_fixture(valid, cells, (2, 1, 2), matrices)
        with pytest.raises(ValueError, match=r"shape \(1, 3\)"):
            read_wannier90_hr(
                str(valid),
                geometry,
                basis,
                jnp.zeros((2, 3)),
            )


class TestReadWannier90Tb:
    """Validate :func:`diffpes.inout.read_wannier90_tb`."""

    def test_round_trips_all_blocks_and_spin_layouts(
        self,
        tmp_path: Path,
    ) -> None:
        """Apply one permutation to H, position matrices, and centres.

        Both supported serialized spin layouts must produce identical native
        carriers.

        Notes
        -----
        Parse literal operator blocks and compare every permuted array.
        """
        basis: OrbitalBasis = _spin_basis()
        lattice: NDArray = np.diag([2.0, 3.0, 4.0]).astype(np.float64)
        cells: tuple[tuple[int, int, int], ...] = (
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        )
        degeneracies: tuple[int, ...] = (2, 1, 2)
        onsite: NDArray = np.asarray([-0.2, 0.3, -0.2, 0.3])
        hopping: NDArray = np.asarray([-0.5, -0.25, -0.5, -0.25])
        hamiltonians: NDArray = np.zeros(
            (3, 4, 4),
            dtype=np.complex128,
        )
        hamiltonians[0] = np.diag(hopping)
        hamiltonians[1] = np.diag(onsite)
        hamiltonians[2] = np.diag(hopping)
        positions: NDArray = np.zeros(
            (3, 4, 4, 3),
            dtype=np.complex128,
        )
        centres: NDArray = np.asarray(
            [
                [0.2, 0.3, 0.4],
                [1.2, 0.6, 0.8],
                [0.2, 0.3, 0.4],
                [1.2, 0.6, 0.8],
            ]
        )
        for orbital, centre in enumerate(centres):
            positions[1, orbital, orbital] = centre
        positions[1, 0, 1] = np.asarray([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
        positions[2, 3, 2] = np.asarray([-0.1 + 0.3j, 0.7 + 0.2j, -0.4 + 0.6j])

        block_path: Path = tmp_path / "block_tb.dat"
        _write_tb_fixture(
            block_path,
            lattice,
            cells,
            degeneracies,
            hamiltonians,
            positions,
        )
        interleaved_path: Path = tmp_path / "interleaved_tb.dat"
        _write_tb_fixture(
            interleaved_path,
            lattice,
            cells,
            degeneracies,
            _serialize_interleaved(hamiltonians),
            _serialize_interleaved(positions),
        )

        block_model: TBModel
        block_data: WannierOperatorData
        block_model, block_data = read_wannier90_tb(
            str(block_path),
            basis,
            "block_down_up",
        )
        interleaved_model: TBModel
        interleaved_data: WannierOperatorData
        interleaved_model, interleaved_data = read_wannier90_tb(
            str(interleaved_path),
            basis,
            "interleaved_up_down",
        )

        chex.assert_trees_all_close(
            block_model.hopping_amplitudes,
            interleaved_model.hopping_amplitudes,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            block_model.onsite_energies,
            interleaved_model.onsite_energies,
            rtol=0.0,
            atol=0.0,
        )
        assert block_model.hopping_pairs == interleaved_model.hopping_pairs
        assert block_model.hopping_cells == interleaved_model.hopping_cells
        chex.assert_trees_all_close(
            block_model.geometry.lattice,
            lattice,
            rtol=0.0,
            atol=0.0,
        )
        expected_fractional_positions: NDArray = np.asarray(
            [centres[0], centres[1]]
        ) @ np.linalg.inv(lattice)
        chex.assert_trees_all_close(
            block_model.geometry.positions,
            expected_fractional_positions,
            rtol=0.0,
            atol=1e-15,
        )
        assert block_data.position_matrices is not None
        assert interleaved_data.position_matrices is not None
        chex.assert_trees_all_close(
            block_data.position_matrices,
            positions,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            interleaved_data.position_matrices,
            positions,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(block_data.centres_cart, centres)
        chex.assert_trees_all_close(interleaved_data.centres_cart, centres)
        assert block_data.cells == cells
        assert block_data.degeneracies == degeneracies
        assert block_data.spin_layout == "block_down_up"
        assert interleaved_data.spin_layout == "interleaved_up_down"

        fractional_k: jax.Array = jnp.asarray(0.19)

        def matrix(kx: jax.Array) -> jax.Array:
            """Return the parsed Bloch matrix along the first reciprocal axis."""
            return bloch_hamiltonian(
                interleaved_model,
                jnp.stack((kx, jnp.asarray(0.0), jnp.asarray(0.0))),
            )

        expected_matrix: jax.Array = jnp.diag(
            jnp.asarray(onsite)
            + 2.0 * jnp.asarray(hopping) * jnp.cos(2.0 * jnp.pi * fractional_k)
        ).astype(jnp.complex128)
        expected_derivative: jax.Array = jnp.diag(
            -4.0
            * jnp.pi
            * jnp.asarray(hopping)
            * jnp.sin(2.0 * jnp.pi * fractional_k)
        ).astype(jnp.complex128)
        chex.assert_trees_all_close(
            matrix(fractional_k),
            expected_matrix,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jax.jacfwd(matrix)(fractional_k),
            expected_derivative,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rejects_missing_position_data_and_bad_spin_layout(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject truncated operator blocks and unsupported spin selectors.

        File completeness and static layout validation must fail
        independently.

        Notes
        -----
        Truncate a literal file and separately pass an invalid selector.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        lattice: NDArray = np.eye(3)
        cells: tuple[tuple[int, int, int], ...] = ((0, 0, 0),)
        hamiltonians: NDArray = np.asarray([[[0.2 + 0.0j]]])
        positions: NDArray = np.asarray([[[[0.1, 0.2, 0.3]]]])
        path: Path = tmp_path / "truncated_tb.dat"
        _write_tb_fixture(
            path,
            lattice,
            cells,
            (1,),
            hamiltonians,
            positions,
        )
        lines: list[str] = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unexpected end of file"):
            read_wannier90_tb(str(path), basis, "block_down_up")

        valid_path: Path = tmp_path / "selector_tb.dat"
        _write_tb_fixture(
            valid_path,
            lattice,
            cells,
            (1,),
            hamiltonians,
            positions,
        )
        with pytest.raises(ValueError, match="spin_layout must be"):
            read_wannier90_tb(str(valid_path), basis, "unknown")
        with pytest.raises(ValueError, match="requires a spinor basis"):
            read_wannier90_tb(
                str(valid_path),
                basis,
                "interleaved_up_down",
            )

        bad_index: Path = tmp_path / "bad_index_tb.dat"
        _write_tb_fixture(
            bad_index,
            lattice,
            cells,
            (1,),
            hamiltonians,
            positions,
        )
        lines = bad_index.read_text(encoding="utf-8").splitlines()
        hamiltonian_fields: list[str] = lines[9].split()
        hamiltonian_fields[0] = "2"
        lines[9] = " ".join(hamiltonian_fields)
        bad_index.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"line 10: matrix indices"):
            read_wannier90_tb(str(bad_index), basis, "block_down_up")

        bad_position_cell: Path = tmp_path / "bad_position_cell_tb.dat"
        _write_tb_fixture(
            bad_position_cell,
            lattice,
            cells,
            (1,),
            hamiltonians,
            positions,
        )
        lines = bad_position_cell.read_text(encoding="utf-8").splitlines()
        lines[11] = "1 0 0"
        bad_position_cell.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="has no Hamiltonian block"):
            read_wannier90_tb(
                str(bad_position_cell),
                basis,
                "block_down_up",
            )

    def test_rejects_inconsistent_spin_copy_centres(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject orbital centres that cannot define one position per atom.

        Spin partners on one atom must carry a consistent spatial centre.

        Notes
        -----
        Alter one origin position diagonal and match the centre diagnostic.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
            spin=(-1, 1),
        )
        hamiltonians: NDArray = np.zeros((1, 2, 2), dtype=np.complex128)
        positions: NDArray = np.zeros(
            (1, 2, 2, 3),
            dtype=np.complex128,
        )
        positions[0, 0, 0] = np.asarray([0.0, 0.0, 0.0])
        positions[0, 1, 1] = np.asarray([0.01, 0.0, 0.0])
        path: Path = tmp_path / "centres_tb.dat"
        _write_tb_fixture(
            path,
            np.eye(3),
            ((0, 0, 0),),
            (1,),
            hamiltonians,
            positions,
        )

        with pytest.raises(ValueError, match="centres assigned to atom 0"):
            read_wannier90_tb(str(path), basis, "block_down_up")


class TestExplicitFormatDispatch:
    """Validate the deliberate absence of generic ``.dat`` dispatch."""

    def test_requires_explicit_format_filenames(self, tmp_path: Path) -> None:
        """Reject generic suffixes before attempting either normative grammar.

        Each reader must enforce its own explicit Wannier90 filename contract.

        Notes
        -----
        Call both readers with one generic path and match their suffix errors.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        geometry, basis = _chain_context()
        generic: Path = tmp_path / "model.dat"
        generic.write_text("not dispatched\n", encoding="utf-8")

        with pytest.raises(ValueError, match="generic .dat dispatch"):
            read_wannier90_hr(
                str(generic),
                geometry,
                basis,
                jnp.zeros((1, 3)),
            )
        with pytest.raises(ValueError, match="generic .dat dispatch"):
            read_wannier90_tb(
                str(generic),
                basis,
                "block_down_up",
            )
        assert not hasattr(tb_files, "read_tb_file")
