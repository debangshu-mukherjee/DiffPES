"""Validate explicit Cartesian hopping-list ingestion.

The tests cover exact cells, complex Hermitian closure, spin permutations,
and malformed hopping-list inputs.
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.inout import read_hopping_list
from diffpes.tightb import bloch_hamiltonian
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
)


def _two_orbital_context() -> Tuple[CrystalGeometry, OrbitalBasis]:
    """PRIVATE: Build a two-site context for Cartesian hopping-list tests.

    Returns
    -------
    geometry : CrystalGeometry
        Identity cubic lattice in Angstrom with atoms X at the origin
        and Y at fractional position (0.25, 0, 0).
    basis : OrbitalBasis
        One s orbital on each atom, labeled ``X_s`` and ``Y_s``.

    Notes
    -----
    Keeps the geometry minimal so Cartesian bond vectors map to cell
    offsets by inspection.
    """
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
    context: Tuple[CrystalGeometry, OrbitalBasis] = (geometry, basis)
    return context


class TestReadHoppingList:
    """Validate :func:`diffpes.inout.read_hopping_list`.

    The cases recover exact cells and reject malformed Hermitian records.
    """

    def test_recovers_exact_cells_and_complex_closed_hoppings(
        self,
        tmp_path: Path,
    ) -> None:
        """Convert Cartesian bonds only after validating integer recovery.

        Complex128 reverse records and real onsite entries must retain their
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
        fractional_k: Float64[Array, ""] = jnp.asarray(0.17)
        actual: Complex128[Array, "2 2"] = bloch_hamiltonian(
            model,
            jnp.asarray([fractional_k, 0.0, 0.0]),
        )
        expected_01: Complex128[Array, ""] = (0.3 + 0.2j) * jnp.exp(
            2j * jnp.pi * fractional_k * 0.25
        ) + (-0.4 + 0.1j) * jnp.exp(2j * jnp.pi * fractional_k * 1.25)
        expected: Complex128[Array, "2 2"] = jnp.asarray(
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
        [
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
        ],
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
