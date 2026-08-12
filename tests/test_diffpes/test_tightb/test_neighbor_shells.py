"""Validate the neighbor shells module.

The cases use analytic values, invariants, and finite differences.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Float64

from diffpes.tightb import (
    neighbor_shells,
)
from diffpes.types import (
    CrystalGeometry,
    make_crystal_geometry,
)

from ._slaterkoster_helpers import (
    _graphene_geometry,
)


class TestNeighborShells:
    """Validate :func:`diffpes.tightb.neighbor_shells`.

    The cases check honeycomb, fcc, small-cell, and skew-lattice shells plus
    explicit-radius rejection.
    """

    def test_honeycomb_has_three_unique_nearest_neighbor_bonds(self) -> None:
        """Verify the three undirected A--B bonds of a honeycomb cell.

        The records retain distinct exact cells for the translated nearest
        neighbors.

        Notes
        -----
        Also derive every fractional displacement from its exact integer cell.
        """
        geometry: CrystalGeometry = _graphene_geometry()
        atom_pairs: Tuple[Tuple[int, int], ...]
        cells: Tuple[Tuple[int, int, int], ...]
        displacements: Float64[Array, "3 3"]
        distances: Float64[Array, " 3"]
        atom_pairs, cells, displacements, distances = neighbor_shells(
            geometry,
            1.5,
        )

        assert atom_pairs == ((0, 1), (0, 1), (0, 1))
        assert cells == ((-1, 0, 0), (0, -1, 0), (0, 0, 0))
        assert all(
            type(component) is int for cell in cells for component in cell
        )
        expected: Float64[Array, "3 3"] = (
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
        lattice: Float64[Array, "3 3"] = (
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
        atom_pairs: Tuple[Tuple[int, int], ...]
        cells: Tuple[Tuple[int, int, int], ...]
        distances: Float64[Array, " 6"]
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
        atom_pairs: Tuple[Tuple[int, int], ...]
        cells: Tuple[Tuple[int, int, int], ...]
        distances: Float64[Array, " 3"]
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
        cells: Tuple[Tuple[int, int, int], ...]
        distances: Float64[Array, " n_bond"]
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
