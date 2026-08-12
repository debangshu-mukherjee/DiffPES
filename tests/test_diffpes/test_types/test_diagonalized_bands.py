"""Validate the diagonalized bands contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from beartype.typing import List
from jaxtyping import Array, Complex128

from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    PyTreeDef,
    make_diagonalized_bands,
)
from tests._assertions import assert_rejects
from tests._factories import (
    make_minimal_crystal_geometry,
    make_minimal_orbital_basis,
)


class TestDiagonalizedBands(chex.TestCase):
    """Validate :class:`~diffpes.types.DiagonalizedBands`.

    The case round-trips numerical leaves and compares the attached geometry
    and basis context.
    """

    def test_pytree_round_trip_preserves_context(self) -> None:
        """Preserve numerical leaves and the static basis on reconstruction.

        The case flattens and rebuilds a geometry-bearing eigensystem.

        Notes
        -----
        Compare all leaves and inspect the restored atom and quantum metadata.
        """
        geometry: CrystalGeometry = make_minimal_crystal_geometry()
        basis: OrbitalBasis = make_minimal_orbital_basis()
        bands: DiagonalizedBands = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0], [2.0]], dtype=jnp.float64),
            eigenvectors=jnp.ones((2, 1, 1), dtype=jnp.complex128),
            kpoints=jnp.zeros((2, 3), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(bands)
        restored: DiagonalizedBands = jax.tree_util.tree_unflatten(
            tree,
            leaves,
        )

        assert len(leaves) == 7
        chex.assert_trees_all_close(restored, bands)
        assert restored.basis.atom_indices == (0,)
        assert restored.basis.n == (1,)


class TestMakeDiagonalizedBands(chex.TestCase):
    """Validate :func:`~diffpes.types.make_diagonalized_bands`.

    The cases check context construction plus eager and compiled rejection of
    structural and numerical defects.
    """

    def test_constructs_geometry_and_basis_context(self) -> None:
        """Store the frozen geometry and basis contract with normalized dtypes.

        The case builds a two-orbital eigensystem on a two-atom geometry.

        Notes
        -----
        Check eigensystem shapes, numerical dtypes, and atom assignments.
        """
        geometry: CrystalGeometry = make_minimal_crystal_geometry(2)
        basis: OrbitalBasis = make_minimal_orbital_basis((0, 1))
        bands: DiagonalizedBands = make_diagonalized_bands(
            eigenvalues=jnp.zeros((5, 3), dtype=jnp.float64),
            eigenvectors=jnp.ones((5, 3, 2), dtype=jnp.complex128),
            kpoints=jnp.zeros((5, 3), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
            fermi_energy=0.5,
        )

        chex.assert_shape(bands.eigenvalues, (5, 3))
        chex.assert_shape(bands.eigenvectors, (5, 3, 2))
        assert bands.eigenvalues.dtype == jnp.float64
        assert bands.eigenvectors.dtype == jnp.complex128
        assert bands.basis.atom_indices == (0, 1)

    @parameterized.named_parameters(
        (
            "k_axis_mismatch",
            "k_axis",
            "eigenvalues and eigenvectors must agree",
        ),
        (
            "basis_axis_mismatch",
            "basis_axis",
            "eigenvector orbital axis must match basis",
        ),
        (
            "atom_mapping_out_of_range",
            "atom_mapping",
            "basis atom_indices must refer",
        ),
    )
    def test_rejects_structural_mismatch(
        self,
        defect: str,
        match: str,
    ) -> None:
        """Reject incompatible eigensystem axes and structural context.

        Parameterized cases vary k axes, orbital axes, and atom mappings.

        Notes
        -----
        Match the factory diagnostic for each isolated structural mismatch.
        """
        eigenvectors: Complex128[Array, "1 1 1"] = jnp.ones(
            (1, 1, 1),
            dtype=jnp.complex128,
        )
        geometry: CrystalGeometry = make_minimal_crystal_geometry()
        basis: OrbitalBasis = make_minimal_orbital_basis()
        if defect == "k_axis":
            eigenvectors = jnp.ones((2, 1, 1), dtype=jnp.complex128)
        elif defect == "basis_axis":
            eigenvectors = jnp.ones((1, 1, 2), dtype=jnp.complex128)
        else:
            basis = make_minimal_orbital_basis((1,))

        assert_rejects(
            make_diagonalized_bands,
            eigenvalues=jnp.zeros((1, 1), dtype=jnp.float64),
            eigenvectors=eigenvectors,
            kpoints=jnp.zeros((1, 3), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
            match=match,
        )

    def test_rejects_nonfinite_data_eager_and_jit(self) -> None:
        """Reject a NaN eigenvector through runtime validation.

        The case injects one nonfinite complex orbital coefficient.

        Notes
        -----
        Use the shared helper for eager and compiled runtime rejection.
        """
        assert_rejects(
            make_diagonalized_bands,
            eigenvalues=jnp.zeros((1, 1), dtype=jnp.float64),
            eigenvectors=jnp.array([[[jnp.nan + 0.0j]]]),
            kpoints=jnp.zeros((1, 3), dtype=jnp.float64),
            geometry=make_minimal_crystal_geometry(),
            basis=make_minimal_orbital_basis(),
            match="eigenvectors finite",
        )
