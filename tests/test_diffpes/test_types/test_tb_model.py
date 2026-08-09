"""Validate native tight-binding and diagonalized-band carriers.

The tests cover PyTree structure, differentiable leaves, exact connectivity,
geometry context, runtime checks, and randomized Hermitian closure.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from hypothesis import given, settings
from hypothesis import strategies as st

from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    SlabTopology,
    TBModel,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
    make_tb_model,
)
from tests._assertions import assert_rejects


def _geometry(n_atoms: int = 1) -> CrystalGeometry:
    """PRIVATE: Create a simple right-handed test geometry.

    Parameters
    ----------
    n_atoms : int, optional
        Number of species-``X`` atoms, all at the origin. Default 1.

    Returns
    -------
    geometry : CrystalGeometry
        Identity cubic lattice in Angstrom with ``n_atoms`` coincident
        atoms.

    Notes
    -----
    The identity lattice keeps the cell volume positive, so factory
    validation of handedness passes without further data.
    """
    positions: jax.Array = jnp.zeros((n_atoms, 3), dtype=jnp.float64)
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=positions,
        species=tuple("X" for _ in range(n_atoms)),
    )
    return geometry


def _basis(
    atom_indices: Tuple[int, ...] = (0,),
    spin: Tuple[int, ...] = (),
) -> OrbitalBasis:
    """PRIVATE: Create one valid s orbital for every supplied atom index.

    Parameters
    ----------
    atom_indices : Tuple[int, ...], optional
        Atom index of each orbital. Default ``(0,)``.
    spin : Tuple[int, ...], optional
        Spin channel of each orbital, empty for a spinless basis.
        Default ``()``.

    Returns
    -------
    basis : OrbitalBasis
        Basis with one (n=1, l=0, m=0) orbital per supplied atom
        index.

    Notes
    -----
    Repeats the s quantum numbers so callers control only the atom
    assignment and the optional spin channel.
    """
    n_orbitals: int = len(atom_indices)
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=atom_indices,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        spin=spin,
    )
    return basis


def _model_arguments() -> dict[str, object]:
    """PRIVATE: Return a minimal complex Hermitian-closed model argument mapping.

    Returns
    -------
    arguments : dict[str, object]
        Keyword arguments for ``make_tb_model``: one orbital, one
        0.25 eV onsite energy, and one complex-conjugate hopping pair
        on opposite ``(+/-1, 0, 0)`` cells.

    Notes
    -----
    The two amplitudes ``1 + 2j`` and ``1 - 2j`` close the Hermitian
    pairing, so tests can mutate single entries to break exactly one
    validation rule at a time.
    """
    arguments: dict[str, object] = {
        "hopping_amplitudes": jnp.array(
            [1.0 + 2.0j, 1.0 - 2.0j],
            dtype=jnp.complex128,
        ),
        "onsite_energies": jnp.array([0.25], dtype=jnp.float64),
        "soc_lambdas": jnp.zeros((0,), dtype=jnp.float64),
        "geometry": _geometry(),
        "basis": _basis(),
        "hopping_pairs": ((0, 0), (0, 0)),
        "hopping_cells": ((1, 0, 0), (-1, 0, 0)),
        "shell_index": (-1,),
        "spinor": False,
    }
    return arguments


class TestDiagonalizedBands(chex.TestCase):
    """Validate :class:`~diffpes.types.DiagonalizedBands`."""

    def test_pytree_round_trip_preserves_context(self) -> None:
        """Preserve numerical leaves and the static basis on reconstruction.

        The case flattens and rebuilds a geometry-bearing eigensystem.

        Notes
        -----
        Compare all leaves and inspect the restored atom and quantum metadata.
        """
        geometry: CrystalGeometry = _geometry()
        basis: OrbitalBasis = _basis()
        bands: DiagonalizedBands = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0], [2.0]], dtype=jnp.float64),
            eigenvectors=jnp.ones((2, 1, 1), dtype=jnp.complex128),
            kpoints=jnp.zeros((2, 3), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(bands)
        restored: DiagonalizedBands = jax.tree_util.tree_unflatten(
            tree,
            leaves,
        )

        assert len(leaves) == 7
        chex.assert_trees_all_close(restored, bands)
        assert restored.basis.atom_indices == (0,)
        assert restored.basis.n == (1,)


class TestTBModel(chex.TestCase):
    """Validate :class:`~diffpes.types.TBModel`."""

    def test_pytree_round_trip_preserves_nine_field_contract(self) -> None:
        """Preserve all differentiable leaves and static connectivity.

        The case flattens and rebuilds the complete nine-field model carrier.

        Notes
        -----
        Compare leaves, exact hopping metadata, shell indices, and spin mode.
        """
        model: TBModel = make_tb_model(**_model_arguments())
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(model)
        restored: TBModel = jax.tree_util.tree_unflatten(tree, leaves)

        assert len(leaves) == 6
        chex.assert_trees_all_close(restored, model)
        assert restored.hopping_pairs == model.hopping_pairs
        assert restored.hopping_cells == model.hopping_cells
        assert restored.shell_index == (-1,)
        assert not restored.spinor

    def test_array_fields_remain_differentiable(self) -> None:
        """Differentiate a real loss through hoppings, onsite, and geometry.

        The case traces a quadratic loss through every relevant numerical leaf.

        Notes
        -----
        Require finite gradients with the original hopping and onsite shapes.
        """
        model: TBModel = make_tb_model(**_model_arguments())

        def loss(candidate: TBModel) -> jax.Array:
            """Return a real quadratic loss over traced model leaves."""
            result: jax.Array = (
                jnp.sum(jnp.abs(candidate.hopping_amplitudes) ** 2)
                + jnp.sum(candidate.onsite_energies**2)
                + jnp.sum(candidate.geometry.positions**2)
            )
            return result

        gradient: TBModel = jax.grad(loss)(model)

        chex.assert_tree_all_finite(jax.tree.leaves(gradient))
        chex.assert_shape(gradient.hopping_amplitudes, (2,))
        chex.assert_shape(gradient.onsite_energies, (1,))


class TestSurfaceCell:
    """Mirror coverage for :class:`diffpes.types.SurfaceCell`."""


class TestSlabSpec:
    """Mirror coverage for :class:`diffpes.types.SlabSpec`."""


class TestMakeSurfaceCell:
    """Mirror coverage for :func:`diffpes.types.make_surface_cell`."""


class TestMakeSlabSpec:
    """Mirror coverage for :func:`diffpes.types.make_slab_spec`."""


class TestSlabTopology:
    """Validate :class:`~diffpes.types.SlabTopology`."""

    def test_fields_are_static_pytree_metadata(self) -> None:
        """Keep every frozen topology choice outside differentiable leaves.

        The carrier stores one primitive single-atom slab selection.

        Notes
        -----
        Flatten the carrier and compare representative integer metadata.
        """
        topology: SlabTopology = SlabTopology(
            miller=(0, 0, 1),
            in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
            stacking_coeffs=(0, 0, 1),
            atom_shifts=((0, 0, 0),),
            bulk_atom_of_slab_atom=(0,),
            layer_of_slab_atom=(0,),
            termination=("X", "X"),
            thickness_ang=0.0,
            vacuum_ang=3.0,
            fine=(0.0, 0.0),
            n_layers=1,
            bulk_atom_count=1,
            basis_atom_indices=(0,),
        )

        assert jax.tree_util.tree_leaves(topology) == []
        assert topology.miller == (0, 0, 1)
        assert topology.n_layers == 1


class TestMakeDiagonalizedBands(chex.TestCase):
    """Validate :func:`~diffpes.types.make_diagonalized_bands`."""

    def test_constructs_geometry_and_basis_context(self) -> None:
        """Store the frozen geometry and basis contract with normalized dtypes.

        The case builds a two-orbital eigensystem on a two-atom geometry.

        Notes
        -----
        Check eigensystem shapes, numerical dtypes, and atom assignments.
        """
        geometry: CrystalGeometry = _geometry(2)
        basis: OrbitalBasis = _basis((0, 1))
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
        eigenvectors: jax.Array = jnp.ones(
            (1, 1, 1),
            dtype=jnp.complex128,
        )
        geometry: CrystalGeometry = _geometry()
        basis: OrbitalBasis = _basis()
        if defect == "k_axis":
            eigenvectors = jnp.ones((2, 1, 1), dtype=jnp.complex128)
        elif defect == "basis_axis":
            eigenvectors = jnp.ones((1, 1, 2), dtype=jnp.complex128)
        else:
            basis = _basis((1,))

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
        Use the shared eager and compiled rejection helper for the runtime gate.
        """
        assert_rejects(
            make_diagonalized_bands,
            eigenvalues=jnp.zeros((1, 1), dtype=jnp.float64),
            eigenvectors=jnp.array([[[jnp.nan + 0.0j]]]),
            kpoints=jnp.zeros((1, 3), dtype=jnp.float64),
            geometry=_geometry(),
            basis=_basis(),
            match="eigenvectors finite",
        )


class TestMakeTBModel(chex.TestCase):
    """Validate :func:`~diffpes.types.make_tb_model`."""

    def test_constructs_complex_nine_field_model(self) -> None:
        """Normalize dtypes while preserving exact closed metadata.

        The case builds a complex one-orbital model with two reverse hoppings.

        Notes
        -----
        Check numerical shapes, complex precision, and exact orbital pairs.
        """
        model: TBModel = make_tb_model(**_model_arguments())

        chex.assert_shape(model.hopping_amplitudes, (2,))
        chex.assert_shape(model.onsite_energies, (1,))
        chex.assert_shape(model.soc_lambdas, (0,))
        assert model.hopping_amplitudes.dtype == jnp.complex128
        assert model.hopping_pairs == ((0, 0), (0, 0))

    @parameterized.named_parameters(
        (
            "open_hopping_list",
            "open",
            "Hermitian-closed",
        ),
        (
            "bad_pair_index",
            "pair_index",
            "hopping pair indices must be in",
        ),
        (
            "duplicate_hopping_record",
            "duplicate",
            "duplicate \\(i, j, R\\) hopping records",
        ),
        (
            "bad_shell_count",
            "shell_count",
            "soc_lambdas length must equal",
        ),
        (
            "noncontiguous_shell_ids",
            "shell_gap",
            "shell_index IDs must be contiguous",
        ),
        (
            "one_id_spans_two_atomic_shells",
            "mixed_shell_group",
            "each shell_index ID must map to one",
        ),
        (
            "one_atomic_shell_has_two_ids",
            "split_shell_group",
            "each \\(atom, n, l\\) group must map to one",
        ),
        (
            "spinor_without_spin",
            "spinor",
            "spinor models require",
        ),
    )
    def test_rejects_static_defects_eager_and_jit(
        self,
        defect: str,
        match: str,
    ) -> None:
        """Reject malformed connectivity, shell, and spin metadata.

        Parameterized cases isolate open records, duplicates, shells, and spin.

        Notes
        -----
        Match the public factory diagnostic for each static defect.
        """
        arguments: dict[str, object] = _model_arguments()
        if defect == "open":
            arguments["hopping_amplitudes"] = jnp.array(
                [1.0 + 0.0j],
                dtype=jnp.complex128,
            )
            arguments["hopping_pairs"] = ((0, 0),)
            arguments["hopping_cells"] = ((1, 0, 0),)
        elif defect == "pair_index":
            arguments["hopping_pairs"] = ((0, 1), (1, 0))
        elif defect == "duplicate":
            arguments["hopping_amplitudes"] = jnp.array(
                [1.0 + 2.0j, 3.0 + 4.0j, 1.0 - 2.0j, 3.0 - 4.0j],
                dtype=jnp.complex128,
            )
            arguments["hopping_pairs"] = ((0, 0),) * 4
            arguments["hopping_cells"] = (
                (1, 0, 0),
                (1, 0, 0),
                (-1, 0, 0),
                (-1, 0, 0),
            )
        elif defect == "shell_count":
            arguments["shell_index"] = (0,)
        elif defect == "shell_gap":
            arguments.update(
                {
                    "hopping_amplitudes": jnp.zeros(
                        (0,),
                        dtype=jnp.complex128,
                    ),
                    "onsite_energies": jnp.zeros(2, dtype=jnp.float64),
                    "soc_lambdas": jnp.zeros(3, dtype=jnp.float64),
                    "basis": _basis((0, 0)),
                    "hopping_pairs": (),
                    "hopping_cells": (),
                    "shell_index": (0, 2),
                }
            )
        elif defect == "mixed_shell_group":
            arguments.update(
                {
                    "hopping_amplitudes": jnp.zeros(
                        (0,),
                        dtype=jnp.complex128,
                    ),
                    "onsite_energies": jnp.zeros(2, dtype=jnp.float64),
                    "soc_lambdas": jnp.zeros(1, dtype=jnp.float64),
                    "geometry": _geometry(2),
                    "basis": _basis((0, 1)),
                    "hopping_pairs": (),
                    "hopping_cells": (),
                    "shell_index": (0, 0),
                }
            )
        elif defect == "split_shell_group":
            arguments.update(
                {
                    "hopping_amplitudes": jnp.zeros(
                        (0,),
                        dtype=jnp.complex128,
                    ),
                    "onsite_energies": jnp.zeros(2, dtype=jnp.float64),
                    "soc_lambdas": jnp.zeros(2, dtype=jnp.float64),
                    "basis": _basis((0, 0)),
                    "hopping_pairs": (),
                    "hopping_cells": (),
                    "shell_index": (0, 1),
                }
            )
        else:
            arguments["spinor"] = True

        assert_rejects(make_tb_model, match=match, **arguments)

    @parameterized.named_parameters(
        (
            "mismatched_conjugates",
            "amplitudes",
            "reverse hopping amplitudes must be complex conjugates",
        ),
        (
            "nonfinite_onsite",
            "onsite",
            "onsite energies finite",
        ),
        (
            "nonfinite_geometry",
            "geometry",
            "geometry positions finite",
        ),
    )
    def test_rejects_traced_defects_eager_and_jit(
        self,
        defect: str,
        match: str,
    ) -> None:
        """Reject numerical defects through runtime checks under compilation.

        Parameterized cases corrupt amplitudes, onsite values, or geometry.

        Notes
        -----
        Use the shared eager and compiled rejection helper for every case.
        """
        arguments: dict[str, object] = _model_arguments()
        if defect == "amplitudes":
            arguments["hopping_amplitudes"] = jnp.array(
                [1.0 + 2.0j, 1.0 - 3.0j],
                dtype=jnp.complex128,
            )
        elif defect == "onsite":
            arguments["onsite_energies"] = jnp.array(
                [jnp.nan],
                dtype=jnp.float64,
            )
        else:
            geometry: CrystalGeometry = _geometry()
            arguments["geometry"] = eqx.tree_at(
                lambda item: item.positions,
                geometry,
                jnp.full((1, 3), jnp.nan),
            )

        assert_rejects(make_tb_model, match=match, **arguments)

    def test_raw_constructor_reasserts_static_invariants(self) -> None:
        """Prevent direct construction from bypassing metadata closure.

        The case removes the reverse partner from otherwise valid arguments.

        Notes
        -----
        Require the raw module constructor to emit the closure diagnostic.
        """
        arguments: dict[str, object] = _model_arguments()
        arguments["hopping_amplitudes"] = jnp.array(
            [1.0 + 0.0j],
            dtype=jnp.complex128,
        )
        arguments["hopping_pairs"] = ((0, 0),)
        arguments["hopping_cells"] = ((1, 0, 0),)

        with pytest.raises(ValueError, match="Hermitian-closed"):
            TBModel(**arguments)

    @settings(max_examples=20, deadline=None)
    @given(
        cells=st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        real_parts=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=6,
            max_size=6,
        ),
        imaginary_parts=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=6,
            max_size=6,
        ),
    )
    def test_random_closed_lists_accept_and_corruption_rejects(
        self,
        cells: list[int],
        real_parts: list[float],
        imaginary_parts: list[float],
    ) -> None:
        """Accept random closed lists and reject one corrupted reverse entry.

        Hypothesis supplies unique cells and finite complex hopping components.

        Notes
        -----
        Build exact reverse entries, then perturb one amplitude and require rejection.
        """
        n_hoppings: int = len(cells)
        forward_values: list[complex] = [
            complex(real_parts[index], imaginary_parts[index])
            for index in range(n_hoppings)
        ]
        amplitudes: jax.Array = jnp.asarray(
            forward_values + [value.conjugate() for value in forward_values],
            dtype=jnp.complex128,
        )
        hopping_cells: Tuple[Tuple[int, int, int], ...] = tuple(
            [(cell, 0, 0) for cell in cells]
            + [(-cell, 0, 0) for cell in cells]
        )
        hopping_pairs: Tuple[Tuple[int, int], ...] = ((0, 0),) * (
            2 * n_hoppings
        )
        arguments: dict[str, object] = _model_arguments()
        arguments["hopping_amplitudes"] = amplitudes
        arguments["hopping_pairs"] = hopping_pairs
        arguments["hopping_cells"] = hopping_cells

        model: TBModel = make_tb_model(**arguments)
        chex.assert_trees_all_close(model.hopping_amplitudes, amplitudes)

        corrupted: jax.Array = amplitudes.at[0].add(0.25j)
        arguments["hopping_amplitudes"] = corrupted
        with pytest.raises(RuntimeError, match="complex conjugates"):
            make_tb_model(**arguments)
