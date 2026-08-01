"""Validate the Plan-05a depth and surface metadata carriers.

The tests exercise Plan-05 numerical and structural contracts.
"""

from collections.abc import Callable
from pathlib import Path

import chex
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any
from jaxtyping import Array

from diffpes.inout import load_from_h5, save_to_h5
from diffpes.tightb import diagonalize_tb, spin_double_model, tb_parameter_view
from diffpes.types import (
    ATTR_NONE,
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    SlabSpec,
    SurfaceCell,
    TBModel,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
    make_slab_spec,
    make_surface_cell,
    make_tb_model,
)
from tests._assertions import assert_rejects


def _geometry(n_atoms: int = 1) -> CrystalGeometry:
    """Return a cubic geometry for carrier validation."""
    species: tuple[str, ...] = tuple(
        "A" if index == 0 else "B" for index in range(n_atoms)
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((n_atoms, 3), dtype=jnp.float64),
        species=species,
    )
    return geometry


def _basis(n_orbitals: int) -> OrbitalBasis:
    """Return a spinless s-orbital basis on one atom."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
    )
    return basis


def _model(depths: Array | None) -> TBModel:
    """Return a diagonal model carrying optional orbital depths."""
    n_orbitals: int = 2 if depths is None else depths.shape[0]
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.arange(n_orbitals, dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=_geometry(),
        basis=_basis(n_orbitals),
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(-1,) * n_orbitals,
        depths=depths,
    )
    return model


def _surface_cell() -> SurfaceCell:
    """Return a primitive (001) surface cell."""
    surface_cell: SurfaceCell = make_surface_cell(
        in_plane_vectors=jnp.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=jnp.float64,
        ),
        stacking_vector=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
        rotation=jnp.eye(3, dtype=jnp.float64),
        interlayer_spacing_ang=1.0,
        miller=(0, 0, 1),
        in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
        stacking_coeffs=(0, 0, 1),
    )
    return surface_cell


class TestDepthCarrier:
    """Validate the optional depth leaf and its exact propagation."""

    def test_none_remains_bulk(self) -> None:
        """Preserve the absent carrier through native diagonalization.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        model: TBModel = _model(None)
        bands: DiagonalizedBands = diagonalize_tb(
            model,
            jnp.zeros((1, 3), dtype=jnp.float64),
        )

        assert model.depths is None
        assert bands.depths is None

    def test_rejects_invalid_depths_eager_and_jitted(self) -> None:
        """Reject negative, nonfinite, and wrong-length depth arrays.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        cases: tuple[tuple[Array, str], ...] = (
            (
                jnp.asarray([0.0, -1e-6], dtype=jnp.float64),
                "depths must be nonnegative",
            ),
            (
                jnp.asarray([0.0, jnp.nan], dtype=jnp.float64),
                "depths must be finite",
            ),
            (
                jnp.asarray([0.0], dtype=jnp.float64),
                "depths must have shape",
            ),
        )
        depths: Array
        match: str
        for depths, match in cases:
            arguments: dict[str, object] = {
                "hopping_amplitudes": jnp.zeros(
                    (0,),
                    dtype=jnp.complex128,
                ),
                "onsite_energies": jnp.zeros(2, dtype=jnp.float64),
                "soc_lambdas": jnp.zeros((0,), dtype=jnp.float64),
                "geometry": _geometry(),
                "basis": _basis(2),
                "hopping_pairs": (),
                "hopping_cells": (),
                "shell_index": (-1, -1),
                "depths": depths,
            }
            assert_rejects(make_tb_model, match=match, **arguments)

        assert_rejects(
            make_diagonalized_bands,
            eigenvalues=jnp.zeros((1, 2), dtype=jnp.float64),
            eigenvectors=jnp.ones((1, 2, 2), dtype=jnp.complex128),
            kpoints=jnp.zeros((1, 3), dtype=jnp.float64),
            geometry=_geometry(),
            basis=_basis(2),
            depths=jnp.asarray([0.0, -1e-6], dtype=jnp.float64),
            match="depths must be nonnegative",
        )

    @pytest.mark.parametrize("n_orbitals", [2, 5])
    def test_diagonalization_depth_jvp_and_vjp_are_identity(
        self,
        n_orbitals: int,
    ) -> None:
        """Verify 05a·D1 for frozen and generic registered directions.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = jnp.linspace(
            0.0,
            3.0,
            n_orbitals,
            dtype=jnp.float64,
        )
        direction: Array = jnp.linspace(
            -0.75,
            1.25,
            n_orbitals,
            dtype=jnp.float64,
        )

        def propagate(candidate: Array) -> Array:
            """Return depths after the native diagonalization seam."""
            output: Array | None = diagonalize_tb(
                _model(candidate),
                jnp.zeros((1, 3), dtype=jnp.float64),
            ).depths
            assert output is not None
            return output

        primal: Array
        tangent: Array
        primal, tangent = jax.jvp(
            propagate,
            (depths,),
            (direction,),
        )
        output: Array
        pullback: object
        output, pullback = jax.vjp(propagate, depths)
        cotangent: Array = pullback(direction)[0]
        jacobian: Array = jax.jacfwd(propagate)(depths)

        chex.assert_trees_all_equal(primal, depths)
        chex.assert_trees_all_equal(output, depths)
        chex.assert_trees_all_equal(tangent, direction)
        chex.assert_trees_all_equal(cotangent, direction)
        chex.assert_trees_all_equal(
            jacobian,
            jnp.eye(n_orbitals, dtype=jnp.float64),
        )

    def test_model_rebuilders_preserve_or_duplicate_depths(self) -> None:
        """Prevent optimizer and spin builders from silently dropping depths.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = jnp.asarray([0.0, 2.5], dtype=jnp.float64)
        model: TBModel = _model(depths)
        parameters: Array
        rebuild: Callable[[Array], TBModel]
        parameters, rebuild = tb_parameter_view(model)
        rebuilt: TBModel = rebuild(parameters)
        doubled: TBModel = spin_double_model(model)

        assert rebuilt.depths is not None
        assert doubled.depths is not None
        chex.assert_trees_all_equal(rebuilt.depths, depths)
        chex.assert_trees_all_equal(
            doubled.depths,
            jnp.concatenate((depths, depths)),
        )


class TestSurfaceCell:
    """Validate the traced surface frame and exact integer metadata."""

    def test_constructs_and_is_a_pytree(self) -> None:
        """Preserve surface leaves and static coefficients on reconstruction.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        surface_cell: SurfaceCell = _surface_cell()
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(surface_cell)
        restored: SurfaceCell = jax.tree_util.tree_unflatten(tree, leaves)

        chex.assert_trees_all_equal(restored, surface_cell)
        assert restored.miller == (0, 0, 1)
        assert restored.stacking_coeffs == (0, 0, 1)

    def test_rejects_nonorthogonal_rotation_eager_and_jitted(self) -> None:
        """Keep the rotation check active under compilation.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        assert_rejects(
            make_surface_cell,
            in_plane_vectors=jnp.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=jnp.float64,
            ),
            stacking_vector=jnp.asarray(
                [0.0, 0.0, 1.0],
                dtype=jnp.float64,
            ),
            rotation=jnp.diag(jnp.asarray([1.0, 1.0, 2.0])),
            interlayer_spacing_ang=1.0,
            miller=(0, 0, 1),
            in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
            stacking_coeffs=(0, 0, 1),
            match="rotation must be orthogonal",
        )

    def test_rejects_nonprimitive_stacking_coefficients(self) -> None:
        """Require exact unit stacking advance along the Miller normal.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        with pytest.raises(ValueError, match="must equal one"):
            make_surface_cell(
                in_plane_vectors=jnp.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=jnp.float64,
                ),
                stacking_vector=jnp.asarray(
                    [0.0, 0.0, 2.0],
                    dtype=jnp.float64,
                ),
                rotation=jnp.eye(3, dtype=jnp.float64),
                interlayer_spacing_ang=1.0,
                miller=(0, 0, 1),
                in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
                stacking_coeffs=(0, 0, 2),
            )


class TestSlabSpec:
    """Validate slab choices and provenance mappings."""

    def test_factory_validates_species_and_provenance(self) -> None:
        """Store valid static provenance and reject unknown terminations.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        geometry: CrystalGeometry = _geometry(2)
        slab_spec: SlabSpec = make_slab_spec(
            surface_cell=_surface_cell(),
            geometry=geometry,
            thickness_ang=10.0,
            vacuum_ang=15.0,
            fine=(0.0, 0.25),
            termination=("A", "B"),
            n_layers=2,
            bulk_atom_of_slab_atom=(0, 1, 0, 1),
            layer_of_slab_atom=(0, 0, 1, 1),
        )

        assert slab_spec.termination == ("A", "B")
        assert slab_spec.bulk_atom_of_slab_atom == (0, 1, 0, 1)
        with pytest.raises(ValueError, match="termination species"):
            make_slab_spec(
                surface_cell=_surface_cell(),
                geometry=geometry,
                thickness_ang=10.0,
                vacuum_ang=15.0,
                fine=(0.0, 0.25),
                termination=("A", "C"),
                n_layers=2,
                bulk_atom_of_slab_atom=(0, 1),
                layer_of_slab_atom=(0, 1),
            )

    def test_hdf5_round_trip_preserves_plan05a_carriers(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist depth leaves, surface arrays, and slab static metadata.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = jnp.asarray([0.0, 2.5], dtype=jnp.float64)
        model: TBModel = _model(depths)
        bands: DiagonalizedBands = diagonalize_tb(
            model,
            jnp.zeros((1, 3), dtype=jnp.float64),
        )
        slab_spec: SlabSpec = make_slab_spec(
            surface_cell=_surface_cell(),
            geometry=_geometry(),
            thickness_ang=8.0,
            vacuum_ang=12.0,
            fine=(0.0, 0.0),
            termination=("A", "A"),
            n_layers=2,
            bulk_atom_of_slab_atom=(0, 0),
            layer_of_slab_atom=(0, 1),
        )
        path: Path = tmp_path / "plan05a.h5"

        save_to_h5(path, model=model, bands=bands, slab_spec=slab_spec)
        restored: dict[str, eqx.Module] = load_from_h5(path)

        assert eqx.tree_equal(restored["model"], model)
        assert eqx.tree_equal(restored["bands"], bands)
        assert eqx.tree_equal(restored["slab_spec"], slab_spec)

    def test_legacy_hdf5_without_depth_dataset_loads_as_bulk(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify legacy missing depths map to the bulk sentinel.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        output: Any
        path: Path = tmp_path / "legacy_bulk.h5"
        save_to_h5(
            path,
            model=_model(jnp.asarray([0.0, 2.5], dtype=jnp.float64)),
        )
        with h5py.File(path, "r+") as output:
            del output["model"]["depths"]
            output["model"].attrs[ATTR_NONE] = '["orbital_positions"]'

        restored: TBModel = load_from_h5(path, name="model")

        assert restored.depths is None
