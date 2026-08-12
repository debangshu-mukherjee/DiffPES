"""Validate shared electronic-structure geometry checks.

The tests exercise common depth, geometry, and surface metadata contracts.
"""

from collections.abc import Callable
from pathlib import Path

import chex
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64

from diffpes.constants import (
    ATTR_NONE,
)
from diffpes.inout import load_from_h5, save_to_h5
from diffpes.tightb import diagonalize_tb, spin_double_model, tb_parameter_view
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    PyTreeDef,
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
    """PRIVATE: Return a cubic geometry for carrier validation.

    Parameters
    ----------
    n_atoms : int, optional
        Number of atoms, all placed at the origin. Default 1.

    Returns
    -------
    geometry : CrystalGeometry
        Identity cubic lattice in Angstrom with species ``A`` for the
        first atom and ``B`` for every further atom.

    Notes
    -----
    Keeps positions degenerate at the origin because the metadata
    carriers under test never consume interatomic distances.
    """
    species: Tuple[str, ...] = tuple(
        "A" if index == 0 else "B" for index in range(n_atoms)
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((n_atoms, 3), dtype=jnp.float64),
        species=species,
    )
    return geometry


def _basis(n_orbitals: int) -> OrbitalBasis:
    """PRIVATE: Return a spinless s-orbital basis on one atom.

    Parameters
    ----------
    n_orbitals : int
        Number of identical 1s orbitals to place on atom 0.

    Returns
    -------
    basis : OrbitalBasis
        Spinless basis with ``n_orbitals`` copies of the (n=1, l=0,
        m=0) orbital on the one atom.

    Notes
    -----
    Repeats one quantum-number tuple so the basis length can track the
    depth arrays under test without further structure.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
    )
    return basis


def _model(depths: Float64[Array, " norb"] | None) -> TBModel:
    """PRIVATE: Return a diagonal model carrying optional orbital depths.

    Parameters
    ----------
    depths : Float64[Array, " norb"] | None
        Optional per-orbital depths below the surface in Angstrom.
        ``None`` builds a bulk model with two orbitals.

    Returns
    -------
    model : TBModel
        Hopping-free model whose onsite energies are ``0, 1, ...`` eV,
        with the ``depths`` leaf attached verbatim.

    Notes
    -----
    Sizes the basis from ``depths`` when present and passes empty
    hopping arrays. Marks every orbital with shell index -1, so the
    depth leaf is the only slab metadata in play.
    """
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
    """PRIVATE: Return a primitive (001) surface cell.

    Returns
    -------
    surface_cell : SurfaceCell
        Identity-oriented cell with unit in-plane vectors along x and
        y and the stacking vector along z. Interlayer spacing is
        1.0 Angstrom, with Miller indices (0, 0, 1) and matching
        integer coefficients.

    Notes
    -----
    Mirrors the trivial cubic surface so carrier tests can assert
    exact round trips of every stored field.
    """
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
    """Validate the optional depth leaf and its exact propagation.

    The cases propagate optional depths through PyTrees, derivatives, model
    rebuilders, and invalid-value guards.
    """

    def test_none_remains_bulk(self) -> None:
        """Preserve the absent carrier through native diagonalization.

        Exercise this slab condition with fixed fixtures.

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

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        cases: Tuple[Tuple[Float64[Array, " norb"], str], ...] = (
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
        depths: Float64[Array, " norb"]
        match: str
        for depths, match in cases:
            arguments: Dict[str, object] = {
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
        """Verify frozen and generic registered slab directions.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Float64[Array, " norb"] = jnp.linspace(
            0.0,
            3.0,
            n_orbitals,
            dtype=jnp.float64,
        )
        direction: Float64[Array, " norb"] = jnp.linspace(
            -0.75,
            1.25,
            n_orbitals,
            dtype=jnp.float64,
        )

        def propagate(
            candidate: Float64[Array, " norb"],
        ) -> Float64[Array, " norb"]:
            """Return depths after the native diagonalization seam."""
            output: Float64[Array, " norb"] | None = diagonalize_tb(
                _model(candidate),
                jnp.zeros((1, 3), dtype=jnp.float64),
            ).depths
            assert output is not None
            return output

        primal: Float64[Array, " norb"]
        tangent: Float64[Array, " norb"]
        primal, tangent = jax.jvp(
            propagate,
            (depths,),
            (direction,),
        )
        output: Float64[Array, " norb"]
        pullback: object
        output, pullback = jax.vjp(propagate, depths)
        cotangent: Float64[Array, " norb"] = pullback(direction)[0]
        jacobian: Float64[Array, "norb norb"] = jax.jacfwd(propagate)(depths)

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

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Float64[Array, " 2"] = jnp.asarray(
            [0.0, 2.5], dtype=jnp.float64
        )
        model: TBModel = _model(depths)
        parameters: Float64[Array, " n_par"]
        rebuild: Callable[[Float64[Array, " n_par"]], TBModel]
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


class TestSurfaceCellIntegration:
    """Validate the traced surface frame and exact integer metadata.

    The cases inspect PyTree structure and reject invalid rotations and
    nonprimitive stacking coefficients.
    """

    def test_constructs_and_is_a_pytree(self) -> None:
        """Preserve surface leaves and static coefficients on reconstruction.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        surface_cell: SurfaceCell = _surface_cell()
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(surface_cell)
        restored: SurfaceCell = jax.tree_util.tree_unflatten(tree, leaves)

        chex.assert_trees_all_equal(restored, surface_cell)
        assert restored.miller == (0, 0, 1)
        assert restored.stacking_coeffs == (0, 0, 1)

    def test_rejects_nonorthogonal_rotation_eager_and_jitted(self) -> None:
        """Keep the rotation check active under compilation.

        Exercise this slab condition with fixed fixtures.

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

        Exercise this slab condition with fixed fixtures.

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


class TestSlabSpecIntegration:
    """Validate slab choices and provenance mappings.

    The cases validate provenance metadata and compare modern and legacy HDF5
    round trips.
    """

    def test_factory_validates_species_and_provenance(self) -> None:
        """Store valid static provenance and reject unknown terminations.

        Exercise this slab condition with fixed fixtures.

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

    def test_hdf5_round_trip_preserves_slab_metadata_carriers(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist depth leaves, surface arrays, and slab static metadata.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Float64[Array, " 2"] = jnp.asarray(
            [0.0, 2.5], dtype=jnp.float64
        )
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
        path: Path = tmp_path / "slab_metadata.h5"

        save_to_h5(path, model=model, bands=bands, slab_spec=slab_spec)
        restored: Dict[str, eqx.Module] = load_from_h5(path)

        assert eqx.tree_equal(restored["model"], model)
        assert eqx.tree_equal(restored["bands"], bands)
        assert eqx.tree_equal(restored["slab_spec"], slab_spec)

    def test_legacy_hdf5_without_depth_dataset_loads_as_bulk(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify legacy missing depths map to the bulk sentinel.

        Exercise this slab condition with fixed fixtures.

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
