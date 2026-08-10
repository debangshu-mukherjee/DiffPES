"""Validate HDF5 PyTree save/load.

Extended Summary
----------------
The tests exercise ``save_to_h5`` and ``load_from_h5`` for all registered
PyTree types. They cover spin and OAM variants and files with multiple
PyTrees. Round-trip tests compare numerical values within tolerance.
Error tests cover unknown types, missing groups, and invalid options.
The tests also verify applicable compression and data set options.

"""

import json
import tempfile
from dataclasses import Field, fields
from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Callable, Tuple
from jaxtyping import Array

import diffpes
from diffpes.inout import load_from_h5, save_to_h5
from diffpes.inout.hdf5 import _decode_static, _encode_static
from diffpes.types import (
    SelfEnergyModel,
    make_arpes_spectrum,
    make_band_structure,
    make_crystal_geometry,
    make_density_of_states,
    make_kpath_info,
    make_orbital_projection,
    make_self_energy_model,
)


def _cartesian_path(n_points: int) -> Tuple[Array, Array]:
    """PRIVATE: Build a straight path and its cumulative coordinate.

    Parameters
    ----------
    n_points : int
        Number of equally spaced path nodes.

    Returns
    -------
    path : Tuple[Array, Array]
        Cumulative coordinate and matching Cartesian three-vectors.
    """
    k_axis: Array = jnp.linspace(0.0, 1.0, n_points)
    kpoints: Array = jnp.stack(
        (k_axis, jnp.zeros_like(k_axis), jnp.zeros_like(k_axis)),
        axis=1,
    )
    return k_axis, kpoints


class TestSelfEnergyModel(chex.TestCase):
    """Round-trip the renamed self-energy carrier.

    :see: :class:`~diffpes.types.SelfEnergyModel`
    """

    def test_round_trip(self) -> None:
        """Preserve every traced and static model field through HDF5.

        The test covers a numerical grid model with its complete causal state.

        Notes
        -----
        The test saves the model, loads it by name, and compares the full tree.
        """
        model: SelfEnergyModel = make_self_energy_model(
            mode="grid",
            coefficients=jnp.array([-2.0, -1.0, 0.0]),
            energy_nodes_rel_fermi_ev=jnp.array([-4.0, 0.0, 4.0]),
            kk_domain_rel_fermi_ev=jnp.array([-4.0, 4.0]),
            tail_coefficients=jnp.array([-1.0, 1.0]),
            subtraction_point_rel_fermi_ev=0.0,
            tail_mode="power2",
        )
        path: Path
        loaded: Any
        td: str
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "self_energy.h5"
            save_to_h5(path, self_energy=model)
            loaded = load_from_h5(path, name="self_energy")
        assert isinstance(loaded, SelfEnergyModel)
        chex.assert_trees_all_equal(loaded, model)


class TestDensityOfStates(chex.TestCase):
    """Round-trip tests for DensityOfStates HDF5 serialization.

    Verifies that energy, total DOS, and Fermi energy arrays
    survive a save-then-load cycle with exact numerical
    fidelity.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip(self) -> None:
        """Verify DensityOfStates survives HDF5 round-trip.

        The test establishes the round trip contract for density of states with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create** a DensityOfStates with 100-point energy
           axis and uniform DOS.
        2. **Save** to a temporary HDF5 file.
        3. **Load** back by group name.

        **Expected assertions**

        All three array fields match to within 1e-12.
        """
        td: str

        dos: diffpes.types.DensityOfStates
        path: Path
        loaded: Any

        dos = make_density_of_states(
            energy=jnp.linspace(-10.0, 5.0, 100),
            total_dos=jnp.ones(100),
            fermi_energy=-1.5,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dos.h5"
            save_to_h5(path, dos=dos)
            loaded = load_from_h5(path, name="dos")
        chex.assert_trees_all_close(loaded.energy, dos.energy, atol=1e-12)
        chex.assert_trees_all_close(
            loaded.total_dos,
            dos.total_dos,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.fermi_energy,
            dos.fermi_energy,
            atol=1e-12,
        )


class TestBandStructure(chex.TestCase):
    """Round-trip tests for BandStructure HDF5 serialization.

    The tests verify preservation of the eigenvalues, k-points, weights,
    and Fermi energy through save and load operations.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip(self) -> None:
        """Verify BandStructure survives HDF5 round-trip.

        The test establishes the round trip contract for band structure with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create** a BandStructure with 10 k-points, 4 bands.
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        All four array fields match to within 1e-12.
        """
        td: str

        nk: int
        nb: int
        bands: diffpes.types.BandStructure
        path: Path
        loaded: Any

        nk, nb = 10, 4
        bands = make_band_structure(
            eigenvalues=jnp.linspace(-2.0, 0.5, nk * nb).reshape(nk, nb),
            kpoints=jnp.zeros((nk, 3)),
            fermi_energy=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bands.h5"
            save_to_h5(path, bands=bands)
            loaded = load_from_h5(path, name="bands")
        chex.assert_trees_all_close(
            loaded.eigenvalues,
            bands.eigenvalues,
            atol=1e-12,
        )
        chex.assert_trees_all_close(loaded.kpoints, bands.kpoints, atol=1e-12)
        chex.assert_trees_all_close(
            loaded.kpoint_weights,
            bands.kpoint_weights,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.fermi_energy,
            bands.fermi_energy,
            atol=1e-12,
        )


class TestArpesSpectrum(chex.TestCase):
    """Round-trip tests for ArpesSpectrum HDF5 serialization.

    Verifies intensity map and energy axis survive the
    save/load cycle.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip(self) -> None:
        """Verify ArpesSpectrum survives HDF5 round-trip.

        The test establishes the round trip contract for arpes spectrum with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create** an ArpesSpectrum with shape (20, 100).
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        Both array fields match to within 1e-12.
        """
        td: str

        spectrum: diffpes.types.ArpesSpectrum
        path: Path
        loaded: Any

        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((20, 100)),
            energy_axis=jnp.linspace(-3.0, 1.0, 100),
            k_axis=_cartesian_path(20)[0],
            kpoints_cart_inv_ang=_cartesian_path(20)[1],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spectrum.h5"
            save_to_h5(path, spectrum=spectrum)
            loaded = load_from_h5(path, name="spectrum")
        chex.assert_trees_all_close(
            loaded.intensity,
            spectrum.intensity,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.energy_axis,
            spectrum.energy_axis,
            atol=1e-12,
        )


class TestOrbitalProjection(chex.TestCase):
    """Round-trip tests for OrbitalProjection HDF5 serialization.

    Verifies projections array and Optional spin/oam fields
    (both None and non-None) survive the cycle.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip_none_optionals(self) -> None:
        """Verify OrbitalProjection with None spin/oam.

        The test establishes the round trip none optionals contract for orbital
        projection with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** with only projections (spin=None,
           oam=None).
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        Projections match; spin and oam are None.
        """
        td: str

        nk: int
        nb: int
        na: int
        orb: diffpes.types.OrbitalProjection
        path: Path
        loaded: Any

        nk, nb, na = 5, 3, 2
        orb = make_orbital_projection(
            projections=jnp.ones((nk, nb, na, 9)) * 0.1,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orb.h5"
            save_to_h5(path, orb=orb)
            loaded = load_from_h5(path, name="orb")
        chex.assert_trees_all_close(
            loaded.projections,
            orb.projections,
            atol=1e-12,
        )
        assert loaded.spin is None
        assert loaded.oam is None

    def test_round_trip_with_spin(self) -> None:
        """Verify OrbitalProjection with non-None spin.

        The test establishes the round trip with spin contract for orbital projection
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** with both projections and spin arrays.
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        Projections and spin match; oam is None.
        """
        td: str

        nk: int
        nb: int
        na: int
        orb: diffpes.types.OrbitalProjection
        path: Path
        loaded: Any

        nk, nb, na = 5, 3, 2
        orb = make_orbital_projection(
            projections=jnp.ones((nk, nb, na, 9)) * 0.1,
            spin=jnp.zeros((nk, nb, na, 6)),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orb_spin.h5"
            save_to_h5(path, orb=orb)
            loaded = load_from_h5(path, name="orb")
        chex.assert_trees_all_close(
            loaded.projections,
            orb.projections,
            atol=1e-12,
        )
        chex.assert_trees_all_close(loaded.spin, orb.spin, atol=1e-12)
        assert loaded.oam is None

    def test_round_trip_with_all(self) -> None:
        """Verify OrbitalProjection with spin and oam.

        The test establishes the round trip with all contract for orbital projection
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** with projections, spin, and oam.
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        All three array fields match.
        """
        td: str

        nk: int
        nb: int
        na: int
        orb: diffpes.types.OrbitalProjection
        path: Path
        loaded: Any

        nk, nb, na = 5, 3, 2
        orb = make_orbital_projection(
            projections=jnp.ones((nk, nb, na, 9)) * 0.1,
            spin=jnp.zeros((nk, nb, na, 6)),
            oam=jnp.ones((nk, nb, na, 3)) * 0.5,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orb_all.h5"
            save_to_h5(path, orb=orb)
            loaded = load_from_h5(path, name="orb")
        chex.assert_trees_all_close(
            loaded.projections,
            orb.projections,
            atol=1e-12,
        )
        chex.assert_trees_all_close(loaded.spin, orb.spin, atol=1e-12)
        chex.assert_trees_all_close(loaded.oam, orb.oam, atol=1e-12)


class TestKPathInfo(chex.TestCase):
    """Round-trip tests for KPathInfo HDF5 serialization.

    Verifies that integer arrays and the ``(mode, labels)``
    tuple aux_data survive the cycle.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip(self) -> None:
        """Verify KPathInfo survives HDF5 round-trip.

        The test establishes the round trip contract for k path info with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Create** with 3 symmetry labels.
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        The integer arrays, mode, and label strings match.
        """
        td: str

        kpath: diffpes.types.KPathInfo
        path: Path
        loaded: Any

        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            segments=2,
            mode="Line-mode",
            labels=("G", "M", "K"),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kpath.h5"
            save_to_h5(path, kpath=kpath)
            loaded = load_from_h5(path, name="kpath")
        chex.assert_trees_all_close(
            loaded.num_kpoints,
            kpath.num_kpoints,
            atol=0,
        )
        chex.assert_trees_all_close(
            loaded.label_indices,
            kpath.label_indices,
            atol=0,
        )
        assert loaded.mode == "Line-mode"
        assert loaded.labels == ("G", "M", "K")

    def test_loads_pre_migration_aux_format(self) -> None:
        """Load K-path metadata written by the NamedTuple-era codec.

        The test establishes the loads pre migration aux format contract for k path
        info with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        temporary_directory: str
        h5_file: h5py.File
        field: Field[Any]

        kpath: diffpes.types.KPathInfo
        value: Array
        loaded: Any

        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            segments=2,
            mode="Line-mode",
            labels=("G", "M", "K"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path: Path = Path(temporary_directory) / "legacy_kpath.h5"
            with h5py.File(path, "w") as h5_file:
                group: h5py.Group = h5_file.create_group("kpath")
                group.attrs["_pytree_type"] = "KPathInfo"
                group.attrs["_aux_data_json"] = json.dumps(
                    [
                        kpath.mode,
                        list(kpath.labels),
                        kpath.comment,
                        kpath.coordinate_mode,
                    ]
                )
                none_fields: list[str] = []
                for field in fields(kpath):
                    if bool(field.metadata.get("static", False)):
                        continue
                    value = getattr(kpath, field.name)
                    if value is None:
                        none_fields.append(field.name)
                    else:
                        group.create_dataset(
                            field.name, data=np.asarray(value)
                        )
                group.attrs["_none_fields"] = json.dumps(none_fields)

            loaded = load_from_h5(path, name="kpath")

        chex.assert_trees_all_equal(loaded.num_kpoints, kpath.num_kpoints)
        chex.assert_equal(loaded.mode, "Line-mode")
        chex.assert_equal(loaded.labels, ("G", "M", "K"))


class TestCrystalGeometry(chex.TestCase):
    """Round-trip tests for CrystalGeometry HDF5 serialization.

    Verifies that lattice, reciprocal lattice, coords,
    atom_counts arrays and the ``symbols`` tuple aux_data
    survive the cycle.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_round_trip(self) -> None:
        """Verify CrystalGeometry survives HDF5 round-trip.

        The test establishes the round trip contract for crystal geometry with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create** a cubic Si structure with 2 atoms.
        2. **Save** and **load** via HDF5.

        **Expected assertions**

        All four arrays and the symbols tuple match.
        """
        td: str

        lattice: Array
        positions: Array
        geo: diffpes.types.CrystalGeometry
        path: Path
        loaded: Any

        lattice = jnp.eye(3) * 5.43
        positions = jnp.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        geo = make_crystal_geometry(
            lattice=lattice,
            positions=positions,
            species=("Si", "Si"),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo.h5"
            save_to_h5(path, geo=geo)
            loaded = load_from_h5(path, name="geo")
        chex.assert_trees_all_close(loaded.lattice, geo.lattice, atol=1e-12)
        chex.assert_trees_all_close(
            loaded.reciprocal,
            geo.reciprocal,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.positions,
            geo.positions,
            atol=1e-12,
        )
        assert loaded.species == ("Si", "Si")


class TestSaveToH5(chex.TestCase):
    """Validate saving/loading multiple PyTrees in one file.

    Verifies multi-group HDF5 files and both loading modes
    (by name, load all).

    :see: :func:`~diffpes.inout.save_to_h5`
    """

    def test_save_load_multiple(self) -> None:
        """Verify two PyTrees coexist in one HDF5 file.

        The test establishes the save load multiple contract for multi py tree with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create** a BandStructure and an OrbitalProjection.
        2. **Save** both to one file under different names.
        3. **Load** each by name.

        **Expected assertions**

        Both round-trip correctly and independently.
        """
        td: str

        nk: int
        nb: int
        na: int
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        path: Path
        loaded_bands: Any
        loaded_orb: Any

        nk, nb, na = 8, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.linspace(-2.0, 0.5, nk * nb).reshape(nk, nb),
            kpoints=jnp.zeros((nk, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.ones((nk, nb, na, 9)) * 0.1,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "multi.h5"
            save_to_h5(path, bands=bands, orb=orb)
            loaded_bands = load_from_h5(path, name="bands")
            loaded_orb = load_from_h5(path, name="orb")
        chex.assert_trees_all_close(
            loaded_bands.eigenvalues,
            bands.eigenvalues,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded_orb.projections,
            orb.projections,
            atol=1e-12,
        )

    def test_load_all_returns_dict(self) -> None:
        """Verify load_from_h5 without name returns dict.

        The test establishes the load all returns dict contract for multi py tree with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Save** two PyTrees to one file.
        2. **Load** without specifying a name.

        **Expected assertions**

        Result is a dict with both group names as keys.
        """
        td: str

        nk: int
        nb: int
        bands: diffpes.types.BandStructure
        spectrum: diffpes.types.ArpesSpectrum
        path: Path
        loaded: Any

        nk, nb = 8, 3
        bands = make_band_structure(
            eigenvalues=jnp.linspace(-2.0, 0.5, nk * nb).reshape(nk, nb),
            kpoints=jnp.zeros((nk, 3)),
        )
        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((20, 50)),
            energy_axis=jnp.linspace(-3.0, 1.0, 50),
            k_axis=_cartesian_path(20)[0],
            kpoints_cart_inv_ang=_cartesian_path(20)[1],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "multi_all.h5"
            save_to_h5(path, bands=bands, spectrum=spectrum)
            loaded = load_from_h5(path)
        assert isinstance(loaded, dict)
        assert "bands" in loaded
        assert "spectrum" in loaded


class TestLoadFromH5(chex.TestCase):
    """Validate :func:`~diffpes.inout.load_from_h5`.

    Covers named-group recovery of a concrete Equinox carrier from a file
    produced by the public HDF5 writer.

    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_rejects_two_field_arpes_spectrum_schema(self) -> None:
        """Reject a legacy spectrum without Cartesian path geometry.

        The old two-field schema stores intensity and energy only. The loader
        must not invent a momentum path or sample frame.

        Notes
        -----
        Write the retired group layout directly and match the actionable
        schema-incompatibility diagnostic.
        """
        temporary_directory: str
        file_handle: h5py.File
        group: h5py.Group
        path: Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy_spectrum.h5"
            with h5py.File(path, "w") as file_handle:
                group = file_handle.create_group("spectrum")
                group.attrs["_pytree_type"] = "ArpesSpectrum"
                group.attrs["_aux_data_json"] = "null"
                group.attrs["_none_fields"] = "[]"
                group.create_dataset("intensity", data=jnp.ones((2, 3)))
                group.create_dataset(
                    "energy_axis", data=jnp.array([-1.0, 0.0, 1.0])
                )
            with pytest.raises(
                ValueError,
                match=(
                    "ArpesSpectrum HDF5 schema is incompatible.*"
                    "Geometry cannot be reconstructed"
                ),
            ):
                load_from_h5(path, name="spectrum")

    def test_load_named_band_structure(self) -> None:
        """Recover a named band structure without changing its arrays.

        The loaded carrier must preserve the eigenvalue and k-point arrays at
        exact float64 precision for this deterministic fixture.

        Notes
        -----
        Saves one two-k-point ``BandStructure``, loads the ``bands`` group by
        name, narrows its runtime type, and compares both array fields exactly.
        """
        temporary_directory: str
        expected: diffpes.types.BandStructure
        path: Path
        loaded: Any

        expected = make_band_structure(
            eigenvalues=jnp.array([[-1.0], [0.5]], dtype=jnp.float64),
            kpoints=jnp.zeros((2, 3), dtype=jnp.float64),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "named.h5"
            save_to_h5(path, bands=expected)
            loaded = load_from_h5(path, name="bands")
        assert isinstance(loaded, diffpes.types.BandStructure)
        chex.assert_trees_all_equal(loaded.eigenvalues, expected.eigenvalues)
        chex.assert_trees_all_equal(loaded.kpoints, expected.kpoints)


class TestErrorHandling(chex.TestCase):
    """Validate error conditions in save/load functions.

    The tests verify the applicable exceptions for invalid inputs and
    missing data.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_no_pytrees_raises(self) -> None:
        """Verify save_to_h5 with no kwargs raises ValueError.

        The test establishes the no pytrees raises contract for error handling with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Call** save_to_h5 with only a path, no PyTrees.

        **Expected assertions**

        The function raises ``ValueError``.
        """
        td: str

        path: Path

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.h5"
            with pytest.raises(ValueError, match="At least"):
                save_to_h5(path)

    def test_unsupported_type_raises(self) -> None:
        """Verify unregistered type raises TypeError.

        The test establishes the unsupported type raises contract for error handling
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Call** save_to_h5 with a plain tuple (not a
           registered PyTree).

        **Expected assertions**

        The function raises ``TypeError``.
        """
        td: str

        path: Path

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.h5"
            with pytest.raises(TypeError, match="Unsupported"):
                save_to_h5(path, bad=(1, 2, 3))

    def test_missing_group_raises(self) -> None:
        """Verify load with nonexistent name raises KeyError.

        The test establishes the missing group raises contract for error handling with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Save** a PyTree under name ``"a"``.
        2. **Load** with name ``"b"``.

        **Expected assertions**

        The function raises ``KeyError``.
        """
        td: str

        dos: diffpes.types.DensityOfStates
        path: Path

        dos = make_density_of_states(
            energy=jnp.linspace(-5.0, 5.0, 50),
            total_dos=jnp.ones(50),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.h5"
            save_to_h5(path, a=dos)
            with pytest.raises(KeyError, match="not found"):
                load_from_h5(path, name="b")

    def test_load_unknown_pytree_type_raises(self) -> None:
        """Verify loading a group with unknown _pytree_type raises TypeError.

        The test establishes the load unknown pytree type raises contract for error
        handling with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** an HDF5 file with a group that has
           _pytree_type = "UnknownType" (not in _PYTREE_REGISTRY).
        2. **Load** that group with load_from_h5(path, name="bad").

        **Expected assertions**

        The function raises ``TypeError`` with the unknown type in its message.
        """
        td: str
        f: h5py.File

        path: Path
        grp: h5py.Group

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unknown_type.h5"
            with h5py.File(path, "w") as f:
                grp = f.create_group("bad")
                grp.attrs["_pytree_type"] = "UnknownType"
                grp.attrs["_aux_data_json"] = "null"
                grp.attrs["_none_fields"] = "[]"
            with pytest.raises(TypeError, match="Unknown PyTree type"):
                load_from_h5(path, name="bad")


class TestDatasetFlags(chex.TestCase):
    """Validate HDF5 dataset storage flags in save_to_h5.

    :see: :func:`~diffpes.inout.save_to_h5`
    """

    def test_compression_flags_applied_to_arrays(self) -> None:
        """Verify application of storage flags to non-scalar data sets.

        The test establishes the compression flags applied to arrays contract for
        dataset flags with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** an ArpesSpectrum (array datasets) and
           BandStructure (including a scalar Fermi-energy dataset).
        2. **Save** both with compression/chunk/checksum flags.
        3. **Inspect** HDF5 dataset properties directly.
        4. **Round-trip load** to verify numerical integrity.

        **Expected assertions**

        - Array dataset has requested filter/chunk settings.
        - Scalar dataset remains uncompressed (safe handling).
        - Loaded spectrum matches original data.
        """
        td: str
        f: h5py.File

        spectrum: diffpes.types.ArpesSpectrum
        bands: diffpes.types.BandStructure
        path: Path
        ds: h5py.Dataset
        scalar_ds: h5py.Dataset
        loaded: Any

        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((12, 30)),
            energy_axis=jnp.linspace(-2.0, 1.0, 30),
            k_axis=_cartesian_path(12)[0],
            kpoints_cart_inv_ang=_cartesian_path(12)[1],
        )
        bands = make_band_structure(
            eigenvalues=jnp.asarray([[-1.0, 0.5]]),
            kpoints=jnp.zeros((1, 3)),
            fermi_energy=0.25,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "flags.h5"
            save_to_h5(
                path,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                chunks=True,
                spectrum=spectrum,
                bands=bands,
            )
            with h5py.File(path, "r") as f:
                ds = f["spectrum"]["intensity"]
                assert ds.compression == "gzip"
                assert ds.compression_opts == 4
                assert ds.shuffle
                assert ds.fletcher32
                assert ds.chunks is not None

                scalar_ds = f["bands"]["fermi_energy"]
                assert scalar_ds.shape == ()
                assert scalar_ds.compression is None
            loaded = load_from_h5(path, name="spectrum")
        chex.assert_trees_all_close(
            loaded.intensity,
            spectrum.intensity,
            atol=1e-12,
        )

    def test_compression_opts_without_compression_raises(self) -> None:
        """Verify invalid compression flag combination raises ValueError.

        The test establishes the compression opts without compression raises contract
        for dataset flags with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create** a simple DensityOfStates PyTree.
        2. **Call** ``save_to_h5`` with ``compression_opts`` only.

        **Expected assertions**

        The function raises ``ValueError`` with an explanatory message.
        """
        td: str

        dos: diffpes.types.DensityOfStates
        path: Path

        dos = make_density_of_states(
            energy=jnp.linspace(-5.0, 5.0, 50),
            total_dos=jnp.ones(50),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad_flags.h5"
            with pytest.raises(ValueError, match="compression_opts"):
                save_to_h5(
                    path,
                    compression_opts=4,
                    dos=dos,
                )


class TestStaticMetadataEncoding:
    """Validate generic static Equinox metadata encoding.

    :see: :func:`~diffpes.inout.save_to_h5`
    :see: :func:`~diffpes.inout.load_from_h5`
    """

    def test_encode_and_decode_round_trip(self) -> None:
        """Preserve nested tuple types through the generic JSON codec.

        The test establishes the encode and decode round trip contract for static
        metadata encoding with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        aux: Tuple[Tuple[int, int, int], Tuple[str, str]]
        encoded: Any
        decoded: Any

        aux = ((8, 8, 8), ("Fe", "Co"))
        encoded = _encode_static(aux)
        decoded = _decode_static(json.loads(json.dumps(encoded)))
        assert decoded == aux

    def test_encode_returns_tagged_json_mapping(self) -> None:
        """Encode tuples as JSON-compatible tagged mappings.

        The test establishes the encode returns tagged json mapping contract for static
        metadata encoding with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        aux: Tuple[Tuple[int, int, int], Tuple[str, str]]
        encoded: Any

        aux = ((4, 6, 8), ("H", "O"))
        encoded = _encode_static(aux)
        assert isinstance(encoded, dict)
        assert "__tuple__" in encoded
        json.dumps(encoded)
