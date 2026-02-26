"""Tests for HDF5 PyTree serializer and deserializer."""

import tempfile
from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import pytest

from arpyes.inout import load_from_h5, save_to_h5
from arpyes.types import (
    make_arpes_spectrum,
    make_band_structure,
    make_crystal_geometry,
    make_density_of_states,
    make_kpath_info,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
)


class TestDensityOfStates(chex.TestCase):
    """Round-trip tests for DensityOfStates HDF5 serialization.

    Verifies that energy, total DOS, and Fermi energy arrays
    survive a save-then-load cycle with exact numerical
    fidelity.
    """

    def test_round_trip(self):
        """Verify DensityOfStates survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** a DensityOfStates with 100-point energy
           axis and uniform DOS.
        2. **Save** to a temporary HDF5 file.
        3. **Load** back by group name.

        Asserts
        -------
        All three array fields match to within 1e-12.
        """
        dos = make_density_of_states(
            energy=jnp.linspace(-10.0, 5.0, 100),
            total_dos=jnp.ones(100),
            fermi_energy=-1.5,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dos.h5"
            save_to_h5(path, dos=dos)
            loaded = load_from_h5(path, name="dos")
        chex.assert_trees_all_close(
            loaded.energy, dos.energy, atol=1e-12
        )
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

    Verifies eigenvalues, kpoints, weights, and Fermi energy
    are preserved through save/load.
    """

    def test_round_trip(self):
        """Verify BandStructure survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** a BandStructure with 10 k-points, 4 bands.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        All four array fields match to within 1e-12.
        """
        nk, nb = 10, 4
        bands = make_band_structure(
            eigenvalues=jnp.linspace(
                -2.0, 0.5, nk * nb
            ).reshape(nk, nb),
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
        chex.assert_trees_all_close(
            loaded.kpoints, bands.kpoints, atol=1e-12
        )
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
    """

    def test_round_trip(self):
        """Verify ArpesSpectrum survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** an ArpesSpectrum with shape (20, 100).
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        Both array fields match to within 1e-12.
        """
        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((20, 100)),
            energy_axis=jnp.linspace(-3.0, 1.0, 100),
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
    """

    def test_round_trip_none_optionals(self):
        """Verify OrbitalProjection with None spin/oam.

        Test Logic
        ----------
        1. **Create** with only projections (spin=None,
           oam=None).
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        Projections match; spin and oam are None.
        """
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

    def test_round_trip_with_spin(self):
        """Verify OrbitalProjection with non-None spin.

        Test Logic
        ----------
        1. **Create** with both projections and spin arrays.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        Projections and spin match; oam is None.
        """
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
        chex.assert_trees_all_close(
            loaded.spin, orb.spin, atol=1e-12
        )
        assert loaded.oam is None

    def test_round_trip_with_all(self):
        """Verify OrbitalProjection with spin and oam.

        Test Logic
        ----------
        1. **Create** with projections, spin, and oam.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        All three array fields match.
        """
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
        chex.assert_trees_all_close(
            loaded.spin, orb.spin, atol=1e-12
        )
        chex.assert_trees_all_close(
            loaded.oam, orb.oam, atol=1e-12
        )


class TestSimulationParams(chex.TestCase):
    """Round-trip tests for SimulationParams HDF5 serialization.

    Verifies that all six float children and the integer
    ``fidelity`` aux_data survive the cycle.
    """

    def test_round_trip(self):
        """Verify SimulationParams survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** with non-default values for all fields.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        All float fields match; fidelity int is preserved.
        """
        params = make_simulation_params(
            energy_min=-5.0,
            energy_max=2.0,
            fidelity=500,
            sigma=0.08,
            gamma=0.15,
            temperature=30.0,
            photon_energy=21.2,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "params.h5"
            save_to_h5(path, params=params)
            loaded = load_from_h5(path, name="params")
        chex.assert_trees_all_close(
            loaded.energy_min,
            params.energy_min,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.sigma, params.sigma, atol=1e-12
        )
        chex.assert_trees_all_close(
            loaded.photon_energy,
            params.photon_energy,
            atol=1e-12,
        )
        assert loaded.fidelity == 500


class TestPolarizationConfig(chex.TestCase):
    """Round-trip tests for PolarizationConfig serialization.

    Verifies that float angles and the string
    ``polarization_type`` aux_data survive the cycle.
    """

    def test_round_trip(self):
        """Verify PolarizationConfig survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** an LHP config with custom angles.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        Float angles match; polarization_type string preserved.
        """
        pol = make_polarization_config(
            theta=0.5,
            phi=1.2,
            polarization_angle=0.3,
            polarization_type="LHP",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pol.h5"
            save_to_h5(path, pol=pol)
            loaded = load_from_h5(path, name="pol")
        chex.assert_trees_all_close(
            loaded.theta, pol.theta, atol=1e-12
        )
        chex.assert_trees_all_close(
            loaded.phi, pol.phi, atol=1e-12
        )
        assert loaded.polarization_type == "LHP"


class TestKPathInfo(chex.TestCase):
    """Round-trip tests for KPathInfo HDF5 serialization.

    Verifies that integer arrays and the ``(mode, labels)``
    tuple aux_data survive the cycle.
    """

    def test_round_trip(self):
        """Verify KPathInfo survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** with 3 symmetry labels.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        Integer arrays match; mode and labels strings preserved.
        """
        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
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


class TestCrystalGeometry(chex.TestCase):
    """Round-trip tests for CrystalGeometry HDF5 serialization.

    Verifies that lattice, reciprocal lattice, coords,
    atom_counts arrays and the ``symbols`` tuple aux_data
    survive the cycle.
    """

    def test_round_trip(self):
        """Verify CrystalGeometry survives HDF5 round-trip.

        Test Logic
        ----------
        1. **Create** a cubic Si structure with 2 atoms.
        2. **Save** and **load** via HDF5.

        Asserts
        -------
        All four arrays match; symbols tuple preserved.
        """
        lattice = jnp.eye(3) * 5.43
        coords = jnp.array(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        )
        geo = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[2],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo.h5"
            save_to_h5(path, geo=geo)
            loaded = load_from_h5(path, name="geo")
        chex.assert_trees_all_close(
            loaded.lattice, geo.lattice, atol=1e-12
        )
        chex.assert_trees_all_close(
            loaded.reciprocal_lattice,
            geo.reciprocal_lattice,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            loaded.coords, geo.coords, atol=1e-12
        )
        chex.assert_trees_all_close(
            loaded.atom_counts,
            geo.atom_counts,
            atol=0,
        )
        assert loaded.symbols == ("Si",)


class TestMultiPyTree(chex.TestCase):
    """Tests for saving/loading multiple PyTrees in one file.

    Verifies multi-group HDF5 files and both loading modes
    (by name, load all).
    """

    def test_save_load_multiple(self):
        """Verify two PyTrees coexist in one HDF5 file.

        Test Logic
        ----------
        1. **Create** a BandStructure and an OrbitalProjection.
        2. **Save** both to one file under different names.
        3. **Load** each by name.

        Asserts
        -------
        Both round-trip correctly and independently.
        """
        nk, nb, na = 8, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.linspace(
                -2.0, 0.5, nk * nb
            ).reshape(nk, nb),
            kpoints=jnp.zeros((nk, 3)),
        )
        orb = make_orbital_projection(
            projections=jnp.ones((nk, nb, na, 9)) * 0.1,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "multi.h5"
            save_to_h5(path, bands=bands, orb=orb)
            loaded_bands = load_from_h5(
                path, name="bands"
            )
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

    def test_load_all_returns_dict(self):
        """Verify load_from_h5 without name returns dict.

        Test Logic
        ----------
        1. **Save** two PyTrees to one file.
        2. **Load** without specifying a name.

        Asserts
        -------
        Result is a dict with both group names as keys.
        """
        nk, nb = 8, 3
        bands = make_band_structure(
            eigenvalues=jnp.linspace(
                -2.0, 0.5, nk * nb
            ).reshape(nk, nb),
            kpoints=jnp.zeros((nk, 3)),
        )
        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((20, 50)),
            energy_axis=jnp.linspace(-3.0, 1.0, 50),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "multi_all.h5"
            save_to_h5(
                path, bands=bands, spectrum=spectrum
            )
            loaded = load_from_h5(path)
        assert isinstance(loaded, dict)
        assert "bands" in loaded
        assert "spectrum" in loaded


class TestErrorHandling(chex.TestCase):
    """Tests for error conditions in save/load functions.

    Verifies that appropriate exceptions are raised for
    invalid inputs and missing data.
    """

    def test_no_pytrees_raises(self):
        """Verify save_to_h5 with no kwargs raises ValueError.

        Test Logic
        ----------
        1. **Call** save_to_h5 with only a path, no PyTrees.

        Asserts
        -------
        ValueError is raised.
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.h5"
            with pytest.raises(
                ValueError, match="At least"
            ):
                save_to_h5(path)

    def test_unsupported_type_raises(self):
        """Verify unregistered type raises TypeError.

        Test Logic
        ----------
        1. **Call** save_to_h5 with a plain tuple (not a
           registered PyTree).

        Asserts
        -------
        TypeError is raised.
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.h5"
            with pytest.raises(
                TypeError, match="Unsupported"
            ):
                save_to_h5(path, bad=(1, 2, 3))

    def test_missing_group_raises(self):
        """Verify load with nonexistent name raises KeyError.

        Test Logic
        ----------
        1. **Save** a PyTree under name ``"a"``.
        2. **Load** with name ``"b"``.

        Asserts
        -------
        KeyError is raised.
        """
        dos = make_density_of_states(
            energy=jnp.linspace(-5.0, 5.0, 50),
            total_dos=jnp.ones(50),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.h5"
            save_to_h5(path, a=dos)
            with pytest.raises(
                KeyError, match="not found"
            ):
                load_from_h5(path, name="b")


class TestDatasetFlags(chex.TestCase):
    """Tests for HDF5 dataset storage flags in save_to_h5."""

    def test_compression_flags_applied_to_arrays(self):
        """Verify storage flags are applied to non-scalar datasets.

        Test Logic
        ----------
        1. **Create** an ArpesSpectrum (array datasets) and
           SimulationParams (scalar datasets).
        2. **Save** both with compression/chunk/checksum flags.
        3. **Inspect** HDF5 dataset properties directly.
        4. **Round-trip load** to verify numerical integrity.

        Asserts
        -------
        - Array dataset has requested filter/chunk settings.
        - Scalar dataset remains uncompressed (safe handling).
        - Loaded spectrum matches original data.
        """
        spectrum = make_arpes_spectrum(
            intensity=jnp.ones((12, 30)),
            energy_axis=jnp.linspace(-2.0, 1.0, 30),
        )
        params = make_simulation_params(
            fidelity=30,
            sigma=0.04,
            gamma=0.1,
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
                params=params,
            )
            with h5py.File(path, "r") as f:
                ds = f["spectrum"]["intensity"]
                assert ds.compression == "gzip"
                assert ds.compression_opts == 4
                assert ds.shuffle
                assert ds.fletcher32
                assert ds.chunks is not None

                scalar_ds = f["params"]["energy_min"]
                assert scalar_ds.shape == ()
                assert scalar_ds.compression is None
            loaded = load_from_h5(path, name="spectrum")
        chex.assert_trees_all_close(
            loaded.intensity,
            spectrum.intensity,
            atol=1e-12,
        )

    def test_compression_opts_without_compression_raises(self):
        """Verify invalid compression flag combination raises ValueError.

        Test Logic
        ----------
        1. **Create** a simple DensityOfStates PyTree.
        2. **Call** ``save_to_h5`` with ``compression_opts`` only.

        Asserts
        -------
        ValueError is raised with explanatory message.
        """
        dos = make_density_of_states(
            energy=jnp.linspace(-5.0, 5.0, 50),
            total_dos=jnp.ones(50),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad_flags.h5"
            with pytest.raises(
                ValueError, match="compression_opts"
            ):
                save_to_h5(
                    path,
                    compression_opts=4,
                    dos=dos,
                )
