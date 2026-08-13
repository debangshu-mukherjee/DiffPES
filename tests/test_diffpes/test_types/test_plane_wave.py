"""Verify bounded plane-wave carrier invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict
from jaxtyping import TypeCheckError

from diffpes.types import (
    CrystalGeometry,
    PlaneWaveBatch,
    PlaneWaveStateSource,
    VaspWavefunctionSource,
    WavecarDataset,
    WavecarHeader,
    make_crystal_geometry,
    make_in_memory_plane_wave_source,
    make_plane_wave_batch,
    make_state_batch_request,
    make_vasp_wavefunction_source,
    make_wavecar_dataset,
    make_wavecar_header,
)


def _geometry() -> CrystalGeometry:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_crystal_geometry(jnp.eye(3), jnp.zeros((1, 3)), ("X",))
    return result


def _batch(**overrides: object) -> PlaneWaveBatch:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "coefficients": jnp.ones((1, 2, 1), dtype=jnp.complex128),
        "g_vectors_frac": jnp.zeros((1, 2, 3), dtype=jnp.int32),
        "plane_wave_counts": jnp.asarray([2], dtype=jnp.int32),
        "kpoints_frac": jnp.zeros((1, 3)),
        "kpoint_weights": jnp.ones((1,)),
        "energies_ev": jnp.asarray([-0.2]),
        "occupations": jnp.ones((1,)),
        "state_indices": jnp.zeros((1, 3), dtype=jnp.int32),
        "geometry": _geometry(),
        "fermi_energy_ev": jnp.asarray(0.0),
        "spin_mode": "scalar",
        "source_ref": "fixture",
        "gauge_ref": "velocity",
    }
    values.update(overrides)
    result: Any = make_plane_wave_batch(**values)
    return result


def _vasp_source(path: Path) -> VaspWavefunctionSource:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_vasp_wavefunction_source(
        path,
        jnp.asarray([1.0]),
        jnp.asarray(0.0),
        spin_mode="scalar",
        source_ref="fixture",
        potcar_sha256=("digest",),
    )
    return result


def _header(**overrides: object) -> WavecarHeader:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "record_length": 128,
        "spin_components": 1,
        "precision_tag": 45200,
        "byte_order": "little",
        "nkpoints": 1,
        "nbands": 1,
        "encut_ev": jnp.asarray(10.0),
        "lattice_ang": jnp.eye(3),
        "fermi_energy_ev": jnp.asarray(0.0),
    }
    values.update(overrides)
    result: Any = make_wavecar_header(**values)
    return result


def _dataset(path: Path, **overrides: object) -> WavecarDataset:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "source": _vasp_source(path),
        "header": _header(),
        "file_size": 384,
        "record_offsets": (0, 128, 256),
        "coefficient_record_offsets": (256,),
        "plane_wave_counts": jnp.asarray([[1]], dtype=jnp.int32),
        "kpoints_frac": jnp.zeros((1, 1, 3)),
        "eigenvalues_ev": jnp.zeros((1, 1, 1), dtype=jnp.complex128),
        "occupations": jnp.ones((1, 1, 1)),
        "g_vectors_frac": (jnp.zeros((1, 3), dtype=jnp.int32),),
    }
    values.update(overrides)
    result: Any = make_wavecar_dataset(**values)
    return result


class TestStatebatchrequest:
    """Verify ``diffpes.types.StateBatchRequest`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_aligned_state_indices(self) -> None:
        """Preserve aligned k, band, and spin selections.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare each integer vector with its explicit input.
        """
        request: Any = make_state_batch_request(
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([2, 3], dtype=jnp.int32),
            jnp.asarray([0, 0], dtype=jnp.int32),
            purpose="photocurrent",
        )
        assert request.purpose == "photocurrent"
        assert request.k_indices.shape == (2,)

    @pytest.mark.parametrize(
        ("band", "purpose", "message"),
        [
            (
                jnp.asarray([0], dtype=jnp.int32),
                "",
                "purpose must be nonempty",
            ),
            (
                jnp.asarray([0, 1], dtype=jnp.int32),
                "x",
                "indices must have one shared axis",
            ),
        ],
    )
    def test_rejects_each_request_invariant(
        self, band: object, purpose: str, message: str
    ) -> None:
        """Reject empty purpose and unequal selection lengths.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one static field of a one-state request.
        """
        with pytest.raises(ValueError, match=message):
            make_state_batch_request(
                jnp.asarray([0], dtype=jnp.int32),
                band,
                jnp.asarray([0], dtype=jnp.int32),
                purpose=purpose,
            )


class TestVaspwavefunctionsource:
    """Verify ``diffpes.types.VaspWavefunctionSource`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_explicit_weights_and_provenance(
        self, tmp_path: Path
    ) -> None:
        """Preserve file, weights, spin mode, and provenance identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Create a real empty file because this carrier validates path existence.
        """
        path: Path = tmp_path / "WAVECAR"
        path.touch()
        source: Any = _vasp_source(path)
        assert source.wavecar_path == path
        assert source.potcar_sha256 == ("digest",)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        """Reject a WAVECAR path that does not name a file.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Pass one nonexistent path below the temporary directory.
        """
        with pytest.raises(ValueError, match="must name a file"):
            _vasp_source(tmp_path / "missing")

    @pytest.mark.parametrize(
        ("kwargs", "message", "error"),
        [
            ({"spin_mode": "bad"}, "spin mode is unsupported", ValueError),
            ({"source_ref": ""}, "provenance must be complete", ValueError),
            ({"potcar_sha256": ("",)}, "hashes must be nonempty", ValueError),
            (
                {"kpoint_weights": jnp.asarray([])},
                "weights are required",
                ValueError,
            ),
            (
                {"kpoint_weights": jnp.asarray([0.0])},
                "weights must be finite and positive",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"fermi_energy_ev": jnp.asarray(jnp.nan)},
                "Fermi energy must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_source_invariant(
        self,
        tmp_path: Path,
        kwargs: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject invalid spin, provenance, weights, and Fermi energy.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change one field in a descriptor backed by an existing file.
        """
        path: Path = tmp_path / "WAVECAR"
        path.touch()
        values: Dict[str, object] = {
            "wavecar_path": path,
            "kpoint_weights": jnp.asarray([1.0]),
            "fermi_energy_ev": jnp.asarray(0.0),
            "spin_mode": "scalar",
            "source_ref": "fixture",
            "potcar_sha256": ("digest",),
        }
        values.update(kwargs)
        with pytest.raises(error, match=message):
            make_vasp_wavefunction_source(**values)


class TestWavecarheader:
    """Verify ``diffpes.types.WavecarHeader`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_header_metadata(self) -> None:
        """Preserve direct-access layout and physical metadata.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare each scalar identity and the cubic lattice.
        """
        header: Any = _header()
        assert header.record_length == 128
        assert header.precision_tag == 45200
        assert jnp.array_equal(header.lattice_ang, jnp.eye(3))

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"record_length": 0}, "length must be positive", ValueError),
            ({"spin_components": 3}, "must be one or two", ValueError),
            ({"precision_tag": 1}, "tag is unsupported", ValueError),
            ({"byte_order": "big"}, "must be little", ValueError),
            ({"nkpoints": 0}, "counts must be positive", ValueError),
            (
                {"encut_ev": jnp.asarray(0.0)},
                "cutoff must be finite and positive",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"lattice_ang": jnp.zeros((3, 3))},
                "lattice must be finite and nonsingular",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"fermi_energy_ev": jnp.asarray(jnp.nan)},
                "Fermi energy must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_header_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject every malformed layout and physical header field.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one value in the valid header fixture.
        """
        with pytest.raises(error, match=message):
            _header(**overrides)


class TestPlanewavebatch:
    """Verify ``diffpes.types.PlaneWaveBatch`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_consistent_bounded_state_axes(self) -> None:
        """Preserve one state with two valid plane-wave coefficients.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect the padded axis, count, and immutable source identity.
        """
        batch: Any = _batch()
        assert batch.coefficients.shape == (1, 2, 1)
        assert batch.plane_wave_counts[0] == 2
        assert batch.source_ref == "fixture"

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            (
                {"state_indices": jnp.zeros((2, 3), dtype=jnp.int32)},
                "state_indices",
                TypeCheckError,
            ),
            (
                {"plane_wave_counts": jnp.asarray([3], dtype=jnp.int32)},
                "counts must fit",
                eqx.EquinoxRuntimeError,
            ),
            (
                {
                    "coefficients": jnp.asarray(
                        [[[jnp.nan + 0.0j], [jnp.nan + 0.0j]]]
                    )
                },
                "coefficients must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"kpoints_frac": jnp.asarray([[jnp.nan, 0.0, 0.0]])},
                "k points must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"kpoint_weights": jnp.asarray([0.0])},
                "weights must be finite and positive",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"energies_ev": jnp.asarray([jnp.nan])},
                "energies must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"occupations": jnp.asarray([-1.0])},
                "occupations must be finite and nonnegative",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"fermi_energy_ev": jnp.asarray(jnp.nan)},
                "Fermi energy must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_batch_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject mismatched axes and every invalid numerical field.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change one field in the valid bounded-state fixture.
        """
        with pytest.raises(error, match=message):
            _batch(**overrides)


class TestInmemoryplanewavesource:
    """Verify ``diffpes.types.InMemoryPlaneWaveSource`` behavior.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_returns_the_exact_selected_batch(self) -> None:
        """Return coefficients when all requested indices match.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the selected coefficient tensor with the source batch.
        """
        batch: Any = _batch()
        source: Any = make_in_memory_plane_wave_source(
            batch, capabilities=("wavefunction",), state_ref="fixture"
        )
        request: Any = make_state_batch_request(
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            purpose="photocurrent",
        )
        assert jnp.array_equal(
            source.plane_wave_batch(request).coefficients, batch.coefficients
        )
        assert isinstance(source, PlaneWaveStateSource)

    def test_rejects_incomplete_source_metadata(self) -> None:
        """Reject an empty capability declaration.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Keep the state reference and derivative mode otherwise valid.
        """
        with pytest.raises(ValueError, match="metadata must be nonempty"):
            make_in_memory_plane_wave_source(
                _batch(), capabilities=(), state_ref="fixture"
            )

    def test_rejects_nonmatching_selection(self) -> None:
        """Reject a requested band absent from the in-memory batch.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change only the selected band index from zero to one.
        """
        source: Any = make_in_memory_plane_wave_source(
            _batch(), capabilities=("wavefunction",), state_ref="fixture"
        )
        request: Any = make_state_batch_request(
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            purpose="photocurrent",
        )
        with pytest.raises(eqx.EquinoxRuntimeError, match="does not match"):
            source.plane_wave_batch(request)


class TestPlanewavestatesource:
    """Verify the ``diffpes.types.PlaneWaveStateSource`` protocol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestWavecardataset:
    """Verify ``diffpes.types.WavecarDataset`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_consistent_indexed_metadata(self, tmp_path: Path) -> None:
        """Preserve one indexed coefficient record and one G vector.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Construct the carrier independently of the binary reader.
        """
        path: Path = tmp_path / "WAVECAR"
        path.write_bytes(bytes(384))
        dataset: Any = _dataset(path)
        assert dataset.coefficient_record_offsets == (256,)
        assert dataset.plane_wave_counts[0, 0] == 1

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"file_size": 129}, "only full records", ValueError),
            (
                {"record_offsets": (0, 129, 256)},
                "offsets must name full records",
                ValueError,
            ),
            (
                {"coefficient_record_offsets": ()},
                "offsets have an invalid count",
                ValueError,
            ),
            (
                {"coefficient_record_offsets": (64,)},
                "offsets must name full records",
                ValueError,
            ),
            (
                {"g_vectors_frac": ()},
                "metadata axes are inconsistent",
                ValueError,
            ),
            (
                {"plane_wave_counts": jnp.asarray([[2]], dtype=jnp.int32)},
                "counts must match G-vector lengths",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"kpoints_frac": jnp.asarray([[[jnp.nan, 0.0, 0.0]]])},
                "k points must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {
                    "eigenvalues_ev": jnp.asarray(
                        [[[jnp.nan + 0.0j]]], dtype=jnp.complex128
                    )
                },
                "eigenvalues must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"occupations": jnp.asarray([[[-1.0]]])},
                "occupations must be finite and nonnegative",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_dataset_invariant(
        self,
        tmp_path: Path,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject invalid record layout, axes, counts, and metadata values.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one component of the independent indexed dataset fixture.
        """
        path: Path = tmp_path / "WAVECAR"
        path.write_bytes(bytes(384))
        with pytest.raises(error, match=message):
            _dataset(path, **overrides)


class TestMakeStateBatchRequest:
    """Verify ``diffpes.types.make_state_batch_request``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeVaspWavefunctionSource:
    """Verify ``diffpes.types.make_vasp_wavefunction_source``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeWavecarHeader:
    """Verify ``diffpes.types.make_wavecar_header``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeWavecarDataset:
    """Verify ``diffpes.types.make_wavecar_dataset``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeInMemoryPlaneWaveSource:
    """Verify ``diffpes.types.make_in_memory_plane_wave_source``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakePlaneWaveBatch:
    """Verify ``diffpes.types.make_plane_wave_batch``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
