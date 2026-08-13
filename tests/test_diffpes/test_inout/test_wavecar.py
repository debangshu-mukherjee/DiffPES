"""Check bounded host-only WAVECAR indexing contracts.

Extended Summary
----------------
These tests create complete synthetic direct-access files independently.
They check format metadata, record offsets, and bounded coefficient reads.

Routine Listings
----------------
:class:`TestWavecarHeader`
    Specify WAVECAR-header validation.
:class:`TestWavecarIndex`
    Specify host-only WAVECAR indexing.
"""

import itertools
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Callable, List, Tuple, Union
from jaxtyping import Complex64, Float64, Int32
from numpy.typing import NDArray

from diffpes.inout import index_wavecar, load_wavecar_records, wavecar_header
from diffpes.types import make_vasp_wavefunction_source

_RECORD_LENGTH = 128
_NKPOINTS = 2
_NBANDS = 2
_HEADER_VALUES = 13
_KINETIC_EV_ANG2 = 3.8099821161548597


def _source(path: Path) -> Any:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_vasp_wavefunction_source(
        path,
        jnp.asarray([0.5, 0.5]),
        jnp.asarray(0.5),
        spin_mode="scalar",
        source_ref="org.diffpes.fixture.wavecar@1",
        potcar_sha256=("fixture-hash",),
    )
    return result


def _independent_g_vectors(
    kpoint: Float64[NDArray, " 3"],
) -> Int32[NDArray, "n_pw 3"]:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    reciprocal: Float64[NDArray, "3 3"] = (
        2.0 * np.pi * np.linalg.inv(np.eye(3) * 10.0).T
    )
    accepted: List[Tuple[int, int, int]] = []
    vector: Tuple[int, int, int]
    for vector in itertools.product((0, 1, -1), repeat=3):
        cartesian: Float64[NDArray, " 3"] = (
            kpoint + np.asarray(vector)
        ) @ reciprocal
        if _KINETIC_EV_ANG2 * np.dot(cartesian, cartesian) <= 0.1:
            accepted.append(vector)
    result: Int32[NDArray, "n_pw 3"] = np.asarray(accepted, dtype=np.int32)
    return result


def _write_complete_wavecar(path: Path) -> None:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    kpoint: Any
    records: List[bytes] = []

    def record(
        values: Union[
            Float64[NDArray, " values"],
            Complex64[NDArray, " values"],
        ],
    ) -> bytes:
        """Check the private helper behavior."""
        payload: bytes = values.tobytes()
        padding: bytes = bytes(_RECORD_LENGTH - len(payload))
        completed: bytes = payload + padding
        return completed

    header_one: Float64[NDArray, " 3"] = np.asarray(
        [_RECORD_LENGTH, 1.0, 45200.0], dtype="<f8"
    )
    records.append(record(header_one))
    header_two: Float64[NDArray, " 13"] = np.asarray(
        [
            _NKPOINTS,
            _NBANDS,
            0.1,
            10.0,
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
            10.0,
            0.5,
        ],
        dtype="<f8",
    )
    records.append(record(header_two))
    kpoints: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.01, 0.0, 0.0),
    )
    k_index: int
    for k_index, kpoint in enumerate(kpoints):
        metadata: Float64[NDArray, " 10"] = np.asarray(
            [
                1.0,
                *kpoint,
                -1.0 - k_index,
                0.0,
                1.0,
                2.0 + k_index,
                0.0,
                0.0,
            ],
            dtype="<f8",
        )
        records.append(record(metadata))
        first: Complex64[NDArray, " 1"] = np.asarray(
            [1.0 + 2.0j + k_index], dtype="<c8"
        )
        second: Complex64[NDArray, " 1"] = np.asarray(
            [3.0 + 4.0j + k_index], dtype="<c8"
        )
        records.append(record(first))
        records.append(record(second))
    path.write_bytes(b"".join(records))


class TestWavecarHeader:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.inout.wavecar_header``
    """

    def test_rejects_unsupported_layout(self, tmp_path: Path) -> None:
        """Reject an unknown precision tag from a complete first record.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        path: Path = tmp_path / "WAVECAR"
        path.write_bytes(struct.pack("<3d", 128.0, 1.0, 99999.0) + bytes(232))
        with pytest.raises(ValueError, match="unsupported"):
            wavecar_header(path)

    def test_rejects_truncated_first_and_second_records(
        self, tmp_path: Path
    ) -> None:
        """Reject files truncated before either required header record.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Write one two-value prefix and one complete first record only.
        """
        first: Path = tmp_path / "WAVECAR-first"
        first.write_bytes(struct.pack("<2d", 128.0, 1.0))
        with pytest.raises(ValueError, match="truncated before the header"):
            wavecar_header(first)
        second: Path = tmp_path / "WAVECAR-second"
        second.write_bytes(
            struct.pack("<3d", 128.0, 1.0, 45200.0) + bytes(104)
        )
        with pytest.raises(ValueError, match="truncated before the second"):
            wavecar_header(second)

    def test_rejects_ambiguous_record_length(self, tmp_path: Path) -> None:
        """Reject a non-float-aligned direct-access record length.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Supply otherwise supported first-record scalar metadata.
        """
        path: Path = tmp_path / "WAVECAR"
        path.write_bytes(struct.pack("<3d", 127.0, 1.0, 45200.0) + bytes(230))
        with pytest.raises(ValueError, match="ambiguous or too small"):
            wavecar_header(path)

    def test_rejects_nonintegral_header_counts(self, tmp_path: Path) -> None:
        """Reject a fractional k-point count in the second record.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace only NKPTS in an otherwise complete valid fixture.
        """
        path: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(path)
        stream: Any
        with path.open("r+b") as stream:
            stream.seek(_RECORD_LENGTH)
            stream.write(np.asarray([1.5], dtype="<f8").tobytes())
        with pytest.raises(ValueError, match="k-point count.*exact"):
            wavecar_header(path)


class TestWavecarIndex:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.inout.index_wavecar``
    """

    def test_indexes_metadata_and_reads_one_record(
        self, tmp_path: Path
    ) -> None:
        """Read every metadata field and one bounded coefficient record.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        source: Any
        dataset: Any
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        source = _source(wavecar)
        dataset = index_wavecar(source)
        assert dataset.source.wavecar_path == wavecar
        assert dataset.header.precision_tag == 45200
        assert dataset.header.nkpoints == _NKPOINTS
        assert dataset.header.nbands == _NBANDS
        assert float(dataset.header.encut_ev) == 0.1
        assert np.array_equal(
            np.asarray(dataset.header.lattice_ang), np.eye(3) * 10.0
        )
        assert float(dataset.header.fermi_energy_ev) == 0.5
        assert jnp.array_equal(dataset.plane_wave_counts, jnp.ones((1, 2)))
        assert np.array_equal(
            np.asarray(dataset.kpoints_frac),
            np.asarray([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]]),
        )
        assert np.array_equal(
            np.asarray(dataset.eigenvalues_ev),
            np.asarray([[[-1.0, 2.0], [-2.0, 3.0]]]),
        )
        assert np.array_equal(
            np.asarray(dataset.occupations),
            np.asarray([[[1.0, 0.0], [1.0, 0.0]]]),
        )
        expected_coefficients: Tuple[complex, ...] = (
            1.0 + 2.0j,
            3.0 + 4.0j,
            2.0 + 2.0j,
            4.0 + 4.0j,
        )
        record_indices: Tuple[int, ...] = (3, 4, 6, 7)
        record_index: int
        expected: complex
        for record_index, expected in zip(
            record_indices, expected_coefficients, strict=True
        ):
            coefficients: Complex64[NDArray, " 1"] = load_wavecar_records(
                dataset, offset=record_index, count=1
            )
            assert np.array_equal(coefficients, np.asarray([expected]))

    def test_regenerates_independent_g_vector_counts(
        self, tmp_path: Path
    ) -> None:
        """Match every indexed G vector with a plain NumPy enumeration.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Enumerate integer triples directly from the fixture lattice and cutoff.
        """
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        dataset: Any = index_wavecar(_source(wavecar))
        index: int
        kpoint: Float64[NDArray, " 3"]
        for index, kpoint in enumerate(np.asarray(dataset.kpoints_frac[0])):
            expected: Int32[NDArray, "n_pw 3"] = _independent_g_vectors(kpoint)
            assert np.array_equal(dataset.g_vectors_frac[index], expected)
            assert expected.shape[0] == dataset.plane_wave_counts[0, index]

    def test_indexing_never_reads_coefficient_payloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep index-time reads bounded to float metadata records.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Record every ``numpy.fromfile`` dtype/count pair during indexing.
        """
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        original: Callable[..., Any] = np.fromfile
        calls: List[Tuple[np.dtype[Any], int]] = []

        def recording_fromfile(*args: Any, **kwargs: Any) -> Any:
            """Check the private helper behavior."""
            dtype: np.dtype[Any] = np.dtype(kwargs.get("dtype", float))
            count: int = int(kwargs.get("count", -1))
            calls.append((dtype, count))
            result: Any = original(*args, **kwargs)
            return result

        monkeypatch.setattr(np, "fromfile", recording_fromfile)
        index_wavecar(_source(wavecar))
        assert calls
        assert all(
            not np.issubdtype(dtype, np.complexfloating) for dtype, _ in calls
        )
        assert max(count for _, count in calls) <= _HEADER_VALUES

    def test_rejects_g_count_mismatch(self, tmp_path: Path) -> None:
        """Reject metadata whose NPLANE disagrees with regenerated vectors.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change the first k-point count from one to two exactly.
        """
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        stream: Any
        with wavecar.open("r+b") as stream:
            stream.seek(2 * _RECORD_LENGTH)
            stream.write(np.asarray([2.0], dtype="<f8").tobytes())
        with pytest.raises(
            ValueError,
            match="G-vector count mismatch.*expected 2, found 1",
        ):
            index_wavecar(_source(wavecar))

    def test_rejects_partial_final_record(self, tmp_path: Path) -> None:
        """Reject a file ending inside its final direct-access record.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Remove one byte from an otherwise complete fixture.
        """
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        wavecar.write_bytes(wavecar.read_bytes()[:-1])
        with pytest.raises(ValueError, match="partial direct-access record"):
            index_wavecar(_source(wavecar))


class TestIndexWavecar:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.inout.index_wavecar``
    """


class TestLoadWavecarRecords:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.inout.load_wavecar_records``
    """

    def test_rejects_noncoefficient_and_cross_record_reads(
        self, tmp_path: Path
    ) -> None:
        """Reject metadata offsets and coefficient reads crossing a record.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Request the first metadata record and seventeen complex64 values.
        """
        wavecar: Path = tmp_path / "WAVECAR"
        _write_complete_wavecar(wavecar)
        dataset: Any = index_wavecar(_source(wavecar))
        with pytest.raises(ValueError, match="select a coefficient record"):
            load_wavecar_records(dataset, offset=2, count=1)
        with pytest.raises(ValueError, match="exceeds one record"):
            load_wavecar_records(dataset, offset=3, count=17)
