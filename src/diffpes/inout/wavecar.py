"""Index and read bounded VASP WAVECAR direct-access records.

Extended Summary
----------------
The WAVECAR reader is a host-side file airlock.  It parses only explicit
little-endian standard-binary records, keeps file provenance in immutable
types, and reads a caller-bounded coefficient range.  It never creates a
global device coefficient tensor.

Routine Listings
----------------
:func:`index_wavecar`
    Construct a bounded host-side index from a VASP source bundle.
:func:`load_wavecar_records`
    Read one validated direct-access coefficient range.
:func:`wavecar_header`
    Parse and validate the mandatory first WAVECAR record.
"""

from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Union
from numpy.typing import NDArray

from diffpes.types import (
    VaspWavefunctionSource,
    WavecarDataset,
    WavecarHeader,
    make_wavecar_dataset,
    make_wavecar_header,
)

_WAVECAR_SINGLE_PRECISION_TAG = 45200


@beartype
def wavecar_header(path: Union[str, Path]) -> WavecarHeader:
    """Parse and validate the mandatory first WAVECAR record.

    Parameters
    ----------
    path : Union[str, Path]
        Standard-binary WAVECAR path.

    Returns
    -------
    WavecarHeader
        Immutable direct-access metadata using explicit little-endian order.

    Raises
    ------
    ValueError
        If the header is truncated, has an unsupported tag, or exceeds the
        file.
    """
    source = Path(path)
    raw = np.fromfile(source, dtype="<f8", count=3)
    if raw.shape != (3,):
        raise ValueError("WAVECAR is truncated before the header record")
    record_length = int(round(float(raw[0])))
    spin_components = int(round(float(raw[1])))
    precision_tag = int(round(float(raw[2])))
    if record_length <= 0:
        raise ValueError("WAVECAR record length must be positive")
    if spin_components not in (1, 2):
        raise ValueError("unsupported WAVECAR spin-component count")
    if precision_tag not in (45200, 45210):
        raise ValueError("unsupported WAVECAR precision tag")
    if source.stat().st_size < record_length:
        raise ValueError("WAVECAR first record exceeds file size")
    return make_wavecar_header(
        record_length, spin_components, precision_tag, byte_order="little"
    )


@beartype
def index_wavecar(source: VaspWavefunctionSource) -> WavecarDataset:
    """Construct a non-materializing random-access WAVECAR descriptor.

    Parameters
    ----------
    source : VaspWavefunctionSource
        VASP source bundle providing file paths and independent provenance.

    Returns
    -------
    WavecarDataset
        Host-side descriptor containing file extent and direct-access offsets.

    Raises
    ------
    ValueError
        If the file has a partial direct-access record.
    """
    header = wavecar_header(source.wavecar_path)
    file_size = source.wavecar_path.stat().st_size
    if file_size % header.record_length != 0:
        raise ValueError(
            "WAVECAR extent contains a partial direct-access record"
        )
    record_count = file_size // header.record_length
    offsets = tuple(
        index * header.record_length for index in range(record_count)
    )
    return make_wavecar_dataset(source, header, file_size, offsets)


@beartype
def load_wavecar_records(
    dataset: WavecarDataset,
    *,
    offset: int,
    count: int,
    coefficient_dtype: Literal["file", "complex64", "complex128"] = "file",
) -> Union[NDArray[np.complex64], NDArray[np.complex128]]:
    """Read one validated bounded coefficient range from an indexed WAVECAR.

    Parameters
    ----------
    dataset : WavecarDataset
        Checked host-side WAVECAR descriptor.
    offset : int
        Direct-access record index containing the first requested coefficient.
    count : int
        Number of complex coefficients to read.
    coefficient_dtype : Literal["file", "complex64", "complex128"]
        Requested output precision, independent from format detection.

    Returns
    -------
    numpy.ndarray
        One bounded complex coefficient vector in the requested precision.

    Raises
    ------
    ValueError
        If the requested range is invalid or exceeds the indexed file extent.
    """
    if offset < 0 or count <= 0:
        raise ValueError("WAVECAR record offset/count must be positive")
    file_dtype = (
        np.dtype("<c8")
        if dataset.header.precision_tag == _WAVECAR_SINGLE_PRECISION_TAG
        else np.dtype("<c16")
    )
    byte_offset = offset * dataset.header.record_length
    byte_count = count * file_dtype.itemsize
    if byte_offset + byte_count > dataset.file_size:
        raise ValueError(
            "WAVECAR coefficient request exceeds indexed file extent"
        )
    values = np.fromfile(
        dataset.source.wavecar_path,
        dtype=file_dtype,
        count=count,
        offset=byte_offset,
    )
    if coefficient_dtype == "complex64":
        return values.astype(np.complex64, copy=False)
    if coefficient_dtype == "complex128":
        return values.astype(np.complex128, copy=False)
    return values


__all__: list[str] = [
    "index_wavecar",
    "load_wavecar_records",
    "wavecar_header",
]
