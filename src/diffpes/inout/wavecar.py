"""Read bounded VASP WAVECAR direct-access records.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:func:`index_wavecar`
    Compute the ``index_wavecar`` public contract.
:func:`load_wavecar_records`
    Compute the ``load_wavecar_records`` public contract.
:func:`wavecar_header`
    Compute the ``wavecar_header`` public contract.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Literal, Tuple, Union
from jaxtyping import Complex64, Complex128, Float64, Int32, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    HBAR_SQ_OVER_2ME_EV_ANG2,
    WAVECAR_SECOND_RECORD_VALUES,
    WAVECAR_SINGLE_PRECISION_TAG,
)
from diffpes.types import (
    VaspWavefunctionSource,
    WavecarDataset,
    WavecarHeader,
    make_wavecar_dataset,
    make_wavecar_header,
)


def _exact_positive_integer(value: float, *, name: str) -> int:
    """PRIVATE: Convert one exactly integral positive format value.

    Parameters
    ----------
    value : float
        Format value to convert.
    name : str
        Field name for failures.

    Returns
    -------
    integer : int
        Exact positive integer.

    Raises
    ------
    ValueError
        If the value lacks an exact positive integer representation.
    """
    integer: int = int(round(value))
    if not np.isfinite(value) or float(integer) != value or integer <= 0:
        raise ValueError(f"WAVECAR {name} must be an exact positive integer")
    return integer


def _fft_order(bound: int) -> Tuple[int, ...]:
    """PRIVATE: Return zero-positive-negative FFT integer order.

    Notes
    -----
    Match the direct-access reciprocal-grid ordering.
    """
    order: Tuple[int, ...] = tuple(range(bound + 1)) + tuple(range(-bound, 0))
    return order


def _regenerate_g_vectors(
    kpoint_frac: Float64[NDArray, " 3"],
    lattice_ang: Float64[NDArray, "3 3"],
    encut_ev: float,
) -> Int32[NDArray, "n_pw 3"]:
    """PRIVATE: Generate cutoff-valid reciprocal vectors in file order.

    Notes
    -----
    Enumerate integer triplets and retain states inside the kinetic cutoff.
    """
    reciprocal: Float64[NDArray, "3 3"] = (
        2.0 * np.pi * np.linalg.inv(lattice_ang).T
    )
    minimum_scale: float = float(np.min(np.linalg.svd(reciprocal).S))
    wavevector_limit: float = float(
        np.sqrt(encut_ev / HBAR_SQ_OVER_2ME_EV_ANG2)
    )
    bound: int = int(
        np.ceil(wavevector_limit / minimum_scale + np.max(np.abs(kpoint_frac)))
    )
    order: Tuple[int, ...] = _fft_order(bound)
    accepted: List[Tuple[int, int, int]] = []
    first: int
    second: int
    third: int
    for first in order:
        for second in order:
            for third in order:
                fractional: Float64[NDArray, " 3"] = np.asarray(
                    (first, second, third), dtype=np.float64
                )
                cartesian: Float64[NDArray, " 3"] = (
                    kpoint_frac + fractional
                ) @ reciprocal
                kinetic_energy: float = HBAR_SQ_OVER_2ME_EV_ANG2 * float(
                    np.dot(cartesian, cartesian)
                )
                if kinetic_energy <= encut_ev:
                    accepted.append((first, second, third))
    vectors: Int32[NDArray, "n_pw 3"] = np.asarray(accepted, dtype=np.int32)
    return vectors


@jaxtyped(typechecker=beartype)
def wavecar_header(path: Union[str, Path]) -> WavecarHeader:
    """Compute the ``wavecar_header`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_wavecar.TestWavecarHeader`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    path : Union[str, Path]
        Input value for this operation.

    Returns
    -------
    result : WavecarHeader
        Validated operation result.

    Raises
    ------
    ValueError
        If metadata records contain truncation, unsupported layouts, or
        nonintegral counts.
    """
    source: Path = Path(path)
    raw: Float64[NDArray, " 3"] = np.fromfile(source, dtype="<f8", count=3)
    if raw.shape != (3,):
        raise ValueError("WAVECAR is truncated before the header record")
    record_length: int = _exact_positive_integer(
        float(raw[0]), name="record length"
    )
    spin_components: int = _exact_positive_integer(
        float(raw[1]), name="spin-component count"
    )
    precision_tag: int = _exact_positive_integer(
        float(raw[2]), name="precision tag"
    )
    if spin_components not in (1, 2):
        raise ValueError("unsupported WAVECAR spin-component count")
    if precision_tag not in (45200, 45210):
        raise ValueError("unsupported WAVECAR precision tag")
    if (
        record_length % np.dtype("<f8").itemsize != 0
        or record_length
        < WAVECAR_SECOND_RECORD_VALUES * np.dtype("<f8").itemsize
    ):
        raise ValueError("WAVECAR record length is ambiguous or too small")
    if source.stat().st_size < record_length:
        raise ValueError("WAVECAR first record exceeds file size")
    second: Float64[NDArray, " 13"] = np.fromfile(
        source,
        dtype="<f8",
        count=WAVECAR_SECOND_RECORD_VALUES,
        offset=record_length,
    )
    if second.shape != (WAVECAR_SECOND_RECORD_VALUES,):
        raise ValueError("WAVECAR is truncated before the second record")
    nkpoints: int = _exact_positive_integer(
        float(second[0]), name="k-point count"
    )
    nbands: int = _exact_positive_integer(float(second[1]), name="band count")
    encut_ev: float = float(second[2])
    lattice_ang: Float64[NDArray, "3 3"] = second[3:12].reshape(3, 3)
    fermi_energy_ev: float = float(second[12])
    result: WavecarHeader = make_wavecar_header(
        record_length,
        spin_components,
        precision_tag,
        byte_order="little",
        nkpoints=nkpoints,
        nbands=nbands,
        encut_ev=jnp.asarray(encut_ev),
        lattice_ang=jnp.asarray(lattice_ang),
        fermi_energy_ev=jnp.asarray(fermi_energy_ev),
    )
    return result


@jaxtyped(typechecker=beartype)
def index_wavecar(source: VaspWavefunctionSource) -> WavecarDataset:
    """Compute the ``index_wavecar`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_wavecar.TestIndexWavecar`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    source : VaspWavefunctionSource
        Input value for this operation.

    Returns
    -------
    result : WavecarDataset
        Validated operation result.

    Raises
    ------
    ValueError
        If record boundaries, metadata axes, source identity, or regenerated
        reciprocal-vector counts disagree.
    """
    header: WavecarHeader = wavecar_header(source.wavecar_path)
    file_size: int = source.wavecar_path.stat().st_size
    if file_size % header.record_length != 0:
        raise ValueError(
            "WAVECAR extent contains a partial direct-access record"
        )
    if source.kpoint_weights.shape != (header.nkpoints,):
        raise ValueError("WAVECAR and source k-point counts disagree")
    record_count: int = file_size // header.record_length
    offsets: Tuple[int, ...] = tuple(
        index * header.record_length for index in range(record_count)
    )
    expected_records: int = 2 + header.spin_components * header.nkpoints * (
        header.nbands + 1
    )
    if record_count != expected_records:
        raise ValueError("WAVECAR direct-access record count is inconsistent")
    metadata_values: int = 4 + 3 * header.nbands
    if metadata_values * np.dtype("<f8").itemsize > header.record_length:
        raise ValueError(
            "WAVECAR band metadata exceeds one direct-access record"
        )
    counts: Int32[NDArray, "n_spin n_k"] = np.empty(
        (header.spin_components, header.nkpoints), dtype=np.int32
    )
    kpoints: Float64[NDArray, "n_spin n_k 3"] = np.empty(
        (header.spin_components, header.nkpoints, 3), dtype=np.float64
    )
    eigenvalues: Complex128[NDArray, "n_spin n_k n_band"] = np.empty(
        (header.spin_components, header.nkpoints, header.nbands),
        dtype=np.complex128,
    )
    occupations: Float64[NDArray, "n_spin n_k n_band"] = np.empty(
        (header.spin_components, header.nkpoints, header.nbands),
        dtype=np.float64,
    )
    coefficient_offsets: List[int] = []
    g_vectors: List[Int32[NDArray, "n_pw 3"]] = []
    spin_index: int
    k_index: int
    for spin_index in range(header.spin_components):
        for k_index in range(header.nkpoints):
            record_index: int = (
                2
                + spin_index * header.nkpoints * (header.nbands + 1)
                + k_index * (header.nbands + 1)
            )
            metadata: Float64[NDArray, " n_metadata"] = np.fromfile(
                source.wavecar_path,
                dtype="<f8",
                count=metadata_values,
                offset=offsets[record_index],
            )
            if metadata.shape != (metadata_values,):
                raise ValueError(
                    "WAVECAR metadata record is truncated at "
                    f"k-index {k_index}"
                )
            plane_wave_count: int = _exact_positive_integer(
                float(metadata[0]), name="plane-wave count"
            )
            coefficient_itemsize: int = (
                np.dtype("<c8").itemsize
                if header.precision_tag == WAVECAR_SINGLE_PRECISION_TAG
                else np.dtype("<c16").itemsize
            )
            if plane_wave_count * coefficient_itemsize > header.record_length:
                raise ValueError(
                    "WAVECAR coefficient payload exceeds its direct-access "
                    f"record at k-index {k_index}"
                )
            counts[spin_index, k_index] = plane_wave_count
            kpoints[spin_index, k_index] = metadata[1:4]
            band_values: Float64[NDArray, "n_band 3"] = metadata[4:].reshape(
                header.nbands, 3
            )
            eigenvalues[spin_index, k_index] = (
                band_values[:, 0] + 1.0j * band_values[:, 1]
            )
            occupations[spin_index, k_index] = band_values[:, 2]
            regenerated: Int32[NDArray, "n_pw 3"] = _regenerate_g_vectors(
                kpoints[spin_index, k_index],
                np.asarray(header.lattice_ang),
                float(header.encut_ev),
            )
            if regenerated.shape[0] != plane_wave_count:
                raise ValueError(
                    "WAVECAR G-vector count mismatch at "
                    f"k-index {k_index}: expected {plane_wave_count}, "
                    f"found {regenerated.shape[0]}"
                )
            g_vectors.append(regenerated)
            band_index: int
            for band_index in range(header.nbands):
                coefficient_offsets.append(
                    offsets[record_index + band_index + 1]
                )
    if not np.isclose(
        float(header.fermi_energy_ev), float(source.fermi_energy_ev)
    ):
        raise ValueError("WAVECAR and source Fermi energies disagree")
    result: WavecarDataset = make_wavecar_dataset(
        source,
        header,
        file_size,
        offsets,
        tuple(coefficient_offsets),
        jnp.asarray(counts),
        jnp.asarray(kpoints),
        jnp.asarray(eigenvalues),
        jnp.asarray(occupations),
        tuple(jnp.asarray(vectors) for vectors in g_vectors),
    )
    return result


@jaxtyped(typechecker=beartype)
def load_wavecar_records(
    dataset: WavecarDataset,
    *,
    offset: int,
    count: int,
    coefficient_dtype: Literal["file", "complex64", "complex128"] = "file",
) -> Union[
    Complex64[NDArray, " count"],
    Complex128[NDArray, " count"],
]:
    """Compute the ``load_wavecar_records`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_wavecar.TestLoadWavecarRecords`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    dataset : WavecarDataset
        Input value for this operation.
    offset : int
        Input value for this operation.
    count : int
        Input value for this operation.
    coefficient_dtype : Literal['file', 'complex64', 'complex128']
        Input value for this operation.

    Returns
    -------
    result : Union[Complex64[NDArray, ' count'], Complex128[NDArray, ' count']]
        Validated operation result.

    Raises
    ------
    ValueError
        If the request does not select one bounded coefficient record.
    """
    if offset < 0 or count <= 0:
        raise ValueError("WAVECAR record offset/count must be positive")
    file_dtype: np.dtype = (
        np.dtype("<c8")
        if dataset.header.precision_tag == WAVECAR_SINGLE_PRECISION_TAG
        else np.dtype("<c16")
    )
    byte_offset: int = offset * dataset.header.record_length
    byte_count: int = count * file_dtype.itemsize
    if byte_offset not in dataset.coefficient_record_offsets:
        raise ValueError("WAVECAR offset must select a coefficient record")
    if byte_count > dataset.header.record_length:
        raise ValueError("WAVECAR coefficient read exceeds one record")
    if byte_offset + byte_count > dataset.file_size:
        raise ValueError(
            "WAVECAR coefficient request exceeds indexed file extent"
        )
    values: Union[
        Complex64[NDArray, " count"], Complex128[NDArray, " count"]
    ] = np.fromfile(
        dataset.source.wavecar_path,
        dtype=file_dtype,
        count=count,
        offset=byte_offset,
    )
    if coefficient_dtype == "complex64":
        result: Union[
            Complex64[NDArray, " count"], Complex128[NDArray, " count"]
        ] = values.astype(np.complex64, copy=False)
        return result  # noqa: RET504
    if coefficient_dtype == "complex128":
        result = values.astype(np.complex128, copy=False)
        return result  # noqa: RET504
    result = values
    return result  # noqa: RET504


__all__: list[str] = [
    "index_wavecar",
    "load_wavecar_records",
    "wavecar_header",
]
