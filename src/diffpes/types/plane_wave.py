"""Define bounded plane-wave and PAW carriers for solver-neutral ARPES.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`InMemoryPlaneWaveSource`
    Define the ``InMemoryPlaneWaveSource`` public contract.
:class:`PlaneWaveBatch`
    Define the ``PlaneWaveBatch`` public contract.
:class:`PlaneWaveStateSource`
    Define the ``PlaneWaveStateSource`` public contract.
:class:`StateBatchRequest`
    Define the ``StateBatchRequest`` public contract.
:class:`VaspWavefunctionSource`
    Define the ``VaspWavefunctionSource`` public contract.
:class:`WavecarDataset`
    Define the ``WavecarDataset`` public contract.
:class:`WavecarHeader`
    Define the ``WavecarHeader`` public contract.
:func:`make_in_memory_plane_wave_source`
    Compute the ``make_in_memory_plane_wave_source`` public contract.
:func:`make_plane_wave_batch`
    Compute the ``make_plane_wave_batch`` public contract.
:func:`make_state_batch_request`
    Compute the ``make_state_batch_request`` public contract.
:func:`make_vasp_wavefunction_source`
    Compute the ``make_vasp_wavefunction_source`` public contract.
:func:`make_wavecar_dataset`
    Compute the ``make_wavecar_dataset`` public contract.
:func:`make_wavecar_header`
    Compute the ``make_wavecar_header`` public contract.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Protocol, Tuple, runtime_checkable
from jaxtyping import Array, Bool, Complex128, Float64, Int32, jaxtyped

from diffpes.constants import ARRAY_MATRIX_NDIM, CARTESIAN_COMPONENTS

from .geometry import CrystalGeometry


class StateBatchRequest(eqx.Module):
    """Define the ``StateBatchRequest`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestStatebatchrequest`

    Attributes
    ----------
    k_indices : Int32[Array, " n_k_selected"]
        Store selected momentum indices.
    band_indices : Int32[Array, " n_band_selected"]
        Store selected band indices.
    spin_indices : Int32[Array, " n_spin_selected"]
        Store selected spin indices.
    purpose : str
        Store the request purpose.

    See Also
    --------
    make_state_batch_request
        Construct a validated batch request.
    """

    k_indices: Int32[Array, " n_k_selected"]
    band_indices: Int32[Array, " n_band_selected"]
    spin_indices: Int32[Array, " n_spin_selected"]
    purpose: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static request intent and selected-state axis agreement."""
        if not self.purpose:
            raise ValueError("state batch request purpose must be nonempty")
        lengths: Tuple[int, int, int] = (
            self.k_indices.shape[0],
            self.band_indices.shape[0],
            self.spin_indices.shape[0],
        )
        if len(set(lengths)) != 1:
            raise ValueError(
                "state batch request indices must have one shared axis"
            )


class VaspWavefunctionSource(eqx.Module):
    """Define the ``VaspWavefunctionSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestVaspwavefunctionsource`

    Attributes
    ----------
    wavecar_path : Path
        Store the WAVECAR path.
    kpoint_weights : Float64[Array, " n_k"]
        Store momentum weights.
    fermi_energy_ev : Float64[Array, ""]
        Store Fermi energy.
    spin_mode : str
        Store the spin mode.
    source_ref : str
        Store the source identity.
    potcar_sha256 : Tuple[str, ...]
        Store POTCAR digests.

    See Also
    --------
    make_vasp_wavefunction_source
        Construct a validated VASP source.
    """

    wavecar_path: Path = eqx.field(static=True)
    kpoint_weights: Float64[Array, " n_k"]
    fermi_energy_ev: Float64[Array, ""]
    spin_mode: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    potcar_sha256: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate required provenance before binary ingestion."""
        if self.spin_mode not in ("scalar", "collinear", "spinor"):
            raise ValueError("VASP source spin mode is unsupported")
        if not self.source_ref or not self.potcar_sha256:
            raise ValueError("VASP source provenance must be complete")
        if any(not digest for digest in self.potcar_sha256):
            raise ValueError("PAW dataset hashes must be nonempty")
        if self.kpoint_weights.ndim != 1 or self.kpoint_weights.shape[0] == 0:
            raise ValueError("explicit VASP k-point weights are required")


class WavecarHeader(eqx.Module):
    """Define the ``WavecarHeader`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestWavecarheader`

    Attributes
    ----------
    record_length : int
        Store direct-record length.
    spin_components : int
        Store the spin-component count.
    precision_tag : int
        Store the precision tag.
    byte_order : str
        Store byte order.
    nkpoints : int
        Store the momentum count.
    nbands : int
        Store the band count.
    encut_ev : Float64[Array, ""]
        Store plane-wave cutoff.
    lattice_ang : Float64[Array, "3 3"]
        Store lattice vectors.
    fermi_energy_ev : Float64[Array, ""]
        Store Fermi energy.

    See Also
    --------
    make_wavecar_header
        Construct a validated WAVECAR header.
    """

    record_length: int = eqx.field(static=True)
    spin_components: int = eqx.field(static=True)
    precision_tag: int = eqx.field(static=True)
    byte_order: str = eqx.field(static=True)
    nkpoints: int = eqx.field(static=True)
    nbands: int = eqx.field(static=True)
    encut_ev: Float64[Array, ""]
    lattice_ang: Float64[Array, "3 3"]
    fermi_energy_ev: Float64[Array, ""]


class WavecarDataset(eqx.Module):
    """Define the ``WavecarDataset`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestWavecardataset`

    Attributes
    ----------
    source : VaspWavefunctionSource
        Store the source declaration.
    header : WavecarHeader
        Store parsed header metadata.
    file_size : int
        Store file size.
    record_offsets : Tuple[int, ...]
        Store metadata-record offsets.
    coefficient_record_offsets : Tuple[int, ...]
        Store coefficient-record offsets.
    plane_wave_counts : Int32[Array, "n_spin n_k"]
        Store plane-wave counts.
    kpoints_frac : Float64[Array, "n_spin n_k 3"]
        Store fractional momenta.
    eigenvalues_ev : Complex128[Array, "n_spin n_k n_band"]
        Store eigenvalues.
    occupations : Float64[Array, "n_spin n_k n_band"]
        Store occupations.
    g_vectors_frac : Tuple[Int32[Array, "n_pw 3"], ...]
        Store reciprocal-grid vectors.

    See Also
    --------
    make_wavecar_dataset
        Construct a validated WAVECAR dataset.
    """

    source: VaspWavefunctionSource
    header: WavecarHeader
    file_size: int = eqx.field(static=True)
    record_offsets: Tuple[int, ...] = eqx.field(static=True)
    coefficient_record_offsets: Tuple[int, ...] = eqx.field(static=True)
    plane_wave_counts: Int32[Array, "n_spin n_k"]
    kpoints_frac: Float64[Array, "n_spin n_k 3"]
    eigenvalues_ev: Complex128[Array, "n_spin n_k n_band"]
    occupations: Float64[Array, "n_spin n_k n_band"]
    g_vectors_frac: Tuple[Int32[Array, "n_pw 3"], ...]


class PlaneWaveBatch(eqx.Module):
    """Define the ``PlaneWaveBatch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestPlanewavebatch`

    Attributes
    ----------
    coefficients : Complex128[Array, "n_state n_pw n_spinor"]
        Store plane-wave coefficients.
    g_vectors_frac : Int32[Array, "n_state n_pw 3"]
        Store reciprocal-grid vectors.
    plane_wave_counts : Int32[Array, " n_state"]
        Store plane-wave counts.
    kpoints_frac : Float64[Array, "n_state 3"]
        Store fractional momenta.
    kpoint_weights : Float64[Array, " n_state"]
        Store momentum weights.
    energies_ev : Float64[Array, " n_state"]
        Store state energies.
    occupations : Float64[Array, " n_state"]
        Store state occupations.
    state_indices : Int32[Array, "n_state 3"]
        Store source indices.
    geometry : CrystalGeometry
        Store crystal geometry.
    fermi_energy_ev : Float64[Array, ""]
        Store Fermi energy.
    spin_mode : str
        Store the spin mode.
    source_ref : str
        Store the source identity.
    gauge_ref : str
        Store the gauge identity.
    augmentation_ref : Optional[str]
        Store the augmentation identity.

    See Also
    --------
    make_plane_wave_batch
        Construct a validated plane-wave batch.
    """

    coefficients: Complex128[Array, "n_state n_pw n_spinor"]
    g_vectors_frac: Int32[Array, "n_state n_pw 3"]
    plane_wave_counts: Int32[Array, " n_state"]
    kpoints_frac: Float64[Array, "n_state 3"]
    kpoint_weights: Float64[Array, " n_state"]
    energies_ev: Float64[Array, " n_state"]
    occupations: Float64[Array, " n_state"]
    state_indices: Int32[Array, "n_state 3"]
    geometry: CrystalGeometry
    fermi_energy_ev: Float64[Array, ""]
    spin_mode: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    gauge_ref: str = eqx.field(static=True)
    augmentation_ref: Optional[str] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate bounded state and padded plane-wave axes."""
        states: int = self.coefficients.shape[0]
        if (
            self.coefficients.ndim != 3  # noqa: PLR2004
            or self.g_vectors_frac.shape != self.coefficients.shape[:2] + (3,)
            or self.plane_wave_counts.shape != (states,)
            or self.kpoints_frac.shape != (states, 3)
            or self.kpoint_weights.shape != (states,)
            or self.energies_ev.shape != (states,)
            or self.occupations.shape != (states,)
            or self.state_indices.shape != (states, 3)
            or self.spin_mode not in ("scalar", "collinear", "spinor")
            or not self.source_ref
            or not self.gauge_ref
        ):
            raise ValueError(
                "plane-wave batch axes or metadata are inconsistent"
            )
        counts: Int32[Array, " n_state"] = eqx.error_if(
            self.plane_wave_counts,
            jnp.any(self.plane_wave_counts < 0)
            | jnp.any(self.plane_wave_counts > self.coefficients.shape[1]),
            "plane-wave counts must fit the padded axis",
        )
        coefficients: Complex128[Array, "n_state n_pw n_spinor"] = (
            eqx.error_if(
                self.coefficients,
                ~jnp.all(jnp.isfinite(self.coefficients)),
                "plane-wave coefficients must be finite",
            )
        )
        kpoints: Float64[Array, "n_state 3"] = eqx.error_if(
            self.kpoints_frac,
            ~jnp.all(jnp.isfinite(self.kpoints_frac)),
            "plane-wave k points must be finite",
        )
        weights: Float64[Array, " n_state"] = eqx.error_if(
            self.kpoint_weights,
            ~jnp.all(jnp.isfinite(self.kpoint_weights))
            | jnp.any(self.kpoint_weights <= 0.0),
            "plane-wave k-point weights must be finite and positive",
        )
        energies: Float64[Array, " n_state"] = eqx.error_if(
            self.energies_ev,
            ~jnp.all(jnp.isfinite(self.energies_ev)),
            "plane-wave energies must be finite",
        )
        occupations: Float64[Array, " n_state"] = eqx.error_if(
            self.occupations,
            ~jnp.all(jnp.isfinite(self.occupations))
            | jnp.any(self.occupations < 0.0),
            "plane-wave occupations must be finite and nonnegative",
        )
        fermi: Float64[Array, ""] = eqx.error_if(
            self.fermi_energy_ev,
            ~jnp.isfinite(self.fermi_energy_ev),
            "plane-wave Fermi energy must be finite",
        )
        object.__setattr__(self, "plane_wave_counts", counts)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "kpoints_frac", kpoints)
        object.__setattr__(self, "kpoint_weights", weights)
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "fermi_energy_ev", fermi)


class InMemoryPlaneWaveSource(eqx.Module):
    """Define the ``InMemoryPlaneWaveSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestInmemoryplanewavesource`

    Attributes
    ----------
    batch : PlaneWaveBatch
        Store the plane-wave batch.
    capabilities : Tuple[str, ...]
        Store supported capabilities.
    state_ref : str
        Store the state identity.
    derivative_mode : str
        Store the derivative mode.

    See Also
    --------
    make_in_memory_plane_wave_source
        Construct a validated in-memory source.
    """

    batch: PlaneWaveBatch
    capabilities: Tuple[str, ...] = eqx.field(static=True)
    state_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)

    @jaxtyped(typechecker=beartype)
    def plane_wave_batch(self, request: StateBatchRequest) -> PlaneWaveBatch:
        """Return the batch when its state selection matches the request."""
        selection_matches: Bool[Array, ""] = (
            jnp.array_equal(request.k_indices, self.batch.state_indices[:, 0])
            & jnp.array_equal(
                request.band_indices, self.batch.state_indices[:, 1]
            )
            & jnp.array_equal(
                request.spin_indices, self.batch.state_indices[:, 2]
            )
        )
        coefficients: Complex128[Array, "n_state n_pw n_spinor"] = (
            eqx.error_if(
                self.batch.coefficients,
                ~selection_matches,
                "in-memory plane-wave selection does not match the request",
            )
        )
        selected_batch: PlaneWaveBatch = eqx.tree_at(
            lambda batch: batch.coefficients,
            self.batch,
            coefficients,
        )
        return selected_batch


@runtime_checkable
class PlaneWaveStateSource(Protocol):
    """Define the ``PlaneWaveStateSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestPlanewavestatesource`
    """

    capabilities: Tuple[str, ...]
    state_ref: str
    derivative_mode: str

    def plane_wave_batch(self, request: StateBatchRequest) -> PlaneWaveBatch:
        """Return a selected bounded plane-wave batch."""
        ...  # noqa: PIE790


@jaxtyped(typechecker=beartype)
def make_state_batch_request(
    k_indices: Int32[Array, " n_k_selected"],
    band_indices: Int32[Array, " n_band_selected"],
    spin_indices: Int32[Array, " n_spin_selected"],
    *,
    purpose: str,
) -> StateBatchRequest:
    """Compute the ``make_state_batch_request`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakeStateBatchRequest`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    k_indices : Int32[Array, ' n_k_selected']
        Input value for this operation.
    band_indices : Int32[Array, ' n_band_selected']
        Input value for this operation.
    spin_indices : Int32[Array, ' n_spin_selected']
        Input value for this operation.
    purpose : str
        Input value for this operation.

    Returns
    -------
    result : StateBatchRequest
        Validated operation result.
    """
    result: StateBatchRequest = StateBatchRequest(
        jnp.asarray(k_indices, dtype=jnp.int32),
        jnp.asarray(band_indices, dtype=jnp.int32),
        jnp.asarray(spin_indices, dtype=jnp.int32),
        purpose,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_vasp_wavefunction_source(
    wavecar_path: Path,
    kpoint_weights: Float64[Array, " n_k"],
    fermi_energy_ev: Float64[Array, ""],
    *,
    spin_mode: str,
    source_ref: str,
    potcar_sha256: Tuple[str, ...],
) -> VaspWavefunctionSource:
    """Compute the ``make_vasp_wavefunction_source`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakeVaspWavefunctionSource`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    wavecar_path : Path
        Input value for this operation.
    kpoint_weights : Float64[Array, ' n_k']
        Input value for this operation.
    fermi_energy_ev : Float64[Array, '']
        Input value for this operation.
    spin_mode : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    potcar_sha256 : Tuple[str, ...]
        Input value for this operation.

    Returns
    -------
    result : VaspWavefunctionSource
        Validated operation result.

    Raises
    ------
    ValueError
        If the WAVECAR path does not name a file.
    """
    if not wavecar_path.is_file():
        raise ValueError("VASP WAVECAR path must name a file")
    weights: Float64[Array, " n_k"] = jnp.asarray(
        kpoint_weights, dtype=jnp.float64
    )
    checked_weights: Float64[Array, " n_k"] = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights <= 0.0),
        "VASP k-point weights must be finite and positive",
    )
    fermi: Float64[Array, ""] = jnp.asarray(fermi_energy_ev, dtype=jnp.float64)
    checked_fermi: Float64[Array, ""] = eqx.error_if(
        fermi,
        ~jnp.isfinite(fermi),
        "VASP Fermi energy must be finite",
    )
    result: VaspWavefunctionSource = VaspWavefunctionSource(
        wavecar_path,
        checked_weights,
        checked_fermi,
        spin_mode,
        source_ref,
        potcar_sha256,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_wavecar_header(
    record_length: int,
    spin_components: int,
    precision_tag: int,
    *,
    byte_order: str = "little",
    nkpoints: int,
    nbands: int,
    encut_ev: Float64[Array, ""],
    lattice_ang: Float64[Array, "3 3"],
    fermi_energy_ev: Float64[Array, ""],
) -> WavecarHeader:
    """Compute the ``make_wavecar_header`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakeWavecarHeader`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    record_length : int
        Input value for this operation.
    spin_components : int
        Input value for this operation.
    precision_tag : int
        Input value for this operation.
    byte_order : str
        Input value for this operation.
    nkpoints : int
        Input value for this operation.
    nbands : int
        Input value for this operation.
    encut_ev : Float64[Array, '']
        Input value for this operation.
    lattice_ang : Float64[Array, '3 3']
        Input value for this operation.
    fermi_energy_ev : Float64[Array, '']
        Input value for this operation.

    Returns
    -------
    result : WavecarHeader
        Validated operation result.

    Raises
    ------
    ValueError
        If direct-access layout metadata or positive counts are invalid.
    """
    if record_length <= 0:
        raise ValueError("WAVECAR record length must be positive")
    if spin_components not in (1, 2):
        raise ValueError("WAVECAR spin components must be one or two")
    if precision_tag not in (45200, 45210):
        raise ValueError("WAVECAR precision tag is unsupported")
    if byte_order != "little":
        raise ValueError("WAVECAR byte order must be little")
    if nkpoints <= 0 or nbands <= 0:
        raise ValueError("WAVECAR k-point and band counts must be positive")
    cutoff: Float64[Array, ""] = jnp.asarray(encut_ev, dtype=jnp.float64)
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        lattice_ang, dtype=jnp.float64
    )
    fermi: Float64[Array, ""] = jnp.asarray(fermi_energy_ev, dtype=jnp.float64)
    cutoff = eqx.error_if(
        cutoff,
        ~jnp.isfinite(cutoff) | (cutoff <= 0.0),
        "WAVECAR cutoff must be finite and positive",
    )
    lattice = eqx.error_if(
        lattice,
        ~jnp.all(jnp.isfinite(lattice))
        | (jnp.abs(jnp.linalg.det(lattice)) <= jnp.finfo(jnp.float64).eps),
        "WAVECAR lattice must be finite and nonsingular",
    )
    fermi = eqx.error_if(
        fermi,
        ~jnp.isfinite(fermi),
        "WAVECAR Fermi energy must be finite",
    )
    result: WavecarHeader = WavecarHeader(
        record_length,
        spin_components,
        precision_tag,
        byte_order,
        nkpoints,
        nbands,
        cutoff,
        lattice,
        fermi,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_wavecar_dataset(
    source: VaspWavefunctionSource,
    header: WavecarHeader,
    file_size: int,
    record_offsets: Tuple[int, ...],
    coefficient_record_offsets: Tuple[int, ...],
    plane_wave_counts: Int32[Array, "n_spin n_k"],
    kpoints_frac: Float64[Array, "n_spin n_k 3"],
    eigenvalues_ev: Complex128[Array, "n_spin n_k n_band"],
    occupations: Float64[Array, "n_spin n_k n_band"],
    g_vectors_frac: Tuple[Int32[Array, "n_pw 3"], ...],
) -> WavecarDataset:
    """Compute the ``make_wavecar_dataset`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakeWavecarDataset`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    source : VaspWavefunctionSource
        Input value for this operation.
    header : WavecarHeader
        Input value for this operation.
    file_size : int
        Input value for this operation.
    record_offsets : Tuple[int, ...]
        Input value for this operation.
    coefficient_record_offsets : Tuple[int, ...]
        Input value for this operation.
    plane_wave_counts : Int32[Array, 'n_spin n_k']
        Input value for this operation.
    kpoints_frac : Float64[Array, 'n_spin n_k 3']
        Input value for this operation.
    eigenvalues_ev : Complex128[Array, 'n_spin n_k n_band']
        Input value for this operation.
    occupations : Float64[Array, 'n_spin n_k n_band']
        Input value for this operation.
    g_vectors_frac : Tuple[Int32[Array, 'n_pw 3'], ...]
        Input value for this operation.

    Returns
    -------
    result : WavecarDataset
        Validated operation result.

    Raises
    ------
    ValueError
        If record offsets, indexed metadata axes, or G-vector shapes disagree.
    """
    if (
        file_size < header.record_length
        or file_size % header.record_length != 0
    ):
        raise ValueError("WAVECAR file size must contain only full records")
    if any(
        offset < 0
        or offset + header.record_length > file_size
        or offset % header.record_length != 0
        for offset in record_offsets
    ):
        raise ValueError(
            "WAVECAR record offsets must name full records within the file"
        )
    coefficient_count: int = (
        header.spin_components * header.nkpoints * header.nbands
    )
    if len(coefficient_record_offsets) != coefficient_count:
        raise ValueError("WAVECAR coefficient offsets have an invalid count")
    if any(
        offset not in record_offsets for offset in coefficient_record_offsets
    ):
        raise ValueError("WAVECAR coefficient offsets must name full records")
    counts: Int32[Array, "n_spin n_k"] = jnp.asarray(
        plane_wave_counts, dtype=jnp.int32
    )
    kpoints: Float64[Array, "n_spin n_k 3"] = jnp.asarray(
        kpoints_frac, dtype=jnp.float64
    )
    eigenvalues: Complex128[Array, "n_spin n_k n_band"] = jnp.asarray(
        eigenvalues_ev, dtype=jnp.complex128
    )
    occupation_values: Float64[Array, "n_spin n_k n_band"] = jnp.asarray(
        occupations, dtype=jnp.float64
    )
    expected_prefix: Tuple[int, int] = (
        header.spin_components,
        header.nkpoints,
    )
    if (
        counts.shape != expected_prefix
        or kpoints.shape != expected_prefix + (3,)
        or eigenvalues.shape != expected_prefix + (header.nbands,)
        or occupation_values.shape != eigenvalues.shape
        or len(g_vectors_frac) != header.spin_components * header.nkpoints
    ):
        raise ValueError("WAVECAR indexed metadata axes are inconsistent")
    checked_counts: Int32[Array, "n_spin n_k"] = counts
    spin_index: int
    k_index: int
    for spin_index in range(header.spin_components):
        for k_index in range(header.nkpoints):
            vector_index: int = spin_index * header.nkpoints + k_index
            vectors: Int32[Array, "n_pw 3"] = g_vectors_frac[vector_index]
            if (
                vectors.ndim != ARRAY_MATRIX_NDIM
                or vectors.shape[1] != CARTESIAN_COMPONENTS
            ):
                raise ValueError(
                    "WAVECAR G vectors must contain integer triples"
                )
            checked_counts = eqx.error_if(
                checked_counts,
                (counts[spin_index, k_index] <= 0)
                | (counts[spin_index, k_index] != vectors.shape[0]),
                "WAVECAR plane-wave counts must match G-vector lengths",
            )
    checked_kpoints: Float64[Array, "n_spin n_k 3"] = eqx.error_if(
        kpoints,
        ~jnp.all(jnp.isfinite(kpoints)),
        "WAVECAR indexed k points must be finite",
    )
    checked_eigenvalues: Complex128[Array, "n_spin n_k n_band"] = eqx.error_if(
        eigenvalues,
        ~jnp.all(jnp.isfinite(eigenvalues)),
        "WAVECAR indexed eigenvalues must be finite",
    )
    checked_occupations: Float64[Array, "n_spin n_k n_band"] = eqx.error_if(
        occupation_values,
        ~jnp.all(jnp.isfinite(occupation_values))
        | jnp.any(occupation_values < 0.0),
        "WAVECAR indexed occupations must be finite and nonnegative",
    )
    result: WavecarDataset = WavecarDataset(
        source,
        header,
        file_size,
        record_offsets,
        coefficient_record_offsets,
        checked_counts,
        checked_kpoints,
        checked_eigenvalues,
        checked_occupations,
        g_vectors_frac,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_in_memory_plane_wave_source(
    batch: PlaneWaveBatch,
    *,
    capabilities: Tuple[str, ...],
    state_ref: str,
    derivative_mode: str = "exact_ad",
) -> InMemoryPlaneWaveSource:
    """Compute the ``make_in_memory_plane_wave_source`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakeInMemoryPlaneWaveSource`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    batch : PlaneWaveBatch
        Input value for this operation.
    capabilities : Tuple[str, ...]
        Input value for this operation.
    state_ref : str
        Input value for this operation.
    derivative_mode : str
        Input value for this operation.

    Returns
    -------
    result : InMemoryPlaneWaveSource
        Validated operation result.

    Raises
    ------
    ValueError
        If capability, state, or derivative metadata is empty.
    """
    if not capabilities or not state_ref or not derivative_mode:
        raise ValueError("plane-wave source metadata must be nonempty")
    result: InMemoryPlaneWaveSource = InMemoryPlaneWaveSource(
        batch, capabilities, state_ref, derivative_mode
    )
    return result


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_plane_wave_batch(  # noqa: PLR0913
    coefficients: Complex128[Array, "n_state n_pw n_spinor"],
    g_vectors_frac: Int32[Array, "n_state n_pw 3"],
    plane_wave_counts: Int32[Array, " n_state"],
    kpoints_frac: Float64[Array, "n_state 3"],
    kpoint_weights: Float64[Array, " n_state"],
    energies_ev: Float64[Array, " n_state"],
    occupations: Float64[Array, " n_state"],
    state_indices: Int32[Array, "n_state 3"],
    geometry: CrystalGeometry,
    fermi_energy_ev: Float64[Array, ""],
    *,
    spin_mode: str,
    source_ref: str,
    gauge_ref: str,
    augmentation_ref: Optional[str] = None,
) -> PlaneWaveBatch:
    """Compute the ``make_plane_wave_batch`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_plane_wave.TestMakePlaneWaveBatch`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    coefficients : Complex128[Array, 'n_state n_pw n_spinor']
        Input value for this operation.
    g_vectors_frac : Int32[Array, 'n_state n_pw 3']
        Input value for this operation.
    plane_wave_counts : Int32[Array, ' n_state']
        Input value for this operation.
    kpoints_frac : Float64[Array, 'n_state 3']
        Input value for this operation.
    kpoint_weights : Float64[Array, ' n_state']
        Input value for this operation.
    energies_ev : Float64[Array, ' n_state']
        Input value for this operation.
    occupations : Float64[Array, ' n_state']
        Input value for this operation.
    state_indices : Int32[Array, 'n_state 3']
        Input value for this operation.
    geometry : CrystalGeometry
        Input value for this operation.
    fermi_energy_ev : Float64[Array, '']
        Input value for this operation.
    spin_mode : str
        Input value for this operation.
    source_ref : str
        Input value for this operation.
    gauge_ref : str
        Input value for this operation.
    augmentation_ref : Optional[str]
        Input value for this operation.

    Returns
    -------
    result : PlaneWaveBatch
        Validated operation result.
    """
    result: PlaneWaveBatch = PlaneWaveBatch(
        jnp.asarray(coefficients, dtype=jnp.complex128),
        jnp.asarray(g_vectors_frac, dtype=jnp.int32),
        jnp.asarray(plane_wave_counts, dtype=jnp.int32),
        jnp.asarray(kpoints_frac, dtype=jnp.float64),
        jnp.asarray(kpoint_weights, dtype=jnp.float64),
        jnp.asarray(energies_ev, dtype=jnp.float64),
        jnp.asarray(occupations, dtype=jnp.float64),
        jnp.asarray(state_indices, dtype=jnp.int32),
        geometry,
        jnp.asarray(fermi_energy_ev, dtype=jnp.float64),
        spin_mode,
        source_ref,
        gauge_ref,
        augmentation_ref,
    )
    return result


__all__: list[str] = [
    "InMemoryPlaneWaveSource",
    "PlaneWaveBatch",
    "PlaneWaveStateSource",
    "StateBatchRequest",
    "VaspWavefunctionSource",
    "WavecarDataset",
    "WavecarHeader",
    "make_in_memory_plane_wave_source",
    "make_plane_wave_batch",
    "make_state_batch_request",
    "make_vasp_wavefunction_source",
    "make_wavecar_dataset",
    "make_wavecar_header",
]
