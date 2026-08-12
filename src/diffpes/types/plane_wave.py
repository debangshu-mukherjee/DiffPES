"""Define bounded plane-wave and PAW carriers for solver-neutral ARPES."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Protocol, Tuple, runtime_checkable
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from .geometry import CrystalGeometry


class StateBatchRequest(eqx.Module):
    """Select a bounded set of state records for host-side loading."""

    k_indices: Int32[Array, " n_k_selected"]
    band_indices: Int32[Array, " n_band_selected"]
    spin_indices: Int32[Array, " n_spin_selected"]
    purpose: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static request intent and selected-state axis agreement."""
        if not self.purpose:
            raise ValueError("state batch request purpose must be nonempty")
        lengths = (
            self.k_indices.shape[0],
            self.band_indices.shape[0],
            self.spin_indices.shape[0],
        )
        if len(set(lengths)) != 1:
            raise ValueError(
                "state batch request indices must have one shared axis"
            )


class VaspWavefunctionSource(eqx.Module):
    """Bind VASP wavefunction files to mandatory external provenance.

    Attributes
    ----------
    wavecar_path : Path
        Path to the standard-binary WAVECAR file.
    kpoint_weights : Float64[Array, " n_k"]
        Explicit irreducible k-point weights in WAVECAR order.
    fermi_energy_ev : Float64[Array, ""]
        Fermi energy supplied by companion metadata, never inferred from
        WAVECAR.
    spin_mode : str
        Declared scalar, collinear, or spinor layout.
    source_ref : str
        Stable source identity.
    potcar_sha256 : Tuple[str, ...]
        Hashes of the PAW datasets without their licensed source text.

    See Also
    --------
    WavecarDataset
        Indexed host-side view created from this immutable source bundle.
    """

    wavecar_path: Path = eqx.field(static=True)
    kpoint_weights: Float64[Array, " n_k"]
    fermi_energy_ev: Float64[Array, ""]
    spin_mode: str = eqx.field(static=True)
    source_ref: str = eqx.field(static=True)
    potcar_sha256: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate provenance that must be known before binary ingestion."""
        if self.spin_mode not in ("scalar", "collinear", "spinor"):
            raise ValueError("VASP source spin mode is unsupported")
        if not self.source_ref or not self.potcar_sha256:
            raise ValueError("VASP source provenance must be complete")
        if any(not digest for digest in self.potcar_sha256):
            raise ValueError("PAW dataset hashes must be nonempty")
        if self.kpoint_weights.ndim != 1 or self.kpoint_weights.shape[0] == 0:
            raise ValueError("explicit VASP k-point weights are required")


class WavecarHeader(eqx.Module):
    """Describe validated direct-access WAVECAR header metadata.

    Attributes
    ----------
    record_length : int
        Direct-access record length in bytes.
    spin_components : int
        Number of stored spin components.
    precision_tag : int
        VASP format and scalar-precision tag.
    byte_order : str
        Explicit byte order used for binary records.
    """

    record_length: int = eqx.field(static=True)
    spin_components: int = eqx.field(static=True)
    precision_tag: int = eqx.field(static=True)
    byte_order: str = eqx.field(static=True)


class WavecarDataset(eqx.Module):
    """Store a checked host-side WAVECAR index without global coefficient
    leaves.

    Attributes
    ----------
    source : VaspWavefunctionSource
        Source bundle whose provenance was checked during indexing.
    header : WavecarHeader
        Validated binary header.
    file_size : int
        Indexed WAVECAR extent in bytes.
    record_offsets : Tuple[int, ...]
        Validated direct-access offsets for available records.
    """

    source: VaspWavefunctionSource
    header: WavecarHeader = eqx.field(static=True)
    file_size: int = eqx.field(static=True)
    record_offsets: Tuple[int, ...] = eqx.field(static=True)


class PlaneWaveBatch(eqx.Module):
    """Store padded selected-state plane-wave coefficients and metadata."""

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
        object.__setattr__(self, "plane_wave_counts", counts)


class InMemoryPlaneWaveSource(eqx.Module):
    """Expose traced plane-wave leaves through the common source protocol."""

    batch: PlaneWaveBatch
    capabilities: Tuple[str, ...] = eqx.field(static=True)
    state_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)

    @jaxtyped(typechecker=beartype)
    def plane_wave_batch(self, request: StateBatchRequest) -> PlaneWaveBatch:
        """Return the bounded in-memory batch; selection is producer-owned."""
        del request
        return self.batch


@runtime_checkable
class PlaneWaveStateSource(Protocol):
    """Capability protocol for sources providing bounded plane-wave states."""

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
    """Create a host-selected bounded state request."""
    return StateBatchRequest(
        jnp.asarray(k_indices, dtype=jnp.int32),
        jnp.asarray(band_indices, dtype=jnp.int32),
        jnp.asarray(spin_indices, dtype=jnp.int32),
        purpose,
    )


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
    """Construct a VASP source bundle with explicit Fermi provenance.

    Parameters
    ----------
    wavecar_path : Path
        Path to the binary wavefunctions.
    kpoint_weights : Float64[Array, " n_k"]
        Explicit irreducible k-point weights in binary-file order.
    fermi_energy_ev : Float64[Array, ""]
        Fermi energy supplied by companion metadata.
    spin_mode : str
        Declared VASP spin layout.
    source_ref : str
        Stable source identity.
    potcar_sha256 : Tuple[str, ...]
        PAW-dataset hashes, excluding licensed source text.

    Returns
    -------
    VaspWavefunctionSource
        Immutable source bundle for bounded host-side WAVECAR indexing.
    """
    weights = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    checked_weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights <= 0.0),
        "VASP k-point weights must be finite and positive",
    )
    fermi = jnp.asarray(fermi_energy_ev, dtype=jnp.float64)
    checked_fermi = eqx.error_if(
        fermi,
        ~jnp.isfinite(fermi),
        "VASP Fermi energy must be finite",
    )
    return VaspWavefunctionSource(
        wavecar_path,
        checked_weights,
        checked_fermi,
        spin_mode,
        source_ref,
        potcar_sha256,
    )


@jaxtyped(typechecker=beartype)
def make_wavecar_header(
    record_length: int,
    spin_components: int,
    precision_tag: int,
    *,
    byte_order: str = "little",
) -> WavecarHeader:
    """Create validated immutable WAVECAR header metadata."""
    if record_length <= 0:
        raise ValueError("WAVECAR record length must be positive")
    if spin_components not in (1, 2):
        raise ValueError("WAVECAR spin components must be one or two")
    if precision_tag not in (45200, 45210):
        raise ValueError("WAVECAR precision tag is unsupported")
    if byte_order != "little":
        raise ValueError("WAVECAR byte order must be little")
    return WavecarHeader(
        record_length, spin_components, precision_tag, byte_order
    )


@jaxtyped(typechecker=beartype)
def make_wavecar_dataset(
    source: VaspWavefunctionSource,
    header: WavecarHeader,
    file_size: int,
    record_offsets: Tuple[int, ...],
) -> WavecarDataset:
    """Create an indexed WAVECAR descriptor with bounded record offsets."""
    if file_size < header.record_length:
        raise ValueError("WAVECAR file size must include its first record")
    if any(offset < 0 or offset >= file_size for offset in record_offsets):
        raise ValueError("WAVECAR record offsets must be within the file")
    return WavecarDataset(source, header, file_size, record_offsets)


@jaxtyped(typechecker=beartype)
def make_in_memory_plane_wave_source(
    batch: PlaneWaveBatch,
    *,
    capabilities: Tuple[str, ...],
    state_ref: str,
    derivative_mode: str = "exact_ad",
) -> InMemoryPlaneWaveSource:
    """Create a traced plane-wave source with declared capabilities."""
    if not capabilities or not state_ref or not derivative_mode:
        raise ValueError("plane-wave source metadata must be nonempty")
    return InMemoryPlaneWaveSource(
        batch, capabilities, state_ref, derivative_mode
    )


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
    """Create a f64/c128 bounded plane-wave compute carrier."""
    return PlaneWaveBatch(
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
