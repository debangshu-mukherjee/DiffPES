"""Compose coherent ARPES and photon-energy-scan drivers.

Extended Summary
----------------
This module exposes the canonical detector-raster boundary.
It also exposes the photon-energy-scan boundary.

Routine Listings
----------------
:func:`hv_map_at_energy`
    Interpolate a photon-energy scan at one sampled binding energy.
:func:`normalize_intensity`
    Return an explicit display-only normalization of carrier values.
:func:`simulate_arpes`
    Simulate the canonical detector raster.
:func:`simulate_arpes_cut`
    Simulate the canonical path-cut detector raster.
:func:`simulate_hv_scan`
    Simulate a single-domain pre-detector photon-energy scan.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Tuple, Union
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    KPath,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SurfaceCell,
    TBModel,
)

from . import effects
from ._kz_spectrum import _bulk_domain_intensity
from ._source_carriers import _physical_cubes, _physical_spectra
from ._spectrum_stream import _stream_domain_intensity
from ._spectrum_validation import _validate_kz_mode_inputs


@jaxtyped(typechecker=beartype)
def simulate_arpes(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kgrid: KGrid,
    energy_axis: Float64[Array, " n_e"],
    detector_calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> DetectorRaster:
    """Simulate the canonical detector raster.

    The driver constructs one physical source cube per static domain through
    the degeneracy-safe resolvent.  It then invokes the single shared detector
    chain.  No normalization, random sampling, fidelity construction, or
    approximation-tier dispatch occurs here.

    :see: :class:`~.test_spectrum.TestSimulateArpes`

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy orbital Hamiltonians in eV, one per domain.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers on exactly ``kgrid.kpoints``.
    radial_spec : RadialSpec
        Shell-shared radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit plane-wave or Coulomb radial final state.
    geometry : ExperimentGeometry
        Traced single-acquisition geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy model.
    kgrid : KGrid
        Fixed-kz separable sample-Cartesian source raster.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned ``E - E_F`` samples in eV.
    detector_calibration : DetectorCalibration
        Explicit target bins, PSF widths, and transmission domain.
    detector_effects : DetectorEffects
        Complete deterministic detector and nuisance parameters.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize live chunks in reverse mode. Default is ``True``.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Per-domain bulk models for ``bulk_direct`` and ``bulk_kz``. Default is
        ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Per-domain exact surface frames for bulk/coherent modes. Default is
        ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The independent
        convergence profile certifies 2048 nodes. Every other uniform count
        is a caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        ``"native_direct"``, ``"bulk_direct"``, ``"bulk_kz"``, or
        ``"coherent_slab"``. Default is ``"native_direct"``.

    Returns
    -------
    raster : DetectorRaster
        Native-axis expected detector counts.

    Raises
    ------
    ValueError
        If domain/static shapes disagree or the source grid lacks separable
        sample-Cartesian axes.
    EquinoxRuntimeError
        If a traced physical, kinematic, spectral, or detector contract fails.

    Notes
    -----
    ``DiagonalizedBands`` is metadata at this seam.  The explicit Hamiltonian
    owns resolvent values and derivatives; it is never reconstructed from the
    carrier's eigensystem.
    """
    _validate_kz_mode_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        bulk_models_by_domain,
        surface_cells_by_domain,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    physical_by_domain: Tuple[ArpesCube, ...] = _physical_cubes(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        geometry,
        self_energy,
        kgrid,
        energy_axis,
        eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        kz_nodes_frac=kz_nodes_frac,
        kz_mode=kz_mode,
    )
    raster: DetectorRaster = effects.apply_detector_effects(
        physical_by_domain,
        geometry,
        detector_calibration,
        detector_effects,
    )
    return raster


@jaxtyped(typechecker=beartype)
def simulate_arpes_cut(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kpath: KPath,
    energy_axis: Float64[Array, " n_e"],
    detector_calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> DetectorRaster:
    """Simulate the canonical path-cut detector raster.

    Every domain becomes an ``ArpesSpectrum`` carrying cumulative distance,
    the complete sample-Cartesian path, and its registered frame identity.
    The slit spans one bin in native detector ``v`` coordinates.  The shared
    detector chain applies all resolution after mapping.

    :see: :class:`~.test_spectrum.TestSimulateArpesCut`

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy orbital Hamiltonians in eV, one per domain.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers on exactly ``kpath.kpoints``.
    radial_spec : RadialSpec
        Shell-shared radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit plane-wave or Coulomb radial final state.
    geometry : ExperimentGeometry
        Traced single-acquisition geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy model.
    kpath : KPath
        Fractional source path retaining complete Cartesian vectors.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned ``E - E_F`` samples in eV.
    detector_calibration : DetectorCalibration
        Explicit slit target bins, PSF widths, and transmission domain.
    detector_effects : DetectorEffects
        Complete deterministic detector and nuisance parameters.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize live chunks in reverse mode. Default is ``True``.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Per-domain bulk models for ``bulk_direct`` and ``bulk_kz``. Default is
        ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Per-domain exact surface frames for bulk/coherent modes. Default is
        ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The independent
        convergence profile certifies 2048 nodes. Every other uniform count
        is a caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        ``"native_direct"``, ``"bulk_direct"``, ``"bulk_kz"``, or
        ``"coherent_slab"``. Default is ``"native_direct"``.

    Returns
    -------
    raster : DetectorRaster
        Native slit-axis expected detector counts.

    Raises
    ------
    ValueError
        If domain/static shapes disagree or the path has fewer than two nodes.
    EquinoxRuntimeError
        If a traced physical, kinematic, spectral, or detector contract fails.

    Notes
    -----
    ``DiagonalizedBands`` remains metadata.  The explicit Hamiltonian owns
    resolvent values and derivatives through the complete cut path.
    """
    _validate_kz_mode_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        bulk_models_by_domain,
        surface_cells_by_domain,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    physical_by_domain: Tuple[ArpesSpectrum, ...] = _physical_spectra(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        geometry,
        self_energy,
        kpath,
        energy_axis,
        eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        kz_nodes_frac=kz_nodes_frac,
        kz_mode=kz_mode,
    )
    raster: DetectorRaster = effects.apply_detector_effects(
        physical_by_domain,
        geometry,
        detector_calibration,
        detector_effects,
    )
    return raster


@jaxtyped(typechecker=beartype)
def simulate_hv_scan(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] | None,
    bands: DiagonalizedBands | None,
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kpath: KPath,
    energy_axis: Float64[Array, " n_e"],
    photon_energies_ev: Float64[Array, " n_hv"],
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_model: TBModel | None = None,
    surface_cell: SurfaceCell | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> Float64[Array, "n_hv n_k n_e"]:
    """Simulate a single-domain pre-detector photon-energy scan.

    The scan keeps the photon-energy axis explicit and re-evaluates exact
    finite-energy kinematics and matrix elements at every row. It carries no
    detector response, transmission, sampling, or display normalization.

    :see: :class:`~.test_spectrum.TestSimulateHvScan`

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_k n_orb n_orb"] | None
        Explicit Hamiltonian for native/coherent modes; ``None`` in bulk
        modes.
    bands : DiagonalizedBands | None
        Metadata paired with explicit H; ``None`` in bulk modes.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final-state model.
    geometry : ExperimentGeometry
        Base experiment geometry; each row replaces only photon energy.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy.
    kpath : KPath
        Fixed-shape source path.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing relative-energy samples.
    photon_energies_ev : Float64[Array, " n_hv"]
        Positive finite photon energies.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Static sampled-energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize scan bodies in reverse mode. Default is ``True``.
    bulk_model : TBModel | None, optional
        Single bulk model for either bulk mode. Default is ``None``.
    surface_cell : SurfaceCell | None, optional
        Exact surface frame for bulk/coherent mode. Default is ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The independent
        convergence profile certifies 2048 nodes. Every other uniform count
        is a caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        Registered mutually exclusive mode. Default is ``"native_direct"``.

    Returns
    -------
    scan : Float64[Array, "n_hv n_k n_e"]
        Intrinsic single-domain intensity for every photon energy.

    Raises
    ------
    ValueError
        If the mode/carrier surface is invalid or an axis is empty.
    EquinoxRuntimeError
        If traced photon energies, kinematics, or physics leave their domain.

    Notes
    -----
    A :func:`jax.lax.scan` owns the photon-energy loop. Node count and all
    chunk choices remain static; photon-energy values remain differentiable.
    """
    if photon_energies_ev.ndim != 1 or photon_energies_ev.shape[0] < 1:
        raise ValueError("photon_energies_ev must be a nonempty vector")
    if kpath.kz is None or kpath.kpoints.shape[0] < 2:  # noqa: PLR2004
        raise ValueError(
            "simulate_hv_scan requires a fixed-kz path with two points"
        )
    checked_photon_energies: Float64[Array, " n_hv"] = eqx.error_if(
        photon_energies_ev,
        ~jnp.all(jnp.isfinite(photon_energies_ev))
        | jnp.any(photon_energies_ev <= 0.0),
        "photon energies must be finite and positive",
    )
    hamiltonian_tuple: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...] = (
        () if hamiltonian is None else (hamiltonian,)
    )
    bands_tuple: Tuple[DiagonalizedBands, ...] = (
        () if bands is None else (bands,)
    )
    bulk_tuple: Tuple[TBModel, ...] | None = (
        None if bulk_model is None else (bulk_model,)
    )
    surface_tuple: Tuple[SurfaceCell, ...] | None = (
        None if surface_cell is None else (surface_cell,)
    )
    _validate_kz_mode_inputs(
        hamiltonian_tuple,
        bands_tuple,
        radial_spec,
        matrix_element_params,
        bulk_tuple,
        surface_tuple,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    bulk_mode: bool = kz_mode in {"bulk_direct", "bulk_kz"}

    def one_photon_energy(
        carry: None,
        photon_energy: Float64[Array, ""],
    ) -> Tuple[None, Float64[Array, "n_k n_e"]]:
        """Evaluate one physical scan row with an updated geometry leaf."""
        row_geometry: ExperimentGeometry = eqx.tree_at(
            lambda item: item.photon_energy_ev,
            geometry,
            photon_energy,
        )
        row_intensity: Float64[Array, "n_k n_e"]
        if bulk_mode:
            if bulk_model is None or surface_cell is None:
                raise ValueError("bulk scan requires a model and surface cell")
            row_intensity, _ = _bulk_domain_intensity(
                bulk_model,
                surface_cell,
                kpath.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                row_geometry,
                self_energy,
                eta,
                kz_nodes_frac,
                kz_mode,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
            )
        else:
            if hamiltonian is None or bands is None:
                raise ValueError("native/coherent scan requires H and bands")
            row_intensity, _ = _stream_domain_intensity(
                hamiltonian,
                bands,
                kpath.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                row_geometry,
                self_energy,
                eta,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
                use_inner_potential=kz_mode == "coherent_slab",
                surface_cell=(
                    surface_cell if kz_mode == "coherent_slab" else None
                ),
            )
        result: Tuple[None, Float64[Array, "n_k n_e"]] = (
            carry,
            row_intensity,
        )
        return result

    scan_step: Any = (
        jax.checkpoint(one_photon_energy) if checkpoint else one_photon_energy
    )
    scan: Float64[Array, "n_hv n_k n_e"]
    _, scan = jax.lax.scan(scan_step, None, checked_photon_energies)
    return scan


@jaxtyped(typechecker=beartype)
def hv_map_at_energy(  # noqa: DOC503
    scan: Float64[Array, "n_hv n_k n_e"],
    energy_axis: Float64[Array, " n_e"],
    energy_ev: ScalarFloat,
) -> Float64[Array, "n_k n_hv"]:
    """Interpolate a photon-energy scan at one sampled binding energy.

    The helper applies piecewise-linear interpolation on the caller-owned
    sampled-energy axis. It then returns momentum as the leading plotting axis.

    :see: :class:`~.test_spectrum.TestHvMapAtEnergy`

    Parameters
    ----------
    scan : Float64[Array, "n_hv n_k n_e"]
        Single-domain pre-detector photon-energy scan.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing sampled relative-energy axis.
    energy_ev : ScalarFloat
        Requested in-domain relative energy in eV.

    Returns
    -------
    hv_map : Float64[Array, "n_k n_hv"]
        Linearly interpolated path-by-photon-energy map.

    Raises
    ------
    ValueError
        If array axes disagree or the energy axis contains fewer than two
        nodes.
    EquinoxRuntimeError
        If the axis/query is non-finite, non-increasing, or out of domain.

    Notes
    -----
    The output orientation puts path momentum first for direct plotting.
    Query derivatives are piecewise linear away from sampled knots.
    """
    minimum_points: int = 2
    if (
        scan.ndim != 3  # noqa: PLR2004
        or energy_axis.ndim != 1
        or energy_axis.shape[0] < minimum_points
        or scan.shape[-1] != energy_axis.shape[0]
    ):
        raise ValueError(
            "scan and energy axis must have compatible sampled axes"
        )
    query: Float64[Array, ""] = jnp.asarray(energy_ev, dtype=jnp.float64)
    checked_axis: Float64[Array, " n_e"] = eqx.error_if(
        energy_axis,
        ~jnp.all(jnp.isfinite(energy_axis))
        | jnp.any(jnp.diff(energy_axis) <= 0.0)
        | ~jnp.isfinite(query)
        | (query < energy_axis[0])
        | (query > energy_axis[-1]),
        "energy axis must increase and the query must lie in its domain",
    )
    upper: Int32[Array, ""] = jnp.clip(
        jnp.searchsorted(checked_axis, query, side="right"),
        1,
        checked_axis.shape[0] - 1,
    )
    lower: Int32[Array, ""] = upper - 1
    fraction: Float64[Array, ""] = (query - checked_axis[lower]) / (
        checked_axis[upper] - checked_axis[lower]
    )
    values: Float64[Array, "n_hv n_k"] = (1.0 - fraction) * scan[
        :, :, lower
    ] + fraction * scan[:, :, upper]
    hv_map: Float64[Array, "n_k n_hv"] = jnp.swapaxes(values, 0, 1)
    return hv_map


@jaxtyped(typechecker=beartype)
def normalize_intensity(  # noqa: DOC105, DOC503
    carrier: Union[ArpesCube, ArpesSpectrum, DetectorRaster],
    mode: str = "none",
) -> Float64[Array, " ..."]:
    """Return an explicit display-only normalization of carrier values.

    The function returns a plain array rather than relabeling normalized or
    z-scored display values as physical intensity or expected counts.  Neither
    canonical driver calls this helper.

    :see: :class:`~.test_spectrum.TestNormalizeIntensity`

    Parameters
    ----------
    carrier : ArpesCube | ArpesSpectrum | DetectorRaster
        Physical source intensity or native expected counts.
    mode : str, optional
        ``"none"``, ``"sum"``, or ``"zscore"``. Default is ``"none"``.

    Returns
    -------
    normalized : Float64[Array, " ..."]
        Plain display array with the carrier's original shape.

    Raises
    ------
    ValueError
        If ``mode`` has an unsupported value.
    EquinoxRuntimeError
        If a requested sum or standard deviation is zero.

    Notes
    -----
    ``"sum"`` divides by the complete-array sum.  ``"zscore"`` subtracts the
    complete-array mean and divides by the population standard deviation.
    These crop-dependent transforms are unsuitable for a physical likelihood.
    """
    if mode not in {"none", "sum", "zscore"}:
        raise ValueError("mode must be 'none', 'sum', or 'zscore'")
    values: Float64[Array, " ..."] = (
        carrier.expected_counts
        if isinstance(carrier, DetectorRaster)
        else carrier.intensity
    )
    if mode == "none":
        return values
    if mode == "sum":
        total: Float64[Array, ""] = jnp.sum(values)
        checked_total: Float64[Array, ""] = eqx.error_if(
            total,
            total == 0.0,
            "sum normalization requires nonzero total intensity",
        )
        normalized_sum: Float64[Array, " ..."] = values / checked_total
        return normalized_sum
    mean: Float64[Array, ""] = jnp.mean(values)
    standard_deviation: Float64[Array, ""] = jnp.std(values)
    checked_deviation: Float64[Array, ""] = eqx.error_if(
        standard_deviation,
        standard_deviation == 0.0,
        "zscore normalization requires nonzero standard deviation",
    )
    normalized_zscore: Float64[Array, " ..."] = (
        values - mean
    ) / checked_deviation
    return normalized_zscore


__all__: list[str] = [
    "hv_map_at_energy",
    "normalize_intensity",
    "simulate_arpes",
    "simulate_arpes_cut",
    "simulate_hv_scan",
]
