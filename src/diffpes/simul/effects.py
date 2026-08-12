"""Compose source mapping and deterministic detector effects.

Extended Summary
----------------
This module exposes the canonical source-to-count detector boundary.

Routine Listings
----------------
:func:`apply_detector_effects`
    Apply the complete deterministic source-to-count detector chain.
:func:`map_source_to_detector`
    Convert one source density to native detector bins conservatively.
"""

from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Float64, jaxtyped

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    ExperimentGeometry,
    make_detector_raster,
)

from ._detector_map import _map_and_mix_domains, _map_source_to_detector
from .detector_response import expected_counts
from .resolution import apply_resolution
from .transmission import apply_transmission


@jaxtyped(typechecker=beartype)
def map_source_to_detector(  # noqa: DOC502, DOC503
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """Convert one source density to native detector bins conservatively.

    The source carries its complete Cartesian axes and registered sample
    frame. The calibration independently owns every native target edge. The
    returned density uses the declared per-native-volume convention, while
    the scalar reports the fraction of source flux captured under the
    calibrated ``loss`` boundary policy.

    :see: :class:`~.test_effects.TestMapSourceToDetector`

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source-coordinate physical intensity.
    geometry : ExperimentGeometry
        Traced sample and photoemission geometry.
    calibration : DetectorCalibration
        Explicit native detector target and boundary convention.

    Returns
    -------
    density : Float64[Array, "u v e"]
        Native detector density before transmission and resolution.
    captured_fraction : Float64[Array, ""]
        Captured source-flux fraction under the ``loss`` policy.

    Raises
    ------
    ValueError
        If the source carrier, registered frame, target dimensionality, or
        slit/map contract is invalid.
    EquinoxRuntimeError
        If a traced geometry, source, or calibration value leaves the valid
        detector chart.

    Notes
    -----
    Production uses four Gauss--Legendre nodes on every active target axis.
    The mapper performs no target inference, row normalization, reflection,
    or source-axis relabeling. Signed diagonal and antidiagonal domain maps
    may cross source-support boundaries: those branches split quadrature at
    every support and exterior-face seam before integration. General domain
    rotations instead require a conservative interval enclosure. Every
    inverse-mapped target bin must lie strictly inside the source exterior
    faces. Eager and compiled calls reject rotations whose enclosures touch or
    cross that boundary. Coordinate derivatives for the general branch
    therefore claim only the smooth, strictly enclosed interior chart, never
    a support crossing or topology switch.

    An :class:`~diffpes.types.ArpesSpectrum` is already a line density along
    its declared path, integrated over exactly one transverse ``v`` aperture.
    Its cumulative path coordinate must be strictly increasing and its full
    Cartesian path must remain inside that aperture. The slit mapper applies
    the absolute path-to-detector Jacobian exactly once. It also divides by the
    declared aperture width once. It never promotes a cut into an inferred 2-D
    source.
    """
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        _map_source_to_detector(source, geometry, calibration)
    )
    return mapped


@jaxtyped(typechecker=beartype)
def apply_detector_effects(  # noqa: DOC502, DOC503
    physical_by_domain: Tuple[Union[ArpesCube, ArpesSpectrum], ...],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> DetectorRaster:
    """Apply the complete deterministic source-to-count detector chain.

    The chain actively rotates and conservatively maps each source domain
    before traced softmax mixing on common detector bins. The mixed density
    then passes through true-kinetic-energy transmission, native-coordinate
    finite-volume resolution, background, sensitivity, exposure, explicit
    bin-volume conversion, and the optional calibrated post-count response.

    :see: :class:`~.test_effects.TestApplyDetectorEffects`

    Parameters
    ----------
    physical_by_domain : Tuple[Union[ArpesCube, ArpesSpectrum], ...]
        Nonempty static tuple of self-describing source densities.
    geometry : ExperimentGeometry
        Traced photoemission and sample geometry.
    calibration : DetectorCalibration
        Explicit native detector target, PSF, and transmission domain.
    effects : DetectorEffects
        Domain, analyser, background, sensitivity, and acquisition state.

    Returns
    -------
    raster : DetectorRaster
        Single-channel native-coordinate expected counts.

    Raises
    ------
    ValueError
        If domains, frames, source carriers, or target dimensions disagree.
    EquinoxRuntimeError
        If a traced physical or detector coordinate is invalid.

    Notes
    -----
    Integer acquisition and display normalization are intentionally separate.
    Evaluate transmission at true kinetic energy before the recorded-bin PSF.
    Apply background and sensitivity only after that resolution. Domain
    mapping inherits :func:`map_source_to_detector`'s complete coordinate
    contract. Signed diagonal and antidiagonal maps split support seams.
    General rotations require strict enclosure and claim smooth-interior
    derivatives only. Slit spectra retain single-aperture line-density
    semantics.
    """
    mixed_density: Float64[Array, "u v e"]
    mixed_density, _ = _map_and_mix_domains(
        physical_by_domain, geometry, calibration, effects
    )
    recorded_energy: Float64[Array, " e"] = 0.5 * (
        calibration.energy_bin_edges_ev[:-1]
        + calibration.energy_bin_edges_ev[1:]
    )
    kinetic_energy: Float64[Array, " e"] = (
        geometry.photon_energy_ev - geometry.work_function_ev + recorded_energy
    )
    transmitted: Float64[Array, "u v e"] = apply_transmission(
        mixed_density,
        kinetic_energy,
        effects.transmission_raw_slopes,
        calibration,
    )
    resolved: Float64[Array, "u v e"] = apply_resolution(
        transmitted, calibration
    )[0]
    rates: Float64[Array, "1 u v e"] = expected_counts(
        resolved[None, ...], calibration, effects
    )
    detector_u: Float64[Array, " u"] = 0.5 * (
        calibration.u_bin_edges[:-1] + calibration.u_bin_edges[1:]
    )
    detector_v: Float64[Array, " v"] = 0.5 * (
        calibration.v_bin_edges[:-1] + calibration.v_bin_edges[1:]
    )
    raster: DetectorRaster = make_detector_raster(
        rates,
        detector_u,
        detector_v,
        recorded_energy,
        channel_labels=("intensity",),
        coordinate_system=calibration.coordinate_system,
    )
    return raster


__all__: list[str] = [
    "apply_detector_effects",
    "map_source_to_detector",
]
