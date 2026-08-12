"""PRIVATE: Dispatch conservative source-to-detector maps.

Extended Summary
----------------
This private module dispatches source carriers and mixes calibrated domains.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Float64

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DetectorCalibration,
    DetectorEffects,
    ExperimentGeometry,
)

from ._detector_cube import _map_cube
from ._detector_geometry import (
    _captured_fraction,
    _gauss_legendre_rule,
    _source_flux,
    _validate_common,
)
from ._detector_spectrum import _map_spectrum


def _map_source_to_detector_with_order(
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    euler_angles_rad: Optional[Float64[Array, " 3"]] = None,
    *,
    order: int = 4,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute one source map with a registered quadrature order.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source density.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.
    euler_angles_rad : Optional[Float64[Array, " 3"]], optional
        Active source-domain z--y--z rotation.  ``None`` selects identity.
    order : int, optional
        Static quadrature order, four for production and eight for the
        convergence comparison. Default is four.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Native detector density and captured source-flux fraction.
    """
    _gauss_legendre_rule(order)
    angles: Float64[Array, " 3"] = (
        jnp.zeros(3, dtype=jnp.float64)
        if euler_angles_rad is None
        else jnp.asarray(euler_angles_rad, dtype=jnp.float64)
    )
    angles = _validate_common(source, geometry, calibration, angles)
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
    if isinstance(source, ArpesCube):
        mapped = _map_cube(source, geometry, calibration, angles, order)
    else:
        mapped = _map_spectrum(source, geometry, calibration, angles, order)
    density: Float64[Array, "u v e"] = eqx.error_if(
        mapped[0],
        ~jnp.all(jnp.isfinite(mapped[0])) | ~jnp.all(mapped[0] >= 0.0),
        "mapped detector density must be finite and nonnegative",
    )
    fraction: Float64[Array, ""] = eqx.error_if(
        mapped[1],
        ~jnp.isfinite(mapped[1]) | (mapped[1] < 0.0) | (mapped[1] > 1.0),
        (
            "captured detector fraction must lie in [0, 1]; refine target "
            "bins if quadrature creates flux"
        ),
    )
    validated: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        density,
        fraction,
    )
    return validated


def _map_source_to_detector(
    source: Union[ArpesCube, ArpesSpectrum],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute one unrotated source map with production quadrature.

    Parameters
    ----------
    source : Union[ArpesCube, ArpesSpectrum]
        Self-describing source density.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Explicit native detector target.

    Returns
    -------
    mapped : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Native detector density and captured source-flux fraction.
    """
    mapped: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        _map_source_to_detector_with_order(
            source, geometry, calibration, order=4
        )
    )
    return mapped


def _map_and_mix_domains(
    physical_by_domain: Tuple[Union[ArpesCube, ArpesSpectrum], ...],
    geometry: ExperimentGeometry,
    calibration: DetectorCalibration,
    effects: DetectorEffects,
) -> Tuple[Float64[Array, "u v e"], Float64[Array, ""]]:
    """PRIVATE: Compute rotated domain maps before detector-space mixing.

    Parameters
    ----------
    physical_by_domain : Tuple[Union[ArpesCube, ArpesSpectrum], ...]
        Nonempty static tuple of self-describing source densities.
    geometry : ExperimentGeometry
        Experiment kinematics and sample azimuth.
    calibration : DetectorCalibration
        Common explicit native detector target.
    effects : DetectorEffects
        Traced domain rotations/logits and registered frame identities.

    Returns
    -------
    mixed : Tuple[Float64[Array, "u v e"], Float64[Array, ""]]
        Softmax-weighted detector density and its flux-weighted captured
        fraction.

    Raises
    ------
    ValueError
        If the domain count or frame identities disagree.

    Notes
    -----
    Mapping precedes mixing.  Differently oriented source arrays are never
    combined pointwise.
    """
    n_domain: int = len(physical_by_domain)
    if n_domain < 1:
        raise ValueError("detector domain tuple cannot be empty")
    if effects.domain_logits.shape[0] != n_domain:
        raise ValueError("detector domain logits and source count disagree")
    if effects.domain_euler_angles_rad.shape != (n_domain, 3):
        raise ValueError("detector rotations and source count disagree")
    if effects.coordinate_density != "per_native_volume":
        raise ValueError("detector mapping requires per-native-volume density")
    if len(effects.domain_frame_ids) != n_domain:
        raise ValueError(
            "detector frame identifiers and source count disagree"
        )
    domain_index: int
    densities: List[Float64[Array, "u v e"]] = []
    source_fluxes: List[Float64[Array, ""]] = []
    for domain_index in range(n_domain):
        source: Union[ArpesCube, ArpesSpectrum] = physical_by_domain[
            domain_index
        ]
        if effects.domain_frame_ids[domain_index] != source.cartesian_frame_id:
            raise ValueError(
                "detector domain and source frame identifiers disagree"
            )
        density: Float64[Array, "u v e"]
        fraction: Float64[Array, ""]
        density, fraction = _map_source_to_detector_with_order(
            source,
            geometry,
            calibration,
            effects.domain_euler_angles_rad[domain_index],
            order=4,
        )
        del fraction
        densities.append(density)
        source_fluxes.append(_source_flux(source))
    density_stack: Float64[Array, "d u v e"] = jnp.stack(densities)
    source_flux_stack: Float64[Array, " d"] = jnp.stack(source_fluxes)
    weights: Float64[Array, " d"] = jax.nn.softmax(effects.domain_logits)
    mixed_density: Float64[Array, "u v e"] = jnp.einsum(
        "d,duve->uve", weights, density_stack
    )
    mixed_source_flux: Float64[Array, ""] = jnp.dot(weights, source_flux_stack)
    mixed_fraction: Float64[Array, ""] = _captured_fraction(
        mixed_density, mixed_source_flux, calibration
    )
    mixed_fraction = eqx.error_if(
        mixed_fraction,
        ~jnp.isfinite(mixed_fraction)
        | (mixed_fraction < 0.0)
        | (mixed_fraction > 1.0),
        "mixed captured detector fraction must lie in [0, 1]",
    )
    mixed: Tuple[Float64[Array, "u v e"], Float64[Array, ""]] = (
        mixed_density,
        mixed_fraction,
    )
    return mixed


__all__: list[str] = []
