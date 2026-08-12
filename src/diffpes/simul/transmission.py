"""Apply calibrated analyser transmission.

Extended Summary
----------------
This module evaluates the fixed-domain positive analyser response.

Routine Listings
----------------
:func:`apply_transmission`
    Apply analyser transmission to intensity at true kinetic energy.
:func:`transmission_shape`
    Evaluate positive monotone analyser transmission with fixed mean one.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List
from jaxtyping import Array, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.types import DetectorCalibration


def _integrated_bernstein(
    x: Float64[Array, "..."],
    degree: int,
) -> Float64[Array, "... q"]:
    r"""PRIVATE: Integrate every Bernstein basis function from zero to x.

    Parameters
    ----------
    x : Float64[Array, "..."]
        Normalized calibration-domain coordinate.
    degree : int
        Static Bernstein degree, one or two for the public model.

    Returns
    -------
    integrals : Float64[Array, "... q"]
        Analytic values :math:`\int_0^x B_j^{degree}(t)\,dt`.
    """
    elevated_degree: int = degree + 1
    elevated: List[Float64[Array, "..."]] = [
        math.comb(elevated_degree, index)
        * x**index
        * (1.0 - x) ** (elevated_degree - index)
        for index in range(elevated_degree + 1)
    ]
    integrals: Float64[Array, "... q"] = jnp.stack(
        [
            sum(elevated[index + 1 :], start=jnp.zeros_like(x))
            / elevated_degree
            for index in range(degree + 1)
        ],
        axis=-1,
    )
    return integrals


def _transmission_log_response(
    normalized_energy: Float64[Array, "..."],
    raw_slopes: Float64[Array, " q"],
    sign: int,
) -> Float64[Array, "..."]:
    """PRIVATE: Evaluate the anchored monotone log-transmission polynomial.

    Parameters
    ----------
    normalized_energy : Float64[Array, "..."]
        Calibration-domain coordinate in ``[0, 1]``.
    raw_slopes : Float64[Array, " q"]
        Two or three unconstrained Bernstein derivative coordinates.
    sign : int
        Registered monotonic direction, ``-1`` or ``1``.

    Returns
    -------
    log_response : Float64[Array, "..."]
        Anchored integrated-Bernstein log response.
    """
    basis_integrals: Float64[Array, "... q"] = _integrated_bernstein(
        normalized_energy, raw_slopes.shape[0] - 1
    )
    slopes: Float64[Array, " q"] = jax.nn.softplus(raw_slopes)
    log_response: Float64[Array, "..."] = sign * jnp.sum(
        basis_integrals * slopes, axis=-1
    )
    return log_response


@jaxtyped(typechecker=beartype)
def transmission_shape(  # noqa: DOC503
    kinetic_energy_axis_ev: Float64[Array, " n_e"],
    raw_slopes: Float64[Array, " q"],
    calibration: DetectorCalibration,
) -> Float64[Array, " n_e"]:
    """Evaluate positive monotone analyser transmission with fixed mean one.

    Two or three softplus slope coordinates weight a Bernstein basis for the
    derivative of log transmission. The calibration fixes its sign. A
    64-point Gauss--Legendre mean over the complete calibration domain removes
    its constant mode. The caller's query window never controls normalization.

    :see: :class:`~.test_transmission.TestTransmissionShape`

    Parameters
    ----------
    kinetic_energy_axis_ev : Float64[Array, " n_e"]
        True kinetic energies inside the fixed calibration domain in eV.
    raw_slopes : Float64[Array, " q"]
        Exactly two or three unconstrained log-slope coordinates.
    calibration : DetectorCalibration
        Fixed domain and registered monotonic direction.

    Returns
    -------
    transmission : Float64[Array, " n_e"]
        Positive normalized analyser response at every query energy.

    Raises
    ------
    ValueError
        If either numerical input is not one-dimensional or ``q`` is invalid.
    EquinoxRuntimeError
        If values are non-finite, outside the domain, or overflow the model.

    Notes
    -----
    Cropping, padding, or rebinning the query cannot change the affine basis or
    normalization. Version one provides no extrapolation mode.
    """
    if kinetic_energy_axis_ev.ndim != 1 or raw_slopes.ndim != 1:
        raise ValueError(
            "transmission energy and slopes must be one-dimensional"
        )
    if raw_slopes.shape[0] not in (2, 3):  # noqa: PLR2004
        raise ValueError("transmission requires exactly two or three slopes")
    domain: Float64[Array, " 2"] = calibration.transmission_reference_domain_ev
    energies: Float64[Array, " n_e"] = eqx.error_if(
        kinetic_energy_axis_ev,
        ~jnp.all(jnp.isfinite(kinetic_energy_axis_ev))
        | jnp.any(kinetic_energy_axis_ev < domain[0])
        | jnp.any(kinetic_energy_axis_ev > domain[1]),
        "transmission queries must stay inside the calibration domain",
    )
    slopes: Float64[Array, " q"] = eqx.error_if(
        raw_slopes,
        ~jnp.all(jnp.isfinite(raw_slopes)),
        "transmission slopes must be finite",
    )
    span: Float64[Array, ""] = domain[1] - domain[0]
    normalized_query: Float64[Array, " n_e"] = (energies - domain[0]) / span
    query_log_response: Float64[Array, " n_e"] = _transmission_log_response(
        normalized_query, slopes, calibration.transmission_monotonic_sign
    )
    gauss_nodes: Float64[NDArray, " n_quad"]
    gauss_weights: Float64[NDArray, " n_quad"]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(64)
    quadrature_nodes: Float64[Array, " n_quad"] = jnp.asarray(
        0.5 * (gauss_nodes + 1.0), dtype=jnp.float64
    )
    quadrature_weights: Float64[Array, " n_quad"] = jnp.asarray(
        0.5 * gauss_weights, dtype=jnp.float64
    )
    quadrature_log_response: Float64[Array, " n_quad"] = (
        _transmission_log_response(
            quadrature_nodes, slopes, calibration.transmission_monotonic_sign
        )
    )
    denominator: Float64[Array, ""] = jnp.sum(
        quadrature_weights * jnp.exp(quadrature_log_response)
    )
    transmission: Float64[Array, " n_e"] = (
        jnp.exp(query_log_response) / denominator
    )
    validated_transmission: Float64[Array, " n_e"] = eqx.error_if(
        transmission,
        ~jnp.all(jnp.isfinite(transmission))
        | ~jnp.all(transmission > 0.0)
        | ~jnp.isfinite(denominator)
        | ~(denominator > 0.0),
        "transmission response and normalization must be finite and positive",
    )
    return validated_transmission


@jaxtyped(typechecker=beartype)
def apply_transmission(  # noqa: DOC503
    intensity: Float64[Array, "... n_e"],
    kinetic_energy_axis_ev: Float64[Array, " n_e"],
    raw_slopes: Float64[Array, " q"],
    calibration: DetectorCalibration,
) -> Float64[Array, "... n_e"]:
    """Apply analyser transmission to intensity at true kinetic energy.

    The fixed-domain shape broadcasts across every leading intensity axis.

    :see: :class:`~.test_transmission.TestApplyTransmission`

    Parameters
    ----------
    intensity : Float64[Array, "... n_e"]
        Finite pre-resolution physical intensity.
    kinetic_energy_axis_ev : Float64[Array, " n_e"]
        True kinetic-energy coordinates for the trailing intensity axis.
    raw_slopes : Float64[Array, " q"]
        Two or three unconstrained analyser shape coordinates.
    calibration : DetectorCalibration
        Fixed transmission calibration domain and monotonic sign.

    Returns
    -------
    transmitted : Float64[Array, "... n_e"]
        Intensity multiplied along its trailing energy axis.

    Raises
    ------
    ValueError
        If the intensity and kinetic-energy dimensions disagree.
    EquinoxRuntimeError
        If the intensity or transmission inputs are invalid.

    Notes
    -----
    The canonical chain calls this function before detector resolution. It
    therefore evaluates throughput at true, not recorded, kinetic energy.
    """
    if (
        intensity.ndim < 1
        or intensity.shape[-1] != kinetic_energy_axis_ev.shape[0]
    ):
        raise ValueError("intensity trailing axis must match kinetic energy")
    values: Float64[Array, "... n_e"] = eqx.error_if(
        intensity,
        ~jnp.all(jnp.isfinite(intensity)),
        "transmission input intensity must be finite",
    )
    shape: Float64[Array, " n_e"] = transmission_shape(
        kinetic_energy_axis_ev, raw_slopes, calibration
    )
    transmitted: Float64[Array, "... n_e"] = values * shape
    return transmitted


__all__: list[str] = [
    "apply_transmission",
    "transmission_shape",
]
