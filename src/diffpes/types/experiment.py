r"""Define the geometry of an ARPES experiment.

Extended Summary
----------------
This module defines a JAX-compatible carrier for the beamline, sample, and
instrument geometry. The numerical fields remain traced for forward and
inverse calculations. The slit selector remains static because it controls
the detector convention.

Routine Listings
----------------
:class:`ExperimentGeometry`
    Store the geometry and resolution of an ARPES experiment.
:func:`make_experiment_geometry`
    Create a validated geometry for an ARPES experiment.

Notes
-----
The factory removes the norm of the polarization vector. This operation fixes
the intensity-scale gauge at the experiment boundary. It also verifies that
the declared electric field is transverse to the photon propagation direction
defined by the incidence angles.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from .aliases import ScalarFloat

_SLIT_ORIENTATIONS: Tuple[str, ...] = ("H", "V")
_TRANSVERSALITY_ATOL: float = 1.0e-10


class ExperimentGeometry(eqx.Module):
    """Store the geometry and resolution of an ARPES experiment.

    This PyTree groups the traced quantities that define one ARPES
    measurement. JAX differentiates every numerical field. The static slit
    field selects the detector frame and causes retracing when it changes.

    :see: :class:`~.test_experiment.TestExperimentGeometry`

    Attributes
    ----------
    photon_energy_ev : Float64[Array, ""]
        Photon energy in eV.
    polarization : Complex128[Array, "3"]
        Unit polarization vector in the laboratory frame, transverse to the
        photon direction declared by ``incidence_theta`` and
        ``incidence_phi``.
    incidence_theta : Float64[Array, ""]
        Incidence angle from the surface normal in radians.
    incidence_phi : Float64[Array, ""]
        Azimuthal incidence angle in radians.
    sample_azimuth : Float64[Array, ""]
        Sample rotation about the surface normal in radians.
    work_function_ev : Float64[Array, ""]
        Work function in eV.
    inner_potential_ev : Float64[Array, ""]
        Inner potential in eV.
    temperature_k : Float64[Array, ""]
        Strictly positive sample temperature in kelvin.
    energy_resolution_ev : Float64[Array, ""]
        Full width at half maximum of the energy resolution in eV.
    momentum_resolution_inv_ang : Float64[Array, ""]
        Full width at half maximum of the momentum resolution in
        1/Angstrom.
    mean_free_path_ang : Float64[Array, ""]
        Mean free path of the photoelectron in Angstrom.
    slit : str
        Detector slit orientation. This field is **static**. A change causes
        JAX to retrace the receiving function.

    Notes
    -----
    The traced fields support calibration and geometry inversion. The
    normalized polarization removes its intensity-scale gauge. The carrier
    represents a propagating electromagnetic field, so its polarization is
    transverse to the incidence direction.

    Treat polarization and incidence angles as constrained coordinates during
    inversion. Build transverse fields from two Jones coefficients at each
    incidence direction, or project updates onto that tangent space.
    Independent PyTree updates can leave the physical manifold and bypass
    factory validation.

    See Also
    --------
    make_experiment_geometry : Create a validated geometry for an ARPES
        experiment.
    """

    photon_energy_ev: Float64[Array, ""]
    polarization: Complex128[Array, "3"]
    incidence_theta: Float64[Array, ""]
    incidence_phi: Float64[Array, ""]
    sample_azimuth: Float64[Array, ""]
    work_function_ev: Float64[Array, ""]
    inner_potential_ev: Float64[Array, ""]
    temperature_k: Float64[Array, ""]
    energy_resolution_ev: Float64[Array, ""]
    momentum_resolution_inv_ang: Float64[Array, ""]
    mean_free_path_ang: Float64[Array, ""]
    slit: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_experiment_geometry(  # noqa: DOC503, PLR0913
    photon_energy_ev: ScalarFloat,
    polarization: Complex128[Array, "3"],
    incidence_theta: ScalarFloat = 0.0,
    incidence_phi: ScalarFloat = 0.0,
    sample_azimuth: ScalarFloat = 0.0,
    work_function_ev: ScalarFloat = 4.0,
    inner_potential_ev: ScalarFloat = 10.0,
    temperature_k: ScalarFloat = 10.0,
    energy_resolution_ev: ScalarFloat = 0.02,
    momentum_resolution_inv_ang: ScalarFloat = 0.01,
    mean_free_path_ang: ScalarFloat = 10.0,
    slit: str = "H",
) -> ExperimentGeometry:
    """Create a validated geometry for an ARPES experiment.

    The factory converts all numerical inputs to JAX arrays. It then fixes the
    intensity-scale gauge by normalizing the complex polarization vector and
    checks that it is transverse to the declared photon direction.

    :see: :class:`~.test_experiment.TestMakeExperimentGeometry`

    Implementation Logic
    --------------------
    1. **Convert the inputs**::

           photon_energy = jnp.asarray(photon_energy_ev, dtype=jnp.float64)

       The conversion gives each traced field a stable JAX dtype.

    2. **Validate the traced values**::

           photon_energy = eqx.error_if(photon_energy, invalid, message)

       The runtime checks remain active during compiled execution.

    3. **Normalize and validate the polarization**::

           normalized_polarization = checked_polarization / safe_norm

       The safe denominator keeps the invalid zero-norm branch finite. The
       normalized field must obey ``abs(k_hat dot epsilon) <= 1e-10`` with
       ``k_hat`` derived from the incidence angles.

    4. **Return the named carrier**::

           return geometry

       The carrier preserves each valid numerical input as a traced leaf.

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    polarization : Complex128[Array, "3"]
        Nonzero complex polarization vector in the laboratory frame. It must
        be transverse to the photon direction declared by the incidence
        angles.
    incidence_theta : ScalarFloat, optional
        Incidence angle from the surface normal in radians. Default is 0.0.
    incidence_phi : ScalarFloat, optional
        Azimuthal incidence angle in radians. Default is 0.0.
    sample_azimuth : ScalarFloat, optional
        Sample rotation about the surface normal in radians. Default is 0.0.
    work_function_ev : ScalarFloat, optional
        Work function in eV. Default is 4.0.
    inner_potential_ev : ScalarFloat, optional
        Inner potential in eV. Default is 10.0.
    temperature_k : ScalarFloat, optional
        Strictly positive sample temperature in kelvin. Default is 10.0.
    energy_resolution_ev : ScalarFloat, optional
        Energy-resolution width in eV. Default is 0.02.
    momentum_resolution_inv_ang : ScalarFloat, optional
        Momentum-resolution width in 1/Angstrom. Default is 0.01.
    mean_free_path_ang : ScalarFloat, optional
        Mean free path in Angstrom. Default is 10.0.
    slit : str, optional
        Detector slit orientation. This value is **static**. A change causes
        retracing. Default is ``"H"``.

    Returns
    -------
    geometry : ExperimentGeometry
        Validated experiment geometry with a unit polarization vector.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.
    EquinoxRuntimeError
        If an input is non-finite or violates its physical range. The factory
        also rejects a zero or non-transverse polarization vector.

    Notes
    -----
    The normalization is differentiable for every accepted polarization.
    The factory assigns no derivative at the rejected zero vector.
    The finite-temperature Fermi-Dirac model requires positive temperature.
    The carrier defines no separate static zero-temperature branch.
    The factory validates values, not later arbitrary PyTree mutations.
    Inversion code must rebuild polarization from a transverse basis when it
    changes incidence angles.
    """
    if slit not in _SLIT_ORIENTATIONS:
        message: str = "slit must be 'H' or 'V'"
        raise ValueError(message)

    photon_energy: Float64[Array, ""] = jnp.asarray(
        photon_energy_ev, dtype=jnp.float64
    )
    polarization_array: Complex128[Array, "3"] = jnp.asarray(
        polarization, dtype=jnp.complex128
    )
    theta: Float64[Array, ""] = jnp.asarray(incidence_theta, dtype=jnp.float64)
    phi: Float64[Array, ""] = jnp.asarray(incidence_phi, dtype=jnp.float64)
    azimuth: Float64[Array, ""] = jnp.asarray(
        sample_azimuth, dtype=jnp.float64
    )
    work_function: Float64[Array, ""] = jnp.asarray(
        work_function_ev, dtype=jnp.float64
    )
    inner_potential: Float64[Array, ""] = jnp.asarray(
        inner_potential_ev, dtype=jnp.float64
    )
    temperature: Float64[Array, ""] = jnp.asarray(
        temperature_k, dtype=jnp.float64
    )
    energy_resolution: Float64[Array, ""] = jnp.asarray(
        energy_resolution_ev, dtype=jnp.float64
    )
    momentum_resolution: Float64[Array, ""] = jnp.asarray(
        momentum_resolution_inv_ang, dtype=jnp.float64
    )
    mean_free_path: Float64[Array, ""] = jnp.asarray(
        mean_free_path_ang, dtype=jnp.float64
    )

    photon_energy = eqx.error_if(
        photon_energy,
        ~jnp.isfinite(photon_energy) | (photon_energy <= 0.0),
        "photon_energy_ev must be finite and positive",
    )
    finite_polarization: Float64[Array, ""] = jnp.all(
        jnp.isfinite(polarization_array)
    )
    polarization_array = eqx.error_if(
        polarization_array,
        ~finite_polarization,
        "polarization must be finite",
    )
    safe_polarization: Complex128[Array, "3"] = jnp.where(
        jnp.isfinite(polarization_array), polarization_array, 0.0 + 0.0j
    )
    polarization_norm: Float64[Array, ""] = jnp.sqrt(
        jnp.real(jnp.vdot(safe_polarization, safe_polarization))
    )
    safe_polarization = eqx.error_if(
        safe_polarization,
        polarization_norm <= 0.0,
        "polarization norm must be positive",
    )
    safe_norm: Float64[Array, ""] = jnp.where(
        polarization_norm > 0.0, polarization_norm, 1.0
    )
    normalized_polarization: Complex128[Array, "3"] = (
        safe_polarization / safe_norm
    )

    theta = eqx.error_if(
        theta, ~jnp.isfinite(theta), "incidence_theta must be finite"
    )
    phi = eqx.error_if(phi, ~jnp.isfinite(phi), "incidence_phi must be finite")
    safe_theta: Float64[Array, ""] = jnp.where(jnp.isfinite(theta), theta, 0.0)
    safe_phi: Float64[Array, ""] = jnp.where(jnp.isfinite(phi), phi, 0.0)
    photon_direction: Float64[Array, "3"] = -jnp.stack(
        (
            jnp.sin(safe_theta) * jnp.cos(safe_phi),
            jnp.sin(safe_theta) * jnp.sin(safe_phi),
            jnp.cos(safe_theta),
        )
    )
    longitudinal_amplitude: Complex128[Array, ""] = jnp.sum(
        photon_direction * normalized_polarization
    )
    normalized_polarization = eqx.error_if(
        normalized_polarization,
        jnp.abs(longitudinal_amplitude) > _TRANSVERSALITY_ATOL,
        "polarization must be transverse to the photon incidence direction",
    )
    azimuth = eqx.error_if(
        azimuth, ~jnp.isfinite(azimuth), "sample_azimuth must be finite"
    )
    work_function = eqx.error_if(
        work_function,
        ~jnp.isfinite(work_function)
        | (work_function < 0.0)
        | (work_function >= photon_energy),
        "work_function_ev must be finite and in [0, photon_energy_ev)",
    )
    inner_potential = eqx.error_if(
        inner_potential,
        ~jnp.isfinite(inner_potential) | (inner_potential < 0.0),
        "inner_potential_ev must be finite and nonnegative",
    )
    temperature = eqx.error_if(
        temperature,
        ~jnp.isfinite(temperature) | (temperature <= 0.0),
        "temperature_k must be finite and positive",
    )
    energy_resolution = eqx.error_if(
        energy_resolution,
        ~jnp.isfinite(energy_resolution) | (energy_resolution < 0.0),
        "energy_resolution_ev must be finite and nonnegative",
    )
    momentum_resolution = eqx.error_if(
        momentum_resolution,
        ~jnp.isfinite(momentum_resolution) | (momentum_resolution < 0.0),
        "momentum_resolution_inv_ang must be finite and nonnegative",
    )
    mean_free_path = eqx.error_if(
        mean_free_path,
        ~jnp.isfinite(mean_free_path) | (mean_free_path <= 0.0),
        "mean_free_path_ang must be finite and positive",
    )

    geometry: ExperimentGeometry = ExperimentGeometry(
        photon_energy_ev=photon_energy,
        polarization=normalized_polarization,
        incidence_theta=theta,
        incidence_phi=phi,
        sample_azimuth=azimuth,
        work_function_ev=work_function,
        inner_potential_ev=inner_potential,
        temperature_k=temperature,
        energy_resolution_ev=energy_resolution,
        momentum_resolution_inv_ang=momentum_resolution,
        mean_free_path_ang=mean_free_path,
        slit=slit,
    )
    return geometry


__all__: list[str] = [
    "ExperimentGeometry",
    "make_experiment_geometry",
]
