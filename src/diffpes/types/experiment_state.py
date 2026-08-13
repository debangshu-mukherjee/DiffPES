"""Define split experiment carriers without hidden workflow state.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`Acquisition`
    Define the ``Acquisition`` public contract.
:class:`Experiment`
    Define the ``Experiment`` public contract.
:class:`PhotonBeam`
    Define the ``PhotonBeam`` public contract.
:class:`SamplePose`
    Define the ``SamplePose`` public contract.
:class:`SampleState`
    Define the ``SampleState`` public contract.
:func:`make_acquisition`
    Compute the ``make_acquisition`` public contract.
:func:`make_experiment`
    Compute the ``make_experiment`` public contract.
:func:`make_photon_beam`
    Compute the ``make_photon_beam`` public contract.
:func:`make_sample_pose`
    Compute the ``make_sample_pose`` public contract.
:func:`make_sample_state`
    Compute the ``make_sample_state`` public contract.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import (
    ARRAY_MATRIX_NDIM,
    CARTESIAN_COMPONENTS,
    HERMITICITY_RELATIVE_TOLERANCE,
)


class PhotonBeam(eqx.Module):
    """Define the ``PhotonBeam`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestPhotonbeam`

    Attributes
    ----------
    photon_energy_ev : Float64[Array, ""]
        Store photon energy.
    polarization_lab : Complex128[Array, " 3"]
        Store laboratory polarization.
    incidence_theta_rad : Float64[Array, ""]
        Store the polar incidence angle.
    incidence_phi_rad : Float64[Array, ""]
        Store the azimuthal incidence angle.

    See Also
    --------
    make_photon_beam
        Construct a validated photon beam.
    """

    photon_energy_ev: Float64[Array, ""]
    polarization_lab: Complex128[Array, " 3"]
    incidence_theta_rad: Float64[Array, ""]
    incidence_phi_rad: Float64[Array, ""]


class SampleState(eqx.Module):
    """Define the ``SampleState`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestSamplestate`

    Attributes
    ----------
    temperature_k : Float64[Array, ""]
        Store sample temperature.
    work_function_ev : Float64[Array, ""]
        Store the work function.
    inner_potential_ev : Float64[Array, ""]
        Store the inner potential.
    mean_free_path_ang : Float64[Array, ""]
        Store the mean free path.
    domain_logits : Float64[Array, " n_domain"]
        Store domain logits.
    domain_frame_ids : Tuple[str, ...]
        Store domain frame identities.

    See Also
    --------
    make_sample_state
        Construct a validated sample state.
    """

    temperature_k: Float64[Array, ""]
    work_function_ev: Float64[Array, ""]
    inner_potential_ev: Float64[Array, ""]
    mean_free_path_ang: Float64[Array, ""]
    domain_logits: Float64[Array, " n_domain"]
    domain_frame_ids: Tuple[str, ...] = eqx.field(static=True)


class SamplePose(eqx.Module):
    """Define the ``SamplePose`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestSamplepose`

    Attributes
    ----------
    sample_azimuth_rad : Float64[Array, ""]
        Store the sample azimuth.
    domain_euler_angles_rad : Float64[Array, "n_domain 3"]
        Store domain Euler angles.

    See Also
    --------
    make_sample_pose
        Construct a validated sample pose.
    """

    sample_azimuth_rad: Float64[Array, ""]
    domain_euler_angles_rad: Float64[Array, "n_domain 3"]


class Acquisition(eqx.Module):
    """Define the ``Acquisition`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestAcquisition`

    Attributes
    ----------
    exposure : Float64[Array, ""]
        Store the exposure.
    statistics_mode : str
        Store the statistics mode.
    gaussian_sigma_counts : Optional[Float64[Array, "..."]]
        Store Gaussian count noise.
    fixed_total_count : Optional[int]
        Store the fixed event total.
    scan_order : str
        Store the scan order.
    acquisition_ref : str
        Store the acquisition identity.

    See Also
    --------
    make_acquisition
        Construct a validated acquisition.
    """

    exposure: Float64[Array, ""]
    statistics_mode: str = eqx.field(static=True)
    gaussian_sigma_counts: Optional[Float64[Array, "..."]]
    fixed_total_count: Optional[int] = eqx.field(static=True)
    scan_order: str = eqx.field(static=True)
    acquisition_ref: str = eqx.field(static=True)


class Experiment(eqx.Module):
    """Define the ``Experiment`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestExperiment`

    Attributes
    ----------
    photon : PhotonBeam
        Store the photon beam.
    sample : SampleState
        Store the sample state.
    pose : SamplePose
        Store the sample pose.
    acquisition : Acquisition
        Store the acquisition state.

    See Also
    --------
    make_experiment
        Construct a validated experiment.
    """

    photon: PhotonBeam
    sample: SampleState
    pose: SamplePose
    acquisition: Acquisition


@jaxtyped(typechecker=beartype)
def make_photon_beam(
    photon_energy_ev: Float64[Array, ""],
    polarization_lab: Complex128[Array, " 3"],
    incidence_theta_rad: Float64[Array, ""],
    incidence_phi_rad: Float64[Array, ""],
) -> PhotonBeam:
    """Compute the ``make_photon_beam`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestMakePhotonBeam`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    photon_energy_ev : Float64[Array, '']
        Input value for this operation.
    polarization_lab : Complex128[Array, ' 3']
        Input value for this operation.
    incidence_theta_rad : Float64[Array, '']
        Input value for this operation.
    incidence_phi_rad : Float64[Array, '']
        Input value for this operation.

    Returns
    -------
    result : PhotonBeam
        Validated operation result.
    """
    energy: Float64[Array, ""] = jnp.asarray(
        photon_energy_ev, dtype=jnp.float64
    )
    polarization: Complex128[Array, " 3"] = jnp.asarray(
        polarization_lab, dtype=jnp.complex128
    )
    energy = eqx.error_if(
        energy,
        ~jnp.isfinite(energy) | (energy <= 0.0),
        "photon energy must be positive",
    )
    norm: Float64[Array, ""] = jnp.linalg.norm(polarization)
    polarization = (
        eqx.error_if(
            polarization,
            ~jnp.isfinite(norm) | (norm <= 0.0),
            "polarization must be finite and nonzero",
        )
        / norm
    )
    theta: Float64[Array, ""] = jnp.asarray(
        incidence_theta_rad, dtype=jnp.float64
    )
    phi: Float64[Array, ""] = jnp.asarray(incidence_phi_rad, dtype=jnp.float64)
    propagation: Float64[Array, " 3"] = jnp.stack(
        (
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        )
    )
    polarization = eqx.error_if(
        polarization,
        ~jnp.isfinite(theta)
        | ~jnp.isfinite(phi)
        | (
            jnp.abs(jnp.vdot(propagation, polarization))
            > HERMITICITY_RELATIVE_TOLERANCE
        ),
        "beam polarization must be transverse to finite incidence",
    )
    result: PhotonBeam = PhotonBeam(
        energy,
        polarization,
        theta,
        phi,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_sample_state(
    temperature_k: Float64[Array, ""],
    work_function_ev: Float64[Array, ""],
    inner_potential_ev: Float64[Array, ""],
    mean_free_path_ang: Float64[Array, ""],
    domain_logits: Float64[Array, " n_domain"],
    *,
    domain_frame_ids: Tuple[str, ...],
) -> SampleState:
    """Compute the ``make_sample_state`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestMakeSampleState`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    temperature_k : Float64[Array, '']
        Input value for this operation.
    work_function_ev : Float64[Array, '']
        Input value for this operation.
    inner_potential_ev : Float64[Array, '']
        Input value for this operation.
    mean_free_path_ang : Float64[Array, '']
        Input value for this operation.
    domain_logits : Float64[Array, ' n_domain']
        Input value for this operation.
    domain_frame_ids : Tuple[str, ...]
        Input value for this operation.

    Returns
    -------
    result : SampleState
        Validated operation result.

    Raises
    ------
    ValueError
        If the domain-logit and domain-frame axes disagree.
    """
    logits: Float64[Array, " n_domain"] = jnp.asarray(
        domain_logits, dtype=jnp.float64
    )
    if logits.shape != (len(domain_frame_ids),) or not domain_frame_ids:
        raise ValueError("sample domain logits and frames must agree")
    temperature: Float64[Array, ""] = jnp.asarray(
        temperature_k, dtype=jnp.float64
    )
    work_function: Float64[Array, ""] = jnp.asarray(
        work_function_ev, dtype=jnp.float64
    )
    inner_potential: Float64[Array, ""] = jnp.asarray(
        inner_potential_ev, dtype=jnp.float64
    )
    mean_free_path: Float64[Array, ""] = jnp.asarray(
        mean_free_path_ang, dtype=jnp.float64
    )
    temperature = eqx.error_if(
        temperature,
        ~jnp.isfinite(temperature) | (temperature < 0.0),
        "sample temperature must be finite and nonnegative",
    )
    work_function = eqx.error_if(
        work_function,
        ~jnp.isfinite(work_function) | (work_function <= 0.0),
        "sample work function must be finite and positive",
    )
    inner_potential = eqx.error_if(
        inner_potential,
        ~jnp.isfinite(inner_potential),
        "sample inner potential must be finite",
    )
    mean_free_path = eqx.error_if(
        mean_free_path,
        ~jnp.isfinite(mean_free_path) | (mean_free_path <= 0.0),
        "sample mean free path must be finite and positive",
    )
    logits = eqx.error_if(
        logits,
        ~jnp.all(jnp.isfinite(logits)),
        "sample domain logits must be finite",
    )
    result: SampleState = SampleState(
        temperature,
        work_function,
        inner_potential,
        mean_free_path,
        logits,
        domain_frame_ids,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_sample_pose(
    sample_azimuth_rad: Float64[Array, ""],
    domain_euler_angles_rad: Float64[Array, "n_domain 3"],
) -> SamplePose:
    """Compute the ``make_sample_pose`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestMakeSamplePose`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    sample_azimuth_rad : Float64[Array, '']
        Input value for this operation.
    domain_euler_angles_rad : Float64[Array, 'n_domain 3']
        Input value for this operation.

    Returns
    -------
    result : SamplePose
        Validated operation result.

    Raises
    ------
    ValueError
        If domain rotations are not stored as Euler triples.
    """
    azimuth: Float64[Array, ""] = jnp.asarray(
        sample_azimuth_rad, dtype=jnp.float64
    )
    angles: Float64[Array, "n_domain 3"] = jnp.asarray(
        domain_euler_angles_rad, dtype=jnp.float64
    )
    if (
        angles.ndim != ARRAY_MATRIX_NDIM
        or angles.shape[1] != CARTESIAN_COMPONENTS
    ):
        raise ValueError(
            "sample pose must contain one Euler triple per domain"
        )
    azimuth = eqx.error_if(
        azimuth,
        ~jnp.isfinite(azimuth),
        "sample azimuth must be finite",
    )
    angles = eqx.error_if(
        angles,
        ~jnp.all(jnp.isfinite(angles)),
        "sample Euler angles must be finite",
    )
    result: SamplePose = SamplePose(azimuth, angles)
    return result


@jaxtyped(typechecker=beartype)
def make_acquisition(
    exposure: Float64[Array, ""],
    *,
    statistics_mode: str = "expected",
    gaussian_sigma_counts: Optional[Float64[Array, "..."]] = None,
    fixed_total_count: Optional[int] = None,
    scan_order: str = "simultaneous",
    acquisition_ref: str = "org.diffpes.acquisition.counting@0.1.0",
) -> Acquisition:
    """Compute the ``make_acquisition`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestMakeAcquisition`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    exposure : Float64[Array, '']
        Input value for this operation.
    statistics_mode : str
        Input value for this operation.
    gaussian_sigma_counts : Optional[Float64[Array, '...']]
        Input value for this operation.
    fixed_total_count : Optional[int]
        Input value for this operation.
    scan_order : str
        Input value for this operation.
    acquisition_ref : str
        Input value for this operation.

    Returns
    -------
    result : Acquisition
        Validated operation result.

    Raises
    ------
    ValueError
        If the statistics mode, its optional parameter, or an identity is
        inconsistent.
    """
    if statistics_mode not in (
        "expected",
        "gaussian",
        "poisson",
        "fixed_total",
    ):
        raise ValueError("unknown acquisition statistics mode")
    if (statistics_mode == "gaussian") != (gaussian_sigma_counts is not None):
        raise ValueError(
            "Gaussian acquisition requires and only permits sigma"
        )
    if (statistics_mode == "fixed_total") != (
        fixed_total_count is not None and fixed_total_count > 0
    ):
        raise ValueError("fixed-total acquisition requires a positive total")
    if not scan_order or not acquisition_ref:
        raise ValueError("acquisition references must be nonempty")
    exposure_value: Float64[Array, ""] = jnp.asarray(
        exposure, dtype=jnp.float64
    )
    exposure_value = eqx.error_if(
        exposure_value,
        ~jnp.isfinite(exposure_value) | (exposure_value <= 0.0),
        "acquisition exposure must be finite and positive",
    )
    sigma: Optional[Float64[Array, "..."]] = (
        None
        if gaussian_sigma_counts is None
        else jnp.asarray(gaussian_sigma_counts, dtype=jnp.float64)
    )
    if sigma is not None:
        sigma = eqx.error_if(
            sigma,
            ~jnp.all(jnp.isfinite(sigma)) | jnp.any(sigma <= 0.0),
            "Gaussian acquisition sigma must be finite and positive",
        )
    result: Acquisition = Acquisition(
        exposure_value,
        statistics_mode,
        sigma,
        fixed_total_count,
        scan_order,
        acquisition_ref,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_experiment(
    photon: PhotonBeam,
    sample: SampleState,
    pose: SamplePose,
    acquisition: Acquisition,
) -> Experiment:
    """Compute the ``make_experiment`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_experiment_state.TestMakeExperiment`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    photon : PhotonBeam
        Input value for this operation.
    sample : SampleState
        Input value for this operation.
    pose : SamplePose
        Input value for this operation.
    acquisition : Acquisition
        Input value for this operation.

    Returns
    -------
    result : Experiment
        Validated operation result.

    Raises
    ------
    ValueError
        If the sample and pose domain counts disagree.
    """
    if pose.domain_euler_angles_rad.shape[0] != sample.domain_logits.shape[0]:
        raise ValueError("experiment pose and sample domain counts must agree")
    checked_work_function: Float64[Array, ""] = eqx.error_if(
        sample.work_function_ev,
        sample.work_function_ev >= photon.photon_energy_ev,
        "sample work function must be below photon energy",
    )
    checked_sample: SampleState = eqx.tree_at(
        lambda state: state.work_function_ev,
        sample,
        checked_work_function,
    )
    result: Experiment = Experiment(photon, checked_sample, pose, acquisition)
    return result


__all__: list[str] = [
    "Acquisition",
    "Experiment",
    "PhotonBeam",
    "SamplePose",
    "SampleState",
    "make_acquisition",
    "make_experiment",
    "make_photon_beam",
    "make_sample_pose",
    "make_sample_state",
]
