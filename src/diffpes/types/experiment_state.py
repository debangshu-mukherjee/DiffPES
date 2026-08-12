"""Define split experiment carriers without hidden workflow state."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped


class PhotonBeam(eqx.Module):
    """Store photon energy, normalized laboratory polarization, and
    incidence.
    """

    photon_energy_ev: Float64[Array, ""]
    polarization_lab: Complex128[Array, " 3"]
    incidence_theta_rad: Float64[Array, ""]
    incidence_phi_rad: Float64[Array, ""]


class SampleState(eqx.Module):
    """Store thermal, vacuum, escape-depth, and domain state."""

    temperature_k: Float64[Array, ""]
    work_function_ev: Float64[Array, ""]
    inner_potential_ev: Float64[Array, ""]
    mean_free_path_ang: Float64[Array, ""]
    domain_logits: Float64[Array, " n_domain"]
    domain_frame_ids: Tuple[str, ...] = eqx.field(static=True)


class SamplePose(eqx.Module):
    """Store a sample azimuth and one Euler triple for each domain."""

    sample_azimuth_rad: Float64[Array, ""]
    domain_euler_angles_rad: Float64[Array, "n_domain 3"]


class Acquisition(eqx.Module):
    """Declare counting semantics and exposure independently of the model."""

    exposure: Float64[Array, ""]
    statistics_mode: str = eqx.field(static=True)
    gaussian_sigma_counts: Optional[Float64[Array, "..."]]
    fixed_total_count: Optional[int] = eqx.field(static=True)
    scan_order: str = eqx.field(static=True)
    acquisition_ref: str = eqx.field(static=True)


class Experiment(eqx.Module):
    """Aggregate split photon, sample, pose, and acquisition carriers."""

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
    """Create a positive-energy beam with nonzero normalized polarization."""
    energy = jnp.asarray(photon_energy_ev, dtype=jnp.float64)
    polarization = jnp.asarray(polarization_lab, dtype=jnp.complex128)
    energy = eqx.error_if(
        energy,
        ~jnp.isfinite(energy) | (energy <= 0.0),
        "photon energy must be positive",
    )
    norm = jnp.linalg.norm(polarization)
    polarization = (
        eqx.error_if(
            polarization,
            ~jnp.isfinite(norm) | (norm <= 0.0),
            "polarization must be finite and nonzero",
        )
        / norm
    )
    return PhotonBeam(
        energy,
        polarization,
        jnp.asarray(incidence_theta_rad, dtype=jnp.float64),
        jnp.asarray(incidence_phi_rad, dtype=jnp.float64),
    )


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
    """Create a split sample state with explicit domain frames."""
    logits = jnp.asarray(domain_logits, dtype=jnp.float64)
    if logits.shape != (len(domain_frame_ids),) or not domain_frame_ids:
        raise ValueError("sample domain logits and frames must agree")
    return SampleState(
        jnp.asarray(temperature_k, dtype=jnp.float64),
        jnp.asarray(work_function_ev, dtype=jnp.float64),
        jnp.asarray(inner_potential_ev, dtype=jnp.float64),
        jnp.asarray(mean_free_path_ang, dtype=jnp.float64),
        logits,
        domain_frame_ids,
    )


@jaxtyped(typechecker=beartype)
def make_sample_pose(
    sample_azimuth_rad: Float64[Array, ""],
    domain_euler_angles_rad: Float64[Array, "n_domain 3"],
) -> SamplePose:
    """Create one sample pose and one rotation per domain."""
    return SamplePose(
        jnp.asarray(sample_azimuth_rad, dtype=jnp.float64),
        jnp.asarray(domain_euler_angles_rad, dtype=jnp.float64),
    )


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
    """Create a mutually exclusive declared counting-acquisition contract."""
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
    return Acquisition(
        jnp.asarray(exposure, dtype=jnp.float64),
        statistics_mode,
        None
        if gaussian_sigma_counts is None
        else jnp.asarray(gaussian_sigma_counts, dtype=jnp.float64),
        fixed_total_count,
        scan_order,
        acquisition_ref,
    )


@jaxtyped(typechecker=beartype)
def make_experiment(
    photon: PhotonBeam,
    sample: SampleState,
    pose: SamplePose,
    acquisition: Acquisition,
) -> Experiment:
    """Create a cross-validated experiment aggregate."""
    if pose.domain_euler_angles_rad.shape[0] != sample.domain_logits.shape[0]:
        raise ValueError("experiment pose and sample domain counts must agree")
    return Experiment(photon, sample, pose, acquisition)


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
