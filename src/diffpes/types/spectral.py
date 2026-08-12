"""Define spectral-tail and streamed-source data structures.

Extended Summary
----------------
This module owns the immutable PyTree carriers used by causal tail
integration and block-local transition-source assembly. Their factories
preserve float64 and complex128 leaves while validating every static axis.

Routine Listings
----------------
:class:`Power2TailSpec`
    Store the six derived coefficients for two causal power-law tails.
:class:`TransitionSourceSchedule`
    Store compact inputs for block-local transition-source assembly.
:func:`make_power2_tail_spec`
    Create a scalar-valued causal-tail carrier.
:func:`make_transition_source_schedule`
    Create a shape-consistent streamed transition-source schedule.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    ENERGY_AXIS_NDIM,
    MATRIX_NDIM,
)

from .experiment import ExperimentGeometry
from .radial_params import MatrixElementParams, RadialSpec
from .radial_profiles import FinalStateSpec, RadialQuadratureSpec


class Power2TailSpec(eqx.Module):
    """Store the six derived coefficients for two causal power-law tails.

    The carrier preserves the left-then-right ordering used by the causal
    self-energy quadrature. Every numerical field is a scalar PyTree leaf.

    :see: :class:`~.test_spectral.TestPower2TailSpec`

    Attributes
    ----------
    amplitude_left : Float64[Array, ""]
        Positive left-edge amplitude in eV.
    alpha_left : Float64[Array, ""]
        Left linear denominator coefficient.
    beta_left : Float64[Array, ""]
        Left quadratic denominator coefficient.
    amplitude_right : Float64[Array, ""]
        Positive right-edge amplitude in eV.
    alpha_right : Float64[Array, ""]
        Right linear denominator coefficient.
    beta_right : Float64[Array, ""]
        Right quadratic denominator coefficient.

    Notes
    -----
    The producing spectral routine derives the coefficients from edge values
    and slopes. Runtime physics validation remains attached to that routine.

    See Also
    --------
    make_power2_tail_spec : Validated factory for this type.
    """

    amplitude_left: Float64[Array, ""]
    alpha_left: Float64[Array, ""]
    beta_left: Float64[Array, ""]
    amplitude_right: Float64[Array, ""]
    alpha_right: Float64[Array, ""]
    beta_right: Float64[Array, ""]

    def __check_init__(self) -> None:
        """Require every tail coefficient to remain scalar-valued.

        Raises
        ------
        ValueError
            If any coefficient has a non-scalar shape.
        """
        coefficient: Float64[Array, ""]
        for coefficient in (
            self.amplitude_left,
            self.alpha_left,
            self.beta_left,
            self.amplitude_right,
            self.alpha_right,
            self.beta_right,
        ):
            if coefficient.ndim != 0:
                message: str = "power2 tail coefficients must be scalars"
                raise ValueError(message)


class TransitionSourceSchedule(eqx.Module):
    """Store compact inputs for block-local transition-source assembly.

    The carrier excludes precomputed energy-by-momentum final states and
    transition tensors. Consumers reconstruct those values only for each
    live momentum and energy block.

    :see: :class:`~.test_spectral.TestTransitionSourceSchedule`

    Attributes
    ----------
    k_i_cart : Float64[Array, "n_k_max 3"]
        Initial sample-frame crystal momenta in inverse Angstrom.
    final_norm : Float64[Array, " n_omega_max"]
        Vacuum final-momentum magnitude for each sampled energy.
    emission_energy_valid : Bool[Array, " n_omega_max"]
        Positive kinetic-energy and final-state-momentum mask.
    positions_cart : Float64[Array, "n_orb 3"]
        Orbital or Wannier centres in Cartesian Angstrom.
    depths : Float64[Array, " n_orb"]
        Orbital depths below the surface in Angstrom.
    polarization_sample_cart : Complex128[Array, " 3"]
        Sample-frame Cartesian polarization.
    mean_free_path_ang : Float64[Array, ""]
        Photoelectron intensity mean free path in Angstrom.
    radial : RadialSpec
        Shell-shared initial-state radial carrier.
    matrix_element : MatrixElementParams
        Shell scales and phase coordinates.
    quadrature : RadialQuadratureSpec
        Fixed radial quadrature.
    final_state : FinalStateSpec
        Plane-wave or Coulomb final-state selection.
    inner_potential_geometry : Optional[ExperimentGeometry]
        Geometry for exact finite-energy internal final momentum.

    Notes
    -----
    Numerical fields remain differentiable leaves. The nested carriers retain
    their own static selectors and validation contracts.

    See Also
    --------
    make_transition_source_schedule : Validated factory for this type.
    """

    k_i_cart: Float64[Array, "n_k_max 3"]
    final_norm: Float64[Array, " n_omega_max"]
    emission_energy_valid: Bool[Array, " n_omega_max"]
    positions_cart: Float64[Array, "n_orb 3"]
    depths: Float64[Array, " n_orb"]
    polarization_sample_cart: Complex128[Array, " 3"]
    mean_free_path_ang: Float64[Array, ""]
    radial: RadialSpec
    matrix_element: MatrixElementParams
    quadrature: RadialQuadratureSpec
    final_state: FinalStateSpec
    inner_potential_geometry: Optional[ExperimentGeometry] = None

    def __check_init__(self) -> None:
        """Require consistent schedule and orbital axes.

        Raises
        ------
        ValueError
            If one schedule field has an incompatible static shape.
        """
        n_orb: int = self.positions_cart.shape[0]
        if (
            self.k_i_cart.ndim != MATRIX_NDIM
            or self.k_i_cart.shape[1] != CARTESIAN_COMPONENTS
            or self.final_norm.ndim != ENERGY_AXIS_NDIM
            or self.emission_energy_valid.shape != self.final_norm.shape
            or self.positions_cart.ndim != MATRIX_NDIM
            or self.positions_cart.shape[1] != CARTESIAN_COMPONENTS
            or self.depths.shape != (n_orb,)
            or self.polarization_sample_cart.shape != (CARTESIAN_COMPONENTS,)
            or self.mean_free_path_ang.ndim != 0
            or len(self.radial.basis.n) != n_orb
        ):
            message: str = (
                "transition source schedule fields have incompatible axes"
            )
            raise ValueError(message)


@jaxtyped(typechecker=beartype)
def make_power2_tail_spec(  # noqa: DOC502
    amplitude_left: Float64[Array, ""],
    alpha_left: Float64[Array, ""],
    beta_left: Float64[Array, ""],
    amplitude_right: Float64[Array, ""],
    alpha_right: Float64[Array, ""],
    beta_right: Float64[Array, ""],
) -> Power2TailSpec:
    """Create a scalar-valued causal-tail carrier.

    Preserve the derived coefficients without changing their numerical values.

    :see: :class:`~.test_spectral.TestMakePower2TailSpec`

    Parameters
    ----------
    amplitude_left : Float64[Array, ""]
        Positive left-edge amplitude in eV.
    alpha_left : Float64[Array, ""]
        Left linear denominator coefficient.
    beta_left : Float64[Array, ""]
        Left quadratic denominator coefficient.
    amplitude_right : Float64[Array, ""]
        Positive right-edge amplitude in eV.
    alpha_right : Float64[Array, ""]
        Right linear denominator coefficient.
    beta_right : Float64[Array, ""]
        Right quadratic denominator coefficient.

    Returns
    -------
    spec : Power2TailSpec
        Immutable six-coefficient tail carrier.

    Raises
    ------
    ValueError
        If any coefficient is not scalar-valued.

    Notes
    -----
    Physical positivity and seam checks depend on the sampled self-energy.
    The consuming spectral routine binds those dynamic checks to its result.
    """
    spec: Power2TailSpec = Power2TailSpec(
        amplitude_left=amplitude_left,
        alpha_left=alpha_left,
        beta_left=beta_left,
        amplitude_right=amplitude_right,
        alpha_right=alpha_right,
        beta_right=beta_right,
    )
    return spec


@jaxtyped(typechecker=beartype)
def make_transition_source_schedule(  # noqa: DOC502, PLR0913, PLR0917
    k_i_cart: Float64[Array, "n_k_max 3"],
    final_norm: Float64[Array, " n_omega_max"],
    emission_energy_valid: Bool[Array, " n_omega_max"],
    positions_cart: Float64[Array, "n_orb 3"],
    depths: Float64[Array, " n_orb"],
    polarization_sample_cart: Complex128[Array, " 3"],
    mean_free_path_ang: Float64[Array, ""],
    radial: RadialSpec,
    matrix_element: MatrixElementParams,
    quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    inner_potential_geometry: Optional[ExperimentGeometry] = None,
) -> TransitionSourceSchedule:
    """Create a shape-consistent streamed transition-source schedule.

    Preserve compact kinematic and matrix-element inputs for block-local use.

    :see: :class:`~.test_spectral.TestMakeTransitionSourceSchedule`

    Parameters
    ----------
    k_i_cart : Float64[Array, "n_k_max 3"]
        Initial sample-frame crystal momenta in inverse Angstrom.
    final_norm : Float64[Array, " n_omega_max"]
        Vacuum final-momentum magnitude for each sampled energy.
    emission_energy_valid : Bool[Array, " n_omega_max"]
        Positive kinetic-energy and final-state-momentum mask.
    positions_cart : Float64[Array, "n_orb 3"]
        Orbital or Wannier centres in Cartesian Angstrom.
    depths : Float64[Array, " n_orb"]
        Orbital depths below the surface in Angstrom.
    polarization_sample_cart : Complex128[Array, " 3"]
        Sample-frame Cartesian polarization.
    mean_free_path_ang : Float64[Array, ""]
        Photoelectron intensity mean free path in Angstrom.
    radial : RadialSpec
        Shell-shared initial-state radial carrier.
    matrix_element : MatrixElementParams
        Shell scales and phase coordinates.
    quadrature : RadialQuadratureSpec
        Fixed radial quadrature.
    final_state : FinalStateSpec
        Plane-wave or Coulomb final-state selection.
    inner_potential_geometry : Optional[ExperimentGeometry], optional
        Geometry for exact finite-energy internal final momentum.

    Returns
    -------
    schedule : TransitionSourceSchedule
        Validated compact transition-source schedule.

    Raises
    ------
    TypeCheckError
        If a jaxtyping axis relation or canonical dtype is invalid.
    ValueError
        If one schedule field has an incompatible static shape.

    Notes
    -----
    The factory stores every input without numerical conversion. Callers
    establish the canonical widths before they build this carrier.
    """
    schedule: TransitionSourceSchedule = TransitionSourceSchedule(
        k_i_cart=k_i_cart,
        final_norm=final_norm,
        emission_energy_valid=emission_energy_valid,
        positions_cart=positions_cart,
        depths=depths,
        polarization_sample_cart=polarization_sample_cart,
        mean_free_path_ang=mean_free_path_ang,
        radial=radial,
        matrix_element=matrix_element,
        quadrature=quadrature,
        final_state=final_state,
        inner_potential_geometry=inner_potential_geometry,
    )
    return schedule


__all__: list[str] = [
    "Power2TailSpec",
    "TransitionSourceSchedule",
    "make_power2_tail_spec",
    "make_transition_source_schedule",
]
