"""Define native finite-slab Kohn--Sham scattering contracts."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex128, Float64, Int32, jaxtyped

from diffpes.constants import CARTESIAN_COMPONENTS

_NormalStencilValues = Complex128[Array, "n_normal_stencil n_slice n_chan"]


class KSScatteringRequest(eqx.Module):
    """Describe outgoing parallel momentum, energy, and incident channels.

    Attributes
    ----------
    k_parallel_cart_inv_ang : Float64[Array, "n_state 2"]
        Parallel momenta.
    kinetic_energy_ev : Float64[Array, " n_state"]
        Kinetic energies.
    outgoing_channel_index : Int32[Array, " n_state"]
        Selected channel indices.
    surface_normal_cart : Float64[Array, " 3"]
        Surface normal.
    energy_block_size : int
        Static energy block size.
    validity_profile_ref : str
        Static validity profile.

    See Also
    --------
    make_ks_scattering_request : Create this request.
    """

    k_parallel_cart_inv_ang: Float64[Array, "n_state 2"]
    kinetic_energy_ev: Float64[Array, " n_state"]
    outgoing_channel_index: Int32[Array, " n_state"]
    surface_normal_cart: Float64[Array, " 3"]
    energy_block_size: int = eqx.field(static=True)
    validity_profile_ref: str = eqx.field(static=True)


class DenseSliceOperator(eqx.Module):
    """Store dense Laue coupled-channel blocks by normal slice.

    Attributes
    ----------
    blocks_ev : Complex128[Array, "n_slice n_chan n_chan"]
        Dense slice blocks.

    See Also
    --------
    make_dense_slice_operator : Create this operator.
    """

    blocks_ev: Complex128[Array, "n_slice n_chan n_chan"]


class SparseSliceOperator(eqx.Module):
    """Store sorted sparse real-space lateral operator values.

    Attributes
    ----------
    values_ev : Complex128[Array, " n_nonzero"]
        Sparse values.
    indices : Int32[Array, "n_nonzero 3"]
        Sparse integer indices.
    shape : Tuple[int, int, int]
        Static sparse-grid shape.

    See Also
    --------
    make_sparse_slice_operator : Create this operator.
    """

    values_ev: Complex128[Array, " n_nonzero"]
    indices: Int32[Array, "n_nonzero 3"]
    shape: Tuple[int, int, int] = eqx.field(static=True)


SliceOperator = Union[DenseSliceOperator, SparseSliceOperator]


class KSScatteringProblem(eqx.Module):
    """Store one lowered finite-slab scattering problem without a global
    matrix.

    Attributes
    ----------
    slice_operator : SliceOperator
        Lowered slice operator.
    normal_stencil_offsets : Int32[Array, " n_normal_stencil"]
        Normal-direction stencil offsets.
    normal_stencil_values_ev : _NormalStencilValues
        Normal-direction stencil values.
    nonlocal_projectors : Complex128[Array, "n_projector n_slice n_chan"]
        Nonlocal projector values.
    nonlocal_couplings_ev : Complex128[Array, "n_projector n_projector"]
        Nonlocal coupling matrix.
    slice_coordinates_ang : Float64[Array, " n_slice"]
        Slice coordinates.
    channel_coordinates : Float64[Array, "n_chan 2"]
        Channel coordinates.
    hamiltonian_ref : str
        Hamiltonian identity.
    basis_kind : str
        Basis declaration.
    channel_coordinate_kind : str
        Channel-coordinate declaration.
    operator_storage_ref : str
        Storage declaration.
    discretization_ref : str
        Discretization identity.

    See Also
    --------
    make_ks_scattering_problem : Create this problem.
    """

    slice_operator: SliceOperator
    normal_stencil_offsets: Int32[Array, " n_normal_stencil"]
    normal_stencil_values_ev: _NormalStencilValues
    nonlocal_projectors: Complex128[Array, "n_projector n_slice n_chan"]
    nonlocal_couplings_ev: Complex128[Array, "n_projector n_projector"]
    slice_coordinates_ang: Float64[Array, " n_slice"]
    channel_coordinates: Float64[Array, "n_chan 2"]
    hamiltonian_ref: str = eqx.field(static=True)
    basis_kind: str = eqx.field(static=True)
    channel_coordinate_kind: str = eqx.field(static=True)
    operator_storage_ref: str = eqx.field(static=True)
    discretization_ref: str = eqx.field(static=True)


class VacuumBoundarySpec(eqx.Module):
    """Specify one unit-normal-flux vacuum lead.

    Attributes
    ----------
    reference_potential_ev : Float64[Array, ""]
        Reference potential.
    direction : str
        Lead direction.
    normalization : str
        Flux normalization.

    See Also
    --------
    make_vacuum_boundary_spec : Create this boundary.
    """

    reference_potential_ev: Float64[Array, ""]
    direction: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)


class BackingAbsorberSpec(eqx.Module):
    """Specify an optional convergence-tested finite-domain absorber.

    Attributes
    ----------
    absorber_strength_ev : Float64[Array, ""]
        Absorber strength.
    absorber_start_ang : Float64[Array, ""]
        Absorber start position.
    absorber_width_ang : Float64[Array, ""]
        Absorber width.
    side : str
        Absorbing boundary side.
    shape : str
        Absorber profile shape.
    profile_ref : str
        Static profile reference.

    See Also
    --------
    make_backing_absorber_spec : Create this absorber.
    """

    absorber_strength_ev: Float64[Array, ""]
    absorber_start_ang: Float64[Array, ""]
    absorber_width_ang: Float64[Array, ""]
    side: str = eqx.field(static=True)
    shape: str = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


class KSScatteringBoundaryProfile(eqx.Module):
    """Bind both vacuum leads and optional backing absorption.

    Attributes
    ----------
    left : VacuumBoundarySpec
        Left vacuum lead.
    right : VacuumBoundarySpec
        Right vacuum lead.
    backing_absorber : Optional[BackingAbsorberSpec]
        Optional backing absorber.
    vacuum_convergence_ref : str
        Vacuum convergence evidence.
    slab_convergence_ref : str
        Slab convergence evidence.
    absorber_convergence_ref : Optional[str]
        Absorber convergence evidence.
    profile_ref : str
        Boundary profile identity.

    See Also
    --------
    make_ks_scattering_boundary_profile : Create this profile.
    """

    left: VacuumBoundarySpec
    right: VacuumBoundarySpec
    backing_absorber: Optional[BackingAbsorberSpec]
    vacuum_convergence_ref: str = eqx.field(static=True)
    slab_convergence_ref: str = eqx.field(static=True)
    absorber_convergence_ref: Optional[str] = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


class LightMatterCouplingSpec(eqx.Module):
    """Declare the nonlocal velocity and spin-contraction convention.

    Attributes
    ----------
    representation : str
        Coupling representation.
    photon_momentum : str
        Photon-momentum convention.
    final_spin_mode : str
        Final-state spin convention.
    profile_ref : str
        Static coupling profile.

    See Also
    --------
    make_light_matter_coupling_spec : Create this convention.
    """

    representation: str = eqx.field(static=True)
    photon_momentum: str = eqx.field(static=True)
    final_spin_mode: str = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


class KSScatteringSolverSpec(eqx.Module):
    """Pin one residual-controlled matrix-free solver execution profile.

    Attributes
    ----------
    relative_residual : float
        Relative residual limit.
    absolute_residual : float
        Absolute residual limit.
    max_iterations : int
        Static iteration limit.
    krylov_dimension : int
        Static Krylov dimension.
    preconditioner_ref : str
        Static preconditioner identity.
    threshold_guard_ev : float
        Threshold guard energy.

    See Also
    --------
    make_ks_scattering_solver_spec : Create this solver profile.
    """

    relative_residual: float = eqx.field(static=True)
    absolute_residual: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    krylov_dimension: int = eqx.field(static=True)
    preconditioner_ref: str = eqx.field(static=True)
    threshold_guard_ev: float = eqx.field(static=True)


class KSScatteringBatch(eqx.Module):
    """Store bounded scattering states and physical flux diagnostics.

    Attributes
    ----------
    states : Complex128[Array, "n_state n_slice n_chan n_out_spin"]
        Computed scattering states.
    reflection_amplitudes : Complex128[Array, "n_state n_open n_out_spin"]
        Reflection amplitudes.
    transmission_amplitudes : Complex128[Array, "n_state n_open n_out_spin"]
        Transmission amplitudes.
    open_channel_mask : Bool[Array, "n_state n_chan"]
        Open-channel selector.
    residual_norm : Float64[Array, " n_state"]
        Residual diagnostics.
    incident_flux : Float64[Array, " n_state"]
        Incident flux.
    reflected_flux : Float64[Array, " n_state"]
        Reflected flux.
    transmitted_flux : Float64[Array, " n_state"]
        Transmitted flux.
    absorbed_flux : Float64[Array, " n_state"]
        Absorbed flux.
    state_ref : str
        State identity.

    See Also
    --------
    make_ks_scattering_batch : Create this result batch.
    """

    states: Complex128[Array, "n_state n_slice n_chan n_out_spin"]
    reflection_amplitudes: Complex128[Array, "n_state n_open n_out_spin"]
    transmission_amplitudes: Complex128[Array, "n_state n_open n_out_spin"]
    open_channel_mask: Bool[Array, "n_state n_chan"]
    residual_norm: Float64[Array, " n_state"]
    incident_flux: Float64[Array, " n_state"]
    reflected_flux: Float64[Array, " n_state"]
    transmitted_flux: Float64[Array, " n_state"]
    absorbed_flux: Float64[Array, " n_state"]
    state_ref: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_ks_scattering_request(
    k_parallel_cart_inv_ang: Float64[Array, "n_state 2"],
    kinetic_energy_ev: Float64[Array, " n_state"],
    outgoing_channel_index: Int32[Array, " n_state"],
    surface_normal_cart: Float64[Array, " 3"],
    *,
    energy_block_size: int,
    validity_profile_ref: str,
) -> KSScatteringRequest:
    """Create a bounded Kohn--Sham scattering request."""
    if energy_block_size <= 0 or not validity_profile_ref:
        raise ValueError("scattering request metadata is invalid")
    return KSScatteringRequest(
        jnp.asarray(k_parallel_cart_inv_ang, dtype=jnp.float64),
        jnp.asarray(kinetic_energy_ev, dtype=jnp.float64),
        jnp.asarray(outgoing_channel_index, dtype=jnp.int32),
        jnp.asarray(surface_normal_cart, dtype=jnp.float64),
        energy_block_size,
        validity_profile_ref,
    )


@jaxtyped(typechecker=beartype)
def make_dense_slice_operator(
    blocks_ev: Complex128[Array, "n_slice n_chan n_chan"],
) -> DenseSliceOperator:
    """Create a dense finite-slab slice operator."""
    blocks = jnp.asarray(blocks_ev, dtype=jnp.complex128)
    if (
        blocks.ndim != CARTESIAN_COMPONENTS
        or blocks.shape[-1] != blocks.shape[-2]
    ):
        raise ValueError("dense slice blocks must be square matrices")
    return DenseSliceOperator(blocks)


@jaxtyped(typechecker=beartype)
def make_sparse_slice_operator(
    values_ev: Complex128[Array, " n_nonzero"],
    indices: Int32[Array, "n_nonzero 3"],
    *,
    shape: Tuple[int, int, int],
) -> SparseSliceOperator:
    """Create a sparse finite-slab slice operator."""
    values = jnp.asarray(values_ev, dtype=jnp.complex128)
    sparse_indices = jnp.asarray(indices, dtype=jnp.int32)
    if len(shape) != CARTESIAN_COMPONENTS or any(size <= 0 for size in shape):
        raise ValueError("sparse slice shape must have three positive entries")
    if sparse_indices.shape != (values.shape[0], CARTESIAN_COMPONENTS):
        raise ValueError("sparse slice indices must align with values")
    return SparseSliceOperator(values, sparse_indices, shape)


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_ks_scattering_problem(  # noqa: PLR0913
    slice_operator: SliceOperator,
    normal_stencil_offsets: Int32[Array, " n_normal_stencil"],
    normal_stencil_values_ev: _NormalStencilValues,
    nonlocal_projectors: Complex128[Array, "n_projector n_slice n_chan"],
    nonlocal_couplings_ev: Complex128[Array, "n_projector n_projector"],
    slice_coordinates_ang: Float64[Array, " n_slice"],
    channel_coordinates: Float64[Array, "n_chan 2"],
    *,
    hamiltonian_ref: str,
    basis_kind: str,
    channel_coordinate_kind: str,
    operator_storage_ref: str,
    discretization_ref: str,
) -> KSScatteringProblem:
    """Create a lowered finite-slab scattering problem."""
    if not all(
        (
            hamiltonian_ref,
            basis_kind,
            channel_coordinate_kind,
            operator_storage_ref,
            discretization_ref,
        )
    ):
        raise ValueError("scattering problem references must be nonempty")
    return KSScatteringProblem(
        slice_operator,
        jnp.asarray(normal_stencil_offsets, dtype=jnp.int32),
        jnp.asarray(normal_stencil_values_ev, dtype=jnp.complex128),
        jnp.asarray(nonlocal_projectors, dtype=jnp.complex128),
        jnp.asarray(nonlocal_couplings_ev, dtype=jnp.complex128),
        jnp.asarray(slice_coordinates_ang, dtype=jnp.float64),
        jnp.asarray(channel_coordinates, dtype=jnp.float64),
        hamiltonian_ref,
        basis_kind,
        channel_coordinate_kind,
        operator_storage_ref,
        discretization_ref,
    )


@jaxtyped(typechecker=beartype)
def make_vacuum_boundary_spec(
    reference_potential_ev: Float64[Array, ""], *, direction: str
) -> VacuumBoundarySpec:
    """Create a unit-normal-flux vacuum boundary declaration."""
    if direction not in ("left", "right"):
        raise ValueError("vacuum boundary direction must be left or right")
    return VacuumBoundarySpec(
        jnp.asarray(reference_potential_ev, dtype=jnp.float64),
        direction,
        "unit_normal_flux",
    )


@jaxtyped(typechecker=beartype)
def make_backing_absorber_spec(
    absorber_strength_ev: Float64[Array, ""],
    absorber_start_ang: Float64[Array, ""],
    absorber_width_ang: Float64[Array, ""],
    *,
    side: str,
    shape: str,
    profile_ref: str,
) -> BackingAbsorberSpec:
    """Create a finite-domain absorber declaration."""
    if not all((side, shape, profile_ref)):
        raise ValueError("absorber metadata must be nonempty")
    return BackingAbsorberSpec(
        jnp.asarray(absorber_strength_ev, dtype=jnp.float64),
        jnp.asarray(absorber_start_ang, dtype=jnp.float64),
        jnp.asarray(absorber_width_ang, dtype=jnp.float64),
        side,
        shape,
        profile_ref,
    )


@jaxtyped(typechecker=beartype)
def make_ks_scattering_boundary_profile(
    left: VacuumBoundarySpec,
    right: VacuumBoundarySpec,
    backing_absorber: Optional[BackingAbsorberSpec],
    *,
    vacuum_convergence_ref: str,
    slab_convergence_ref: str,
    absorber_convergence_ref: Optional[str],
    profile_ref: str,
) -> KSScatteringBoundaryProfile:
    """Create a complete finite-slab boundary profile."""
    if not all((vacuum_convergence_ref, slab_convergence_ref, profile_ref)):
        raise ValueError("boundary profile references must be nonempty")
    return KSScatteringBoundaryProfile(
        left,
        right,
        backing_absorber,
        vacuum_convergence_ref,
        slab_convergence_ref,
        absorber_convergence_ref,
        profile_ref,
    )


@jaxtyped(typechecker=beartype)
def make_light_matter_coupling_spec(
    *,
    representation: str,
    photon_momentum: str,
    final_spin_mode: str,
    profile_ref: str,
) -> LightMatterCouplingSpec:
    """Create a light-matter coupling convention declaration."""
    if not all(
        (representation, photon_momentum, final_spin_mode, profile_ref)
    ):
        raise ValueError("light-matter metadata must be nonempty")
    return LightMatterCouplingSpec(
        representation, photon_momentum, final_spin_mode, profile_ref
    )


@jaxtyped(typechecker=beartype)
def make_ks_scattering_solver_spec(
    *,
    relative_residual: float = 1.0e-10,
    absolute_residual: float = 1.0e-12,
    max_iterations: int = 500,
    krylov_dimension: int = 32,
    preconditioner_ref: str = "org.diffpes.preconditioner.kinetic@1.0.0",
    threshold_guard_ev: float = 1.0e-5,
) -> KSScatteringSolverSpec:
    """Create a residual-controlled native scattering solver profile."""
    if (
        min(relative_residual, absolute_residual, threshold_guard_ev) <= 0.0
        or max_iterations <= 0
        or krylov_dimension <= 0
    ):
        raise ValueError(
            "scattering solver tolerances and dimensions must be positive"
        )
    return KSScatteringSolverSpec(
        relative_residual,
        absolute_residual,
        max_iterations,
        krylov_dimension,
        preconditioner_ref,
        threshold_guard_ev,
    )


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_ks_scattering_batch(  # noqa: PLR0913
    states: Complex128[Array, "n_state n_slice n_chan n_out_spin"],
    reflection_amplitudes: Complex128[Array, "n_state n_open n_out_spin"],
    transmission_amplitudes: Complex128[Array, "n_state n_open n_out_spin"],
    open_channel_mask: Bool[Array, "n_state n_chan"],
    residual_norm: Float64[Array, " n_state"],
    incident_flux: Float64[Array, " n_state"],
    reflected_flux: Float64[Array, " n_state"],
    transmitted_flux: Float64[Array, " n_state"],
    absorbed_flux: Float64[Array, " n_state"],
    *,
    state_ref: str,
) -> KSScatteringBatch:
    """Create bounded scattering states with explicit flux diagnostics."""
    if not state_ref:
        raise ValueError("scattering batch state_ref must be nonempty")
    return KSScatteringBatch(
        jnp.asarray(states, dtype=jnp.complex128),
        jnp.asarray(reflection_amplitudes, dtype=jnp.complex128),
        jnp.asarray(transmission_amplitudes, dtype=jnp.complex128),
        jnp.asarray(open_channel_mask, dtype=jnp.bool),
        jnp.asarray(residual_norm, dtype=jnp.float64),
        jnp.asarray(incident_flux, dtype=jnp.float64),
        jnp.asarray(reflected_flux, dtype=jnp.float64),
        jnp.asarray(transmitted_flux, dtype=jnp.float64),
        jnp.asarray(absorbed_flux, dtype=jnp.float64),
        state_ref,
    )


__all__: list[str] = [
    "BackingAbsorberSpec",
    "DenseSliceOperator",
    "KSScatteringBatch",
    "KSScatteringBoundaryProfile",
    "KSScatteringProblem",
    "KSScatteringRequest",
    "KSScatteringSolverSpec",
    "LightMatterCouplingSpec",
    "SliceOperator",
    "SparseSliceOperator",
    "VacuumBoundarySpec",
    "make_backing_absorber_spec",
    "make_dense_slice_operator",
    "make_ks_scattering_batch",
    "make_ks_scattering_boundary_profile",
    "make_ks_scattering_problem",
    "make_ks_scattering_request",
    "make_ks_scattering_solver_spec",
    "make_light_matter_coupling_spec",
    "make_sparse_slice_operator",
    "make_vacuum_boundary_spec",
]
