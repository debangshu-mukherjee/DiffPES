"""Define native finite-slab Kohn--Sham scattering contracts.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`BackingAbsorberSpec`
    Define the ``BackingAbsorberSpec`` public contract.
:class:`DenseSliceOperator`
    Define the ``DenseSliceOperator`` public contract.
:class:`KSScatteringBoundaryProfile`
    Define the ``KSScatteringBoundaryProfile`` public contract.
:class:`KSScatteringProblem`
    Define the ``KSScatteringProblem`` public contract.
:class:`KSScatteringRequest`
    Define the ``KSScatteringRequest`` public contract.
:class:`LightMatterCouplingSpec`
    Define the ``LightMatterCouplingSpec`` public contract.
:class:`SparseSliceOperator`
    Define the ``SparseSliceOperator`` public contract.
:class:`VacuumBoundarySpec`
    Define the ``VacuumBoundarySpec`` public contract.
:func:`make_backing_absorber_spec`
    Compute the ``make_backing_absorber_spec`` public contract.
:func:`make_dense_slice_operator`
    Compute the ``make_dense_slice_operator`` public contract.
:func:`make_ks_scattering_boundary_profile`
    Compute the ``make_ks_scattering_boundary_profile`` public contract.
:func:`make_ks_scattering_problem`
    Compute the ``make_ks_scattering_problem`` public contract.
:func:`make_ks_scattering_request`
    Compute the ``make_ks_scattering_request`` public contract.
:func:`make_light_matter_coupling_spec`
    Compute the ``make_light_matter_coupling_spec`` public contract.
:func:`make_sparse_slice_operator`
    Compute the ``make_sparse_slice_operator`` public contract.
:func:`make_vacuum_boundary_spec`
    Compute the ``make_vacuum_boundary_spec`` public contract.
"""

# Exact pydoclint attribute types cannot split across physical lines.
# ruff: noqa: E501

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.constants import CARTESIAN_COMPONENTS


class KSScatteringRequest(eqx.Module):
    """Define the ``KSScatteringRequest`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestKsscatteringrequest`

    Attributes
    ----------
    k_parallel_cart_inv_ang : Float64[Array, "n_state 2"]
        Store parallel momenta.
    kinetic_energy_ev : Float64[Array, " n_state"]
        Store kinetic energies.
    outgoing_channel_index : Int32[Array, " n_state"]
        Store outgoing-channel indices.
    surface_normal_cart : Float64[Array, " 3"]
        Store the surface normal.
    energy_block_size : int
        Store the energy block size.
    validity_profile_ref : str
        Store the validity-profile identity.

    See Also
    --------
    make_ks_scattering_request
        Construct a validated request.
    """

    k_parallel_cart_inv_ang: Float64[Array, "n_state 2"]
    kinetic_energy_ev: Float64[Array, " n_state"]
    outgoing_channel_index: Int32[Array, " n_state"]
    surface_normal_cart: Float64[Array, " 3"]
    energy_block_size: int = eqx.field(static=True)
    validity_profile_ref: str = eqx.field(static=True)


class DenseSliceOperator(eqx.Module):
    """Define the ``DenseSliceOperator`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestDensesliceoperator`

    Attributes
    ----------
    blocks_ev : Complex128[Array, "n_slice n_chan n_chan"]
        Store dense operator blocks.

    See Also
    --------
    make_dense_slice_operator
        Construct a validated dense operator.
    """

    blocks_ev: Complex128[Array, "n_slice n_chan n_chan"]


class SparseSliceOperator(eqx.Module):
    """Define the ``SparseSliceOperator`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestSparsesliceoperator`

    Attributes
    ----------
    values_ev : Complex128[Array, " n_nonzero"]
        Store sparse values.
    indices : Int32[Array, "n_nonzero 3"]
        Store sparse indices.
    shape : Tuple[int, int, int]
        Store the dense shape.

    See Also
    --------
    make_sparse_slice_operator
        Construct a validated sparse operator.
    """

    values_ev: Complex128[Array, " n_nonzero"]
    indices: Int32[Array, "n_nonzero 3"]
    shape: Tuple[int, int, int] = eqx.field(static=True)


class KSScatteringProblem(eqx.Module):
    """Define the ``KSScatteringProblem`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestKsscatteringproblem`

    Attributes
    ----------
    slice_operator : Union[DenseSliceOperator, SparseSliceOperator]
        Store the slice operator.
    normal_stencil_offsets : Int32[Array, " n_normal_stencil"]
        Store normal-stencil offsets.
    normal_stencil_values_ev : Complex128[Array, "n_normal_stencil n_slice n_chan"]
        Store normal-stencil values.
    nonlocal_projectors : Complex128[Array, "n_projector n_slice n_chan"]
        Store nonlocal projectors.
    nonlocal_couplings_ev : Complex128[Array, "n_projector n_projector"]
        Store nonlocal couplings.
    slice_coordinates_ang : Float64[Array, " n_slice"]
        Store slice coordinates.
    channel_coordinates : Float64[Array, "n_chan 2"]
        Store channel coordinates.
    hamiltonian_ref : str
        Store the Hamiltonian identity.
    basis_kind : str
        Store the basis kind.
    channel_coordinate_kind : str
        Store the channel-coordinate kind.
    operator_storage_ref : str
        Store the storage identity.
    discretization_ref : str
        Store the discretization identity.

    See Also
    --------
    make_ks_scattering_problem
        Construct a validated problem.
    """

    slice_operator: Union[DenseSliceOperator, SparseSliceOperator]
    normal_stencil_offsets: Int32[Array, " n_normal_stencil"]
    normal_stencil_values_ev: Complex128[
        Array, "n_normal_stencil n_slice n_chan"
    ]
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
    """Define the ``VacuumBoundarySpec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestVacuumboundaryspec`

    Attributes
    ----------
    reference_potential_ev : Float64[Array, ""]
        Store the reference potential.
    direction : str
        Store the propagation direction.
    normalization : str
        Store the normalization convention.

    See Also
    --------
    make_vacuum_boundary_spec
        Construct a validated vacuum boundary.
    """

    reference_potential_ev: Float64[Array, ""]
    direction: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)


class BackingAbsorberSpec(eqx.Module):
    """Define the ``BackingAbsorberSpec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestBackingabsorberspec`

    Attributes
    ----------
    absorber_strength_ev : Float64[Array, ""]
        Store absorber strength.
    absorber_start_ang : Float64[Array, ""]
        Store absorber start position.
    absorber_width_ang : Float64[Array, ""]
        Store absorber width.
    side : str
        Store absorber side.
    shape : str
        Store the profile shape.
    profile_ref : str
        Store the profile identity.

    See Also
    --------
    make_backing_absorber_spec
        Construct a validated absorber.
    """

    absorber_strength_ev: Float64[Array, ""]
    absorber_start_ang: Float64[Array, ""]
    absorber_width_ang: Float64[Array, ""]
    side: str = eqx.field(static=True)
    shape: str = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


class KSScatteringBoundaryProfile(eqx.Module):
    """Define the ``KSScatteringBoundaryProfile`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestKsscatteringboundaryprofile`

    Attributes
    ----------
    left : VacuumBoundarySpec
        Store the left boundary.
    right : VacuumBoundarySpec
        Store the right boundary.
    backing_absorber : Optional[BackingAbsorberSpec]
        Store the backing absorber.
    vacuum_convergence_ref : str
        Store vacuum-convergence evidence.
    slab_convergence_ref : str
        Store slab-convergence evidence.
    absorber_convergence_ref : Optional[str]
        Store absorber-convergence evidence.
    profile_ref : str
        Store the profile identity.

    See Also
    --------
    make_ks_scattering_boundary_profile
        Construct a validated boundary profile.
    """

    left: VacuumBoundarySpec
    right: VacuumBoundarySpec
    backing_absorber: Optional[BackingAbsorberSpec]
    vacuum_convergence_ref: str = eqx.field(static=True)
    slab_convergence_ref: str = eqx.field(static=True)
    absorber_convergence_ref: Optional[str] = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


class LightMatterCouplingSpec(eqx.Module):
    """Define the ``LightMatterCouplingSpec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestLightmattercouplingspec`

    Attributes
    ----------
    representation : str
        Store the coupling representation.
    photon_momentum : str
        Store the photon-momentum policy.
    final_spin_mode : str
        Store the final-spin mode.
    profile_ref : str
        Store the profile identity.

    See Also
    --------
    make_light_matter_coupling_spec
        Construct a validated coupling specification.
    """

    representation: str = eqx.field(static=True)
    photon_momentum: str = eqx.field(static=True)
    final_spin_mode: str = eqx.field(static=True)
    profile_ref: str = eqx.field(static=True)


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
    """Compute the ``make_ks_scattering_request`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeKsScatteringRequest`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    k_parallel_cart_inv_ang : Float64[Array, 'n_state 2']
        Input value for this operation.
    kinetic_energy_ev : Float64[Array, ' n_state']
        Input value for this operation.
    outgoing_channel_index : Int32[Array, ' n_state']
        Input value for this operation.
    surface_normal_cart : Float64[Array, ' 3']
        Input value for this operation.
    energy_block_size : int
        Input value for this operation.
    validity_profile_ref : str
        Input value for this operation.

    Returns
    -------
    result : KSScatteringRequest
        Validated operation result.

    Raises
    ------
    ValueError
        If request metadata or state axes are inconsistent.
    """
    if energy_block_size <= 0 or not validity_profile_ref:
        raise ValueError("scattering request metadata is invalid")
    parallel: Float64[Array, "n_state 2"] = jnp.asarray(
        k_parallel_cart_inv_ang, dtype=jnp.float64
    )
    energy: Float64[Array, " n_state"] = jnp.asarray(
        kinetic_energy_ev, dtype=jnp.float64
    )
    channels: Int32[Array, " n_state"] = jnp.asarray(
        outgoing_channel_index, dtype=jnp.int32
    )
    normal: Float64[Array, " 3"] = jnp.asarray(
        surface_normal_cart, dtype=jnp.float64
    )
    if parallel.shape[0] != energy.shape[0] or channels.shape != energy.shape:
        raise ValueError("scattering request state axes must agree")
    parallel = eqx.error_if(
        parallel,
        ~jnp.all(jnp.isfinite(parallel)),
        "parallel momenta must be finite",
    )
    energy = eqx.error_if(
        energy,
        ~jnp.all(jnp.isfinite(energy)) | jnp.any(energy <= 0.0),
        "kinetic energies must be finite and positive",
    )
    channels = eqx.error_if(
        channels,
        jnp.any(channels < 0),
        "outgoing channel indices must be nonnegative",
    )
    normal = eqx.error_if(
        normal,
        ~jnp.all(jnp.isfinite(normal))
        | ~jnp.isclose(jnp.linalg.norm(normal), 1.0),
        "surface normal must be finite and normalized",
    )
    result: KSScatteringRequest = KSScatteringRequest(
        parallel,
        energy,
        channels,
        normal,
        energy_block_size,
        validity_profile_ref,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_dense_slice_operator(
    blocks_ev: Complex128[Array, "n_slice n_chan n_chan"],
) -> DenseSliceOperator:
    """Compute the ``make_dense_slice_operator`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeDenseSliceOperator`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    blocks_ev : Complex128[Array, 'n_slice n_chan n_chan']
        Input value for this operation.

    Returns
    -------
    result : DenseSliceOperator
        Validated operation result.

    Raises
    ------
    ValueError
        If the dense blocks are not square matrices.
    """
    blocks: Complex128[Array, "n_slice n_chan n_chan"] = jnp.asarray(
        blocks_ev, dtype=jnp.complex128
    )
    if (
        blocks.ndim != CARTESIAN_COMPONENTS
        or blocks.shape[-1] != blocks.shape[-2]
    ):
        raise ValueError("dense slice blocks must be square matrices")
    blocks = eqx.error_if(
        blocks,
        ~jnp.all(jnp.isfinite(blocks)),
        "dense slice blocks must be finite",
    )
    result: DenseSliceOperator = DenseSliceOperator(blocks)
    return result


@jaxtyped(typechecker=beartype)
def make_sparse_slice_operator(
    values_ev: Complex128[Array, " n_nonzero"],
    indices: Int32[Array, "n_nonzero 3"],
    *,
    shape: Tuple[int, int, int],
) -> SparseSliceOperator:
    """Compute the ``make_sparse_slice_operator`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeSparseSliceOperator`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    values_ev : Complex128[Array, ' n_nonzero']
        Input value for this operation.
    indices : Int32[Array, 'n_nonzero 3']
        Input value for this operation.
    shape : Tuple[int, int, int]
        Input value for this operation.

    Returns
    -------
    result : SparseSliceOperator
        Validated operation result.

    Raises
    ------
    ValueError
        If the declared shape or sparse entry axes are inconsistent.
    """
    values: Complex128[Array, " n_nonzero"] = jnp.asarray(
        values_ev, dtype=jnp.complex128
    )
    sparse_indices: Int32[Array, "n_nonzero 3"] = jnp.asarray(
        indices, dtype=jnp.int32
    )
    if len(shape) != CARTESIAN_COMPONENTS or any(size <= 0 for size in shape):
        raise ValueError("sparse slice shape must have three positive entries")
    if sparse_indices.shape != (values.shape[0], CARTESIAN_COMPONENTS):
        raise ValueError("sparse slice indices must align with values")
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "sparse slice values must be finite",
    )
    sparse_indices = eqx.error_if(
        sparse_indices,
        jnp.any(sparse_indices < 0)
        | jnp.any(sparse_indices >= jnp.asarray(shape)[None, :]),
        "sparse slice indices must lie within the declared shape",
    )
    result: SparseSliceOperator = SparseSliceOperator(
        values, sparse_indices, shape
    )
    return result


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_ks_scattering_problem(  # noqa: DOC105, PLR0913
    slice_operator: Union[DenseSliceOperator, SparseSliceOperator],
    normal_stencil_offsets: Int32[Array, " n_normal_stencil"],
    normal_stencil_values_ev: Complex128[
        Array, "n_normal_stencil n_slice n_chan"
    ],
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
    """Compute the ``make_ks_scattering_problem`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeKsScatteringProblem`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    slice_operator : SliceOperator
        Input value for this operation.
    normal_stencil_offsets : Int32[Array, ' n_normal_stencil']
        Input value for this operation.
    normal_stencil_values_ev : _NormalStencilValues
        Input value for this operation.
    nonlocal_projectors : Complex128[Array, 'n_projector n_slice n_chan']
        Input value for this operation.
    nonlocal_couplings_ev : Complex128[Array, 'n_projector n_projector']
        Input value for this operation.
    slice_coordinates_ang : Float64[Array, ' n_slice']
        Input value for this operation.
    channel_coordinates : Float64[Array, 'n_chan 2']
        Input value for this operation.
    hamiltonian_ref : str
        Input value for this operation.
    basis_kind : str
        Input value for this operation.
    channel_coordinate_kind : str
        Input value for this operation.
    operator_storage_ref : str
        Input value for this operation.
    discretization_ref : str
        Input value for this operation.

    Returns
    -------
    result : KSScatteringProblem
        Validated operation result.

    Raises
    ------
    ValueError
        If identities, numerical axes, or slice-operator axes disagree.
    """
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
    offsets: Int32[Array, " n_normal_stencil"] = jnp.asarray(
        normal_stencil_offsets, dtype=jnp.int32
    )
    stencil: Complex128[Array, "n_normal_stencil n_slice n_chan"] = (
        jnp.asarray(normal_stencil_values_ev, dtype=jnp.complex128)
    )
    projectors: Complex128[Array, "n_projector n_slice n_chan"] = jnp.asarray(
        nonlocal_projectors, dtype=jnp.complex128
    )
    couplings: Complex128[Array, "n_projector n_projector"] = jnp.asarray(
        nonlocal_couplings_ev, dtype=jnp.complex128
    )
    slices: Float64[Array, " n_slice"] = jnp.asarray(
        slice_coordinates_ang, dtype=jnp.float64
    )
    channels: Float64[Array, "n_chan 2"] = jnp.asarray(
        channel_coordinates, dtype=jnp.float64
    )
    if (
        stencil.shape[0] != offsets.shape[0]
        or stencil.shape[1] != slices.shape[0]
        or stencil.shape[2] != channels.shape[0]
        or projectors.shape[1:] != stencil.shape[1:]
        or couplings.shape != (projectors.shape[0], projectors.shape[0])
    ):
        raise ValueError("scattering problem numerical axes are inconsistent")
    operator_shape: Tuple[int, int, int] = (
        slice_operator.blocks_ev.shape
        if isinstance(slice_operator, DenseSliceOperator)
        else slice_operator.shape
    )
    if operator_shape != (
        slices.shape[0],
        channels.shape[0],
        channels.shape[0],
    ):
        raise ValueError(
            "slice operator axes must match problem slices and channels"
        )
    stencil = eqx.error_if(
        stencil,
        ~jnp.all(jnp.isfinite(stencil))
        | ~jnp.all(jnp.isfinite(projectors))
        | ~jnp.all(jnp.isfinite(couplings))
        | ~jnp.all(jnp.isfinite(slices))
        | ~jnp.all(jnp.isfinite(channels)),
        "scattering problem numerical values must be finite",
    )
    result: KSScatteringProblem = KSScatteringProblem(
        slice_operator,
        offsets,
        stencil,
        projectors,
        couplings,
        slices,
        channels,
        hamiltonian_ref,
        basis_kind,
        channel_coordinate_kind,
        operator_storage_ref,
        discretization_ref,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_vacuum_boundary_spec(
    reference_potential_ev: Float64[Array, ""], *, direction: str
) -> VacuumBoundarySpec:
    """Compute the ``make_vacuum_boundary_spec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeVacuumBoundarySpec`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    reference_potential_ev : Float64[Array, '']
        Input value for this operation.
    direction : str
        Input value for this operation.

    Returns
    -------
    result : VacuumBoundarySpec
        Validated operation result.

    Raises
    ------
    ValueError
        If the boundary direction lacks support.
    """
    if direction not in ("left", "right"):
        raise ValueError("vacuum boundary direction must be left or right")
    potential: Float64[Array, ""] = jnp.asarray(
        reference_potential_ev, dtype=jnp.float64
    )
    potential = eqx.error_if(
        potential,
        ~jnp.isfinite(potential),
        "vacuum reference potential must be finite",
    )
    result: VacuumBoundarySpec = VacuumBoundarySpec(
        potential,
        direction,
        "unit_normal_flux",
    )
    return result


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
    """Compute the ``make_backing_absorber_spec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeBackingAbsorberSpec`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    absorber_strength_ev : Float64[Array, '']
        Input value for this operation.
    absorber_start_ang : Float64[Array, '']
        Input value for this operation.
    absorber_width_ang : Float64[Array, '']
        Input value for this operation.
    side : str
        Input value for this operation.
    shape : str
        Input value for this operation.
    profile_ref : str
        Input value for this operation.

    Returns
    -------
    result : BackingAbsorberSpec
        Validated operation result.

    Raises
    ------
    ValueError
        If absorber identity, side, or shape metadata is invalid.
    """
    if not all((side, shape, profile_ref)):
        raise ValueError("absorber metadata must be nonempty")
    if side not in ("left", "right") or shape not in (
        "polynomial",
        "cosine",
    ):
        raise ValueError("absorber side or shape is unsupported")
    strength: Float64[Array, ""] = jnp.asarray(
        absorber_strength_ev, dtype=jnp.float64
    )
    start: Float64[Array, ""] = jnp.asarray(
        absorber_start_ang, dtype=jnp.float64
    )
    width: Float64[Array, ""] = jnp.asarray(
        absorber_width_ang, dtype=jnp.float64
    )
    strength = eqx.error_if(
        strength,
        ~jnp.isfinite(strength) | (strength < 0.0),
        "absorber strength must be finite and nonnegative",
    )
    start = eqx.error_if(
        start,
        ~jnp.isfinite(start),
        "absorber start must be finite",
    )
    width = eqx.error_if(
        width,
        ~jnp.isfinite(width) | (width <= 0.0),
        "absorber width must be finite and positive",
    )
    result: BackingAbsorberSpec = BackingAbsorberSpec(
        strength,
        start,
        width,
        side,
        shape,
        profile_ref,
    )
    return result


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
    """Compute the ``make_ks_scattering_boundary_profile`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeKsScatteringBoundaryProfile`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    left : VacuumBoundarySpec
        Input value for this operation.
    right : VacuumBoundarySpec
        Input value for this operation.
    backing_absorber : Optional[BackingAbsorberSpec]
        Input value for this operation.
    vacuum_convergence_ref : str
        Input value for this operation.
    slab_convergence_ref : str
        Input value for this operation.
    absorber_convergence_ref : Optional[str]
        Input value for this operation.
    profile_ref : str
        Input value for this operation.

    Returns
    -------
    result : KSScatteringBoundaryProfile
        Validated operation result.

    Raises
    ------
    ValueError
        If profile identities, lead directions, or absorber evidence disagree.
    """
    if not all((vacuum_convergence_ref, slab_convergence_ref, profile_ref)):
        raise ValueError("boundary profile references must be nonempty")
    if left.direction != "left" or right.direction != "right":
        raise ValueError("boundary profile lead directions are inconsistent")
    if (backing_absorber is None) != (absorber_convergence_ref is None):
        raise ValueError("absorber convergence must match absorber presence")
    result: KSScatteringBoundaryProfile = KSScatteringBoundaryProfile(
        left,
        right,
        backing_absorber,
        vacuum_convergence_ref,
        slab_convergence_ref,
        absorber_convergence_ref,
        profile_ref,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_light_matter_coupling_spec(
    *,
    representation: str,
    photon_momentum: str,
    final_spin_mode: str,
    profile_ref: str,
) -> LightMatterCouplingSpec:
    """Compute the ``make_light_matter_coupling_spec`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_ks_scattering.TestMakeLightMatterCouplingSpec`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    representation : str
        Input value for this operation.
    photon_momentum : str
        Input value for this operation.
    final_spin_mode : str
        Input value for this operation.
    profile_ref : str
        Input value for this operation.

    Returns
    -------
    result : LightMatterCouplingSpec
        Validated operation result.

    Raises
    ------
    ValueError
        If coupling metadata is empty or its final-spin mode lacks support.
    """
    if not all(
        (representation, photon_momentum, final_spin_mode, profile_ref)
    ):
        raise ValueError("light-matter metadata must be nonempty")
    if final_spin_mode not in ("scalar", "spinor"):
        raise ValueError("light-matter final spin mode is unsupported")
    result: LightMatterCouplingSpec = LightMatterCouplingSpec(
        representation, photon_momentum, final_spin_mode, profile_ref
    )
    return result


__all__: list[str] = [
    "BackingAbsorberSpec",
    "DenseSliceOperator",
    "KSScatteringBoundaryProfile",
    "KSScatteringProblem",
    "KSScatteringRequest",
    "LightMatterCouplingSpec",
    "SparseSliceOperator",
    "VacuumBoundarySpec",
    "make_backing_absorber_spec",
    "make_dense_slice_operator",
    "make_ks_scattering_boundary_profile",
    "make_ks_scattering_problem",
    "make_ks_scattering_request",
    "make_light_matter_coupling_spec",
    "make_sparse_slice_operator",
    "make_vacuum_boundary_spec",
]
