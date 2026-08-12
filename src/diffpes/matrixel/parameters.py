r"""Compute matrix-element parameters and sensitivity diagnostics.

Extended Summary
----------------
The module packs active radial and phase parameters into one real optimizer
vector and reconstructs validated carriers from that vector. Named unit
directions expose the overall phase gauge and every normalized
Slater-contraction scale gauge. Complete isolated band-group weights and
their Jacobians report which matrix-element coordinates the sampled
momentum coverage can determine.

Routine Listings
----------------
:func:`band_group_weight_sensitivity`
    Compute complete isolated band-group weights and their Jacobian.
:func:`log_band_group_weight_sensitivity`
    Convert positive group-weight derivatives to logarithmic derivatives.
:func:`matrix_element_phase_gauge_direction`
    Build the unit overall-phase tangent in packed coordinates.
:func:`pack_matrixel_params`
    Pack active matrix-element parameters into one real vector.
:func:`radial_coefficient_scale_gauge_directions`
    Build normalized radial coefficient-scale gauge tangents.
:func:`unpack_matrixel_params`
    Construct active matrix-element parameters from one real vector.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Dict, List, Tuple, Union
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    BAND_GROUP_COMPLEMENT_GAP_MIN_EV,
    CARTESIAN_COMPONENTS,
)
from diffpes.types import (
    DiagonalizedBands,
    ExperimentGeometry,
    MatrixElementParams,
    PyTreeDef,
    RadialSpec,
)
from diffpes.utils import pack_complex, unpack_complex

from .transition import _basis_key, matrix_element_intensity


def _active_parameter_tree(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float64[Array, ""],
) -> Dict[
    str,
    Union[Float64[Array, "..."], Complex128[Array, "..."]],
]:
    """PRIVATE: Collect the mode-active matrix-element parameter leaves.

    Notes
    -----
    The mode string selects which leaves stay active.
    """
    active: Dict[
        str,
        Union[Float64[Array, "..."], Complex128[Array, "..."]],
    ] = {}
    if radial.mode == "slater":
        active["zeta_shell"] = radial.zeta_shell
        active["coefficients_shell"] = radial.coefficients_shell
    elif radial.mode == "hydrogenic":
        active["effective_charge_shell"] = radial.effective_charge_shell
    active["phase_shift_angles_shell"] = me_params.phase_shift_angles_shell
    active["sigma_shell"] = me_params.sigma_shell
    active["mean_free_path_ang"] = jnp.asarray(
        mean_free_path_ang,
        dtype=jnp.float64,
    )
    return active


def _pack_active_tree(
    active: Dict[
        str,
        Union[Float64[Array, "..."], Complex128[Array, "..."]],
    ],
) -> Tuple[
    Float64[Array, " n_theta"],
    PyTreeDef,
    Tuple[Tuple[Tuple[int, ...], bool], ...],
]:
    """PRIVATE: Pack one active tree with stacked complex coordinates.

    Notes
    -----
    The packing stacks real and imaginary parts as one vector.
    """
    leaves: List[Union[Float64[Array, "..."], Complex128[Array, "..."]]]
    tree_definition: PyTreeDef
    leaves, tree_definition = jax.tree_util.tree_flatten(active)
    packed_leaves: List[Float64[Array, " n_leaf"]] = []
    metadata: List[Tuple[Tuple[int, ...], bool]] = []
    leaf: Union[Float64[Array, "..."], Complex128[Array, "..."]]
    for leaf in leaves:
        is_complex: bool = bool(jnp.iscomplexobj(leaf))
        shape: Tuple[int, ...] = tuple(leaf.shape)
        packed_leaf: Float64[Array, " n_leaf"] = (
            pack_complex(leaf).reshape(-1)
            if is_complex
            else jnp.asarray(leaf, dtype=jnp.float64).reshape(-1)
        )
        packed_leaves.append(packed_leaf)
        metadata.append((shape, is_complex))
    flat: Float64[Array, " n_theta"] = jnp.concatenate(packed_leaves)
    packing_metadata: Tuple[Tuple[Tuple[int, ...], bool], ...] = tuple(
        metadata
    )
    result: Tuple[
        Float64[Array, " n_theta"],
        PyTreeDef,
        Tuple[Tuple[Tuple[int, ...], bool], ...],
    ] = (flat, tree_definition, packing_metadata)
    return result


def _validate_band_groups(
    bands: DiagonalizedBands,
    band_groups: Tuple[Tuple[int, ...], ...],
) -> None:
    """PRIVATE: Validate complete groups against each eigenspectrum.

    Notes
    -----
    The check rejects overlapping or incomplete band groups.
    """
    if type(band_groups) is not tuple or not band_groups:
        message: str = "band_groups must be a nonempty tuple"
        raise ValueError(message)
    n_bands: int = bands.eigenvalues.shape[1]
    occupied: set[int] = set()
    energies: Float64[NDArray, "nkpt nband"] = np.asarray(bands.eigenvalues)
    group: Tuple[int, ...]
    for group in band_groups:
        if (
            type(group) is not tuple
            or not group
            or any(type(index) is not int for index in group)
            or len(set(group)) != len(group)
            or any(index < 0 or index >= n_bands for index in group)
        ):
            message = "each band group must contain unique valid indices"
            raise ValueError(message)
        if occupied.intersection(group):
            message = "band groups must not overlap"
            raise ValueError(message)
        occupied.update(group)
        complement: Tuple[int, ...] = tuple(
            index for index in range(n_bands) if index not in group
        )
        if not complement:
            continue
        cross_gaps: Float64[NDArray, "nkpt n_group n_complement"] = np.abs(
            energies[:, np.asarray(group), None]
            - energies[:, None, np.asarray(complement)]
        )
        if bool(np.any(cross_gaps < BAND_GROUP_COMPLEMENT_GAP_MIN_EV)):
            message = (
                "band group cuts a degeneracy or lacks complement isolation"
            )
            raise ValueError(message)


@jaxtyped(typechecker=beartype)
def pack_matrixel_params(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float64[Array, ""],
) -> Tuple[
    Float64[Array, " n_theta"],
    PyTreeDef,
    Tuple[Tuple[Tuple[int, ...], bool], ...],
]:
    """Pack active matrix-element parameters into one real vector.

    Preserve the optimizer boundary independently of radial mode.

    Slater mode packs exponents and contraction coefficients.  Hydrogenic
    mode packs effective charges.  Every mode packs shell scales, physical
    channel phases, and mean free path.  Grid samples and calibrated fixed
    channel ratios remain outside the inversion view.  Complex leaves use
    stacked real and imaginary coordinates.

    :see: :class:`~.test_parameters.TestPackMatrixelParams`

    Parameters
    ----------
    radial : RadialSpec
        Radial template and active mode.
    me_params : MatrixElementParams
        Shell scales and channel phases.
    mean_free_path_ang : Float64[Array, ""]
        Scalar intensity mean free path in Angstrom.

    Returns
    -------
    flat : Float64[Array, "n_theta"]
        Flat real optimizer coordinates.
    tree_definition : PyTreeDef
        Active parameter-tree definition.
    packing_metadata : Tuple[Tuple[Tuple[int, ...], bool], ...]
        Original leaf shapes and complex flags.

    Raises
    ------
    ValueError
        If carrier metadata disagree or mean free path is not scalar.

    Notes
    -----
    Gather active leaves before applying the stacked-real packing rule.
    """
    if (
        _basis_key(radial.basis) != _basis_key(me_params.basis)
        or radial.radial_shell_index != me_params.radial_shell_index
    ):
        message: str = (
            "radial and matrix-element parameter metadata must agree"
        )
        raise ValueError(message)
    if mean_free_path_ang.ndim != 0:
        message = "mean_free_path_ang must be scalar"
        raise ValueError(message)
    active: Dict[
        str,
        Union[Float64[Array, "..."], Complex128[Array, "..."]],
    ] = _active_parameter_tree(
        radial,
        me_params,
        mean_free_path_ang,
    )
    result: Tuple[
        Float64[Array, " n_theta"],
        PyTreeDef,
        Tuple[Tuple[Tuple[int, ...], bool], ...],
    ] = _pack_active_tree(active)
    return result


@jaxtyped(typechecker=beartype)
def unpack_matrixel_params(
    flat: Float64[Array, " n_theta"],
    tree_definition: PyTreeDef,
    packing_metadata: Tuple[Tuple[Tuple[int, ...], bool], ...],
    radial_template: RadialSpec,
    me_params_template: MatrixElementParams,
) -> Tuple[RadialSpec, MatrixElementParams, Float64[Array, ""]]:
    """Construct active matrix-element parameters from one real vector.

    Reuse static metadata and excluded calibration leaves from the templates.

    The tree definition restores named active leaves.  Shape metadata removes
    each flat slice, and the complex flags join stacked coordinates.  The
    reconstruction writes only mode-active fields and the compact physical
    phase vector.

    :see: :class:`~.test_parameters.TestUnpackMatrixelParams`

    Parameters
    ----------
    flat : Float64[Array, "n_theta"]
        Flat real optimizer coordinates.
    tree_definition : PyTreeDef
        Tree definition returned by :func:`pack_matrixel_params`.
    packing_metadata : Tuple[Tuple[Tuple[int, ...], bool], ...]
        Leaf shapes and complex flags returned by packing.
    radial_template : RadialSpec
        Radial template that retains static and excluded leaves.
    me_params_template : MatrixElementParams
        Matrix-element template that retains static metadata.

    Returns
    -------
    radial : RadialSpec
        Reconstructed radial carrier.
    me_params : MatrixElementParams
        Reconstructed matrix-element carrier.
    mean_free_path_ang : Float64[Array, ""]
        Reconstructed scalar mean free path.

    Raises
    ------
    ValueError
        If metadata or flat-vector lengths disagree.

    Notes
    -----
    Split the real vector before restoring the active named tree.
    """
    if flat.ndim != 1:
        message: str = "flat matrix-element parameters must be a vector"
        raise ValueError(message)
    if tree_definition.num_leaves != len(packing_metadata):
        message = "packing metadata must match the parameter tree"
        raise ValueError(message)
    leaves: List[Union[Float64[Array, "..."], Complex128[Array, "..."]]] = []
    offset: int = 0
    shape: Tuple[int, ...]
    is_complex: bool
    for shape, is_complex in packing_metadata:
        scalar_count: int = math.prod(shape)
        packed_count: int = scalar_count * (2 if is_complex else 1)
        next_offset: int = offset + packed_count
        if next_offset > flat.shape[0]:
            message = "flat vector is shorter than its packing metadata"
            raise ValueError(message)
        packed_leaf: Float64[Array, " n_leaf"] = flat[offset:next_offset]
        leaf: Union[Float64[Array, "..."], Complex128[Array, "..."]] = (
            unpack_complex(packed_leaf.reshape(shape + (2,)))
            if is_complex
            else packed_leaf.reshape(shape)
        )
        leaves.append(leaf)
        offset = next_offset
    if offset != flat.shape[0]:
        message = "flat vector is longer than its packing metadata"
        raise ValueError(message)
    active: Dict[
        str,
        Union[Float64[Array, "..."], Complex128[Array, "..."]],
    ] = jax.tree_util.tree_unflatten(
        tree_definition,
        leaves,
    )

    radial: RadialSpec = radial_template
    if "zeta_shell" in active:
        radial = eqx.tree_at(
            lambda item: item.zeta_shell,
            radial,
            active["zeta_shell"],
        )
    if "coefficients_shell" in active:
        radial = eqx.tree_at(
            lambda item: item.coefficients_shell,
            radial,
            active["coefficients_shell"],
        )
    if "effective_charge_shell" in active:
        radial = eqx.tree_at(
            lambda item: item.effective_charge_shell,
            radial,
            active["effective_charge_shell"],
        )

    me_params: MatrixElementParams = eqx.tree_at(
        lambda item: (item.sigma_shell, item.phase_shift_angles_shell),
        me_params_template,
        (
            active["sigma_shell"],
            active["phase_shift_angles_shell"],
        ),
    )
    mean_free_path_ang: Float64[Array, ""] = active["mean_free_path_ang"]
    result: Tuple[RadialSpec, MatrixElementParams, Float64[Array, ""]] = (
        radial,
        me_params,
        mean_free_path_ang,
    )
    return result


@jaxtyped(typechecker=beartype)
def matrix_element_phase_gauge_direction(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float64[Array, ""],
) -> Float64[Array, " n_theta"]:
    """Build the unit overall-phase tangent in packed coordinates.

    Shift every physical final-state channel phase by the same angle.

    The tangent generates one common phase on the complete transition
    amplitude.  It excludes the nonexistent lower channel of each s shell.
    Its Euclidean normalization gives Fisher analyses a convention-free null
    direction.

    :see: :class:`~.test_parameters.TestMatrixElementPhaseGaugeDirection`

    Parameters
    ----------
    radial : RadialSpec
        Radial parameter carrier.
    me_params : MatrixElementParams
        Matrix-element parameter carrier.
    mean_free_path_ang : Float64[Array, ""]
        Scalar intensity mean free path.

    Returns
    -------
    direction : Float64[Array, "n_theta"]
        Unit packed phase-gauge tangent.

    Notes
    -----
    Pack a unit common phase displacement and subtract the base vector.
    """
    base: Float64[Array, " n_theta"] = pack_matrixel_params(
        radial,
        me_params,
        mean_free_path_ang,
    )[0]
    shifted_angles: Float64[Array, " n_valid_phase"] = (
        me_params.phase_shift_angles_shell + 1.0
    )
    shifted_params: MatrixElementParams = eqx.tree_at(
        lambda item: item.phase_shift_angles_shell,
        me_params,
        shifted_angles,
    )
    displaced: Float64[Array, " n_theta"] = pack_matrixel_params(
        radial,
        shifted_params,
        mean_free_path_ang,
    )[0]
    tangent: Float64[Array, " n_theta"] = displaced - base
    direction: Float64[Array, " n_theta"] = tangent / jnp.linalg.norm(tangent)
    return direction


@jaxtyped(typechecker=beartype)
def radial_coefficient_scale_gauge_directions(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float64[Array, ""],
) -> Float64[Array, "n_gauge n_theta"]:
    """Build normalized radial coefficient-scale gauge tangents.

    Return one tangent for every normalized Slater contraction shell.

    Multiplying all coefficients in one shell by a positive common scale does
    not change its normalized radial wavefunction.  Hydrogenic, grid, and
    fixed modes expose no contraction coefficient coordinate.

    :see: :class:`~.test_parameters.TestRadialCoefficientScaleGaugeDirections`

    Parameters
    ----------
    radial : RadialSpec
        Radial parameter carrier.
    me_params : MatrixElementParams
        Matrix-element parameter carrier.
    mean_free_path_ang : Float64[Array, ""]
        Scalar intensity mean free path.

    Returns
    -------
    directions : Float64[Array, "n_gauge n_theta"]
        Unit packed coefficient-scale tangents.

    Notes
    -----
    Differentiate the finite common rescaling in each shell coordinate block.
    """
    base: Float64[Array, " n_theta"] = pack_matrixel_params(
        radial,
        me_params,
        mean_free_path_ang,
    )[0]
    if radial.mode != "slater":
        directions: Float64[Array, "n_gauge n_theta"] = jnp.zeros(
            (0, base.shape[0]),
            dtype=jnp.float64,
        )
        return directions
    tangents: List[Float64[Array, " n_theta"]] = []
    shell: int
    for shell in range(radial.coefficients_shell.shape[0]):
        displaced_coefficients: Float64[Array, "n_shell n_contraction"] = (
            radial.coefficients_shell.at[shell].add(
                radial.coefficients_shell[shell]
            )
        )
        displaced_radial: RadialSpec = eqx.tree_at(
            lambda item: item.coefficients_shell,
            radial,
            displaced_coefficients,
        )
        displaced: Float64[Array, " n_theta"] = pack_matrixel_params(
            displaced_radial,
            me_params,
            mean_free_path_ang,
        )[0]
        tangent: Float64[Array, " n_theta"] = displaced - base
        tangents.append(tangent / jnp.linalg.norm(tangent))
    directions = jnp.stack(tangents)
    return directions  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def band_group_weight_sensitivity(  # noqa: DOC105, DOC502
    flat_params: Float64[Array, " n_theta"],
    rebuild: Callable[
        [Float64[Array, " n_theta"], DiagonalizedBands, ExperimentGeometry],
        Complex128[Array, "n_k n_bands n_spin"],
    ],
    bands: DiagonalizedBands,
    experiment: ExperimentGeometry,
    band_groups: Tuple[Tuple[int, ...], ...],
) -> Tuple[
    Float64[Array, "n_k n_group"],
    Float64[Array, "n_theta n_k n_group"],
]:
    """Compute complete isolated band-group weights and their Jacobian.

    Apply ``jacfwd`` only after the static physical group validation.

    The rebuild callback returns polarization-contracted outgoing-spin band
    amplitudes.  The helper sums spin modulus squares and then sums all members
    of each complete group.  It assigns no spectral, exposure, background, or
    detector-count interpretation.

    :see: :class:`~.test_parameters.TestBandGroupWeightSensitivity`

    Parameters
    ----------
    flat_params : Float64[Array, "n_theta"]
        Real packed parameter vector.
    rebuild : Callable
        Callback from parameters, bands, and experiment to ``[K,B,S]``
        complex amplitudes.
    bands : DiagonalizedBands
        Eigensystem whose energies define complete isolated groups.
    experiment : ExperimentGeometry
        Experiment carrier passed unchanged to ``rebuild``.
    band_groups : Tuple[Tuple[int, ...], ...]
        Nonoverlapping static complete band groups.

    Returns
    -------
    band_group_weights : Float64[Array, "n_k n_group"]
        Unresolved-spin complete-group matrix-element weights.
    weight_jacobian : Float64[Array, "n_theta n_k n_group"]
        Forward-mode derivative of every group weight.

    Raises
    ------
    ValueError
        If a group is partial, overlapping, invalid, or insufficiently
        isolated, or if callback axes disagree.

    Notes
    -----
    Validate group topology before differentiating the group-summed weights.
    """
    _validate_band_groups(bands, band_groups)

    def group_weights(
        candidate: Float64[Array, " n_theta"],
    ) -> Float64[Array, "n_k n_group"]:
        """Return unresolved-spin weights summed over static band groups."""
        spin_amplitudes: Complex128[Array, "n_k n_bands n_spin"] = rebuild(
            candidate,
            bands,
            experiment,
        )
        if (
            spin_amplitudes.ndim != CARTESIAN_COMPONENTS
            or spin_amplitudes.shape[:2] != bands.eigenvalues.shape
        ):
            message: str = "rebuild must return amplitudes with shape [K,B,S]"
            raise ValueError(message)
        band_weights: Float64[Array, "n_k n_bands"] = matrix_element_intensity(
            spin_amplitudes
        )
        weights: Float64[Array, "n_k n_group"] = jnp.stack(
            tuple(
                jnp.sum(
                    band_weights[:, jnp.asarray(group, dtype=jnp.int32)],
                    axis=-1,
                )
                for group in band_groups
            ),
            axis=-1,
        )
        return weights

    band_group_weights: Float64[Array, "n_k n_group"] = group_weights(
        flat_params
    )
    output_first_jacobian: Float64[Array, "n_k n_group n_theta"] = jax.jacfwd(
        group_weights
    )(flat_params)
    weight_jacobian: Float64[Array, "n_theta n_k n_group"] = jnp.moveaxis(
        output_first_jacobian,
        -1,
        0,
    )
    result: Tuple[
        Float64[Array, "n_k n_group"],
        Float64[Array, "n_theta n_k n_group"],
    ] = (band_group_weights, weight_jacobian)
    return result


@jaxtyped(typechecker=beartype)
def log_band_group_weight_sensitivity(
    band_group_weights: Float64[Array, " ..."],
    weight_jacobian: Float64[Array, "n_theta ..."],
    min_band_group_weight: float,
) -> Tuple[Float64[Array, "n_theta ..."], Bool[Array, " ..."]]:
    """Convert positive group-weight derivatives to logarithmic derivatives.

    Mark dark or sub-floor weights invalid without dividing by them.

    The helper returns a zero derivative sentinel outside its positive domain.
    Consumers must use the validity mask rather than interpreting that sentinel
    as physical logarithmic information.

    :see: :class:`~.test_parameters.TestLogBandGroupWeightSensitivity`

    Parameters
    ----------
    band_group_weights : Float64[Array, "..."]
        Complete-group matrix-element weights.
    weight_jacobian : Float64[Array, "n_theta ..."]
        Derivatives of those weights.
    min_band_group_weight : float
        Static strictly positive validity floor.

    Returns
    -------
    log_weight_jacobian : Float64[Array, "n_theta ..."]
        ``dw / w`` on valid weights and zero elsewhere.
    valid : Bool[Array, "..."]
        Positive-domain validity mask.

    Raises
    ------
    ValueError
        If the floor or array axes are invalid.

    Notes
    -----
    Replace invalid denominators before division and apply the mask afterward.
    """
    if (
        type(min_band_group_weight) is not float
        or not math.isfinite(min_band_group_weight)
        or min_band_group_weight <= 0.0
    ):
        message: str = "min_band_group_weight must be a finite positive float"
        raise ValueError(message)
    if weight_jacobian.shape[1:] != band_group_weights.shape:
        message = "weight Jacobian trailing axes must match group weights"
        raise ValueError(message)
    valid: Bool[Array, " ..."] = band_group_weights >= min_band_group_weight
    safe_weights: Float64[Array, " ..."] = jnp.where(
        valid,
        band_group_weights,
        1.0,
    )
    log_weight_jacobian: Float64[Array, "n_theta ..."] = jnp.where(
        valid[None, ...],
        weight_jacobian / safe_weights[None, ...],
        0.0,
    )
    result: Tuple[
        Float64[Array, "n_theta ..."],
        Bool[Array, " ..."],
    ] = (log_weight_jacobian, valid)
    return result


__all__: list[str] = [
    "band_group_weight_sensitivity",
    "log_band_group_weight_sensitivity",
    "matrix_element_phase_gauge_direction",
    "pack_matrixel_params",
    "radial_coefficient_scale_gauge_directions",
    "unpack_matrixel_params",
]
