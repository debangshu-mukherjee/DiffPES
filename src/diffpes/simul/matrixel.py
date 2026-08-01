r"""Assemble coherent orbital and band photoemission matrix elements.

Extended Summary
----------------
The module keeps the photoemission transition in orbital space until the
last possible stage.  For outgoing-spin channel :math:`\sigma`, real
orbital :math:`a`, and real dipole channel :math:`d=(y,z,x)`, it evaluates

.. math::

   d^\sigma_{ad} =
   \sigma_a e^{-z_a/(2\lambda)}
   e^{i(\mathbf k_i-\mathbf k_f)\cdot\mathbf R_a}
   \sum_{c,y} B_{ac}e^{i\delta_{ac}}G_{acdy}Y_y(\hat{\mathbf k}_f).

Callers supply the vacuum final momentum explicitly.  The module never
reconstructs it from the inner potential.  Orbital positions use explicit
Wannier centres before atom-derived centres.  Attenuation acts on amplitude
with a half exponent.  Cartesian laboratory polarization rotates to the
sample frame before the late linear contraction.  Band projection uses the
stored basis-position-gauge coefficients without conjugating them.  Only
:func:`matrix_element_intensity` takes a modulus square.  Unresolved outgoing
spins sum incoherently there.

Cartesian recurrences evaluate the final harmonics as normalized solid
harmonics.  The algorithm avoids azimuth construction and defines both pole
values and transverse directional derivatives.

The inversion surface packs active parameters into one real optimizer vector.
It excludes calibrated fixed-radial ratios and invalid s-shell lower phases.
Named unit tangents expose the overall phase gauge and each normalized
Slater-contraction scale gauge.  Complete isolated band-group weights answer
which matrix-element coordinates the sampled k coverage can determine.
Plans 07 and 08 add spectral and detector-count semantics.

Routine Listings
----------------
:func:`assemble_orbital_transition_channels`
    Assemble the validated orbital transition tensor.
:func:`band_group_weight_sensitivity`
    Compute complete isolated band-group weights and their Jacobian.
:func:`contract_experiment_polarization`
    Rotate laboratory polarization to the sample and contract it late.
:func:`contract_polarization`
    Compute the sample-frame polarization contraction.
:func:`matrix_element_intensity`
    Sum outgoing-spin modulus squares exactly once.
:func:`log_band_group_weight_sensitivity`
    Convert positive group-weight derivatives to logarithmic derivatives.
:func:`matrix_element_phase_gauge_direction`
    Build the unit overall-phase tangent in packed coordinates.
:func:`orbital_transition_channels`
    Assemble coherent orbital transition channels.
:func:`pack_matrixel_params`
    Pack active matrix-element parameters into one real vector.
:func:`project_band_channels`
    Compute band channels without conjugating orbital coefficients.
:func:`radial_coefficient_scale_gauge_directions`
    Build normalized radial coefficient-scale gauge tangents.
:func:`real_spherical_harmonics_cartesian_all`
    Evaluate all real spherical harmonics from Cartesian directions.
:func:`resolve_orbital_positions_cart`
    Resolve orbital centres in Cartesian Angstrom coordinates.
:func:`transition_source`
    Build conjugated outgoing-spin rows as full source kets.
:func:`unpack_matrixel_params`
    Construct active matrix-element parameters from one real vector.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Bool, Complex, Float, jaxtyped
from numpy.typing import NDArray

from diffpes.maths import channel_tables, polarization_cart_to_real
from diffpes.radial import momentum_inv_ang_to_bohr_inv, radial_bvals
from diffpes.types import (
    BAND_GROUP_COMPLEMENT_GAP_MIN_EV,
    CARTESIAN_COMPONENTS,
    EPS,
    G_PARALLEL_ATOL_INV_ANG,
    ISPIN2_BLOCKS,
    L_MAX,
    MATRIX_NDIM,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
)
from diffpes.utils import pack_complex, unpack_complex

from .polarization import (
    lab_polarization_to_sample,
    sample_azimuth_rotation,
)


def _basis_key(basis: OrbitalBasis) -> tuple[tuple[object, ...], ...]:
    """Return the exact static identity of an orbital basis."""
    key: tuple[tuple[object, ...], ...] = (
        basis.atom_indices,
        basis.n,
        basis.l,
        basis.m,
        basis.spin,
        basis.labels,
    )
    return key


def _spin_layout(basis: OrbitalBasis) -> tuple[int, int]:
    """Validate and return ``(n_spin, n_orbitals_per_spin)``."""
    n_orbitals: int = len(basis.n)
    if not basis.spin:
        layout: tuple[int, int] = (1, n_orbitals)
        return layout
    if n_orbitals % ISPIN2_BLOCKS != 0:
        message: str = "a spinor basis must contain two equal spin blocks"
        raise ValueError(message)
    n_spatial: int = n_orbitals // ISPIN2_BLOCKS
    expected_spin: tuple[int, ...] = (-1,) * n_spatial + (1,) * n_spatial
    if basis.spin != expected_spin:
        message = "spinor basis must use block-down then block-up ordering"
        raise ValueError(message)
    paired_fields: tuple[tuple[object, ...], ...] = (
        basis.atom_indices,
        basis.n,
        basis.l,
        basis.m,
    )
    if any(field[:n_spatial] != field[n_spatial:] for field in paired_fields):
        message = "spin blocks must contain paired spatial orbitals"
        raise ValueError(message)
    layout = (ISPIN2_BLOCKS, n_spatial)
    return layout  # noqa: RET504 -- assign-before-return is required.


def _solid_harmonic_component(
    direction_unit: Float[Array, "... 3"],
    degree: int,
    order: int,
) -> Float[Array, " ..."]:
    """Evaluate one normalized real solid harmonic on the unit sphere."""
    absolute_order: int = abs(order)
    x_component: Float[Array, " ..."] = direction_unit[..., 0]
    y_component: Float[Array, " ..."] = direction_unit[..., 1]
    z_component: Float[Array, " ..."] = direction_unit[..., 2]
    transverse: Complex[Array, " ..."] = x_component.astype(
        jnp.complex128
    ) + 1j * y_component.astype(jnp.complex128)
    double_factorial: float = 1.0
    factor_index: int
    for factor_index in range(1, absolute_order + 1):
        double_factorial *= 2.0 * factor_index - 1.0
    sectoral_complex: Complex[Array, " ..."] = transverse**absolute_order
    if order > 0:
        sectoral: Float[Array, " ..."] = double_factorial * jnp.real(
            sectoral_complex
        )
    elif order < 0:
        sectoral = double_factorial * jnp.imag(sectoral_complex)
    else:
        sectoral = jnp.ones_like(z_component)

    polynomial: Float[Array, " ..."] = sectoral
    if degree > absolute_order:
        previous_two: Float[Array, " ..."] = sectoral
        previous_one: Float[Array, " ..."] = (
            (2.0 * absolute_order + 1.0) * z_component * sectoral
        )
        if degree == absolute_order + 1:
            polynomial = previous_one
        else:
            recurrence_degree: int
            for recurrence_degree in range(absolute_order + 2, degree + 1):
                current: Float[Array, " ..."] = (
                    (2.0 * recurrence_degree - 1.0)
                    * z_component
                    * previous_one
                    - (recurrence_degree + absolute_order - 1.0) * previous_two
                ) / (recurrence_degree - absolute_order)
                previous_two, previous_one = previous_one, current
            polynomial = previous_one

    normalization: float = math.sqrt(
        (2 * degree + 1)
        / (4.0 * math.pi)
        * math.factorial(degree - absolute_order)
        / math.factorial(degree + absolute_order)
    )
    if order != 0:
        normalization *= math.sqrt(2.0)
    value: Float[Array, " ..."] = normalization * polynomial
    return value


def _orbital_phase_indices(
    me_params: MatrixElementParams,
) -> tuple[tuple[int, int], ...]:
    """Return each orbital branch's compact phase index or zero sentinel."""
    compact_index: dict[tuple[int, int], int] = {
        key: index for index, key in enumerate(me_params.phase_channel_keys)
    }
    zero_sentinel: int = len(me_params.phase_channel_keys)
    result: tuple[tuple[int, int], ...] = tuple(
        (
            compact_index.get(
                (shell, angular - 1),
                zero_sentinel,
            ),
            compact_index[(shell, angular + 1)],
        )
        for shell, angular in zip(
            me_params.radial_shell_index,
            me_params.basis.l,
            strict=True,
        )
    )
    return result


def _active_parameter_tree(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float[Array, ""],
) -> dict[str, Array]:
    """Collect the mode-active matrix-element parameter leaves."""
    active: dict[str, Array] = {}
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
    active: dict[str, Array],
) -> tuple[
    Float[Array, " n_theta"],
    PyTreeDef,
    tuple[tuple[tuple[int, ...], bool], ...],
]:
    """Pack one active parameter tree with stacked complex coordinates."""
    leaves: list[Array]
    tree_definition: PyTreeDef
    leaves, tree_definition = jax.tree_util.tree_flatten(active)
    packed_leaves: list[Float[Array, " n_leaf"]] = []
    metadata: list[tuple[tuple[int, ...], bool]] = []
    leaf: Array
    for leaf in leaves:
        is_complex: bool = bool(jnp.iscomplexobj(leaf))
        shape: tuple[int, ...] = tuple(leaf.shape)
        packed_leaf: Float[Array, " n_leaf"] = (
            pack_complex(leaf).reshape(-1)
            if is_complex
            else jnp.asarray(leaf, dtype=jnp.float64).reshape(-1)
        )
        packed_leaves.append(packed_leaf)
        metadata.append((shape, is_complex))
    flat: Float[Array, " n_theta"] = jnp.concatenate(packed_leaves)
    packing_metadata: tuple[tuple[tuple[int, ...], bool], ...] = tuple(
        metadata
    )
    result: tuple[
        Float[Array, " n_theta"],
        PyTreeDef,
        tuple[tuple[tuple[int, ...], bool], ...],
    ] = (flat, tree_definition, packing_metadata)
    return result


def _validate_band_groups(
    bands: DiagonalizedBands,
    band_groups: tuple[tuple[int, ...], ...],
) -> None:
    """Validate complete static groups against every sampled eigenspectrum."""
    if type(band_groups) is not tuple or not band_groups:
        message: str = "band_groups must be a nonempty tuple"
        raise ValueError(message)
    n_bands: int = bands.eigenvalues.shape[1]
    occupied: set[int] = set()
    energies: Float[NDArray, "nkpt nband"] = np.asarray(bands.eigenvalues)
    group: tuple[int, ...]
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
        complement: tuple[int, ...] = tuple(
            index for index in range(n_bands) if index not in group
        )
        if not complement:
            continue
        cross_gaps: Float[NDArray, "nkpt n_group n_complement"] = np.abs(
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
    mean_free_path_ang: Float[Array, ""],
) -> tuple[
    Float[Array, " n_theta"],
    PyTreeDef,
    tuple[tuple[tuple[int, ...], bool], ...],
]:
    """Pack active matrix-element parameters into one real vector.

    Preserve the optimizer boundary independently of radial mode.

    Slater mode packs exponents and contraction coefficients.  Hydrogenic
    mode packs effective charges.  Every mode packs shell scales, physical
    channel phases, and mean free path.  Grid samples and calibrated fixed
    channel ratios remain outside the inversion view.  Complex leaves use
    stacked real and imaginary coordinates.

    :see: :class:`~.test_matrixel.TestPackMatrixelParams`

    Parameters
    ----------
    radial : RadialSpec
        Radial template and active mode.
    me_params : MatrixElementParams
        Shell scales and channel phases.
    mean_free_path_ang : Float[Array, ""]
        Scalar intensity mean free path in Angstrom.

    Returns
    -------
    flat : Float[Array, "n_theta"]
        Flat real optimizer coordinates.
    tree_definition : PyTreeDef
        Active parameter-tree definition.
    packing_metadata : tuple[tuple[tuple[int, ...], bool], ...]
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
    active: dict[str, Array] = _active_parameter_tree(
        radial,
        me_params,
        mean_free_path_ang,
    )
    result: tuple[
        Float[Array, " n_theta"],
        PyTreeDef,
        tuple[tuple[tuple[int, ...], bool], ...],
    ] = _pack_active_tree(active)
    return result


@jaxtyped(typechecker=beartype)
def unpack_matrixel_params(
    flat: Float[Array, " n_theta"],
    tree_definition: PyTreeDef,
    packing_metadata: tuple[tuple[tuple[int, ...], bool], ...],
    radial_template: RadialSpec,
    me_params_template: MatrixElementParams,
) -> tuple[RadialSpec, MatrixElementParams, Float[Array, ""]]:
    """Construct active matrix-element parameters from one real vector.

    Reuse static metadata and excluded calibration leaves from the templates.

    The tree definition restores named active leaves.  Shape metadata removes
    each flat slice, and the complex flags join stacked coordinates.  The
    reconstruction writes only mode-active fields and the compact physical
    phase vector.

    :see: :class:`~.test_matrixel.TestUnpackMatrixelParams`

    Parameters
    ----------
    flat : Float[Array, "n_theta"]
        Flat real optimizer coordinates.
    tree_definition : PyTreeDef
        Tree definition returned by :func:`pack_matrixel_params`.
    packing_metadata : tuple[tuple[tuple[int, ...], bool], ...]
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
    mean_free_path_ang : Float[Array, ""]
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
    leaves: list[Array] = []
    offset: int = 0
    shape: tuple[int, ...]
    is_complex: bool
    for shape, is_complex in packing_metadata:
        scalar_count: int = math.prod(shape)
        packed_count: int = scalar_count * (2 if is_complex else 1)
        next_offset: int = offset + packed_count
        if next_offset > flat.shape[0]:
            message = "flat vector is shorter than its packing metadata"
            raise ValueError(message)
        packed_leaf: Float[Array, " n_leaf"] = flat[offset:next_offset]
        leaf: Array = (
            unpack_complex(packed_leaf.reshape(shape + (2,)))
            if is_complex
            else packed_leaf.reshape(shape)
        )
        leaves.append(leaf)
        offset = next_offset
    if offset != flat.shape[0]:
        message = "flat vector is longer than its packing metadata"
        raise ValueError(message)
    active: dict[str, Array] = jax.tree_util.tree_unflatten(
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
    mean_free_path_ang: Float[Array, ""] = active["mean_free_path_ang"]
    result: tuple[RadialSpec, MatrixElementParams, Float[Array, ""]] = (
        radial,
        me_params,
        mean_free_path_ang,
    )
    return result


@jaxtyped(typechecker=beartype)
def matrix_element_phase_gauge_direction(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float[Array, ""],
) -> Float[Array, " n_theta"]:
    """Build the unit overall-phase tangent in packed coordinates.

    Shift every physical final-state channel phase by the same angle.

    The tangent generates one common phase on the complete transition
    amplitude.  It excludes the nonexistent lower channel of each s shell.
    Its Euclidean normalization gives Fisher analyses a convention-free null
    direction.

    :see: :class:`~.test_matrixel.TestMatrixElementPhaseGaugeDirection`

    Parameters
    ----------
    radial : RadialSpec
        Radial parameter carrier.
    me_params : MatrixElementParams
        Matrix-element parameter carrier.
    mean_free_path_ang : Float[Array, ""]
        Scalar intensity mean free path.

    Returns
    -------
    direction : Float[Array, "n_theta"]
        Unit packed phase-gauge tangent.

    Notes
    -----
    Pack a unit common phase displacement and subtract the base vector.
    """
    base: Float[Array, " n_theta"] = pack_matrixel_params(
        radial,
        me_params,
        mean_free_path_ang,
    )[0]
    shifted_angles: Float[Array, " n_valid_phase"] = (
        me_params.phase_shift_angles_shell + 1.0
    )
    shifted_params: MatrixElementParams = eqx.tree_at(
        lambda item: item.phase_shift_angles_shell,
        me_params,
        shifted_angles,
    )
    displaced: Float[Array, " n_theta"] = pack_matrixel_params(
        radial,
        shifted_params,
        mean_free_path_ang,
    )[0]
    tangent: Float[Array, " n_theta"] = displaced - base
    direction: Float[Array, " n_theta"] = tangent / jnp.linalg.norm(tangent)
    return direction


@jaxtyped(typechecker=beartype)
def radial_coefficient_scale_gauge_directions(
    radial: RadialSpec,
    me_params: MatrixElementParams,
    mean_free_path_ang: Float[Array, ""],
) -> Float[Array, "n_gauge n_theta"]:
    """Build normalized radial coefficient-scale gauge tangents.

    Return one tangent for every normalized Slater contraction shell.

    Multiplying all coefficients in one shell by a positive common scale does
    not change its normalized radial wavefunction.  Hydrogenic, grid, and
    fixed modes expose no contraction coefficient coordinate.

    :see: :class:`~.test_matrixel.TestRadialCoefficientScaleGaugeDirections`

    Parameters
    ----------
    radial : RadialSpec
        Radial parameter carrier.
    me_params : MatrixElementParams
        Matrix-element parameter carrier.
    mean_free_path_ang : Float[Array, ""]
        Scalar intensity mean free path.

    Returns
    -------
    directions : Float[Array, "n_gauge n_theta"]
        Unit packed coefficient-scale tangents.

    Notes
    -----
    Differentiate the finite common rescaling in each shell coordinate block.
    """
    base: Float[Array, " n_theta"] = pack_matrixel_params(
        radial,
        me_params,
        mean_free_path_ang,
    )[0]
    if radial.mode != "slater":
        directions: Float[Array, "n_gauge n_theta"] = jnp.zeros(
            (0, base.shape[0]),
            dtype=jnp.float64,
        )
        return directions
    tangents: list[Float[Array, " n_theta"]] = []
    shell: int
    for shell in range(radial.coefficients_shell.shape[0]):
        displaced_coefficients: Float[Array, "n_shell n_contraction"] = (
            radial.coefficients_shell.at[shell].add(
                radial.coefficients_shell[shell]
            )
        )
        displaced_radial: RadialSpec = eqx.tree_at(
            lambda item: item.coefficients_shell,
            radial,
            displaced_coefficients,
        )
        displaced: Float[Array, " n_theta"] = pack_matrixel_params(
            displaced_radial,
            me_params,
            mean_free_path_ang,
        )[0]
        tangent: Float[Array, " n_theta"] = displaced - base
        tangents.append(tangent / jnp.linalg.norm(tangent))
    directions = jnp.stack(tangents)
    return directions  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def band_group_weight_sensitivity(  # noqa: DOC105, DOC502
    flat_params: Float[Array, " n_theta"],
    rebuild: Callable[
        [Float[Array, " n_theta"], DiagonalizedBands, ExperimentGeometry],
        Complex[Array, "n_k n_bands n_spin"],
    ],
    bands: DiagonalizedBands,
    experiment: ExperimentGeometry,
    band_groups: tuple[tuple[int, ...], ...],
) -> tuple[
    Float[Array, "n_k n_group"],
    Float[Array, "n_theta n_k n_group"],
]:
    """Compute complete isolated band-group weights and their Jacobian.

    Apply ``jacfwd`` only after the static physical group validation.

    The rebuild callback returns polarization-contracted outgoing-spin band
    amplitudes.  The helper sums spin modulus squares and then sums all members
    of each complete group.  It assigns no spectral, exposure, background, or
    detector-count interpretation.

    :see: :class:`~.test_matrixel.TestBandGroupWeightSensitivity`

    Parameters
    ----------
    flat_params : Float[Array, "n_theta"]
        Real packed parameter vector.
    rebuild : Callable
        Callback from parameters, bands, and experiment to ``[K,B,S]``
        complex amplitudes.
    bands : DiagonalizedBands
        Eigensystem whose energies define complete isolated groups.
    experiment : ExperimentGeometry
        Experiment carrier passed unchanged to ``rebuild``.
    band_groups : tuple[tuple[int, ...], ...]
        Nonoverlapping static complete band groups.

    Returns
    -------
    band_group_weights : Float[Array, "n_k n_group"]
        Unresolved-spin complete-group matrix-element weights.
    weight_jacobian : Float[Array, "n_theta n_k n_group"]
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
        candidate: Float[Array, " n_theta"],
    ) -> Float[Array, "n_k n_group"]:
        """Return unresolved-spin weights summed over static band groups."""
        spin_amplitudes: Complex[Array, "n_k n_bands n_spin"] = rebuild(
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
        band_weights: Float[Array, "n_k n_bands"] = matrix_element_intensity(
            spin_amplitudes
        )
        weights: Float[Array, "n_k n_group"] = jnp.stack(
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

    band_group_weights: Float[Array, "n_k n_group"] = group_weights(
        flat_params
    )
    output_first_jacobian: Float[Array, "n_k n_group n_theta"] = jax.jacfwd(
        group_weights
    )(flat_params)
    weight_jacobian: Float[Array, "n_theta n_k n_group"] = jnp.moveaxis(
        output_first_jacobian,
        -1,
        0,
    )
    result: tuple[
        Float[Array, "n_k n_group"],
        Float[Array, "n_theta n_k n_group"],
    ] = (band_group_weights, weight_jacobian)
    return result


@jaxtyped(typechecker=beartype)
def log_band_group_weight_sensitivity(
    band_group_weights: Float[Array, " ..."],
    weight_jacobian: Float[Array, "n_theta ..."],
    min_band_group_weight: float,
) -> tuple[Float[Array, "n_theta ..."], Bool[Array, " ..."]]:
    """Convert positive group-weight derivatives to logarithmic derivatives.

    Mark dark or sub-floor weights invalid without dividing by them.

    The helper returns a zero derivative sentinel outside its positive domain.
    Consumers must use the validity mask rather than interpreting that sentinel
    as physical logarithmic information.

    :see: :class:`~.test_matrixel.TestLogBandGroupWeightSensitivity`

    Parameters
    ----------
    band_group_weights : Float[Array, "..."]
        Complete-group matrix-element weights.
    weight_jacobian : Float[Array, "n_theta ..."]
        Derivatives of those weights.
    min_band_group_weight : float
        Static strictly positive validity floor.

    Returns
    -------
    log_weight_jacobian : Float[Array, "n_theta ..."]
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
    safe_weights: Float[Array, " ..."] = jnp.where(
        valid,
        band_group_weights,
        1.0,
    )
    log_weight_jacobian: Float[Array, "n_theta ..."] = jnp.where(
        valid[None, ...],
        weight_jacobian / safe_weights[None, ...],
        0.0,
    )
    result: tuple[
        Float[Array, "n_theta ..."],
        Bool[Array, " ..."],
    ] = (log_weight_jacobian, valid)
    return result


@jaxtyped(typechecker=beartype)
def resolve_orbital_positions_cart(
    bands: DiagonalizedBands,
) -> Float[Array, "n_orb 3"]:
    """Resolve orbital centres in Cartesian Angstrom coordinates.

    Preserve the basis-position gauge at the structural boundary.

    Explicit fractional Wannier centres remain authoritative.  Otherwise,
    ``basis.atom_indices`` gathers fractional atom positions.  One final
    matrix product maps the selected fractional array through the lattice.

    :see: :class:`~.test_matrixel.TestResolveOrbitalPositionsCart`

    Parameters
    ----------
    bands : DiagonalizedBands
        Eigensystem with geometry, basis, and optional orbital centres.

    Returns
    -------
    positions_cart : Float[Array, "n_orb 3"]
        Orbital centres in Cartesian Angstrom coordinates.

    Notes
    -----
    Select one fractional source before applying the lattice multiplication.
    """
    positions_fractional: Float[Array, "n_orb 3"]
    if bands.orbital_positions is not None:
        positions_fractional = bands.orbital_positions
    else:
        atom_indices: Array = jnp.asarray(
            bands.basis.atom_indices,
            dtype=jnp.int32,
        )
        positions_fractional = bands.geometry.positions[atom_indices]
    positions_cart: Float[Array, "n_orb 3"] = (
        positions_fractional @ bands.geometry.lattice
    )
    return positions_cart


@jaxtyped(typechecker=beartype)
def real_spherical_harmonics_cartesian_all(  # noqa: DOC503
    direction_cart: Float[Array, "... 3"],
    l_max: int,
) -> Float[Array, "... n_y"]:
    """Evaluate all real spherical harmonics from Cartesian directions.

    Avoid the azimuthal coordinate singularity at normal emission.

    The degree-then-order index is ``l*l + l + m``.  The implementation uses
    homogeneous Cartesian polynomials and therefore never reconstructs an
    azimuth at the poles.

    :see: :class:`~.test_matrixel.TestRealSphericalHarmonicsCartesianAll`

    Parameters
    ----------
    direction_cart : Float[Array, "... 3"]
        Nonzero Cartesian direction vectors.
    l_max : int
        Static maximum degree in the inclusive range zero through five.

    Returns
    -------
    harmonics : Float[Array, "... n_y"]
        Normalized real harmonics with ``n_y = (l_max + 1)**2``.

    Raises
    ------
    ValueError
        If the static degree or trailing Cartesian axis is invalid.
    EquinoxRuntimeError
        If a direction is non-finite or zero.

    Notes
    -----
    Normalize each vector and evaluate finite Cartesian recurrences.
    """
    if type(l_max) is not int or not 0 <= l_max <= L_MAX + 1:
        message: str = "l_max must be an integer in [0, 5]"
        raise ValueError(message)
    if (
        direction_cart.ndim < 1
        or direction_cart.shape[-1] != CARTESIAN_COMPONENTS
    ):
        message = "direction_cart must have a trailing axis of length 3"
        raise ValueError(message)
    directions: Float[Array, "... 3"] = jnp.asarray(
        direction_cart,
        dtype=jnp.float64,
    )
    norms: Float[Array, " ..."] = jnp.linalg.norm(directions, axis=-1)
    directions = eqx.error_if(
        directions,
        ~jnp.all(jnp.isfinite(directions)) | jnp.any(norms <= 0.0),
        "Cartesian harmonic directions must be finite and nonzero",
    )
    safe_norms: Float[Array, " ..."] = jnp.where(norms > 0.0, norms, 1.0)
    direction_unit: Float[Array, "... 3"] = directions / safe_norms[..., None]
    values: list[Float[Array, " ..."]] = []
    degree: int
    order: int
    for degree in range(l_max + 1):
        for order in range(-degree, degree + 1):
            values.append(
                _solid_harmonic_component(direction_unit, degree, order)
            )
    harmonics: Float[Array, "... n_y"] = jnp.stack(values, axis=-1)
    return harmonics


@jaxtyped(typechecker=beartype)
def orbital_transition_channels(  # noqa: DOC503
    k_i_cart: Float[Array, "n_k 3"],
    k_f_cart: Float[Array, "n_k 3"],
    positions_cart: Float[Array, "n_orb 3"],
    depths: Float[Array, " n_orb"],
    bvals: Complex[Array, "n_k n_orb 2"],
    me_params: MatrixElementParams,
    mean_free_path_ang: Float[Array, ""],
    basis: OrbitalBasis,
) -> Complex[Array, "n_k n_spin n_orb_per_spin 3"]:
    r"""Assemble coherent orbital transition channels.

    Keep every orbital factor at amplitude level.

    Position, attenuation, radial, atomic-phase, angular, and shell-scale
    factors remain complex amplitudes.  The function performs no modulus
    square and leaves outgoing spin as an explicit axis.

    :see: :class:`~.test_matrixel.TestOrbitalTransitionChannels`

    Parameters
    ----------
    k_i_cart : Float[Array, "n_k 3"]
        Initial crystal momenta in sample Cartesian inverse Angstrom.
    k_f_cart : Float[Array, "n_k 3"]
        Explicit vacuum final momenta in the same frame and units.
    positions_cart : Float[Array, "n_orb 3"]
        Cartesian orbital centres in Angstrom.
    depths : Float[Array, "n_orb"]
        Orbital depths below the surface in Angstrom.
    bvals : Complex[Array, "n_k n_orb 2"]
        Radial channels in static order ``(l-1, l+1)``.
    me_params : MatrixElementParams
        Shell-shared amplitude scales and channel phase angles.
    mean_free_path_ang : Float[Array, ""]
        Positive photoelectron intensity mean free path in Angstrom.
    basis : OrbitalBasis
        Static real-orbital and spin layout.

    Returns
    -------
    transition_channels : Complex[Array, "n_k n_spin n_orb_per_spin 3"]
        Real dipole channels in order ``(y,z,x)``.

    Raises
    ------
    ValueError
        If static carriers or array axes disagree.
    EquinoxRuntimeError
        If traced physical values are invalid.

    Notes
    -----
    Derive a dense orbital branch view from compact physical phase coordinates.
    Contract both radial branches with one tensor sum.
    """
    n_spin: int
    n_spatial: int
    n_spin, n_spatial = _spin_layout(basis)
    n_orbitals: int = len(basis.n)
    n_kpoints: int = k_i_cart.shape[0]
    if _basis_key(me_params.basis) != _basis_key(basis):
        message: str = "matrix-element parameters and basis must agree"
        raise ValueError(message)
    if len(me_params.radial_shell_index) != n_orbitals:
        message = "matrix-element shell map must match the orbital basis"
        raise ValueError(message)
    if (
        k_i_cart.ndim != MATRIX_NDIM
        or k_f_cart.shape != k_i_cart.shape
        or k_i_cart.shape[-1] != CARTESIAN_COMPONENTS
        or positions_cart.shape != (n_orbitals, CARTESIAN_COMPONENTS)
        or depths.shape != (n_orbitals,)
        or bvals.shape != (n_kpoints, n_orbitals, 2)
        or mean_free_path_ang.ndim != 0
    ):
        message = "matrix-element array axes are inconsistent"
        raise ValueError(message)
    if (
        n_spin == ISPIN2_BLOCKS
        and me_params.radial_shell_index[:n_spatial]
        != me_params.radial_shell_index[n_spatial:]
    ):
        message = "spin blocks must share paired radial shells"
        raise ValueError(message)

    initial_momentum: Float[Array, "n_k 3"] = jnp.asarray(
        k_i_cart,
        dtype=jnp.float64,
    )
    final_momentum: Float[Array, "n_k 3"] = jnp.asarray(
        k_f_cart,
        dtype=jnp.float64,
    )
    positions: Float[Array, "n_orb 3"] = jnp.asarray(
        positions_cart,
        dtype=jnp.float64,
    )
    depth_values: Float[Array, " n_orb"] = jnp.asarray(
        depths,
        dtype=jnp.float64,
    )
    radial_values: Complex[Array, "n_k n_orb 2"] = jnp.asarray(
        bvals,
        dtype=jnp.complex128,
    )
    mean_free_path: Float[Array, ""] = jnp.asarray(
        mean_free_path_ang,
        dtype=jnp.float64,
    )
    checked_arrays: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    checked_arrays = eqx.error_if(
        checked_arrays,
        ~jnp.all(jnp.isfinite(initial_momentum))
        | ~jnp.all(jnp.isfinite(final_momentum))
        | ~jnp.all(jnp.isfinite(positions))
        | ~jnp.all(jnp.isfinite(depth_values))
        | ~jnp.all(jnp.isfinite(radial_values))
        | ~jnp.isfinite(mean_free_path)
        | (mean_free_path <= 0.0)
        | jnp.any(depth_values < -EPS),
        "matrix-element inputs must be finite with physical depth and mfp",
    )
    initial_momentum = initial_momentum + checked_arrays
    depth_nonnegative: Float[Array, " n_orb"] = jnp.maximum(depth_values, 0.0)
    shell_indices: Array = jnp.asarray(
        me_params.radial_shell_index,
        dtype=jnp.int32,
    )
    sigma_shell: Float[Array, " n_shell"] = eqx.error_if(
        me_params.sigma_shell,
        ~jnp.all(jnp.isfinite(me_params.sigma_shell)),
        "matrix-element shell scales must be finite",
    )
    compact_phase_angles: Float[Array, " n_valid_phase"] = eqx.error_if(
        me_params.phase_shift_angles_shell,
        ~jnp.all(jnp.isfinite(me_params.phase_shift_angles_shell)),
        "matrix-element phase angles must be finite",
    )
    phase_indices: Array = jnp.asarray(
        _orbital_phase_indices(me_params),
        dtype=jnp.int32,
    )
    phase_angles: Float[Array, "n_valid_phase_plus_zero"] = jnp.concatenate(
        (
            compact_phase_angles,
            jnp.zeros((1,), dtype=jnp.float64),
        )
    )
    orbital_scales: Float[Array, " n_orb"] = sigma_shell[shell_indices]
    channel_phases: Complex[Array, "n_orb 2"] = jnp.exp(
        1j * phase_angles[phase_indices]
    )
    attenuation: Float[Array, " n_orb"] = jnp.exp(
        -depth_nonnegative / (2.0 * mean_free_path)
    )
    momentum_difference: Float[Array, "n_k 3"] = (
        initial_momentum - final_momentum
    )
    position_phase: Complex[Array, "n_k n_orb"] = jnp.exp(
        1j * jnp.einsum("kd,od->ko", momentum_difference, positions)
    )
    harmonics: Float[Array, "n_k 36"] = real_spherical_harmonics_cartesian_all(
        final_momentum,
        L_MAX + 1,
    )
    coupling_coefficients: Float[Array, "n_orb 2 3 36"]
    channel_valid: Float[Array, "n_orb 2 3 36"]
    coupling_coefficients, channel_valid = channel_tables(basis)
    angular_channels: Complex[Array, "n_k n_orb 3"] = jnp.einsum(
        "koc,oc,ocdy,ky->kod",
        radial_values,
        channel_phases,
        coupling_coefficients * channel_valid,
        harmonics,
    )
    prefactor: Complex[Array, "n_k n_orb"] = (
        orbital_scales[None, :] * attenuation[None, :] * position_phase
    )
    flat_channels: Complex[Array, "n_k n_orb 3"] = (
        prefactor[..., None] * angular_channels
    )
    transition_channels: Complex[Array, "n_k n_spin n_orb_per_spin 3"] = (
        flat_channels.reshape(
            n_kpoints,
            n_spin,
            n_spatial,
            CARTESIAN_COMPONENTS,
        )
    )
    return transition_channels


@jaxtyped(typechecker=beartype)
def contract_polarization(
    transition_channels: Complex[Array, "... 3"],
    polarization_sample_cart: Complex[Array, " 3"],
) -> Complex[Array, " ..."]:
    """Compute the sample-frame polarization contraction.

    Preserve generic complex optical phases through the late linear step.

    :see: :class:`~.test_matrixel.TestContractPolarization`

    Parameters
    ----------
    transition_channels : Complex[Array, "... 3"]
        Real dipole channels in order ``(y,z,x)``.
    polarization_sample_cart : Complex[Array, "3"]
        Cartesian complex polarization in order ``(x,y,z)``.

    Returns
    -------
    polarized_transition : Complex[Array, "..."]
        Complex amplitude with the channel axis contracted once.

    Notes
    -----
    Permute Cartesian components into real-harmonic order before summing.
    """
    polarization: Complex[Array, " 3"] = jnp.asarray(
        polarization_sample_cart,
        dtype=jnp.complex128,
    )
    polarization_real: Complex[Array, " 3"] = polarization_cart_to_real(
        polarization
    )
    polarized_transition: Complex[Array, " ..."] = jnp.sum(
        transition_channels * polarization_real,
        axis=-1,
    )
    return polarized_transition


@jaxtyped(typechecker=beartype)
def contract_experiment_polarization(
    transition_channels: Complex[Array, "... 3"],
    experiment: ExperimentGeometry,
) -> Complex[Array, " ..."]:
    """Rotate laboratory polarization to the sample and contract it late.

    Keep the physical beam fixed while the sample azimuth changes.

    :see: :class:`~.test_matrixel.TestContractExperimentPolarization`

    Parameters
    ----------
    transition_channels : Complex[Array, "... 3"]
        Real dipole channels in order ``(y,z,x)``.
    experiment : ExperimentGeometry
        Experiment whose stored polarization is in the laboratory frame.

    Returns
    -------
    polarized_transition : Complex[Array, "..."]
        Complex sample-frame polarization contraction.

    Notes
    -----
    Apply the inverse sample orientation exactly once before contraction.
    """
    sample_orientation: Float[Array, "3 3"] = sample_azimuth_rotation(
        experiment.sample_azimuth
    )
    polarization_sample: Complex[Array, " 3"] = lab_polarization_to_sample(
        experiment.polarization,
        sample_orientation,
    )
    polarized_transition: Complex[Array, " ..."] = contract_polarization(
        transition_channels,
        polarization_sample,
    )
    return polarized_transition


@jaxtyped(typechecker=beartype)
def transition_source(
    transition_row: Complex[Array, "... n_spin n_orb_per_spin"],
) -> Complex[Array, "... n_spin n_orb"]:
    """Build conjugated outgoing-spin rows as full source kets.

    Retain separate sources for each outgoing-spin measurement channel.

    :see: :class:`~.test_matrixel.TestTransitionSource`

    Parameters
    ----------
    transition_row : Complex[Array, "... n_spin n_orb_per_spin"]
        Polarization-contracted outgoing-spin orbital rows.

    Returns
    -------
    source : Complex[Array, "... n_spin n_orb"]
        Conjugated source kets, with exact zero in the opposite spin block.

    Raises
    ------
    ValueError
        If the outgoing-spin axis is not one or two.

    Notes
    -----
    Multiply conjugated rows by a spin identity before flattening the blocks.
    """
    n_spin: int = transition_row.shape[-2]
    if n_spin not in (1, ISPIN2_BLOCKS):
        message: str = "transition rows must contain one or two spin channels"
        raise ValueError(message)
    n_spatial: int = transition_row.shape[-1]
    spin_identity: Float[Array, "n_spin n_spin"] = jnp.eye(
        n_spin,
        dtype=jnp.float64,
    )
    blocked_source: Complex[Array, "... n_spin n_spin n_orb_per_spin"] = (
        jnp.einsum(
            "...sa,st->...sta",
            jnp.conj(transition_row),
            spin_identity,
        )
    )
    source: Complex[Array, "... n_spin n_orb"] = blocked_source.reshape(
        transition_row.shape[:-2] + (n_spin, n_spin * n_spatial)
    )
    return source


@jaxtyped(typechecker=beartype)
def project_band_channels(
    transition_channels: Complex[Array, "n_k n_spin n_orb_per_spin 3"],
    eigenvectors: Complex[Array, "n_k n_bands n_orb"],
) -> Complex[Array, "n_k n_bands n_spin 3"]:
    """Compute band channels without conjugating orbital coefficients.

    Follow the stored ket-coefficient convention for generic complex bands.

    :see: :class:`~.test_matrixel.TestProjectBandChannels`

    Parameters
    ----------
    transition_channels : Complex[Array, "n_k n_spin n_orb_per_spin 3"]
        Orbital-space transition channels.
    eigenvectors : Complex[Array, "n_k n_bands n_orb"]
        Basis-position-gauge band coefficients.

    Returns
    -------
    band_channels : Complex[Array, "n_k n_bands n_spin 3"]
        Band amplitudes with outgoing spin retained.

    Raises
    ------
    ValueError
        If the k-point or orbital axes disagree.

    Notes
    -----
    Reshape coefficients into spin blocks and apply one direct complex sum.
    """
    n_kpoints: int = transition_channels.shape[0]
    n_spin: int = transition_channels.shape[1]
    n_spatial: int = transition_channels.shape[2]
    if n_spin not in (1, ISPIN2_BLOCKS):
        message: str = "transition channels must contain one or two spins"
        raise ValueError(message)
    if (
        eigenvectors.ndim != CARTESIAN_COMPONENTS
        or eigenvectors.shape[0] != n_kpoints
        or eigenvectors.shape[2] != n_spin * n_spatial
    ):
        message = "eigenvector axes must match transition channels"
        raise ValueError(message)
    eigenvector_blocks: Complex[Array, "n_k n_bands n_spin n_orb_per_spin"] = (
        eigenvectors.reshape(
            eigenvectors.shape[0],
            eigenvectors.shape[1],
            n_spin,
            n_spatial,
        )
    )
    band_channels: Complex[Array, "n_k n_bands n_spin 3"] = jnp.einsum(
        "ksad,kbsa->kbsd",
        transition_channels,
        eigenvector_blocks,
    )
    return band_channels


@jaxtyped(typechecker=beartype)
def matrix_element_intensity(
    spin_amplitude: Complex[Array, "... n_spin"],
) -> Float[Array, " ..."]:
    """Sum outgoing-spin modulus squares exactly once.

    Preserve interference within spins and exclude interference across spins.

    :see: :class:`~.test_matrixel.TestMatrixElementIntensity`

    Parameters
    ----------
    spin_amplitude : Complex[Array, "... n_spin"]
        Polarization-contracted amplitude for each outgoing spin.

    Returns
    -------
    intensity : Float[Array, "..."]
        Incoherent unresolved-spin intensity.

    Notes
    -----
    Form one modulus square and reduce only the final outgoing-spin axis.
    """
    intensity: Float[Array, " ..."] = jnp.sum(
        jnp.real(jnp.conj(spin_amplitude) * spin_amplitude),
        axis=-1,
    )
    return intensity


@jaxtyped(typechecker=beartype)
def assemble_orbital_transition_channels(  # noqa: DOC503
    bands: DiagonalizedBands,
    radial: RadialSpec,
    me_params: MatrixElementParams,
    quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    experiment: ExperimentGeometry,
    k_f_cart: Float[Array, "n_k 3"],
    emission_valid: Bool[Array, " n_k"],
) -> Complex[Array, "n_k n_spin n_orb_per_spin 3"]:
    """Assemble the validated orbital transition tensor.

    Bind carrier data to the scalar-energy orbital transition primitive.

    The function consumes explicit valid vacuum momentum in the sample frame.
    It accepts only the registered zero in-plane reciprocal shift and never
    reads ``experiment.inner_potential_ev``.

    :see: :class:`~.test_matrixel.TestAssembleOrbitalTransitionChannels`

    Parameters
    ----------
    bands : DiagonalizedBands
        Eigensystem, geometry, basis, orbital centres, and optional depths.
    radial : RadialSpec
        Shell-shared radial wavefunctions.
    me_params : MatrixElementParams
        Shell scales and channel phase angles.
    quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final-state model.
    experiment : ExperimentGeometry
        Supplies the traced intensity mean free path.
    k_f_cart : Float[Array, "n_k 3"]
        Explicit vacuum final momentum in sample inverse Angstrom.
    emission_valid : Bool[Array, "n_k"]
        Explicit Plan-03 emission validity mask.

    Returns
    -------
    transition_channels : Complex[Array, "n_k n_spin n_orb_per_spin 3"]
        Coherent orbital transition channels in real order ``(y,z,x)``.

    Raises
    ------
    ValueError
        If static carriers or array axes disagree.
    EquinoxRuntimeError
        If momentum is invalid, zero, or outside the zero-umklapp seam.

    Notes
    -----
    Validate the vacuum seam before evaluating radial and angular channels.
    """
    if _basis_key(radial.basis) != _basis_key(bands.basis) or _basis_key(
        me_params.basis
    ) != _basis_key(bands.basis):
        message: str = "bands, radial, and matrix-element bases must agree"
        raise ValueError(message)
    if radial.radial_shell_index != me_params.radial_shell_index:
        message = "radial and matrix-element shell maps must agree"
        raise ValueError(message)
    if (
        k_f_cart.ndim != MATRIX_NDIM
        or k_f_cart.shape != bands.kpoints.shape
        or emission_valid.shape != (bands.kpoints.shape[0],)
    ):
        message = "vacuum momentum and validity axes must match band kpoints"
        raise ValueError(message)
    final_momentum: Float[Array, "n_k 3"] = jnp.asarray(
        k_f_cart,
        dtype=jnp.float64,
    )
    validity: Bool[Array, " n_k"] = jnp.asarray(
        emission_valid,
        dtype=jnp.bool_,
    )
    initial_momentum: Float[Array, "n_k 3"] = (
        bands.kpoints @ bands.geometry.reciprocal
    )
    final_norm: Float[Array, " n_k"] = jnp.linalg.norm(
        final_momentum,
        axis=-1,
    )
    final_momentum = eqx.error_if(
        final_momentum,
        ~jnp.all(jnp.isfinite(final_momentum))
        | ~jnp.all(validity)
        | jnp.any(final_norm <= 0.0)
        | jnp.any(
            jnp.abs(final_momentum[:, :2] - initial_momentum[:, :2])
            > G_PARALLEL_ATOL_INV_ANG
        ),
        (
            "matrix-element assembly requires valid nonzero vacuum momentum "
            "in the registered G_parallel=0 channel"
        ),
    )
    positions_cart: Float[Array, "n_orb 3"] = resolve_orbital_positions_cart(
        bands
    )
    n_orbitals: int = len(bands.basis.n)
    depths: Float[Array, " n_orb"] = (
        jnp.zeros((n_orbitals,), dtype=jnp.float64)
        if bands.depths is None
        else bands.depths
    )
    momentum_bohr_inv: Float[Array, " n_k"] = momentum_inv_ang_to_bohr_inv(
        final_norm
    )
    bvals: Complex[Array, "n_k n_orb 2"] = radial_bvals(
        radial,
        momentum_bohr_inv,
        quadrature,
        final_state,
    )
    transition_channels: Complex[Array, "n_k n_spin n_orb_per_spin 3"] = (
        orbital_transition_channels(
            initial_momentum,
            final_momentum,
            positions_cart,
            depths,
            bvals,
            me_params,
            experiment.mean_free_path_ang,
            bands.basis,
        )
    )
    return transition_channels


__all__: list[str] = [
    "assemble_orbital_transition_channels",
    "band_group_weight_sensitivity",
    "contract_experiment_polarization",
    "contract_polarization",
    "log_band_group_weight_sensitivity",
    "matrix_element_intensity",
    "matrix_element_phase_gauge_direction",
    "orbital_transition_channels",
    "pack_matrixel_params",
    "project_band_channels",
    "radial_coefficient_scale_gauge_directions",
    "real_spherical_harmonics_cartesian_all",
    "resolve_orbital_positions_cart",
    "transition_source",
    "unpack_matrixel_params",
]
