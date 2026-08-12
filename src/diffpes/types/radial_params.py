"""Define radial-wavefunction and matrix-element parameters.

Extended Summary
----------------
This module defines differentiable shell-shared radial parameters
and coherent matrix-element channel scales and phases.

Routine Listings
----------------
:class:`MatrixElementParams`
    Store shell-shared matrix-element scales and channel phases.
:class:`RadialSpec`
    Store shell-shared radial-wavefunction parameters.
:func:`make_matrix_element_params`
    Create validated shell-shared matrix-element parameters.
:func:`make_radial_spec`
    Create a validated shell-shared radial specification.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import (
    ARRAY_MATRIX_NDIM,
    CERTIFIED_R_MAX_BOHR,
    CERTIFIED_TAIL_ENVELOPE_ID,
    MAX_COEFFICIENT_CONDITION,
    MAX_DECAY_PARAMETER,
    MAX_EFFECTIVE_PRINCIPAL,
    MAX_HYDROGENIC_PRINCIPAL,
    MAX_MATRIXEL_L,
    MIN_COMPACT_GRID_POINTS,
    MIN_DECAY_PARAMETER,
    RADIAL_MODES,
)

from .orbital_basis import OrbitalBasis


def _shell_representatives(
    radial_shell_index: Tuple[int, ...],
) -> Tuple[int, ...]:
    """PRIVATE: Return the first orbital index assigned to every shell.

    Parameters
    ----------
    radial_shell_index : Tuple[int, ...]
        Orbital-to-shell map with contiguous shell identifiers.

    Returns
    -------
    representatives : Tuple[int, ...]
        First orbital index of each shell, ordered by shell identifier.

    Implementation Logic
    --------------------
    Derive the shell count from the maximum identifier plus one. Then
    apply ``tuple.index`` per shell identifier, which returns the first
    match. An empty map yields an empty tuple.
    """
    n_shells: int = max(radial_shell_index, default=-1) + 1
    representatives: Tuple[int, ...] = tuple(
        radial_shell_index.index(shell_index)
        for shell_index in range(n_shells)
    )
    return representatives


def _matrixel_phase_channel_keys(
    basis: "OrbitalBasis",
    radial_shell_index: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    """PRIVATE: Return the compact ``(shell, l_prime)`` phase-channel keys.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital quantum-number metadata for every orbital.
    radial_shell_index : Tuple[int, ...]
        Orbital-to-shell map with contiguous shell identifiers.

    Returns
    -------
    channel_keys : Tuple[Tuple[int, int], ...]
        Canonical ordered ``(shell_index, l_prime)`` keys for exactly
        the physical dipole channels.

    Implementation Logic
    --------------------
    Walk the shell representatives in shell order and read each shell
    angular momentum ``l``. Emit ``(shell, l - 1)`` only when ``l > 0``
    and always emit ``(shell, l + 1)``, following the dipole selection
    rule ``l_prime = l +- 1``.
    """
    keys: List[Tuple[int, int]] = []
    shell_index: int
    orbital_index: int
    for shell_index, orbital_index in enumerate(
        _shell_representatives(radial_shell_index)
    ):
        angular: int = basis.l[orbital_index]
        if angular > 0:
            keys.append((shell_index, angular - 1))
        keys.append((shell_index, angular + 1))
    channel_keys: Tuple[Tuple[int, int], ...] = tuple(keys)
    return channel_keys


def _default_n_star(principal: int) -> float:
    """PRIVATE: Return Slater's effective principal number.

    Parameters
    ----------
    principal : int
        Hydrogenic principal quantum number ``n``, at least 1.

    Returns
    -------
    value : float
        Dimensionless Slater effective principal number ``n*``.

    Notes
    -----
    Apply Slater's rules table ``(1, 2, 3, 3.7, 4, 4.2)`` indexed by
    ``n - 1``. Every ``n`` of 6 or more maps to the last entry, 4.2.
    """
    values: Tuple[float, ...] = (1.0, 2.0, 3.0, 3.7, 4.0, 4.2)
    value: float = values[min(principal, len(values)) - 1]
    return value


def _slater_norm_squared(
    zeta_row: Float64[Array, " n_contraction"],
    coefficient_row: Float64[Array, " n_contraction"],
    effective_principal: float,
) -> Float64[Array, ""]:
    """PRIVATE: Return the analytic squared norm of one contracted STO row.

    Parameters
    ----------
    zeta_row : Float64[Array, " n_contraction"]
        Slater exponents of one shell in inverse Bohr.
    coefficient_row : Float64[Array, " n_contraction"]
        Dimensionless contraction coefficients of the same shell.
    effective_principal : float
        Dimensionless Slater effective principal number ``n*``.

    Returns
    -------
    norm_squared : Float64[Array, ""]
        Dimensionless squared radial norm ``c^T S c`` of the contracted
        row, where ``S`` is the primitive overlap matrix.

    Implementation Logic
    --------------------
    Each primitive is the normalized Slater orbital
    ``N r^(n* - 1) exp(-zeta r)`` with normalization constant
    ``N = (2 zeta)^(n* + 1/2) / sqrt(Gamma(2 n* + 1))``. The closed-form
    radial overlap of two primitives is
    ``N_i N_j Gamma(2 n* + 1) / (zeta_i + zeta_j)^(2 n* + 1)``. Contract
    this overlap matrix with the coefficient row on both sides through
    ``einsum``.
    """
    gamma_value: Float64[Array, ""] = jnp.asarray(
        math.gamma(2.0 * effective_principal + 1.0),
        dtype=jnp.float64,
    )
    primitive_norms: Float64[Array, " n_contraction"] = (
        (2.0 * zeta_row) ** (effective_principal + 0.5)
    ) / jnp.sqrt(gamma_value)
    denominator: Float64[Array, "n_contraction n_contraction"] = (
        zeta_row[:, None] + zeta_row[None, :]
    ) ** (2.0 * effective_principal + 1.0)
    overlap: Float64[Array, "n_contraction n_contraction"] = (
        primitive_norms[:, None]
        * primitive_norms[None, :]
        * gamma_value
        / denominator
    )
    norm_squared: Float64[Array, ""] = jnp.einsum(
        "i,ij,j->",
        coefficient_row,
        overlap,
        coefficient_row,
    )
    return norm_squared


def _slater_coefficient_condition(
    zeta_row: Float64[Array, " n_contraction"],
    coefficient_row: Float64[Array, " n_contraction"],
    effective_principal: float,
) -> Float64[Array, ""]:
    """PRIVATE: Return the scale-invariant normalized-contraction condition.

    Parameters
    ----------
    zeta_row : Float64[Array, " n_contraction"]
        Slater exponents of one shell in inverse Bohr.
    coefficient_row : Float64[Array, " n_contraction"]
        Dimensionless contraction coefficients of the same shell.
    effective_principal : float
        Dimensionless Slater effective principal number ``n*``.

    Returns
    -------
    condition : Float64[Array, ""]
        Dimensionless tail condition ``sum(|c|) / sqrt(c^T S c)``.

    Notes
    -----
    The ratio measures signed-coefficient cancellation and is invariant
    under a common rescale of the coefficients. The factory bounds it by
    the certified maximum so that the tail envelope stays valid.
    """
    norm_squared: Float64[Array, ""] = _slater_norm_squared(
        zeta_row,
        coefficient_row,
        effective_principal,
    )
    condition: Float64[Array, ""] = jnp.sum(
        jnp.abs(coefficient_row)
    ) / jnp.sqrt(norm_squared)
    return condition


def _validate_radial_shell_structure(
    basis: OrbitalBasis,
    radial_shell_index: Tuple[int, ...],
) -> int:
    """PRIVATE: Validate one shell partition and return its shell count.

    Implementation Logic
    --------------------
    After the tuple and range checks, walk every orbital and record the
    ``(atom_index, n, l)`` signature of its shell in two dictionaries.
    Reject a shell that mixes signatures and a signature that is split
    across shells. The bijection makes rotational partners share one
    contraction row.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital quantum-number metadata for every orbital.
    radial_shell_index : Tuple[int, ...]
        Orbital-to-shell map with contiguous shell identifiers.

    Returns
    -------
    n_shells : int
        Number of radial shells, ``max(radial_shell_index) + 1``.

    Raises
    ------
    ValueError
        If ``radial_shell_index`` lacks one valid integer entry per
        orbital. Also if a basis orbital has ``l > 4``. If shell
        identifiers are not contiguous from zero. If the partition
        lacks a one-to-one map to ``(atom, n, l)`` groups. This is the
        static construction-time contract.
    """
    if type(radial_shell_index) is not tuple:
        message: str = "radial_shell_index must be a tuple"
        raise ValueError(message)
    n_orbitals: int = len(basis.n)
    if len(radial_shell_index) != n_orbitals:
        message = "radial_shell_index must have one entry per orbital"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 for index in radial_shell_index
    ):
        message = "radial_shell_index must contain non-negative integers"
        raise ValueError(message)
    if any(angular > MAX_MATRIXEL_L for angular in basis.l):
        message = "matrix-element radial bases require l <= 4"
        raise ValueError(message)
    n_shells: int = max(radial_shell_index, default=-1) + 1
    if set(radial_shell_index) != set(range(n_shells)):
        message = "radial_shell_index must use contiguous shell identifiers"
        raise ValueError(message)
    shell_quantum_numbers: Dict[int, Tuple[int, int, int]] = {}
    quantum_number_shells: Dict[Tuple[int, int, int], int] = {}
    orbital_index: int
    shell_index: int
    for orbital_index, shell_index in enumerate(radial_shell_index):
        signature: Tuple[int, int, int] = (
            basis.atom_indices[orbital_index],
            basis.n[orbital_index],
            basis.l[orbital_index],
        )
        previous_signature: Tuple[int, int, int] | None = (
            shell_quantum_numbers.get(shell_index)
        )
        if previous_signature is not None and previous_signature != signature:
            message = "one radial shell cannot mix site, n, or l"
            raise ValueError(message)
        previous_shell: int | None = quantum_number_shells.get(signature)
        if previous_shell is not None and previous_shell != shell_index:
            message = "one site, n, l shell cannot be split across parameters"
            raise ValueError(message)
        shell_quantum_numbers[shell_index] = signature
        quantum_number_shells[signature] = shell_index
    return n_shells


def _validate_radial_array_shapes(
    zeta_shell: Float64[Array, "n_shell n_contraction"],
    coefficients_shell: Float64[Array, "n_shell n_contraction"],
    effective_charge_shell: Float64[Array, " n_shell"],
    n_shells: int,
) -> None:
    """PRIVATE: Validate common shell and contraction axes.

    Parameters
    ----------
    zeta_shell : Float64[Array, "n_shell n_contraction"]
        Slater exponents in inverse Bohr, one row per shell.
    coefficients_shell : Float64[Array, "n_shell n_contraction"]
        Dimensionless contraction coefficients, one row per shell.
    effective_charge_shell : Float64[Array, " n_shell"]
        Hydrogenic effective charges in elementary-charge units.
    n_shells : int
        Shell count from the validated shell partition.

    Raises
    ------
    ValueError
        If either contraction array is not a matrix or their shapes
        differ. If the shell axis differs from ``n_shells`` or the
        contraction axis is empty. If ``effective_charge_shell`` lacks
        shape ``(n_shells,)``. This is the static construction-time
        contract.

    Notes
    -----
    Checks only static shape metadata here. Numerical content checks
    stay traced inside the factory.
    """
    if (
        zeta_shell.ndim != ARRAY_MATRIX_NDIM
        or coefficients_shell.ndim != ARRAY_MATRIX_NDIM
    ):
        message: str = "zeta_shell and coefficients_shell must be matrices"
        raise ValueError(message)
    if zeta_shell.shape != coefficients_shell.shape:
        message = "zeta_shell and coefficients_shell must have equal shape"
        raise ValueError(message)
    if zeta_shell.shape[0] != n_shells or zeta_shell.shape[1] < 1:
        message = "radial shell arrays must match the shell partition"
        raise ValueError(message)
    if effective_charge_shell.shape != (n_shells,):
        message = "effective_charge_shell must have one entry per shell"
        raise ValueError(message)


class RadialSpec(eqx.Module):
    """Store shell-shared radial-wavefunction parameters.

    Rotational partners share one contraction row. Static mode and shell
    metadata define the traced program, while active numerical rows remain
    differentiable.

    :see: :class:`~.test_radial_params.TestRadialSpec`

    Attributes
    ----------
    zeta_shell : Float64[Array, "n_shell n_contraction"]
        Slater exponents in inverse Bohr.
    coefficients_shell : Float64[Array, "n_shell n_contraction"]
        Real contraction coefficients.
    effective_charge_shell : Float64[Array, "n_shell"]
        Hydrogenic effective charges in elementary-charge units.
    r_grid : Optional[Float64[Array, "n_r"]]
        Uniform compact-support grid for ``mode="grid"``.
    grid_values_shell : Optional[Float64[Array, "n_shell n_r"]]
        Sampled compact-support radial rows.
    fixed_integrals_shell : Optional[Float64[Array, "n_shell 2"]]
        Real phase-free fixed radial integrals for the ``l-1`` and ``l+1``
        channels.
    radial_shell_index : Tuple[int, ...]
        Orbital-to-shell map (**static**).
    basis : OrbitalBasis
        Orbital metadata (**static**).
    mode : str
        Radial mode (**static**).
    n_star_shell : Tuple[float, ...]
        Slater effective principal numbers (**static**).
    tail_envelope_id : str
        Certified tail-envelope identity (**static**).

    Notes
    -----
    Evaluation normalizes every non-fixed shell. The factory normalizes fixed
    rows at construction and excludes radial phases.

    See Also
    --------
    make_radial_spec : Validated factory for this type.
    """

    zeta_shell: Float64[Array, "n_shell n_contraction"]
    coefficients_shell: Float64[Array, "n_shell n_contraction"]
    effective_charge_shell: Float64[Array, " n_shell"]
    r_grid: Optional[Float64[Array, " n_r"]]
    grid_values_shell: Optional[Float64[Array, "n_shell n_r"]]
    fixed_integrals_shell: Optional[Float64[Array, "n_shell 2"]]
    radial_shell_index: Tuple[int, ...] = eqx.field(static=True)
    basis: OrbitalBasis = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    n_star_shell: Tuple[float, ...] = eqx.field(static=True)
    tail_envelope_id: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static shell metadata and numerical array shapes."""
        n_shells: int = _validate_radial_shell_structure(
            self.basis,
            self.radial_shell_index,
        )
        _validate_radial_array_shapes(
            self.zeta_shell,
            self.coefficients_shell,
            self.effective_charge_shell,
            n_shells,
        )
        if self.mode not in RADIAL_MODES:
            message: str = f"mode must be one of {RADIAL_MODES}"
            raise ValueError(message)
        if len(self.n_star_shell) != n_shells:
            message = "n_star_shell must have one entry per shell"
            raise ValueError(message)
        if self.tail_envelope_id != CERTIFIED_TAIL_ENVELOPE_ID:
            message = "tail_envelope_id is not a certified radial envelope"
            raise ValueError(message)


class MatrixElementParams(eqx.Module):
    """Store shell-shared matrix-element scales and channel phases.

    The static shell map forbids independent magnetic-component parameters.
    Traced phase angles generate unit-modulus channel factors downstream.

    :see: :class:`~.test_radial_params.TestMatrixElementParams`

    Attributes
    ----------
    sigma_shell : Float64[Array, "n_shell"]
        Real shell amplitude scales.
    phase_shift_angles_shell : Float64[Array, " n_valid_phase"]
        Final-state phase angles for exactly the physical channels.
    phase_channel_keys : Tuple[Tuple[int, int], ...]
        Compact ``(radial_shell, l_prime)`` coordinate keys (**static**).
    radial_shell_index : Tuple[int, ...]
        Orbital-to-shell map (**static**).
    basis : OrbitalBasis
        Orbital metadata (**static**).

    See Also
    --------
    make_matrix_element_params : Validated factory for this type.
    """

    sigma_shell: Float64[Array, " n_shell"]
    phase_shift_angles_shell: Float64[Array, " n_valid_phase"]
    phase_channel_keys: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    radial_shell_index: Tuple[int, ...] = eqx.field(static=True)
    basis: OrbitalBasis = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate shell metadata and parameter axes."""
        n_shells: int = _validate_radial_shell_structure(
            self.basis,
            self.radial_shell_index,
        )
        if self.sigma_shell.shape != (n_shells,):
            message: str = "sigma_shell must have one entry per shell"
            raise ValueError(message)
        expected_keys: Tuple[Tuple[int, int], ...] = (
            _matrixel_phase_channel_keys(
                self.basis,
                self.radial_shell_index,
            )
        )
        if self.phase_channel_keys != expected_keys:
            message = (
                "phase_channel_keys must contain exactly the canonical "
                "physical shell channels"
            )
            raise ValueError(message)
        if self.phase_shift_angles_shell.shape != (len(expected_keys),):
            message = (
                "phase_shift_angles_shell must have one entry per valid "
                "phase channel"
            )
            raise ValueError(message)


@jaxtyped(typechecker=beartype)
def make_radial_spec(  # noqa: DOC105, DOC502, DOC503, PLR0912, PLR0913, PLR0915, PLR0917
    basis: OrbitalBasis,
    radial_shell_index: Tuple[int, ...],
    mode: str = "slater",
    zeta_shell: Optional[Float64[Array, "n_shell n_contraction"]] = None,
    coefficients_shell: Optional[
        Float64[Array, "n_shell n_contraction"]
    ] = None,
    effective_charge_shell: Optional[Float64[Array, " n_shell"]] = None,
    r_grid: Optional[Float64[Array, " n_r"]] = None,
    grid_values_shell: Optional[Float64[Array, "n_shell n_r"]] = None,
    fixed_integrals_shell: Optional[Float64[Array, "n_shell 2"]] = None,
    n_star_shell: Optional[Tuple[float, ...]] = None,
    tail_envelope_id: str = CERTIFIED_TAIL_ENVELOPE_ID,
) -> RadialSpec:
    """Create a validated shell-shared radial specification.

    The factory freezes one shell partition, validates its active mode, and
    retains only real phase-free fixed integrals. Runtime checks remain active
    under JIT for every traced physical parameter.

    :see: :class:`~.test_radial_params.TestMakeRadialSpec`

    Parameters
    ----------
    basis : OrbitalBasis
        Static orbital metadata.
    radial_shell_index : Tuple[int, ...]
        Static orbital-to-shell partition.
    mode : str, optional
        ``"slater"``, ``"hydrogenic"``, ``"grid"``, or ``"fixed"``.
    zeta_shell : Optional[Float64[Array, "n_shell n_contraction"]], optional
        Slater exponents in inverse Bohr.
    coefficients_shell : Optional[Float64[Array, "S C"]], optional
        Real Slater contraction coefficients.
    effective_charge_shell : Optional[Float64[Array, " n_shell"]], optional
        Hydrogenic effective charges.
    r_grid : Optional[Float64[Array, " n_r"]], optional
        Uniform compact-support grid in Bohr.
    grid_values_shell : Optional[Float64[Array, "n_shell n_r"]], optional
        Sampled radial rows on ``r_grid``.
    fixed_integrals_shell : Optional[Float64[Array, "n_shell 2"]], optional
        Real phase-free fixed channel integrals.
    n_star_shell : Optional[Tuple[float, ...]], optional
        Slater effective principal numbers.
    tail_envelope_id : str, optional
        Certified tail-envelope identity.

    Returns
    -------
    spec : RadialSpec
        Validated radial specification.

    Raises
    ------
    ValueError
        If static shell, mode, or active-array structure is invalid.
    EquinoxRuntimeError
        If traced parameters leave the certified domain or have zero norm.

    Notes
    -----
    The initial certified envelope requires active decay parameters in
    ``[0.5, 4]`` and compact grids ending no later than 120 Bohr.
    """
    if mode not in RADIAL_MODES:
        message: str = f"mode must be one of {RADIAL_MODES}"
        raise ValueError(message)
    n_shells: int = _validate_radial_shell_structure(
        basis,
        radial_shell_index,
    )
    representatives: Tuple[int, ...] = _shell_representatives(
        radial_shell_index,
    )
    resolved_n_star: Tuple[float, ...] = (
        tuple(_default_n_star(basis.n[index]) for index in representatives)
        if n_star_shell is None
        else n_star_shell
    )
    if len(resolved_n_star) != n_shells or any(
        type(value) not in (float, int)
        or not 1.0 <= float(value) <= MAX_EFFECTIVE_PRINCIPAL
        for value in resolved_n_star
    ):
        message = (
            "n_star_shell must contain values from 1 through 4.2, "
            "one per shell"
        )
        raise ValueError(message)
    if tail_envelope_id != CERTIFIED_TAIL_ENVELOPE_ID:
        message = "tail_envelope_id is not a certified radial envelope"
        raise ValueError(message)

    if zeta_shell is None:
        zeta_array: Float64[Array, "n_shell n_contraction"] = jnp.ones(
            (n_shells, 1),
            dtype=jnp.float64,
        )
    else:
        zeta_array = jnp.asarray(zeta_shell, dtype=jnp.float64)
    if coefficients_shell is None:
        coefficient_array: Float64[Array, "n_shell n_contraction"] = (
            jnp.ones_like(zeta_array)
        )
    else:
        coefficient_array = jnp.asarray(
            coefficients_shell,
            dtype=jnp.float64,
        )
    if effective_charge_shell is None:
        charge_array: Float64[Array, " n_shell"] = jnp.ones(
            (n_shells,),
            dtype=jnp.float64,
        )
    else:
        charge_array = jnp.asarray(
            effective_charge_shell,
            dtype=jnp.float64,
        )
    _validate_radial_array_shapes(
        zeta_array,
        coefficient_array,
        charge_array,
        n_shells,
    )

    grid_array: Optional[Float64[Array, " n_r"]] = None
    grid_value_array: Optional[Float64[Array, "n_shell n_r"]] = None
    fixed_array: Optional[Float64[Array, "n_shell 2"]] = None
    if mode == "grid":
        if r_grid is None or grid_values_shell is None:
            message = "grid mode requires r_grid and grid_values_shell"
            raise ValueError(message)
        if fixed_integrals_shell is not None:
            message = "grid mode does not accept fixed_integrals_shell"
            raise ValueError(message)
        grid_array = jnp.asarray(r_grid, dtype=jnp.float64)
        grid_value_array = jnp.asarray(
            grid_values_shell,
            dtype=jnp.float64,
        )
        if (
            grid_array.ndim != 1
            or grid_array.shape[0] < MIN_COMPACT_GRID_POINTS
            or grid_value_array.shape != (n_shells, grid_array.shape[0])
        ):
            message = "grid mode arrays have inconsistent shapes"
            raise ValueError(message)
        spacings: Float64[Array, " n_interval"] = jnp.diff(grid_array)
        grid_array = eqx.error_if(
            grid_array,
            ~jnp.all(jnp.isfinite(grid_array))
            | (grid_array[0] != 0.0)
            | (grid_array[-1] > CERTIFIED_R_MAX_BOHR)
            | ~jnp.all(spacings > 0.0)
            | ~jnp.allclose(spacings, spacings[0], rtol=1.0e-12, atol=0.0),
            (
                "grid mode requires a finite uniform grid from zero through "
                "at most 120 Bohr"
            ),
        )
        grid_value_array = eqx.error_if(
            grid_value_array,
            ~jnp.all(jnp.isfinite(grid_value_array))
            | ~jnp.all(grid_value_array[:, -1] == 0.0),
            "grid rows must be finite and exactly compact-supported",
        )
        grid_norms: Float64[Array, " n_shell"] = jnp.trapezoid(
            grid_value_array**2 * grid_array[None, :] ** 2,
            x=grid_array,
            axis=-1,
        )
        grid_value_array = eqx.error_if(
            grid_value_array,
            ~jnp.all(jnp.isfinite(grid_norms)) | jnp.any(grid_norms <= 0.0),
            "grid radial rows must have positive finite norm",
        )
        grid_value_array = grid_value_array / jnp.sqrt(grid_norms)[:, None]
    elif r_grid is not None or grid_values_shell is not None:
        message = "only grid mode accepts r_grid and grid_values_shell"
        raise ValueError(message)

    if mode == "fixed":
        if fixed_integrals_shell is None:
            message = "fixed mode requires fixed_integrals_shell"
            raise ValueError(message)
        fixed_array = jnp.asarray(
            fixed_integrals_shell,
            dtype=jnp.float64,
        )
        if fixed_array.shape != (n_shells, 2):
            message = "fixed_integrals_shell must have shape (n_shell, 2)"
            raise ValueError(message)
        fixed_norms: Float64[Array, " n_shell"] = jnp.linalg.norm(
            fixed_array,
            axis=-1,
        )
        fixed_array = eqx.error_if(
            fixed_array,
            ~jnp.all(jnp.isfinite(fixed_array))
            | ~jnp.all(jnp.isfinite(fixed_norms))
            | jnp.any(fixed_norms <= 0.0),
            "fixed integral rows must have positive finite norm",
        )
        fixed_array = fixed_array / fixed_norms[:, None]
    elif fixed_integrals_shell is not None:
        message = "only fixed mode accepts fixed_integrals_shell"
        raise ValueError(message)

    zeta_array = eqx.error_if(
        zeta_array,
        ~jnp.all(jnp.isfinite(zeta_array)),
        "zeta_shell must be finite",
    )
    coefficient_array = eqx.error_if(
        coefficient_array,
        ~jnp.all(jnp.isfinite(coefficient_array)),
        "coefficients_shell must be finite",
    )
    charge_array = eqx.error_if(
        charge_array,
        ~jnp.all(jnp.isfinite(charge_array)),
        "effective_charge_shell must be finite",
    )
    if mode == "slater":
        zeta_array = eqx.error_if(
            zeta_array,
            jnp.any(zeta_array < MIN_DECAY_PARAMETER)
            | jnp.any(zeta_array > MAX_DECAY_PARAMETER),
            "slater zeta_shell leaves the certified tail envelope",
        )
        shell_norms: List[Float64[Array, ""]] = []
        coefficient_conditions: List[Float64[Array, ""]] = []
        shell_index: int
        for shell_index in range(n_shells):
            effective_principal: float = resolved_n_star[shell_index]
            shell_norms.append(
                _slater_norm_squared(
                    zeta_array[shell_index],
                    coefficient_array[shell_index],
                    effective_principal,
                )
            )
            coefficient_conditions.append(
                _slater_coefficient_condition(
                    zeta_array[shell_index],
                    coefficient_array[shell_index],
                    effective_principal,
                )
            )
        contraction_norms: Float64[Array, " n_shell"] = jnp.stack(shell_norms)
        contraction_conditions: Float64[Array, " n_shell"] = jnp.stack(
            coefficient_conditions
        )
        coefficient_array = eqx.error_if(
            coefficient_array,
            ~jnp.all(jnp.isfinite(contraction_norms))
            | jnp.any(contraction_norms <= 0.0)
            | ~jnp.all(jnp.isfinite(contraction_conditions))
            | jnp.any(contraction_conditions > MAX_COEFFICIENT_CONDITION),
            (
                "slater contraction rows must have positive finite norm "
                "and coefficient condition at most 32"
            ),
        )
    if mode == "hydrogenic":
        if zeta_array.shape[1] != 1 or coefficient_array.shape[1] != 1:
            message = "hydrogenic mode has exactly one radial row per shell"
            raise ValueError(message)
        if any(
            basis.n[orbital_index] > MAX_HYDROGENIC_PRINCIPAL
            for orbital_index in representatives
        ):
            message = "hydrogenic mode is certified only through n=7"
            raise ValueError(message)
        principal_array: Float64[Array, " n_shell"] = jnp.asarray(
            tuple(basis.n[index] for index in representatives),
            dtype=jnp.float64,
        )
        charge_array = eqx.error_if(
            charge_array,
            jnp.any(charge_array / principal_array < MIN_DECAY_PARAMETER)
            | jnp.any(charge_array / principal_array > MAX_DECAY_PARAMETER),
            "hydrogenic effective charge leaves the certified tail envelope",
        )

    spec: RadialSpec = RadialSpec(
        zeta_shell=zeta_array,
        coefficients_shell=coefficient_array,
        effective_charge_shell=charge_array,
        r_grid=grid_array,
        grid_values_shell=grid_value_array,
        fixed_integrals_shell=fixed_array,
        radial_shell_index=radial_shell_index,
        basis=basis,
        mode=mode,
        n_star_shell=tuple(float(value) for value in resolved_n_star),
        tail_envelope_id=tail_envelope_id,
    )
    return spec


@jaxtyped(typechecker=beartype)
def make_matrix_element_params(  # noqa: DOC502, DOC503
    basis: OrbitalBasis,
    radial_shell_index: Tuple[int, ...],
    sigma_shell: Optional[Float64[Array, "n_shell"]] = None,
    phase_shift_angles_shell: Optional[Float64[Array, "n_phase"]] = None,
) -> MatrixElementParams:
    """Create validated shell-shared matrix-element parameters.

    The factory validates the rotational-shell partition and derives a
    compact static key for every physical final-state channel. Nonexistent
    channels never enter the carrier PyTree.

    :see: :class:`~.test_radial_params.TestMakeMatrixElementParams`

    Notes
    -----
    Validate shell sharing, then derive the compact physical-channel axis.

    Parameters
    ----------
    basis : OrbitalBasis
        Static orbital metadata.
    radial_shell_index : Tuple[int, ...]
        Static orbital-to-shell partition.
    sigma_shell : Optional[Float64[Array, "n_shell"]], optional
        Real shell scales, defaulting to one.
    phase_shift_angles_shell : Optional[Float64[Array, "n_phase"]], optional
        Real compact channel phase angles, defaulting to zero. Their static
        coordinates follow shell order and then increasing ``l_prime``.

    Returns
    -------
    params : MatrixElementParams
        Validated matrix-element parameter carrier.

    Raises
    ------
    ValueError
        If shell metadata or array shapes are invalid.
    EquinoxRuntimeError
        If traced values are non-finite.
    """
    n_shells: int = _validate_radial_shell_structure(
        basis,
        radial_shell_index,
    )
    sigma_array: Float64[Array, " n_shell"] = (
        jnp.ones((n_shells,), dtype=jnp.float64)
        if sigma_shell is None
        else jnp.asarray(sigma_shell, dtype=jnp.float64)
    )
    phase_channel_keys: Tuple[Tuple[int, int], ...] = (
        _matrixel_phase_channel_keys(
            basis,
            radial_shell_index,
        )
    )
    phase_array: Float64[Array, " n_valid_phase"] = (
        jnp.zeros((len(phase_channel_keys),), dtype=jnp.float64)
        if phase_shift_angles_shell is None
        else jnp.asarray(phase_shift_angles_shell, dtype=jnp.float64)
    )
    if sigma_array.shape != (n_shells,):
        message: str = "sigma_shell must have one entry per shell"
        raise ValueError(message)
    if phase_array.shape != (len(phase_channel_keys),):
        message = (
            "phase_shift_angles_shell must have one entry per valid "
            "phase channel"
        )
        raise ValueError(message)
    sigma_array = eqx.error_if(
        sigma_array,
        ~jnp.all(jnp.isfinite(sigma_array)),
        "sigma_shell must be finite",
    )
    phase_array = eqx.error_if(
        phase_array,
        ~jnp.all(jnp.isfinite(phase_array)),
        "phase_shift_angles_shell must be finite",
    )
    params: MatrixElementParams = MatrixElementParams(
        sigma_shell=sigma_array,
        phase_shift_angles_shell=phase_array,
        phase_channel_keys=phase_channel_keys,
        radial_shell_index=radial_shell_index,
        basis=basis,
    )
    return params


__all__: list[str] = [
    "MatrixElementParams",
    "RadialSpec",
    "make_matrix_element_params",
    "make_radial_spec",
]
