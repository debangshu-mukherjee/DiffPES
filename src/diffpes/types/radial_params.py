"""Define radial-wavefunction parameter structures.

Extended Summary
----------------
This module defines PyTree types for orbital basis metadata and Slater-type
radial wavefunction parameters used by the differentiable dipole
matrix element pipeline.

Routine Listings
----------------
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:class:`FinalStateSpec`
    Store a certified radial final-state selection.
:class:`MatrixElementParams`
    Store shell-shared matrix-element scales and channel phases.
:class:`RadialQuadratureSpec`
    Store one immutable certified radial-quadrature profile.
:class:`RadialSpec`
    Store shell-shared radial-wavefunction parameters.
:class:`SlaterKosterParams`
    Store differentiable Slater--Koster two-center integrals.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_final_state_spec`
    Create a validated radial final-state selection.
:func:`make_matrix_element_params`
    Create validated shell-shared matrix-element parameters.
:func:`make_radial_quadrature_spec`
    Select one immutable certified quadrature profile.
:func:`make_radial_spec`
    Create a validated shell-shared radial specification.
:func:`make_slater_koster_params`
    Create validated Slater--Koster two-center parameters.

Notes
-----
``OrbitalBasis`` contains only static auxiliary data. Atom assignments,
quantum numbers, spin channels, and labels define the traced program shape.
``RadialSpec`` wraps differentiable shell-shared radial parameters alongside
the static orbital basis. ``SlaterKosterParams`` separates differentiable
two-center values from their static material/channel identifiers.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple
from jaxtyping import Array, Float, Float64, jaxtyped

_ARRAY_MATRIX_NDIM: int = 2
_MIN_COMPACT_GRID_POINTS: int = 3
_RADIAL_MODES: Tuple[str, ...] = (
    "slater",
    "hydrogenic",
    "grid",
    "fixed",
)
_FINAL_STATE_MODES: Tuple[str, ...] = ("plane_wave", "coulomb")
_RADIAL_ACCELERATORS: Tuple[str, ...] = ("direct", "hermite")
_HERMITE_TABLE_POINTS: Tuple[int, ...] = (257, 513, 1025, 2049)
_CERTIFIED_RADIAL_PROFILES: Dict[
    str,
    Tuple[
        int,
        float,
        float,
        int,
        float,
        float,
        str,
        float,
        float,
        float,
    ],
] = {
    "gl1024-r120-k4-l9-v1": (
        1024,
        120.0,
        4.0,
        9,
        1.0e-10,
        1.0e-8,
        "analytic-exp-r120-or-compact-v1",
        32.0,
        0.5,
        4.0,
    ),
    "gl2048-r120-k4-l9-reference-v1": (
        2048,
        120.0,
        4.0,
        9,
        5.0e-11,
        5.0e-9,
        "analytic-exp-r120-or-compact-v1",
        32.0,
        0.5,
        4.0,
    ),
}
_CERTIFIED_TAIL_ENVELOPE_ID: str = "r120-zeta0p5-to4-v1"
_CERTIFIED_R_MAX_BOHR: float = 120.0
_MIN_DECAY_PARAMETER: float = 0.5
_MAX_DECAY_PARAMETER: float = 4.0
_MAX_COEFFICIENT_CONDITION: float = 32.0
_MAX_EFFECTIVE_PRINCIPAL: float = 4.2
_MAX_HYDROGENIC_PRINCIPAL: int = 7
_MAX_MATRIXEL_L: int = 4


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
    keys: list[Tuple[int, int]] = []
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


def _validate_orbital_basis_structure(
    atom_indices: Tuple[int, ...],
    n: Tuple[int, ...],
    l: Tuple[int, ...],  # noqa: E741
    m: Tuple[int, ...],
    spin: Tuple[int, ...],
    labels: Tuple[str, ...],
) -> None:
    """PRIVATE: Validate static orbital-basis metadata.

    Implementation Logic
    --------------------
    Use exact ``type`` comparisons to reject bools and NumPy integers.
    Check quantum-number consistency pairwise with
    ``zip(..., strict=True)`` over ``(n, l)`` and ``(l, m)``.

    Parameters
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row index for each orbital.
    n : Tuple[int, ...]
        Principal quantum numbers, one per orbital.
    l : Tuple[int, ...]
        Angular momentum quantum numbers, one per orbital.
    m : Tuple[int, ...]
        Magnetic quantum numbers, one per orbital.
    spin : Tuple[int, ...]
        Spin channels: empty for a spinless basis, otherwise one ``+1``
        or ``-1`` entry per orbital.
    labels : Tuple[str, ...]
        Human-readable orbital labels.

    Raises
    ------
    ValueError
        If a field has the wrong container type or length. If an atom,
        principal, or angular quantum number is invalid. If ``spin``
        has the wrong length or channel values. If a label is not a
        string. This is the static construction-time contract.
    """
    if any(
        type(values) is not tuple
        for values in (atom_indices, n, l, m, spin, labels)
    ):
        message: str = "all OrbitalBasis fields must be tuples"
        raise ValueError(message)
    n_orbitals: int = len(n)
    if not (
        len(atom_indices) == len(l) == len(m) == len(labels) == n_orbitals
    ):
        message: str = (
            "atom_indices, n, l, m, and labels must have the same length"
        )
        raise ValueError(message)
    if any(type(index) is not int or index < 0 for index in atom_indices):
        message = "atom_indices must contain non-negative integers"
        raise ValueError(message)
    if any(type(value) is not int or value < 1 for value in n):
        message = "n must contain integers of at least 1"
        raise ValueError(message)
    if any(
        type(angular) is not int or angular < 0 or angular >= principal
        for principal, angular in zip(n, l, strict=True)
    ):
        message = "l must contain integers satisfying 0 <= l < n"
        raise ValueError(message)
    if any(
        type(magnetic) is not int or abs(magnetic) > angular
        for angular, magnetic in zip(l, m, strict=True)
    ):
        message = "m must contain integers satisfying abs(m) <= l"
        raise ValueError(message)
    if spin and len(spin) != n_orbitals:
        message = "spin must be empty or have one entry per orbital"
        raise ValueError(message)
    if any(
        type(channel) is not int or channel not in (-1, 1) for channel in spin
    ):
        message = "spin entries must be +1 or -1"
        raise ValueError(message)
    if any(type(label) is not str for label in labels):
        message = "labels must contain strings"
        raise ValueError(message)


class OrbitalBasis(eqx.Module):
    """Store orbital quantum-number metadata in a JAX PyTree.

    This type describes the orbital basis for dipole matrix-element
    calculations for the differentiable Chinook pipeline. The quantum
    numbers (n, l, m) parameterize the radial wavefunctions (via
    Slater-type orbitals) and angular parts (spherical harmonics) that
    enter the photoemission matrix element.

    All fields contain static auxiliary data because quantum numbers control
    code paths. They determine recurrence depths in spherical Bessel functions
    and associated Legendre polynomials. They also index the Gaunt coefficient
    table. A quantum-number change alters the computational graph. JAX must
    therefore recompile after this change.


    :see: :class:`~.test_radial_params.TestOrbitalBasis`

    Attributes
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row index for each orbital. Each entry refers to a row of
        :attr:`~diffpes.types.CrystalGeometry.positions` (**static** -- a
        compile-time constant; changing it triggers retracing).
    n : Tuple[int, ...]
        Principal quantum numbers, one per orbital. Each value controls the
        radial node count and the power of *r*. The Slater form
        R_nl(r) ~ r^{n-1} exp(-zeta*r) uses static compile-time values;
        changing them triggers retracing.
    l : Tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral (**static** -- compile-time constants;
        changing them triggers retracing).
    m : Tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component (**static** -- compile-time constants; changing them
        triggers retracing).
    spin : Tuple[int, ...]
        Spin channel for each orbital. The empty tuple denotes a spinless
        basis; a spinor basis stores ``+1`` or ``-1`` for every orbital
        (**static** -- a compile-time constant; changing it triggers
        retracing).
    labels : Tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).
        Used for plotting and debugging (**static** -- compile-time constants;
        changing them triggers retracing).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    All fields are auxiliary data (no JAX array children) because
    changing any quantum number changes the computational graph and
    requires JIT recompilation. The children tuple is always empty.

    See Also
    --------
    RadialSpec : Wraps shell-shared radial parameters alongside this basis.
    make_orbital_basis : Factory function with length validation and
        default label generation.
    """

    atom_indices: Tuple[int, ...] = eqx.field(static=True)
    n: Tuple[int, ...] = eqx.field(static=True)
    l: Tuple[int, ...] = eqx.field(static=True)  # noqa: E741
    m: Tuple[int, ...] = eqx.field(static=True)
    spin: Tuple[int, ...] = eqx.field(static=True)
    labels: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the static orbital-basis invariants again."""
        _validate_orbital_basis_structure(
            self.atom_indices,
            self.n,
            self.l,
            self.m,
            self.spin,
            self.labels,
        )


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
    if any(angular > _MAX_MATRIXEL_L for angular in basis.l):
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
        zeta_shell.ndim != _ARRAY_MATRIX_NDIM
        or coefficients_shell.ndim != _ARRAY_MATRIX_NDIM
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
        if self.mode not in _RADIAL_MODES:
            message: str = f"mode must be one of {_RADIAL_MODES}"
            raise ValueError(message)
        if len(self.n_star_shell) != n_shells:
            message = "n_star_shell must have one entry per shell"
            raise ValueError(message)
        if self.tail_envelope_id != _CERTIFIED_TAIL_ENVELOPE_ID:
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


class RadialQuadratureSpec(eqx.Module):
    """Store one immutable certified radial-quadrature profile.

    Callers select a registered identity. They cannot self-assert numerical
    tolerances or enlarge its domain.

    :see: :class:`~.test_radial_params.TestRadialQuadratureSpec`

    Attributes
    ----------
    profile_id : str
        Registered profile identity (**static**).
    n_nodes : int
        Gauss--Legendre node count (**static**).
    r_max_bohr : float
        Certified radial cutoff in Bohr (**static**).
    k_max_bohr_inv : float
        Certified momentum limit in inverse Bohr (**static**).
    l_prime_max : int
        Certified final angular-momentum limit (**static**).
    value_rtol : float
        Registered value tolerance (**static**).
    gradient_rtol : float
        Registered derivative tolerance (**static**).
    tail_bound_method_id : str
        Registered tail-bound method (**static**).
    coefficient_condition_max : float
        Maximum certified normalized-contraction condition (**static**).
    min_decay_parameter : float
        Minimum certified exponential decay in inverse Bohr (**static**).
    max_decay_parameter : float
        Maximum certified exponential decay in inverse Bohr (**static**).
    """

    profile_id: str = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    r_max_bohr: float = eqx.field(static=True)
    k_max_bohr_inv: float = eqx.field(static=True)
    l_prime_max: int = eqx.field(static=True)
    value_rtol: float = eqx.field(static=True)
    gradient_rtol: float = eqx.field(static=True)
    tail_bound_method_id: str = eqx.field(static=True)
    coefficient_condition_max: float = eqx.field(static=True)
    min_decay_parameter: float = eqx.field(static=True)
    max_decay_parameter: float = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Require exact agreement with the selected certified profile."""
        expected: (
            Tuple[
                int,
                float,
                float,
                int,
                float,
                float,
                str,
                float,
                float,
                float,
            ]
            | None
        ) = _CERTIFIED_RADIAL_PROFILES.get(self.profile_id)
        actual: Tuple[
            int,
            float,
            float,
            int,
            float,
            float,
            str,
            float,
            float,
            float,
        ] = (
            self.n_nodes,
            self.r_max_bohr,
            self.k_max_bohr_inv,
            self.l_prime_max,
            self.value_rtol,
            self.gradient_rtol,
            self.tail_bound_method_id,
            self.coefficient_condition_max,
            self.min_decay_parameter,
            self.max_decay_parameter,
        )
        if expected is None or actual != expected:
            message: str = (
                "quadrature properties must match a certified profile"
            )
            raise ValueError(message)


class FinalStateSpec(eqx.Module):
    """Store a certified radial final-state selection.

    The numerical effective charge remains differentiable. Static mode and
    accelerator choices determine the compiled radial kernel.

    :see: :class:`~.test_radial_params.TestFinalStateSpec`

    Attributes
    ----------
    effective_charge : Float64[Array, ""]
        Coulomb effective charge in elementary-charge units.
    mode : str
        ``"plane_wave"`` or ``"coulomb"`` (**static**).
    radial_accelerator : str
        ``"direct"`` (**static**). The schema retains ``"hermite"`` for
        validation and raises because the frozen radial accelerator fails.
    table_n_points : int
        Registered Hermite table size (**static**).
    """

    effective_charge: Float64[Array, ""]
    mode: str = eqx.field(static=True)
    radial_accelerator: str = eqx.field(static=True)
    table_n_points: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static final-state choices."""
        if self.mode not in _FINAL_STATE_MODES:
            message: str = f"mode must be one of {_FINAL_STATE_MODES}"
            raise ValueError(message)
        if self.radial_accelerator not in _RADIAL_ACCELERATORS:
            message = (
                f"radial_accelerator must be one of {_RADIAL_ACCELERATORS}"
            )
            raise ValueError(message)
        if self.radial_accelerator == "hermite":
            message = (
                "Hermite mode failed the frozen radial accelerator "
                "1025-to-2049 next-rung certification"
            )
            raise ValueError(message)
        if self.table_n_points not in _HERMITE_TABLE_POINTS:
            message = f"table_n_points must be one of {_HERMITE_TABLE_POINTS}"
            raise ValueError(message)
        if self.mode == "coulomb" and self.radial_accelerator != "direct":
            message = "coulomb final states require direct radial evaluation"
            raise ValueError(message)


def _validate_slater_koster_structure(
    values: Float64[Array, " n_sk"],
    keys: Tuple[str, ...],
) -> None:
    """PRIVATE: Validate Slater--Koster parameter axes and identifiers.

    Implementation Logic
    --------------------
    Check the traced axis only through ``ndim`` and ``shape`` so that no
    numerical value leaves the traced domain. Compare the key-set size
    against the tuple length to reject duplicates.

    Parameters
    ----------
    values : Float64[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV.
    keys : Tuple[str, ...]
        Static material/channel identifiers, one per value.

    Raises
    ------
    ValueError
        If ``values`` is not one-dimensional. If ``keys`` disagrees
        with ``values`` on length or contains invalid or duplicate
        strings. This is the static construction-time contract.
    """
    if values.ndim != 1:
        message: str = "SlaterKosterParams values must be one-dimensional"
        raise ValueError(message)
    if type(keys) is not tuple:
        message = "SlaterKosterParams keys must be a tuple"
        raise ValueError(message)
    if len(keys) != values.shape[0]:
        message = (
            "SlaterKosterParams values and keys must have the same length"
        )
        raise ValueError(message)
    if any(type(key) is not str or not key for key in keys):
        message = "SlaterKosterParams keys must contain non-empty strings"
        raise ValueError(message)
    if len(set(keys)) != len(keys):
        message = "SlaterKosterParams keys must be unique"
        raise ValueError(message)


class SlaterKosterParams(eqx.Module):
    """Store differentiable Slater--Koster two-center integrals.

    The numerical values are the flat-real optimization coordinates for a
    Slater--Koster material model. Their keys are static identifiers such as
    ``"C-C:pp_sigma"`` or ``"Ru-O:pd_pi"``. A key change alters the material
    topology and therefore triggers JAX retracing.

    :see: :class:`~.test_radial_params.TestSlaterKosterParams`

    Attributes
    ----------
    values : Float64[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV. These values remain
        differentiable JAX leaves.
    keys : Tuple[str, ...]
        Unique material/channel identifiers (**static** -- changing them
        triggers retracing).

    Notes
    -----
    The carrier deliberately does not prescribe distance scaling. The
    Slater--Koster builder interprets the identifiers and assigns them to
    frozen neighbor shells.

    See Also
    --------
    make_slater_koster_params : Validating factory for this carrier.
    """

    values: Float64[Array, " n_sk"]
    keys: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the traced axis against the static key tuple."""
        _validate_slater_koster_structure(self.values, self.keys)


@jaxtyped(typechecker=beartype)
def make_orbital_basis(  # noqa: DOC502
    atom_indices: Tuple[int, ...],
    n: Tuple[int, ...],
    l: Tuple[int, ...],  # noqa: E741
    m: Tuple[int, ...],
    spin: Tuple[int, ...] = (),
    labels: Optional[Tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated ``OrbitalBasis`` instance.

    The factory validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is absent, the factory generates generic labels such as
    ``"orb_0"`` and ``"orb_1"``.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    :see: :class:`~.test_radial_params.TestMakeOrbitalBasis`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           n_orbitals = len(n)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           _validate_orbital_basis_structure(...)

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Return the named instance**::

           return basis

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    atom_indices : Tuple[int, ...]
        Atom-row indices (**static** -- compile-time constants; changing them
        triggers retracing), one per orbital.
    n : Tuple[int, ...]
        Principal quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    l : Tuple[int, ...]
        Angular momentum quantum numbers (**static** -- compile-time
        constants; changing them triggers retracing), one per orbital.
    m : Tuple[int, ...]
        Magnetic quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    spin : Tuple[int, ...], optional
        Spin channels (**static** -- compile-time constants; changing them
        triggers retracing). The empty tuple denotes a spinless basis;
        otherwise every entry must be ``+1`` or ``-1``. Default is empty.
    labels : Optional[Tuple[str, ...]], optional
        Human-readable orbital labels (**static** -- compile-time constants;
        changing them triggers retracing). Defaults to
        ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis with consistent lengths.

    Raises
    ------
    ValueError
        If any per-orbital tuple has a different length. The function also
        rejects invalid atom indices, quantum numbers, or spin channels.

    Notes
    -----
    Every ``OrbitalBasis`` field uses ``eqx.field(static=True)``, so the
    factory performs static validation. Invalid tuple lengths or quantum
    numbers raise ``ValueError`` before tracing. No ``eqx.error_if`` checks
    apply.

    See Also
    --------
    OrbitalBasis : The PyTree class constructed by this factory.
    """
    n_orbitals: int = len(n)
    resolved_labels: Tuple[str, ...] = (
        tuple(f"orb_{i}" for i in range(n_orbitals))
        if labels is None
        else labels
    )
    _validate_orbital_basis_structure(
        atom_indices,
        n,
        l,
        m,
        spin,
        resolved_labels,
    )
    basis: OrbitalBasis = OrbitalBasis(
        atom_indices=atom_indices,
        n=n,
        l=l,
        m=m,
        spin=spin,
        labels=resolved_labels,
    )
    return basis


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
    tail_envelope_id: str = _CERTIFIED_TAIL_ENVELOPE_ID,
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
    if mode not in _RADIAL_MODES:
        message: str = f"mode must be one of {_RADIAL_MODES}"
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
        or not 1.0 <= float(value) <= _MAX_EFFECTIVE_PRINCIPAL
        for value in resolved_n_star
    ):
        message = (
            "n_star_shell must contain values from 1 through 4.2, "
            "one per shell"
        )
        raise ValueError(message)
    if tail_envelope_id != _CERTIFIED_TAIL_ENVELOPE_ID:
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
            or grid_array.shape[0] < _MIN_COMPACT_GRID_POINTS
            or grid_value_array.shape != (n_shells, grid_array.shape[0])
        ):
            message = "grid mode arrays have inconsistent shapes"
            raise ValueError(message)
        spacings: Float64[Array, " n_interval"] = jnp.diff(grid_array)
        grid_array = eqx.error_if(
            grid_array,
            ~jnp.all(jnp.isfinite(grid_array))
            | (grid_array[0] != 0.0)
            | (grid_array[-1] > _CERTIFIED_R_MAX_BOHR)
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
            jnp.any(zeta_array < _MIN_DECAY_PARAMETER)
            | jnp.any(zeta_array > _MAX_DECAY_PARAMETER),
            "slater zeta_shell leaves the certified tail envelope",
        )
        shell_norms: list[Float64[Array, ""]] = []
        coefficient_conditions: list[Float64[Array, ""]] = []
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
            | jnp.any(contraction_conditions > _MAX_COEFFICIENT_CONDITION),
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
            basis.n[orbital_index] > _MAX_HYDROGENIC_PRINCIPAL
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
            jnp.any(charge_array / principal_array < _MIN_DECAY_PARAMETER)
            | jnp.any(charge_array / principal_array > _MAX_DECAY_PARAMETER),
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


@jaxtyped(typechecker=beartype)
def make_radial_quadrature_spec(
    profile_id: str = "gl1024-r120-k4-l9-v1",
) -> RadialQuadratureSpec:
    """Select one immutable certified quadrature profile.

    The profile identity resolves every numerical property. Callers cannot
    override tolerances or domain limits.

    :see: :class:`~.test_radial_params.TestMakeRadialQuadratureSpec`

    Notes
    -----
    Resolve every domain and tolerance field from the immutable profile map.

    Parameters
    ----------
    profile_id : str, optional
        Registered profile identity.

    Returns
    -------
    spec : RadialQuadratureSpec
        Immutable certified profile.

    Raises
    ------
    ValueError
        If ``profile_id`` is not registered.
    """
    profile: (
        Tuple[
            int,
            float,
            float,
            int,
            float,
            float,
            str,
            float,
            float,
            float,
        ]
        | None
    ) = _CERTIFIED_RADIAL_PROFILES.get(profile_id)
    if profile is None:
        message: str = "unknown certified radial quadrature profile"
        raise ValueError(message)
    spec: RadialQuadratureSpec = RadialQuadratureSpec(
        profile_id=profile_id,
        n_nodes=profile[0],
        r_max_bohr=profile[1],
        k_max_bohr_inv=profile[2],
        l_prime_max=profile[3],
        value_rtol=profile[4],
        gradient_rtol=profile[5],
        tail_bound_method_id=profile[6],
        coefficient_condition_max=profile[7],
        min_decay_parameter=profile[8],
        max_decay_parameter=profile[9],
    )
    return spec


@jaxtyped(typechecker=beartype)
def make_final_state_spec(  # noqa: DOC503
    mode: str = "plane_wave",
    effective_charge: float | Float64[Array, ""] = 0.0,
    radial_accelerator: str = "direct",
    table_n_points: int = 257,
) -> FinalStateSpec:
    """Create a validated radial final-state selection.

    Plane waves require zero charge. All final states require direct radial
    evaluation because the frozen Hermite convergence gate failed.

    :see: :class:`~.test_radial_params.TestMakeFinalStateSpec`

    Notes
    -----
    Validate static mode compatibility before checking the traced charge.

    Parameters
    ----------
    mode : str, optional
        ``"plane_wave"`` or ``"coulomb"``.
    effective_charge : float | Float64[Array, ""], optional
        Final-state effective charge.
    radial_accelerator : str, optional
        ``"direct"``. The factory recognizes ``"hermite"`` but raises.
    table_n_points : int, optional
        Registered Hermite table size.

    Returns
    -------
    spec : FinalStateSpec
        Validated final-state carrier.

    Raises
    ------
    ValueError
        When a static choice falls outside the registered options.
    EquinoxRuntimeError
        If the charge is non-finite or nonzero for a plane wave.
    """
    if mode not in _FINAL_STATE_MODES:
        message: str = f"mode must be one of {_FINAL_STATE_MODES}"
        raise ValueError(message)
    if radial_accelerator not in _RADIAL_ACCELERATORS:
        message = f"radial_accelerator must be one of {_RADIAL_ACCELERATORS}"
        raise ValueError(message)
    if radial_accelerator == "hermite":
        message = (
            "Hermite mode failed the frozen radial accelerator "
            "1025-to-2049 next-rung certification"
        )
        raise ValueError(message)
    if table_n_points not in _HERMITE_TABLE_POINTS:
        message = f"table_n_points must be one of {_HERMITE_TABLE_POINTS}"
        raise ValueError(message)
    if mode == "coulomb" and radial_accelerator != "direct":
        message = "coulomb final states require direct radial evaluation"
        raise ValueError(message)
    charge: Float64[Array, ""] = jnp.asarray(
        effective_charge,
        dtype=jnp.float64,
    )
    charge = eqx.error_if(
        charge,
        ~jnp.isfinite(charge),
        "effective_charge must be finite",
    )
    if mode == "plane_wave":
        charge = eqx.error_if(
            charge,
            charge != 0.0,
            "plane-wave final states require zero effective charge",
        )
    spec: FinalStateSpec = FinalStateSpec(
        effective_charge=charge,
        mode=mode,
        radial_accelerator=radial_accelerator,
        table_n_points=table_n_points,
    )
    return spec


@jaxtyped(typechecker=beartype)
def make_slater_koster_params(  # noqa: DOC502, DOC503
    values: Float[Array, " n_sk"],
    keys: Tuple[str, ...],
) -> SlaterKosterParams:
    """Create validated Slater--Koster two-center parameters.

    The factory normalizes numerical values and validates every static channel
    identifier before constructing the carrier.

    :see: :class:`~.test_radial_params.TestMakeSlaterKosterParams`

    Parameters
    ----------
    values : Float[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV.
    keys : Tuple[str, ...]
        Unique static identifiers, one for every value. Material builders use
        identifiers such as ``"C-C:pp_sigma"``.

    Returns
    -------
    params : SlaterKosterParams
        Parameter carrier with float64 differentiable values and static keys.

    Raises
    ------
    ValueError
        If values are not one-dimensional, keys are not a tuple, lengths
        differ, or a key is empty or duplicated.
    EquinoxRuntimeError
        If any value is non-finite, in eager or compiled execution.

    Notes
    -----
    Values may have either sign and may be zero. Only finiteness is a
    numerical invariant; channel and material semantics belong to the
    Slater--Koster model builder.

    See Also
    --------
    SlaterKosterParams : Carrier constructed by this factory.
    """
    value_array: Float64[Array, " n_sk"] = jnp.asarray(
        values,
        dtype=jnp.float64,
    )
    _validate_slater_koster_structure(value_array, keys)
    value_array = eqx.error_if(
        value_array,
        ~jnp.all(jnp.isfinite(value_array)),
        "make_slater_koster_params: values finite",
    )
    params: SlaterKosterParams = SlaterKosterParams(
        values=value_array,
        keys=keys,
    )
    return params


__all__: list[str] = [
    "FinalStateSpec",
    "MatrixElementParams",
    "OrbitalBasis",
    "RadialQuadratureSpec",
    "RadialSpec",
    "SlaterKosterParams",
    "make_final_state_spec",
    "make_matrix_element_params",
    "make_orbital_basis",
    "make_radial_quadrature_spec",
    "make_radial_spec",
    "make_slater_koster_params",
]
