r"""Evaluate dipole radial integrals with fixed differentiable quadrature.

Extended Summary
----------------
The public kernels evaluate

.. math::

    B_{l'}(k)=i^{l'}\int R(r)r^3j_{l'}(kr)\,dr

on either host-generated Gauss--Legendre nodes or a uniform compact-support
grid. Momentum arguments are explicitly in inverse Bohr. The routine applies
the partial-wave phase exactly once after the real quadrature.

Routine Listings
----------------
:func:`gauss_legendre_nodes`
    Construct Gauss--Legendre nodes and weights on ``[0, r_max_bohr]``.
:func:`momentum_inv_ang_to_bohr_inv`
    Convert momentum from inverse Angstrom to inverse Bohr.
:func:`radial_bvals`
    Assemble direct final-state radial channels for every orbital.
:func:`radial_integral`
    Evaluate a weighted :math:`R(r)r^3j_{l'}(kr)` radial integral.
:func:`radial_integral_simpson`
    Evaluate a radial integral by composite Simpson quadrature.
"""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    BOHR_TO_ANGSTROM,
    FinalStateSpec,
    RadialQuadratureSpec,
    RadialSpec,
)

from .bessel import spherical_bessel_jl
from .coulomb import final_state_radial
from .wavefunctions import evaluate_radial


def _validate_l_prime(l_prime: int) -> None:
    """PRIVATE: Validate one static final-state angular momentum.

    Parameters
    ----------
    l_prime : int
        Static final-state angular momentum to check.

    Raises
    ------
    ValueError
        If ``l_prime`` is negative.

    Notes
    -----
    Accepts any nonnegative integer and returns nothing on success.
    """
    if l_prime < 0:
        message: str = "l_prime must be non-negative"
        raise ValueError(message)


def _partial_wave_phase(l_prime: int) -> Complex128[Array, ""]:
    """PRIVATE: Return the single canonical partial-wave phase.

    Parameters
    ----------
    l_prime : int
        Nonnegative static final-state angular momentum.

    Returns
    -------
    phase : Complex128[Array, ""]
        Dimensionless scalar phase :math:`i^{l'}`.

    Notes
    -----
    Raises the Python complex unit to the static integer power on the
    host and converts the result to a complex128 scalar once.
    """
    phase: Complex128[Array, ""] = jnp.asarray(
        (1j) ** l_prime,
        dtype=jnp.complex128,
    )
    return phase


def _weighted_real_integral(
    k_bohr_inv: Float64[Array, " ..."],
    r_bohr: Float64[Array, " n_r"],
    weights_bohr: Float64[Array, " n_r"],
    radial_values: Float64[Array, " n_r"],
    l_prime: int,
) -> Float64[Array, " ..."]:
    r"""PRIVATE: Compute one real radial contraction with fixed weights.

    Parameters
    ----------
    k_bohr_inv : Float64[Array, " ..."]
        Momentum in inverse Bohr.
    r_bohr : Float64[Array, " n_r"]
        Fixed radial nodes in Bohr.
    weights_bohr : Float64[Array, " n_r"]
        Fixed integration weights in Bohr.
    radial_values : Float64[Array, " n_r"]
        Real radial wavefunction values in inverse Bohr to the power
        3/2.
    l_prime : int
        Static nonnegative final angular momentum.

    Returns
    -------
    integral : Float64[Array, " ..."]
        Real quadrature values of :math:`\int R(r)r^3j_{l'}(kr)\,dr`
        with the momentum shape.

    Notes
    -----
    Broadcasts ``k_bohr_inv`` against the radial axis, evaluates the
    spherical Bessel factor at ``k r``, and contracts the fixed measure
    ``weights_bohr * radial_values * r_bohr**3`` over the last axis.
    The caller applies the partial-wave phase.
    """
    kr: Float64[Array, "... n_r"] = (
        jnp.expand_dims(k_bohr_inv, axis=-1) * r_bohr
    )
    bessel_values: Float64[Array, "... n_r"] = spherical_bessel_jl(l_prime, kr)
    radial_measure: Float64[Array, " n_r"] = (
        weights_bohr * radial_values * r_bohr**3
    )
    integral: Float64[Array, " ..."] = jnp.sum(
        bessel_values * radial_measure,
        axis=-1,
    )
    return integral


def _simpson_weights(
    r_bohr: Float64[Array, " n_r"],
) -> Float64[Array, " n_r"]:
    """PRIVATE: Return composite Simpson weights for a validated uniform grid.

    Parameters
    ----------
    r_bohr : Float64[Array, " n_r"]
        Uniform ascending radial grid in Bohr with an odd point count.

    Returns
    -------
    weights : Float64[Array, " n_r"]
        Composite Simpson weights in Bohr.

    Notes
    -----
    Reads the uniform spacing from the first two nodes and scales the
    coefficient pattern ``1, 4, 2, 4, ..., 4, 1`` by one third of the
    spacing.  Callers validate uniformity and the odd point count.
    """
    n_points: int = r_bohr.shape[0]
    spacing: Float64[Array, ""] = r_bohr[1] - r_bohr[0]
    coefficients: Float64[Array, " n_r"] = jnp.ones(
        (n_points,), dtype=jnp.float64
    )
    coefficients = coefficients.at[1:-1:2].set(4.0)
    coefficients = coefficients.at[2:-1:2].set(2.0)
    weights: Float64[Array, " n_r"] = (spacing / 3.0) * coefficients
    return weights


@jaxtyped(typechecker=beartype)
def gauss_legendre_nodes(
    n_nodes: int,
    r_max_bohr: float,
) -> Tuple[Float64[Array, " n_r"], Float64[Array, " n_r"]]:
    """Construct Gauss--Legendre nodes and weights on ``[0, r_max_bohr]``.

    Host-side setup maps canonical nodes onto the requested finite interval.

    :see: :class:`~.test_integrate.TestGaussLegendreNodes`

    Parameters
    ----------
    n_nodes : int
        Static positive node count.
    r_max_bohr : float
        Static positive upper radial bound in Bohr.

    Returns
    -------
    r_bohr : Float64[Array, " n_r"]
        Ascending quadrature nodes in Bohr.
    weights_bohr : Float64[Array, " n_r"]
        Positive quadrature weights in Bohr.

    Raises
    ------
    ValueError
        If the node count or radial bound is invalid.

    Notes
    -----
    Node construction is host-side setup.  The returned arrays become fixed
    JAX constants in compiled radial evaluations.
    """
    if type(n_nodes) is not int or n_nodes < 1:
        message: str = "n_nodes must be a positive integer"
        raise ValueError(message)
    if not np.isfinite(r_max_bohr) or r_max_bohr <= 0.0:
        message = "r_max_bohr must be finite and positive"
        raise ValueError(message)
    canonical_pair: Tuple[
        Float64[NDArray, " n_r"], Float64[NDArray, " n_r"]
    ] = np.polynomial.legendre.leggauss(n_nodes)
    canonical_nodes: Float64[NDArray, " n_r"] = canonical_pair[0]
    canonical_weights: Float64[NDArray, " n_r"] = canonical_pair[1]
    scale: float = 0.5 * r_max_bohr
    shifted_nodes: Float64[NDArray, " n_r"] = scale * (canonical_nodes + 1.0)
    shifted_weights: Float64[NDArray, " n_r"] = scale * canonical_weights
    quadrature: Tuple[Float64[Array, " n_r"], Float64[Array, " n_r"]] = (
        jnp.asarray(shifted_nodes, dtype=jnp.float64),
        jnp.asarray(shifted_weights, dtype=jnp.float64),
    )
    return quadrature


@jaxtyped(typechecker=beartype)
def momentum_inv_ang_to_bohr_inv(
    momentum_inv_ang: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    """Convert momentum from inverse Angstrom to inverse Bohr.

    The conversion applies the exact project-wide Bohr-radius seam once.

    :see: :class:`~.test_integrate.TestMomentumInvAngToBohrInv`

    Parameters
    ----------
    momentum_inv_ang : Float64[Array, " ..."]
        Momentum in inverse Angstrom.

    Returns
    -------
    momentum_bohr_inv : Float64[Array, " ..."]
        Momentum multiplied by ``0.529177210903`` in inverse Bohr.

    Notes
    -----
    Multiply by the project-wide Bohr radius exactly once.
    """
    momentum_bohr_inv: Float64[Array, " ..."] = (
        jnp.asarray(momentum_inv_ang, dtype=jnp.float64) * BOHR_TO_ANGSTROM
    )
    return momentum_bohr_inv


@jaxtyped(typechecker=beartype)
def radial_integral(
    k_bohr_inv: Float64[Array, " ..."],
    r_bohr: Float64[Array, " n_r"],
    weights_bohr: Float64[Array, " n_r"],
    radial_values: Float64[Array, " n_r"],
    l_prime: int,
) -> Complex128[Array, " ..."]:
    r"""Evaluate a weighted :math:`R(r)r^3j_{l'}(kr)` radial integral.

    The contraction accepts fixed nodes, weights, and normalized radial values.

    :see: :class:`~.test_integrate.TestRadialIntegral`

    Parameters
    ----------
    k_bohr_inv : Float64[Array, " ..."]
        Momentum in inverse Bohr.
    r_bohr : Float64[Array, " n_r"]
        Fixed radial nodes in Bohr.
    weights_bohr : Float64[Array, " n_r"]
        Fixed integration weights in Bohr.
    radial_values : Float64[Array, " n_r"]
        Real radial wavefunction values in inverse Bohr to the power 3/2.
    l_prime : int
        Static nonnegative final angular momentum.

    Returns
    -------
    values : Complex128[Array, " ..."]
        Complex128 channel integrals with the leading momentum shape.

    Raises
    ------
    ValueError
        If ``l_prime`` is negative or radial axes are inconsistent.

    Notes
    -----
    The function accepts already converted inverse-Bohr momentum.  It applies
    :math:`i^{l'}` exactly once after the real fixed-weight contraction.
    """
    _validate_l_prime(l_prime)
    if (
        r_bohr.ndim != 1
        or weights_bohr.ndim != 1
        or radial_values.ndim != 1
        or r_bohr.shape != weights_bohr.shape
        or r_bohr.shape != radial_values.shape
    ):
        message: str = (
            "radial nodes, weights, and values must be equal vectors"
        )
        raise ValueError(message)
    k_array: Float64[Array, " ..."] = jnp.asarray(
        k_bohr_inv, dtype=jnp.float64
    )
    r_array: Float64[Array, " n_r"] = jnp.asarray(r_bohr, dtype=jnp.float64)
    weight_array: Float64[Array, " n_r"] = jnp.asarray(
        weights_bohr, dtype=jnp.float64
    )
    radial_array: Float64[Array, " n_r"] = jnp.asarray(
        radial_values, dtype=jnp.float64
    )
    r_array = eqx.error_if(
        r_array,
        ~jnp.all(jnp.isfinite(r_array))
        | ~jnp.all(jnp.diff(r_array) > 0.0)
        | (r_array[0] < 0.0),
        "radial nodes must be finite, nonnegative, and strictly ascending",
    )
    weight_array = eqx.error_if(
        weight_array,
        ~jnp.all(jnp.isfinite(weight_array)) | ~jnp.all(weight_array > 0.0),
        "radial weights must be finite and positive",
    )
    radial_array = eqx.error_if(
        radial_array,
        ~jnp.all(jnp.isfinite(radial_array)),
        "radial values must be finite",
    )
    real_integral: Float64[Array, " ..."] = _weighted_real_integral(
        k_array,
        r_array,
        weight_array,
        radial_array,
        l_prime,
    )
    values: Complex128[Array, " ..."] = _partial_wave_phase(
        l_prime
    ) * real_integral.astype(jnp.complex128)
    return values


@jaxtyped(typechecker=beartype)
def radial_integral_simpson(
    k_bohr_inv: Float64[Array, " ..."],
    r_bohr: Float64[Array, " n_r"],
    radial_values: Float64[Array, " n_r"],
    l_prime: int,
) -> Complex128[Array, " ..."]:
    """Evaluate a radial integral by composite Simpson quadrature.

    The routine constructs deterministic weights for an odd uniform grid.

    :see: :class:`~.test_integrate.TestRadialIntegralSimpson`

    Parameters
    ----------
    k_bohr_inv : Float64[Array, " ..."]
        Momentum in inverse Bohr.
    r_bohr : Float64[Array, " n_r"]
        Uniform ascending compact-support grid in Bohr.
    radial_values : Float64[Array, " n_r"]
        Real radial values sampled on ``r_bohr``.
    l_prime : int
        Static nonnegative final angular momentum.

    Returns
    -------
    values : Complex128[Array, " ..."]
        Complex128 channel integrals with the leading momentum shape.

    Raises
    ------
    ValueError
        If the grid has fewer than three points, an even point count, or
        inconsistent radial axes.

    Notes
    -----
    Composite Simpson quadrature requires an even number of uniform
    subintervals. Grid-mode setup checks uniformity and exact compact support
    before calling this traced kernel.
    """
    _validate_l_prime(l_prime)
    if (
        r_bohr.ndim != 1
        or radial_values.ndim != 1
        or r_bohr.shape != radial_values.shape
    ):
        message: str = "radial grid and values must be equal vectors"
        raise ValueError(message)
    n_points: int = r_bohr.shape[0]
    minimum_points: int = 3
    if n_points < minimum_points or n_points % 2 == 0:
        message = (
            "Simpson quadrature requires an odd point count of at least 3"
        )
        raise ValueError(message)
    r_array: Float64[Array, " n_r"] = jnp.asarray(r_bohr, dtype=jnp.float64)
    spacings: Float64[Array, " n_interval"] = jnp.diff(r_array)
    r_array = eqx.error_if(
        r_array,
        ~jnp.all(jnp.isfinite(r_array))
        | ~jnp.all(spacings > 0.0)
        | ~jnp.allclose(spacings, spacings[0], rtol=1.0e-12, atol=0.0),
        "Simpson radial grid must be finite, ascending, and uniform",
    )
    weights: Float64[Array, " n_r"] = _simpson_weights(r_array)
    values: Complex128[Array, " ..."] = radial_integral(
        k_bohr_inv,
        r_array,
        weights,
        radial_values,
        l_prime,
    )
    return values


@jaxtyped(typechecker=beartype)
def radial_bvals(  # noqa: DOC503, PLR0912, PLR0915
    spec: RadialSpec,
    k_bohr_inv: Float64[Array, " ..."],
    quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
) -> Complex128[Array, "... n_orb 2"]:
    """Assemble direct final-state radial channels for every orbital.

    The routine evaluates each shell once and gathers the results onto
    orbitals.

    :see: :class:`~.test_integrate.TestRadialBvals`

    Parameters
    ----------
    spec : RadialSpec
        Shell-shared radial-wavefunction carrier.
    k_bohr_inv : Float64[Array, " ..."]
        Nonnegative momentum in inverse Bohr.
    quadrature : RadialQuadratureSpec
        Immutable certified quadrature profile.
    final_state : FinalStateSpec
        Direct plane-wave or Coulomb final-state selection.

    Returns
    -------
    values : Complex128[Array, "... n_orb 2"]
        Channels in static order ``(l-1, l+1)``. The nonexistent lower
        channel of an s shell is exactly zero.

    Raises
    ------
    ValueError
        If static metadata exceed the certified envelope.
    NotImplementedError
        When a caller requests uncertified Hermite acceleration.
    EquinoxRuntimeError
        If traced inputs leave the certified envelope.

    Notes
    -----
    The evaluator normalizes non-fixed rows before their dipole contraction.
    Fixed inputs contain normalized, real, phase-free calibration shapes. The
    routine applies the canonical :math:`i^{l'}` phase exactly once.
    """
    if final_state.radial_accelerator != "direct":
        message: str = (
            "Hermite radial acceleration is not certified for this profile"
        )
        raise NotImplementedError(message)
    if max(spec.basis.l, default=0) + 1 > quadrature.l_prime_max:
        message = "basis angular momentum exceeds the quadrature profile"
        raise ValueError(message)
    maximum_n_star: float = 4.2
    if any(
        n_star <= 0.0 or n_star > maximum_n_star
        for n_star in spec.n_star_shell
    ):
        message = "Slater n_star leaves the certified radial envelope"
        raise ValueError(message)

    momentum: Float64[Array, " ..."] = jnp.asarray(
        k_bohr_inv, dtype=jnp.float64
    )
    momentum = eqx.error_if(
        momentum,
        ~jnp.all(jnp.isfinite(momentum))
        | jnp.any(momentum < 0.0)
        | jnp.any(momentum > quadrature.k_max_bohr_inv),
        "k_bohr_inv leaves the certified quadrature domain",
    )
    shell_indices: Int32[Array, " n_orb"] = jnp.asarray(
        spec.radial_shell_index, dtype=jnp.int32
    )
    n_shells: int = max(spec.radial_shell_index, default=-1) + 1
    representatives: Tuple[int, ...] = tuple(
        spec.radial_shell_index.index(shell) for shell in range(n_shells)
    )

    if spec.mode == "fixed":
        if final_state.mode != "plane_wave":
            message = (
                "fixed radial calibration requires a plane-wave final state"
            )
            raise ValueError(message)
        if spec.fixed_integrals_shell is None:
            message = "fixed mode requires fixed integral calibration rows"
            raise ValueError(message)
        fixed_integrals: Float64[Array, "n_shell 2"] = eqx.error_if(
            spec.fixed_integrals_shell,
            ~jnp.all(jnp.isfinite(spec.fixed_integrals_shell)),
            "fixed integral calibration rows must remain finite",
        )
        fixed_norms: Float64[Array, " n_shell"] = jnp.linalg.norm(
            fixed_integrals,
            axis=-1,
        )
        fixed_integrals = eqx.error_if(
            fixed_integrals,
            ~jnp.all(jnp.isfinite(fixed_norms)) | jnp.any(fixed_norms <= 0.0),
            "fixed integral calibration rows must have positive norm",
        )
        fixed_integrals = fixed_integrals / fixed_norms[:, None]
        fixed_shell_rows: List[Complex128[Array, " 2"]] = []
        shell: int
        orbital: int
        for shell, orbital in enumerate(representatives):
            angular: int = spec.basis.l[orbital]
            if angular == 0:
                fixed_integrals = eqx.error_if(
                    fixed_integrals,
                    fixed_integrals[shell, 0] != 0.0,
                    (
                        "the nonexistent s-shell lower radial channel "
                        "must be zero"
                    ),
                )
            lower: Complex128[Array, ""] = (
                jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
                if angular == 0
                else _partial_wave_phase(angular - 1)
                * fixed_integrals[shell, 0]
            )
            upper: Complex128[Array, ""] = (
                _partial_wave_phase(angular + 1) * fixed_integrals[shell, 1]
            )
            fixed_shell_rows.append(jnp.stack((lower, upper)))
        fixed_shell_values: Complex128[Array, "n_shell 2"] = jnp.stack(
            fixed_shell_rows
        )
        orbital_values: Complex128[Array, "n_orb 2"] = fixed_shell_values[
            shell_indices
        ]
        broadcast_values: Complex128[Array, "... n_orb 2"] = jnp.broadcast_to(
            orbital_values,
            momentum.shape + orbital_values.shape,
        )
        return broadcast_values

    radial_grid: Float64[Array, " n_r"]
    radial_weights: Float64[Array, " n_r"]
    if spec.mode == "grid":
        if spec.r_grid is None or spec.grid_values_shell is None:
            message = "grid mode requires compact-support sampled rows"
            raise ValueError(message)
        minimum_points: int = 3
        if spec.r_grid.shape[0] < minimum_points:
            message = "grid mode requires at least three points"
            raise ValueError(message)
        if spec.r_grid.shape[0] % 2 == 0:
            message = "grid mode requires an odd Simpson point count"
            raise ValueError(message)
        grid_spacings: Float64[Array, " n_interval"] = jnp.diff(spec.r_grid)
        radial_grid = eqx.error_if(
            spec.r_grid,
            ~jnp.all(jnp.isfinite(spec.r_grid))
            | ~jnp.all(grid_spacings > 0.0)
            | ~jnp.allclose(
                grid_spacings,
                grid_spacings[0],
                rtol=1.0e-12,
                atol=0.0,
            )
            | (grid_spacings[0] > math.pi / 20.0)
            | (spec.r_grid[-1] > quadrature.r_max_bohr),
            "grid radial spacing or support leaves the certified envelope",
        )
        radial_weights = _simpson_weights(radial_grid)
    else:
        radial_grid, radial_weights = gauss_legendre_nodes(
            quadrature.n_nodes,
            quadrature.r_max_bohr,
        )

    orbital_radial_rows: Float64[Array, "n_orb n_r"] = evaluate_radial(
        spec,
        radial_grid,
    )

    shell_values: List[Complex128[Array, "... 2"]] = []
    orbital: int
    for orbital in representatives:
        angular = spec.basis.l[orbital]
        radial_row: Float64[Array, " n_r"] = orbital_radial_rows[orbital]

        if final_state.mode == "plane_wave":
            lower_values: Complex128[Array, " ..."] = (
                jnp.zeros_like(momentum, dtype=jnp.complex128)
                if angular == 0
                else radial_integral(
                    momentum,
                    radial_grid,
                    radial_weights,
                    radial_row,
                    angular - 1,
                )
            )
            upper_values: Complex128[Array, " ..."] = radial_integral(
                momentum,
                radial_grid,
                radial_weights,
                radial_row,
                angular + 1,
            )
        else:
            radial_measure: Float64[Array, " n_r"] = (
                radial_weights * radial_row * radial_grid**3
            )
            if angular == 0:
                lower_values = jnp.zeros_like(
                    momentum,
                    dtype=jnp.complex128,
                )
            else:
                lower_final: Complex128[Array, "... n_r"] = final_state_radial(
                    angular - 1,
                    momentum,
                    radial_grid,
                    final_state,
                )
                lower_values = _partial_wave_phase(angular - 1) * jnp.sum(
                    lower_final * radial_measure,
                    axis=-1,
                )
            upper_final: Complex128[Array, "... n_r"] = final_state_radial(
                angular + 1,
                momentum,
                radial_grid,
                final_state,
            )
            upper_values = _partial_wave_phase(angular + 1) * jnp.sum(
                upper_final * radial_measure,
                axis=-1,
            )
        shell_values.append(jnp.stack((lower_values, upper_values), axis=-1))
    stacked_shell_values: Complex128[Array, "... n_shell 2"] = jnp.stack(
        shell_values, axis=-2
    )
    values: Complex128[Array, "... n_orb 2"] = stacked_shell_values[
        ..., shell_indices, :
    ]
    return values


__all__: list[str] = [
    "gauss_legendre_nodes",
    "momentum_inv_ang_to_bohr_inv",
    "radial_bvals",
    "radial_integral",
    "radial_integral_simpson",
]
