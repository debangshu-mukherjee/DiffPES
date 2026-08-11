"""Evaluate atomic radial wavefunction models in JAX.

Extended Summary
----------------
The module provides normalized Slater-type and hydrogenic radial
wavefunctions for differentiable ARPES matrix element computations.

Routine Listings
----------------
:func:`evaluate_radial`
    Evaluate normalized shell-shared radial rows on their declared grid.
:func:`hydrogenic_radial`
    Evaluate normalized hydrogenic radial function.
:func:`slater_radial`
    Evaluate normalized Slater-type radial function.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64, Integer, jaxtyped

from diffpes.types import RadialSpec, ScalarFloat


def _associated_laguerre(
    order: int,
    alpha: int | ScalarFloat,
    x: Float64[Array, " ..."],
) -> Float64[Array, " ..."]:
    r"""PRIVATE: Evaluate associated Laguerre polynomial.

    The function computes :math:`L_n^\alpha(x)`.

    The function uses the standard three-term recurrence for the generalized
    Laguerre polynomial. This recurrence remains numerically stable during
    upward iteration in the polynomial order.

    **Seed values:**

    .. math::

        L_0^\alpha(x) = 1

        L_1^\alpha(x) = 1 + \alpha - x

    **Upward recurrence** (for n >= 2):

    .. math::

        n \, L_n^\alpha(x) = (2n - 1 + \alpha - x) \, L_{n-1}^\alpha(x)
                            - (n - 1 + \alpha) \, L_{n-2}^\alpha(x)

    The function implements this recurrence with ``jax.lax.fori_loop``. The
    loop carries :math:`(L_{n-2}^\alpha, L_{n-1}^\alpha)`. Each iteration
    advances one order from n=2 through ``order``.

    The generalized Laguerre polynomials appear in the hydrogenic
    radial wavefunctions as :math:`L_{n-l-1}^{2l+1}(\rho)` where
    :math:`\rho = 2 Z_{\text{eff}} r / n`. They are orthogonal on
    :math:`[0, \infty)` with weight :math:`x^\alpha e^{-x}`:

    .. math::

        \int_0^\infty x^\alpha e^{-x} L_n^\alpha(x) L_m^\alpha(x) \, dx
        = \frac{\Gamma(n + \alpha + 1)}{n!} \, \delta_{nm}

    Parameters
    ----------
    order : int
        Polynomial order (n >= 0).
    alpha : int | ScalarFloat
        Generalization parameter (alpha >= 0). For hydrogenic
        wavefunctions, alpha = 2*l + 1.
    x : Float64[Array, " ..."]
        Evaluation points.

    Returns
    -------
    values : Float64[Array, " ..."]
        :math:`L_n^\alpha(x)` evaluated element-wise.

    Raises
    ------
    ValueError
        If ``order`` or ``alpha`` is negative.
    """
    if order < 0:
        msg: str = "order must be non-negative"
        raise ValueError(msg)
    if alpha < 0:
        msg: str = "alpha must be non-negative"
        raise ValueError(msg)

    x_arr: Float64[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    laguerre_zero: Float64[Array, " ..."] = jnp.ones_like(x_arr)
    if order == 0:
        return laguerre_zero

    alpha_arr: Float64[Array, " "] = jnp.asarray(alpha, dtype=jnp.float64)
    laguerre_one: Float64[Array, " ..."] = 1.0 + alpha_arr - x_arr
    if order == 1:
        return laguerre_one

    def _recurrence_step(
        current_order: Integer[Array, ""],
        state: Tuple[Float64[Array, " ..."], Float64[Array, " ..."]],
    ) -> Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]:
        r"""PRIVATE: Apply one step of the Laguerre recurrence.

        Parameters
        ----------
        current_order : Integer[Array, ""]
            Order ``n`` of the polynomial this step produces.
        state : Tuple[Float64[Array, " ..."], Float64[Array, " ..."]]
            Pair :math:`(L_{n-2}^\alpha, L_{n-1}^\alpha)` of the two
            preceding polynomials.

        Returns
        -------
        recurrence_state : tuple of Float64[Array, " ..."]
            Shifted pair :math:`(L_{n-1}^\alpha, L_n^\alpha)`.

        Notes
        -----
        Casts the loop index to float64 and applies the three-term
        recurrence
        :math:`n L_n^\alpha = (2n-1+\alpha-x) L_{n-1}^\alpha
        - (n-1+\alpha) L_{n-2}^\alpha`.
        """
        laguerre_prev_prev: Float64[Array, " ..."]
        laguerre_prev: Float64[Array, " ..."]
        laguerre_prev_prev, laguerre_prev = state
        order_arr: Float64[Array, " "] = jnp.asarray(
            current_order, dtype=jnp.float64
        )
        prefactor: Float64[Array, " ..."] = (
            2.0 * order_arr - 1.0 + alpha_arr - x_arr
        ) / order_arr
        correction: Float64[Array, " ..."] = (
            (order_arr - 1.0 + alpha_arr) / order_arr
        ) * laguerre_prev_prev
        laguerre_curr: Float64[Array, " ..."] = (
            prefactor * laguerre_prev - correction
        )
        recurrence_state: Tuple[
            Float64[Array, " ..."], Float64[Array, " ..."]
        ] = (
            laguerre_prev,
            laguerre_curr,
        )
        return recurrence_state

    recurrence_result: Tuple[
        Float64[Array, " ..."], Float64[Array, " ..."]
    ] = jax.lax.fori_loop(
        2,
        order + 1,
        _recurrence_step,
        (laguerre_zero, laguerre_one),
    )
    laguerre_final: Float64[Array, " ..."] = recurrence_result[1]
    return laguerre_final


@jaxtyped(typechecker=beartype)
def slater_radial(
    r: Float64[Array, " ..."],
    n: int,
    zeta: ScalarFloat,
) -> Float64[Array, " ..."]:
    r"""Evaluate normalized Slater-type radial function.

    The function computes the Slater-type orbital (STO) radial function:

    .. math::

        R(r) = N \, r^{n-1} \, e^{-\zeta r}

    The normalization constant :math:`N` satisfies
    :math:`\int_0^\infty |R(r)|^2 r^2 dr = 1`:

    .. math::

        N = \frac{(2\zeta)^{n + 1/2}}{\sqrt{(2n)!}}

    **Slater vs. hydrogenic models:**

    Slater-type orbitals are simpler than hydrogenic radial functions
    because they lack the associated Laguerre polynomial factor. They
    have the correct exponential decay and cusp behavior at the
    nucleus, making them popular as basis functions in quantum
    chemistry. However, they do not possess radial nodes (except at
    r = 0 and r = infinity), unlike the exact hydrogenic solutions.

    The Slater exponent :math:`\zeta` represents the effective nuclear charge
    and screening. A fit to Hartree-Fock atomic orbitals usually determines
    this exponent. Variational optimization provides another method.

    **Normalization derivation:**

    The radial normalization integral is:

    .. math::

        \int_0^\infty r^{2(n-1)} e^{-2\zeta r} r^2 dr
        = \int_0^\infty r^{2n} e^{-2\zeta r} dr
        = \frac{(2n)!}{(2\zeta)^{2n+1}}

    Setting :math:`N^2 \cdot (2n)! / (2\zeta)^{2n+1} = 1` gives the
    formula above.

    :see: :class:`~.test_wavefunctions.TestSlaterRadial`

    Parameters
    ----------
    r : Float64[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number (``n >= 1``).
    zeta : ScalarFloat
        Slater exponent.

    Returns
    -------
    values : Float64[Array, " ..."]
        Normalized radial function
        ``R(r) = N r^(n-1) exp(-zeta * r)``.

    Raises
    ------
    ValueError
        If ``n`` is less than one.

    Notes
    -----
    The ``zeta`` parameter is a JAX array, not a Python float. Therefore,
    automatic differentiation can include this parameter. Inverse workflows
    can use its gradient to optimize Slater exponents.
    """
    if n < 1:
        msg: str = "n must be >= 1"
        raise ValueError(msg)

    r_arr: Float64[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    zeta_arr: Float64[Array, " "] = jnp.asarray(zeta, dtype=jnp.float64)
    factorial_term: Float64[Array, " "] = jnp.asarray(
        math.factorial(2 * n), dtype=jnp.float64
    )
    norm: Float64[Array, " "] = ((2.0 * zeta_arr) ** (n + 0.5)) / jnp.sqrt(
        factorial_term
    )
    values: Float64[Array, " ..."] = (
        norm * (r_arr ** (n - 1)) * jnp.exp(-zeta_arr * r_arr)
    )
    return values


@jaxtyped(typechecker=beartype)
def hydrogenic_radial(
    r: Float64[Array, " ..."],
    n: int,
    angular_momentum: int,
    z_eff: ScalarFloat,
) -> Float64[Array, " ..."]:
    r"""Evaluate normalized hydrogenic radial function.

    The function computes the exact radial wavefunction for a hydrogenic atom.
    The atom has one electron and an effective nuclear charge
    :math:`Z_{\text{eff}}`:

    .. math::

        R_{n,l}(r) = N_{n,l} \, e^{-\rho/2} \, \rho^l \,
            L_{n-l-1}^{2l+1}(\rho)

    Here, :math:`\rho = 2 Z_{\text{eff}} r / n` is the scaled radial
    coordinate. The `_associated_laguerre` function computes the generalized
    Laguerre polynomial :math:`L_{n-l-1}^{2l+1}`.

    **Normalization:**

    The normalization constant is:

    .. math::

        N_{n,l} = \left(\frac{2 Z_{\text{eff}}}{n}\right)^{3/2}
            \sqrt{\frac{(n - l - 1)!}{2n \cdot (n + l)!}}

    This ensures :math:`\int_0^\infty |R_{n,l}(r)|^2 r^2 dr = 1`.
    The function computes the factorial ratio with Python's ``math.factorial``.
    This operation uses exact integer arithmetic. The function then converts
    the ratio to a JAX scalar with ``jnp.sqrt``.

    **Hydrogenic vs. Slater model:**

    Unlike Slater-type orbitals (which are node-free exponentials),
    hydrogenic radial functions have :math:`n - l - 1` radial nodes
    encoded by the zeros of the Laguerre polynomial. This makes them
    exact solutions for hydrogen-like atoms but less commonly used as
    basis functions in multi-electron calculations.

    **Laguerre polynomial recurrence:**

    The `_associated_laguerre` function computes
    :math:`L_{n-l-1}^{2l+1}(\rho)` with an upward three-term recurrence.
    The recurrence starts at order 0 and ends at :math:`n - l - 1`.
    It remains stable in the upward direction. The implementation uses
    ``jax.lax.fori_loop`` for JAX transformations.

    :see: :class:`~.test_wavefunctions.TestHydrogenicRadial`

    Parameters
    ----------
    r : Float64[Array, " ..."]
        Radial coordinate in atomic units.
    n : int
        Principal quantum number.
    angular_momentum : int
        Angular momentum quantum number (``0 <= angular_momentum < n``).
    z_eff : ScalarFloat
        Effective nuclear charge.

    Returns
    -------
    values : Float64[Array, " ..."]
        ``R_{n,l}(r)`` for hydrogenic orbitals.

    Raises
    ------
    ValueError
        If ``n`` is less than one or ``angular_momentum`` lies outside
        ``[0, n)``.

    Notes
    -----
    The ``z_eff`` parameter is a JAX array that supports automatic
    differentiation. The quantum numbers ``n`` and ``angular_momentum`` are
    Python integers. They control the Laguerre polynomial order and remain
    static in the traced computation graph.
    """
    if n < 1:
        msg: str = "n must be >= 1"
        raise ValueError(msg)
    if angular_momentum < 0 or angular_momentum >= n:
        msg: str = "angular_momentum must satisfy 0 <= angular_momentum < n"
        raise ValueError(msg)

    r_arr: Float64[Array, " ..."] = jnp.asarray(r, dtype=jnp.float64)
    z_arr: Float64[Array, " "] = jnp.asarray(z_eff, dtype=jnp.float64)
    n_float: float = float(n)
    rho: Float64[Array, " ..."] = 2.0 * z_arr * r_arr / n_float

    laguerre_order: int = n - angular_momentum - 1
    laguerre_alpha: int = 2 * angular_momentum + 1
    laguerre_values: Float64[Array, " ..."] = _associated_laguerre(
        laguerre_order, laguerre_alpha, rho
    )

    factorial_ratio: float = math.factorial(laguerre_order) / (
        2.0 * n_float * math.factorial(n + angular_momentum)
    )
    prefactor: Float64[Array, " "] = ((2.0 * z_arr) / n_float) ** 1.5
    norm: Float64[Array, " "] = prefactor * jnp.sqrt(
        jnp.asarray(factorial_ratio, dtype=jnp.float64)
    )
    values: Float64[Array, " ..."] = (
        norm * jnp.exp(-0.5 * rho) * (rho**angular_momentum) * laguerre_values
    )
    return values


def _contracted_slater_row(
    r: Float64[Array, " n_r"],
    effective_principal: float,
    zeta_row: Float64[Array, " n_contraction"],
    coefficient_row: Float64[Array, " n_contraction"],
) -> Float64[Array, " n_r"]:
    """PRIVATE: Evaluate and analytically normalize one contracted Slater row.

    Implementation Logic
    --------------------
    1. **Evaluate the normalized primitives**::

           primitive_rows = (
               primitive_norms[:, None]
               * radial_power[None, :]
               * jnp.exp(-zeta_row[:, None] * r[None, :])
           )

       Each analytic norm uses ``Gamma(2 * n_star + 1)``.

    2. **Compute the contraction norm**::

           norm_squared = jnp.einsum(
               "i,ij,j->",
               coefficient_row,
               overlap,
               coefficient_row,
           )

       The analytic primitive overlap matrix defines the quadratic form.

    3. **Validate and normalize the row**::

           values = (
               checked_coefficients @ primitive_rows
           ) / jnp.sqrt(norm_squared)

       The runtime check rejects nonpositive norms and condition numbers above
       32. The final division normalizes the contracted radial function.

    Parameters
    ----------
    r : Float64[Array, " n_r"]
        Nonnegative radial nodes in Bohr.
    effective_principal : float
        Static Slater effective principal number ``n_star``.
    zeta_row : Float64[Array, " n_contraction"]
        Primitive Slater exponents in inverse Bohr.
    coefficient_row : Float64[Array, " n_contraction"]
        Dimensionless contraction coefficients.

    Returns
    -------
    values : Float64[Array, " n_r"]
        Normalized contracted radial values in inverse Bohr to the
        power 3/2.

    """
    gamma_value: Float64[Array, ""] = jnp.asarray(
        math.gamma(2.0 * effective_principal + 1.0),
        dtype=jnp.float64,
    )
    primitive_norms: Float64[Array, " n_contraction"] = (
        (2.0 * zeta_row) ** (effective_principal + 0.5)
    ) / jnp.sqrt(gamma_value)
    radial_power: Float64[Array, " n_r"] = jnp.where(
        r == 0.0,
        jnp.asarray(
            1.0 if effective_principal == 1.0 else 0.0,
            dtype=jnp.float64,
        ),
        r ** (effective_principal - 1.0),
    )
    primitive_rows: Float64[Array, "n_contraction n_r"] = (
        primitive_norms[:, None]
        * radial_power[None, :]
        * jnp.exp(-zeta_row[:, None] * r[None, :])
    )
    overlap: Float64[Array, "n_contraction n_contraction"] = (
        primitive_norms[:, None]
        * primitive_norms[None, :]
        * gamma_value
        / (
            (zeta_row[:, None] + zeta_row[None, :])
            ** (2.0 * effective_principal + 1.0)
        )
    )
    norm_squared: Float64[Array, ""] = jnp.einsum(
        "i,ij,j->",
        coefficient_row,
        overlap,
        coefficient_row,
    )
    checked_coefficients: Float64[Array, " n_contraction"] = eqx.error_if(
        coefficient_row,
        ~jnp.isfinite(norm_squared)
        | (norm_squared <= 0.0)
        | (
            jnp.sum(jnp.abs(coefficient_row)) / jnp.sqrt(norm_squared) > 32.0  # noqa: PLR2004
        ),
        (
            "slater contraction rows must have positive finite norm "
            "and coefficient condition at most 32"
        ),
    )
    values: Float64[Array, " n_r"] = (
        checked_coefficients @ primitive_rows
    ) / jnp.sqrt(norm_squared)
    return values


@jaxtyped(typechecker=beartype)
def evaluate_radial(  # noqa: DOC503
    spec: RadialSpec,
    r: Float64[Array, " n_r"],
) -> Float64[Array, "n_orb n_r"]:
    """Evaluate normalized shell-shared radial rows on their declared grid.

    Slater contractions use their analytic overlap matrix. Hydrogenic rows
    retain their analytic normalization. The grid path normalizes rows on the
    exact stored compact-support grid and never interpolates.

    :see: :class:`~.test_wavefunctions.TestEvaluateRadial`

    Parameters
    ----------
    spec : RadialSpec
        Validated radial carrier.
    r : Float64[Array, "n_r"]
        Nonnegative radial evaluation points in Bohr. Grid mode requires the
        exact stored grid.

    Returns
    -------
    values : Float64[Array, "n_orb n_r"]
        Normalized radial row gathered onto every orbital.

    Raises
    ------
    ValueError
        If ``r`` has invalid shape or fixed mode requests a radial function.
    EquinoxRuntimeError
        If traced values violate the certified tail envelope, grid identity,
        finiteness, or positive-norm contract.

    Notes
    -----
    The active Slater and hydrogenic decay parameters must remain in
    ``[0.5, 4]`` after any traced PyTree update.
    """
    radial_grid: Float64[Array, " n_r"] = jnp.asarray(r, dtype=jnp.float64)
    if radial_grid.ndim != 1:
        message: str = "r must be a one-dimensional radial grid"
        raise ValueError(message)
    if spec.mode == "fixed":
        message = "fixed mode supplies integrals and has no radial function"
        raise ValueError(message)
    radial_grid = eqx.error_if(
        radial_grid,
        ~jnp.all(jnp.isfinite(radial_grid)) | jnp.any(radial_grid < 0.0),
        "r must contain finite nonnegative radii",
    )
    n_shells: int = max(spec.radial_shell_index, default=-1) + 1
    representatives: Tuple[int, ...] = tuple(
        spec.radial_shell_index.index(shell_index)
        for shell_index in range(n_shells)
    )
    shell_rows: Float64[Array, "n_shell n_r"]
    if spec.mode == "slater":
        checked_zeta: Float64[Array, "n_shell n_contraction"] = eqx.error_if(
            spec.zeta_shell,
            ~jnp.all(jnp.isfinite(spec.zeta_shell))
            | jnp.any(spec.zeta_shell < 0.5)  # noqa: PLR2004
            | jnp.any(spec.zeta_shell > 4.0),  # noqa: PLR2004
            "slater zeta_shell leaves the certified tail envelope",
        )
        checked_coefficients: Float64[Array, "n_shell n_contraction"] = (
            eqx.error_if(
                spec.coefficients_shell,
                ~jnp.all(jnp.isfinite(spec.coefficients_shell)),
                "coefficients_shell must be finite",
            )
        )
        slater_rows: List[Float64[Array, " n_r"]] = []
        shell_index: int
        for shell_index in range(n_shells):
            slater_rows.append(
                _contracted_slater_row(
                    radial_grid,
                    spec.n_star_shell[shell_index],
                    checked_zeta[shell_index],
                    checked_coefficients[shell_index],
                )
            )
        shell_rows = jnp.stack(slater_rows, axis=0)
    elif spec.mode == "hydrogenic":
        principal_array: Float64[Array, " n_shell"] = jnp.asarray(
            tuple(
                spec.basis.n[representatives[shell_index]]
                for shell_index in range(n_shells)
            ),
            dtype=jnp.float64,
        )
        checked_charge: Float64[Array, " n_shell"] = eqx.error_if(
            spec.effective_charge_shell,
            ~jnp.all(jnp.isfinite(spec.effective_charge_shell))
            | jnp.any(
                spec.effective_charge_shell / principal_array < 0.5  # noqa: PLR2004
            )
            | jnp.any(
                spec.effective_charge_shell / principal_array > 4.0  # noqa: PLR2004
            ),
            "hydrogenic effective charge leaves the certified tail envelope",
        )
        hydrogenic_rows: List[Float64[Array, " n_r"]] = []
        for shell_index in range(n_shells):
            orbital_index: int = representatives[shell_index]
            hydrogenic_rows.append(
                hydrogenic_radial(
                    radial_grid,
                    spec.basis.n[orbital_index],
                    spec.basis.l[orbital_index],
                    checked_charge[shell_index],
                )
            )
        shell_rows = jnp.stack(hydrogenic_rows, axis=0)
    else:
        if spec.r_grid is None or spec.grid_values_shell is None:
            message = "grid mode requires stored grid arrays"
            raise ValueError(message)
        if radial_grid.shape != spec.r_grid.shape:
            message = "grid mode evaluates only on its stored grid"
            raise ValueError(message)
        checked_grid: Float64[Array, " n_r"] = eqx.error_if(
            spec.r_grid,
            ~jnp.all(radial_grid == spec.r_grid),
            "grid mode performs no interpolation",
        )
        checked_values: Float64[Array, "n_shell n_r"] = eqx.error_if(
            spec.grid_values_shell,
            ~jnp.all(jnp.isfinite(spec.grid_values_shell))
            | ~jnp.all(spec.grid_values_shell[:, -1] == 0.0),
            "grid rows must remain finite and compact-supported",
        )
        grid_norms: Float64[Array, " n_shell"] = jnp.trapezoid(
            checked_values**2 * checked_grid[None, :] ** 2,
            x=checked_grid,
            axis=-1,
        )
        checked_values = eqx.error_if(
            checked_values,
            ~jnp.all(jnp.isfinite(grid_norms)) | jnp.any(grid_norms <= 0.0),
            "grid radial rows must have positive finite norm",
        )
        shell_rows = checked_values / jnp.sqrt(grid_norms)[:, None]
    gather_indices: Integer[Array, " n_orb"] = jnp.asarray(
        spec.radial_shell_index,
        dtype=jnp.int32,
    )
    values: Float64[Array, "n_orb n_r"] = shell_rows[gather_indices]
    return values


__all__: list[str] = [
    "evaluate_radial",
    "hydrogenic_radial",
    "slater_radial",
]
