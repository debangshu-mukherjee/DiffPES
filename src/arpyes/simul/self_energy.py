"""Energy-dependent self-energy evaluation for ARPES simulations.

Extended Summary
----------------
Evaluates the imaginary part of the electronic self-energy
(lifetime broadening) as a function of energy. The self-energy
determines the Lorentzian component of the spectral linewidth
and is critical for modelling correlated materials where
quasiparticle lifetimes vary strongly with energy.

Three parametric models are supported:

- **constant**: Uniform broadening at all energies.
- **polynomial**: Polynomial expansion in energy (e.g., Fermi-liquid
  quadratic dependence).
- **tabulated**: Piecewise-linear interpolation from user-supplied
  (energy, gamma) pairs (e.g., from GW or DMFT calculations).

All modes are JAX-differentiable with respect to the model
coefficients, enabling gradient-based fitting of self-energy
parameters to experimental linewidths.

Routine Listings
----------------
:func:`evaluate_self_energy`
    Evaluate the energy-dependent Lorentzian broadening from
    a ``SelfEnergyConfig``.

Notes
-----
The mode string is a Python-level dispatch (not JAX-traced),
so only one code path is compiled per JIT invocation.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from arpyes.types import SelfEnergyConfig


@jaxtyped(typechecker=beartype)
def evaluate_self_energy(
    energy: Float[Array, " ..."],
    config: SelfEnergyConfig,
) -> Float[Array, " ..."]:
    r"""Evaluate the imaginary self-energy :math:`\Gamma(E)`.

    Computes the energy-dependent Lorentzian broadening width from
    the specified self-energy model. The result replaces the constant
    ``params.gamma`` in the Voigt profile when energy-dependent
    lifetime effects are important (e.g., near the Fermi level or in
    correlated materials).

    Extended Summary
    ----------------
    The imaginary part of the electron self-energy determines the
    quasiparticle lifetime and thus the Lorentzian component of the
    spectral linewidth. Three parametric models are supported:

    - **constant**: A single scalar broadening applied uniformly at
      all energies. Equivalent to using ``params.gamma`` directly.
    - **polynomial**: :math:`\Gamma(E) = \sum_n c_n E^n` where
      ``coefficients`` are ordered highest-degree first (as expected
      by ``jnp.polyval``). Useful for capturing the quadratic
      Fermi-liquid self-energy :math:`\Gamma \propto (E - E_F)^2`.
    - **tabulated**: Piecewise-linear interpolation of user-supplied
      :math:`(\varepsilon_i, \Gamma_i)` pairs via ``jnp.interp``.
      Suitable for self-energies obtained from many-body calculations
      (e.g., GW or DMFT).

    Implementation Logic
    --------------------
    1. **Read mode** from ``config.mode`` (a Python string, not
       traced).
    2. **Dispatch on mode**:

       - ``"constant"``: Broadcast ``coefficients[0]`` to the shape
         of ``energy``.
       - ``"polynomial"``: Evaluate ``jnp.polyval(coefficients,
         energy)`` for a polynomial of degree
         ``len(coefficients) - 1``.
       - ``"tabulated"``: Interpolate ``jnp.interp(energy,
         config.energy_nodes, config.coefficients)`` using the
         tabulated energy-gamma pairs.

    3. **Error on unknown mode**: Raises ``ValueError`` with the
       unrecognized mode string.

    Parameters
    ----------
    energy : Float[Array, " ..."]
        Energy values in eV at which to evaluate the self-energy.
        Can be any shape; the output will match.
    config : SelfEnergyConfig
        Self-energy model specification containing:

        - ``mode`` : str -- ``"constant"``, ``"polynomial"``, or
          ``"tabulated"``.
        - ``coefficients`` : array -- Model parameters (scalar for
          constant, polynomial coefficients for polynomial, gamma
          values for tabulated).
        - ``energy_nodes`` : array (only for ``"tabulated"``) --
          Energy grid for the tabulated gamma values.

    Returns
    -------
    gamma : Float[Array, " ..."]
        Energy-dependent Lorentzian broadening in eV, same shape
        as ``energy``.

    Raises
    ------
    ValueError
        If ``config.mode`` is not one of ``"constant"``,
        ``"polynomial"``, or ``"tabulated"``.

    Notes
    -----
    All three modes are fully JAX-differentiable with respect to
    ``config.coefficients``. The ``"constant"`` and ``"polynomial"``
    modes are also differentiable with respect to ``energy``. The
    ``"tabulated"`` mode uses ``jnp.interp`` which has limited
    gradient support (piecewise-constant gradients).
    """
    mode = config.mode

    if mode == "constant":
        return jnp.broadcast_to(config.coefficients[0], energy.shape)
    if mode == "polynomial":
        return jnp.polyval(config.coefficients, energy)
    if mode == "tabulated":
        assert config.energy_nodes is not None
        return jnp.interp(energy, config.energy_nodes, config.coefficients)
    msg = f"Unknown self-energy mode: {mode}"
    raise ValueError(msg)


__all__: list[str] = ["evaluate_self_energy"]
