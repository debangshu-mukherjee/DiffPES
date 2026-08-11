r"""Provide differentiable radial primitives for ARPES matrix elements.

Extended Summary
----------------
The subpackage provides stable Bessel functions, normalized atomic radial
models, certified fixed quadrature, and Slater screening estimates. The
central dipole quantity is

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r)\, r^3\, j_{l'}(kr)\, dr

The following list describes the submodules:

- :mod:`bessel`
    Evaluate spherical Bessel functions with stable JAX primitives.
- :mod:`coulomb`
    Evaluate regular and irregular Coulomb radial functions.
- :mod:`integrate`
    Evaluate dipole radial integrals with fixed differentiable quadrature.
- :mod:`screening`
    Compute static atomic configurations and Slater screening estimates.
- :mod:`wavefunctions`
    Evaluate atomic radial wavefunction models in JAX.

Routine Listings
----------------
:func:`coulomb_fg`
    Evaluate normalized Coulomb functions and radial derivatives.
:func:`coulomb_phase_shift`
    Evaluate the continuous Coulomb arg-Gamma phase.
:func:`electron_configuration`
    Return the neutral ground-state subshell configuration.
:func:`evaluate_radial`
    Evaluate normalized shell-shared radial rows on their declared grid.
:func:`final_state_radial`
    Evaluate a plane-wave or Coulomb final-state radial row.
:func:`gauss_legendre_nodes`
    Construct Gauss--Legendre nodes and weights on ``[0, r_max_bohr]``.
:func:`hydrogenic_radial`
    Evaluate normalized hydrogenic radial function.
:func:`momentum_inv_ang_to_bohr_inv`
    Convert momentum from inverse Angstrom to inverse Bohr.
:func:`radial_bvals`
    Assemble direct final-state radial channels for every orbital.
:func:`radial_integral`
    Evaluate a weighted :math:`R(r)r^3j_{l'}(kr)` radial integral.
:func:`radial_integral_simpson`
    Evaluate a radial integral by composite Simpson quadrature.
:func:`slater_radial`
    Evaluate normalized Slater-type radial function.
:func:`slater_zeff`
    Compute a subshell effective charge from Slater screening.
:func:`slater_zeta`
    Compute a Slater exponent from the effective principal number.
:func:`spherical_bessel_jl`
    Evaluate the spherical Bessel function :math:`j_l(x)`.
:func:`spherical_bessel_jl_derivative`
    Evaluate :math:`d j_l(x)/dx`.

Notes
-----
All functions support JAX transformations and automatic differentiation.
The Bessel and Laguerre recurrences use ``jax.lax.fori_loop``.
Plane-wave radial channels use direct fixed-node quadrature. The optional
Hermite table fails its frozen convergence ladder, so the factory rejects it.
"""

from .bessel import spherical_bessel_jl, spherical_bessel_jl_derivative
from .coulomb import coulomb_fg, coulomb_phase_shift, final_state_radial
from .integrate import (
    gauss_legendre_nodes,
    momentum_inv_ang_to_bohr_inv,
    radial_bvals,
    radial_integral,
    radial_integral_simpson,
)
from .screening import electron_configuration, slater_zeff, slater_zeta
from .wavefunctions import evaluate_radial, hydrogenic_radial, slater_radial

__all__: list[str] = [
    "coulomb_fg",
    "coulomb_phase_shift",
    "electron_configuration",
    "evaluate_radial",
    "final_state_radial",
    "gauss_legendre_nodes",
    "hydrogenic_radial",
    "momentum_inv_ang_to_bohr_inv",
    "radial_bvals",
    "radial_integral",
    "radial_integral_simpson",
    "slater_radial",
    "slater_zeff",
    "slater_zeta",
    "spherical_bessel_jl",
    "spherical_bessel_jl_derivative",
]
