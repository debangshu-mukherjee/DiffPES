"""Eigenvector to orbital weight conversions.

Extended Summary
----------------
Provides utilities to extract orbital weights and coefficients
from diagonalized band structures.  The two functions in this
module form the bridge between the tight-binding eigenvectors
(complex coefficients in the orbital basis) and the physical
observables used in ARPES simulations -- namely the orbital-
resolved spectral weight and the full complex amplitudes needed
for coherent photoemission matrix elements.

Routine Listings
----------------
:func:`eigenvector_orbital_weights`
    Compute orbital weights from eigenvectors.
:func:`orbital_coefficients`
    Return the raw complex orbital coefficients.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def eigenvector_orbital_weights(
    eigenvectors: Complex[Array, "K B O"],
) -> Float[Array, "K B O"]:
    r"""Compute orbital weights from eigenvectors.

    Extended Summary
    ----------------
    For each eigenstate ``|psi_{k,b}>`` expanded in the orbital basis

    .. math::

        |\psi_{k,b}\rangle = \sum_o c_{k,b,o} |o\rangle,

    the orbital weight of orbital ``o`` in band ``b`` at k-point ``k``
    is the squared modulus of the expansion coefficient:

    .. math::

        w_{k,b,o} = |c_{k,b,o}|^2.

    This is the probability of finding the electron in orbital ``o``
    given that it occupies eigenstate ``(k, b)``.  By construction,
    the weights sum to 1 over orbitals for each ``(k, b)`` pair when
    the eigenvectors are properly normalized.

    Orbital weights are the fundamental quantity behind fat-band
    plots, orbital-resolved DOS, and the starting point for
    photoemission matrix element calculations (where the full complex
    coefficients are also needed; see ``orbital_coefficients``).

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb}.

    Returns
    -------
    weights : Float[Array, "K B O"]
        |c_{k,b,orb}|^2 per orbital.

    Notes
    -----
    The implementation uses ``jnp.abs(eigenvectors) ** 2`` rather than
    ``(eigenvectors * eigenvectors.conj()).real`` for clarity.  Both
    expressions are mathematically identical for complex arrays and
    produce the same JAX trace; the former is marginally more readable.
    """
    return jnp.abs(eigenvectors) ** 2


@jaxtyped(typechecker=beartype)
def orbital_coefficients(
    eigenvectors: Complex[Array, "K B O"],
) -> Complex[Array, "K B O"]:
    """Return the raw complex orbital coefficients.

    For full matrix element calculation the complex coefficients
    c_{k,b,orb} are needed (not just |c|^2).

    Extended Summary
    ----------------
    This is an **identity function**: it returns its input unchanged.
    Its purpose is purely semantic -- to make call sites self-documenting
    when both ``eigenvector_orbital_weights`` (which squares the
    modulus) and the raw coefficients are needed in the same pipeline.
    Callers can write::

        weights = eigenvector_orbital_weights(evecs)
        coeffs  = orbital_coefficients(evecs)

    making it immediately obvious which downstream code path uses
    magnitudes only and which requires full phase information.

    In the photoemission matrix element calculation (Chinook-style),
    the full complex coefficients ``c_{k,b,o}`` are multiplied by the
    one-electron dipole matrix elements for each orbital ``o`` and
    then coherently summed.  Interference between orbital channels
    depends on the relative phases of these coefficients, which is
    why the complex values -- not just ``|c|^2`` -- are essential.

    When the eigenvectors originate from ``vasp_to_diagonalized``
    (the PROCAR adapter), the phases are lost and all coefficients
    are real and non-negative.  In that regime, coherent interference
    terms are approximate.

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients.

    Returns
    -------
    coefficients : Complex[Array, "K B O"]
        Same as input (identity, for API clarity).

    Notes
    -----
    Because this is an identity, ``jax.grad`` through
    ``orbital_coefficients`` adds zero overhead -- the function
    compiles away entirely.
    """
    return eigenvectors


__all__: list[str] = [
    "eigenvector_orbital_weights",
    "orbital_coefficients",
]
