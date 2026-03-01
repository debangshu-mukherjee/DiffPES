"""Eigenvector to orbital weight conversions.

Extended Summary
----------------
Provides utilities to extract orbital weights and coefficients
from diagonalized band structures.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def eigenvector_orbital_weights(
    eigenvectors: Complex[Array, "K B O"],
) -> Float[Array, "K B O"]:
    """Compute orbital weights from eigenvectors.

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb}.

    Returns
    -------
    weights : Float[Array, "K B O"]
        |c_{k,b,orb}|^2 per orbital.
    """
    return jnp.abs(eigenvectors) ** 2


@jaxtyped(typechecker=beartype)
def orbital_coefficients(
    eigenvectors: Complex[Array, "K B O"],
) -> Complex[Array, "K B O"]:
    """Return the raw complex orbital coefficients.

    For full matrix element calculation the complex coefficients
    c_{k,b,orb} are needed (not just |c|^2).

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients.

    Returns
    -------
    coefficients : Complex[Array, "K B O"]
        Same as input (identity, for API clarity).
    """
    return eigenvectors


__all__: list[str] = [
    "eigenvector_orbital_weights",
    "orbital_coefficients",
]
