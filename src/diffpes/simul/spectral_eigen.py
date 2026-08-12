"""Evaluate nondegenerate eigenvalue spectral observables.

Extended Summary
----------------
This module uses gauge-invariant band weights.
It rejects degenerate eigensystem derivatives.

Routine Listings
----------------
:func:`spectral_intensity_eigen`
    Evaluate spectral intensity from eigenvalues and invariant weights.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.constants import EPS_DEG
from diffpes.types import ScalarBool, ScalarFloat

from .spectral_resolvent import _checked_resolvent_scalars


def _checked_eigenvalue_domain(
    eigenvalues_ev: Float64[Array, "... n_bands"],
    allow_degenerate_value_only: ScalarBool,
    *,
    context: str,
) -> Float64[Array, "... n_bands"]:
    """PRIVATE: Enforce the differentiated eigen-path gap floor.

    Parameters
    ----------
    eigenvalues_ev : Float64[Array, "... n_bands"]
        Finite eigenvalues in eV, with any leading batch axes.
    allow_degenerate_value_only : ScalarBool
        Whether to admit a degenerate primal with no derivative claim.
    context : str
        Public caller name included in a rejection message.

    Returns
    -------
    checked : Float64[Array, "... n_bands"]
        Eigenvalues carrying the traced nondegenerate-domain guard.
    """
    if eigenvalues_ev.shape[-1] < 2:  # noqa: PLR2004 -- a gap needs a pair.
        return eigenvalues_ev
    minimum_gap_ev: float = 1.0e3 * EPS_DEG
    sorted_eigenvalues: Float64[Array, "... n_bands"] = jnp.sort(
        eigenvalues_ev,
        axis=-1,
    )
    adjacent_gaps: Float64[Array, "... n_gap"] = jnp.diff(
        sorted_eigenvalues,
        axis=-1,
    )
    minimum_gap: Float64[Array, ""] = jnp.min(adjacent_gaps)
    enforce_gap: Bool[Array, ""] = ~jnp.asarray(
        allow_degenerate_value_only,
        dtype=jnp.bool_,
    )
    checked: Float64[Array, "... n_bands"] = eqx.error_if(
        eigenvalues_ev,
        enforce_gap & (minimum_gap < minimum_gap_ev),
        f"{context}: differentiated eigen path requires every adjacent band "
        f"gap to be at least {minimum_gap_ev:.1e} eV; use the "
        "resolvent for gradients or set allow_degenerate_value_only=True "
        "only for primal evaluation",
    )
    return checked


def _spectral_intensity_eigen_unchecked(
    eigenvalues_rel_fermi_ev: Float64[Array, " n_bands"],
    band_weights: Float64[Array, " n_bands"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Sum already-validated Lorentzian band contributions.

    Notes
    -----
    The caller validates weights, eigenvalues, and linewidth. This helper
    contains only the normalized Lorentzian arithmetic.
    """
    linewidth: Float64[Array, ""] = eta - jnp.imag(sigma_omega)
    displacement: Float64[Array, " n_bands"] = (
        omega_rel_fermi_ev - eigenvalues_rel_fermi_ev - jnp.real(sigma_omega)
    )
    intensity: Float64[Array, ""] = jnp.sum(
        band_weights * linewidth / (jnp.pi * (displacement**2 + linewidth**2))
    )
    return intensity


@jaxtyped(typechecker=beartype)
def spectral_intensity_eigen(  # noqa: DOC502 -- traced domain guards.
    eigenvalues_rel_fermi_ev: Float64[Array, " n_bands"],
    band_weights: Float64[Array, " n_bands"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
    *,
    allow_degenerate_value_only: ScalarBool = False,
) -> Float64[Array, ""]:
    """Evaluate spectral intensity from eigenvalues and invariant weights.

    This fast path sums one normalized Lorentzian per band. Its inputs are
    gauge-invariant band weights, so raw eigenvector phases never reach the
    observable. The resolvent path remains the certified choice at an exact
    degeneracy.

    :see: :class:`~.test_spectral_eigen.TestSpectralIntensityEigen`

    Parameters
    ----------
    eigenvalues_rel_fermi_ev : Float64[Array, " n_bands"]
        Band energies relative to the Fermi level in eV.
    band_weights : Float64[Array, " n_bands"]
        Finite, nonnegative gauge-invariant transition weights.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive regulator in eV.
    allow_degenerate_value_only : ScalarBool, optional
        Admit an exact or near-degenerate primal without certifying JVPs,
        VJPs, finite differences, or Hamiltonian-parameter derivatives.
        Default is ``False``.

    Returns
    -------
    intensity : Float64[Array, ""]
        Intrinsic spectral intensity in inverse eV.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, a band weight is negative, or the total
        linewidth is not strictly positive. Also raised when the minimum band
        gap is below ``1e3 * EPS_DEG`` unless value-only evaluation is
        explicit.

    Notes
    -----
    The linewidth is exactly ``eta - imag(sigma_omega)`` and the pole
    displacement is ``omega - eigenvalue - real(sigma_omega)``. Equal poles
    have a gauge-invariant primal when their supplied weights form a complete
    invariant group. Only the resolvent path owns derivatives at such a
    degeneracy.
    """
    checked_eigenvalues: Float64[Array, " n_bands"] = eqx.error_if(
        eigenvalues_rel_fermi_ev,
        ~jnp.all(jnp.isfinite(eigenvalues_rel_fermi_ev)),
        "spectral_intensity_eigen: eigenvalues must be finite",
    )
    checked_eigenvalues = _checked_eigenvalue_domain(
        checked_eigenvalues,
        allow_degenerate_value_only,
        context="spectral_intensity_eigen",
    )
    checked_weights: Float64[Array, " n_bands"] = eqx.error_if(
        band_weights,
        ~jnp.all(jnp.isfinite(band_weights) & (band_weights >= 0.0)),
        "spectral_intensity_eigen: band_weights must be finite and "
        "nonnegative",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="spectral_intensity_eigen",
    )
    intensity: Float64[Array, ""] = _spectral_intensity_eigen_unchecked(
        checked_eigenvalues,
        checked_weights,
        omega_checked,
        sigma_checked,
        eta_checked,
    )
    return intensity


__all__: list[str] = [
    "spectral_intensity_eigen",
]
