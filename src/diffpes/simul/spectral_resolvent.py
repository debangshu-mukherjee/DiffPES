"""Evaluate degeneracy-safe resolvent spectral observables.

Extended Summary
----------------
This module uses complex128 linear solves for gauge-safe spectral intensities.

Routine Listings
----------------
:func:`projected_spectral_density_resolvent`
    Compute the projected Hermitian resolvent spectral density.
:func:`spectral_intensity_resolvent`
    Evaluate degeneracy-safe spectral intensity through a linear solve.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import EPS
from diffpes.types import ScalarFloat


def _checked_spectral_hamiltonian(
    hamiltonian: Complex128[Array, "n_orb n_orb"],
    *,
    context: str,
) -> Complex128[Array, "n_orb n_orb"]:
    """PRIVATE: Validate one finite Hermitian Hamiltonian.

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_orb n_orb"]
        Candidate Hamiltonian in eV.
    context : str
        Public caller name used in error messages.

    Returns
    -------
    checked : Complex128[Array, "n_orb n_orb"]
        Unchanged Hamiltonian carrying both runtime guards.

    Notes
    -----
    The types-owned ``EPS`` tolerance matches the Hermitian validation
    used by the tight-binding eigensolver. Both checks survive JIT.
    """
    checked: Complex128[Array, "n_orb n_orb"] = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.isfinite(hamiltonian)),
        f"{context}: Hamiltonian entries must be finite",
    )
    checked = eqx.error_if(
        checked,
        ~jnp.allclose(checked, checked.conj().T, rtol=EPS, atol=EPS),
        f"{context}: Hamiltonian must be Hermitian",
    )
    return checked  # noqa: RET504 -- the returned value carries both guards.


def _checked_resolvent_scalars(
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
    *,
    context: str,
) -> Tuple[
    Float64[Array, ""],
    Complex128[Array, ""],
    Float64[Array, ""],
]:
    """PRIVATE: Validate one retarded resolvent coordinate.

    Parameters
    ----------
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Positive regulator in eV.
    context : str
        Public caller name used in error messages.

    Returns
    -------
    checked : Tuple[Float64[Array, ""], Complex128[Array, ""],
        Float64[Array, ""]]
        Finite sampled energy, retarded self-energy with a positive total
        linewidth, and a positive float64 regulator.

    Notes
    -----
    The physical denominator width is ``eta - imag(sigma)``. Requiring it
    to remain positive rejects an advanced or singular resolvent.
    """
    omega_checked: Float64[Array, ""] = eqx.error_if(
        omega_rel_fermi_ev,
        ~jnp.isfinite(omega_rel_fermi_ev),
        f"{context}: omega must be finite",
    )
    eta_array: Float64[Array, ""] = jnp.asarray(eta, dtype=jnp.float64)
    eta_checked: Float64[Array, ""] = eqx.error_if(
        eta_array,
        ~jnp.isfinite(eta_array) | (eta_array <= 0.0),
        f"{context}: eta must be finite and strictly positive",
    )
    sigma_checked: Complex128[Array, ""] = eqx.error_if(
        sigma_omega,
        ~jnp.isfinite(sigma_omega),
        f"{context}: sigma_omega must be finite",
    )
    sigma_checked = eqx.error_if(
        sigma_checked,
        jnp.imag(sigma_checked) > 0.0,
        f"{context}: retarded sigma_omega must have a nonpositive "
        "imaginary part",
    )
    sigma_checked = eqx.error_if(
        sigma_checked,
        eta_checked - jnp.imag(sigma_checked) <= 0.0,
        f"{context}: eta - imag(sigma_omega) must be strictly positive",
    )
    checked: Tuple[
        Float64[Array, ""],
        Complex128[Array, ""],
        Float64[Array, ""],
    ] = (omega_checked, sigma_checked, eta_checked)
    return checked


def _resolvent_solution(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    source: Complex128[Array, " n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Complex128[Array, " n_orb"]:
    """PRIVATE: Apply the complex128 retarded resolvent to one source.

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Hermitian Hamiltonian relative to the Fermi level in eV.
    source : Complex128[Array, " n_orb"]
        Right-hand side source ket.
    omega_rel_fermi_ev : Float64[Array, ""]
        Relative sampled energy in eV.
    sigma_omega : Complex128[Array, ""]
        Retarded self-energy at that energy in eV.
    eta : Float64[Array, ""]
        Positive regulator in eV.

    Returns
    -------
    solution : Complex128[Array, " n_orb"]
        ``((omega + i*eta - sigma) I - H)^{-1} source``.

    Notes
    -----
    Lineax owns the transpose rule, so reverse mode uses the corresponding
    adjoint solve without a hand-written custom derivative.
    """
    identity: Complex128[Array, "n_orb n_orb"] = jnp.eye(
        hamiltonian_rel_fermi_k.shape[0], dtype=jnp.complex128
    )
    operator_matrix: Complex128[Array, "n_orb n_orb"] = (
        omega_rel_fermi_ev + 1.0j * eta - sigma_omega
    ) * identity - hamiltonian_rel_fermi_k
    solution: Complex128[Array, " n_orb"] = lx.linear_solve(
        lx.MatrixLinearOperator(operator_matrix),
        source,
        lx.LU(),
    ).value
    return solution


def _spectral_intensity_resolvent_unchecked(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_source: Complex128[Array, " n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate one already-validated resolvent quadratic form.

    Notes
    -----
    The caller owns all domain checks. This helper performs one complex128
    solve and contracts the source with its response.
    """
    solution: Complex128[Array, " n_orb"] = _resolvent_solution(
        hamiltonian_rel_fermi_k,
        transition_source,
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
    )
    intensity: Float64[Array, ""] = (
        -jnp.imag(jnp.vdot(transition_source, solution)) / jnp.pi
    )
    return intensity


def _summed_spectral_intensity_resolvent_unchecked(
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_sources: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate every outgoing source before incoherent reduction.

    Notes
    -----
    Vectorization applies the scalar resolvent to each source separately.
    The helper sums only after it forms each real quadratic response.
    """
    per_output: Float64[Array, " n_out"] = jax.vmap(
        _spectral_intensity_resolvent_unchecked,
        in_axes=(None, 0, None, None, None),
    )(
        hamiltonian_rel_fermi_k,
        transition_sources,
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
    )
    intensity: Float64[Array, ""] = jnp.sum(per_output)
    return intensity


@jaxtyped(typechecker=beartype)
def spectral_intensity_resolvent(  # noqa: DOC502, DOC503 -- traced guards.
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_sources: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
) -> Float64[Array, ""]:
    r"""Evaluate degeneracy-safe spectral intensity through a linear solve.

    For every outgoing channel :math:`\alpha`, the primitive computes
    :math:`-\operatorname{Im}[s_\alpha^\dagger G(\omega)s_\alpha]/\pi`,
    where :math:`G=[(\omega+i\eta-\Sigma)I-H]^{-1}`, and then sums the real
    responses. It never coherently combines sources before solving and never
    differentiates an eigenvector, so exact band degeneracies remain regular.

    :see: :class:`~.test_spectral_resolvent.TestSpectralIntensityResolvent`

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian Hamiltonian relative to the Fermi level in eV.
    transition_sources : Complex128[Array, "n_out n_orb"]
        Nonempty outgoing-channel source kets ``s = d.conj()`` from the
        matrix-element seam. ``n_out=1`` is the spinless case.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive resolvent regulator in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        Intrinsic spectral intensity in inverse eV.

    Raises
    ------
    ValueError
        If the outgoing-channel axis is empty.
    EquinoxRuntimeError
        If an input is non-finite, the Hamiltonian is non-Hermitian, or the
        total linewidth is not strictly positive.

    Notes
    -----
    Each source enters an independent scalar-RHS solve. The contraction uses
    :func:`jax.numpy.vdot`, not ``dot``. The helper reduces only after forming
    the real quadratic responses. Lineax keeps the operator, right-hand side,
    LU factorization, and result in complex128. It supplies exact forward- and
    reverse-mode rules.
    """
    if transition_sources.shape[0] == 0:
        raise ValueError("transition_sources n_out axis must be nonempty")
    checked_hamiltonian: Complex128[Array, "n_orb n_orb"] = (
        _checked_spectral_hamiltonian(
            hamiltonian_rel_fermi_k,
            context="spectral_intensity_resolvent",
        )
    )
    checked_sources: Complex128[Array, "n_out n_orb"] = eqx.error_if(
        transition_sources,
        ~jnp.all(jnp.isfinite(transition_sources)),
        "spectral_intensity_resolvent: transition_sources must be finite",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="spectral_intensity_resolvent",
    )
    intensity: Float64[Array, ""] = (
        _summed_spectral_intensity_resolvent_unchecked(
            checked_hamiltonian,
            checked_sources,
            omega_checked,
            sigma_checked,
            eta_checked,
        )
    )
    return intensity


@jaxtyped(typechecker=beartype)
def projected_spectral_density_resolvent(  # noqa: DOC502 -- traced guards.
    hamiltonian_rel_fermi_k: Complex128[Array, "n_orb n_orb"],
    transition_operator: Complex128[Array, "n_out n_orb"],
    omega_rel_fermi_ev: Float64[Array, ""],
    sigma_omega: Complex128[Array, ""],
    eta: ScalarFloat,
) -> Complex128[Array, "n_out n_out"]:
    r"""Compute the projected Hermitian resolvent spectral density.

    The returned matrix is
    :math:`D[-(G-G^\dagger)/(2\pi i)]D^\dagger`. This polynomial projector
    form preserves off-diagonal spin and channel coherences at degeneracies.

    :see: :class:`.TestProjectedSpectralDensityResolvent`

    Parameters
    ----------
    hamiltonian_rel_fermi_k : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian Hamiltonian relative to the Fermi level in eV.
    transition_operator : Complex128[Array, "n_out n_orb"]
        Output-channel rows ``D`` in the orbital basis.
    omega_rel_fermi_ev : Float64[Array, ""]
        Sampled energy relative to the Fermi level in eV.
    sigma_omega : Complex128[Array, ""]
        Complex retarded self-energy at the sampled energy in eV.
    eta : ScalarFloat
        Finite, strictly positive regulator in eV.

    Returns
    -------
    density : Complex128[Array, "n_out n_out"]
        Hermitian positive-semidefinite projected spectral density.

    Raises
    ------
    EquinoxRuntimeError
        If an input is non-finite, the Hamiltonian is non-Hermitian, or the
        total linewidth is not strictly positive.

    Notes
    -----
    A static ``vmap`` applies the same Lineax operator to every column of
    ``D.dagger``. Antisymmetrizing the projected Green function as a matrix
    preserves its off-diagonal coherences. An elementwise imaginary part
    corrupts them.
    """
    checked_hamiltonian: Complex128[Array, "n_orb n_orb"] = (
        _checked_spectral_hamiltonian(
            hamiltonian_rel_fermi_k,
            context="projected_spectral_density_resolvent",
        )
    )
    checked_operator: Complex128[Array, "n_out n_orb"] = eqx.error_if(
        transition_operator,
        ~jnp.all(jnp.isfinite(transition_operator)),
        "projected_spectral_density_resolvent: transition_operator "
        "must be finite",
    )
    omega_checked: Float64[Array, ""]
    sigma_checked: Complex128[Array, ""]
    eta_checked: Float64[Array, ""]
    omega_checked, sigma_checked, eta_checked = _checked_resolvent_scalars(
        omega_rel_fermi_ev,
        sigma_omega,
        eta,
        context="projected_spectral_density_resolvent",
    )
    right_hand_sides: Complex128[Array, "n_orb n_out"] = (
        checked_operator.conj().T
    )
    solved: Complex128[Array, "n_orb n_out"] = jax.vmap(
        lambda source: _resolvent_solution(
            checked_hamiltonian,
            source,
            omega_checked,
            sigma_checked,
            eta_checked,
        ),
        in_axes=1,
        out_axes=1,
    )(right_hand_sides)
    projected_green: Complex128[Array, "n_out n_out"] = (
        checked_operator @ solved
    )
    density: Complex128[Array, "n_out n_out"] = -(
        projected_green - projected_green.conj().T
    ) / (2.0j * jnp.pi)
    return density


__all__: list[str] = [
    "projected_spectral_density_resolvent",
    "spectral_intensity_resolvent",
]
