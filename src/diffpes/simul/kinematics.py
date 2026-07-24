r"""Compute free-electron photoemission kinematics.

Extended Summary
----------------
This module maps photon energy, Fermi-relative electron energy, and detector
angles to photoelectron momenta. It implements the free-electron final-state
model with an inner potential.
All array operations support JAX transformations.

The work function and Fermi-relative energy enter kinematics only through
``photon_energy_ev - work_function_ev + omega_rel_fermi_ev``. Consequently,
``J_work_function = -J_omega`` exactly, and a photon-energy scan does not lift
that gauge without an external energy or work-function reference. The scan
can still constrain the inner potential through the ``kz`` dispersion.

Routine Listings
----------------
:func:`detector_angles_to_kpar`
    Convert detector angles to parallel momentum.
:func:`emission_angles`
    Convert Cartesian momentum to emission angles.
:func:`final_state_k_inv_ang`
    Convert kinetic energy to momentum and return its validity mask.
:func:`kinetic_energy_ev`
    Compute signed photoelectron kinetic energy and its validity mask.
:func:`kpar_to_detector_angles`
    Convert parallel momentum to detector angles.
:func:`kz_from_inner_potential`
    Compute complex out-of-plane momentum from the inner potential.
:func:`kz_from_inner_potential_at_fermi`
    Evaluate the named Fermi-level ``kz`` approximation.

Notes
-----
Public angles use radians because they are detector-frame coordinates. The
horizontal slit uses ``Rx(ty) @ Ry(tx)``. The vertical slit uses
``Rx(tx) @ Ry(ty)``.

DiffPES maps Chinook's horizontal ``tilt.k_mesh`` angles as ``T=-tx`` and
``P=ty``. For the vertical slit, the corresponding mapping is
``T=-ty, P=tx``. These mappings define one active detector frame.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from diffpes.maths import safe_arctan2, safe_divide, safe_norm, safe_sqrt
from diffpes.types import (
    K_PREFACTOR_INV_ANG_SQRT_EV,
    TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2,
    ScalarFloat,
)


@jaxtyped(typechecker=beartype)
def kinetic_energy_ev(
    photon_energy_ev: ScalarFloat,
    work_function_ev: ScalarFloat,
    omega_rel_fermi_ev: Float[Array, " ..."],
) -> Tuple[Float[Array, " ..."], Bool[Array, " ..."]]:
    r"""Compute signed photoelectron kinetic energy and its validity mask.

    The function applies energy conservation in the three-step photoemission
    model [3]_. It does not alter forbidden energies.

    :see: :class:`~.test_kinematics.TestKineticEnergyEv`

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    omega_rel_fermi_ev : Float[Array, " ..."]
        Electron energy relative to the Fermi level, ``E - E_F``, in eV.

    Returns
    -------
    kinetic_energies : Float[Array, " ..."]
        Signed kinetic energies in eV.
    emission_valid : Bool[Array, " ..."]
        Mask selecting strictly positive kinetic energies.

    Notes
    -----
    The frozen convention is
    :math:`E_{\mathrm{kin}}=h\nu-W+(E-E_F)`. The validity mask rejects zero
    and negative energies. The returned raw energy retains unit derivative on
    both sides of the threshold. The Boolean mask carries nondifferentiable
    metadata.

    References
    ----------
    .. [3] A. Damascelli, Z. Hussain, and Z.-X. Shen, Rev. Mod. Phys. 75,
       473 (2003).
    """
    photon_energy_array: Float[Array, ""] = jnp.asarray(photon_energy_ev)
    work_function_array: Float[Array, ""] = jnp.asarray(work_function_ev)
    kinetic_energies: Float[Array, " ..."] = (
        photon_energy_array + omega_rel_fermi_ev - work_function_array
    )
    emission_valid: Bool[Array, " ..."] = kinetic_energies > 0.0
    result: Tuple[Float[Array, " ..."], Bool[Array, " ..."]] = (
        kinetic_energies,
        emission_valid,
    )
    return result


@jaxtyped(typechecker=beartype)
def final_state_k_inv_ang(
    kinetic_energy_ev: Float[Array, " ..."],
) -> Tuple[Float[Array, " ..."], Bool[Array, " ..."]]:
    """Convert kinetic energy to momentum and return its validity mask.

    The function applies the free-electron dispersion. Forbidden inputs map
    to a zero sentinel rather than a fabricated positive momentum.

    :see: :class:`~.test_kinematics.TestFinalStateKInvAng`

    Parameters
    ----------
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.

    Returns
    -------
    momentum_magnitudes : Float[Array, " ..."]
        Final-state momentum magnitudes in 1/Angstrom.
    emission_valid : Bool[Array, " ..."]
        Mask selecting strictly positive kinetic energies.

    Notes
    -----
    The function computes ``K_PREFACTOR_INV_ANG_SQRT_EV * sqrt(E_kin)`` on
    the physical domain. At and below threshold it returns zero with a zero
    selected derivative. Consumers must propagate ``emission_valid`` rather
    than interpreting the zero sentinel as an emitted state.
    """
    emission_valid: Bool[Array, " ..."] = kinetic_energy_ev > 0.0
    sanitized_energies: Float[Array, " ..."] = jnp.where(
        emission_valid,
        kinetic_energy_ev,
        1.0,
    )
    rooted_momenta: Float[Array, " ..."] = (
        K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(sanitized_energies)
    )
    momentum_magnitudes: Float[Array, " ..."] = jnp.where(
        emission_valid,
        rooted_momenta,
        0.0,
    )
    result: Tuple[Float[Array, " ..."], Bool[Array, " ..."]] = (
        momentum_magnitudes,
        emission_valid,
    )
    return result


@jaxtyped(typechecker=beartype)
def kz_from_inner_potential(
    photon_energy_ev: ScalarFloat,
    work_function_ev: ScalarFloat,
    inner_potential_ev: ScalarFloat,
    omega_rel_fermi_ev: Float[Array, " ..."],
    k_par_inv_ang: Float[Array, " ..."],
) -> Tuple[Complex[Array, " ..."], Bool[Array, " ..."]]:
    r"""Compute complex out-of-plane momentum from the inner potential.

    The function implements the free-electron final-state approximation [4]_.
    Its principal complex root retains evanescent channels.

    :see: :class:`~.test_kinematics.TestKzFromInnerPotential`

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    inner_potential_ev : ScalarFloat
        Inner potential in eV.
    omega_rel_fermi_ev : Float[Array, " ..."]
        Electron energy relative to the Fermi level, ``E - E_F``, in eV.
    k_par_inv_ang : Float[Array, " ..."]
        Parallel momentum magnitudes in 1/Angstrom.

    Returns
    -------
    kz_values : Complex[Array, " ..."]
        Principal out-of-plane momenta in 1/Angstrom.
    propagating : Bool[Array, " ..."]
        Mask requiring both valid photoemission and a positive real radicand.

    Notes
    -----
    The radicand equals
    :math:`(2m_e/\hbar^2)(h\nu-W+\omega+V_0)-k_\parallel^2`.
    Negative radicands give positive imaginary roots. The propagation mask
    also rejects forbidden surface emission. The branch point has no assigned
    derivative.

    For a propagating channel,
    :math:`\partial k_z/\partial V_0=(2\,\hbar^2/2m_e)^{-1}/k_z`.

    References
    ----------
    .. [4] A. Damascelli, Z. Hussain, and Z.-X. Shen, Rev. Mod. Phys. 75,
       473 (2003).
    """
    surface_kinetic_energies: Float[Array, " ..."]
    energy_valid: Bool[Array, " ..."]
    surface_kinetic_energies, energy_valid = kinetic_energy_ev(
        photon_energy_ev,
        work_function_ev,
        omega_rel_fermi_ev,
    )
    inner_potential_array: Float[Array, ""] = jnp.asarray(inner_potential_ev)
    surface_aperture_valid: Bool[Array, " ..."] = energy_valid & (
        k_par_inv_ang * k_par_inv_ang
        < TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2 * surface_kinetic_energies
    )
    radicand: Float[Array, " ..."] = (
        TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2
        * (surface_kinetic_energies + inner_potential_array)
        - k_par_inv_ang * k_par_inv_ang
    )
    propagating: Bool[Array, " ..."] = surface_aperture_valid & (
        radicand > 0.0
    )
    sanitized_radicand: Float[Array, " ..."] = jnp.where(
        surface_aperture_valid,
        radicand,
        1.0,
    )
    complex_radicand: Complex[Array, " ..."] = sanitized_radicand.astype(
        jnp.complex128
    )
    rooted_values: Complex[Array, " ..."] = jnp.sqrt(complex_radicand)
    kz_values: Complex[Array, " ..."] = jnp.where(
        surface_aperture_valid,
        rooted_values,
        jnp.zeros_like(rooted_values),
    )
    kinematics_result: Tuple[Complex[Array, " ..."], Bool[Array, " ..."]] = (
        kz_values,
        propagating,
    )
    return kinematics_result


@jaxtyped(typechecker=beartype)
def kz_from_inner_potential_at_fermi(
    photon_energy_ev: ScalarFloat,
    work_function_ev: ScalarFloat,
    inner_potential_ev: ScalarFloat,
    k_par_inv_ang: Float[Array, " ..."],
) -> Tuple[Complex[Array, " ..."], Bool[Array, " ..."]]:
    """Evaluate the named Fermi-level ``kz`` approximation.

    This compatibility helper evaluates :func:`kz_from_inner_potential` at
    ``omega_rel_fermi_ev = 0``. Production paths with an energy axis must call
    the exact function and supply that axis.

    :see: :class:`~.test_kinematics.TestKzFromInnerPotentialAtFermi`

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    inner_potential_ev : ScalarFloat
        Inner potential in eV.
    k_par_inv_ang : Float[Array, " ..."]
        Parallel momentum magnitudes in 1/Angstrom.

    Returns
    -------
    kz_values : Complex[Array, " ..."]
        Principal at-Fermi out-of-plane momenta in 1/Angstrom.
    propagating : Bool[Array, " ..."]
        Physical propagation mask.

    Notes
    -----
    The helper constructs a zero energy array and delegates all validity,
    aperture, and branch handling to the exact function.
    """
    omega_at_fermi: Float[Array, " ..."] = jnp.zeros_like(k_par_inv_ang)
    result: Tuple[Complex[Array, " ..."], Bool[Array, " ..."]] = (
        kz_from_inner_potential(
            photon_energy_ev,
            work_function_ev,
            inner_potential_ev,
            omega_at_fermi,
            k_par_inv_ang,
        )
    )
    return result


@jaxtyped(typechecker=beartype)
def emission_angles(
    k_cart_inv_ang: Float[Array, "... 3"],
) -> Tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Convert Cartesian momentum to emission angles.

    The function returns the polar angle from positive z and the azimuth from
    positive x. It selects zero azimuth at normal emission.

    :see: :class:`~.test_kinematics.TestEmissionAngles`

    Parameters
    ----------
    k_cart_inv_ang : Float[Array, "... 3"]
        Cartesian momentum vectors in 1/Angstrom.

    Returns
    -------
    theta : Float[Array, " ..."]
        Polar emission angles in radians.
    phi : Float[Array, " ..."]
        Azimuthal emission angles in radians.

    Notes
    -----
    The polar angle uses ``arctan2(norm([kx, ky]), kz)``. The azimuth uses
    ``arctan2(ky, kx)``. Safe primitives give zero coordinate gradients at
    their undefined origins.
    """
    k_parallel: Float[Array, " ..."] = safe_norm(k_cart_inv_ang[..., :2])
    theta: Float[Array, " ..."] = safe_arctan2(
        k_parallel,
        k_cart_inv_ang[..., 2],
    )
    phi: Float[Array, " ..."] = safe_arctan2(
        k_cart_inv_ang[..., 1],
        k_cart_inv_ang[..., 0],
    )
    angles: Tuple[Float[Array, " ..."], Float[Array, " ..."]] = (theta, phi)
    return angles


@jaxtyped(typechecker=beartype)
def detector_angles_to_kpar(
    tx: Float[Array, " ..."],
    ty: Float[Array, " ..."],
    kinetic_energy_ev: Float[Array, " ..."],
    slit: str,
) -> Float[Array, "... 2"]:
    """Convert detector angles to parallel momentum.

    The function rotates the positive z direction with the Plan 03 detector
    convention. It broadcasts all traced inputs over their leading axes.

    :see: :class:`~.test_kinematics.TestDetectorAnglesToKpar`

    Parameters
    ----------
    tx : Float[Array, " ..."]
        First detector angles in radians.
    ty : Float[Array, " ..."]
        Second detector angles in radians.
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.
    slit : str
        Static slit orientation, ``"H"`` or ``"V"``. A change causes
        retracing.

    Returns
    -------
    k_parallel : Float[Array, "... 2"]
        Cartesian parallel momenta ``(kx, ky)`` in 1/Angstrom.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    The horizontal slit uses ``Rx(ty) @ Ry(tx)``. The vertical slit uses
    ``Rx(tx) @ Ry(ty)``. These active rotations act on the positive z vector.
    The public chart requires strictly positive kinetic energy and both angles
    in ``(-pi/2, pi/2)``.
    """
    if slit not in {"H", "V"}:
        message: str = "slit must be 'H' or 'V'"
        raise ValueError(message)
    broadcast_tx: Float[Array, " ..."]
    broadcast_ty: Float[Array, " ..."]
    broadcast_energy: Float[Array, " ..."]
    broadcast_tx, broadcast_ty, broadcast_energy = jnp.broadcast_arrays(
        tx,
        ty,
        kinetic_energy_ev,
    )
    chart_valid: Bool[Array, " ..."] = (
        (broadcast_energy > 0.0)
        & (jnp.abs(broadcast_tx) < jnp.pi / 2.0)
        & (jnp.abs(broadcast_ty) < jnp.pi / 2.0)
    )
    checked_tx: Float[Array, " ..."] = eqx.error_if(
        broadcast_tx,
        ~jnp.all(chart_valid),
        (
            "detector_angles_to_kpar requires Ekin > 0 and "
            "tx, ty in (-pi/2, pi/2)"
        ),
    )
    checked_ty: Float[Array, " ..."] = jnp.where(
        chart_valid,
        broadcast_ty,
        0.0,
    )
    momentum_magnitudes: Float[Array, " ..."]
    momentum_magnitudes, _ = final_state_k_inv_ang(broadcast_energy)
    if slit == "H":
        kx: Float[Array, " ..."] = momentum_magnitudes * jnp.sin(checked_tx)
        ky: Float[Array, " ..."] = (
            -momentum_magnitudes * jnp.cos(checked_tx) * jnp.sin(checked_ty)
        )
    else:
        kx = momentum_magnitudes * jnp.sin(checked_ty)
        ky = -momentum_magnitudes * jnp.sin(checked_tx) * jnp.cos(checked_ty)
    k_parallel: Float[Array, "... 2"] = jnp.stack((kx, ky), axis=-1)
    return k_parallel


@jaxtyped(typechecker=beartype)
def kpar_to_detector_angles(
    k_par_inv_ang: Float[Array, "... 2"],
    kinetic_energy_ev: Float[Array, " ..."],
    slit: str,
) -> Tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Convert parallel momentum to detector angles.

    The function gives the exact inverse detector map on the physical domain.
    This domain requires ``norm(k_parallel) < k_f``.

    :see: :class:`~.test_kinematics.TestKparToDetectorAngles`

    Parameters
    ----------
    k_par_inv_ang : Float[Array, "... 2"]
        Cartesian parallel momenta ``(kx, ky)`` in 1/Angstrom.
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.
    slit : str
        Static slit orientation, ``"H"`` or ``"V"``. A change causes
        retracing.

    Returns
    -------
    tx : Float[Array, " ..."]
        First detector angles in radians.
    ty : Float[Array, " ..."]
        Second detector angles in radians.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    The inverse uses the positive detector-normal branch, corresponding to
    ``tx, ty`` in ``(-pi/2, pi/2)``. It rejects nonpositive kinetic energy and
    ``norm(k_parallel) >= k_f``. The function fabricates no boundary value
    outside that open chart.
    """
    if slit not in {"H", "V"}:
        message: str = "slit must be 'H' or 'V'"
        raise ValueError(message)
    target_shape: tuple[int, ...] = jnp.broadcast_shapes(
        k_par_inv_ang.shape[:-1],
        kinetic_energy_ev.shape,
    )
    broadcast_k_parallel: Float[Array, "... 2"] = jnp.broadcast_to(
        k_par_inv_ang,
        (*target_shape, 2),
    )
    broadcast_energy: Float[Array, " ..."] = jnp.broadcast_to(
        kinetic_energy_ev,
        target_shape,
    )
    momentum_magnitudes: Float[Array, " ..."]
    emission_valid: Bool[Array, " ..."]
    momentum_magnitudes, emission_valid = final_state_k_inv_ang(
        broadcast_energy
    )
    aperture_sq: Float[Array, " ..."] = jnp.sum(
        broadcast_k_parallel * broadcast_k_parallel,
        axis=-1,
    )
    momentum_sq: Float[Array, " ..."] = (
        momentum_magnitudes * momentum_magnitudes
    )
    chart_valid: Bool[Array, " ..."] = emission_valid & (
        aperture_sq < momentum_sq
    )
    checked_k_parallel: Float[Array, "... 2"] = eqx.error_if(
        broadcast_k_parallel,
        ~jnp.all(chart_valid),
        "kpar_to_detector_angles requires Ekin > 0 and norm(kpar) < kf",
    )
    normalized_k_parallel: Float[Array, "... 2"] = safe_divide(
        checked_k_parallel,
        momentum_magnitudes[..., None],
    )
    normalized_kx: Float[Array, " ..."] = normalized_k_parallel[..., 0]
    normalized_ky: Float[Array, " ..."] = normalized_k_parallel[..., 1]
    normal_component: Float[Array, " ..."] = jnp.sqrt(
        jnp.maximum(
            1.0
            - normalized_kx * normalized_kx
            - normalized_ky * normalized_ky,
            0.0,
        )
    )
    if slit == "H":
        tx: Float[Array, " ..."] = safe_arctan2(
            normalized_kx,
            safe_sqrt(1.0 - normalized_kx * normalized_kx),
        )
        ty: Float[Array, " ..."] = safe_arctan2(
            -normalized_ky,
            normal_component,
        )
    else:
        tx = safe_arctan2(-normalized_ky, normal_component)
        ty = safe_arctan2(
            normalized_kx,
            safe_sqrt(1.0 - normalized_kx * normalized_kx),
        )
    detector_angles: Tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
        tx,
        ty,
    )
    return detector_angles


__all__: list[str] = [
    "detector_angles_to_kpar",
    "emission_angles",
    "final_state_k_inv_ang",
    "kinetic_energy_ev",
    "kpar_to_detector_angles",
    "kz_from_inner_potential",
    "kz_from_inner_potential_at_fermi",
]
