"""Certify complete-shell covariance and named radial gauges.

Extended Summary
----------------
The tests rotate complete p and d shells through every tensor leg. They also
verify the two named radial coefficient-scale gauge directions.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.maths import (
    channel_tables,
    real_harmonic_unitary,
    wigner_d,
)
from diffpes.radial import radial_bvals
from diffpes.simul import (
    contract_polarization,
    orbital_transition_channels,
    pack_matrixel_params,
    radial_coefficient_scale_gauge_directions,
    unpack_matrixel_params,
)
from diffpes.types import (
    MatrixElementParams,
    OrbitalBasis,
    PyTreeDef,
    RadialSpec,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)


def _complete_pd_fixture() -> Tuple[
    OrbitalBasis,
    Tuple[int, ...],
    RadialSpec,
    MatrixElementParams,
]:
    """PRIVATE: Return complete p and d shells with normalized contractions.

    Returns
    -------
    basis : OrbitalBasis
        One-atom basis with the complete n=2 p and n=3 d shells.
    shell_index : Tuple[int, ...]
        Shell assignment mapping the three p orbitals to shell 0 and
        the five d orbitals to shell 1.
    radial : RadialSpec
        Two-term contracted Slater radial spec for both shells.
    params : MatrixElementParams
        Parameter carrier with per-shell sigma scales and four phase
        shift angles.

    Notes
    -----
    Fixes generic zeta pairs, contraction coefficients, sigma scales,
    and phase-shift angles so no shell or channel is degenerate.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 8,
        n=(2,) * 3 + (3,) * 5,
        l=(1,) * 3 + (2,) * 5,
        m=(-1, 0, 1, -2, -1, 0, 1, 2),
    )
    shell_index: Tuple[int, ...] = (0, 0, 0, 1, 1, 1, 1, 1)
    radial: RadialSpec = make_radial_spec(
        basis,
        shell_index,
        zeta_shell=jnp.asarray(((0.8, 1.7), (0.65, 1.35))),
        coefficients_shell=jnp.asarray(((0.7, -0.3), (0.4, 0.9))),
    )
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        shell_index,
        sigma_shell=jnp.asarray((1.2, 0.75)),
        phase_shift_angles_shell=jnp.asarray((0.1, -0.3, 0.4, -0.2)),
    )
    returned: Tuple[
        OrbitalBasis,
        Tuple[int, ...],
        RadialSpec,
        MatrixElementParams,
    ] = basis, shell_index, radial, params
    return returned


def _real_wigner(
    degree: int,
    angles: Tuple[float, float, float],
) -> Float64[Array, "m1 m2"]:
    """PRIVATE: Return one Wigner rotation in the real-harmonic basis.

    Parameters
    ----------
    degree : int
        Tensor degree l of the rotated shell.
    angles : Tuple[float, float, float]
        The z-y-z Euler angles in radians.

    Returns
    -------
    real_rotation : Float64[Array, "m1 m2"]
        Real rotation matrix acting on the real harmonics of the
        degree.

    Notes
    -----
    Conjugates the complex Wigner D-matrix with the real-harmonic
    unitary and takes the real part of U.conj() @ D @ U.T.
    """
    unitary: Complex128[Array, "m1 m2"] = real_harmonic_unitary(degree)
    complex_rotation: Complex128[Array, "m1 m2"] = wigner_d(degree, *angles)
    real_rotation: Float64[Array, "m1 m2"] = jnp.real(
        unitary.conj() @ complex_rotation @ unitary.T
    )
    return real_rotation


def test_random_complete_p_d_shell_wigner_covariance() -> None:
    """Rotate every tensor leg and preserve contracted p/d amplitudes.

    Random proper rotations cover complete p and d shells with generic
    complex coefficients, polarization, and final harmonics.

    Notes
    -----
    Compare the original contraction against a simultaneous three-leg rotation.
    """
    basis: OrbitalBasis
    params: MatrixElementParams
    basis, _, _, params = _complete_pd_fixture()
    assert params.phase_channel_keys == (
        (0, 0),
        (0, 2),
        (1, 1),
        (1, 3),
    )
    coupling: Float64[Array, "n_orb 2 3 36"] = channel_tables(basis)[0]
    generator: np.random.Generator = np.random.default_rng(20260728)
    degree: int
    start: int
    for degree, start in ((1, 0), (2, 3)):
        shell_size: int = 2 * degree + 1
        for _ in range(4):
            angles: Tuple[float, float, float] = tuple(
                float(value)
                for value in generator.uniform(
                    low=(-np.pi, 0.2, -np.pi),
                    high=(np.pi, np.pi - 0.2, np.pi),
                )
            )
            initial_rotation: Float64[Array, "m1 m2"] = _real_wigner(
                degree,
                angles,
            )
            photon_rotation: Float64[Array, "3 3"] = _real_wigner(1, angles)
            initial_coefficients: Complex128[Array, " m"] = jnp.asarray(
                generator.normal(size=shell_size)
                + 1j * generator.normal(size=shell_size)
            )
            polarization: Complex128[Array, " 3"] = jnp.asarray(
                generator.normal(size=3) + 1j * generator.normal(size=3)
            )
            branch: int
            final_degree: int
            for branch, final_degree in enumerate((degree - 1, degree + 1)):
                final_size: int = 2 * final_degree + 1
                block: Float64[Array, "m 3 mf"] = coupling[
                    start : start + shell_size,
                    branch,
                    :,
                    final_degree**2 : (final_degree + 1) ** 2,
                ]
                final_harmonics: Complex128[Array, " mf"] = jnp.asarray(
                    generator.normal(size=final_size)
                    + 1j * generator.normal(size=final_size)
                )
                final_rotation: Float64[Array, "mf1 mf2"] = _real_wigner(
                    final_degree,
                    angles,
                )
                amplitude: Complex128[Array, ""] = jnp.einsum(
                    "aqb,a,q,b->",
                    block,
                    initial_coefficients,
                    polarization,
                    final_harmonics,
                )
                rotated: Complex128[Array, ""] = jnp.einsum(
                    "aqb,a,q,b->",
                    block,
                    initial_rotation @ initial_coefficients,
                    photon_rotation @ polarization,
                    final_rotation @ final_harmonics,
                )
                chex.assert_trees_all_close(
                    rotated,
                    amplitude,
                    rtol=1.0e-12,
                    atol=1.0e-13,
                )


def test_each_coefficient_scale_direction_nulls_full_intensity() -> None:
    """Verify null intensity slopes for both radial coefficient gauges.

    The complete coherent matrix element includes radial evaluation,
    attenuation, polarization contraction, and complex orbital coefficients.

    Notes
    -----
    Apply each named gauge tangent and contrast it with a physical sigma
    tangent.
    """
    basis: OrbitalBasis
    radial: RadialSpec
    params: MatrixElementParams
    basis, _, radial, params = _complete_pd_fixture()
    mean_free_path: Float64[Array, ""] = jnp.asarray(8.4)
    flat: Float64[Array, " n_theta"]
    tree_definition: PyTreeDef
    metadata: Tuple[Tuple[Tuple[int, ...], bool], ...]
    flat, tree_definition, metadata = pack_matrixel_params(
        radial,
        params,
        mean_free_path,
    )
    directions: Float64[Array, "2 n_theta"] = (
        radial_coefficient_scale_gauge_directions(
            radial,
            params,
            mean_free_path,
        )
    )
    coefficients: Complex128[Array, " 8"] = jnp.asarray(
        (
            0.2 + 0.1j,
            -0.4 + 0.3j,
            0.7 - 0.2j,
            -0.1 + 0.5j,
            0.3 + 0.4j,
            -0.6 + 0.2j,
            0.8 - 0.1j,
            0.25 + 0.35j,
        )
    )

    def intensity(candidate: Float64[Array, " n_theta"]) -> Float64[Array, ""]:
        """Return one fully composed coherent shell intensity."""
        rebuilt_radial: RadialSpec
        rebuilt_params: MatrixElementParams
        rebuilt_mfp: Float64[Array, ""]
        rebuilt_radial, rebuilt_params, rebuilt_mfp = unpack_matrixel_params(
            candidate,
            tree_definition,
            metadata,
            radial,
            params,
        )
        bvals: Complex128[Array, "1 8 2"] = radial_bvals(
            rebuilt_radial,
            jnp.asarray((0.9,)),
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        channels: Complex128[Array, "1 1 8 3"] = orbital_transition_channels(
            jnp.asarray(((0.12, -0.08, 0.03),)),
            jnp.asarray(((0.12, -0.08, 1.7),)),
            jnp.zeros((8, 3)),
            jnp.linspace(0.0, 2.0, 8),
            bvals,
            rebuilt_params,
            rebuilt_mfp,
            basis,
        )
        polarized: Complex128[Array, " 8"] = contract_polarization(
            channels[0, 0],
            jnp.asarray((0.3 + 0.2j, -0.4j, 0.7 - 0.1j)),
        )
        amplitude: Complex128[Array, ""] = jnp.sum(polarized * coefficients)
        returned: Float64[Array, ""] = jnp.abs(amplitude) ** 2
        return returned

    direction: Float64[Array, " n_theta"]
    for direction in directions:
        derivative: Float64[Array, ""] = jax.jvp(
            intensity,
            (flat,),
            (direction,),
        )[1]
        chex.assert_trees_all_close(
            derivative,
            jnp.asarray(0.0),
            rtol=0.0,
            atol=1.0e-11,
        )

    sigma_start: int = next(
        offset
        for offset in range(flat.shape[0] - 1)
        if bool(
            jnp.allclose(
                flat[offset : offset + 2],
                params.sigma_shell,
                rtol=0.0,
                atol=0.0,
            )
        )
    )
    sigma_tangent: Float64[Array, " n_theta"] = (
        jnp.zeros_like(flat).at[sigma_start].set(1.0)
    )
    physical_sigma_derivative: Float64[Array, ""] = jax.jvp(
        intensity,
        (flat,),
        (sigma_tangent,),
    )[1]
    assert float(jnp.abs(physical_sigma_derivative)) > 1.0e-7
