"""Certify Plan 06 complete-shell covariance and named radial gauges.

Extended Summary
----------------
The tests rotate complete p and d shells through every tensor leg. They also
verify the two named radial coefficient-scale gauge directions.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Complex, Float

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
    RadialSpec,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)


def _complete_pd_fixture() -> tuple[
    OrbitalBasis,
    tuple[int, ...],
    RadialSpec,
    MatrixElementParams,
]:
    """Return complete p and d shells with two normalized contractions."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 8,
        n=(2,) * 3 + (3,) * 5,
        l=(1,) * 3 + (2,) * 5,
        m=(-1, 0, 1, -2, -1, 0, 1, 2),
    )
    shell_index: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 1, 1)
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
    return basis, shell_index, radial, params


def _real_wigner(
    degree: int,
    angles: tuple[float, float, float],
) -> Float[Array, "m1 m2"]:
    """Return one complex Wigner matrix in the real-harmonic basis."""
    unitary: Complex[Array, "m1 m2"] = real_harmonic_unitary(degree)
    complex_rotation: Complex[Array, "m1 m2"] = wigner_d(degree, *angles)
    real_rotation: Float[Array, "m1 m2"] = jnp.real(
        unitary.conj() @ complex_rotation @ unitary.T
    )
    return real_rotation


def test_g15_random_complete_p_d_shell_wigner_covariance() -> None:
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
    coupling: Float[Array, "n_orb 2 3 36"] = channel_tables(basis)[0]
    generator: np.random.Generator = np.random.default_rng(20260728)
    degree: int
    start: int
    for degree, start in ((1, 0), (2, 3)):
        shell_size: int = 2 * degree + 1
        for _ in range(4):
            angles: tuple[float, float, float] = tuple(
                float(value)
                for value in generator.uniform(
                    low=(-np.pi, 0.2, -np.pi),
                    high=(np.pi, np.pi - 0.2, np.pi),
                )
            )
            initial_rotation: Float[Array, "m1 m2"] = _real_wigner(
                degree,
                angles,
            )
            photon_rotation: Float[Array, "3 3"] = _real_wigner(1, angles)
            initial_coefficients: Complex[Array, " m"] = jnp.asarray(
                generator.normal(size=shell_size)
                + 1j * generator.normal(size=shell_size)
            )
            polarization: Complex[Array, " 3"] = jnp.asarray(
                generator.normal(size=3) + 1j * generator.normal(size=3)
            )
            branch: int
            final_degree: int
            for branch, final_degree in enumerate((degree - 1, degree + 1)):
                final_size: int = 2 * final_degree + 1
                block: Float[Array, "m 3 mf"] = coupling[
                    start : start + shell_size,
                    branch,
                    :,
                    final_degree**2 : (final_degree + 1) ** 2,
                ]
                final_harmonics: Complex[Array, " mf"] = jnp.asarray(
                    generator.normal(size=final_size)
                    + 1j * generator.normal(size=final_size)
                )
                final_rotation: Float[Array, "mf1 mf2"] = _real_wigner(
                    final_degree,
                    angles,
                )
                amplitude: Complex[Array, ""] = jnp.einsum(
                    "aqb,a,q,b->",
                    block,
                    initial_coefficients,
                    polarization,
                    final_harmonics,
                )
                rotated: Complex[Array, ""] = jnp.einsum(
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


def test_g15_each_coefficient_scale_direction_nulls_full_intensity() -> None:
    """Verify null intensity slopes for both radial coefficient gauges.

    The complete coherent matrix element includes radial evaluation,
    attenuation, polarization contraction, and complex orbital coefficients.

    Notes
    -----
    Apply each named gauge tangent and contrast it with a physical sigma tangent.
    """
    basis: OrbitalBasis
    radial: RadialSpec
    params: MatrixElementParams
    basis, _, radial, params = _complete_pd_fixture()
    mean_free_path: Float[Array, ""] = jnp.asarray(8.4)
    flat: Float[Array, " n_theta"]
    tree_definition: PyTreeDef
    metadata: tuple[tuple[tuple[int, ...], bool], ...]
    flat, tree_definition, metadata = pack_matrixel_params(
        radial,
        params,
        mean_free_path,
    )
    directions: Float[Array, "2 n_theta"] = (
        radial_coefficient_scale_gauge_directions(
            radial,
            params,
            mean_free_path,
        )
    )
    coefficients: Complex[Array, " 8"] = jnp.asarray(
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

    def intensity(candidate: Float[Array, " n_theta"]) -> Float[Array, ""]:
        """Return one fully composed coherent shell intensity."""
        rebuilt_radial: RadialSpec
        rebuilt_params: MatrixElementParams
        rebuilt_mfp: Float[Array, ""]
        rebuilt_radial, rebuilt_params, rebuilt_mfp = unpack_matrixel_params(
            candidate,
            tree_definition,
            metadata,
            radial,
            params,
        )
        bvals: Complex[Array, "1 8 2"] = radial_bvals(
            rebuilt_radial,
            jnp.asarray((0.9,)),
            make_radial_quadrature_spec(),
            make_final_state_spec(),
        )
        channels: Complex[Array, "1 1 8 3"] = orbital_transition_channels(
            jnp.asarray(((0.12, -0.08, 0.03),)),
            jnp.asarray(((0.12, -0.08, 1.7),)),
            jnp.zeros((8, 3)),
            jnp.linspace(0.0, 2.0, 8),
            bvals,
            rebuilt_params,
            rebuilt_mfp,
            basis,
        )
        polarized: Complex[Array, " 8"] = contract_polarization(
            channels[0, 0],
            jnp.asarray((0.3 + 0.2j, -0.4j, 0.7 - 0.1j)),
        )
        amplitude: Complex[Array, ""] = jnp.sum(polarized * coefficients)
        return jnp.abs(amplitude) ** 2

    direction: Float[Array, " n_theta"]
    for direction in directions:
        derivative: Float[Array, ""] = jax.jvp(
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
    sigma_tangent: Float[Array, " n_theta"] = (
        jnp.zeros_like(flat).at[sigma_start].set(1.0)
    )
    physical_sigma_derivative: Float[Array, ""] = jax.jvp(
        intensity,
        (flat,),
        (sigma_tangent,),
    )[1]
    assert float(jnp.abs(physical_sigma_derivative)) > 1.0e-7
