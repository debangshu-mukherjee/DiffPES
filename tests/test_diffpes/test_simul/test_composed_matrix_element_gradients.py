"""Certify composed matrix-element differentiability across all seams.

Extended Summary
----------------
The tests apply the shared forward/reverse and central-finite-
difference harness to generic-complex matrix-element compositions. Analytic
JVPs separately pin holomorphic centre phases and complete-group covariance.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.matrixel import (
    assemble_orbital_transition_channels,
    matrix_element_intensity,
    orbital_transition_channels,
    project_band_channels,
    resolve_orbital_positions_cart,
)
from diffpes.simul import (
    contract_experiment_polarization,
    final_state_k_inv_ang,
    kinetic_energy_ev,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)
from tests._gradients import assert_grad_matches_fd, assert_nonzero_grad


def _basis() -> OrbitalBasis:
    """PRIVATE: Build one complete mixed s-and-p real-orbital basis.

    Returns
    -------
    basis : OrbitalBasis
        One atom with the 1s orbital and the complete n=2 p shell.

    Notes
    -----
    Lists the three p orbitals in ascending m order after the s
    orbital.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 2, 2),
        l=(0, 1, 1, 1),
        m=(0, -1, 0, 1),
    )
    return basis


def _experiment(
    photon_energy: Float64[Array, ""] = jnp.asarray(24.0),
    polarization: Complex128[Array, " 3"] = jnp.asarray(
        [0.6 + 0.2j, -0.3 + 0.7j, 0.0 + 0.0j]
    ),
    azimuth: Float64[Array, ""] = jnp.asarray(0.31),
    mean_free_path: Float64[Array, ""] = jnp.asarray(8.4),
    inner_potential: Float64[Array, ""] = jnp.asarray(11.0),
) -> ExperimentGeometry:
    """PRIVATE: Build a generic transverse experiment carrier.

    Parameters
    ----------
    photon_energy : Float64[Array, ""]
        Photon energy in eV.
    polarization : Complex128[Array, " 3"]
        Cartesian complex polarization; the default is transverse with
        a zero z component.
    azimuth : Float64[Array, ""]
        Sample azimuth in radians.
    mean_free_path : Float64[Array, ""]
        Inelastic mean free path in Angstrom.
    inner_potential : Float64[Array, ""]
        Inner potential in eV.

    Returns
    -------
    experiment : ExperimentGeometry
        Experiment carrier with the work function fixed at 4.5 eV.

    Notes
    -----
    Passes every argument through the public factory and pins only the
    work function.
    """
    experiment: ExperimentGeometry = make_experiment_geometry(
        photon_energy,
        polarization,
        sample_azimuth=azimuth,
        work_function_ev=4.5,
        mean_free_path_ang=mean_free_path,
        inner_potential_ev=inner_potential,
    )
    return experiment


def _bands(
    basis: OrbitalBasis,
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    atom_positions: Float64[Array, "1 3"] = jnp.asarray([[0.07, 0.11, 0.13]]),
    orbital_positions: Float64[Array, "4 3"] | None = jnp.asarray(
        [
            [0.07, 0.11, 0.13],
            [0.21, 0.08, 0.17],
            [0.12, 0.26, 0.09],
            [0.31, 0.18, 0.23],
        ]
    ),
    depths: Float64[Array, " 4"] | None = jnp.asarray([0.4, 1.2, 2.1, 3.4]),
) -> DiagonalizedBands:
    """PRIVATE: Build one generic-complex nondegenerate band carrier.

    Parameters
    ----------
    basis : OrbitalBasis
        Four-orbital metadata for the single atom.
    lattice : Float64[Array, "3 3"]
        Lattice rows in Angstrom.
    atom_positions : Float64[Array, "1 3"]
        Fractional position of the one atom.
    orbital_positions : Float64[Array, "4 3"] | None
        Fractional orbital centres; None falls back to the host atom
        sites.
    depths : Float64[Array, " 4"] | None
        Orbital depths in Angstrom; None drops the depth carrier.

    Returns
    -------
    bands : DiagonalizedBands
        One zone-center k-point with one band at -0.37 eV and a fixed
        generic complex eigenvector.

    Notes
    -----
    The generic default positions and depths keep every orbital factor
    distinct so no gradient path cancels.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        atom_positions,
        ("X",),
    )
    eigenvectors: Complex128[Array, "1 1 4"] = jnp.asarray(
        [[[0.43 + 0.17j, -0.28 + 0.51j, 0.36 - 0.22j, 0.19 + 0.47j]]]
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        jnp.asarray([[-0.37]]),
        eigenvectors,
        jnp.zeros((1, 3)),
        geometry,
        basis,
        orbital_positions=orbital_positions,
        depths=depths,
    )
    return bands


def _matrix_params(
    basis: OrbitalBasis,
    phases: Float64[Array, " 3"] = jnp.asarray([0.23, -0.41, 0.67]),
) -> MatrixElementParams:
    """PRIVATE: Build compact physical phase and scale coordinates.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata for the mixed s-and-p basis.
    phases : Float64[Array, " 3"]
        Phase-shift angles in radians for the three dipole branch
        channels of the two shells.

    Returns
    -------
    params : MatrixElementParams
        Parameter carrier with per-shell sigma scales 1.1 and 0.83.

    Notes
    -----
    Maps the four orbitals onto the s shell and one p shell with shell
    map (0, 1, 1, 1).
    """
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        (0, 1, 1, 1),
        sigma_shell=jnp.asarray([1.1, 0.83]),
        phase_shift_angles_shell=phases,
    )
    return params


def _generic_channels(
    *,
    positions: Float64[Array, "4 3"] | None = None,
    mean_free_path: Float64[Array, ""] = jnp.asarray(8.4),
    phases: Float64[Array, " 3"] = jnp.asarray([0.23, -0.41, 0.67]),
) -> Complex128[Array, "1 1 4 3"]:
    """PRIVATE: Assemble lower-level generic-complex orbital channels.

    Parameters
    ----------
    positions : Float64[Array, "4 3"] | None
        Cartesian orbital positions in Angstrom; None selects fixed
        generic values.
    mean_free_path : Float64[Array, ""]
        Inelastic mean free path in Angstrom.
    phases : Float64[Array, " 3"]
        Branch phase-shift angles in radians.

    Returns
    -------
    channels : Complex128[Array, "1 1 4 3"]
        Cartesian orbital transition channels at one k-point.

    Notes
    -----
    Fixes one initial and one final momentum in 1/Angstrom. Uses generic
    complex radial branches with the s lower branch zero. Uses depths
    from 0.4 to 3.4 Angstrom.
    """
    basis: OrbitalBasis = _basis()
    resolved_positions: Float64[Array, "4 3"] = (
        jnp.asarray(
            [
                [0.13, 0.07, 0.19],
                [0.31, 0.11, 0.23],
                [0.17, 0.29, 0.09],
                [0.27, 0.21, 0.37],
            ]
        )
        if positions is None
        else positions
    )
    radial_values: Complex128[Array, "1 4 2"] = jnp.asarray(
        [
            [
                [0.0 + 0.0j, 0.71 + 0.23j],
                [0.43 - 0.19j, -0.27 + 0.61j],
                [-0.38 + 0.47j, 0.52 + 0.16j],
                [0.29 + 0.34j, -0.41 - 0.22j],
            ]
        ]
    )
    channels: Complex128[Array, "1 1 4 3"] = orbital_transition_channels(
        jnp.asarray([[0.17, -0.09, 0.05]]),
        jnp.asarray([[0.17, -0.09, 1.31]]),
        resolved_positions,
        jnp.asarray([0.4, 1.2, 2.1, 3.4]),
        radial_values,
        _matrix_params(basis, phases),
        mean_free_path,
        basis,
    )
    return channels


def _intensity(
    channels: Complex128[Array, "1 1 4 3"],
    experiment: ExperimentGeometry,
    eigenvectors: Complex128[Array, "1 1 4"] | None = None,
) -> Float64[Array, ""]:
    """PRIVATE: Reduce generic orbital channels through one modulus square.

    Parameters
    ----------
    channels : Complex128[Array, "1 1 4 3"]
        Cartesian orbital transition channels.
    experiment : ExperimentGeometry
        Carrier that supplies the late polarization contraction.
    eigenvectors : Complex128[Array, "1 1 4"] | None
        Band projection coefficients; None selects the fixed generic
        vector.

    Returns
    -------
    intensity : Float64[Array, ""]
        Summed physical intensity of the projected amplitudes.

    Notes
    -----
    Projects the channels onto the band, contracts the experiment
    polarization late, and sums the modulus-squared amplitudes.
    """
    coefficients: Complex128[Array, "1 1 4"] = (
        jnp.asarray(
            [[[0.43 + 0.17j, -0.28 + 0.51j, 0.36 - 0.22j, 0.19 + 0.47j]]]
        )
        if eigenvectors is None
        else eigenvectors
    )
    band_channels: Complex128[Array, "1 1 1 3"] = project_band_channels(
        channels,
        coefficients,
    )
    amplitudes: Complex128[Array, "1 1 1"] = contract_experiment_polarization(
        band_channels,
        experiment,
    )
    intensity: Float64[Array, ""] = jnp.sum(
        matrix_element_intensity(amplitudes)
    )
    return intensity


class TestRadialAndChannelGradients:
    """Certify composed radial and photon-energy derivative checks.

    The cases compare finite differences with gradients for the Slater data,
    radial coefficients, channel coefficients, and the vacuum momentum.
    """

    @pytest.mark.slow
    @pytest.mark.rss_limit_mb(1024)
    def test_slater_exponents_and_coefficients(self) -> None:
        """Match autodiff for normalized multi-zeta intensity.

        The composition includes radial quadrature, orbital assembly, generic
        complex projection, late polarization, and the final intensity.

        Notes
        -----
        Apply both AD modes and the central-FD census to every coordinate.
        """
        basis: OrbitalBasis = _basis()
        bands: DiagonalizedBands = _bands(basis)
        bands = eqx.tree_at(
            lambda carrier: carrier.kpoints,
            bands,
            jnp.asarray([[0.27, -0.19, 0.0]]) / (2.0 * jnp.pi),
        )
        params: MatrixElementParams = _matrix_params(basis)
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        final_state: FinalStateSpec = make_final_state_spec()
        experiment: ExperimentGeometry = _experiment()
        final_momentum: Float64[Array, "1 3"] = jnp.asarray(
            [[0.27, -0.19, 1.31]]
        )
        validity: Bool[Array, " 1"] = jnp.asarray([True])
        initial: Float64[Array, " 8"] = jnp.asarray(
            [1.1, 1.9, 0.9, 1.6, 0.8, -0.27, 0.61, 0.39]
        )

        def loss(candidate: Float64[Array, " 8"]) -> Float64[Array, ""]:
            """Return composed intensity for radial physical coordinates."""
            radial: RadialSpec = make_radial_spec(
                basis,
                (0, 1, 1, 1),
                zeta_shell=candidate[:4].reshape(2, 2),
                coefficients_shell=candidate[4:].reshape(2, 2),
            )
            channels: Complex128[Array, "1 1 4 3"] = (
                assemble_orbital_transition_channels(
                    bands,
                    radial,
                    params,
                    quadrature,
                    final_state,
                    experiment,
                    final_momentum,
                    validity,
                )
            )
            value: Float64[Array, ""] = _intensity(channels, experiment)
            return value

        assert_grad_matches_fd(loss, initial, modes=("fwd", "rev"))
        assert_nonzero_grad(loss, initial, elementwise=True)

    @pytest.mark.slow
    @pytest.mark.rss_limit_mb(768)
    def test_photon_energy_to_explicit_vacuum_momentum(self) -> None:
        """Match derivatives through energy conservation and vacuum momentum.

        The inner potential remains absent from the fixed-vacuum assembly
        path while photon energy changes its explicit final momentum.

        Notes
        -----
        Apply the shared harness away from threshold and assert the exact V0
        zero.
        """
        basis: OrbitalBasis = _basis()
        bands: DiagonalizedBands = _bands(basis)
        radial: RadialSpec = make_radial_spec(
            basis,
            (0, 1, 1, 1),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray([[0.0, 1.0], [0.6, 0.8]]),
        )
        params: MatrixElementParams = _matrix_params(basis)
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        final_state: FinalStateSpec = make_final_state_spec()

        def loss(photon_energy: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return intensity after the explicit vacuum-momentum map."""
            kinetic: Float64[Array, " 1"]
            energy_valid: Bool[Array, " 1"]
            kinetic, energy_valid = kinetic_energy_ev(
                photon_energy,
                jnp.asarray(4.5),
                jnp.asarray([-0.37]),
            )
            momentum: Float64[Array, " 1"]
            momentum_valid: Bool[Array, " 1"]
            momentum, momentum_valid = final_state_k_inv_ang(kinetic)
            final_momentum: Float64[Array, "1 3"] = jnp.stack(
                (jnp.zeros_like(momentum), jnp.zeros_like(momentum), momentum),
                axis=-1,
            )
            experiment: ExperimentGeometry = _experiment(photon_energy)
            channels: Complex128[Array, "1 1 4 3"] = (
                assemble_orbital_transition_channels(
                    bands,
                    radial,
                    params,
                    quadrature,
                    final_state,
                    experiment,
                    final_momentum,
                    energy_valid & momentum_valid,
                )
            )
            value: Float64[Array, ""] = _intensity(channels, experiment)
            return value

        photon_energy: Float64[Array, ""] = jnp.asarray(24.0)
        assert_grad_matches_fd(loss, photon_energy, modes=("fwd", "rev"))
        assert_nonzero_grad(loss, photon_energy)

        fixed_momentum: Float64[Array, "1 3"] = jnp.asarray([[0.0, 0.0, 2.1]])

        def fixed_vacuum(
            inner_potential: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Return assembly intensity at fixed vacuum momentum."""
            experiment: ExperimentGeometry = _experiment(
                inner_potential=inner_potential
            )
            channels: Complex128[Array, "1 1 4 3"] = (
                assemble_orbital_transition_channels(
                    bands,
                    radial,
                    params,
                    quadrature,
                    final_state,
                    experiment,
                    fixed_momentum,
                    jnp.asarray([True]),
                )
            )
            value: Float64[Array, ""] = _intensity(channels, experiment)
            return value

        zero: Float64[Array, ""] = jax.grad(fixed_vacuum)(jnp.asarray(11.0))
        chex.assert_trees_all_equal(zero, jnp.asarray(0.0))


class TestProjectionAndPolarizationGradients:
    """Certify optical, centre, attenuation, and phase derivatives.

    The cases perturb the polarization, lattice, orbital centres, attenuation,
    and compact phase coordinates through the composed matrix element.
    """

    def test_complex_polarization_real_view_and_azimuth(self) -> None:
        """Match derivatives on generic complex optical coordinates.

        Four stacked real quadratures construct the transverse laboratory
        vector, and a fifth coordinate rotates the sample before contraction.

        Notes
        -----
        Apply the shared forward/reverse and central-FD harness to the real
        view.
        """
        channels: Complex128[Array, "1 1 4 3"] = _generic_channels()
        initial: Float64[Array, " 5"] = jnp.asarray(
            [0.61, -0.27, 0.23, 0.69, 0.31]
        )

        def loss(candidate: Float64[Array, " 5"]) -> Float64[Array, ""]:
            """Return intensity for a stacked-real optical chart."""
            polarization: Complex128[Array, " 3"] = jnp.asarray(
                [
                    candidate[0] + 1j * candidate[2],
                    candidate[1] + 1j * candidate[3],
                    0.0 + 0.0j,
                ]
            )
            experiment: ExperimentGeometry = _experiment(
                polarization=polarization,
                azimuth=candidate[4],
            )
            value: Float64[Array, ""] = _intensity(channels, experiment)
            return value

        assert_grad_matches_fd(loss, initial, modes=("fwd", "rev"))
        assert_nonzero_grad(loss, initial, elementwise=True)

    @pytest.mark.slow
    def test_fractional_centres_and_lattice(self) -> None:
        """Match derivatives through explicit and atom-fallback centre maps.

        Both carrier routes apply one fractional-to-Cartesian lattice product
        before the coherent centre phase.

        Notes
        -----
        Check both routes with the shared harness and pin translation JVP
        analytically.
        """
        basis: OrbitalBasis = _basis()
        centre: Float64[Array, "4 3"] = jnp.asarray(
            [
                [0.07, 0.11, 0.13],
                [0.21, 0.08, 0.17],
                [0.12, 0.26, 0.09],
                [0.31, 0.18, 0.23],
            ]
        )
        lattice: Float64[Array, "3 3"] = jnp.asarray(
            [[1.7, 0.1, 0.0], [0.0, 1.4, 0.2], [0.1, 0.0, 1.9]]
        )

        def explicit_loss(
            candidate: Tuple[Float64[Array, "4 3"], Float64[Array, "3 3"]],
        ) -> Float64[Array, ""]:
            """Return intensity through explicit Wannier centres."""
            centres: Float64[Array, "4 3"]
            trial_lattice: Float64[Array, "3 3"]
            centres, trial_lattice = candidate
            bands: DiagonalizedBands = _bands(
                basis,
                lattice=trial_lattice,
                orbital_positions=centres,
            )
            cartesian: Float64[Array, "4 3"] = resolve_orbital_positions_cart(
                bands
            )
            value: Float64[Array, ""] = _intensity(
                _generic_channels(positions=cartesian),
                _experiment(),
            )
            return value

        assert_grad_matches_fd(
            explicit_loss,
            (centre, lattice),
            modes=("fwd", "rev"),
        )

        def fallback_loss(
            candidate: Tuple[Float64[Array, "1 3"], Float64[Array, "3 3"]],
        ) -> Float64[Array, ""]:
            """Return intensity through atom-derived centres."""
            atoms: Float64[Array, "1 3"]
            trial_lattice: Float64[Array, "3 3"]
            atoms, trial_lattice = candidate
            bands: DiagonalizedBands = _bands(
                basis,
                lattice=trial_lattice,
                atom_positions=atoms,
                orbital_positions=None,
            )
            cartesian: Float64[Array, "4 3"] = resolve_orbital_positions_cart(
                bands
            )
            value: Float64[Array, ""] = _intensity(
                _generic_channels(positions=cartesian),
                _experiment(),
            )
            return value

        assert_grad_matches_fd(
            fallback_loss,
            (jnp.asarray([[0.07, 0.11, 0.13]]), lattice),
            modes=("fwd", "rev"),
        )

        direction: Float64[Array, " 3"] = jnp.asarray([0.13, -0.17, 0.09])
        base_channels: Complex128[Array, "1 1 4 3"] = _generic_channels()

        def translated(
            amount: Float64[Array, ""],
        ) -> Complex128[Array, "1 1 4 3"]:
            """Return channels after a common Cartesian translation."""
            positions: Float64[Array, "4 3"] = jnp.asarray(
                [
                    [0.13, 0.07, 0.19],
                    [0.31, 0.11, 0.23],
                    [0.17, 0.29, 0.09],
                    [0.27, 0.21, 0.37],
                ]
            )
            channels: Complex128[Array, "1 1 4 3"] = _generic_channels(
                positions=positions + amount * direction
            )
            return channels

        derivative: Complex128[Array, "1 1 4 3"] = jax.jvp(
            translated,
            (jnp.asarray(0.0),),
            (jnp.asarray(1.0),),
        )[1]
        momentum_difference: Float64[Array, " 3"] = jnp.asarray(
            [0.0, 0.0, -1.26]
        )
        expected: Complex128[Array, "1 1 4 3"] = (
            1j * jnp.dot(momentum_difference, direction) * base_channels
        )
        chex.assert_trees_all_close(
            derivative,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_nonzero_depth_attenuation(self) -> None:
        """Match mean-free-path sensitivity at positive depths.

        The fixture has several distinct nonzero depths, so attenuation must
        contribute a useful physical derivative.

        Notes
        -----
        Apply the shared harness and its nonzero-gradient tripwire.
        """

        def loss(mean_free_path: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return attenuated composed intensity."""
            channels: Complex128[Array, "1 1 4 3"] = _generic_channels(
                mean_free_path=mean_free_path
            )
            value: Float64[Array, ""] = _intensity(channels, _experiment())
            return value

        initial: Float64[Array, ""] = jnp.asarray(8.4)
        assert_grad_matches_fd(loss, initial, modes=("fwd", "rev"))
        assert_nonzero_grad(loss, initial, elementwise=True)

    def test_compact_physical_phase_coordinates(self) -> None:
        """Match derivatives for every compact physical channel phase.

        Generic radial values, centres, eigenvectors, and polarization expose
        phase sensitivities without padded or invalid channel coordinates.

        Notes
        -----
        Apply the shared harness to the three valid s-and-p phase coordinates.
        """

        def loss(phases: Float64[Array, " 3"]) -> Float64[Array, ""]:
            """Return composed intensity for compact physical phases."""
            channels: Complex128[Array, "1 1 4 3"] = _generic_channels(
                phases=phases
            )
            value: Float64[Array, ""] = _intensity(channels, _experiment())
            return value

        initial: Float64[Array, " 3"] = jnp.asarray([0.23, -0.41, 0.67])
        assert_grad_matches_fd(loss, initial, modes=("fwd", "rev"))
        assert_nonzero_grad(loss, initial, elementwise=True)


class TestIntensityAndGroupWeightGradients:
    """Certify holomorphic centre phases and complex band derivatives.

    The cases compare finite differences with the centre-phase gradient and
    with real views of generic complex band and group-weight directions.
    """

    def test_holomorphic_centre_phase(self) -> None:
        """Match intensity derivatives to four independent truths.

        A non-real baseline forbids a subtract-free imaginary-step shortcut
        and exercises the actual production centre-phase sub-block.

        Notes
        -----
        Apply the shared scalar harness and compare complex directional
        derivatives.
        """
        direction: Float64[Array, " 3"] = jnp.asarray([0.13, -0.17, 0.09])
        positions: Float64[Array, "4 3"] = jnp.asarray(
            [
                [0.13, 0.07, 0.19],
                [0.31, 0.11, 0.23],
                [0.17, 0.29, 0.09],
                [0.27, 0.21, 0.37],
            ]
        )
        weight: Complex128[Array, "1 1 4 3"] = jnp.asarray(
            jnp.arange(1, 13).reshape(1, 1, 4, 3)
        ) * (0.07 + 0.03j)

        def channels(
            amount: Float64[Array, ""],
        ) -> Complex128[Array, "1 1 4 3"]:
            """Return production channels along one centre direction."""
            result: Complex128[Array, "1 1 4 3"] = _generic_channels(
                positions=positions + amount * direction
            )
            return result

        def scalar_loss(amount: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return one generic real view of the complex phase block."""
            value: Float64[Array, ""] = jnp.real(
                jnp.sum(weight * channels(amount))
            )
            return value

        assert_grad_matches_fd(
            scalar_loss,
            jnp.asarray(0.0),
            modes=("fwd", "rev"),
        )
        derivative: Complex128[Array, "1 1 4 3"] = jax.jvp(
            channels,
            (jnp.asarray(0.0),),
            (jnp.asarray(1.0),),
        )[1]
        baseline: Complex128[Array, "1 1 4 3"] = channels(jnp.asarray(0.0))
        phase_rate: Float64[Array, ""] = jnp.dot(
            jnp.asarray([0.0, 0.0, -1.26]),
            direction,
        )
        expected: Complex128[Array, "1 1 4 3"] = 1j * phase_rate * baseline
        chex.assert_trees_all_close(
            derivative,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        mpmath.mp.dps = 80
        rate_mp: mpmath.mpf = mpmath.mpf(str(float(phase_rate)))
        baseline_mp: mpmath.mpc = mpmath.mpc(
            str(float(jnp.real(baseline[0, 0, 0, 0]))),
            str(float(jnp.imag(baseline[0, 0, 0, 0]))),
        )
        step_mp: mpmath.mpf = mpmath.mpf("1e-30")
        quotient_mp: mpmath.mpc = (
            baseline_mp * mpmath.exp(1j * rate_mp * step_mp)
            - baseline_mp * mpmath.exp(-1j * rate_mp * step_mp)
        ) / (2 * step_mp)
        quotient: Complex128[Array, ""] = jnp.asarray(
            complex(quotient_mp),
            dtype=jnp.complex128,
        )
        chex.assert_trees_all_close(
            derivative[0, 0, 0, 0],
            quotient,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_generic_eigenvectors_and_group_directions(self) -> None:
        """Match raw-band derivatives and complete-group JVPs.

        The nondegenerate input populates both complex quadratures. Separate
        U(2) and U(3) tangent directions rotate complete degenerate groups.

        Notes
        -----
        Apply both harness modes, then require exact first-order group
        covariance.
        """
        channels: Complex128[Array, "1 1 4 3"] = _generic_channels()
        experiment: ExperimentGeometry = _experiment()
        eigenvectors: Complex128[Array, "1 1 4"] = jnp.asarray(
            [[[0.43 + 0.17j, -0.28 + 0.51j, 0.36 - 0.22j, 0.19 + 0.47j]]]
        )

        def raw_loss(
            candidate: Complex128[Array, "1 1 4"],
        ) -> Float64[Array, ""]:
            """Return one registered nondegenerate raw-band weight."""
            value: Float64[Array, ""] = _intensity(
                channels,
                experiment,
                candidate,
            )
            return value

        assert_grad_matches_fd(
            raw_loss,
            eigenvectors,
            modes=("fwd", "rev"),
        )
        assert_nonzero_grad(raw_loss, eigenvectors, elementwise=True)

        group_size: int
        for group_size in (2, 3):
            transition: Complex128[Array, "1 1 4 3"] = channels
            rows: Complex128[Array, "1 n_group 4"] = jnp.asarray(
                [
                    [
                        [
                            complex(
                                0.17 * (band + 1) * (orbital + 1),
                                0.11 * (band + orbital + 1),
                            )
                            for orbital in range(4)
                        ]
                        for band in range(group_size)
                    ]
                ],
                dtype=jnp.complex128,
            )
            generator: Complex128[Array, "n_group n_group"] = jnp.asarray(
                [
                    [
                        complex(0.0, 0.13 * (row + 1))
                        if row == column
                        else complex(
                            0.07 * (row - column),
                            0.05 * (row + column + 1),
                        )
                        for column in range(group_size)
                    ]
                    for row in range(group_size)
                ],
                dtype=jnp.complex128,
            )
            generator = 0.5 * (generator - jnp.conj(generator.T))
            tangent: Complex128[Array, "1 n_group 4"] = (generator @ rows[0])[
                None, ...
            ]

            def _group_weight(
                candidate: Complex128[Array, "1 n_group 4"],
                transition_channels: Complex128[Array, "1 1 4 3"] = transition,
            ) -> Float64[Array, ""]:
                """PRIVATE: Check the private helper behavior.

                Parameters
                ----------
                candidate : Complex128[Array, "1 n_group 4"]
                    Candidate eigenvectors for the unresolved group.
                transition_channels : Complex128[Array, "1 1 4 3"]
                    Fixed orbital transition channels.

                Returns
                -------
                value : Float64[Array, ""]
                    Summed group intensity.

                Notes
                -----
                Projects every candidate band before contracting polarization.
                """
                band_channels: Complex128[Array, "1 n_group 1 3"] = (
                    project_band_channels(transition_channels, candidate)
                )
                amplitudes: Complex128[Array, "1 n_group 1"] = (
                    contract_experiment_polarization(
                        band_channels,
                        experiment,
                    )
                )
                value: Float64[Array, ""] = jnp.sum(
                    matrix_element_intensity(amplitudes)
                )
                return value

            derivative: Float64[Array, ""] = jax.jvp(
                _group_weight,
                (rows,),
                (tangent,),
            )[1]
            chex.assert_trees_all_close(
                derivative,
                jnp.asarray(0.0),
                rtol=0.0,
                atol=1.0e-12,
            )
