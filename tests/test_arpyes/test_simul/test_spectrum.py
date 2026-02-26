"""Tests for ARPES simulation spectrum functions.

Validates the five simulation levels in :mod:`arpyes.simul.spectrum`
(novice, basic, basicplus, advanced, expert) by checking output shapes,
value constraints (non-negativity, finiteness), and sensitivity to
physical parameters (photon energy, polarization type).
"""

import chex
import jax.numpy as jnp

from arpyes.simul.spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
)
from arpyes.types import (
    make_band_structure,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
)


def _make_synthetic_data(nk=20, nb=5, na=2):
    """Generate synthetic band structure and orbital projections for testing.

    Creates a minimal but physically plausible dataset: eigenvalues are
    linearly spaced from -2.0 to 0.5 eV across ``nk * nb`` entries (then
    reshaped to ``(nk, nb)``), k-points lie along the x-axis from 0 to 1,
    and all orbital projections are set to a uniform value of 0.1.

    Parameters
    ----------
    nk : int, optional
        Number of k-points. Default is 20.
    nb : int, optional
        Number of bands. Default is 5.
    na : int, optional
        Number of atoms. Default is 2.

    Returns
    -------
    bands : BandStructure
        Band structure with linearly spaced eigenvalues, 1-D k-path,
        and Fermi energy at 0.0 eV.
    orb_proj : OrbitalProjection
        Uniform orbital projections of shape ``(nk, nb, na, 9)`` with
        all entries set to 0.1.
    """
    eigenvalues = jnp.linspace(-2.0, 0.5, nk * nb).reshape(
        nk, nb
    )
    kpoints = jnp.zeros((nk, 3))
    kpoints = kpoints.at[:, 0].set(
        jnp.linspace(0.0, 1.0, nk)
    )
    bands = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        fermi_energy=0.0,
    )
    projections = jnp.ones((nk, nb, na, 9)) * 0.1
    orb_proj = make_orbital_projection(projections=projections)
    return bands, orb_proj


class TestSimulateNovice(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_novice`.

    Verifies the novice-level simulation (Voigt broadening with uniform
    orbital weights) including output tensor shapes and non-negativity of
    the intensity map.
    """

    def test_output_shape(self):
        """Verify that intensity and energy axis have the expected shapes.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500 energy points.

        2. **Simulate**:
           Run ``simulate_novice`` to produce an ``ArpesSpectrum``.

        3. **Check shapes**:
           Assert that ``intensity`` is ``(20, 500)`` (k-points by energy)
           and ``energy_axis`` is ``(500,)``.

        Asserts
        -------
        The intensity shape matches ``(n_kpoints, fidelity)`` and the
        energy axis length matches ``fidelity``.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_novice(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))
        chex.assert_shape(spectrum.energy_axis, (500,))

    def test_nonnegative_intensity(self):
        """Verify that all intensity values are non-negative.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data and simulation parameters with
           fidelity=200.

        2. **Simulate**:
           Run ``simulate_novice`` to produce an ``ArpesSpectrum``.

        3. **Check non-negativity**:
           Assert that the minimum intensity value is at least ``-1e-10``
           (allowing for negligible floating-point undershoot).

        Asserts
        -------
        The minimum intensity is effectively non-negative, confirming that
        the Voigt convolution with Fermi-Dirac occupation does not produce
        physically impossible negative spectral weight.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_novice(bands, orb_proj, params)
        assert float(jnp.min(spectrum.intensity)) >= -1e-10


class TestSimulateBasic(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_basic`.

    Verifies the basic-level simulation (Gaussian broadening with heuristic
    orbital weights) including output tensor shape and finiteness of all
    intensity values.
    """

    def test_output_shape(self):
        """Verify that the intensity array has the expected shape.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500.

        2. **Simulate**:
           Run ``simulate_basic`` to produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        Asserts
        -------
        The intensity shape matches ``(n_kpoints, fidelity)``.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_finite_values(self):
        """Verify that all intensity values are finite (no NaN or Inf).

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data and simulation parameters with
           fidelity=200.

        2. **Simulate**:
           Run ``simulate_basic`` to produce an ``ArpesSpectrum``.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm that no element
           of the intensity array is NaN or infinite.

        Asserts
        -------
        Every element of the intensity array is finite, confirming that
        the Gaussian convolution and heuristic weighting do not produce
        numerical overflow or undefined values.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateBasicplus(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_basicplus`.

    Verifies the basicplus-level simulation (Gaussian broadening with
    Yeh-Lindau photoionization cross-sections) including output tensor
    shape and sensitivity of the spectrum to changes in photon energy.
    """

    def test_output_shape(self):
        """Verify that the intensity array has the expected shape.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500.

        2. **Simulate**:
           Run ``simulate_basicplus`` to produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        Asserts
        -------
        The intensity shape matches ``(n_kpoints, fidelity)``.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basicplus(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_yeh_lindau_affects_weights(self):
        """Verify that different photon energies produce different spectra.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data and create two sets of simulation
           parameters that differ only in photon energy (20 eV vs 60 eV).

        2. **Simulate both**:
           Run ``simulate_basicplus`` with each parameter set to produce
           two spectra.

        3. **Compare**:
           Compute the total absolute difference between the two
           intensity arrays.

        Asserts
        -------
        The summed absolute intensity difference is strictly positive,
        confirming that Yeh-Lindau cross-section weights are
        photon-energy-dependent and produce measurably different orbital
        weightings at 20 eV versus 60 eV.
        """
        bands, orb_proj = _make_synthetic_data()
        params_low = make_simulation_params(
            fidelity=200, photon_energy=20.0
        )
        params_high = make_simulation_params(
            fidelity=200, photon_energy=60.0
        )
        spec_low = simulate_basicplus(
            bands, orb_proj, params_low
        )
        spec_high = simulate_basicplus(
            bands, orb_proj, params_high
        )
        diff = jnp.sum(
            jnp.abs(spec_low.intensity - spec_high.intensity)
        )
        assert float(diff) > 0.0


class TestSimulateAdvanced(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_advanced`.

    Verifies the advanced-level simulation (Gaussian broadening with
    Yeh-Lindau cross-sections and polarization selection rules) including
    output tensor shape and finiteness under unpolarized light conditions.
    """

    def test_output_shape(self):
        """Verify that intensity has the expected shape under LVP polarization.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, create
           simulation parameters with fidelity=500, and configure linear
           vertical polarization (LVP).

        2. **Simulate**:
           Run ``simulate_advanced`` with the polarization config to
           produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        Asserts
        -------
        The intensity shape matches ``(n_kpoints, fidelity)``, confirming
        that the polarization pathway produces correctly shaped output.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        pol = make_polarization_config(
            polarization_type="LVP"
        )
        spectrum = simulate_advanced(
            bands, orb_proj, params, pol
        )
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_unpolarized(self):
        """Verify that unpolarized light produces finite intensity values.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure unpolarized light.

        2. **Simulate**:
           Run ``simulate_advanced`` with unpolarized configuration,
           which averages s- and p-polarization contributions.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        Asserts
        -------
        All intensity values are finite, verifying that the unpolarized
        code path (averaging over orthogonal polarization vectors) does
        not introduce numerical instabilities.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(
            polarization_type="unpolarized"
        )
        spectrum = simulate_advanced(
            bands, orb_proj, params, pol
        )
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateExpert(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_expert`.

    Verifies the expert-level simulation (Voigt broadening with Yeh-Lindau
    cross-sections, polarization selection rules, and dipole matrix
    elements) including output tensor shape, finiteness under unpolarized
    light, and finiteness under circular (RCP) polarization.
    """

    def test_output_shape(self):
        """Verify that intensity has the expected shape under LHP polarization.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, create
           simulation parameters with fidelity=500, and configure linear
           horizontal polarization (LHP).

        2. **Simulate**:
           Run ``simulate_expert`` with the polarization config to
           produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        Asserts
        -------
        The intensity shape matches ``(n_kpoints, fidelity)``, confirming
        that the expert-level polarization and Voigt broadening pathway
        produces correctly shaped output.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        pol = make_polarization_config(
            polarization_type="LHP"
        )
        spectrum = simulate_expert(
            bands, orb_proj, params, pol
        )
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_unpolarized(self):
        """Verify that unpolarized light produces finite intensity values.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure unpolarized light.

        2. **Simulate**:
           Run ``simulate_expert`` with unpolarized configuration,
           which averages s- and p-polarization dipole contributions.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        Asserts
        -------
        All intensity values are finite, verifying that the unpolarized
        code path with Voigt broadening and dipole matrix elements does
        not introduce numerical instabilities.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(
            polarization_type="unpolarized"
        )
        spectrum = simulate_expert(
            bands, orb_proj, params, pol
        )
        chex.assert_tree_all_finite(spectrum.intensity)

    def test_circular_polarization(self):
        """Verify that right circular polarization (RCP) produces finite values.

        Test Logic
        ----------
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure right circular polarization (RCP).

        2. **Simulate**:
           Run ``simulate_expert`` with RCP configuration, which
           builds a complex electric-field vector and evaluates dipole
           matrix elements with Voigt broadening.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        Asserts
        -------
        All intensity values are finite, verifying that the circular
        polarization code path (complex-valued electric field) does not
        produce numerical overflow or undefined values in the expert-level
        simulation.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(
            polarization_type="RCP"
        )
        spectrum = simulate_expert(
            bands, orb_proj, params, pol
        )
        chex.assert_tree_all_finite(spectrum.intensity)
