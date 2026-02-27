"""Tests for ARPES simulation spectrum functions.

Extended Summary
----------------
Validates the simulation levels in :mod:`arpyes.simul.spectrum`:
novice, basic, basicplus, advanced, expert, and soc. Tests check
output tensor shapes (intensity and energy_axis), value constraints
(non-negativity, finiteness), and sensitivity to physical parameters
(photon energy, polarization type, spin-orbit scale). Synthetic band
structure and orbital projections are built via _make_synthetic_data.
SOC tests verify that spin is required and that ls_scale=0 matches
expert. All test logic and assertions are documented in the
docstrings of each test class and method.

Routine Listings
----------------
:class:`TestSimulateAdvanced`
    Tests for simulate_advanced.
:class:`TestSimulateBasic`
    Tests for simulate_basic.
:class:`TestSimulateBasicplus`
    Tests for simulate_basicplus.
:class:`TestSimulateExpert`
    Tests for simulate_expert.
:class:`TestSimulateNovice`
    Tests for simulate_novice.
:class:`TestSimulateSoc`
    Tests for simulate_soc.
:func:`_make_synthetic_data`
    Helper to build synthetic bands and orbital projections.
:func:`_make_synthetic_data_with_spin`
    Helper to build synthetic data including spin for SOC tests.
"""

import chex
import jax.numpy as jnp

from arpyes.simul.spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
    simulate_soc,
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


def _make_synthetic_data_with_spin(nk=20, nb=5, na=2):
    """Generate synthetic band and orbital data including spin projections.

    Same as :func:`_make_synthetic_data` but adds a non-zero spin array
    of shape ``(nk, nb, na, 6)`` (x up/down, y up/down, z up/down) and
    returns an :class:`~arpyes.types.OrbitalProjection` built with
    :func:`~arpyes.types.make_orbital_projection` including the spin
    argument. Used by SOC simulation tests.

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
        Band structure from :func:`_make_synthetic_data`.
    orb_with_spin : OrbitalProjection
        Orbital projections with ``spin`` set to a synthetic
        (K, B, A, 6) array.
    """
    bands, orb_proj = _make_synthetic_data(nk=nk, nb=nb, na=na)
    spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
    spin = spin.at[..., 0].set(0.1)
    spin = spin.at[..., 1].set(0.05)
    spin = spin.at[..., 2].set(0.0)
    spin = spin.at[..., 3].set(0.0)
    spin = spin.at[..., 4].set(0.2)
    spin = spin.at[..., 5].set(0.1)
    orb_with_spin = make_orbital_projection(
        projections=orb_proj.projections,
        spin=spin,
    )
    return bands, orb_with_spin


class TestSimulateSoc(chex.TestCase):
    """Tests for :func:`arpyes.simul.spectrum.simulate_soc`.

    Verifies that the SOC simulation requires non-None spin data,
    produces the expected output shapes, that ``ls_scale=0`` recovers
    the expert result, and that non-zero ``ls_scale`` changes the
    intensity relative to expert.
    """

    def test_requires_spin_raises(self):
        """Verify that simulate_soc raises when orb_proj.spin is None.

        Test Logic
        ----------
        1. **Setup**: Use synthetic data without spin (orb_proj from
           :func:`_make_synthetic_data` has ``spin=None``).
        2. **Call**: Invoke ``simulate_soc`` with bands, orb_proj,
           params, and pol_config.
        3. **Check**: Assert that a ``ValueError`` is raised and that
           the exception message contains ``"spin"``.

        Asserts
        -------
        Calling ``simulate_soc`` with no spin data raises
        ``ValueError`` with a message referring to spin.
        """
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        with self.assertRaises(ValueError) as ctx:
            simulate_soc(bands, orb_proj, params, pol)
        self.assertIn("spin", str(ctx.exception))

    def test_output_shape(self):
        """Verify SOC spectrum has intensity (K, E) and energy_axis (E).

        Test Logic
        ----------
        1. **Setup**: Build bands and orb_proj with spin via
           :func:`_make_synthetic_data_with_spin`; params with
           fidelity=100; unpolarized polarization config.
        2. **Simulate**: Run ``simulate_soc`` with ls_scale=0.01.
        3. **Check shapes**: Assert intensity shape ``(n_kpoints, 100)``
           and energy_axis shape ``(100,)``; assert all intensity values
           are finite.

        Asserts
        -------
        Output shapes match the expert convention and the spectrum
        contains no NaN or Inf.
        """
        bands, orb_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_soc(
            bands, orb_proj, params, pol, ls_scale=0.01
        )
        chex.assert_shape(spectrum.intensity, (bands.eigenvalues.shape[0], 100))
        chex.assert_shape(spectrum.energy_axis, (100,))
        chex.assert_tree_all_finite(spectrum.intensity)

    def test_soc_ls_scale_zero_matches_expert(self):
        """With ls_scale=0, SOC intensity should match expert (same orbital part).

        Test Logic
        ----------
        1. **Setup**: Synthetic data with spin; params and
           unpolarized pol_config.
        2. **Run both**: Call ``simulate_expert`` and
           ``simulate_soc(..., ls_scale=0.0)``.
        3. **Compare**: Assert intensity and energy_axis from both
           match to within 1e-12 absolute tolerance.

        Asserts
        -------
        When the spin-orbit correction is disabled (ls_scale=0),
        the SOC simulation reproduces the expert result exactly.
        """
        bands, orb_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        expert_spec = simulate_expert(bands, orb_proj, params, pol)
        soc_spec = simulate_soc(
            bands, orb_proj, params, pol, ls_scale=0.0
        )
        chex.assert_trees_all_close(
            expert_spec.intensity, soc_spec.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            expert_spec.energy_axis, soc_spec.energy_axis, atol=1e-12
        )

    def test_soc_nonzero_ls_scale_differs_from_expert(self):
        """With ls_scale != 0, SOC intensity can differ from expert.

        Test Logic
        ----------
        1. **Setup**: Synthetic data with spin; params and
           unpolarized pol_config.
        2. **Run both**: Call ``simulate_expert`` and
           ``simulate_soc(..., ls_scale=0.1)``.
        3. **Compare**: Compute the maximum absolute difference
           between the two intensity arrays; assert it is greater
           than 1e-10.

        Asserts
        -------
        A non-zero ls_scale produces a different intensity map
        from the expert-only simulation, confirming the spin
        correction is applied.
        """
        bands, orb_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        expert_spec = simulate_expert(bands, orb_proj, params, pol)
        soc_spec = simulate_soc(
            bands, orb_proj, params, pol, ls_scale=0.1
        )
        diff = jnp.abs(soc_spec.intensity - expert_spec.intensity)
        self.assertGreater(jnp.max(diff), 1e-10)

    def test_soc_polarized_light(self):
        """Verify SOC runs with polarized light (hits polarized branch in simulate_soc)."""
        bands, orb_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(
            polarization_type="LHP",
        )
        spectrum = simulate_soc(
            bands, orb_proj, params, pol, ls_scale=0.02
        )
        chex.assert_shape(spectrum.intensity, (bands.eigenvalues.shape[0], 100))
        chex.assert_tree_all_finite(spectrum.intensity)
