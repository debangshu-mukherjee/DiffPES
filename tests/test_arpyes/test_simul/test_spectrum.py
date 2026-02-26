"""Tests for ARPES simulation spectrum functions."""

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

    def test_output_shape(self):
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_novice(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))
        chex.assert_shape(spectrum.energy_axis, (500,))

    def test_nonnegative_intensity(self):
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_novice(bands, orb_proj, params)
        assert float(jnp.min(spectrum.intensity)) >= -1e-10


class TestSimulateBasic(chex.TestCase):

    def test_output_shape(self):
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_finite_values(self):
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateBasicplus(chex.TestCase):

    def test_output_shape(self):
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basicplus(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_yeh_lindau_affects_weights(self):
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

    def test_output_shape(self):
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

    def test_output_shape(self):
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
        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(
            polarization_type="RCP"
        )
        spectrum = simulate_expert(
            bands, orb_proj, params, pol
        )
        chex.assert_tree_all_finite(spectrum.intensity)
