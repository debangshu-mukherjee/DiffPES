"""Tests for expanded-input simulation wrappers."""

import chex
import jax.numpy as jnp

from arpyes.simul import (
    make_expanded_simulation_params,
    simulate_advanced,
    simulate_advanced_expanded,
    simulate_basic,
    simulate_basic_expanded,
    simulate_expert_expanded,
    simulate_expanded,
)
from arpyes.types import (
    make_band_structure,
    make_orbital_projection,
    make_polarization_config,
)


def _make_synthetic_data(nk=12, nb=4, na=3):
    eigenbands = jnp.linspace(
        -2.5, 0.75, nk * nb, dtype=jnp.float64
    ).reshape(nk, nb)
    surface_orb = jnp.ones((nk, nb, na, 9), dtype=jnp.float64)
    surface_orb = surface_orb * 0.1
    return eigenbands, surface_orb


class TestExpandedParams(chex.TestCase):

    def test_energy_window_matches_expanded_default(self):
        eigenbands = jnp.array(
            [[-2.0, 0.25], [1.0, -1.0]], dtype=jnp.float64
        )
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=100,
        )
        chex.assert_trees_all_close(
            params.energy_min, jnp.float64(-3.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            params.energy_max, jnp.float64(2.0), atol=1e-12
        )
        chex.assert_equal(params.fidelity, 100)


class TestExpandedBasicWrapper(chex.TestCase):

    def test_matches_core_basic_simulation(self):
        eigenbands, surface_orb = _make_synthetic_data()
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=240,
            sigma=0.06,
            temperature=20.0,
            photon_energy=35.0,
        )
        kpoints = jnp.zeros(
            (eigenbands.shape[0], 3), dtype=jnp.float64
        )
        bands = make_band_structure(
            eigenvalues=eigenbands,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        orb_proj = make_orbital_projection(
            projections=surface_orb
        )
        expected = simulate_basic(bands, orb_proj, params)
        wrapped = simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.06,
            fidelity=240,
            temperature=20.0,
            photon_energy=35.0,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            wrapped.energy_axis, expected.energy_axis, atol=1e-12
        )


class TestExpandedAdvancedWrapper(chex.TestCase):

    def test_degree_conversion_matches_core_advanced(self):
        eigenbands, surface_orb = _make_synthetic_data()
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=220,
            sigma=0.05,
            temperature=25.0,
            photon_energy=21.2,
        )
        kpoints = jnp.zeros(
            (eigenbands.shape[0], 3), dtype=jnp.float64
        )
        bands = make_band_structure(
            eigenvalues=eigenbands,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        orb_proj = make_orbital_projection(
            projections=surface_orb
        )
        pol = make_polarization_config(
            theta=jnp.deg2rad(jnp.float64(45.0)),
            phi=jnp.deg2rad(jnp.float64(30.0)),
            polarization_angle=jnp.float64(0.25),
            polarization_type="LHP",
        )
        expected = simulate_advanced(bands, orb_proj, params, pol)
        wrapped = simulate_advanced_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.05,
            fidelity=220,
            temperature=25.0,
            photon_energy=21.2,
            polarization="LHP",
            incident_theta=45.0,
            incident_phi=30.0,
            polarization_angle=0.25,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )


class TestExpandedDispatch(chex.TestCase):

    def test_dispatch_expert_matches_direct_wrapper(self):
        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_expert_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        dispatched = simulate_expanded(
            level="Expert",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )
