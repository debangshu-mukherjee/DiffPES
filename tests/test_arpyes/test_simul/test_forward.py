"""Tests for the end-to-end differentiable forward model."""

import jax
import jax.numpy as jnp
import pytest

from arpyes.simul.forward import simulate_tb_radial
from arpyes.tightb import (
    diagonalize_tb,
    make_1d_chain_model,
    make_graphene_model,
)
from arpyes.types import (
    make_orbital_basis,
    make_polarization_config,
    make_self_energy_config,
    make_simulation_params,
    make_slater_params,
)


@pytest.fixture
def chain_setup():
    """1D chain model with Slater s-orbital."""
    model = make_1d_chain_model(t=-1.0)
    kpoints = jnp.linspace(-0.4, 0.4, 10)[:, None] * jnp.array(
        [[1.0, 0.0, 0.0]]
    )
    diag = diagonalize_tb(model, kpoints)
    basis = make_orbital_basis(
        n_values=(1,),
        l_values=(0,),
        m_values=(0,),
        labels=("1s",),
    )
    slater = make_slater_params(
        zeta=jnp.array([1.0]),
        orbital_basis=basis,
    )
    params = make_simulation_params(
        energy_min=-3.0,
        energy_max=3.0,
        fidelity=500,
        sigma=0.1,
        gamma=0.1,
        temperature=30.0,
        photon_energy=21.2,
    )
    pol = make_polarization_config(polarization_type="LHP")
    return diag, slater, params, pol


class TestSimulateTBRadial:
    """Tests for simulate_tb_radial."""

    def test_output_shape(self, chain_setup):
        """Output has correct shape."""
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert spectrum.intensity.shape == (10, 500)
        assert spectrum.energy_axis.shape == (500,)

    def test_output_finite(self, chain_setup):
        """All output values are finite."""
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))
        assert jnp.all(jnp.isfinite(spectrum.energy_axis))

    def test_output_non_negative(self, chain_setup):
        """Intensity is non-negative."""
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(spectrum.intensity >= -1e-15)

    def test_unpolarized(self, chain_setup):
        """Unpolarized mode runs without error."""
        diag, slater, params, _ = chain_setup
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_self_energy(self, chain_setup):
        """Self-energy mode runs without error."""
        diag, slater, params, pol = chain_setup
        se = make_self_energy_config(gamma=0.15, mode="constant")
        spectrum = simulate_tb_radial(
            diag, slater, params, pol, self_energy=se
        )
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_momentum_broadening(self, chain_setup):
        """Momentum broadening runs without error."""
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol, dk=0.05)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_work_function_effect(self, chain_setup):
        """Changing work function changes the spectrum."""
        diag, slater, params, pol = chain_setup
        spec1 = simulate_tb_radial(
            diag, slater, params, pol, work_function=4.0
        )
        spec2 = simulate_tb_radial(
            diag, slater, params, pol, work_function=6.0
        )
        # Intensities should differ
        assert not jnp.allclose(spec1.intensity, spec2.intensity)

    def test_gradient_wrt_zeta(self, chain_setup):
        """Gradient w.r.t. Slater exponent is finite."""
        diag, _, params, pol = chain_setup
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )

        def loss(zeta_val):
            sp = make_slater_params(
                zeta=jnp.array([zeta_val]),
                orbital_basis=basis,
            )
            spectrum = simulate_tb_radial(
                diag,
                sp,
                params,
                pol,
                r_grid=jnp.linspace(1e-6, 30.0, 2000),
            )
            return jnp.sum(spectrum.intensity)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad), f"Gradient w.r.t. zeta is {grad}"

    def test_gradient_wrt_hopping(self):
        """End-to-end gradient through TB diag + simulate is finite."""
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.3, 0.3, 5)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        slater = make_slater_params(
            zeta=jnp.array([1.0]),
            orbital_basis=basis,
        )
        params = make_simulation_params(
            energy_min=-3.0,
            energy_max=3.0,
            fidelity=200,
            sigma=0.1,
            gamma=0.1,
            temperature=30.0,
            photon_energy=21.2,
        )
        pol = make_polarization_config(polarization_type="LHP")

        def loss(hop):
            m = model._replace(hopping_params=hop)
            diag = diagonalize_tb(m, kpoints)
            spec = simulate_tb_radial(
                diag,
                slater,
                params,
                pol,
                r_grid=jnp.linspace(1e-6, 30.0, 2000),
            )
            return jnp.sum(spec.intensity)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(
            jnp.isfinite(grad)
        ), f"Gradient w.r.t. hopping is {grad}"

    def test_graphene_runs(self):
        """Graphene model with two pz orbitals runs end-to-end."""
        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 / 3, 1.0 / 3, 0.0],
                [2.0 / 3, 1.0 / 3, 0.0],
            ]
        )
        diag = diagonalize_tb(model, kpoints)
        basis = make_orbital_basis(
            n_values=(2, 2),
            l_values=(1, 1),
            m_values=(0, 0),
            labels=("A_pz", "B_pz"),
        )
        slater = make_slater_params(
            zeta=jnp.array([1.625, 1.625]),
            orbital_basis=basis,
        )
        params = make_simulation_params(
            energy_min=-10.0,
            energy_max=10.0,
            fidelity=300,
            sigma=0.2,
            gamma=0.2,
            temperature=30.0,
            photon_energy=21.2,
        )
        pol = make_polarization_config(polarization_type="LHP")
        spectrum = simulate_tb_radial(
            diag,
            slater,
            params,
            pol,
            r_grid=jnp.linspace(1e-6, 30.0, 2000),
        )
        assert spectrum.intensity.shape == (3, 300)
        assert jnp.all(jnp.isfinite(spectrum.intensity))
