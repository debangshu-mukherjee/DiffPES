"""Tests for the end-to-end differentiable forward model.

Extended Summary
----------------
Validates :func:`diffpes.simul.forward.simulate_tb_radial`, the Chinook-style
tight-binding forward model that combines tight-binding diagonalization,
Slater radial wavefunctions, dipole matrix elements, and Voigt broadening
into a single differentiable pipeline. Tests cover output shape and value
constraints (finiteness, non-negativity), polarization modes (LHP,
unpolarized), self-energy and momentum-broadening extensions, sensitivity
to the work function parameter, and -- critically -- end-to-end
differentiability with respect to Slater exponents and hopping parameters
via ``jax.grad``. The graphene model test verifies a multi-orbital,
multi-atom system runs end-to-end without error.

Routine Listings
----------------
:func:`chain_setup`
    Pytest fixture providing a 1D chain tight-binding model.
:class:`TestSimulateTBRadial`
    Tests for simulate_tb_radial.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.simul.forward import simulate_tb_radial
from diffpes.tightb import (
    diagonalize_tb,
    make_1d_chain_model,
    make_graphene_model,
)
from diffpes.types import (
    make_orbital_basis,
    make_polarization_config,
    make_self_energy_config,
    make_simulation_params,
    make_slater_params,
)


@pytest.fixture
def chain_setup():
    """Provide a 1D chain tight-binding model with a single Slater s-orbital.

    Constructs the simplest possible tight-binding system: a 1D chain with
    nearest-neighbour hopping t = -1.0 eV, diagonalized at 10 k-points along
    the x-axis from -0.4 to 0.4 (in reciprocal lattice units). The orbital
    basis is a single hydrogen-like 1s orbital (n=1, l=0, m=0) with Slater
    exponent zeta = 1.0. Simulation parameters use a wide energy window
    [-3, 3] eV, fidelity 500, Voigt broadening (sigma=0.1, gamma=0.1),
    temperature 30 K, and photon energy 21.2 eV (He-I). Polarization is
    set to LHP (linear horizontal).

    This fixture is shared across most tests in ``TestSimulateTBRadial``
    because the 1D chain model is analytically tractable (cosine band)
    and fast to diagonalize, while still exercising all stages of the
    forward pipeline.

    Returns
    -------
    diag : TBDiagonalization
        Diagonalized tight-binding model at 10 k-points.
    slater : SlaterParams
        Slater radial parameters with zeta=1.0 for a single 1s orbital.
    params : SimulationParams
        Simulation parameters (energy window, broadening, temperature,
        photon energy).
    pol : PolarizationConfig
        LHP polarization configuration.
    """
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
    """Tests for :func:`diffpes.simul.forward.simulate_tb_radial`.

    Validates the Chinook-style differentiable forward model that computes
    ARPES spectra from tight-binding Hamiltonians and Slater radial
    wavefunctions. Tests cover basic output correctness (shape, finiteness,
    non-negativity), multiple polarization modes, optional self-energy and
    momentum broadening extensions, work-function sensitivity, and -- most
    importantly -- end-to-end differentiability via ``jax.grad`` with respect
    to both Slater exponents (zeta) and tight-binding hopping parameters.
    A graphene test confirms multi-orbital, multi-atom generality.
    """

    def test_output_shape(self, chain_setup):
        """Verify that the spectrum intensity and energy axis have expected shapes.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture (10 k-points,
           fidelity=500, 1D chain with one s-orbital, LHP polarization).
        2. **Simulate**: Call ``simulate_tb_radial`` with the fixture
           inputs to produce an ``ArpesSpectrum``.
        3. **Check shapes**: Assert ``intensity`` shape is ``(10, 500)``
           (n_kpoints x fidelity) and ``energy_axis`` shape is ``(500,)``.

        Asserts
        -------
        The intensity has shape ``(n_kpoints, fidelity)`` and the energy
        axis has length ``fidelity``, confirming that the forward model
        correctly maps k-points and energy grid dimensions.
        """
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert spectrum.intensity.shape == (10, 500)
        assert spectrum.energy_axis.shape == (500,)

    def test_output_finite(self, chain_setup):
        """Verify that all intensity and energy axis values are finite.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture (1D chain, LHP).
        2. **Simulate**: Call ``simulate_tb_radial`` to produce a spectrum.
        3. **Check finiteness**: Assert every element of both ``intensity``
           and ``energy_axis`` is finite (no NaN or Inf), which could arise
           from overflow in the Slater radial integration or division by
           zero in the Voigt profile evaluation.

        Asserts
        -------
        All elements of ``intensity`` and ``energy_axis`` are finite,
        confirming numerical stability of the radial integration, dipole
        matrix element computation, and broadening stages.
        """
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))
        assert jnp.all(jnp.isfinite(spectrum.energy_axis))

    def test_output_non_negative(self, chain_setup):
        """Verify that the simulated ARPES intensity is non-negative.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture (1D chain, LHP).
        2. **Simulate**: Call ``simulate_tb_radial`` to produce a spectrum.
        3. **Check non-negativity**: Assert all intensity values are
           >= -1e-15 (allowing negligible floating-point undershoot).

        Asserts
        -------
        All intensity values are effectively non-negative, consistent with
        the physical interpretation of ARPES intensity as a squared
        matrix element weighted by spectral function and occupation.
        """
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(spectrum.intensity >= -1e-15)

    def test_unpolarized(self, chain_setup):
        """Verify that unpolarized light mode produces finite intensity.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture but override the
           polarization config to ``"unpolarized"``, which averages
           contributions from orthogonal s- and p-polarization vectors.
        2. **Simulate**: Call ``simulate_tb_radial`` with the unpolarized
           configuration.
        3. **Check finiteness**: Assert all intensity values are finite.

        Asserts
        -------
        All intensity values are finite under the unpolarized code path,
        confirming that the polarization-averaging branch does not
        introduce numerical instabilities in the dipole matrix element
        computation.
        """
        diag, slater, params, _ = chain_setup
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_self_energy(self, chain_setup):
        """Verify that the optional self-energy extension produces finite output.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture and create a
           constant-mode self-energy config with gamma=0.15 eV, which
           adds an energy-independent imaginary part to the Green's
           function, broadening the spectral features.
        2. **Simulate**: Call ``simulate_tb_radial`` with the
           ``self_energy`` keyword argument.
        3. **Check finiteness**: Assert all intensity values are finite.

        Asserts
        -------
        All intensity values are finite when a constant self-energy is
        applied, confirming that the self-energy evaluation integrates
        correctly with the forward model pipeline.
        """
        diag, slater, params, pol = chain_setup
        se = make_self_energy_config(gamma=0.15, mode="constant")
        spectrum = simulate_tb_radial(
            diag, slater, params, pol, self_energy=se
        )
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_momentum_broadening(self, chain_setup):
        """Verify that the optional momentum-broadening extension produces finite output.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture and pass dk=0.05
           (inverse angstroms), which applies Gaussian broadening along
           the k-axis to simulate finite angular resolution of the
           analyser.
        2. **Simulate**: Call ``simulate_tb_radial`` with the ``dk``
           keyword argument.
        3. **Check finiteness**: Assert all intensity values are finite.

        Asserts
        -------
        All intensity values are finite when momentum broadening is
        applied, confirming that ``apply_momentum_broadening`` integrates
        correctly with the forward model without introducing NaN or Inf.
        """
        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol, dk=0.05)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_work_function_effect(self, chain_setup):
        """Verify that different work function values produce different spectra.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture and run the
           simulation twice with work_function=4.0 eV and
           work_function=6.0 eV. The work function enters the
           photoelectron kinetic energy calculation (Ek = hv - phi - |Eb|),
           which affects the Slater radial matrix element via the
           outgoing plane-wave momentum.
        2. **Simulate both**: Call ``simulate_tb_radial`` twice with
           different work functions.
        3. **Compare**: Assert the two intensity arrays are not
           element-wise close (``jnp.allclose`` returns False).

        Asserts
        -------
        The intensity maps differ for different work functions,
        confirming that the work function parameter propagates through
        the radial matrix element calculation and affects the final
        ARPES intensity.
        """
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
        """Verify that the gradient of total intensity w.r.t. Slater exponent is finite.

        Test Logic
        ----------
        1. **Setup**: Use the ``chain_setup`` fixture (discarding its
           slater params) and define a scalar loss function that creates
           new ``SlaterParams`` from a single zeta value, runs
           ``simulate_tb_radial`` with an explicit r_grid of 2000 points
           from 1e-6 to 30.0 bohr, and returns the sum of all
           intensities.
        2. **Differentiate**: Call ``jax.grad(loss)(1.0)`` to compute
           the gradient of the loss with respect to the Slater exponent.
        3. **Check finiteness**: Assert the gradient is finite.

        Asserts
        -------
        The gradient w.r.t. the Slater exponent zeta is finite, proving
        that the radial integration (Slater wavefunction times spherical
        Bessel function) is differentiable through JAX and that no
        numerical singularities arise during backpropagation. This is
        essential for inverse fitting of orbital parameters.
        """
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
        """Verify end-to-end gradient through TB diagonalization and simulation.

        Test Logic
        ----------
        1. **Setup**: Build a fresh 1D chain model (t=-1.0), 5 k-points,
           a single 1s Slater orbital (zeta=1.0), simulation params with
           fidelity=200, and LHP polarization. Define a scalar loss
           function that replaces the model's hopping_params, re-
           diagonalizes, runs ``simulate_tb_radial`` with an explicit
           r_grid, and returns the sum of all intensities.
        2. **Differentiate**: Call ``jax.grad(loss)(model.hopping_params)``
           to compute the gradient of the total intensity with respect to
           the tight-binding hopping parameter(s).
        3. **Check finiteness**: Assert all gradient components are finite.

        Asserts
        -------
        All gradient components w.r.t. hopping parameters are finite,
        demonstrating that the full pipeline -- Hamiltonian construction,
        eigenvalue decomposition via ``diagonalize_tb``, radial matrix
        elements, broadening, and summation -- is end-to-end
        differentiable through JAX. This is the core requirement for
        inverse tight-binding fitting.
        """
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
        """Verify that a graphene model with two pz orbitals runs end-to-end.

        Test Logic
        ----------
        1. **Setup**: Build a graphene tight-binding model with
           nearest-neighbour hopping t=-2.7 eV (standard graphene value).
           Use 3 high-symmetry k-points (Gamma, K, M) and define two
           pz orbitals (n=2, l=1, m=0) for the A and B sublattice atoms,
           each with Slater exponent zeta=1.625 (carbon 2p). Simulation
           params use a wide energy window [-10, 10] eV, fidelity=300,
           broader Voigt widths (sigma=0.2, gamma=0.2), temperature
           30 K, photon energy 21.2 eV, and an explicit r_grid of 2000
           points.
        2. **Simulate**: Call ``simulate_tb_radial`` with LHP
           polarization.
        3. **Check shape and finiteness**: Assert intensity shape is
           ``(3, 300)`` and all values are finite.

        Asserts
        -------
        The intensity has shape ``(n_kpoints, fidelity)`` and is
        entirely finite, confirming that the forward model generalizes
        beyond the single-orbital 1D chain to a multi-atom, multi-orbital
        system with known physical relevance (graphene's Dirac cone band
        structure).
        """
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
