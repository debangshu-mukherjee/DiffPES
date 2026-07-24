"""Validate the end-to-end differentiable forward model.

Extended Summary
----------------
Validates :func:`diffpes.simul.forward.simulate_tb_radial`, the Chinook-style
tight-binding forward model that combines tight-binding diagonalization,
Slater radial wavefunctions, dipole matrix elements, and Voigt broadening
into a single differentiable pipeline. The tests cover shapes, value
constraints, polarization modes, and optional broadening. They verify work
function sensitivity and derivatives for Slater exponents and hopping
parameters. The graphene model test verifies a multi-orbital,
multi-atom system runs end-to-end without error.

"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

import diffpes
from diffpes.simul import simulate_tb_radial
from diffpes.tightb import (
    diagonalize_tb,
)
from diffpes.types import (
    ArpesSpectrum,
    DiagonalizedBands,
    OrbitalBasis,
    PolarizationConfig,
    SimulationParams,
    SlaterParams,
    TBModel,
    make_orbital_basis,
    make_polarization_config,
    make_self_energy_config,
    make_simulation_params,
    make_slater_params,
)
from tests._factories import make_1d_chain_model, make_graphene_model


@pytest.fixture
@jaxtyped(typechecker=beartype)
def chain_setup() -> tuple[
    DiagonalizedBands,
    SlaterParams,
    SimulationParams,
    PolarizationConfig,
]:
    """Provide a 1D chain tight-binding model with a single Slater s-orbital.

    The fixture constructs a 1D chain with -1.0 eV nearest-neighbor hopping.
    It diagonalizes the model at ten k-points along the x-axis. The orbital
    basis is a single hydrogen-like 1s orbital (n=1, l=0, m=0) with Slater
    exponent zeta = 1.0. The simulation uses a [-3, 3] eV energy window,
    fidelity 500, Voigt broadening, 30 K, and 21.2 eV photons. The fixture
    uses LHP polarization.

    Most tests in ``TestSimulateTBRadial`` share this fixture. The analytical
    cosine band keeps diagonalization fast while exercising the complete
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
    model: TBModel = make_1d_chain_model(t=-1.0)
    kpoints: Float[Array, "10 3"] = jnp.linspace(-0.4, 0.4, 10)[
        :, None
    ] * jnp.array([[1.0, 0.0, 0.0]])
    diag: DiagonalizedBands = diagonalize_tb(model, kpoints)
    basis: OrbitalBasis = diag.basis
    slater: SlaterParams = make_slater_params(
        zeta=jnp.array([1.0]),
        orbital_basis=basis,
    )
    params: SimulationParams = make_simulation_params(
        energy_min=-3.0,
        energy_max=3.0,
        fidelity=500,
        sigma=0.1,
        gamma=0.1,
        temperature=30.0,
        photon_energy=21.2,
    )
    pol: PolarizationConfig = make_polarization_config(polarization_type="LHP")
    result: tuple[
        DiagonalizedBands,
        SlaterParams,
        SimulationParams,
        PolarizationConfig,
    ] = (diag, slater, params, pol)
    return result


class TestSimulateTBRadial:
    """Validate :func:`diffpes.simul.forward.simulate_tb_radial`.

    Validates the Chinook-style differentiable forward model that computes
    ARPES spectra from tight-binding Hamiltonians and Slater radial
    wavefunctions. The tests cover shapes, finite nonnegative values, and
    polarization modes. They cover optional broadening and work-function
    sensitivity. They reject mismatched Slater and diagonalized orbital
    metadata. They also verify derivatives for Slater exponents and
    tight-binding hopping amplitudes.
    A graphene test confirms multi-orbital, multi-atom generality.

    :see: :func:`~diffpes.simul.simulate_tb_radial`
    """

    def test_nonzero_fermi_level_controls_emission_threshold(
        self,
        chain_setup,
    ) -> None:
        """Verify Fermi-level subtraction before photoelectron kinematics.

        Set the absolute Fermi level to 30 eV. Every band then lies below the
        emission threshold for 21.2 eV photons and a 4.5 eV work function.
        The complete spectrum must vanish.

        Notes
        -----
        Replace only the immutable Fermi-level leaf and run the complete
        radial forward path at the original photon energy.
        """
        diag: DiagonalizedBands
        slater: SlaterParams
        params: SimulationParams
        pol: PolarizationConfig
        diag, slater, params, pol = chain_setup
        shifted_fermi: DiagonalizedBands = eqx.tree_at(
            lambda bands: bands.fermi_energy,
            diag,
            jnp.array(30.0),
        )
        spectrum: ArpesSpectrum = simulate_tb_radial(
            shifted_fermi,
            slater,
            params,
            pol,
            work_function=4.5,
        )
        assert bool(jnp.all(spectrum.intensity == 0.0))

    def test_nonorthogonal_reciprocal_map_sets_cartesian_direction(
        self,
        chain_setup,
    ) -> None:
        """Verify Cartesian direction handling on a nonorthogonal lattice.

        Two carriers encode the same Cartesian momentum direction with
        different fractional coordinates and reciprocal bases. Their radial
        spectra must agree.

        Notes
        -----
        Replace reciprocal and fractional leaves in one carrier. Construct a
        second carrier with equivalent coordinates in the original basis.
        """
        diag: DiagonalizedBands
        slater: SlaterParams
        params: SimulationParams
        pol: PolarizationConfig
        diag, slater, params, pol = chain_setup
        skew_reciprocal: Float[Array, "3 3"] = jnp.array(
            [
                [2.0 * jnp.pi, 0.0, 0.0],
                [jnp.pi, 2.0 * jnp.pi, 0.0],
                [0.0, 0.0, 2.0 * jnp.pi],
            ]
        )
        skew_fractional: Float[Array, "10 3"] = jnp.broadcast_to(
            jnp.array([0.2, 0.3, 0.1]),
            diag.kpoints.shape,
        )
        cartesian: Float[Array, "10 3"] = skew_fractional @ skew_reciprocal
        equivalent_fractional: Float[Array, "10 3"] = (
            cartesian @ jnp.linalg.inv(diag.geometry.reciprocal)
        )
        skew_diag: DiagonalizedBands = eqx.tree_at(
            lambda bands: (bands.geometry.reciprocal, bands.kpoints),
            diag,
            (skew_reciprocal, skew_fractional),
        )
        reference_diag: DiagonalizedBands = eqx.tree_at(
            lambda bands: bands.kpoints,
            diag,
            equivalent_fractional,
        )
        skew_spectrum: ArpesSpectrum = simulate_tb_radial(
            skew_diag,
            slater,
            params,
            pol,
        )
        reference_spectrum: ArpesSpectrum = simulate_tb_radial(
            reference_diag,
            slater,
            params,
            pol,
        )
        chex.assert_trees_all_close(
            skew_spectrum.intensity,
            reference_spectrum.intensity,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_output_shape(self, chain_setup) -> None:
        """Verify that the spectrum intensity and energy axis have expected shapes.

        The test establishes the output shape contract for simulate t b radial with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the ``chain_setup`` fixture (10 k-points,
           fidelity=500, 1D chain with one s-orbital, LHP polarization).
        2. **Simulate**: Call ``simulate_tb_radial`` with the fixture
           inputs to produce an ``ArpesSpectrum``.
        3. **Check shapes**: Assert ``intensity`` shape is ``(10, 500)``
           (n_kpoints x fidelity) and ``energy_axis`` shape is ``(500,)``.

        **Expected assertions**

        The intensity has shape ``(n_kpoints, fidelity)`` and the energy
        axis has length ``fidelity``, confirming that the forward model
        correctly maps k-points and energy grid dimensions.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert spectrum.intensity.shape == (10, 500)
        assert spectrum.energy_axis.shape == (500,)

    def test_rejects_slater_basis_mismatch(self, chain_setup) -> None:
        """Reject Slater metadata that differs from the diagonalized basis.

        The eigensystem coefficients and radial parameters must describe the
        same ordered orbitals. A same-sized but different principal quantum
        number must therefore fail before numerical simulation begins.

        Notes
        -----
        Build a valid one-orbital ``2s`` Slater basis against the fixture's
        ``1s`` diagonalized basis and require the explicit ``ValueError``.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        mismatch_basis: diffpes.types.OrbitalBasis
        mismatch_slater: diffpes.types.SlaterParams

        diag, slater, params, pol = chain_setup
        del slater
        mismatch_basis = make_orbital_basis(
            atom_indices=diag.basis.atom_indices,
            n=(2,),
            l=(0,),
            m=(0,),
            spin=diag.basis.spin,
            labels=("2s",),
        )
        mismatch_slater = make_slater_params(
            zeta=jnp.array([1.0]),
            orbital_basis=mismatch_basis,
        )
        with pytest.raises(
            ValueError,
            match="orbital_basis must equal diag_bands.basis",
        ):
            simulate_tb_radial(
                diag,
                mismatch_slater,
                params,
                pol,
                r_grid=jnp.linspace(1e-6, 5.0, 16),
            )

    def test_output_finite(self, chain_setup) -> None:
        """Verify that all intensity and energy axis values are finite.

        The test establishes the output finite contract for simulate t b radial with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the ``chain_setup`` fixture (1D chain, LHP).
        2. **Simulate**: Call ``simulate_tb_radial`` to produce a spectrum.
        3. **Check finiteness**: Check every intensity and energy-axis value.

        **Expected assertions**

        All elements of ``intensity`` and ``energy_axis`` are finite,
        confirming numerical stability of the radial integration, dipole
        matrix element computation, and broadening stages.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))
        assert jnp.all(jnp.isfinite(spectrum.energy_axis))

    def test_output_non_negative(self, chain_setup) -> None:
        """Verify that the simulated ARPES intensity is non-negative.

        The test establishes the output non negative contract for simulate t b radial
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the ``chain_setup`` fixture (1D chain, LHP).
        2. **Simulate**: Call ``simulate_tb_radial`` to produce a spectrum.
        3. **Check non-negativity**: Assert all intensity values are
           >= -1e-15 (allowing negligible floating-point undershoot).

        **Expected assertions**

        All intensity values are effectively non-negative, consistent with
        the physical interpretation of ARPES intensity as a squared
        matrix element weighted by spectral function and occupation.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(spectrum.intensity >= -1e-15)

    def test_unpolarized(self, chain_setup) -> None:
        """Verify that unpolarized light mode produces finite intensity.

        The test establishes the unpolarized contract for simulate t b radial with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the ``chain_setup`` fixture but override the
           polarization config to ``"unpolarized"``, which averages
           contributions from orthogonal s- and p-polarization vectors.
        2. **Simulate**: Call ``simulate_tb_radial`` with the unpolarized
           configuration.
        3. **Check finiteness**: Assert all intensity values are finite.

        **Expected assertions**

        The unpolarized path produces finite intensity values. This result
        verifies stable averaging of the polarization contributions.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, _ = chain_setup
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_tb_radial(diag, slater, params, pol)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_self_energy(self, chain_setup) -> None:
        """Verify that the optional self-energy extension produces finite output.

        The test establishes the with self energy contract for simulate t b radial with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the fixture and a constant 0.15 eV self-energy.
        2. **Simulate**: Call ``simulate_tb_radial`` with the
           ``self_energy`` keyword argument.
        3. **Check finiteness**: Assert all intensity values are finite.

        **Expected assertions**

        A constant self-energy produces finite intensity values. This result
        verifies its integration with the forward model.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        se: diffpes.types.SelfEnergyConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        se = make_self_energy_config(gamma=0.15, mode="constant")
        spectrum = simulate_tb_radial(
            diag, slater, params, pol, self_energy=se
        )
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_with_momentum_broadening(self, chain_setup) -> None:
        """Verify that the optional momentum-broadening extension produces finite output.

        The test establishes the with momentum broadening contract for simulate t b
        radial with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the fixture and ``dk=0.05`` inverse Angstrom.
        2. **Simulate**: Call ``simulate_tb_radial`` with the ``dk``
           keyword argument.
        3. **Check finiteness**: Assert all intensity values are finite.

        **Expected assertions**

        Momentum broadening produces finite intensity values. This result
        verifies its integration with the forward model.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        spectrum = simulate_tb_radial(diag, slater, params, pol, dk=0.05)
        assert jnp.all(jnp.isfinite(spectrum.intensity))

    def test_work_function_effect(self, chain_setup) -> None:
        """Verify that different work function values produce different spectra.

        The test establishes the work function effect contract for simulate t b radial
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the ``chain_setup`` fixture and run the
           simulation twice with work_function=4.0 eV and
           work_function=6.0 eV. The work function enters the
           photoelectron kinetic energy calculation
           (``Ek = hv - phi - |Eb|``),
           which affects the Slater radial matrix element via the
           outgoing plane-wave momentum.
        2. **Simulate both**: Call ``simulate_tb_radial`` twice with
           different work functions.
        3. **Compare**: Assert the two intensity arrays are not
           element-wise close (``jnp.allclose`` returns False).

        **Expected assertions**

        Different work functions produce different intensity maps. This
        result verifies propagation through the radial matrix element.
        """
        diag: diffpes.types.DiagonalizedBands
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spec1: diffpes.types.ArpesSpectrum
        spec2: diffpes.types.ArpesSpectrum

        diag, slater, params, pol = chain_setup
        spec1 = simulate_tb_radial(
            diag, slater, params, pol, work_function=4.0
        )
        spec2 = simulate_tb_radial(
            diag, slater, params, pol, work_function=6.0
        )

        assert not jnp.allclose(spec1.intensity, spec2.intensity)

    def test_gradient_wrt_zeta(self, chain_setup) -> None:
        """Verify that the gradient of total intensity w.r.t. Slater exponent is finite.

        The test establishes the gradient wrt zeta contract for simulate t b radial
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Use the fixture without its Slater parameters.
           Define a scalar loss with new parameters and an explicit radial grid.
        2. **Differentiate**: Call ``jax.grad(loss)(1.0)`` to compute
           the gradient of the loss with respect to the Slater exponent.
        3. **Check finiteness**: Assert the gradient is finite.

        **Expected assertions**

        The Slater-exponent gradient is finite. This result verifies JAX
        differentiation through the radial integration without singularities.
        """
        diag: diffpes.types.DiagonalizedBands
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        basis: diffpes.types.OrbitalBasis
        grad: Array

        diag, _, params, pol = chain_setup
        basis = diag.basis

        def loss(zeta_val):
            sp: diffpes.types.SlaterParams
            spectrum: diffpes.types.ArpesSpectrum

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

    def test_gradient_wrt_hopping(self) -> None:
        """Verify end-to-end gradient through TB diagonalization and simulation.

        The test establishes the gradient wrt hopping contract for simulate t b radial
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Build a five-point chain with one 1s orbital.
           Define a scalar loss that replaces the hopping parameters.
           Diagonalize the model and simulate with an explicit radial grid.
        2. **Differentiate**: Call
           ``jax.grad(loss)(model.hopping_amplitudes)``
           to compute the gradient of the total intensity with respect to
           the tight-binding hopping parameter(s).
        3. **Check finiteness**: Assert all gradient components are finite.

        **Expected assertions**

        All hopping gradients are finite. This result verifies JAX
        differentiation through the complete forward model.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        basis: diffpes.types.OrbitalBasis
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        grad: Array

        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.3, 0.3, 5)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )
        basis = model.basis
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
            m: Array
            diag: diffpes.types.DiagonalizedBands
            spec: diffpes.types.ArpesSpectrum

            m = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                hop,
            )
            diag = diagonalize_tb(m, kpoints)
            spec = simulate_tb_radial(
                diag,
                slater,
                params,
                pol,
                r_grid=jnp.linspace(1e-6, 30.0, 2000),
            )
            return jnp.sum(spec.intensity)

        grad = jax.grad(loss)(model.hopping_amplitudes)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Gradient w.r.t. hopping is {grad}"
        )

    def test_graphene_runs(self) -> None:
        """Verify that a graphene model with two pz orbitals runs end-to-end.

        The test establishes the graphene runs contract for simulate t b radial with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Build a graphene tight-binding model with
           nearest-neighbour hopping t=-2.7 eV (standard graphene value).
           Use three high-symmetry k-points and two carbon pz orbitals.
           Use a wide energy window and an explicit radial grid.
        2. **Simulate**: Call ``simulate_tb_radial`` with LHP
           polarization.
        3. **Check shape and finiteness**: Assert intensity shape is
           ``(3, 300)`` and all values are finite.

        **Expected assertions**

        The intensity has the expected shape and finite values. This result
        verifies the forward model for a graphene system with multiple atoms.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands
        basis: diffpes.types.OrbitalBasis
        slater: diffpes.types.SlaterParams
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 / 3, 1.0 / 3, 0.0],
                [2.0 / 3, 1.0 / 3, 0.0],
            ]
        )
        diag = diagonalize_tb(model, kpoints)
        basis = diag.basis
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
