"""Tests for expanded-input simulation wrappers.

Validates the convenience wrappers in :mod:`arpyes.simul.expanded` that
accept plain arrays and scalars instead of PyTree structures. Tests
verify that auto-derived energy windows are correct, that each expanded
wrapper produces results identical to the corresponding core simulation
function, and that the level-dispatch function routes correctly.
"""

import chex
import jax.numpy as jnp

from arpyes.simul import (
    make_expanded_simulation_params,
    simulate_advanced,
    simulate_advanced_expanded,
    simulate_basic,
    simulate_basic_expanded,
    simulate_expanded,
    simulate_expert_expanded,
    simulate_soc,
    simulate_soc_expanded,
)
from arpyes.types import (
    make_band_structure,
    make_orbital_projection,
    make_polarization_config,
)


def _make_synthetic_data(nk=12, nb=4, na=3):
    """Generate synthetic eigenband and orbital projection arrays for testing.

    Creates raw arrays (not wrapped in PyTrees) suitable for the
    expanded-input API. Eigenvalues are linearly spaced from -2.5 to
    0.75 eV in float64 precision, and all orbital projections are set
    to a uniform value of 0.1.

    Parameters
    ----------
    nk : int, optional
        Number of k-points. Default is 12.
    nb : int, optional
        Number of bands. Default is 4.
    na : int, optional
        Number of atoms. Default is 3.

    Returns
    -------
    eigenbands : jnp.ndarray
        Band eigenvalues of shape ``(nk, nb)`` in float64, linearly
        spaced from -2.5 to 0.75 eV.
    surface_orb : jnp.ndarray
        Uniform orbital projections of shape ``(nk, nb, na, 9)`` in
        float64 with all entries set to 0.1.
    """
    eigenbands = jnp.linspace(
        -2.5, 0.75, nk * nb, dtype=jnp.float64
    ).reshape(nk, nb)
    surface_orb = jnp.ones((nk, nb, na, 9), dtype=jnp.float64)
    surface_orb = surface_orb * 0.1
    return eigenbands, surface_orb


class TestExpandedParams(chex.TestCase):
    """Tests for :func:`arpyes.simul.expanded.make_expanded_simulation_params`.

    Verifies that the auto-derived energy window is correctly computed from
    the eigenband extrema with default padding, and that scalar parameters
    are forwarded accurately.
    """

    def test_energy_window_matches_expanded_default(self):
        """Verify that energy_min and energy_max are derived from eigenbands.

        Test Logic
        ----------
        1. **Setup**:
           Create a small eigenband array with known extrema:
           min = -2.0, max = 1.0. Use the default energy padding of 1.0.

        2. **Build params**:
           Call ``make_expanded_simulation_params`` with fidelity=100.

        3. **Check energy bounds**:
           Assert that ``energy_min`` equals ``min(eigenbands) - 1.0 = -3.0``
           and ``energy_max`` equals ``max(eigenbands) + 1.0 = 2.0``,
           each within a tolerance of 1e-12.

        4. **Check fidelity**:
           Assert that the fidelity parameter is forwarded correctly as 100.

        Asserts
        -------
        The auto-derived energy window matches ``[min - padding, max + padding]``
        and the fidelity value is preserved exactly.
        """
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
    """Tests for :func:`arpyes.simul.expanded.simulate_basic_expanded`.

    Verifies that the expanded basic wrapper produces results identical to
    manually constructing PyTree inputs and calling the core
    :func:`~arpyes.simul.spectrum.simulate_basic` function directly.
    """

    def test_matches_core_basic_simulation(self):
        """Verify that the expanded wrapper matches the core basic simulation.

        Test Logic
        ----------
        1. **Build reference manually**:
           Construct ``BandStructure``, ``OrbitalProjection``, and
           ``SimulationParams`` PyTrees by hand from raw arrays, using
           zero-filled k-points and the same scalar parameters
           (sigma=0.06, fidelity=240, temperature=20, photon_energy=35).

        2. **Run core simulation**:
           Call ``simulate_basic`` directly with the PyTree inputs to
           obtain the expected spectrum.

        3. **Run expanded wrapper**:
           Call ``simulate_basic_expanded`` with the same raw arrays and
           scalar parameters.

        4. **Compare**:
           Assert that both intensity arrays and energy axes match to
           within 1e-12 absolute tolerance.

        Asserts
        -------
        The intensity and energy axis from the expanded wrapper are
        numerically identical to those from the core function, confirming
        that the wrapper correctly constructs all intermediate PyTrees.
        """
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
    """Tests for :func:`arpyes.simul.expanded.simulate_advanced_expanded`.

    Verifies that the expanded advanced wrapper correctly converts incident
    angles from degrees to radians and produces results identical to manually
    calling the core :func:`~arpyes.simul.spectrum.simulate_advanced` with
    pre-converted radian angles.
    """

    def test_degree_conversion_matches_core_advanced(self):
        """Verify that degree-input angles produce the same result as radian-input.

        Test Logic
        ----------
        1. **Build reference manually**:
           Construct PyTree inputs by hand, explicitly converting
           incident_theta=45 and incident_phi=30 from degrees to radians
           via ``jnp.deg2rad``, and set polarization_angle=0.25 (already
           in radians). Use LHP polarization, sigma=0.05, fidelity=220,
           temperature=25, and photon_energy=21.2.

        2. **Run core simulation**:
           Call ``simulate_advanced`` directly with the manually built
           PyTrees and polarization config to obtain the expected spectrum.

        3. **Run expanded wrapper**:
           Call ``simulate_advanced_expanded`` with the same angles in
           degrees (45.0 and 30.0), relying on the wrapper to convert
           them to radians internally.

        4. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        Asserts
        -------
        The intensity from the expanded wrapper (degree input) is
        numerically identical to the core function (radian input),
        confirming that the degree-to-radian conversion is applied
        correctly to both theta and phi.
        """
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
    """Tests for :func:`arpyes.simul.expanded.simulate_expanded`.

    Verifies that the level-based dispatch function correctly routes to
    the appropriate expanded wrapper and produces identical results.
    """

    def test_dispatch_expert_matches_direct_wrapper(self):
        """Verify that dispatching with level="Expert" matches the direct wrapper.

        Test Logic
        ----------
        1. **Run direct wrapper**:
           Call ``simulate_expert_expanded`` with explicit parameters
           (sigma=0.04, gamma=0.1, fidelity=200, temperature=15,
           photon_energy=11, unpolarized, theta=45, phi=0) to produce
           the expected spectrum.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="Expert"`` and the
           same parameters. Note the capitalized level string, which
           tests case-insensitive dispatch.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        Asserts
        -------
        The dispatched intensity is numerically identical to the direct
        expert wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_expert_expanded`` and that level matching
        is case-insensitive.
        """
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

    def test_dispatch_soc_requires_surface_spin(self):
        """Verify that level='soc' without surface_spin raises ValueError.

        Test Logic
        ----------
        1. **Setup**: Synthetic eigenbands and surface_orb (no
           surface_spin passed).
        2. **Call**: Invoke ``simulate_expanded(level="soc", ...)``
           with only the required array args and ef.
        3. **Check**: Assert that a ``ValueError`` is raised and
           that the message contains ``"surface_spin"``.

        Asserts
        -------
        Dispatching to the SOC level without providing spin data
        raises ``ValueError`` with a clear requirement for
        surface_spin.
        """
        eigenbands, surface_orb = _make_synthetic_data()
        with self.assertRaises(ValueError) as ctx:
            simulate_expanded(
                level="soc",
                eigenbands=eigenbands,
                surface_orb=surface_orb,
                ef=0.0,
            )
        self.assertIn("surface_spin", str(ctx.exception))

    def test_dispatch_soc_matches_direct_wrapper(self):
        """Verify that level='soc' with surface_spin matches simulate_soc_expanded.

        Test Logic
        ----------
        1. **Setup**: Synthetic eigenbands and surface_orb; build a
           surface_spin array of shape (K, B, A, 6) with non-zero
           z components.
        2. **Run direct**: Call ``simulate_soc_expanded`` with the
           same parameters (sigma=0.04, gamma=0.1, fidelity=200,
           ls_scale=0.01, etc.) to get the expected spectrum.
        3. **Run dispatcher**: Call ``simulate_expanded(level="soc",
           ..., surface_spin=surface_spin, ls_scale=0.01)``.
        4. **Compare**: Assert dispatched and expected intensities
           match to within 1e-12.

        Asserts
        -------
        The level-based dispatcher with ``level="soc"`` and
        surface_spin produces the same result as calling
        ``simulate_soc_expanded`` directly.
        """
        eigenbands, surface_orb = _make_synthetic_data()
        nk, nb, na = surface_orb.shape[0], surface_orb.shape[1], surface_orb.shape[2]
        surface_spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        surface_spin = surface_spin.at[..., 4].set(0.15)
        surface_spin = surface_spin.at[..., 5].set(0.08)
        expected = simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
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
            ls_scale=0.01,
        )
        dispatched = simulate_expanded(
            level="soc",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
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
            ls_scale=0.01,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )


class TestExpandedSocWrapper(chex.TestCase):
    """Tests for :func:`arpyes.simul.expanded.simulate_soc_expanded`.

    Verifies that the expanded-input SOC wrapper constructs the same
    PyTrees (bands, orb_proj with spin, params, pol) as would be
    built by hand and produces results identical to the core
    :func:`~arpyes.simul.simulate_soc` function.
    """

    def test_matches_core_soc_simulation(self):
        """Verify that the expanded SOC wrapper matches the core simulate_soc.

        Test Logic
        ----------
        1. **Build reference manually**: Use :func:`_build_inputs` with
           surface_spin to get bands and orb_proj; build params via
           :func:`make_expanded_simulation_params` (fidelity=180,
           sigma=0.05, gamma=0.08, etc.); build pol via
           :func:`_build_polarization` (unpolarized, 45°, 0°). Call
           ``simulate_soc(bands, orb_proj, params, pol, ls_scale=0.02)``
           to get the expected spectrum.
        2. **Run expanded wrapper**: Call ``simulate_soc_expanded`` with
           the same raw arrays and scalar parameters (including
           ls_scale=0.02).
        3. **Compare**: Assert that wrapped and expected intensity
           and energy_axis match to within 1e-12.

        Asserts
        -------
        The expanded SOC wrapper produces numerically identical
        output to the core SOC simulation when given the same
        physical parameters.
        """
        eigenbands, surface_orb = _make_synthetic_data()
        nk, nb, na = surface_orb.shape[0], surface_orb.shape[1], surface_orb.shape[2]
        surface_spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        surface_spin = surface_spin.at[..., 0].set(0.1)
        surface_spin = surface_spin.at[..., 4].set(0.2)
        from arpyes.simul.expanded import _build_inputs, _build_polarization

        bands, orb_proj = _build_inputs(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            surface_spin=surface_spin,
        )
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=180,
            sigma=0.05,
            gamma=0.08,
            temperature=20.0,
            photon_energy=12.0,
        )
        pol = _build_polarization(
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
        )
        expected = simulate_soc(
            bands, orb_proj, params, pol, ls_scale=0.02
        )
        wrapped = simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
            ef=0.0,
            sigma=0.05,
            gamma=0.08,
            fidelity=180,
            temperature=20.0,
            photon_energy=12.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
            ls_scale=0.02,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            wrapped.energy_axis, expected.energy_axis, atol=1e-12
        )
