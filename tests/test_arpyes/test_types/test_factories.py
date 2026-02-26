"""Tests for PyTree factory functions."""

import chex
import jax
import jax.numpy as jnp

from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    CrystalGeometry,
    DensityOfStates,
    KPathInfo,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    make_arpes_spectrum,
    make_band_structure,
    make_crystal_geometry,
    make_density_of_states,
    make_kpath_info,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
)


class TestMakeCrystalGeometry(chex.TestCase):

    def test_basic_creation(self):
        lattice = jnp.eye(3) * 3.0
        coords = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[2],
        )
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.reciprocal_lattice, (3, 3))
        chex.assert_shape(geom.coords, (2, 3))
        chex.assert_equal(geom.symbols, ("Si",))

    def test_reciprocal_lattice_orthogonal(self):
        a = 5.0
        lattice = jnp.eye(3) * a
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=jnp.zeros((1, 3)),
            symbols=("X",),
            atom_counts=[1],
        )
        expected = jnp.eye(3) * 2.0 * jnp.pi / a
        chex.assert_trees_all_close(
            geom.reciprocal_lattice, expected, atol=1e-10
        )

    def test_pytree_flatten_unflatten(self):
        lattice = jnp.eye(3) * 3.0
        coords = jnp.array([[0.0, 0.0, 0.0]])
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[1],
        )
        leaves, treedef = jax.tree.flatten(geom)
        restored = jax.tree.unflatten(treedef, leaves)
        chex.assert_trees_all_close(
            restored.lattice, geom.lattice
        )
        chex.assert_equal(restored.symbols, geom.symbols)


class TestMakeBandStructure(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        nk, nb = 10, 5
        eigenvalues = jnp.zeros((nk, nb))
        kpoints = jnp.zeros((nk, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(
            eigenvalues=eigenvalues,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        chex.assert_shape(bands.eigenvalues, (nk, nb))
        chex.assert_shape(bands.kpoints, (nk, 3))
        chex.assert_shape(bands.kpoint_weights, (nk,))
        chex.assert_shape(bands.fermi_energy, ())

    @chex.variants(with_jit=True, without_jit=True)
    def test_default_weights(self):
        nk = 8
        eigenvalues = jnp.zeros((nk, 3))
        kpoints = jnp.zeros((nk, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(eigenvalues=eigenvalues, kpoints=kpoints)
        expected = jnp.ones(nk, dtype=jnp.float64)
        chex.assert_trees_all_close(
            bands.kpoint_weights, expected
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_type_conversion(self):
        eigenvalues = jnp.ones((4, 2))
        kpoints = jnp.zeros((4, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(
            eigenvalues=eigenvalues,
            kpoints=kpoints,
            fermi_energy=-1.5,
        )
        chex.assert_equal(
            isinstance(bands.fermi_energy, jax.Array), True
        )


class TestMakeOrbitalProjection(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        nk, nb, na = 10, 5, 2
        proj = jnp.zeros((nk, nb, na, 9))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj)
        chex.assert_shape(orb.projections, (nk, nb, na, 9))
        chex.assert_equal(orb.spin, None)
        chex.assert_equal(orb.oam, None)

    @chex.variants(with_jit=True, without_jit=True)
    def test_with_spin(self):
        nk, nb, na = 4, 3, 1
        proj = jnp.ones((nk, nb, na, 9))
        spin = jnp.zeros((nk, nb, na, 6))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj, spin=spin)
        chex.assert_shape(orb.spin, (nk, nb, na, 6))


class TestMakeSimulationParams(chex.TestCase):

    def test_defaults(self):
        params = make_simulation_params()
        chex.assert_trees_all_close(
            params.energy_min, jnp.float64(-3.0)
        )
        chex.assert_trees_all_close(
            params.energy_max, jnp.float64(1.0)
        )
        chex.assert_equal(params.fidelity, 25000)
        chex.assert_trees_all_close(
            params.sigma, jnp.float64(0.04)
        )
        chex.assert_trees_all_close(
            params.gamma, jnp.float64(0.1)
        )

    def test_custom_values(self):
        params = make_simulation_params(
            energy_min=-5.0,
            energy_max=2.0,
            fidelity=1000,
            sigma=0.08,
            gamma=0.2,
            temperature=300.0,
            photon_energy=21.2,
        )
        chex.assert_trees_all_close(
            params.temperature, jnp.float64(300.0)
        )

    def test_pytree_compatible(self):
        params = make_simulation_params()
        leaves, treedef = jax.tree.flatten(params)
        restored = jax.tree.unflatten(treedef, leaves)
        chex.assert_trees_all_close(
            restored.sigma, params.sigma
        )


class TestMakePolarizationConfig(chex.TestCase):

    def test_defaults(self):
        config = make_polarization_config()
        chex.assert_equal(
            config.polarization_type, "unpolarized"
        )
        chex.assert_shape(config.theta, ())
        chex.assert_shape(config.phi, ())

    def test_lvp(self):
        config = make_polarization_config(
            theta=0.7854,
            phi=0.0,
            polarization_type="LVP",
        )
        chex.assert_equal(config.polarization_type, "LVP")


class TestMakeArpesSpectrum(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        nk, ne = 10, 100
        intensity = jnp.zeros((nk, ne))
        energy_axis = jnp.linspace(-3.0, 1.0, ne)
        var_fn = self.variant(make_arpes_spectrum)
        spec = var_fn(
            intensity=intensity,
            energy_axis=energy_axis,
        )
        chex.assert_shape(spec.intensity, (nk, ne))
        chex.assert_shape(spec.energy_axis, (ne,))


class TestMakeDensityOfStates(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        ne = 500
        energy = jnp.linspace(-10.0, 5.0, ne)
        dos = jnp.ones(ne)
        var_fn = self.variant(make_density_of_states)
        result = var_fn(
            energy=energy, total_dos=dos, fermi_energy=-1.5
        )
        chex.assert_shape(result.energy, (ne,))
        chex.assert_shape(result.total_dos, (ne,))
        chex.assert_trees_all_close(
            result.fermi_energy, jnp.float64(-1.5)
        )


class TestMakeKPathInfo(chex.TestCase):

    def test_basic_creation(self):
        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            mode="Line-mode",
            labels=("G", "M", "K"),
        )
        chex.assert_shape(kpath.label_indices, (3,))
        chex.assert_equal(kpath.mode, "Line-mode")
        chex.assert_equal(kpath.labels, ("G", "M", "K"))
