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
    """Tests for :func:`arpyes.types.geometry.make_crystal_geometry`.

    Verifies correct construction of ``CrystalGeometry`` PyTrees including
    output shape validation, automatic reciprocal lattice computation for
    orthogonal lattices, and JAX PyTree round-trip (flatten/unflatten)
    fidelity.
    """

    def test_basic_creation(self):
        """Verify that a CrystalGeometry is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct geometry**:
           Build a simple cubic lattice (3 Angstrom) with two atoms
           using ``make_crystal_geometry``.

        2. **Assert shapes**:
           Check that ``lattice`` is (3, 3), ``reciprocal_lattice`` is
           (3, 3), and ``coords`` is (2, 3).

        3. **Assert static field**:
           Confirm that ``symbols`` is preserved as ``("Si",)``.

        Asserts
        -------
        Output array shapes match expected dimensions and the ``symbols``
        tuple is stored unchanged as auxiliary data.
        """
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
        """Verify that the reciprocal lattice is correct for a cubic cell.

        Test Logic
        ----------
        1. **Construct orthogonal lattice**:
           Create a simple cubic lattice with parameter ``a = 5.0``
           Angstroms (diagonal identity matrix scaled by ``a``).

        2. **Compute expected reciprocal lattice**:
           For a cubic cell, ``b_i = 2 pi / a`` along each axis, so
           the expected reciprocal lattice is ``eye(3) * 2 pi / a``.

        3. **Compare numerically**:
           Assert element-wise closeness between the factory-computed
           ``reciprocal_lattice`` and the analytical expectation.

        Asserts
        -------
        ``geom.reciprocal_lattice`` matches the analytical ``2 pi / a``
        diagonal matrix to within ``atol=1e-10``.
        """
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
        """Verify that CrystalGeometry survives a JAX PyTree round-trip.

        Test Logic
        ----------
        1. **Create geometry**:
           Build a minimal CrystalGeometry with one atom.

        2. **Flatten and unflatten**:
           Use ``jax.tree.flatten`` and ``jax.tree.unflatten`` to
           simulate the round-trip JAX performs during ``jit``/``grad``.

        3. **Compare restored fields**:
           Assert that the numeric ``lattice`` array and the auxiliary
           ``symbols`` tuple are identical after reconstruction.

        Asserts
        -------
        ``restored.lattice`` is close to the original and
        ``restored.symbols`` equals the original, confirming both
        children and auxiliary data survive the round-trip.
        """
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
    """Tests for :func:`arpyes.types.bands.make_band_structure`.

    Verifies correct construction of ``BandStructure`` PyTrees including
    output shape validation under both JIT and eager modes, default
    uniform k-point weight generation, and automatic type conversion of
    the ``fermi_energy`` scalar to a JAX array.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        """Verify that a BandStructure is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct band structure**:
           Create zero-filled eigenvalues (10 k-points, 5 bands) and
           k-points arrays, then call the factory via ``self.variant``
           to test both JIT-compiled and eager execution paths.

        2. **Assert shapes**:
           Check that ``eigenvalues`` is (10, 5), ``kpoints`` is (10, 3),
           ``kpoint_weights`` is (10,), and ``fermi_energy`` is scalar.

        Asserts
        -------
        All output fields have the expected shapes, confirming that the
        factory correctly allocates default weights and casts the Fermi
        energy to a 0-D array.
        """
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
        """Verify that omitting kpoint_weights produces uniform weights.

        Test Logic
        ----------
        1. **Construct without explicit weights**:
           Call ``make_band_structure`` with 8 k-points and no
           ``kpoint_weights`` argument.

        2. **Compare to expected uniform vector**:
           Build an expected weight vector of all ones (float64) with
           length matching the number of k-points.

        Asserts
        -------
        ``bands.kpoint_weights`` is element-wise close to a uniform
        ``ones(8)`` vector, confirming the default-weight logic.
        """
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
        """Verify that the fermi_energy Python float is cast to a JAX array.

        Test Logic
        ----------
        1. **Pass a Python float**:
           Supply ``fermi_energy=-1.5`` as a plain Python float to the
           factory function.

        2. **Check result type**:
           Inspect whether ``bands.fermi_energy`` is an instance of
           ``jax.Array``, confirming the factory performed the cast.

        Asserts
        -------
        ``bands.fermi_energy`` is an instance of ``jax.Array``, not a
        raw Python float, verifying the float64 conversion logic.
        """
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
    """Tests for :func:`arpyes.types.bands.make_orbital_projection`.

    Verifies correct construction of ``OrbitalProjection`` PyTrees including
    output shape validation, default ``None`` handling for optional spin and
    OAM fields, and proper shape storage when spin data is provided.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        """Verify that an OrbitalProjection is created with correct defaults.

        Test Logic
        ----------
        1. **Construct with projections only**:
           Create a zero-filled projections array with shape
           (10 k-points, 5 bands, 2 atoms, 9 orbitals) and call the
           factory without providing optional ``spin`` or ``oam``.

        2. **Assert projection shape**:
           Check that ``orb.projections`` has shape (10, 5, 2, 9).

        3. **Assert optional fields are None**:
           Confirm that both ``orb.spin`` and ``orb.oam`` are ``None``
           when not supplied.

        Asserts
        -------
        Projections shape is correct and optional fields default to
        ``None``, confirming the factory's sentinel-preserving logic.
        """
        nk, nb, na = 10, 5, 2
        proj = jnp.zeros((nk, nb, na, 9))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj)
        chex.assert_shape(orb.projections, (nk, nb, na, 9))
        chex.assert_equal(orb.spin, None)
        chex.assert_equal(orb.oam, None)

    @chex.variants(with_jit=True, without_jit=True)
    def test_with_spin(self):
        """Verify that spin projections are stored with the correct shape.

        Test Logic
        ----------
        1. **Construct with spin data**:
           Create projection and spin arrays with compatible leading
           dimensions (4 k-points, 3 bands, 1 atom) and pass both to
           the factory.

        2. **Assert spin shape**:
           Check that ``orb.spin`` has shape (4, 3, 1, 6), matching the
           6 spin-projection channels (up/down for x, y, z).

        Asserts
        -------
        ``orb.spin`` has the expected 4-D shape, confirming that the
        factory correctly casts and stores the optional spin array.
        """
        nk, nb, na = 4, 3, 1
        proj = jnp.ones((nk, nb, na, 9))
        spin = jnp.zeros((nk, nb, na, 6))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj, spin=spin)
        chex.assert_shape(orb.spin, (nk, nb, na, 6))


class TestMakeSimulationParams(chex.TestCase):
    """Tests for :func:`arpyes.types.params.make_simulation_params`.

    Verifies correct construction of ``SimulationParams`` PyTrees including
    default parameter values, custom value passthrough, and JAX PyTree
    round-trip (flatten/unflatten) fidelity.
    """

    def test_defaults(self):
        """Verify that default parameter values match expected constants.

        Test Logic
        ----------
        1. **Construct with no arguments**:
           Call ``make_simulation_params()`` with all defaults.

        2. **Assert each default value**:
           Check ``energy_min`` (-3.0), ``energy_max`` (1.0),
           ``fidelity`` (25000), ``sigma`` (0.04), and ``gamma`` (0.1)
           against their documented default values.

        Asserts
        -------
        Each default parameter matches the expected constant, confirming
        the factory's default-value specification is correct.
        """
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
        """Verify that custom parameter values are stored correctly.

        Test Logic
        ----------
        1. **Construct with custom arguments**:
           Call the factory with non-default values for all parameters,
           including ``temperature=300.0`` and ``photon_energy=21.2``.

        2. **Spot-check one custom field**:
           Assert that ``params.temperature`` equals the supplied
           ``300.0`` as a float64 JAX scalar.

        Asserts
        -------
        ``params.temperature`` matches the custom input value,
        confirming that user-supplied arguments override defaults and
        are correctly cast to float64.
        """
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
        """Verify that SimulationParams survives a JAX PyTree round-trip.

        Test Logic
        ----------
        1. **Create params**:
           Build a default ``SimulationParams`` instance.

        2. **Flatten and unflatten**:
           Use ``jax.tree.flatten`` and ``jax.tree.unflatten`` to
           simulate the round-trip JAX performs during ``jit``/``grad``.

        3. **Compare restored field**:
           Assert that ``restored.sigma`` is close to the original
           ``params.sigma``.

        Asserts
        -------
        ``restored.sigma`` matches the original value, confirming that
        both JAX-traced children and the auxiliary ``fidelity`` int
        survive the flatten/unflatten round-trip.
        """
        params = make_simulation_params()
        leaves, treedef = jax.tree.flatten(params)
        restored = jax.tree.unflatten(treedef, leaves)
        chex.assert_trees_all_close(
            restored.sigma, params.sigma
        )


class TestMakePolarizationConfig(chex.TestCase):
    """Tests for :func:`arpyes.types.params.make_polarization_config`.

    Verifies correct construction of ``PolarizationConfig`` PyTrees including
    default polarization type and angular values, as well as explicit LVP
    (linear vertical polarization) configuration.
    """

    def test_defaults(self):
        """Verify that default polarization config is unpolarized with scalar angles.

        Test Logic
        ----------
        1. **Construct with no arguments**:
           Call ``make_polarization_config()`` with all defaults.

        2. **Assert polarization type**:
           Check that ``polarization_type`` defaults to ``"unpolarized"``.

        3. **Assert angle shapes**:
           Confirm that ``theta`` and ``phi`` are 0-D scalar arrays.

        Asserts
        -------
        Default polarization type is ``"unpolarized"`` and angular fields
        are scalar JAX arrays, confirming the factory's default behavior.
        """
        config = make_polarization_config()
        chex.assert_equal(
            config.polarization_type, "unpolarized"
        )
        chex.assert_shape(config.theta, ())
        chex.assert_shape(config.phi, ())

    def test_lvp(self):
        """Verify that an LVP polarization config stores the correct type string.

        Test Logic
        ----------
        1. **Construct with LVP settings**:
           Call the factory with ``theta=0.7854``, ``phi=0.0``, and
           ``polarization_type="LVP"`` (linear vertical polarization,
           i.e., s-polarization).

        2. **Assert polarization type**:
           Check that ``config.polarization_type`` is ``"LVP"``.

        Asserts
        -------
        The auxiliary ``polarization_type`` string is stored as ``"LVP"``,
        confirming that user-supplied string arguments are passed through
        unchanged.
        """
        config = make_polarization_config(
            theta=0.7854,
            phi=0.0,
            polarization_type="LVP",
        )
        chex.assert_equal(config.polarization_type, "LVP")


class TestMakeArpesSpectrum(chex.TestCase):
    """Tests for :func:`arpyes.types.bands.make_arpes_spectrum`.

    Verifies correct construction of ``ArpesSpectrum`` PyTrees including
    output shape validation for the 2-D intensity map and 1-D energy axis
    under both JIT and eager execution modes.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        """Verify that an ArpesSpectrum is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct spectrum**:
           Create a zero-filled intensity map (10 k-points, 100 energy
           points) and a linearly spaced energy axis, then call the
           factory via ``self.variant`` to test both JIT and eager modes.

        2. **Assert shapes**:
           Check that ``intensity`` is (10, 100) and ``energy_axis`` is
           (100,).

        Asserts
        -------
        Both output fields have the expected shapes, confirming that the
        factory correctly casts inputs to float64 arrays and preserves
        dimensionality.
        """
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
    """Tests for :func:`arpyes.types.dos.make_density_of_states`.

    Verifies correct construction of ``DensityOfStates`` PyTrees including
    output shape validation for the energy axis and total DOS arrays, and
    correct casting of the Fermi energy scalar to a float64 JAX array.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self):
        """Verify that a DensityOfStates is created with correct fields.

        Test Logic
        ----------
        1. **Construct DOS**:
           Create a linearly spaced energy axis (500 points from -10 to
           5 eV), a uniform total DOS, and a Fermi energy of -1.5 eV,
           then call the factory via ``self.variant``.

        2. **Assert shapes**:
           Check that ``energy`` and ``total_dos`` both have shape (500,).

        3. **Assert Fermi energy value**:
           Confirm that ``fermi_energy`` is close to the supplied -1.5
           as a float64 scalar.

        Asserts
        -------
        Array shapes match the input dimensions and the scalar Fermi
        energy is correctly cast and stored.
        """
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
    """Tests for :func:`arpyes.types.kpath.make_kpath_info`.

    Verifies correct construction of ``KPathInfo`` PyTrees including
    output shape validation for label indices, and correct storage of
    the ``mode`` string and ``labels`` tuple as auxiliary data.
    """

    def test_basic_creation(self):
        """Verify that a KPathInfo is created with correct fields and shapes.

        Test Logic
        ----------
        1. **Construct k-path info**:
           Call ``make_kpath_info`` with 100 k-points, three label
           indices (start, midpoint, end), ``"Line-mode"`` mode, and
           symmetry labels ``("G", "M", "K")``.

        2. **Assert label_indices shape**:
           Check that ``label_indices`` has shape (3,), matching the
           three supplied indices.

        3. **Assert auxiliary fields**:
           Confirm that ``mode`` is ``"Line-mode"`` and ``labels`` is
           ``("G", "M", "K")``.

        Asserts
        -------
        ``label_indices`` has the expected 1-D shape and string fields
        are stored unchanged as auxiliary data.
        """
        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            mode="Line-mode",
            labels=("G", "M", "K"),
        )
        chex.assert_shape(kpath.label_indices, (3,))
        chex.assert_equal(kpath.mode, "Line-mode")
        chex.assert_equal(kpath.labels, ("G", "M", "K"))
