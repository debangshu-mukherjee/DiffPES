"""Tests for new PyTree types: OrbitalBasis, SlaterParams, DiagonalizedBands, TBModel, SelfEnergyConfig.

Extended Summary
----------------
Exercises the Chinook-pipeline PyTree factories: make_orbital_basis,
make_slater_params, make_diagonalized_bands, make_tb_model, and
make_self_energy_config. Tests cover correct construction and field
storage, default value generation (e.g. auto-labels), input
validation (length mismatches, required fields, invalid modes),
JAX PyTree round-trip (flatten/unflatten) fidelity, gradient flow
through differentiable fields, and dtype enforcement (float64
casting). All test logic and assertions are documented in the
docstrings of each test class and method.

Routine Listings
----------------
:class:`TestOrbitalBasis`
    Tests for make_orbital_basis.
:class:`TestSlaterParams`
    Tests for make_slater_params.
:class:`TestDiagonalizedBands`
    Tests for make_diagonalized_bands.
:class:`TestTBModel`
    Tests for make_tb_model.
:class:`TestSelfEnergyConfig`
    Tests for make_self_energy_config.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.types import (
    DiagonalizedBands,
    OrbitalBasis,
    SelfEnergyConfig,
    SlaterParams,
    TBModel,
    make_diagonalized_bands,
    make_orbital_basis,
    make_self_energy_config,
    make_slater_params,
    make_spin_orbital_projection,
    make_tb_model,
)


class TestOrbitalBasis:
    """Tests for :func:`diffpes.types.make_orbital_basis`.

    Validates construction of ``OrbitalBasis`` PyTrees that describe
    the quantum-number basis (n, l, m) for Chinook tight-binding
    calculations. Covers explicit creation with provided quantum
    numbers, automatic label generation, JAX PyTree round-trip
    (flatten/unflatten) fidelity, and input validation when quantum
    number tuple lengths are mismatched.
    """

    def test_creation(self):
        """Verify OrbitalBasis stores quantum number tuples unchanged.

        Constructs an OrbitalBasis with two orbitals: (n=1, l=0, m=0)
        and (n=2, l=1, m=0). Asserts that ``n_values`` is stored as
        ``(1, 2)`` and ``l_values`` as ``(0, 1)``, confirming the
        factory passes through the tuple auxiliary data without
        modification.
        """
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        assert basis.n_values == (1, 2)
        assert basis.l_values == (0, 1)

    def test_default_labels(self):
        """Verify auto-generated labels follow the ``orb_N`` naming convention.

        Constructs a single-orbital OrbitalBasis without supplying
        explicit labels. Asserts that ``labels`` defaults to
        ``("orb_0",)``, confirming the factory's automatic label
        generation produces zero-indexed names matching the number of
        orbitals.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        assert basis.labels == ("orb_0",)

    def test_pytree_flatten_unflatten(self):
        """Verify OrbitalBasis survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a 3-orbital basis with explicit labels ``("s", "p", "d")``.
        Flattens via ``jax.tree_util.tree_flatten`` and reconstructs via
        ``jax.tree_util.tree_unflatten``. Asserts that ``n_values`` and
        ``labels`` on the restored object match the originals, confirming
        that the auxiliary data (tuples of ints and strings) is correctly
        encoded in the tree definition and recovered on reconstruction.
        """
        basis = make_orbital_basis(
            n_values=(1, 2, 3),
            l_values=(0, 1, 2),
            m_values=(0, 0, 0),
            labels=("s", "p", "d"),
        )
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        basis2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert basis2.n_values == basis.n_values
        assert basis2.labels == basis.labels

    def test_length_mismatch_raises(self):
        """Verify ValueError is raised when quantum number tuple lengths disagree.

        Calls ``make_orbital_basis`` with ``n_values`` of length 2,
        ``l_values`` of length 1, and ``m_values`` of length 2.
        Asserts that ``ValueError`` is raised with a message matching
        ``"same length"``, confirming the factory validates that all
        three quantum number tuples have consistent lengths before
        construction.
        """
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis(
                n_values=(1, 2),
                l_values=(0,),
                m_values=(0, 0),
            )


class TestSlaterParams:
    """Tests for :func:`diffpes.types.make_slater_params`.

    Validates construction of ``SlaterParams`` PyTrees that hold
    Slater-type orbital exponents and expansion coefficients. Covers
    basic creation with shape verification, gradient flow through
    the ``zeta`` field (essential for inverse fitting), and automatic
    float64 dtype casting of input arrays.
    """

    def test_creation(self):
        """Verify SlaterParams stores zeta and default coefficients with correct shapes.

        Constructs a SlaterParams with a single orbital (zeta = [1.5])
        and a 1-orbital basis. Asserts ``zeta`` shape is (1,) and
        ``coefficients`` shape is (1, 1), confirming that the factory
        creates a default single-term expansion coefficient matrix when
        none is supplied.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.5]),
            orbital_basis=basis,
        )
        assert sp.zeta.shape == (1,)
        assert sp.coefficients.shape == (1, 1)

    def test_pytree_gradient(self):
        """Verify JAX gradients flow through the zeta field of SlaterParams.

        Defines a loss function ``loss(sp) = sum(zeta^2)`` and computes
        ``jax.grad(loss)`` with respect to a SlaterParams having
        ``zeta = [2.0]``. Asserts the gradient of zeta is 4.0
        (``d/d(zeta) zeta^2 = 2*zeta = 4.0``), confirming that
        SlaterParams is a valid differentiable JAX PyTree and that
        ``jax.grad`` can trace through its leaf arrays. This is
        critical for Chinook inverse fitting workflows.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )

        def loss(sp):
            return jnp.sum(sp.zeta**2)

        sp = make_slater_params(
            zeta=jnp.array([2.0]),
            orbital_basis=basis,
        )
        grad = jax.grad(loss)(sp)
        assert jnp.allclose(grad.zeta, 4.0)

    def test_float64_casting(self):
        """Verify that float32 input arrays are automatically promoted to float64.

        Constructs a SlaterParams with ``zeta`` supplied as a float32
        array. Asserts that the stored ``zeta`` has dtype ``jnp.float64``,
        confirming the factory enforces 64-bit precision regardless of
        the input dtype. This is important for numerical accuracy in
        Slater integrals and gradient-based optimization.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0], dtype=jnp.float32),
            orbital_basis=basis,
        )
        assert sp.zeta.dtype == jnp.float64


class TestDiagonalizedBands:
    """Tests for :func:`diffpes.types.make_diagonalized_bands`.

    Validates construction of ``DiagonalizedBands`` PyTrees that store
    eigenvalues and eigenvectors from tight-binding Hamiltonian
    diagonalization. Covers basic creation with shape verification and
    JAX PyTree flatten/unflatten round-trip fidelity for both real
    eigenvalue and complex eigenvector arrays.
    """

    def test_creation(self):
        """Verify DiagonalizedBands stores eigenvalues and eigenvectors with correct shapes.

        Constructs a DiagonalizedBands with 5 k-points, 3 bands, and
        4 orbitals. Asserts ``eigenvalues`` shape is (5, 3) and
        ``eigenvectors`` shape is (5, 3, 4), confirming the factory
        correctly stores the real eigenvalue matrix and the complex
        eigenvector tensor with the expected dimensions
        (K, B) and (K, B, O) respectively.
        """
        K, B, O = 5, 3, 4
        diag = make_diagonalized_bands(
            eigenvalues=jnp.zeros((K, B)),
            eigenvectors=jnp.ones((K, B, O), dtype=jnp.complex128),
            kpoints=jnp.zeros((K, 3)),
            fermi_energy=0.0,
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, O)

    def test_pytree_round_trip(self):
        """Verify DiagonalizedBands survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a 2-k-point, 2-band, 2-orbital DiagonalizedBands
        with specific eigenvalues [[1, 2], [3, 4]] and identity
        eigenvectors. Flattens and reconstructs via JAX tree utilities.
        Asserts that the restored ``eigenvalues`` match the originals
        via ``jnp.allclose``, confirming both the real eigenvalue and
        complex eigenvector leaf arrays, plus the scalar Fermi energy
        auxiliary data, survive the round-trip.
        """
        K, B, O = 2, 2, 2
        diag = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            eigenvectors=jnp.eye(2, dtype=jnp.complex128)[None].repeat(
                2, axis=0
            ),
            kpoints=jnp.zeros((2, 3)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(diag)
        diag2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(diag2.eigenvalues, diag.eigenvalues)


class TestTBModel:
    """Tests for :func:`diffpes.types.make_tb_model`.

    Validates construction of ``TBModel`` PyTrees that hold
    tight-binding Hamiltonian parameters: hopping amplitudes, lattice
    vectors, hopping connectivity indices, and an orbital basis. Covers
    basic creation with shape verification and gradient flow through
    the differentiable ``hopping_params`` field (essential for
    inverse fitting of tight-binding models).
    """

    def test_creation(self):
        """Verify TBModel stores hopping parameters with the correct shape.

        Constructs a minimal TBModel with 2 hopping parameters
        (forward and backward nearest-neighbor along x), a cubic
        lattice, and a single-orbital basis. Asserts
        ``hopping_params`` shape is (2,), confirming the factory
        correctly stores the differentiable hopping array alongside
        the static auxiliary data (hopping_indices, n_orbitals,
        orbital_basis).
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0, 1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)), (0, 0, (-1, 0, 0))),
            n_orbitals=1,
            orbital_basis=basis,
        )
        assert model.hopping_params.shape == (2,)

    def test_pytree_gradient(self):
        """Verify JAX gradients flow through the hopping_params field of TBModel.

        Defines a loss function ``loss(m) = sum(hopping_params^2)`` and
        computes ``jax.grad(loss)`` with respect to a TBModel having a
        single hopping parameter of value 1.0. Asserts the gradient is
        2.0 (``d/d(t) t^2 = 2*t = 2.0``), confirming that TBModel is
        a valid differentiable JAX PyTree and that gradients correctly
        propagate through its leaf arrays while treating hopping_indices
        and other structure as static auxiliary data.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)),),
            n_orbitals=1,
            orbital_basis=basis,
        )

        def loss(m):
            return jnp.sum(m.hopping_params**2)

        grad = jax.grad(loss)(model)
        assert jnp.allclose(grad.hopping_params, 2.0)


class TestSelfEnergyConfig:
    """Tests for :func:`diffpes.types.make_self_energy_config`.

    Validates construction of ``SelfEnergyConfig`` PyTrees that
    parameterize the imaginary part of the electron self-energy for
    ARPES spectral function broadening. Covers the default constant
    mode, polynomial mode, validation errors for tabulated mode
    without energy nodes and for invalid mode strings, and JAX PyTree
    flatten/unflatten round-trip fidelity.
    """

    def test_constant_default(self):
        """Verify default SelfEnergyConfig uses constant mode with gamma=0.1.

        Calls ``make_self_energy_config()`` with no arguments. Asserts
        ``mode`` is ``"constant"``, ``coefficients`` shape is (1,),
        and the single coefficient value is approximately 0.1 (the
        default gamma broadening), verified via ``pytest.approx``.
        This confirms the factory's default initialization path.
        """
        config = make_self_energy_config()
        assert config.mode == "constant"
        assert config.coefficients.shape == (1,)
        assert float(config.coefficients[0]) == pytest.approx(0.1)

    def test_polynomial(self):
        """Verify polynomial mode accepts explicit coefficients and stores mode string.

        Constructs a SelfEnergyConfig with ``mode="polynomial"`` and
        two coefficients [0.01, 0.1] (representing a linear polynomial
        in energy). Asserts ``mode`` is ``"polynomial"``, confirming
        the factory accepts this mode and stores the mode string as
        auxiliary data without error.
        """
        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array([0.01, 0.1]),
        )
        assert config.mode == "polynomial"

    def test_tabulated_requires_nodes(self):
        """Verify tabulated mode raises ValueError when energy_nodes are not provided.

        Calls ``make_self_energy_config(mode="tabulated")`` without
        supplying ``energy_nodes``. Asserts ``ValueError`` is raised
        with a message matching ``"energy_nodes required"``, confirming
        the factory enforces that tabulated interpolation mode requires
        an explicit set of energy grid nodes.
        """
        with pytest.raises(ValueError, match="energy_nodes required"):
            make_self_energy_config(mode="tabulated")

    def test_invalid_mode_raises(self):
        """Verify an unrecognized mode string raises ValueError.

        Calls ``make_self_energy_config(mode="invalid")``. Asserts
        ``ValueError`` is raised with a message matching
        ``"mode must be"``, confirming the factory validates the mode
        string against the allowed set (``"constant"``,
        ``"polynomial"``, ``"tabulated"``) and rejects unknown values.
        """
        with pytest.raises(ValueError, match="mode must be"):
            make_self_energy_config(mode="invalid")

    def test_pytree_round_trip(self):
        """Verify SelfEnergyConfig survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a constant-mode SelfEnergyConfig with ``gamma=0.2``.
        Flattens via ``jax.tree_util.tree_flatten`` and reconstructs
        via ``jax.tree_util.tree_unflatten``. Asserts that the restored
        ``mode`` string matches the original and the ``coefficients``
        array matches via ``jnp.allclose``, confirming that both the
        auxiliary string data and the differentiable leaf arrays survive
        the round-trip.
        """
        config = make_self_energy_config(gamma=0.2)
        leaves, treedef = jax.tree_util.tree_flatten(config)
        config2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert config2.mode == config.mode
        assert jnp.allclose(config2.coefficients, config.coefficients)


class TestSpinOrbitalProjectionWithOAM:
    """Tests for make_spin_orbital_projection with explicit OAM data.

    Exercises the branch in ``make_spin_orbital_projection`` where
    ``oam`` is not None, covering the OAM array casting code path.
    """

    def test_with_oam_stores_array(self):
        """Verify that providing oam creates a float64 OAM array in the result.

        Constructs a SpinOrbitalProjection with explicit OAM and asserts
        the ``oam`` field is not None and has the correct shape.
        """
        K, B, A = 2, 3, 1
        proj = jnp.ones((K, B, A, 9), dtype=jnp.float64)
        spin = jnp.zeros((K, B, A, 6), dtype=jnp.float64)
        oam = jnp.ones((K, B, A, 3), dtype=jnp.float64) * 0.5
        sop = make_spin_orbital_projection(
            projections=proj, spin=spin, oam=oam
        )
        assert sop.oam is not None
        assert sop.oam.shape == (K, B, A, 3)
        assert sop.oam.dtype == jnp.float64


class TestVolumetricPyTree:
    """PyTree round-trip tests for VolumetricData and SOCVolumetricData.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` methods and
    the ``atom_counts=None`` default path in both factories.
    """

    def test_volumetric_data_pytree_round_trip(self):
        """Verify VolumetricData survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a minimal VolumetricData, flattens and unflattens it,
        then asserts the ``charge`` array is preserved.
        """
        from diffpes.types import make_volumetric_data

        lattice = jnp.eye(3, dtype=jnp.float64)
        coords = jnp.zeros((2, 3), dtype=jnp.float64)
        charge = jnp.ones((4, 4, 4), dtype=jnp.float64)
        vol = make_volumetric_data(
            lattice=lattice,
            coords=coords,
            charge=charge,
            grid_shape=(4, 4, 4),
            symbols=("Fe", "Co"),
        )
        leaves, treedef = jax.tree_util.tree_flatten(vol)
        vol2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(vol2.charge, vol.charge)
        assert vol2.grid_shape == vol.grid_shape
        assert vol2.symbols == vol.symbols

    def test_volumetric_data_atom_counts_none_default(self):
        """Verify that atom_counts=None creates a zero-length int32 array.

        When ``atom_counts`` is omitted, the factory should default to
        ``jnp.zeros(0, dtype=jnp.int32)`` so the field is always an array.
        """
        from diffpes.types import make_volumetric_data

        vol = make_volumetric_data(
            lattice=jnp.eye(3, dtype=jnp.float64),
            coords=jnp.zeros((1, 3), dtype=jnp.float64),
            charge=jnp.ones((2, 2, 2), dtype=jnp.float64),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        assert vol.atom_counts.dtype == jnp.int32
        assert vol.atom_counts.shape == (0,)

    def test_soc_volumetric_data_pytree_round_trip(self):
        """Verify SOCVolumetricData survives a JAX PyTree flatten/unflatten round-trip."""
        from diffpes.types import make_soc_volumetric_data

        lattice = jnp.eye(3, dtype=jnp.float64)
        coords = jnp.zeros((1, 3), dtype=jnp.float64)
        charge = jnp.ones((2, 2, 2), dtype=jnp.float64)
        mag = jnp.zeros((2, 2, 2), dtype=jnp.float64)
        mag_vec = jnp.zeros((2, 2, 2, 3), dtype=jnp.float64)
        vol = make_soc_volumetric_data(
            lattice=lattice,
            coords=coords,
            charge=charge,
            magnetization=mag,
            magnetization_vector=mag_vec,
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        leaves, treedef = jax.tree_util.tree_flatten(vol)
        vol2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(vol2.charge, vol.charge)
        assert vol2.grid_shape == vol.grid_shape

    def test_soc_volumetric_data_atom_counts_none_default(self):
        """Verify that atom_counts=None creates a zero-length int32 array in SOCVolumetricData."""
        from diffpes.types import make_soc_volumetric_data

        vol = make_soc_volumetric_data(
            lattice=jnp.eye(3, dtype=jnp.float64),
            coords=jnp.zeros((1, 3), dtype=jnp.float64),
            charge=jnp.ones((2, 2, 2), dtype=jnp.float64),
            magnetization=jnp.zeros((2, 2, 2), dtype=jnp.float64),
            magnetization_vector=jnp.zeros((2, 2, 2, 3), dtype=jnp.float64),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        assert vol.atom_counts.dtype == jnp.int32
        assert vol.atom_counts.shape == (0,)


class TestFullDensityOfStatesPyTree:
    """PyTree round-trip tests for FullDensityOfStates.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` of
    ``FullDensityOfStates``, both with and without optional fields.
    """

    def test_pytree_round_trip(self):
        """Verify FullDensityOfStates survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a minimal FullDensityOfStates (spin-up only),
        flattens and unflattens it, then asserts the ``fermi_energy``
        and ``total_dos_up`` arrays are preserved.
        """
        from diffpes.types import make_full_density_of_states

        energy = jnp.linspace(-3, 1, 50, dtype=jnp.float64)
        dos_up = jnp.ones(50, dtype=jnp.float64)
        idos_up = jnp.cumsum(dos_up)
        full_dos = make_full_density_of_states(
            energy=energy,
            total_dos_up=dos_up,
            integrated_dos_up=idos_up,
            fermi_energy=-0.5,
        )
        leaves, treedef = jax.tree_util.tree_flatten(full_dos)
        full_dos2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(full_dos2.total_dos_up, full_dos.total_dos_up)
        assert jnp.allclose(full_dos2.fermi_energy, full_dos.fermi_energy)
        assert full_dos2.natoms == full_dos.natoms


class TestBandsPyTree:
    """PyTree round-trip tests for SpinOrbitalProjection and SpinBandStructure.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` methods for
    the spin-aware band and projection types.
    """

    def test_spin_orbital_projection_pytree_round_trip(self):
        """Verify SpinOrbitalProjection survives a JAX PyTree flatten/unflatten round-trip."""
        from diffpes.types import make_spin_orbital_projection

        K, B, A = 3, 4, 2
        proj = jnp.ones((K, B, A, 9), dtype=jnp.float64)
        spin = jnp.zeros((K, B, A, 6), dtype=jnp.float64)
        sop = make_spin_orbital_projection(projections=proj, spin=spin)
        leaves, treedef = jax.tree_util.tree_flatten(sop)
        sop2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(sop2.projections, sop.projections)
        assert jnp.allclose(sop2.spin, sop.spin)
        assert sop2.oam is None

    def test_spin_band_structure_pytree_round_trip(self):
        """Verify SpinBandStructure survives a JAX PyTree flatten/unflatten round-trip."""
        from diffpes.types import make_spin_band_structure

        K, B = 5, 3
        evals_up = jnp.zeros((K, B), dtype=jnp.float64)
        evals_down = jnp.ones((K, B), dtype=jnp.float64)
        kpoints = jnp.zeros((K, 3), dtype=jnp.float64)
        sbs = make_spin_band_structure(
            eigenvalues_up=evals_up,
            eigenvalues_down=evals_down,
            kpoints=kpoints,
            fermi_energy=-1.0,
        )
        leaves, treedef = jax.tree_util.tree_flatten(sbs)
        sbs2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(sbs2.eigenvalues_up, sbs.eigenvalues_up)
        assert jnp.allclose(sbs2.eigenvalues_down, sbs.eigenvalues_down)
        assert jnp.allclose(sbs2.fermi_energy, sbs.fermi_energy)
