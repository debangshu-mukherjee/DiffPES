"""Tests for tight-binding Hamiltonian builder.

Extended Summary
----------------
Validates the k-space Hamiltonian construction function
``build_hamiltonian_k`` and the two built-in model factories
``make_1d_chain_model`` and ``make_graphene_model``.  Tests verify
Hermiticity of H(k), correct matrix shape, JIT compatibility, the
analytically known cosine dispersion of the 1D chain, the eigenvalue
bandwidth [-2|t|, 2|t|], the graphene Gamma-point eigenvalues (+/-3|t|),
the Dirac-point degeneracy at K=(2/3, 1/3, 0), and differentiability of
eigenvalues with respect to hopping parameters.

Routine Listings
----------------
:class:`TestBuildHamiltonianK`
    Tests for build_hamiltonian_k.
:class:`TestMake1DChainModel`
    Tests for the 1D chain model factory.
:class:`TestMakeGrapheneModel`
    Tests for the graphene model factory.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.tightb.hamiltonian import (
    build_hamiltonian_k,
    make_1d_chain_model,
    make_graphene_model,
)


class TestBuildHamiltonianK:
    """Tests for ``build_hamiltonian_k``.

    Validates the Bloch Hamiltonian H(k) = sum_R t_R * exp(i k.R)
    construction from hopping parameters and lattice vectors.  Tests
    verify Hermiticity (H = H^dag), correct matrix shape, and JIT
    compatibility.
    """

    def test_hermitian(self):
        """Verify H(k) is Hermitian at an arbitrary k-point.

        Builds the graphene Hamiltonian at k=(0.3, 0.2, 0.0) and asserts
        H == H^dag to within 1e-12.  Hermiticity is a fundamental
        requirement for a physical Hamiltonian and ensures real
        eigenvalues upon diagonalization.
        """
        model = make_graphene_model()
        k = jnp.array([0.3, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert jnp.allclose(H, H.conj().T, atol=1e-12)

    def test_correct_shape(self):
        """Verify H(k) has shape (n_orbitals, n_orbitals).

        Builds the graphene Hamiltonian (2 orbitals: pz on sublattices A
        and B) at an arbitrary k-point and asserts the output shape is
        (2, 2).  This confirms the Hamiltonian matrix dimension matches
        the model's orbital count.
        """
        model = make_graphene_model()
        k = jnp.array([0.1, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert H.shape == (2, 2)

    def test_jit_compatible(self):
        """Verify ``build_hamiltonian_k`` is JAX-JIT-compatible.

        Wraps the Hamiltonian construction for the 1D chain model in
        ``jax.jit`` with k as the traced argument.  Evaluates at
        k=(0.25, 0, 0) and asserts all matrix elements are finite,
        confirming no Python-side control flow or non-JAX operations
        break tracing.
        """
        model = make_1d_chain_model()
        f = jax.jit(
            lambda k: build_hamiltonian_k(
                k,
                model.hopping_params,
                model.hopping_indices,
                model.n_orbitals,
                model.lattice_vectors,
            )
        )
        H = f(jnp.array([0.25, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(H))


class TestMake1DChainModel:
    """Tests for the 1D chain tight-binding model factory.

    Validates ``make_1d_chain_model`` by checking the analytically known
    dispersion E(k) = 2t*cos(2*pi*k) and the eigenvalue bandwidth
    [-2|t|, 2|t|].  The 1D chain is the simplest tight-binding system
    and serves as a regression test for the Hamiltonian builder.
    """

    def test_cosine_dispersion(self):
        """Verify E(k) = 2t*cos(2*pi*k) for the 1D chain.

        Constructs the 1D chain with t=-1.0 and diagonalizes at 101
        k-points spanning the full Brillouin zone [-0.5, 0.5].  The
        single-band dispersion is E(k) = -2*cos(2*pi*k) (since t=-1).
        Asserts the numerical eigenvalues match this analytical cosine
        to within 1e-10 across the entire BZ, validating both the
        Hamiltonian construction (Fourier sum of hoppings) and the
        diagonalization.
        """
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.5, 0.5, 101)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        expected = -2.0 * jnp.cos(2.0 * jnp.pi * jnp.linspace(-0.5, 0.5, 101))
        assert jnp.allclose(diag.eigenvalues[:, 0], expected, atol=1e-10)

    def test_eigenvalue_range(self):
        """Verify eigenvalue bandwidth spans [-2|t|, 2|t|].

        Constructs the 1D chain with t=-1.5 and diagonalizes at 201
        k-points spanning the full BZ.  The bandwidth of a single-
        orbital nearest-neighbor chain is exactly 4|t| = 6.0, with
        extrema at -3.0 and +3.0.  Asserts min and max eigenvalues
        match these expected values to within 0.05 (accounting for
        finite k-grid sampling that may not exactly hit the BZ center
        and edge).
        """
        model = make_1d_chain_model(t=-1.5)
        kpoints = jnp.linspace(-0.5, 0.5, 201)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        assert float(diag.eigenvalues.min()) == pytest.approx(-3.0, abs=0.05)
        assert float(diag.eigenvalues.max()) == pytest.approx(3.0, abs=0.05)


class TestMakeGrapheneModel:
    """Tests for the graphene tight-binding model factory.

    Validates ``make_graphene_model`` using analytically known properties
    of the graphene pi-band structure: the Gamma-point eigenvalue
    splitting +/-3|t|, the Dirac-point degeneracy at K=(2/3, 1/3, 0),
    and differentiability of eigenvalues with respect to the hopping
    parameter t.
    """

    def test_gamma_point(self):
        """Verify Gamma-point eigenvalues are +/-3|t| for graphene.

        Constructs the graphene model with t=-2.7 eV and diagonalizes
        at Gamma=(0,0,0).  The two-orbital nearest-neighbor model yields
        E(Gamma) = +/-3|t| = +/-8.1 eV, corresponding to the bonding
        and anti-bonding combinations of the two sublattice pz orbitals.
        Asserts sorted eigenvalues match -8.1 and +8.1 to within 0.01 eV.
        """
        model = make_graphene_model(t=-2.7)
        Gamma = jnp.array([[0.0, 0.0, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, Gamma)
        evals = jnp.sort(diag.eigenvalues[0])
        assert float(evals[0]) == pytest.approx(-8.1, abs=0.01)
        assert float(evals[1]) == pytest.approx(8.1, abs=0.01)

    def test_k_point_dirac(self):
        """Verify the Dirac point at K=(2/3, 1/3, 0) has zero-energy eigenvalues.

        Constructs the graphene model with t=-2.7 eV and diagonalizes at
        the high-symmetry K-point in fractional reciprocal coordinates.
        At this point the off-diagonal Bloch sum vanishes identically,
        producing a two-fold degeneracy at E=0 (the Dirac cone apex).
        Asserts both eigenvalues are zero to within 1e-10 eV.
        """
        model = make_graphene_model(t=-2.7)
        K = jnp.array([[2.0 / 3, 1.0 / 3, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, K)
        assert jnp.allclose(diag.eigenvalues[0], 0.0, atol=1e-10)

    def test_gradient_wrt_hopping(self):
        """Verify gradient of eigenvalue sum-of-squares w.r.t. hopping is finite.

        Defines loss = sum(E^2) for the graphene model at k=(0.1, 0.2, 0)
        and differentiates with respect to the hopping parameter array
        using ``jax.grad``.  Asserts all gradient elements are finite,
        confirming end-to-end differentiability of the Hamiltonian
        construction and eigendecomposition for the multi-orbital case.
        """
        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array([[0.1, 0.2, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        def loss(hop):
            m = model._replace(hopping_params=hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues**2)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))
