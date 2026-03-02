"""Tests for differentiable diagonalization and VASP adapter.

Extended Summary
----------------
Validates the tight-binding diagonalization pipeline: the single-k
solver ``diagonalize_single_k``, the multi-k batch solver
``diagonalize_tb``, and the VASP-to-diagonalized adapter
``vasp_to_diagonalized``.  Tests cover eigenvalue dtype (real),
eigenvector orthonormality, output shapes, eigenvalue ordering,
differentiability with respect to hopping parameters, eigenvector
shape conventions, the VASP adapter's approximate eigenvector
normalization from DFT projections, and the phase-loss warning/error
policy for VASP projection data.

Routine Listings
----------------
:class:`TestDiagonalizeSingleK`
    Tests for diagonalize_single_k.
:class:`TestDiagonalizeTB`
    Tests for diagonalize_tb.
:class:`TestVaspToDiagonalized`
    Tests for vasp_to_diagonalized.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.tightb.diagonalize import (
    diagonalize_single_k,
    diagonalize_tb,
    vasp_to_diagonalized,
)
from diffpes.tightb.hamiltonian import make_1d_chain_model, make_graphene_model
from diffpes.types import (
    make_band_structure,
    make_orbital_basis,
    make_orbital_projection,
)


class TestDiagonalizeSingleK:
    """Tests for ``diagonalize_single_k``.

    Validates the core single-k-point Hermitian eigensolver.  Tests
    verify that eigenvalues are real-valued (as guaranteed by the
    spectral theorem for Hermitian matrices) and that eigenvectors
    form an orthonormal set (U^dag U = I).
    """

    def test_eigenvalues_real(self):
        """Verify eigenvalues of a Hermitian matrix are real-valued.

        Constructs a 2x2 Hermitian matrix with off-diagonal complex
        entries (0.5 +/- 0.1j) and on-site energies 1.0 and 2.0.
        Diagonalizes and asserts the eigenvalue dtype is ``jnp.float64``,
        confirming the solver correctly extracts real eigenvalues from
        the complex Hermitian input.
        """
        H = jnp.array(
            [[1.0, 0.5 + 0.1j], [0.5 - 0.1j, 2.0]], dtype=jnp.complex128
        )
        evals, evecs = diagonalize_single_k(H)
        assert evals.dtype == jnp.float64

    def test_eigenvectors_orthogonal(self):
        """Verify eigenvectors are orthonormal (U^dag U = I).

        Constructs a real-symmetric 2x2 matrix with on-site energies 1.0
        and 2.0 and off-diagonal coupling 0.5.  Diagonalizes and computes
        the overlap matrix U^dag U.  Asserts it equals the 2x2 identity
        to within 1e-10, confirming the eigenvectors form a unitary
        transformation.
        """
        H = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=jnp.complex128)
        evals, evecs = diagonalize_single_k(H)
        overlap = evecs.conj().T @ evecs
        assert jnp.allclose(overlap, jnp.eye(2), atol=1e-10)


class TestDiagonalizeTB:
    """Tests for ``diagonalize_tb``.

    Validates the multi-k-point tight-binding diagonalization function
    using the built-in graphene and 1D chain model factories.  Tests
    cover output shapes (K x B eigenvalues, K x B x O eigenvectors),
    ascending eigenvalue ordering, differentiability with respect to
    hopping parameters, and the eigenvector indexing convention
    [k, band, orbital].
    """

    def test_output_shapes(self):
        """Verify output shapes match (K, B) eigenvalues and (K, B, O) eigenvectors.

        Constructs a graphene model (2 orbitals) and diagonalizes at 2
        k-points.  Asserts eigenvalues shape is (2, 2), eigenvectors
        shape is (2, 2, 2), and kpoints shape is (2, 3).  This confirms
        the function correctly maps K k-points through a B=O=2 orbital
        tight-binding model.
        """
        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        assert diag.eigenvalues.shape == (2, 2)
        assert diag.eigenvectors.shape == (2, 2, 2)
        assert diag.kpoints.shape == (2, 3)

    def test_eigenvalues_sorted(self):
        """Verify eigenvalues are sorted in ascending order at each k-point.

        Diagonalizes the graphene model at 3 k-points and checks that
        eigenvalues[i, 0] <= eigenvalues[i, 1] for each k-point.  This
        ascending-energy convention is required by downstream band-
        structure analysis code that assumes band indices correspond to
        energy ordering.
        """
        model = make_graphene_model()
        kpoints = jnp.array(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]
        )
        diag = diagonalize_tb(model, kpoints)
        for i in range(3):
            assert float(diag.eigenvalues[i, 0]) <= float(
                diag.eigenvalues[i, 1]
            )

    def test_differentiable(self):
        """Verify eigenvalue sum is differentiable w.r.t. hopping parameters.

        Uses the 1D chain model (t=-1.0) at a single k-point (0.25, 0, 0).
        Defines loss = sum(eigenvalues) and differentiates with respect to
        the hopping parameter array using ``jax.grad``.  Asserts all
        gradient elements are finite, confirming the Hamiltonian
        construction and eigendecomposition are end-to-end differentiable
        -- a prerequisite for inverse band-structure fitting.
        """
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.array([[0.25, 0.0, 0.0]])

        def loss(hop):
            m = model._replace(hopping_params=hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))

    def test_eigenvectors_shape_convention(self):
        """Verify eigenvector indexing: eigenvectors[k, band, orbital].

        Diagonalizes the graphene model at a single k-point and asserts
        the eigenvectors array has shape (1, 2, 2) -- (K=1, B=2, O=2).
        Also verifies that each band's eigenvector is normalized
        (sum |c_o|^2 = 1) to within 1e-10, confirming the convention
        that the last axis indexes orbital coefficients.
        """
        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        # K=1, B=2, O=2
        assert diag.eigenvectors.shape == (1, 2, 2)
        # Eigenvectors should be normalized per band
        norms = jnp.sum(jnp.abs(diag.eigenvectors[0]) ** 2, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)


class TestVaspToDiagonalized:
    """Tests for ``vasp_to_diagonalized``.

    Validates the adapter that constructs a ``DiagonalizedTB`` result
    from VASP band structure and orbital projection data.  This path
    uses DFT-computed eigenvalues directly and approximates eigenvectors
    from PROCAR-style orbital projections, enabling ARPES simulation
    from first-principles inputs.  Tests cover output shapes, eigenvector
    normalization, the default warning on phase-information loss, and the
    error-mode policy that rejects phase-less approximation.
    """

    def test_output_shapes(self):
        """Verify output shapes match the input band structure dimensions.

        Constructs synthetic VASP-like inputs with K=5 k-points, B=3
        bands, A=2 atoms, and a 3-orbital basis (1s, 2pz, 2px).  The
        orbital projections have shape (K, B, A, 9) with 9 = s + 3p + 5d
        projection channels per atom.  Uses ``phase_loss="ignore"`` to
        suppress the phase-loss warning.  Asserts the output eigenvalues
        shape is (5, 3) and eigenvectors shape is (5, 3, 3), where the
        last dimension matches the number of orbitals in the basis.
        """
        K, B, A = 5, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((K, B, A, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1, 2, 2),
            l_values=(0, 1, 1),
            m_values=(0, 0, 1),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, basis, phase_loss="ignore"
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, 3)

    def test_eigenvectors_normalized(self):
        """Verify approximate eigenvectors from VASP projections are normalized.

        Constructs a 2-orbital basis (1s, 2pz) with synthetic projections
        where the s-channel weight is 0.3 and the pz-channel weight is
        0.7 for a single atom across K=3 k-points and B=2 bands.  Uses
        ``phase_loss="ignore"`` to suppress the phase-loss warning.  The
        adapter should re-normalize these weights to unit length.  Asserts
        that sum(|c_o|^2, axis=-1) = 1.0 for all (k, band) pairs to
        within 1e-10, confirming the normalization step works correctly
        even when raw VASP projections do not sum to unity.
        """
        K, B, A = 3, 2, 1
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        proj = jnp.zeros((K, B, A, 9))
        proj = proj.at[:, :, :, 0].set(0.3)  # s
        proj = proj.at[:, :, :, 2].set(0.7)  # pz
        orb_proj = make_orbital_projection(projections=proj)
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, basis, phase_loss="ignore"
        )
        norms = jnp.sum(jnp.abs(diag.eigenvectors) ** 2, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)

    def test_warns_by_default(self):
        """Verify the default phase_loss policy emits a RuntimeWarning.

        Calls ``vasp_to_diagonalized`` without specifying ``phase_loss``
        (default is "warn") for a minimal 1-k, 1-band, 1-atom, 1-orbital
        setup.  Asserts a ``RuntimeWarning`` matching "cannot recover
        complex" is raised, informing the user that VASP PROCAR
        projections are real-valued and cannot reconstruct the complex
        eigenvector phases.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.warns(RuntimeWarning, match="cannot recover complex"):
            _ = vasp_to_diagonalized(bands, orb_proj, basis)

    def test_phase_loss_error_mode_raises(self):
        """Verify phase_loss="error" raises ValueError instead of warning.

        Calls ``vasp_to_diagonalized`` with ``phase_loss="error"`` for
        the same minimal setup.  Asserts a ``ValueError`` matching
        "cannot recover complex" is raised, providing a strict mode for
        callers who require complex eigenvector phases and should not
        silently proceed with a phase-less approximation.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.raises(ValueError, match="cannot recover complex"):
            _ = vasp_to_diagonalized(
                bands, orb_proj, basis, phase_loss="error"
            )

    def test_invalid_phase_loss_raises(self):
        """Verify that an unrecognised phase_loss value raises an error.

        The ``phase_loss`` parameter is annotated ``Literal['warn',
        'ignore', 'error']``, so beartype rejects any other string.
        Passes ``phase_loss="bad"`` (via an unsafe cast) and asserts
        that any exception is raised.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.raises(Exception):
            _ = vasp_to_diagonalized(bands, orb_proj, basis, phase_loss="bad")

    def test_f_orbital_raises(self):
        """Verify that an f-orbital (l=3) in the basis raises ValueError.

        VASP PROCAR covers only s, p, d channels (9-orbital set).
        Passing an f-orbital (l=3, m=0) to ``vasp_to_diagonalized``
        should raise a ``ValueError`` indicating the orbital is not in
        the VASP 9-orbital set.
        """
        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(4,),
            l_values=(3,),
            m_values=(0,),
        )
        with pytest.raises(ValueError, match="not in VASP 9-orbital set"):
            _ = vasp_to_diagonalized(
                bands, orb_proj, basis, phase_loss="ignore"
            )
