"""Validate differentiable diagonalization and the VASP adapter.

Extended Summary
----------------
The tests validate the single-k solver, the multi-k solver, and the VASP
adapter. They cover eigensystem properties, gradients, output shapes, and
the policy for phase loss.

"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.tightb import (
    diagonalize_tb,
    eigh_safe,
    eigvalsh_bands,
    eigvalsh_bands_chunked,
    vasp_to_diagonalized,
)
from diffpes.types import (
    EPS_DEG,
    make_band_structure,
    make_crystal_geometry,
    make_orbital_basis,
    make_orbital_projection,
)
from tests._factories import make_1d_chain_model, make_graphene_model
from tests._gradients import gradient_gate


def _geometry(n_atoms: int) -> diffpes.types.CrystalGeometry:
    """Build a minimal geometry for atom-resolved adapter tests."""
    geometry: diffpes.types.CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((n_atoms, 3), dtype=jnp.float64),
        species=tuple("X" for _ in range(n_atoms)),
    )
    return geometry


class TestEighSafe:
    """Validate ``eigh_safe``.

    The tests verify the real eigenvalues and the orthonormal eigenvectors of
    a Hermitian matrix.

    :see: :func:`~diffpes.tightb.eigh_safe`
    """

    def test_eigenvalues_real(self) -> None:
        """Verify the Hermitian solver returns real eigenvalues.

        The Hermitian eigensolver produces a float64 eigenvalue array from a
        complex128 input matrix.

        Notes
        -----
        The test diagonalizes a 2x2 matrix with complex off-diagonal entries.
        It compares the eigenvalue dtype with ``jnp.float64``.
        """
        H: Array
        evals: Array
        evecs: Array

        H = jnp.array(
            [[1.0, 0.5 + 0.1j], [0.5 - 0.1j, 2.0]], dtype=jnp.complex128
        )
        evals, evecs = eigh_safe(H)
        assert evals.dtype == jnp.float64

    def test_eigenvectors_orthogonal(self) -> None:
        """Verify the Hermitian solver returns orthonormal eigenvectors.

        The eigenvectors of the Hermitian matrix form a unitary
        transformation.

        Notes
        -----
        The test diagonalizes a real-symmetric 2x2 matrix. It compares
        ``U^dag U`` with the identity at an absolute tolerance of 1e-10.
        """
        H: Array
        evals: Array
        evecs: Array
        overlap: Array

        H = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=jnp.complex128)
        evals, evecs = eigh_safe(H)
        overlap = evecs.conj().T @ evecs
        assert jnp.allclose(overlap, jnp.eye(2), atol=1e-10)

    def test_degenerate_jvp_is_finite(self) -> None:
        """Verify the regularized JVP remains finite at exact degeneracy.

        The case differentiates an identity Hamiltonian along a Hermitian direction.

        Notes
        -----
        Require finite eigenvalue and eigenvector tangent arrays.
        """
        hamiltonian: Array = jnp.eye(2, dtype=jnp.complex128)
        direction: Array = jnp.asarray(
            [[0.2, 0.1 + 0.3j], [0.1 - 0.3j, -0.2]],
            dtype=jnp.complex128,
        )
        tangents: tuple[Array, Array]
        _, tangents = jax.jvp(eigh_safe, (hamiltonian,), (direction,))
        assert jnp.all(jnp.isfinite(tangents[0]))
        assert jnp.all(jnp.isfinite(tangents[1]))

    def test_rejects_invalid_matrices_eager_and_jitted(self) -> None:
        """Reject non-finite and non-Hermitian eigensolver inputs.

        The same numerical contracts remain active in eager and compiled
        execution instead of relying on an unchecked LAPACK triangle.

        Notes
        -----
        Exercise one NaN matrix and one finite matrix with unequal conjugate
        off-diagonal entries through both execution modes.
        """
        cases: tuple[tuple[Array, str], ...] = (
            (
                jnp.asarray(
                    [[jnp.nan, 0.0], [0.0, 1.0]],
                    dtype=jnp.complex128,
                ),
                "entries must be finite",
            ),
            (
                jnp.asarray(
                    [[1.0, 0.2 + 0.4j], [0.2 + 0.4j, 2.0]],
                    dtype=jnp.complex128,
                ),
                "must be Hermitian",
            ),
        )
        under_jit: bool
        hamiltonian: Array
        expected_message: str
        for under_jit in (False, True):
            solver: Callable[[Array], tuple[Array, Array]] = (
                eqx.filter_jit(eigh_safe) if under_jit else eigh_safe
            )
            for hamiltonian, expected_message in cases:
                with pytest.raises(RuntimeError, match=expected_message):
                    solver(hamiltonian)

    def test_degenerate_projector_gradient_matches_finite_difference(
        self,
    ) -> None:
        """Match a generic-complex degenerate-projector derivative to FD.

        The loss consumes a gauge-invariant two-state projector; exact
        degeneracy leaves its individual eigenvectors without a unique form.

        Notes
        -----
        Differentiate a Hermitian three-level model through both automatic
        modes and the shared central-finite-difference harness.
        """
        base: Array = jnp.diag(
            jnp.asarray([0.0, 0.0, 2.0], dtype=jnp.complex128)
        )
        direction: Array = jnp.asarray(
            [
                [0.3, 0.2 + 0.4j, -0.15 + 0.21j],
                [0.2 - 0.4j, -0.1, 0.37 - 0.19j],
                [-0.15 - 0.21j, 0.37 + 0.19j, 0.25],
            ],
            dtype=jnp.complex128,
        )
        observable: Array = jnp.asarray(
            [
                [0.4, -0.3 + 0.11j, 0.28 + 0.23j],
                [-0.3 - 0.11j, -0.2, -0.17 + 0.31j],
                [0.28 - 0.23j, -0.17 - 0.31j, 0.7],
            ],
            dtype=jnp.complex128,
        )

        def loss(theta: Array) -> Array:
            """Return a real expectation of the lowest-group projector."""
            eigensystem: tuple[Array, Array] = eigh_safe(
                base + theta * direction
            )
            eigenvectors: Array = eigensystem[1]
            occupied: Array = eigenvectors[:, :2]
            projector: Array = occupied @ occupied.conj().T
            value: Array = jnp.real(jnp.trace(observable @ projector))
            return value

        theta: Array = jnp.asarray(0.0, dtype=jnp.float64)
        gradient_gate(loss, theta, regime="smooth")

    def test_regularization_bias_matches_analytic_gap_law(self) -> None:
        r"""Measure the eigenvector-gradient bias over an avoided-gap sweep.

        Rotating a two-level Hamiltonian with fixed gap ``g`` gives the exact
        projector observable :math:`I(\theta)=\cos^2\theta`. The regularized
        inverse gap in :func:`eigh_safe` scales its derivative by
        :math:`g^2/(g^2+\epsilon^2)`, so the relative bias is exactly
        :math:`\epsilon^2/(g^2+\epsilon^2)`.

        Notes
        -----
        Sweep gaps from ``2`` through ``1e6`` times ``EPS_DEG``. Match the
        measured bias to the analytic law and require relative error below
        ``1.1e-10`` once ``g >= 1e5 * EPS_DEG``. The ``1e5`` threshold follows
        directly from the declared quadratic law. A ``1e3`` threshold implies
        ``1e-6`` relative bias instead of ``1e-10``.
        """
        theta: Array = jnp.asarray(0.37, dtype=jnp.float64)
        gap_ratios: Array = jnp.asarray(
            [2.0, 5.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6],
            dtype=jnp.float64,
        )
        exact_derivative: Array = -jnp.sin(2.0 * theta)

        def derivative_for_gap(gap: Array) -> Array:
            def loss(angle: Array) -> Array:
                cosine: Array = jnp.cos(angle)
                sine: Array = jnp.sin(angle)
                rotation: Array = jnp.asarray(
                    [[cosine, -sine], [sine, cosine]],
                    dtype=jnp.float64,
                )
                diagonal: Array = jnp.diag(
                    jnp.asarray(
                        [-gap / 2.0, gap / 2.0],
                        dtype=jnp.float64,
                    )
                )
                hamiltonian: Array = (rotation @ diagonal @ rotation.T).astype(
                    jnp.complex128
                )
                eigenvectors: Array = eigh_safe(hamiltonian)[1]
                observable: Array = jnp.abs(eigenvectors[0, 0]) ** 2
                return observable

            derivative: Array = jax.grad(loss)(theta)
            return derivative

        gaps: Array = gap_ratios * EPS_DEG
        derivatives: Array = jax.vmap(derivative_for_gap)(gaps)
        measured_bias: Array = jnp.abs(
            (derivatives - exact_derivative) / exact_derivative
        )
        expected_bias: Array = EPS_DEG**2 / (gaps**2 + EPS_DEG**2)

        assert jnp.allclose(
            measured_bias,
            expected_bias,
            rtol=5e-6,
            atol=5e-15,
        )
        assert measured_bias[-2] <= 1.1e-10
        assert measured_bias[-1] <= 1.1e-12


class TestEigvalshBands:
    """Validate :func:`~diffpes.tightb.eigvalsh_bands`."""

    def test_matches_full_diagonalization(self) -> None:
        """Verify the fast path matches full band eigenvalues.

        The case evaluates both APIs for graphene at two generic k-points.

        Notes
        -----
        Compare ascending eigenvalues at absolute tolerance ``1e-13``.
        """
        model: diffpes.types.TBModel = make_graphene_model()
        kpoints: Array = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]], dtype=jnp.float64
        )
        actual: Array = eigvalsh_bands(model, kpoints)
        expected: Array = diagonalize_tb(model, kpoints).eigenvalues
        assert jnp.allclose(actual, expected, atol=1e-13)


class TestEigvalshBandsChunked:
    """Validate :func:`~diffpes.tightb.eigvalsh_bands_chunked`."""

    def test_matches_ordinary_eigenvalue_path(self) -> None:
        """Match the ordinary solver without a full k-point Hamiltonian batch.

        The case evaluates eight graphene momenta in two chunks.

        Notes
        -----
        Compare every sorted eigenvalue with the non-chunked public path.
        """
        model: diffpes.types.TBModel = make_graphene_model()
        k_x: Array = jnp.linspace(-0.4, 0.4, 8)
        kpoints: Array = jnp.stack(
            (k_x, 0.2 * k_x, jnp.zeros_like(k_x)),
            axis=-1,
        )
        actual: Array = eigvalsh_bands_chunked(model, kpoints, 4)
        expected: Array = eigvalsh_bands(model, kpoints)

        assert jnp.allclose(actual, expected, rtol=1e-13, atol=1e-13)


class TestDiagonalizeTB:
    """Validate ``diagonalize_tb``.

    The tests use graphene and 1D chain models. They cover output shapes,
    eigenvalue order, hopping gradients, and eigenvector indexing.

    :see: :func:`~diffpes.tightb.diagonalize_tb`
    """

    def test_output_shapes(self) -> None:
        """Verify output shapes match (K, B) eigenvalues and (K, B, O) eigenvectors.

        The solver retains both k-points and both orbitals in the
        ``DiagonalizedBands`` fields.

        Notes
        -----
        The test diagonalizes a two-orbital graphene model at two k-points.
        It compares all three output shapes with their specified values.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        assert diag.eigenvalues.shape == (2, 2)
        assert diag.eigenvectors.shape == (2, 2, 2)
        assert diag.kpoints.shape == (2, 3)

    def test_preserves_explicit_orbital_positions(self) -> None:
        """Carry Wannier centres through the electronic-structure seam.

        Distinct per-orbital centres define the gauge of the returned
        eigenvectors and must remain available to coherent consumers.

        Notes
        -----
        Rebuild graphene with explicit centres, diagonalize it, and compare
        the downstream carrier field exactly.
        """
        template: diffpes.types.TBModel = make_graphene_model()
        centres: Array = jnp.asarray(
            [[0.13, 0.07, 0.0], [0.41, 0.22, 0.0]],
            dtype=jnp.float64,
        )
        model: diffpes.types.TBModel = diffpes.types.make_tb_model(
            hopping_amplitudes=template.hopping_amplitudes,
            onsite_energies=template.onsite_energies,
            soc_lambdas=template.soc_lambdas,
            geometry=template.geometry,
            basis=template.basis,
            hopping_pairs=template.hopping_pairs,
            hopping_cells=template.hopping_cells,
            shell_index=template.shell_index,
            spinor=template.spinor,
            orbital_positions=centres,
        )

        bands: diffpes.types.DiagonalizedBands = diagonalize_tb(
            model,
            jnp.asarray([[0.17, 0.09, 0.0]], dtype=jnp.float64),
        )

        assert bands.orbital_positions is not None
        assert jnp.array_equal(bands.orbital_positions, centres)

    def test_eigenvalues_sorted(self) -> None:
        """Verify eigenvalues have ascending order at each k-point.

        The lower band precedes the upper band at every k-point. Downstream
        band analysis depends on this energy order.

        Notes
        -----
        The test diagonalizes the graphene model at three k-points. It compares
        the first eigenvalue with the second eigenvalue at each point.
        """
        i: int

        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model()
        kpoints = jnp.array(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]
        )
        diag = diagonalize_tb(model, kpoints)
        for i in range(3):
            assert float(diag.eigenvalues[i, 0]) <= float(
                diag.eigenvalues[i, 1]
            )

    def test_differentiable(self) -> None:
        """Verify JAX differentiates the eigenvalue sum with respect to hopping parameters.

        The eigenspectrum carries hopping sensitivity through the Hamiltonian
        and the eigendecomposition.

        Notes
        -----
        The test differentiates the eigenvalue sum for a 1D chain with
        ``jax.grad``. It checks every hopping gradient for a finite value.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        grad: Array

        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.array([[0.25, 0.0, 0.0]])

        def loss(hop):
            m: diffpes.types.TBModel
            d: diffpes.types.DiagonalizedBands

            m = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                hop,
            )
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues)

        grad = jax.grad(loss)(model.hopping_amplitudes)
        assert jnp.all(jnp.isfinite(grad))

    def test_eigenvectors_shape_convention(self) -> None:
        """Verify eigenvector indexing: eigenvectors[k, band, orbital].

        The last eigenvector axis contains the orbital coefficients. Each
        vector has unit norm across this axis.

        Notes
        -----
        The test diagonalizes graphene at one k-point. It checks shape
        ``(1, 2, 2)`` and compares each norm with 1.0 at a tolerance of 1e-10.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands
        norms: Array

        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)

        assert diag.eigenvectors.shape == (1, 2, 2)

        norms = jnp.sum(jnp.abs(diag.eigenvectors[0]) ** 2, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)


class TestVaspToDiagonalized:
    """Validate ``vasp_to_diagonalized``.

    The adapter uses VASP eigenvalues and approximates eigenvectors from
    orbital projections. The tests cover shapes, normalization, and the
    policy for phase loss.

    :see: :func:`~diffpes.tightb.vasp_to_diagonalized`
    """

    def test_output_shapes(self) -> None:
        """Verify output shapes match the input band structure dimensions.

        The adapter retains five k-points and three bands. Its eigenvector
        result also contains the three selected orbitals.

        Notes
        -----
        The test supplies projections for two atoms and nine channels. It
        checks eigenvalue shape ``(5, 3)`` and eigenvector shape ``(5, 3, 3)``.
        """
        K: int
        B: int
        A: int
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis
        diag: diffpes.types.DiagonalizedBands

        K, B, A = 5, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((K, B, A, 9)) / 9.0
        )
        basis = make_orbital_basis(
            atom_indices=(0, 0, 1),
            n=(1, 2, 2),
            l=(0, 1, 1),
            m=(0, 0, 1),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, _geometry(A), basis, phase_loss="ignore"
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, 3)

    def test_eigenvectors_normalized(self) -> None:
        """Verify the VASP projection adapter normalizes approximate eigenvectors.

        The adapter produces unit eigenvector norms from approximate orbital
        weights. This property holds when the raw weights do not sum to one.

        Notes
        -----
        The test supplies s and pz weights of 0.3 and 0.7. It compares every
        squared norm with 1.0 at an absolute tolerance of 1e-10.
        """
        K: int
        B: int
        A: int
        bands: diffpes.types.BandStructure
        proj: Array
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis
        diag: diffpes.types.DiagonalizedBands
        norms: Array

        K, B, A = 3, 2, 1
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        proj = jnp.zeros((K, B, A, 9))
        proj = proj.at[:, :, :, 0].set(0.3)
        proj = proj.at[:, :, :, 2].set(0.7)
        orb_proj = make_orbital_projection(projections=proj)
        basis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 2),
            l=(0, 1),
            m=(0, 0),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, _geometry(A), basis, phase_loss="ignore"
        )
        norms = jnp.sum(jnp.abs(diag.eigenvectors) ** 2, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)

    def test_selects_each_registered_atom_and_channel(self) -> None:
        """Select projection values by paired atom and orbital channel.

        Distinct distractor weights expose any accidental reduction or
        independent indexing along the atom and channel axes.

        Notes
        -----
        Compare the normalized p-x, s, and d-xy coefficients with their
        hand-selected square-root weights in a deliberately permuted basis.
        """
        bands: diffpes.types.BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        projections: Array = jnp.zeros((1, 1, 2, 9), dtype=jnp.float64)
        projections = projections.at[0, 0, 1, 3].set(0.09)
        projections = projections.at[0, 0, 0, 0].set(0.16)
        projections = projections.at[0, 0, 1, 4].set(0.25)
        projections = projections.at[0, 0, 0, 3].set(0.81)
        projections = projections.at[0, 0, 1, 0].set(0.64)
        projections = projections.at[0, 0, 0, 4].set(0.49)
        orb_proj: diffpes.types.OrbitalProjection = make_orbital_projection(
            projections=projections
        )
        basis: diffpes.types.OrbitalBasis = make_orbital_basis(
            atom_indices=(1, 0, 1),
            n=(2, 1, 3),
            l=(1, 0, 2),
            m=(1, 0, -2),
        )
        diagonalized: diffpes.types.DiagonalizedBands = vasp_to_diagonalized(
            bands,
            orb_proj,
            _geometry(2),
            basis,
            phase_loss="ignore",
        )
        expected: Array = jnp.asarray([0.3, 0.4, 0.5]) / jnp.sqrt(0.5)
        assert jnp.allclose(
            diagonalized.eigenvectors[0, 0],
            expected,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_rejects_zero_selected_norm_eager_and_jitted(self) -> None:
        """Reject an all-zero selected projection vector in both modes.

        Returning a finite zero vector falsely presents an invalid band as a
        normalized approximate eigenstate.

        Notes
        -----
        Send the same zero-weight carrier through eager execution and an
        Equinox-filtered JIT wrapper, matching the value-sensitive diagnostic.
        """
        bands: diffpes.types.BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj: diffpes.types.OrbitalProjection = make_orbital_projection(
            projections=jnp.zeros((1, 1, 1, 9))
        )
        basis: diffpes.types.OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )

        def adapt(candidate: diffpes.types.OrbitalProjection) -> Array:
            """Return coefficients after adapting one projection carrier."""
            diagonalized: diffpes.types.DiagonalizedBands = (
                vasp_to_diagonalized(
                    bands,
                    candidate,
                    _geometry(1),
                    basis,
                    phase_loss="ignore",
                )
            )
            return diagonalized.eigenvectors

        under_jit: bool
        for under_jit in (False, True):
            adapter: Callable[[diffpes.types.OrbitalProjection], Array] = (
                eqx.filter_jit(adapt) if under_jit else adapt
            )
            with pytest.raises(
                RuntimeError,
                match="selected projection norm must be nonzero",
            ):
                adapter(orb_proj)

    def test_rejects_geometry_projection_atom_mismatch(self) -> None:
        """Reject geometry and PROCAR carriers with different atom counts.

        Atom indices have no unambiguous physical meaning when the geometry
        and projection atom axes describe different structures.

        Notes
        -----
        Pair a one-atom geometry with a two-atom projection carrier and match
        the structural diagnostic before coefficient construction.
        """
        bands: diffpes.types.BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj: diffpes.types.OrbitalProjection = make_orbital_projection(
            projections=jnp.ones((1, 1, 2, 9))
        )
        basis: diffpes.types.OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        with pytest.raises(
            ValueError,
            match="geometry and projection atom counts must agree",
        ):
            vasp_to_diagonalized(
                bands,
                orb_proj,
                _geometry(1),
                basis,
                phase_loss="ignore",
            )

    def test_rejects_spin_resolved_basis(self) -> None:
        """Reject a spin-labelled basis while the adapter lacks spin support.

        Scalar PROCAR weights cannot determine which spin-resolved basis
        coefficient should receive a selected orbital weight.

        Notes
        -----
        Supply one explicitly spin-up basis orbital and require the adapter's
        temporary unsupported-feature diagnostic.
        """
        bands: diffpes.types.BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj: diffpes.types.OrbitalProjection = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9))
        )
        basis: diffpes.types.OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
            spin=(1,),
        )
        with pytest.raises(
            ValueError,
            match="spin-resolved orbital bases are not supported",
        ):
            vasp_to_diagonalized(
                bands,
                orb_proj,
                _geometry(1),
                basis,
                phase_loss="ignore",
            )

    def test_warns_by_default(self) -> None:
        """Verify the default phase_loss policy emits a RuntimeWarning.

        The default policy warns the user because real VASP projections do
        not contain complex eigenvector phases.

        Notes
        -----
        The test calls the adapter with a minimal input and no policy argument.
        It expects a ``RuntimeWarning`` that contains ``cannot recover complex``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        with pytest.warns(RuntimeWarning, match="cannot recover complex"):
            _ = vasp_to_diagonalized(bands, orb_proj, _geometry(1), basis)

    def test_phase_loss_error_mode_raises(self) -> None:
        """Verify phase_loss="error" raises ValueError instead of warning.

        The error policy rejects approximate eigenvectors that lack complex
        phase information.

        Notes
        -----
        The test calls the adapter with a minimal input and the error policy.
        It expects a ``ValueError`` that contains ``cannot recover complex``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        with pytest.raises(ValueError, match="cannot recover complex"):
            _ = vasp_to_diagonalized(
                bands,
                orb_proj,
                _geometry(1),
                basis,
                phase_loss="error",
            )

    def test_invalid_phase_loss_raises(self) -> None:
        """Verify an unrecognized phase_loss value raises an error.

        The type contract permits only ``warn``, ``ignore``, and ``error``.

        Notes
        -----
        The test passes ``phase_loss="bad"`` to the adapter with a minimal
        input. It expects the runtime type checker to raise an exception.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        with pytest.raises(Exception):
            _ = vasp_to_diagonalized(
                bands,
                orb_proj,
                _geometry(1),
                basis,
                phase_loss="bad",
            )

    def test_f_orbital_raises(self) -> None:
        """Verify an f-orbital (l=3) in the basis raises ValueError.

        The VASP PROCAR format contains only s, p, and d channels. The adapter
        rejects an f-orbital outside this set.

        Notes
        -----
        The test supplies an orbital with ``l=3`` and ``m=0``. It expects a
        ``ValueError`` that contains ``not in VASP 9-orbital set``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            atom_indices=(0,),
            n=(4,),
            l=(3,),
            m=(0,),
        )
        with pytest.raises(ValueError, match="not in VASP 9-orbital set"):
            _ = vasp_to_diagonalized(
                bands,
                orb_proj,
                _geometry(1),
                basis,
                phase_loss="ignore",
            )
