r"""Validate tight-binding differentiation and eigensystem evidence.

The tests close the parameter-class matrix not already covered by the
all-channel Slater--Koster gate in
``test_slaterkoster.TestBuildSlaterKosterModel.
test_every_integral_has_fd_correct_band_spectral_gradient``. They check
generic-k atomic-position and atomic-SOC derivatives. Further tests cover
exact-degeneracy invariants, gauge budgets, and independent NumPy truth.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.tightb import (
    bloch_hamiltonian,
    eigh_safe,
    eigvalsh_bands,
    group_projector,
    group_trace,
    sk_block,
    tb_parameter_view,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
    make_slater_koster_params,
    make_tb_model,
)
from tests._factories import make_t2g_soc_model
from tests._gradients import assert_grad_matches_fd, gradient_gate

_DIRAC_K: Float64[Array, " 3"] = jnp.asarray(
    (2.0 / 3.0, 1.0 / 3.0, 0.0),
    dtype=jnp.float64,
)
_GENERIC_K: Float64[Array, " 3"] = jnp.asarray(
    (0.173, -0.219, 0.083),
    dtype=jnp.float64,
)
_GRAPHENE_CELLS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (-1, 0, 0),
    (0, -1, 0),
)


def _honeycomb_geometry(
    second_position: Float64[Array, " 3"],
) -> CrystalGeometry:
    """PRIVATE: Build a graphene lattice with a traced second-atom position.

    Parameters
    ----------
    second_position : Float64[Array, " 3"]
        Fractional position of the B sublattice atom; the position
        gradient tests differentiate through it.

    Returns
    -------
    geometry : CrystalGeometry
        Hexagonal two-atom cell with lattice constant 2.46 Angstrom, a
        10 Angstrom vacuum axis, atom A at the origin, and atom B at
        ``second_position``.

    Notes
    -----
    Stacks the traced position with the static origin, so autodiff
    through the geometry reaches only the second atom.
    """
    lattice_constant: float = 2.46
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        (
            (lattice_constant, 0.0, 0.0),
            (
                lattice_constant / 2.0,
                lattice_constant * np.sqrt(3.0) / 2.0,
                0.0,
            ),
            (0.0, 0.0, 10.0),
        ),
        dtype=jnp.float64,
    )
    positions: Float64[Array, "2 3"] = jnp.stack(
        (
            jnp.zeros((3,), dtype=jnp.float64),
            second_position,
        )
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice,
        positions,
        ("X", "X"),
    )
    return geometry


def _pz_honeycomb_model(
    sk_values: Float64[Array, " 2"],
    second_position: Float64[Array, " 3"],
    *,
    spinful: bool,
    onsite_offset: float = 0.0,
) -> TBModel:
    """PRIVATE: Build a fixed-topology pz honeycomb through the public SK
    kernel.

    Parameters
    ----------
    sk_values : Float64[Array, " 2"]
        The ``pp_sigma`` and ``pp_pi`` Slater--Koster integrals in eV.
    second_position : Float64[Array, " 3"]
        Fractional position of the B atom inside the honeycomb cell.
    spinful : bool
        If true, duplicate the pz pair into a four-orbital spinor basis.
    onsite_offset : float
        Sublattice-staggered onsite energy in eV; atom A receives
        ``+onsite_offset`` and atom B receives ``-onsite_offset``.

    Returns
    -------
    model : TBModel
        Honeycomb pz model whose hoppings come from :func:`sk_block` on
        the three nearest-neighbor bonds.

    Notes
    -----
    The three nearest-neighbor cells are exact static metadata. Each
    two-center block uses ``R + tau_B - tau_A``. This matches the
    production builder away from cutoff crossings. The small gate
    excludes topology discovery. Reverse hoppings reuse the forward
    values because the pz--pz element is direction-even.
    """
    geometry: CrystalGeometry = _honeycomb_geometry(second_position)
    sk_params: SlaterKosterParams = make_slater_koster_params(
        sk_values,
        ("X-X:pp_sigma", "X-X:pp_pi"),
    )
    forward: list[Float64[Array, ""]] = []
    cell: Tuple[int, int, int]
    for cell in _GRAPHENE_CELLS:
        displacement: Float64[Array, " 3"] = (
            jnp.asarray(cell, dtype=jnp.float64)
            + geometry.positions[1]
            - geometry.positions[0]
        )
        cartesian: Float64[Array, " 3"] = displacement @ geometry.lattice
        direction: Float64[Array, " 3"] = cartesian / jnp.linalg.norm(
            cartesian
        )
        block: Float64[Array, "3 3"] = sk_block(
            1,
            1,
            sk_params.values,
            direction,
        )
        forward.append(block[1, 1])

    if spinful:
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1, 0, 1),
            n=(2, 2, 2, 2),
            l=(1, 1, 1, 1),
            m=(0, 0, 0, 0),
            spin=(-1, -1, 1, 1),
            labels=("A_down", "B_down", "A_up", "B_up"),
        )
        pairs: Tuple[Tuple[int, int], ...] = (
            ((0, 1),) * 3 + ((1, 0),) * 3 + ((2, 3),) * 3 + ((3, 2),) * 3
        )
        cells: Tuple[Tuple[int, int, int], ...] = (
            _GRAPHENE_CELLS
            + tuple(
                tuple(-component for component in item)
                for item in _GRAPHENE_CELLS
            )
            + _GRAPHENE_CELLS
            + tuple(
                tuple(-component for component in item)
                for item in _GRAPHENE_CELLS
            )
        )
        one_spin: list[Float64[Array, ""]] = forward + forward
        hopping: Complex128[Array, " 12"] = jnp.asarray(
            one_spin + one_spin,
            dtype=jnp.complex128,
        )
        onsite: Float64[Array, " 4"] = jnp.asarray(
            (
                onsite_offset,
                -onsite_offset,
                onsite_offset,
                -onsite_offset,
            ),
            dtype=jnp.float64,
        )
        shell_index: Tuple[int, ...] = (-1,) * 4
    else:
        basis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(2, 2),
            l=(1, 1),
            m=(0, 0),
            labels=("A_pz", "B_pz"),
        )
        pairs = ((0, 1),) * 3 + ((1, 0),) * 3
        cells = _GRAPHENE_CELLS + tuple(
            tuple(-component for component in item) for item in _GRAPHENE_CELLS
        )
        hopping = jnp.asarray(
            forward + forward,
            dtype=jnp.complex128,
        )
        onsite = jnp.asarray(
            (onsite_offset, -onsite_offset),
            dtype=jnp.float64,
        )
        shell_index = (-1, -1)

    model: TBModel = make_tb_model(
        hopping,
        onsite,
        jnp.zeros((0,), dtype=jnp.float64),
        geometry,
        basis,
        pairs,
        cells,
        shell_index,
        spinful,
    )
    return model


def _spectral_square(
    model: TBModel,
    kpoint: Float64[Array, " 3"],
) -> Float64[Array, ""]:
    r"""PRIVATE: Return ``Tr(H**2)``, a symmetric spectral polynomial.

    Parameters
    ----------
    model : TBModel
        Tight-binding model under differentiation.
    kpoint : Float64[Array, " 3"]
        Fractional k-point of the Bloch Hamiltonian.

    Returns
    -------
    value : Float64[Array, ""]
        The real trace of :math:`H(k)^2` in eV squared.

    Notes
    -----
    The trace of a spectral polynomial is invariant under any unitary
    eigenbasis choice. This loss therefore gives a degeneracy-safe
    gradient path through :func:`bloch_hamiltonian` without eigenvector
    gauge ambiguity.
    """
    hamiltonian: Complex128[Array, "n n"] = bloch_hamiltonian(model, kpoint)
    value: Float64[Array, ""] = jnp.real(jnp.trace(hamiltonian @ hamiltonian))
    return value


def _minimal_bands(
    eigenvalues: Float64[Array, "n_k n_bands"],
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"],
) -> DiagonalizedBands:
    """PRIVATE: Attach a minimal geometry and basis to a synthetic eigensystem.

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Synthetic band energies in eV.
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Band-major eigenvector rows for each k-point.

    Returns
    -------
    bands : DiagonalizedBands
        Carrier that wraps the supplied eigensystem with zero k-points,
        a one-site cubic geometry, and an all-s orbital basis.

    Notes
    -----
    The projector and group-trace tests only need a well-formed carrier;
    the placeholder geometry and basis carry no physics.
    """
    n_orbitals: int = eigenvectors.shape[-1]
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        labels=tuple(f"o{index}" for index in range(n_orbitals)),
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues,
        eigenvectors,
        jnp.zeros((eigenvalues.shape[0], 3), dtype=jnp.float64),
        geometry,
        basis,
    )
    return bands


def _normwise_roundoff_budget(
    reference: Array,
    scale: Array,
    *,
    contraction_dimension: int,
    contractions: int,
) -> float:
    """PRIVATE: Derive a normwise f64 budget from the standard gamma-n bound.

    Parameters
    ----------
    reference : Array
        Reference matrix stack whose trailing two axes set the matrix
        dimension.
    scale : Array
        Companion array whose norm also bounds the input magnitude.
    contraction_dimension : int
        Inner dimension of one matrix contraction.
    contractions : int
        Number of chained contractions in the compared expression.

    Returns
    -------
    budget : float
        Normwise roundoff allowance
        ``gamma_n * sqrt(matrix_dimension) * magnitude``.

    Notes
    -----
    Uses the standard Higham bound ``gamma_n = n*eps / (1 - n*eps)``
    with ``n = contraction_dimension * contractions``. Converts the
    entrywise bound to a Frobenius-norm bound through the square root
    of the matrix size. Scales the result by the largest of one and the
    two input norms.
    """
    epsilon: float = np.finfo(np.float64).eps
    operation_count: int = contraction_dimension * contractions
    gamma: float = (
        operation_count * epsilon / (1.0 - operation_count * epsilon)
    )
    matrix_dimension: int = int(np.prod(reference.shape[-2:]))
    entry_to_norm: float = np.sqrt(matrix_dimension)
    magnitude: float = max(
        1.0,
        float(jnp.linalg.norm(reference)),
        float(jnp.linalg.norm(scale)),
    )
    return gamma * entry_to_norm * magnitude


class TestD1GenericK:
    """Validate generic-k derivatives missing from the all-SK-key gate."""

    def test_soc_lambda_forward_reverse_fd_and_nonzero(self) -> None:
        """Differentiate one nondegenerate magnetic t2g+SOC band.

        A small exchange/crystal-field pattern lifts the atomic Kramers
        pairs. This gives the physical individual ``dE_n/dlambda`` case. The
        unmodified t2g fixture instead belongs to the invariant SOC check.

        Notes
        -----
        Compare forward and reverse derivatives with finite differences.
        """
        model: TBModel = make_t2g_soc_model(coupling=0.41)
        parameters: Float64[Array, " 7"]
        rebuild: Callable[[Float64[Array, " 7"]], TBModel]
        parameters, rebuild = tb_parameter_view(model)
        onsite: Float64[Array, " 6"] = jnp.asarray(
            (-0.31, 0.08, 0.22, 0.29, -0.14, 0.37),
            dtype=jnp.float64,
        )
        magnetic: Float64[Array, " 7"] = parameters.at[:6].set(onsite)
        coupling: Float64[Array, ""] = magnetic[-1]

        def loss(value: Float64[Array, ""]) -> Float64[Array, ""]:
            candidate: TBModel = rebuild(magnetic.at[-1].set(value))
            eigenvalues: Float64[Array, " 6"] = eigvalsh_bands(
                candidate,
                _GENERIC_K[None, :],
            )[0]
            return eigenvalues[4]

        initial_bands: Float64[Array, " 6"] = eigvalsh_bands(
            rebuild(magnetic),
            _GENERIC_K[None, :],
        )[0]
        assert float(jnp.min(jnp.diff(initial_bands))) > 1e-2
        gradient_gate(
            loss,
            coupling,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        assert abs(float(jax.grad(loss)(coupling))) > 1e-3

    @pytest.mark.rss_limit_mb(900)
    def test_atomic_position_forward_reverse_fd_and_nonzero(self) -> None:
        """Differentiate an off-crossing buckled-honeycomb band.

        Planar graphene pz hoppings are first-order insensitive to in-plane
        bond rotations and materialized-position changes are Bloch-gauge
        transformations.  A buckled pz honeycomb with unequal pp-sigma and
        pp-pi integrals is the minimal physical fixture with genuine atomic
        position sensitivity.

        Notes
        -----
        Compare forward and reverse derivatives with finite differences.
        """
        sk_values: Float64[Array, " 2"] = jnp.asarray(
            (1.35, -2.7),
            dtype=jnp.float64,
        )
        position: Float64[Array, " 3"] = jnp.asarray(
            (1.0 / 3.0, 1.0 / 3.0, 0.035),
            dtype=jnp.float64,
        )

        def loss(
            candidate_position: Float64[Array, " 3"],
        ) -> Float64[Array, ""]:
            model: TBModel = _pz_honeycomb_model(
                sk_values,
                candidate_position,
                spinful=False,
                onsite_offset=0.17,
            )
            eigenvalues: Float64[Array, " 2"] = eigvalsh_bands(
                model,
                _GENERIC_K[None, :],
            )[0]
            return eigenvalues[1]

        gradient_gate(
            loss,
            position,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        derivative: Float64[Array, " 3"] = jax.grad(loss)(position)
        assert float(jnp.linalg.norm(derivative)) > 1e-2


class TestD2ExactDegeneracy:
    """Differentiate only invariant losses at exact band degeneracies."""

    def test_graphene_k_sk_invariant_matches_fd(self) -> None:
        """Match FD for pp-pi at the exact graphene Dirac point.

        At K, ``Tr(H**2)`` is quadratic in the Dirac splitting. The derivative
        may vanish without invalidating this invariant gate.

        Notes
        -----
        Avoid sorted-band derivatives and energy-threshold masks at crossing.
        """
        planar_position: Float64[Array, " 3"] = jnp.asarray(
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            dtype=jnp.float64,
        )
        sigma: Float64[Array, ""] = jnp.asarray(1.2, dtype=jnp.float64)
        pi: Float64[Array, ""] = jnp.asarray(-2.7, dtype=jnp.float64)

        def loss(value: Float64[Array, ""]) -> Float64[Array, ""]:
            model: TBModel = _pz_honeycomb_model(
                jnp.stack((sigma, value)),
                planar_position,
                spinful=False,
            )
            return _spectral_square(model, _DIRAC_K)

        model: TBModel = _pz_honeycomb_model(
            jnp.stack((sigma, pi)),
            planar_position,
            spinful=False,
        )
        eigenvalues: Float64[Array, " 2"] = eigvalsh_bands(
            model,
            _DIRAC_K[None, :],
        )[0]
        assert float(jnp.max(jnp.abs(eigenvalues))) < 1e-12
        assert_grad_matches_fd(
            loss,
            pi,
            regime="stiff",
            modes=("fwd", "rev"),
        )
        assert bool(jnp.isfinite(jax.grad(loss)(pi)))

    def test_kramers_sk_and_position_invariant_is_nonzero(self) -> None:
        """Validate SK and position classes through exact Kramers pairs.

        The spin-independent buckled honeycomb is two identical spin blocks,
        so every band is exactly double-degenerate at arbitrary k.  Its
        ``Tr(H**2)`` loss is invariant under pair rotations and has nonzero
        sensitivity to pp-sigma, pp-pi, and the second-atom position.

        Notes
        -----
        Require exact pair equality and nonzero invariant gradients.
        """
        initial: Float64[Array, " 5"] = jnp.asarray(
            (1.35, -2.7, 0.34, 0.31, 0.035),
            dtype=jnp.float64,
        )

        def loss(values: Float64[Array, " 5"]) -> Float64[Array, ""]:
            model: TBModel = _pz_honeycomb_model(
                values[:2],
                values[2:],
                spinful=True,
                onsite_offset=0.11,
            )
            return _spectral_square(model, _GENERIC_K)

        model: TBModel = _pz_honeycomb_model(
            initial[:2],
            initial[2:],
            spinful=True,
            onsite_offset=0.11,
        )
        eigenvalues: Float64[Array, " 4"] = eigvalsh_bands(
            model,
            _GENERIC_K[None, :],
        )[0]
        assert jnp.allclose(
            eigenvalues[0::2],
            eigenvalues[1::2],
            rtol=0.0,
            atol=2e-12,
        )
        gradient_gate(
            loss,
            initial,
            regime="stiff",
            modes=("fwd", "rev"),
        )
        derivative: Float64[Array, " 5"] = jax.grad(loss)(initial)
        assert jnp.all(jnp.abs(derivative) > 1e-4)

    def test_kramers_soc_lambda_invariant_is_nonzero(self) -> None:
        """Validate lambda through an exact Kramers-degenerate polynomial.

        The case differentiates a symmetric t2g spectral loss.

        Notes
        -----
        Require exact Kramers pairs and a nonzero invariant gradient.
        """
        model: TBModel = make_t2g_soc_model(coupling=0.41)
        parameters: Float64[Array, " 7"]
        rebuild: Callable[[Float64[Array, " 7"]], TBModel]
        parameters, rebuild = tb_parameter_view(model)
        coupling: Float64[Array, ""] = parameters[-1]

        def loss(value: Float64[Array, ""]) -> Float64[Array, ""]:
            candidate: TBModel = rebuild(parameters.at[-1].set(value))
            return _spectral_square(candidate, _GENERIC_K)

        eigenvalues: Float64[Array, " 6"] = eigvalsh_bands(
            model,
            _GENERIC_K[None, :],
        )[0]
        assert jnp.allclose(
            eigenvalues[0::2],
            eigenvalues[1::2],
            rtol=0.0,
            atol=1e-12,
        )
        gradient_gate(
            loss,
            coupling,
            regime="stiff",
            modes=("fwd", "rev"),
        )
        assert abs(float(jax.grad(loss)(coupling))) > 1e-2


class TestG8GaugeInvariance:
    """Apply phases and independent rotations to every degenerate block."""

    def test_multiple_blocks_obey_dimension_derived_budget(self) -> None:
        """Preserve fixed-group projectors and traces within a gamma-n bound.

        The case rotates three exact two-band blocks independently.

        Notes
        -----
        Derive each tolerance from contraction dimensions and float64 epsilon.
        """
        n_k: int = 2
        n_orbitals: int = 6
        group_size: int = 2
        key: Array = jax.random.key(408)
        keys: list[Array] = list(jax.random.split(key, 16))
        raw: Complex128[Array, "2 6 6"] = jax.random.normal(
            keys[0], (n_k, n_orbitals, n_orbitals)
        ) + 1j * jax.random.normal(keys[1], (n_k, n_orbitals, n_orbitals))
        columns: Complex128[Array, "2 6 6"] = jax.vmap(
            lambda matrix: jnp.linalg.qr(matrix)[0]
        )(raw)
        vectors: Complex128[Array, "2 6 6"] = jnp.swapaxes(columns, -1, -2)
        energies: Float64[Array, "2 6"] = jnp.broadcast_to(
            jnp.asarray((-1.7, -1.7, 0.2, 0.2, 1.4, 1.4)),
            (n_k, n_orbitals),
        )
        phases: Complex128[Array, "2 6"] = jnp.exp(
            1j
            * jax.random.uniform(
                keys[2],
                (n_k, n_orbitals),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=jnp.float64,
            )
        )
        rotated: Complex128[Array, "2 6 6"] = phases[:, :, None] * vectors
        key_index: int = 3
        k_index: int
        start: int
        for k_index in range(n_k):
            for start in range(0, n_orbitals, group_size):
                block_raw: Complex128[Array, "2 2"] = jax.random.normal(
                    keys[key_index], (group_size, group_size)
                ) + 1j * jax.random.normal(
                    keys[key_index + 1],
                    (group_size, group_size),
                )
                rotation: Complex128[Array, "2 2"] = jnp.linalg.qr(block_raw)[
                    0
                ]
                rotated = rotated.at[
                    k_index,
                    start : start + group_size,
                    :,
                ].set(
                    rotation
                    @ rotated[
                        k_index,
                        start : start + group_size,
                        :,
                    ]
                )
                key_index += 2

        observable_raw: Complex128[Array, "6 6"] = jax.random.normal(
            keys[15], (n_orbitals, n_orbitals)
        ) + 1j * jax.random.normal(
            jax.random.fold_in(keys[15], 1),
            (n_orbitals, n_orbitals),
        )
        observable: Complex128[Array, "6 6"] = (
            observable_raw + observable_raw.conj().T
        ) / 2.0
        baseline: DiagonalizedBands = _minimal_bands(energies, vectors)
        transformed: DiagonalizedBands = _minimal_bands(energies, rotated)

        for start in range(0, n_orbitals, group_size):
            group: Tuple[int, ...] = (start, start + 1)
            reference_projector: Array = group_projector(baseline, group)
            actual_projector: Array = group_projector(transformed, group)
            projector_budget: float = _normwise_roundoff_budget(
                reference_projector,
                actual_projector,
                contraction_dimension=group_size,
                contractions=4,
            )
            assert (
                float(jnp.linalg.norm(actual_projector - reference_projector))
                <= projector_budget
            )

            reference_trace: Array = group_trace(
                baseline,
                observable,
                group,
            )
            actual_trace: Array = group_trace(
                transformed,
                observable,
                group,
            )
            trace_budget: float = _normwise_roundoff_budget(
                reference_trace[:, None, None],
                observable,
                contraction_dimension=n_orbitals,
                contractions=6,
            )
            assert (
                float(jnp.linalg.norm(actual_trace - reference_trace))
                <= trace_budget
            )


class TestEighSafeNumPyTruth:
    """Compare production eigenpairs with independent NumPy/LAPACK truth."""

    def test_random_complex_hermitian_eigenpairs(self) -> None:
        """Match eigenvalues, projectors, residuals, and orthonormality.

        The cases span three independently generated complex dimensions.

        Notes
        -----
        Compare production results with NumPy and explicit residual equations.
        """
        rng: np.random.Generator = np.random.default_rng(407)
        dimension: int
        for dimension in (2, 5, 8):
            raw: Complex128[NDArray, "dim dim"] = rng.normal(
                size=(dimension, dimension)
            ) + 1j * (0.73 * rng.normal(size=(dimension, dimension)))
            hamiltonian: Complex128[NDArray, "dim dim"] = (
                raw + raw.conj().T
            ) / 2.0
            hamiltonian += np.diag(np.linspace(-0.4, 0.6, dimension))
            expected_values: Float64[NDArray, " dim"]
            expected_vectors: Complex128[NDArray, "dim dim"]
            expected_values, expected_vectors = np.linalg.eigh(hamiltonian)

            actual_values: Array
            actual_vectors: Array
            actual_values, actual_vectors = eigh_safe(
                jnp.asarray(hamiltonian, dtype=jnp.complex128)
            )
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2e-13,
                atol=2e-13,
            )
            expected_projectors: Complex128[NDArray, "dim dim dim"] = (
                np.einsum(
                    "ib,jb->bij",
                    expected_vectors,
                    expected_vectors.conj(),
                )
            )
            actual_projectors: Complex128[NDArray, "dim dim dim"] = np.einsum(
                "ib,jb->bij",
                np.asarray(actual_vectors),
                np.asarray(actual_vectors).conj(),
            )
            np.testing.assert_allclose(
                actual_projectors,
                expected_projectors,
                rtol=3e-12,
                atol=3e-12,
            )
            residual: Array = (
                jnp.asarray(hamiltonian) @ actual_vectors
                - actual_vectors * actual_values[None, :]
            )
            np.testing.assert_allclose(
                residual,
                0.0,
                rtol=0.0,
                atol=3e-12,
            )
            np.testing.assert_allclose(
                actual_vectors.conj().T @ actual_vectors,
                jnp.eye(dimension, dtype=jnp.complex128),
                rtol=3e-12,
                atol=3e-12,
            )


__all__: list[str] = []
