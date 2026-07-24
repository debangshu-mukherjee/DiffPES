"""Validate the tight-binding Hamiltonian builder.

Extended Summary
----------------
The tests validate ``bloch_hamiltonian`` and the two model fixtures.
They cover Hermiticity, the matrix shape, JIT, analytic dispersions, and
gradients with respect to hopping parameters.

"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.tightb import bloch_hamiltonian, bloch_hamiltonian_batch
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)
from tests._assertions import assert_rejects
from tests._factories import make_1d_chain_model, make_graphene_model
from tests._gradients import gradient_gate


def _make_empty_model() -> TBModel:
    """Build a two-orbital model with onsite energies and no hoppings."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        species=("A", "B"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(-1, -1),
    )
    return model


def _make_gauge_probe_model() -> TBModel:
    """Build a two-site complex model with nontrivial basis positions."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.asarray(
            [[0.1, 0.0, 0.0], [0.37, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        species=("A", "B"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
    )
    hopping: complex = 0.7 + 0.4j
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray(
            [hopping, hopping.conjugate()],
            dtype=jnp.complex128,
        ),
        onsite_energies=jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1), (1, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0)),
        shell_index=(-1, -1),
    )
    return model


class TestBlochHamiltonian:
    """Validate ``bloch_hamiltonian``.

    The tests cover the Bloch Hamiltonian ``H(k)`` from hopping parameters
    and lattice vectors. They verify Hermiticity, the matrix shape, and JIT.

    :see: :func:`~diffpes.tightb.bloch_hamiltonian`
    """

    def test_hermitian(self) -> None:
        """Verify H(k) satisfies Hermiticity at the specified k-point.

        The graphene Hamiltonian satisfies ``H = H^dag``. This property
        permits a real eigenspectrum.

        Notes
        -----
        The test computes ``H`` at ``k=(0.3, 0.2, 0.0)``. It compares ``H``
        with its conjugate transpose at an absolute tolerance of 1e-12.
        """
        model: diffpes.types.TBModel
        k: Array
        H: Array

        model = make_graphene_model()
        k = jnp.array([0.3, 0.2, 0.0])
        H = bloch_hamiltonian(model, k)
        assert jnp.allclose(H, H.conj().T, atol=1e-12)

    def test_correct_shape(self) -> None:
        """Verify H(k) has shape (n_orbitals, n_orbitals).

        The Hamiltonian has one row and one column for each model orbital.

        Notes
        -----
        The test computes the two-orbital graphene Hamiltonian at one k-point.
        It compares the output shape with ``(2, 2)``.
        """
        model: diffpes.types.TBModel
        k: Array
        H: Array

        model = make_graphene_model()
        k = jnp.array([0.1, 0.2, 0.0])
        H = bloch_hamiltonian(model, k)
        assert H.shape == (2, 2)

    def test_jit_compatible(self) -> None:
        """Verify JAX compiles ``bloch_hamiltonian``.

        The compiled builder accepts a traced k-point and produces finite
        matrix elements.

        Notes
        -----
        The test compiles the builder for the 1D chain model with ``jax.jit``.
        It checks the result at ``k=(0.25, 0, 0)`` for finite values.
        """
        model: diffpes.types.TBModel
        f: Callable[..., Any]
        H: Array

        model = make_1d_chain_model()
        f = jax.jit(lambda k: bloch_hamiltonian(model, k))
        H = f(jnp.array([0.25, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(H))

    def test_basis_position_gauge_matches_diagonal_unitary(self) -> None:
        """Match cell-origin assembly after the predicted gauge transform.

        The case compares basis-position assembly with an explicit diagonal transform.

        Notes
        -----
        Build the cell-origin matrix directly and compare matrices and spectra.
        """
        model: TBModel = _make_gauge_probe_model()
        kpoint: Array = jnp.asarray([0.23, 0.0, 0.0], dtype=jnp.float64)
        actual: Array = bloch_hamiltonian(model, kpoint)
        cell_origin: Array = jnp.diag(
            model.onsite_energies.astype(jnp.complex128)
        )
        index: int
        orbital_i: int
        orbital_j: int
        for index, (orbital_i, orbital_j) in enumerate(model.hopping_pairs):
            cell: Array = jnp.asarray(
                model.hopping_cells[index],
                dtype=jnp.float64,
            )
            phase: Array = jnp.exp(2j * jnp.pi * jnp.dot(kpoint, cell))
            cell_origin = cell_origin.at[orbital_i, orbital_j].add(
                model.hopping_amplitudes[index] * phase
            )
        orbital_positions: Array = model.geometry.positions[
            jnp.asarray(model.basis.atom_indices)
        ]
        unitary: Array = jnp.diag(
            jnp.exp(2j * jnp.pi * (orbital_positions @ kpoint))
        )
        expected: Array = unitary.conj().T @ cell_origin @ unitary

        assert jnp.allclose(actual, expected, atol=1e-13)
        assert jnp.allclose(
            jnp.linalg.eigvalsh(actual),
            jnp.linalg.eigvalsh(cell_origin),
            atol=1e-13,
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_closed_hoppings_remain_hermitian(self, seed: int) -> None:
        """Assemble Hermitian matrices from generic complex closed lists.

        Each seed creates independent complex hoppings and a random k-point.

        Notes
        -----
        Construct reverse amplitudes by conjugation and compare the matrix with its adjoint.
        """
        model: TBModel = make_graphene_model()
        key: Array = jax.random.key(seed)
        real_key: Array
        imaginary_key: Array
        kpoint_key: Array
        real_key, imaginary_key, kpoint_key = jax.random.split(key, 3)
        forward: Array = jax.random.normal(real_key, (3,)) + 1j * (
            0.7 * jax.random.normal(imaginary_key, (3,))
        )
        amplitudes: Array = jnp.concatenate((forward, jnp.conj(forward)))
        candidate: TBModel = eqx.tree_at(
            lambda item: item.hopping_amplitudes,
            model,
            amplitudes,
        )
        kpoint: Array = jax.random.uniform(
            kpoint_key,
            (3,),
            minval=-0.5,
            maxval=0.5,
        )
        hamiltonian: Array = bloch_hamiltonian(candidate, kpoint)

        assert jnp.allclose(
            hamiltonian,
            hamiltonian.conj().T,
            atol=1e-12,
        )

    def test_rejects_mutated_hoppings_eager_and_jit(self) -> None:
        """Reject non-finite and no-longer-closed differentiable updates.

        The cases bypass factory checks through direct PyTree leaf replacement.

        Notes
        -----
        Run eager and compiled rejection gates for open and nonfinite amplitudes.
        """
        model: TBModel = make_1d_chain_model()
        kpoint: Array = jnp.asarray([0.25, 0.0, 0.0], dtype=jnp.float64)
        open_model: TBModel = eqx.tree_at(
            lambda item: item.hopping_amplitudes,
            model,
            jnp.asarray([1.0 + 0.0j, 2.0 + 0.0j]),
        )
        assert_rejects(
            bloch_hamiltonian,
            open_model,
            kpoint,
            match="must remain Hermitian-closed",
        )
        nonfinite_model: TBModel = eqx.tree_at(
            lambda item: item.hopping_amplitudes,
            model,
            jnp.asarray([jnp.nan + 0.0j, jnp.nan + 0.0j]),
        )
        assert_rejects(
            bloch_hamiltonian,
            nonfinite_model,
            kpoint,
            match="hopping amplitudes finite",
        )

    def test_rejects_soc_and_accepts_spinor_kinetic_model(self) -> None:
        """Reject SOC while accepting an already doubled spinor basis.

        A spinor carrier supplies its full Hamiltonian dimension and explicit
        spin-block hoppings; the assembler must never double it again.

        Notes
        -----
        Reject nonzero shell couplings, then compare a zero-SOC two-spin
        kinetic model with its independent diagonal cosine bands.
        """
        empty_model: TBModel = _make_empty_model()
        soc_basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
        )
        soc_model: TBModel = make_tb_model(
            hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
            onsite_energies=empty_model.onsite_energies,
            soc_lambdas=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
            geometry=empty_model.geometry,
            basis=soc_basis,
            hopping_pairs=(),
            hopping_cells=(),
            shell_index=(0, 1),
        )
        kpoint: Array = jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float64)
        assert_rejects(
            bloch_hamiltonian,
            soc_model,
            kpoint,
            match="nonzero SOC requires",
        )
        spin_basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
            spin=(-1, 1),
        )
        spin_hoppings: Array = jnp.asarray(
            [-1.0, -1.0, -2.0, -2.0],
            dtype=jnp.complex128,
        )
        spinor_model: TBModel = make_tb_model(
            hopping_amplitudes=spin_hoppings,
            onsite_energies=jnp.asarray([0.3, -0.2], dtype=jnp.float64),
            soc_lambdas=jnp.zeros(1, dtype=jnp.float64),
            geometry=empty_model.geometry,
            basis=spin_basis,
            hopping_pairs=((0, 0), (0, 0), (1, 1), (1, 1)),
            hopping_cells=(
                (1, 0, 0),
                (-1, 0, 0),
                (1, 0, 0),
                (-1, 0, 0),
            ),
            shell_index=(0, 0),
            spinor=True,
        )
        hamiltonian: Array = bloch_hamiltonian(spinor_model, kpoint)
        cosine: Array = jnp.cos(2.0 * jnp.pi * kpoint[0])
        expected: Array = jnp.diag(
            jnp.asarray(
                [0.3 - 2.0 * cosine, -0.2 - 4.0 * cosine],
                dtype=jnp.complex128,
            )
        )
        assert hamiltonian.shape == (2, 2)
        assert jnp.allclose(hamiltonian, expected, atol=1e-13)


class TestBlochHamiltonianBatch:
    """Validate :func:`~diffpes.tightb.bloch_hamiltonian_batch`."""

    def test_matches_single_point_vmap(self) -> None:
        """Verify batched assembly equals explicit single-point stacking.

        The case evaluates two graphene k-points through both assembly paths.

        Notes
        -----
        Compare the complex matrices at absolute tolerance ``1e-13``.
        """
        model: diffpes.types.TBModel = make_graphene_model()
        kpoints: Array = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]], dtype=jnp.float64
        )
        actual: Array = bloch_hamiltonian_batch(model, kpoints)
        expected: Array = jnp.stack(
            tuple(bloch_hamiltonian(model, point) for point in kpoints)
        )
        assert jnp.allclose(actual, expected, atol=1e-13)

    def test_empty_hopping_list_and_empty_batch(self) -> None:
        """Preserve onsite energies and exact zero-length batch shapes.

        The case evaluates an onsite-only model and an empty k-point array.

        Notes
        -----
        Compare the single matrix with the onsite diagonal and check batch shape.
        """
        model: TBModel = _make_empty_model()
        hamiltonian: Array = bloch_hamiltonian(
            model,
            jnp.zeros(3, dtype=jnp.float64),
        )
        batch: Array = bloch_hamiltonian_batch(
            model,
            jnp.zeros((0, 3), dtype=jnp.float64),
        )

        assert jnp.allclose(
            hamiltonian,
            jnp.diag(model.onsite_energies),
            atol=0.0,
        )
        assert batch.shape == (0, 2, 2)


class TestMake1DChainModel:
    """Validate the 1D chain tight-binding model factory.

    The tests compare the model with its analytic cosine dispersion.
    They also verify the eigenvalue interval ``[-2|t|, 2|t|]``.

    :see: :func:`~tests._factories.make_1d_chain_model`
    """

    def test_cosine_dispersion(self) -> None:
        """Verify E(k) = 2t*cos(2*pi*k) for the 1D chain.

        The single-band model with ``t=-1.0`` has the dispersion
        ``E(k) = -2*cos(2*pi*k)``.

        Notes
        -----
        The test diagonalizes 101 k-points in the Brillouin zone
        ``[-0.5, 0.5]``. It compares all eigenvalues at an absolute tolerance
        of 1e-10.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands
        expected: Array

        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.5, 0.5, 101)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        expected = -2.0 * jnp.cos(2.0 * jnp.pi * jnp.linspace(-0.5, 0.5, 101))
        assert jnp.allclose(
            diag.eigenvalues[:, 0],
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    def test_eigenvalue_range(self) -> None:
        """Verify eigenvalue bandwidth spans [-2|t|, 2|t|].

        A chain with ``t=-1.5`` has eigenvalue limits of -3.0 and 3.0 eV.
        The corresponding bandwidth is 6.0 eV.

        Notes
        -----
        The test diagonalizes 201 k-points across the Brillouin zone. It
        compares both extrema at an absolute tolerance of 0.05 eV.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_1d_chain_model(t=-1.5)
        kpoints = jnp.linspace(-0.5, 0.5, 201)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        assert float(diag.eigenvalues.min()) == pytest.approx(-3.0, abs=0.05)
        assert float(diag.eigenvalues.max()) == pytest.approx(3.0, abs=0.05)


class TestMakeGrapheneModel:
    """Validate the graphene tight-binding model factory.

    The tests use analytic properties of the graphene pi-band structure.
    They cover the Gamma point, the Dirac point, and hopping gradients.

    :see: :func:`~tests._factories.make_graphene_model`
    """

    def test_gamma_point(self) -> None:
        """Verify graphene has Gamma-point eigenvalues of +/-3|t|.

        The two-orbital model with ``t=-2.7`` eV has eigenvalues of -8.1
        and 8.1 eV at the Gamma point.

        Notes
        -----
        The test diagonalizes the model at ``Gamma=(0, 0, 0)``. It compares
        both eigenvalues at an absolute tolerance of 0.01 eV.
        """
        model: diffpes.types.TBModel
        Gamma: Array
        diag: diffpes.types.DiagonalizedBands
        evals: Array

        model = make_graphene_model(t=-2.7)
        Gamma = jnp.array([[0.0, 0.0, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, Gamma)
        evals = jnp.sort(diag.eigenvalues[0])
        assert float(evals[0]) == pytest.approx(-8.1, abs=1e-12)
        assert float(evals[1]) == pytest.approx(8.1, abs=1e-12)

    def test_k_point_dirac(self) -> None:
        """Verify the Dirac point at K=(2/3, 1/3, 0) has zero-energy eigenvalues.

        The off-diagonal Bloch sum vanishes at this k-point. Both eigenvalues
        consequently equal zero.

        Notes
        -----
        The test diagonalizes the model with ``t=-2.7`` eV at the Dirac
        point. It compares both eigenvalues with zero at a tolerance of 1e-10 eV.
        """
        model: diffpes.types.TBModel
        K: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model(t=-2.7)
        K = jnp.array([[2.0 / 3, 1.0 / 3, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, K)
        assert jnp.allclose(
            diag.eigenvalues[0],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_matches_closed_form_along_gamma_k_m(self) -> None:
        r"""Match both graphene bands along the full Gamma--K--M path.

        The nearest-neighbour honeycomb spectrum is
        :math:`E_\pm(k)=\pm |t|\,|1+e^{-2\pi i k_1}
        +e^{-2\pi i k_2}|` in the cell-origin gauge. The basis-position
        gauge used by production changes eigenvector phases but not these
        eigenvalues.

        Notes
        -----
        Build 65 points on each path segment, including both endpoints.
        Require the complete two-band path to agree at absolute tolerance
        ``1e-12`` with zero relative tolerance.
        """
        hopping: float = -2.7
        model: TBModel = make_graphene_model(t=hopping)
        gamma: Array = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
        k_point: Array = jnp.asarray(
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            dtype=jnp.float64,
        )
        m_point: Array = jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float64)
        interpolation: Array = jnp.linspace(0.0, 1.0, 65)[:, None]
        gamma_to_k: Array = gamma + interpolation * (k_point - gamma)
        k_to_m: Array = k_point + interpolation[1:] * (m_point - k_point)
        kpoints: Array = jnp.concatenate((gamma_to_k, k_to_m), axis=0)

        from diffpes.tightb.diagonalize import eigvalsh_bands

        actual: Array = eigvalsh_bands(model, kpoints)
        structure_factor: Array = (
            1.0
            + jnp.exp(-2j * jnp.pi * kpoints[:, 0])
            + jnp.exp(-2j * jnp.pi * kpoints[:, 1])
        )
        magnitude: Array = jnp.abs(hopping) * jnp.abs(structure_factor)
        expected: Array = jnp.stack((-magnitude, magnitude), axis=-1)

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_gradient_wrt_hopping(self) -> None:
        """Match hopping autodiff to FD on the conjugate-closed manifold.

        The graphene eigenspectrum carries sensitivity to the hopping
        parameters through the Hamiltonian and the eigendecomposition.

        Notes
        -----
        The test parameterizes three generic complex forward hoppings by six
        real values. It reconstructs reverse hoppings by conjugation before
        differentiating ``sum(E**2)`` at ``k=(0.13, 0.21, 0)``.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array([[0.13, 0.21, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        def loss(parameters: Array) -> Array:
            m: diffpes.types.TBModel
            d: diffpes.types.DiagonalizedBands
            forward: Array = parameters[:, 0] + 1j * parameters[:, 1]
            hopping: Array = jnp.concatenate((forward, jnp.conj(forward)))
            m = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                hopping,
            )
            d = diagonalize_tb(m, kpoints)
            result: Array = jnp.sum(d.eigenvalues**2)
            return result

        parameters: Array = jnp.asarray(
            [[-2.7, 0.2], [-2.5, -0.15], [-2.8, 0.35]],
            dtype=jnp.float64,
        )
        gradient_gate(loss, parameters, regime="smooth", atol=1e-7)
