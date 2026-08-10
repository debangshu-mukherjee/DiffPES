r"""Validate spin doubling and atomic spin--orbit coupling.

Extended Summary
----------------
The tests pin complex-basis ladder matrices and the real-cubic operator
transform. They also cover down--up spin order, shell placement, atomic
multiplets, Kramers degeneracy, and shell-coupling differentiation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Complex128

from diffpes.maths import real_harmonic_unitary
from diffpes.tightb import (
    bloch_hamiltonian,
    eigh_safe,
    l_matrices,
    soc_matrix,
    soc_shell_block,
    spin_double_basis,
    spin_double_model,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)
from tests._gradients import gradient_gate


def _one_atom_geometry() -> CrystalGeometry:
    """PRIVATE: Build a one-atom orthogonal-cell geometry.

    Returns
    -------
    geometry : CrystalGeometry
        Unit-cubic-cell geometry in Angstrom with one species-X atom at
        the origin.

    Notes
    -----
    Atomic spin--orbit coupling acts on-site, so one atom in a unit
    cell is the smallest carrier the SOC fixtures need.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    return geometry


def _make_atomic_shell_model(
    angular_momentum: int,
    coupling: float,
) -> TBModel:
    """PRIVATE: Build a spin-doubled isolated complete shell.

    Parameters
    ----------
    angular_momentum : int
        Orbital angular momentum ``l`` of the complete shell.
    coupling : float
        Atomic spin--orbit coupling strength in eV.

    Returns
    -------
    model : TBModel
        Spin-doubled model of one isolated ``2l + 1``-orbital shell
        with zero onsite energies and no hoppings.

    Notes
    -----
    Without hoppings the Hamiltonian is exactly ``lambda L.S``, whose
    spectrum splits into the two atomic multiplets ``j = l + 1/2`` and
    ``j = l - 1/2`` with the closed eigenvalues ``lambda*l/2`` and
    ``-lambda*(l + 1)/2``. The multiplet tests compare against these
    degeneracies and energies.
    """
    magnetic_numbers: Tuple[int, ...] = tuple(
        range(-angular_momentum, angular_momentum + 1)
    )
    shell_size: int = 2 * angular_momentum + 1
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * shell_size,
        n=(angular_momentum + 1,) * shell_size,
        l=(angular_momentum,) * shell_size,
        m=magnetic_numbers,
        labels=tuple(f"l{angular_momentum}_m{m}" for m in magnetic_numbers),
    )
    spinless: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((shell_size,), dtype=jnp.float64),
        soc_lambdas=jnp.asarray([coupling], dtype=jnp.float64),
        geometry=_one_atom_geometry(),
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(0,) * shell_size,
    )
    model: TBModel = spin_double_model(spinless)
    return model


def _make_dispersive_p_model(coupling: float = 0.37) -> TBModel:
    """PRIVATE: Build an inversion- and time-reversal-symmetric p-shell
    model.

    Parameters
    ----------
    coupling : float
        Atomic spin--orbit coupling strength in eV.

    Returns
    -------
    model : TBModel
        Spin-doubled one-atom p-shell model with orbital-diagonal
        conjugate hopping pairs ``(-0.7, -1.1, -1.6)`` eV along x,
        onsite energies ``(0.2, -0.1, 0.4)`` eV, and the given SOC
        strength.

    Notes
    -----
    Real orbital-diagonal hoppings on ``(+/-1, 0, 0)`` preserve both
    inversion and time reversal. Every band of this dispersive model
    must therefore stay two-fold Kramers degenerate at each k-point.
    The degeneracy tests certify this property.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
        labels=("p_y", "p_z", "p_x"),
    )
    hopping_values: Tuple[float, ...] = (-0.7, -1.1, -1.6)
    amplitudes: list[complex] = []
    pairs: list[Tuple[int, int]] = []
    cells: list[Tuple[int, int, int]] = []
    orbital: int
    hopping: float
    for orbital, hopping in enumerate(hopping_values):
        amplitudes.extend((complex(hopping), complex(hopping)))
        pairs.extend(((orbital, orbital), (orbital, orbital)))
        cells.extend(((1, 0, 0), (-1, 0, 0)))
    spinless: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray(amplitudes, dtype=jnp.complex128),
        onsite_energies=jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float64),
        soc_lambdas=jnp.asarray([coupling], dtype=jnp.float64),
        geometry=_one_atom_geometry(),
        basis=basis,
        hopping_pairs=tuple(pairs),
        hopping_cells=tuple(cells),
        shell_index=(0, 0, 0),
    )
    model: TBModel = spin_double_model(spinless)
    return model


class TestLMatrices:
    """Validate :func:`diffpes.tightb.l_matrices`."""

    @pytest.mark.parametrize("angular_momentum", range(5))
    def test_ladder_algebra(self, angular_momentum: int) -> None:
        r"""Match Hermiticity and the :math:`[L_+,L_-]=2L_z` algebra.

        Every supported angular momentum must satisfy both matrix identities.

        Notes
        -----
        Compare the adjoint and commutator directly at strict tolerance.
        """
        lz: Array
        raising: Array
        lowering: Array
        lz, raising, lowering = l_matrices(angular_momentum)

        assert jnp.allclose(lowering, raising.conj().T, atol=0.0, rtol=0.0)
        assert jnp.allclose(
            raising @ lowering - lowering @ raising,
            2.0 * lz,
            atol=1e-13,
            rtol=1e-13,
        )

    def test_selected_ladder_entries(self) -> None:
        """Pin ascending magnetic-number indices and square-root factors.

        The d-shell provides distinct inner and outer ladder coefficients.

        Notes
        -----
        Compare selected entries and the complete diagonal with literals.
        """
        lz: Array
        raising: Array
        lowering: Array
        lz, raising, lowering = l_matrices(2)
        expected_diagonal: Array = jnp.asarray(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            dtype=jnp.complex128,
        )

        assert jnp.array_equal(jnp.diag(lz), expected_diagonal)
        assert raising[1, 0] == pytest.approx(2.0)
        assert raising[2, 1] == pytest.approx(jnp.sqrt(6.0))
        assert raising[4, 3] == pytest.approx(2.0)

    @pytest.mark.parametrize("invalid", (-1, 5))
    def test_rejects_unsupported_l(self, invalid: int) -> None:
        """Reject angular momenta outside the package-wide supported range.

        Both sides of the closed supported interval must fail validation.

        Notes
        -----
        Match the shared angular-momentum diagnostic for each boundary case.
        """
        with pytest.raises(ValueError, match="0 <= l <="):
            l_matrices(invalid)


class TestSocShellBlock:
    """Validate :func:`diffpes.tightb.soc_shell_block`."""

    def test_p_block_matches_canonical_analytic_matrix(self) -> None:
        """Pin every p-shell matrix element in down--up block order.

        The local orbital order is ``(p_y, p_z, p_x)`` within each spin
        sector. The literal matrix checks spin-block placement, imaginary
        signs, and the factor of one half independently of the spectrum.

        Notes
        -----
        Compare every complex entry with the canonical literal matrix.
        """
        expected: Complex128[Array, "6 6"] = 0.5 * jnp.asarray(
            [
                [0, 0, -1j, 0, -1j, 0],
                [0, 0, 0, 1j, 0, 1],
                [1j, 0, 0, 0, -1, 0],
                [0, -1j, 0, 0, 0, 1j],
                [1j, 0, -1, 0, 0, 0],
                [0, 1, 0, -1j, 0, 0],
            ],
            dtype=jnp.complex128,
        )
        actual: Complex128[Array, "6 6"] = soc_shell_block(1)

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("angular_momentum", (1, 2))
    def test_commutes_with_total_jz(self, angular_momentum: int) -> None:
        r"""Verify :math:`[L\cdot S,J_z]=0` after the real-basis transform.

        Both complete p and d shells must preserve total angular momentum.

        Notes
        -----
        Transform orbital :math:`L_z`, form :math:`J_z`, and evaluate the
        commutator.
        """
        lz_complex: Array
        raising: Array
        lowering: Array
        lz_complex, raising, lowering = l_matrices(angular_momentum)
        del raising, lowering
        unitary: Array = real_harmonic_unitary(angular_momentum)
        lz_real: Array = unitary.conj() @ lz_complex @ unitary.T
        orbital_size: int = 2 * angular_momentum + 1
        spin_z: Array = jnp.diag(
            jnp.asarray([-0.5, 0.5], dtype=jnp.complex128)
        )
        total_jz: Array = jnp.kron(jnp.eye(2), lz_real) + jnp.kron(
            spin_z,
            jnp.eye(orbital_size),
        )
        block: Array = soc_shell_block(angular_momentum)
        commutator: Array = block @ total_jz - total_jz @ block

        assert jnp.allclose(commutator, 0.0, rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("angular_momentum", range(5))
    def test_is_hermitian_and_jittable(self, angular_momentum: int) -> None:
        """Preserve Hermiticity through static shell-block compilation.

        Every supported shell size must compile to the same physical matrix.

        Notes
        -----
        Compile a zero-argument closure and compare it with its adjoint.
        """
        block: Array = jax.jit(lambda: soc_shell_block(angular_momentum))()

        assert jnp.allclose(
            block,
            block.conj().T,
            rtol=0.0,
            atol=1e-13,
        )


class TestSpinDoubleBasis:
    """Validate :func:`diffpes.tightb.spin_double_basis`."""

    def test_basis_uses_declared_down_then_up_order(self) -> None:
        """Keep the original orbital block down and append an up copy.

        Every static orbital field must repeat in the same partner order.

        Notes
        -----
        Compare spin tags, quantum numbers, atom indices, and labels exactly.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(1, 0),
            n=(2, 3),
            l=(1, 2),
            m=(0, -2),
            labels=("p_z", "d_xy"),
        )
        doubled: OrbitalBasis = spin_double_basis(basis)

        assert doubled.spin == (-1, -1, 1, 1)
        assert doubled.atom_indices == (1, 0, 1, 0)
        assert doubled.n == (2, 3, 2, 3)
        assert doubled.l == (1, 2, 1, 2)
        assert doubled.m == (0, -2, 0, -2)
        assert doubled.labels == ("p_z", "d_xy", "p_z", "d_xy")


class TestSpinDoubleModel:
    """Validate :func:`diffpes.tightb.spin_double_model`."""

    def test_model_duplicates_only_spin_diagonal_hoppings(self) -> None:
        """Verify onsite, hopping, cell, and shell duplication exactly once.

        The spin-up copy must contain no accidental spin-flip hopping.

        Notes
        -----
        Compare all static metadata and numerical arrays with explicit values.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
            labels=("s",),
        )
        original: TBModel = make_tb_model(
            hopping_amplitudes=jnp.asarray(
                [-0.8 + 0.2j, -0.8 - 0.2j],
                dtype=jnp.complex128,
            ),
            onsite_energies=jnp.asarray([0.3], dtype=jnp.float64),
            soc_lambdas=jnp.asarray([0.0], dtype=jnp.float64),
            geometry=_one_atom_geometry(),
            basis=basis,
            hopping_pairs=((0, 0), (0, 0)),
            hopping_cells=((1, 0, 0), (-1, 0, 0)),
            shell_index=(0,),
        )
        doubled: TBModel = spin_double_model(original)

        assert doubled.spinor
        assert doubled.basis.spin == (-1, 1)
        assert doubled.hopping_pairs == (
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        assert doubled.hopping_cells == (
            (1, 0, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
        )
        assert doubled.shell_index == (0, 0)
        assert jnp.array_equal(
            doubled.onsite_energies,
            jnp.asarray([0.3, 0.3], dtype=jnp.float64),
        )
        assert jnp.array_equal(
            doubled.hopping_amplitudes,
            jnp.asarray(
                [
                    -0.8 + 0.2j,
                    -0.8 - 0.2j,
                    -0.8 + 0.2j,
                    -0.8 - 0.2j,
                ],
                dtype=jnp.complex128,
            ),
        )

    def test_rejects_an_already_spinful_basis_or_model(self) -> None:
        """Prevent accidental second doubling of a Hamiltonian dimension.

        Both public doubling helpers must recognize existing spin metadata.

        Notes
        -----
        Reuse one doubled atomic model and match each targeted diagnostic.
        """
        model: TBModel = _make_atomic_shell_model(1, 0.2)

        with pytest.raises(ValueError, match="spinless basis"):
            spin_double_basis(model.basis)
        with pytest.raises(ValueError, match="spinless model"):
            spin_double_model(model)


class TestSocMatrix:
    """Validate :func:`diffpes.tightb.soc_matrix`."""

    def test_places_two_shell_blocks_at_their_global_indices(self) -> None:
        """Place independent shell strengths without cross-shell coupling.

        Two atoms must receive disjoint blocks with different scalar weights.

        Notes
        -----
        Build the expected global matrix by explicit indexed updates.
        """
        spinless_basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0, 0, 1, 1, 1),
            n=(2,) * 6,
            l=(1,) * 6,
            m=(-1, 0, 1, -1, 0, 1),
            labels=("a_py", "a_pz", "a_px", "b_py", "b_pz", "b_px"),
        )
        basis: OrbitalBasis = spin_double_basis(spinless_basis)
        shell_index: Tuple[int, ...] = (0, 0, 0, 1, 1, 1) * 2
        lambdas: Array = jnp.asarray([0.2, -0.35], dtype=jnp.float64)
        actual: Array = soc_matrix(basis, shell_index, lambdas)
        block: Array = soc_shell_block(1)
        first: Array = jnp.asarray([0, 1, 2, 6, 7, 8])
        second: Array = jnp.asarray([3, 4, 5, 9, 10, 11])
        expected: Array = jnp.zeros((12, 12), dtype=jnp.complex128)
        expected = expected.at[first[:, None], first[None, :]].set(
            lambdas[0] * block
        )
        expected = expected.at[second[:, None], second[None, :]].set(
            lambdas[1] * block
        )

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-13)
        assert jnp.allclose(
            actual[first[:, None], second[None, :]],
            0.0,
            rtol=0.0,
            atol=0.0,
        )

    def test_supports_projected_t2g_subshell(self) -> None:
        """Verify the d-shell projection onto dxy, dyz, and dxz channels.

        The projected t2g spectrum must retain its analytic multiplicities.

        Notes
        -----
        Diagonalize the six-state block and compare all eigenvalues.
        """
        basis: OrbitalBasis = spin_double_basis(
            make_orbital_basis(
                atom_indices=(0, 0, 0),
                n=(3, 3, 3),
                l=(2, 2, 2),
                m=(-2, -1, 1),
                labels=("d_xy", "d_yz", "d_xz"),
            )
        )
        coupling: float = 0.4
        matrix: Array = soc_matrix(
            basis,
            (0,) * 6,
            jnp.asarray([coupling], dtype=jnp.float64),
        )
        eigenvalues: Array = jnp.linalg.eigvalsh(matrix)
        expected: Array = jnp.asarray(
            [-coupling / 2.0] * 4 + [coupling] * 2,
            dtype=jnp.float64,
        )

        assert jnp.allclose(
            eigenvalues,
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    def test_rejects_missing_spin_partners(self) -> None:
        """Reject active shells lacking the matching opposite-spin channel.

        An incomplete magnetic-number pairing violates the shell contract.

        Notes
        -----
        Construct mismatched spin metadata and match the validation message.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(2, 2),
            l=(1, 1),
            m=(0, 1),
            spin=(-1, 1),
        )

        with pytest.raises(ValueError, match="matching, unique"):
            soc_matrix(
                basis,
                (0, 0),
                jnp.asarray([0.2], dtype=jnp.float64),
            )

    @pytest.mark.parametrize(
        ("angular_momentum", "lower_multiplicity", "upper_multiplicity"),
        ((1, 2, 4), (2, 4, 6)),
    )
    def test_atomic_multiplets_through_bloch_hamiltonian(
        self,
        angular_momentum: int,
        lower_multiplicity: int,
        upper_multiplicity: int,
    ) -> None:
        """Match analytic p- and d-shell energies and multiplicities.

        Bloch assembly must preserve the isolated atomic multiplet structure.

        Notes
        -----
        Compare sorted eigenvalues and the total analytic splitting.
        """
        coupling: float = 0.4
        model: TBModel = _make_atomic_shell_model(
            angular_momentum,
            coupling,
        )
        hamiltonian: Array = bloch_hamiltonian(
            model,
            jnp.asarray([0.19, -0.23, 0.11], dtype=jnp.float64),
        )
        actual: Array = jnp.linalg.eigvalsh(hamiltonian)
        lower: float = -0.5 * (angular_momentum + 1) * coupling
        upper: float = 0.5 * angular_momentum * coupling
        expected: Array = jnp.asarray(
            [lower] * lower_multiplicity + [upper] * upper_multiplicity,
            dtype=jnp.float64,
        )

        assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-12)
        assert float(actual[-1] - actual[0]) == pytest.approx(
            (angular_momentum + 0.5) * coupling,
            abs=1e-12,
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_kramers_degeneracy_at_random_k(self, seed: int) -> None:
        """Keep every band paired in a nontrivial TRS+inversion model.

        Random k-points exercise dispersive orbital channels away from symmetry
        anchors.

        Notes
        -----
        Compare adjacent sorted eigenvalues across five deterministic keys.
        """
        model: TBModel = _make_dispersive_p_model()
        kpoint: Array = jax.random.uniform(
            jax.random.key(seed),
            (3,),
            minval=-0.5,
            maxval=0.5,
            dtype=jnp.float64,
        )
        hamiltonian: Array = bloch_hamiltonian(model, kpoint)
        eigenvalues: Array = jnp.linalg.eigvalsh(hamiltonian)

        assert jnp.allclose(
            eigenvalues[0::2],
            eigenvalues[1::2],
            rtol=0.0,
            atol=1e-10,
        )

    def test_lambda_gradient_through_degenerate_spectral_invariant(
        self,
    ) -> None:
        """Match AD and FD for a symmetric loss at Kramers degeneracy.

        A spectral invariant avoids choosing vectors inside each degenerate
        pair.

        Notes
        -----
        Apply the shared smooth gradient gate and check the analytic result.
        """
        model: TBModel = _make_atomic_shell_model(1, 0.37)
        kpoint: Array = jnp.asarray([0.17, -0.21, 0.09], dtype=jnp.float64)

        def loss(coupling: Array) -> Array:
            candidate: TBModel = eqx.tree_at(
                lambda item: item.soc_lambdas,
                model,
                coupling[None],
            )
            hamiltonian: Array = bloch_hamiltonian(candidate, kpoint)
            eigenvalues: Array = eigh_safe(hamiltonian)[0]
            value: Array = jnp.sum(eigenvalues**2)
            return value

        coupling: Array = jnp.asarray(0.37, dtype=jnp.float64)
        gradient_gate(loss, coupling, regime="smooth")
        derivative: Array = jax.grad(loss)(coupling)

        assert float(derivative) == pytest.approx(
            6.0 * float(coupling),
            rel=1e-12,
            abs=1e-12,
        )
