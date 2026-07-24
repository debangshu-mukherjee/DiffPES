r"""Validate Hermitian tight-binding observable builders.

The tests pin the down--up Pauli convention, arbitrary basis ordering,
orbital projectors, unit-strength atomic L dot S, and analytic Rashba spin
texture.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array

from diffpes.tightb import expectation_path
from diffpes.tightb.operators import (
    ls_operator,
    orbital_projector,
    spin_operator,
)
from diffpes.types import (
    DiagonalizedBands,
    OrbitalBasis,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
)


def _spin_basis() -> OrbitalBasis:
    """Build two spatial orbitals in a deliberately interleaved spin order."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 1, 2),
        l=(0, 1, 0, 1),
        m=(0, 0, 0, 0),
        spin=(-1, 1, 1, -1),
        labels=("s_down", "p_up", "s_up", "p_down"),
    )
    return basis


def _rashba_bands() -> tuple[DiagonalizedBands, Array]:
    """Diagonalize analytic two-band Rashba matrices away from Gamma."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        spin=(-1, 1),
        labels=("s_down", "s_up"),
    )
    kpoints: Array = jnp.asarray(
        [[0.13, 0.21, 0.0], [-0.27, 0.08, 0.0], [0.19, -0.31, 0.0]],
        dtype=jnp.float64,
    )
    sigma_x: Array = jnp.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=jnp.complex128,
    )
    sigma_y: Array = jnp.asarray(
        [[0.0, 1.0j], [-1.0j, 0.0]],
        dtype=jnp.complex128,
    )
    direction: Array = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * kpoints[:, 1]),
            -jnp.sin(2.0 * jnp.pi * kpoints[:, 0]),
        ),
        axis=-1,
    )
    hamiltonians: Array = (
        direction[:, 0, None, None] * sigma_x
        + direction[:, 1, None, None] * sigma_y
    )
    values: Array
    columns: Array
    values, columns = jnp.linalg.eigh(hamiltonians)
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=values,
        eigenvectors=jnp.swapaxes(columns, -1, -2),
        kpoints=kpoints,
        geometry=make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        ),
        basis=basis,
    )
    return bands, direction


class TestSpinOperator:
    """Validate :func:`diffpes.tightb.spin_operator`."""

    @pytest.mark.parametrize(
        ("axis", "expected_block"),
        (
            (
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([[0.0, 0.5], [0.5, 0.0]]),
            ),
            (
                jnp.asarray([0.0, 1.0, 0.0]),
                jnp.asarray([[0.0, 0.5j], [-0.5j, 0.0]]),
            ),
            (
                jnp.asarray([0.0, 0.0, 1.0]),
                jnp.asarray([[-0.5, 0.0], [0.0, 0.5]]),
            ),
        ),
    )
    def test_pauli_blocks_follow_down_up_convention(
        self,
        axis: Array,
        expected_block: Array,
    ) -> None:
        """Match each physical Pauli matrix despite interleaved storage.

        Every Cartesian axis must act identically on both spatial partner
        pairs.

        Notes
        -----
        Extract each two-state block and compare it with a literal matrix.
        """
        operator: Array = spin_operator(_spin_basis(), axis)
        s_indices: Array = jnp.asarray([0, 2])
        p_indices: Array = jnp.asarray([3, 1])

        assert jnp.allclose(
            operator[jnp.ix_(s_indices, s_indices)],
            expected_block,
            rtol=0.0,
            atol=0.0,
        )
        assert jnp.allclose(
            operator[jnp.ix_(p_indices, p_indices)],
            expected_block,
            rtol=0.0,
            atol=0.0,
        )

    def test_arbitrary_unit_axis_has_half_spin_spectrum(self) -> None:
        """Give eigenvalues plus and minus one half for every partner pair.

        A normalized generic axis exercises all three Pauli components.

        Notes
        -----
        Check Hermiticity and the complete four-state spectrum directly.
        """
        axis: Array = jnp.asarray([1.0, -2.0, 2.0]) / 3.0
        operator: Array = spin_operator(_spin_basis(), axis)

        assert jnp.allclose(
            operator,
            operator.conj().T,
            rtol=0.0,
            atol=1e-15,
        )
        assert jnp.allclose(
            jnp.linalg.eigvalsh(operator),
            jnp.asarray([-0.5, -0.5, 0.5, 0.5]),
            rtol=0.0,
            atol=1e-14,
        )

    def test_rejects_spinless_or_incomplete_bases(self) -> None:
        """Require exactly one state in each spin sector per spatial orbital.

        Missing spin tags and mismatched spatial partners must fail separately.

        Notes
        -----
        Construct both malformed basis shapes and match their diagnostics.
        """
        spinless: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        incomplete: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 2),
            l=(0, 1),
            m=(0, 0),
            spin=(-1, 1),
        )

        with pytest.raises(ValueError, match="spinor basis"):
            spin_operator(spinless, jnp.asarray([0.0, 0.0, 1.0]))
        with pytest.raises(ValueError, match="exactly one"):
            spin_operator(incomplete, jnp.asarray([0.0, 0.0, 1.0]))

    def test_rejects_nonunit_axis_eager_and_compiled(self) -> None:
        """Reject a finite axis whose norm is not one.

        Runtime validation must survive both eager and compiled evaluation.

        Notes
        -----
        Use the shared rejection helper around a doubled-length x axis.
        """
        from tests._assertions import assert_rejects

        assert_rejects(
            spin_operator,
            _spin_basis(),
            jnp.asarray([2.0, 0.0, 0.0]),
            match="unit vector",
        )


class TestOrbitalProjector:
    """Validate :func:`diffpes.tightb.orbital_projector`."""

    def test_orbital_projector_is_diagonal_and_idempotent(self) -> None:
        """Select exactly the requested static basis indices.

        The diagonal mask must also satisfy the projector identity.

        Notes
        -----
        Compare the complete literal matrix and its square exactly.
        """
        projector: Array = orbital_projector(_spin_basis(), (0, 3))
        expected: Array = jnp.diag(
            jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
        )

        assert jnp.array_equal(projector, expected)
        assert jnp.array_equal(projector @ projector, projector)

    @pytest.mark.parametrize("selection", ((), (0, 0), (-1,), (4,)))
    def test_orbital_projector_rejects_invalid_selection(
        self,
        selection: tuple[int, ...],
    ) -> None:
        """Reject empty, duplicated, and out-of-range orbital selections.

        Every invalid static tuple must reach the same selection contract.

        Notes
        -----
        Parameterize all malformed tuple classes and match the routine name.
        """
        with pytest.raises(ValueError, match="orbital_select"):
            orbital_projector(_spin_basis(), selection)


class TestLsOperator:
    """Validate :func:`diffpes.tightb.ls_operator`."""

    def test_complete_p_shell_has_atomic_ls_multiplets(self) -> None:
        """Recover j=1/2 and j=3/2 L dot S eigenvalues.

        A complete real-cubic p shell must produce both analytic multiplets.

        Notes
        -----
        Check Hermiticity before comparing the sorted six-state spectrum.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,) * 6,
            n=(2,) * 6,
            l=(1,) * 6,
            m=(-1, 0, 1, -1, 0, 1),
            spin=(-1, -1, -1, 1, 1, 1),
            labels=("py", "pz", "px", "py", "pz", "px"),
        )
        operator: Array = ls_operator(basis, (0,) * 6)

        assert jnp.allclose(
            operator,
            operator.conj().T,
            rtol=0.0,
            atol=1e-14,
        )
        assert jnp.allclose(
            jnp.linalg.eigvalsh(operator),
            jnp.asarray([-1.0, -1.0, 0.5, 0.5, 0.5, 0.5]),
            rtol=0.0,
            atol=1e-13,
        )


class TestRashbaSpinTexture:
    """Validate spin texture against the analytic two-band Rashba model."""

    def test_texture_is_perpendicular_and_has_half_magnitude(self) -> None:
        """Match both helical branches away from Gamma.

        Generic momenta must produce opposite in-plane spin expectation
        values with zero out-of-plane component.

        Notes
        -----
        Compare all Cartesian expectations with the analytic Rashba direction.
        """
        bands: DiagonalizedBands
        direction: Array
        bands, direction = _rashba_bands()
        sx: Array = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([1.0, 0.0, 0.0])),
        )
        sy: Array = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([0.0, 1.0, 0.0])),
        )
        sz: Array = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([0.0, 0.0, 1.0])),
        )
        scale: Array = jnp.linalg.norm(direction, axis=-1)
        expected_upper: Array = 0.5 * direction / scale[:, None]
        expected: Array = jnp.stack((-expected_upper, expected_upper), axis=1)

        assert jnp.allclose(sx, expected[:, :, 0], rtol=0.0, atol=1e-10)
        assert jnp.allclose(sy, expected[:, :, 1], rtol=0.0, atol=1e-10)
        assert jnp.allclose(sz, 0.0, rtol=0.0, atol=1e-12)
        magnitude: Array = jnp.sqrt(sx * sx + sy * sy + sz * sz)
        assert jnp.allclose(magnitude, 0.5, rtol=0.0, atol=1e-10)


__all__: list[str] = []
