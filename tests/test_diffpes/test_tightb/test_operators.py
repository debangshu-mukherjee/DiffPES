r"""Validate Hermitian tight-binding observable builders.

The tests pin the down--up Pauli convention, arbitrary basis ordering,
orbital projectors, unit-strength atomic L dot S, and analytic Rashba spin
texture.
"""

import jax.numpy as jnp
import pytest
from beartype.typing import Tuple, Union
from jaxtyping import Array, Complex128, Float64, Int64

from diffpes.tightb import (
    expectation_path,
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
from tests._assertions import assert_rejects


def _spin_basis() -> OrbitalBasis:
    """PRIVATE: Build two spatial orbitals in an interleaved spin order.

    Returns
    -------
    basis : OrbitalBasis
        Four-orbital basis whose spin labels run ``(-1, 1, 1, -1)``, so
        neither spin sector occupies one contiguous block.

    Notes
    -----
    The deliberate interleaving forces the operator builders to place
    Pauli blocks by the per-orbital spin metadata rather than by an
    assumed block layout.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 1, 2),
        l=(0, 1, 0, 1),
        m=(0, 0, 0, 0),
        spin=(-1, 1, 1, -1),
        labels=("s_down", "p_up", "s_up", "p_down"),
    )
    return basis


def _rashba_bands() -> Tuple[DiagonalizedBands, Float64[Array, "3 2"]]:
    """PRIVATE: Diagonalize analytic two-band Rashba matrices away from Gamma.

    Returns
    -------
    bands_and_direction : Tuple[DiagonalizedBands, Float64[Array, "3 2"]]
        The diagonalized two-band eigensystem at three generic in-plane
        k-points and the in-plane Rashba field direction
        ``(sin 2*pi*k_y, -sin 2*pi*k_x)`` at each point.

    Notes
    -----
    Builds ``H(k) = d_x(k) sigma_x + d_y(k) sigma_y`` in the down--up
    basis, diagonalizes with ``jnp.linalg.eigh``, and swaps the column
    eigenvectors into the band-major carrier layout. The returned field
    direction is the analytic truth for the in-plane spin texture of
    the two chiral bands.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        spin=(-1, 1),
        labels=("s_down", "s_up"),
    )
    kpoints: Float64[Array, "3 3"] = jnp.asarray(
        [[0.13, 0.21, 0.0], [-0.27, 0.08, 0.0], [0.19, -0.31, 0.0]],
        dtype=jnp.float64,
    )
    sigma_x: Complex128[Array, "2 2"] = jnp.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=jnp.complex128,
    )
    sigma_y: Complex128[Array, "2 2"] = jnp.asarray(
        [[0.0, 1.0j], [-1.0j, 0.0]],
        dtype=jnp.complex128,
    )
    direction: Float64[Array, "3 2"] = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * kpoints[:, 1]),
            -jnp.sin(2.0 * jnp.pi * kpoints[:, 0]),
        ),
        axis=-1,
    )
    hamiltonians: Complex128[Array, "3 2 2"] = (
        direction[:, 0, None, None] * sigma_x
        + direction[:, 1, None, None] * sigma_y
    )
    values: Float64[Array, "3 2"]
    columns: Complex128[Array, "3 2 2"]
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
    result: Tuple[DiagonalizedBands, Float64[Array, "3 2"]] = (
        bands,
        direction,
    )
    return result


class TestSpinOperator:
    """Validate :func:`diffpes.tightb.spin_operator`.

    The cases check basis order, arbitrary axes, spin spectra, and eager or
    compiled rejection.
    """

    @pytest.mark.parametrize(
        ("axis", "expected_block"),
        [
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
        ],
    )
    def test_pauli_blocks_follow_down_up_convention(
        self,
        axis: Float64[Array, " 3"],
        expected_block: Union[Float64[Array, "2 2"], Complex128[Array, "2 2"]],
    ) -> None:
        """Match each physical Pauli matrix despite interleaved storage.

        Every Cartesian axis must act identically on both spatial partner
        pairs.

        Notes
        -----
        Extract each two-state block and compare it with a literal matrix.
        """
        operator: Complex128[Array, "4 4"] = spin_operator(_spin_basis(), axis)
        s_indices: Int64[Array, " 2"] = jnp.asarray([0, 2])
        p_indices: Int64[Array, " 2"] = jnp.asarray([3, 1])

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
        axis: Float64[Array, " 3"] = jnp.asarray([1.0, -2.0, 2.0]) / 3.0
        operator: Complex128[Array, "4 4"] = spin_operator(_spin_basis(), axis)

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
        assert_rejects(
            spin_operator,
            _spin_basis(),
            jnp.asarray([2.0, 0.0, 0.0]),
            match="unit vector",
        )


class TestOrbitalProjector:
    """Validate :func:`diffpes.tightb.orbital_projector`.

    The cases check diagonal idempotence and reject invalid orbital selections.
    """

    def test_orbital_projector_is_diagonal_and_idempotent(self) -> None:
        """Select exactly the requested static basis indices.

        The diagonal mask must also satisfy the projector identity.

        Notes
        -----
        Compare the complete literal matrix and its square exactly.
        """
        projector: Complex128[Array, "4 4"] = orbital_projector(
            _spin_basis(), (0, 3)
        )
        expected: Complex128[Array, "4 4"] = jnp.diag(
            jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
        )

        assert jnp.array_equal(projector, expected)
        assert jnp.array_equal(projector @ projector, projector)

    @pytest.mark.parametrize("selection", [(), (0, 0), (-1,), (4,)])
    def test_orbital_projector_rejects_invalid_selection(
        self,
        selection: Tuple[int, ...],
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
    """Validate :func:`diffpes.tightb.ls_operator`.

    The case compares a complete p-shell operator with the analytic atomic
    multiplets.
    """

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
        operator: Complex128[Array, "6 6"] = ls_operator(basis, (0,) * 6)

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
    """Validate spin texture against the analytic two-band Rashba model.

    The case checks the analytic perpendicular direction and half-spin
    magnitude of the Rashba texture.
    """

    def test_texture_is_perpendicular_and_has_half_magnitude(self) -> None:
        """Match both helical branches away from Gamma.

        Generic momenta must produce opposite in-plane spin expectation
        values with zero out-of-plane component.

        Notes
        -----
        Compare all Cartesian expectations with the analytic Rashba direction.
        """
        bands: DiagonalizedBands
        direction: Float64[Array, "3 2"]
        bands, direction = _rashba_bands()
        sx: Float64[Array, "3 2"] = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([1.0, 0.0, 0.0])),
        )
        sy: Float64[Array, "3 2"] = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([0.0, 1.0, 0.0])),
        )
        sz: Float64[Array, "3 2"] = expectation_path(
            bands,
            spin_operator(bands.basis, jnp.asarray([0.0, 0.0, 1.0])),
        )
        scale: Float64[Array, " 3"] = jnp.linalg.norm(direction, axis=-1)
        expected_upper: Float64[Array, "3 2"] = (
            0.5 * direction / scale[:, None]
        )
        expected: Float64[Array, "3 2 2"] = jnp.stack(
            (-expected_upper, expected_upper), axis=1
        )

        assert jnp.allclose(sx, expected[:, :, 0], rtol=0.0, atol=1e-10)
        assert jnp.allclose(sy, expected[:, :, 1], rtol=0.0, atol=1e-10)
        assert jnp.allclose(sz, 0.0, rtol=0.0, atol=1e-12)
        magnitude: Float64[Array, "3 2"] = jnp.sqrt(
            sx * sx + sy * sy + sz * sz
        )
        assert jnp.allclose(magnitude, 0.5, rtol=0.0, atol=1e-10)


class TestSurfaceProjector:
    """Mirror coverage for :func:`diffpes.tightb.surface_projector`.

    The detailed surface cases check projector support, idempotence, and
    slab-layer selection.
    """


class TestLayerResolvedWeights:
    """Mirror coverage for :func:`diffpes.tightb.layer_resolved_weights`.

    The detailed surface cases check normalized weights for each registered
    slab layer.
    """


class TestLayerResolvedGroupTraces:
    """Mirror coverage for fixed-group surface traces.

    :see: :func:`diffpes.tightb.layer_resolved_group_traces`
    """
