"""Validate exact surface cells and whole-model orbital rotations.

The tests exercise slab numerical and structural contracts.
"""

import math

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any
from jaxtyping import Array, Int
from numpy.typing import NDArray

from diffpes.maths import rodrigues_rotation
from diffpes.tightb.hamiltonian import bloch_hamiltonian
from diffpes.tightb.slab import (
    find_surface_cell,
    freeze_slab_topology,
    rebuild_slab,
    rotate_tb_model,
)
from diffpes.tightb.soc import spin_double_model
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)


def _geometry(lattice: Array | None = None) -> CrystalGeometry:
    """Build a one-site primitive geometry."""
    resolved_lattice: Array = (
        jnp.eye(3, dtype=jnp.float64) if lattice is None else lattice
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=resolved_lattice,
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    return geometry


def _complete_p_model() -> TBModel:
    """Build a generic complex complete-shell model."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
    )
    forward: Array = jnp.asarray(
        (
            (0.3 + 0.1j, 0.2 + 0.4j, -0.1 + 0.05j),
            (0.7 - 0.2j, -0.4 + 0.3j, 0.6 + 0.1j),
            (-0.2 + 0.5j, 0.1 - 0.4j, 0.8 + 0.2j),
        ),
        dtype=jnp.complex128,
    )
    hopping_pairs: tuple[tuple[int, int], ...] = tuple(
        (row, column)
        for _cell in range(2)
        for row in range(3)
        for column in range(3)
    )
    hopping_cells: tuple[tuple[int, int, int], ...] = tuple(
        (1, 0, 0) if cell == 0 else (-1, 0, 0)
        for cell in range(2)
        for _row in range(3)
        for _column in range(3)
    )
    amplitudes: Array = jnp.concatenate(
        (forward.reshape(-1), forward.conj().T.reshape(-1))
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=amplitudes,
        onsite_energies=jnp.asarray((0.2, -0.3, 0.7)),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=_geometry(
            jnp.asarray(
                (
                    (2.0, 0.1, 0.2),
                    (0.2, 1.7, 0.1),
                    (0.1, 0.3, 2.1),
                ),
                dtype=jnp.float64,
            )
        ),
        basis=basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=(-1, -1, -1),
    )
    return model


def _z_chain_model(lattice_scale: Array | float) -> TBModel:
    """Build a one-orbital chain with a differentiable z spacing."""
    scale: Array = jnp.asarray(lattice_scale, dtype=jnp.float64)
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.diag(jnp.stack((jnp.ones_like(scale),) * 2 + (scale,))),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray((-0.7, -0.7), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )
    return model


class TestFindSurfaceCell:
    """Validate exact Miller-index topology and traced geometry.

    :see: :func:`~diffpes.tightb.find_surface_cell`
    """

    @pytest.mark.parametrize(
        ("miller", "expected_spacing"),
        (
            ((0, 0, 1), 1.0),
            ((1, 1, 0), 1.0 / math.sqrt(2.0)),
            ((1, 1, 1), 1.0 / math.sqrt(3.0)),
        ),
    )
    def test_cubic_external_truth(
        self,
        miller: tuple[int, int, int],
        expected_spacing: float,
    ) -> None:
        """Match cubic interplanar spacing and exact integer identities.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        cell: Any
        cell = find_surface_cell(_geometry(), miller)
        miller_array: Int[NDArray, " 3"] = np.asarray(miller, dtype=np.int64)
        coefficients: Int[NDArray, "3 3"] = np.asarray(
            (*cell.in_plane_coeffs, cell.stacking_coeffs),
            dtype=np.int64,
        )

        assert np.array_equal(coefficients[:2] @ miller_array, (0, 0))
        assert int(coefficients[2] @ miller_array) == 1
        assert abs(round(np.linalg.det(coefficients))) == 1
        assert float(cell.interlayer_spacing_ang) == pytest.approx(
            expected_spacing,
            rel=1e-12,
        )
        assert jnp.allclose(
            cell.rotation @ cell.rotation.T,
            jnp.eye(3),
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.linalg.det(cell.rotation) == pytest.approx(
            1.0,
            rel=1e-12,
        )
        assert jnp.allclose(
            cell.in_plane_vectors[:, 2],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_hexagonal_001_preserves_basal_metric(self) -> None:
        """Keep a primitive basal cell for a hexagonal (001) surface.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        cell: Any
        lattice: Array = jnp.asarray(
            (
                (2.5, 0.0, 0.0),
                (-1.25, 1.25 * math.sqrt(3.0), 0.0),
                (0.0, 0.0, 4.2),
            ),
            dtype=jnp.float64,
        )
        cell = find_surface_cell(_geometry(lattice), (0, 0, 1))
        metric: Array = cell.in_plane_vectors @ cell.in_plane_vectors.T

        assert jnp.diag(metric) == pytest.approx((6.25, 6.25), rel=1e-12)
        assert abs(float(metric[0, 1])) == pytest.approx(3.125, rel=1e-12)
        assert float(cell.interlayer_spacing_ang) == pytest.approx(
            4.2,
            rel=1e-12,
        )

    @pytest.mark.parametrize(
        "miller",
        ((0, 0, 0), (2, 2, 0)),
    )
    def test_rejects_invalid_miller(
        self,
        miller: tuple[int, int, int],
    ) -> None:
        """Reject zero and nonprimitive integer Miller tuples.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        with pytest.raises(ValueError, match="miller"):
            find_surface_cell(_geometry(), miller)


class TestRotateTbModel:
    """Validate complete-shell covariance and incomplete-shell guards.

    :see: :func:`~diffpes.tightb.rotate_tb_model`
    """

    def test_generic_complex_spectrum_is_invariant(self) -> None:
        """Preserve sorted energies after full translation-block conjugation.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        model: TBModel = _complete_p_model()
        rotation: Array = rodrigues_rotation(
            jnp.asarray((0.2, 0.5, 0.7), dtype=jnp.float64),
            0.63,
        )
        rotated: TBModel = rotate_tb_model(model, rotation)
        kpoint: Array = jnp.asarray((0.13, -0.27, 0.31))
        expected: Array = jnp.linalg.eigvalsh(bloch_hamiltonian(model, kpoint))
        actual: Array = jnp.linalg.eigvalsh(bloch_hamiltonian(rotated, kpoint))

        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(
            rotated.geometry.lattice,
            model.geometry.lattice @ rotation.T,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_complete_d_shell_soc_spectrum_is_invariant(self) -> None:
        """Rotate orbital and spin frames together without changing L dot S.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,) * 5,
            n=(3,) * 5,
            l=(2,) * 5,
            m=(-2, -1, 0, 1, 2),
        )
        spinless: TBModel = make_tb_model(
            hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
            onsite_energies=jnp.asarray((0.1, 0.2, 0.3, 0.4, 0.5)),
            soc_lambdas=jnp.asarray((0.7,)),
            geometry=_geometry(),
            basis=basis,
            hopping_pairs=(),
            hopping_cells=(),
            shell_index=(0,) * 5,
        )
        model: TBModel = spin_double_model(spinless)
        rotation: Array = rodrigues_rotation(
            jnp.asarray((0.3, -0.4, 0.8)),
            0.71,
        )
        rotated: TBModel = rotate_tb_model(model, rotation)
        kpoint: Array = jnp.asarray((0.17, -0.09, 0.23))

        assert jnp.allclose(
            jnp.linalg.eigvalsh(bloch_hamiltonian(rotated, kpoint)),
            jnp.linalg.eigvalsh(bloch_hamiltonian(model, kpoint)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_incomplete_shell_identity_is_exact_noop(self) -> None:
        """Allow only the exact identity path for an incomplete p shell.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(2, 2),
            l=(1, 1),
            m=(-1, 1),
        )
        model: TBModel = make_tb_model(
            hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
            onsite_energies=jnp.asarray((0.1, 0.2)),
            soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
            geometry=_geometry(),
            basis=basis,
            hopping_pairs=(),
            hopping_cells=(),
            shell_index=(-1, -1),
        )

        assert rotate_tb_model(model, jnp.eye(3)) is model
        with pytest.raises(ValueError, match=r"missing m=.*0"):
            rotate_tb_model(
                model,
                rodrigues_rotation(
                    jnp.asarray((1.0, 0.0, 0.0)),
                    0.3,
                ),
            )


class TestFreezeSlabTopology:
    """Validate the eager, host-only topology-selection stage.

    :see: :func:`~diffpes.tightb.freeze_slab_topology`
    """

    def test_natural_fine_preserves_requested_minimum_span(self) -> None:
        """Expand before trimming so the post-fine slab keeps its minimum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        topology: Any
        topology = freeze_slab_topology(
            _z_chain_model(1.0),
            miller=(0, 0, 1),
            thickness_ang=4.0,
            vacuum_ang=3.0,
            fine=(0.6, 0.6),
        )

        assert topology.n_layers == 5
        assert topology.fine == (0.6, 0.6)


class TestRebuildSlab(chex.TestCase):
    """Validate the pure-JAX reconstruction side of the topology airlock.

    :see: :func:`~diffpes.tightb.rebuild_slab`
    """

    def test_depth_gradient_uses_frozen_topology(self) -> None:
        """Match the analytic layer-depth derivative through rebuild_slab.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        topology: Any
        topology = freeze_slab_topology(
            _z_chain_model(1.0),
            miller=(0, 0, 1),
            thickness_ang=4.0,
            vacuum_ang=3.0,
        )

        def depth_sum(scale: Array) -> Array:
            slab: Any
            slab, _ = rebuild_slab(_z_chain_model(scale), topology)
            assert slab.depths is not None
            return jnp.sum(slab.depths)

        derivative: Array = jax.grad(depth_sum)(jnp.asarray(1.0))
        finite_difference: Array = (
            depth_sum(jnp.asarray(1.0 + 1e-5))
            - depth_sum(jnp.asarray(1.0 - 1e-5))
        ) / 2e-5

        assert derivative == pytest.approx(10.0, rel=1e-12)
        assert derivative == pytest.approx(
            float(finite_difference),
            rel=1e-9,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_variants_preserve_frozen_topology_values(self) -> None:
        """Match slab depths under eager and compiled continuous rebuilding.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Any
        rebuild: Any
        topology: Any
        topology = freeze_slab_topology(
            _z_chain_model(1.0),
            miller=(0, 0, 1),
            thickness_ang=4.0,
            vacuum_ang=3.0,
            fine=(0.6, 0.6),
        )
        rebuild = self.variant(
            lambda scale: (
                rebuild_slab(_z_chain_model(scale), topology)[0].depths
            )
        )
        depths = rebuild(jnp.asarray(1.0))

        assert jnp.array_equal(
            depths,
            jnp.asarray((4.0, 3.0, 2.0, 1.0, 0.0)),
        )

    def test_vmap_preserves_frozen_topology_shape(self) -> None:
        """Batch continuous lattice scales without reselecting topology.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        topology: Any
        topology = freeze_slab_topology(
            _z_chain_model(1.0),
            miller=(0, 0, 1),
            thickness_ang=4.0,
            vacuum_ang=3.0,
        )

        def depths(scale: Array) -> Array:
            slab: Any
            slab, _ = rebuild_slab(_z_chain_model(scale), topology)
            assert slab.depths is not None
            return slab.depths

        batched: Array = jax.vmap(depths)(
            jnp.asarray((0.9, 1.0, 1.1), dtype=jnp.float64)
        )

        assert batched.shape == (3, 5)
        assert jnp.allclose(
            batched[1],
            jnp.asarray((4.0, 3.0, 2.0, 1.0, 0.0)),
        )
