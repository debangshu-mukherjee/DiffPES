"""Validate the radial params contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import List
from jaxtyping import Array, Float64

from diffpes.types import (
    MatrixElementParams,
    OrbitalBasis,
    RadialSpec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_spec,
)
from tests._assertions import assert_rejects


def _complete_p_basis() -> OrbitalBasis:
    """PRIVATE: Create one complete real p shell.

    Returns
    -------
    basis : OrbitalBasis
        Three n=2, l=1 orbitals on one atom with m = -1, 0, 1, labeled
        ``py``, ``pz``, ``px``.

    Notes
    -----
    Provides the single complete shell that the shell-resolved radial
    and matrix-element carriers partition into one group.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
        labels=("py", "pz", "px"),
    )
    return basis


class TestRadialSpec(chex.TestCase):
    """Validate :class:`diffpes.types.RadialSpec`.

    The case distinguishes traced radial rows from static shell metadata in the
    PyTree.
    """

    def test_separates_traced_rows_from_static_shell_metadata(self) -> None:
        """Expose active numerical leaves and preserve one shell partition.

        The carrier holds one two-term contraction for a complete p shell.

        Notes
        -----
        Inspect the PyTree leaves and exact static metadata.
        """
        spec: RadialSpec = make_radial_spec(
            _complete_p_basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((0.8, 1.6),)),
            coefficients_shell=jnp.asarray(((0.6, -0.8),)),
        )
        leaves: List[Float64[Array, "..."]] = jax.tree.leaves(spec)
        assert spec.radial_shell_index == (0, 0, 0)
        assert spec.mode == "slater"
        assert any(leaf.shape == (1, 2) for leaf in leaves)


class TestMatrixElementParams(chex.TestCase):
    """Validate :class:`diffpes.types.MatrixElementParams`.

    The case compares shell-shared scales, channel phases, and static orbital
    mappings.
    """

    def test_preserves_shell_shared_scale_and_phases(self) -> None:
        """Store one scale and two phases for a complete p shell.

        The parameters remain numerical leaves while the partition is static.

        Notes
        -----
        Compare arrays and shell metadata exactly.
        """
        params: MatrixElementParams = make_matrix_element_params(
            _complete_p_basis(),
            (0, 0, 0),
            sigma_shell=jnp.asarray((1.2,)),
            phase_shift_angles_shell=jnp.asarray((0.1, -0.2)),
        )
        chex.assert_trees_all_close(params.sigma_shell, jnp.asarray((1.2,)))
        chex.assert_trees_all_close(
            params.phase_shift_angles_shell,
            jnp.asarray((0.1, -0.2)),
        )
        assert params.phase_channel_keys == ((0, 0), (0, 2))
        assert params.radial_shell_index == (0, 0, 0)
        leaves: List[Float64[Array, "..."]] = jax.tree.leaves(params)
        assert all(leaf.shape != (1, 2) for leaf in leaves)


class TestMakeRadialSpec(chex.TestCase):
    """Validate :func:`diffpes.types.make_radial_spec`.

    The cases check row normalization, compact grids, shell grouping, and
    certified tail constraints.
    """

    def test_normalizes_fixed_rows_and_compact_grid_rows(self) -> None:
        """Normalize phase-free fixed data and one compact sampled radial.

        The two modes exercise their distinct active storage contracts.

        Notes
        -----
        Compare Euclidean and radial-volume norms with unity.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        fixed: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray(((3.0, 4.0),)),
        )
        grid: Float64[Array, " 101"] = jnp.linspace(0.0, 10.0, 101)
        samples: Float64[Array, "1 101"] = (
            jnp.exp(-grid).at[-1].set(0.0)[None, :]
        )
        sampled: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="grid",
            r_grid=grid,
            grid_values_shell=samples,
        )
        assert fixed.fixed_integrals_shell is not None
        assert sampled.grid_values_shell is not None
        chex.assert_trees_all_close(
            jnp.linalg.norm(fixed.fixed_integrals_shell, axis=-1),
            jnp.ones((1,)),
        )
        grid_norm: Float64[Array, ""] = jnp.trapezoid(
            sampled.grid_values_shell[0] ** 2 * grid**2,
            x=grid,
        )
        chex.assert_trees_all_close(grid_norm, jnp.asarray(1.0))

    def test_rejects_shell_split_and_uncertified_tail_updates(self) -> None:
        """Reject per-m splitting and decay parameters outside their envelope.

        A complete p shell cannot acquire independent radial scalar rows.

        Notes
        -----
        Exercise static rejection directly and traced rejection through JIT.
        """
        with pytest.raises(ValueError, match="cannot be split"):
            make_radial_spec(_complete_p_basis(), (0, 1, 1))
        assert_rejects(
            make_radial_spec,
            _complete_p_basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((0.49,),)),
            match="certified tail envelope",
        )
        assert_rejects(
            make_radial_spec,
            _complete_p_basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((4.01,),)),
            match="certified tail envelope",
        )
        hydrogen_basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(2,),
            l=(1,),
            m=(0,),
        )
        charge: float
        for charge in (0.99, 8.01):
            assert_rejects(
                make_radial_spec,
                hydrogen_basis,
                (0,),
                mode="hydrogenic",
                effective_charge_shell=jnp.asarray((charge,)),
                match="certified tail envelope",
            )
        assert_rejects(
            make_radial_spec,
            _complete_p_basis(),
            (0, 0, 0),
            zeta_shell=jnp.asarray(((0.8, 0.801),)),
            coefficients_shell=jnp.asarray(((1.0, -0.999999),)),
            n_star_shell=(3.7,),
            match="coefficient condition",
        )

    def test_rejects_noncompact_or_nonuniform_grids(self) -> None:
        """Reject two finite grid inputs that violate compact-grid identity.

        One row has a nonzero endpoint and one grid has unequal spacing.

        Notes
        -----
        Match the compact-support and uniform-grid diagnostics separately.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        grid: Float64[Array, " 3"] = jnp.asarray((0.0, 1.0, 2.0))
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="compact-supported",
        ):
            make_radial_spec(
                basis,
                (0,),
                mode="grid",
                r_grid=grid,
                grid_values_shell=jnp.asarray(((1.0, 0.5, 0.1),)),
            )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="uniform grid",
        ):
            make_radial_spec(
                basis,
                (0,),
                mode="grid",
                r_grid=jnp.asarray((0.0, 1.0, 2.1)),
                grid_values_shell=jnp.asarray(((1.0, 0.5, 0.0),)),
            )


class TestMakeMatrixElementParams(chex.TestCase):
    """Validate :func:`diffpes.types.make_matrix_element_params`.

    The cases check physical s-shell channels and reject noncanonical radial
    phases.
    """

    def test_s_shell_exposes_only_its_physical_upper_channel(self) -> None:
        """Store only the s-to-p phase without a padded lower coordinate.

        The compact carrier excludes every nonexistent lower channel.

        Notes
        -----
        Inspect static keys and require the compact traced axis under JIT.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        params: MatrixElementParams = jax.jit(
            make_matrix_element_params,
            static_argnums=(0, 1),
        )(
            basis,
            (0,),
            phase_shift_angles_shell=jnp.asarray((0.2,)),
        )
        assert params.phase_channel_keys == ((0, 1),)
        chex.assert_trees_all_close(
            params.phase_shift_angles_shell,
            jnp.asarray((0.2,)),
        )
        with pytest.raises(ValueError, match="one entry per valid"):
            make_matrix_element_params(
                basis,
                (0,),
                phase_shift_angles_shell=jnp.asarray((0.1, 0.2)),
            )

    def test_raw_constructor_rejects_noncanonical_phase_keys(self) -> None:
        """Reject fabricated or reordered compact phase coordinates.

        The raw constructor enforces the factory's canonical static keys.

        Notes
        -----
        Change only the static key while retaining valid numerical axes.
        """
        basis: OrbitalBasis = _complete_p_basis()
        with pytest.raises(ValueError, match="canonical physical"):
            MatrixElementParams(
                sigma_shell=jnp.asarray((1.0,)),
                phase_shift_angles_shell=jnp.asarray((0.1, 0.2)),
                phase_channel_keys=((0, -1), (0, 2)),
                radial_shell_index=(0, 0, 0),
                basis=basis,
            )
