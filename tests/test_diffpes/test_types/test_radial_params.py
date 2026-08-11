"""Validate orbital-basis and Slater-parameter carriers.

The tests cover static PyTree metadata, differentiable Slater leaves,
factory defaults, dtype normalization, and eager or compiled rejection.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Dict, List
from jaxtyping import Array, Float64

from diffpes.types import (
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    PyTreeDef,
    RadialQuadratureSpec,
    RadialSpec,
    SlaterKosterParams,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_slater_koster_params,
)
from tests._assertions import assert_rejects


def _basis() -> OrbitalBasis:
    """PRIVATE: Create a two-orbital, two-atom spinless test basis.

    Returns
    -------
    basis : OrbitalBasis
        A 1s orbital on atom 0 and a 2pz orbital on atom 1, with no
        spin channel.

    Notes
    -----
    Uses the public factory so the radial-parameter carriers under
    test receive a validated basis with two distinct (n, l) shells.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 2),
        l=(0, 1),
        m=(0, 0),
        labels=("1s", "2pz"),
    )
    return basis


class TestOrbitalBasis(chex.TestCase):
    """Validate :class:`~diffpes.types.OrbitalBasis`.

    The case round-trips the PyTree and compares every static orbital field.
    """

    def test_pytree_round_trip_preserves_all_static_fields(self) -> None:
        """Preserve atom, quantum-number, spin, and label tuples exactly.

        The case flattens and rebuilds a spinful two-atom orbital basis.

        Notes
        -----
        Compare every static tuple after reconstruction and require no leaves.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0, 1, 1),
            n=(2, 2, 2, 2),
            l=(1, 1, 1, 1),
            m=(-1, 0, -1, 0),
            spin=(1, 1, -1, -1),
            labels=("a_px_up", "a_pz_up", "b_px_dn", "b_pz_dn"),
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(basis)
        restored: OrbitalBasis = jax.tree_util.tree_unflatten(tree, leaves)

        assert leaves == []
        assert restored.atom_indices == basis.atom_indices
        assert restored.n == basis.n
        assert restored.l == basis.l
        assert restored.m == basis.m
        assert restored.spin == basis.spin
        assert restored.labels == basis.labels


class TestSlaterKosterParams(chex.TestCase):
    """Validate :class:`~diffpes.types.SlaterKosterParams`.

    The cases inspect PyTree leaves and exercise eager and compiled validation
    for keys and values.
    """

    def test_values_are_the_only_differentiable_leaf(self) -> None:
        """Keep material keys static while differentiating every value.

        The case constructs two carbon hopping channels and differentiates a
        weighted quadratic loss.

        Notes
        -----
        Require one float64 leaf and compare its gradient analytically.
        """
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-2.7, 0.8), dtype=jnp.float32),
            ("C-C:pp_pi", "C-C:pp_sigma"),
        )
        leaves: List[Float64[Array, "..."]] = jax.tree.leaves(params)

        def loss(candidate: SlaterKosterParams) -> Float64[Array, ""]:
            """Return a weighted quadratic parameter loss."""
            result: Float64[Array, ""] = jnp.sum(
                jnp.asarray((1.0, 2.0)) * candidate.values**2
            )
            return result

        gradient: SlaterKosterParams = jax.grad(loss)(params)

        assert len(leaves) == 1
        assert leaves[0].dtype == jnp.float64
        assert params.keys == ("C-C:pp_pi", "C-C:pp_sigma")
        chex.assert_trees_all_close(
            gradient.values,
            jnp.asarray((-5.4, 3.2)),
        )

    def test_rejects_invalid_keys_and_values_eager_and_jit(self) -> None:
        """Reject duplicate keys, length mismatches, and non-finite values.

        The cases isolate static carrier defects and one traced numerical
        defect.

        Notes
        -----
        Route factory failures through the shared eager and compiled check.
        """
        assert_rejects(
            make_slater_koster_params,
            jnp.ones((2,), dtype=jnp.float64),
            ("X-X:ss_sigma",),
            match="same length",
        )
        assert_rejects(
            make_slater_koster_params,
            jnp.ones((2,), dtype=jnp.float64),
            ("X-X:ss_sigma", "X-X:ss_sigma"),
            match="must be unique",
        )
        assert_rejects(
            make_slater_koster_params,
            jnp.asarray((jnp.nan,), dtype=jnp.float64),
            ("X-X:ss_sigma",),
            match="values finite",
        )


class TestMakeSlaterKosterParams(chex.TestCase):
    """Validate :func:`~diffpes.types.make_slater_koster_params`.

    The case checks value normalization and exact preservation of
    Slater--Koster keys.
    """

    def test_normalizes_values_and_preserves_keys(self) -> None:
        """Normalize input values while preserving static channel identifiers.

        The case separates the factory contract from the carrier leaf test.

        Notes
        -----
        Require float64 values and exact static keys after construction.
        """
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-1.0, 2.0), dtype=jnp.float32),
            ("X-X:ss_sigma", "X-X:pp_pi"),
        )

        assert params.values.dtype == jnp.float64
        assert params.keys == ("X-X:ss_sigma", "X-X:pp_pi")


class TestMakeOrbitalBasis(chex.TestCase):
    """Validate :func:`~diffpes.types.make_orbital_basis`.

    The cases check label generation, spin defaults, static rejection, and
    direct-constructor invariants.
    """

    def test_generates_labels_and_spinless_default(self) -> None:
        """Generate stable labels and an empty spin tuple by default.

        The case omits optional metadata for a two-orbital basis.

        Notes
        -----
        Compare generated labels by position and require an empty spin tuple.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(1, 2),
            l=(0, 1),
            m=(0, 0),
        )

        assert basis.labels == ("orb_0", "orb_1")
        assert basis.spin == ()

    @parameterized.named_parameters(
        (
            "length_mismatch",
            "length",
            "must have the same length",
        ),
        (
            "negative_atom_index",
            "atom",
            "atom_indices must contain non-negative integers",
        ),
        (
            "invalid_principal_quantum_number",
            "principal",
            "n must contain integers of at least 1",
        ),
        (
            "invalid_angular_quantum_number",
            "angular",
            "l must contain integers satisfying",
        ),
        (
            "invalid_spin_length",
            "spin_length",
            "spin must be empty or have one entry per orbital",
        ),
        (
            "invalid_spin_channel",
            "spin_channel",
            r"spin entries must be \+1 or -1",
        ),
    )
    def test_rejects_invalid_static_metadata_eager_and_jit(
        self,
        defect: str,
        match: str,
    ) -> None:
        """Reject malformed atom, quantum-number, and spin tuples.

        Parameterized cases isolate one structural metadata defect at a time.

        Notes
        -----
        Route every case through the shared eager and compiled rejection check.
        """
        arguments: Dict[str, object] = {
            "atom_indices": (0,),
            "n": (1,),
            "l": (0,),
            "m": (0,),
            "spin": (),
        }
        if defect == "length":
            arguments["m"] = (0, 0)
        elif defect == "atom":
            arguments["atom_indices"] = (-1,)
        elif defect == "principal":
            arguments["n"] = (0,)
        elif defect == "angular":
            arguments["l"] = (1,)
        elif defect == "spin_length":
            arguments["atom_indices"] = (0, 0)
            arguments["n"] = (1, 1)
            arguments["l"] = (0, 0)
            arguments["m"] = (0, 0)
            arguments["spin"] = (1,)
        else:
            arguments["spin"] = (0,)

        assert_rejects(make_orbital_basis, match=match, **arguments)

    def test_raw_constructor_reasserts_static_invariants(self) -> None:
        """Prevent direct construction from bypassing spin validation.

        The case supplies an invalid spin channel to the raw module
        constructor.

        Notes
        -----
        Require the same validation error that the public factory emits.
        """
        with pytest.raises(ValueError, match="spin entries"):
            OrbitalBasis(
                atom_indices=(0,),
                n=(1,),
                l=(0,),
                m=(0,),
                spin=(0,),
                labels=("s",),
            )


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


class TestRadialQuadratureSpec(chex.TestCase):
    """Validate :class:`diffpes.types.RadialQuadratureSpec`.

    The case rejects a direct construction that asserts an uncertified
    quadrature tolerance.
    """

    def test_raw_constructor_rejects_self_asserted_tolerance(self) -> None:
        """Reject a profile whose claimed tolerance differs from its identity.

        The false control changes only ``value_rtol`` on the initial profile.

        Notes
        -----
        Require exact registry-property matching in the raw constructor.
        """
        with pytest.raises(ValueError, match="certified profile"):
            RadialQuadratureSpec(
                profile_id="gl1024-r120-k4-l9-v1",
                n_nodes=1024,
                r_max_bohr=120.0,
                k_max_bohr_inv=4.0,
                l_prime_max=9,
                value_rtol=1.0e-14,
                gradient_rtol=1.0e-8,
                tail_bound_method_id="analytic-exp-r120-or-compact-v1",
                coefficient_condition_max=32.0,
                min_decay_parameter=0.5,
                max_decay_parameter=4.0,
            )


class TestFinalStateSpec(chex.TestCase):
    """Validate :class:`diffpes.types.FinalStateSpec`.

    The case separates the traced Coulomb charge from the static final-state
    mode.
    """

    def test_keeps_charge_traced_and_mode_static(self) -> None:
        """Preserve a Coulomb charge as the carrier's only numerical leaf.

        The direct Coulomb selector stays static.

        Notes
        -----
        Flatten the carrier and inspect its exact fields.
        """
        spec: FinalStateSpec = make_final_state_spec(
            mode="coulomb",
            effective_charge=1.5,
        )
        leaves: List[Float64[Array, "..."]] = jax.tree.leaves(spec)
        assert len(leaves) == 1
        chex.assert_trees_all_close(leaves[0], jnp.asarray(1.5))
        assert spec.mode == "coulomb"


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


class TestMakeRadialQuadratureSpec(chex.TestCase):
    """Validate :func:`diffpes.types.make_radial_quadrature_spec`.

    The case selects both certified profiles and rejects an unknown profile
    identity.
    """

    def test_selects_both_profiles_and_rejects_unknown_identity(self) -> None:
        """Resolve the production and reference profiles without overrides.

        An invented identifier provides the false control.

        Notes
        -----
        Compare node counts and require explicit unknown-profile rejection.
        """
        production: RadialQuadratureSpec = make_radial_quadrature_spec()
        reference: RadialQuadratureSpec = make_radial_quadrature_spec(
            "gl2048-r120-k4-l9-reference-v1"
        )
        assert production.n_nodes == 1024
        assert reference.n_nodes == 2048
        assert production.coefficient_condition_max == 32.0
        assert production.min_decay_parameter == 0.5
        assert production.max_decay_parameter == 4.0
        with pytest.raises(ValueError, match="unknown certified"):
            make_radial_quadrature_spec("gl128-unverified")


class TestMakeFinalStateSpec(chex.TestCase):
    """Validate :func:`diffpes.types.make_final_state_spec`.

    The case rejects an incompatible plane-wave charge and an uncertified
    radial accelerator.
    """

    def test_rejects_plane_wave_charge_and_uncertified_acceleration(
        self,
    ) -> None:
        """Reject incompatible numerical and static final-state selections.

        The cases prevent a charged plane wave and tabulated Coulomb radial.

        Notes
        -----
        Exercise the traced charge check and the eager mode check.
        """
        assert_rejects(
            make_final_state_spec,
            effective_charge=jnp.asarray(0.1),
            match="require zero effective charge",
        )
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(radial_accelerator="hermite")
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            FinalStateSpec(
                effective_charge=jnp.asarray(0.0),
                mode="plane_wave",
                radial_accelerator="hermite",
                table_n_points=1025,
            )
        with pytest.raises(
            ValueError, match="failed the frozen radial accelerator"
        ):
            make_final_state_spec(
                mode="coulomb",
                effective_charge=1.0,
                radial_accelerator="hermite",
            )
