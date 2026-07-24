"""Validate orbital-basis and Slater-parameter carriers.

The tests cover static PyTree metadata, differentiable Slater leaves,
factory defaults, dtype normalization, and eager or compiled rejection.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from diffpes.types import (
    OrbitalBasis,
    SlaterParams,
    make_orbital_basis,
    make_slater_params,
)
from diffpes.types.radial_params import (
    SlaterKosterParams,
    make_slater_koster_params,
)
from tests._assertions import assert_rejects


def _basis() -> OrbitalBasis:
    """Create a two-orbital, two-atom spinless test basis."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 2),
        l=(0, 1),
        m=(0, 0),
        labels=("1s", "2pz"),
    )
    return basis


class TestOrbitalBasis(chex.TestCase):
    """Validate :class:`~diffpes.types.OrbitalBasis`."""

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
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(basis)
        restored: OrbitalBasis = jax.tree_util.tree_unflatten(tree, leaves)

        assert leaves == []
        assert restored.atom_indices == basis.atom_indices
        assert restored.n == basis.n
        assert restored.l == basis.l
        assert restored.m == basis.m
        assert restored.spin == basis.spin
        assert restored.labels == basis.labels


class TestSlaterParams(chex.TestCase):
    """Validate :class:`~diffpes.types.SlaterParams`."""

    def test_zeta_gradient(self) -> None:
        """Differentiate a quadratic loss through Slater exponents.

        The case traces one positive exponent through a scalar square loss.

        Notes
        -----
        Compare the resulting derivative with the analytic value four.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([2.0], dtype=jnp.float64),
            orbital_basis=basis,
        )

        def loss(candidate: SlaterParams) -> jax.Array:
            """Return the quadratic Slater-exponent loss."""
            result: jax.Array = jnp.sum(candidate.zeta**2)
            return result

        gradient: SlaterParams = jax.grad(loss)(params)

        chex.assert_trees_all_close(gradient.zeta, 4.0)


class TestSlaterKosterParams(chex.TestCase):
    """Validate :class:`~diffpes.types.SlaterKosterParams`."""

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
        leaves: list[jax.Array] = jax.tree.leaves(params)

        def loss(candidate: SlaterKosterParams) -> jax.Array:
            """Return a weighted quadratic parameter loss."""
            result: jax.Array = jnp.sum(
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
        Route factory failures through the shared eager/compiled gate.
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
    """Validate :func:`~diffpes.types.make_slater_koster_params`."""

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
    """Validate :func:`~diffpes.types.make_orbital_basis`."""

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
        Route every case through the shared eager and compiled rejection gate.
        """
        arguments: dict[str, object] = {
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

        The case supplies an invalid spin channel to the raw module constructor.

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


class TestMakeSlaterParams(chex.TestCase):
    """Validate :func:`~diffpes.types.make_slater_params`."""

    def test_supplies_unit_coefficients(self) -> None:
        """Create one unit coefficient per orbital when omitted.

        The case constructs two single-zeta orbitals without coefficients.

        Notes
        -----
        Check the generated column shape, dtype, and unit values.
        """
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([1.0, 1.5], dtype=jnp.float64),
            orbital_basis=_basis(),
        )

        chex.assert_shape(params.coefficients, (2, 1))
        assert params.coefficients.dtype == jnp.float64
        chex.assert_trees_all_close(params.coefficients, jnp.ones((2, 1)))

    def test_casts_explicit_coefficients_to_float64(self) -> None:
        """Promote explicit float32 coefficients to project precision.

        The case supplies a two-column contraction in lower precision.

        Notes
        -----
        Check the preserved shape and required float64 output dtype.
        """
        coefficients: jax.Array = jnp.array(
            [[0.8, 0.2], [0.6, 0.4]],
            dtype=jnp.float32,
        )
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([1.0, 1.5], dtype=jnp.float64),
            orbital_basis=_basis(),
            coefficients=coefficients,
        )

        chex.assert_shape(params.coefficients, (2, 2))
        assert params.coefficients.dtype == jnp.float64

    def test_rejects_invalid_radial_arrays_eager_and_jit(self) -> None:
        """Reject nonpositive exponents and coefficient-axis mismatches.

        The cases exercise traced value checks and static leading-axis checks.

        Notes
        -----
        Use the shared rejection helper and match each diagnostic fragment.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(1,),
            l=(0,),
            m=(0,),
        )
        assert_rejects(
            make_slater_params,
            zeta=jnp.array([0.0], dtype=jnp.float64),
            orbital_basis=basis,
            match="zeta positive",
        )
        assert_rejects(
            make_slater_params,
            zeta=jnp.array([1.0], dtype=jnp.float64),
            orbital_basis=basis,
            coefficients=jnp.ones((2, 1), dtype=jnp.float64),
            match="coefficients first dimension must match",
        )
