"""Validate the slater koster params contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import List
from jaxtyping import Array, Float64

from diffpes.types import (
    SlaterKosterParams,
    make_slater_koster_params,
)
from tests._assertions import assert_rejects


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
