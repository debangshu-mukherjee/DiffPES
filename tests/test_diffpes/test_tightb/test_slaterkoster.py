"""Validate the slaterkoster module.

The cases use analytic values, invariants, and finite differences.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.tightb import (
    sk_block,
)

from ._slaterkoster_helpers import (
    _TABLE_DIRECTIONS,
    _table_i_blocks,
)


class TestSkBlock:
    """Validate :func:`diffpes.tightb.sk_block`.

    The cases check table channels, parity, Hermiticity, pole gradients,
    compilation, and validation.
    """

    def test_all_table_i_channels_on_fifty_random_directions(self) -> None:
        """Match direction-cosine polynomials for all ten s/p/d channels.

        The deterministic directions are generic and the parameter values are
        distinct, preventing accidental sigma/pi/delta interchange.

        Notes
        -----
        Compare every entry of all six canonical shell-pair blocks at the
        slater-koster-analytic-tolerance.
        """
        generator: np.random.Generator = np.random.default_rng(93281)
        raw: Float64[NDArray, "n_direction 3"] = generator.normal(
            size=(_TABLE_DIRECTIONS, 3)
        )
        directions: Float64[NDArray, "n_direction 3"] = raw / np.linalg.norm(
            raw,
            axis=1,
            keepdims=True,
        )
        values: Float64[Array, " 10"] = jnp.asarray(
            (0.37, -1.1, 0.83, 2.3, -0.61, 1.7, -0.42, 3.1, -0.91, 0.28),
            dtype=jnp.float64,
        )
        channel_vectors: Dict[Tuple[int, int], Float64[Array, " n_m"]] = {
            (0, 0): values[0:1],
            (0, 1): values[1:2],
            (0, 2): values[2:3],
            (1, 1): values[3:5],
            (1, 2): values[5:7],
            (2, 2): values[7:10],
        }

        direction: Float64[NDArray, " 3"]
        for direction in directions:
            bond: Float64[Array, " 3"] = jnp.asarray(
                direction,
                dtype=jnp.float64,
            )
            references: Dict[Tuple[int, int], Float64[Array, "m1 m2"]] = (
                _table_i_blocks(bond, values)
            )
            angular_pair: Tuple[int, int]
            integrals: Float64[Array, " n_m"]
            for angular_pair, integrals in channel_vectors.items():
                actual: Float64[Array, "m1 m2"] = sk_block(
                    angular_pair[0],
                    angular_pair[1],
                    integrals,
                    bond,
                )
                np.testing.assert_allclose(
                    actual,
                    references[angular_pair],
                    rtol=1e-12,
                    atol=2e-14,
                )

    @pytest.mark.parametrize(
        ("l1", "l2"),
        tuple((l1, l2) for l1 in range(3) for l2 in range(3)),
    )
    def test_parity_and_swapped_shell_hermiticity(
        self,
        l1: int,
        l2: int,
    ) -> None:
        """Preserve fixed-shell parity and reverse-bond Hermiticity.

        The nine shell-order pairs include separately dimensioned rectangular
        blocks.

        Notes
        -----
        Apply the radial-integral reversal convention automatically for a
        swapped shell order.
        """
        bond: Float64[Array, " 3"] = jnp.asarray(
            (0.31, -0.47, 0.73),
            dtype=jnp.float64,
        )
        integrals: Float64[Array, " n_m"] = jnp.arange(
            1,
            min(l1, l2) + 2,
            dtype=jnp.float64,
        )
        block: Float64[Array, "m1 m2"] = sk_block(
            l1,
            l2,
            integrals,
            bond,
        )
        reversed_bond: Float64[Array, "m1 m2"] = sk_block(
            l1,
            l2,
            integrals,
            -bond,
        )
        swapped: Float64[Array, "m2 m1"] = sk_block(
            l2,
            l1,
            integrals,
            -bond,
        )

        np.testing.assert_allclose(
            reversed_bond,
            (-1) ** (l1 + l2) * block,
            rtol=1e-13,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            swapped.T,
            block,
            rtol=1e-13,
            atol=2e-14,
        )

    @pytest.mark.parametrize("pole", [1.0, -1.0])
    def test_transverse_gradient_is_analytic_at_bond_poles(
        self,
        pole: float,
    ) -> None:
        """Retain the nonzero s--px derivative at positive and negative z.

        The s--px Table-I element is ``x / norm(bond) * V_sp_sigma``.

        Notes
        -----
        Compare reverse-mode AD with its exact transverse pole derivative.
        """
        integral: float = 1.7
        bond: Float64[Array, " 3"] = jnp.asarray(
            (0.0, 0.0, 2.0 * pole),
            dtype=jnp.float64,
        )

        def s_px(candidate: Float64[Array, " 3"]) -> Float64[Array, ""]:
            """Return the s--px matrix element."""
            value: Float64[Array, ""] = sk_block(
                0,
                1,
                jnp.asarray((integral,), dtype=jnp.float64),
                candidate,
            )[0, 2]
            return value

        gradient: Float64[Array, " 3"] = jax.grad(s_px)(bond)

        np.testing.assert_allclose(
            gradient,
            jnp.asarray((integral / 2.0, 0.0, 0.0)),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_jit_shape_dtype_and_rejections(self) -> None:
        """Compile a d--p block and reject malformed physical inputs.

        The successful path fixes the rectangular shell dimensions and
        double-precision output contract.

        Notes
        -----
        Pin the rectangular real float64 output and zero-bond diagnostic.
        """
        compiled: Callable[
            [
                int,
                int,
                Float64[Array, " n_integral"],
                Float64[Array, " 3"],
            ],
            Float64[Array, "n_left n_right"],
        ] = jax.jit(
            sk_block,
            static_argnums=(0, 1),
        )
        block: Float64[Array, "5 3"] = compiled(
            2,
            1,
            jnp.asarray((0.8, -0.3), dtype=jnp.float64),
            jnp.asarray((0.2, 0.4, 0.7), dtype=jnp.float64),
        )

        assert block.shape == (5, 3)
        assert block.dtype == jnp.float64
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="bond nonzero",
        ):
            sk_block(
                1,
                1,
                jnp.asarray((1.0, -0.2), dtype=jnp.float64),
                jnp.zeros((3,), dtype=jnp.float64),
            )
        with pytest.raises(ValueError, match="v_llm length"):
            sk_block(
                2,
                2,
                jnp.ones((2,), dtype=jnp.float64),
                jnp.ones((3,), dtype=jnp.float64),
            )
