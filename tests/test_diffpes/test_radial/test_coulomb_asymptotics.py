"""Validate the coulomb asymptotics module.

The cases use analytic values, invariants, and finite differences.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict
from jaxtyping import Array, Float64, Shaped
from numpy.typing import NDArray

from diffpes.radial import (
    coulomb_phase_shift,
)

from ._coulomb_helpers import (
    _reference,
)


class TestCoulombPhaseShift:
    """Validate :func:`diffpes.radial.coulomb_phase_shift`.

    The cases cover values and derivatives for orders zero through four on
    the frozen Sommerfeld grid. They compare arbitrary-precision rows through
    JVPs and check continuity on a dense no-wrap sweep.

    :see: :func:`~diffpes.radial.coulomb_phase_shift`
    """

    def test_values_derivatives_and_continuous_branch(self) -> None:
        """Match phase values and derivatives on a continuous branch.

        The test covers every frozen order and a dense no-wrap sweep.

        Notes
        -----
        It compares production outputs with stored arbitrary-precision values.
        """
        reference: Dict[str, Shaped[NDArray, "..."]] = _reference()
        etas: Float64[Array, " n_eta"] = jnp.asarray(reference["etas"])
        order: int
        for order in range(5):
            values: Float64[Array, " n_eta"] = coulomb_phase_shift(order, etas)
            derivatives: Float64[Array, " n_eta"] = jax.jvp(
                partial(coulomb_phase_shift, order),
                (etas,),
                (jnp.ones_like(etas),),
            )[1]
            np.testing.assert_allclose(
                values,
                reference["phase"][order],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                derivatives,
                reference["phase_eta"][order],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        dense_eta: Float64[Array, " n_dense"] = jnp.linspace(-3.0, 3.0, 601)
        dense_phase: Float64[Array, " n_dense"] = coulomb_phase_shift(
            4,
            dense_eta,
        )
        assert float(jnp.max(jnp.abs(jnp.diff(dense_phase)))) < 0.1
        assert float(coulomb_phase_shift(4, jnp.asarray(0.0))) == 0.0
