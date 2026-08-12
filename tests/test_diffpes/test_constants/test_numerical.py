"""Validate frozen coefficient tables for numerical kernels.

The tests inspect the exact registered shapes and basic support structure.
Independent kernel tests compare the tables with their mathematical rebuilds.
"""

import jax.numpy as jnp

from diffpes.constants import FADDEEVA_WEIDEMAN_COEFFICIENTS, GAUNT_TABLE


class TestNumericalConstants:
    """Validate the frozen Gaunt and Weideman tables.

    The cases check the registered dimensions, float64 storage, and sparse
    support count.

    Notes
    -----
    Mathematical rebuild tests remain with the owning numerical kernels.
    """

    def test_frozen_tables_have_registered_layouts(self) -> None:
        """Preserve the table shapes and float64 representation.

        The check inspects both public arrays and counts the Gaunt support.

        Notes
        -----
        Shape and support checks detect truncation during source generation.
        """
        assert FADDEEVA_WEIDEMAN_COEFFICIENTS.shape == (40,)
        assert FADDEEVA_WEIDEMAN_COEFFICIENTS.dtype == jnp.float64
        assert GAUNT_TABLE.shape == (5, 9, 3, 6, 11)
        assert GAUNT_TABLE.dtype == jnp.float64
        assert int(jnp.count_nonzero(GAUNT_TABLE)) == 173
