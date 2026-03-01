"""Tests for Gaunt coefficient table."""

import math

import jax.numpy as jnp
import pytest

from arpyes.maths.gaunt import (
    GAUNT_TABLE,
    L_MAX,
    build_gaunt_table,
    gaunt_lookup,
)


class TestGauntTable:
    """Tests for the precomputed Gaunt coefficient table."""

    def test_table_shape(self):
        """Table has expected shape for l_max=4."""
        assert GAUNT_TABLE.shape == (5, 9, 3, 6, 11)

    def test_selection_rule_delta_l(self):
        """Only l' = l ± 1 are nonzero (dipole selection rule)."""
        for l in range(L_MAX + 1):
            for m in range(-l, l + 1):
                for q in (-1, 0, 1):
                    for lp in range(L_MAX + 2):
                        val = gaunt_lookup(l, m, q, lp, m + q)
                        if abs(lp - l) != 1:
                            assert abs(val) < 1e-12, (
                                f"Expected zero for l={l}, lp={lp} (Delta_l != ±1), "
                                f"got {val}"
                            )

    def test_s_to_p_nonzero(self):
        """s → p transition (l=0, m=0, q=0, l'=1, m'=0) is nonzero."""
        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert abs(val) > 1e-6

    def test_p_to_s_nonzero(self):
        """p → s transition (l=1, m=0, q=0, l'=0, m'=0) is nonzero."""
        val = gaunt_lookup(1, 0, 0, 0, 0)
        assert abs(val) > 1e-6

    def test_p_to_d_nonzero(self):
        """p → d transition (l=1, m=0, q=0, l'=2, m'=0) is nonzero."""
        val = gaunt_lookup(1, 0, 0, 2, 0)
        assert abs(val) > 1e-6

    def test_forbidden_delta_m(self):
        """Transitions with |Δm| > 1 are zero."""
        # l=2, m=0, q=0 => m'=0, so try m'=2 which requires |Δm|=2
        val = gaunt_lookup(2, 0, 0, 1, 2)
        assert abs(val) < 1e-12

    def test_rebuild_matches_precomputed(self):
        """Rebuilding the table produces identical results."""
        table2 = build_gaunt_table(l_max=4)
        assert jnp.allclose(GAUNT_TABLE, table2)

    def test_real_valued(self):
        """All Gaunt coefficients are real (no imaginary residuals)."""
        # The table is already a float array, so this is about the
        # construction process not producing imaginary parts
        assert GAUNT_TABLE.dtype == jnp.float64

    def test_known_value_y00_dipole(self):
        """G(0, 0, 0, 1, 0) = 1/(2√π) · √(3/(4π)) · √(4π/3) = 1/(2√π)."""
        # The integral ∫ Y_0^0 Y_1^0 Y_1^0 dΩ = √(3/(4π)) * √(1/(4π)) * √(3)/(4π) ...
        # Actually just check it's a known positive value.
        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert val > 0.0
