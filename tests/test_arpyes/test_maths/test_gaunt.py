"""Tests for Gaunt coefficient table.

Extended Summary
----------------
Validates the precomputed Gaunt coefficient table used for angular
coupling in dipole matrix elements.  The Gaunt coefficients
G(l, m, q, l', m') encode the integral of three spherical harmonics
over the unit sphere and enforce dipole selection rules (Delta l = +/-1,
Delta m = q).  Tests verify the table shape, the Delta-l and Delta-m
selection rules, nonzero values for allowed transitions (s->p, p->s,
p->d), reproducibility of the build procedure, real-valuedness of all
entries, and a known positive value for the fundamental s-p coupling.

Routine Listings
----------------
:class:`TestGauntTable`
    Tests for the precomputed Gaunt coefficient table and lookup.
"""

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
    """Tests for the precomputed Gaunt coefficient table.

    Validates the module-level ``GAUNT_TABLE`` array and the
    ``gaunt_lookup`` accessor function.  The table is built at import
    time by ``build_gaunt_table(l_max=4)`` and stores real-valued Gaunt
    coefficients indexed by (l, m, q, l', m').  Tests systematically
    check selection rules, allowed transitions, table shape, dtype,
    reproducibility, and a known analytical value.
    """

    def test_table_shape(self):
        """Verify the precomputed table has the expected 5-D shape.

        For l_max=4 the table axes are: l in [0..4] (5), m in [-4..4]
        (9), q in [-1..1] (3), l' in [0..5] (6), m' in [-5..5] (11).
        Asserts the shape is exactly ``(5, 9, 3, 6, 11)``.
        """
        assert GAUNT_TABLE.shape == (5, 9, 3, 6, 11)

    def test_selection_rule_delta_l(self):
        """Verify the dipole selection rule Delta l = +/-1.

        Iterates over every valid combination of (l, m, q, l', m') in the
        table and asserts that whenever |l' - l| != 1 the Gaunt
        coefficient is zero (< 1e-12).  This is the fundamental angular
        momentum selection rule for electric-dipole transitions.
        """
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
        """Verify the s -> p allowed transition has a nonzero Gaunt coefficient.

        Looks up G(l=0, m=0, q=0, l'=1, m'=0), the prototypical
        electric-dipole transition from an s-orbital to pz.  Asserts
        |G| > 1e-6, confirming the table correctly encodes the coupling.
        """
        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert abs(val) > 1e-6

    def test_p_to_s_nonzero(self):
        """Verify the p -> s allowed transition has a nonzero Gaunt coefficient.

        Looks up G(l=1, m=0, q=0, l'=0, m'=0), the reverse of the s->p
        transition.  Asserts |G| > 1e-6, confirming reciprocity of the
        Gaunt integral.
        """
        val = gaunt_lookup(1, 0, 0, 0, 0)
        assert abs(val) > 1e-6

    def test_p_to_d_nonzero(self):
        """Verify the p -> d allowed transition has a nonzero Gaunt coefficient.

        Looks up G(l=1, m=0, q=0, l'=2, m'=0), a higher-l allowed dipole
        transition.  Asserts |G| > 1e-6, confirming that the table covers
        transitions beyond the lowest-order s<->p coupling.
        """
        val = gaunt_lookup(1, 0, 0, 2, 0)
        assert abs(val) > 1e-6

    def test_forbidden_delta_m(self):
        """Verify the magnetic selection rule forbids |Delta m| > 1.

        Looks up G(l=2, m=0, q=0, l'=1, m'=2) where m' - m = 2 != q = 0.
        The selection rule requires m' = m + q, so m' = 2 is forbidden.
        Asserts the coefficient is zero (< 1e-12).
        """
        # l=2, m=0, q=0 => m'=0, so try m'=2 which requires |Δm|=2
        val = gaunt_lookup(2, 0, 0, 1, 2)
        assert abs(val) < 1e-12

    def test_rebuild_matches_precomputed(self):
        """Verify that rebuilding the table reproduces the precomputed values.

        Calls ``build_gaunt_table(l_max=4)`` at test time and compares to
        the module-level ``GAUNT_TABLE`` with ``jnp.allclose``.  This
        guards against silent corruption of the cached table and confirms
        deterministic construction.
        """
        table2 = build_gaunt_table(l_max=4)
        assert jnp.allclose(GAUNT_TABLE, table2)

    def test_real_valued(self):
        """Verify the table dtype is float64 (no imaginary residuals).

        The Gaunt coefficients for real spherical harmonics are purely
        real.  Asserts the table dtype is ``jnp.float64``, confirming
        the construction did not accidentally introduce complex values.
        """
        # The table is already a float array, so this is about the
        # construction process not producing imaginary parts
        assert GAUNT_TABLE.dtype == jnp.float64

    def test_known_value_y00_dipole(self):
        """Verify the s-to-p Gaunt coefficient is a known positive value.

        The integral G(0, 0, 0, 1, 0) = integral Y_0^0 * Y_1^0 * Y_1^0 dOmega
        is analytically positive.  While the exact numerical value depends
        on normalization conventions, the sign is unambiguous.  Asserts
        val > 0.0 as a consistency check against sign errors in the
        Condon-Shortley phase convention.
        """
        # The integral ∫ Y_0^0 Y_1^0 Y_1^0 dΩ = √(3/(4π)) * √(1/(4π)) * √(3)/(4π) ...
        # Actually just check it's a known positive value.
        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert val > 0.0
