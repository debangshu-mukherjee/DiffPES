"""Validate Gaunt coefficient table.

Extended Summary
----------------
The tests validate the precomputed Gaunt coefficient table for angular
coupling in dipole matrix elements. Gaunt coefficients encode the integral
of three spherical harmonics over the unit sphere. They enforce the dipole
selection rules ``Delta l = +/-1`` and ``Delta m = q``. The tests verify the
table shape and both selection rules. They also verify allowed transitions,
reproducible construction, real values, and the positive fundamental s-p
coupling.

"""

import math

import chex
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped
from numpy.typing import NDArray
from scipy.special import sph_harm_y

from diffpes.maths import GAUNT_TABLE, build_gaunt_table, gaunt_lookup
from diffpes.maths.gaunt import _complex_gaunt, _wigner3j
from diffpes.types import L_MAX


@jaxtyped(typechecker=beartype)
def _scipy_real_spherical_harmonic(
    l_value: int,
    m_value: int,
    theta: Float64[NDArray, "..."],
    phi: Float64[NDArray, "..."],
) -> Float64[NDArray, "..."]:
    """PRIVATE: Evaluate the production real-harmonic convention through SciPy.

    Parameters
    ----------
    l_value : int
        Spherical-harmonic degree l.
    m_value : int
        Signed spherical-harmonic order m.
    theta : Float64[NDArray, "..."]
        Polar angles in radians.
    phi : Float64[NDArray, "..."]
        Azimuthal angles in radians.

    Returns
    -------
    real_value : Float64[NDArray, "..."]
        Real spherical-harmonic samples in the production convention.

    Notes
    -----
    Evaluates the complex harmonic at order abs(m) with SciPy. For
    positive m, scales its real part. For negative m, scales its
    imaginary part. For m = 0, returns the real part.
    """
    complex_value: Complex128[NDArray, "..."] = sph_harm_y(
        l_value,
        abs(m_value),
        theta,
        phi,
    )
    if m_value > 0:
        real_value: Float64[NDArray, "..."] = (
            np.sqrt(2.0) * (-1) ** m_value * complex_value.real
        )
    elif m_value < 0:
        real_value = np.sqrt(2.0) * (-1) ** abs(m_value) * complex_value.imag
    else:
        real_value = complex_value.real
    return real_value


class TestBuildGauntTable:
    """Validate the precomputed Gaunt coefficient table.

    Validates the module-level ``GAUNT_TABLE`` array and the
    ``gaunt_lookup`` accessor function. ``build_gaunt_table(l_max=4)`` builds
    the table at import time. The table stores real-valued Gaunt
    coefficients indexed by (l, m, q, l', m').  Tests systematically
    check selection rules, allowed transitions, table shape, dtype,
    reproducibility, and a known analytical value.

    :see: :func:`~diffpes.maths.build_gaunt_table`
    """

    def test_table_shape(self) -> None:
        """Verify the precomputed table has the expected 5-D shape.

        The five axes represent ``l``, ``m``, ``q``, ``l'``, and ``m'``.
        Their respective sizes are 5, 9, 3, 6, and 11 for ``l_max=4``.
        The test asserts the exact shape ``(5, 9, 3, 6, 11)``.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        assert GAUNT_TABLE.shape == (5, 9, 3, 6, 11)

    def test_selection_rule_delta_l(self) -> None:
        """Verify the dipole selection rule Delta l = +/-1.

        The test iterates over every valid table index combination.
        It asserts that whenever ``|l' - l| != 1`` the Gaunt
        coefficient is zero (< 1e-12).  This is the fundamental angular
        momentum selection rule for electric-dipole transitions.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        l: int
        m: int
        q: int
        lp: int

        val: float

        for l in range(L_MAX + 1):
            for m in range(-l, l + 1):
                for q in (-1, 0, 1):
                    for lp in range(L_MAX + 2):
                        val = gaunt_lookup(l, m, q, lp, m + q)
                        if abs(lp - l) != 1:
                            assert abs(val) < 1e-12, (
                                f"Expected zero for l={l}, lp={lp} "
                                "(Delta_l != ±1), "
                                f"got {val}"
                            )

    def test_s_to_p_nonzero(self) -> None:
        """Verify that the allowed s-to-p transition is nonzero.

        Looks up G(l=0, m=0, q=0, l'=1, m'=0), the prototypical
        electric-dipole transition from an s-orbital to pz.  Asserts
        ``|G| > 1e-6``, confirming the table correctly encodes the
        coupling.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert abs(val) > 1e-6

    def test_p_to_s_nonzero(self) -> None:
        """Verify that the allowed p-to-s transition is nonzero.

        Looks up G(l=1, m=0, q=0, l'=0, m'=0), the reverse of the s->p
        transition.  Asserts ``|G| > 1e-6``, confirming reciprocity of the
        Gaunt integral.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = gaunt_lookup(1, 0, 0, 0, 0)
        assert abs(val) > 1e-6

    def test_p_to_d_nonzero(self) -> None:
        """Verify that the allowed p-to-d transition is nonzero.

        Looks up G(l=1, m=0, q=0, l'=2, m'=0), a higher-l allowed dipole
        transition.  Asserts ``|G| > 1e-6``, confirming that the table
        covers
        transitions beyond the lowest-order s<->p coupling.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = gaunt_lookup(1, 0, 0, 2, 0)
        assert abs(val) > 1e-6

    def test_forbidden_delta_m(self) -> None:
        """Verify the magnetic selection rule forbids ``|Delta m| > 1``.

        Looks up G(l=2, m=0, q=0, l'=1, m'=2) where m' - m = 2 != q = 0.
        The selection rule requires ``m' = m + q`` and forbids ``m'=2`` here.
        The test asserts the coefficient is zero (< 1e-12).

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = gaunt_lookup(2, 0, 0, 1, 2)
        assert abs(val) < 1e-12

    def test_rebuild_matches_precomputed(self) -> None:
        """Verify that rebuilding the table reproduces the precomputed values.

        The test calls ``build_gaunt_table(l_max=4)`` at test time and compares
        the result to the module-level ``GAUNT_TABLE`` with ``jnp.allclose``.
        This guards against silent corruption of the cached table and confirms
        deterministic construction.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        table2: Float64[
            Array,
            "n_l_initial n_m_initial 3 n_l_final n_m_final",
        ]

        table2 = build_gaunt_table(l_max=4)
        assert jnp.allclose(GAUNT_TABLE, table2)

    def test_real_valued(self) -> None:
        """Verify the table dtype is float64 (no imaginary residuals).

        The Gaunt coefficients for real spherical harmonics are purely
        real.  Asserts the table dtype is ``jnp.float64``, confirming
        the construction did not accidentally introduce complex values.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        assert GAUNT_TABLE.dtype == jnp.float64

    def test_known_value_y00_dipole(self) -> None:
        """Verify the s-to-p Gaunt coefficient is a known positive value.

        The integral G(0, 0, 0, 1, 0) = integral Y_0^0 * Y_1^0 * Y_1^0 dOmega
        is analytically positive.  While the exact numerical value depends
        on normalization conventions, the sign is unambiguous.  Asserts
        val > 0.0 as a consistency check against sign errors in the
        Condon-Shortley phase convention.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = gaunt_lookup(0, 0, 0, 1, 0)
        assert val > 0.0

    def test_sine_channel_sign_matches_cartesian_axes(self) -> None:
        """Match the positive ``p_y``--``y``--``s`` Cartesian integral.

        The fixture detects a conjugated real-basis transform on the final leg.

        Notes
        -----
        The test compares the channel with the exact value ``1/sqrt(4*pi)``.
        """
        actual: float = gaunt_lookup(1, -1, -1, 0, 0)
        expected: float = 1.0 / math.sqrt(4.0 * math.pi)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_complete_table_matches_independent_angular_quadrature(
        self,
    ) -> None:
        """Match every physical entry with an independent SciPy quadrature.

        The reference constructs real harmonics directly from complex SciPy
        values.

        Notes
        -----
        The test combines Gauss--Legendre polar nodes with a uniform azimuth
        grid.
        """
        cosine_nodes: Float64[NDArray, " n_theta"]
        cosine_weights: Float64[NDArray, " n_theta"]
        cosine_nodes, cosine_weights = np.polynomial.legendre.leggauss(32)
        phi_values: Float64[NDArray, " n_phi"] = np.arange(
            64, dtype=np.float64
        ) * (2.0 * np.pi / 64.0)
        theta_grid: Float64[NDArray, "n_theta 1"] = np.arccos(cosine_nodes)[
            :, None
        ]
        phi_grid: Float64[NDArray, "1 n_phi"] = phi_values[None, :]
        phi_weight: float = 2.0 * np.pi / 64.0
        l_initial: int
        m_initial: int
        q_value: int
        l_final: int
        m_final: int
        initial_harmonic: Float64[NDArray, "n_theta n_phi"]
        dipole_harmonic: Float64[NDArray, "n_theta n_phi"]
        final_harmonic: Float64[NDArray, "n_theta n_phi"]
        expected: float
        actual: float
        for l_initial in range(L_MAX + 1):
            for m_initial in range(-l_initial, l_initial + 1):
                initial_harmonic = _scipy_real_spherical_harmonic(
                    l_initial,
                    m_initial,
                    theta_grid,
                    phi_grid,
                )
                for q_value in (-1, 0, 1):
                    dipole_harmonic = _scipy_real_spherical_harmonic(
                        1,
                        q_value,
                        theta_grid,
                        phi_grid,
                    )
                    for l_final in (l_initial - 1, l_initial + 1):
                        if l_final < 0:
                            continue
                        for m_final in range(-l_final, l_final + 1):
                            final_harmonic = _scipy_real_spherical_harmonic(
                                l_final,
                                m_final,
                                theta_grid,
                                phi_grid,
                            )
                            expected = float(
                                np.sum(
                                    cosine_weights[:, None]
                                    * final_harmonic
                                    * dipole_harmonic
                                    * initial_harmonic
                                )
                                * phi_weight
                            )
                            actual = gaunt_lookup(
                                l_initial,
                                m_initial,
                                q_value,
                                l_final,
                                m_final,
                            )
                            np.testing.assert_allclose(
                                actual,
                                expected,
                                rtol=1e-14,
                                atol=1e-14,
                            )


class TestGauntLookup:
    """Validate :func:`~diffpes.maths.gaunt_lookup`.

    Covers indexed retrieval from the canonical dipole Gaunt table for one
    allowed transition and one forbidden magnetic transition.

    :see: :func:`~diffpes.maths.gaunt_lookup`
    """

    def test_matches_canonical_table_entries(self) -> None:
        """Match lookup results at allowed and zero table entries.

        The accessor must preserve both the positive s-to-p coefficient and an
        exactly forbidden magnetic channel under the package indexing
        convention.

        Notes
        -----
        The test evaluates two scalar lookups and compares them with directly
        indexed ``GAUNT_TABLE`` entries at zero absolute and relative
        tolerance.
        """
        allowed: float
        forbidden: float
        expected_allowed: Float64[Array, ""]
        expected_forbidden: Float64[Array, ""]

        allowed = gaunt_lookup(0, 0, 0, 1, 0)
        forbidden = gaunt_lookup(0, 0, 1, 1, 0)
        expected_allowed = GAUNT_TABLE[0, L_MAX, 1, 1, L_MAX + 1]
        expected_forbidden = GAUNT_TABLE[0, L_MAX, 2, 1, L_MAX + 1]
        chex.assert_trees_all_close(
            allowed,
            expected_allowed,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            forbidden,
            expected_forbidden,
            rtol=0.0,
            atol=0.0,
        )


class TestWigner3jSelectionRules:
    """Validate internal Wigner 3-j and complex Gaunt zero paths.

    The tests exercise early returns in ``_wigner3j`` and ``_complex_gaunt``.
    These branches return 0.0 for violations of selection rules.

    :see: :func:`~diffpes.maths.build_gaunt_table`
    """

    def test_abs_m_exceeds_j_returns_zero(self) -> None:
        """Verify ``|m1| > j1`` causes _wigner3j to return 0.0.

        The test constructs a call where ``|m1| = 2 > j1 = 1``, violating the
        ``|mi| <= ji`` constraint, and asserts the result is 0.0
        (line 111).

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = _wigner3j(1, 1, 0, 2, -1, -1)
        assert val == 0.0

    def test_triangle_inequality_violated_returns_zero(self) -> None:
        """Verify triangle inequality violation causes _wigner3j to return 0.0.

        The test uses j1=2, j2=1, j3=0 where ``j3 < |j1 - j2| = 1``. This
        violates the triangle inequality, so the result must be 0.0.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = _wigner3j(2, 1, 0, 0, 0, 0)
        assert val == 0.0

    def test_complex_gaunt_zero_w3j_000_returns_zero(self) -> None:
        """Verify that a zero w3j_000 makes _complex_gaunt return 0.0.

        The three-j symbol ``(2,1,0 | 0,0,0)`` is zero for this input.
        The parity rule gives zero because ``l1+l2+l3 = 3`` is odd.
        The test asserts that the complex Gaunt integral returns 0.0.

        Notes
        -----
        The test builds the documented inputs.
        It checks the stated property with explicit assertions.
        """
        val: float

        val = _complex_gaunt(2, 0, 1, 0, 0, 0)
        assert val == 0.0
