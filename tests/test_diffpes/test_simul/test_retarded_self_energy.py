"""Validate the retarded self energy module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

from types import ModuleType

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    evaluate_self_energy,
)
from diffpes.types import (
    SelfEnergyModel,
    make_self_energy_model,
)
from tests._assertions import assert_rejects

from ._spectral_helpers import (
    _ANALYTIC_ROW_ATOL,
    _BASE_SPACING_EV,
    _DERIVATIVE_ROW_RTOL,
    _DOMAIN_HIGH_EV,
    _DOMAIN_LOW_EV,
    _FL_PARAMETERS_PHYSICAL,
    _FL_REAL_ROW_ATOL,
    _FL_TAIL_RAW,
    _FL_VALUE_ATOL_EV,
    _FL_VALUE_RTOL,
    _GRID_EXACTNESS_ATOL_EV,
    _GRID_EXACTNESS_RTOL,
    _IDENTITY_SCALE_BOUND,
    _MODELS_ARCHIVE_PATH,
    _MODELS_ARCHIVE_SHA256,
    _N_KK,
    _QUERY_INVARIANCE_ATOL_EV,
    _authenticated_npz,
    _committed_operator,
    _fermi_liquid_model,
    _fl_dsigma_imag_dynamic,
    _fl_dsigma_real_domega,
    _fl_sigma_imag_dynamic,
    _fl_tail_spec,
    _hand_power2_tail,
    _hat_core_pv,
    _instrument_composite_derivative,
    _instrument_subtracted,
    _scaled_model,
    _softplus_inverse_np,
)


class TestEvaluateSelfEnergy(chex.TestCase):
    """Test the public complex retarded evaluation contract.

    :see: :func:`diffpes.simul.retarded_self_energy.evaluate_self_energy`
    :see: :func:`~diffpes.simul.evaluate_self_energy`
    """

    def test_constant_mode_returns_complex_retarded_sigma(self) -> None:
        """Require ``Sigma = -i * softplus(c0)`` for the constant carrier.

        Acceptance: the output dtype is complex. The subtracted real part
        equals zero exactly. The imaginary part equals ``-gamma`` within
        ``1e-14`` through the stored inverse-softplus coordinate.

        Notes
        -----
        The test builds the ``gamma`` shortcut carrier and evaluates it on
        a small query grid. It then checks dtype, real part, and imaginary
        part separately.
        """
        model: SelfEnergyModel = make_self_energy_model(gamma=0.1)
        omega: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 21)
        sigma: Complex128[Array, " n"] = evaluate_self_energy(omega, model)
        assert jnp.iscomplexobj(sigma), (
            "constant-mode evaluation must return the complex retarded "
            "self-energy"
        )
        np.testing.assert_array_equal(np.real(np.asarray(sigma)), np.zeros(21))
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            np.full(21, -0.1),
            rtol=0.0,
            atol=1e-14,
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_fermi_liquid_subtracted_real_part_matches_frozen_truth(
        self,
    ) -> None:
        """Match the frozen full-line Fermi-liquid subtracted real part.

        Acceptance: per query row ``|err| <= 2e-8 + 1e-6 * |truth|``
        against ``fl_sigma_real_subtracted_analytic``. The committed
        instrument measures maximum error ``1.6529e-9 eV`` (worst mixed
        ratio ``0.0497``) here. The imaginary part must reproduce
        ``fl_sigma_imag_grid`` at roundoff.

        Notes
        -----
        The test evaluates the frozen carrier on the frozen grid with the
        default committed geometry. It then applies the registered mixed
        criterion per row.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"], dtype=jnp.float64
        )
        model: SelfEnergyModel = _fermi_liquid_model()
        sigma: Complex128[Array, " n"] = evaluate_self_energy(
            queries, model, n_kk=_N_KK
        )
        assert jnp.iscomplexobj(sigma), (
            "fermi_liquid evaluation must return the complex retarded "
            "self-energy"
        )
        truth: Float64[NDArray, " n"] = models[
            "fl_sigma_real_subtracted_analytic"
        ]
        error: Float64[NDArray, " n"] = np.abs(
            np.real(np.asarray(sigma)) - truth
        )
        assert np.all(
            error <= _FL_VALUE_ATOL_EV + _FL_VALUE_RTOL * np.abs(truth)
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            models["fl_sigma_imag_grid"],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_bosonic_kink_complex_pole_pair_matches_frozen_truth(
        self,
    ) -> None:
        """Match the frozen analytic bosonic-kink complex pole pair.

        Acceptance: the real part matches ``kink_sigma_real_grid`` at
        rtol ``1e-8``. The imaginary part reproduces the frozen
        ``kink_sigma_imag_grid`` parameter truth at roundoff.

        Notes
        -----
        The test builds the frozen kink carrier from raw coordinates and
        evaluates the analytic pole pair on the frozen grid. It compares
        both parts against the archive rows.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"], dtype=jnp.float64
        )
        model: SelfEnergyModel = _scaled_model(
            "bosonic_kink", jnp.ones(4, dtype=jnp.float64)
        )
        sigma: Complex128[Array, " n"] = evaluate_self_energy(queries, model)
        assert jnp.iscomplexobj(sigma), (
            "bosonic_kink evaluation must return the complex retarded "
            "self-energy"
        )
        np.testing.assert_allclose(
            np.real(np.asarray(sigma)),
            models["kink_sigma_real_grid"],
            rtol=1e-8,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            models["kink_sigma_imag_grid"],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_appending_distant_query_points_preserves_existing_values(
        self,
    ) -> None:
        """Require query-set invariance of the existing outputs.

        Acceptance: appended distant trusted query points change no
        existing output by more than ``1e-15 eV``. The invariance holds on
        the complex retarded contract.

        Notes
        -----
        The test evaluates one base query set and one extended query set.
        It then compares the shared prefix of both outputs.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        base: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 41)
        extended: Float64[Array, " m"] = jnp.concatenate(
            [base, jnp.asarray([5.0, -5.5, 6.5])]
        )
        first: Complex128[Array, " n"] = evaluate_self_energy(base, model)
        second: Complex128[Array, " m"] = evaluate_self_energy(extended, model)
        assert jnp.iscomplexobj(first), (
            "the invariance check runs on the complex retarded contract"
        )
        np.testing.assert_allclose(
            np.asarray(second)[: base.shape[0]],
            np.asarray(first),
            rtol=0.0,
            atol=_QUERY_INVARIANCE_ATOL_EV,
        )

    def test_query_window_never_defines_the_quadrature_grid(self) -> None:
        """Require the quadrature domain to come from the carrier.

        Acceptance: two disjoint query windows agree within ``1e-15 eV``
        at shared points. The grid derives from
        ``kk_domain_rel_fermi_ev`` and never from the query extrema. A
        planted query-window grid fails this bound.

        Notes
        -----
        The test evaluates one narrow and one wide query window with a
        shared prefix. It then compares the shared outputs bitwise-close.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        shared: Float64[Array, " k"] = jnp.linspace(-0.4, 0.4, 17)
        narrow: Float64[Array, " n"] = jnp.concatenate(
            [shared, jnp.asarray([-0.9, 1.1])]
        )
        wide: Float64[Array, " m"] = jnp.concatenate(
            [shared, jnp.asarray([-6.5, 6.9])]
        )
        narrow_out: Complex128[Array, " n"] = evaluate_self_energy(
            narrow, model
        )
        wide_out: Complex128[Array, " m"] = evaluate_self_energy(wide, model)
        assert jnp.iscomplexobj(narrow_out), (
            "the window check runs on the complex retarded contract"
        )
        np.testing.assert_allclose(
            np.asarray(wide_out)[: shared.shape[0]],
            np.asarray(narrow_out)[: shared.shape[0]],
            rtol=0.0,
            atol=_QUERY_INVARIANCE_ATOL_EV,
        )

    def test_trusted_interval_is_enforced_eagerly_and_under_jit(
        self,
    ) -> None:
        """Reject queries and subtraction points outside the trusted band.

        Acceptance: with base spacing ``h = 16/4095 eV`` the trusted
        interval equals ``[a + 2h, b - 2h]``. A query at ``b - h/2``, a
        query beyond the domain, and an out-of-band subtraction point each
        raise eagerly and under jit.

        Notes
        -----
        The test drives the shared rejection helper, which repeats each
        call through ``eqx.filter_jit`` and matches the error text.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        outside_query: Float64[Array, " one"] = jnp.asarray(
            [_DOMAIN_HIGH_EV - 0.5 * _BASE_SPACING_EV]
        )
        assert_rejects(
            evaluate_self_energy, outside_query, model, match="trusted"
        )
        beyond_domain: Float64[Array, " one"] = jnp.asarray([9.0])
        assert_rejects(
            evaluate_self_energy, beyond_domain, model, match="trusted"
        )
        raw: Float64[NDArray, " three"] = _softplus_inverse_np(
            np.asarray(_FL_PARAMETERS_PHYSICAL)
        )
        edge_subtraction: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray(raw),
            mode="fermi_liquid",
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray(
                [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV]
            ),
            tail_coefficients=jnp.asarray(_FL_TAIL_RAW),
            subtraction_point_rel_fermi_ev=(
                _DOMAIN_HIGH_EV - 0.5 * _BASE_SPACING_EV
            ),
            tail_mode="power2",
        )
        inside_query: Float64[Array, " one"] = jnp.asarray([0.25])
        assert_rejects(
            evaluate_self_energy,
            inside_query,
            edge_subtraction,
            match="trusted",
        )

    def test_fermi_liquid_trusted_boundary_evaluation_is_certified(
        self,
    ) -> None:
        """Certify boundary evaluation for a non-decaying imaginary part.

        The Fermi-liquid imaginary part does not decay inside the core
        domain. No selection-battery fixture covers this boundary class.
        Acceptance: a query just inside ``b - 2h`` evaluates and matches
        the committed operator within ``1e-9 * max(1, |value|)``. Queries
        at ``a + 1.5h`` and ``b - 1.5h`` raise eagerly and under jit.

        Notes
        -----
        The test evaluates production near the trusted boundary and
        compares against the authenticated instrument value. It then
        drives the shared rejection helper for both edges.
        """
        model: SelfEnergyModel = _fermi_liquid_model()
        boundary_query: Float64[Array, " one"] = jnp.asarray(
            [_DOMAIN_HIGH_EV - 2.5 * _BASE_SPACING_EV]
        )
        sigma: Complex128[Array, " one"] = evaluate_self_energy(
            boundary_query, model
        )
        assert jnp.iscomplexobj(sigma), (
            "the boundary check runs on the complex retarded contract"
        )
        common: ModuleType
        common, _ = _committed_operator()
        spec: Any = _fl_tail_spec(common)
        expected: Float64[Array, " one"] = _instrument_subtracted(
            _fl_sigma_imag_dynamic, spec, boundary_query, 0.0
        )
        deviation: float = float(
            np.abs(np.real(np.asarray(sigma))[0] - np.asarray(expected)[0])
        )
        assert deviation <= _IDENTITY_SCALE_BOUND * max(
            1.0, float(np.abs(np.asarray(expected)[0]))
        )
        assert_rejects(
            evaluate_self_energy,
            jnp.asarray([_DOMAIN_HIGH_EV - 1.5 * _BASE_SPACING_EV]),
            model,
            match="trusted",
        )
        assert_rejects(
            evaluate_self_energy,
            jnp.asarray([_DOMAIN_LOW_EV + 1.5 * _BASE_SPACING_EV]),
            model,
            match="trusted",
        )


class TestEvaluateSelfEnergyDerivatives(chex.TestCase):
    """Test the composite derivative route on the public callable.

    :see: :func:`diffpes.simul.retarded_self_energy.evaluate_self_energy`
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_public_forward_and_reverse_ad_follow_the_composite_route(
        self,
    ) -> None:
        """Require the composite route from public ``jvp`` and ``grad``.

        Acceptance: ``jax.jvp`` and ``jax.grad`` of the public callable in
        the query coordinate reproduce the committed composite route
        within ``1e-9 * max(1, |composite|)`` per query. Both also match
        the analytic derivative within ``2e-8 + 1e-6 * |truth|`` per
        query. The committed instrument measures ``3.0722e-9`` (mixed
        ratio ``0.0459``) for the composite route here. Direct
        differentiation of the transform reaches ``7.46e-7`` and fails
        the identity bound.

        Notes
        -----
        The test runs forward mode with a unit query tangent and reverse
        mode through the summed real part. It rebuilds the composite truth
        from the authenticated instrument modules.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            models["eval_grid_ev"][::10], dtype=jnp.float64
        )
        model: SelfEnergyModel = _fermi_liquid_model()

        def public_map(
            points: Float64[Array, " n"],
        ) -> Complex128[Array, " n"]:
            returned: Complex128[Array, " n"] = evaluate_self_energy(
                points, model
            )
            return returned

        primal: Complex128[Array, " n"]
        tangent: Complex128[Array, " n"]
        primal, tangent = jax.jvp(
            public_map, (queries,), (jnp.ones_like(queries),)
        )
        assert jnp.iscomplexobj(primal), (
            "the public derivative demonstrator requires the complex "
            "retarded contract"
        )
        forward: Float64[NDArray, " n"] = np.real(np.asarray(tangent))

        def real_sum(points: Float64[Array, " n"]) -> Float64[Array, ""]:
            returned: Float64[Array, ""] = jnp.sum(
                jnp.real(evaluate_self_energy(points, model))
            )
            return returned

        reverse: Float64[NDArray, " n"] = np.asarray(
            jax.grad(real_sum)(queries)
        )

        common: ModuleType
        common, _ = _committed_operator()
        spec: Any = _fl_tail_spec(common)
        composite: Float64[NDArray, " n"] = np.asarray(
            _instrument_composite_derivative(
                _fl_sigma_imag_dynamic,
                _fl_dsigma_imag_dynamic,
                spec,
                queries,
            )
        )
        identity_bound: Float64[NDArray, " n"] = (
            _IDENTITY_SCALE_BOUND * np.maximum(1.0, np.abs(composite))
        )
        assert np.all(np.abs(forward - composite) <= identity_bound)
        assert np.all(np.abs(reverse - composite) <= identity_bound)
        assert np.all(np.abs(forward - reverse) <= identity_bound)

        analytic: Float64[NDArray, " n"] = _fl_dsigma_real_domega(
            np.asarray(queries)
        )
        mixed_bound: Float64[NDArray, " n"] = (
            _FL_VALUE_ATOL_EV + _FL_VALUE_RTOL * np.abs(analytic)
        )
        assert np.all(np.abs(forward - analytic) <= mixed_bound)
        assert np.all(np.abs(reverse - analytic) <= mixed_bound)

        imag_tangent: Float64[NDArray, " n"] = np.imag(np.asarray(tangent))
        np.testing.assert_allclose(
            imag_tangent,
            np.asarray(_fl_dsigma_imag_dynamic(queries)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_parameter_derivative_rows_match_frozen_complex_step_truths(
        self,
    ) -> None:
        """Match every frozen parameter-derivative row on the public map.

        Acceptance: for each dimensionless coordinate ``q_p = p / p_fix``
        the public forward-mode derivative matches the frozen complex-step
        rows at rtol ``1e-6``. The numerical Fermi-liquid real rows carry
        atol ``5e-8`` (committed instrument measurement ``2.579e-8``).
        Every analytic row carries atol ``1e-10``. A separate test covers
        the structural-zero columns.

        Notes
        -----
        The test rescales each fixture through its dimensionless
        coordinates and differentiates the public map per column. It then
        compares real and imaginary rows against the archive.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        cases: Tuple[Tuple[str, str, int], ...] = (
            ("fermi_liquid", "fl", 3),
            ("bosonic_kink", "kink", 4),
        )
        fixture: str
        prefix: str
        count: int
        column: int
        for fixture, prefix, count in cases:
            probes: Float64[Array, " n"] = jnp.asarray(
                models[f"{prefix}_probe_frequencies_ev"],
                dtype=jnp.float64,
            )
            real_rows: Float64[NDArray, " n n_param"] = models[
                f"{prefix}_dsigma_real_dq"
            ]
            imag_rows: Float64[NDArray, " n n_param"] = models[
                f"{prefix}_dsigma_imag_dq"
            ]

            def scaled_map(
                scale: Float64[Array, " n_param"],
                fixture_name: str = fixture,
                points: Float64[Array, " n"] = probes,
            ) -> Complex128[Array, " n"]:
                returned: Complex128[Array, " n"] = evaluate_self_energy(
                    points, _scaled_model(fixture_name, scale)
                )
                return returned

            ones: Float64[Array, " n_param"] = jnp.ones(
                count, dtype=jnp.float64
            )
            primal: Complex128[Array, " n"] = scaled_map(ones)
            assert jnp.iscomplexobj(primal), (
                "the parameter rows run on the complex retarded contract"
            )
            for column in range(count):
                tangent_direction: Float64[Array, " n_param"] = (
                    jnp.zeros(count, dtype=jnp.float64).at[column].set(1.0)
                )
                tangent: Complex128[Array, " n"]
                _, tangent = jax.jvp(scaled_map, (ones,), (tangent_direction,))
                real_atol: float = (
                    _FL_REAL_ROW_ATOL
                    if fixture == "fermi_liquid"
                    else _ANALYTIC_ROW_ATOL
                )
                if column > 0:
                    np.testing.assert_allclose(
                        np.real(np.asarray(tangent)),
                        real_rows[:, column],
                        rtol=_DERIVATIVE_ROW_RTOL,
                        atol=real_atol,
                    )
                np.testing.assert_allclose(
                    np.imag(np.asarray(tangent)),
                    imag_rows[:, column],
                    rtol=_DERIVATIVE_ROW_RTOL,
                    atol=_ANALYTIC_ROW_ATOL,
                )

    def test_structural_zero_gamma0_real_response_is_exactly_zero(
        self,
    ) -> None:
        """Assert the documented structural zeros with exact-zero rows.

        Acceptance: ``dSigma'/dGamma0`` equals zero exactly for both
        fixtures. The constant baseline stays imaginary-only, and the
        subtraction removes it before the Kramers-Kronig map. The rows
        carry exact-zero assertions, never nonzero tripwires.

        Notes
        -----
        The test differentiates the public map along the ``gamma0``
        coordinate for both fixtures. It then requires an exactly zero
        real tangent row.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        fixture: str
        prefix: str
        count: int
        for fixture, prefix, count in (
            ("fermi_liquid", "fl", 3),
            ("bosonic_kink", "kink", 4),
        ):
            probes: Float64[Array, " n"] = jnp.asarray(
                models[f"{prefix}_probe_frequencies_ev"],
                dtype=jnp.float64,
            )

            def scaled_map(
                scale: Float64[Array, " n_param"],
                fixture_name: str = fixture,
                points: Float64[Array, " n"] = probes,
            ) -> Complex128[Array, " n"]:
                returned: Complex128[Array, " n"] = evaluate_self_energy(
                    points, _scaled_model(fixture_name, scale)
                )
                return returned

            ones: Float64[Array, " n_param"] = jnp.ones(
                count, dtype=jnp.float64
            )
            gamma_direction: Float64[Array, " n_param"] = (
                jnp.zeros(count, dtype=jnp.float64).at[0].set(1.0)
            )
            primal: Complex128[Array, " n"]
            tangent: Complex128[Array, " n"]
            primal, tangent = jax.jvp(scaled_map, (ones,), (gamma_direction,))
            assert jnp.iscomplexobj(primal), (
                "the structural zeros run on the complex retarded contract"
            )
            np.testing.assert_array_equal(
                np.real(np.asarray(tangent)),
                np.zeros(probes.shape[0]),
            )


class TestSelfEnergyModelFactoryGuards(chex.TestCase):
    """Test the remaining carrier factory guard.

    :see: :func:`diffpes.types.make_self_energy_model`
    """

    def test_gamma_with_explicit_coefficients_is_rejected(self) -> None:
        """Reject a ``gamma`` shortcut next to explicit coefficients.

        Acceptance: the factory accepts the ``gamma`` shortcut only when
        ``coefficients`` stays absent and ``mode='constant'``. An explicit
        ``gamma`` together with ``coefficients`` raises instead of a
        silent drop.

        Notes
        -----
        The test drives the shared rejection helper with both arguments
        supplied. The helper repeats the call under ``eqx.filter_jit``.
        """
        assert_rejects(
            make_self_energy_model,
            coefficients=jnp.asarray([-2.0]),
            gamma=0.2,
            match="gamma",
        )


class TestGridModeHatTransform(chex.TestCase):
    """Test the exact grid-mode hat-interpolant transform.

    :see: :func:`diffpes.simul.retarded_self_energy.evaluate_self_energy`
    """

    def test_three_node_hat_evaluation_matches_closed_form_transform(
        self,
    ) -> None:
        """Match a hand-computed closed-form hat principal value.

        Acceptance: for a three-node hat carrier the grid-mode real part
        equals the exact segment-wise principal value plus the frozen
        256-node power2 tail quadrature within
        ``1e-12 + 1e-12 * |truth|``. The committed selection evidence
        measures ``3.886e-16 eV`` for this operator class. The hat tail
        slopes come from the hat interpolant's edge segments.

        Notes
        -----
        The test computes the closed-form truth with NumPy only, using the
        documented per-segment formula and the frozen tail rule. It then
        compares the production grid-mode output per query.
        """
        nodes: Float64[NDArray, " three"] = np.asarray([-4.0, 0.0, 4.0])
        raw: Float64[NDArray, " three"] = np.asarray([-1.1, 0.4, -0.7])
        ordinates: Float64[NDArray, " three"] = -np.logaddexp(raw, 0.0)
        tail_raw: Tuple[float, float] = (-2.0, -1.5)
        subtraction_point: float = 0.3
        queries: Float64[NDArray, " four"] = np.asarray(
            [-1.3, -0.45, 0.62, 1.85]
        )

        amplitude_left: float = float(-ordinates[0])
        amplitude_right: float = float(-ordinates[-1])
        slope_left: float = float(
            (ordinates[1] - ordinates[0]) / (nodes[1] - nodes[0])
        )
        slope_right: float = float(
            (ordinates[2] - ordinates[1]) / (nodes[2] - nodes[1])
        )
        alpha_left: float = -slope_left / amplitude_left
        alpha_right: float = slope_right / amplitude_right
        beta_left: float = alpha_left**2 / 4.0 + float(
            np.logaddexp(tail_raw[0], 0.0)
        )
        beta_right: float = alpha_right**2 / 4.0 + float(
            np.logaddexp(tail_raw[1], 0.0)
        )

        evaluation_points: Float64[NDArray, " five"] = np.concatenate(
            [queries, np.asarray([subtraction_point])]
        )
        core: Float64[NDArray, " five"] = _hat_core_pv(
            nodes, ordinates, evaluation_points
        )
        tails: Float64[NDArray, " five"] = _hand_power2_tail(
            (float(nodes[0]), float(nodes[-1])),
            (amplitude_left, amplitude_right),
            (alpha_left, alpha_right),
            (beta_left, beta_right),
            evaluation_points,
        )
        total: Float64[NDArray, " five"] = core + tails
        truth: Float64[NDArray, " four"] = total[:-1] - total[-1]

        model: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray(raw),
            mode="grid",
            energy_nodes_rel_fermi_ev=jnp.asarray(nodes),
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
            tail_coefficients=jnp.asarray(tail_raw),
            subtraction_point_rel_fermi_ev=subtraction_point,
            tail_mode="power2",
        )
        sigma: Complex128[Array, " four"] = evaluate_self_energy(
            jnp.asarray(queries), model
        )
        assert jnp.iscomplexobj(sigma), (
            "grid-mode evaluation must return the complex retarded self-energy"
        )
        error: Float64[NDArray, " four"] = np.abs(
            np.real(np.asarray(sigma)) - truth
        )
        assert np.all(
            error
            <= _GRID_EXACTNESS_ATOL_EV + _GRID_EXACTNESS_RTOL * np.abs(truth)
        )
        np.testing.assert_allclose(
            np.imag(np.asarray(sigma)),
            np.interp(queries, nodes, ordinates),
            rtol=0.0,
            atol=1e-14,
        )
