"""Validate the private principal-value module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

from types import ModuleType

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Dict, List
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import (
    voigt,
)
from diffpes.utils import faddeeva

from ._spectral_helpers import (
    _ANALYTIC_ARCHIVE_SHA256,
    _ANALYTIC_DIRECTORY,
    _ANALYTIC_GENERATOR_SHA256,
    _ANALYTIC_MANIFEST_SHA256,
    _COMMON_MODULE_SHA256,
    _COMPOSITE_DERIVATIVE_BOUND,
    _COMPOSITE_EXPECTED_MAX_ERROR,
    _CONTROL_MODULE_SHA256,
    _CUBIC_MODULE_SHA256,
    _FL_PARAMETERS_PHYSICAL,
    _KINK_PARAMETERS_PHYSICAL,
    _LINEAR_MODULE_SHA256,
    _MODELS_ARCHIVE_PATH,
    _MODELS_ARCHIVE_SHA256,
    _MODELS_GENERATOR_SHA256,
    _MODELS_MANIFEST_PATH,
    _MODELS_MANIFEST_SHA256,
    _PAIR_TRUTH_ATOL_EV,
    _PAIR_TRUTH_EXPECTED_MAX_ERROR_EV,
    _PAIR_TRUTH_EXPECTED_MAX_RATIO,
    _PAIR_TRUTH_RTOL,
    _POLE_COUPLING_EV2,
    _POLE_GAMMA_EV,
    _POLE_OMEGA0_EV,
    _QUADRATIC_MODULE_SHA256,
    _SELECTION_ARCHIVE_PATH,
    _SELECTION_ARCHIVE_SHA256,
    _SELECTION_GENERATOR_SHA256,
    _SELECTION_MANIFEST_PATH,
    _SELECTION_MANIFEST_SHA256,
    _TAIL_RULE_BOUND,
    _TOOLS_DIRECTORY,
    _authenticated_json,
    _authenticated_npz,
    _committed_operator,
    _instrument_composite_derivative,
    _instrument_subtracted,
    _pole_dsigma_imag,
    _pole_sigma_imag,
    _pole_tail_spec,
    _sha256,
)


class TestKramersKronigEvidence(chex.TestCase):
    """Validate the frozen independent artifacts before production edits.

    The cases authenticate the manifests and replay analytic causal models,
    selected operators, frozen outputs, and the certified Faddeeva envelope.
    """

    def test_reference_manifests_and_archives_are_authenticated(
        self,
    ) -> None:
        """Verify every truth manifest, archive, and generator digest.

        The test recomputes each SHA-256 digest and compares it against
        its frozen pin. It also checks each manifest against the digest of
        its own archive.

        Notes
        -----
        The test reads the three manifests, the three archives, both
        generators, and the committed instrument modules. It then compares
        the registered budget constants against the frozen module values.
        """
        selection: Dict[str, Any] = _authenticated_json(
            _SELECTION_MANIFEST_PATH, _SELECTION_MANIFEST_SHA256
        )
        assert selection["schema"] == "diffpes.kk-operator-selection.v1"
        assert selection["archive_sha256"] == _SELECTION_ARCHIVE_SHA256
        assert _sha256(_SELECTION_ARCHIVE_PATH) == _SELECTION_ARCHIVE_SHA256
        executable: Dict[str, Any] = selection["executable_inputs"]
        assert executable["generator"]["sha256"] == _SELECTION_GENERATOR_SHA256
        assert (
            _sha256(_TOOLS_DIRECTORY / "generate_kk_operator_selection.py")
            == _SELECTION_GENERATOR_SHA256
        )
        assert executable["common"]["sha256"] == _COMMON_MODULE_SHA256
        assert (
            _sha256(_TOOLS_DIRECTORY / "_kk_candidate_common.py")
            == _COMMON_MODULE_SHA256
        )
        assert executable["piecewise_cubic"]["sha256"] == _CUBIC_MODULE_SHA256
        assert (
            executable["piecewise_linear"]["sha256"] == _LINEAR_MODULE_SHA256
        )
        assert (
            executable["piecewise_quadratic"]["sha256"]
            == _QUADRATIC_MODULE_SHA256
        )
        assert (
            executable["maclaurin_control"]["sha256"] == _CONTROL_MODULE_SHA256
        )
        name: str
        digest: str
        for name, digest in (
            ("_kk_candidate_piecewise_cubic.py", _CUBIC_MODULE_SHA256),
            ("_kk_candidate_piecewise_linear.py", _LINEAR_MODULE_SHA256),
            ("_kk_candidate_piecewise_quadratic.py", _QUADRATIC_MODULE_SHA256),
            (
                "_kk_control_opposite_parity_maclaurin.py",
                _CONTROL_MODULE_SHA256,
            ),
        ):
            assert _sha256(_TOOLS_DIRECTORY / name) == digest
        assert selection["production_imports"] == []

        analytic: Dict[str, Any] = _authenticated_json(
            _ANALYTIC_DIRECTORY / "manifest.json", _ANALYTIC_MANIFEST_SHA256
        )
        assert analytic["schema"] == "diffpes.kk-analytic-reference.v1"
        assert analytic["archive_sha256"] == _ANALYTIC_ARCHIVE_SHA256
        assert (
            _sha256(_ANALYTIC_DIRECTORY / analytic["archive"])
            == _ANALYTIC_ARCHIVE_SHA256
        )
        assert (
            _sha256(_ANALYTIC_DIRECTORY / analytic["generator"])
            == _ANALYTIC_GENERATOR_SHA256
        )
        assert analytic["generator_sha256"] == _ANALYTIC_GENERATOR_SHA256
        assert (
            executable["analytic_arbiter"]["generator_sha256"]
            == _ANALYTIC_GENERATOR_SHA256
        )
        assert (
            executable["analytic_arbiter"]["manifest_sha256"]
            == _ANALYTIC_MANIFEST_SHA256
        )
        assert int(analytic["arbiter"]["decimal_digits"]) == 80

        models: Dict[str, Any] = _authenticated_json(
            _MODELS_MANIFEST_PATH, _MODELS_MANIFEST_SHA256
        )
        assert models["schema"] == "diffpes.self-energy-models-reference.v1"
        recorded: str = models["archives"]["self_energy_models_reference"][
            "sha256"
        ]
        assert recorded == _MODELS_ARCHIVE_SHA256
        assert _sha256(_MODELS_ARCHIVE_PATH) == _MODELS_ARCHIVE_SHA256
        assert models["generator_sha256"] == _MODELS_GENERATOR_SHA256
        assert (
            _sha256(
                _TOOLS_DIRECTORY / "generate_self_energy_models_reference.py"
            )
            == _MODELS_GENERATOR_SHA256
        )
        zeros: List[str] = models["derivatives"]["structural_zeros"]
        assert any("dSigma'/dGamma0" in entry for entry in zeros)

        budgets: Dict[str, float] = selection["registered_budgets"]
        assert budgets["pair_truth_atol_ev"] == _PAIR_TRUTH_ATOL_EV
        assert budgets["pair_truth_rtol"] == _PAIR_TRUTH_RTOL
        assert budgets["pole_tail_only_max_delta_sigma_ev"] == _TAIL_RULE_BOUND
        assert budgets["pole_tail_only_max_delta_dsigma"] == _TAIL_RULE_BOUND

    def test_operator_selection_measurements_replicate_frozen_numbers(
        self,
    ) -> None:
        """Reproduce the committed acceptance measurements from the archive.

        The frozen operator meets the per-row pair-truth mixed criterion
        ``|err| <= 2e-8 + 1e-6 * |truth|``. The recomputed maximum error
        equals ``3.393963743381079e-8 eV`` with worst ratio
        ``0.8239596852267533``. The recomputed composite query-derivative
        error equals ``9.342723950034326e-7``.

        Notes
        -----
        The test loads the frozen arrays, recomputes each measurement, and
        compares it against the recorded manifest number. It also checks
        the 256-to-512 tail rule deltas against the ``1e-13`` budget.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        analytic: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _ANALYTIC_DIRECTORY / "kk_reference.npz",
            _ANALYTIC_ARCHIVE_SHA256,
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        truth: Float64[NDArray, " n"] = selection[
            "truth_pole_sigma_real_sub_ev"
        ]
        np.testing.assert_array_equal(queries, analytic["pole_omega"])
        np.testing.assert_array_equal(truth, analytic["pole_sigma_real"])

        offset: Float64[NDArray, " n"] = queries - _POLE_OMEGA0_EV
        closed: Float64[NDArray, " n"] = (
            _POLE_COUPLING_EV2 * offset / (offset**2 + _POLE_GAMMA_EV**2)
        ) - (
            _POLE_COUPLING_EV2
            * (0.0 - _POLE_OMEGA0_EV)
            / ((0.0 - _POLE_OMEGA0_EV) ** 2 + _POLE_GAMMA_EV**2)
        )
        assert np.max(np.abs(closed - truth)) <= 5.0e-16

        error: Float64[NDArray, " n"] = np.abs(
            selection["pwcubic_pole_base_sigma_sub_ev"] - truth
        )
        bound: Float64[NDArray, " n"] = (
            _PAIR_TRUTH_ATOL_EV + _PAIR_TRUTH_RTOL * np.abs(truth)
        )
        assert int(np.sum(error > bound)) == 0
        np.testing.assert_allclose(
            np.max(error), _PAIR_TRUTH_EXPECTED_MAX_ERROR_EV, rtol=1e-12
        )
        np.testing.assert_allclose(
            np.max(error / bound),
            _PAIR_TRUTH_EXPECTED_MAX_RATIO,
            rtol=1e-12,
        )

        composite_error: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_base_dsigma_composite"]
                    - selection["truth_pole_dsigma_domega"]
                )
            )
        )
        np.testing.assert_allclose(
            composite_error, _COMPOSITE_EXPECTED_MAX_ERROR, rtol=1e-12
        )
        assert composite_error <= _COMPOSITE_DERIVATIVE_BOUND

        tail_value_delta: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_tail512_sigma_sub_ev"]
                    - selection["pwcubic_pole_base_sigma_sub_ev"]
                )
            )
        )
        tail_derivative_delta: float = float(
            np.max(
                np.abs(
                    selection["pwcubic_pole_tail512_dsigma_composite"]
                    - selection["pwcubic_pole_base_dsigma_composite"]
                )
            )
        )
        assert tail_value_delta <= _TAIL_RULE_BOUND
        assert tail_derivative_delta <= _TAIL_RULE_BOUND

    def test_causal_model_truth_archive_replicates_closed_forms(
        self,
    ) -> None:
        """Reproduce the frozen causal-model truths from their formulas.

        The raw coordinates reproduce the physical parameters through
        softplus exactly. Both imaginary-part grids replicate their closed
        forms. The Fermi-liquid subtracted real part replicates its
        analytic identity. The documented structural-zero columns
        ``dSigma'/dGamma0`` equal zero exactly for both fixtures.

        Notes
        -----
        The test evaluates each closed form on the frozen grids with
        NumPy. It then compares the results against the archive rows and
        checks the recorded complex-step cross-check scalars.
        """
        models: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _MODELS_ARCHIVE_PATH, _MODELS_ARCHIVE_SHA256
        )
        fl_raw: Float64[NDArray, " three"] = models["fl_parameters_raw"]
        np.testing.assert_allclose(
            np.logaddexp(fl_raw, 0.0),
            np.asarray(_FL_PARAMETERS_PHYSICAL),
            rtol=1e-15,
        )
        kink_raw: Float64[NDArray, " four"] = models["kink_parameters_raw"]
        np.testing.assert_allclose(
            np.logaddexp(kink_raw, 0.0),
            np.asarray(_KINK_PARAMETERS_PHYSICAL),
            rtol=1e-15,
        )

        grid: Float64[NDArray, " n"] = models["eval_grid_ev"]
        gamma0: float
        beta: float
        omega_c: float
        gamma0, beta, omega_c = _FL_PARAMETERS_PHYSICAL
        fl_imag: Float64[NDArray, " n"] = -gamma0 - beta * grid**2 / (
            1.0 + (grid / omega_c) ** 4
        )
        np.testing.assert_allclose(
            fl_imag, models["fl_sigma_imag_grid"], rtol=1e-13, atol=1e-16
        )
        fl_real: Float64[NDArray, " n"] = (
            (np.sqrt(2.0) / 2.0)
            * beta
            * omega_c**3
            * grid
            * (grid**2 - omega_c**2)
            / (grid**4 + omega_c**4)
        )
        np.testing.assert_allclose(
            fl_real,
            models["fl_sigma_real_subtracted_analytic"],
            rtol=1e-13,
            atol=1e-16,
        )

        k_gamma0: float
        k_g: float
        k_omega0: float
        k_width: float
        k_gamma0, k_g, k_omega0, k_width = _KINK_PARAMETERS_PHYSICAL
        pair: Complex128[NDArray, " n"] = k_g**2 * (
            1.0 / (grid - k_omega0 + 1j * k_width)
            + 1.0 / (grid + k_omega0 + 1j * k_width)
        )
        np.testing.assert_allclose(
            -k_gamma0 + np.imag(pair),
            models["kink_sigma_imag_grid"],
            rtol=1e-13,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            np.real(pair),
            models["kink_sigma_real_grid"],
            rtol=1e-13,
            atol=1e-16,
        )

        np.testing.assert_array_equal(
            models["fl_dsigma_real_dq"][:, 0],
            np.zeros_like(models["fl_probe_frequencies_ev"]),
        )
        np.testing.assert_array_equal(
            models["kink_dsigma_real_dq"][:, 0],
            np.zeros_like(models["kink_probe_frequencies_ev"]),
        )
        assert (
            float(models["fl_derivative_crosscheck_max_abs_error"]) <= 1.0e-10
        )
        assert (
            float(models["kink_derivative_crosscheck_max_abs_error"])
            <= 1.0e-10
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_committed_instrument_replicates_frozen_operator_outputs(
        self,
    ) -> None:
        """Reproduce the frozen operator outputs with the instrument.

        The authenticated instrument modules reproduce the frozen
        subtracted values and the frozen composite derivative on the pole
        fixture. This certifies the in-file composite-route truth for the
        red autodiff demonstrator.

        Notes
        -----
        The test rebuilds the tail contract from the recorded raw
        coordinates and the committed edge stencils. It then compares both
        instrument outputs against the frozen arrays at ``1e-15``.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )
        spec: Any = _pole_tail_spec(common)
        values: Float64[Array, " n"] = _instrument_subtracted(
            _pole_sigma_imag, spec, queries, 0.0
        )
        np.testing.assert_allclose(
            np.asarray(values),
            selection["pwcubic_pole_base_sigma_sub_ev"],
            rtol=0.0,
            atol=1e-15,
        )
        derivative: Float64[Array, " n"] = _instrument_composite_derivative(
            _pole_sigma_imag, _pole_dsigma_imag, spec, queries
        )
        np.testing.assert_allclose(
            np.asarray(derivative),
            selection["pwcubic_pole_base_dsigma_composite"],
            rtol=0.0,
            atol=1e-15,
        )

    def test_positive_width_voigt_consumes_certified_faddeeva_envelope(
        self,
    ) -> None:
        """Require the positive-width Voigt path to use the Faddeeva map.

        The strictly positive branch must equal
        ``Re w((E - E0 + i*gamma) / (sigma*sqrt(2))) / (sigma*sqrt(2*pi))``
        with the certified :func:`diffpes.utils.faddeeva` envelope. The
        check guards this composition when :mod:`diffpes.simul.spectral`
        lands next to the broadening module.

        Notes
        -----
        The test composes the Faddeeva map directly and compares the
        production Voigt values against the composition at ``1e-13``.
        """
        energy: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 41)
        center: float = 0.137
        sigma: float = 0.04
        gamma: float = 0.1
        argument: Complex128[Array, " n"] = (energy - center + 1j * gamma) / (
            sigma * jnp.sqrt(2.0)
        )
        composed: Float64[Array, " n"] = jnp.real(faddeeva(argument)) / (
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )
        produced: Float64[Array, " n"] = voigt(energy, center, sigma, gamma)
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(composed),
            rtol=1e-13,
            atol=1e-16,
        )
