"""Validate the private Kramers--Kronig module.

The cases use analytic values, invariants, and finite differences.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Callable, Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.simul import _kramers_kronig, evaluate_self_energy
from diffpes.simul._kramers_kronig import _kk_transform
from diffpes.simul._principal_value import (
    _cubic_core_pv,
    _cubic_edge_slopes,
    _power2_spec_from_edges,
)
from diffpes.types import (
    SelfEnergyModel,
    make_self_energy_model,
)
from tests._assertions import assert_rejects

from ._spectral_helpers import (
    _COMPOSITE_DERIVATIVE_BOUND,
    _DOMAIN_HIGH_EV,
    _DOMAIN_LOW_EV,
    _N_KK,
    _N_TAIL,
    _PAIR_TRUTH_ATOL_EV,
    _PAIR_TRUTH_RTOL,
    _POLE_GAMMA_EV,
    _POLE_OMEGA0_EV,
    _POLE_TAIL_RAW,
    _SELECTION_ARCHIVE_PATH,
    _SELECTION_ARCHIVE_SHA256,
    _authenticated_npz,
    _base_grid,
    _committed_operator,
    _pole_sigma_imag,
    _pole_tail_spec,
)


class TestKkTransformSeam(chex.TestCase):
    """Test the private cell-integrated transform seam.

    :see: :func:`diffpes.simul._kramers_kronig._kk_transform`
    """

    def test_private_transform_seam_signature_and_no_kernel_matrix(
        self,
    ) -> None:
        """Require the committed seam and forbid the kernel-matrix API.

        Acceptance: ``diffpes.simul._kramers_kronig._kk_transform`` exists with
        the exact parameters ``(core_grid, model_domain, tail_spec,
        queries, n_tail)``. The module defines no ``build_kk_kernel``
        dense ``[n_kk, n_kk]`` constructor.

        Notes
        -----
        The test imports the production module, inspects the seam
        signature, and checks the retired kernel name stays absent.
        """
        seam: Callable[..., Any] = _kk_transform
        parameters: List[str] = list(inspect.signature(seam).parameters)
        assert parameters == [
            "core_grid",
            "model_domain",
            "tail_spec",
            "queries",
            "n_tail",
        ]
        assert not hasattr(_kramers_kronig, "build_kk_kernel")

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_retarded_pole_real_part_matches_analytic_truth(self) -> None:
        """Match the 80-digit pole truth through the production seam.

        Acceptance: on the 1001 frozen queries with the committed base
        configuration, ``|err| <= 2e-8 + 1e-6 * |truth|`` per query row.
        The committed operator measures maximum error ``3.3940e-8 eV``
        with worst mixed ratio ``0.8240`` here. ``core_grid`` carries the
        frozen node positions and the sampled imaginary part.

        Notes
        -----
        The test samples the analytic pole on the frozen grid, attaches
        the committed tail contract, and subtracts at the frozen point. It
        then applies the registered mixed criterion per row.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )
        stacked: Float64[Array, " m"] = jnp.concatenate(
            [queries, jnp.asarray([0.0], dtype=jnp.float64)]
        )
        transformed: Float64[Array, " m"] = _kk_transform(
            (grid, _pole_sigma_imag(grid)),
            domain,
            spec,
            stacked,
            _N_TAIL,
        )
        subtracted: Float64[NDArray, " n"] = np.asarray(
            transformed[:-1] - transformed[-1]
        )
        truth: Float64[NDArray, " n"] = selection[
            "truth_pole_sigma_real_sub_ev"
        ]
        error: Float64[NDArray, " n"] = np.abs(subtracted - truth)
        assert np.all(
            error <= _PAIR_TRUTH_ATOL_EV + _PAIR_TRUTH_RTOL * np.abs(truth)
        )

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_retarded_pole_query_derivative_follows_composite_route(
        self,
    ) -> None:
        """Require the composite-route class from the seam derivative.

        Acceptance: forward-mode differentiation of the seam in the query
        coordinate matches the analytic pole derivative within ``2e-6``.
        The committed composite route measures ``9.3427e-7`` on the base
        configuration. Direct differentiation of the transform measures
        ``8.87e-5`` and fails this bound.

        Notes
        -----
        The test differentiates the seam with a unit query tangent on the
        1001 frozen queries. It then compares against the frozen analytic
        derivative array.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.asarray(
            selection["queries_ev"], dtype=jnp.float64
        )

        def seam_map(
            points: Float64[Array, " n"],
        ) -> Float64[Array, " n"]:
            returned: Float64[Array, " n"] = _kk_transform(
                (grid, _pole_sigma_imag(grid)),
                domain,
                spec,
                points,
                _N_TAIL,
            )
            return returned

        derivative: Float64[Array, " n"]
        _, derivative = jax.jvp(
            seam_map, (queries,), (jnp.ones_like(queries),)
        )
        error: float = float(
            np.max(
                np.abs(
                    np.asarray(derivative)
                    - selection["truth_pole_dsigma_domega"]
                )
            )
        )
        assert error <= _COMPOSITE_DERIVATIVE_BOUND


class TestPlantedNoncompliantConstructions(chex.TestCase):
    """Test rejection of the planted noncompliant constructions.

    :see: :func:`diffpes.simul._kramers_kronig._kk_transform`
    """

    def _seam_arguments(
        self,
    ) -> Tuple[
        Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
        Float64[Array, " 2"],
        Any,
        Float64[Array, " n"],
    ]:
        """PRIVATE: Return one compliant seam argument set for mutation.

        Notes
        -----
        The set reuses the pole fixture and the committed tail contract.
        """
        common: ModuleType
        common, _ = _committed_operator()
        grid: Float64[Array, " n_kk"] = _base_grid()
        spec: Any = _pole_tail_spec(common)
        domain: Float64[Array, " 2"] = jnp.asarray(
            [_DOMAIN_LOW_EV, _DOMAIN_HIGH_EV], dtype=jnp.float64
        )
        queries: Float64[Array, " n"] = jnp.linspace(-1.0, 1.0, 11)
        returned: Tuple[
            Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]],
            Float64[Array, " 2"],
            Any,
            Float64[Array, " n"],
        ] = (grid, _pole_sigma_imag(grid)), domain, spec, queries
        return returned

    def test_discontinuous_tail_edge_value_is_rejected(self) -> None:
        """Reject a tail whose edge value breaks the C1 match.

        Acceptance: a planted tail amplitude away from the core edge
        sample violates the continuity contract. Production raises on this
        construction.

        Notes
        -----
        The test scales one committed tail amplitude by 1.05 and drives
        the shared rejection helper on the seam.
        """
        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        planted: Any = spec._replace(amplitude_left=spec.amplitude_left * 1.05)
        assert_rejects(
            _kk_transform,
            core_grid,
            domain,
            planted,
            queries,
            _N_TAIL,
            match="edge",
        )

    def test_sign_crossing_tail_denominator_is_rejected(self) -> None:
        """Reject a tail denominator that can cross zero.

        Acceptance: a planted negative quadratic tail coefficient lets
        ``1 + alpha*t + beta*t**2`` cross zero. Production raises on this
        construction.

        Notes
        -----
        The test replaces one committed ``beta`` with ``-0.01`` and drives
        the shared rejection helper on the seam.
        """
        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        planted: Any = spec._replace(
            beta_left=jnp.asarray(-0.01, dtype=jnp.float64)
        )
        assert_rejects(
            _kk_transform,
            core_grid,
            domain,
            planted,
            queries,
            _N_TAIL,
            match="denominator|positive",
        )

    def test_silently_truncated_tail_is_rejected(self) -> None:
        """Reject a truncated semi-infinite tail quadrature.

        Acceptance: a zero tail order truncates both semi-infinite
        contributions. Production raises instead of evaluating the
        truncated transform.

        Notes
        -----
        The test passes ``n_tail=0`` and drives the shared rejection
        helper on the seam.
        """
        core_grid: Tuple[Float64[Array, " n_kk"], Float64[Array, " n_kk"]]
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        core_grid, domain, spec, queries = self._seam_arguments()
        assert_rejects(
            _kk_transform,
            core_grid,
            domain,
            spec,
            queries,
            0,
            match="n_tail|tail",
        )

    def test_query_window_built_grid_is_rejected(self) -> None:
        """Reject a quadrature grid built from the query extrema.

        Acceptance: a planted core grid that spans only the query window
        contradicts the declared model domain. Production raises because
        the domain always comes from the carrier.

        Notes
        -----
        The test rebuilds the grid from the query extrema and drives the
        shared rejection helper on the seam. The planted grid violates
        the domain check and the frozen tail-edge check together. The
        compiled program reports one of the two traced rejections, and
        the order depends on the XLA schedule.
        """
        domain: Float64[Array, " 2"]
        spec: Any
        queries: Float64[Array, " n"]
        _, domain, spec, queries = self._seam_arguments()
        window_grid: Float64[Array, " n_kk"] = jnp.linspace(
            float(jnp.min(queries)), float(jnp.max(queries)), _N_KK
        )
        assert_rejects(
            _kk_transform,
            (window_grid, _pole_sigma_imag(window_grid)),
            domain,
            spec,
            queries,
            _N_TAIL,
            match="domain|grid|tail edge",
        )


class TestProductionKkConvergence(chex.TestCase):
    """Run the frozen refinement battery through production operators.

    :see: :func:`diffpes.simul._kramers_kronig._kk_transform`
    """

    @staticmethod
    def _uniform_grid(
        low: float, high: float, count: int
    ) -> Float64[NDArray, " n"]:
        """PRIVATE: Construct the frozen index-defined uniform grid.

        Notes
        -----
        The construction matches the selection instrument exactly and avoids
        endpoint redistribution by ``linspace``.
        """
        spacing: float = (high - low) / (count - 1)
        returned: Float64[NDArray, " n"] = (
            low + np.arange(count, dtype=np.float64) * spacing
        )
        return returned

    @staticmethod
    def _production_core_only_subtracted(
        grid_np: Float64[NDArray, " n_kk"],
        queries_np: Float64[NDArray, " n_query"],
    ) -> Float64[NDArray, " n_query"]:
        """PRIVATE: Evaluate the production cubic under a zero-tail contract.

        Notes
        -----
        The Wigner fixture has compact support and exactly zero edge values on
        every registered domain. The specification explicitly routes this
        test-only analytic exception through the production cubic core, not
        through the positive-amplitude ``power2`` tail seam. Query chunks keep
        the matrix-free working set proportional to ``chunk * n_kk``.
        """
        grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
        half_width: float = 1.5
        coupling: float = 0.2
        scale: float = 2.0 * coupling / half_width**2
        radicand: Float64[Array, " n_kk"] = jnp.maximum(
            half_width**2 - grid**2, 0.0
        )
        values: Float64[Array, " n_kk"] = jnp.where(
            jnp.abs(grid) < half_width,
            -scale * jnp.sqrt(radicand),
            0.0,
        )
        subtraction: float = float(
            _cubic_core_pv(
                grid,
                values,
                jnp.asarray([0.0], dtype=jnp.float64),
            )[0]
        )
        chunks: List[Float64[NDArray, " chunk"]] = []
        start: int
        chunk_size: int = 64
        for start in range(0, queries_np.shape[0], chunk_size):
            points: Float64[Array, " chunk"] = jnp.asarray(
                queries_np[start : start + chunk_size]
            )
            chunks.append(
                np.asarray(_cubic_core_pv(grid, values, points)) - subtraction
            )
        returned: Float64[NDArray, " n_query"] = np.concatenate(chunks)
        return returned

    @staticmethod
    def _pole_tail_raw(
        grid: Float64[Array, " n_kk"],
        *,
        hold_base_raw: bool,
    ) -> Tuple[float, float]:
        """PRIVATE: Return the frozen or domain-recomputed pole tail raws.

        Notes
        -----
        Same-domain refinements retain the base carrier coordinates. The
        phase-aligned domain extension recomputes them from the analytic pole
        before inspecting any production output.
        """
        if hold_base_raw:
            return _POLE_TAIL_RAW
        values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
        spacing: Float64[Array, ""] = grid[1] - grid[0]
        slope_left: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        slope_left, slope_right = _cubic_edge_slopes(values, spacing)
        raw: List[float] = []
        edge: float
        amplitude: float
        alpha: float
        for edge, amplitude, alpha in (
            (
                float(grid[0]),
                float(-values[0]),
                float(-slope_left / (-values[0])),
            ),
            (
                float(grid[-1]),
                float(-values[-1]),
                float(slope_right / (-values[-1])),
            ),
        ):
            del amplitude
            beta_target: float = 1.0 / (
                (edge - _POLE_OMEGA0_EV) ** 2 + _POLE_GAMMA_EV**2
            )
            delta_beta: float = beta_target - alpha**2 / 4.0
            raw.append(float(np.log(np.expm1(delta_beta))))
        returned: Tuple[float, float] = raw[0], raw[1]
        return returned

    @classmethod
    def _production_pole_subtracted(
        cls,
        grid_np: Float64[NDArray, " n_kk"],
        queries_np: Float64[NDArray, " n_query"],
        *,
        n_tail: int,
        hold_base_raw: bool,
    ) -> Float64[NDArray, " n_query"]:
        """PRIVATE: Evaluate one production pole refinement in query chunks.

        Notes
        -----
        The helper constructs the production C1 tail from the registered raw
        carrier coordinates, evaluates the mandatory seam directly, and
        subtracts the independently evaluated zero-frequency value.
        """
        grid: Float64[Array, " n_kk"] = jnp.asarray(grid_np)
        values: Float64[Array, " n_kk"] = _pole_sigma_imag(grid)
        spacing: Float64[Array, ""] = grid[1] - grid[0]
        slope_left: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        slope_left, slope_right = _cubic_edge_slopes(values, spacing)
        raw_left: float
        raw_right: float
        raw_left, raw_right = cls._pole_tail_raw(
            grid, hold_base_raw=hold_base_raw
        )
        spec: Any = _power2_spec_from_edges(
            values[0],
            slope_left,
            values[-1],
            slope_right,
            jnp.asarray(raw_left),
            jnp.asarray(raw_right),
        )
        domain: Float64[Array, " 2"] = jnp.asarray(
            [grid_np[0], grid_np[-1]], dtype=jnp.float64
        )
        subtraction: float = float(
            _kk_transform(
                (grid, values),
                domain,
                spec,
                jnp.asarray([0.0], dtype=jnp.float64),
                n_tail,
            )[0]
        )
        chunks: List[Float64[NDArray, " chunk"]] = []
        start: int
        chunk_size: int = 64
        for start in range(0, queries_np.shape[0], chunk_size):
            points: Float64[Array, " chunk"] = jnp.asarray(
                queries_np[start : start + chunk_size]
            )
            values_chunk: Float64[Array, " chunk"] = _kk_transform(
                (grid, values), domain, spec, points, n_tail
            )
            chunks.append(np.asarray(values_chunk) - subtraction)
        returned: Float64[NDArray, " n_query"] = np.concatenate(chunks)
        return returned

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_wigner_stress_witness_runs_through_production_cubic(self) -> None:
        """Certify Wigner convergence, order, and phase-aligned extension.

        Acceptance: production reproduces every frozen cubic array within
        ``2e-12 eV``. Errors decrease monotonically from 4096 to 8192 to
        16384 nodes, both observed orders reach ``1.4``, and the base error
        stays below ``1e-5 eV``. The phase-aligned extension embeds every
        base node bitwise and leaves the compact-support result unchanged.

        Notes
        -----
        Evaluate the explicit test-only zero-tail exception through
        :func:`diffpes.simul._principal_value._cubic_core_pv`. This is the
        production smooth core for the singularity witness. Routing a zero edge
        through the positive-amplitude power2 carrier violates the contract.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        base_grid: Float64[NDArray, " n_base"] = self._uniform_grid(
            -8.0, 8.0, 4096
        )
        grid_8192: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 8192
        )
        grid_16384: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 16384
        )
        spacing: float = 16.0 / 4095.0
        extension_grid: Float64[NDArray, " n_extended"] = (
            -8.0 + (np.arange(8192, dtype=np.float64) - 2048.0) * spacing
        )
        np.testing.assert_array_equal(extension_grid[2048:6144], base_grid)

        produced: Dict[str, Float64[NDArray, " n"]] = {
            "base": self._production_core_only_subtracted(base_grid, queries),
            "grid8192": self._production_core_only_subtracted(
                grid_8192, queries
            ),
            "grid16384": self._production_core_only_subtracted(
                grid_16384, queries
            ),
            "domain_extension": self._production_core_only_subtracted(
                extension_grid, queries
            ),
        }
        configuration: str
        values: Float64[NDArray, " n"]
        for configuration, values in produced.items():
            np.testing.assert_allclose(
                values,
                selection[f"pwcubic_wigner_{configuration}_sigma_sub_ev"],
                rtol=0.0,
                atol=2.0e-12,
            )
        truth: Float64[NDArray, " n"] = selection[
            "truth_wigner_sigma_real_sub_ev"
        ]
        errors: Tuple[float, float, float] = tuple(
            float(np.max(np.abs(produced[name] - truth)))
            for name in ("base", "grid8192", "grid16384")
        )
        assert errors[0] <= 1.0e-5
        assert errors[0] > errors[1] > errors[2]
        assert np.log2(errors[0] / errors[1]) >= 1.4
        assert np.log2(errors[1] / errors[2]) >= 1.4
        np.testing.assert_allclose(
            produced["domain_extension"],
            produced["base"],
            rtol=0.0,
            atol=2.0e-12,
        )

    @pytest.mark.slow
    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_pole_refinement_domain_and_tail_rules_use_production_seam(
        self,
    ) -> None:
        """Certify the pole value refinements through the production seam.

        Acceptance: each production configuration reproduces its frozen cubic
        array within ``2e-12 eV``. The 4096-to-8192 and phase-aligned-domain
        changes stay below ``2e-6 eV``; the 256-to-512 tail change stays below
        ``1e-13 eV``.

        Notes
        -----
        Same-domain refinement holds raw tail coordinates fixed. Domain
        extension recomputes them from the analytic fixture, exactly as the
        frozen carrier convention requires.
        """
        selection: Dict[str, Float64[NDArray, "..."]] = _authenticated_npz(
            _SELECTION_ARCHIVE_PATH, _SELECTION_ARCHIVE_SHA256
        )
        queries: Float64[NDArray, " n"] = selection["queries_ev"]
        base_grid: Float64[NDArray, " n_base"] = self._uniform_grid(
            -8.0, 8.0, 4096
        )
        grid_8192: Float64[NDArray, " n_refined"] = self._uniform_grid(
            -8.0, 8.0, 8192
        )
        spacing: float = 16.0 / 4095.0
        extension_grid: Float64[NDArray, " n_extended"] = (
            -8.0 + (np.arange(8192, dtype=np.float64) - 2048.0) * spacing
        )
        np.testing.assert_array_equal(extension_grid[2048:6144], base_grid)
        produced: Dict[str, Float64[NDArray, " n"]] = {
            "base": self._production_pole_subtracted(
                base_grid, queries, n_tail=256, hold_base_raw=True
            ),
            "grid8192": self._production_pole_subtracted(
                grid_8192, queries, n_tail=256, hold_base_raw=True
            ),
            "domain_extension": self._production_pole_subtracted(
                extension_grid, queries, n_tail=256, hold_base_raw=False
            ),
            "tail512": self._production_pole_subtracted(
                base_grid, queries, n_tail=512, hold_base_raw=True
            ),
        }
        configuration: str
        values: Float64[NDArray, " n"]
        for configuration, values in produced.items():
            np.testing.assert_allclose(
                values,
                selection[f"pwcubic_pole_{configuration}_sigma_sub_ev"],
                rtol=0.0,
                atol=2.0e-12,
            )
        assert (
            float(np.max(np.abs(produced["grid8192"] - produced["base"])))
            <= 2.0e-6
        )
        assert (
            float(
                np.max(np.abs(produced["domain_extension"] - produced["base"]))
            )
            <= 2.0e-6
        )
        assert (
            float(np.max(np.abs(produced["tail512"] - produced["base"])))
            <= 1.0e-13
        )


class TestProductionKkContinuityAndDerivatives(chex.TestCase):
    """Certify C1 seams, parameter derivatives, JIT, and VMAP success.

    The cases match tail values and slopes at each seam, compare parameter
    Jacobians with finite differences, and exercise compiled vectorization.
    """

    def test_power2_tails_match_production_edge_values_and_slopes(
        self,
    ) -> None:
        """Match smooth and hat tail values at rtol 1e-12 and slopes at 1e-9.

        The test probes both interpolation families at each domain boundary.

        Notes
        -----
        Construct one smooth pole seam and one grid-hat seam with production
        helpers. Evaluate the analytic tail value and outward-coordinate
        derivative at zero distance. Compare both with the owning core
        interpolant.
        """
        smooth_grid: Float64[Array, " n_kk"] = _base_grid()
        smooth_values: Float64[Array, " n_kk"] = _pole_sigma_imag(smooth_grid)
        smooth_left: Float64[Array, ""]
        smooth_right: Float64[Array, ""]
        smooth_left, smooth_right = _cubic_edge_slopes(
            smooth_values, smooth_grid[1] - smooth_grid[0]
        )

        raw: Float64[Array, " four"] = jnp.asarray([-1.1, 0.4, -0.7, -1.0])
        nodes: Float64[Array, " four"] = jnp.asarray([-4.0, -1.2, 1.1, 4.0])
        hat_values: Float64[Array, " four"] = -jnp.logaddexp(raw, 0.0)
        hat_left: Float64[Array, ""] = (hat_values[1] - hat_values[0]) / (
            nodes[1] - nodes[0]
        )
        hat_right: Float64[Array, ""] = (hat_values[-1] - hat_values[-2]) / (
            nodes[-1] - nodes[-2]
        )
        cases: Tuple[
            Tuple[
                Float64[Array, ""],
                Float64[Array, ""],
                Float64[Array, ""],
                Float64[Array, ""],
            ],
            ...,
        ] = (
            (smooth_values[0], smooth_left, smooth_values[-1], smooth_right),
            (hat_values[0], hat_left, hat_values[-1], hat_right),
        )
        edge_left: Float64[Array, ""]
        slope_left: Float64[Array, ""]
        edge_right: Float64[Array, ""]
        slope_right: Float64[Array, ""]
        for edge_left, slope_left, edge_right, slope_right in cases:
            spec: Any = _power2_spec_from_edges(
                edge_left,
                slope_left,
                edge_right,
                slope_right,
                jnp.asarray(-2.0),
                jnp.asarray(-1.5),
            )
            np.testing.assert_allclose(
                [-float(spec.amplitude_left), -float(spec.amplitude_right)],
                [float(edge_left), float(edge_right)],
                rtol=1.0e-12,
                atol=0.0,
            )
            np.testing.assert_allclose(
                [
                    -float(spec.amplitude_left * spec.alpha_left),
                    float(spec.amplitude_right * spec.alpha_right),
                ],
                [float(slope_left), float(slope_right)],
                rtol=1.0e-9,
                atol=1.0e-14,
            )
            assert float(spec.beta_left) >= float(spec.alpha_left**2 / 4.0)
            assert float(spec.beta_right) >= float(spec.alpha_right**2 / 4.0)

    def _assert_parameter_jacobian_matches_central_fd(self, mode: str) -> None:
        """PRIVATE: Match each coefficient column away from grid knots.

        Acceptance: production ``jacfwd`` matches a central finite-difference
        Jacobian at rtol ``1e-6`` and atol ``2e-8``. Every coefficient has a
        nonzero complex-response column. Grid queries and the subtraction point
        remain at least ``0.1 eV`` from every interpolation knot.

        Notes
        -----
        Rebuild the immutable carrier from each perturbed raw coordinate
        vector. The grid case also plants the knot derivative jump and verifies
        that the
        certified points lie strictly away from that nonsmooth set.
        """
        if mode == "poly":
            coefficients: Float64[Array, " n_coef"] = jnp.asarray(
                [-0.04, 0.08, -1.4]
            )
            queries: Float64[Array, " n"] = jnp.asarray(
                [-0.77, -0.21, 0.38, 0.93]
            )
            nodes: Float64[Array, " n_nodes"] | None = None
            subtraction: float = 0.13
        else:
            coefficients = jnp.asarray([-1.1, 0.4, -0.7, -1.0])
            queries = jnp.asarray([-0.83, -0.17, 0.46, 0.88])
            nodes = jnp.asarray([-4.0, -1.2, 1.1, 4.0])
            subtraction = 0.2
            distances: Float64[Array, "n n_nodes"] = jnp.abs(
                queries[:, None] - nodes[None, :]
            )
            assert float(jnp.min(distances)) > 0.1
            assert float(jnp.min(jnp.abs(nodes - subtraction))) > 0.1
            ordinates: Float64[Array, " n_nodes"] = -jnp.logaddexp(
                coefficients, 0.0
            )
            left_slope: float = float(
                (ordinates[1] - ordinates[0]) / (nodes[1] - nodes[0])
            )
            right_slope: float = float(
                (ordinates[2] - ordinates[1]) / (nodes[2] - nodes[1])
            )
            assert abs(left_slope - right_slope) > 1.0e-3

        def response(
            raw: Float64[Array, " n_coef"],
        ) -> Complex128[Array, " n"]:
            """Evaluate one coefficient vector on frozen geometry."""
            model: SelfEnergyModel = make_self_energy_model(
                coefficients=raw,
                mode=mode,
                energy_nodes_rel_fermi_ev=nodes,
                kk_consistent=True,
                kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
                tail_coefficients=jnp.asarray([-2.2, -1.7]),
                subtraction_point_rel_fermi_ev=subtraction,
                tail_mode="power2",
            )
            returned: Complex128[Array, " n"] = evaluate_self_energy(
                queries, model, n_kk=256
            )
            return returned

        automatic: Complex128[Array, "n n_coef"] = jax.jacfwd(response)(
            coefficients
        )
        step: float = 2.0**-15
        fd_columns: List[Complex128[Array, " n"]] = []
        column: int
        for column in range(coefficients.shape[0]):
            direction: Float64[Array, " n_coef"] = (
                jnp.zeros_like(coefficients).at[column].set(step)
            )
            fd_columns.append(
                (
                    response(coefficients + direction)
                    - response(coefficients - direction)
                )
                / (2.0 * step)
            )
        finite_difference: Complex128[Array, "n n_coef"] = jnp.stack(
            fd_columns, axis=-1
        )
        np.testing.assert_allclose(
            np.asarray(automatic),
            np.asarray(finite_difference),
            rtol=1.0e-6,
            atol=2.0e-8,
        )
        column_norms: Float64[NDArray, " n_coef"] = np.max(
            np.abs(np.asarray(automatic)), axis=0
        )
        assert np.all(column_norms > 1.0e-8)

    @pytest.mark.slow
    def test_poly_parameter_jacobian_matches_central_fd(self) -> None:
        """Match every smooth polynomial coefficient against central FD.

        The shared helper compares the complete complex response Jacobian.

        Notes
        -----
        Polynomial coordinates remain smooth throughout the frozen domain.
        """
        self._assert_parameter_jacobian_matches_central_fd("poly")

    @pytest.mark.slow
    def test_grid_parameter_jacobian_matches_central_fd_away_from_knots(
        self,
    ) -> None:
        """Match every hat ordinate against central FD away from knots.

        The shared helper compares the complete complex response Jacobian.

        Notes
        -----
        Queries and the subtraction point stay outside every knot neighborhood.
        """
        self._assert_parameter_jacobian_matches_central_fd("grid")

    def test_numerical_mode_succeeds_under_jit_and_vmap(self) -> None:
        """Run a grid-mode success path through nested JIT and VMAP.

        Acceptance: compiled batched values equal the eager per-row reference
        at rtol ``1e-12``; the output stays finite, complex128, and retarded.

        Notes
        -----
        VMAP batches complete query vectors because the public API owns a
        one-dimensional energy axis. JIT wraps the batched numerical-KK path,
        including its traced validation predicates.
        """
        model: SelfEnergyModel = make_self_energy_model(
            coefficients=jnp.asarray([-1.1, 0.4, -0.7, -1.0]),
            mode="grid",
            energy_nodes_rel_fermi_ev=jnp.asarray([-4.0, -1.2, 1.1, 4.0]),
            kk_consistent=True,
            kk_domain_rel_fermi_ev=jnp.asarray([-4.0, 4.0]),
            tail_coefficients=jnp.asarray([-2.2, -1.7]),
            subtraction_point_rel_fermi_ev=0.2,
            tail_mode="power2",
        )
        batches: Float64[Array, "batch n"] = jnp.asarray(
            [[-0.9, -0.2, 0.5], [-0.7, 0.1, 0.8]]
        )

        def batched(
            points: Float64[Array, "batch n"],
        ) -> Complex128[Array, "batch n"]:
            """Batch the complete production evaluator by query row."""
            returned: Complex128[Array, "batch n"] = jax.vmap(
                lambda row: evaluate_self_energy(row, model, n_kk=64)
            )(points)
            return returned

        expected: Complex128[Array, "batch n"] = jnp.stack(
            [evaluate_self_energy(row, model, n_kk=64) for row in batches]
        )
        produced: Complex128[Array, "batch n"] = jax.jit(batched)(batches)
        np.testing.assert_allclose(
            np.asarray(produced),
            np.asarray(expected),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert produced.dtype == jnp.complex128
        assert bool(jnp.all(jnp.isfinite(produced)))
        assert bool(jnp.all(jnp.imag(produced) < 0.0))
