"""Verify generalized spectral-source carrier invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple
from jaxtyping import TypeCheckError

from diffpes.types import (
    MeasurementCoordinates,
    ParametricSelfEnergy,
    SpectralEvaluationRequest,
    TabulatedMatrixSelfEnergy,
    TabulatedRetardedGreenFunctionSource,
    make_dyson_spectral_source,
    make_measurement_coordinates,
    make_parametric_self_energy,
    make_retarded_green_batch,
    make_retarded_validation_report,
    make_self_energy_batch,
    make_self_energy_model,
    make_spectral_evaluation_request,
    make_tabulated_matrix_self_energy,
    make_tabulated_retarded_green_function_source,
)


def _coordinates(
    *,
    k_points: object = None,
    omega: object = None,
    temperature: object = None,
) -> MeasurementCoordinates:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    resolved_k: Any = (
        jnp.asarray([[0.0, 0.0, 0.0]]) if k_points is None else k_points
    )
    resolved_omega: Any = jnp.asarray([-0.5, 0.5]) if omega is None else omega
    resolved_temperature: Any = (
        jnp.asarray([20.0]) if temperature is None else temperature
    )
    result: Any = make_measurement_coordinates(
        (resolved_k, resolved_omega, resolved_temperature),
        coordinate_names=(
            "k_points_frac",
            "omega_rel_fermi_ev",
            "temperature_k",
        ),
        coordinate_units=("1", "eV", "K"),
        coordinate_dimensions=(("k", "cart"), ("omega",), ("temperature",)),
        dimension_names=("k", "cart", "omega", "temperature"),
        coordinate_system="fractional_energy_temperature",
        frame_id="fixture",
    )
    return result


def _request(**overrides: object) -> SpectralEvaluationRequest:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "coordinates": _coordinates(),
        "omega_rel_fermi_ev": jnp.asarray([-0.5, 0.5]),
        "temperature_k": jnp.asarray([20.0]),
        "eta_ev": jnp.asarray(0.02),
        "basis_ref": "basis",
    }
    values.update(overrides)
    result: Any = make_spectral_evaluation_request(**values)
    return result


def _parametric() -> ParametricSelfEnergy:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_parametric_self_energy(
        make_self_energy_model(gamma=0.05),
        source_ref="sigma",
        basis_ref="scalar",
        provenance_ref="native",
    )
    return result


def _table_values(sign: float = -1.0) -> object:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = sign * 0.1j * jnp.ones((1, 1, 2, 1, 1), dtype=jnp.complex128)
    return result


def _self_energy_table(**overrides: object) -> TabulatedMatrixSelfEnergy:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "values_ev": _table_values(),
        "k_points_frac": jnp.asarray([[0.0, 0.0, 0.0]]),
        "omega_rel_fermi_ev": jnp.asarray([-0.5, 0.5]),
        "temperature_k": jnp.asarray([20.0]),
        "basis_ref": "basis",
        "k_frame_id": "frame",
        "interpolation": "exact_nodes_v1",
        "source_ref": "sigma-table",
        "provenance_ref": "fixture",
        "source_sha256": "digest",
        "derivative_mode": "stopped",
        "validation_policy_ref": "validation",
    }
    values.update(overrides)
    result: Any = make_tabulated_matrix_self_energy(**values)
    return result


def _green_table(
    **overrides: object,
) -> TabulatedRetardedGreenFunctionSource:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "values_per_ev": _table_values(),
        "overlap": jnp.ones((1, 1, 1), dtype=jnp.complex128),
        "k_points_frac": jnp.asarray([[0.0, 0.0, 0.0]]),
        "omega_rel_fermi_ev": jnp.asarray([-0.5, 0.5]),
        "temperature_k": jnp.asarray([20.0]),
        "basis_ref": "basis",
        "k_frame_id": "frame",
        "interpolation": "exact_nodes_v1",
        "source_ref": "green-table",
        "provenance_ref": "fixture",
        "source_sha256": "digest",
        "derivative_mode": "stopped",
        "validation_policy_ref": "validation",
    }
    values.update(overrides)
    result: Any = make_tabulated_retarded_green_function_source(**values)
    return result


class TestRetardedvalidationreport:
    """Verify ``diffpes.types.RetardedValidationReport`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_aligned_measured_diagnostics(self) -> None:
        """Preserve aligned check identifiers, metrics, and tolerances.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare each explicit diagnostic tuple after construction.
        """
        report: Any = make_retarded_validation_report(
            report_ref="report",
            check_ids=("residual",),
            metric_values=(1.0e-14,),
            tolerance_values=(1.0e-12,),
            metric_units=("1",),
        )
        assert report.metric_values == (1.0e-14,)
        assert report.tolerance_values == (1.0e-12,)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"report_ref": ""}, "identity must be nonempty"),
            (
                {
                    "check_ids": ("x",),
                    "metric_values": (),
                    "tolerance_values": (0.0,),
                    "metric_units": ("1",),
                },
                "metrics must align",
            ),
            (
                {
                    "check_ids": ("",),
                    "metric_values": (0.0,),
                    "tolerance_values": (0.0,),
                    "metric_units": ("1",),
                },
                "identifiers are required",
            ),
            (
                {
                    "check_ids": ("x",),
                    "metric_values": (float("nan"),),
                    "tolerance_values": (0.0,),
                    "metric_units": ("1",),
                },
                "metrics must be finite",
            ),
            (
                {
                    "check_ids": ("x",),
                    "metric_values": (0.0,),
                    "tolerance_values": (-1.0,),
                    "metric_units": ("1",),
                },
                "tolerances must be nonnegative",
            ),
        ],
    )
    def test_rejects_each_report_invariant(
        self, overrides: Dict[str, object], message: str
    ) -> None:
        """Reject empty, misaligned, nonfinite, and negative diagnostics.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Apply one malformed report field in each parameterized case.
        """
        values: Dict[str, object] = {"report_ref": "report"}
        values.update(overrides)
        with pytest.raises(ValueError, match=message):
            make_retarded_validation_report(**values)


class TestSpectralevaluationrequest:
    """Verify ``diffpes.types.SpectralEvaluationRequest`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_increasing_finite_axes(self) -> None:
        """Preserve energy, temperature, regulator, and basis identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare all scalar and vector request fields with explicit inputs.
        """
        request: Any = _request()
        assert jnp.array_equal(
            request.omega_rel_fermi_ev, jnp.asarray([-0.5, 0.5])
        )
        assert request.basis_ref == "basis"

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"basis_ref": ""}, "basis_ref must be nonempty", ValueError),
            (
                {"omega_rel_fermi_ev": jnp.zeros((1, 1))},
                "omega_rel_fermi_ev",
                TypeCheckError,
            ),
            (
                {"omega_rel_fermi_ev": jnp.asarray([0.0, 0.0])},
                "energy axis.*strictly increasing",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"temperature_k": jnp.asarray([-1.0])},
                "temperature axis.*nonnegative",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"eta_ev": jnp.asarray(0.0)},
                "eta must be finite and strictly positive",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_request_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject empty basis, bad ranks, axes, temperatures, and eta.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one field in the valid request fixture.
        """
        with pytest.raises(error, match=message):
            _request(**overrides)


class TestParametricselfenergy:
    """Verify ``diffpes.types.ParametricSelfEnergy`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_scalar_source_identity(self) -> None:
        """Preserve source, basis, provenance, and exact-AD identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect each static field of the constant source.
        """
        source: Any = _parametric()
        assert source.source_ref == "sigma"
        assert source.derivative_mode == "exact_ad"

    @pytest.mark.parametrize(
        "field", ["source_ref", "basis_ref", "provenance_ref"]
    )
    def test_rejects_each_empty_source_identity(self, field: str) -> None:
        """Reject each empty parametric self-energy identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Blank one static field while retaining a valid parameterization.
        """
        values: Dict[str, object] = {
            "source_ref": "s",
            "basis_ref": "b",
            "provenance_ref": "p",
        }
        values[field] = ""
        with pytest.raises(
            ValueError, match="identity fields must be nonempty"
        ):
            make_parametric_self_energy(
                make_self_energy_model(gamma=0.1), **values
            )


class TestSelfenergybatch:
    """Verify ``diffpes.types.SelfEnergyBatch`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_request_aligned_square_matrices(self) -> None:
        """Preserve a one-k two-energy evaluated self-energy batch.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare its complete shape and source identity.
        """
        batch: Any = make_self_energy_batch(
            _table_values(),
            _request(),
            basis_ref="basis",
            source_ref="source",
            derivative_mode="exact_ad",
        )
        assert batch.values_ev.shape == (1, 1, 2, 1, 1)

    @pytest.mark.parametrize(
        ("values", "basis", "source", "message"),
        [
            (
                jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex128),
                "basis",
                "source",
                "axes do not match",
            ),
            (
                _table_values(),
                "other",
                "source",
                "basis identities must match",
            ),
            (_table_values(), "basis", "", "identity fields must be nonempty"),
        ],
    )
    def test_rejects_each_batch_invariant(
        self, values: object, basis: str, source: str, message: str
    ) -> None:
        """Reject bad axes, mismatched basis, and empty identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one field of the evaluated self-energy batch.
        """
        with pytest.raises(ValueError, match=message):
            make_self_energy_batch(
                values,
                _request(),
                basis_ref=basis,
                source_ref=source,
                derivative_mode="exact_ad",
            )


class TestRetardedgreenbatch:
    """Verify ``diffpes.types.RetardedGreenBatch`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_request_and_overlap_aligned_matrices(self) -> None:
        """Preserve Green matrices and the corresponding overlap metric.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the exact batch and metric shapes.
        """
        batch: Any = make_retarded_green_batch(
            _table_values(),
            jnp.ones((1, 1, 1), dtype=jnp.complex128),
            _request(),
            basis_ref="basis",
            source_ref="source",
            derivative_mode="exact_ad",
            validation_ref="validation",
        )
        assert batch.values_per_ev.shape == (1, 1, 2, 1, 1)
        assert batch.overlap.shape == (1, 1, 1)

    @pytest.mark.parametrize(
        ("overlap", "basis", "validation", "message"),
        [
            (
                jnp.ones((1, 1, 1), dtype=jnp.complex128),
                "other",
                "v",
                "basis identities must match",
            ),
            (
                jnp.ones((1, 1, 1), dtype=jnp.complex128),
                "basis",
                "",
                "identity fields must be nonempty",
            ),
        ],
    )
    def test_rejects_each_green_batch_invariant(
        self, overlap: object, basis: str, validation: str, message: str
    ) -> None:
        """Reject bad metric axes, basis identity, and evidence identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one field of the valid Green batch fixture.
        """
        with pytest.raises(ValueError, match=message):
            make_retarded_green_batch(
                _table_values(),
                overlap,
                _request(),
                basis_ref=basis,
                source_ref="source",
                derivative_mode="exact_ad",
                validation_ref=validation,
            )

    def test_rejects_overlap_axis_mismatch_at_runtime_boundary(self) -> None:
        """Reject a two-k overlap paired with a one-k Green tensor.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Keep the complex dtype valid so the named-axis check is the oracle.
        """
        with pytest.raises(TypeCheckError, match="overlap"):
            make_retarded_green_batch(
                _table_values(),
                jnp.ones((2, 1, 1), dtype=jnp.complex128),
                _request(),
                basis_ref="basis",
                source_ref="source",
                derivative_mode="exact_ad",
                validation_ref="validation",
            )


class TestDysonspectralsource:
    """Verify ``diffpes.types.DysonSpectralSource`` capability inference.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_keys_overlap_requirement_on_state_orthonormality(self) -> None:
        """Require overlap only for a nonorthonormal electronic-state basis.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare sources built with and without the orthonormal declaration.
        """
        orthonormal: Any = make_dyson_spectral_source(
            _parametric(),
            electronic_state_capabilities=("hamiltonian", "orthonormal_basis"),
            source_ref="dyson",
        )
        general: Any = make_dyson_spectral_source(
            _parametric(),
            electronic_state_capabilities=("hamiltonian",),
            source_ref="dyson",
        )
        assert orthonormal.required_capabilities == ("hamiltonian",)
        assert general.required_capabilities == ("hamiltonian", "overlap")

    @pytest.mark.parametrize(
        ("capabilities", "source_ref", "message"),
        [
            ((), "dyson", "capabilities must be nonempty"),
            (("hamiltonian",), "", "source_ref"),
        ],
    )
    def test_rejects_each_dyson_identity_invariant(
        self, capabilities: Tuple[str, ...], source_ref: str, message: str
    ) -> None:
        """Reject absent state capabilities and empty Dyson identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Blank one static declaration per parameterized case.
        """
        with pytest.raises(ValueError, match=message):
            make_dyson_spectral_source(
                _parametric(),
                electronic_state_capabilities=capabilities,
                source_ref=source_ref,
            )


class TestTabulatedmatrixselfenergy:
    """Verify ``diffpes.types.TabulatedMatrixSelfEnergy`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_causal_exact_node_table_with_measured_report(
        self,
    ) -> None:
        """Record the actual causal-loss eigenvalue for a valid table.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the measured minimum with the analytic scalar loss 0.1 eV.
        """
        table: Any = _self_energy_table()
        assert table.interpolation == "exact_nodes_v1"
        assert table.extrapolation == "reject"
        assert table.validation.check_ids[-1] == "causal_loss"
        assert table.validation.metric_values[-1] == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            (
                {"k_points_frac": jnp.asarray([[jnp.nan, 0.0, 0.0]])},
                "axes must be finite",
            ),
            (
                {"omega_rel_fermi_ev": jnp.asarray([0.5, -0.5])},
                "axes must be ordered",
            ),
            ({"interpolation": "linear"}, "exact-node evaluation only"),
            ({"source_ref": ""}, "identity fields must be nonempty"),
            (
                {"values_ev": _table_values(sign=1.0)},
                "causal_loss check failed",
            ),
        ],
    )
    def test_rejects_each_table_invariant(
        self, overrides: Dict[str, object], message: str
    ) -> None:
        """Reject bad axes, identity, interpolation, and noncausal loss.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one component of the valid exact-node table.
        """
        with pytest.raises(ValueError, match=message):
            _self_energy_table(**overrides)

    def test_rejects_table_axis_mismatch_at_runtime_boundary(self) -> None:
        """Reject a one-energy tensor paired with a two-energy axis.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Preserve the required complex dtype while changing the named axis.
        """
        with pytest.raises(TypeCheckError, match="values_ev"):
            _self_energy_table(
                values_ev=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex128)
            )


class TestTabulatedretardedgreenfunctionsource:
    """Verify ``diffpes.types.TabulatedRetardedGreenFunctionSource``.

    Exercise direct retarded-Green table invariants and exact-node selection.
    """

    def test_serves_only_exact_coordinates_and_records_psd_metric(
        self,
    ) -> None:
        """Return the singleton-temperature table at its exact nodes.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the selected tensor and analytic spectral eigenvalue.
        """
        table: Any = _green_table()
        values: Any = table.retarded_green_function(_coordinates())
        assert values.shape == (1, 2, 1, 1)
        assert table.validation.check_ids[-1] == "spectral_psd"
        assert table.validation.metric_values[-1] == pytest.approx(
            0.1 / jnp.pi
        )

    def test_rejects_nonmatching_exact_coordinates(self) -> None:
        """Reject a requested energy axis different from the stored nodes.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Shift both requested energies without changing their length.
        """
        coordinates: Any = _coordinates(omega=jnp.asarray([-0.4, 0.6]))
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="exact-node evaluation"
        ):
            _green_table().retarded_green_function(coordinates)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"overlap": jnp.asarray([[[1.0 + 1.0j]]])}, "Hermiticity failed"),
            (
                {"overlap": -jnp.ones((1, 1, 1), dtype=jnp.complex128)},
                "positive-definiteness failed",
            ),
            ({"interpolation": "linear"}, "exact nodes only"),
            ({"source_ref": ""}, "identity fields must be nonempty"),
            (
                {"values_per_ev": _table_values(sign=1.0)},
                "spectral_psd check failed",
            ),
        ],
    )
    def test_rejects_each_direct_table_invariant(
        self, overrides: Dict[str, object], message: str
    ) -> None:
        """Reject metric, identity, interpolation, and spectral-PSD failures.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one component of the valid direct table.
        """
        with pytest.raises(ValueError, match=message):
            _green_table(**overrides)

    def test_rejects_overlap_axis_mismatch_at_runtime_boundary(self) -> None:
        """Reject a two-k overlap paired with a one-k direct table.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Preserve the complex dtype so named-axis unification rejects the input.
        """
        with pytest.raises(TypeCheckError, match="overlap"):
            _green_table(overlap=jnp.ones((2, 1, 1), dtype=jnp.complex128))


class TestMakeRetardedValidationReport:
    """Verify ``diffpes.types.make_retarded_validation_report``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSpectralEvaluationRequest:
    """Verify ``diffpes.types.make_spectral_evaluation_request``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeParametricSelfEnergy:
    """Verify ``diffpes.types.make_parametric_self_energy``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSelfEnergyBatch:
    """Verify ``diffpes.types.make_self_energy_batch``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeRetardedGreenBatch:
    """Verify ``diffpes.types.make_retarded_green_batch``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeDysonSpectralSource:
    """Verify ``diffpes.types.make_dyson_spectral_source``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeTabulatedMatrixSelfEnergy:
    """Verify ``diffpes.types.make_tabulated_matrix_self_energy``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeTabulatedRetardedGreenFunctionSource:
    """Verify ``diffpes.types.make_tabulated_retarded_green_function_source``.

    Bind the factory to the direct retarded-Green table tests above.
    """
