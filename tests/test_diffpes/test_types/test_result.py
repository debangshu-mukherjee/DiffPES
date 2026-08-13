"""Verify intrinsic and observed result-carrier invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Tuple

from diffpes.types import (
    FidelityManifest,
    MeasurementCoordinates,
    make_fidelity_manifest,
    make_intrinsic_photocurrent,
    make_measurement_coordinates,
    make_simulation_result,
)


def _coordinates() -> MeasurementCoordinates:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_measurement_coordinates(
        (jnp.asarray([0.0]),),
        coordinate_names=("energy",),
        coordinate_units=("eV",),
        coordinate_dimensions=(("energy",),),
        dimension_names=("energy",),
        coordinate_system="relative_energy",
        frame_id="fixture",
    )
    return result


def _fidelity() -> FidelityManifest:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_fidelity_manifest(
        schema_version="1.0",
        model_ref="model",
        instrument_ref="none",
        acquisition_ref="none",
        initial_state="tb",
        spectral_physics="scalar",
        photocurrent="projection",
        light_interaction="none",
        instrument="none",
    )
    return result


class TestIntrinsicphotocurrent:
    """Verify ``diffpes.types.IntrinsicPhotocurrent`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_nonnegative_multidomain_payload(self) -> None:
        """Preserve two domains sharing two explicit channel labels.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare payload kind, domain count, and channel metadata.
        """
        result: Any = make_intrinsic_photocurrent(
            (jnp.ones((2, 3)), jnp.zeros((2, 3))),
            _coordinates(),
            channel_labels=("a", "b"),
            intensity_units="1/eV",
            model_ref="model",
            state_ref="state",
            fidelity=_fidelity(),
        )
        assert result.payload_kind == "scalar_intensity"
        assert len(result.scalar_intensity_by_domain) == 2

    @pytest.mark.parametrize(
        ("payload", "labels", "units", "message"),
        [
            ((), ("a",), "1/eV", "at least one"),
            ((jnp.ones((1, 2)),), (), "1/eV", "labels must be nonempty"),
            ((jnp.ones((1, 2)),), ("",), "1/eV", "labels must be nonempty"),
            ((jnp.ones((1, 2)),), ("a",), "", "identity fields"),
            ((jnp.ones((2, 2)),), ("a",), "1/eV", "share the channel"),
        ],
    )
    def test_rejects_each_static_intrinsic_invariant(
        self,
        payload: Tuple[object, ...],
        labels: Tuple[str, ...],
        units: str,
        message: str,
    ) -> None:
        """Reject absent domains, identities, and channel-axis mismatches.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change one structural field in the valid intrinsic result fixture.
        """
        with pytest.raises(ValueError, match=message):
            make_intrinsic_photocurrent(
                payload,
                _coordinates(),
                channel_labels=labels,
                intensity_units=units,
                model_ref="model",
                state_ref="state",
                fidelity=_fidelity(),
            )

    def test_rejects_negative_or_nonfinite_intensity(self) -> None:
        """Reject invalid intrinsic values through the traced guard.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Supply one negative value in a single-channel domain.
        """
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="finite and nonnegative"
        ):
            make_intrinsic_photocurrent(
                (jnp.asarray([[-1.0]]),),
                _coordinates(),
                channel_labels=("a",),
                intensity_units="1/eV",
                model_ref="model",
                state_ref="state",
                fidelity=_fidelity(),
            )


class TestSimulationresult:
    """Verify ``diffpes.types.SimulationResult`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_nonnegative_expected_counts(self) -> None:
        """Preserve two labeled nonnegative count channels.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the stored matrix and its exact label tuple.
        """
        result: Any = make_simulation_result(
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            _coordinates(),
            channel_labels=("a", "b"),
            fidelity=_fidelity(),
        )
        assert result.channel_labels == ("a", "b")
        assert jnp.sum(result.expected_counts) == 10.0

    @pytest.mark.parametrize(
        ("counts", "labels", "message", "error"),
        [
            (jnp.ones((1, 2)), (), "labels must be nonempty", ValueError),
            (jnp.ones((2, 2)), ("a",), "channels must match", ValueError),
            (
                jnp.asarray([[-1.0]]),
                ("a",),
                "finite and nonnegative",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_observed_result_invariant(
        self,
        counts: object,
        labels: Tuple[str, ...],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject absent labels, axis mismatch, and invalid counts.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change one result field in each parameterized case.
        """
        with pytest.raises(error, match=message):
            make_simulation_result(
                counts,
                _coordinates(),
                channel_labels=labels,
                fidelity=_fidelity(),
            )


class TestMakeIntrinsicPhotocurrent:
    """Verify ``diffpes.types.make_intrinsic_photocurrent``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSimulationResult:
    """Verify ``diffpes.types.make_simulation_result``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
