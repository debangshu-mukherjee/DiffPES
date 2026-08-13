"""Validate matrix-valued retarded spectral evidence.

Extended Summary
----------------
Provide eager validation records for tabulated self-energy and Green sources.

Routine Listings
----------------
:class:`RetardedValidationReport`
    Define the ``RetardedValidationReport`` public contract.
:func:`make_retarded_validation_report`
    Compute the ``make_retarded_validation_report`` public contract.
"""

import equinox as eqx
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import HERMITICITY_RELATIVE_TOLERANCE


class RetardedValidationReport(eqx.Module):
    """Define the ``RetardedValidationReport`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_generalized_spectral.TestRetardedvalidationreport`

    Attributes
    ----------
    report_ref : str
        Store the report identity.
    check_ids : Tuple[str, ...]
        Store check identities.
    metric_values : Tuple[float, ...]
        Store measured values.
    tolerance_values : Tuple[float, ...]
        Store tolerance values.
    metric_units : Tuple[str, ...]
        Store metric units.
    assumptions : Tuple[str, ...]
        Store assumptions.
    excluded_claims : Tuple[str, ...]
        Store excluded claims.
    evidence_refs : Tuple[str, ...]
        Store evidence identities.
    schema_version : str
        Store the schema version.

    See Also
    --------
    make_retarded_validation_report
        Construct a validated report.
    """

    report_ref: str = eqx.field(static=True)
    check_ids: Tuple[str, ...] = eqx.field(static=True)
    metric_values: Tuple[float, ...] = eqx.field(static=True)
    tolerance_values: Tuple[float, ...] = eqx.field(static=True)
    metric_units: Tuple[str, ...] = eqx.field(static=True)
    assumptions: Tuple[str, ...] = eqx.field(static=True)
    excluded_claims: Tuple[str, ...] = eqx.field(static=True)
    evidence_refs: Tuple[str, ...] = eqx.field(static=True)
    schema_version: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate aligned numeric report fields."""
        if not self.report_ref or not self.schema_version:
            raise ValueError("validation report identity must be nonempty")
        size: int = len(self.check_ids)
        if any(
            len(values) != size
            for values in (
                self.metric_values,
                self.tolerance_values,
                self.metric_units,
            )
        ):
            raise ValueError(
                "validation report metrics must align with check_ids"
            )
        if any(not check_id for check_id in self.check_ids):
            raise ValueError(
                "validation report check identifiers are required"
            )
        if not all(np.isfinite(self.metric_values)):
            raise ValueError("validation report metrics must be finite")
        if any(value < 0.0 for value in self.tolerance_values):
            raise ValueError(
                "validation report tolerances must be nonnegative"
            )


def _validate_table_axes(
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    k_points: Float64[Array, "n_k 3"],
    omega: Float64[Array, " n_omega"],
    temperature: Float64[Array, " n_temperature"],
) -> None:
    """PRIVATE: Validate common table dimensions before construction.

    Parameters
    ----------
    values : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Matrix table to validate.
    k_points : Float64[Array, "n_k 3"]
        Fractional momenta to validate.
    omega : Float64[Array, " n_omega"]
        Energy axis to validate.
    temperature : Float64[Array, " n_temperature"]
        Temperature axis to validate.

    Raises
    ------
    ValueError
        If array axes, finite values, or ordering checks fail.
    """
    if (
        values.ndim != 5  # noqa: PLR2004
        or values.shape[-1] != values.shape[-2]
        or values.shape[:3]
        != (temperature.shape[0], k_points.shape[0], omega.shape[0])
        or k_points.shape[1:] != (3,)
        or omega.ndim != 1
        or temperature.ndim != 1
    ):
        raise ValueError(
            "spectral table axes and square matrix axes must agree"
        )
    k_table: Float64[NDArray, "n_k 3"] = np.asarray(k_points)
    omega_table: Float64[NDArray, " n_omega"] = np.asarray(omega)
    temperature_table: Float64[NDArray, " n_temperature"] = np.asarray(
        temperature
    )
    if not all(
        np.all(np.isfinite(axis))
        for axis in (k_table, omega_table, temperature_table)
    ):
        raise ValueError("spectral table axes must be finite")
    if (
        not np.all(np.diff(omega_table) > 0.0)
        or not np.all(temperature_table >= 0.0)
        or not np.all(np.diff(temperature_table) > 0.0)
    ):
        raise ValueError("spectral table axes must be ordered")


def _eager_overlap_validation(
    overlap: Complex128[Array, "n_k n_orb n_orb"],
) -> None:
    """PRIVATE: Require a finite Hermitian positive-definite overlap.

    Parameters
    ----------
    overlap : Complex128[Array, "n_k n_orb n_orb"]
        Overlap matrices to validate.

    Raises
    ------
    ValueError
        If finite, Hermitian, or positive-definite checks fail.
    """
    metric: Complex128[NDArray, "n_k n_orb n_orb"] = np.asarray(overlap)
    if not np.all(np.isfinite(metric)):
        raise ValueError("direct Green overlap must be finite")
    dagger: Complex128[NDArray, "n_k n_orb n_orb"] = np.swapaxes(
        np.conj(metric), -1, -2
    )
    scale: Float64[NDArray, " n_k"] = np.maximum(
        np.linalg.norm(metric, axis=(-2, -1)), np.finfo(np.float64).eps
    )
    residual: float = float(
        np.max(np.linalg.norm(metric - dagger, axis=(-2, -1)) / scale)
    )
    if residual > HERMITICITY_RELATIVE_TOLERANCE:
        raise ValueError(
            f"direct Green overlap Hermiticity failed: residual={residual}"
        )
    minimum_eigenvalue: float = float(np.min(np.linalg.eigvalsh(metric)))
    if minimum_eigenvalue <= 0.0:
        raise ValueError(
            "direct Green overlap positive-definiteness failed: "
            f"minimum_eigenvalue={minimum_eigenvalue}"
        )


def _eager_matrix_validation(
    values: Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"],
    *,
    matrix_kind: str,
) -> RetardedValidationReport:
    """PRIVATE: Validate symmetry and semidefinite physics eagerly.

    Parameters
    ----------
    values : Complex128[Array, "n_temperature n_k n_omega n_orb n_orb"]
        Matrix table to validate.
    matrix_kind : str
        Derived matrix kind.

    Returns
    -------
    result : RetardedValidationReport
        Measured validation report.

    Raises
    ------
    ValueError
        If finite, Hermitian, or semidefinite checks fail.
    """
    table: Complex128[NDArray, "n_temperature n_k n_omega n_orb n_orb"] = (
        np.asarray(values)
    )
    if not np.all(np.isfinite(table)):
        raise ValueError("finite_table check failed: nonfinite values")
    dagger: Complex128[NDArray, "n_temperature n_k n_omega n_orb n_orb"] = (
        np.swapaxes(np.conj(table), -1, -2)
    )
    derived: Complex128[NDArray, "n_temperature n_k n_omega n_orb n_orb"]
    check_id: str
    unit: str
    if matrix_kind == "self_energy":
        derived = -(table - dagger) / (2.0j)
        check_id = "causal_loss"
        unit = "eV"
    elif matrix_kind == "green":
        derived = -(table - dagger) / (2.0j * np.pi)
        check_id = "spectral_psd"
        unit = "1/eV"
    else:
        raise ValueError("unknown matrix validation kind")
    derived_dagger: Complex128[
        NDArray, "n_temperature n_k n_omega n_orb n_orb"
    ] = np.swapaxes(np.conj(derived), -1, -2)
    derived_residual: float = float(
        np.max(
            np.linalg.norm(derived - derived_dagger, axis=(-2, -1))
            / np.maximum(
                np.linalg.norm(derived, axis=(-2, -1)),
                np.finfo(np.float64).eps,
            )
        )
    )
    minimum_eigenvalue: float = float(np.min(np.linalg.eigvalsh(derived)))
    scale: float = max(
        float(np.max(np.linalg.norm(derived, axis=(-2, -1)))), 1.0
    )
    tolerance: float = HERMITICITY_RELATIVE_TOLERANCE * scale
    if derived_residual > HERMITICITY_RELATIVE_TOLERANCE:
        raise ValueError(
            f"{check_id} Hermiticity check failed: residual={derived_residual}"
        )
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"{check_id} check failed: minimum_eigenvalue={minimum_eigenvalue}"
        )
    result: RetardedValidationReport = RetardedValidationReport(
        "pending",
        ("finite_table", "derived_hermiticity", check_id),
        (0.0, derived_residual, minimum_eigenvalue),
        (0.0, HERMITICITY_RELATIVE_TOLERANCE, tolerance),
        ("1", "1", unit),
        ("eager_node_validation",),
        ("analyticity_proven",),
        (),
        "1.0",
    )
    return result


@jaxtyped(typechecker=beartype)
def make_retarded_validation_report(
    *,
    report_ref: str,
    check_ids: Tuple[str, ...] = (),
    metric_values: Tuple[float, ...] = (),
    tolerance_values: Tuple[float, ...] = (),
    metric_units: Tuple[str, ...] = (),
    assumptions: Tuple[str, ...] = (),
    excluded_claims: Tuple[str, ...] = (),
    evidence_refs: Tuple[str, ...] = (),
    schema_version: str = "1.0",
) -> RetardedValidationReport:
    """Compute the ``make_retarded_validation_report`` public contract.

    Validate aligned metrics and preserve the declared report identity.

    :see: :class:`~.test_generalized_spectral.TestMakeRetardedValidationReport`

    Notes
    -----
    Construct the immutable carrier after Equinox validates its field groups.

    Parameters
    ----------
    report_ref : str
        Report identity.
    check_ids : Tuple[str, ...]
        Check identities.
    metric_values : Tuple[float, ...]
        Measured values.
    tolerance_values : Tuple[float, ...]
        Tolerance values.
    metric_units : Tuple[str, ...]
        Metric units.
    assumptions : Tuple[str, ...]
        Validation assumptions.
    excluded_claims : Tuple[str, ...]
        Excluded claims.
    evidence_refs : Tuple[str, ...]
        Evidence identities.
    schema_version : str
        Schema version.

    Returns
    -------
    result : RetardedValidationReport
        Validated report.
    """
    result: RetardedValidationReport = RetardedValidationReport(
        report_ref,
        check_ids,
        metric_values,
        tolerance_values,
        metric_units,
        assumptions,
        excluded_claims,
        evidence_refs,
        schema_version,
    )
    return result


__all__: list[str] = [
    "RetardedValidationReport",
    "make_retarded_validation_report",
]
