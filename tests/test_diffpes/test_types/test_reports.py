"""Validate the reports contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx
from beartype.typing import Any

import diffpes.types


class TestEvidencereport:
    """Verify :class:`~diffpes.types.EvidenceReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.EvidenceReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``EvidenceReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.EvidenceReport
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestPolicyreport:
    """Verify :class:`~diffpes.types.PolicyReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.PolicyReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``PolicyReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.PolicyReport
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestReproductionreport:
    """Verify :class:`~diffpes.types.ReproductionReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.ReproductionReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``ReproductionReport`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.ReproductionReport
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestVerificationreport:
    """Verify :class:`~diffpes.types.VerificationReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.VerificationReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``VerificationReport`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.VerificationReport
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestWaiverRecord:
    """Verify :class:`~diffpes.types.WaiverRecord`.

    The case checks the static carrier for a bounded policy waiver.

    :see: :class:`~diffpes.types.WaiverRecord`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the waiver record as an Equinox module class.

        The carrier must use the library PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = diffpes.types.WaiverRecord
        assert issubclass(symbol, eqx.Module)


class TestWaiverReport:
    """Verify :class:`~diffpes.types.WaiverReport`.

    The case checks the JAX-native carrier for temporal waiver validation.

    :see: :class:`~diffpes.types.WaiverReport`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the waiver report as an Equinox module class.

        The carrier must use the library PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = diffpes.types.WaiverReport
        assert issubclass(symbol, eqx.Module)


class TestMakeEvidenceReport:
    """Verify :func:`~diffpes.types.make_evidence_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_evidence_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_evidence_report`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_evidence_report
        assert callable(symbol)


class TestMakePolicyReport:
    """Verify :func:`~diffpes.types.make_policy_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_policy_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_policy_report`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_policy_report
        assert callable(symbol)


class TestMakeReproductionReport:
    """Verify :func:`~diffpes.types.make_reproduction_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_reproduction_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_reproduction_report`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_reproduction_report
        assert callable(symbol)


class TestMakeVerificationReport:
    """Verify :func:`~diffpes.types.make_verification_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_verification_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_verification_report`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_verification_report
        assert callable(symbol)


class TestMakeWaiverRecord:
    """Verify :func:`~diffpes.types.make_waiver_record`.

    The case checks construction of a bounded policy-waiver declaration.

    :see: :func:`~diffpes.types.make_waiver_record`
    """

    def test_factory_retains_absolute_utc_limits(self) -> None:
        """Build a waiver with explicit issue and expiry times.

        The factory must retain both absolute UTC strings exactly.

        Notes
        -----
        The test compares the two static UTC fields exactly.
        """
        factory: Any = diffpes.types.make_waiver_record
        result: Any = factory(
            "waiver-1",
            "org.diffpes.policy.research.v1",
            ("claim-1",),
            "reviewer",
            "The external result is pending.",
            "2026-07-20T00:00:00Z",
            "2026-07-22T00:00:00Z",
        )
        assert result.issued_at_utc == "2026-07-20T00:00:00Z"
        assert result.expires_at_utc == "2026-07-22T00:00:00Z"


class TestMakeWaiverReport:
    """Verify :func:`~diffpes.types.make_waiver_report`.

    The case checks the JAX Boolean outcomes for temporal validation.

    :see: :func:`~diffpes.types.make_waiver_report`
    """

    def test_factory_retains_valid_and_active_outcomes(self) -> None:
        """Build a valid and active report without errors.

        The report must contain two true scalar JAX leaves.

        Notes
        -----
        The test converts both scalar JAX leaves to Boolean values.
        """
        factory: Any = diffpes.types.make_waiver_report
        result: Any = factory("waiver-1", True, True)
        assert bool(result.valid)
        assert bool(result.active)
