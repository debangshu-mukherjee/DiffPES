"""Validate executable failure classification and exit mapping.

The tests cover classified input, numerical, timeout, and unknown failures.
They also cover the stable exit code for each public error category.
"""

from beartype.typing import Any

from diffpes.harness import (
    AutomatonError,
    DeadlineExceededError,
    classify_exception,
    exit_code_for,
)


class TestAutomatonError:
    """Validate :class:`~diffpes.harness.AutomatonError` metadata.

    The case scope covers stable error-kind and field attributes.
    """

    def test_retains_error_kind_and_field(self) -> None:
        """Retain classified error metadata on the public exception.

        A parameter error must expose both its category and affected field.

        Notes
        -----
        Creates one public error and compares the dynamic metadata attributes.
        """
        error: Any = AutomatonError(
            "invalid parameter",
            error_kind="InvalidInput",
            field="energy",
        )

        assert error.error_kind == "InvalidInput"
        assert error.field == "energy"


class TestDeadlineExceededError:
    """Validate :class:`~diffpes.harness.DeadlineExceededError` metadata.

    The case scope covers stable timeout classification.
    """

    def test_sets_the_timeout_category(self) -> None:
        """Set the stable timeout error category.

        A deadline exception must always map to the timeout result category.

        Notes
        -----
        Builds one public deadline exception and reads its metadata attribute.
        """
        error: Any = DeadlineExceededError()

        assert error.error_kind == "Timeout"


class TestClassifyException:
    """Validate :func:`~diffpes.harness.classify_exception` mappings.

    The case scope covers standard input and numerical exception families.
    """

    def test_maps_value_errors_to_invalid_input(self) -> None:
        """Map ValueError to the invalid-input category.

        A body validation error must become a parseable executable failure.

        Notes
        -----
        Classifies one ValueError and inspects the public category attribute.
        """
        error: Any = classify_exception(ValueError("bad input"))

        assert error.error_kind == "InvalidInput"


class TestExitCodeFor:
    """Validate :func:`~diffpes.harness.exit_code_for` mappings.

    The case scope covers stable parameter and timeout process codes.
    """

    def test_maps_timeout_and_parameter_ranges(self) -> None:
        """Map timeout and range categories to documented process codes.

        Timeout uses 124 while a parameter range failure uses 2.

        Notes
        -----
        Reads the public error-code function for two stable categories.
        """
        timeout_code: int = exit_code_for("Timeout")
        range_code: int = exit_code_for("ParamOutOfRange")

        assert timeout_code == 124
        assert range_code == 2
