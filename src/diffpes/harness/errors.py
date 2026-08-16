"""Map failures from executable experiment runs.

Extended Summary
----------------
This module maps Python failures to stable result categories and process exit
codes. It exposes exception types lazily because production carrier classes
must remain in :mod:`diffpes.types` under the repository architecture rules.

Routine Listings
----------------
:class:`AutomatonError`
    Report one classified executable experiment failure.
:class:`DeadlineExceededError`
    Report a wall-time deadline failure.
:func:`classify_exception`
    Map one Python exception to an executable result.
:func:`exit_code_for`
    Return the process exit code for one error category.
"""

from __future__ import annotations

from beartype import beartype
from beartype.typing import Any, Optional, Tuple
from jaxtyping import jaxtyped

from diffpes.constants import AUTOMATON_ERROR_KINDS, AUTOMATON_EXIT_CODES


def _error_types() -> Tuple[type[Exception], type[Exception]]:
    """PRIVATE: Build and cache the public executable exception types.

    Returns
    -------
    error_types : Tuple[type[Exception], type[Exception]]
        Stable base and deadline exception classes.

    Notes
    -----
    Stores the classes on this function instead of a module assignment. The
    arrangement preserves the repository rule for top-level carrier classes.
    """
    cached: object | None = getattr(_error_types, "cached", None)
    if cached is not None:
        error_types: Tuple[type[Exception], type[Exception]] = cached
        return error_types

    class _AutomatonError(Exception):
        def __init__(
            self,
            message: str,
            error_kind: str = "Unknown",
            field: Optional[str] = None,
        ) -> None:
            super().__init__(message)
            self.error_kind: str = error_kind
            self.field: Optional[str] = field

    class _DeadlineExceededError(_AutomatonError):
        def __init__(
            self,
            message: str = "deadline exceeded",
            field: Optional[str] = None,
        ) -> None:
            super().__init__(message, error_kind="Timeout", field=field)

    _AutomatonError.__name__ = "AutomatonError"
    _AutomatonError.__qualname__ = "AutomatonError"
    _AutomatonError.__module__ = __name__
    _DeadlineExceededError.__name__ = "DeadlineExceededError"
    _DeadlineExceededError.__qualname__ = "DeadlineExceededError"
    _DeadlineExceededError.__module__ = __name__
    error_types = (_AutomatonError, _DeadlineExceededError)
    setattr(_error_types, "cached", error_types)  # noqa: B010
    return error_types


def __getattr__(name: str) -> Any:
    """Return a lazily built public executable exception type."""
    error_type: type[Exception]
    deadline_type: type[Exception]
    error_type, deadline_type = _error_types()
    if name == "AutomatonError":
        value: type[Exception] = error_type
    elif name == "DeadlineExceededError":
        value = deadline_type
    else:
        message: str = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    return value


@jaxtyped(typechecker=beartype)
def classify_exception(exc: Exception) -> Exception:
    """Map one Python exception to an executable result.

    The function preserves an already classified failure. It maps common I/O,
    numerical, resource, and timeout failures to stable error categories.

    :see: :class:`~.test_errors.TestClassifyException`

    Implementation Logic
    --------------------
    1. **Preserve classified failures**::

           if isinstance(exc, error_type): return exc

       The operation keeps field metadata from parameter validation intact.

    2. **Map standard exception families**::

           error = error_type(message, error_kind=error_kind)

       The mapping produces a portable result record without a traceback.

    Parameters
    ----------
    exc : Exception
        Python exception raised while parsing or running an experiment.

    Returns
    -------
    error : Exception
        Stable executable exception with ``error_kind`` and ``field`` values.
    """
    error_type: type[Exception]
    deadline_type: type[Exception]
    error_type, deadline_type = _error_types()
    if isinstance(exc, error_type):
        error: Exception = exc
        return error  # noqa: RET504 -- assign-before-return is required.
    text: str = str(exc)
    normalized_text: str = text.lower()
    error_kind: str
    if isinstance(exc, TimeoutError):
        error: Exception = deadline_type(text or "deadline exceeded")
        return error
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, ValueError)):
        error_kind = "InvalidInput"
    elif isinstance(
        exc, (FloatingPointError, OverflowError, ZeroDivisionError)
    ):
        error_kind = "NumericalFailure"
    elif isinstance(exc, MemoryError) or (
        "memory" in normalized_text or "resource exhausted" in normalized_text
    ):
        error_kind = "ResourceExhausted"
    else:
        error_kind = "Unknown"
    field: Optional[str] = getattr(exc, "field", None)
    error = error_type(
        text or type(exc).__name__, error_kind=error_kind, field=field
    )
    return error  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def exit_code_for(error_kind: str) -> int:
    """Return the process exit code for one error category.

    The function falls back to the unknown-error code for unrecognized input.
    This behavior keeps exception reporting safe for third-party body errors.

    :see: :class:`~.test_errors.TestExitCodeFor`

    Notes
    -----
    Reads the centrally declared mapping. It does not raise while handling a
    failure because callers must still emit one JSON result line.

    Parameters
    ----------
    error_kind : str
        Stable executable error category.

    Returns
    -------
    exit_code : int
        Process exit code for the selected category.
    """
    selected_kind: str = (
        error_kind if error_kind in AUTOMATON_ERROR_KINDS else "Unknown"
    )
    exit_code: int = AUTOMATON_EXIT_CODES[selected_kind]
    return exit_code


__all__: list[str] = [
    "AutomatonError",  # noqa: F822 -- resolved by module __getattr__.
    "DeadlineExceededError",  # noqa: F822 -- resolved by module __getattr__.
    "classify_exception",
    "exit_code_for",
]
