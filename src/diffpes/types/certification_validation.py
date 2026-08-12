"""Validate shared certification values.

Extended Summary
----------------
This private module normalizes static text and traced arrays for
the certification carrier factories.
"""

import json

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Any, Optional, Tuple
from jaxtyping import Array, Bool, Float64, Int32


def _require_text(value: str, name: str) -> str:
    """PRIVATE: Reject empty static vocabulary entries.

    Parameters
    ----------
    value : str
        Candidate static vocabulary string.
    name : str
        Field name used in the static error message.

    Returns
    -------
    value : str
        The validated input string, unchanged.

    Raises
    ------
    ValueError
        If ``value`` contains only whitespace or is empty. This is the
        static construction-time contract.

    Notes
    -----
    Apply ``str.strip`` for the check and keep the original
    representation on success.
    """
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_optional_text(value: Optional[str], name: str) -> Optional[str]:
    """PRIVATE: Reject an explicitly supplied empty optional string.

    Parameters
    ----------
    value : Optional[str]
        Candidate optional string, or ``None`` when absent.
    name : str
        Field name passed to the static error message.

    Returns
    -------
    value : Optional[str]
        ``None`` when absent, otherwise the validated nonblank string.

    Notes
    -----
    Pass ``None`` through untouched. Delegate a present value to
    ``_require_text``, which raises the static ``ValueError`` for blank
    text.
    """
    if value is not None:
        result: Optional[str] = _require_text(value, name)
        return result
    return value


def _text_tuple(values: Tuple[str, ...], name: str) -> Tuple[str, ...]:
    """PRIVATE: Normalize and validate a tuple of identifiers.

    Implementation Logic
    --------------------
    Validate every entry through ``_require_text`` while rebuilding the
    tuple. Then compare the set size against the tuple length to reject
    duplicates.

    Parameters
    ----------
    values : Tuple[str, ...]
        Candidate identifier entries.
    name : str
        Field name used in the static error messages.

    Returns
    -------
    result : Tuple[str, ...]
        The validated identifiers frozen into a tuple in input order.

    Raises
    ------
    ValueError
        If the entries are not unique. ``_require_text`` also raises
        ``ValueError`` for a blank entry. This is the static
        construction-time contract.
    """
    result: Tuple[str, ...] = tuple(
        _require_text(value, name) for value in values
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_object(value: str, name: str) -> str:
    """PRIVATE: Require a JSON object while preserving the supplied text.

    Implementation Logic
    --------------------
    Decode with ``json.loads`` only to validate. Return the supplied
    string so that checksums over the stored representation stay
    stable.

    Parameters
    ----------
    value : str
        Candidate JSON document as text.
    name : str
        Field name used in the static error messages.

    Returns
    -------
    value : str
        The original JSON text, byte-for-byte unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not valid JSON, or if the decoded document is
        not a JSON object. This is the static construction-time
        contract.
    """
    error: json.JSONDecodeError
    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return value


def _float(value: Any, name: str, ndim: int) -> Float64[Array, "..."]:
    """PRIVATE: Cast a numerical value to float64 and enforce its rank.

    Parameters
    ----------
    value : Any
        Numerical value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Float64[Array, "..."]
        The value cast to a float64 JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The numerical content
    stays a traced leaf.
    """
    array: Float64[Array, "..."] = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _bool(value: Any, name: str, ndim: int) -> Bool[Array, "..."]:
    """PRIVATE: Cast a logical value to bool and enforce its rank.

    Parameters
    ----------
    value : Any
        Logical value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Bool[Array, "..."]
        The value cast to a boolean JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The logical content stays
    a traced leaf.
    """
    array: Bool[Array, "..."] = jnp.asarray(value, dtype=jnp.bool_)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _int(value: Any, name: str, ndim: int) -> Int32[Array, "..."]:
    """PRIVATE: Cast an integer value to int32 and enforce its rank.

    Parameters
    ----------
    value : Any
        Integer value convertible by ``jnp.asarray``.
    name : str
        Field name used in the static error message.
    ndim : int
        Required array rank.

    Returns
    -------
    array : Int32[Array, "..."]
        The value cast to an int32 JAX array.

    Raises
    ------
    ValueError
        If the cast array does not have rank ``ndim``. This is the
        static construction-time contract.

    Notes
    -----
    The rank check is static shape metadata. The integer content stays
    a traced leaf.
    """
    array: Int32[Array, "..."] = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _nonnegative(
    array: Float64[Array, "..."], name: str
) -> Float64[Array, "..."]:
    """PRIVATE: Require finite, nonnegative tolerance-like leaves under JIT.

    Parameters
    ----------
    array : Float64[Array, "..."]
        Traced numerical leaf to guard.
    name : str
        Field name used in the traced error message.

    Returns
    -------
    result : Float64[Array, "..."]
        The same values with the runtime check attached.

    Notes
    -----
    Attach a traced ``eqx.error_if`` guard instead of raising a static
    ``ValueError``. The check runs under JIT and fails at run time when
    any element is nonfinite or negative.
    """
    result: Float64[Array, "..."] = eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array >= 0.0)),
        f"{name} must be finite and nonnegative",
    )
    return result


def _positive(
    array: Float64[Array, "..."], name: str
) -> Float64[Array, "..."]:
    """PRIVATE: Require finite, positive scale leaves under JIT.

    Parameters
    ----------
    array : Float64[Array, "..."]
        Traced numerical leaf to guard.
    name : str
        Field name used in the traced error message.

    Returns
    -------
    result : Float64[Array, "..."]
        The same values with the runtime check attached.

    Notes
    -----
    Attach a traced ``eqx.error_if`` guard instead of raising a static
    ``ValueError``. The check runs under JIT and fails at run time when
    any element is nonfinite or not strictly positive.
    """
    result: Float64[Array, "..."] = eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array > 0.0)),
        f"{name} must be finite and positive",
    )
    return result


__all__: list[str] = []
