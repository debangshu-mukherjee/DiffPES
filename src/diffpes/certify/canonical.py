"""Represent scientific records canonically for certification.

Extended Summary
----------------
This module turns the certification layer's supported values into
deterministic byte records. The representation distinguishes lists from
tuples and records each array dtype and shape. It also distinguishes Python
scalars from zero-dimensional arrays. The module normalizes text to NFC.
Numerical arrays use little-endian C-order bytes.

Canonicalization provides bookkeeping at the Python/JAX boundary. Do not call
it from a traced numerical kernel. It contributes no physical or numerical
certification claim.

Routine Listings
----------------
:func:`canonical_json`
    Return deterministic typed JSON bytes for ``value``.
:func:`canonical_pytree`
    Return canonical bytes for a supported carrier or PyTree.
:func:`iter_canonical_pytree_chunks`
    Yield canonical carrier bytes in bounded chunks.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import struct
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import Any, List, Tuple, cast
from jaxtyping import Bool, Num, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import (
    CANONICAL_ARRAY_CHUNK_BYTES,
    CANONICAL_JSON_PREFIX,
    CANONICAL_PYTREE_PREFIX,
    CANONICAL_SUPPORTED_ARRAY_KINDS,
)


def _normalize_text(value: str) -> str:
    """PRIVATE: Return the NFC-normalized form of ``value``.

    Parameters
    ----------
    value : str
        Text in any Unicode normalization state.

    Returns
    -------
    normalized : str
        The NFC-normalized text.

    Notes
    -----
    Applies :func:`unicodedata.normalize` with the ``NFC`` form so that
    canonically equivalent text produces identical canonical bytes.
    """
    normalized: str = unicodedata.normalize("NFC", value)
    return normalized


def _length(value: int) -> bytes:
    """PRIVATE: Encode a nonnegative record length as an unsigned 64-bit
    integer.

    Parameters
    ----------
    value : int
        Nonnegative length or count of a canonical record component.

    Returns
    -------
    encoded : bytes
        Eight big-endian bytes that encode ``value``.

    Raises
    ------
    ValueError
        If ``value`` is negative.

    Notes
    -----
    Packs ``value`` with the ``struct`` format ``>Q`` so every length
    field in a canonical record has a fixed width and byte order.
    """
    if value < 0:
        msg: str = "canonical record lengths must be nonnegative"
        raise ValueError(msg)
    encoded: bytes = struct.pack(">Q", value)
    return encoded


def _text_record(tag: bytes, value: str) -> bytes:
    """PRIVATE: Encode a tagged normalized UTF-8 text record.

    Parameters
    ----------
    tag : bytes
        One-byte type tag that names the record kind.
    value : str
        Text payload for the record.

    Returns
    -------
    record : bytes
        The tag, the big-endian payload length, and the NFC-normalized
        UTF-8 payload, in that order.

    Notes
    -----
    Normalizes ``value`` with :func:`_normalize_text` before encoding so
    that equivalent text yields one record. The length prefix makes the
    record self-delimiting.
    """
    encoded: bytes = _normalize_text(value).encode("utf-8")
    record: bytes = tag + _length(len(encoded)) + encoded
    return record


def _qualified_name(value_type: type[Any]) -> str:
    """PRIVATE: Return a stable module-qualified type name.

    Parameters
    ----------
    value_type : type[Any]
        Type whose identity enters a canonical record.

    Returns
    -------
    name : str
        The ``module.qualname`` string for ``value_type``.

    Notes
    -----
    Joins ``__module__`` and ``__qualname__`` with a dot so dataclass and
    enum records name their type without ambiguity across modules.
    """
    name: str = f"{value_type.__module__}.{value_type.__qualname__}"
    return name


def _float_bits(value: float) -> str:
    """PRIVATE: Return exact IEEE-754 binary64 bits as lowercase
    hexadecimal.

    Parameters
    ----------
    value : float
        Finite floating-point number.

    Returns
    -------
    bits : str
        Sixteen lowercase hexadecimal digits of the big-endian binary64
        encoding of ``value``.

    Raises
    ------
    ValueError
        If ``value`` is NaN or infinite.

    Notes
    -----
    Packs ``value`` with the ``struct`` format ``>d`` and hex-encodes the
    result. The bit-exact encoding keeps the canonical record free of
    decimal rounding.
    """
    if not math.isfinite(value):
        msg: str = "canonical records reject NaN and infinite floats"
        raise ValueError(msg)
    bits: str = struct.pack(">d", value).hex()
    return bits


def _json_node(value: object) -> object:  # noqa: PLR0911
    """PRIVATE: Convert JSON-like input to an explicitly typed JSON tree.

    Implementation Logic
    --------------------
    Dispatches on the concrete type. Integers become decimal strings and
    floats become binary64 hex through :func:`_float_bits`, so the JSON
    layer cannot round them. Mapping entries get NFC-normalized keys and
    a sort by the UTF-8 byte order of the keys. Tuples and lists keep
    distinct tags. The function recurses into containers.

    Parameters
    ----------
    value : object
        JSON-like value: ``None``, ``bool``, ``int``, ``float``, ``str``,
        ``tuple``, ``list``, or a mapping with string keys.

    Returns
    -------
    node : object
        A tree of single-key dictionaries. Each key states the value
        type: ``$none``, ``$bool``, ``$int``, ``$float64``, ``$str``,
        ``$tuple``, ``$list``, or ``$map``.

    Raises
    ------
    ValueError
        If a mapping key is not a string, if two mapping keys collide
        after Unicode normalization, or if the value type is not
        supported.
    """
    key: Any
    item: Any
    if value is None:
        node: object = {"$none": True}
    elif isinstance(value, bool):
        node = {"$bool": value}
    elif isinstance(value, int):
        node = {"$int": str(value)}
    elif isinstance(value, float):
        node = {"$float64": _float_bits(value)}
    elif isinstance(value, str):
        node = {"$str": _normalize_text(value)}
    elif isinstance(value, tuple):
        node = {"$tuple": [_json_node(item) for item in value]}
    elif isinstance(value, list):
        node = {"$list": [_json_node(item) for item in value]}
    elif isinstance(value, Mapping):
        normalized: List[Tuple[str, object]] = []
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                msg: str = "canonical JSON mappings require string keys"
                raise ValueError(msg)
            normalized_key: str = _normalize_text(key)
            if normalized_key in seen:
                msg: str = "mapping keys collide after Unicode normalization"
                raise ValueError(msg)
            seen.add(normalized_key)
            normalized.append((normalized_key, item))
        normalized.sort(key=lambda pair: pair[0].encode("utf-8"))
        node = {
            "$map": [
                [{"$str": key}, _json_node(item)] for key, item in normalized
            ]
        }
    else:
        msg: str = f"unsupported canonical JSON value: {type(value)!r}"
        raise ValueError(msg)
    return node


@jaxtyped(typechecker=beartype)
def canonical_json(value: object) -> bytes:
    """Return deterministic typed JSON bytes for ``value``.

    The record preserves scalar, container, array-dtype, and array-shape
    identity. It rejects values that have no finite deterministic
    representation.

    :see: :class:`~.test_canonical.TestCanonicalJson`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           encoded: bytes = CANONICAL_JSON_PREFIX + payload

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    value : object
        JSON-like data. Mapping keys must be strings. The function accepts
        tuples and distinguishes them from lists.

    Returns
    -------
    encoded : bytes
        Versioned canonical UTF-8 JSON record.
    """
    node: object = _json_node(value)
    payload: bytes = json.dumps(
        node,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded: bytes = CANONICAL_JSON_PREFIX + payload
    return encoded


def _is_array(value: object) -> bool:
    """PRIVATE: Return whether ``value`` exposes a concrete NumPy/JAX
    array protocol.

    Parameters
    ----------
    value : object
        Candidate value from a canonical PyTree.

    Returns
    -------
    is_array : bool
        ``True`` when the value is a NumPy array or scalar, or when it
        has ``__array__``, ``dtype``, and ``shape`` attributes together.

    Notes
    -----
    The attribute triple accepts JAX arrays without an import of JAX.
    The check is structural only; :func:`_canonical_array` still rejects
    tracers when it materializes the value.
    """
    if isinstance(value, np.ndarray | np.generic):
        is_array: bool = True
        return is_array
    is_array: bool = all(
        hasattr(value, attr) for attr in ("__array__", "dtype", "shape")
    )
    return is_array


def _canonical_array(
    value: object,
) -> Bool[NDArray, "..."] | Num[NDArray, "..."]:
    """PRIVATE: Materialize one array in the canonical dtype and memory
    order.

    Implementation Logic
    --------------------
    Calls :func:`numpy.asarray` and wraps any failure as a
    :class:`ValueError`. Validates the dtype kind against
    ``CANONICAL_SUPPORTED_ARRAY_KINDS`` and checks finiteness for float
    and complex kinds. Then re-materializes the array with the
    little-endian dtype and C memory order so the raw bytes are
    platform-independent.

    Parameters
    ----------
    value : object
        Array-like value that passed :func:`_is_array`.

    Returns
    -------
    canonical : Bool[NDArray, "..."] | Num[NDArray, "..."]
        NumPy array with a little-endian dtype in C order.

    Raises
    ------
    ValueError
        If the value cannot become a concrete NumPy array, for example
        a JAX tracer. Also if the dtype kind lacks support, or a float
        or complex array contains NaN or infinity.
    """
    exc: Exception
    try:
        array: Bool[NDArray, "..."] | Num[NDArray, "..."] = np.asarray(value)
    except Exception as exc:
        msg: str = (
            "canonicalization requires concrete arrays and cannot consume "
            "a JAX tracer"
        )
        raise ValueError(msg) from exc
    if array.dtype.kind not in CANONICAL_SUPPORTED_ARRAY_KINDS:
        msg: str = (
            f"unsupported array dtype for canonicalization: {array.dtype}"
        )
        raise ValueError(msg)
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg: str = "canonical records reject arrays containing NaN or infinity"
        raise ValueError(msg)
    dtype: np.dtype[Any] = array.dtype.newbyteorder("<")
    canonical: Bool[NDArray, "..."] | Num[NDArray, "..."] = np.asarray(
        array, dtype=dtype, order="C"
    )
    return canonical


def _iter_array_chunks(
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[bytes | memoryview]:
    """PRIVATE: Yield a canonical typed header followed by bounded array
    chunks.

    Implementation Logic
    --------------------
    Materializes the array through :func:`_canonical_array`, records the
    dtype string and shape with fixed-width lengths, and then yields
    zero-copy :class:`memoryview` slices over the C-order payload. An
    empty array yields no payload slice.

    Parameters
    ----------
    value : object
        Array-like value to serialize.
    chunk_bytes : int
        Maximum payload bytes per yielded array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        The ``A`` tag with the dtype string record, the dimension count,
        each dimension length, and the payload byte count. Then payload
        slices of at most ``chunk_bytes`` bytes.
    """
    dimension: Any
    offset: Any
    array: Bool[NDArray, "..."] | Num[NDArray, "..."] = _canonical_array(value)
    dtype_text: bytes = array.dtype.str.encode("ascii")
    yield b"A" + _length(len(dtype_text)) + dtype_text
    yield _length(array.ndim)
    for dimension in array.shape:
        yield _length(int(dimension))
    yield _length(array.nbytes)
    if array.nbytes == 0:
        return
    payload: memoryview = memoryview(array).cast("B")
    for offset in range(0, array.nbytes, chunk_bytes):
        yield payload[offset : offset + chunk_bytes]


def _iter_mapping_chunks(
    value: Mapping[object, object],
    *,
    chunk_bytes: int,
) -> Iterator[bytes | memoryview]:
    """PRIVATE: Yield a mapping sorted by normalized text keys.

    Implementation Logic
    --------------------
    NFC-normalizes every key, rejects collisions, and sorts entries by
    the UTF-8 byte order of the normalized keys. Entry values recurse
    through :func:`_iter_value_chunks`.

    Parameters
    ----------
    value : Mapping[object, object]
        Mapping whose keys must all be strings.
    chunk_bytes : int
        Maximum payload bytes per yielded array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        The ``M`` tag with the entry count, then for each entry a ``K``
        key record followed by the canonical chunks of the value.

    Raises
    ------
    ValueError
        If a key is not a string, or if two keys collide after Unicode
        normalization.
    """
    key: Any
    item: Any
    normalized: List[Tuple[str, object]] = []
    seen: set[str] = set()
    for key, item in value.items():
        if not isinstance(key, str):
            msg: str = "canonical PyTree mappings require string keys"
            raise ValueError(msg)
        normalized_key: str = _normalize_text(key)
        if normalized_key in seen:
            msg: str = "mapping keys collide after Unicode normalization"
            raise ValueError(msg)
        seen.add(normalized_key)
        normalized.append((normalized_key, item))
    normalized.sort(key=lambda pair: pair[0].encode("utf-8"))
    yield b"M" + _length(len(normalized))
    for key, item in normalized:
        yield _text_record(b"K", key)
        yield from _iter_value_chunks(item, chunk_bytes=chunk_bytes)


def _iter_dataclass_chunks(
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[bytes | memoryview]:
    """PRIVATE: Yield dataclass or Equinox fields in declaration order.

    Implementation Logic
    --------------------
    Reads the fields with :func:`dataclasses.fields`, which returns them
    in declaration order, and recurses into each field value through
    :func:`_iter_value_chunks`. The type-name record separates equal
    field contents of different dataclass types.

    Parameters
    ----------
    value : object
        Dataclass instance, including any Equinox module.
    chunk_bytes : int
        Maximum payload bytes per yielded array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        An ``O`` record with the module-qualified type name and the
        field count. Then, per field, a ``K`` name record followed by
        the canonical chunks of the field value.
    """
    field: Any
    fields: Tuple[dataclasses.Field[Any], ...] = dataclasses.fields(
        cast(Any, value)
    )
    yield _text_record(b"O", _qualified_name(type(value)))
    yield _length(len(fields))
    for field in fields:
        yield _text_record(b"K", field.name)
        yield from _iter_value_chunks(
            getattr(value, field.name),
            chunk_bytes=chunk_bytes,
        )


def _iter_sequence_chunks(
    value: Sequence[object],
    *,
    tag: bytes,
    chunk_bytes: int,
) -> Iterator[bytes | memoryview]:
    """PRIVATE: Yield one tagged list or tuple record.

    Parameters
    ----------
    value : Sequence[object]
        List or tuple to serialize.
    tag : bytes
        One-byte container tag: ``T`` for tuples, ``L`` for lists.
    chunk_bytes : int
        Maximum payload bytes per yielded array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        The tag with the element count, then the canonical chunks of
        each element in order.

    Notes
    -----
    The distinct tags keep the list/tuple identity in the record. The
    elements recurse through :func:`_iter_value_chunks`.
    """
    item: Any
    yield tag + _length(len(value))
    for item in value:
        yield from _iter_value_chunks(item, chunk_bytes=chunk_bytes)


def _iter_value_chunks(  # noqa: PLR0912
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[bytes | memoryview]:
    """PRIVATE: Yield canonical chunks for one supported value.

    Implementation Logic
    --------------------
    Dispatches on the concrete type and assigns distinct tags. Assigns
    ``N``, ``B``, ``I``, ``F``, and ``C`` to none, bool, integer,
    binary64, and complex values. Assigns ``S``, ``Y``, ``P``, and
    ``E`` to text, bytes, paths, and enums. Assigns ``A``, ``O``,
    ``T``, ``L``, and ``M`` to arrays, dataclasses, tuples, lists, and
    mappings. The bool check runs before the int check because
    ``bool`` subclasses ``int``. Containers recurse through the
    dedicated ``_iter_*_chunks`` helpers.

    Parameters
    ----------
    value : object
        Supported scalar, container, array, dataclass, enum, path, or
        bytes value.
    chunk_bytes : int
        Maximum payload bytes per yielded array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        Tagged canonical chunks for the value.

    Raises
    ------
    ValueError
        If a complex value is not finite, or if the value type is not
        supported.
    """
    if value is None:
        yield b"N"
    elif isinstance(value, bool):
        yield b"B\x01" if value else b"B\x00"
    elif isinstance(value, int):
        yield _text_record(b"I", str(value))
    elif isinstance(value, float):
        yield b"F" + bytes.fromhex(_float_bits(value))
    elif isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            msg: str = "canonical records reject nonfinite complex values"
            raise ValueError(msg)
        yield b"C" + struct.pack(">dd", value.real, value.imag)
    elif isinstance(value, str):
        yield _text_record(b"S", value)
    elif isinstance(value, bytes):
        yield b"Y" + _length(len(value)) + value
    elif isinstance(value, Path):
        yield _text_record(b"P", value.as_posix())
    elif isinstance(value, enum.Enum):
        yield _text_record(b"E", _qualified_name(type(value)))
        yield from _iter_value_chunks(value.value, chunk_bytes=chunk_bytes)
    elif _is_array(value):
        yield from _iter_array_chunks(value, chunk_bytes=chunk_bytes)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield from _iter_dataclass_chunks(value, chunk_bytes=chunk_bytes)
    elif isinstance(value, tuple):
        yield from _iter_sequence_chunks(
            value,
            tag=b"T",
            chunk_bytes=chunk_bytes,
        )
    elif isinstance(value, list):
        yield from _iter_sequence_chunks(
            value,
            tag=b"L",
            chunk_bytes=chunk_bytes,
        )
    elif isinstance(value, Mapping):
        mapping: Mapping[object, object] = cast(
            "Mapping[object, object]",
            value,
        )
        yield from _iter_mapping_chunks(mapping, chunk_bytes=chunk_bytes)
    else:
        msg: str = f"unsupported canonical PyTree value: {type(value)!r}"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def iter_canonical_pytree_chunks(
    tree: object,
    *,
    chunk_bytes: int = CANONICAL_ARRAY_CHUNK_BYTES,
) -> Iterator[bytes | memoryview]:
    """Yield canonical carrier bytes in bounded chunks.

    The record preserves scalar, container, array-dtype, and array-shape
    identity. It rejects values that have no finite deterministic
    representation.

    :see: :class:`~.test_canonical.TestIterCanonicalPytreeChunks`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           yield from _iter_value_chunks(tree, chunk_bytes=chunk_bytes)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    tree : object
        A supported scientific carrier or nested PyTree-like value.
    chunk_bytes : int, optional
        Maximum payload bytes yielded for each numerical-array chunk.

    Yields
    ------
    chunk : bytes | memoryview
        Consecutive chunks of the canonical representation.

    Raises
    ------
    ValueError
        If ``chunk_bytes`` is not positive. This error also occurs if the tree
        contains an unsupported or nonfinite value.
    """
    if chunk_bytes <= 0:
        msg: str = "chunk_bytes must be positive"
        raise ValueError(msg)
    yield CANONICAL_PYTREE_PREFIX
    yield from _iter_value_chunks(tree, chunk_bytes=chunk_bytes)


@jaxtyped(typechecker=beartype)
def canonical_pytree(tree: object) -> bytes:
    """Return canonical bytes for a supported carrier or PyTree.

    The record preserves scalar, container, array-dtype, and array-shape
    identity. It rejects values that have no finite deterministic
    representation.

    :see: :class:`~.test_canonical.TestCanonicalPytree`


    Parameters
    ----------
    tree : object
        Supported nested scientific content. The record represents Equinox
        modules through their dataclass fields, including static metadata.

    Returns
    -------
    encoded : bytes
        Complete versioned canonical record.

    Notes
    -----
    Use :func:`iter_canonical_pytree_chunks` for a streaming checksum of a
    large array.
    """
    encoded: bytes = b"".join(iter_canonical_pytree_chunks(tree))
    return encoded


__all__: list[str] = [
    "canonical_json",
    "canonical_pytree",
    "iter_canonical_pytree_chunks",
]
