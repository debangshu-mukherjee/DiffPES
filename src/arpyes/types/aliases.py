"""Scalar type aliases for JAX-compatible numeric types.

Extended Summary
----------------
Provides type aliases that accept both native Python scalars and
zero-dimensional JAX arrays, enabling flexible function signatures
that work seamlessly with JAX transformations.

Routine Listings
----------------
:data:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:data:`ScalarBool`
    Union of ``bool`` and ``Bool[Array, " "]``.
:data:`ScalarComplex`
    Union of ``complex`` and ``Complex[Array, " "]``.
:data:`ScalarFloat`
    Union of ``float`` and ``Float[Array, " "]``.
:data:`ScalarInteger`
    Union of ``int`` and ``Int[Array, " "]``.
:data:`ScalarNumeric`
    Union of ``int``, ``float``, ``complex``, and ``Num[Array, " "]``.

Notes
-----
These aliases mirror those in ``janssen.types`` to maintain a
consistent type annotation style across JAX-based projects.
"""

from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Num

NonJaxNumber: TypeAlias = Union[int, float, complex]

ScalarBool: TypeAlias = Union[bool, Bool[Array, " "]]

ScalarComplex: TypeAlias = Union[complex, Complex[Array, " "]]

ScalarFloat: TypeAlias = Union[float, Float[Array, " "]]

ScalarInteger: TypeAlias = Union[int, Int[Array, " "]]

ScalarNumeric: TypeAlias = Union[int, float, complex, Num[Array, " "]]

__all__: list[str] = [
    "NonJaxNumber",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
]
