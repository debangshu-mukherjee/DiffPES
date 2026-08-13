"""Define scalar type aliases for JAX-compatible numeric types.

Extended Summary
----------------
This module provides type aliases that accept both native Python scalars and
zero-dimensional JAX arrays, enabling flexible function signatures
that work seamlessly with JAX transformations.

Routine Listings
----------------
:class:`PyTreeDef`
    Runtime pytree definition with a typed static-analysis stand-in.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:obj:`RetardedSelfEnergySource`
    Union of parametric and tabulated retarded self-energy sources.
:obj:`ScalarBool`
    Union of ``bool`` and ``Bool[Array, " "]``.
:obj:`ScalarComplex`
    Union of ``complex`` and ``Complex[Array, " "]``.
:obj:`ScalarFloat`
    Union of ``float`` and ``Float[Array, " "]``.
:obj:`ScalarInteger`
    Union of ``int`` and ``Int[Array, " "]``.
:obj:`ScalarNumeric`
    Union of ``int``, ``float``, ``complex``, and ``Num[Array, " "]``.
:obj:`SliceOperator`
    Union of dense and sparse finite-slice operators.

Notes
-----
These aliases mirror those in ``janssen.types`` to maintain a
consistent type annotation style across JAX-based code.
"""

from beartype.typing import TYPE_CHECKING, List, TypeAlias, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Num

if TYPE_CHECKING:
    from beartype.typing import Any, Iterable

    from .generalized_spectral import (
        ParametricSelfEnergy,  # noqa: F401
        TabulatedMatrixSelfEnergy,  # noqa: F401
    )
    from .ks_scattering import (  # noqa: F401
        DenseSliceOperator,
        SparseSliceOperator,
    )

    class PyTreeDef:
        """Represent the unstubbed jaxlib pytree definition statically.

        Notes
        -----
        The runtime name binds the genuine class from
        :mod:`jax.tree_util`, which lives in a compiled extension module
        without type stubs and therefore cannot appear in static type
        expressions.  This stand-in mirrors the members DiffPES uses.
        """

        @property
        def num_leaves(self) -> int:
            """Return the number of leaves in the flattened pytree."""
            ...

        @property
        def num_nodes(self) -> int:
            """Return the number of nodes in the pytree."""
            ...

        def unflatten(self, leaves: Iterable[Any]) -> Any:
            """Build a pytree from this definition and its leaves."""
            ...

        def children(self) -> List["PyTreeDef"]:
            """Return the definitions of the direct subtrees."""
            ...

        def flatten_up_to(self, xs: Any) -> List[Any]:
            """Flatten ``xs`` down to the depth of this definition."""
            ...

else:
    from jax.tree_util import PyTreeDef

NonJaxNumber: TypeAlias = Union[int, float, complex]
ScalarBool: TypeAlias = Union[bool, Bool[Array, " "]]
ScalarComplex: TypeAlias = Union[complex, Complex[Array, " "]]
ScalarFloat: TypeAlias = Union[float, Float[Array, " "]]
ScalarInteger: TypeAlias = Union[int, Int[Array, " "]]
ScalarNumeric: TypeAlias = Union[int, float, complex, Num[Array, " "]]
RetardedSelfEnergySource: TypeAlias = Union[
    "ParametricSelfEnergy", "TabulatedMatrixSelfEnergy"
]
SliceOperator: TypeAlias = Union["DenseSliceOperator", "SparseSliceOperator"]

__all__: list[str] = [
    "PyTreeDef",
    "NonJaxNumber",
    "RetardedSelfEnergySource",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SliceOperator",
]
