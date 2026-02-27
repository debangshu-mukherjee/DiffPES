"""K-point path information data structure.

Extended Summary
----------------
Defines the :class:`KPathInfo` PyTree for storing Brillouin zone
path metadata from VASP KPOINTS files, including symmetry point
labels and their indices along the path.

Routine Listings
----------------
:class:`KPathInfo`
    PyTree for k-point path metadata.
:func:`make_kpath_info`
    Factory function for KPathInfo.

Notes
-----
String fields (``mode``, ``labels``) are stored as PyTree auxiliary
data since JAX cannot trace Python strings.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Int, jaxtyped


@register_pytree_node_class
class KPathInfo(NamedTuple):
    """PyTree for k-point path metadata.

    Stores Brillouin zone path information parsed from VASP KPOINTS
    files. This includes the total number of k-points, the integer
    indices at which high-symmetry labels fall along the path, the
    KPOINTS file mode string, and the human-readable symmetry point
    labels (e.g., Gamma, M, K). This metadata is needed to annotate
    the k-axis of ARPES band-structure plots.

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``. Integer array fields
    (``num_kpoints``, ``label_indices``) are stored as JAX children,
    while the Python string fields (``mode``, ``labels``) are stored
    as auxiliary data because JAX cannot trace Python strings.

    Attributes
    ----------
    num_kpoints : Int[Array, " "]
        Total number of k-points along the path.
    label_indices : Int[Array, " L"]
        Indices of symmetry points along the path.
    mode : str
        KPOINTS file mode (Automatic, Line-mode, Explicit).
    labels : tuple[str, ...]
        Symmetry point labels (e.g., Gamma, M, K).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    The ``mode`` string and ``labels`` tuple of strings are stored as
    auxiliary data (not children). JAX treats auxiliary data as
    compile-time constants: changing either value triggers
    recompilation of any ``jit``-compiled function that receives
    this PyTree.

    See Also
    --------
    make_kpath_info : Factory function with validation and int32
        casting.
    """

    num_kpoints: Int[Array, " "]
    label_indices: Int[Array, " L"]
    mode: str
    labels: tuple[str, ...]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Int[Array, " "], Int[Array, " L"]],
        Tuple[str, tuple[str, ...]],
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For KPathInfo, the
        two integer array fields are children and the two string
        fields are auxiliary data.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in tracing):
           ``(num_kpoints, label_indices)``
        2. **Auxiliary data** (static, not traced by JAX):
           ``(mode, labels)`` -- a pair of Python string values.
           JAX treats these as compile-time constants; any change
           triggers JIT recompilation.

        Returns
        -------
        children : tuple of Array
            Tuple of ``(num_kpoints, label_indices)`` JAX int arrays.
        aux_data : tuple[str, tuple[str, ...]]
            Pair of ``(mode, labels)`` stored outside JAX tracing.
        """
        return (
            (self.num_kpoints, self.label_indices),
            (self.mode, self.labels),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[str, tuple[str, ...]],
        children: Tuple[Int[Array, " "], Int[Array, " L"]],
    ) -> "KPathInfo":
        """Reconstruct a KPathInfo from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into two JAX int arrays:
           ``(num_kpoints, label_indices)``.
        2. Unpack ``aux_data`` into the two string fields:
           ``(mode, labels)``.
        3. Pass all four fields to the constructor, restoring the
           original KPathInfo layout.

        Parameters
        ----------
        aux_data : tuple[str, tuple[str, ...]]
            Pair of ``(mode, labels)`` recovered from auxiliary data.
        children : tuple of Array
            Tuple of ``(num_kpoints, label_indices)`` JAX int arrays.

        Returns
        -------
        kpath : KPathInfo
            Reconstructed instance with identical data.
        """
        num_kpoints, label_indices = children
        mode, labels = aux_data
        return cls(
            num_kpoints=num_kpoints,
            label_indices=label_indices,
            mode=mode,
            labels=labels,
        )


@jaxtyped(typechecker=beartype)
def make_kpath_info(
    num_kpoints: Union[int, Int[Array, " "]],
    label_indices: Union[Int[Array, " L"], "list[int]"],
    mode: str = "Line-mode",
    labels: tuple[str, ...] = (),
) -> KPathInfo:
    """Create a validated KPathInfo instance.

    Validates and normalises the inputs before constructing the
    KPathInfo PyTree. Integer inputs are cast to int32 JAX arrays,
    while string fields are passed through unchanged. Accepts both
    JAX arrays and plain Python ints / lists for convenience.

    Implementation Logic
    --------------------
    1. **Cast num_kpoints** to a 0-D ``jnp.int32`` array via
       ``jnp.asarray``. Accepts both Python ``int`` and existing
       JAX integer arrays.
    2. **Cast label_indices** to a 1-D ``jnp.int32`` array via
       ``jnp.asarray``. Accepts both JAX arrays and Python
       ``list[int]``.
    3. **Pass through** the ``mode`` string and ``labels`` tuple
       without modification (these become auxiliary data in the
       PyTree).
    4. **Construct** the ``KPathInfo`` NamedTuple from all four
       fields and return it.

    Parameters
    ----------
    num_kpoints : Union[int, Int[Array, " "]]
        Total number of k-points along the path.
    label_indices : Union[Int[Array, " L"], list[int]]
        Indices of symmetry points along the path.
    mode : str, optional
        KPOINTS file mode. Default is ``"Line-mode"``.
    labels : tuple[str, ...], optional
        Symmetry point labels. Default is empty tuple.

    Returns
    -------
    kpath : KPathInfo
        Validated k-path info instance.

    See Also
    --------
    KPathInfo : The PyTree class constructed by this factory.
    """
    nkpts_arr: Int[Array, " "] = jnp.asarray(num_kpoints, dtype=jnp.int32)
    indices_arr: Int[Array, " L"] = jnp.asarray(label_indices, dtype=jnp.int32)
    kpath: KPathInfo = KPathInfo(
        num_kpoints=nkpts_arr,
        label_indices=indices_arr,
        mode=mode,
        labels=labels,
    )
    return kpath


__all__: list[str] = [
    "KPathInfo",
    "make_kpath_info",
]
