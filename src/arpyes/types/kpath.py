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
        """Flatten into JAX children and auxiliary data."""
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
        """Reconstruct from flattened components."""
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
    """
    nkpts_arr: Int[Array, " "] = jnp.asarray(
        num_kpoints, dtype=jnp.int32
    )
    indices_arr: Int[Array, " L"] = jnp.asarray(
        label_indices, dtype=jnp.int32
    )
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
