"""K-point path information data structure.

Extended Summary
----------------
Defines the :class:`KPathInfo` PyTree for storing Brillouin-zone
metadata parsed from VASP KPOINTS files, including both plotting
labels and mode-specific metadata (automatic grids, explicit weights,
line-mode segments/endpoints).
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, jaxtyped


@register_pytree_node_class
class KPathInfo(NamedTuple):
    """PyTree for k-point path metadata.

    Stores Brillouin-zone path information parsed from VASP KPOINTS
    files. Includes plotting fields (labels + label indices) and
    mode-specific metadata needed for full parser completeness:
    automatic-mode grid/shift, explicit-mode k-points/weights, and
    line-mode segment endpoints.

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``. Array-valued metadata is stored
    as children. String metadata is stored as auxiliary data because
    JAX cannot trace Python strings.

    Attributes
    ----------
    num_kpoints : Int[Array, " "]
        Total number of k-points in the path (line mode) or header
        count (explicit mode).
    label_indices : Int[Array, " L"]
        Indices of symmetry points along the path.
    points_per_segment : Int[Array, " "]
        Raw integer from line 2 of KPOINTS (line mode: points per
        segment).
    segments : Int[Array, " "]
        Number of line segments in line mode.
    kpoints : Optional[Float[Array, "K 3"]]
        Mode-specific k-points:
        line mode -> segment endpoints (segments + 1),
        explicit mode -> listed k-points,
        automatic mode -> None.
    weights : Optional[Float[Array, " K"]]
        Explicit-mode per-k-point weights (None otherwise).
    grid : Optional[Int[Array, " 3"]]
        Automatic-mode Monkhorst-Pack/Gamma grid (None otherwise).
    shift : Optional[Float[Array, " 3"]]
        Automatic-mode grid shift (None otherwise).
    mode : str
        KPOINTS file mode (Automatic, Line-mode, Explicit).
    labels : tuple[str, ...]
        Symmetry point labels (e.g., Gamma, M, K).
    comment : str
        Raw comment from KPOINTS line 1.
    coordinate_mode : str
        Coordinate/scheme line metadata:
        line/explicit -> Reciprocal/Cartesian line,
        automatic -> scheme line (e.g., Monkhorst-Pack).

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
    points_per_segment: Int[Array, " "]
    segments: Int[Array, " "]
    kpoints: Optional[Float[Array, "K 3"]]
    weights: Optional[Float[Array, " K"]]
    grid: Optional[Int[Array, " 3"]]
    shift: Optional[Float[Array, " 3"]]
    mode: str
    labels: tuple[str, ...]
    comment: str
    coordinate_mode: str

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Int[Array, " "],
            Int[Array, " L"],
            Int[Array, " "],
            Int[Array, " "],
            Optional[Float[Array, "K 3"]],
            Optional[Float[Array, " K"]],
            Optional[Int[Array, " 3"]],
            Optional[Float[Array, " 3"]],
        ],
        Tuple[str, tuple[str, ...], str, str],
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For KPathInfo, all
        array-valued metadata is stored in children and string metadata
        in aux_data.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in tracing):
           ``(num_kpoints, label_indices, points_per_segment,
           segments, kpoints, weights, grid, shift)``.
        2. **Auxiliary data** (static, not traced by JAX):
           ``(mode, labels, comment, coordinate_mode)``.
           JAX treats these as compile-time constants; any change
           triggers JIT recompilation.

        Returns
        -------
        children : tuple of Array
            Tuple of array-valued metadata fields.
        aux_data : tuple[str, tuple[str, ...], str, str]
            String metadata stored outside JAX tracing.
        """
        return (
            (
                self.num_kpoints,
                self.label_indices,
                self.points_per_segment,
                self.segments,
                self.kpoints,
                self.weights,
                self.grid,
                self.shift,
            ),
            (
                self.mode,
                self.labels,
                self.comment,
                self.coordinate_mode,
            ),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[str, tuple[str, ...], str, str],
        children: Tuple[
            Int[Array, " "],
            Int[Array, " L"],
            Int[Array, " "],
            Int[Array, " "],
            Optional[Float[Array, "K 3"]],
            Optional[Float[Array, " K"]],
            Optional[Int[Array, " 3"]],
            Optional[Float[Array, " 3"]],
        ],
    ) -> "KPathInfo":
        """Reconstruct a KPathInfo from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into the eight array-valued fields.
        2. Unpack ``aux_data`` into the four string fields.
        3. Pass all fields to the constructor.

        Parameters
        ----------
        aux_data : tuple[str, tuple[str, ...], str, str]
            String metadata recovered from auxiliary data.
        children : tuple of Array
            Array-valued metadata tuple.

        Returns
        -------
        kpath : KPathInfo
            Reconstructed instance with identical data.
        """
        (
            num_kpoints,
            label_indices,
            points_per_segment,
            segments,
            kpoints,
            weights,
            grid,
            shift,
        ) = children
        mode, labels, comment, coordinate_mode = aux_data
        return cls(
            num_kpoints=num_kpoints,
            label_indices=label_indices,
            points_per_segment=points_per_segment,
            segments=segments,
            kpoints=kpoints,
            weights=weights,
            grid=grid,
            shift=shift,
            mode=mode,
            labels=labels,
            comment=comment,
            coordinate_mode=coordinate_mode,
        )


@jaxtyped(typechecker=beartype)
def make_kpath_info(  # noqa: PLR0913
    num_kpoints: Union[int, Int[Array, " "]],
    label_indices: Union[Int[Array, " L"], "list[int]"],
    points_per_segment: Union[int, Int[Array, " "]] = 0,
    segments: Union[int, Int[Array, " "]] = 0,
    kpoints: Optional[Float[Array, "K 3"]] = None,
    weights: Optional[Float[Array, " K"]] = None,
    grid: Optional[Union[Int[Array, " 3"], "list[int]"]] = None,
    shift: Optional[Float[Array, " 3"]] = None,
    mode: str = "Line-mode",
    labels: tuple[str, ...] = (),
    comment: str = "",
    coordinate_mode: str = "",
) -> KPathInfo:
    """Create a validated KPathInfo instance.

    Validates and normalises the inputs before constructing the
    KPathInfo PyTree. Integer inputs are cast to int32 JAX arrays,
    float inputs to float64 arrays, and optional array fields are
    cast when provided.

    Implementation Logic
    --------------------
    1. Cast integer scalar/array fields to ``jnp.int32``.
    2. Cast optional float/array metadata to ``jnp.float64`` if set.
    3. Pass string metadata through unchanged (stored as aux data).
    4. Construct and return ``KPathInfo``.

    Parameters
    ----------
    num_kpoints : Union[int, Int[Array, " "]]
        Total number of k-points along the path.
    label_indices : Union[Int[Array, " L"], list[int]]
        Indices of symmetry points along the path.
    points_per_segment : Union[int, Int[Array, " "]], optional
        Raw value from line 2 of KPOINTS. Default is 0.
    segments : Union[int, Int[Array, " "]], optional
        Number of path segments in line mode. Default is 0.
    kpoints : Optional[Float[Array, "K 3"]], optional
        Mode-specific k-point coordinates. Default is None.
    weights : Optional[Float[Array, " K"]], optional
        Explicit-mode weights. Default is None.
    grid : Optional[Union[Int[Array, " 3"], list[int]]], optional
        Automatic-mode MP/Gamma grid. Default is None.
    shift : Optional[Float[Array, " 3"]], optional
        Automatic-mode grid shift. Default is None.
    mode : str, optional
        KPOINTS file mode. Default is ``"Line-mode"``.
    labels : tuple[str, ...], optional
        Symmetry point labels. Default is empty tuple.
    comment : str, optional
        KPOINTS comment line. Default is empty string.
    coordinate_mode : str, optional
        Coordinate/scheme line metadata. Default is empty string.

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
    pps_arr: Int[Array, " "] = jnp.asarray(points_per_segment, dtype=jnp.int32)
    segments_arr: Int[Array, " "] = jnp.asarray(segments, dtype=jnp.int32)
    kpoints_arr: Optional[Float[Array, "K 3"]] = None
    if kpoints is not None:
        kpoints_arr = jnp.asarray(kpoints, dtype=jnp.float64)
    weights_arr: Optional[Float[Array, " K"]] = None
    if weights is not None:
        weights_arr = jnp.asarray(weights, dtype=jnp.float64)
    grid_arr: Optional[Int[Array, " 3"]] = None
    if grid is not None:
        grid_arr = jnp.asarray(grid, dtype=jnp.int32)
    shift_arr: Optional[Float[Array, " 3"]] = None
    if shift is not None:
        shift_arr = jnp.asarray(shift, dtype=jnp.float64)
    kpath: KPathInfo = KPathInfo(
        num_kpoints=nkpts_arr,
        label_indices=indices_arr,
        points_per_segment=pps_arr,
        segments=segments_arr,
        kpoints=kpoints_arr,
        weights=weights_arr,
        grid=grid_arr,
        shift=shift_arr,
        mode=mode,
        labels=labels,
        comment=comment,
        coordinate_mode=coordinate_mode,
    )
    return kpath


__all__: list[str] = [
    "KPathInfo",
    "make_kpath_info",
]
