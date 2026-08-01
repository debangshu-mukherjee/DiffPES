"""Define k-space path and grid data structures.

Extended Summary
----------------
This module defines carriers for generated k-space paths and fixed-shape
grids. It also keeps :class:`KPathInfo` for metadata from VASP KPOINTS files.

Routine Listings
----------------
:class:`KPathInfo`
    Store k-point path metadata in a JAX PyTree.
:class:`KPath`
    Store a generated path through fractional k-space.
:class:`KGrid`
    Store a fixed-shape raster in fractional k-space.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
:func:`make_kpath`
    Create a validated path through fractional k-space.
:func:`make_kgrid`
    Create a validated fixed-shape k-space raster.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from jax.core import Tracer
from jaxtyping import Array, Float, Float64, Int, Int32, jaxtyped

from .aliases import ScalarFloat

_KPATH_MODES: tuple[str, ...] = ("Automatic", "Line-mode", "Explicit")


class KPathInfo(eqx.Module):
    """Store k-point path metadata in a JAX PyTree.

    This type stores Brillouin-zone path information from VASP KPOINTS files.
    It includes plotting labels and label indices. Mode-specific metadata
    includes an automatic grid and shift, explicit k-points and weights, and
    line-mode segment endpoints.

    JAX stores numerical fields as PyTree children. It stores string metadata
    as auxiliary data because JAX cannot trace Python strings.


    :see: :class:`~.test_kpath.TestKPathInfo`

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
    kpoints : Optional[Float64[Array, "K 3"]]
        Mode-specific k-points:
        line mode -> segment endpoints (segments + 1),
        explicit mode -> listed k-points,
        automatic mode -> None.
    weights : Optional[Float64[Array, " K"]]
        Explicit-mode per-k-point weights (None otherwise).
    grid : Optional[Int[Array, " 3"]]
        Automatic-mode Monkhorst-Pack/Gamma grid (None otherwise).
    shift : Optional[Float64[Array, " 3"]]
        Automatic-mode grid shift (None otherwise).
    mode : str
        KPOINTS file mode (Automatic, Line-mode, Explicit; **static** -- a
        compile-time constant; changing it triggers retracing).
    labels : tuple[str, ...]
        Symmetry point labels (e.g., Gamma, M, K; **static** -- compile-time
        constants; changing them triggers retracing).
    comment : str
        Raw comment from KPOINTS line 1 (**static** -- a compile-time
        constant; changing it triggers retracing).
    coordinate_mode : str
        Coordinate/scheme line metadata:
        line/explicit -> Reciprocal/Cartesian line,
        automatic -> scheme line (e.g., Monkhorst-Pack; **static** -- a
        compile-time constant; changing it triggers retracing).

    Notes
    -----
    ``eqx.field(static=True)`` declares string metadata as auxiliary data
    instead of traced leaves. Changing any static value triggers
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
    kpoints: Optional[Float64[Array, "K 3"]]
    weights: Optional[Float64[Array, " K"]]
    grid: Optional[Int[Array, " 3"]]
    shift: Optional[Float64[Array, " 3"]]
    mode: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    comment: str = eqx.field(static=True)
    coordinate_mode: str = eqx.field(static=True)


class KPath(eqx.Module):
    """Store a generated path through fractional k-space.

    This PyTree carries the dense path from a k-space builder. The numerical
    coordinates remain traced. Labels, their indices, and the segment size
    remain static because they describe the compiled path shape.

    :see: :class:`~.test_kpath.TestKPath`

    Attributes
    ----------
    kpoints : Float64[Array, "n_k 3"]
        Fractional k-points.
    kz : Optional[Float64[Array, ""]]
        Fixed Cartesian out-of-plane momentum in 1/Angstrom. ``None`` means
        that the path has no separate fixed value.
    labels : tuple[str, ...]
        Labels for the path anchors. This field is **static**. A change causes
        JAX to retrace the receiving function.
    label_indices : tuple[int, ...]
        Indices for the labels. This field is **static**. A change causes JAX
        to retrace the receiving function.
    n_per_segment : int
        Number of points in each segment. This field is **static**. A change
        causes JAX to retrace the receiving function.

    See Also
    --------
    KPathInfo : Store k-point path metadata in a JAX PyTree.
    make_kpath : Create a validated path through fractional k-space.
    """

    kpoints: Float64[Array, "n_k 3"]
    kz: Optional[Float64[Array, ""]]
    labels: tuple[str, ...] = eqx.field(static=True)
    label_indices: tuple[int, ...] = eqx.field(static=True)
    n_per_segment: int = eqx.field(static=True)


class KGrid(eqx.Module):
    """Store a fixed-shape raster in fractional k-space.

    The flattened coordinates remain traced for geometry inversion. The
    static mesh shape lets compiled consumers restore the raster without a
    data-dependent shape operation.

    :see: :class:`~.test_kpath.TestKGrid`

    Attributes
    ----------
    kpoints : Float64[Array, "n_k 3"]
        Flattened fractional k-points.
    kz : Optional[Float64[Array, ""]]
        Fixed Cartesian out-of-plane momentum in 1/Angstrom. ``None`` marks a
        grid with varying or unspecified out-of-plane momentum.
    photon_energy_axis_ev : Optional[Float64[Array, " n_rows"]]
        Photon energy for each raster row in eV. ``None`` marks a grid without
        a photon-energy axis.
    mesh_shape : tuple[int, int]
        Raster shape as ``(n_rows, n_cols)``. This field is **static**. A
        change causes JAX to retrace the receiving function.

    See Also
    --------
    make_kgrid : Create a validated fixed-shape k-space raster.
    """

    kpoints: Float64[Array, "n_k 3"]
    kz: Optional[Float64[Array, ""]]
    photon_energy_axis_ev: Optional[Float64[Array, " n_rows"]]
    mesh_shape: tuple[int, int] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_kpath_info(  # noqa: DOC503, PLR0913
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

    The factory validates and normalizes raw k-path
    metadata before constructing a ``KPathInfo`` PyTree. Integer
    The factory casts integer scalars and arrays to ``int32``. It casts float
    arrays to ``float64``. The factory casts optional fields only when they
    are present. Thus, ``None`` continues to identify modes that omit them.

    ``@jaxtyped(typechecker=beartype)`` checks shape constraints at call time.
    The factory passes string metadata unchanged and stores it as PyTree
    auxiliary data.

    Use this factory when constructing ``KPathInfo`` from parsed
    KPOINTS data or when building synthetic k-paths for testing. The
    factory ensures consistent dtypes and handles the three KPOINTS
    modes (Line-mode, Explicit, Automatic) through the optional
    fields.

    :see: :class:`~.test_kpath.TestMakeKPathInfo`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           nkpts_arr = jnp.asarray(num_kpoints, dtype=jnp.int32)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           mode not in _KPATH_MODES

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~(nkpts_arr >= 0)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return kpath

       The explicit name keeps the implementation and the Returns section
       synchronized.

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
        KPOINTS file mode (**static** -- a compile-time constant; changing it
        triggers retracing). Default is ``"Line-mode"``.
    labels : tuple[str, ...], optional
        Symmetry point labels (**static** -- compile-time constants; changing
        them triggers retracing). Default is empty tuple.
    comment : str, optional
        KPOINTS comment line (**static** -- a compile-time constant; changing
        it triggers retracing). Default is empty string.
    coordinate_mode : str, optional
        Coordinate/scheme line metadata (**static** -- a compile-time
        constant; changing it triggers retracing). Default is empty string.

    Returns
    -------
    kpath : KPathInfo
        Validated k-path info instance.

    Raises
    ------
    ValueError
        If ``mode`` has an unsupported value. The function also rejects
        incomplete or inconsistent line-mode metadata.
    EquinoxRuntimeError
        If ``num_kpoints`` is negative, a traced line-mode segment count is
        less than one, or supplied ``kpoints``, ``weights``, or ``shift``
        values are non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction for an
    unsupported mode or inconsistent line-mode metadata. Traced validation
    uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for negative
    k-point counts, invalid traced segment counts, and non-finite optional
    floating-point arrays.

    See Also
    --------
    KPathInfo : The PyTree class constructed by this factory.
    """
    if mode not in _KPATH_MODES:
        msg: str = f"make_kpath_info: mode must be one of {_KPATH_MODES}"
        raise ValueError(msg)
    if mode == "Line-mode" and len(label_indices) < 1:
        msg: str = "make_kpath_info: at least one label index is required"
        raise ValueError(msg)
    if mode == "Line-mode" and labels and len(labels) != len(label_indices):
        msg: str = "make_kpath_info: labels and label_indices must agree"
        raise ValueError(msg)

    nkpts_arr: Int32[Array, " "] = jnp.asarray(num_kpoints, dtype=jnp.int32)
    indices_arr: Int32[Array, " L"] = jnp.asarray(
        label_indices, dtype=jnp.int32
    )
    pps_arr: Int32[Array, " "] = jnp.asarray(
        points_per_segment, dtype=jnp.int32
    )
    segments_arr: Int32[Array, " "] = jnp.asarray(segments, dtype=jnp.int32)
    if (
        mode == "Line-mode"
        and not isinstance(segments_arr, Tracer)
        and int(segments_arr) < 1
    ):
        msg: str = "make_kpath_info: line mode requires at least one segment"
        raise ValueError(msg)
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

    def validate_and_create() -> KPathInfo:
        nonlocal kpoints_arr, nkpts_arr, segments_arr, shift_arr, weights_arr
        nkpts_arr = eqx.error_if(
            nkpts_arr,
            ~(nkpts_arr >= 0),
            "make_kpath_info: num kpoints non negative",
        )
        if mode == "Line-mode":
            segments_arr = eqx.error_if(
                segments_arr,
                segments_arr < 1,
                "make_kpath_info: line mode requires at least one segment",
            )
        if kpoints_arr is not None:
            kpoints_arr = eqx.error_if(
                kpoints_arr,
                ~jnp.all(jnp.isfinite(kpoints_arr)),
                "make_kpath_info: kpoints must be finite",
            )
        if weights_arr is not None:
            weights_arr = eqx.error_if(
                weights_arr,
                ~jnp.all(jnp.isfinite(weights_arr)),
                "make_kpath_info: weights must be finite",
            )
        if shift_arr is not None:
            shift_arr = eqx.error_if(
                shift_arr,
                ~jnp.all(jnp.isfinite(shift_arr)),
                "make_kpath_info: shift must be finite",
            )
        validated_kpath: KPathInfo = KPathInfo(
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
        return validated_kpath

    kpath: KPathInfo = validate_and_create()
    return kpath


@jaxtyped(typechecker=beartype)
def make_kpath(  # noqa: DOC503
    kpoints: Float[Array, "n_k 3"],
    labels: tuple[str, ...] = (),
    label_indices: tuple[int, ...] = (),
    n_per_segment: int = 1,
    kz: Optional[ScalarFloat] = None,
) -> KPath:
    """Create a validated path through fractional k-space.

    The factory validates the static plotting metadata and the traced path
    coordinates. It preserves an optional fixed Cartesian out-of-plane value.

    :see: :class:`~.test_kpath.TestMakeKPath`

    Implementation Logic
    --------------------
    1. **Validate the path structure**::

           label_indices[-1] < kpoints.shape[0]

       The static checks keep labels within the fixed path shape.

    2. **Validate the traced values**::

           checked_kpoints = eqx.error_if(kpoints_array, invalid, message)

       The runtime checks remain active during compiled execution.

    3. **Return the named path**::

           return kpath

       The result keeps only the plotting metadata static.

    Parameters
    ----------
    kpoints : Float[Array, "n_k 3"]
        Fractional k-points.
    labels : tuple[str, ...], optional
        Labels for the path anchors. This value is **static**. A change causes
        retracing. Default is an empty tuple.
    label_indices : tuple[int, ...], optional
        Indices for the labels. This value is **static**. A change causes
        retracing. Default is an empty tuple.
    n_per_segment : int, optional
        Number of points in each segment. This value is **static**. A change
        causes retracing. Default is 1.
    kz : Optional[ScalarFloat], optional
        Fixed Cartesian out-of-plane momentum in 1/Angstrom. Default is
        ``None``.

    Returns
    -------
    kpath : KPath
        Validated path with traced fractional coordinates.

    Raises
    ------
    ValueError
        If the static metadata has invalid lengths, indices, or segment size.
    EquinoxRuntimeError
        If a traced k-point or fixed out-of-plane value is non-finite.

    Notes
    -----
    The k-points and optional ``kz`` value carry gradients. The labels,
    indices, and segment size do not carry gradients.
    """
    if n_per_segment < 1:
        message: str = "n_per_segment must be positive"
        raise ValueError(message)
    if len(labels) != len(label_indices):
        message = "labels and label_indices must have equal lengths"
        raise ValueError(message)
    if any(index < 0 or index >= kpoints.shape[0] for index in label_indices):
        message = "label_indices must be within the k-point range"
        raise ValueError(message)
    if any(
        next_index <= index
        for index, next_index in zip(
            label_indices, label_indices[1:], strict=False
        )
    ):
        message = "label_indices must be strictly increasing"
        raise ValueError(message)

    kpoints_array: Float64[Array, "n_k 3"] = jnp.asarray(
        kpoints, dtype=jnp.float64
    )
    checked_kpoints: Float[Array, "n_k 3"] = eqx.error_if(
        kpoints_array,
        ~jnp.all(jnp.isfinite(kpoints_array)),
        "kpoints must be finite",
    )
    kz_array: Optional[Float[Array, ""]] = None
    if kz is not None:
        kz_array = jnp.asarray(kz, dtype=jnp.float64)
        kz_array = eqx.error_if(
            kz_array, ~jnp.isfinite(kz_array), "kz must be finite"
        )
    kpath: KPath = KPath(
        kpoints=checked_kpoints,
        kz=kz_array,
        labels=labels,
        label_indices=label_indices,
        n_per_segment=n_per_segment,
    )
    return kpath


@jaxtyped(typechecker=beartype)
def make_kgrid(  # noqa: DOC503
    kpoints: Float[Array, "n_k 3"],
    mesh_shape: tuple[int, int],
    kz: Optional[ScalarFloat] = None,
    photon_energy_axis_ev: Optional[Float[Array, " n_rows"]] = None,
) -> KGrid:
    """Create a validated fixed-shape k-space raster.

    The factory validates the static raster shape and traced numerical data.
    It preserves an optional photon-energy axis for a photon-energy map.

    :see: :class:`~.test_kpath.TestMakeKGrid`

    Implementation Logic
    --------------------
    1. **Validate the raster shape**::

           mesh_shape[0] * mesh_shape[1] == kpoints.shape[0]

       This check keeps the flattened point count consistent with the raster.

    2. **Validate the traced values**::

           checked_kpoints = eqx.error_if(kpoints_array, invalid, message)

       The runtime checks remain active during compiled execution.

    3. **Return the named grid**::

           return kgrid

       The result keeps only the raster shape static.

    Parameters
    ----------
    kpoints : Float[Array, "n_k 3"]
        Flattened fractional k-points.
    mesh_shape : tuple[int, int]
        Raster shape as ``(n_rows, n_cols)``. This value is **static**. A
        change causes retracing.
    kz : Optional[ScalarFloat], optional
        Fixed Cartesian out-of-plane momentum in 1/Angstrom. Default is
        ``None``.
    photon_energy_axis_ev : Optional[Float[Array, " n_rows"]], optional
        Photon energy for each raster row in eV. Default is ``None``.

    Returns
    -------
    kgrid : KGrid
        Validated grid with traced fractional coordinates.

    Raises
    ------
    ValueError
        If the mesh dimensions are not positive or their product is wrong.
        The factory also rejects a photon-energy axis with the wrong length.
    EquinoxRuntimeError
        If traced data is non-finite or a photon energy is not positive.

    Notes
    -----
    The mesh shape controls array reshaping and remains static. All numerical
    fields carry gradients when they are present.
    """
    if mesh_shape[0] < 1 or mesh_shape[1] < 1:
        message: str = "mesh_shape dimensions must be positive"
        raise ValueError(message)
    if mesh_shape[0] * mesh_shape[1] != kpoints.shape[0]:
        message = "mesh_shape product must equal the k-point count"
        raise ValueError(message)
    if (
        photon_energy_axis_ev is not None
        and photon_energy_axis_ev.shape[0] != mesh_shape[0]
    ):
        message = "photon_energy_axis_ev length must equal n_rows"
        raise ValueError(message)

    kpoints_array: Float64[Array, "n_k 3"] = jnp.asarray(
        kpoints, dtype=jnp.float64
    )
    checked_kpoints: Float[Array, "n_k 3"] = eqx.error_if(
        kpoints_array,
        ~jnp.all(jnp.isfinite(kpoints_array)),
        "kpoints must be finite",
    )
    kz_array: Optional[Float[Array, ""]] = None
    if kz is not None:
        kz_array = jnp.asarray(kz, dtype=jnp.float64)
        kz_array = eqx.error_if(
            kz_array, ~jnp.isfinite(kz_array), "kz must be finite"
        )
    photon_energies: Optional[Float[Array, " n_rows"]] = None
    if photon_energy_axis_ev is not None:
        photon_energies = jnp.asarray(photon_energy_axis_ev, dtype=jnp.float64)
        photon_energies = eqx.error_if(
            photon_energies,
            ~jnp.all(jnp.isfinite(photon_energies))
            | ~jnp.all(photon_energies > 0.0),
            "photon_energy_axis_ev must be finite and positive",
        )
    kgrid: KGrid = KGrid(
        kpoints=checked_kpoints,
        kz=kz_array,
        photon_energy_axis_ev=photon_energies,
        mesh_shape=mesh_shape,
    )
    return kgrid


__all__: list[str] = [
    "KGrid",
    "KPath",
    "KPathInfo",
    "make_kgrid",
    "make_kpath",
    "make_kpath_info",
]
