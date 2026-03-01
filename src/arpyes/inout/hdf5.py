"""HDF5 serializer and deserializer for arpyes PyTrees.

Extended Summary
----------------
Provides functions for saving and loading arpyes PyTree objects
(NamedTuples registered with ``@register_pytree_node_class``)
to and from HDF5 files via ``h5py``. Each PyTree's named fields
become HDF5 datasets, and auxiliary data (non-JAX metadata) is
stored as HDF5 group attributes in JSON format.

Routine Listings
----------------
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.

Notes
-----
All eight arpyes PyTree types are supported:
``DensityOfStates``, ``BandStructure``, ``ArpesSpectrum``,
``OrbitalProjection``, ``SimulationParams``,
``PolarizationConfig``, ``KPathInfo``, ``CrystalGeometry``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Union

from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    CrystalGeometry,
    DensityOfStates,
    KPathInfo,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    SOCVolumetricData,
    SpinOrbitalProjection,
    VolumetricData,
)

_ATTR_TYPE: str = "_pytree_type"
_ATTR_AUX: str = "_aux_data_json"
_ATTR_NONE: str = "_none_fields"
_KPATH_AUX_WITH_COMMENT_LEN: int = 3
_KPATH_AUX_WITH_COORD_MODE_LEN: int = 4


@dataclass(frozen=True)
class _PyTreeMeta:
    """Serialization metadata for a registered PyTree type.

    Stores the class reference, the ordered names of JAX-traced
    children fields (matching the order returned by
    ``tree_flatten``), and encoder/decoder callables for
    converting auxiliary data to and from JSON-serializable form.

    Parameters
    ----------
    cls : type
        The NamedTuple PyTree class.
    children_fields : tuple[str, ...]
        Ordered field names of JAX array children.
    aux_encoder : Callable[[Any], Any]
        Converts aux_data to a JSON-serializable value.
    aux_decoder : Callable[[Any], Any]
        Converts JSON-decoded value back to the Python type
        expected by ``tree_unflatten``.
    """

    cls: Any  # PyTree NamedTuple class
    children_fields: tuple[str, ...]
    aux_encoder: Callable[[Any], Any]
    aux_decoder: Callable[[Any], Any]


def _encode_none(
    _aux: None,  # noqa: ARG001
) -> None:
    """Encode PyTree auxiliary data ``None`` for JSON storage.

    Used for PyTree types that have no auxiliary data (e.g. DensityOfStates).
    The value is returned unchanged; when written to JSON it becomes
    ``null``.

    Parameters
    ----------
    _aux : None
        The auxiliary data to encode (must be None).

    Returns
    -------
    None
        Unchanged; serializers write this as JSON ``null``.
    """


def _decode_none(
    _val: None,  # noqa: ARG001
) -> None:
    """Decode JSON ``null`` back to PyTree auxiliary data ``None``.

    Inverse of ``_encode_none``. Used when loading PyTrees that have
    no auxiliary data.

    Parameters
    ----------
    _val : None
        The decoded value (expected to be None / JSON null).

    Returns
    -------
    None
        The reconstructed auxiliary data for the PyTree.
    """


def _encode_int(aux: int) -> int:
    """Encode PyTree auxiliary integer for JSON storage.

    Converts the Python int to a JSON-serialisable integer. Used for
    types that store a single int in auxiliary data (e.g. SimulationParams
    fidelity).

    Parameters
    ----------
    aux : int
        The auxiliary integer to encode.

    Returns
    -------
    int
        The same value, guaranteed to be a plain Python int for JSON.
    """
    return int(aux)


def _decode_int(val: Any) -> int:  # noqa: ANN401
    """Decode a JSON integer back to Python int for PyTree auxiliary data.

    Inverse of ``_encode_int``. Accepts any value and casts to int so that
    JSON number types are correctly restored.

    Parameters
    ----------
    val : Any
        The value read from JSON (typically an int or float).

    Returns
    -------
    int
        The reconstructed auxiliary integer.
    """
    return int(val)


def _encode_str(aux: str) -> str:
    """Encode PyTree auxiliary string for JSON storage.

    Returns the string unchanged so it can be written as a JSON string.
    Used for single-string auxiliary fields.

    Parameters
    ----------
    aux : str
        The auxiliary string to encode.

    Returns
    -------
    str
        The same string for JSON serialisation.
    """
    return str(aux)


def _decode_str(val: Any) -> str:  # noqa: ANN401
    """Decode a JSON string back to Python str for PyTree auxiliary data.

    Inverse of ``_encode_str``. Converts the loaded value to str so that
    non-string JSON types (if any) are normalised.

    Parameters
    ----------
    val : Any
        The value read from JSON (typically a string).

    Returns
    -------
    str
        The reconstructed auxiliary string.
    """
    return str(val)


def _encode_tuple_str(
    aux: tuple[str, ...],
) -> list[str]:
    """Encode PyTree auxiliary tuple of strings for JSON storage.

    JSON does not support tuples; the tuple is converted to a list of
    strings so that it can be serialised. Used for types that store
    sequences of strings (e.g. KPathInfo labels).

    Parameters
    ----------
    aux : tuple[str, ...]
        The auxiliary tuple of strings to encode.

    Returns
    -------
    list[str]
        A list of the same strings for JSON array serialisation.
    """
    return list(aux)


def _decode_tuple_str(
    val: Any,  # noqa: ANN401
) -> tuple[str, ...]:
    """Decode a JSON array of strings to a tuple for PyTree auxiliary data.

    Inverse of ``_encode_tuple_str``. Each element is coerced to str and
    the result is returned as an immutable tuple.

    Parameters
    ----------
    val : Any
        The value read from JSON (typically a list of strings).

    Returns
    -------
    tuple[str, ...]
        The reconstructed auxiliary tuple of strings.
    """
    return tuple(str(s) for s in val)


def _encode_kpath_aux(
    aux: tuple[str, tuple[str, ...], str, str],
) -> list[Any]:
    """Encode KPathInfo auxiliary string metadata for JSON storage.

    KPathInfo stores ``(mode, labels, comment, coordinate_mode)`` as
    auxiliary data.
    """
    mode, labels, comment, coordinate_mode = aux
    return [
        str(mode),
        list(labels),
        str(comment),
        str(coordinate_mode),
    ]


def _decode_kpath_aux(
    val: Any,  # noqa: ANN401
) -> tuple[str, tuple[str, ...], str, str]:
    """Decode JSON list back to KPathInfo auxiliary string metadata.

    Supports both the legacy format ``[mode, labels]`` and the new
    format ``[mode, labels, comment, coordinate_mode]``.
    """
    mode: str = str(val[0])
    labels: tuple[str, ...] = tuple(str(s) for s in val[1])
    comment: str = (
        str(val[2]) if len(val) >= _KPATH_AUX_WITH_COMMENT_LEN else ""
    )
    coordinate_mode: str = (
        str(val[3]) if len(val) >= _KPATH_AUX_WITH_COORD_MODE_LEN else ""
    )
    return (mode, labels, comment, coordinate_mode)


def _encode_volumetric_aux(
    aux: tuple[tuple[int, int, int], tuple[str, ...]],
) -> list[Any]:
    """Encode VolumetricData auxiliary data for JSON storage.

    VolumetricData stores ``(grid_shape, symbols)`` as auxiliary data.
    """
    grid_shape, symbols = aux
    return [list(grid_shape), list(symbols)]


def _decode_volumetric_aux(
    val: Any,  # noqa: ANN401
) -> tuple[tuple[int, int, int], tuple[str, ...]]:
    """Decode JSON list back to VolumetricData auxiliary data."""
    grid_shape: tuple[int, int, int] = (
        int(val[0][0]),
        int(val[0][1]),
        int(val[0][2]),
    )
    symbols: tuple[str, ...] = tuple(str(s) for s in val[1])
    return (grid_shape, symbols)


_PYTREE_REGISTRY: dict[str, _PyTreeMeta] = {
    "DensityOfStates": _PyTreeMeta(
        cls=DensityOfStates,
        children_fields=(
            "energy",
            "total_dos",
            "fermi_energy",
        ),
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "BandStructure": _PyTreeMeta(
        cls=BandStructure,
        children_fields=(
            "eigenvalues",
            "kpoints",
            "kpoint_weights",
            "fermi_energy",
        ),
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "ArpesSpectrum": _PyTreeMeta(
        cls=ArpesSpectrum,
        children_fields=(
            "intensity",
            "energy_axis",
        ),
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "OrbitalProjection": _PyTreeMeta(
        cls=OrbitalProjection,
        children_fields=(
            "projections",
            "spin",
            "oam",
        ),
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "SpinOrbitalProjection": _PyTreeMeta(
        cls=SpinOrbitalProjection,
        children_fields=(
            "projections",
            "spin",
            "oam",
        ),
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "SimulationParams": _PyTreeMeta(
        cls=SimulationParams,
        children_fields=(
            "energy_min",
            "energy_max",
            "sigma",
            "gamma",
            "temperature",
            "photon_energy",
        ),
        aux_encoder=_encode_int,
        aux_decoder=_decode_int,
    ),
    "PolarizationConfig": _PyTreeMeta(
        cls=PolarizationConfig,
        children_fields=(
            "theta",
            "phi",
            "polarization_angle",
        ),
        aux_encoder=_encode_str,
        aux_decoder=_decode_str,
    ),
    "KPathInfo": _PyTreeMeta(
        cls=KPathInfo,
        children_fields=(
            "num_kpoints",
            "label_indices",
            "points_per_segment",
            "segments",
            "kpoints",
            "weights",
            "grid",
            "shift",
        ),
        aux_encoder=_encode_kpath_aux,
        aux_decoder=_decode_kpath_aux,
    ),
    "CrystalGeometry": _PyTreeMeta(
        cls=CrystalGeometry,
        children_fields=(
            "lattice",
            "reciprocal_lattice",
            "coords",
            "atom_counts",
        ),
        aux_encoder=_encode_tuple_str,
        aux_decoder=_decode_tuple_str,
    ),
    "VolumetricData": _PyTreeMeta(
        cls=VolumetricData,
        children_fields=(
            "lattice",
            "coords",
            "charge",
            "magnetization",
            "atom_counts",
        ),
        aux_encoder=_encode_volumetric_aux,
        aux_decoder=_decode_volumetric_aux,
    ),
    "SOCVolumetricData": _PyTreeMeta(
        cls=SOCVolumetricData,
        children_fields=(
            "lattice",
            "coords",
            "charge",
            "magnetization",
            "magnetization_vector",
            "atom_counts",
        ),
        aux_encoder=_encode_volumetric_aux,
        aux_decoder=_decode_volumetric_aux,
    ),
}


@beartype
def _dataset_write_kwargs(
    data: np.ndarray,
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, tuple[int, ...]]],
) -> dict[str, Any]:
    """Build ``h5py.create_dataset`` kwargs for one child array.

    Implementation Logic
    --------------------
    1. For scalar datasets (shape ``()``), return an empty dict since
       HDF5 filter/chunk flags are invalid for scalar datasets.
    2. For non-scalar datasets, forward supported storage flags:
       ``compression``, ``compression_opts``, ``shuffle``,
       ``fletcher32``, and ``chunks``.
    """
    if data.ndim == 0:
        return {}

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    if shuffle:
        kwargs["shuffle"] = True
    if fletcher32:
        kwargs["fletcher32"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return kwargs


@beartype
def save_to_h5(
    path: Union[str, Path],
    /,
    *,
    compression: Optional[str] = None,
    compression_opts: Any = None,  # noqa: ANN401
    shuffle: bool = False,
    fletcher32: bool = False,
    chunks: Optional[Union[bool, tuple[int, ...]]] = None,
    **pytrees: Any,  # noqa: ANN401
) -> None:
    """Save one or more named PyTrees to an HDF5 file.

    Serializes each keyword-argument PyTree into a named HDF5
    group. JAX array fields become HDF5 datasets (named by the
    NamedTuple field name), and auxiliary data is stored as a
    JSON-encoded group attribute.

    Implementation Logic
    --------------------
    1. **Validate inputs**:
       Ensure at least one PyTree is provided.

    2. **Iterate over keyword arguments**:
       For each ``(group_name, pytree)`` pair:

       a. Look up ``type(pytree).__name__`` in
          ``_PYTREE_REGISTRY`` to obtain serialization metadata.

       b. Call ``pytree.tree_flatten()`` to separate the PyTree
          into ``(children, aux_data)``.

       c. Create an HDF5 group named ``group_name``.

       d. Store the type name as ``_pytree_type`` attribute and
          the JSON-encoded aux_data as ``_aux_data_json``.

       e. For each child field: if the value is ``None``
          (Optional field), record the field name in
          ``_none_fields``; otherwise create an HDF5 dataset
          from ``numpy.asarray(child)`` with optional storage
          flags (compression/chunk/checksum) for non-scalar
          datasets.

       f. Store the ``_none_fields`` list as a JSON attribute.

    Parameters
    ----------
    path : Union[str, Path]
        File path for the HDF5 file to create.
    compression : Optional[str], optional
        HDF5 compression filter name (e.g. ``"gzip"``, ``"lzf"``).
        Applied to non-scalar datasets only.
    compression_opts : Any, optional
        Compression options passed through to h5py (e.g. gzip level).
        Must be ``None`` when ``compression`` is ``None``.
    shuffle : bool, optional
        If True, enable HDF5 shuffle filter on non-scalar datasets.
    fletcher32 : bool, optional
        If True, enable HDF5 Fletcher32 checksum on non-scalar datasets.
    chunks : Optional[Union[bool, tuple[int, ...]]], optional
        Chunking policy for non-scalar datasets. ``True`` enables
        auto-chunking, or provide an explicit chunk-shape tuple.
    **pytrees : PyTree
        Named PyTree instances. Each keyword argument name
        becomes an HDF5 group name.

    Raises
    ------
    ValueError
        If no PyTrees are provided.
    ValueError
        If ``compression_opts`` is provided without ``compression``.
    TypeError
        If a PyTree's class is not in the registry.

    Notes
    -----
    Scalar datasets (shape ``()``) are always written without HDF5
    filter/chunk flags because those options are invalid for scalar
    dataspace in HDF5.
    """
    if not pytrees:
        msg = "At least one PyTree must be provided."
        raise ValueError(msg)
    if compression is None and compression_opts is not None:
        msg = "compression_opts requires compression to be set."
        raise ValueError(msg)

    file_path: Path = Path(path)
    with h5py.File(file_path, "w") as f:
        for group_name, pytree in pytrees.items():
            type_name: str = type(pytree).__name__
            if type_name not in _PYTREE_REGISTRY:
                msg = f"Unsupported PyTree type: {type_name}"
                raise TypeError(msg)

            meta: _PyTreeMeta = _PYTREE_REGISTRY[type_name]
            children, aux_data = pytree.tree_flatten()

            grp: h5py.Group = f.create_group(group_name)
            grp.attrs[_ATTR_TYPE] = type_name
            aux_serializable = meta.aux_encoder(aux_data)
            grp.attrs[_ATTR_AUX] = json.dumps(aux_serializable)

            none_fields: list[str] = []
            for field_name, child in zip(
                meta.children_fields,
                children,
                strict=True,
            ):
                if child is None:
                    none_fields.append(field_name)
                else:
                    child_arr: np.ndarray = np.asarray(child)
                    ds_kwargs: dict[str, Any] = _dataset_write_kwargs(
                        data=child_arr,
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=shuffle,
                        fletcher32=fletcher32,
                        chunks=chunks,
                    )
                    grp.create_dataset(
                        field_name,
                        data=child_arr,
                        **ds_kwargs,
                    )
            grp.attrs[_ATTR_NONE] = json.dumps(none_fields)


@beartype
def load_from_h5(
    path: Union[str, Path],
    name: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load PyTrees from an HDF5 file.

    Deserializes HDF5 groups back into arpyes PyTree objects
    by reading datasets as JAX arrays and reconstructing the
    NamedTuple via ``tree_unflatten``.

    Implementation Logic
    --------------------
    1. **Open the HDF5 file** for reading.

    2. **Select groups to load**:
       If ``name`` is provided, load only that group. If
       ``name`` is ``None``, load all top-level groups.

    3. **For each group**:

       a. Read ``_pytree_type`` attribute and look up the class
          in ``_PYTREE_REGISTRY``.

       b. Read ``_aux_data_json`` attribute and decode it via
          the type-specific ``aux_decoder``.

       c. Read ``_none_fields`` attribute (defaulting to empty
          list).

       d. For each children field name: if the name appears in
          ``_none_fields``, set the child to ``None``;
          otherwise read the HDF5 dataset and convert to a JAX
          array via ``jnp.asarray``.

       e. Call ``cls.tree_unflatten(aux_data, children)`` to
          reconstruct the PyTree.

    Parameters
    ----------
    path : Union[str, Path]
        File path to the HDF5 file to read.
    name : Optional[str], optional
        Name of a specific group to load. If ``None``, all
        groups are loaded and returned as a dict.

    Returns
    -------
    result : PyTree or dict[str, PyTree]
        A single PyTree if ``name`` is given, otherwise a dict
        mapping group names to PyTree instances.

    Raises
    ------
    KeyError
        If ``name`` is specified but does not exist in the file.
    TypeError
        If a group's ``_pytree_type`` is not in the registry.
    """
    file_path: Path = Path(path)

    def _load_group(
        grp: h5py.Group,
    ) -> Any:  # noqa: ANN401
        type_name: str = str(grp.attrs[_ATTR_TYPE])
        if type_name not in _PYTREE_REGISTRY:
            msg = f"Unknown PyTree type: {type_name}"
            raise TypeError(msg)

        meta: _PyTreeMeta = _PYTREE_REGISTRY[type_name]
        aux_json = json.loads(str(grp.attrs[_ATTR_AUX]))
        aux_data = meta.aux_decoder(aux_json)

        none_fields: list[str] = json.loads(str(grp.attrs[_ATTR_NONE]))

        children: list[Any] = []
        for field_name in meta.children_fields:
            if field_name in none_fields:
                children.append(None)
            else:
                arr = grp[field_name][()]
                children.append(jnp.asarray(arr))

        return meta.cls.tree_unflatten(aux_data, tuple(children))

    with h5py.File(file_path, "r") as f:
        if name is not None:
            if name not in f:
                msg = f"Group '{name}' not found in {path}"
                raise KeyError(msg)
            return _load_group(f[name])

        result: dict[str, Any] = {}
        for group_name in f:
            result[group_name] = _load_group(f[group_name])
        return result


__all__: list[str] = [
    "load_from_h5",
    "save_to_h5",
]
