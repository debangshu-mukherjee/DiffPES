"""Serialize and deserialize diffpes PyTrees in HDF5.

Extended Summary
----------------
The module saves and loads diffpes Equinox PyTrees in HDF5 files through
``h5py``. Each named array field becomes an HDF5 dataset. The codec stores
static metadata as JSON in HDF5 group attributes.

Routine Listings
----------------
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.

Notes
-----
The codec supports the complete registered set of numerical types-owned
carriers.
Dataclass fields define the serialization metadata.
The codec stores dynamic fields as datasets or recursive module groups.
It encodes ``eqx.field(static=True)`` values as tuple-preserving JSON.
Consequently, tight-binding carriers preserve nested crystal geometry as
recursive numerical children and their orbital basis as static metadata
without carrier-specific serialization rules.
Files predating tight-binding depth metadata load absent ``depths`` datasets
as the bulk sentinel ``None``.
"""

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Mapping, Optional, Tuple, Union
from jaxtyping import Shaped, jaxtyped
from numpy.typing import NDArray

from diffpes.types import (
    ATTR_AUX,
    ATTR_NONE,
    ATTR_TYPE,
    ArpesSpectrum,
    BandStructure,
    CrystalGeometry,
    DensityOfStates,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    FullDensityOfStates,
    KGrid,
    KPath,
    KPathInfo,
    MatrixElementParams,
    OrbitalBasis,
    OrbitalProjection,
    RadialQuadratureSpec,
    RadialSpec,
    SelfEnergyModel,
    SimulationParams,
    SlabSpec,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    SurfaceCell,
    TBModel,
    VolumetricData,
    WannierOperatorData,
    WorkflowContext,
)


def _pytree_classes() -> Tuple[type[eqx.Module], ...]:
    """PRIVATE: Return the complete carrier class set used by the codec.

    Returns
    -------
    classes : Tuple[type[eqx.Module], ...]
        Every registered Equinox carrier class the HDF5 codec can
        serialize.

    Notes
    -----
    The module-level ``_PYTREE_REGISTRY`` maps each class name from
    this tuple to its serialization metadata.  Registering a new
    carrier means listing it here.
    """
    classes: Tuple[type[eqx.Module], ...] = (
        ArpesSpectrum,
        BandStructure,
        CrystalGeometry,
        DensityOfStates,
        DiagonalizedBands,
        ExperimentGeometry,
        FullDensityOfStates,
        KGrid,
        KPath,
        KPathInfo,
        OrbitalBasis,
        OrbitalProjection,
        RadialQuadratureSpec,
        RadialSpec,
        MatrixElementParams,
        FinalStateSpec,
        SelfEnergyModel,
        SimulationParams,
        SlabSpec,
        SOCVolumetricData,
        SpinBandStructure,
        SpinOrbitalProjection,
        SurfaceCell,
        TBModel,
        VolumetricData,
        WannierOperatorData,
        WorkflowContext,
    )
    return classes


def _encode_static(value: Any) -> Any:  # noqa: ANN401
    """PRIVATE: Encode nested static Equinox metadata without losing tuple
    types.

    Parameters
    ----------
    value : Any
        Static field value: a tuple, an Equinox dataclass module, a
        list, a dict, a NumPy scalar, or a JSON-ready leaf.

    Returns
    -------
    encoded : Any
        JSON-serializable structure with tuples tagged as
        ``{"__tuple__": [...]}`` and nested modules tagged as
        ``{"__module__": <class name>, "fields": {...}}``.

    Implementation Logic
    --------------------
    Recurses structurally.  A tuple maps to the tagged list form.  An
    Equinox dataclass module maps to its class name plus encoded
    fields.  Lists and dicts encode their items with string keys.
    NumPy scalars convert to Python scalars with ``item()``.  Any
    other value passes through unchanged.
    """
    encoded: Any
    if isinstance(value, tuple):
        encoded = {"__tuple__": [_encode_static(item) for item in value]}
    elif isinstance(value, eqx.Module) and is_dataclass(value):
        encoded = {
            "__module__": type(value).__name__,
            "fields": {
                field.name: _encode_static(getattr(value, field.name))
                for field in fields(value)
            },
        }
    elif isinstance(value, list):
        encoded = [_encode_static(item) for item in value]
    elif isinstance(value, dict):
        encoded = {
            str(key): _encode_static(item) for key, item in value.items()
        }
    elif isinstance(value, np.generic):
        encoded = value.item()
    else:
        encoded = value
    return encoded


def _decode_static(value: Any) -> Any:  # noqa: ANN401
    """PRIVATE: Decode tuple-preserving and nested-module static metadata.

    Parameters
    ----------
    value : Any
        Parsed JSON value that may contain the ``__tuple__`` and
        ``__module__`` tags of :func:`_encode_static`.

    Returns
    -------
    decoded : Any
        Original static structure with tuples, nested carrier
        instances, lists, dicts, and leaves restored.

    Implementation Logic
    --------------------
    Recurses structurally and inverts every :func:`_encode_static`
    tag.  A ``__tuple__`` dict becomes a tuple.  A ``__module__``
    dict looks up its class in ``_PYTREE_REGISTRY`` and instantiates
    it from the decoded fields.  Untagged lists and dicts decode
    elementwise, and leaves pass through unchanged.
    """
    decoded: Any
    if isinstance(value, dict) and "__tuple__" in value:
        decoded = tuple(_decode_static(item) for item in value["__tuple__"])
    elif isinstance(value, dict) and "__module__" in value:
        class_name: str = str(value["__module__"])
        module_class: type[eqx.Module] = _PYTREE_REGISTRY[class_name]["cls"]
        module_fields: Dict[str, Any] = {
            str(name): _decode_static(item)
            for name, item in value["fields"].items()
        }
        decoded = module_class(**module_fields)
    elif isinstance(value, list):
        decoded = [_decode_static(item) for item in value]
    elif isinstance(value, dict):
        decoded = {
            str(key): _decode_static(item) for key, item in value.items()
        }
    else:
        decoded = value
    return decoded


def _decode_aux_data(type_name: str, value: Any) -> Any:  # noqa: ANN401
    """PRIVATE: Decode current static metadata and supported legacy HDF5 aux
    data.

    The pre-migration codec wrote plain JSON lists for tuple-valued
    metadata. Current files use explicit tuple tags. The three conversions
    below retain read compatibility with those pinned files without restoring
    a per-carrier write registry.

    Parameters
    ----------
    type_name : str
        Registered carrier class name read from the group attributes.
    value : Any
        Parsed JSON auxiliary payload of that carrier.

    Returns
    -------
    auxiliary_data : Any
        Decoded static payload ready for the carrier constructor.

    Implementation Logic
    --------------------
    Tagged dicts and ``None`` decode through :func:`_decode_static`.
    A legacy untagged list converts per carrier.  ``CrystalGeometry``
    restores its symbol tuple.  ``KPathInfo`` restores four string
    fields with a string tuple in position two.  ``SOCVolumetricData``
    and ``VolumetricData`` restore the integer grid-shape tuple and
    the symbol tuple.  Every other carrier keeps the plainly decoded
    value.
    """
    decoded: Any = _decode_static(value)
    auxiliary_data: Any
    if isinstance(value, dict) or value is None:
        auxiliary_data = decoded
    elif type_name == "CrystalGeometry":
        auxiliary_data = tuple(str(item) for item in value)
    elif type_name == "KPathInfo":
        auxiliary_data = (
            str(value[0]),
            tuple(str(item) for item in value[1]),
            str(value[2]),
            str(value[3]),
        )
    elif type_name in {"SOCVolumetricData", "VolumetricData"}:
        auxiliary_data = (
            tuple(int(item) for item in value[0]),
            tuple(str(item) for item in value[1]),
        )
    else:
        auxiliary_data = decoded
    return auxiliary_data


def _module_meta(module_class: type[eqx.Module]) -> Mapping[str, Any]:
    """PRIVATE: Build serialization metadata from Equinox dataclass fields.

    Parameters
    ----------
    module_class : type[eqx.Module]
        Registered carrier class to inspect.

    Returns
    -------
    metadata : Mapping[str, Any]
        Read-only mapping with the class under ``"cls"``, the dynamic
        field names under ``"children_fields"``, and the static field
        names under ``"static_fields"``.

    Notes
    -----
    Splits the dataclass fields on the Equinox ``static`` marker in
    ``field.metadata`` and keeps the declaration order inside each
    group.  A ``MappingProxyType`` wraps the result, so registry
    entries stay immutable.
    """
    module_fields: Tuple[Any, ...] = fields(module_class)
    children_fields: Tuple[str, ...] = tuple(
        field.name
        for field in module_fields
        if not bool(field.metadata.get("static", False))
    )
    static_fields: Tuple[str, ...] = tuple(
        field.name
        for field in module_fields
        if bool(field.metadata.get("static", False))
    )
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "cls": module_class,
            "children_fields": children_fields,
            "static_fields": static_fields,
        }
    )
    return metadata


_PYTREE_REGISTRY: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {cls.__name__: _module_meta(cls) for cls in _pytree_classes()}
)


def _optional_migration_fields() -> Mapping[str, frozenset[str]]:
    """PRIVATE: Return fields absent from historical serialized carriers.

    Returns
    -------
    fields_by_type : Mapping[str, frozenset[str]]
        Read-only map from carrier class name to the child fields a
        legacy file may omit; currently the ``depths`` dataset of
        ``DiagonalizedBands`` and ``TBModel``.

    Notes
    -----
    The loader treats a listed field that is absent from a group as
    ``None``, so files that predate the field still load.
    """
    fields_by_type: Mapping[str, frozenset[str]] = MappingProxyType(
        {
            "DiagonalizedBands": frozenset({"depths"}),
            "TBModel": frozenset({"depths"}),
        }
    )
    return fields_by_type


@beartype
def _dataset_write_kwargs(
    data: Shaped[NDArray, "..."],
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, Tuple[int, ...]]],
) -> Dict[str, Any]:
    """PRIVATE: Build ``h5py.create_dataset`` keyword arguments for one
    child array.

    Extended Summary
    ----------------
    HDF5 storage filters and chunking apply only to datasets with nonscalar
    dataspaces. This helper checks the array dimensions. It returns the
    applicable keyword dictionary for ``h5py.Group.create_dataset``.

    Implementation Logic
    --------------------
    1. For scalar datasets (``data.ndim == 0``), return an empty dict
       since HDF5 filter/chunk flags are invalid for scalar dataspace.
    2. For non-scalar datasets, conditionally include each supported
       storage flag (``compression``, ``compression_opts``, ``shuffle``,
       ``fletcher32``, ``chunks``) only when the corresponding argument
       is not ``None`` / ``False``.

    Parameters
    ----------
    data : Shaped[NDArray, "..."]
        The NumPy array for the dataset. Its ``ndim`` determines whether the
        filters apply.
    compression : Optional[str]
        HDF5 compression filter name, for example ``"gzip"`` or ``"lzf"``.
    compression_opts : Any
        Compression-specific options, for example a gzip level from 1 to 9.
    shuffle : bool
        Whether to enable the HDF5 byte-shuffle filter.
    fletcher32 : bool
        Whether to enable the Fletcher32 checksum filter.
    chunks : Optional[Union[bool, Tuple[int, ...]]]
        Chunking policy: ``True`` for auto-chunking, or an explicit
        chunk shape tuple.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments to pass to ``h5py.Group.create_dataset``.
        Empty dict for scalar datasets.
    """
    kwargs: Dict[str, Any] = {}
    if data.ndim != 0:
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


@jaxtyped(typechecker=beartype)
def save_to_h5(  # noqa: DOC503 -- recursive helper raises TypeError.
    path: Union[str, Path],
    /,
    *,
    compression: Optional[str] = None,
    compression_opts: Any = None,  # noqa: ANN401
    shuffle: bool = False,
    fletcher32: bool = False,
    chunks: Optional[Union[bool, Tuple[int, ...]]] = None,
    **pytrees: Any,  # noqa: ANN401
) -> None:
    """Save one or more named PyTrees to an HDF5 file.

    The function serializes each keyword PyTree into a named HDF5 group. JAX
    array fields become datasets with their Equinox field names. The codec
    stores static metadata in a JSON group attribute.

    :see: :class:`~.test_hdf5.TestSaveToH5`

    Implementation Logic
    --------------------
    1. **Reject an empty save request**::

           if not pytrees:
               msg: str = "At least one PyTree must be provided."
               raise ValueError(msg)

       This prevents creation of a file with no registered carrier groups.

    2. **Write each carrier through the registry codec**::

           file_path: Path = Path(path)
           with h5py.File(file_path, "w") as f:
               for group_name, pytree in pytrees.items():
                   grp: h5py.Group = f.create_group(group_name)
                   _write_module(grp, pytree)

       The recursive writer preserves child arrays and static metadata.
       It applies the storage flags under one group name.

    Parameters
    ----------
    path : Union[str, Path]
        File path for the HDF5 file to create.
    compression : Optional[str], optional
        HDF5 compression filter name, for example ``"gzip"`` or ``"lzf"``.
        Applied to non-scalar datasets only.
    compression_opts : Any, optional
        Compression options for h5py, for example the gzip level.
        Must be ``None`` when ``compression`` is ``None``.
    shuffle : bool, optional
        If True, enable HDF5 shuffle filter on non-scalar datasets.
    fletcher32 : bool, optional
        If True, enable HDF5 Fletcher32 checksum on non-scalar datasets.
    chunks : Optional[Union[bool, Tuple[int, ...]]], optional
        Chunking policy for non-scalar datasets. ``True`` enables
        auto-chunking, or provide an explicit chunk-shape tuple.
    **pytrees : Any
        Named PyTree instances. Each keyword argument name
        becomes an HDF5 group name.

    Raises
    ------
    ValueError
        If the caller provides no PyTrees.
    ValueError
        If the caller provides ``compression_opts`` without ``compression``.
    TypeError
        If a PyTree's class is not in the registry.

    Notes
    -----
    Scalar datasets (shape ``()``) are always written without HDF5
    filter/chunk flags because those options are invalid for scalar
    dataspace in HDF5.
    """
    f: h5py.File
    group_name: str
    pytree: eqx.Module

    if not pytrees:
        msg: str = "At least one PyTree must be provided."
        raise ValueError(msg)
    if compression is None and compression_opts is not None:
        msg: str = "compression_opts requires compression to be set."
        raise ValueError(msg)

    def _write_module(
        grp: h5py.Group,
        pytree: Any,  # noqa: ANN401
    ) -> None:
        """PRIVATE: Write one Equinox module, recursively storing module
        children.

        Implementation Logic
        --------------------
        Stores the class name in the type attribute and the encoded
        static payload as one JSON attribute.  One static field saves
        bare; several save as a tuple.  A ``None`` child writes as an
        entry in the none-field JSON attribute.  A nested module
        writes as a recursive subgroup.  Any other child writes as
        one dataset with the storage flags from
        :func:`_dataset_write_kwargs`.

        Parameters
        ----------
        grp : h5py.Group
            Open destination group for this module.
        pytree : Any
            Registered carrier instance to serialize.

        Raises
        ------
        TypeError
            If the class of ``pytree`` is not in ``_PYTREE_REGISTRY``.
        """
        field_name: str

        type_name: str = type(pytree).__name__
        if type_name not in _PYTREE_REGISTRY:
            msg: str = f"Unsupported PyTree type: {type_name}"
            raise TypeError(msg)
        meta: Mapping[str, Any] = _PYTREE_REGISTRY[type_name]
        static_values: Tuple[Any, ...] = tuple(
            getattr(pytree, field_name) for field_name in meta["static_fields"]
        )
        aux_data: Any = None
        if len(static_values) == 1:
            aux_data = static_values[0]
        elif static_values:
            aux_data = static_values
        grp.attrs[ATTR_TYPE] = type_name
        grp.attrs[ATTR_AUX] = json.dumps(_encode_static(aux_data))

        none_fields: list[str] = []
        for field_name in meta["children_fields"]:
            child: Any = getattr(pytree, field_name)
            if child is None:
                none_fields.append(field_name)
            elif isinstance(child, eqx.Module):
                child_group: h5py.Group = grp.create_group(field_name)
                _write_module(child_group, child)
            else:
                child_arr: Shaped[NDArray, "..."] = np.asarray(child)
                ds_kwargs: Dict[str, Any] = _dataset_write_kwargs(
                    data=child_arr,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                    fletcher32=fletcher32,
                    chunks=chunks,
                )
                grp.create_dataset(field_name, data=child_arr, **ds_kwargs)
        grp.attrs[ATTR_NONE] = json.dumps(none_fields)

    file_path: Path = Path(path)
    with h5py.File(file_path, "w") as f:
        for group_name, pytree in pytrees.items():
            grp: h5py.Group = f.create_group(group_name)
            _write_module(grp, pytree)


@jaxtyped(typechecker=beartype)
def load_from_h5(  # noqa: DOC502 -- raises occur under the HDF5 context.
    path: Union[str, Path],
    name: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load PyTrees from an HDF5 file.

    The function deserializes HDF5 groups into diffpes PyTrees. It reads the
    datasets as JAX arrays and reconstructs each Equinox module with keyword
    arguments.

    :see: :class:`~.test_hdf5.TestLoadFromH5`

    Implementation Logic
    --------------------
    1. **Open the requested HDF5 path**::

           file_path: Path = Path(path)
           with h5py.File(file_path, "r") as f:

       This gives named and all-group loads the same read-only file boundary.

    2. **Reconstruct registered groups recursively**::

           loaded: Any = _load_group(f[name])

       The nested loader restores child arrays, optional fields, and metadata.
       It then calls the registered Equinox carrier class.

    3. **Return the selected load result**::

           return loaded

       A named request returns one carrier. Other requests return a mapping.

    Parameters
    ----------
    path : Union[str, Path]
        File path to the HDF5 file to read.
    name : Optional[str], optional
        Name of a specific group to load. If ``None``, the function loads all
        groups and returns a dictionary.

    Returns
    -------
    loaded : PyTree or Dict[str, PyTree]
        One PyTree when ``name`` identifies a group. Otherwise, a dictionary
        that maps group names to PyTree instances.

    Raises
    ------
    KeyError
        If ``name`` identifies no group in the file.
    TypeError
        If a group's ``_pytree_type`` is not in the registry.
    """
    f: h5py.File
    group_name: str

    file_path: Path = Path(path)

    def _load_group(
        grp: h5py.Group,
    ) -> Any:  # noqa: ANN401
        """PRIVATE: Build one registered carrier from an HDF5 group.

        Implementation Logic
        --------------------
        Reads the type attribute, looks up the registry metadata, and
        decodes the JSON static payload with :func:`_decode_aux_data`.
        A child listed in the none-field attribute loads as ``None``;
        a subgroup recurses; a dataset loads as a JAX array.  A
        dataset that is absent but listed by
        :func:`_optional_migration_fields` also loads as ``None``.
        The loader calls the carrier class with children and static
        fields as keyword arguments.  One static field takes the
        payload bare; several unpack it as a tuple.

        Parameters
        ----------
        grp : h5py.Group
            Open source group that holds one serialized carrier.

        Returns
        -------
        loaded : Any
            Reconstructed carrier instance.

        Raises
        ------
        TypeError
            If the stored type name is not in ``_PYTREE_REGISTRY``.
        """
        field_name: str

        type_name: str = str(grp.attrs[ATTR_TYPE])
        if type_name not in _PYTREE_REGISTRY:
            msg: str = f"Unknown PyTree type: {type_name}"
            raise TypeError(msg)

        meta: Mapping[str, Any] = _PYTREE_REGISTRY[type_name]
        aux_json: Any = json.loads(str(grp.attrs[ATTR_AUX]))
        aux_data: Any = _decode_aux_data(type_name, aux_json)

        none_fields: list[str] = json.loads(str(grp.attrs[ATTR_NONE]))

        children: list[Any] = []
        for field_name in meta["children_fields"]:
            if field_name in none_fields or (
                field_name not in grp
                and field_name
                in _optional_migration_fields().get(type_name, frozenset())
            ):
                children.append(None)
            elif isinstance(grp[field_name], h5py.Group):
                children.append(_load_group(grp[field_name]))
            else:
                arr: Shaped[NDArray, "..."] = grp[field_name][()]
                children.append(jnp.asarray(arr))

        constructor_fields: Dict[str, Any] = dict(
            zip(meta["children_fields"], children, strict=True)
        )
        static_values: Tuple[Any, ...]
        if not meta["static_fields"]:
            static_values = ()
        elif len(meta["static_fields"]) == 1:
            static_values = (aux_data,)
        else:
            static_values = tuple(aux_data)
        constructor_fields.update(
            zip(meta["static_fields"], static_values, strict=True)
        )
        module_class: type[eqx.Module] = meta["cls"]
        loaded: Any = module_class(**constructor_fields)
        return loaded

    with h5py.File(file_path, "r") as f:
        if name is not None:
            if name not in f:
                msg: str = f"Group '{name}' not found in {path}"
                raise KeyError(msg)
            loaded: Any = _load_group(f[name])
            return loaded

        result: Dict[str, Any] = {}
        for group_name in f:
            result[group_name] = _load_group(f[group_name])
        loaded: Dict[str, Any] = result
        return loaded


__all__: list[str] = [
    "load_from_h5",
    "save_to_h5",
]
