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
The codec supports all twenty-three numerical types-owned carriers.
Dataclass fields define the serialization metadata.
The codec stores dynamic fields as datasets or recursive module groups.
It encodes ``eqx.field(static=True)`` values as tuple-preserving JSON.
Consequently, tight-binding carriers preserve nested crystal geometry as
recursive numerical children and their orbital basis as static metadata
without carrier-specific serialization rules.
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
from beartype.typing import Any, Mapping, Optional, Union
from jaxtyping import Shaped, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812

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
    FullDensityOfStates,
    KGrid,
    KPath,
    KPathInfo,
    OrbitalBasis,
    OrbitalProjection,
    PolarizationConfig,
    SelfEnergyConfig,
    SimulationParams,
    SlaterParams,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    TBModel,
    VolumetricData,
    WannierOperatorData,
    WorkflowContext,
)


def _pytree_classes() -> tuple[type[eqx.Module], ...]:
    """Return the complete carrier class set used by the codec."""
    classes: tuple[type[eqx.Module], ...] = (
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
        PolarizationConfig,
        SelfEnergyConfig,
        SimulationParams,
        SlaterParams,
        SOCVolumetricData,
        SpinBandStructure,
        SpinOrbitalProjection,
        TBModel,
        VolumetricData,
        WannierOperatorData,
        WorkflowContext,
    )
    return classes


def _encode_static(value: Any) -> Any:  # noqa: ANN401
    """Encode nested static Equinox metadata without losing tuple types."""
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
    """Decode tuple-preserving and nested-module static metadata."""
    decoded: Any
    if isinstance(value, dict) and "__tuple__" in value:
        decoded = tuple(_decode_static(item) for item in value["__tuple__"])
    elif isinstance(value, dict) and "__module__" in value:
        class_name: str = str(value["__module__"])
        module_class: type[eqx.Module] = _PYTREE_REGISTRY[class_name]["cls"]
        module_fields: dict[str, Any] = {
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
    """Decode current static metadata and supported legacy HDF5 aux data.

    Plan 02's pre-migration codec wrote plain JSON lists for tuple-valued
    metadata. Current files use explicit tuple tags. The three conversions
    below retain read compatibility with those pinned files without restoring
    a per-carrier write registry.
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
    """Build serialization metadata from Equinox dataclass fields."""
    module_fields: tuple[Any, ...] = fields(module_class)
    children_fields: tuple[str, ...] = tuple(
        field.name
        for field in module_fields
        if not bool(field.metadata.get("static", False))
    )
    static_fields: tuple[str, ...] = tuple(
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


@beartype
def _dataset_write_kwargs(
    data: Shaped[NDArray, "..."],
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, tuple[int, ...]]],
) -> dict[str, Any]:
    """Build ``h5py.create_dataset`` keyword arguments for one child array.

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
    chunks : Optional[Union[bool, tuple[int, ...]]]
        Chunking policy: ``True`` for auto-chunking, or an explicit
        chunk shape tuple.

    Returns
    -------
    dict[str, Any]
        Keyword arguments to pass to ``h5py.Group.create_dataset``.
        Empty dict for scalar datasets.
    """
    kwargs: dict[str, Any] = {}
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
    chunks: Optional[Union[bool, tuple[int, ...]]] = None,
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
    chunks : Optional[Union[bool, tuple[int, ...]]], optional
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
        """Write one Equinox module, recursively storing module children."""
        field_name: str

        type_name: str = type(pytree).__name__
        if type_name not in _PYTREE_REGISTRY:
            msg: str = f"Unsupported PyTree type: {type_name}"
            raise TypeError(msg)
        meta: Mapping[str, Any] = _PYTREE_REGISTRY[type_name]
        static_values: tuple[Any, ...] = tuple(
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
                ds_kwargs: dict[str, Any] = _dataset_write_kwargs(
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
    loaded : PyTree or dict[str, PyTree]
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
            if field_name in none_fields:
                children.append(None)
            elif isinstance(grp[field_name], h5py.Group):
                children.append(_load_group(grp[field_name]))
            else:
                arr: Shaped[NDArray, "..."] = grp[field_name][()]
                children.append(jnp.asarray(arr))

        constructor_fields: dict[str, Any] = dict(
            zip(meta["children_fields"], children, strict=True)
        )
        static_values: tuple[Any, ...]
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

        result: dict[str, Any] = {}
        for group_name in f:
            result[group_name] = _load_group(f[group_name])
        loaded: dict[str, Any] = result
        return loaded


__all__: list[str] = [
    "load_from_h5",
    "save_to_h5",
]
