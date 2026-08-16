"""Write and record artifacts from executable experiments.

Extended Summary
----------------
This module confines every artifact path below one experiment output root. It
writes JSON, compressed arrays, images, figures, and HDF5 carriers. Each
writer returns an immutable manifest record for the result payload.

Routine Listings
----------------
:func:`artifact_path`
    Resolve one safe artifact path below an experiment output root.
:func:`artifact_record_as_dict`
    Convert one artifact record to a JSON-ready dictionary.
:func:`log_message`
    Write one human-readable experiment message to standard error.
:func:`record_artifact`
    Record an existing artifact below an experiment output root.
:func:`save_array_artifact`
    Save compressed NumPy arrays and return a manifest record.
:func:`save_carrier_artifact`
    Save one diffpes carrier in HDF5 and return a manifest record.
:func:`save_figure_artifact`
    Save and close one Matplotlib figure and return a manifest record.
:func:`save_image_artifact`
    Save one image array and return a manifest record.
:func:`save_json_artifact`
    Save JSON data and return a manifest record.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Mapping
from jaxtyping import jaxtyped

from diffpes.constants import AUTOMATON_PREVIEW_MAX_BYTES
from diffpes.inout import save_to_h5
from diffpes.types import (
    ArtifactRecord,
    AutomatonContext,
    make_artifact_record,
)

from .results import json_ready


def _named_path(name: str, suffix: str) -> str:
    """PRIVATE: Add one required suffix when an artifact name has none.

    Parameters
    ----------
    name : str
        Requested relative artifact name.
    suffix : str
        Required filename suffix, including the leading period.

    Returns
    -------
    resolved_name : str
        Requested name or the requested name with ``suffix`` appended.

    Notes
    -----
    Preserves an explicit suffix because callers can choose a more specific
    extension. Writers still use their own documented file format.
    """
    requested_path: Path = Path(name)
    resolved_name: str = name if requested_path.suffix else f"{name}{suffix}"
    return resolved_name


def _relative_path(ctx: AutomatonContext, path: Path) -> Path:
    """PRIVATE: Validate an existing artifact path below the output root.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the allowed artifact root.
    path : Path
        Candidate artifact path in absolute or root-relative form.

    Returns
    -------
    relative : Path
        Safe path relative to the context output root.

    Raises
    ------
    ValueError
        If the candidate path escapes the output root or does not exist.

    Notes
    -----
    Resolves both paths before comparison. This operation rejects ``..`` and
    symlink escapes before it creates a manifest record.
    """
    exc: ValueError
    root: Path = Path(ctx.outdir).resolve()
    candidate: Path = path if path.is_absolute() else root / path
    resolved: Path = candidate.resolve()
    try:
        relative: Path = resolved.relative_to(root)
    except ValueError as exc:
        message: str = "artifact path escapes the output directory"
        raise ValueError(message) from exc
    if not resolved.is_file():
        message = "artifact path must name an existing file"
        raise ValueError(message)
    return relative


def _scaled_image(image: Any, intensity_scale: str) -> Any:
    """PRIVATE: Scale image intensity with one declared display scale.

    Parameters
    ----------
    image : Any
        Array-like image intensity data.
    intensity_scale : str
        ``"linear"``, ``"sqrt"``, or ``"log"`` display scale.

    Returns
    -------
    scaled : Any
        NumPy image data after the requested intensity transformation.

    Raises
    ------
    ValueError
        Raise for an unsupported scale.

    Notes
    -----
    Uses a safe positive maximum for logarithmic scaling. The logarithm uses
    a fixed gain of 25.0 to preserve the executable display contract.
    """
    if intensity_scale not in {"linear", "sqrt", "log"}:
        message: str = "intensity_scale must be linear, sqrt, or log"
        raise ValueError(message)
    array: Any = np.asarray(image)
    if intensity_scale == "linear":
        scaled: Any = array
    elif intensity_scale == "sqrt":
        scaled = np.sqrt(np.clip(array, a_min=0.0, a_max=None))
    else:
        maximum: float = float(np.nanmax(array)) if array.size else 0.0
        safe_maximum: float = maximum if maximum > 0.0 else 1.0
        scaled = np.log1p(
            25.0 * np.clip(array, a_min=0.0, a_max=None) / safe_maximum
        )
    return scaled


@jaxtyped(typechecker=beartype)
def artifact_path(ctx: AutomatonContext, name: str) -> Path:
    """Resolve one safe artifact path below an experiment output root.

    The function rejects absolute paths and path traversal. It creates missing
    parent directories below the allowed output root.

    :see: :class:`~.test_artifacts.TestArtifactPath`

    Implementation Logic
    --------------------
    1. **Resolve the candidate path**::

           resolved = (root / requested).resolve()

       The resolution exposes parent and symlink escapes before file writing.

    2. **Create parent directories**::

           resolved.parent.mkdir(parents=True, exist_ok=True)

       The operation supports nested artifact names without external setup.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Relative artifact filename or nested relative path.

    Returns
    -------
    path : Path
        Safe writable path below the context output root.

    Raises
    ------
    ValueError
        If the requested path is empty, absolute, or escapes the output root.
    """
    requested: Path = Path(name)
    if not name or requested.is_absolute():
        message: str = "artifact name must be a nonempty relative path"
        raise ValueError(message)
    exc: ValueError
    root: Path = Path(ctx.outdir).resolve()
    candidate: Path = root / requested
    resolved: Path = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        message = "artifact path escapes the output directory"
        raise ValueError(message) from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    path: Path = resolved
    return path


@jaxtyped(typechecker=beartype)
def record_artifact(  # noqa: DOC502 -- helper raises preserve containment logic.
    ctx: AutomatonContext,
    path: str | Path,
    *,
    role: str,
    mime: str,
    preview: bool,
) -> ArtifactRecord:
    """Record an existing artifact below an experiment output root.

    The function stores a relative POSIX path. It embeds a base64 preview only
    for files no larger than the configured preview threshold.

    :see: :class:`~.test_artifacts.TestRecordArtifact`

    Implementation Logic
    --------------------
    1. **Validate containment**::

           relative = _relative_path(ctx, Path(path))

       The check prevents result manifests from referring outside the run root.

    2. **Build an optional preview**::

           preview_b64 = base64.b64encode(file_bytes).decode("ascii")

       The size ceiling keeps JSON results compact and deterministic.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    path : str | Path
        Existing artifact path under the output root.
    role : str
        Domain role declared by the executable experiment.
    mime : str
        Internet media type for the artifact.
    preview : bool
        Whether a small file can embed a base64 preview.

    Returns
    -------
    record : ArtifactRecord
        Immutable relative artifact manifest record.

    Raises
    ------
    ValueError
        If the path escapes the output root or does not name a file.
    """
    candidate: Path = Path(path)
    relative: Path = _relative_path(ctx, candidate)
    root: Path = Path(ctx.outdir).resolve()
    resolved: Path = root / relative
    preview_b64: str = ""
    if preview and resolved.stat().st_size <= AUTOMATON_PREVIEW_MAX_BYTES:
        content: bytes = resolved.read_bytes()
        preview_b64 = base64.b64encode(content).decode("ascii")
    record: ArtifactRecord = make_artifact_record(
        role=role,
        mime=mime,
        path=relative.as_posix(),
        preview_b64=preview_b64,
    )
    return record


@jaxtyped(typechecker=beartype)
def save_json_artifact(
    ctx: AutomatonContext,
    name: str,
    data: Any,
    *,
    role: str = "metrics",
) -> ArtifactRecord:
    """Save JSON data and return a manifest record.

    The writer sanitizes arrays, paths, and non-finite values before it emits
    sorted JSON. It always uses the ``application/json`` media type.

    :see: :class:`~.test_artifacts.TestSaveJsonArtifact`

    Notes
    -----
    Uses ``allow_nan=False`` after ``json_ready`` converts non-finite values to
    null. The result preserves reproducible object key ordering.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Requested JSON artifact name.
    data : Any
        JSON-like data, arrays, paths, or scalar values.
    role : str, optional
        Declared artifact role. Default is ``"metrics"``.

    Returns
    -------
    record : ArtifactRecord
        Manifest record for the written JSON artifact.
    """
    output_name: str = _named_path(name, ".json")
    output_path: Path = artifact_path(ctx, output_name)
    ready_data: Any = json_ready(data)
    output_path.write_text(
        json.dumps(ready_data, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    record: ArtifactRecord = record_artifact(
        ctx,
        output_path,
        role=role,
        mime="application/json",
        preview=True,
    )
    return record


@jaxtyped(typechecker=beartype)
def save_array_artifact(
    ctx: AutomatonContext,
    name: str,
    data: Any,
    *,
    role: str = "array",
) -> ArtifactRecord:
    """Save compressed NumPy arrays and return a manifest record.

    Mappings become named arrays in an ``.npz`` archive. Other values become
    one array named ``"array"`` in the archive.

    :see: :class:`~.test_artifacts.TestSaveArrayArtifact`

    Notes
    -----
    Uses NumPy compressed archive output. The writer does not pickle object
    data, so callers should provide numerical arrays or plain scalars.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Requested array artifact name.
    data : Any
        Mapping of named arrays or one array-like value.
    role : str, optional
        Declared artifact role. Default is ``"array"``.

    Returns
    -------
    record : ArtifactRecord
        Manifest record for the written compressed archive.
    """
    output_name: str = _named_path(name, ".npz")
    output_path: Path = artifact_path(ctx, output_name)
    if isinstance(data, Mapping):
        named_arrays: Dict[str, Any] = {
            str(key): np.asarray(value) for key, value in data.items()
        }
        np.savez_compressed(output_path, **named_arrays)
    else:
        np.savez_compressed(output_path, array=np.asarray(data))
    record: ArtifactRecord = record_artifact(
        ctx,
        output_path,
        role=role,
        mime="application/npz",
        preview=True,
    )
    return record


@jaxtyped(typechecker=beartype)
def save_image_artifact(  # noqa: DOC502 -- helper raises validate the scale.
    ctx: AutomatonContext,
    name: str,
    image: Any,
    *,
    role: str = "image",
    cmap: str = "magma",
    intensity_scale: str = "linear",
    preview: bool = True,
) -> ArtifactRecord:
    """Save one image array and return a manifest record.

    The writer renders linear, square-root, or logarithmic intensity data to
    PNG. Its logarithmic display transform uses the fixed gain in the contract.

    :see: :class:`~.test_artifacts.TestSaveImageArtifact`

    Notes
    -----
    The image writer serves visualization only. It does not replace a raw
    array artifact when a consumer needs numerical source data.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Requested image artifact name.
    image : Any
        Array-like intensity data.
    role : str, optional
        Declared artifact role. Default is ``"image"``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    intensity_scale : str, optional
        ``"linear"``, ``"sqrt"``, or ``"log"``. Default is ``"linear"``.
    preview : bool, optional
        Whether a small image embeds a base64 preview. Default is ``True``.

    Returns
    -------
    record : ArtifactRecord
        Manifest record for the written PNG image.

    Raises
    ------
    ValueError
        Raise an error for an unsupported ``intensity_scale`` value.
    """
    output_name: str = _named_path(name, ".png")
    output_path: Path = artifact_path(ctx, output_name)
    scaled: Any = _scaled_image(image, intensity_scale)
    plt.imsave(output_path, scaled, cmap=cmap)
    record: ArtifactRecord = record_artifact(
        ctx,
        output_path,
        role=role,
        mime="image/png",
        preview=preview,
    )
    return record


@jaxtyped(typechecker=beartype)
def save_figure_artifact(
    ctx: AutomatonContext,
    name: str,
    figure: Any,
    *,
    role: str = "figure",
    preview: bool = True,
) -> ArtifactRecord:
    """Save and close one Matplotlib figure and return a manifest record.

    The writer uses a tight bounding box and always closes the supplied figure.
    Closing releases figure resources during repeated executable smoke runs.

    :see: :class:`~.test_artifacts.TestSaveFigureArtifact`

    Notes
    -----
    The caller controls the filename suffix. PNG defaults apply when the name
    has no suffix, and the manifest media type follows the final suffix.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Requested figure artifact name.
    figure : Any
        Matplotlib figure object with a ``savefig`` method.
    role : str, optional
        Declared artifact role. Default is ``"figure"``.
    preview : bool, optional
        Whether a small figure embeds a base64 preview. Default is ``True``.

    Returns
    -------
    record : ArtifactRecord
        Manifest record for the written figure.
    """
    output_name: str = _named_path(name, ".png")
    output_path: Path = artifact_path(ctx, output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    suffix: str = output_path.suffix.lower()
    mime: str = "image/svg+xml" if suffix == ".svg" else "image/png"
    record: ArtifactRecord = record_artifact(
        ctx,
        output_path,
        role=role,
        mime=mime,
        preview=preview,
    )
    return record


@jaxtyped(typechecker=beartype)
def save_carrier_artifact(
    ctx: AutomatonContext,
    name: str,
    carrier: Any,
    *,
    role: str,
) -> ArtifactRecord:
    """Save one diffpes carrier in HDF5 and return a manifest record.

    The writer delegates HDF5 persistence to the public DiffPES I/O surface.
    It stores the supplied carrier under the stable HDF5 name ``"carrier"``.

    :see: :class:`~.test_artifacts.TestSaveCarrierArtifact`

    Notes
    -----
    Carrier classes own their serialization registrations. This wrapper only
    controls the executable artifact path and manifest record.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that defines the artifact output root.
    name : str
        Requested HDF5 artifact name.
    carrier : Any
        DiffPES carrier supported by ``save_to_h5``.
    role : str
        Declared artifact role.

    Returns
    -------
    record : ArtifactRecord
        Manifest record for the written HDF5 artifact.
    """
    output_name: str = _named_path(name, ".h5")
    output_path: Path = artifact_path(ctx, output_name)
    save_to_h5(output_path, carrier=carrier)
    record: ArtifactRecord = record_artifact(
        ctx,
        output_path,
        role=role,
        mime="application/x-hdf5",
        preview=False,
    )
    return record


@jaxtyped(typechecker=beartype)
def log_message(ctx: AutomatonContext, message: str) -> None:
    """Write one human-readable experiment message to standard error.

    The function suppresses the message when the context requests JSON-only
    output. It never writes host-side status text to standard output.

    :see: :class:`~.test_artifacts.TestLogMessage`

    Notes
    -----
    Prefixes every message with the experiment identifier. The prefix helps
    agents distinguish concurrent experiment output streams.

    Parameters
    ----------
    ctx : AutomatonContext
        Runtime context that supplies the experiment identifier and mode.
    message : str
        Human-readable status text.
    """
    if not ctx.json_mode:
        print(f"[{ctx.experiment}] {message}", file=sys.stderr, flush=True)


@jaxtyped(typechecker=beartype)
def artifact_record_as_dict(record: ArtifactRecord) -> Dict[str, Any]:
    """Convert one artifact record to a JSON-ready dictionary.

    The returned dictionary preserves the manifest field names exactly. It
    includes an empty preview string when no preview exists.

    :see: :class:`~.test_artifacts.TestArtifactRecordAsDict`

    Notes
    -----
    Uses only static carrier fields. No file-system access occurs during this
    conversion.

    Parameters
    ----------
    record : ArtifactRecord
        Immutable artifact manifest record.

    Returns
    -------
    payload : Dict[str, Any]
        JSON-ready artifact dictionary.
    """
    payload: Dict[str, Any] = {
        "role": record.role,
        "mime": record.mime,
        "path": record.path,
        "preview_b64": record.preview_b64,
    }
    return payload


__all__: list[str] = [
    "artifact_path",
    "artifact_record_as_dict",
    "log_message",
    "record_artifact",
    "save_array_artifact",
    "save_carrier_artifact",
    "save_figure_artifact",
    "save_image_artifact",
    "save_json_artifact",
]
