r"""Persist forward-model certificates in portable storage formats.

Extended Summary
----------------
This module saves canonical certificate JSON atomically and embeds the same
exact bytes in HDF5 result files. A CRC32 checksum detects accidental
storage mismatches but does not authenticate scientific claims.

Routine Listings
----------------
:func:`attach_certificate_h5`
    Attach a certificate atomically to an HDF5 result file.
:func:`load_certificate_h5`
    Load a certificate embedded in an HDF5 result file.
:func:`load_certificate_json`
    Load a validated forward certificate from canonical JSON.
:func:`save_certificate_json`
    Save a forward certificate atomically as canonical JSON.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from jaxtyping import UInt8, jaxtyped
from numpy.typing import NDArray

from diffpes.constants import CERTIFICATE_FORMAT, CERTIFICATE_H5_GROUP
from diffpes.types import ForwardCertificate

from .certificate import _certificate_document, _json_bytes
from .certificate_decoding import _certificate_from_document, _read_document


def _atomic_write(path: Path, data: bytes) -> None:
    """PRIVATE: Write bytes through a same-directory temporary and atomic
    replace.

    Parameters
    ----------
    path : Path
        Destination file path; its parent directory must exist.
    data : bytes
        Exact bytes to publish.

    Raises
    ------
    BaseException
        If temporary creation, writing, syncing, or replacement fails;
        the handler removes the temporary file first.

    Notes
    -----
    Writes into a ``mkstemp`` file in the destination directory,
    flushes and fsyncs it, and publishes with :func:`os.replace`.  The
    same-directory temporary keeps the replace atomic on one
    filesystem, so readers never observe a partial record.
    """
    stream: Any
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary_record: Tuple[int, str] = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    descriptor: int = temporary_record[0]
    temporary_name: str = temporary_record[1]
    temporary: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@jaxtyped(typechecker=beartype)
def save_certificate_json(
    certificate: ForwardCertificate,
    path: str | Path,
) -> None:
    """Save a forward certificate atomically as canonical JSON.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate_storage.TestSaveCertificateJson`


    Implementation Logic
    --------------------
    1. **Build the certificate document**::

           document = _certificate_document(certificate)
           data = _json_bytes(document, newline=True)

       The document includes the schema and a non-security consistency check.
    2. **Replace the destination atomically**::

           _atomic_write(Path(path), data)

       A same-directory temporary prevents a partial JSON record.

    Parameters
    ----------
    certificate : ForwardCertificate
        Validated scientific-assurance record to persist.
    path : str | Path
        Destination JSON path. Its parent directory must already exist.
    """
    document: Dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    _atomic_write(Path(path), data)


@jaxtyped(typechecker=beartype)
def load_certificate_json(path: str | Path) -> ForwardCertificate:
    """Load a validated forward certificate from canonical JSON.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate_storage.TestLoadCertificateJson`


    Implementation Logic
    --------------------
    1. **Read and validate the document**::

           data = Path(path).read_bytes()
           document = _read_document(data)

       The decoder checks the schema and consistency checksum before use.
    2. **Reconstruct the carrier**::

           certificate = _certificate_from_document(document)

       The decoder restores persisted numerical leaves as JAX arrays.

    Parameters
    ----------
    path : str | Path
        Source JSON path.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed certificate with numerical leaves restored as JAX
        arrays.
    """
    data: bytes = Path(path).read_bytes()
    document: Dict[str, Any] = _read_document(data)
    certificate: ForwardCertificate = _certificate_from_document(document)
    return certificate


def _validate_h5_name(name: str) -> None:
    """PRIVATE: Reject ambiguous or path-like HDF5 certificate names.

    Parameters
    ----------
    name : str
        Requested certificate entry name.

    Raises
    ------
    ValueError
        If the name is empty, ``"."``, ``".."``, or contains a slash
        or NUL character.

    Notes
    -----
    The name must stay one plain group component, so a caller cannot
    address groups outside the certificate index.
    """
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        msg: str = "HDF5 certificate name must be one nonblank group component"
        raise ValueError(msg)


def _write_h5_record(
    path: Path,
    name: str,
    data: bytes,
    certificate: ForwardCertificate,
) -> None:
    """PRIVATE: Write one exact JSON record and its convenience index
    attributes.

    Parameters
    ----------
    path : Path
        HDF5 container to open in append mode.
    name : str
        Validated certificate entry name.
    data : bytes
        Canonical JSON bytes of the certificate document.
    certificate : ForwardCertificate
        Finalized certificate that supplies the index attributes.

    Notes
    -----
    Revalidates ``data`` through :func:`_read_document` first.  The
    entry replaces any same-named group under the certificate index
    group and stores the exact bytes as one compressed,
    checksummed ``uint8`` dataset.  Convenience attributes copy the
    format, schema version, model identity, policy, execution ID, and
    storage checksum for quick inspection.
    """
    file: Any
    document: Dict[str, Any] = _read_document(data)
    with h5py.File(path, "a") as file:
        root: h5py.Group = file.require_group(CERTIFICATE_H5_GROUP)
        if name in root:
            del root[name]
        group: h5py.Group = root.create_group(name)
        group.create_dataset(
            "canonical_json",
            data=np.frombuffer(data, dtype=np.uint8),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        group.attrs["format"] = CERTIFICATE_FORMAT
        group.attrs["schema_version"] = certificate.manifest.schema_version
        group.attrs["model_id"] = certificate.model.model_id
        group.attrs["model_version"] = certificate.model.model_version
        group.attrs["policy_id"] = certificate.policy_id
        group.attrs["execution_id"] = certificate.manifest.execution_id
        group.attrs["consistency_checksum"] = document["consistency_checksum"]
        file.flush()


@jaxtyped(typechecker=beartype)
def attach_certificate_h5(
    path: str | Path,
    name: str,
    certificate: ForwardCertificate,
) -> None:
    """Attach a certificate atomically to an HDF5 result file.

    The function updates the complete file through a same-directory temporary.
    It preserves existing numerical result groups.

    :see: :class:`~.test_certificate_storage.TestAttachCertificateH5`


    Implementation Logic
    --------------------
    1. **Encode the certificate**::

           document = _certificate_document(certificate)
           data = _json_bytes(document, newline=True)

       The HDF5 record stores the same canonical bytes as JSON persistence.
    2. **Copy the current container**::

           shutil.copy2(destination, temporary)

       An existing result file remains intact while the copy changes.
    3. **Write and replace the container**::

           _write_h5_record(temporary, name, data, certificate)
           os.replace(temporary, destination)
           temporary.unlink(missing_ok=True)

       Replacement publishes the complete file. Failure removes the temporary.

    Parameters
    ----------
    path : str | Path
        Existing HDF5 result path, or a path for a new HDF5 container.
    name : str
        Name of one result under the certificate index group.
    certificate : ForwardCertificate
        Certificate associated with the named result.

    Raises
    ------
    BaseException
        If copying, writing, or replacing the HDF5 file fails.
    """
    _validate_h5_name(name)
    destination: Path = Path(path)
    destination.parent.mkdir(parents=False, exist_ok=True)
    document: Dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    temporary_record: Tuple[int, str] = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    descriptor: int = temporary_record[0]
    temporary_name: str = temporary_record[1]
    os.close(descriptor)
    temporary: Path = Path(temporary_name)
    try:
        if destination.exists():
            shutil.copy2(destination, temporary)
        _write_h5_record(temporary, name, data, certificate)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@jaxtyped(typechecker=beartype)
def load_certificate_h5(
    path: str | Path,
    name: str,
) -> ForwardCertificate:
    """Load a certificate embedded in an HDF5 result file.

    The persistence operation retains the complete scientific-assurance record
    and its JAX array leaves. Consistency checks detect accidental storage
    corruption.

    :see: :class:`~.test_certificate_storage.TestLoadCertificateH5`


    Implementation Logic
    --------------------
    1. **Resolve the stored record**::

           root = file[CERTIFICATE_H5_GROUP]
           group = root[name]

       Missing groups or names raise ``KeyError`` before decoding.
    2. **Decode the canonical bytes**::

           data = stored.tobytes()
           document = _read_document(data)
           certificate = _certificate_from_document(document)

       The decoder validates the persisted schema and consistency check.
    3. **Validate the convenience index**::

           msg: str = f"HDF5 certificate index mismatch for {key!r}"

       Every HDF5 attribute must agree with the canonical JSON record.

    Parameters
    ----------
    path : str | Path
        HDF5 result path.
    name : str
        Certificate name supplied to :func:`attach_certificate_h5`.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed and validated certificate.

    Raises
    ------
    KeyError
        If the certificate group or named record is absent.
    ValueError
        If the exact JSON bytes or HDF5 convenience index are inconsistent.
    """
    file: Any
    key: Any
    expected: Any
    _validate_h5_name(name)
    source: Path = Path(path)
    with h5py.File(source, "r") as file:
        if CERTIFICATE_H5_GROUP not in file:
            msg: str = f"No certificates found in {source}"
            raise KeyError(msg)
        root: h5py.Group = file[CERTIFICATE_H5_GROUP]
        if name not in root:
            msg: str = f"Certificate '{name}' not found in {source}"
            raise KeyError(msg)
        group: h5py.Group = root[name]
        if "canonical_json" not in group:
            msg: str = "HDF5 certificate record has no canonical_json dataset"
            raise ValueError(msg)
        stored: UInt8[NDArray, " n_byte"] = np.asarray(
            group["canonical_json"][()]
        )
        if stored.dtype != np.dtype(np.uint8) or stored.ndim != 1:
            msg: str = (
                "HDF5 canonical_json dataset must be one-dimensional uint8"
            )
            raise ValueError(msg)
        data: bytes = stored.tobytes()
        document: Dict[str, Any] = _read_document(data)
        certificate: ForwardCertificate = _certificate_from_document(document)
        expected_attrs: Dict[str, str] = {
            "format": CERTIFICATE_FORMAT,
            "schema_version": certificate.manifest.schema_version,
            "model_id": certificate.model.model_id,
            "model_version": certificate.model.model_version,
            "policy_id": certificate.policy_id,
            "execution_id": certificate.manifest.execution_id,
            "consistency_checksum": document["consistency_checksum"],
        }
        for key, expected in expected_attrs.items():
            actual: Any = group.attrs.get(key)
            if actual is None or str(actual) != expected:
                msg: str = f"HDF5 certificate index mismatch for {key!r}"
                raise ValueError(msg)
    return certificate


__all__: list[str] = [
    "attach_certificate_h5",
    "load_certificate_h5",
    "load_certificate_json",
    "save_certificate_json",
]
