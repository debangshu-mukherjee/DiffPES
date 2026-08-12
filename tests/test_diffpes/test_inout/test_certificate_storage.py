"""Validate portable forward-certificate persistence.

The tests cover deterministic JSON and HDF5 storage, validation, and
atomic replacement in the supported certification regime.
"""

import json
import os
from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict

from diffpes.constants import CERTIFICATE_FORMAT
from diffpes.inout import (
    attach_certificate_h5,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)
from diffpes.inout.certificate import _storage_checksum
from diffpes.types import ForwardCertificate
from tests._factories import sample_forward_certificate


def _read_json(path: Path) -> Dict[str, Any]:
    """PRIVATE: Load one certificate JSON document from disk.

    Parameters
    ----------
    path : Path
        Path of the JSON file to read.

    Returns
    -------
    document : Dict[str, Any]
        Parsed top-level JSON object of the file.

    Notes
    -----
    Reads the file as UTF-8 text and parses it with ``json.loads``.
    """
    document: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _write_document(path: Path, document: Dict[str, Any]) -> None:
    """PRIVATE: Write one mutated certificate document back to disk.

    Parameters
    ----------
    path : Path
        Destination path of the JSON file.
    document : Dict[str, Any]
        Certificate document, usually mutated by the calling test.

    Notes
    -----
    Recomputes ``consistency_checksum`` with the storage checksum
    helper so the mutated document stays internally consistent, then
    writes the compact key-sorted JSON with one trailing newline.
    """
    document["consistency_checksum"] = _storage_checksum(document)
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class TestSaveCertificateJson:
    """Verify :func:`~diffpes.inout.save_certificate_json`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.inout.save_certificate_json`
    """

    def test_json_preserves_model_identity(self, tmp_path: Path) -> None:
        """Verify JSON round-trip preserves exact scientific model identity.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test writes the shared complete certificate fixture and reloads it
        before comparing the permanent model identity.
        """
        path: Path = tmp_path / "certificate-class.json"
        certificate: ForwardCertificate = sample_forward_certificate()
        save_certificate_json(certificate, path)
        restored: ForwardCertificate = load_certificate_json(path)
        assert restored.model.model_id == certificate.model.model_id

    def test_json_round_trip_is_byte_stable_and_lossless(
        self, tmp_path: Any
    ) -> None:
        """Verify json round trip is byte stable and lossless.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        certificate: Any
        first: Any
        second: Any
        loaded: Any
        document: Any
        jvp_node: Any
        certificate = sample_forward_certificate()
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"

        save_certificate_json(certificate, first)
        loaded = load_certificate_json(first)
        save_certificate_json(loaded, second)

        assert first.read_bytes() == second.read_bytes()
        assert loaded.model.model_id == certificate.model.model_id
        assert json.loads(loaded.extensions_json) == json.loads(
            certificate.extensions_json
        )
        chex.assert_trees_all_equal(
            loaded.derivatives.jvp_probes,
            certificate.derivatives.jvp_probes,
        )
        document = _read_json(first)
        assert document["format"] == CERTIFICATE_FORMAT
        assert document["consistency_checksum"] == (
            "crc32:certificate-json-v1:1c1a6c25"
        )
        jvp_node = document["certificate"]["fields"]["derivatives"]["fields"][
            "jvp_probes"
        ]
        assert jvp_node["dtype"] == "<f8"
        assert jvp_node["shape"] == [2, 1]
        assert jvp_node["encoding"] == "base64"

    def test_json_write_failure_keeps_previous_file(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Verify json write failure keeps previous file.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        path: Any
        path = tmp_path / "certificate.json"
        path.write_bytes(b"previous contents")

        def fail_replace(source: Any, destination: Any) -> Any:
            del source, destination
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            save_certificate_json(sample_forward_certificate(), path)

        assert path.read_bytes() == b"previous contents"
        assert list(tmp_path.iterdir()) == [path]


class TestLoadCertificateJson:
    """Verify :func:`~diffpes.inout.load_certificate_json`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.inout.load_certificate_json`
    """

    def test_json_corruption_fails_consistency_check(
        self, tmp_path: Any
    ) -> None:
        """Verify json corruption fails consistency check.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_forward_certificate(), path)
        document = _read_json(path)
        document["certificate"]["fields"]["model"]["fields"]["model_id"] = (
            "org.diffpes.model.changed"
        )
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ValueError, match="checksum mismatch"):
            load_certificate_json(path)

    def test_internal_identity_mismatch_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Reject an identity mismatch after CRC recalculation.

        The reader must validate the internal identity independently.

        Notes
        -----
        The test changes only the identity field and updates the outer CRC32.
        """
        path: Path = tmp_path / "identity.json"
        save_certificate_json(sample_forward_certificate(), path)
        document: Dict[str, Any] = _read_json(path)
        document["certificate"]["fields"]["certificate_checksum"] = (
            "sha256:1:certificate:"
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        _write_document(path, document)
        with pytest.raises(ValueError, match="canonical identity mismatch"):
            load_certificate_json(path)

    def test_legacy_crc32_scientific_identity_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Reject legacy CRC32 even when the outer transport CRC is valid.

        Replace the scientific identity while preserving transport validity.

        Notes
        -----
        Load the modified document and require an identity-mismatch error.
        """
        path: Path = tmp_path / "legacy-identity.json"
        save_certificate_json(sample_forward_certificate(), path)
        document: Dict[str, Any] = _read_json(path)
        document["certificate"]["fields"]["certificate_checksum"] = (
            "crc32:canonical-1:certificate:00000000"
        )
        _write_document(path, document)

        with pytest.raises(ValueError, match="canonical identity mismatch"):
            load_certificate_json(path)

    def test_unknown_schema_major_is_rejected_before_interpretation(
        self,
        tmp_path: Any,
    ) -> None:
        """Verify rejection of an unknown schema major before interpretation.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_forward_certificate(), path)
        document = _read_json(path)
        document["schema_version"] = "2.0.0"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ValueError, match="schema major 2"):
            load_certificate_json(path)

    def test_unknown_minor_extensions_survive_round_trip(
        self, tmp_path: Any
    ) -> None:
        """Verify unknown minor extensions survive round trip.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        source: Any
        restored: Any
        document: Any
        manifest_fields: Any
        loaded: Any
        extensions: Any
        reloaded: Any
        source = tmp_path / "future.json"
        restored = tmp_path / "restored.json"
        save_certificate_json(sample_forward_certificate(), source)
        document = _read_json(source)
        document["schema_version"] = "1.1.0"
        manifest_fields = document["certificate"]["fields"]["manifest"][
            "fields"
        ]
        manifest_fields["schema_version"] = "1.1.0"
        document["extensions"]["future_quantity"] = {"units": "eV", "value": 2}
        document["future_top_level"] = {"meaning": "retained"}
        document["certificate"]["fields"]["model"]["fields"][
            "future_model_field"
        ] = {"kind": "tuple", "items": ["alpha"]}
        _write_document(source, document)

        loaded = load_certificate_json(source)
        extensions = json.loads(loaded.extensions_json)
        save_certificate_json(loaded, restored)
        reloaded = load_certificate_json(restored)

        assert extensions["future_quantity"]["units"] == "eV"
        assert (
            extensions["org.diffpes.persistence.unknown_document_fields"][
                "future_top_level"
            ]["meaning"]
            == "retained"
        )
        assert (
            "certificate.model"
            in extensions["org.diffpes.persistence.unknown_module_fields"]
        )
        assert json.loads(reloaded.extensions_json) == extensions

    def test_current_minor_rejects_unknown_structural_field(
        self, tmp_path: Any
    ) -> None:
        """Verify current minor rejects unknown structural field.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_forward_certificate(), path)
        document = _read_json(path)
        document["unexpected"] = True
        _write_document(path, document)

        with pytest.raises(ValueError, match="unknown current-schema"):
            load_certificate_json(path)


class TestAttachCertificateH5:
    """Verify :func:`~diffpes.inout.attach_certificate_h5`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.inout.attach_certificate_h5`
    """

    def test_hdf5_attach_preserves_results_and_round_trips(
        self, tmp_path: Any
    ) -> None:
        """Verify hdf5 attach preserves results and round trips.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        file: Any
        path: Any
        certificate: Any
        loaded: Any
        group: Any
        path = tmp_path / "spectrum.h5"
        certificate = sample_forward_certificate()
        with h5py.File(path, "w") as file:
            file.create_dataset(
                "spectrum/intensity", data=jnp.arange(6).reshape(2, 3)
            )

        attach_certificate_h5(path, "spectrum", certificate)
        loaded = load_certificate_h5(path, "spectrum")

        assert loaded.model.model_id == certificate.model.model_id
        chex.assert_trees_all_equal(
            loaded.information.singular_values,
            certificate.information.singular_values,
        )
        with h5py.File(path, "r") as file:
            chex.assert_trees_all_equal(
                file["spectrum/intensity"][()],
                jnp.arange(6).reshape(2, 3),
            )
            group = file["_diffpes_certificates/spectrum"]
            assert group.attrs["model_id"] == certificate.model.model_id

    def test_hdf5_write_failure_keeps_previous_file(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Verify hdf5 write failure keeps previous file.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        file: Any
        path: Any
        previous: Any
        path = tmp_path / "spectrum.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("sentinel", data=jnp.array([1.0, 2.0]))
        previous = path.read_bytes()

        def fail_write(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("simulated HDF5 write failure")

        monkeypatch.setattr(
            "diffpes.inout.certificate_storage._write_h5_record",
            fail_write,
        )
        with pytest.raises(OSError, match="simulated HDF5 write failure"):
            attach_certificate_h5(
                path, "spectrum", sample_forward_certificate()
            )

        assert path.read_bytes() == previous
        assert list(tmp_path.iterdir()) == [path]

    @pytest.mark.parametrize(
        "name", ["", ".", "..", "nested/name", "bad\x00name"]
    )
    def test_hdf5_name_rejects_path_components(
        self, tmp_path: Any, name: Any
    ) -> None:
        """Verify hdf5 name rejects path components.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        with pytest.raises(ValueError, match="one nonblank group component"):
            attach_certificate_h5(
                tmp_path / "result.h5", name, sample_forward_certificate()
            )


class TestLoadCertificateH5:
    """Verify :func:`~diffpes.inout.load_certificate_h5`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.inout.load_certificate_h5`
    """

    def test_hdf5_replace_and_missing_name(self, tmp_path: Any) -> None:
        """Verify hdf5 replace and missing name.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        path: Any
        newer: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_forward_certificate())
        newer = sample_forward_certificate(execution_id="run-002")
        attach_certificate_h5(path, "spectrum", newer)

        assert (
            load_certificate_h5(path, "spectrum").manifest.execution_id
            == "run-002"
        )
        with pytest.raises(KeyError, match="missing"):
            load_certificate_h5(path, "missing")

    def test_hdf5_corruption_and_index_disagreement_fail_closed(
        self,
        tmp_path: Any,
    ) -> None:
        """Verify hdf5 corruption and index disagreement fail closed.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        file: Any
        path: Any
        group: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_forward_certificate())
        with h5py.File(path, "a") as file:
            group = file["_diffpes_certificates/spectrum"]
            group.attrs["model_version"] = "99.0.0"

        with pytest.raises(ValueError, match="index mismatch"):
            load_certificate_h5(path, "spectrum")

    def test_hdf5_corrupt_authoritative_json_fails_closed(
        self, tmp_path: Any
    ) -> None:
        """Verify hdf5 corrupt authoritative json fails closed.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        file: Any
        path: Any
        dataset: Any
        damaged: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_forward_certificate())
        with h5py.File(path, "a") as file:
            dataset = file["_diffpes_certificates/spectrum/canonical_json"]
            damaged = dataset[()]
            damaged[0] ^= 1
            dataset[...] = damaged

        with pytest.raises(ValueError, match="valid UTF-8 JSON"):
            load_certificate_h5(path, "spectrum")
