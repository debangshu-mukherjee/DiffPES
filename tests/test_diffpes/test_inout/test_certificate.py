"""Validate portable forward-certificate persistence.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import json
import os
from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict

from diffpes.inout import (
    attach_certificate_h5,
    certificate_identity,
    finalize_certificate,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)
from diffpes.inout.certificate import _storage_checksum
from diffpes.types import (
    CERTIFICATE_FORMAT,
    ForwardCertificate,
    make_artifact_ref,
    make_certification_claim,
    make_convention_ref,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_human_attestation_ref,
    make_information_spectrum,
    make_policy_report,
    make_sensitivity_map,
    make_transformation_record,
)


def sample_certificate(
    *,
    execution_id: str = "run-001",
    started_at_utc: str = "2026-07-21T12:00:00Z",
    model_version: str = "1.0.0",
    environment_checksum: str = (
        "sha256:1:environment:"
        "89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef"
    ),
    extensions_json: str = '{"project":"demo","unicode":"Å"}',
) -> ForwardCertificate:
    """Return one small, fully populated certificate test fixture."""
    convention: Any
    predicate: Any
    model: Any
    manifest: Any
    artifact: Any
    transformation: Any
    attestation: Any
    evidence: Any
    claim: Any
    domain: Any
    derivatives: Any
    dependencies: Any
    sensitivities: Any
    information: Any
    policy: Any
    convention = make_convention_ref(
        "org.diffpes.convention.energy.fermi_referenced_ev",
        "1.0.0",
    )
    predicate = make_domain_predicate(
        "org.diffpes.domain.photon_energy.positive",
        "photon_energy_ev > 0",
        units="eV",
    )
    model = make_forward_model_spec(
        model_id="org.diffpes.model.arpes.test",
        model_version=model_version,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="tests.forward",
        assumptions=("dipole_approximation",),
        conventions=(convention,),
        domain=(predicate,),
        differentiable_paths=("params.sigma", "params.temperature"),
    )
    manifest = make_execution_manifest(
        execution_id=execution_id,
        model_ref=f"{model.model_id}@{model.model_version}",
        schema_version="1.0.0",
        package_version="2026.06.02",
        source_checksum=(
            "sha256:1:source:"
            "0123456701234567012345670123456701234567012345670123456701234567"
        ),
        environment_checksum=environment_checksum,
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc=started_at_utc,
    )
    artifact = make_artifact_ref(
        artifact_id="bands",
        media_type="application/x-vasp-eigenval",
        byte_checksum=(
            "sha256:1:artifact-bytes:"
            "1020304010203040102030401020304010203040102030401020304010203040"
        ),
        content_checksum=(
            "sha256:1:normalized-content:"
            "2030405020304050203040502030405020304050203040502030405020304050"
        ),
        semantic_checksum=(
            "sha256:1:semantic:"
            "3040506030405060304050603040506030405060304050603040506030405060"
        ),
        locator="/private/data/EIGENVAL",
        role="initial_state",
    )
    transformation = make_transformation_record(
        transformation_id="org.diffpes.transform.amplitude.intensity",
        transformation_version="1.0.0",
        parent_ids=("amplitude",),
        output_ids=("intensity",),
        preserves=("energy_reference",),
        destroys=("overall_phase",),
        invalidates_claims=("claim.phase",),
        parameters_checksum=(
            "sha256:1:parameters:"
            "4050607040506070405060704050607040506070405060704050607040506070"
        ),
    )
    attestation = make_human_attestation_ref(
        attestation_id="attestation.reference-review",
        reviewer_ref="reviewer.example",
        scope_ids=("reference-spectrum",),
        statement="Reviewed the named evidence lineage.",
        recorded_at_utc="2026-07-24T12:00:00Z",
    )
    evidence = make_evidence_ref(
        evidence_id="reference-spectrum",
        method_id="org.diffpes.method.reference",
        source_type="analytic_reference",
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1e-8),
        lineage=make_evidence_lineage(
            implementation_refs=("reference.impl",),
            generator_refs=("reference.generator",),
            artifact_refs=("bands",),
            derivation_refs=("reference.derivation",),
            relationship_ids=(
                "independent-derivation:reference.derivation",
                "resolves-node:reference.impl",
                "resolves-node:reference.generator",
                "resolves-node:reference.derivation",
            ),
        ),
        human_attestation_refs=(attestation.attestation_id,),
    )
    claim = make_certification_claim(
        claim_id="claim.output.finite",
        subject_id=model.observable_id,
        predicate_id="output.finite",
        evidence_ids=(evidence.evidence_id,),
        measured=jnp.zeros(1),
        reference=jnp.zeros(1),
        residual=jnp.zeros(1),
        tolerance=jnp.zeros(1),
        passed=True,
        checked=True,
        in_domain=True,
        margin=0.5,
        severity_code=1,
    )
    domain = make_domain_result(
        predicate_id=predicate.predicate_id,
        measured=21.2,
        reference=20.0,
        residual=1.2,
        tolerance=20.0,
        margin=18.8,
        passed=True,
        checked=True,
        in_domain=True,
        severity_code=1,
    )
    derivatives = make_derivative_evidence(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        method="jax.linearize+jvp+vjp+central_fd",
        scales=jnp.array([0.05, 30.0]),
        jvp_probes=jnp.array([[1.0], [0.5]]),
        vjp_probes=jnp.array([[1.0, 0.5], [0.2, 0.1]]),
        reference_derivatives=jnp.array([[1.0], [0.5]]),
        derivative_residuals=jnp.zeros((2, 1)),
        singular_values=jnp.array([2.0, 0.25]),
        effective_rank=2,
        condition_estimate=8.0,
        finite=True,
        fd_correct=True,
    )
    dependencies = make_dependency_map(
        model_id=model.model_id,
        input_paths=("params.sigma", "params.temperature"),
        output_paths=("spectrum.intensity",),
        structural=jnp.array([[True, True]]),
        traced=jnp.array([[True, True]]),
    )
    sensitivities = make_sensitivity_map(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        scales=jnp.array([0.05, 30.0]),
        sensitivities=jnp.array([[1.0, 0.5]]),
        threshold=1e-12,
        active=jnp.array([[True, True]]),
    )
    information = make_information_spectrum(
        input_paths=("params.sigma", "params.temperature"),
        singular_values=jnp.array([2.0, 0.25]),
        right_singular_vectors=jnp.eye(2),
        effective_rank=2,
        condition_estimate=8.0,
        threshold=1e-10,
    )
    policy = make_policy_report(
        policy_id="org.diffpes.policy.research.v1",
        level_ids=(
            "identified",
            "validated",
            "differentiable",
            "verified",
            "benchmarked",
            "reproducible",
        ),
        required_claim_ids=(claim.claim_id,),
        claim_passed=jnp.array([True]),
        claim_checked=jnp.array([True]),
        claim_in_domain=jnp.array([True]),
        achieved=jnp.array([True, True, True, True, False, False]),
    )
    return make_forward_certificate(
        manifest=manifest,
        model=model,
        artifacts=(artifact,),
        transformations=(transformation,),
        evidence=(evidence,),
        attestations=(attestation,),
        claims=(claim,),
        domains=(domain,),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy,
        policy_id=policy.policy_id,
        certificate_checksum=(
            "sha256:1:certificate:"
            "5060708050607080506070805060708050607080506070805060708050607080"
        ),
        extensions_json=extensions_json,
    )


def _read_json(path: Path) -> Dict:
    """PRIVATE: Load one certificate JSON document from disk.

    Parameters
    ----------
    path : Path
        Path of the JSON file to read.

    Returns
    -------
    document : dict
        Parsed top-level JSON object of the file.

    Notes
    -----
    Reads the file as UTF-8 text and parses it with ``json.loads``.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _write_document(path: Path, document: Dict) -> None:
    """PRIVATE: Write one mutated certificate document back to disk.

    Parameters
    ----------
    path : Path
        Destination path of the JSON file.
    document : dict
        Certificate document, usually mutated by the calling test.

    Notes
    -----
    Recomputes ``consistency_checksum`` with the storage checksum
    helper so the mutated document stays internally consistent, then
    writes the compact key-sorted JSON with one trailing newline.
    """
    document: Any
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
        The test writes the shared complete certificate fixture and reloads it before
        comparing the permanent model identity.
        """
        path: Path = tmp_path / "certificate-class.json"
        certificate: ForwardCertificate = sample_certificate()
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
        The test compares the result with explicit numerical or structural assertions.
        """
        certificate: Any
        first: Any
        second: Any
        loaded: Any
        document: Any
        jvp_node: Any
        certificate = sample_certificate()
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
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        path = tmp_path / "certificate.json"
        path.write_bytes(b"previous contents")

        def fail_replace(source: Any, destination: Any) -> Any:
            del source, destination
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            save_certificate_json(sample_certificate(), path)

        assert path.read_bytes() == b"previous contents"
        assert list(tmp_path.iterdir()) == [path]


class TestCertificateIdentity:
    """Verify :func:`~diffpes.inout.certificate_identity`.

    The cases separate scientific identity from audit execution fields.

    :see: :func:`~diffpes.inout.certificate_identity`
    """

    def test_audit_fields_do_not_change_scientific_identity(self) -> None:
        """Keep one identity across distinct execution IDs and timestamps.

        The scientific identity must exclude both declared audit fields.

        Notes
        -----
        The test changes only the two fields classified as audit metadata.
        """
        left: ForwardCertificate = sample_certificate(
            execution_id="audit-left",
            started_at_utc="2026-07-20T00:00:00Z",
        )
        right: ForwardCertificate = sample_certificate(
            execution_id="audit-right",
            started_at_utc="2026-07-21T00:00:00Z",
        )
        assert left.manifest.execution_id != right.manifest.execution_id
        assert certificate_identity(left) == certificate_identity(right)


class TestFinalizeCertificate:
    """Verify :func:`~diffpes.inout.finalize_certificate`.

    The case replaces the compiled placeholder at the canonical I/O boundary.

    :see: :func:`~diffpes.inout.finalize_certificate`
    """

    def test_final_identity_matches_canonical_record(self) -> None:
        """Replace an arbitrary checksum with the computed scientific identity.

        The stored identity must equal a new computation from the result.

        Notes
        -----
        The test finalizes the complete shared certificate fixture once.
        """
        certificate: ForwardCertificate = sample_certificate()
        finalized: ForwardCertificate = finalize_certificate(certificate)
        assert finalized.certificate_checksum == certificate_identity(
            finalized
        )


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
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_certificate(), path)
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
        """Reject an internal identity mismatch after storage CRC recalculation.

        The reader must validate the internal identity independently.

        Notes
        -----
        The test changes only the identity field and updates the outer CRC32.
        """
        path: Path = tmp_path / "identity.json"
        save_certificate_json(sample_certificate(), path)
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
        save_certificate_json(sample_certificate(), path)
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
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_certificate(), path)
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
        The test compares the result with explicit numerical or structural assertions.
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
        save_certificate_json(sample_certificate(), source)
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
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        document: Any
        path = tmp_path / "certificate.json"
        save_certificate_json(sample_certificate(), path)
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
        The test compares the result with explicit numerical or structural assertions.
        """
        file: Any
        path: Any
        certificate: Any
        loaded: Any
        group: Any
        path = tmp_path / "spectrum.h5"
        certificate = sample_certificate()
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
        The test compares the result with explicit numerical or structural assertions.
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
            "diffpes.inout.certificate._write_h5_record",
            fail_write,
        )
        with pytest.raises(OSError, match="simulated HDF5 write failure"):
            attach_certificate_h5(path, "spectrum", sample_certificate())

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
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="one nonblank group component"):
            attach_certificate_h5(
                tmp_path / "result.h5", name, sample_certificate()
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
        The test compares the result with explicit numerical or structural assertions.
        """
        path: Any
        newer: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_certificate())
        newer = sample_certificate(execution_id="run-002")
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
        The test compares the result with explicit numerical or structural assertions.
        """
        file: Any
        path: Any
        group: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_certificate())
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
        The test compares the result with explicit numerical or structural assertions.
        """
        file: Any
        path: Any
        dataset: Any
        damaged: Any
        path = tmp_path / "spectrum.h5"
        attach_certificate_h5(path, "spectrum", sample_certificate())
        with h5py.File(path, "a") as file:
            dataset = file["_diffpes_certificates/spectrum/canonical_json"]
            damaged = dataset[()]
            damaged[0] ^= 1
            dataset[...] = damaged

        with pytest.raises(ValueError, match="valid UTF-8 JSON"):
            load_certificate_h5(path, "spectrum")
