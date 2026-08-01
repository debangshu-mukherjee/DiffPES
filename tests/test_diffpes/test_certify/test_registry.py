"""Validate deterministic immutable certification registries.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import subprocess
import sys

import pytest
from beartype.typing import Any

from diffpes.certify import (
    freeze_registry,
    get_model,
    get_transformation,
    list_handshakes,
    list_models,
    list_registered_models,
    list_transformations,
    packaged_model_card,
    register_builtin_models,
    register_handshake,
    register_model,
    register_transformation,
    registry_manifest,
    registry_snapshot,
    render_model_card,
    validate_handshake,
    validate_registry,
    validate_registry_manifest,
)
from diffpes.types import (
    make_convention_ref,
    make_forward_model_spec,
    make_registration_handshake,
    make_transformation_contract,
)


def _model_spec(name: str) -> Any:
    return make_forward_model_spec(
        model_id=f"org.diffpes.model.registry_test.{name}",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref=f"tests.registry:{name}",
        differentiable_paths=("parameters.scale",),
    )


class TestValidateRegistry:
    """Verify :func:`~diffpes.certify.validate_registry`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.validate_registry`
    """

    def test_snapshot_is_structurally_valid(self) -> None:
        """Verify a snapshot satisfies ordering and checksum invariants.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        Recomputes validation against the current process-local registry.
        """
        assert validate_registry().valid

    def test_registry_report_recomputes_structural_consistency(self) -> None:
        """Report a stable checksum and successful internal validation.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = validate_registry()
        assert report.valid, report.errors
        assert report.model_count >= 0
        assert report.transformation_count >= 0
        assert report.checksum.startswith("sha256:1:registry:")


class TestRegistrySnapshot:
    """Verify :func:`~diffpes.certify.registry_snapshot`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.registry_snapshot`
    """

    def test_models_are_sorted_and_resolved_independent_of_registration_order(
        self,
    ) -> None:
        """Expose deterministic immutable snapshots after reverse-order inserts.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        zulu: Any
        alpha: Any
        ids: Any
        snapshot: Any
        zulu = _model_spec("zulu")
        alpha = _model_spec("alpha")
        register_model(zulu, lambda value: value)
        register_model(alpha, lambda value: value)

        ids = tuple(spec.model_id for spec in list_models())
        assert ids == tuple(sorted(ids))
        assert get_model(alpha.model_id, alpha.model_version).spec is alpha
        snapshot = registry_snapshot()
        assert snapshot.models == tuple(
            sorted(
                snapshot.models,
                key=lambda item: (item.spec.model_id, item.spec.model_version),
            )
        )
        with pytest.raises(TypeError):
            snapshot.models[0] = snapshot.models[-1]


class TestRegisterModel:
    """Verify :func:`~diffpes.certify.register_model`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_model`
    """

    def test_duplicate_model_identity_is_rejected_even_for_same_spec(
        self,
    ) -> None:
        """Prevent import order from replacing an existing scientific identity.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        spec: Any
        spec = _model_spec("duplicate")
        register_model(spec, lambda value: value)
        with pytest.raises(ValueError, match="duplicate model identity"):
            register_model(spec, lambda value: value)

    @pytest.mark.parametrize(
        ("model_id", "model_version"),
        [("invalid", "1.0.0"), ("org.diffpes.model.registry_test.bad", "v1")],
    )
    def test_invalid_model_identity_is_rejected(
        self, model_id: Any, model_version: Any
    ) -> None:
        """Enforce permanent reverse-DNS IDs and semantic model versions.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        spec: Any
        spec = make_forward_model_spec(
            model_id=model_id,
            model_version=model_version,
            observable_id="org.diffpes.observable.arpes.intensity",
            implementation_ref="tests.registry:invalid",
        )
        with pytest.raises(ValueError):
            register_model(spec, lambda value: value)


class TestRegisterTransformation:
    """Verify :func:`~diffpes.certify.register_transformation`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_transformation`
    """

    def test_transformation_registry_is_sorted_and_rejects_duplicates(
        self,
    ) -> None:
        """Apply the same append-only identity rule to semantic contracts.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        zulu: Any
        alpha: Any
        identities: Any
        zulu = make_transformation_contract(
            "org.diffpes.transform.registry_test.zulu",
            "1.0.0",
            produces=("zulu-output",),
        )
        alpha = make_transformation_contract(
            "org.diffpes.transform.registry_test.alpha",
            "1.0.0",
            produces=("alpha-output",),
        )
        register_transformation(zulu)
        register_transformation(alpha)
        identities = tuple(
            contract.transformation_id for contract in list_transformations()
        )
        assert identities == tuple(sorted(identities))
        assert (
            get_transformation(alpha.transformation_id, "1.0.0").contract
            is alpha
        )
        with pytest.raises(ValueError, match="duplicate transformation"):
            register_transformation(alpha)


class TestGetModel:
    """Verify :func:`~diffpes.certify.get_model`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.get_model`
    """

    def test_unknown_registry_entries_raise_key_error(self) -> None:
        """Require an exact scientific identity and version on lookup.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(KeyError, match="unknown model"):
            get_model("org.diffpes.model.registry_test.absent", "1.0.0")
        with pytest.raises(KeyError, match="unknown transformation"):
            get_transformation(
                "org.diffpes.transform.registry_test.absent",
                "1.0.0",
            )


class TestGetTransformation:
    """Verify :func:`~diffpes.certify.get_transformation`.

    The cases cover exact lookup of a registered semantic contract.

    :see: :func:`~diffpes.certify.get_transformation`
    """

    def test_registered_contract_is_resolved_exactly(self) -> None:
        """Resolve one transformation by its permanent identity and version.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        contract: Any
        resolved: Any
        contract = make_transformation_contract(
            "org.diffpes.transform.registry_test.lookup",
            "1.0.0",
            produces=("lookup-output",),
        )
        register_transformation(contract)
        resolved = get_transformation(contract.transformation_id, "1.0.0")
        assert resolved.contract is contract


class TestListModels:
    """Verify :func:`~diffpes.certify.list_models`.

    The cases cover stable ordering without exposing executor callables.

    :see: :func:`~diffpes.certify.list_models`
    """

    def test_model_specs_are_sorted(self) -> None:
        """Return model specifications in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        models: Any
        models = list_models()
        identities: tuple[tuple[str, str], ...] = tuple(
            (model.model_id, model.model_version) for model in models
        )
        assert identities == tuple(sorted(identities))


class TestListRegisteredModels:
    """Verify :func:`~diffpes.certify.list_registered_models`.

    The cases cover stable ordering of model specifications and executors.

    :see: :func:`~diffpes.certify.list_registered_models`
    """

    def test_registered_bindings_are_sorted(self) -> None:
        """Return complete registered bindings in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        models: Any
        models = list_registered_models()
        identities: tuple[tuple[str, str], ...] = tuple(
            (model.spec.model_id, model.spec.model_version) for model in models
        )
        assert identities == tuple(sorted(identities))


class TestListTransformations:
    """Verify :func:`~diffpes.certify.list_transformations`.

    The cases cover stable ordering of transformation contracts.

    :see: :func:`~diffpes.certify.list_transformations`
    """

    def test_contracts_are_sorted(self) -> None:
        """Return transformation contracts in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        contracts: Any
        contracts = list_transformations()
        identities: tuple[tuple[str, str], ...] = tuple(
            (item.transformation_id, item.transformation_version)
            for item in contracts
        )
        assert identities == tuple(sorted(identities))


class TestFreezeRegistry:
    """Verify :func:`~diffpes.certify.freeze_registry`.

    The case isolates the process-global freeze operation in a child process.

    :see: :func:`~diffpes.certify.freeze_registry`
    """

    def test_freeze_rejects_later_registration(self) -> None:
        """Reject registry mutation after an application freezes its snapshot.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        program: str = """
from diffpes.certify import freeze_registry, register_model
from diffpes.types import make_forward_model_spec
freeze_registry()
spec = make_forward_model_spec(
    'org.diffpes.model.registry_test.frozen',
    '1.0.0',
    'org.diffpes.observable.test.result',
    'tests.registry:frozen',
)
try:
    register_model(spec, lambda value: value)
except ValueError as exc:
    assert 'frozen' in str(exc)
else:
    raise AssertionError('registration unexpectedly succeeded')
"""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


class TestRegisterHandshake:
    """Verify :func:`~diffpes.certify.register_handshake`.

    The case registers one owner without importing its scientific modules.

    :see: :func:`~diffpes.certify.register_handshake`
    """

    def test_registers_one_exact_owner(self) -> None:
        """Register one unique owner handshake in process-local state.

        The next registry snapshot must contain the same immutable record.

        Notes
        -----
        The test uses an empty requirement set and a unique owner suffix.
        """
        owner_id: str = f"registration-test-{len(list_handshakes())}"
        handshake: Any = make_registration_handshake(owner_id)
        register_handshake(handshake)
        assert handshake in list_handshakes()


class TestListHandshakes:
    """Verify :func:`~diffpes.certify.list_handshakes`.

    The case checks deterministic owner ordering after registration.

    :see: :func:`~diffpes.certify.list_handshakes`
    """

    def test_returns_owner_sorted_records(self) -> None:
        """Return all handshake declarations in sorted owner order.

        Registry insertion order must not change the returned order.

        Notes
        -----
        The test compares the owner sequence with its sorted copy.
        """
        owners: tuple[str, ...] = tuple(
            item.owner_id for item in list_handshakes()
        )
        assert owners == tuple(sorted(owners))


class TestValidateHandshake:
    """Verify :func:`~diffpes.certify.validate_handshake`.

    The case validates model, convention, and evidence references explicitly.

    :see: :func:`~diffpes.certify.validate_handshake`
    """

    def test_reports_missing_then_complete_references(self) -> None:
        """Report missing evidence and then complete the same handshake.

        The same declaration must become complete when evidence becomes available.

        Notes
        -----
        The test registers one model and supplies its external evidence later.
        """
        suffix: str = str(len(list_models()))
        convention: Any = make_convention_ref(
            f"org.diffpes.convention.registry_test.{suffix}",
            "1.0.0",
            "{}",
        )
        spec: Any = make_forward_model_spec(
            model_id=f"org.diffpes.model.registry_test.handshake{suffix}",
            model_version="1.0.0",
            observable_id="org.diffpes.observable.test.result",
            implementation_ref="tests.registry:handshake",
            conventions=(convention,),
        )
        register_model(spec, lambda value: value)
        handshake: Any = make_registration_handshake(
            owner_id=f"registration-{suffix}",
            model_refs=(f"{spec.model_id}@{spec.model_version}",),
            convention_refs=(
                f"{convention.convention_id}@{convention.version}",
            ),
            evidence_ids=("evidence-registration",),
        )
        missing: Any = validate_handshake(handshake)
        complete: Any = validate_handshake(
            handshake,
            evidence_ids=("evidence-registration",),
        )
        assert missing.missing_ids == ("evidence-registration",)
        assert bool(complete.complete)

    def test_kinematics_handshake_is_green_with_declared_evidence(
        self,
    ) -> None:
        """Verify the built-in kinematics handshake with exact evidence IDs.

        The registered transformation contracts and supplied evidence must suffice.

        Notes
        -----
        The test reads evidence IDs from the packaged handshake declaration.
        """
        register_builtin_models()
        manifest: dict[str, Any] = registry_manifest()
        declaration: dict[str, Any] = manifest["handshakes"][0]
        handshake: Any = next(
            item
            for item in list_handshakes()
            if item.owner_id == "org.diffpes.kspace"
        )
        expected_refs: tuple[str, ...] = (
            "org.diffpes.transform.kspace.fractional_cartesian@1.0.0",
            "org.diffpes.transform.kinematics.detector_angle_kpar@1.0.0",
            "org.diffpes.transform.kinematics.inner_potential@1.0.0",
            "org.diffpes.transform.polarization.lab_polarization_to_sample@1.0.0",
            "org.diffpes.transform.geometry.detector_axis_to_sample@1.0.0",
        )
        assert handshake.transformation_refs == expected_refs
        assert tuple(declaration["transformation_refs"]) == expected_refs
        photon: Any = get_transformation(
            "org.diffpes.transform.polarization.lab_polarization_to_sample",
            "1.0.0",
        ).contract
        detector_axis: Any = get_transformation(
            "org.diffpes.transform.geometry.detector_axis_to_sample",
            "1.0.0",
        ).contract
        assert "fixed_beam_across_detector_pixels" in photon.preserves
        assert "detector_orientation" not in photon.requires
        assert "detector_orientation" in detector_axis.requires
        report: Any = validate_handshake(
            handshake,
            evidence_ids=tuple(declaration["evidence_ids"]),
        )
        assert bool(report.complete), report.missing_ids

    def test_tight_binding_handshake_is_idempotent_and_green_with_declared_evidence(
        self,
    ) -> None:
        """Validate tight-binding contracts and all declared evidence.

        The tight-binding registration records transformation semantics without inventing a new
        executable model identity. Repeated built-in registration must leave
        the complete process-local registry unchanged.

        Notes
        -----
        Resolve the packaged declaration by owner instead of list position.
        Require evidence for every declared verification category. Validate exactly that
        evidence.
        """
        register_builtin_models()
        transformations_before: tuple[Any, ...] = list_transformations()
        handshakes_before: tuple[Any, ...] = list_handshakes()
        register_builtin_models()
        assert list_transformations() == transformations_before
        assert list_handshakes() == handshakes_before

        manifest: dict[str, Any] = registry_manifest()
        declaration: dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.tightb"
        )
        handshake: Any = next(
            item
            for item in list_handshakes()
            if item.owner_id == "org.diffpes.tightb"
        )
        expected_refs: tuple[str, ...] = (
            "org.diffpes.transform.tightb.bloch_basis_position@1.0.0",
            "org.diffpes.transform.tightb.eigensystem_fixed_group@1.0.0",
            "org.diffpes.transform.tightb.dos_gaussian@1.0.0",
            "org.diffpes.transform.tightb.filling_fermi_level@1.0.0",
        )
        assert handshake.model_refs == ()
        assert handshake.transformation_refs == expected_refs
        assert tuple(declaration["transformation_refs"]) == expected_refs
        assert declaration["model_refs"] == []

        evidence_ids: tuple[str, ...] = tuple(declaration["evidence_ids"])
        assert len(evidence_ids) == 17
        assert len(set(evidence_ids)) == 17
        assert all(
            item.startswith("org.diffpes.evidence.tightb.")
            for item in evidence_ids
        )
        assert (
            "org.diffpes.evidence.tightb.chinook_k_compatibility_resolved"
            in evidence_ids
        )
        assert (
            "org.diffpes.evidence.tightb.wannier90_normative_ingestion"
            in evidence_ids
        )

        reference: str
        for reference in expected_refs:
            transformation_id: str
            version: str
            transformation_id, version = reference.rsplit("@", maxsplit=1)
            contract: Any = get_transformation(
                transformation_id,
                version,
            ).contract
            assert contract.jax_pure
        bloch: Any = get_transformation(
            "org.diffpes.transform.tightb.bloch_basis_position",
            "1.0.0",
        ).contract
        eigensystem: Any = get_transformation(
            "org.diffpes.transform.tightb.eigensystem_fixed_group",
            "1.0.0",
        ).contract
        dos: Any = get_transformation(
            "org.diffpes.transform.tightb.dos_gaussian",
            "1.0.0",
        ).contract
        filling: Any = get_transformation(
            "org.diffpes.transform.tightb.filling_fermi_level",
            "1.0.0",
        ).contract
        assert "convention.bloch.basis_position_gauge" in bloch.introduces
        assert "degenerate_subspace_basis_choice" in eigensystem.destroys
        assert "delta_resolved_spectral_information" in dos.destroys
        assert "implicit_root_differential" in filling.introduces

        report: Any = validate_handshake(
            handshake,
            evidence_ids=evidence_ids,
        )
        assert bool(report.complete), report.missing_ids
        assert report.missing_ids == ()
        assert {item.owner_id for item in list_handshakes()} >= {
            "org.diffpes.kspace",
            "org.diffpes.tightb",
        }

    def test_slab_split_handshakes_are_complete_and_acyclic(self) -> None:
        """Validate the separate carrier and full-slab lifecycle records.

        The carrier registration must certify only the depth-carrier release. The slab
        registration enumerates every retained verification requirement. It must not acquire
        amplitude, intensity, or matrix-element dependencies.

        Notes
        -----
        Compare the live declarations with the packaged manifest exactly,
        then validate each handshake using only its own declared evidence.
        """
        register_builtin_models()
        manifest: dict[str, Any] = registry_manifest()
        declarations: dict[str, dict[str, Any]] = {
            item["owner_id"]: item for item in manifest["handshakes"]
        }
        handshakes: dict[str, Any] = {
            item.owner_id: item for item in list_handshakes()
        }
        carrier_owner: str = "org.diffpes.slab"
        slab_owner: str = "org.diffpes.surface"
        assert {carrier_owner, slab_owner} <= declarations.keys()
        assert {carrier_owner, slab_owner} <= handshakes.keys()

        carrier_evidence: tuple[str, ...] = (
            "org.diffpes.evidence.slab.depth_carrier_persistence",
            "org.diffpes.evidence.slab.depth_identity_jacobian",
        )
        carrier_refs: tuple[str, ...] = (
            "org.diffpes.transform.tightb.depth_carrier@1.0.0",
        )
        slab_evidence: tuple[str, ...] = (
            "org.diffpes.evidence.surface.finite_chain",
            "org.diffpes.evidence.surface.rotation_covariance",
            "org.diffpes.evidence.surface.graphene_edges",
            "org.diffpes.evidence.surface.chinook_slab",
            "org.diffpes.evidence.surface.primitive_depths",
            "org.diffpes.evidence.surface.inversion_covariance",
            "org.diffpes.evidence.surface.open_surface",
            "org.diffpes.evidence.surface.depth_handoff",
            "org.diffpes.evidence.surface.surface_projection",
            "org.diffpes.evidence.surface.exact_operator_gather",
            "org.diffpes.evidence.surface.incomplete_shell_rejection",
            "org.diffpes.evidence.surface.fixed_group_gauge",
            "org.diffpes.evidence.surface.unfolded_graph",
            "org.diffpes.evidence.surface.acyclic_lifecycle",
            "org.diffpes.evidence.surface.bulk_parameter_gradients",
            "org.diffpes.evidence.surface.lattice_depth_gradients",
            "org.diffpes.evidence.surface.probe_depth_gradients",
            "org.diffpes.evidence.surface.random_group_gauges",
            "org.diffpes.evidence.surface.chunked_memory",
            "org.diffpes.evidence.surface.compile_count",
        )
        slab_refs: tuple[str, ...] = (
            "org.diffpes.transform.tightb.slab_surface@1.0.0",
            "org.diffpes.transform.tightb.surface_projection@1.0.0",
        )
        manifest_transformation_refs: set[str] = {
            f"{item['transformation_id']}@{item['transformation_version']}"
            for item in manifest["transformations"]
        }
        assert set((*carrier_refs, *slab_refs)) <= manifest_transformation_refs
        assert tuple(declarations) == tuple(sorted(declarations))

        expected: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
            (carrier_owner, carrier_refs, carrier_evidence),
            (slab_owner, slab_refs, slab_evidence),
        )
        owner: str
        transformation_refs: tuple[str, ...]
        evidence_ids: tuple[str, ...]
        for owner, transformation_refs, evidence_ids in expected:
            declaration: dict[str, Any] = declarations[owner]
            handshake: Any = handshakes[owner]
            assert handshake.model_refs == ()
            assert handshake.convention_refs == ()
            assert handshake.transformation_refs == transformation_refs
            assert handshake.evidence_ids == evidence_ids
            assert declaration["model_refs"] == []
            assert declaration["convention_refs"] == []
            assert (
                tuple(declaration["transformation_refs"])
                == transformation_refs
            )
            assert tuple(declaration["evidence_ids"]) == evidence_ids
            report: Any = validate_handshake(
                handshake,
                evidence_ids=evidence_ids,
            )
            assert bool(report.complete), report.missing_ids
            assert report.missing_ids == ()

        assert len(slab_evidence[:14]) == 14
        assert len(set(slab_evidence[:14])) == 14
        assert len(slab_evidence[14:]) == 6
        assert len(set(slab_evidence[14:])) == 6
        slab_identifiers: tuple[str, ...] = tuple(
            identifier
            for owner in (carrier_owner, slab_owner)
            for identifier in (
                *handshakes[owner].model_refs,
                *handshakes[owner].transformation_refs,
                *handshakes[owner].convention_refs,
                *handshakes[owner].evidence_ids,
            )
        )
        assert not any(
            ".matrixel" in identifier for identifier in slab_identifiers
        )
        assert not any(
            forbidden in identifier
            for identifier in slab_identifiers
            for forbidden in ("amplitude", "matrix_element")
        )
        assert (
            "org.diffpes.evidence.surface.acyclic_lifecycle" in slab_evidence
        )

        depth_carrier: Any = get_transformation(
            "org.diffpes.transform.tightb.depth_carrier",
            "1.0.0",
        ).contract
        slab_surface: Any = get_transformation(
            "org.diffpes.transform.tightb.slab_surface",
            "1.0.0",
        ).contract
        surface_projection: Any = get_transformation(
            "org.diffpes.transform.tightb.surface_projection",
            "1.0.0",
        ).contract
        assert "depth_values_exactly" in depth_carrier.preserves
        assert "open_normal_boundary" in slab_surface.introduces
        assert "normal_translation_symmetry" in slab_surface.destroys
        assert "complete_group_unitary_gauge" in surface_projection.preserves
        assert (
            "exponential_intensity_depth_weight"
            in surface_projection.introduces
        )


class TestRegistryManifest:
    """Verify :func:`~diffpes.certify.registry_manifest`.

    The case reads the packaged manifest without process-local mutation.

    :see: :func:`~diffpes.certify.registry_manifest`
    """

    def test_manifest_has_matrix_element_handshake_and_no_retired_model(
        self,
    ) -> None:
        """Read the schema and current owner handshakes from package resources.

        The manifest omits the stale radial model and declares matrix-element ownership explicitly.

        Notes
        -----
        Compare manifest identities before validating the complete live drift.
        """
        manifest: dict[str, Any] = registry_manifest()
        assert manifest["schema_version"] == "1.0.0"
        assert manifest["models"] == []
        owners: tuple[str, ...] = tuple(
            item["owner_id"] for item in manifest["handshakes"]
        )
        assert owners == tuple(sorted(owners))
        assert "org.diffpes.matrixel" in owners
        matrix_element: dict[str, Any] = next(
            item
            for item in manifest["handshakes"]
            if item["owner_id"] == "org.diffpes.matrixel"
        )
        assert len(matrix_element["evidence_ids"]) == 41
        assert (
            "org.diffpes.evidence.matrixel.orbital_position_vacuum_momentum"
            in matrix_element["evidence_ids"]
        )
        assert (
            "org.diffpes.evidence.matrixel.hermite_acceleration_not_applicable"
            in matrix_element["evidence_ids"]
        )
        assert len(set(matrix_element["evidence_ids"])) == 41
        assert (
            "org.diffpes.evidence.matrixel.late_polarization_performance"
            in matrix_element["evidence_ids"]
        )


class TestRenderModelCard:
    """Verify :func:`~diffpes.certify.render_model_card`.

    The cases render Markdown directly from a model specification.

    :see: :func:`~diffpes.certify.render_model_card`
    """

    def test_card_contains_exact_model_identity(self) -> None:
        """Render an exact model identity and its scientific fields.

        The case uses an isolated model specification.

        Notes
        -----
        Compare the generated header and required registry fields.
        """
        spec: Any = _model_spec("render_card")
        card: str = render_model_card(spec)
        assert card.startswith("# org.diffpes.model.registry_test.render_card")
        assert "Version: `1.0.0`." in card
        assert "Observable: `org.diffpes.observable.arpes.intensity`." in card
        assert "Implementation: `tests.registry:render_card`." in card


class TestPackagedModelCard:
    """Verify :func:`~diffpes.certify.packaged_model_card`.

    The cases read generated model cards from package resources.

    :see: :func:`~diffpes.certify.packaged_model_card`
    """

    def test_missing_card_raises_file_not_found(self) -> None:
        """Reject a model identity without a packaged card.

        The case uses an identity outside the packaged manifest.

        Notes
        -----
        Confirm the resource layer reports a missing generated card.
        """
        with pytest.raises(FileNotFoundError):
            packaged_model_card(
                "org.diffpes.model.registry_test.missing",
                "1.0.0",
            )


class TestValidateRegistryManifest:
    """Verify :func:`~diffpes.certify.validate_registry_manifest`.

    The case checks every packaged entry and generated model card for drift.

    :see: :func:`~diffpes.certify.validate_registry_manifest`
    """

    def test_builtin_registry_has_no_packaged_drift(self) -> None:
        """Find no missing built-in entry or changed generated model card.

        The validator must return an empty tuple after built-in registration.

        Notes
        -----
        The test registers all built-ins before it validates the package files.
        """
        register_builtin_models()
        assert validate_registry_manifest() == ()
