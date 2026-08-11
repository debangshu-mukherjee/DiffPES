"""Validate provenance DAG validation and information propagation.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import pytest
from beartype.typing import Any, Tuple
from hypothesis import given
from hypothesis import strategies as st

from diffpes.certify import (
    build_provenance,
    effective_information,
    invalidated_claims,
    lineage,
    validate_provenance,
)
from diffpes.types import TransformationRecord, make_transformation_record


def _record(
    transformation_id: str,
    parents: Tuple[str, ...],
    outputs: Tuple[str, ...],
    *,
    preserves: Tuple[str, ...] = (),
    introduces: Tuple[str, ...] = (),
    destroys: Tuple[str, ...] = (),
    invalidates: Tuple[str, ...] = (),
) -> TransformationRecord:
    """PRIVATE: Build one transformation record for provenance tests.

    Parameters
    ----------
    transformation_id : str
        Stable transformation identity.
    parents : Tuple[str, ...]
        Parent artifact identities.
    outputs : Tuple[str, ...]
        Output artifact identities.
    preserves : Tuple[str, ...]
        Information keys the transformation preserves.
    introduces : Tuple[str, ...]
        Information keys the transformation introduces.
    destroys : Tuple[str, ...]
        Information keys the transformation destroys.
    invalidates : Tuple[str, ...]
        Claim identities the transformation invalidates.

    Returns
    -------
    record : TransformationRecord
        Transformation record at version 1.0.0 with the parameters
        checksum fixed to ``none``.

    Notes
    -----
    Forwards each argument to the transformation-record constructor and
    fixes the version and the parameters checksum.
    """
    record: TransformationRecord = make_transformation_record(
        transformation_id=transformation_id,
        transformation_version="1.0.0",
        parent_ids=parents,
        output_ids=outputs,
        preserves=preserves,
        introduces=introduces,
        destroys=destroys,
        invalidates_claims=invalidates,
        parameters_checksum="none",
    )
    return record


class TestValidateProvenance:
    """Verify :func:`~diffpes.certify.validate_provenance`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.validate_provenance`
    """

    def test_empty_graph_is_valid(self) -> None:
        """Verify an empty declared graph has no structural errors.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test builds and independently validates the minimal graph.
        """
        graph: Any
        graph = build_provenance(())
        assert validate_provenance(graph).valid

    @given(
        st.lists(
            st.integers(min_value=0, max_value=31), min_size=1, max_size=12
        )
    )
    def test_generated_acyclic_graphs_validate(
        self, parent_choices: Any
    ) -> None:
        """Validate generated DAGs regardless of supplied record order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        index: Any
        choice: Any
        records: Any
        parents: Any
        graph: Any
        report: Any
        records = []
        for index, choice in enumerate(parent_choices):
            parents = () if index == 0 else (f"node-{choice % index}",)
            records.append(
                _record(
                    f"org.diffpes.transform.generated.step_{index}",
                    parents,
                    (f"node-{index}",),
                    preserves=("axis",),
                    introduces=(f"stage-{index}",),
                )
            )
        graph = build_provenance(tuple(reversed(records)))
        report = validate_provenance(graph)
        assert report.valid, report.errors
        assert set(graph.topological_order) == {
            f"node-{index}" for index in range(len(records))
        }


class TestEffectiveInformation:
    """Verify :func:`~diffpes.certify.effective_information`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.effective_information`
    """

    def test_dag_is_topological_and_propagates_losses_and_claims(self) -> None:
        """Propagate inherited loss and claim invalidation to terminal results.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        amplitude: Any
        intensity: Any
        normalization: Any
        graph: Any
        report: Any
        state: Any
        amplitude = _record(
            "org.diffpes.transform.test.amplitude",
            ("bands",),
            ("amplitude",),
            preserves=("energy-axis", "absolute-scale", "phase"),
            introduces=("matrix-element-amplitude",),
        )
        intensity = _record(
            "org.diffpes.transform.test.intensity",
            ("amplitude",),
            ("intensity",),
            preserves=("energy-axis", "absolute-scale"),
            introduces=("intensity-observable",),
            destroys=("phase",),
            invalidates=("org.diffpes.claim.phase.resolved",),
        )
        normalization = _record(
            "org.diffpes.transform.test.normalize",
            ("intensity",),
            ("normalized",),
            preserves=("energy-axis", "intensity-observable"),
            introduces=("dimensionless-scale",),
            destroys=("absolute-scale",),
            invalidates=("org.diffpes.claim.intensity.absolute",),
        )
        graph = build_provenance(
            (normalization, intensity, amplitude),
            external_inputs={
                "bands": ("energy-axis", "absolute-scale", "phase")
            },
        )
        report = validate_provenance(graph)
        assert report.valid, report.errors
        assert graph.topological_order == (
            "amplitude",
            "intensity",
            "normalized",
        )
        state = effective_information(graph, "normalized")
        assert state.active_semantics == (
            "dimensionless-scale",
            "energy-axis",
            "intensity-observable",
        )
        assert set(state.destroyed_information) == {
            "absolute-scale",
            "matrix-element-amplitude",
            "phase",
        }
        assert invalidated_claims(graph, "normalized") == (
            "org.diffpes.claim.intensity.absolute",
            "org.diffpes.claim.phase.resolved",
        )
        assert lineage(graph, "normalized") == (
            "amplitude",
            "bands",
            "intensity",
        )

    def test_source_record_and_multiple_outputs_are_supported(self) -> None:
        """Allow source nodes and transformations with several outputs.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        source: Any
        graph: Any
        report: Any
        source = _record(
            "org.diffpes.transform.test.source_multi",
            (),
            ("left", "right"),
            introduces=("parsed",),
        )
        graph = build_provenance((source,))
        report = validate_provenance(graph)
        assert report.valid
        assert report.roots == ("left", "right")
        assert report.terminal_outputs == ("left", "right")
        assert effective_information(graph, "left").active_semantics == (
            "parsed",
        )


class TestBuildProvenance:
    """Verify :func:`~diffpes.certify.build_provenance`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.build_provenance`
    """

    def test_cycle_is_rejected_and_remains_inspectable_in_nonstrict_mode(
        self,
    ) -> None:
        """Detect a multi-step cycle without hanging topological analysis.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        first: Any
        second: Any
        graph: Any
        report: Any
        first = _record(
            "org.diffpes.transform.test.cycle_a",
            ("b",),
            ("a",),
        )
        second = _record(
            "org.diffpes.transform.test.cycle_b",
            ("a",),
            ("b",),
        )
        with pytest.raises(ValueError, match="cycle"):
            build_provenance((first, second))
        graph = build_provenance((first, second), strict=False)
        report = validate_provenance(graph)
        assert not report.valid
        assert any("cycle" in error for error in report.errors)

    def test_missing_parent_and_multiple_producer_are_rejected(self) -> None:
        """Require unique non-source parents and output producers.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        missing: Any
        first: Any
        second: Any
        missing = _record(
            "org.diffpes.transform.test.missing",
            ("not-declared",),
            ("result",),
        )
        with pytest.raises(ValueError, match="missing parents"):
            build_provenance((missing,))

        first = _record("org.diffpes.transform.test.one", (), ("same",))
        second = _record("org.diffpes.transform.test.two", (), ("same",))
        with pytest.raises(ValueError, match="multiple transformations"):
            build_provenance((first, second))

    def test_unused_external_input_is_rejected(self) -> None:
        """Expose declared required inputs that no transformation consumes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        source: Any
        source = _record("org.diffpes.transform.test.source", (), ("result",))
        with pytest.raises(ValueError, match="not consumed"):
            build_provenance((source,), external_inputs=("unused",))


class TestLineage:
    """Verify :func:`~diffpes.certify.lineage`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.lineage`
    """

    def test_unknown_node_inspection_is_rejected(self) -> None:
        """Avoid silently returning empty state for a misspelled artifact ID.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        source: Any
        graph: Any
        source = _record("org.diffpes.transform.test.known", (), ("known",))
        graph = build_provenance((source,))
        with pytest.raises(KeyError, match="unknown provenance node"):
            effective_information(graph, "unknown")
        with pytest.raises(KeyError, match="unknown provenance node"):
            lineage(graph, "unknown")


class TestInvalidatedClaims:
    """Verify :func:`~diffpes.certify.invalidated_claims`.

    The cases cover sorted claim invalidation at a derived result node.

    :see: :func:`~diffpes.certify.invalidated_claims`
    """

    def test_node_reports_inherited_invalidated_claims(self) -> None:
        """Return all claim identities invalidated along the node lineage.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test checks the result with explicit assertions.
        """
        record: Any
        graph: Any
        record = _record(
            "org.diffpes.transform.test.invalidate",
            (),
            ("result",),
            invalidates=("org.diffpes.claim.test.invalid",),
        )
        graph = build_provenance((record,))
        assert invalidated_claims(graph, "result") == (
            "org.diffpes.claim.test.invalid",
        )
