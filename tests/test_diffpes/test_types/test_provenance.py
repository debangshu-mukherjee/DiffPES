"""Validate provenance and information-flow carriers and factories.

The cases cover immutable semantic states, graph-boundary invariants, and
validity consistency without invoking the higher-level propagation engine.
"""

import pytest
from beartype.typing import Tuple

from diffpes.types import (
    InformationState,
    ProvenanceGraph,
    ProvenanceReport,
    make_information_state,
    make_provenance_graph,
    make_provenance_report,
)


class TestInformationState:
    """Validate :class:`~diffpes.types.InformationState` storage.

    The carrier must keep available semantics separate from destroyed
    information and invalidated claims.

    :see: :class:`~diffpes.types.InformationState`
    """

    def test_separates_semantic_categories(self) -> None:
        """Preserve active, destroyed, and invalidated semantic categories.

        The check compares each immutable category against independently
        supplied labels for one artifact node.

        Notes
        -----
        The test constructs the state through its public factory and reads every static
        information-flow field directly from the resulting carrier.
        """
        state: InformationState = make_information_state(
            "spectrum",
            active_semantics=("intensity",),
            destroyed_information=("phase",),
            invalidated_claims=("coherent-amplitude",),
        )

        assert state.active_semantics == ("intensity",)
        assert state.destroyed_information == ("phase",)
        assert state.invalidated_claims == ("coherent-amplitude",)


class TestProvenanceGraph:
    """Validate :class:`~diffpes.types.ProvenanceGraph` storage.

    The carrier must retain deterministic root semantics, propagated states,
    topological order, and graph identity.

    :see: :class:`~diffpes.types.ProvenanceGraph`
    """

    def test_stores_empty_valid_graph(self) -> None:
        """Represent a valid external root with no transformation edges.

        The check verifies the minimal graph boundary independently of graph
        propagation behavior owned by ``diffpes.certify``.

        Notes
        -----
        The test builds one root state and an edge-free graph, then compares root,
        order, state, and checksum fields with their explicit inputs.
        """
        state: InformationState = make_information_state(
            "input", active_semantics=("bands",)
        )
        graph: ProvenanceGraph = make_provenance_graph(
            records=(),
            external_inputs=("input",),
            initial_semantics=(("input", ("bands",)),),
            topological_order=("input",),
            information=(state,),
            validation_errors=(),
            graph_checksum="sha256:test",
        )

        assert graph.external_inputs == ("input",)
        assert graph.information == (state,)
        assert graph.topological_order == ("input",)
        assert graph.graph_checksum == "sha256:test"


class TestProvenanceReport:
    """Validate :class:`~diffpes.types.ProvenanceReport` storage.

    The carrier must retain graph endpoints and validation status without
    introducing differentiable leaves.

    :see: :class:`~diffpes.types.ProvenanceReport`
    """

    def test_stores_valid_graph_summary(self) -> None:
        """Preserve graph roots, terminals, order, and checksum.

        The check verifies the complete successful-report boundary for a
        single-root, single-terminal graph.

        Notes
        -----
        The test constructs a valid report through the public factory and compares its
        static endpoint fields with independently specified tuples.
        """
        report: ProvenanceReport = make_provenance_report(
            valid=True,
            errors=(),
            roots=("input",),
            terminal_outputs=("spectrum",),
            orphaned_inputs=(),
            topological_order=("input", "spectrum"),
            graph_checksum="sha256:test",
        )

        assert report.valid is True
        assert report.roots == ("input",)
        assert report.terminal_outputs == ("spectrum",)
        assert report.topological_order == ("input", "spectrum")


class TestMakeInformationState:
    """Validate :func:`~diffpes.types.make_information_state`.

    The factory must reject blank node identities and duplicate semantic
    labels before constructing an immutable state.

    :see: :func:`~diffpes.types.make_information_state`
    """

    @pytest.mark.parametrize(
        ("node_id", "active_semantics", "message"),
        [
            ("", (), "nonblank string"),
            ("node", ("bands", "bands"), "duplicates"),
        ],
    )
    def test_rejects_invalid_state(
        self,
        node_id: str,
        active_semantics: Tuple[str, ...],
        message: str,
    ) -> None:
        """Reject an invalid node identity or duplicated semantic label.

        The parameter table covers the two independent validation boundaries
        of the semantic-state factory.

        Notes
        -----
        The test calls the factory for each malformed static input and matches the
        diagnostic emitted by the corresponding validation branch.
        """
        with pytest.raises(ValueError, match=message):
            make_information_state(node_id, active_semantics)


class TestMakeProvenanceGraph:
    """Validate :func:`~diffpes.types.make_provenance_graph`.

    The factory must require exactly one initial-semantic entry for every
    external root and unique propagated state identities.

    :see: :func:`~diffpes.types.make_provenance_graph`
    """

    def test_rejects_incomplete_root_semantics(self) -> None:
        """Reject a graph whose initial semantics omit an external root.

        The check isolates the root-coverage invariant at the carrier factory
        rather than relying on the higher-level graph builder.

        Notes
        -----
        Supplies one external input and no initial-semantic pairs, then matches
        the explicit root-coverage diagnostic.
        """
        with pytest.raises(ValueError, match="every external input"):
            make_provenance_graph(
                records=(),
                external_inputs=("input",),
                initial_semantics=(),
                topological_order=("input",),
                information=(),
                validation_errors=(),
                graph_checksum="sha256:test",
            )


class TestMakeProvenanceReport:
    """Validate :func:`~diffpes.types.make_provenance_report`.

    The factory must keep the report validity flag equivalent to an empty
    validation-error sequence.

    :see: :func:`~diffpes.types.make_provenance_report`
    """

    def test_rejects_inconsistent_validity(self) -> None:
        """Reject a successful provenance report containing errors.

        The check verifies the Boolean consistency invariant independently of
        graph construction and semantic propagation.

        Notes
        -----
        Supplies ``valid=True`` with one error and matches the consistency
        diagnostic raised before construction.
        """
        with pytest.raises(ValueError, match="valid must be true"):
            make_provenance_report(
                valid=True,
                errors=("cycle",),
                roots=(),
                terminal_outputs=(),
                orphaned_inputs=(),
                topological_order=(),
                graph_checksum="sha256:test",
            )
