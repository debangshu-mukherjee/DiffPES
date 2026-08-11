"""Store types-owned carriers for artifact provenance and information flow.

Extended Summary
----------------
The carriers in this module contain the immutable result of provenance graph
analysis.  Graph construction and semantic propagation remain owned by
``diffpes.certify``; this module owns only their validated data boundary.

Routine Listings
----------------
:class:`InformationState`
    Store effective semantic state for one artifact or result node.
:class:`ProvenanceAnalysis`
    Store the complete result of one provenance-graph analysis.
:class:`ProvenanceGraph`
    Store a validated lineage graph and its propagated semantics.
:class:`ProvenanceReport`
    Store a structural and semantic provenance-validation report.
:func:`make_information_state`
    Create a validated semantic-information state for one graph node.
:func:`make_provenance_analysis`
    Create an immutable provenance-analysis carrier.
:func:`make_provenance_graph`
    Create a validated immutable provenance graph carrier.
:func:`make_provenance_report`
    Create a validated structural and semantic provenance report.
"""

from collections.abc import Sequence

import equinox as eqx
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped

from .certification import TransformationRecord


class InformationState(eqx.Module):
    """Store effective semantic state for one artifact or result node.

    This carrier separates semantics that remain available from information
    and claims invalidated along the node's provenance path.

    :see: :class:`~.test_provenance.TestInformationState`

    Attributes
    ----------
    node_id : str
        Artifact or result identifier (**static** -- a compile-time constant;
        changing it triggers retracing).
    active_semantics : Tuple[str, ...]
        Available scientific semantics (**static** -- compile-time constants;
        changing them triggers retracing).
    destroyed_information : Tuple[str, ...]
        Information lost before this node (**static** -- compile-time
        constants; changing them triggers retracing).
    invalidated_claims : Tuple[str, ...]
        Claims invalidated before this node (**static** -- compile-time
        constants; changing them triggers retracing).

    Notes
    -----
    The state is declarative static metadata. It records information flow but
    does not alter or differentiate through the associated physical arrays.

    See Also
    --------
    make_information_state : Create a validated semantic-information state
        for one graph node.
    """

    node_id: str = eqx.field(static=True)
    active_semantics: Tuple[str, ...] = eqx.field(static=True)
    destroyed_information: Tuple[str, ...] = eqx.field(static=True)
    invalidated_claims: Tuple[str, ...] = eqx.field(static=True)


class ProvenanceAnalysis(eqx.Module):
    """Store the complete result of one provenance-graph analysis.

    This carrier retains the ordered transformations, propagated information,
    graph endpoints, and structural diagnostics from one deterministic walk.

    :see: :class:`~.test_provenance.TestProvenanceAnalysis`

    Attributes
    ----------
    ordered_records : Tuple[TransformationRecord, ...]
        Transformation records in deterministic graph order.
    topological_order : Tuple[str, ...]
        Output node identities in deterministic order (**static** -- changing
        them triggers retracing).
    information : Tuple[InformationState, ...]
        Propagated semantic state for every known graph node.
    errors : Tuple[str, ...]
        Structural and semantic diagnostics (**static** -- changing them
        triggers retracing).
    roots : Tuple[str, ...]
        Root node identities (**static** -- changing them triggers retracing).
    terminal_outputs : Tuple[str, ...]
        Terminal output identities (**static** -- changing them triggers
        retracing).
    orphaned_inputs : Tuple[str, ...]
        Unconsumed external input identities (**static** -- changing them
        triggers retracing).

    Notes
    -----
    The carrier records graph analysis only. It does not alter physical arrays
    or introduce a differentiable reduction.

    See Also
    --------
    make_provenance_analysis : Create an immutable provenance-analysis
        carrier.
    """

    ordered_records: Tuple[TransformationRecord, ...]
    topological_order: Tuple[str, ...] = eqx.field(static=True)
    information: Tuple[InformationState, ...]
    errors: Tuple[str, ...] = eqx.field(static=True)
    roots: Tuple[str, ...] = eqx.field(static=True)
    terminal_outputs: Tuple[str, ...] = eqx.field(static=True)
    orphaned_inputs: Tuple[str, ...] = eqx.field(static=True)


class ProvenanceGraph(eqx.Module):
    """Store a validated lineage graph and its propagated semantics.

    The graph retains every transformation edge, external root, and effective
    semantic state needed to inspect information flow without reevaluation.

    :see: :class:`~.test_provenance.TestProvenanceGraph`

    Attributes
    ----------
    records : Tuple[TransformationRecord, ...]
        Transformation records in graph order.
    external_inputs : Tuple[str, ...]
        External root identifiers (**static** -- compile-time constants;
        changing them triggers retracing).
    initial_semantics : Tuple[Tuple[str, Tuple[str, ...]], ...]
        Initial semantics per external root (**static** -- compile-time
        constants; changing them triggers retracing).
    topological_order : Tuple[str, ...]
        Validated node order (**static** -- compile-time constants; changing
        them triggers retracing).
    information : Tuple[InformationState, ...]
        Propagated semantic state for graph nodes.
    validation_errors : Tuple[str, ...]
        Structural or semantic validation errors (**static** -- compile-time
        constants; changing them triggers retracing).
    graph_checksum : str
        Deterministic consistency checksum (**static** -- a compile-time
        constant; changing it triggers retracing).

    Notes
    -----
    The graph is an immutable audit carrier evaluated outside forward kernels.
    Its records and information states contain no differentiable array leaves.

    See Also
    --------
    make_provenance_graph : Create a validated immutable provenance graph
        carrier.
    """

    records: Tuple[TransformationRecord, ...]
    external_inputs: Tuple[str, ...] = eqx.field(static=True)
    initial_semantics: Tuple[Tuple[str, Tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    topological_order: Tuple[str, ...] = eqx.field(static=True)
    information: Tuple[InformationState, ...]
    validation_errors: Tuple[str, ...] = eqx.field(static=True)
    graph_checksum: str = eqx.field(static=True)


class ProvenanceReport(eqx.Module):
    """Store a structural and semantic provenance-validation report.

    This carrier summarizes graph validity and exposes its roots, terminal
    outputs, orphaned inputs, and deterministic traversal identity.

    :see: :class:`~.test_provenance.TestProvenanceReport`

    Attributes
    ----------
    valid : bool
        Whether validation succeeded (**static** -- a compile-time constant;
        changing it triggers retracing).
    errors : Tuple[str, ...]
        Validation failures (**static** -- compile-time constants; changing
        them triggers retracing).
    roots : Tuple[str, ...]
        Root node identifiers (**static** -- compile-time constants; changing
        them triggers retracing).
    terminal_outputs : Tuple[str, ...]
        Terminal output identifiers (**static** -- compile-time constants;
        changing them triggers retracing).
    orphaned_inputs : Tuple[str, ...]
        Unconsumed external input identifiers (**static** -- compile-time
        constants; changing them triggers retracing).
    topological_order : Tuple[str, ...]
        Validated node order (**static** -- compile-time constants; changing
        them triggers retracing).
    graph_checksum : str
        Deterministic consistency checksum (**static** -- a compile-time
        constant; changing it triggers retracing).

    Notes
    -----
    The report contains only static graph metadata and therefore contributes no
    gradient path to a certified forward execution.

    See Also
    --------
    make_provenance_report : Create a validated structural and semantic
        provenance report.
    """

    valid: bool = eqx.field(static=True)
    errors: Tuple[str, ...] = eqx.field(static=True)
    roots: Tuple[str, ...] = eqx.field(static=True)
    terminal_outputs: Tuple[str, ...] = eqx.field(static=True)
    orphaned_inputs: Tuple[str, ...] = eqx.field(static=True)
    topological_order: Tuple[str, ...] = eqx.field(static=True)
    graph_checksum: str = eqx.field(static=True)


def _require_text(value: str, name: str) -> str:
    """PRIVATE: Require one nonblank string.

    Parameters
    ----------
    value : str
        Candidate text value.
    name : str
        Field name used in the static error message.

    Returns
    -------
    value : str
        The validated input string, unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not a ``str`` or contains only whitespace. This
        is the static construction-time contract.

    Notes
    -----
    Apply ``isinstance`` and ``str.strip`` so that wrong types fail
    together with whitespace-only text.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _text_tuple(
    values: Sequence[str],
    name: str,
    *,
    unique: bool = True,
) -> Tuple[str, ...]:
    """PRIVATE: Validate and freeze one string sequence.

    Implementation Logic
    --------------------
    Validate every entry through ``_require_text`` while freezing the
    sequence into a tuple. Then compare the set size against the tuple
    length when ``unique`` is true.

    Parameters
    ----------
    values : Sequence[str]
        Candidate sequence of text entries.
    name : str
        Field name used in the static error messages.
    unique : bool
        Reject duplicate entries when true. Default is True.

    Returns
    -------
    result : Tuple[str, ...]
        The validated entries frozen into a tuple in input order.

    Raises
    ------
    ValueError
        If duplicates are present while ``unique`` is true.
        ``_require_text`` also raises ``ValueError`` for an entry that
        is not a nonblank string. This is the static construction-time
        contract.
    """
    result: Tuple[str, ...] = tuple(
        _require_text(value, name) for value in values
    )
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@jaxtyped(typechecker=beartype)
def make_information_state(  # noqa: DOC502
    node_id: str,
    active_semantics: Tuple[str, ...] = (),
    destroyed_information: Tuple[str, ...] = (),
    invalidated_claims: Tuple[str, ...] = (),
) -> InformationState:
    """Create a validated semantic-information state for one graph node.

    Freeze named semantic sets while rejecting blank or duplicate entries.

    :see: :class:`~.test_provenance.TestMakeInformationState`

    Implementation Logic
    --------------------
    1. **Validate node identity**::

           node_id=_require_text(node_id, "node_id")

       Reject a blank node identifier.
    2. **Freeze semantic sets**::

           active_semantics=_text_tuple(active_semantics, "active_semantics")

       Convert each sequence to an immutable, duplicate-free tuple.
    3. **Construct the state**::

           state = InformationState(...)

       Bind and return the semantic state carrier.

    Parameters
    ----------
    node_id : str
        Artifact or result identifier (**static** -- a compile-time constant;
        changing it triggers retracing).
    active_semantics : Tuple[str, ...]
        Available semantics (**static** -- compile-time constants; changing
        them triggers retracing). Default is empty.
    destroyed_information : Tuple[str, ...]
        Lost information labels (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    invalidated_claims : Tuple[str, ...]
        Invalidated claim identifiers (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.

    Returns
    -------
    state : InformationState
        Validated immutable semantic-information state.

    Raises
    ------
    ValueError
        If an identifier is blank or a semantic sequence contains duplicates.

    Notes
    -----
    Validation is static; the resulting carrier contains no numerical leaves.
    """
    state: InformationState = InformationState(
        node_id=_require_text(node_id, "node_id"),
        active_semantics=_text_tuple(
            active_semantics,
            "active_semantics",
        ),
        destroyed_information=_text_tuple(
            destroyed_information,
            "destroyed_information",
        ),
        invalidated_claims=_text_tuple(
            invalidated_claims,
            "invalidated_claims",
        ),
    )
    return state


@jaxtyped(typechecker=beartype)
def make_provenance_analysis(  # noqa: PLR0913
    ordered_records: Tuple[TransformationRecord, ...],
    topological_order: Tuple[str, ...],
    information: Tuple[InformationState, ...],
    errors: Tuple[str, ...],
    roots: Tuple[str, ...],
    terminal_outputs: Tuple[str, ...],
    orphaned_inputs: Tuple[str, ...],
) -> ProvenanceAnalysis:
    """Create an immutable provenance-analysis carrier.

    Freeze the complete result of one deterministic graph walk without a
    second policy or physics reduction.

    :see: :class:`~.test_provenance.TestMakeProvenanceAnalysis`

    Implementation Logic
    --------------------
    1. **Freeze the analysis fields**::

           analysis = ProvenanceAnalysis(...)

       Convert each input sequence to a tuple and construct the canonical
       carrier.

    Parameters
    ----------
    ordered_records : Tuple[TransformationRecord, ...]
        Transformation records in deterministic graph order.
    topological_order : Tuple[str, ...]
        Output node identities in deterministic order (**static** -- changing
        them triggers retracing).
    information : Tuple[InformationState, ...]
        Propagated semantic state for every known graph node.
    errors : Tuple[str, ...]
        Structural and semantic diagnostics (**static** -- changing them
        triggers retracing).
    roots : Tuple[str, ...]
        Root node identities (**static** -- changing them triggers retracing).
    terminal_outputs : Tuple[str, ...]
        Terminal output identities (**static** -- changing them triggers
        retracing).
    orphaned_inputs : Tuple[str, ...]
        Unconsumed external input identities (**static** -- changing them
        triggers retracing).

    Returns
    -------
    analysis : ProvenanceAnalysis
        Immutable result of the provenance-graph analysis.

    Notes
    -----
    The factory preserves diagnostic order and repeated errors. Invalid graph
    analyses therefore remain inspectable by the certification layer.
    """
    analysis: ProvenanceAnalysis = ProvenanceAnalysis(
        ordered_records=tuple(ordered_records),
        topological_order=tuple(topological_order),
        information=tuple(information),
        errors=tuple(errors),
        roots=tuple(roots),
        terminal_outputs=tuple(terminal_outputs),
        orphaned_inputs=tuple(orphaned_inputs),
    )
    return analysis


@jaxtyped(typechecker=beartype)
def make_provenance_graph(  # noqa: PLR0913
    records: Tuple[TransformationRecord, ...],
    external_inputs: Tuple[str, ...],
    initial_semantics: Tuple[Tuple[str, Tuple[str, ...]], ...],
    topological_order: Tuple[str, ...],
    information: Tuple[InformationState, ...],
    validation_errors: Tuple[str, ...],
    graph_checksum: str,
) -> ProvenanceGraph:
    """Create a validated immutable provenance graph carrier.

    Validate carrier types, root coverage, and unique semantic-state IDs,
    then freeze the graph in deterministic topological order.

    :see: :class:`~.test_provenance.TestMakeProvenanceGraph`

    Implementation Logic
    --------------------
    1. **Validate carrier sequences**::

           frozen_records = tuple(records)

       Require transformation records and information states to use their
       canonical types.
    2. **Validate root semantics**::

           if set(semantic_nodes) != set(inputs):

       Require one unique initial-semantic entry for every external input.
    3. **Validate state identities**::

           if len(state_ids) != len(set(state_ids)):

       Prevent ambiguous propagated semantic states.
    4. **Construct the graph**::

           graph = ProvenanceGraph(...)

       Freeze the validated graph and bind the result.

    Parameters
    ----------
    records : Tuple[TransformationRecord, ...]
        Transformation records in graph order.
    external_inputs : Tuple[str, ...]
        External root identifiers (**static** -- compile-time constants;
        changing them triggers retracing).
    initial_semantics : Tuple[Tuple[str, Tuple[str, ...]], ...]
        Initial semantics for every root (**static** -- compile-time constants;
        changing them triggers retracing).
    topological_order : Tuple[str, ...]
        Validated node order (**static** -- compile-time constants; changing
        them triggers retracing).
    information : Tuple[InformationState, ...]
        Propagated semantic states for graph nodes.
    validation_errors : Tuple[str, ...]
        Graph validation errors (**static** -- compile-time constants;
        changing them triggers retracing).
    graph_checksum : str
        Deterministic consistency checksum (**static** -- a compile-time
        constant; changing it triggers retracing).

    Returns
    -------
    graph : ProvenanceGraph
        Validated immutable provenance graph.

    Raises
    ------
    TypeError
        If ``records`` or ``information`` contains the wrong carrier type.
    ValueError
        If text values are blank or duplicated. The function also rejects
        incomplete initial semantics or duplicate information-state node IDs.

    Notes
    -----
    Graph validation uses only static identities and carrier structure; it does
    not inspect or reduce physical model arrays.
    """
    frozen_records: Tuple[TransformationRecord, ...] = tuple(records)
    if any(not isinstance(record, TransformationRecord) for record in records):
        raise TypeError("records must contain TransformationRecord instances")
    inputs: Tuple[str, ...] = _text_tuple(external_inputs, "external_inputs")
    semantic_pairs: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
        (
            _require_text(node_id, "initial_semantics node_id"),
            _text_tuple(semantics, "initial_semantics values"),
        )
        for node_id, semantics in initial_semantics
    )
    semantic_nodes: Tuple[str, ...] = tuple(
        node_id for node_id, _ in semantic_pairs
    )
    if len(semantic_nodes) != len(set(semantic_nodes)):
        raise ValueError("initial_semantics node IDs must be unique")
    if set(semantic_nodes) != set(inputs):
        raise ValueError(
            "initial_semantics must describe every external input"
        )
    states: Tuple[InformationState, ...] = tuple(information)
    if any(not isinstance(state, InformationState) for state in states):
        raise TypeError("information must contain InformationState instances")
    state_ids: Tuple[str, ...] = tuple(state.node_id for state in states)
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("information node IDs must be unique")
    graph: ProvenanceGraph = ProvenanceGraph(
        records=frozen_records,
        external_inputs=inputs,
        initial_semantics=semantic_pairs,
        topological_order=_text_tuple(
            topological_order,
            "topological_order",
        ),
        information=states,
        validation_errors=_text_tuple(
            validation_errors,
            "validation_errors",
            unique=False,
        ),
        graph_checksum=_require_text(graph_checksum, "graph_checksum"),
    )
    return graph


@jaxtyped(typechecker=beartype)
def make_provenance_report(
    valid: bool,
    errors: Tuple[str, ...],
    roots: Tuple[str, ...],
    terminal_outputs: Tuple[str, ...],
    orphaned_inputs: Tuple[str, ...],
    topological_order: Tuple[str, ...],
    graph_checksum: str,
) -> ProvenanceReport:
    """Create a validated structural and semantic provenance report.

    Freeze the graph summary and require the validity flag to agree exactly
    with whether validation errors are present.

    :see: :class:`~.test_provenance.TestMakeProvenanceReport`

    Implementation Logic
    --------------------
    1. **Normalize errors**::

           normalized_errors = _text_tuple(errors, "errors", unique=False)

       Freeze validation failures without hiding repeated diagnostics.
    2. **Check validity consistency**::

           if valid == bool(normalized_errors):

       Require success exactly when no graph error is present.
    3. **Construct the report**::

           report = ProvenanceReport(...)

       Freeze graph endpoints and bind the result.

    Parameters
    ----------
    valid : bool
        Whether graph validation succeeded (**static** -- a compile-time
        constant; changing it triggers retracing).
    errors : Tuple[str, ...]
        Validation failures (**static** -- compile-time constants; changing
        them triggers retracing).
    roots : Tuple[str, ...]
        Root identifiers (**static** -- compile-time constants; changing them
        triggers retracing).
    terminal_outputs : Tuple[str, ...]
        Terminal output identifiers (**static** -- compile-time constants;
        changing them triggers retracing).
    orphaned_inputs : Tuple[str, ...]
        Unconsumed input identifiers (**static** -- compile-time constants;
        changing them triggers retracing).
    topological_order : Tuple[str, ...]
        Validated node order (**static** -- compile-time constants; changing
        them triggers retracing).
    graph_checksum : str
        Deterministic consistency checksum (**static** -- a compile-time
        constant; changing it triggers retracing).

    Returns
    -------
    report : ProvenanceReport
        Validated immutable provenance report.

    Raises
    ------
    ValueError
        If a text value is blank, a unique sequence contains duplicates, or
        ``valid`` does not agree with whether ``errors`` is empty.

    Notes
    -----
    Report construction is static and introduces no differentiable leaves.
    """
    normalized_errors: Tuple[str, ...] = _text_tuple(
        errors,
        "errors",
        unique=False,
    )
    if valid == bool(normalized_errors):
        raise ValueError("valid must be true exactly when errors is empty")
    report: ProvenanceReport = ProvenanceReport(
        valid=valid,
        errors=normalized_errors,
        roots=_text_tuple(roots, "roots"),
        terminal_outputs=_text_tuple(terminal_outputs, "terminal_outputs"),
        orphaned_inputs=_text_tuple(orphaned_inputs, "orphaned_inputs"),
        topological_order=_text_tuple(
            topological_order,
            "topological_order",
        ),
        graph_checksum=_require_text(graph_checksum, "graph_checksum"),
    )
    return report


__all__: list[str] = [
    "InformationState",
    "ProvenanceAnalysis",
    "ProvenanceGraph",
    "ProvenanceReport",
    "make_information_state",
    "make_provenance_analysis",
    "make_provenance_graph",
    "make_provenance_report",
]
