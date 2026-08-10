"""Trace artifact lineage and semantic information loss.

Extended Summary
----------------
This module builds a deterministic directed acyclic graph from immutable
``TransformationRecord`` carriers. It rejects missing parents, cycles,
multiple producers, and unused declared inputs. The module propagates semantic
properties, information losses, and claim invalidations to every output.

Propagation is deliberately conservative: inherited semantics remain active
only when a transformation explicitly lists them in ``preserves``.  A missing
preservation declaration therefore becomes a visible information loss rather
than an accidental scientific promise.

Routine Listings
----------------
:func:`build_provenance`
    Build a deterministic provenance DAG and propagate information.
:func:`validate_provenance`
    Re-evaluate graph structure and derived state independently.
:func:`effective_information`
    Return propagated semantics, losses, and invalidations for one node.
:func:`invalidated_claims`
    Return every claim invalidated at or upstream of one output.
:func:`lineage`
    Return the transitive parent-node lineage of one output.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence

import equinox as eqx
from beartype import beartype
from beartype.typing import Any, Dict, Tuple, cast
from jaxtyping import jaxtyped

from diffpes.types import (
    InformationState,
    ProvenanceGraph,
    ProvenanceReport,
    TransformationRecord,
    make_information_state,
    make_provenance_graph,
    make_provenance_report,
)

from .checksums import checksum_pytree


class _Analysis(eqx.Module):
    """Store complete internal analysis before construction or comparison."""

    ordered_records: Tuple[TransformationRecord, ...]
    topological_order: Tuple[str, ...] = eqx.field(static=True)
    information: Tuple[InformationState, ...]
    errors: Tuple[str, ...] = eqx.field(static=True)
    roots: Tuple[str, ...] = eqx.field(static=True)
    terminal_outputs: Tuple[str, ...] = eqx.field(static=True)
    orphaned_inputs: Tuple[str, ...] = eqx.field(static=True)


def _normalize_terms(
    values: Iterable[str],
    *,
    field_name: str,
) -> Tuple[str, ...]:
    """PRIVATE: Validate and deterministically order one
    identifier/property set.

    Parameters
    ----------
    values : Iterable[str]
        Identifiers or semantic properties for one field.
    field_name : str
        Field label used in error messages.

    Returns
    -------
    result : Tuple[str, ...]
        The entries in sorted order.

    Raises
    ------
    ValueError
        If an entry is not a nonblank string, or if entries repeat.

    Notes
    -----
    The sort removes input-order sensitivity, so equal sets produce
    equal graph records and equal checksums.
    """
    normalized: Tuple[str, ...] = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() for value in normalized
    ):
        msg: str = f"{field_name} entries must be nonblank strings"
        raise ValueError(msg)
    if len(set(normalized)) != len(normalized):
        msg: str = f"{field_name} entries must be unique"
        raise ValueError(msg)
    result: Tuple[str, ...] = tuple(sorted(normalized))
    return result


def _normalize_external_inputs(
    external_inputs: Mapping[str, Iterable[str]] | Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    """PRIVATE: Normalize external node identities and their input
    semantics.

    Parameters
    ----------
    external_inputs : Mapping[str, Iterable[str]] | Iterable[str]
        External node identities, either as a mapping from node identity
        to its initial semantic properties or as a bare identity
        iterable.

    Returns
    -------
    result : Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]
        Sorted external node identities and one ``(node_id, semantics)``
        pair per node. Bare iterables yield empty semantics for every
        node.

    Notes
    -----
    Both the identities and each semantics set pass through
    :func:`_normalize_terms`, so blank or repeated entries fail before
    graph construction.
    """
    node_ids: Tuple[str, ...]
    semantic_pairs: Tuple[Tuple[str, Tuple[str, ...]], ...]
    if isinstance(external_inputs, Mapping):
        input_mapping: Mapping[str, Iterable[str]] = cast(
            "Mapping[str, Iterable[str]]",
            external_inputs,
        )
        node_ids = _normalize_terms(
            input_mapping.keys(),
            field_name="external_inputs",
        )
        semantic_pairs = tuple(
            (
                node_id,
                _normalize_terms(
                    input_mapping[node_id],
                    field_name=f"initial semantics for {node_id}",
                ),
            )
            for node_id in node_ids
        )
    else:
        node_ids = _normalize_terms(
            external_inputs,
            field_name="external_inputs",
        )
        semantic_pairs = tuple((node_id, ()) for node_id in node_ids)
    result: Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]] = (
        node_ids,
        semantic_pairs,
    )
    return result


def _record_key(
    record: TransformationRecord,
    original_index: int,
) -> Tuple[str, str, Tuple[str, ...], int]:
    """PRIVATE: Return a deterministic ordering key for one
    transformation execution.

    Parameters
    ----------
    record : TransformationRecord
        Transformation execution record.
    original_index : int
        Position of the record in the caller-supplied sequence.

    Returns
    -------
    key : Tuple[str, str, Tuple[str, ...], int]
        Transformation identity, version, output identities, and the
        original index, in comparison order.

    Notes
    -----
    The key sorts ready records in the Kahn walk. The trailing original
    index breaks ties between otherwise identical records, so the
    topological order is total and deterministic.
    """
    key: Tuple[str, str, Tuple[str, ...], int] = (
        record.transformation_id,
        record.transformation_version,
        record.output_ids,
        original_index,
    )
    return key


def _record_errors(  # noqa: PLR0912
    records: Sequence[TransformationRecord],
    external_inputs: Tuple[str, ...],
) -> Tuple[
    list[str],
    Dict[str, int],
    set[str],
    Tuple[str, ...],
]:
    """PRIVATE: Collect local record and node-identity errors.

    Implementation Logic
    --------------------
    Walks the records once and reports blank identities, missing
    outputs, repetitions, self-consumption, and contradictory terms.
    Also reports blank outputs and multiple producers. It then reports
    external inputs that
    internal records also produce, parents that no known node supplies,
    and declared external inputs that no record consumes.

    Parameters
    ----------
    records : Sequence[TransformationRecord]
        Transformation execution records in caller order.
    external_inputs : Tuple[str, ...]
        Normalized external node identities.

    Returns
    -------
    result : Tuple[list[str], Dict[str, int], set[str], Tuple[str, ...]]
        Deterministic error messages, the producer-record index per
        output identity, the set of consumed node identities, and the
        sorted orphaned external inputs.
    """
    index: Any
    record: Any
    output_id: Any
    errors: list[str] = []
    producer: Dict[str, int] = {}
    consumed: set[str] = set()
    all_outputs: list[str] = []
    for index, record in enumerate(records):
        reference: str = (
            f"{record.transformation_id}@{record.transformation_version}"
        )
        if not record.transformation_id.strip():
            errors.append(f"record {index} has a blank transformation ID")
        if not record.transformation_version.strip():
            errors.append(f"record {index} has a blank transformation version")
        if not record.output_ids:
            errors.append(f"record {index} ({reference}) has no outputs")
        if len(set(record.parent_ids)) != len(record.parent_ids):
            errors.append(f"record {index} ({reference}) repeats a parent")
        if len(set(record.output_ids)) != len(record.output_ids):
            errors.append(f"record {index} ({reference}) repeats an output")
        overlap: set[str] = set(record.parent_ids) & set(record.output_ids)
        if overlap:
            details: str = ", ".join(sorted(overlap))
            errors.append(
                f"record {index} ({reference}) directly consumes its own "
                f"output: {details}"
            )
        contradictory: set[str] = set(record.preserves) & set(record.destroys)
        contradictory |= set(record.introduces) & set(record.destroys)
        if contradictory:
            details = ", ".join(sorted(contradictory))
            errors.append(
                f"record {index} ({reference}) both retains and destroys: "
                f"{details}"
            )
        for output_id in record.output_ids:
            if not output_id.strip():
                errors.append(
                    f"record {index} ({reference}) has a blank output"
                )
                continue
            if output_id in producer:
                errors.append(
                    f"multiple transformations produce output {output_id!r}"
                )
            else:
                producer[output_id] = index
            all_outputs.append(output_id)
        consumed.update(record.parent_ids)
    collision: set[str] = set(external_inputs) & set(all_outputs)
    if collision:
        details = ", ".join(sorted(collision))
        errors.append(
            f"external inputs are also produced internally: {details}"
        )
    known_nodes: set[str] = set(external_inputs) | set(all_outputs)
    for index, record in enumerate(records):
        missing: set[str] = set(record.parent_ids) - known_nodes
        if missing:
            details = ", ".join(sorted(missing))
            errors.append(f"record {index} has missing parents: {details}")
    orphaned: Tuple[str, ...] = (
        tuple(sorted(set(external_inputs) - consumed)) if records else ()
    )
    if orphaned:
        errors.append(
            "declared external inputs are not consumed: " + ", ".join(orphaned)
        )
    result: Tuple[list[str], Dict[str, int], set[str], Tuple[str, ...]] = (
        errors,
        producer,
        consumed,
        orphaned,
    )
    return result


def _topological_indices(
    records: Sequence[TransformationRecord],
    producer: Mapping[str, int],
) -> Tuple[Tuple[int, ...], bool]:
    """PRIVATE: Return deterministic Kahn ordering and whether a cycle
    remains.

    Implementation Logic
    --------------------
    Builds record-level dependency and child sets from the producer map,
    then runs Kahn's algorithm with a heap ordered by
    :func:`_record_key`, so equal-indegree records leave the queue in
    one deterministic order. Records that never reach zero indegree
    form a cycle. Appends those records by the same sorted key so
    callers still receive a total order.

    Parameters
    ----------
    records : Sequence[TransformationRecord]
        Transformation execution records in caller order.
    producer : Mapping[str, int]
        Producer-record index per output identity.

    Returns
    -------
    result : Tuple[Tuple[int, ...], bool]
        Record indices in deterministic topological order and the cycle
        flag.
    """
    record: Any
    parent_index: Any
    degree: Any
    child: Any
    dependencies: list[set[int]] = []
    children: list[set[int]] = [set() for _ in records]
    for index, record in enumerate(records):
        parents: set[int] = {
            producer[parent]
            for parent in record.parent_ids
            if parent in producer and producer[parent] != index
        }
        dependencies.append(parents)
        for parent_index in parents:
            children[parent_index].add(index)
    indegree: list[int] = [len(parents) for parents in dependencies]
    ready: list[Tuple[Tuple[str, str, Tuple[str, ...], int], int]] = []
    for index, degree in enumerate(indegree):
        if degree == 0:
            heapq.heappush(ready, (_record_key(records[index], index), index))
    ordered: list[int] = []
    while ready:
        ready_item: Tuple[Tuple[str, str, Tuple[str, ...], int], int] = (
            heapq.heappop(ready)
        )
        index: int = ready_item[1]
        ordered.append(index)
        for child in sorted(children[index]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(
                    ready,
                    (_record_key(records[child], child), child),
                )
    has_cycle: bool = len(ordered) != len(records)
    if has_cycle:
        remaining: list[int] = sorted(
            set(range(len(records))) - set(ordered),
            key=lambda index: _record_key(records[index], index),
        )
        ordered.extend(remaining)
    result: Tuple[Tuple[int, ...], bool] = (tuple(ordered), has_cycle)
    return result


def _propagate_information(
    ordered_indices: Sequence[int],
    records: Sequence[TransformationRecord],
    initial_semantics: Tuple[Tuple[str, Tuple[str, ...]], ...],
    *,
    has_cycle: bool,
) -> Tuple[InformationState, ...]:
    """PRIVATE: Propagate semantics only when the graph admits a
    complete ordering.

    Implementation Logic
    --------------------
    Seeds each external node with its declared initial semantics. On a
    cyclic graph, returns only those seed states and propagates nothing.
    Otherwise walks the records in topological order. A node inherits
    its parents' active semantics, losses, and invalidated claims.
    ``preserves`` keeps inherited properties active. Unpreserved and
    destroyed properties join the losses. Introduced properties join
    the active set. ``invalidates_claims`` adds invalidations. Every
    output receives the same propagated state.

    Parameters
    ----------
    ordered_indices : Sequence[int]
        Record indices in topological order.
    records : Sequence[TransformationRecord]
        Transformation execution records in caller order.
    initial_semantics : Tuple[Tuple[str, Tuple[str, ...]], ...]
        Per external node, its initial semantic properties.
    has_cycle : bool
        Whether the topological ordering is incomplete.

    Returns
    -------
    information : Tuple[InformationState, ...]
        Propagated states sorted by node identity, or only the seed
        states when the graph has a cycle.
    """
    index: Any
    output_id: Any
    state: Dict[str, InformationState] = {
        node_id: make_information_state(
            node_id=node_id,
            active_semantics=semantics,
            destroyed_information=(),
            invalidated_claims=(),
        )
        for node_id, semantics in initial_semantics
    }
    if has_cycle:
        information: Tuple[InformationState, ...] = tuple(
            state[node_id] for node_id, _ in initial_semantics
        )
        return information  # noqa: RET504
    for index in ordered_indices:
        record: TransformationRecord = records[index]
        parent_states: Tuple[InformationState, ...] = tuple(
            state[parent_id]
            for parent_id in record.parent_ids
            if parent_id in state
        )
        inherited: set[str] = {
            term
            for parent_state in parent_states
            for term in parent_state.active_semantics
        }
        losses: set[str] = {
            term
            for parent_state in parent_states
            for term in parent_state.destroyed_information
        }
        invalidated: set[str] = {
            claim
            for parent_state in parent_states
            for claim in parent_state.invalidated_claims
        }
        retained: set[str] = inherited & set(record.preserves)
        losses.update(inherited - retained)
        losses.update(record.destroys)
        retained.difference_update(record.destroys)
        retained.update(record.introduces)
        invalidated.update(record.invalidates_claims)
        for output_id in record.output_ids:
            state[output_id] = make_information_state(
                node_id=output_id,
                active_semantics=tuple(sorted(retained)),
                destroyed_information=tuple(sorted(losses)),
                invalidated_claims=tuple(sorted(invalidated)),
            )
    information = tuple(state[node_id] for node_id in sorted(state))
    return information  # noqa: RET504


def _analyze(
    records: Sequence[TransformationRecord],
    external_inputs: Tuple[str, ...],
    initial_semantics: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> _Analysis:
    """PRIVATE: Perform deterministic structural analysis and semantic
    propagation.

    Implementation Logic
    --------------------
    Collects local errors and the producer map with
    :func:`_record_errors`, orders the records with
    :func:`_topological_indices` (a cycle adds one explicit error), and
    propagates information states with :func:`_propagate_information`.
    Derives roots from external inputs and outputs of parentless
    records. Derives terminal outputs from nodes that no record
    consumes. Stores everything in one immutable analysis carrier.

    Parameters
    ----------
    records : Sequence[TransformationRecord]
        Transformation execution records in caller order.
    external_inputs : Tuple[str, ...]
        Normalized external node identities.
    initial_semantics : Tuple[Tuple[str, Tuple[str, ...]], ...]
        Per external node, its initial semantic properties.

    Returns
    -------
    analysis : _Analysis
        Ordered records, topological output order, information states,
        errors, roots, terminal outputs, and orphaned inputs.

    Notes
    -----
    Both graph construction and independent validation call this one
    analysis, so the two paths cannot disagree.
    """
    record_analysis: Tuple[
        list[str], Dict[str, int], set[str], Tuple[str, ...]
    ] = _record_errors(
        records,
        external_inputs,
    )
    errors: list[str] = record_analysis[0]
    producer: Dict[str, int] = record_analysis[1]
    consumed: set[str] = record_analysis[2]
    orphaned: Tuple[str, ...] = record_analysis[3]
    topology: Tuple[Tuple[int, ...], bool] = _topological_indices(
        records,
        producer,
    )
    ordered_indices: Tuple[int, ...] = topology[0]
    has_cycle: bool = topology[1]
    if has_cycle:
        errors.append("provenance graph contains a cycle")
    ordered_records: Tuple[TransformationRecord, ...] = tuple(
        records[index] for index in ordered_indices
    )
    output_order: Tuple[str, ...] = tuple(
        output_id
        for record in ordered_records
        for output_id in record.output_ids
    )
    information: Tuple[InformationState, ...] = _propagate_information(
        ordered_indices,
        records,
        initial_semantics,
        has_cycle=has_cycle,
    )
    all_outputs: set[str] = set(producer)
    roots: Tuple[str, ...] = tuple(
        sorted(
            set(external_inputs)
            | {
                output_id
                for record in records
                if not record.parent_ids
                for output_id in record.output_ids
            }
        )
    )
    terminal_outputs: Tuple[str, ...] = tuple(sorted(all_outputs - consumed))
    analysis: _Analysis = _Analysis(
        ordered_records=ordered_records,
        topological_order=output_order,
        information=information,
        errors=tuple(errors),
        roots=roots,
        terminal_outputs=terminal_outputs,
        orphaned_inputs=orphaned,
    )
    return analysis


@jaxtyped(typechecker=beartype)
def build_provenance(
    records: Sequence[TransformationRecord],
    *,
    external_inputs: Mapping[str, Iterable[str]] | Iterable[str] = (),
    strict: bool = True,
) -> ProvenanceGraph:
    """Build a deterministic provenance DAG and propagate information.

    The operation propagates semantic state, information loss, and claim
    invalidation through a directed artifact graph. It reports contradictions
    explicitly.

    :see: :class:`~.test_provenance.TestBuildProvenance`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           graph: ProvenanceGraph = make_provenance_graph(
                   records=normalized_records,
                   external_inputs=input_ids,
                   initial_semantics=semantic_pairs,
                   topological_order=analysis.topological_order,
                   information=analysis.information,
                   validation_errors=analysis.errors,
                   graph_checksum=checksum,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    records : Sequence[TransformationRecord]
        Transformation executions linking parent and output artifact IDs.
    external_inputs : Mapping[str, Iterable[str]] | Iterable[str], optional
        Known source-node IDs. A mapping additionally declares the semantic
        properties initially available on each source.
    strict : bool, optional
        Raise :class:`ValueError` for any invalid graph. ``False`` is
        useful to construct an inspectable rejected graph.

    Returns
    -------
    graph : ProvenanceGraph
        Immutable graph with propagated per-node semantic state.

    Raises
    ------
    ValueError
        If strict validation detects a cycle or an invalid node. This error
        also identifies an unused input or contradictory information.
    """
    normalized_records: Tuple[TransformationRecord, ...] = tuple(records)
    normalized_inputs: Tuple[
        Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]
    ] = _normalize_external_inputs(external_inputs)
    input_ids: Tuple[str, ...] = normalized_inputs[0]
    semantic_pairs: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        normalized_inputs[1]
    )
    analysis: _Analysis = _analyze(
        normalized_records,
        input_ids,
        semantic_pairs,
    )
    if strict and analysis.errors:
        msg: str = "; ".join(analysis.errors)
        raise ValueError(msg)
    checksum: str = checksum_pytree(
        (normalized_records, semantic_pairs),
        record_kind="provenance",
    )
    graph: ProvenanceGraph = make_provenance_graph(
        records=normalized_records,
        external_inputs=input_ids,
        initial_semantics=semantic_pairs,
        topological_order=analysis.topological_order,
        information=analysis.information,
        validation_errors=analysis.errors,
        graph_checksum=checksum,
    )
    return graph


@jaxtyped(typechecker=beartype)
def validate_provenance(graph: ProvenanceGraph) -> ProvenanceReport:
    """Re-evaluate graph structure and derived state independently.

    The operation propagates semantic state, information loss, and claim
    invalidation through a directed artifact graph. It reports contradictions
    explicitly.

    :see: :class:`~.test_provenance.TestValidateProvenance`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           report: ProvenanceReport = make_provenance_report(
                   valid=not errors,
                   errors=tuple(errors),
                   roots=analysis.roots,
                   terminal_outputs=analysis.terminal_outputs,
                   orphaned_inputs=analysis.orphaned_inputs,
                   topological_order=analysis.topological_order,
                   graph_checksum=expected_checksum,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    graph : ProvenanceGraph
        Graph returned by :func:`build_provenance` or read from storage.

    Returns
    -------
    report : ProvenanceReport
        Complete deterministic validation result.
    """
    analysis: _Analysis = _analyze(
        graph.records,
        graph.external_inputs,
        graph.initial_semantics,
    )
    errors: list[str] = list(analysis.errors)
    expected_checksum: str = checksum_pytree(
        (graph.records, graph.initial_semantics),
        record_kind="provenance",
    )
    if graph.graph_checksum != expected_checksum:
        errors.append("provenance consistency checksum does not match records")
    if graph.topological_order != analysis.topological_order:
        errors.append(
            "stored topological order does not match graph structure"
        )
    if graph.information != analysis.information:
        errors.append("stored information propagation does not match records")
    report: ProvenanceReport = make_provenance_report(
        valid=not errors,
        errors=tuple(errors),
        roots=analysis.roots,
        terminal_outputs=analysis.terminal_outputs,
        orphaned_inputs=analysis.orphaned_inputs,
        topological_order=analysis.topological_order,
        graph_checksum=expected_checksum,
    )
    return report


@jaxtyped(typechecker=beartype)
def effective_information(
    graph: ProvenanceGraph,
    output_id: str,
) -> InformationState:
    """Return propagated semantics, losses, and invalidations for one node.

    The operation propagates semantic state, information loss, and claim
    invalidation through a directed artifact graph. It reports contradictions
    explicitly.

    :see: :class:`~.test_provenance.TestEffectiveInformation`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = f"unknown provenance node: {output_id}"

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    graph : ProvenanceGraph
        Validated provenance graph to inspect.
    output_id : str
        Artifact or result node identifier.

    Returns
    -------
    information : InformationState
        Propagated semantic state at the requested node.

    Raises
    ------
    KeyError
        If ``output_id`` is absent from the graph.
    """
    state: Any
    for state in graph.information:
        if state.node_id == output_id:
            information: InformationState = state
            return information  # noqa: RET504
    msg: str = f"unknown provenance node: {output_id}"
    raise KeyError(msg)


@jaxtyped(typechecker=beartype)
def invalidated_claims(
    graph: ProvenanceGraph,
    output_id: str,
) -> Tuple[str, ...]:
    """Return every claim invalidated at or upstream of one output.

    The operation propagates semantic state, information loss, and claim
    invalidation through a directed artifact graph. It reports contradictions
    explicitly.

    :see: :class:`~.test_provenance.TestInvalidatedClaims`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           claims: Tuple[str, ...] = effective_information(
                   graph,
                   output_id,
               ).invalidated_claims

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    graph : ProvenanceGraph
        Validated provenance graph to inspect.
    output_id : str
        Artifact or result node identifier.

    Returns
    -------
    claims : Tuple[str, ...]
        Sorted claim identifiers invalidated along the upstream lineage.
    """
    claims: Tuple[str, ...] = effective_information(
        graph,
        output_id,
    ).invalidated_claims
    return claims


@jaxtyped(typechecker=beartype)
def lineage(graph: ProvenanceGraph, output_id: str) -> Tuple[str, ...]:
    """Return the transitive parent-node lineage of one output.

    The operation propagates semantic state, information loss, and claim
    invalidation through a directed artifact graph. It reports contradictions
    explicitly.

    :see: :class:`~.test_provenance.TestLineage`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           result: Tuple[str, ...] = tuple(sorted(ancestors))

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    graph : ProvenanceGraph
        Validated provenance graph to inspect.
    output_id : str
        Artifact or result node identifier.

    Returns
    -------
    result : Tuple[str, ...]
        Sorted transitive ancestor identifiers.

    Raises
    ------
    KeyError
        If ``output_id`` is absent from the graph.
    """
    parent_id: Any
    producer: Dict[str, TransformationRecord] = {
        produced: record
        for record in graph.records
        for produced in record.output_ids
    }
    known: set[str] = set(graph.external_inputs) | set(producer)
    if output_id not in known:
        msg: str = f"unknown provenance node: {output_id}"
        raise KeyError(msg)
    ancestors: set[str] = set()
    pending: list[str] = [output_id]
    while pending:
        current: str = pending.pop()
        record: TransformationRecord | None = producer.get(current)
        if record is None:
            continue
        for parent_id in record.parent_ids:
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                pending.append(parent_id)
    result: Tuple[str, ...] = tuple(sorted(ancestors))
    return result


__all__: list[str] = [
    "build_provenance",
    "effective_information",
    "invalidated_claims",
    "lineage",
    "validate_provenance",
]
