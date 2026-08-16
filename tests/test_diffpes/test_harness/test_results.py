"""Validate executable JSON descriptions, results, and result keys.

The tests cover non-finite sanitization, deterministic identities, description
payload fields, standard result fields, extras, and sorted JSON emission.
"""

import json
from pathlib import Path

import numpy as np
from beartype.typing import Any, Dict

from diffpes.harness import (
    build_result,
    describe_payload,
    emit,
    json_ready,
    result_key,
)
from diffpes.types import ArtifactRecord, AutomatonSpec, make_automaton_spec


class TestJsonReady:
    """Validate :func:`~diffpes.harness.json_ready` sanitization.

    The case scope covers arrays, paths, and non-finite scalar values.
    """

    def test_converts_arrays_paths_and_nonfinite_values(self) -> None:
        """Convert arrays paths and non-finite values to JSON data.

        A strict payload must not contain NaN or infinity literals.

        Notes
        -----
        Convert one nested object with a path and NumPy array values.
        """
        ready: Any = json_ready(
            {
                "path": Path("output/file.txt"),
                "values": np.array([1.0, np.nan]),
            }
        )

        assert ready == {"path": "output/file.txt", "values": [1.0, None]}


class TestResultKey:
    """Validate :func:`~diffpes.harness.result_key` determinism.

    The case scope covers mapping key order and stable SHA-256 output.
    """

    def test_compares_equal_across_mapping_insertion_orders(self) -> None:
        """Compare result keys across parameter mapping insertion orders.

        Equivalent parameter objects must produce one content-addressed digest.

        Notes
        -----
        Compute two keys from equal mappings with opposite insertion orders.
        """
        first: str = result_key("example", {"a": 1, "b": 2}, 0, "1.0")
        second: str = result_key("example", {"b": 2, "a": 1}, 0, "1.0")

        assert first == second
        assert len(first) == 64


class TestBuildResult:
    """Validate :func:`~diffpes.harness.build_result` payload fields.

    The case scope covers standard fields artifacts and top-level extras.
    """

    def test_builds_standard_fields_and_preserves_body_extras(self) -> None:
        """Build standard fields and preserve body extras.

        A result must carry metrics artifacts returns and a SHA-256 key.

        Notes
        -----
        Build one success payload with an artifact record and fit extra.
        """
        spec: AutomatonSpec = make_automaton_spec(
            "example",
            (),
            returns={"metrics": {"value": {"type": "number"}}},
        )
        artifact: ArtifactRecord = ArtifactRecord(
            role="array",
            mime="application/npz",
            path="arrays.npz",
            preview_b64="",
        )
        result: Dict[str, Any] = build_result(
            spec,
            "ok",
            {},
            0,
            {"value": 1.0},
            (artifact,),
            {"fit": {"converged": True}, "metrics": "ignored"},
            0.1,
        )

        assert result["status"] == "ok"
        assert result["metrics"] == {"value": 1.0}
        assert result["artifacts"][0]["path"] == "arrays.npz"
        assert result["fit"] == {"converged": True}
        assert result["result_key"] == result_key(
            "example", {}, 0, result["diffpes_version"]
        )


class TestDescribePayload:
    """Validate :func:`~diffpes.harness.describe_payload` output.

    The case scope covers fixed keys schema fields and inherited flags.
    """

    def test_builds_the_declared_description_keys_and_flag_pairs(self) -> None:
        """Build declared description keys and inherited flag pairs.

        An agent must discover schema summary returns and inherited flags.

        Notes
        -----
        Build one empty specification and compare its payload key set.
        """
        spec: AutomatonSpec = make_automaton_spec("example", ())
        payload: Dict[str, Any] = describe_payload(spec)

        assert set(payload) == {
            "schema_version",
            "experiment",
            "summary",
            "params_schema",
            "returns",
            "inherited_flags",
        }
        assert all(len(pair) == 2 for pair in payload["inherited_flags"])


class TestEmit:
    """Validate :func:`~diffpes.harness.emit` standard-output behavior.

    The case scope covers one sorted strict JSON output line.
    """

    def test_writes_one_sorted_json_line(self, capsys: Any) -> None:
        """Write one sorted JSON line to standard output.

        Consumers must parse the final line without a human-readable prefix.

        Notes
        -----
        Emit two reverse-ordered keys and load the captured final JSON line.
        """
        emit({"z": 1, "a": 2})
        captured: Any = capsys.readouterr()
        payload: Dict[str, Any] = json.loads(captured.out)

        assert captured.out == '{"a": 2, "z": 1}\n'
        assert payload == {"a": 2, "z": 1}
