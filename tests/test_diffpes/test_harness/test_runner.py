"""Validate the executable experiment parser, runner, and decorator.

The tests cover declared flags, schemas, validation, estimates, deadlines,
cache setup, unchecked execution, and decorated in-process execution.
"""

import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from beartype.typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from diffpes.harness import (
    DeadlineExceededError,
    build_parser,
    deadline_context,
    enable_compilation_cache,
    experiment,
    run_automaton,
)
from diffpes.types import (
    AutomatonContext,
    AutomatonParam,
    AutomatonSpec,
    make_automaton_param,
    make_automaton_spec,
)
from tests.test_automatons.test_contract import _schema_errors


def _payload_from_capture(capsys: Any) -> Dict[str, Any]:
    """PRIVATE: Parse the final captured stdout line as a JSON object.

    Parameters
    ----------
    capsys : Any
        Pytest capture fixture for the current in-process run.

    Returns
    -------
    payload : Dict[str, Any]
        Decoded final JSON object from standard output.

    Notes
    -----
    Selects the final nonempty line because the executable contract reserves it
    for machine-readable output.
    """
    captured: Any = capsys.readouterr()
    lines: List[str] = [line for line in captured.out.splitlines() if line]
    payload: Dict[str, Any] = json.loads(lines[-1])
    return payload


def _schema_document(name: str) -> Dict[str, Any]:
    """PRIVATE: Load one committed executable schema document.

    Parameters
    ----------
    name : str
        Filename below the executable schema directory.

    Returns
    -------
    schema : Dict[str, Any]
        Parsed committed JSON Schema object.

    Notes
    -----
    Resolves the repository root from this test module. The helper reads only
    committed metadata and creates no files.
    """
    repository_root: Path = Path(__file__).resolve().parents[3]
    path: Path = repository_root / "automatons" / "schema" / name
    schema: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return schema


class TestBuildParser:
    """Validate :func:`~diffpes.harness.build_parser` declared flags.

    The case scope covers aliases and BooleanOptionalAction behavior.
    """

    def test_accepts_parameter_aliases_and_boolean_negation(self) -> None:
        """Accept parameter aliases and BooleanOptionalAction negation.

        Both aliases must set one destination and boolean negation must parse.

        Notes
        -----
        Build one integer and one boolean declaration before parsing flags.
        """
        count: AutomatonParam = make_automaton_param("n_k", int, default=2)
        flag: AutomatonParam = make_automaton_param(
            "cache_data", bool, default=True
        )
        spec: AutomatonSpec = make_automaton_spec("example", (count, flag))
        parser: Any = build_parser(spec)
        hyphenated: Any = parser.parse_args(["--n-k", "3", "--no-cache-data"])
        underscored: Any = parser.parse_args(["--n_k", "4", "--cache_data"])

        assert hyphenated.n_k == 3
        assert hyphenated.cache_data is False
        assert underscored.n_k == 4
        assert underscored.cache_data is True


class TestDeadlineContext:
    """Validate :func:`~diffpes.harness.deadline_context` behavior.

    The case scope covers no-op, invalid, and POSIX timeout contexts.
    """

    def test_allows_a_noop_deadline_context(self) -> None:
        """Allow a no-op context without a deadline.

        The portable no-op path must execute the enclosed block successfully.

        Notes
        -----
        Enter the public context with ``None`` and bind one local value.
        """
        with deadline_context(None):
            value: int = 1

        assert value == 1

    def test_rejects_a_nonpositive_deadline(self) -> None:
        """Reject a nonpositive deadline value.

        A deadline must have a positive wall-time limit when it exists.

        Notes
        -----
        Request the context with zero seconds.
        """
        with pytest.raises(ValueError, match=r"deadline must be positive"):
            deadline_context(0.0)

    def test_installs_the_handler_before_arming_the_timer(
        self, monkeypatch: Any
    ) -> None:
        """Install the handler before arming the POSIX timer.

        The context must never arm a timer while a default handler remains.

        Notes
        -----
        Replace signal calls with ordered mocks and enter one deadline context.
        """
        previous_handler: Any = object()
        events: Mock = Mock()
        get_signal: Mock = Mock(return_value=previous_handler)
        set_signal: Mock = Mock(return_value=None)
        set_timer: Mock = Mock(return_value=(0.0, 0.0))
        events.attach_mock(get_signal, "getsignal")
        events.attach_mock(set_signal, "signal")
        events.attach_mock(set_timer, "setitimer")
        monkeypatch.setattr(
            "diffpes.harness.runner.signal.getsignal",
            get_signal,
        )
        monkeypatch.setattr(
            "diffpes.harness.runner.signal.signal",
            set_signal,
        )
        monkeypatch.setattr(
            "diffpes.harness.runner.signal.setitimer",
            set_timer,
        )

        with deadline_context(1.0):
            ran_body: bool = True

        event_names: List[str] = [item[0] for item in events.mock_calls]
        first_handler: Any = set_signal.call_args_list[0].args[1]
        first_timer_seconds: float = set_timer.call_args_list[0].args[1]
        restored_handler: Any = set_signal.call_args_list[1].args[1]
        disarmed_seconds: float = set_timer.call_args_list[1].args[1]

        assert ran_body is True
        assert event_names[:3] == ["getsignal", "signal", "setitimer"]
        assert callable(first_handler)
        assert first_timer_seconds == 1.0
        assert disarmed_seconds == 0.0
        assert restored_handler is previous_handler

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM is unavailable on this platform.",
    )
    def test_raises_the_deadline_error_after_a_posix_timeout(self) -> None:
        """Raise the deadline error after a POSIX timeout.

        The context must interrupt a body that outlives its configured limit.

        Notes
        -----
        Sleep longer than a short timer inside the public context.
        """
        with (
            pytest.raises(
                DeadlineExceededError,
                match=r"deadline exceeded",
            ),
            deadline_context(0.01),
        ):
            time.sleep(0.05)


class TestEnableCompilationCache:
    """Validate :func:`~diffpes.harness.enable_compilation_cache` setup.

    The case scope covers creation of an explicit cache directory.
    """

    def test_creates_the_requested_cache_directory(
        self, tmp_path: Path
    ) -> None:
        """Create the requested JAX compilation cache directory.

        Cache setup must return the path that it creates for JAX.

        Notes
        -----
        Request a nested temporary cache directory through the public helper.
        """
        expected: Path = tmp_path / "jax" / "cache"
        actual: Path = enable_compilation_cache(expected)

        assert actual == expected
        assert actual.is_dir()


class TestRunAutomaton:
    """Validate :func:`~diffpes.harness.run_automaton` in-process execution.

    The case scope covers schemas, precedence, errors, estimates, and timeouts.
    """

    def test_emits_description_and_result_payloads_that_match_schemas(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Build throwaway payloads that match the committed schemas.

        The runner must emit both discovery and success objects structurally.

        Notes
        -----
        Run one throwaway body through describe and normal execution paths.
        """
        count: AutomatonParam = make_automaton_param("count", int, default=2)
        spec: AutomatonSpec = make_automaton_spec(
            "throwaway",
            (count,),
            returns={"metrics": {"count": {"type": "integer"}}},
            description="Run a throwaway executable body.",
        )

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Return a metrics payload from one throwaway body."""
            del ctx
            payload: Dict[str, Any] = {"metrics": {"count": args.count}}
            return payload

        describe_code: int = run_automaton(
            spec, main, ["--describe", "--json"]
        )
        describe_payload: Dict[str, Any] = _payload_from_capture(capsys)
        result_code: int = run_automaton(
            spec,
            main,
            ["--outdir", str(tmp_path), "--json"],
        )
        result_payload: Dict[str, Any] = _payload_from_capture(capsys)
        describe_schema: Dict[str, Any] = _schema_document(
            "automaton_params.schema.json"
        )
        result_schema: Dict[str, Any] = _schema_document(
            "automaton_result.schema.json"
        )

        assert describe_code == 0
        assert result_code == 0
        assert _schema_errors(describe_payload, describe_schema) == []
        assert _schema_errors(result_payload, result_schema) == []
        assert result_payload["metrics"] == {"count": 2}

    def test_applies_document_then_command_line_precedence(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Apply document values before explicit command-line values.

        The body must receive the command-line value after every merge step.

        Notes
        -----
        Run one body with a default document value and explicit flag value.
        """
        count: AutomatonParam = make_automaton_param("count", int, default=1)
        spec: AutomatonSpec = make_automaton_spec("precedence", (count,))

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Return the merged count from one executable body."""
            del ctx
            payload: Dict[str, Any] = {"metrics": {"count": args.count}}
            return payload

        exit_code: int = run_automaton(
            spec,
            main,
            [
                "--params",
                '{"count": 2}',
                "--count",
                "3",
                "--outdir",
                str(tmp_path),
                "--json",
            ],
        )
        payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert exit_code == 0
        assert payload["params"] == {"count": 3}
        assert payload["metrics"] == {"count": 3}

    def test_validation_mode_skips_the_body_and_emits_a_result(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Validate parameters without invoking the experiment body.

        Validation mode must return zero and report a true valid metric.

        Notes
        -----
        Run one body that raises if validation mode invokes it.
        """
        spec: AutomatonSpec = make_automaton_spec("validation", ())

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Reject an unexpected body call during validation mode."""
            del args, ctx
            raise AssertionError("validation mode must not call the body")

        exit_code: int = run_automaton(
            spec,
            main,
            ["--validate", "--outdir", str(tmp_path), "--json"],
        )
        payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert exit_code == 0
        assert payload["metrics"] == {"valid": True}

    def test_reports_range_and_choice_failures_with_structured_fields(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Report range and choice failures with structured error fields.

        Both failures must return process code two and name their source field.

        Notes
        -----
        Run a bounded integer and a choice parameter through invalid flags.
        """
        count: AutomatonParam = make_automaton_param(
            "count",
            int,
            default=2,
            bounds=(1.0, 4.0),
        )
        model: AutomatonParam = make_automaton_param(
            "model",
            str,
            default="chain",
            choices=("chain", "graphene"),
        )
        spec: AutomatonSpec = make_automaton_spec("errors", (count, model))

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Return an empty body payload after validated input."""
            del args, ctx
            payload: Dict[str, Any] = {"metrics": {}}
            return payload

        range_code: int = run_automaton(
            spec,
            main,
            ["--count", "0", "--outdir", str(tmp_path), "--json"],
        )
        range_payload: Dict[str, Any] = _payload_from_capture(capsys)
        choice_code: int = run_automaton(
            spec,
            main,
            ["--model", "other", "--outdir", str(tmp_path), "--json"],
        )
        choice_payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert range_code == 2
        assert range_payload["error_kind"] == "ParamOutOfRange"
        assert range_payload["field"] == "count"
        assert choice_code == 2
        assert choice_payload["error_kind"] == "Unsupported"
        assert choice_payload["field"] is None

    def test_emits_the_raw_declared_estimate_mapping(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Build the raw declared estimate mapping.

        Estimate mode must not wrap declared estimate fields inside metrics.

        Notes
        -----
        Run one specification that supplies a fixed estimate callable.
        """
        estimate: Mapping[str, Any] = {
            "est_wall_s": 1.25,
            "needs_gpu": False,
            "est_mem_gb": 0.5,
            "cache_warm": True,
        }

        def estimate_body(args: Any) -> Mapping[str, Any]:
            """Return the fixed estimate for one executable specification."""
            del args
            output: Mapping[str, Any] = estimate
            return output

        spec: AutomatonSpec = make_automaton_spec(
            "estimate",
            (),
            estimate=estimate_body,
        )

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Reject an unexpected body call during estimate mode."""
            del args, ctx
            raise AssertionError("estimate mode must not call the body")

        exit_code: int = run_automaton(
            spec,
            main,
            ["--estimate", "--outdir", str(tmp_path), "--json"],
        )
        payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert exit_code == 0
        assert payload == estimate

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM is unavailable on this platform.",
    )
    def test_reports_a_deadline_as_a_timeout_result(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Report a deadline as a timeout result.

        The runner must map an elapsed POSIX timer to exit code 124.

        Notes
        -----
        Run one sleeping body with a deadline shorter than its sleep.
        """
        spec: AutomatonSpec = make_automaton_spec("timeout", ())

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Run past the configured deadline before returning."""
            del args, ctx
            time.sleep(0.05)
            payload: Dict[str, Any] = {"metrics": {}}
            return payload

        exit_code: int = run_automaton(
            spec,
            main,
            [
                "--deadline",
                "0.01",
                "--outdir",
                str(tmp_path),
                "--json",
            ],
        )
        payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert exit_code == 124
        assert payload["status"] == "timeout"
        assert payload["error_kind"] == "Timeout"

    def test_reexecutes_once_for_unchecked_mode(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        """Re-execute once for unchecked runtime validation mode.

        The runner must set the guard variable before it invokes ``os.execv``.

        Notes
        -----
        Replace ``os.execv`` with a local recorder and run unchecked mode.
        """
        calls: List[Tuple[str, Tuple[str, ...]]] = []

        def _fake_execv(program: str, arguments: Sequence[str]) -> None:
            """PRIVATE: Record one re-execution call without replacing Python.

            Parameters
            ----------
            program : str
                Interpreter path selected by the runner.
            arguments : Sequence[str]
                Interpreter arguments selected by the runner.

            Notes
            -----
            Appends immutable values so the test can inspect the re-execution.
            """
            calls.append((program, tuple(arguments)))

        monkeypatch.delenv("JAXTYPING_DISABLE", raising=False)
        monkeypatch.setattr("diffpes.harness.runner.os.execv", _fake_execv)
        spec: AutomatonSpec = make_automaton_spec("unchecked", ())

        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Return an empty payload when runtime checks remain enabled."""
            del args, ctx
            payload: Dict[str, Any] = {"metrics": {}}
            return payload

        exit_code: int = run_automaton(
            spec,
            main,
            ["--unchecked", "--outdir", str(tmp_path), "--json"],
        )

        assert exit_code == 1
        assert os.environ["JAXTYPING_DISABLE"] == "1"
        assert len(calls) == 1


class TestExperiment:
    """Validate :func:`~diffpes.harness.experiment` wrapper behavior.

    The case scope covers attached specifications and explicit argv exit codes.
    """

    def test_wraps_a_body_with_an_attached_specification(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Decorate a body with its static executable specification.

        Explicit arguments must return zero and expose the specification
        attribute.

        Notes
        -----
        Decorate one minimal body and execute its validation-only code path.
        """

        @experiment(name="wrapped", params=())
        def main(
            args: SimpleNamespace,
            ctx: AutomatonContext,
        ) -> Optional[Mapping[str, Any]]:
            """Return an empty executable body payload."""
            del args, ctx
            payload: Dict[str, Any] = {"metrics": {}, "artifacts": []}
            return payload

        exit_code: int = main(
            ["--validate", "--outdir", str(tmp_path), "--json"]
        )
        payload: Dict[str, Any] = _payload_from_capture(capsys)

        assert exit_code == 0
        assert hasattr(main, "__automaton_spec__")
        assert payload["status"] == "ok"
