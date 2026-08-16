"""Run executable experiments through one command-line contract.

Extended Summary
----------------
This module builds parsers, merges parameters, creates run contexts, and
emits structured JSON results. The decorator binds an ordinary experiment body
to the reusable command-line process boundary.

Routine Listings
----------------
:func:`build_parser`
    Build an argument parser for one executable experiment.
:func:`deadline_context`
    Enforce an optional POSIX wall-time deadline.
:func:`enable_compilation_cache`
    Set the JAX persistent compilation cache.
:func:`experiment`
    Decorate one experiment body with the executable command-line contract.
:func:`run_automaton`
    Run one executable experiment with optional argument text.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import time
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

from beartype import beartype
from beartype.typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from jax.experimental.compilation_cache import compilation_cache
from jaxtyping import jaxtyped

from diffpes.constants import (
    AUTOMATON_CACHE_ENV_VAR,
    AUTOMATON_DEFAULT_OUTDIR,
    AUTOMATON_RUNTIME_CHECK_ENV_VAR,
)
from diffpes.types import (
    ArtifactRecord,
    AutomatonContext,
    AutomatonParam,
    AutomatonSpec,
    make_automaton_context,
    make_automaton_spec,
)

from .errors import (
    AutomatonError,
    DeadlineExceededError,
    classify_exception,
    exit_code_for,
)
from .parameters import merge_params, read_params_document
from .results import build_result, describe_payload, emit


def _argument_flags(name: str) -> Sequence[str]:
    """PRIVATE: Build hyphenated and underscored flag aliases for one name.

    Parameters
    ----------
    name : str
        Declared executable parameter identifier.

    Returns
    -------
    flags : Sequence[str]
        One or two command-line flag spellings.

    Notes
    -----
    Keeps one spelling when the parameter name contains no underscore. The
    returned order favors conventional hyphenated command-line names.
    """
    hyphenated: str = f"--{name.replace('_', '-')}"
    underscored: str = f"--{name}"
    flags: Sequence[str] = (
        (hyphenated,)
        if hyphenated == underscored
        else (hyphenated, underscored)
    )
    return flags


def _parser_choices(param: AutomatonParam) -> Optional[Sequence[Any]]:
    """PRIVATE: Select parser choices that accept raw command-line values.

    Parameters
    ----------
    param : AutomatonParam
        Declared executable parameter metadata.

    Returns
    -------
    choices : Optional[Sequence[Any]]
        Choices suitable for argparse, or ``None`` for structured values.

    Notes
    -----
    Leaves structured JSON values for harness validation. Scalar choices pass
    through argparse only after their scalar converter runs.
    """
    if param.choices is None or param.python_type in (bool, dict, list):
        choices: Optional[Sequence[Any]] = None
    else:
        choices = param.choices
    return choices


def _error_result(
    spec: AutomatonSpec,
    seed: int,
    params: Mapping[str, Any],
    error: Exception,
    wall_seconds: float,
) -> int:
    """PRIVATE: Build one classified failure result and return its exit code.

    Parameters
    ----------
    spec : AutomatonSpec
        Static metadata for the executable experiment.
    seed : int
        Parsed deterministic random seed.
    params : Mapping[str, Any]
        Validated values available before the failure.
    error : Exception
        Failure that requires stable classification.
    wall_seconds : float
        Elapsed body time before the failure.

    Returns
    -------
    exit_code : int
        Process exit code for the classified failure.

    Notes
    -----
    Converts the exception once, then builds one result object. The function
    keeps errors on standard output parseable by agent callers.
    """
    classified: Exception = classify_exception(error)
    error_kind: str = getattr(classified, "error_kind", "Unknown")
    status: str = "timeout" if error_kind == "Timeout" else "error"
    result: Dict[str, Any] = build_result(
        spec,
        status,
        params,
        seed,
        {},
        (),
        {},
        wall_seconds,
        error=classified,
    )
    emit(result)
    exit_code: int = exit_code_for(error_kind)
    return exit_code


def _body_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """PRIVATE: Normalize one optional experiment-body payload mapping.

    Parameters
    ----------
    payload : Optional[Mapping[str, Any]]
        Body return value, or ``None`` when the body has no result data.

    Returns
    -------
    normalized : Dict[str, Any]
        Mutable copy of the body payload with required defaults.

    Raises
    ------
    TypeError
        If the payload, metrics, or artifacts have an invalid structure.

    Notes
    -----
    Adds empty metric and artifact collections. It verifies artifact records
    before the result builder serializes them.
    """
    if payload is None:
        normalized: Dict[str, Any] = {}
    elif isinstance(payload, Mapping):
        normalized = dict(payload)
    else:
        message: str = "experiment body must return a mapping or None"
        raise TypeError(message)
    metrics: Any = normalized.get("metrics", {})
    artifacts: Any = normalized.get("artifacts", ())
    if not isinstance(metrics, Mapping):
        message = "experiment metrics must be a mapping"
        raise TypeError(message)
    if not isinstance(artifacts, Sequence) or isinstance(
        artifacts, (str, bytes)
    ):
        message = "experiment artifacts must be a sequence"
        raise TypeError(message)
    if not all(isinstance(artifact, ArtifactRecord) for artifact in artifacts):
        message = "experiment artifacts must contain ArtifactRecord values"
        raise TypeError(message)
    normalized["metrics"] = dict(metrics)
    normalized["artifacts"] = tuple(artifacts)
    return normalized


def _unchecked_reexec(argv: Optional[Sequence[str]]) -> int:
    """PRIVATE: Re-execute once with jaxtyping runtime checks disabled.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Explicit command-line text, or ``None`` for process arguments.

    Returns
    -------
    exit_code : int
        Fallback exit code after a monkeypatched execution call returns.

    Notes
    -----
    Sets the documented environment marker before ``os.execv`` replaces this
    process. The fallback supports deterministic in-process test doubles.
    """
    if argv is None:
        arguments: Sequence[str] = tuple(sys.argv)
    else:
        arguments = (sys.argv[0], *argv)
    os.environ[AUTOMATON_RUNTIME_CHECK_ENV_VAR] = "1"
    os.execv(sys.executable, (sys.executable, *arguments))  # noqa: S606
    exit_code: int = 1
    return exit_code


@jaxtyped(typechecker=beartype)
def build_parser(spec: AutomatonSpec) -> argparse.ArgumentParser:
    """Build an argument parser for one executable experiment.

    The parser adds inherited flags first, followed by each declared parameter.
    Every parameter accepts hyphenated and underscored flag spellings.

    :see: :class:`~.test_runner.TestBuildParser`

    Implementation Logic
    --------------------
    1. **Add inherited controls**::

           parser.add_argument("--describe", action="store_true")

       The controls implement discovery, validation, runtime, and output modes.

    2. **Add declared parameters**::

           parser.add_argument(
               *flags, dest=param.name, default=argparse.SUPPRESS
           )

       Suppressed defaults let merge logic distinguish explicit overrides.

    Parameters
    ----------
    spec : AutomatonSpec
        Static metadata for the executable experiment.

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured parser for inherited and declared flags.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=spec.description,
        exit_on_error=False,
    )
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--params", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--outdir", default=AUTOMATON_DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--unchecked", action="store_true")
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    param: AutomatonParam
    for param in spec.params:
        flags: Sequence[str] = _argument_flags(param.name)
        kwargs: Dict[str, Any] = {
            "dest": param.name,
            "default": argparse.SUPPRESS,
            "help": param.help,
        }
        if param.python_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            if param.python_type in (int, float, str):
                kwargs["type"] = param.python_type
            else:
                kwargs["type"] = str
            choices: Optional[Sequence[Any]] = _parser_choices(param)
            if choices is not None:
                kwargs["choices"] = choices
        parser.add_argument(*flags, **kwargs)
    return parser


@jaxtyped(typechecker=beartype)
def enable_compilation_cache(directory: Optional[str | Path] = None) -> Path:
    """Set the JAX persistent compilation cache.

    The function creates the selected directory and delegates cache activation
    to JAX. It reads the documented environment variable when no path exists.

    :see: :class:`~.test_runner.TestEnableCompilationCache`

    Notes
    -----
    Uses the JAX public compilation-cache surface. It does not compile an
    experiment or mutate JAX numerical precision settings.

    Parameters
    ----------
    directory : Optional[str | Path], optional
        Explicit cache directory. Default reads the environment or home cache.

    Returns
    -------
    cache_directory : Path
        Existing directory configured for JAX persistent compilation cache.
    """
    selected: str | Path
    if directory is None:
        selected = os.environ.get(
            AUTOMATON_CACHE_ENV_VAR,
            str(Path.home() / ".cache" / "diffpes" / "jax"),
        )
    else:
        selected = directory
    cache_directory: Path = Path(selected)
    cache_directory.mkdir(parents=True, exist_ok=True)
    compilation_cache.set_cache_dir(str(cache_directory))
    return cache_directory


@jaxtyped(typechecker=beartype)
def deadline_context(
    seconds: Optional[float],
) -> AbstractContextManager[None]:
    """Enforce an optional POSIX wall-time deadline.

    The context raises ``DeadlineExceededError`` on POSIX hosts. It uses a
    no-op context on platforms that do not expose interval alarms.

    :see: :class:`~.test_runner.TestDeadlineContext`

    Notes
    -----
    Restores the previous alarm handler and timer after the body exits. A
    ``None`` deadline creates a no-op context on every platform.

    Parameters
    ----------
    seconds : Optional[float]
        Positive wall-time limit in seconds, or ``None`` for no limit.

    Returns
    -------
    context : AbstractContextManager[None]
        Context that enforces the requested deadline where POSIX supports it.

    Raises
    ------
    ValueError
        If a supplied deadline is not positive.
    """
    if seconds is not None and seconds <= 0.0:
        message: str = "deadline must be positive"
        raise ValueError(message)
    if seconds is None or not hasattr(signal, "SIGALRM"):
        context: AbstractContextManager[None] = contextlib.nullcontext()
        return context

    @contextlib.contextmanager
    def managed() -> Iterator[None]:
        """Preserve one POSIX timer and its previous state.

        Yields
        ------
        None
            Control to the executable body.

        Notes
        -----
        Replaces the alarm handler only while the body runs. The finalizer
        restores both the previous handler and the previous interval timer.
        """

        def timeout_handler(signum: int, frame: Any) -> None:
            """Raise the stable deadline error from a POSIX alarm signal.

            Parameters
            ----------
            signum : int
                POSIX signal number supplied by the operating system.
            frame : Any
                Interrupted Python frame supplied by the signal runtime.

            Raises
            ------
            DeadlineExceededError
                Always after the configured wall-time limit expires.

            Notes
            -----
            Ignores signal details because the context owns only one alarm
            category. The raised error drives the executable timeout result.
            """
            del signum, frame
            raise DeadlineExceededError()

        previous_handler: Any = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        previous_timer: Tuple[float, float] = signal.setitimer(
            signal.ITIMER_REAL,
            seconds,
        )
        try:
            yield None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(
                signal.ITIMER_REAL, previous_timer[0], previous_timer[1]
            )

    context = managed()
    return context  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def run_automaton(  # noqa: DOC105, PLR0911, PLR0915
    spec: AutomatonSpec,
    main: Callable[
        [SimpleNamespace, AutomatonContext], Optional[Mapping[str, Any]]
    ],
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run one executable experiment with optional argument text.

    The runner parses flags, validates parameters, builds a context, and runs
    the body. It emits a final JSON result for success, validation, or runtime
    failure without writing a traceback to standard output.

    :see: :class:`~.test_runner.TestRunAutomaton`

    Implementation Logic
    --------------------
    1. **Parse and validate input**::

           params = merge_params(spec.params, document, cli_overrides)

       The merge applies defaults, JSON input, and explicit flags in order.

    2. **Run the experiment body**::

           payload = main(args, context)

       The context supplies a deterministic random key and safe artifact root.

    3. **Emit the final result**::

           emit(
               build_result(
                   spec, "ok", params, seed, metrics, artifacts, extras, wall
               )
           )

       The JSON line gives agents one process-boundary result record.

    Parameters
    ----------
    spec : AutomatonSpec
        Static metadata for the executable experiment.
    main : Callable
        Experiment body that returns metrics, artifacts, and optional extras.
    argv : Optional[Sequence[str]], optional
        Explicit argument text. Default reads process command-line arguments.

    Returns
    -------
    exit_code : int
        Process-compatible success or classified failure exit code.
    """
    parser: argparse.ArgumentParser = build_parser(spec)
    parse_error: argparse.ArgumentError
    exc: SystemExit
    parsed: argparse.Namespace
    try:
        parsed = parser.parse_args(None if argv is None else list(argv))
    except argparse.ArgumentError as parse_error:
        parse_message: str = str(parse_error)
        parse_kind: str = (
            "Unsupported"
            if "invalid choice" in parse_message
            else "InvalidInput"
        )
        parse_failure: Exception = AutomatonError(
            parse_message,
            error_kind=parse_kind,
        )
        exit_code: int = _error_result(spec, 0, {}, parse_failure, 0.0)
        return exit_code  # noqa: RET504 -- assign-before-return is required.
    except SystemExit as exc:
        exit_code: int = int(exc.code)
        return exit_code  # noqa: RET504 -- assign-before-return is required.
    seed: int = parsed.seed
    if (
        parsed.unchecked
        and os.environ.get(AUTOMATON_RUNTIME_CHECK_ENV_VAR) != "1"
    ):
        exit_code = _unchecked_reexec(argv)
        return exit_code  # noqa: RET504 -- assign-before-return is required.
    if parsed.describe:
        description: Dict[str, Any] = describe_payload(spec)
        emit(description)
        exit_code = 0
        return exit_code  # noqa: RET504 -- assign-before-return is required.
    runtime_error: Exception
    params: Dict[str, Any] = {}
    start_time: float = time.perf_counter()
    try:
        document: Dict[str, Any] = (
            read_params_document(parsed.params)
            if parsed.params is not None
            else {}
        )
        cli_overrides: Dict[str, Any] = {
            parameter.name: getattr(parsed, parameter.name)
            for parameter in spec.params
            if hasattr(parsed, parameter.name)
        }
        params = merge_params(spec.params, document, cli_overrides)
        argument_values: Dict[str, Any] = vars(parsed).copy()
        argument_values.update(params)
        args: SimpleNamespace = SimpleNamespace(**argument_values)
        context: AutomatonContext = make_automaton_context(
            parsed.outdir,
            seed,
            spec.name,
            json_mode=parsed.json,
        )
        if parsed.cache:
            enable_compilation_cache()
        if parsed.validate:
            result: Dict[str, Any] = build_result(
                spec,
                "ok",
                params,
                seed,
                {"valid": True},
                (),
                {},
                0.0,
            )
            emit(result)
            exit_code = 0
            return exit_code  # noqa: RET504 -- assign-before-return is required.
        if parsed.estimate:
            default_estimate: Dict[str, Any] = {
                "est_wall_s": None,
                "needs_gpu": False,
                "est_mem_gb": None,
                "cache_warm": None,
            }
            estimated: Mapping[str, Any] = (
                spec.estimate(args)
                if spec.estimate is not None
                else default_estimate
            )
            emit(estimated)
            exit_code = 0
            return exit_code  # noqa: RET504 -- assign-before-return is required.
        with deadline_context(parsed.deadline):
            raw_payload: Optional[Mapping[str, Any]] = main(args, context)
        wall_seconds: float = time.perf_counter() - start_time
        payload: Dict[str, Any] = _body_payload(raw_payload)
        metrics: Mapping[str, Any] = payload.pop("metrics")
        artifacts: Sequence[ArtifactRecord] = payload.pop("artifacts")
        result = build_result(
            spec,
            "ok",
            params,
            seed,
            metrics,
            artifacts,
            payload,
            wall_seconds,
        )
        emit(result)
        exit_code = 0
    except Exception as runtime_error:
        wall_seconds = time.perf_counter() - start_time
        exit_code = _error_result(
            spec, seed, params, runtime_error, wall_seconds
        )
    return exit_code


@jaxtyped(typechecker=beartype)
def experiment(
    *,
    name: str,
    params: Sequence[AutomatonParam],
    returns: Optional[Mapping[str, Any]] = None,
    estimate: Optional[Callable[[Any], Mapping[str, Any]]] = None,
) -> Callable[
    [
        Callable[
            [SimpleNamespace, AutomatonContext], Optional[Mapping[str, Any]]
        ]
    ],
    Callable[[Optional[Sequence[str]]], int],
]:
    """Decorate one experiment body with the executable command-line contract.

    The decorator builds one static specification from the body declaration.
    The wrapped function returns an exit code for explicit arguments and raises
    ``SystemExit`` when a script invokes it without arguments.

    :see: :class:`~.test_runner.TestExperiment`

    Notes
    -----
    Copies the body docstring summary into the executable description. The
    wrapper stores the immutable specification as ``__automaton_spec__``.

    Parameters
    ----------
    name : str
        Stable executable experiment identifier.
    params : Sequence[AutomatonParam]
        Ordered static parameter declarations.
    returns : Optional[Mapping[str, Any]], optional
        Declared result fields. Default is an empty mapping.
    estimate : Optional[Callable[[Any], Mapping[str, Any]]], optional
        Optional host-side resource estimate callable. Default is ``None``.

    Returns
    -------
    decorator : Callable
        Decorator that wraps one body in the executable process boundary.
    """

    def decorate(
        main: Callable[
            [SimpleNamespace, AutomatonContext], Optional[Mapping[str, Any]]
        ],
    ) -> Callable[[Optional[Sequence[str]]], int]:
        summary: str = (main.__doc__ or "").splitlines()[0].strip()
        spec: AutomatonSpec = make_automaton_spec(
            name,
            params,
            returns=returns,
            description=summary,
            estimate=estimate,
        )

        @wraps(main)
        def wrapped(argv: Optional[Sequence[str]] = None) -> int:
            """Run the decorated experiment through its process boundary."""
            exit_code: int = run_automaton(spec, main, argv)
            if argv is None:
                raise SystemExit(exit_code)
            return exit_code

        setattr(wrapped, "__automaton_spec__", spec)  # noqa: B010
        return wrapped

    return decorate


__all__: list[str] = [
    "build_parser",
    "deadline_context",
    "enable_compilation_cache",
    "experiment",
    "run_automaton",
]
