"""Validate the public contract for agent-runnable experiments.

The module checks the frozen catalog, schema documents, pin rewriter, CI job,
and every experiment that exists in the checkout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from beartype.typing import Any, Dict, List, Set, Tuple

pytestmark: List[pytest.MarkDecorator] = [
    pytest.mark.big_mem,
    pytest.mark.xdist_group("automatons_cpu"),
]


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
AUTOMATON_DIRECTORY: Path = REPOSITORY_ROOT / "automatons"
PARAMETER_SCHEMA_PATH: Path = (
    AUTOMATON_DIRECTORY / "schema" / "automaton_params.schema.json"
)
RESULT_SCHEMA_PATH: Path = (
    AUTOMATON_DIRECTORY / "schema" / "automaton_result.schema.json"
)
EXPECTED_FILENAMES: Tuple[str, ...] = (
    "arpes_ingest.py",
    "audit_derivatives.py",
    "bump_pin.py",
    "certify_forward.py",
    "convergence_study.py",
    "counting_statistics.py",
    "experiment_design_compare.py",
    "export_model.py",
    "fit_experiment_geometry.py",
    "fit_hopping_parameters.py",
    "fit_self_energy.py",
    "forward_arpes_cube.py",
    "forward_bands.py",
    "forward_detector_acquisition.py",
    "forward_spectral_cut.py",
    "information_spectrum.py",
    "match_measured_to_simulated.py",
    "parameter_grid.py",
    "photon_energy_scan.py",
    "polarization_dichroism.py",
    "resolution_sweep.py",
    "vasp_bands_to_arpes.py",
)


def _experiment_scripts() -> Tuple[Path, ...]:
    """PRIVATE: List experiment scripts that currently exist in the checkout.

    Returns
    -------
    scripts : Tuple[Path, ...]
        Sorted direct scripts except the pin maintenance tool.

    Notes
    -----
    Reads the directory during collection. Later additions participate in a
    fresh pytest process without changing this test module.
    """
    scripts: Tuple[Path, ...] = tuple(
        path
        for path in sorted(AUTOMATON_DIRECTORY.glob("*.py"))
        if path.name != "bump_pin.py"
    )
    return scripts


def _load_document(path: Path) -> Dict[str, Any]:
    """PRIVATE: Load one JSON document as an object.

    Parameters
    ----------
    path : Path
        JSON document path.

    Returns
    -------
    document : Dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    AssertionError
        Raised when the document does not contain an object.

    Notes
    -----
    Parses UTF-8 text with the standard library. The assertion keeps schema
    checks focused on object documents.
    """
    decoded: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    document: Dict[str, Any] = decoded
    return document


def _last_json(stdout: str) -> Dict[str, Any]:
    """PRIVATE: Parse the final nonempty stdout line as an object.

    Parameters
    ----------
    stdout : str
        Captured command stdout.

    Returns
    -------
    payload : Dict[str, Any]
        Parsed final JSON object.

    Raises
    ------
    AssertionError
        Raised when stdout lacks a JSON object line.

    Notes
    -----
    Drops empty lines before parsing. The command contract reserves the final
    line for the result object.
    """
    lines: List[str] = [line for line in stdout.splitlines() if line]
    assert lines
    decoded: Any = json.loads(lines[-1])
    assert isinstance(decoded, dict)
    payload: Dict[str, Any] = decoded
    return payload


def _schema_errors(  # noqa: PLR0912
    value: Any,
    schema: Dict[str, Any],
    location: str = "$",
) -> List[str]:
    """PRIVATE: Validate a JSON value against the committed schema subset.

    Parameters
    ----------
    value : Any
        Value to validate.
    schema : Dict[str, Any]
        JSON Schema object with primitive constraints.
    location : str
        Display path for reported failures. Default ``"$"``.

    Returns
    -------
    errors : List[str]
        Human-readable validation failures.

    Notes
    -----
    Checks required keys, primitive types, arrays, constants, enums, and
    string constraints. The committed schemas intentionally use this subset.
    """
    errors: List[str] = []
    expected_type: Any = schema.get("type")
    expected_types: Tuple[str, ...] = (
        (expected_type,)
        if isinstance(expected_type, str)
        else tuple(expected_type)
        if isinstance(expected_type, list)
        else ()
    )
    type_matches: Dict[str, bool] = {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, int | float)
        and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }
    if expected_types and not any(
        type_matches.get(name, False) for name in expected_types
    ):
        errors.append(f"{location}: expected type {expected_types}")
        return errors
    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
    allowed_values: Any = schema.get("enum")
    if isinstance(allowed_values, list) and value not in allowed_values:
        errors.append(f"{location}: expected one allowed value")
    if isinstance(value, str):
        minimum_length: Any = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{location}: string is too short")
        pattern: Any = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            errors.append(f"{location}: string misses the required pattern")
    if isinstance(value, list):
        minimum_items: Any = schema.get("minItems")
        maximum_items: Any = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{location}: array has too few items")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            errors.append(f"{location}: array has too many items")
        item_schema: Any = schema.get("items")
        if isinstance(item_schema, dict):
            index: int
            item: Any
            for index, item in enumerate(value):
                errors.extend(
                    _schema_errors(item, item_schema, f"{location}[{index}]")
                )
    if isinstance(value, dict):
        required: Any = schema.get("required", [])
        required_name: Any
        for required_name in required:
            if isinstance(required_name, str) and required_name not in value:
                errors.append(f"{location}: missing {required_name}")
        properties: Any = schema.get("properties", {})
        if isinstance(properties, dict):
            property_name: Any
            property_schema: Any
            for property_name, property_schema in properties.items():
                if property_name in value and isinstance(
                    property_schema, dict
                ):
                    errors.extend(
                        _schema_errors(
                            value[property_name],
                            property_schema,
                            f"{location}.{property_name}",
                        )
                    )
            if schema.get("additionalProperties") is False:
                unexpected_names: Set[str] = set(value) - set(properties)
                if unexpected_names:
                    errors.append(
                        f"{location}: unexpected {sorted(unexpected_names)}"
                    )
    return errors


def _run_command(
    arguments: Sequence[str], timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    """PRIVATE: Run one automaton command with the CPU environment.

    Parameters
    ----------
    arguments : Sequence[str]
        Executable path and command arguments.
    timeout : float
        Maximum command duration in seconds. Default 300.0.

    Returns
    -------
    completed : subprocess.CompletedProcess[str]
        Captured command completion record.

    Notes
    -----
    Sets the CPU platform and a temporary Matplotlib directory. Captures text
    streams so each assertion can print a concise failure record.
    """
    environment: Dict[str, str] = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MPLCONFIGDIR"] = "/tmp/dp-mpl"  # noqa: S108
    completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        list(arguments),
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        timeout=timeout,
    )
    return completed


def _document_filenames(path: Path) -> Set[str]:
    """PRIVATE: Collect backticked Python filenames from one catalog document.

    Parameters
    ----------
    path : Path
        Markdown catalog path.

    Returns
    -------
    filenames : Set[str]
        Filenames named inside backticks.

    Notes
    -----
    Matches direct Python names only. Links and prose do not change the
    catalog comparison.
    """
    document: str = path.read_text(encoding="utf-8")
    filenames: Set[str] = set(re.findall(r"`([a-z_]+\.py)`", document))
    return filenames


class TestAutomatonContract:
    """Validate the automaton catalog and process-boundary contract.

    The case checks static repository records first. It then exercises every
    experiment file that exists in a fresh collection session.
    """

    def test_catalog_documents_list_the_frozen_filenames(self) -> None:
        """Keep both catalog documents aligned with the complete filename set.

        The test compares backticked Python names against the frozen catalog.
        It checks both agent-facing documents before experiment files exist.

        Notes
        -----
        Reads the README and index as UTF-8 text. Set equality rejects missing
        entries, duplicate spelling variants, and undocumented additions.
        """
        expected: Set[str] = set(EXPECTED_FILENAMES)
        readme_names: Set[str] = _document_filenames(
            AUTOMATON_DIRECTORY / "README.md"
        )
        index_names: Set[str] = _document_filenames(
            AUTOMATON_DIRECTORY / "INDEX.md"
        )

        assert readme_names == expected
        assert index_names == expected

    def test_schemas_declare_the_contract_envelopes(self) -> None:
        """Keep committed schemas on the Draft 2020-12 contract format.

        The test checks required envelope fields for descriptions and results.
        It provides static coverage before the first experiment script exists.

        Notes
        -----
        Parses both documents with the standard library. Required-field sets
        describe the minimum payload that every runner must emit.
        """
        parameter_schema: Dict[str, Any] = _load_document(
            PARAMETER_SCHEMA_PATH
        )
        result_schema: Dict[str, Any] = _load_document(RESULT_SCHEMA_PATH)
        draft_url: str = "https://json-schema.org/draft/2020-12/schema"
        description_fields: Set[str] = {
            "schema_version",
            "experiment",
            "summary",
            "params_schema",
            "returns",
            "inherited_flags",
        }
        result_fields: Set[str] = {
            "schema_version",
            "status",
            "experiment",
            "diffpes_version",
            "jax_backend",
            "seed",
            "params",
            "metrics",
            "artifacts",
            "wall_seconds",
            "returns",
            "result_key",
        }

        assert parameter_schema["$schema"] == draft_url
        assert result_schema["$schema"] == draft_url
        assert description_fields <= set(parameter_schema["required"])
        assert result_fields <= set(result_schema["required"])

    def test_existing_experiment_files_match_the_catalog(self) -> None:
        """Reject an undocumented experiment file after the catalog arrives.

        The test permits an empty experiment directory during initial setup.
        It requires an exact catalog match once experiment files appear.

        Notes
        -----
        Lists only direct Python files. The pin maintenance tool remains part
        of the catalog but does not implement an experiment body.
        """
        scripts: Tuple[Path, ...] = _experiment_scripts()
        observed: Set[str] = {path.name for path in scripts}
        expected: Set[str] = set(EXPECTED_FILENAMES) - {"bump_pin.py"}

        if observed:
            assert observed == expected

    def test_existing_experiment_headers_pins_and_imports_match_contract(
        self,
    ) -> None:
        """Keep each existing experiment header, pin, and import canonical.

        The test accepts the empty initial directory. It checks every later
        experiment's four-line header, pin, and one permitted diffpes import.

        Notes
        -----
        Reads the project TOML and direct script text. Regex checks reject
        stale pins, duplicate pins, deep imports, and renamed package imports.
        """
        configuration: Dict[str, Any] = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        version: str = configuration["project"]["version"]
        expected_header: Tuple[str, ...] = (
            "# /// script",
            '# requires-python = ">=3.12,<3.15"',
            f'# dependencies = ["diffpes=={version}"]',
            "# ///",
        )
        script_path: Path
        for script_path in _experiment_scripts():
            source: str = script_path.read_text(encoding="utf-8")
            pins: List[str] = re.findall(
                r'diffpes(?:\[cuda\])?==([0-9][^"\']*)', source
            )
            canonical_imports: List[str] = re.findall(
                r"^import diffpes as dp$", source, flags=re.MULTILINE
            )
            diffpes_imports: List[str] = re.findall(
                r"^\s*(?:from\s+diffpes(?:\b|\.)[^\n]*|"
                r"import\s+diffpes(?:\b|\.)[^\n]*)$",
                source,
                flags=re.MULTILINE,
            )
            from_imports: List[str] = re.findall(
                r"^\s*from\s+diffpes(?:\b|\.)", source, flags=re.MULTILINE
            )
            submodule_imports: List[str] = re.findall(
                r"^\s*import\s+diffpes\.", source, flags=re.MULTILINE
            )

            assert tuple(source.splitlines()[:4]) == expected_header
            assert pins == [version]
            assert canonical_imports == ["import diffpes as dp"]
            assert len(diffpes_imports) == 1
            assert not from_imports
            assert not submodule_imports

    def test_bump_pin_rewrites_and_repeats_without_a_diff(
        self, tmp_path: Path
    ) -> None:
        """Keep the pin maintenance command deterministic and idempotent.

        The test builds a disposable project with an old pin. It runs the
        command twice and compares the script text after each completion.

        Notes
        -----
        Uses the active virtual-environment interpreter and CPU environment.
        The fixture stays below pytest's temporary directory.
        """
        project_file: Path = tmp_path / "pyproject.toml"
        scripts_directory: Path = tmp_path / "automatons"
        script_file: Path = scripts_directory / "sample.py"
        tool_path: Path = AUTOMATON_DIRECTORY / "bump_pin.py"
        project_file.write_text(
            '[project]\nversion = "2026.06.13"\n', encoding="utf-8"
        )
        scripts_directory.mkdir()
        script_file.write_text(
            '# /// script\n# dependencies = ["diffpes==0.0.0"]\n# ///\n',
            encoding="utf-8",
        )
        first: subprocess.CompletedProcess[str] = _run_command(
            [sys.executable, str(tool_path), "--root", str(tmp_path)]
        )
        first_text: str = script_file.read_text(encoding="utf-8")
        second: subprocess.CompletedProcess[str] = _run_command(
            [sys.executable, str(tool_path), "--root", str(tmp_path)]
        )
        second_text: str = script_file.read_text(encoding="utf-8")

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert "updated 1 file(s)" in first.stdout
        assert "updated 0 file(s)" in second.stdout
        assert "diffpes==2026.06.13" in first_text
        assert second_text == first_text

    def test_wheel_excludes_the_automaton_directory(
        self, tmp_path: Path
    ) -> None:
        """Keep standalone experiments outside the built package wheel.

        The test builds a wheel when the local virtual environment exposes uv.
        It then lists archive members and rejects an automaton path.

        Notes
        -----
        Skips only when the active interpreter lacks the local build module.
        The release command performs the same archive inspection separately.
        """
        if importlib.util.find_spec("uv") is None:
            pytest.skip("The active virtual environment does not expose uv.")
        output_directory: Path = tmp_path / "wheel"
        completed: subprocess.CompletedProcess[str] = _run_command(
            [
                sys.executable,
                "-m",
                "uv",
                "build",
                "--wheel",
                "-o",
                str(output_directory),
            ],
            timeout=300.0,
        )
        wheel_paths: List[Path] = list(output_directory.glob("*.whl"))
        wheel_path: Path
        archive_names: List[str]
        archive: zipfile.ZipFile

        assert completed.returncode == 0, completed.stderr
        assert len(wheel_paths) == 1
        wheel_path = wheel_paths[0]
        with zipfile.ZipFile(wheel_path) as archive:
            archive_names = archive.namelist()
        assert not any(
            name.startswith("automatons/") for name in archive_names
        )

    def test_ci_defines_a_blocking_cpu_smoke_job(self) -> None:
        """Keep continuous integration responsible for CPU smoke experiments.

        The test checks the dedicated job, its CPU environment, and its loop.
        It rejects a permissive failure setting in the job declaration.

        Notes
        -----
        Parses the checked-in YAML document. The assertions inspect static CI
        configuration without contacting an external service.
        """
        workflow_path: Path = (
            REPOSITORY_ROOT / ".github" / "workflows" / "tests.yml"
        )
        workflow: Dict[str, Any] = yaml.safe_load(
            workflow_path.read_text(encoding="utf-8")
        )
        jobs: Dict[str, Any] = workflow["jobs"]
        smoke_job: Dict[str, Any] = jobs["automatons-smoke"]
        environment: Dict[str, Any] = smoke_job["env"]
        steps: List[Dict[str, Any]] = smoke_job["steps"]
        commands: List[str] = [step["run"] for step in steps if "run" in step]
        command_text: str = "\n".join(commands)

        assert environment["JAX_PLATFORMS"] == "cpu"
        assert "continue-on-error" not in smoke_job
        assert '"$script" --smoke' in command_text
        assert "bump_pin.py" in command_text

    @pytest.mark.parametrize(
        "script_path",
        _experiment_scripts(),
        ids=lambda script_path: script_path.stem,
    )
    def test_describe_validate_and_smoke(
        self,
        script_path: Path,
        tmp_path: Path,
    ) -> None:
        """Require every existing experiment to honor the command protocol.

        The test invokes description, validation, and smoke modes for one file.
        It validates JSON envelopes and checks declared artifacts on disk.

        Notes
        -----
        Runs each file through the active interpreter with the CPU environment.
        A fresh collection includes every script added to the directory.
        """
        parameter_schema: Dict[str, Any] = _load_document(
            PARAMETER_SCHEMA_PATH
        )
        result_schema: Dict[str, Any] = _load_document(RESULT_SCHEMA_PATH)
        describe: subprocess.CompletedProcess[str] = _run_command(
            [sys.executable, str(script_path), "--describe", "--json"]
        )
        describe_payload: Dict[str, Any] = _last_json(describe.stdout)
        validate: subprocess.CompletedProcess[str] = _run_command(
            [sys.executable, str(script_path), "--validate", "--json"]
        )
        validate_payload: Dict[str, Any] = _last_json(validate.stdout)
        output_directory: Path = tmp_path / script_path.stem
        smoke: subprocess.CompletedProcess[str] = _run_command(
            [
                sys.executable,
                str(script_path),
                "--smoke",
                "--seed",
                "123",
                "--outdir",
                str(output_directory),
                "--json",
            ]
        )
        smoke_payload: Dict[str, Any] = _last_json(smoke.stdout)
        description_errors: List[str] = _schema_errors(
            describe_payload, parameter_schema
        )
        result_errors: List[str] = _schema_errors(smoke_payload, result_schema)
        returns: Dict[str, Any] = describe_payload["returns"]
        artifact_specification: Dict[str, Any] = returns.get("artifacts", {})
        declared_roles: Set[str] = set(artifact_specification.get("roles", []))
        artifacts: List[Dict[str, Any]] = smoke_payload["artifacts"]
        observed_roles: Set[str] = {artifact["role"] for artifact in artifacts}
        artifact: Dict[str, Any]
        artifact_path: Path

        assert describe.returncode == 0, describe.stderr
        assert validate.returncode == 0, validate.stderr
        assert smoke.returncode == 0, smoke.stderr
        assert not description_errors
        assert describe_payload["params_schema"]["type"] == "object"
        assert describe_payload["params_schema"]["properties"]
        assert validate_payload["metrics"]["valid"] is True
        assert smoke_payload["status"] == "ok"
        assert not result_errors
        assert smoke_payload["wall_seconds"] <= 60.0
        assert declared_roles <= observed_roles
        for artifact in artifacts:
            artifact_path = output_directory / artifact["path"]
            assert artifact_path.is_file()
            assert artifact_path.resolve().is_relative_to(
                output_directory.resolve()
            )
        assert (output_directory / "metrics.json").is_file()
