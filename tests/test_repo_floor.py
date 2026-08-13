"""Verify the repository dependency and runtime foundation.

The module verifies the dependencies for the differentiable solver stack.
It confirms that diffpes enables JAX 64-bit precision before numerical work.
It also checks Equinox module structure through JAX PyTree operations.
"""

import ast
import hashlib
import re
import tomllib
from pathlib import Path

# ruff: noqa: I001 -- diffpes must configure JAX before stack imports.
import diffpes  # noqa: F401 -- configures JAX before numerical imports.

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optax
import optimistix
import pytest
import yaml
import numpy as np
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped

from tests._assertions import (
    assert_tree_finite,
    assert_trees_close,
)
from tests._factories import (
    toy_band_structure,
    toy_chain_diagonalized,
    toy_graphene_diagonalized,
    toy_orbital_projection,
)
from diffpes.types import PyTreeDef
from diffpes.types import (
    BandStructure,
    DiagonalizedBands,
    OrbitalProjection,
    TBModel,
)


class TestConftest:
    """Validate the shared pytest numerical and resource contracts.

    The class covers the x64 session invariant and stable node-derived random
    keys. It also covers RSS leak failures in an isolated pytest subprocess.
    """

    def test_x64_and_rng_key(
        self,
        request: pytest.FixtureRequest,
        rng_key: PRNGKeyArray,
    ) -> None:
        """Keep x64 precision and random keys stable across workers.

        The test confirms the default scalar dtype is float64. It independently
        derives the expected SHA-256 seed to verify the fixture key.

        Notes
        -----
        The test hashes the fully qualified pytest node ID. It converts its
        first four bytes to an integer and compares the typed JAX key exactly.
        """
        precision_probe: Float[Array, ""] = jnp.zeros(())
        digest: bytes = hashlib.sha256(request.node.nodeid.encode()).digest()
        expected_seed: int = int.from_bytes(digest[:4], byteorder="big")
        expected_key: PRNGKeyArray = jax.random.key(expected_seed)

        chex.assert_equal(precision_probe.dtype, jnp.float64)
        chex.assert_trees_all_equal(rng_key, expected_key)

    def test_rss_leak_guard_trips(self, pytester: pytest.Pytester) -> None:
        """Reject a retained allocation larger than its marked RSS limit.

        The test confirms the real plugin reports a teardown error for retained
        memory above 100 MiB. It exercises the actual resource measurement.

        Notes
        -----
        The test copies the repository conftest into an isolated subprocess.
        It touches a retained 160 MiB byte array page-by-page. The test
        requires the measured-RSS diagnostic and teardown error.
        """
        conftest_path: Path = Path(__file__).with_name("conftest.py")
        pytester.makeconftest(conftest_path.read_text())
        pytester.makepyfile(
            """
            import pytest

            _RETAINED = []

            @pytest.mark.rss_limit_mb(100)
            def test_retained_allocation():
                allocation = bytearray(160 * 1024 * 1024)
                for offset in range(0, len(allocation), 4096):
                    allocation[offset] = 1
                _RETAINED.append(allocation)
            """
        )
        result: Any = pytester.runpytest_subprocess(
            "-q",
            "-n",
            "0",
        )

        result.assert_outcomes(passed=1, errors=1)
        result.stdout.fnmatch_lines(["*retained*MiB RSS*limit is 100.0 MiB*"])

    def test_chinook_import_firewall_trips(
        self,
        pytester: pytest.Pytester,
    ) -> None:
        """Reject even an import-or-skip request for Chinook.

        The case proves that ``pytest.importorskip`` cannot turn the runtime
        firewall into a passing skip.

        Notes
        -----
        Run an isolated session with the repository conftest. Require its
        planted Chinook request to fail with the boundary diagnostic.
        """
        conftest_path: Path = Path(__file__).with_name("conftest.py")
        pytester.makeconftest(conftest_path.read_text())
        pytester.makepyfile(
            """
            import pytest

            def test_forbidden_reference_import():
                pytest.importorskip("chinook")
            """
        )
        result: Any = pytester.runpytest_subprocess(
            "-q",
            "-n",
            "0",
        )

        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(
            ["*tests must consume frozen Chinook artifacts*forbidden*"]
        )


class TestHelpers:
    """Validate deterministic shared factories and assertion wrappers.

    The class covers each shared factory's carrier, shape, and finite leaves.
    It also covers fixed-seed reproducibility with strict shared assertions.
    """

    def test_factories_and_assertions(self, rng_key: PRNGKeyArray) -> None:
        """Build finite, correctly shaped, reproducible toy carriers.

        The test confirms that all seven factories return their declared
        production types. Random factories repeat bit-for-bit for one key.
        Analytic tight-binding paths expose the requested number of k-points.

        Notes
        -----
        The test builds reduced-size carriers and checks dimensions with Chex.
        It verifies every leaf is finite. The test compares repeated random
        trees at zero relative and absolute tolerance.
        """
        bands: BandStructure = toy_band_structure(rng_key, n_k=5, n_bands=3)
        repeated_bands: BandStructure = toy_band_structure(
            rng_key,
            n_k=5,
            n_bands=3,
        )
        projections: OrbitalProjection = toy_orbital_projection(
            rng_key,
            n_k=5,
            n_bands=3,
            n_atoms=2,
        )
        repeated_projections: OrbitalProjection = toy_orbital_projection(
            rng_key,
            n_k=5,
            n_bands=3,
            n_atoms=2,
        )
        graphene_model: TBModel
        graphene_bands: DiagonalizedBands
        graphene_model, graphene_bands = toy_graphene_diagonalized(n_k=6)
        chain_model: TBModel
        chain_bands: DiagonalizedBands
        chain_model, chain_bands = toy_chain_diagonalized(n_k=7)
        all_carriers: Tuple[object, ...] = (
            bands,
            projections,
            graphene_model,
            graphene_bands,
            chain_model,
            chain_bands,
        )

        assert isinstance(bands, BandStructure)
        assert isinstance(projections, OrbitalProjection)
        assert isinstance(graphene_model, TBModel)
        assert isinstance(graphene_bands, DiagonalizedBands)
        assert isinstance(chain_model, TBModel)
        assert isinstance(chain_bands, DiagonalizedBands)
        chex.assert_shape(bands.eigenvalues, (5, 3))
        chex.assert_shape(projections.projections, (5, 3, 2, 9))
        chex.assert_shape(graphene_bands.kpoints, (6, 3))
        chex.assert_shape(chain_bands.kpoints, (7, 3))
        assert_tree_finite(all_carriers)
        assert_trees_close(bands, repeated_bands, rtol=0.0, atol=0.0)
        assert_trees_close(
            projections,
            repeated_projections,
            rtol=0.0,
            atol=0.0,
        )


class TestMetadata(chex.TestCase):
    """Validate the install, tooling, and Python metadata contract.

    The class covers standalone dependency purity and unconditional JAX
    installation. It also covers supported Python versions and tooling scope.
    """

    def test_project_metadata(self) -> None:
        """Keep project metadata consistent with the standalone test floor.

        The test confirms that retired dependencies and configuration are
        absent. JAX has one unconditional runtime constraint. The project
        supports Python 3.12 and declares program-wide tooling settings.

        Notes
        -----
        The test parses ``pyproject.toml`` with the standard-library TOML
        reader. It compares its values with the repository metadata contract.
        """
        project_file: Path = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        )
        configuration: Dict[str, Any] = tomllib.loads(project_file.read_text())
        project: Dict[str, Any] = configuration["project"]
        runtime_dependencies: List[str] = project["dependencies"]
        optional_dependencies: Dict[str, List[str]] = project[
            "optional-dependencies"
        ]
        dependency_groups: Tuple[str, ...] = tuple(
            runtime_dependencies
        ) + tuple(
            dependency
            for group in optional_dependencies.values()
            for dependency in group
        )
        retired_names: Tuple[str, ...] = (
            "diff" + "tb",
            "chinook",
            "black",
            "isort",
            "twine",
        )
        jax_constraints: List[str] = [
            dependency
            for dependency in runtime_dependencies
            if re.match(r"^jax(?:\[|[<>=!~]|$)", dependency) is not None
        ]
        tool_configuration: Dict[str, Any] = configuration["tool"]
        pytest_options: Dict[str, Any] = tool_configuration["pytest"][
            "ini_options"
        ]
        ruff_configuration: Dict[str, Any] = tool_configuration["ruff"]
        ruff_lint_configuration: Dict[str, Any] = ruff_configuration["lint"]
        ruff_per_file_ignores: Dict[str, List[str]] = ruff_lint_configuration[
            "per-file-ignores"
        ]
        interrogate_configuration: Dict[str, Any] = tool_configuration[
            "interrogate"
        ]

        self.assertFalse(
            any(
                retired_name in dependency.lower()
                for retired_name in retired_names
                for dependency in dependency_groups
            )
        )
        self.assertEqual(jax_constraints, ["jax>=0.7.0"])
        self.assertTrue(project["requires-python"].startswith(">="))
        self.assertIn(
            "Programming Language :: Python :: 3.12",
            project["classifiers"],
        )
        self.assertNotIn("setuptools", tool_configuration)
        self.assertNotIn("style", interrogate_configuration)
        self.assertIn("tests/**/*.py", ruff_configuration["include"])
        self.assertEqual(
            ruff_per_file_ignores,
            {
                "tests/**/*.py": [
                    "S101",
                    "RET504",
                    "PLR2004",
                    "PT009",
                    "ARG001",
                    "E741",
                    "N803",
                    "N806",
                ],
                "src/diffpes/maths/*.py": ["E741", "N803", "N806"],
                "src/diffpes/tightb/*.py": ["E741", "N803", "N806"],
            },
        )
        self.assertEqual(
            pytest_options["addopts"],
            "-n auto --dist loadgroup "
            "--jaxtyping-packages=diffpes,beartype.beartype",
        )


class TestCI(chex.TestCase):
    """Validate the continuous-integration workflow.

    The class covers workflow syntax and repository triggers. It also covers
    the complete supported-Python matrix declared by package metadata.
    """

    def test_workflow_matrix(self) -> None:  # noqa: PLR0915
        """Exercise CI on every supported Python minor version.

        The test confirms the workflow exists and parses as YAML. It runs for
        pushes and pull requests. It tests Python 3.12, 3.13, and 3.14 exactly.

        Notes
        -----
        The test loads the checked-in workflow with PyYAML. It compares the
        triggers and test matrix with the external configuration truth.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        workflow_path: Path = repository_root / ".github/workflows/tests.yml"
        workflow: Dict[str, Any] = yaml.safe_load(workflow_path.read_text())
        triggers: List[str] = workflow["on"]
        python_versions: List[str] = workflow["jobs"]["test"]["strategy"][
            "matrix"
        ]["python-version"]

        self.assertTrue(workflow_path.is_file())
        chex.assert_equal(
            triggers,
            ["push", "pull_request", "workflow_dispatch"],
        )
        chex.assert_equal(python_versions, ["3.12", "3.13", "3.14"])
        documentation_path: Path = (
            repository_root / ".github/workflows/documentation.yml"
        )
        documentation: Dict[str, Any] = yaml.safe_load(
            documentation_path.read_text()
        )
        docs_job: Dict[str, Any] = documentation["jobs"]["docs"]
        docs_commands: Tuple[str, ...] = tuple(
            step["run"] for step in docs_job["steps"] if "run" in step
        )
        self.assertEqual(documentation["on"], triggers)
        self.assertIn(
            "uv sync --frozen --extra docs --extra test --extra notebooks",
            docs_commands,
        )
        self.assertTrue(
            any(
                "sphinx-build -W -a -E --keep-going -b html" in command
                and "docs/source docs/build/html" in command
                for command in docs_commands
            )
        )
        tutorial_check_command: str = next(
            command
            for command in docs_commands
            if "python tests/_tutorials.py" in command
        )
        self.assertNotIn("jupytext", tutorial_check_command)
        export_command: str = next(
            command
            for command in docs_commands
            if "jupyter nbconvert" in command
        )
        self.assertIn("tutorials/*.ipynb", export_command)
        self.assertIn("--to markdown --execute", export_command)
        self.assertIn("--output-dir docs/source/tutorials", export_command)
        self.assertIn("git diff --exit-code", export_command)
        cache_steps: List[Dict[str, Any]] = [
            step
            for step in docs_job["steps"]
            if step.get("uses") == "actions/cache@v4"
        ]
        self.assertEqual(cache_steps, [])

        conf_path: Path = repository_root / "docs/source/conf.py"
        conf_tree: ast.Module = ast.parse(conf_path.read_text())
        assignments: Dict[str, Any] = {}
        node: ast.stmt
        for node in conf_tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target: ast.expr = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
        self.assertNotIn("myst_nb", assignments["extensions"])
        self.assertIn("myst_parser", assignments["extensions"])
        self.assertEqual(assignments["source_suffix"][".md"], "markdown")
        self.assertNotIn(".ipynb", assignments["source_suffix"])
        self.assertFalse(
            any(name.startswith("nb_execution_") for name in assignments)
        )

        rtd_path: Path = repository_root / ".readthedocs.yaml"
        rtd: Dict[str, Any] = yaml.safe_load(rtd_path.read_text())
        self.assertIs(rtd["sphinx"]["fail_on_warning"], True)

        tutorial_paths: List[Path] = sorted(
            (repository_root / "docs/source/tutorials").glob("*.md")
        )
        tutorial_path: Path
        for tutorial_path in tutorial_paths:
            tutorial_text: str = tutorial_path.read_text()
            self.assertNotIn("kernelspec:", tutorial_text)
            self.assertNotIn("```{code-cell}", tutorial_text)

        matrix_tutorial: Path = (
            repository_root
            / "docs/source/tutorials/matrix-element-sensitivity.md"
        )
        matrix_tutorial_text: str = matrix_tutorial.read_text()
        self.assertGreaterEqual(matrix_tutorial_text.count("```python"), 4)

    def test_pypi_release_workflow(self) -> None:
        """Publish matching version tags through trusted PyPI identity.

        The test confirms the dedicated release workflow accepts only tags. The
        workflow uses the protected ``pypi`` environment with job-scoped OIDC
        permission. It smoke-tests both distribution formats and requires uv
        trusted publishing.

        Notes
        -----
        The test parses the workflow as YAML. It inspects triggers,
        permissions, and commands without contacting PyPI.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        workflow_path: Path = repository_root / ".github/workflows/release.yml"
        workflow: Dict[str, Any] = yaml.safe_load(workflow_path.read_text())
        publish_job: Dict[str, Any] = workflow["jobs"]["publish"]
        run_commands: Tuple[str, ...] = tuple(
            step["run"] for step in publish_job["steps"] if "run" in step
        )
        combined_commands: str = "\n".join(run_commands)

        chex.assert_equal(workflow["on"]["push"]["tags"], ["v*"])
        chex.assert_equal(publish_job["environment"]["name"], "pypi")
        chex.assert_equal(
            publish_job["permissions"],
            {"contents": "read", "id-token": "write"},
        )
        self.assertIn("uv build --no-sources", combined_commands)
        self.assertIn("--with dist/*.whl", combined_commands)
        self.assertIn("--with dist/*.tar.gz", combined_commands)
        self.assertIn(
            "uv publish --trusted-publishing always dist/*",
            combined_commands,
        )


class TestRegressionReferences(chex.TestCase):
    """Validate historical novice artifact integrity after tier removal.

    The class keeps the frozen true- and pseudo-Voigt archives as provenance
    evidence without replaying either deleted production assembler.
    """

    def test_historical_artifacts_match_manifest(self) -> None:
        """Match historical archive bytes and array metadata to the manifest.

        The check deliberately makes no behavioral claim for a live forward
        path. It preserves both retired archives as immutable evidence.

        Notes
        -----
        Each archive must retain its recorded digest, float64 arrays, and
        pickle-free loading contract.
        """
        reference_directory: Path = (
            Path(__file__).parent / "test_diffpes" / "_reference_data"
        )
        manifest: str = (reference_directory / "MANIFEST.md").read_text()

        artifact_name: str
        for artifact_name in (
            "novice_toy_true_voigt",
            "novice_toy_pseudo_voigt",
        ):
            artifact_path: Path = reference_directory / f"{artifact_name}.npz"
            digest: str = hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest()
            self.assertIn(f"`{digest}`", manifest)
            archive: Any
            with np.load(artifact_path, allow_pickle=False) as archive:
                self.assertTrue(
                    all(
                        array.dtype == np.float64 for array in archive.values()
                    )
                )


class TestRepositoryArchitecture(chex.TestCase):
    """Enforce the production architecture rules from CONTRIBUTING.

    The class covers carrier and factory ownership, import boundaries, and
    runtime type checking. It covers explicit returns and package listings.
    """

    def test_reference_tools_do_not_use_root_scripts_directory(self) -> None:
        """Keep reproducibility tooling under the test evidence boundary.

        The test requires key generators under ``tests/_reference_tools``.

        Notes
        -----
        It also rejects the retired repository-root ``scripts`` directory.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        reference_tools: Path = repository_root / "tests/_reference_tools"

        self.assertFalse((repository_root / "scripts").exists())
        self.assertTrue(reference_tools.is_dir())
        self.assertTrue(
            (reference_tools / "generate_regression_references.py").is_file()
        )
        self.assertTrue(
            (reference_tools / "verify_coulomb_reference.py").is_file()
        )

    def test_detector_zero_legacy_surface_is_absent(self) -> None:
        """Keep retired tier and parameter APIs out of live source.

        The witness checks exact deleted module paths, every former six-tier
        assembler and expanded dispatcher symbol, ``SimulationParams``
        consumers, and live tier identifiers. Fidelity metadata remains valid
        only in its manifest carrier and factorized evaluator. The witness
        inspects production ASTs and literal public exports.

        Notes
        -----
        Live production syntax forbids exact string-tier literals only.
        Docstrings can record the legacy-surface removal.
        """
        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        deleted_paths: Tuple[Path, ...] = (
            Path("simul/expanded.py"),
            Path("simul/forward.py"),
            Path("types/params.py"),
        )
        forbidden_symbols: frozenset[str] = frozenset(
            {
                "SimulationParams",
                "apply_momentum_broadening",
                "make_expanded_simulation_params",
                "make_simulation_params",
                "simulate_advanced",
                "simulate_advanced_expanded",
                "simulate_basic",
                "simulate_basic_expanded",
                "simulate_basicplus",
                "simulate_basicplus_expanded",
                "simulate_context",
                "simulate_expanded",
                "simulate_expert",
                "simulate_expert_expanded",
                "simulate_novice",
                "simulate_novice_expanded",
                "simulate_soc",
                "simulate_soc_expanded",
                "simulate_tb_radial",
            }
        )
        tier_literals: frozenset[str] = frozenset(
            {"advanced", "basic", "basicplus", "expert", "novice", "soc"}
        )
        fidelity_paths: frozenset[Path] = frozenset(
            {Path("simul/factorized.py"), Path("types/result.py")}
        )
        violations: set[str] = {
            f"live deleted path: {relative_path}"
            for relative_path in deleted_paths
            if (source_root / relative_path).exists()
        }
        path: Path
        module: ast.Module
        for path, module in self._production_modules():
            relative_path: Path = path.relative_to(source_root)
            stale_exports: set[str] = (
                self._literal_exports(module) & forbidden_symbols
            )
            if stale_exports:
                violations.add(
                    f"{relative_path}: stale public exports "
                    f"{sorted(stale_exports)}"
                )
            node: ast.AST
            for node in ast.walk(module):
                symbol: str | None = None
                if isinstance(node, ast.Name):
                    symbol = node.id
                elif isinstance(node, ast.Attribute):
                    symbol = node.attr
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    symbol = node.name
                elif isinstance(node, ast.arg | ast.keyword):
                    symbol = node.arg
                elif isinstance(node, ast.alias):
                    symbol = (
                        node.asname or node.name.rsplit(".", maxsplit=1)[-1]
                    )
                forbidden_dispatch_name: bool = symbol == "tier" or (
                    symbol == "fidelity"
                    and relative_path not in fidelity_paths
                )
                if symbol in forbidden_symbols or forbidden_dispatch_name:
                    violations.add(
                        f"{relative_path}:{getattr(node, 'lineno', 0)}:"
                        f"retired symbol {symbol}"
                    )
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and (
                        node.value in forbidden_symbols
                        or node.value in tier_literals
                        or node.value == "tier"
                        or (
                            node.value == "fidelity"
                            and relative_path not in fidelity_paths
                        )
                    )
                ):
                    violations.add(
                        f"{relative_path}:{node.lineno}:"
                        f"retired dispatch literal {node.value}"
                    )
        self.assertEqual(sorted(violations), [])

    @staticmethod
    def _production_modules() -> Tuple[Tuple[Path, ast.Module], ...]:
        """PRIVATE: Parse production Python modules in deterministic order.

        Returns
        -------
        modules : Tuple[Tuple[Path, ast.Module], ...]
            Pairs of source path and parsed tree for every file under
            ``src/diffpes``, in sorted path order.

        Notes
        -----
        Reads each file as UTF-8 text and parses it with
        ``ast.parse``, so every architecture check walks one shared
        representation.
        """
        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        modules: Tuple[Tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(source_root.rglob("*.py"))
        )
        return modules

    @staticmethod
    def _test_modules() -> Tuple[Tuple[Path, ast.Module], ...]:
        """PRIVATE: Parse collected-test modules in deterministic order.

        Returns
        -------
        modules : Tuple[Tuple[Path, ast.Module], ...]
            Pairs of source path and parsed tree for every file under
            ``tests`` outside ``_reference_tools``, in sorted path
            order.

        Notes
        -----
        Excludes ``_reference_tools`` because those generators live
        outside the collected-test documentation contract.
        """
        test_root: Path = Path(__file__).resolve().parents[1] / "tests"
        modules: Tuple[Tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(test_root.rglob("*.py"))
            if "_reference_tools" not in path.parts
        )
        return modules

    @staticmethod
    def _test_tree_modules() -> Tuple[Tuple[Path, ast.Module], ...]:
        """PRIVATE: Parse every Python module in the complete test tree.

        Returns
        -------
        modules : Tuple[Tuple[Path, ast.Module], ...]
            Pairs of source path and parsed tree for every file under
            ``tests``, including non-collected helpers, in sorted
            path order.

        Notes
        -----
        Keeps ``_reference_tools`` in scope, so the boundary checks
        see the whole tree.
        """
        test_root: Path = Path(__file__).resolve().parents[1] / "tests"
        modules: Tuple[Tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(test_root.rglob("*.py"))
        )
        return modules

    def test_chinook_import_boundary(self) -> None:
        """Keep Chinook outside production and the complete test tree.

        The test rejects direct imports and literal dynamic imports through
        ``importlib.import_module``, ``__import__``, or
        ``pytest.importorskip``. Offline generators belong in the separate
        external verification area; DiffPES consumes only their
        immutable artifacts.

        Notes
        -----
        Parse every production and test Python file, including non-collected
        helpers under ``tests/``, and report each forbidden module reference.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        dynamic_importers: set[str] = {
            "__import__",
            "importlib.import_module",
            "pytest.importorskip",
        }
        offenders: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    imported: ast.alias
                    for imported in node.names:
                        if (
                            imported.name.split(".", maxsplit=1)[0]
                            == "chinook"
                        ):
                            offenders.append(
                                f"{path.relative_to(repository_root)}:"
                                f"{node.lineno}"
                            )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.split(".", maxsplit=1)[0] == "chinook"
                ):
                    offenders.append(
                        f"{path.relative_to(repository_root)}:{node.lineno}"
                    )
                elif isinstance(node, ast.Call) and node.args:
                    importer: str = ast.unparse(node.func)
                    requested: ast.expr = node.args[0]
                    if (
                        importer in dynamic_importers
                        and isinstance(requested, ast.Constant)
                        and isinstance(requested.value, str)
                        and requested.value.split(".", maxsplit=1)[0]
                        == "chinook"
                    ):
                        offenders.append(
                            f"{path.relative_to(repository_root)}:"
                            f"{node.lineno}"
                        )

        self.assertEqual(offenders, [])

    @staticmethod
    def _literal_exports(module: ast.Module) -> set[str]:
        """PRIVATE: Return literal names from one module-level ``__all__``.

        Parameters
        ----------
        module : ast.Module
            Parsed module to inspect.

        Returns
        -------
        exports : set[str]
            String constants inside the last ``__all__`` list or
            tuple, empty when the module declares none.

        Notes
        -----
        Accepts plain and annotated assignments to ``__all__`` and
        ignores every non-literal element.
        """
        exports: set[str] = set()
        node: ast.stmt
        for node in module.body:
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and any(  # noqa: SIM114
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            ):
                value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                exports = {
                    entry.value
                    for entry in value.elts
                    if isinstance(entry, ast.Constant)
                    and isinstance(entry.value, str)
                }
        return exports

    @staticmethod
    def _routine_listing_summaries(docstring: str) -> Dict[str, str]:
        """PRIVATE: Return names and summaries from one routine listing.

        Parameters
        ----------
        docstring : str
            Package docstring that carries the Routine Listings
            entries.

        Returns
        -------
        summaries : Dict[str, str]
            Mapping from each referenced public name to the indented
            summary line after it, or to an empty string.

        Notes
        -----
        Matches ``:func:``, ``:class:``, and ``:obj:`` roles line by
        line and strips any leading module path from the captured
        name.
        """
        summaries: Dict[str, str] = {}
        lines: List[str] = docstring.splitlines()
        index: int
        line: str
        for index, line in enumerate(lines):
            match: re.Match[str] | None = re.match(
                r":(?:func|class|obj):`(?:~[^`]*\.)?([^`]+)`",
                line.strip(),
            )
            if match is None:
                continue
            summary: str = ""
            if index + 1 < len(lines) and lines[index + 1].startswith("    "):
                summary = lines[index + 1].strip()
            summaries[match.group(1)] = summary
        return summaries

    @staticmethod
    def _markdown_prose(  # noqa: PLR0912
        path: Path,
    ) -> Tuple[Tuple[int, str], ...]:
        """PRIVATE: Return line-numbered prose blocks from one Markdown file.

        Parameters
        ----------
        path : Path
            Markdown file to segment.

        Returns
        -------
        prose : Tuple[Tuple[int, str], ...]
            Start line and joined text of every prose block, with
            table cells and list items as separate blocks.

        Notes
        -----
        Skips code fences, math blocks, front matter, headings,
        comments, directives, and horizontal rules. Splits table
        rows into cells, starts a new block at each list item, and
        strips block-quote markers before it joins continuation
        lines.
        """
        lines: List[str] = path.read_text(encoding="utf-8").splitlines()
        paragraphs: List[Tuple[int, str]] = []
        current_lines: List[str] = []
        current_start: int = 0
        in_fence: bool = False
        in_math: bool = False
        in_front_matter: bool = bool(lines and lines[0].strip() == "---")
        line_number: int
        raw_line: str
        for line_number, raw_line in enumerate(lines, start=1):
            stripped: str = raw_line.strip()
            starts_fence: bool = stripped.startswith(("```", "~~~"))
            starts_math: bool = stripped == "$$"
            ends_front_matter: bool = (
                in_front_matter and line_number > 1 and stripped == "---"
            )
            boundary: bool = (
                not stripped
                or starts_fence
                or starts_math
                or in_fence
                or in_math
                or in_front_matter
                or stripped.startswith(("#", "<!--", ":::", ":"))
                or stripped in {"---", "***", "___"}
            )
            if boundary:
                if current_lines:
                    paragraphs.append((current_start, " ".join(current_lines)))
                    current_lines = []
                if starts_fence:
                    in_fence = not in_fence
                if starts_math:
                    in_math = not in_math
                if ends_front_matter:
                    in_front_matter = False
                continue

            if stripped.startswith("|"):
                if current_lines:
                    paragraphs.append((current_start, " ".join(current_lines)))
                    current_lines = []
                table_cells: List[str] = [
                    cell.strip()
                    for cell in stripped.strip("|").split("|")
                    if cell.strip()
                    and re.fullmatch(r":?-{3,}:?", cell.strip()) is None
                ]
                paragraphs.extend((line_number, cell) for cell in table_cells)
                continue

            list_match: re.Match[str] | None = re.match(
                r"^(?:[-*+] (?:\[[ xX]\] )?|\d+\. )(.*)",
                stripped,
            )
            if list_match is not None:
                if current_lines:
                    paragraphs.append((current_start, " ".join(current_lines)))
                current_start = line_number
                current_lines = [list_match.group(1)]
                continue

            if stripped.startswith(">"):
                stripped = stripped.removeprefix(">").strip()
            if not current_lines:
                current_start = line_number
            current_lines.append(stripped)

        if current_lines:
            paragraphs.append((current_start, " ".join(current_lines)))
        prose: Tuple[Tuple[int, str], ...] = tuple(paragraphs)
        return prose

    @staticmethod
    def _markdown_sentences(paragraph: str) -> Tuple[str, ...]:
        """PRIVATE: Return normalized sentences from one Markdown prose block.

        Parameters
        ----------
        paragraph : str
            Joined prose text of one block.

        Returns
        -------
        sentences : Tuple[str, ...]
            Non-empty sentences split at terminal punctuation.

        Notes
        -----
        Drops images, keeps link text, and replaces roles, code
        spans, and inline math with a ``TECH`` placeholder. Removes
        HTML tags and emphasis markers, expands the Latin
        abbreviations, and collapses whitespace before the split.
        """
        normalized: str = re.sub(
            r"!\[[^\]]*\]\([^)]*\)",
            " ",
            paragraph,
        )
        normalized = re.sub(
            r"\[([^\]]*)\]\([^)]*\)",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            r":[A-Za-z0-9_-]+:`[^`]+`",
            " TECH ",
            normalized,
        )
        normalized = re.sub(r"`+[^`]+`+", " TECH ", normalized)
        normalized = re.sub(r"\$[^$]+\$", " TECH ", normalized)
        normalized = re.sub(r"<[^>]+>", " ", normalized)
        normalized = re.sub(r"[*_]", "", normalized)
        normalized = normalized.replace("e.g.", "for example")
        normalized = normalized.replace("i.e.", "that is")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        sentences: Tuple[str, ...] = tuple(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        )
        return sentences

    @staticmethod
    def _markdown_instruction(sentence: str) -> bool:
        """PRIVATE: Return whether one Markdown sentence gives an instruction.

        Parameters
        ----------
        sentence : str
            Normalized sentence to classify.

        Returns
        -------
        is_instruction : bool
            True when the first word is an imperative verb or
            ``you``, or when the sentence carries a directive modal.

        Notes
        -----
        Compares the lowercased first word against the frozen
        imperative list and searches for the directive modals with
        one regular expression.
        """
        imperative_verbs: frozenset[str] = frozenset(
            {
                "accept",
                "achieve",
                "accumulate",
                "add",
                "advance",
                "align",
                "allow",
                "annotate",
                "append",
                "apply",
                "assert",
                "assemble",
                "attach",
                "avoid",
                "batch",
                "bind",
                "build",
                "calculate",
                "carry",
                "cast",
                "certify",
                "check",
                "clear",
                "clone",
                "collect",
                "compare",
                "compile",
                "compose",
                "compute",
                "consider",
                "construct",
                "contain",
                "consume",
                "convolve",
                "convert",
                "create",
                "decode",
                "declare",
                "decorate",
                "define",
                "delete",
                "derive",
                "detect",
                "diagonalize",
                "differentiate",
                "discuss",
                "dispatch",
                "distinguish",
                "divide",
                "do",
                "document",
                "encode",
                "ensure",
                "enforce",
                "estimate",
                "evaluate",
                "execute",
                "exercise",
                "exclude",
                "expand",
                "explain",
                "expose",
                "export",
                "extract",
                "fail",
                "find",
                "flatten",
                "follow",
                "forbid",
                "format",
                "freeze",
                "generate",
                "give",
                "guard",
                "identify",
                "import",
                "include",
                "install",
                "integrate",
                "interleave",
                "interpolate",
                "keep",
                "list",
                "limit",
                "load",
                "look",
                "make",
                "map",
                "mark",
                "match",
                "materialize",
                "measure",
                "mirror",
                "name",
                "never",
                "note",
                "normalize",
                "omit",
                "open",
                "pack",
                "pair",
                "parse",
                "pass",
                "perform",
                "persist",
                "pin",
                "place",
                "plan",
                "plant",
                "plot",
                "prefer",
                "prepare",
                "prevent",
                "preserve",
                "produce",
                "promote",
                "propagate",
                "provide",
                "publish",
                "raise",
                "ravel",
                "read",
                "rebuild",
                "recompute",
                "record",
                "recover",
                "reduce",
                "re-evaluate",
                "re-execute",
                "refuse",
                "register",
                "reject",
                "remove",
                "render",
                "replace",
                "repeat",
                "report",
                "replay",
                "represent",
                "reproduce",
                "require",
                "resolve",
                "retain",
                "return",
                "reuse",
                "round",
                "round-trip",
                "rotate",
                "run",
                "sanitize",
                "sample",
                "save",
                "scale",
                "select",
                "serialize",
                "set",
                "show",
                "simulate",
                "skip",
                "stage",
                "start",
                "state",
                "store",
                "stream",
                "subset",
                "sum",
                "synchronize",
                "test",
                "treat",
                "trace",
                "unpack",
                "update",
                "use",
                "validate",
                "vectorize",
                "verify",
                "visit",
                "write",
                "yield",
            }
        )
        words: List[str] = re.findall(
            r"[A-Za-z]+(?:-[A-Za-z]+)*",
            sentence.lower(),
        )
        first_word: str = words[0] if words else ""
        has_directive_modal: bool = (
            re.search(
                r"\b(?:must|shall|should|required)\b",
                sentence,
                flags=re.IGNORECASE,
            )
            is not None
        )
        is_instruction: bool = (
            first_word in imperative_verbs
            or first_word == "you"
            or has_directive_modal
        )
        return is_instruction

    @staticmethod
    def _docstring_prose(  # noqa: PLR0912, PLR0915
        docstring: str,
    ) -> Tuple[str, ...]:
        """PRIVATE: Return prose blocks from one Python docstring.

        Parameters
        ----------
        docstring : str
            Docstring text as ``ast.get_docstring`` yields it.

        Returns
        -------
        prose : Tuple[str, ...]
            Joined prose paragraphs in reading order.

        Notes
        -----
        Tracks the current numpydoc section and keeps only indented
        description lines inside structured sections. Skips section
        rules, directives, Sphinx fields, doctests, and literal
        blocks after a double colon, and starts a new paragraph at
        each list item.
        """
        lines: List[str] = docstring.splitlines()
        paragraphs: List[str] = []
        current_lines: List[str] = []
        current_section: str = ""
        skip_indented_block: bool = False
        structured_sections: frozenset[str] = frozenset(
            {
                "Attributes",
                "Other Parameters",
                "Parameters",
                "Raises",
                "Returns",
                "See Also",
                "Yields",
            }
        )
        line_index: int
        raw_line: str
        for line_index, raw_line in enumerate(lines):
            stripped: str = raw_line.strip()
            next_line: str = (
                lines[line_index + 1].strip()
                if line_index + 1 < len(lines)
                else ""
            )
            starts_section: bool = bool(
                stripped and re.fullmatch(r"-{3,}", next_line)
            )
            is_section_rule: bool = bool(re.fullmatch(r"-{3,}", stripped))
            if starts_section:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                current_section = stripped
                skip_indented_block = False
                continue
            if is_section_rule:
                continue

            starts_rst_directive: bool = stripped.startswith(".. ")
            starts_sphinx_field: bool = (
                re.match(r"^:[A-Za-z0-9_-]+:", stripped) is not None
            )
            starts_doctest: bool = stripped.startswith((">>>", "..."))
            if starts_rst_directive or starts_sphinx_field or starts_doctest:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                skip_indented_block = starts_rst_directive
                continue

            if skip_indented_block:
                if not stripped or raw_line.startswith((" ", "\t")):
                    continue
                skip_indented_block = False

            if not stripped:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                continue

            if (
                current_section in structured_sections
                and not raw_line.startswith((" ", "\t"))
            ):
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                continue

            list_match: re.Match[str] | None = re.match(
                r"^(?:[-*+] |\d+\. )(.*)",
                stripped,
            )
            if list_match is not None:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                current_lines = [list_match.group(1)]
            else:
                current_lines.append(stripped)

            if stripped.endswith("::"):
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                skip_indented_block = True

        if current_lines:
            paragraphs.append(" ".join(current_lines))
        prose: Tuple[str, ...] = tuple(paragraphs)
        return prose

    def test_markdown_prose_obeys_ste_sentence_limits(self) -> None:
        """Keep repository Markdown within the STE sentence limits.

        The test confirms descriptions contain at most 25 words. Instructions
        contain at most 20 words across repository-authored Markdown files.

        Notes
        -----
        The test parses prose paragraphs and table cells. It excludes generated
        files, code fences, math blocks, front matter, and technical literals.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        excluded_roots: Tuple[Path, ...] = (
            repository_root / ".git",
            repository_root / ".venv",
            repository_root / ".pytest_cache",
            repository_root / "docs/build",
        )
        markdown_paths: Tuple[Path, ...] = tuple(
            path
            for path in sorted(repository_root.rglob("*.md"))
            if path.is_file()
            and not any(root in path.parents for root in excluded_roots)
            and not any(
                (parent / ".git").exists()
                for parent in path.parents
                if parent != repository_root
            )
        )
        violations: List[str] = []
        path: Path
        for path in markdown_paths:
            line_number: int
            paragraph: str
            for line_number, paragraph in self._markdown_prose(path):
                sentence: str
                for sentence in self._markdown_sentences(paragraph):
                    words: List[str] = re.findall(
                        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                        sentence,
                    )
                    limit: int = (
                        20 if self._markdown_instruction(sentence) else 25
                    )
                    if len(words) > limit:
                        relative_path: Path = path.relative_to(repository_root)
                        violations.append(
                            f"{relative_path}:{line_number}: "
                            f"{len(words)} words (limit {limit}): {sentence}"
                        )
        self.assertEqual(violations, [])

    def test_python_docstrings_obey_ste_prose_rules(self) -> None:
        """Keep Python docstrings within the measurable STE prose rules.

        The test confirms each docstring sentence meets its applicable word
        limit. It also rejects passive voice and non-present tense candidates.

        Notes
        -----
        The test parses all source and test docstrings through the AST. It
        excludes structured signatures, directives, code blocks, and technical
        literals before it checks the prose.
        """
        modules: Tuple[Tuple[Path, ast.Module], ...] = (
            self._production_modules() + self._test_modules()
        )
        passive_pattern: re.Pattern[str] = re.compile(
            r"\b(?:am|is|are|was|were|be|been|being)\s+"
            r"(?:\w+ly\s+)?(?:\w+(?:ed|en)|built|done|found|given|kept|known|"
            r"made|put|run|set|shown|told|written)\b",
            flags=re.IGNORECASE,
        )
        tense_pattern: re.Pattern[str] = re.compile(
            r"\b(?:will|would|was|were|had)\b",
            flags=re.IGNORECASE,
        )
        violations: List[str] = []
        path: Path
        module: ast.Module
        for path, module in modules:
            node: ast.AST
            for node in ast.walk(module):
                if not isinstance(
                    node,
                    (
                        ast.Module,
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue
                docstring: str | None = ast.get_docstring(node)
                if docstring is None:
                    continue
                summary: str = docstring.splitlines()[0]
                symbol_name: str = getattr(node, "name", "<module>")
                location: str = f"{path}:{getattr(node, 'lineno', 1)}"
                summary_text: str = summary
                is_private_module: bool = isinstance(
                    node, ast.Module
                ) and path.name.startswith("_")
                if symbol_name.startswith("_") or is_private_module:
                    summary_text = summary_text.removeprefix("PRIVATE: ")
                if not self._markdown_instruction(summary_text):
                    violations.append(
                        f"{location}:{symbol_name}: non-imperative summary: "
                        f"{summary}"
                    )
                paragraph: str
                for paragraph in self._docstring_prose(docstring):
                    sentence: str
                    for sentence in self._markdown_sentences(paragraph):
                        words: List[str] = re.findall(
                            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
                            sentence,
                        )
                        limit: int = (
                            20 if self._markdown_instruction(sentence) else 25
                        )
                        if len(words) > limit:
                            violations.append(
                                f"{location}:{symbol_name}: "
                                f"{len(words)} words "
                                f"(limit {limit}): {sentence}"
                            )
                        if passive_pattern.search(sentence) is not None:
                            violations.append(
                                f"{location}:{symbol_name}: passive voice: "
                                f"{sentence}"
                            )
                        if tense_pattern.search(sentence) is not None:
                            violations.append(
                                f"{location}:{symbol_name}: "
                                "non-present tense: "
                                f"{sentence}"
                            )
        self.assertEqual(violations, [])

    def test_private_functions_have_private_docstrings(self) -> None:
        """Require fully fledged PRIVATE docstrings on private callables.

        The test confirms every single-underscore function or method at any
        nesting depth carries a docstring whose summary starts with the
        ``PRIVATE:`` marker.

        Notes
        -----
        The test walks production and complete-test-tree modules, excludes
        dunder names, and rejects summary-only docstrings unless the body
        is a bare ``pass`` or ellipsis stub. It reports each violation as
        a path, line, name, and reason row.
        """
        section_markers: Tuple[str, ...] = (
            "Parameters\n",
            "Returns\n",
            "Yields\n",
            "Notes\n",
            "Implementation Logic\n",
            "Raises\n",
        )
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if not node.name.startswith("_") or node.name.startswith("__"):
                    continue
                docstring: str | None = ast.get_docstring(node)
                if docstring is None:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}:missing-docstring"
                    )
                    continue
                if not docstring.startswith("PRIVATE: "):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}:no-PRIVATE-prefix"
                    )
                body_after_doc: List[ast.stmt] = node.body[1:]
                is_stub: bool = not body_after_doc or all(
                    isinstance(statement, ast.Pass)
                    or (
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and statement.value.value is Ellipsis
                    )
                    for statement in body_after_doc
                )
                has_section: bool = any(
                    marker in docstring for marker in section_markers
                )
                if not has_section and not is_stub:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}:summary-only"
                    )
        self.assertEqual(violations, [])

    def test_annotations_use_beartype_collection_generics(  # noqa: PLR0912
        self,
    ) -> None:
        """Require beartype collection generics in annotations.

        The test rejects builtin ``tuple``, ``dict``, and ``list`` generics.
        It permits builtin ``list`` only for a module ``__all__`` annotation.
        It requires the other forms from ``beartype.typing``. The test also
        rejects charter-owned imports from the standard ``typing`` module.

        Notes
        -----
        The test walks every argument, return, and variable annotation in
        the production and complete test trees. It reports one row per
        offending annotation as a path, line, and reason. Runtime uses of
        The check reads annotation and type-alias expressions only. Runtime
        collection calls, checks, and literals stay valid.
        """
        banned_typing_names: Tuple[str, ...] = (
            "Dict",
            "List",
            "Optional",
            "Tuple",
            "TypeAlias",
            "Union",
        )
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            imported_beartype_names: set[str] = {
                alias.name
                for statement in module.body
                if isinstance(statement, ast.ImportFrom)
                and statement.module == "beartype.typing"
                for alias in statement.names
            }
            required_beartype_names: Dict[str, int] = {}
            for node in ast.walk(module):
                annotations: List[Tuple[int, ast.AST, bool]] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arguments: ast.arguments = node.args
                    argument: ast.arg
                    for argument in (
                        list(arguments.posonlyargs)
                        + list(arguments.args)
                        + list(arguments.kwonlyargs)
                        + (
                            [arguments.vararg]
                            if arguments.vararg is not None
                            else []
                        )
                        + (
                            [arguments.kwarg]
                            if arguments.kwarg is not None
                            else []
                        )
                    ):
                        if argument.annotation is not None:
                            annotations.append(
                                (argument.lineno, argument.annotation, False)
                            )
                    if node.returns is not None:
                        annotations.append((node.lineno, node.returns, False))
                elif isinstance(node, ast.TypeAlias):
                    annotations.append((node.lineno, node.value, False))
                elif isinstance(node, ast.AnnAssign):
                    is_all_annotation: bool = (
                        isinstance(node.target, ast.Name)
                        and node.target.id == "__all__"
                    )
                    annotations.append(
                        (node.lineno, node.annotation, is_all_annotation)
                    )
                    is_type_alias: bool = (
                        isinstance(node.annotation, ast.Name)
                        and node.annotation.id == "TypeAlias"
                    ) or (
                        isinstance(node.annotation, ast.Attribute)
                        and node.annotation.attr == "TypeAlias"
                    )
                    if is_type_alias and node.value is not None:
                        annotations.append((node.lineno, node.value, False))
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "typing":
                        imported_banned: List[str] = sorted(
                            alias.name
                            for alias in node.names
                            if alias.name in banned_typing_names
                        )
                        if imported_banned:
                            violations.append(
                                f"{path}:{node.lineno}:stdlib-typing-import:"
                                + ",".join(imported_banned)
                            )
                lineno: int
                annotation: ast.AST
                allow_builtin_list: bool
                for lineno, annotation, allow_builtin_list in annotations:
                    inner: ast.AST
                    builtin_names: set[str] = {
                        inner.id
                        for inner in ast.walk(annotation)
                        if isinstance(inner, ast.Name)
                        and inner.id in {"dict", "list", "tuple"}
                        and not (inner.id == "list" and allow_builtin_list)
                    }
                    builtin_name: str
                    for builtin_name in sorted(builtin_names):
                        violations.append(
                            f"{path}:{lineno}:"
                            f"builtin-{builtin_name}-annotation"
                        )
                    for inner in ast.walk(annotation):
                        if isinstance(inner, ast.Name) and inner.id in {
                            "Dict",
                            "List",
                            "Tuple",
                        }:
                            required_beartype_names.setdefault(
                                inner.id, lineno
                            )
            required_name: str
            for required_name in sorted(required_beartype_names):
                if required_name not in imported_beartype_names:
                    violations.append(
                        f"{path}:"
                        f"{required_beartype_names[required_name]}:"
                        "missing-beartype-typing-import:"
                        f"{required_name}"
                    )
        self.assertEqual(violations, [])

    def test_legacy_pytree_carriers_are_forbidden(self) -> None:
        """Reject legacy PyTree carrier and registration machinery.

        The test confirms production carriers do not use ``NamedTuple`` or
        manual JAX flattening hooks instead of the Equinox carrier contract.

        Notes
        -----
        The test parses production classes and call expressions. It reports the
        source location of each forbidden base, method, or registration call.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef):
                    bases: set[str] = {
                        ast.unparse(base) for base in node.bases
                    }
                    methods: set[str] = {
                        child.name
                        for child in node.body
                        if isinstance(child, ast.FunctionDef)
                    }
                    if "NamedTuple" in bases or methods & {
                        "tree_flatten",
                        "tree_unflatten",
                    }:
                        violations.append(f"{path}:{node.lineno}:{node.name}")
                if isinstance(node, ast.Call):
                    called: str = ast.unparse(node.func)
                    if called.endswith("register_pytree_node_class"):
                        violations.append(f"{path}:{node.lineno}:{called}")
        self.assertEqual(violations, [])

    def test_all_production_carriers_are_types_equinox_modules(self) -> None:
        """Keep data carriers in ``diffpes.types`` and retain pure interfaces.

        The test confirms every data-holding production class is an Equinox
        module in the types subpackage. A pure Protocol is an interface, not
        a carrier, when it has direct Protocol inheritance, no default values,
        and only ellipsis method bodies.

        Notes
        -----
        The test parses every class declaration, including private operational
        state, and compares its direct bases and source directory with the
        type-ownership rule.
        """
        child: ast.stmt
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            in_types: bool = path.parent.name == "types"
            for node in module.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                bases: set[str] = {ast.unparse(base) for base in node.bases}
                is_protocol: bool = "Protocol" in bases
                if is_protocol:
                    if not in_types:
                        violations.append(
                            f"{path}:{node.lineno}:{node.name}:"
                            "protocol-outside-types"
                        )
                    has_defaults: bool = any(
                        (
                            isinstance(child, ast.AnnAssign)
                            and child.value is not None
                        )
                        or isinstance(child, ast.Assign)
                        for child in node.body
                    )
                    if has_defaults:
                        violations.append(
                            f"{path}:{node.lineno}:{node.name}:"
                            "protocol-attribute-default"
                        )
                    for child in node.body:
                        if not isinstance(
                            child,
                            ast.FunctionDef | ast.AsyncFunctionDef,
                        ):
                            continue
                        body: List[ast.stmt] = [
                            item
                            for item in child.body
                            if not (
                                isinstance(item, ast.Expr)
                                and isinstance(item.value, ast.Constant)
                                and isinstance(item.value.value, str)
                            )
                        ]
                        if not all(
                            isinstance(item, ast.Expr)
                            and isinstance(item.value, ast.Constant)
                            and item.value.value is Ellipsis
                            for item in body
                        ):
                            violations.append(
                                f"{path}:{child.lineno}:{node.name}:"
                                "protocol-method-body"
                            )
                    continue
                if not in_types or "eqx.Module" not in bases:
                    violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_protocol_outside_types_remains_a_carrier_violation(self) -> None:
        """Reject a Protocol outside types by inspecting direct inheritance.

        The fixture places a direct Protocol subclass under a simulation path.

        Notes
        -----
        Compare the fixture path with the single types ownership directory.
        """
        fixture: Any
        protocol: Any
        fixture = ast.parse("class External(Protocol):\n    value: str\n")
        protocol = fixture.body[0]
        assert isinstance(protocol, ast.ClassDef)
        bases: set[str] = {ast.unparse(base) for base in protocol.bases}
        fixture_path: Path = Path("src/diffpes/simul/fixture.py")
        is_violation: bool = (
            "Protocol" in bases and fixture_path.parent.name != "types"
        )
        self.assertTrue(is_violation)

    def test_protocol_logic_remains_a_carrier_violation(self) -> None:
        """Reject a Protocol method that defines non-ellipsis behavior.

        The fixture places executable return logic inside an interface method.

        Notes
        -----
        Inspect every non-docstring statement and require an ellipsis literal.
        """
        fixture: Any
        protocol: Any
        method: Any
        fixture = ast.parse(
            "class External(Protocol):\n"
            "    def value(self):\n"
            "        return 1\n"
        )
        protocol = fixture.body[0]
        assert isinstance(protocol, ast.ClassDef)
        method = protocol.body[0]
        assert isinstance(method, ast.FunctionDef)
        is_violation: bool = not all(
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and item.value.value is Ellipsis
            for item in method.body
        )
        self.assertTrue(is_violation)

    def test_make_factories_are_types_owned(self) -> None:
        """Forbid ``make_*`` factories outside ``diffpes.types``.

        The test confirms consumers cannot create another construction
        contract for a public carrier in a production subpackage.

        Notes
        -----
        Scans top-level production callables and reports each ``make_*`` name
        whose module is not owned by the types subpackage.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            for node in module.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("make_"):
                    violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_all_equinox_carriers_have_factories_and_external_factory_use(  # noqa: PLR0912
        self,
    ) -> None:
        """Bind every Equinox carrier to one types-owned construction API.

        The test maps factory return annotations to carrier classes. It then
        rejects direct carrier construction from production consumer modules.

        Notes
        -----
        Construction inside ``diffpes.types`` remains visible for factory
        implementations. Consumer packages must call the matching public
        ``make_*`` function instead.
        """
        type_classes: Dict[str, Tuple[Path, ast.ClassDef]] = {}
        factories_by_type: Dict[str, List[str]] = {}
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name != "types":
                continue
            for node in module.body:
                if isinstance(node, ast.ClassDef) and any(
                    ast.unparse(base) == "eqx.Module" for base in node.bases
                ):
                    type_classes[node.name] = (path, node)
                if not isinstance(
                    node, ast.FunctionDef
                ) or not node.name.startswith("make_"):
                    continue
                return_name: str = (
                    ast.unparse(node.returns)
                    if node.returns is not None
                    else ""
                ).strip("'\"")
                factories_by_type.setdefault(return_name, []).append(node.name)

        violations: List[str] = []
        type_name: str
        class_record: Tuple[Path, ast.ClassDef]
        for type_name, class_record in sorted(type_classes.items()):
            if type_name not in factories_by_type:
                violations.append(
                    f"{class_record[0]}:{class_record[1].lineno}:"
                    f"{type_name}: missing make_* factory"
                )

        called_name: str
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            call: ast.AST
            for call in ast.walk(module):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name):
                    called_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    called_name = call.func.attr
                else:
                    called_name = ""
                if called_name in type_classes:
                    violations.append(
                        f"{path}:{call.lineno}:{called_name}: "
                        "direct construction"
                    )
        self.assertEqual(violations, [])

    def test_equinox_carrier_docs_bind_fields_and_factories(  # noqa: PLR0912
        self,
    ) -> None:
        """Require complete field and factory documentation on every carrier.

        The test compares class fields with ordered ``Attributes`` entries.
        It also requires ``See Also`` to name a factory returning that class.

        Notes
        -----
        Parse types modules only. Exact field order and a literal factory name
        keep the construction surface auditable without importing JAX.
        """
        type_classes: Dict[str, Tuple[Path, ast.ClassDef]] = {}
        factories_by_type: Dict[str, List[str]] = {}
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name != "types":
                continue
            for node in module.body:
                if isinstance(node, ast.ClassDef) and any(
                    ast.unparse(base) == "eqx.Module" for base in node.bases
                ):
                    type_classes[node.name] = (path, node)
                if not isinstance(
                    node, ast.FunctionDef
                ) or not node.name.startswith("make_"):
                    continue
                return_name: str = (
                    ast.unparse(node.returns)
                    if node.returns is not None
                    else ""
                ).strip("'\"")
                factories_by_type.setdefault(return_name, []).append(node.name)

        violations: List[str] = []
        type_name: str
        class_record: Tuple[Path, ast.ClassDef]
        for type_name, class_record in sorted(type_classes.items()):
            class_path: Path = class_record[0]
            class_node: ast.ClassDef = class_record[1]
            docstring: str = ast.get_docstring(class_node) or ""
            declared_fields: List[str] = [
                child.target.id
                for child in class_node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            ]
            lines: List[str] = docstring.splitlines()
            attribute_fields: List[str] = []
            in_attributes: bool = False
            line_index: int
            line: str
            for line_index, line in enumerate(lines):
                stripped: str = line.strip()
                next_line: str = (
                    lines[line_index + 1].strip()
                    if line_index + 1 < len(lines)
                    else ""
                )
                if stripped == "Attributes" and re.fullmatch(
                    r"-{3,}", next_line
                ):
                    in_attributes = True
                    continue
                if (
                    in_attributes
                    and stripped
                    and re.fullmatch(r"-{3,}", next_line)
                ):
                    break
                if in_attributes:
                    field_match: re.Match[str] | None = re.match(
                        r"^([A-Za-z_][A-Za-z0-9_]*)\s*:",
                        stripped,
                    )
                    if field_match is not None:
                        attribute_fields.append(field_match.group(1))
            if attribute_fields != declared_fields:
                violations.append(
                    f"{class_path}:{class_node.lineno}:{type_name}: "
                    f"Attributes={attribute_fields}, fields={declared_fields}"
                )
            factory_names: List[str] = factories_by_type.get(type_name, [])
            if not factory_names or not any(
                factory_name in docstring for factory_name in factory_names
            ):
                violations.append(
                    f"{class_path}:{class_node.lineno}:{type_name}: "
                    f"See Also factory={factory_names}"
                )
        self.assertEqual(violations, [])

    def test_pytest_raises_uses_exact_exceptions_and_messages(self) -> None:
        """Require specific exception classes and message contracts in tests.

        The test rejects broad ``Exception`` or ``BaseException`` assertions.
        Every ``pytest.raises`` context must also state a nonempty match regex.

        Notes
        -----
        Scan the complete test tree, including non-collected helpers, through
        the AST. Production cleanup handlers remain outside this assertion.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._test_tree_modules():
            for node in ast.walk(module):
                if not isinstance(node, ast.Call):
                    continue
                if ast.unparse(node.func) != "pytest.raises" or not node.args:
                    continue
                exception_names: set[str] = {
                    candidate.id
                    for candidate in ast.walk(node.args[0])
                    if isinstance(candidate, ast.Name)
                }
                if exception_names & {"BaseException", "Exception"}:
                    violations.append(f"{path}:{node.lineno}: broad exception")
                match_values: List[ast.expr | None] = [
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "match"
                ]
                if not match_values or (
                    isinstance(match_values[0], ast.Constant)
                    and match_values[0].value == ""
                ):
                    violations.append(f"{path}:{node.lineno}: missing match")
        self.assertEqual(violations, [])

    def test_declarative_constants_are_centrally_owned(self) -> None:
        """Keep declarative constants under ``diffpes.constants``.

        The test confirms other modules contain only approved generated data,
        runtime state, package metadata, type aliases, and public export lists.

        Notes
        -----
        The test parses module-level assignments. It compares them with the
        narrow allowlist for version, registry, generated data, and aliases.
        """
        allowed: Dict[str, set[str]] = {
            "__init__.py": {"__version__"},
            "inout/hdf5.py": {"_PYTREE_REGISTRY"},
            "types/aliases.py": {
                "NonJaxNumber",
                "RetardedSelfEnergySource",
                "ScalarBool",
                "ScalarComplex",
                "ScalarFloat",
                "ScalarInteger",
                "ScalarNumeric",
                "SliceOperator",
            },
            "types/context.py": {"DosType", "ProjectionType"},
        }
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        name: str
        for path, module in self._production_modules():
            if "constants" in path.parts:
                continue
            relative_path: str = path.as_posix().split("/src/diffpes/", 1)[1]
            for node in module.body:
                names: List[str] = []
                if isinstance(node, ast.Assign):
                    names = [
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    ]
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    names = [node.target.id]
                for name in names:
                    if name == "__all__" or name in allowed.get(
                        relative_path, set()
                    ):
                        continue
                    violations.append(f"{path}:{node.lineno}:{name}")
        self.assertEqual(violations, [])

    def test_production_modules_follow_the_source_line_limit(self) -> None:
        """Limit each production implementation file to 1000 lines.

        The test accepts one function that starts before the limit and ends
        after it. Only the required literal export list may follow that
        function.

        Notes
        -----
        Exclude package initializers because their public listings cannot
        split. Report every other oversized file that lacks the narrow
        single-function exception.
        """
        line_limit: int = 1000
        violations: List[str] = []
        path: Path
        module: ast.Module
        for path, module in self._production_modules():
            if path.name == "__init__.py":
                continue
            line_count: int = len(
                path.read_text(encoding="utf-8").splitlines()
            )
            if line_count <= line_limit:
                continue
            crossing_functions: List[
                ast.FunctionDef | ast.AsyncFunctionDef
            ] = [
                node
                for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and min(
                    [node.lineno]
                    + [decorator.lineno for decorator in node.decorator_list]
                )
                <= line_limit
                and node.end_lineno is not None
                and node.end_lineno > line_limit
            ]
            if len(crossing_functions) != 1:
                violations.append(
                    f"{path}: lines={line_count}, "
                    f"crossing_functions={len(crossing_functions)}"
                )
                continue
            crossing_function: ast.FunctionDef | ast.AsyncFunctionDef = (
                crossing_functions[0]
            )
            trailing_nodes: List[ast.stmt] = []
            node: ast.stmt
            for node in module.body:
                if (
                    crossing_function.end_lineno is not None
                    and node.lineno <= crossing_function.end_lineno
                ):
                    continue
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "__all__"
                    and isinstance(node.value, ast.List)
                ):
                    continue
                trailing_nodes.append(node)
            if trailing_nodes:
                locations: List[str] = [
                    f"{type(node).__name__}:{node.lineno}"
                    for node in trailing_nodes
                ]
                violations.append(
                    f"{path}: lines={line_count}, trailing={locations}"
                )
        self.assertEqual(violations, [])

    def test_source_modules_have_exact_mirrored_test_modules(self) -> None:
        """Mirror every source module with its literal test-module name.

        The test preserves leading underscores when it prefixes a source
        filename with ``test_``. Package initializers map to the matching test
        package initializer.

        Notes
        -----
        Walk the complete source tree without importing it. Resolve each path
        under ``tests/test_diffpes`` and report every missing mirror.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        source_root: Path = repository_root / "src/diffpes"
        tests_root: Path = repository_root / "tests/test_diffpes"
        violations: List[str] = []
        source_path: Path
        for source_path in sorted(source_root.rglob("*.py")):
            relative_path: Path = source_path.relative_to(source_root)
            if len(relative_path.parts) == 1:
                test_parent: Path = tests_root
            else:
                test_parent = tests_root / f"test_{relative_path.parts[0]}"
                test_parent = test_parent.joinpath(*relative_path.parts[1:-1])
            test_name: str = (
                "__init__.py"
                if source_path.name == "__init__.py"
                else f"test_{source_path.name}"
            )
            test_path: Path = test_parent / test_name
            if not test_path.is_file():
                violations.append(f"{relative_path} -> {test_path}")
        self.assertEqual(violations, [])

    def test_type_aliases_are_types_owned(self) -> None:
        """Keep every production type alias under ``diffpes.types``.

        The test confirms PEP 695 declarations and legacy ``TypeAlias``
        annotations do not create local type vocabularies in consumers.

        Notes
        -----
        The test parses module-level declarations. It reports the source
        location of each alias found outside the types subpackage.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            for node in module.body:
                if isinstance(node, ast.TypeAlias):
                    alias_name: str = ast.unparse(node.name)
                    violations.append(f"{path}:{node.lineno}:{alias_name}")
                elif isinstance(node, ast.AnnAssign) and ast.unparse(
                    node.annotation
                ).endswith("TypeAlias"):
                    target_name: str = ast.unparse(node.target)
                    violations.append(f"{path}:{node.lineno}:{target_name}")
        self.assertEqual(violations, [])

    def test_public_functions_are_runtime_typechecked(self) -> None:
        """Require the project decorator on every public production function.

        The test confirms public module-level callables use the exact
        ``@jaxtyped(typechecker=beartype)`` stack required by CONTRIBUTING.

        Notes
        -----
        The test compares normalized decorator syntax through the AST. It
        reports each missing function with its source line.
        """
        violations: List[str] = []
        required_decorator: str = "jaxtyped(typechecker=beartype)"
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            for node in module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or node.name.startswith("_"):
                    continue
                decorators: set[str] = {
                    ast.unparse(decorator) for decorator in node.decorator_list
                }
                if required_decorator not in decorators:
                    violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_functions_assign_before_returning(self) -> None:
        """Require source and test functions to return annotated names.

        The test confirms each value-returning path binds its result before
        returning. It includes paths in private and nested helpers.

        Notes
        -----
        The test walks every function while excluding nested bodies. It reports
        non-name return expressions by source line.
        """

        class ReturnVisitor(ast.NodeVisitor):
            """Collect bare returns without entering nested callables."""

            def __init__(
                self,
                root: ast.FunctionDef | ast.AsyncFunctionDef,
            ) -> None:
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.violations: List[int] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression bodies from the outer contract."""
                del node

            def visit_Return(self, node: ast.Return) -> None:
                """Record returns whose value is not a bound local name."""
                if node.value is not None and not isinstance(
                    node.value, ast.Name
                ):
                    self.violations.append(node.lineno)

        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: ReturnVisitor = ReturnVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}" for line in visitor.violations
                )
        self.assertEqual(violations, [])

    def test_function_intermediates_are_annotated(self) -> None:
        """Require explicit types for production intermediate variables.

        The test confirms assignment, loop, context, walrus, and exception
        targets have an annotation in their scope while respecting
        ``nonlocal``.

        Notes
        -----
        The test walks one callable scope at a time. It excludes nested
        callables and ``_`` bindings and reports each unannotated local target.
        """

        class AssignmentVisitor(ast.NodeVisitor):
            """Collect annotations and assignments in one callable scope."""

            def __init__(
                self,
                root: ast.FunctionDef | ast.AsyncFunctionDef,
            ) -> None:
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.annotated: set[str] = set()
                self.nonlocal_names: set[str] = set()
                self.assignments: List[Tuple[int, str]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression scopes from the outer contract."""
                del node

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Record a directly annotated local name."""
                if isinstance(node.target, ast.Name):
                    self.annotated.add(node.target.id)
                self.generic_visit(node)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                """Record names whose annotations belong to an outer scope."""
                self.nonlocal_names.update(node.names)

            def _record_target(self, target: ast.expr, line: int) -> None:
                """PRIVATE: Record names within one assignment-like target.

                Parameters
                ----------
                target : ast.expr
                    Assignment, loop, context, or walrus target
                    expression.
                line : int
                    Source line to attach to each recorded name.

                Notes
                -----
                Walks the target and appends every stored
                ``ast.Name`` except the throwaway ``_`` to the
                collected assignments.
                """
                candidate: ast.AST
                for candidate in ast.walk(target):
                    if (
                        isinstance(candidate, ast.Name)
                        and isinstance(candidate.ctx, ast.Store)
                        and candidate.id != "_"
                    ):
                        self.assignments.append((line, candidate.id))

            def visit_Assign(self, node: ast.Assign) -> None:
                """Record plain local-name assignment targets."""
                target: ast.expr
                for target in node.targets:
                    self._record_target(target, node.lineno)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                """Record an ordinary loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                """Record an asynchronous loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                """Record context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                """Record asynchronous context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                """Record an assignment-expression target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                """Record an exception-handler binding."""
                if node.name is not None and node.name != "_":
                    self.assignments.append((node.lineno, node.name))
                self.generic_visit(node)

        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: AssignmentVisitor = AssignmentVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}:{name}"
                    for line, name in visitor.assignments
                    if name not in visitor.annotated
                    and name not in visitor.nonlocal_names
                )
        self.assertEqual(violations, [])

    def test_cross_subpackage_imports_use_public_surfaces(self) -> None:
        """Forbid deep imports across production subpackage boundaries.

        The test confirms consumers import through public subpackage surfaces.
        It rejects access to another subpackage's implementation file.

        Notes
        -----
        The test compares each absolute DiffPES import with the file's owning
        subpackage. It reports cross-boundary modules deeper than one level.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            relative_parts: Tuple[str, ...] = tuple(
                path.as_posix().split("/src/diffpes/", 1)[1].split("/")
            )
            owner: str = relative_parts[0]
            for node in ast.walk(module):
                if not isinstance(node, ast.ImportFrom) or node.level != 0:
                    continue
                imported_module: str = node.module or ""
                parts: List[str] = imported_module.split(".")
                if (
                    len(parts) > 2
                    and parts[0] == "diffpes"
                    and parts[1] != owner
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{imported_module}"
                    )
        self.assertEqual(violations, [])

    def test_diffpes_imports_are_not_renamed(self) -> None:
        """Forbid aliases for names imported from DiffPES surfaces.

        The test confirms each internal DiffPES name has one spelling at every
        production or test consumer. It excludes private aliases for shared
        constants.

        Notes
        -----
        The test inspects absolute DiffPES imports across production and the
        complete test tree. It reports every ``as`` binding. Canonical
        third-party aliases such as ``jnp`` are outside this scan.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        imported_name: ast.alias
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom) and (
                    node.module or ""
                ).startswith("diffpes"):
                    for imported_name in node.names:
                        if imported_name.asname is not None:
                            violations.append(
                                f"{path}:{node.lineno}:{imported_name.name}"
                            )
                elif isinstance(node, ast.Import):
                    for imported_name in node.names:
                        if (
                            imported_name.name.startswith("diffpes")
                            and imported_name.asname is not None
                        ):
                            violations.append(
                                f"{path}:{node.lineno}:{imported_name.name}"
                            )
        self.assertEqual(violations, [])

    def test_typing_constructs_use_beartype_typing(self) -> None:
        """Forbid source and test imports from the standard typing module.

        The test confirms runtime-visible typing constructs come from
        ``beartype.typing`` as required by the package type-checking contract.

        Notes
        -----
        The test reports ``import typing`` and ``from typing import ...`` at
        their source or test-tree locations.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in module.body:
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "typing"
                ):
                    violations.append(f"{path}:{node.lineno}:from typing")
                elif isinstance(node, ast.Import) and any(
                    imported_name.name == "typing"
                    for imported_name in node.names
                ):
                    violations.append(f"{path}:{node.lineno}:import typing")
        self.assertEqual(violations, [])

    def test_annotations_do_not_use_bare_ndarray(self) -> None:
        """Forbid dtype-free and shape-free NumPy array annotations.

        The test confirms each NumPy array annotation carries a jaxtyping dtype
        and shape. NumPy arrays receive the same contract as JAX arrays.

        Notes
        -----
        The test inspects annotation positions only and excludes ``isinstance``
        checks. It reports bare and parameterized array annotations across
        production and tests. Generators follow the same contract.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        inner: ast.AST

        def _names_array(annotation: ast.AST) -> bool:
            """PRIVATE: Detect a node that names the NumPy array type.

            Parameters
            ----------
            annotation : ast.AST
                Annotation node to test.

            Returns
            -------
            is_name : bool
                True for an attribute access ending in ``ndarray``
                or for the bare name ``NDArray``.

            Notes
            -----
            Checks the two spellings separately, so dotted and
            imported forms match the same rule.
            """
            if isinstance(annotation, ast.Attribute):
                is_attribute: bool = annotation.attr == "ndarray"
                return is_attribute
            is_name: bool = (
                isinstance(annotation, ast.Name) and annotation.id == "NDArray"
            )
            return is_name

        def _qualified(annotation: ast.AST) -> set[int]:
            """PRIVATE: Collect arrays qualified by jaxtyping specifications.

            Parameters
            ----------
            annotation : ast.AST
                Full annotation expression to scan.

            Returns
            -------
            allowed : set[int]
                ``id`` values of array-type nodes in the first slot
                of a two-part jaxtyping subscript with a string
                shape.

            Notes
            -----
            Walks every subscript and accepts exactly the
            two-element dtype-and-shape pattern, so any other
            parameterization stays reportable.
            """
            allowed: set[int] = set()
            candidate: ast.AST
            for candidate in ast.walk(annotation):
                if not isinstance(candidate, ast.Subscript):
                    continue
                arguments: ast.expr = candidate.slice
                if (
                    isinstance(arguments, ast.Tuple)
                    and len(arguments.elts) == 2  # noqa: PLR2004
                    and _names_array(arguments.elts[0])
                    and isinstance(arguments.elts[1], ast.Constant)
                    and isinstance(arguments.elts[1].value, str)
                ):
                    allowed.add(id(arguments.elts[0]))
            return allowed

        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                annotation: ast.AST | None = getattr(node, "annotation", None)
                if annotation is None and isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    annotation = node.returns
                if annotation is None:
                    continue
                allowed_nodes: set[int] = _qualified(annotation)
                for inner in ast.walk(annotation):
                    if _names_array(inner) and id(inner) not in allowed_nodes:
                        violations.append(f"{path}:{inner.lineno}")
        self.assertEqual(violations, [])

    def test_annotations_do_not_use_bare_jax_array(self) -> None:
        """Forbid dtype-free and shape-free JAX array annotations.

        The test confirms every JAX array annotation carries a jaxtyping dtype
        and shape. Genuinely dtype-polymorphic code uses ``Shaped[Array, ...]``
        while fixed-storage contracts use a width-qualified dtype.

        Notes
        -----
        The test inspects every annotation in production and the complete test
        tree. It accepts ``Array`` only as the first slot of a two-part
        jaxtyping subscript whose second slot is a shape string.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        inner: ast.AST

        def _names_array(annotation: ast.AST) -> bool:
            """PRIVATE: Return whether one annotation node names JAX Array.

            Parameters
            ----------
            annotation : ast.AST
                Annotation node to test.

            Returns
            -------
            is_name : bool
                True for imported ``Array`` or the bare ``jax.Array`` form.

            Notes
            -----
            Both spellings omit dtype and shape unless a jaxtyping subscript
            qualifies the backend array name.
            """
            is_name: bool = (
                isinstance(annotation, ast.Name) and annotation.id == "Array"
            ) or (
                isinstance(annotation, ast.Attribute)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id == "jax"
                and annotation.attr == "Array"
            )
            return is_name

        def _qualified(annotation: ast.AST) -> set[int]:
            """PRIVATE: Collect JAX Array nodes qualified by dtype and shape.

            Parameters
            ----------
            annotation : ast.AST
                Full annotation expression to scan.

            Returns
            -------
            allowed : set[int]
                Object identities for qualified backend-array nodes.

            Notes
            -----
            A qualified node occupies the first slot of a two-element
            jaxtyping subscript. The second slot must be a string shape.
            """
            allowed: set[int] = set()
            candidate: ast.AST
            for candidate in ast.walk(annotation):
                if not isinstance(candidate, ast.Subscript):
                    continue
                arguments: ast.expr = candidate.slice
                if (
                    isinstance(arguments, ast.Tuple)
                    and len(arguments.elts) == 2  # noqa: PLR2004
                    and _names_array(arguments.elts[0])
                    and isinstance(arguments.elts[1], ast.Constant)
                    and isinstance(arguments.elts[1].value, str)
                ):
                    allowed.add(id(arguments.elts[0]))
            return allowed

        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                annotation: ast.AST | None = getattr(node, "annotation", None)
                if annotation is None and isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    annotation = node.returns
                if annotation is None:
                    continue
                allowed_nodes: set[int] = _qualified(annotation)
                for inner in ast.walk(annotation):
                    if _names_array(inner) and id(inner) not in allowed_nodes:
                        violations.append(f"{path}:{inner.lineno}")
        self.assertEqual(violations, [])

    def test_literal_array_locals_use_exact_shapes(self) -> None:  # noqa: PLR0912
        """Require exact shape strings for statically sized array locals.

        The test identifies annotated arrays built from literal values, literal
        shape constructors, or a literal-size linspace. Such locals must name
        their known axes instead of using the polymorphic ellipsis shape.

        Notes
        -----
        Parse production and the complete test tree without importing either.
        Dynamic expressions remain outside this narrow static-size rule.
        """
        literal_constructors: set[str] = {
            "jnp.array",
            "jnp.asarray",
            "jnp.empty",
            "jnp.full",
            "jnp.ones",
            "jnp.zeros",
            "np.array",
            "np.asarray",
            "np.empty",
            "np.full",
            "np.ones",
            "np.zeros",
        }
        linspace_constructors: set[str] = {"jnp.linspace", "np.linspace"}
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in (
            self._production_modules() + self._test_tree_modules()
        ):
            for node in ast.walk(module):
                if (
                    not isinstance(node, ast.AnnAssign)
                    or node.value is None
                    or not isinstance(node.value, ast.Call)
                ):
                    continue
                has_ellipsis_shape: bool = any(
                    isinstance(annotation_node, ast.Constant)
                    and annotation_node.value == "..."
                    for annotation_node in ast.walk(node.annotation)
                )
                if not has_ellipsis_shape:
                    continue
                constructor: str = ast.unparse(node.value.func)
                literal_size: bool = False
                if constructor in literal_constructors and node.value.args:
                    try:
                        ast.literal_eval(node.value.args[0])
                    except (TypeError, ValueError, SyntaxError):
                        pass
                    else:
                        literal_size = True
                elif constructor in linspace_constructors:
                    num_expression: ast.expr | None = (
                        node.value.args[2]
                        if len(node.value.args) > 2  # noqa: PLR2004
                        else next(
                            (
                                keyword.value
                                for keyword in node.value.keywords
                                if keyword.arg == "num"
                            ),
                            None,
                        )
                    )
                    if num_expression is not None:
                        try:
                            literal_num: object = ast.literal_eval(
                                num_expression
                            )
                        except (TypeError, ValueError, SyntaxError):
                            pass
                        else:
                            literal_size = isinstance(literal_num, int)
                if literal_size:
                    target: str = ast.unparse(node.target)
                    violations.append(f"{path}:{node.lineno}:{target}")
        self.assertEqual(violations, [])

    def test_package_docstrings_list_every_submodule(self) -> None:
        """Keep package ``Extended Summary`` submodule lists exact.

        The test confirms each package docstring contains one ``- :mod:`` entry
        for every sibling module. Each entry repeats the module summary.

        Notes
        -----
        The test compares filenames and summaries with Sphinx module roles. It
        parses descriptions from each production package docstring.
        """
        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        violations: List[str] = []
        path: Path
        module: ast.Module
        for path, module in self._production_modules():
            if path.name != "__init__.py":
                continue
            module_paths: Dict[str, Path] = {
                sibling.stem: sibling
                for sibling in path.parent.glob("*.py")
                if sibling.name != "__init__.py"
            }
            if path.parent == source_root:
                module_paths.update(
                    {
                        sibling.name: sibling / "__init__.py"
                        for sibling in path.parent.iterdir()
                        if sibling.is_dir()
                        and (sibling / "__init__.py").is_file()
                    }
                )
            actual_modules: set[str] = set(module_paths)
            module_docstring: str = (
                ast.get_docstring(module, clean=False) or ""
            )
            listed_modules: set[str] = set(
                re.findall(r"- :mod:`([^`]+)`", module_docstring)
            )
            listed_descriptions: Dict[str, str] = dict(
                re.findall(
                    r"(?m)^- :mod:`([^`]+)`\n    ([^\n]+)$",
                    module_docstring,
                )
            )
            if actual_modules != listed_modules:
                violations.append(
                    f"{path}: missing="
                    f"{sorted(actual_modules - listed_modules)} "
                    f"stale={sorted(listed_modules - actual_modules)}"
                )
            module_name: str
            sibling: Path
            for module_name, sibling in sorted(module_paths.items()):
                sibling_module: ast.Module = ast.parse(sibling.read_text())
                sibling_docstring: str = (
                    ast.get_docstring(sibling_module, clean=False) or ""
                )
                sibling_summary: str = sibling_docstring.splitlines()[0]
                if listed_descriptions.get(module_name) != sibling_summary:
                    violations.append(
                        f"{path}: submodule summary mismatch: {module_name}"
                    )
        self.assertEqual(violations, [])

    def test_public_api_uses_three_place_documentation(  # noqa: PLR0912
        self,
    ) -> None:
        """Keep exports and summaries synchronized in all three locations.

        The test confirms each public definition has an export. Each module and
        subpackage surface lists exactly the same names and summary sentences.

        Notes
        -----
        The test parses literal export lists and Sphinx routine listings. It
        compares defining, module, and subpackage summaries verbatim.
        """
        parsed_modules: Tuple[Tuple[Path, ast.Module], ...] = (
            self._production_modules()
        )
        module_records: Dict[
            Path, Tuple[ast.Module, set[str], Dict[str, str]]
        ] = {}
        violations: List[str] = []
        path: Path
        module: ast.Module
        name: str
        for path, module in parsed_modules:
            module_docstring: str = (
                ast.get_docstring(module, clean=False) or ""
            )
            exports: set[str] = self._literal_exports(module)
            listings: Dict[str, str] = self._routine_listing_summaries(
                module_docstring
            )
            module_records[path] = (module, exports, listings)
            if path.name == "__init__.py":
                continue
            public_definitions: Dict[str, str] = {
                node.name: (ast.get_docstring(node) or "").splitlines()[0]
                for node in module.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and not node.name.startswith("_")
            }
            missing_exports: set[str] = set(public_definitions) - exports
            if missing_exports:
                violations.append(
                    f"{path}: public definitions missing from __all__: "
                    f"{sorted(missing_exports)}"
                )
            if exports != set(listings):
                violations.append(
                    f"{path}: __all__/listing mismatch: "
                    f"missing={sorted(exports - set(listings))}, "
                    f"stale={sorted(set(listings) - exports)}"
                )
            for name in exports & set(listings) & set(public_definitions):
                if public_definitions[name] != listings[name]:
                    violations.append(f"{path}: summary mismatch: {name}")

        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        package_path: Path
        for package_path in sorted(source_root.iterdir()):
            init_path: Path = package_path / "__init__.py"
            if not package_path.is_dir() or init_path not in module_records:
                continue
            package_module: ast.Module
            package_exports: set[str]
            package_listings: Dict[str, str]
            package_module, package_exports, package_listings = module_records[
                init_path
            ]
            del package_module
            submodule_exports: set[str] = set()
            submodule_summaries: Dict[str, str] = {}
            for path, (_, exports, listings) in module_records.items():
                if path.parent == package_path and path.name != "__init__.py":
                    submodule_exports.update(exports)
                    submodule_summaries.update(listings)
            if package_exports != submodule_exports:
                violations.append(
                    f"{init_path}: package/module export mismatch: "
                    f"missing={sorted(submodule_exports - package_exports)}, "
                    f"extra={sorted(package_exports - submodule_exports)}"
                )
            if package_exports != set(package_listings):
                violations.append(
                    f"{init_path}: __all__/listing mismatch: "
                    "missing="
                    f"{sorted(package_exports - set(package_listings))}, "
                    f"stale={sorted(set(package_listings) - package_exports)}"
                )
            for name in (
                package_exports
                & set(package_listings)
                & set(submodule_summaries)
            ):
                if package_listings[name] != submodule_summaries[name]:
                    violations.append(f"{init_path}: summary mismatch: {name}")
        self.assertEqual(violations, [])

    def test_public_docstrings_follow_house_process_format(self) -> None:
        """Keep public source docstrings on the house process format.

        Extended Summary
        ----------------
        The test confirms functions and classes use untitled extended prose.
        Each public function explains its process in Notes or literal steps.

        Notes
        -----
        The test parses source docstrings and checks numbered bold logic steps.
        Each step needs a double-colon heading and an indented literal
        expression.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.name == "__init__.py":
                continue
            for node in module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) or node.name.startswith("_"):
                    continue
                docstring: str = ast.get_docstring(node) or ""
                if "\nExtended Summary\n" in docstring:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: titled extended "
                        "summary"
                    )
                summary_end: int = docstring.find("\n")
                see_position: int = docstring.find("\n:see:")
                extended_summary: str = docstring[
                    summary_end:see_position
                ].strip()
                if summary_end < 0 or see_position < 0 or not extended_summary:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: no untitled "
                        "extended summary before :see:"
                    )
                if isinstance(node, ast.ClassDef):
                    continue
                has_logic: bool = "\nImplementation Logic\n" in docstring
                has_notes: bool = "\nNotes\n" in docstring
                if not has_logic and not has_notes:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: no process section"
                    )
                    continue
                if not has_logic:
                    continue
                section_match: re.Match[str] | None = re.search(
                    r"(?ms)^Implementation Logic\n-+\n"
                    r"(.*?)(?=^[A-Z][A-Za-z ]+\n-+\n|\Z)",
                    docstring,
                )
                if section_match is None:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: malformed logic "
                        "section"
                    )
                    continue
                logic_section: str = section_match.group(1)
                step_headings: List[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*:+$", logic_section
                )
                valid_headings: List[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*::$", logic_section
                )
                literal_steps: List[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*::\n\n {7}\S",
                    logic_section,
                )
                if (
                    not valid_headings
                    or step_headings != valid_headings
                    or len(literal_steps) != len(valid_headings)
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: "
                        f"steps={len(step_headings)}, "
                        f"valid={len(valid_headings)}, "
                        f"literal={len(literal_steps)}"
                    )
        self.assertEqual(violations, [])

    def test_public_objects_have_symbol_owned_tests(self) -> None:
        """Require one reciprocal ``Test<Symbol>`` class per public object.

        The test confirms each public production object links to its exact
        symbol-owned class in the mirrored test module. That class links back.

        Notes
        -----
        The test normalizes underscores and capitalization for scientific
        abbreviations. It rejects generic multi-symbol test classes.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        source_root: Path = repository_root / "src/diffpes"
        tests_root: Path = repository_root / "tests/test_diffpes"
        violations: List[str] = []
        source_path: Path
        source_module: ast.Module
        node: ast.stmt
        for source_path, source_module in self._production_modules():
            if source_path.name == "__init__.py":
                continue
            relative_path: Path = source_path.relative_to(source_root)
            subpackage: str = relative_path.parts[0]
            test_path: Path = (
                tests_root / f"test_{subpackage}" / f"test_{source_path.name}"
            )
            test_classes: Dict[str, ast.ClassDef] = {}
            if test_path.is_file():
                test_module: ast.Module = ast.parse(
                    test_path.read_text(encoding="utf-8")
                )
                test_classes = {
                    node.name: node
                    for node in test_module.body
                    if isinstance(node, ast.ClassDef)
                }
            for node in source_module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) or node.name.startswith("_"):
                    continue
                source_docstring: str = ast.get_docstring(node) or ""
                targets: List[str] = re.findall(
                    r":see:\s+:class:`[^`]*\.(Test\w+)`",
                    source_docstring,
                )
                if len(targets) != 1:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"test targets={targets}"
                    )
                    continue
                target_name: str = targets[0]
                expected_normalized: str = "test" + re.sub(
                    r"[^a-z0-9]", "", node.name.lower()
                )
                actual_normalized: str = re.sub(
                    r"[^a-z0-9]", "", target_name.lower()
                )
                if actual_normalized != expected_normalized:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"target={target_name}"
                    )
                    continue
                test_class: ast.ClassDef | None = test_classes.get(target_name)
                if test_class is None:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"missing {test_path}:{target_name}"
                    )
                    continue
                class_docstring: str = ast.get_docstring(test_class) or ""
                reciprocal_name: str = f"diffpes.{subpackage}.{node.name}"
                if reciprocal_name not in class_docstring:
                    violations.append(
                        f"{test_path}:{test_class.lineno}:{target_name}: "
                        f"missing {reciprocal_name}"
                    )
        self.assertEqual(violations, [])

    def test_test_docstrings_specify_what_and_how(self) -> None:
        """Require complete reader-facing specifications on every test.

        The test confirms each test module has an extended summary. Every test
        callable has ``-> None``, extended prose, and how-focused Notes.

        Notes
        -----
        The test parses published test docstrings and reports missing
        structural parts. Semantic prose quality remains a review
        responsibility.
        """
        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._test_modules():
            module_docstring: str = ast.get_docstring(module) or ""
            if (
                path.name != "__init__.py"
                and len(
                    [
                        line
                        for line in module_docstring.splitlines()
                        if line.strip()
                    ]
                )
                < 2
            ):
                violations.append(f"{path}: module extended summary")
            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef) and node.name.startswith(
                    "Test"
                ):
                    class_docstring: str = ast.get_docstring(node) or ""
                    class_lines: List[str] = [
                        line
                        for line in class_docstring.splitlines()
                        if line.strip()
                    ]
                    if len(class_lines) < 2:
                        violations.append(
                            f"{path}:{node.lineno}:{node.name}: "
                            "class case scope"
                        )
                    continue
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or not node.name.startswith("test_"):
                    continue
                if not (
                    isinstance(node.returns, ast.Constant)
                    and node.returns.value is None
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: -> None"
                    )
                docstring: str = ast.get_docstring(node) or ""
                before_notes: str = docstring.split("Notes\n", 1)[0]
                if (
                    len(
                        [
                            line
                            for line in before_notes.splitlines()
                            if line.strip()
                        ]
                    )
                    < 2
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: extended summary"
                    )
                if "\nNotes\n" not in docstring:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: Notes"
                    )
        self.assertEqual(violations, [])

    def test_test_intermediates_are_annotated(self) -> None:
        """Require explicit types for intermediate variables in tests.

        The test confirms assignment, loop, context, walrus, and exception
        targets in each test callable carry a local type annotation.

        Notes
        -----
        The test excludes nested callables, legal ``nonlocal`` assignments, and
        the ``_`` name. It reports every other local target.
        """

        class TestAssignmentVisitor(ast.NodeVisitor):
            """Collect annotations and assignments in one test callable.

            The visitor isolates one function scope and records every local
            target that the annotation contract must classify.
            """

            def __init__(
                self,
                root: ast.FunctionDef | ast.AsyncFunctionDef,
            ) -> None:
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.annotated: set[str] = set()
                self.nonlocal_names: set[str] = set()
                self.assignments: List[Tuple[int, str]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression scopes from the outer contract."""
                del node

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Record a directly annotated local name."""
                if isinstance(node.target, ast.Name):
                    self.annotated.add(node.target.id)
                self.generic_visit(node)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                """Record names whose annotations belong to an outer scope."""
                self.nonlocal_names.update(node.names)

            def _record_target(self, target: ast.expr, line: int) -> None:
                """PRIVATE: Record names within one assignment-like target.

                Parameters
                ----------
                target : ast.expr
                    Assignment, loop, context, or walrus target
                    expression.
                line : int
                    Source line to attach to each recorded name.

                Notes
                -----
                Walks the target and appends every stored
                ``ast.Name`` except the throwaway ``_`` to the
                collected assignments.
                """
                candidate: ast.AST
                for candidate in ast.walk(target):
                    if (
                        isinstance(candidate, ast.Name)
                        and isinstance(candidate.ctx, ast.Store)
                        and candidate.id != "_"
                    ):
                        self.assignments.append((line, candidate.id))

            def visit_Assign(self, node: ast.Assign) -> None:
                """Record plain local-name assignment targets."""
                target: ast.expr
                for target in node.targets:
                    self._record_target(target, node.lineno)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                """Record an ordinary loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                """Record an asynchronous loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                """Record context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                """Record asynchronous context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                """Record an assignment-expression target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                """Record an exception-handler binding."""
                if node.name is not None and node.name != "_":
                    self.assignments.append((node.lineno, node.name))
                self.generic_visit(node)

        violations: List[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._test_tree_modules():
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: TestAssignmentVisitor = TestAssignmentVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}:{name}"
                    for line, name in visitor.assignments
                    if name not in visitor.annotated
                    and name not in visitor.nonlocal_names
                )
        self.assertEqual(violations, [])

    def test_public_symbols_have_one_owning_subpackage(self) -> None:
        """Forbid compatibility re-exports across subpackage surfaces.

        The test confirms a public name appears in one non-root subpackage
        ``__all__``. Moves cannot leave aliases or secondary import paths.

        Notes
        -----
        The test reads literal ``__all__`` entries from each subpackage. It
        reports names claimed by more than one owner.
        """
        owners: Dict[str, List[str]] = {}
        path: Path
        module: ast.Module
        node: ast.stmt
        entry: ast.expr
        for path, module in self._production_modules():
            if path.name != "__init__.py" or path.parent.name == "diffpes":
                continue
            for node in module.body:
                target: ast.expr | None = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                if (
                    not isinstance(target, ast.Name)
                    or target.id != "__all__"
                    or not isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    continue
                for entry in node.value.elts:
                    if isinstance(entry, ast.Constant) and isinstance(
                        entry.value, str
                    ):
                        owners.setdefault(entry.value, []).append(
                            path.parent.name
                        )
        violations: List[str] = [
            f"{name}:{sorted(subpackages)}"
            for name, subpackages in sorted(owners.items())
            if len(subpackages) > 1
        ]
        self.assertEqual(violations, [])


class TestStack(chex.TestCase):
    """Validate the differentiable runtime stack and its JAX contracts.

    The class covers import availability for Equinox, Optimistix, Lineax, and
    Optax. It covers float64 configuration and PyTree reconstruction.
    """

    def test_stack_imports(self) -> None:
        """Preserve stack imports, x64 precision, and PyTree structure.

        The test confirms that each selected solver and type library imports in
        the runtime. Scalar JAX arrays default to float64. A native Equinox
        module round-trips through JAX tree flattening.

        Notes
        -----
        The test imports runtime packages at module collection after diffpes.
        It constructs a scalar Equinox linear layer with a fixed key. The test
        checks the reconstructed module type and leaves exactly.
        """
        runtime_modules: Tuple[object, ...] = (
            eqx,
            optimistix,
            lineax,
            optax,
        )
        module_names: Tuple[str, ...] = tuple(
            module.__name__ for module in runtime_modules
        )
        precision_probe: Float[Array, ""] = jnp.zeros(())
        linear_module: eqx.Module = eqx.nn.Linear(
            "scalar",
            "scalar",
            key=jax.random.PRNGKey(0),
        )
        flattened: Tuple[List[Shaped[Array, "..."]], PyTreeDef] = (
            jax.tree_util.tree_flatten(linear_module)
        )
        leaves: List[Shaped[Array, "..."]]
        tree_definition: PyTreeDef
        leaves, tree_definition = flattened
        reconstructed: eqx.Module = jax.tree_util.tree_unflatten(
            tree_definition,
            leaves,
        )
        reconstructed_leaves: List[Shaped[Array, "..."]] = (
            jax.tree_util.tree_leaves(reconstructed)
        )

        chex.assert_equal(
            module_names,
            ("equinox", "optimistix", "lineax", "optax"),
        )
        chex.assert_equal(precision_probe.dtype, jnp.float64)
        self.assertIsInstance(reconstructed, eqx.Module)
        chex.assert_trees_all_equal(reconstructed_leaves, leaves)
