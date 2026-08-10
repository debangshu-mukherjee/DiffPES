"""Verify paired tutorial and executable-documentation policy.

The tests cover pair discovery, source synchronization, stripped outputs,
content-keyed cache reuse, and hard execution failure.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import nbformat
import pytest
from beartype.typing import Any, Dict, List

from tests._tutorials import check_tutorial_pairs, strip_tutorial_outputs


class TestTutorialPairs:
    """Keep tutorial inputs synchronized and notebook outputs uncommitted."""

    def test_repository_pairs_are_synchronized_and_output_free(self) -> None:
        """Accept the registered tutorial pairs without defects.

        The repository fixture must keep every notebook synchronized with its
        reviewable percent script and free of committed outputs.

        Notes
        -----
        Run the public pair checker at the repository root and require an
        empty defect list.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        assert check_tutorial_pairs(repository_root) == []

    def test_pair_gate_rejects_drift_outputs_and_missing_sides(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject source drift, retained output, and an absent pair.

        The check exercises each pairing and version-control failure policy in
        an isolated directory.

        Notes
        -----
        Copy the registered pair, plant each defect, and match the checker or
        stripping result after every mutation.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        notebook_source: Path = (
            repository_root
            / "docs/source/tutorials/geometry-and-kinematics.ipynb"
        )
        script_source: Path = (
            repository_root / "tutorials/geometry-and-kinematics.py"
        )
        notebook_target: Path = (
            tmp_path / "docs/source/tutorials/geometry-and-kinematics.ipynb"
        )
        script_target: Path = tmp_path / "tutorials/geometry-and-kinematics.py"
        notebook_target.parent.mkdir(parents=True)
        script_target.parent.mkdir(parents=True)
        shutil.copyfile(notebook_source, notebook_target)
        shutil.copyfile(script_source, script_target)

        script_target.write_text(
            script_target.read_text() + "\n# planted drift\n"
        )
        defects: List[str] = check_tutorial_pairs(tmp_path)
        assert any("inputs differ" in defect for defect in defects)

        shutil.copyfile(script_source, script_target)
        notebook: Dict[str, Any] = json.loads(notebook_target.read_text())
        code_cell: Dict[str, Any] = next(
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        code_cell["execution_count"] = 1
        code_cell["outputs"] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["planted output\n"],
            }
        ]
        notebook_target.write_text(json.dumps(notebook))
        defects = check_tutorial_pairs(tmp_path)
        assert any("committed output" in defect for defect in defects)
        assert strip_tutorial_outputs(tmp_path) == [notebook_target]
        assert check_tutorial_pairs(tmp_path) == []

        script_target.unlink()
        defects = check_tutorial_pairs(tmp_path)
        assert any("missing percent script" in defect for defect in defects)


class TestMystNbExecutionPolicy:
    """Exercise cache reuse and hard failure with planted notebooks."""

    @staticmethod
    def _write_notebook(path: Path, source: str) -> Any:
        """PRIVATE: Write one stripped notebook with a single code cell.

        Parameters
        ----------
        path : Path
            Destination notebook path.
        source : str
            Python source for the only code cell.

        Returns
        -------
        loaded : Any
            Parsed notebook node loaded through ``nbformat``.
        """
        notebook: Dict[str, Any] = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "policy-cell",
                    "metadata": {},
                    "outputs": [],
                    "source": [source],
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path.write_text(json.dumps(notebook))
        loaded: Any = nbformat.read(path, as_version=4)
        return loaded

    def test_cache_reuses_success_and_execution_error_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reuse cached inputs and reject a planted execution error.

        The check distinguishes one successful execution plus cache reuse from
        a hard notebook failure.

        Notes
        -----
        Replace the execution seam with deterministic stubs and inspect the
        client metadata and raised ``ExecutionError``.
        """
        from myst_nb.core.config import NbParserConfig
        from myst_nb.core.execute.base import ExecutionError
        from myst_nb.core.execute.cache import NotebookClientCache

        cache: Path = tmp_path / "build/.jupyter_cache"
        notebook_path: Path = tmp_path / "success.ipynb"
        notebook: Any = self._write_notebook(
            notebook_path,
            "print('success')\n",
        )
        config: Any = NbParserConfig(
            execution_mode="cache",
            execution_cache_path=str(cache),
            execution_allow_errors=False,
            execution_raise_on_error=True,
        )
        executions: List[str] = []

        def execute_success(*args: Any, **kwargs: Any) -> SimpleNamespace:
            executions.append("success")
            return SimpleNamespace(err=None, time=0.01, exc_string=None)

        monkeypatch.setattr(
            "myst_nb.core.execute.cache.single_nb_execution",
            execute_success,
        )
        first: Any = NotebookClientCache(
            notebook,
            notebook_path,
            config,
            MagicMock(),
        )
        first.start_client()
        second: Any = NotebookClientCache(
            nbformat.read(notebook_path, as_version=4),
            notebook_path,
            config,
            MagicMock(),
        )
        second.start_client()
        assert executions == ["success"]
        assert first.exec_metadata["succeeded"] is True
        assert second.exec_metadata["succeeded"] is True

        failure_path: Path = tmp_path / "failure.ipynb"
        failure_notebook: Any = self._write_notebook(
            failure_path,
            "raise RuntimeError('planted')\n",
        )

        def execute_failure(*args: Any, **kwargs: Any) -> SimpleNamespace:
            error: RuntimeError = RuntimeError("planted")
            return SimpleNamespace(
                err=error,
                time=0.01,
                exc_string="RuntimeError: planted",
            )

        monkeypatch.setattr(
            "myst_nb.core.execute.cache.single_nb_execution",
            execute_failure,
        )
        failed: Any = NotebookClientCache(
            failure_notebook,
            failure_path,
            config,
            MagicMock(),
        )
        with pytest.raises(ExecutionError):
            failed.start_client()
