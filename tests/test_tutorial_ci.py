"""Verify the canonical tutorial and executable-documentation policy.

The tests cover notebook discovery, the output-free canon, the presence of
executed Markdown exports, content-keyed cache reuse, and hard execution
failure.
"""

import json
import shutil
from pathlib import Path

from beartype.typing import Any, Dict, List

from tests._tutorials import (
    MAXIMUM_TUTORIAL_FIGURES,
    check_tutorials,
    strip_tutorial_outputs,
)


class TestTutorialPolicy:
    """Keep canonical notebooks output-free with present Markdown exports.

    The cases exercise repository discovery, defect reporting, and stripping.
    """

    def test_repository_tutorials_are_output_free_with_exports(self) -> None:
        """Accept the registered canonical tutorials without defects.

        The repository must keep every canonical notebook stripped and give
        each one an executed Markdown export in the documentation tree.

        Notes
        -----
        Run the public tutorial checker at the repository root and require
        an empty defect list.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        assert check_tutorials(repository_root) == []

    def test_check_rejects_outputs_exports_figure_bounds_and_scripts(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject retained output, export defects, and a stray script.

        The check exercises each output and layout failure policy in an
        isolated directory.

        Notes
        -----
        Copy one nested registered notebook and its export, plant each defect,
        and match the checker or stripping result after every mutation.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        tutorial_relative_path: Path = Path(
            "01-intrinsic/01-simulate-an-arpes-spectrum.ipynb"
        )
        export_relative_path: Path = tutorial_relative_path.with_suffix(".md")
        notebook_source: Path = (
            repository_root / "tutorials" / tutorial_relative_path
        )
        export_source: Path = (
            repository_root / "docs/source/tutorials" / export_relative_path
        )
        asset_source: Path = export_source.with_name(
            f"{export_source.stem}_files"
        )
        notebook_target: Path = tmp_path / "tutorials" / tutorial_relative_path
        export_target: Path = (
            tmp_path / "docs/source/tutorials" / export_relative_path
        )
        asset_target: Path = export_target.with_name(
            f"{export_target.stem}_files"
        )
        notebook_target.parent.mkdir(parents=True)
        export_target.parent.mkdir(parents=True)
        shutil.copyfile(notebook_source, notebook_target)
        shutil.copyfile(export_source, export_target)
        shutil.copytree(asset_source, asset_target)
        assert check_tutorials(tmp_path) == []

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
        defects: List[str] = check_tutorials(tmp_path)
        assert any("committed output" in defect for defect in defects)
        assert strip_tutorial_outputs(tmp_path) == [notebook_target]
        assert check_tutorials(tmp_path) == []

        export_target.unlink()
        defects: List[str] = check_tutorials(tmp_path)
        assert any("missing markdown export" in defect for defect in defects)
        shutil.copyfile(export_source, export_target)

        export_text: str = export_target.read_text(encoding="utf-8")
        export_lines: List[str] = [
            line
            for line in export_text.splitlines()
            if f"{asset_target.name}/" not in line
        ]
        export_target.write_text("\n".join(export_lines), encoding="utf-8")
        defects = check_tutorials(tmp_path)
        assert any("requires at least 10" in defect for defect in defects)
        shutil.copyfile(export_source, export_target)

        seed_asset: Path = next(asset_target.rglob("*.png"))
        excess_assets: List[Path] = [
            asset_target / f"over-limit-{index}.png"
            for index in range(MAXIMUM_TUTORIAL_FIGURES + 1)
        ]
        excess_asset: Path
        for excess_asset in excess_assets:
            shutil.copyfile(seed_asset, excess_asset)
        excess_references: str = "".join(
            f"\n![png]({asset_target.name}/{excess_asset.name})\n"
            for excess_asset in excess_assets
        )
        export_target.write_text(
            export_text + excess_references,
            encoding="utf-8",
        )
        defects = check_tutorials(tmp_path)
        assert any("requires at most 25" in defect for defect in defects)
        shutil.rmtree(asset_target)
        shutil.copytree(asset_source, asset_target)
        shutil.copyfile(export_source, export_target)

        stray_script: Path = tmp_path / "tutorials/helper.py"
        stray_script.write_text("print('stray')\n")
        defects: List[str] = check_tutorials(tmp_path)
        assert any(
            "forbidden canonical tutorial file" in defect for defect in defects
        )
