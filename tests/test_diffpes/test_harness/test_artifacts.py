"""Validate artifact writing and manifest records for executable experiments.

The tests cover safe paths, JSON and array files, previews, figures, carrier
storage, logs, and immutable-record conversion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from beartype.typing import Any, Dict

from diffpes.constants import AUTOMATON_PREVIEW_MAX_BYTES
from diffpes.harness import (
    artifact_path,
    artifact_record_as_dict,
    log_message,
    record_artifact,
    save_array_artifact,
    save_carrier_artifact,
    save_figure_artifact,
    save_image_artifact,
    save_json_artifact,
)
from diffpes.types import (
    ArtifactRecord,
    AutomatonContext,
    make_automaton_context,
    make_self_energy_model,
)


class TestArtifactPath:
    """Validate :func:`~diffpes.harness.artifact_path` containment.

    The case scope covers nested paths and traversal rejection.
    """

    def test_creates_nested_paths_below_the_output_root(
        self, tmp_path: Path
    ) -> None:
        """Create nested artifact paths below the output root.

        A safe requested name must create its parent directory inside the run.

        Notes
        -----
        Build one context and request a nested JSON artifact path.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        path: Path = artifact_path(context, "nested/metrics.json")

        assert path.parent.is_dir()
        assert path.parent.parent == tmp_path

    def test_rejects_a_path_that_escapes_the_output_root(
        self, tmp_path: Path
    ) -> None:
        """Reject a relative path that escapes the output root.

        Artifact writers must never create files above their supplied root.

        Notes
        -----
        Build one context and request a parent-directory traversal path.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )

        with pytest.raises(
            ValueError,
            match=r"artifact path escapes the output directory",
        ):
            artifact_path(context, "../outside.json")


class TestRecordArtifact:
    """Validate :func:`~diffpes.harness.record_artifact` manifest output.

    The case scope covers relative paths and preview-size behavior.
    """

    def test_records_a_small_file_with_a_base64_preview(
        self, tmp_path: Path
    ) -> None:
        """Record a small file with a base64 preview.

        A manifest record must retain the requested role and relative path.

        Notes
        -----
        Write one small text file below the context root before recording it.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        path: Path = artifact_path(context, "data/value.txt")
        path.write_text("value", encoding="utf-8")
        record: ArtifactRecord = record_artifact(
            context,
            path,
            role="value",
            mime="text/plain",
            preview=True,
        )

        assert record.role == "value"
        assert record.path == "data/value.txt"
        assert record.preview_b64

    def test_omits_a_preview_above_the_threshold(self, tmp_path: Path) -> None:
        """Omit a preview for a file above the byte threshold.

        A large artifact must retain an empty preview string in its manifest.

        Notes
        -----
        Write one file that exceeds the configured preview ceiling by one byte.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        path: Path = artifact_path(context, "large.bin")
        content: bytes = b"x" * (AUTOMATON_PREVIEW_MAX_BYTES + 1)
        path.write_bytes(content)
        record: ArtifactRecord = record_artifact(
            context,
            path,
            role="binary",
            mime="application/octet-stream",
            preview=True,
        )

        assert record.preview_b64 == ""


class TestSaveJsonArtifact:
    """Validate :func:`~diffpes.harness.save_json_artifact` output.

    The case scope covers sorted JSON writing and manifest records.
    """

    def test_saves_json_and_embeds_a_small_preview(
        self, tmp_path: Path
    ) -> None:
        """Save JSON data and embed a small base64 preview.

        The returned record must name an existing relative JSON file.

        Notes
        -----
        Save one metrics object below a temporary executable context root.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        record: ArtifactRecord = save_json_artifact(
            context,
            "metrics",
            {"z": 1.0, "a": 2.0},
        )
        payload: str = (tmp_path / record.path).read_text(encoding="utf-8")

        assert payload == '{"a": 2.0, "z": 1.0}'
        assert record.mime == "application/json"
        assert record.preview_b64


class TestSaveArrayArtifact:
    """Validate :func:`~diffpes.harness.save_array_artifact` output.

    The case scope covers compressed named-array archive creation.
    """

    def test_saves_named_arrays_in_a_compressed_archive(
        self, tmp_path: Path
    ) -> None:
        """Save named arrays in a compressed archive.

        The archive must expose each mapping key as one NumPy array name.

        Notes
        -----
        Save two small arrays and inspect the generated NPZ archive.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        record: ArtifactRecord = save_array_artifact(
            context,
            "arrays",
            {"left": np.array([1, 2]), "right": np.array([3])},
        )
        archive: Any
        with np.load(tmp_path / record.path) as archive:
            names: set[str] = set(archive.files)
            left: Any = archive["left"]

        assert names == {"left", "right"}
        assert left.tolist() == [1, 2]
        assert record.mime == "application/npz"


class TestSaveImageArtifact:
    """Validate :func:`~diffpes.harness.save_image_artifact` output.

    The case scope covers image rendering and intensity-scale validation.
    """

    def test_saves_a_log_scaled_png(self, tmp_path: Path) -> None:
        """Save a logarithmically scaled PNG image.

        The image writer must create a PNG record under the output root.

        Notes
        -----
        Render a positive two-by-two intensity array with logarithmic scaling.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        record: ArtifactRecord = save_image_artifact(
            context,
            "image",
            np.array([[0.0, 1.0], [2.0, 4.0]]),
            intensity_scale="log",
        )

        assert (tmp_path / record.path).is_file()
        assert record.mime == "image/png"

    def test_rejects_an_unknown_intensity_scale(self, tmp_path: Path) -> None:
        """Reject an unknown image intensity scale.

        The writer must reject a scale outside its three display modes.

        Notes
        -----
        Request an image with an unsupported scale name.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )

        with pytest.raises(
            ValueError,
            match=r"intensity_scale must be linear, sqrt, or log",
        ):
            save_image_artifact(
                context,
                "image.png",
                np.ones((2, 2)),
                intensity_scale="power",
            )


class TestSaveFigureArtifact:
    """Validate :func:`~diffpes.harness.save_figure_artifact` output.

    The case scope covers tight figure writing and resource closure.
    """

    def test_saves_and_closes_a_matplotlib_figure(
        self, tmp_path: Path
    ) -> None:
        """Save and close one Matplotlib figure.

        The writer must close the supplied figure after it stores the PNG.

        Notes
        -----
        Draw one line on a new figure and save it through the helper.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        figure: Any = plt.figure()
        axis: Any = figure.add_subplot(111)
        axis.plot([0.0, 1.0], [1.0, 0.0])
        figure_number: int = figure.number
        record: ArtifactRecord = save_figure_artifact(
            context,
            "figure",
            figure,
        )

        assert (tmp_path / record.path).is_file()
        assert not plt.fignum_exists(figure_number)


class TestSaveCarrierArtifact:
    """Validate :func:`~diffpes.harness.save_carrier_artifact` output.

    The case scope covers HDF5 persistence through the public I/O surface.
    """

    def test_saves_a_registered_diffpes_carrier(self, tmp_path: Path) -> None:
        """Save a registered diffpes carrier in HDF5.

        The writer must produce an HDF5 manifest record without a preview.

        Notes
        -----
        Save one constant self-energy carrier through the public wrapper.
        """
        context: AutomatonContext = make_automaton_context(
            tmp_path, 0, "example"
        )
        carrier: Any = make_self_energy_model(gamma=0.1)
        record: ArtifactRecord = save_carrier_artifact(
            context,
            "carrier",
            carrier,
            role="self_energy",
        )

        assert (tmp_path / record.path).is_file()
        assert record.mime == "application/x-hdf5"
        assert record.preview_b64 == ""


class TestLogMessage:
    """Validate :func:`~diffpes.harness.log_message` output control.

    The case scope covers standard-error prefixes and JSON suppression.
    """

    def test_writes_a_prefixed_message_unless_json_mode_is_set(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Write a prefixed message unless JSON mode suppresses it.

        A JSON-mode context must leave standard error unchanged.

        Notes
        -----
        Emit one normal message, clear capture, and emit one JSON-mode message.
        """
        normal: AutomatonContext = make_automaton_context(
            tmp_path / "normal", 0, "normal"
        )
        json_context: AutomatonContext = make_automaton_context(
            tmp_path / "json", 0, "json", json_mode=True
        )
        log_message(normal, "ready")
        normal_capture: Any = capsys.readouterr()
        log_message(json_context, "quiet")
        json_capture: Any = capsys.readouterr()

        assert normal_capture.err == "[normal] ready\n"
        assert json_capture.err == ""


class TestArtifactRecordAsDict:
    """Validate :func:`~diffpes.harness.artifact_record_as_dict` conversion.

    The case scope covers manifest field preservation.
    """

    def test_preserves_each_manifest_field(self) -> None:
        """Preserve each immutable manifest field in JSON data.

        The dictionary must keep role, MIME type, relative path, and preview.

        Notes
        -----
        Convert one explicit artifact record through the public helper.
        """
        record: ArtifactRecord = ArtifactRecord(
            role="array",
            mime="application/npz",
            path="data.npz",
            preview_b64="",
        )
        payload: Dict[str, Any] = artifact_record_as_dict(record)

        assert payload == {
            "role": "array",
            "mime": "application/npz",
            "path": "data.npz",
            "preview_b64": "",
        }
