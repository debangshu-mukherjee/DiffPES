"""Generate the packaged Yeh--Lindau cross-section dataset.

The input is the Regoutz-group digitisation of the Yeh--Lindau (1985)
tables.  This script deliberately uses only the Python standard library to
read the XLSX container so that regenerating the data does not require an
Excel dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Bool, Float64, Int16, Int32
from numpy.typing import NDArray

_MAIN_NS: str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS: str = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_PACKAGE_REL_NS: str = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
_ORBITAL_RE: re.Pattern[str] = re.compile(r"^([1-7])([spdf])$")
_L_BY_LETTER: Dict[str, int] = {"s": 0, "p": 1, "d": 2, "f": 3}
_SOURCE_URL: str = "https://ndownloader.figshare.com/files/22867790"
_SOURCE_DOI: str = "10.6084/m9.figshare.12389750.v3"
_PAPER_DOI: str = "10.1016/0092-640X(85)90016-6"
_SOURCE_FILE_ID: str = "22867790"
_SOURCE_FILENAME: str = "Excel_Yeh_Lindau_1985_PICS.xlsx"
_MIN_INTERPOLATION_NODES: int = 2
_DIGITISATION_REPLAY_SPOT_CHECKS: Tuple[
    Tuple[int, int, int, float, float], ...
] = (
    (6, 2, 1, 21.2, 6.128),
    (6, 2, 1, 40.8, 1.875),
    (8, 2, 1, 40.8, 6.816),
    (29, 3, 2, 80.0, 8.712),
)


def _cell_column(reference: str) -> int:
    """PRIVATE: Return the zero-based column of an XLSX cell reference.

    Parameters
    ----------
    reference : str
        Cell reference such as ``"B12"``.

    Returns
    -------
    column : int
        Zero-based column index.

    Notes
    -----
    The letters form a bijective base-26 number (``A``=1 through
    ``Z``=26); the digits drop out and one subtracts to zero-based.
    """
    letters: str = "".join(
        character for character in reference if character.isalpha()
    )
    column: int = 0
    character: str
    for character in letters:
        column = column * 26 + ord(character.upper()) - ord("A") + 1
    zero_based_column: int = column - 1
    return zero_based_column


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    """PRIVATE: Read the workbook shared-string table.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open XLSX container.

    Returns
    -------
    strings : List[str]
        Shared strings in table order.

    Notes
    -----
    Each shared-string item concatenates the text of its ``t`` nodes,
    which joins rich-text runs into one plain string.
    """
    root: ET.Element = ET.fromstring(  # noqa: S314 - authenticated input
        archive.read("xl/sharedStrings.xml")
    )
    tag: str = f"{{{_MAIN_NS}}}t"
    strings: List[str] = [
        "".join(node.text or "" for node in item.iter(tag)) for item in root
    ]
    return strings


def _cell_value(
    cell: ET.Element,
    shared_strings: List[str],
) -> str | None:
    """PRIVATE: Return one cell value with shared strings resolved.

    Parameters
    ----------
    cell : ET.Element
        Worksheet ``c`` cell element.
    shared_strings : List[str]
        Workbook shared-string table.

    Returns
    -------
    value : str | None
        The cell text, the resolved shared string for type ``s``
        cells, or ``None`` for empty cells.

    Notes
    -----
    Only the ``v`` child carries the stored value; a missing node or
    missing text means the cell is blank.
    """
    value_node: ET.Element | None = cell.find(f"{{{_MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        value: str | None = None
        return value
    value = value_node.text
    if cell.attrib.get("t") == "s":
        value = shared_strings[int(value)]
    return value


def _pchip_slopes(
    x_values: Float64[NDArray, " n_node"],
    y_values: Float64[NDArray, " n_node"],
) -> Float64[NDArray, " n_node"]:
    """PRIVATE: Compute shape-preserving cubic Hermite derivatives.

    Parameters
    ----------
    x_values : Float64[NDArray, " n_node"]
        Strictly increasing abscissas (log photon energy).
    y_values : Float64[NDArray, " n_node"]
        Ordinates at the abscissas (log cross section).

    Returns
    -------
    slopes : Float64[NDArray, " n_node"]
        PCHIP node derivatives.

    Implementation Logic
    --------------------
    Two-node runs return the secant. Interior nodes use the weighted harmonic
    mean of adjacent secants. Set them to zero at extrema or sign changes.
    This follows the Fritsch--Carlson monotonicity rule. The endpoint formula
    uses the non-centered
    three-point stencil, clipped to zero on sign disagreement and to
    three times the edge secant on overshoot.
    """
    count: int = len(x_values)
    intervals: Float64[NDArray, " n_interval"] = np.diff(x_values)
    secants: Float64[NDArray, " n_interval"] = np.diff(y_values) / intervals
    slopes: Float64[NDArray, " n_node"] = np.empty(count, dtype=np.float64)
    if count == _MIN_INTERPOLATION_NODES:
        slopes[:] = secants[0]
        return slopes

    left: np.float64 = (
        (2.0 * intervals[0] + intervals[1]) * secants[0]
        - intervals[0] * secants[1]
    ) / (intervals[0] + intervals[1])
    if np.sign(left) != np.sign(secants[0]):
        left = 0.0
    elif np.sign(secants[0]) != np.sign(secants[1]) and abs(left) > abs(
        3.0 * secants[0]
    ):
        left = 3.0 * secants[0]
    slopes[0] = left

    index: int
    for index in range(1, count - 1):
        previous: np.float64 = secants[index - 1]
        following: np.float64 = secants[index]
        if (
            previous == 0.0
            or following == 0.0
            or np.sign(previous) != np.sign(following)
        ):
            slopes[index] = 0.0
        else:
            first_weight: np.float64 = (
                2.0 * intervals[index] + intervals[index - 1]
            )
            second_weight: np.float64 = (
                intervals[index] + 2.0 * intervals[index - 1]
            )
            slopes[index] = (first_weight + second_weight) / (
                first_weight / previous + second_weight / following
            )

    right: np.float64 = (
        (2.0 * intervals[-1] + intervals[-2]) * secants[-1]
        - intervals[-1] * secants[-2]
    ) / (intervals[-1] + intervals[-2])
    if np.sign(right) != np.sign(secants[-1]):
        right = 0.0
    elif np.sign(secants[-1]) != np.sign(secants[-2]) and abs(right) > abs(
        3.0 * secants[-1]
    ):
        right = 3.0 * secants[-1]
    slopes[-1] = right
    return slopes


def _workbook_rows(  # noqa: PLR0915 -- XLSX relation and sheet replay.
    source: Path,
) -> List[
    Tuple[
        Tuple[int, int, int],
        Float64[NDArray, " n_node"],
        Float64[NDArray, " n_node"],
    ]
]:
    """PRIVATE: Extract subshell rows with missing and zero entries kept.

    Implementation Logic
    --------------------
    Every element sheet after the first resolves through the workbook
    relationships. Map ``ns/np/nd/nf`` header columns to ``(n, l)``. Read
    energies and values from the first 16 data rows. Drop subshells with fewer
    than two positive finite values.

    Parameters
    ----------
    source : Path
        Authenticated Regoutz-group XLSX workbook.

    Returns
    -------
    rows : List[Tuple[Tuple[int, int, int], Float64[NDArray, " n_node"],
        Float64[NDArray, " n_node"]]]
        Sorted ``(Z, n, l)`` keys with photon energies in eV and cross
        sections in megabarn; blank cells stay ``NaN``.

    Raises
    ------
    ValueError
        If the XLSX workbook carries no sheets element.
    """
    rows: List[
        Tuple[
            Tuple[int, int, int],
            Float64[NDArray, " n_node"],
            Float64[NDArray, " n_node"],
        ]
    ] = []
    archive: zipfile.ZipFile
    with zipfile.ZipFile(source) as archive:
        strings: List[str] = _shared_strings(archive)
        workbook: ET.Element = ET.fromstring(  # noqa: S314
            archive.read("xl/workbook.xml")
        )
        relationships: ET.Element = ET.fromstring(  # noqa: S314
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets: Dict[str, str] = {
            node.attrib["Id"]: node.attrib["Target"]
            for node in relationships.findall(
                f"{{{_PACKAGE_REL_NS}}}Relationship"
            )
        }
        sheets: ET.Element | None = workbook.find(f"{{{_MAIN_NS}}}sheets")
        if sheets is None:
            raise ValueError("XLSX workbook has no sheets")

        sheet: ET.Element
        for sheet in list(sheets)[1:]:
            name: str = sheet.attrib["name"]
            atomic_number: int = int(name.split("_", maxsplit=1)[0])
            relationship_id: str = sheet.attrib[f"{{{_REL_NS}}}id"]
            worksheet_path: str = "xl/" + relationship_targets[relationship_id]
            worksheet: ET.Element = ET.fromstring(  # noqa: S314
                archive.read(worksheet_path)
            )
            sheet_data: ET.Element | None = worksheet.find(
                f".//{{{_MAIN_NS}}}sheetData"
            )
            if sheet_data is None:
                continue
            xml_rows: List[ET.Element] = list(sheet_data)
            if len(xml_rows) < _MIN_INTERPOLATION_NODES:
                continue

            headers: Dict[int, Tuple[int, int]] = {}
            cell: ET.Element
            for cell in xml_rows[0]:
                raw_header: str | None = _cell_value(cell, strings)
                if raw_header is None:
                    continue
                match: re.Match[str] | None = _ORBITAL_RE.fullmatch(
                    raw_header.strip()
                )
                if match is None:
                    continue
                principal: int = int(match.group(1))
                angular: int = _L_BY_LETTER[match.group(2)]
                headers[_cell_column(cell.attrib["r"])] = (principal, angular)

            energies: List[float] = []
            column_values: Dict[int, List[float]] = {
                column: [] for column in headers
            }
            xml_row: ET.Element
            for xml_row in xml_rows[1:17]:
                values_by_column: Dict[int, str | None] = {
                    _cell_column(cell.attrib["r"]): _cell_value(cell, strings)
                    for cell in xml_row
                }
                energy: str | None = values_by_column.get(0)
                if energy is None:
                    continue
                energies.append(float(energy))
                column: int
                for column in headers:
                    raw_value: str | None = values_by_column.get(column)
                    column_values[column].append(
                        np.nan if raw_value is None else float(raw_value)
                    )

            energy_array: Float64[NDArray, " n_node"] = np.asarray(
                energies, dtype=np.float64
            )
            principal: int
            angular: int
            for column, (principal, angular) in headers.items():
                sigma: Float64[NDArray, " n_node"] = np.asarray(
                    column_values[column], dtype=np.float64
                )
                positive: Bool[NDArray, " n_node"] = np.isfinite(sigma) & (
                    sigma > 0.0
                )
                if np.count_nonzero(positive) < _MIN_INTERPOLATION_NODES:
                    continue
                rows.append(
                    (
                        (atomic_number, principal, angular),
                        energy_array,
                        sigma,
                    )
                )
    rows.sort(key=lambda item: item[0])
    return rows


def generate(source: Path, output_directory: Path) -> None:
    """Generate the compressed cross-section archive and provenance.

    Parameters
    ----------
    source : Path
        Authenticated Regoutz-group XLSX workbook.
    output_directory : Path
        Destination directory for the NPZ archive and JSON manifest.

    Notes
    -----
    The generator preserves missing values and contiguous positive domains.
    It computes PCHIP slopes in logarithmic coordinates for each domain.
    """
    rows: List[
        Tuple[
            Tuple[int, int, int],
            Float64[NDArray, " n_node"],
            Float64[NDArray, " n_node"],
        ]
    ] = _workbook_rows(source)
    keys: Int16[NDArray, "n_subshell 3"] = np.asarray(
        [row[0] for row in rows], dtype=np.int16
    )
    offsets: List[int] = [0]
    energy_parts: List[Float64[NDArray, " n_node"]] = []
    sigma_parts: List[Float64[NDArray, " n_node"]] = []
    slope_parts: List[Float64[NDArray, " n_node"]] = []
    domains: Dict[str, List[List[float]]] = {}
    key: Tuple[int, int, int]
    energies: Float64[NDArray, " n_node"]
    sigmas: Float64[NDArray, " n_node"]
    for key, energies, sigmas in rows:
        log_energies: Float64[NDArray, " n_node"] = np.log(energies)
        positive: Bool[NDArray, " n_node"] = np.isfinite(sigmas) & (
            sigmas > 0.0
        )
        slopes: Float64[NDArray, " n_node"] = np.full_like(sigmas, np.nan)
        intervals: List[List[float]] = []
        start: int = 0
        while start < len(sigmas):
            if not positive[start]:
                start += 1
                continue
            stop: int = start + 1
            while stop < len(sigmas) and positive[stop]:
                stop += 1
            if stop - start >= _MIN_INTERPOLATION_NODES:
                slopes[start:stop] = _pchip_slopes(
                    log_energies[start:stop],
                    np.log(sigmas[start:stop]),
                )
                intervals.append(
                    [float(energies[start]), float(energies[stop - 1])]
                )
            start = stop
        energy_parts.append(energies)
        sigma_parts.append(sigmas)
        slope_parts.append(slopes)
        offsets.append(offsets[-1] + len(energies))
        domains[f"{key[0]}-{key[1]}-{key[2]}"] = intervals

    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path: Path = output_directory / "yeh_lindau_1985.npz"
    offset_array: Int32[NDArray, " n_offset"] = np.asarray(
        offsets, dtype=np.int32
    )
    photon_energy: Float64[NDArray, " n_value"] = np.concatenate(energy_parts)
    cross_section: Float64[NDArray, " n_value"] = np.concatenate(sigma_parts)
    log_slopes: Float64[NDArray, " n_value"] = np.concatenate(slope_parts)
    np.savez_compressed(
        archive_path,
        keys=keys,
        offsets=offset_array,
        photon_energy_ev=photon_energy,
        sigma_megabarn=cross_section,
        log_slopes=log_slopes,
    )
    source_sha256: str = hashlib.sha256(source.read_bytes()).hexdigest()
    archive_sha256: str = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    generator_sha256: str = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest: Dict[str, Any] = {
        "archive_sha256": archive_sha256,
        "data_license": "CC BY 4.0",
        "digitisation_method": {
            "input_format": "XLSX workbook",
            "row_scope": (
                "the first 16 photon-energy rows from each element sheet"
            ),
            "subshell_mapping": (
                "sheet atomic number plus workbook ns/np/nd/nf headers"
            ),
            "missing_value_policy": (
                "blank cells become NaN; zeros remain zero"
            ),
            "interpolation_preparation": (
                "split contiguous positive runs and compute PCHIP slopes in "
                "log(sigma_megabarn) versus log(photon_energy_ev)"
            ),
        },
        "digitisation_replay_spot_checks": [
            {
                "atomic_number": atomic_number,
                "n": principal,
                "l": angular,
                "photon_energy_ev": energy,
                "sigma_megabarn": sigma,
                "provenance": "Regoutz-group source workbook replay",
            }
            for atomic_number, principal, angular, energy, sigma in (
                _DIGITISATION_REPLAY_SPOT_CHECKS
            )
        ],
        "digitisation_doi": _SOURCE_DOI,
        "generator": "tests/_reference_tools/generate_yeh_lindau_data.py",
        "generator_sha256": generator_sha256,
        "interpolation": (
            "PCHIP in log(sigma_megabarn) versus log(photon_energy_ev)"
        ),
        "paper_doi": _PAPER_DOI,
        "reference_authority": {
            "numerical_authority": (
                "versioned Regoutz-group Figshare workbook manually mined "
                "from the Yeh-Lindau tabulated data"
            ),
            "authentication": (
                "Figshare DOI, immutable file ID and SHA-256; the publisher "
                "DOI binds the cited primary paper"
            ),
            "review": (
                "Regoutz-group internal peer review; the project page records "
                "Prof. Lindau's agreement to make the dataset available"
            ),
            "project_url": (
                "https://regoutzgroup.org/research/cross-sections/"
            ),
            "figshare_metadata_sha256": (
                "c908a3c855ffe98dabd4660fa4d3c17849ac0b9563c26c7bde0197026"
                "a7bda44"
            ),
            "project_page_sha256": (
                "2e4c3cc0dbb73cecced5d8608fa44286ac444636e8d9142f4bbe4b042"
                "d236703"
            ),
            "scope": (
                "tabulated cross-section values only; no claim of an "
                "independent cell-by-cell transcription from a paper PDF"
            ),
        },
        "primary_source_locator": {
            "journal": "Atomic Data and Nuclear Data Tables",
            "volume": 32,
            "page_range": "1-155",
            "table_identifiers": None,
            "table_identifier_status": (
                "not published in the authenticated workbook metadata; "
                "cell-level authority is the versioned Figshare dataset"
            ),
            "independent_primary_spot_checks": [],
            "independent_primary_spot_check_status": (
                "not claimed; executable spot checks replay the authenticated "
                "versioned workbook"
            ),
        },
        "source_file_id": _SOURCE_FILE_ID,
        "source_filename": _SOURCE_FILENAME,
        "source_sha256": source_sha256,
        "source_url": _SOURCE_URL,
        "supported_domains_ev": domains,
        "units": {"photon_energy": "eV", "cross_section": "megabarn"},
    }
    manifest_path: Path = output_directory / "yeh_lindau_1985.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse command-line arguments and generate the package data.

    Notes
    -----
    The command requires the authenticated XLSX source path. It writes to the
    packaged simulation-data directory unless the caller changes the target.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("src/diffpes/simul/data"),
    )
    arguments: argparse.Namespace = parser.parse_args()
    generate(arguments.source, arguments.output_directory)


if __name__ == "__main__":
    main()
