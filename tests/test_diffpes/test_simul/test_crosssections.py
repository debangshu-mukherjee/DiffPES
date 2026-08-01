"""Validate the authenticated Yeh--Lindau cross-section implementation.

Extended Summary
----------------
The tests check published entries, provenance, interpolation gradients,
domain rejection, and orbital-basis gathering.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray

from diffpes.simul import (
    yeh_lindau_cross_section,
    yeh_lindau_cross_section_table,
    yeh_lindau_orbital_weights,
)
from diffpes.types import OrbitalBasis, make_orbital_basis
from tests._gradients import assert_grad_matches_fd


class TestYehLindauCrossSectionTable:
    """Validate :func:`diffpes.simul.yeh_lindau_cross_section_table`.

    :see: :func:`~diffpes.simul.yeh_lindau_cross_section_table`
    """

    def test_published_rows_and_units(self) -> None:
        """Recover published C 2p values and the manifest unit.

        The checksum also authenticates the exact packed archive.

        Notes
        -----
        Read one table row, compare three nodes, and hash the adjacent data.
        """
        energies: Float[NDArray, " node"]
        sigma: Float[NDArray, " node"]
        slopes: Float[NDArray, " node"]
        energies, sigma, slopes = yeh_lindau_cross_section_table(6, 2, 1)
        indices: dict[float, int] = {
            float(value): index for index, value in enumerate(energies)
        }
        assert sigma[indices[21.2]] == pytest.approx(6.128)
        assert sigma[indices[40.8]] == pytest.approx(1.875)
        assert sigma[indices[80.0]] == pytest.approx(0.3266)
        assert np.all(np.isfinite(slopes[np.isfinite(sigma)]))

        manifest_path: Path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "diffpes"
            / "simul"
            / "data"
            / "yeh_lindau_1985.json"
        )
        manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["units"]["cross_section"] == "megabarn"
        assert manifest["data_license"] == "CC BY 4.0"
        assert manifest["digitisation_doi"].endswith("12389750.v3")
        archive_path: Path = manifest_path.with_suffix(".npz")
        archive_sha256: str = hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()
        assert archive_sha256 == manifest["archive_sha256"]

    def test_missing_entries_and_unsupported_key(self) -> None:
        """Preserve a real table gap and reject an absent subshell.

        This check prevents a logarithmic floor from inventing data.

        Notes
        -----
        Inspect the missing Li 1s node and request an absent hydrogen shell.
        """
        energies: Float[NDArray, " node"]
        sigma: Float[NDArray, " node"]
        slopes: Float[NDArray, " node"]
        energies, sigma, slopes = yeh_lindau_cross_section_table(3, 1, 0)
        index_200: int = int(np.flatnonzero(energies == 200.0)[0])
        assert np.isnan(sigma[index_200])
        assert np.isnan(slopes[index_200])
        with pytest.raises(ValueError, match="unsupported"):
            yeh_lindau_cross_section_table(1, 7, 3)

    def test_manifest_authenticates_archive_domains_and_provenance(
        self,
    ) -> None:
        """Validate every packed domain and its numerical authority.

        The manifest binds the generator and all positive interpolation runs.
        It scopes workbook replay separately from an unclaimed PDF transcription.

        Notes
        -----
        Recompute domains from the archive and replay each recorded spot check.
        """
        data_directory: Path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "diffpes"
            / "simul"
            / "data"
        )
        manifest_path: Path = data_directory / "yeh_lindau_1985.json"
        archive_path: Path = data_directory / "yeh_lindau_1985.npz"
        generator_path: Path = (
            Path(__file__).resolve().parents[3]
            / "tests"
            / "_reference_tools"
            / "generate_yeh_lindau_data.py"
        )
        manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        assert (
            manifest["generator_sha256"]
            == hashlib.sha256(generator_path.read_bytes()).hexdigest()
        )
        assert manifest["source_file_id"] == "22867790"
        assert manifest["source_filename"] == (
            "Excel_Yeh_Lindau_1985_PICS.xlsx"
        )
        assert str(manifest["source_url"]).endswith(
            str(manifest["source_file_id"])
        )

        archive: Any
        with np.load(archive_path) as archive:
            keys: Int[NDArray, "n_row 3"] = archive["keys"]
            offsets: Int[NDArray, " n_row_plus_one"] = archive["offsets"]
            energies: Float[NDArray, " n_packed"] = archive["photon_energy_ev"]
            sigma: Float[NDArray, " n_packed"] = archive["sigma_megabarn"]
        derived_domains: dict[str, list[list[float]]] = {}
        row_index: int
        key: Int[NDArray, " 3"]
        for row_index, key in enumerate(keys):
            start: int = int(offsets[row_index])
            stop: int = int(offsets[row_index + 1])
            row_energies: Float[NDArray, " node"] = energies[start:stop]
            positive: Bool[NDArray, " node"] = np.isfinite(
                sigma[start:stop]
            ) & (sigma[start:stop] > 0.0)
            intervals: list[list[float]] = []
            interval_start: int = 0
            while interval_start < positive.shape[0]:
                if not positive[interval_start]:
                    interval_start += 1
                    continue
                interval_stop: int = interval_start + 1
                while (
                    interval_stop < positive.shape[0]
                    and positive[interval_stop]
                ):
                    interval_stop += 1
                if interval_stop - interval_start >= 2:
                    intervals.append(
                        [
                            float(row_energies[interval_start]),
                            float(row_energies[interval_stop - 1]),
                        ]
                    )
                interval_start = interval_stop
            key_string: str = "-".join(str(int(value)) for value in key)
            derived_domains[key_string] = intervals
        assert manifest["supported_domains_ev"] == derived_domains

        spot_checks: list[dict[str, Any]] = manifest[
            "digitisation_replay_spot_checks"
        ]
        assert len(spot_checks) >= 4
        check: dict[str, Any]
        for check in spot_checks:
            table_energies: Float[NDArray, " node"]
            table_sigma: Float[NDArray, " node"]
            table_energies, table_sigma, _ = yeh_lindau_cross_section_table(
                int(check["atomic_number"]),
                int(check["n"]),
                int(check["l"]),
            )
            energy_index: int = int(
                np.flatnonzero(
                    table_energies == float(check["photon_energy_ev"])
                )[0]
            )
            assert table_sigma[energy_index] == pytest.approx(
                float(check["sigma_megabarn"]),
                rel=0.0,
                abs=0.0,
            )
            assert check["provenance"] == (
                "Regoutz-group source workbook replay"
            )

        primary_locator: dict[str, Any] = manifest["primary_source_locator"]
        assert primary_locator["page_range"] == "1-155"
        assert primary_locator["table_identifiers"] is None
        assert "versioned Figshare dataset" in str(
            primary_locator["table_identifier_status"]
        )
        assert primary_locator["independent_primary_spot_checks"] == []
        assert "not claimed" in str(
            primary_locator["independent_primary_spot_check_status"]
        )
        authority: dict[str, Any] = manifest["reference_authority"]
        assert "Figshare workbook" in str(authority["numerical_authority"])
        assert "file ID and SHA-256" in str(authority["authentication"])
        assert "internal peer review" in str(authority["review"])
        assert str(authority["project_url"]).startswith("https://")
        assert "no claim" in str(authority["scope"])

        authority_directory: Path = (
            Path(__file__).resolve().parents[1]
            / "_reference_data"
            / "plan06_yeh_lindau_authority"
        )
        figshare_path: Path = (
            authority_directory / "plan06_figshare_12389750_v3.json"
        )
        project_path: Path = (
            authority_directory / "plan06_regoutz_cross_sections.html"
        )
        assert (
            hashlib.sha256(figshare_path.read_bytes()).hexdigest()
            == authority["figshare_metadata_sha256"]
        )
        assert (
            hashlib.sha256(project_path.read_bytes()).hexdigest()
            == authority["project_page_sha256"]
        )
        figshare: dict[str, Any] = json.loads(
            figshare_path.read_text(encoding="utf-8")
        )
        assert figshare["doi"] == manifest["digitisation_doi"]
        assert figshare["license"]["name"] == manifest["data_license"]
        assert str(figshare["files"][0]["id"]) == manifest["source_file_id"]
        assert "manually mined" in str(figshare["description"])
        project_html: str = project_path.read_text(encoding="utf-8")
        assert "internal peer review process" in project_html
        assert "very grateful to Prof. Lindau" in project_html


class TestYehLindauCrossSection(chex.TestCase):
    """Validate :func:`diffpes.simul.yeh_lindau_cross_section`.

    :see: :func:`~diffpes.simul.yeh_lindau_cross_section`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_registered_parity_battery(self) -> None:
        """Match exact C 2p, O 2p, and Cu 3d table nodes.

        Both eager and compiled paths replay the registered parity battery.

        Notes
        -----
        Evaluate three photon energies for each registered subshell.
        """
        expected: tuple[
            tuple[
                tuple[int, int, int],
                tuple[float, float, float],
            ],
            ...,
        ] = (
            ((6, 2, 1), (6.128, 1.875, 0.3266)),
            ((8, 2, 1), (10.67, 6.816, 2.064)),
            ((29, 3, 2), (7.553, 9.934, 8.712)),
        )
        key: tuple[int, int, int]
        values: tuple[float, float, float]
        for key, values in expected:
            function: Callable[[float], jax.Array] = self.variant(
                lambda energy: yeh_lindau_cross_section(energy, *key)
            )
            actual: jax.Array = jnp.stack(
                tuple(function(energy) for energy in (21.2, 40.8, 80.0))
            )
            chex.assert_trees_all_close(
                actual,
                jnp.asarray(values),
                rtol=2e-14,
                atol=2e-14,
            )

    def test_gradient_matches_fd_and_old_step_false_control(self) -> None:
        """Verify the smooth energy derivative is finite and nonzero.

        A hard energy-step heuristic fails the nonzero-gradient assertion.

        Notes
        -----
        Compare automatic differentiation with the shared finite-difference harness.
        """

        def function(energy: jax.Array) -> jax.Array:
            result: jax.Array = yeh_lindau_cross_section(energy, 6, 2, 1)
            return result

        point: jax.Array = jnp.asarray(55.0, dtype=jnp.float64)
        assert_grad_matches_fd(function, point, atol=1e-9)
        gradient: jax.Array = jax.grad(function)(point)
        assert jnp.isfinite(gradient)
        assert abs(float(gradient)) > 1e-4

    def test_no_extrapolation_or_gap_crossing(self) -> None:
        """Reject values beyond a row and inside a missing-data gap.

        Runtime checks preserve the scoped domain under compilation.

        Notes
        -----
        Query beyond carbon and at the missing Li 1s 200 eV node.
        """
        checked: Callable[..., jax.Array] = eqx.filter_jit(
            yeh_lindau_cross_section
        )
        with pytest.raises(Exception, match="positive Yeh--Lindau intervals"):
            checked(jnp.asarray(20_000.0), 6, 2, 1)
        with pytest.raises(Exception, match="positive Yeh--Lindau intervals"):
            checked(jnp.asarray(200.0), 3, 1, 0)


class TestYehLindauOrbitalWeights(chex.TestCase):
    """Validate :func:`diffpes.simul.yeh_lindau_orbital_weights`.

    :see: :func:`~diffpes.simul.yeh_lindau_orbital_weights`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basis_gather(self) -> None:
        """Verify atom-row and subshell mapping onto orbitals.

        Repeated carbon orbitals share one subshell cross section.

        Notes
        -----
        Build a three-orbital carbon/oxygen basis and gather at 40.8 eV.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0, 1),
            n=(2, 2, 2),
            l=(1, 1, 1),
            m=(-1, 0, 1),
        )
        function: Callable[[float], jax.Array] = self.variant(
            lambda energy: yeh_lindau_orbital_weights(
                energy,
                basis,
                (6, 8),
            )
        )
        weights: jax.Array = function(40.8)
        chex.assert_shape(weights, (3,))
        chex.assert_trees_all_close(
            weights,
            jnp.asarray([1.875, 1.875, 6.816]),
            rtol=2e-14,
            atol=2e-14,
        )

    def test_atomic_number_coverage(self) -> None:
        """Reject a basis whose atom mapping exceeds the supplied tuple.

        Static validation prevents an accidental element assignment.

        Notes
        -----
        Reference atom row one while supplying only atom row zero.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(1,),
            n=(2,),
            l=(1,),
            m=(0,),
        )
        with pytest.raises(ValueError, match="does not cover"):
            yeh_lindau_orbital_weights(40.8, basis, (6,))
