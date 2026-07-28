"""Validate the two retained incoherent spectrum tiers.

The tests cover numerical output, metadata validation, and removed exports.
"""

import importlib

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import diffpes
from diffpes.simul import simulate_basic, simulate_novice
from diffpes.types import (
    OrbitalBasis,
    make_band_structure,
    make_orbital_basis,
    make_orbital_projection,
    make_simulation_params,
)


def _inputs() -> tuple[
    diffpes.types.BandStructure,
    diffpes.types.OrbitalProjection,
    OrbitalBasis,
]:
    """Build a one-atom Cu projection fixture."""
    eigenvalues: Array = jnp.asarray([[-0.4, 0.1], [-0.2, 0.3]])
    bands: diffpes.types.BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=jnp.zeros((2, 3)),
        fermi_energy=0.0,
    )
    projection: diffpes.types.OrbitalProjection = make_orbital_projection(
        projections=jnp.full((2, 2, 1, 9), 0.1)
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 9,
        n=(3,) * 9,
        l=(0, 1, 1, 1, 2, 2, 2, 2, 2),
        m=(0, -1, 0, 1, -2, -1, 0, 1, 2),
    )
    return bands, projection, basis


class TestSimulateNovice(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_novice`."""

    def test_shape_and_single_incoherent_reduction(self) -> None:
        """Return a finite spectrum proportional to projection weight.

        Doubling every retained projection probability must double the
        novice intensity.

        Notes
        -----
        Evaluate the baseline and doubled fixtures. Compare shapes, finite
        values, and the exact linear scaling law.
        """
        bands: diffpes.types.BandStructure
        projection: diffpes.types.OrbitalProjection
        bands, projection, _ = _inputs()
        params: diffpes.types.SimulationParams = make_simulation_params(
            fidelity=64
        )
        spectrum: diffpes.types.ArpesSpectrum = simulate_novice(
            bands, projection, params, 15.0
        )
        doubled: diffpes.types.OrbitalProjection = make_orbital_projection(
            projections=2.0 * projection.projections
        )
        doubled_spectrum: diffpes.types.ArpesSpectrum = simulate_novice(
            bands, doubled, params, 15.0
        )
        chex.assert_shape(spectrum.intensity, (2, 64))
        chex.assert_tree_all_finite(spectrum.intensity)
        chex.assert_trees_all_close(
            doubled_spectrum.intensity,
            2.0 * spectrum.intensity,
        )


class TestSimulateBasic(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_basic`."""

    def test_yeh_lindau_weighted_spectrum(self) -> None:
        """Use explicit subshell and atomic metadata.

        The basic tier must return finite intensity for a supported Cu basis.

        Notes
        -----
        Build a Cu subshell fixture at 200 eV. Evaluate the spectrum and check
        its array contract.
        """
        bands: diffpes.types.BandStructure
        projection: diffpes.types.OrbitalProjection
        basis: OrbitalBasis
        bands, projection, basis = _inputs()
        params: diffpes.types.SimulationParams = make_simulation_params(
            fidelity=64,
        )
        spectrum: diffpes.types.ArpesSpectrum = simulate_basic(
            bands,
            projection,
            params,
            basis,
            (29,),
            15.0,
            200.0,
        )
        chex.assert_shape(spectrum.intensity, (2, 64))
        chex.assert_tree_all_finite(spectrum.intensity)

    def test_rejects_misaligned_basis(self) -> None:
        """Reject a basis that does not cover all projection channels.

        A one-row basis cannot describe a nine-channel projection atom.

        Notes
        -----
        Build the undersized basis and call the basic tier inside a matching
        exception context.
        """
        bands: diffpes.types.BandStructure
        projection: diffpes.types.OrbitalProjection
        bands, projection, _ = _inputs()
        short_basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0,),
            n=(4,),
            l=(0,),
            m=(0,),
        )
        params: diffpes.types.SimulationParams = make_simulation_params()
        with pytest.raises(ValueError, match="one row"):
            simulate_basic(
                bands,
                projection,
                params,
                short_basis,
                (29,),
                15.0,
                200.0,
            )


class TestZeroLegacySpectrum:
    """Verify that deleted heuristic spectrum levels have no exports."""

    @pytest.mark.parametrize(
        "name",
        (
            "simulate_basicplus",
            "simulate_advanced",
            "simulate_expert",
            "simulate_soc",
            "simulate_basicplus_expanded",
            "simulate_advanced_expanded",
            "simulate_expert_expanded",
            "simulate_soc_expanded",
            "simulate_tb_radial",
            "dipole_matrix_elements",
            "heuristic_weights",
            "yeh_lindau_weights",
        ),
    )
    def test_removed_level_is_absent(self, name: str) -> None:
        """Keep deleted level names absent from the public package.

        Each former heuristic tier must remain outside the simulation surface.

        Notes
        -----
        Parameterize the retired names and query the public package with
        ``hasattr``.
        """
        assert not hasattr(diffpes.simul, name)

    def test_removed_forward_module_is_absent(self) -> None:
        """Keep the retired forward module absent.

        Importing the obsolete module path must fail without a compatibility
        shim.

        Notes
        -----
        Call the standard import mechanism inside a matching exception
        context.
        """
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("diffpes.simul.forward")

    @pytest.mark.parametrize(
        "name",
        (
            "SlaterParams",
            "make_slater_params",
            "ORBITAL_DIRS_NORMALIZED",
            "CROSS_SECTION_ENERGIES",
            "CROSS_SECTION_SIGMA_S",
            "CROSS_SECTION_SIGMA_P",
            "CROSS_SECTION_SIGMA_D",
        ),
    )
    def test_removed_type_symbol_is_absent(self, name: str) -> None:
        """Keep retired radial and toy-table names absent.

        The types surface must expose only the replacement radial carriers and
        authenticated table APIs.

        Notes
        -----
        Parameterize each retired name and query the public types package with
        ``hasattr``.
        """
        assert not hasattr(diffpes.types, name)

    @pytest.mark.parametrize(
        "name",
        (
            "dipole_matrix_element_single",
            "dipole_intensity_orbital",
            "dipole_intensities_all_orbitals",
        ),
    )
    def test_removed_dipole_symbol_is_absent(self, name: str) -> None:
        """Keep retired scalar dipole helpers absent.

        The mathematics surface must expose only the replacement channel
        algebra.

        Notes
        -----
        Parameterize each retired name and query the public mathematics
        package with ``hasattr``.
        """
        assert not hasattr(diffpes.maths, name)
