"""Validate projected band-scatter plotting utilities.

The tests cover presets, projected-band rendering, k-path annotation,
carrier validation, and plotting edge cases.
"""

import chex
import equinox as eqx
import jax.numpy as jnp
import matplotlib
import pytest
from beartype.typing import List, Tuple
from jaxtyping import Array, Float64

import diffpes

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from diffpes.inout import (
    list_band_scatter_presets,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
)
from diffpes.types import (
    BandStructure,
    OrbitalProjection,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
)


def _make_band_and_projection(
    nk: int = 12,
    nb: int = 3,
    na: int = 2,
) -> Tuple[BandStructure, OrbitalProjection]:
    """PRIVATE: Build minimal band/projection inputs for band-scatter tests.

    Parameters
    ----------
    nk : int, optional
        Number of k-points along the linear path. Default 12.
    nb : int, optional
        Number of bands. Default 3.
    na : int, optional
        Number of atoms in the projection. Default 2.

    Returns
    -------
    result : Tuple[BandStructure, OrbitalProjection]
        A band structure with ``fermi_energy=0.15`` eV on a straight
        k-path, plus an orbital projection of shape
        ``(nk, nb, na, 9)``. The projection carries p-orbital weight
        and a spin array whose dominant channel flips at ``nk // 2``.

    Notes
    -----
    Uses evenly spaced eigenvalues in eV between -1.2 and 0.8 so the
    scatter presets have deterministic colors and marker sizes.
    """
    eigen: Float64[Array, "nk nb"] = jnp.linspace(
        -1.2, 0.8, nk * nb, dtype=jnp.float64
    ).reshape(nk, nb)
    kx: Float64[Array, " nk"] = jnp.linspace(0.0, 1.0, nk, dtype=jnp.float64)
    kpoints: Float64[Array, "nk 3"] = jnp.stack(
        [kx, jnp.zeros_like(kx), jnp.zeros_like(kx)],
        axis=1,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigen,
        kpoints=kpoints,
        fermi_energy=0.15,
    )

    projections: Float64[Array, "nk nb na 9"] = (
        jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.05
    )
    projections = projections.at[..., 1:4].set(0.2)
    spin: Float64[Array, "nk nb na 6"] = jnp.zeros(
        (nk, nb, na, 6), dtype=jnp.float64
    )
    spin = spin.at[: nk // 2, ..., 4].set(0.2)
    spin = spin.at[nk // 2 :, ..., 5].set(0.3)
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections=projections,
        spin=spin,
    )
    result: Tuple[BandStructure, OrbitalProjection] = (
        bands,
        orbital_projection,
    )
    return result


class TestListBandScatterPresets(chex.TestCase):
    """Validate :func:`~diffpes.inout.list_band_scatter_presets`.

    Covers the stable public names for orbital, spin, and orbital-angular-
    momentum scatter modes.

    :see: :func:`~diffpes.inout.list_band_scatter_presets`
    """

    def test_returns_each_preset_family(self) -> None:
        """Return at least one name from each supported preset family.

        The result must expose orbital, signed-spin, and OAM choices so callers
        can build selection controls without reading private tables.

        Notes
        -----
        The test calls the listing function once and checks representative
        public names in the returned immutable tuple.
        """
        presets: Tuple[str, ...]

        presets = list_band_scatter_presets()
        assert "p" in presets
        assert "spin_z" in presets
        assert "oam_total" in presets


class TestPlotBandScatterPreset(chex.TestCase):
    """Validate projected-band scatter plotting presets.

    :see: :func:`~diffpes.inout.plot_band_scatter_preset`
    """

    def test_lists_presets(self) -> None:
        """list_band_scatter_presets returns known keys.

        The test establishes the preset-listing contract with the concrete
        values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        presets: Tuple[str, ...]

        presets = list_band_scatter_presets()
        assert "p" in presets
        assert "d" in presets
        assert "spin_z" in presets
        assert "oam_total" in presets

    def test_orbital_preset_plot(self) -> None:
        """Verify the orbital preset scatter for each k-point and band.

        The test establishes the orbital-preset contract with the concrete
        values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        bands, orb = _make_band_and_projection(nk=10, nb=4, na=2)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 40)
        chex.assert_equal(ax.get_xlabel(), "Momentum (k)")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        plt.close(fig)

    def test_signed_spin_preset_with_colorbar(self) -> None:
        """Verify a signed-spin preset with a color bar.

        The test establishes the signed-spin color-bar contract with the
        concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure

        bands, orb = _make_band_and_projection(nk=8, nb=2, na=2)
        fig, _, _ = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="spin_z",
            colorbar=True,
        )

        chex.assert_equal(len(fig.axes), 2)
        plt.close(fig)

    def test_spin_preset_requires_spin_data(self) -> None:
        """Verify that spin presets require a spin field.

        The test establishes the spin-data requirement with the concrete
        values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        no_spin: diffpes.types.OrbitalProjection

        bands, orb = _make_band_and_projection(nk=8, nb=2, na=1)
        no_spin = make_orbital_projection(
            projections=orb.projections, spin=None
        )
        with pytest.raises(ValueError, match="requires spin data"):
            plot_band_scatter_preset(
                bands=bands,
                orb_proj=no_spin,
                preset="spin_z",
            )

    def test_band_scatter_with_kpath(self) -> None:
        """Verify symmetry labels on the projected-band scatter.

        The test establishes the k-path scatter contract with the concrete
        values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        scatter: PathCollection
        labels: List[str]

        bands, orb = _make_band_and_projection(nk=10, nb=3, na=2)
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 4, 9],
            segments=2,
            labels=("G", "M", "K"),
        )
        fig, ax, scatter = plot_band_scatter_with_kpath(
            bands=bands,
            orb_proj=orb,
            kpath=kpath,
            preset="d",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 30)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        plt.close(fig)


class TestPlotBandScatterWithKpath(chex.TestCase):
    """Validate :func:`~diffpes.inout.plot_band_scatter_with_kpath`.

    Covers composition of projected-band marker weights with line-mode
    symmetry labels on the shared momentum axis.

    :see: :func:`~diffpes.inout.plot_band_scatter_with_kpath`
    """

    def test_applies_labels_to_projected_bands(self) -> None:
        """Apply all supplied symmetry labels to the scatter axis.

        A ten-point, three-band fixture must produce thirty offsets and retain
        the three requested high-symmetry labels in order.

        Notes
        -----
        The test builds deterministic band and projection carriers, applies a
        three-label ``KPathInfo``, and checks its collection size and tick
        text.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        scatter: PathCollection
        labels: List[str]

        bands, orb = _make_band_and_projection(nk=10, nb=3, na=2)
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 4, 9],
            segments=2,
            labels=("G", "M", "K"),
        )
        fig, ax, scatter = plot_band_scatter_with_kpath(
            bands=bands,
            orb_proj=orb,
            kpath=kpath,
            preset="d",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 30)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        plt.close(fig)


class TestPlotBandScatterEdgeCases(chex.TestCase):
    """Validate additional paths in the band-scatter plotting helpers.

    The tests cover invalid array ranks and incompatible weight shapes.
    They cover selections by atom, orbital, spin, and OAM. They also cover
    unknown presets, missing OAM data, and an existing axis.

    :see: :func:`~diffpes.inout.plot_band_scatter_preset`
    """

    def _make_bands_1d(self, nk: int = 4, nb: int = 2) -> BandStructure:
        """PRIVATE: Build BandStructure with one-dimensional eigenvalues.

        Parameters
        ----------
        nk : int, optional
            Number of k-points. Default 4.
        nb : int, optional
            Number of bands. Default 2.

        Returns
        -------
        malformed_bands : BandStructure
            Carrier whose ``eigenvalues`` leaf is a flat length
            ``nk * nb`` array instead of the required (K, B) matrix.

        Notes
        -----
        Builds a valid carrier through the factory, then swaps the
        ``eigenvalues`` leaf with ``eqx.tree_at`` so the malformed
        rank reaches the plotting code without factory validation.
        """
        valid_bands: BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((nk, nb), dtype=jnp.float64),
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
            kpoint_weights=jnp.zeros(nk, dtype=jnp.float64),
            fermi_energy=jnp.float64(0.0),
        )
        malformed_bands: BandStructure = eqx.tree_at(
            lambda candidate: candidate.eigenvalues,
            valid_bands,
            jnp.zeros(nk * nb, dtype=jnp.float64),
        )
        return malformed_bands

    def _make_orb_with_spin_and_oam(
        self, nk: int = 4, nb: int = 2, na: int = 1
    ) -> OrbitalProjection:
        """PRIVATE: Build OrbitalProjection with spin and OAM attached.

        Parameters
        ----------
        nk : int, optional
            Number of k-points. Default 4.
        nb : int, optional
            Number of bands. Default 2.
        na : int, optional
            Number of atoms. Default 1.

        Returns
        -------
        orbital_projection : OrbitalProjection
            Carrier with uniform orbital weights of shape
            ``(nk, nb, na, 9)``. The spin array has weight in
            channels 0 and 4; the OAM array is uniform with shape
            ``(nk, nb, na, 3)``.

        Notes
        -----
        Passes all three arrays through the public factory so spin and
        OAM selections in the scatter presets have data to read.
        """
        proj: Float64[Array, "nk nb na 9"]
        spin: Float64[Array, "nk nb na 6"]
        oam: Float64[Array, "nk nb na 3"]

        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        spin = spin.at[..., 0].set(0.3)
        spin = spin.at[..., 4].set(0.2)
        oam = jnp.ones((nk, nb, na, 3), dtype=jnp.float64) * 0.05
        orbital_projection: OrbitalProjection = make_orbital_projection(
            projections=proj, spin=spin, oam=oam
        )
        return orbital_projection

    def test_prepare_band_arrays_wrong_ndim_raises(self) -> None:
        """Reject one-dimensional band eigenvalues.

        The test constructs a BandStructure with 1D eigenvalues (bypassing the
        factory), then calls ``plot_band_scatter_preset``. Asserts a
        ``ValueError`` matching ``"shape (K, B)"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "4 2 1 9"]
        orb: diffpes.types.OrbitalProjection

        bands = self._make_bands_1d(nk=4, nb=2)
        proj = jnp.ones((4, 2, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="shape"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="p")

    def test_subset_atom_axis_with_indices(self) -> None:
        """Verify atom-axis selection with explicit atom indices.

        The test plots the p preset with ``atom_indices=[0]``.
        Thus, ``_subset_atom_axis`` receives an index array.
        The test asserts the scatter point count equals nk * nb.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        na: int
        eigen: Float64[Array, "nk nb"]
        kpoints: Float64[Array, "nk 3"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb na 9"]
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb, na = 6, 2, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        kpoints = jnp.zeros((nk, 3), dtype=jnp.float64)
        bands = make_band_structure(eigenvalues=eigen, kpoints=kpoints)
        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            atom_indices=[0],
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_s_orbital_preset(self) -> None:
        """Verify the s-orbital preset branch.

        The test calls ``plot_band_scatter_preset`` with ``preset='s'``. It
        asserts that the scatter renders without error and the point count is
        correct (exercises the ``ORBITAL_INDEX[key]`` branch).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="s", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_total_preset(self) -> None:
        """Verify that the total preset sums all orbital channels.

        The test calls ``plot_band_scatter_preset`` with ``preset='total'``.
        The test asserts the scatter renders and the point count is correct
        (exercises the ``elif key == 'total'`` branch at line 501-502).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 3
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.05
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_spin_channel_preset(self) -> None:
        """Verify the spin-channel preset branch.

        The test calls ``plot_band_scatter_preset`` with
        ``preset='spin_z_up'``.
        This exercises the ``if key in spin_channel`` branch (line 527-528).
        The test asserts the scatter renders without error.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        spin: Float64[Array, "nk nb 1 6"]
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        spin = jnp.ones((nk, nb, 1, 6), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj, spin=spin)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="spin_z_up", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_with_oam_data(self) -> None:
        """Verify OAM selection and the component branch.

        The test calls ``plot_band_scatter_preset`` with ``preset='oam_total'``
        and an OrbitalProjection that has ``oam`` data. This exercises
        line 541 (``oam_arr = _subset_atom_axis(...)``) and lines
        554-556 (``if key in oam_component: weights = ...``).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_abs_total_preset(self) -> None:
        """Verify the absolute-total OAM preset.

        The test calls ``plot_band_scatter_preset`` with
        ``preset='oam_abs_total'`` and an OrbitalProjection with OAM data.
        This exercises
        the ``elif key == 'oam_abs_total'`` branch at lines 557-559,
        which computes ``np.sum(np.abs(oam_arr[..., 2]), axis=2)`` and
        sets ``signed = False``. Asserts the scatter renders without error.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_abs_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_without_oam_data_raises(self) -> None:
        """Verify that an OAM preset requires OAM data.

        The test calls ``plot_band_scatter_preset`` with ``preset='oam_p'`` but
        provides an OrbitalProjection with ``oam=None``. Asserts
        ``ValueError`` matching ``"requires OAM data"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        orb: diffpes.types.OrbitalProjection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="requires OAM data"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="oam_p")

    def test_unknown_preset_raises(self) -> None:
        """Verify that an unknown preset raises ``ValueError``.

        The test calls ``plot_band_scatter_preset`` with an unrecognized preset
        name. Asserts ``ValueError`` matching ``"Unknown preset"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        orb: diffpes.types.OrbitalProjection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Unknown preset"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="not_a_real_preset"
            )

    def test_weight_shape_mismatch_raises(self) -> None:
        """Verify rejection of weights with an incompatible shape.

        The test creates incompatible band and projection shapes.
        The ``"p"`` preset produces weights with shape ``(4, 3)`` instead of
        ``(4, 2)``. The test expects a ``ValueError`` that matches
        ``"Preset weights must have shape"``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb_bands: int
        nb_proj: int
        eigen: Float64[Array, "nk nb_bands"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb_proj 1 9"]
        orb: diffpes.types.OrbitalProjection

        nk = 4
        nb_bands = 2
        nb_proj = 3
        eigen = jnp.linspace(
            -1.0, 0.5, nk * nb_bands, dtype=jnp.float64
        ).reshape(nk, nb_bands)
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb_proj, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Preset weights must have shape"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="p")

    def test_uses_provided_ax(self) -> None:
        """Verify reuse of a given axis for a band scatter.

        The test creates a figure and passes its axis to the plotting function.
        It verifies the identity of the returned figure. This check covers
        the existing-axis path instead of the new-figure path.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        nk: int
        nb: int
        eigen: Float64[Array, "nk nb"]
        bands: diffpes.types.BandStructure
        proj: Float64[Array, "nk nb 1 9"]
        orb: diffpes.types.OrbitalProjection
        fig0: Figure
        ax0: Axes
        out_fig: Figure
        out_ax: Axes

        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig0, ax0 = plt.subplots()
        out_fig, out_ax, _ = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="p", ax=ax0, colorbar=False
        )
        chex.assert_equal(out_fig is fig0, True)
        plt.close(fig0)
