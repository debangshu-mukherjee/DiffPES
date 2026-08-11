"""Verify slab metadata carriers across every owned legacy consumer.

The tests exercise slab numerical and structural contracts.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from beartype.typing import Any
from matplotlib import pyplot as plt

from diffpes.inout import (
    load_from_h5,
    plot_band_scatter_preset,
    save_to_h5,
)
from diffpes.tightb import (
    diagonalize_tb,
    fat_bands,
    gen_slab,
    orbital_weights,
)
from diffpes.types import (
    ORBITAL_INDEX,
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    TBModel,
    make_band_structure,
    make_crystal_geometry,
    make_orbital_basis,
    make_orbital_projection,
    make_tb_model,
)


def _bulk_chain() -> TBModel:
    """PRIVATE: Build a one-orbital bulk chain normal to the requested surface.

    Returns
    -------
    model : TBModel
        One-atom cubic-cell model with a Hermitian conjugate hopping
        pair of ``-0.8`` eV along ``(0, 0, +/-1)`` and zero onsite
        energy.

    Notes
    -----
    The chain disperses only along z, the (001) extrusion direction.
    The resulting slab carries a clean layer structure for the depth
    and persistence handoff checks.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.asarray((-0.8, -0.8), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )
    return model


class TestSlabCarrierHandoff:
    """Certify slab metadata across persistence and projection consumers.

    The case propagates depth metadata through consumers, persistence,
    projections, and plotting views.
    """

    @pytest.mark.rss_limit_mb(640)
    def test_depths_survive_consumers_and_plot_accepts_slab_views(
        self,
        tmp_path: Path,
    ) -> None:
        """Preserve depths in consumers and plot slab-sized legacy views.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        expected_depths: Any
        figure: Any
        kpoints: Any
        plot_bands: Any
        plot_projection: Any
        projections: Any
        raw_weights: Any
        scatter: Any
        selected_weights: Any
        slab: Any
        slab, _ = gen_slab(
            _bulk_chain(),
            miller=(0, 0, 1),
            thickness_ang=3.0,
            vacuum_ang=5.0,
        )
        kpoints = jnp.asarray(
            ((0.0, 0.0, 0.0), (0.17, -0.09, 0.0)),
            dtype=jnp.float64,
        )
        bands: DiagonalizedBands = diagonalize_tb(slab, kpoints)
        assert bands.depths is not None
        expected_depths = bands.depths

        raw_weights = orbital_weights(bands.eigenvectors)
        selected_weights = fat_bands(
            bands,
            tuple(range(len(slab.basis.n))),
        )
        assert raw_weights.shape == (
            kpoints.shape[0],
            len(slab.basis.n),
            len(slab.basis.n),
        )
        assert jnp.allclose(selected_weights, 1.0, rtol=1e-12, atol=1e-12)
        assert jnp.array_equal(bands.depths, expected_depths)

        path: Path = tmp_path / "slab_bands.h5"
        save_to_h5(path, bands=bands)
        restored: DiagonalizedBands = load_from_h5(path, "bands")
        assert restored.depths is not None
        assert jnp.array_equal(restored.depths, expected_depths)

        projections = jnp.zeros(
            (
                kpoints.shape[0],
                len(slab.basis.n),
                slab.geometry.positions.shape[0],
                9,
            ),
            dtype=jnp.float64,
        )
        projections = projections.at[
            :,
            :,
            :,
            ORBITAL_INDEX["s"],
        ].set(raw_weights)
        plot_bands = make_band_structure(
            bands.eigenvalues,
            bands.kpoints,
            fermi_energy=bands.fermi_energy,
        )
        plot_projection = make_orbital_projection(projections)
        figure, _, scatter = plot_band_scatter_preset(
            plot_bands,
            plot_projection,
            preset="s",
            colorbar=False,
        )
        assert scatter.get_sizes().shape == (
            kpoints.shape[0] * len(slab.basis.n),
        )
        assert jnp.array_equal(bands.depths, expected_depths)
        plt.close(figure)
