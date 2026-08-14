"""Plot the ARPES spectra and band structures.

Extended Summary
----------------
This subpackage renders diffpes carriers with Matplotlib. Every function
ingests native diffpes types or physical axes and returns the Matplotlib
objects that it creates. The functions operate on host-side NumPy arrays
outside JAX-compiled code.

The subpackage contains the following submodules:

- :mod:`arpes_maps`
    Plot ARPES spectra with analysis utilities.
- :mod:`band_lines`
    Plot band dispersions and weighted band scatters along a path.
- :mod:`band_scatter`
    Plot projected tight-binding bands with analysis utilities.
- :mod:`comparison_panels`
    Compare spectral maps in shared-scale panel rows and difference maps.
- :mod:`detector_views`
    Plot calibrated detector rasters and count comparisons.
- :mod:`distribution_curves`
    Plot energy and momentum distribution curves from spectral maps.
- :mod:`scalar_curves`
    Plot scalar curve families, densities of states, and axis profiles.
- :mod:`volume_views`
    Render three-dimensional views of ARPES cubes and band surfaces.

Routine Listings
----------------
:func:`apply_kpath_ticks`
    Apply symmetry-point ticks/labels from KPathInfo to an axis.
:func:`list_band_scatter_presets`
    Return supported preset names for projected band scatter plots.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot ARPES spectrum and annotate k-axis using KPathInfo.
:func:`plot_band_dispersion`
    Plot band dispersions as lines along a momentum path.
:func:`plot_band_scatter_preset`
    Plot projected bands as marker-size-weighted scatter points.
:func:`plot_band_scatter_weights`
    Plot bands as weight-encoded scatter points along a path.
:func:`plot_band_scatter_with_kpath`
    Plot projected band scatter and annotate x-axis with k-path labels.
:func:`plot_band_surface`
    Plot band eigenvalues as three-dimensional surfaces on a momentum mesh.
:func:`plot_bands_over_spectrum`
    Plot band dispersions over an energy-momentum intensity image.
:func:`plot_cube_faces`
    Plot the three outer faces of an ARPES cube as opaque surfaces.
:func:`plot_cube_scatter`
    Plot an ARPES cube as a translucent three-dimensional point cloud.
:func:`plot_curve_family`
    Plot a labeled family of curves over one shared axis.
:func:`plot_detector_comparison`
    Plot expected and observed detector counts side by side.
:func:`plot_detector_energy_cut`
    Plot an energy-versus-angle detector cut image.
:func:`plot_detector_image`
    Plot an energy-summed angular detector image.
:func:`plot_detector_residual`
    Plot the standardized Poisson residual image of detector counts.
:func:`plot_difference_map`
    Plot a signed difference map with a symmetric diverging scale.
:func:`plot_distribution_curves`
    Plot a stack of EDCs or MDCs from an intensity map.
:func:`plot_dos`
    Plot one total density of states with Fermi-level context.
:func:`plot_dos_overlay`
    Plot several density-of-states curves on one axis.
:func:`plot_edc_mdc_panels`
    Plot one EDC panel beside one MDC panel from an intensity map.
:func:`plot_momentum_map`
    Plot a momentum-momentum intensity map with Cartesian axes.
:func:`plot_momentum_map_grid`
    Plot momentum-momentum maps as one row of shared-scale panels.
:func:`plot_momentum_profile`
    Plot the momentum profile integrated over an energy window.
:func:`plot_planar_average`
    Plot a planar-averaged profile along the stacking axis.
:func:`plot_spectral_cut`
    Plot an energy-momentum intensity map on physical axes.
:func:`plot_spectral_cut_series`
    Plot energy-momentum cuts as one row of shared-scale panels.
:func:`spectrum_extent`
    Compute the imshow extent of a momentum and an energy axis.
"""

from .arpes_maps import (
    apply_kpath_ticks,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
    plot_momentum_map,
    plot_spectral_cut,
    spectrum_extent,
)
from .band_lines import (
    plot_band_dispersion,
    plot_band_scatter_weights,
    plot_bands_over_spectrum,
)
from .band_scatter import (
    list_band_scatter_presets,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
)
from .comparison_panels import (
    plot_difference_map,
    plot_momentum_map_grid,
    plot_spectral_cut_series,
)
from .detector_views import (
    plot_detector_comparison,
    plot_detector_energy_cut,
    plot_detector_image,
    plot_detector_residual,
)
from .distribution_curves import (
    plot_distribution_curves,
    plot_edc_mdc_panels,
    plot_momentum_profile,
)
from .scalar_curves import (
    plot_curve_family,
    plot_dos,
    plot_dos_overlay,
    plot_planar_average,
)
from .volume_views import (
    plot_band_surface,
    plot_cube_faces,
    plot_cube_scatter,
)

__all__: list[str] = [
    "apply_kpath_ticks",
    "list_band_scatter_presets",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "plot_band_dispersion",
    "plot_band_scatter_preset",
    "plot_band_scatter_weights",
    "plot_band_scatter_with_kpath",
    "plot_band_surface",
    "plot_bands_over_spectrum",
    "plot_cube_faces",
    "plot_cube_scatter",
    "plot_curve_family",
    "plot_detector_comparison",
    "plot_detector_energy_cut",
    "plot_detector_image",
    "plot_detector_residual",
    "plot_difference_map",
    "plot_distribution_curves",
    "plot_dos",
    "plot_dos_overlay",
    "plot_edc_mdc_panels",
    "plot_momentum_map",
    "plot_momentum_map_grid",
    "plot_momentum_profile",
    "plot_planar_average",
    "plot_spectral_cut",
    "plot_spectral_cut_series",
    "spectrum_extent",
]
