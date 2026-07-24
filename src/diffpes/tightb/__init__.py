"""Provide native tight-binding tools and ARPES-side adapters.

Extended Summary
----------------
The native tight-binding layer provides model construction and Slater-Koster
coupling. It adds spin-orbit coupling, slabs, and degeneracy-safe
diagonalization as the plan series progresses. It also consumes
``DiagonalizedBands`` from other electronic-structure sources.

The package exposes native basis-position-gauge Hamiltonian assembly,
degeneracy-regularized diagonalization, an eigenvalues-only fast path, and
ARPES-side adapters. Analytic chain and graphene models live only in the test
fixture layer.

The following list describes the submodules:

- :mod:`diagonalize`
    Diagonalize native bands and adapt atom-resolved VASP projections.
- :mod:`hamiltonian`
    Assemble native tight-binding Bloch Hamiltonians.
- :mod:`kspace`
    Build differentiable paths and fixed-shape rasters in k-space.
- :mod:`projections`
    Reduce eigenvectors to gauge-invariant observables.
- :mod:`operators`
    Construct spin, orbital, and atomic spin--orbit observables.
- :mod:`parameters`
    Expose independent flat-real optimizer parameter views.
- :mod:`slaterkoster`
    Build s/p/d two-center hopping blocks and materialized models.
- :mod:`soc`
    Construct and scatter atomic spin-orbit coupling.
- :mod:`dos`
    Compute broadened densities of states and fixed-filling Fermi levels.

Routine Listings
----------------
:func:`bloch_hamiltonian`
    Assemble one basis-position-gauge Bloch Hamiltonian.
:func:`bloch_hamiltonian_batch`
    Assemble Bloch Hamiltonians for a batch of fractional k-points.
:func:`build_sk_model`
    Build a tight-binding model from Slater-Koster fundamentals.
:func:`build_arpes_kmesh`
    Build a fixed-kz ARPES raster in fractional coordinates.
:func:`build_bz_mesh`
    Build a fixed-shape reciprocal mesh and its first-zone mask.
:func:`build_kmesh_hv`
    Build a photon-energy raster in fractional coordinates.
:func:`build_kpath`
    Build a labeled path between k-space anchors.
:func:`diagonalize_tb`
    Diagonalize a native tight-binding model over k-points.
:func:`eigh_safe`
    Diagonalize a Hermitian matrix with a regularized eigenvector JVP.
:func:`eigvalsh_bands`
    Compute only native tight-binding eigenvalues over k-points.
:func:`band_projectors`
    Materialize rank-one band projectors.
:func:`dos_gaussian`
    Compute a Gaussian-broadened density of states.
:func:`expectation_path`
    Compute degeneracy-averaged operator expectations.
:func:`fat_bands`
    Compute selected-orbital fat-band weights.
:func:`fermi_level_from_filling`
    Solve for the chemical potential at a fixed filling.
:func:`first_bz_mask`
    Mark Cartesian points inside the first Brillouin zone.
:func:`kpath_arc_length`
    Compute cumulative Cartesian distance along a k-path.
:func:`kpoints_cart_to_frac`
    Convert Cartesian momenta to fractional k-points.
:func:`kpoints_frac_to_cart`
    Convert fractional k-points to Cartesian momenta.
:func:`group_projector`
    Construct a fixed-group projector.
:func:`group_trace`
    Trace an operator over a fixed band group.
:func:`ls_operator`
    Construct atomic L dot S operators.
:func:`l_matrices`
    Construct complex-harmonic orbital angular-momentum matrices.
:func:`neighbor_shells`
    Discover unique undirected neighbor bonds with exact integer cells.
:func:`orbital_projector`
    Construct an orbital-selection projector.
:func:`orbital_weights`
    Compute squared orbital amplitudes.
:func:`sk_block`
    Construct an s/p/d Slater-Koster hopping block.
:func:`sk_model_parameter_view`
    Pack Slater-Koster fundamentals into optimizer coordinates.
:func:`soc_matrix`
    Scatter shell-resolved atomic spin-orbit coupling.
:func:`soc_shell_block`
    Construct a unit-strength real-cubic atomic SOC block.
:func:`spin_double_basis`
    Duplicate a spinless basis in down-up block order.
:func:`spin_double_model`
    Duplicate a spinless model into spin-diagonal blocks.
:func:`spin_operator`
    Construct spin along a Cartesian unit axis.
:func:`tb_parameter_view`
    Pack a materialized model into independent optimizer coordinates.
:func:`vasp_to_diagonalized`
    Convert atom-resolved VASP projections to approximate band vectors.
"""

from .diagonalize import (
    diagonalize_tb,
    eigh_safe,
    eigvalsh_bands,
    vasp_to_diagonalized,
)
from .dos import dos_gaussian, fermi_level_from_filling
from .hamiltonian import (
    bloch_hamiltonian,
    bloch_hamiltonian_batch,
)
from .kspace import (
    build_arpes_kmesh,
    build_bz_mesh,
    build_kmesh_hv,
    build_kpath,
    first_bz_mask,
    kpath_arc_length,
    kpoints_cart_to_frac,
    kpoints_frac_to_cart,
)
from .operators import ls_operator, orbital_projector, spin_operator
from .parameters import sk_model_parameter_view, tb_parameter_view
from .projections import (
    band_projectors,
    expectation_path,
    fat_bands,
    group_projector,
    group_trace,
    orbital_weights,
)
from .slaterkoster import build_sk_model, neighbor_shells, sk_block
from .soc import (
    l_matrices,
    soc_matrix,
    soc_shell_block,
    spin_double_basis,
    spin_double_model,
)

__all__: list[str] = [
    "band_projectors",
    "bloch_hamiltonian",
    "bloch_hamiltonian_batch",
    "build_arpes_kmesh",
    "build_bz_mesh",
    "build_kmesh_hv",
    "build_kpath",
    "build_sk_model",
    "diagonalize_tb",
    "dos_gaussian",
    "eigh_safe",
    "eigvalsh_bands",
    "expectation_path",
    "fat_bands",
    "fermi_level_from_filling",
    "first_bz_mask",
    "kpath_arc_length",
    "kpoints_cart_to_frac",
    "kpoints_frac_to_cart",
    "l_matrices",
    "group_projector",
    "group_trace",
    "ls_operator",
    "neighbor_shells",
    "orbital_projector",
    "orbital_weights",
    "sk_block",
    "sk_model_parameter_view",
    "soc_matrix",
    "soc_shell_block",
    "spin_double_basis",
    "spin_double_model",
    "spin_operator",
    "tb_parameter_view",
    "vasp_to_diagonalized",
]
