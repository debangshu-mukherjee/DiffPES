r"""Provide native tight-binding tools and ARPES-side adapters.

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
    Reduce tight-binding eigenvectors to gauge-invariant observables.
- :mod:`operators`
    Construct Hermitian observables in a tight-binding orbital basis.
- :mod:`parameters`
    Expose independent real optimizer coordinates for tight-binding models.
- :mod:`slaterkoster`
    Build tight-binding hoppings from Slater--Koster integrals.
- :mod:`slab`
    Construct exact Miller-index surface cells and rotate TB models.
- :mod:`soc`
    Construct atomic spin--orbit coupling in the real-cubic basis.
- :mod:`dos`
    Construct broadened tight-binding densities of states and fillings.

Routine Listings
----------------
:func:`bloch_hamiltonian`
    Assemble one basis-position-gauge Bloch Hamiltonian.
:func:`bloch_hamiltonian_batch`
    Assemble Bloch Hamiltonians for a batch of fractional k-points.
:func:`build_sk_model`
    Build a validated tight-binding model from two-center integrals.
:func:`build_arpes_kmesh`
    Build a fixed-kz ARPES raster in fractional coordinates.
:func:`build_bz_mesh`
    Build a fixed-shape reciprocal mesh and its first-zone mask.
:func:`build_kmesh_hv_at_fermi`
    Build an at-Fermi photon-energy raster in fractional coordinates.
:func:`build_kpath`
    Build a labeled path between k-space anchors.
:func:`diagonalize_tb`
    Diagonalize a native tight-binding model over k-points.
:func:`eigh_safe`
    Diagonalize a Hermitian matrix with a regularized eigenvector JVP.
:func:`eigvalsh_bands`
    Compute only native tight-binding eigenvalues over k-points.
:func:`eigvalsh_bands_chunked`
    Compute eigenvalues with bounded live Hamiltonian storage.
:func:`band_projectors`
    Materialize each U(1)-gauge-invariant rank-one band projector.
:func:`dos_gaussian`
    Evaluate a Gaussian-broadened tight-binding density of states.
:func:`expectation_path`
    Compute operator expectations with diagnostic degeneracy averaging.
:func:`fat_bands`
    Compute degeneracy-averaged weights of selected model orbitals.
:func:`fermi_level_from_filling`
    Compute the finite-temperature Fermi level from the filling equation.
:func:`first_bz_mask`
    Mark Cartesian points inside the first Brillouin zone.
:func:`find_surface_cell`
    Build an exact primitive surface cell for one Miller plane.
:func:`freeze_slab_topology`
    Freeze every discrete choice required to rebuild one slab.
:func:`gen_slab`
    Construct a finite Miller-index slab with exact open-normal topology.
:func:`gen_slab_with_operators`
    Construct a slab while preserving its Wannier operator sidecar.
:func:`kpath_arc_length`
    Compute cumulative Cartesian distance along a k-path.
:func:`kpoints_cart_to_frac`
    Convert Cartesian momenta to fractional k-points.
:func:`kpoints_frac_to_cart`
    Convert fractional k-points to Cartesian momenta.
:func:`group_projector`
    Construct the projector onto one registered, fixed band group.
:func:`group_trace`
    Trace a Hermitian operator over one fixed band group.
:func:`ls_operator`
    Construct unit-strength atomic :math:`L\cdot S` by shell.
:func:`layer_resolved_group_traces`
    Compute surface traces over complete, isolated fixed band groups.
:func:`layer_resolved_weights`
    Compute per-band surface weights as an off-degeneracy diagnostic.
:func:`l_matrices`
    Construct orbital angular-momentum matrices in the complex basis.
:func:`neighbor_shells`
    Find unique undirected neighbor bonds at host setup time.
:func:`orbital_projector`
    Construct a diagonal projector onto selected basis orbitals.
:func:`orbital_weights`
    Compute the squared orbital amplitudes of normalized eigenvectors.
:func:`rebuild_slab`
    Construct a slab from frozen topology using only JAX geometry.
:func:`sk_block`
    Construct a real-harmonic Slater--Koster hopping block.
:func:`sk_model_parameter_view`
    Pack Slater--Koster fundamentals and return a rebuilding closure.
:func:`soc_matrix`
    Assemble shell-resolved atomic SOC in an arbitrary spinor basis.
:func:`soc_shell_block`
    Construct a unit-strength real-cubic :math:`\mathbf L\cdot\mathbf S`.
:func:`spin_double_basis`
    Create a spin-doubled basis in the declared down--up block order.
:func:`spin_double_model`
    Create a spin-doubled model with spin-diagonal down--up blocks.
:func:`spin_operator`
    Construct :math:`S_{\widehat n}=\widehat n\cdot\sigma/2`.
:func:`surface_projector`
    Construct surface-sensitive orbital probability weights.
:func:`rotate_tb_model`
    Construct a rotated complete-shell tight-binding model.
:func:`tb_parameter_view`
    Pack a materialized tight-binding model into independent coordinates.
:func:`vasp_to_diagonalized`
    Convert atom-resolved VASP projections to approximate band vectors.
:func:`validate_open_surface_adjacency`
    Reject direct or component-propagated periodic normal images.
"""

from .diagonalize import (
    diagonalize_tb,
    eigh_safe,
    eigvalsh_bands,
    eigvalsh_bands_chunked,
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
    build_kmesh_hv_at_fermi,
    build_kpath,
    first_bz_mask,
    kpath_arc_length,
    kpoints_cart_to_frac,
    kpoints_frac_to_cart,
)
from .operators import (
    layer_resolved_group_traces,
    layer_resolved_weights,
    ls_operator,
    orbital_projector,
    spin_operator,
    surface_projector,
)
from .parameters import sk_model_parameter_view, tb_parameter_view
from .projections import (
    band_projectors,
    expectation_path,
    fat_bands,
    group_projector,
    group_trace,
    orbital_weights,
)
from .slab import (
    find_surface_cell,
    freeze_slab_topology,
    gen_slab,
    gen_slab_with_operators,
    rebuild_slab,
    rotate_tb_model,
    validate_open_surface_adjacency,
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
    "build_kmesh_hv_at_fermi",
    "build_kpath",
    "build_sk_model",
    "diagonalize_tb",
    "dos_gaussian",
    "eigh_safe",
    "eigvalsh_bands",
    "eigvalsh_bands_chunked",
    "expectation_path",
    "fat_bands",
    "fermi_level_from_filling",
    "find_surface_cell",
    "first_bz_mask",
    "freeze_slab_topology",
    "gen_slab",
    "gen_slab_with_operators",
    "kpath_arc_length",
    "kpoints_cart_to_frac",
    "kpoints_frac_to_cart",
    "l_matrices",
    "layer_resolved_group_traces",
    "layer_resolved_weights",
    "group_projector",
    "group_trace",
    "ls_operator",
    "neighbor_shells",
    "orbital_projector",
    "orbital_weights",
    "rebuild_slab",
    "rotate_tb_model",
    "sk_block",
    "sk_model_parameter_view",
    "soc_matrix",
    "soc_shell_block",
    "spin_double_basis",
    "spin_double_model",
    "spin_operator",
    "surface_projector",
    "tb_parameter_view",
    "validate_open_surface_adjacency",
    "vasp_to_diagonalized",
]
