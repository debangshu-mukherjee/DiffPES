"""Expose the public :mod:`diffpes.inout` surface.

Extended Summary
----------------
- :mod:`band_plotting`
    Plot projected tight-binding bands with analysis utilities.
- :mod:`certificate`
    Encode and identify portable forward-model certificates.
- :mod:`certificate_decoding`
    Decode portable forward-model certificate documents.
- :mod:`certificate_storage`
    Persist forward-model certificates in portable storage formats.
- :mod:`chgcar`
    Parse a VASP CHGCAR file.
- :mod:`doscar`
    Parse a VASP DOSCAR file.
- :mod:`eigenval`
    Parse a VASP EIGENVAL file.
- :mod:`hdf5`
    Serialize and deserialize diffpes PyTrees in HDF5.
- :mod:`helpers`
    Provide workflow helpers for simulation-ready parser arrays.
- :mod:`kpoints`
    Parse a VASP KPOINTS file.
- :mod:`plotting`
    Plot ARPES spectra with analysis utilities.
- :mod:`poscar`
    Parse a VASP POSCAR/CONTCAR file.
- :mod:`procar`
    Parse a VASP PROCAR file.
- :mod:`tb_files`
    Parse explicit Cartesian tight-binding hopping lists.
- :mod:`wannier90`
    Read normative Wannier90 tight-binding files.
- :mod:`wannier90_parser`
    Parse normative Wannier90 text records.
- :mod:`wavecar`
    Read bounded VASP WAVECAR direct-access records.

Routine Listings
----------------
:func:`aggregate_atoms`
    Sum orbital projections over a set of atoms.
:func:`apply_kpath_ticks`
    Apply symmetry-point ticks/labels from KPathInfo to an axis.
:func:`attach_certificate_h5`
    Attach a certificate atomically to an HDF5 result file.
:func:`certificate_identity`
    Compute the scientific identity of a canonical certificate.
:func:`check_consistency`
    Check dimension agreement across parsed VASP files.
:func:`finalize_certificate`
    Replace the kernel placeholder with the canonical identity.
:func:`list_band_scatter_presets`
    Return supported preset names for projected band scatter plots.
:func:`load_certificate_h5`
    Load a certificate embedded in an HDF5 result file.
:func:`load_certificate_json`
    Load a validated forward certificate from canonical JSON.
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot ARPES spectrum and annotate k-axis using KPathInfo.
:func:`plot_band_scatter_preset`
    Plot projected bands as marker-size-weighted scatter points.
:func:`plot_band_scatter_with_kpath`
    Plot projected band scatter and annotate x-axis with k-path labels.
:func:`read_chgcar`
    Parse a VASP CHGCAR file.
:func:`read_doscar`
    Parse a VASP DOSCAR file.
:func:`read_eigenval`
    Parse a VASP EIGENVAL file.
:func:`read_hopping_list`
    Parse a zero-based Cartesian tight-binding hopping list.
:func:`read_kpoints`
    Parse a VASP KPOINTS file.
:func:`read_poscar`
    Parse a VASP POSCAR/CONTCAR file.
:func:`read_procar`
    Parse a VASP PROCAR file.
:func:`read_wannier90_hr`
    Parse a normative Wannier90 ``seedname_hr.dat`` file.
:func:`read_wannier90_tb`
    Parse a normative Wannier90 ``seedname_tb.dat`` file.
:func:`reduce_orbitals`
    Reduce 9 orbital channels to s/p/d totals.
:func:`save_certificate_json`
    Save a forward certificate atomically as canonical JSON.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.
:func:`select_atoms`
    Extract orbital projections for a subset of atoms.
:func:`index_wavecar`
    Compute the ``index_wavecar`` public contract.
:func:`load_wavecar_records`
    Compute the ``load_wavecar_records`` public contract.
:func:`wavecar_header`
    Compute the ``wavecar_header`` public contract.
"""

from .band_plotting import (
    list_band_scatter_presets,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
)
from .certificate import (
    certificate_identity,
    finalize_certificate,
)
from .certificate_storage import (
    attach_certificate_h5,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)
from .chgcar import read_chgcar
from .doscar import read_doscar
from .eigenval import read_eigenval
from .hdf5 import load_from_h5, save_to_h5
from .helpers import (
    aggregate_atoms,
    check_consistency,
    reduce_orbitals,
    select_atoms,
)
from .kpoints import read_kpoints
from .plotting import (
    apply_kpath_ticks,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
)
from .poscar import read_poscar
from .procar import read_procar
from .tb_files import read_hopping_list
from .wannier90 import read_wannier90_hr, read_wannier90_tb
from .wavecar import (
    index_wavecar,
    load_wavecar_records,
    wavecar_header,
)

__all__: list[str] = [
    "aggregate_atoms",
    "apply_kpath_ticks",
    "attach_certificate_h5",
    "certificate_identity",
    "check_consistency",
    "finalize_certificate",
    "list_band_scatter_presets",
    "load_certificate_h5",
    "load_certificate_json",
    "load_from_h5",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "plot_band_scatter_preset",
    "plot_band_scatter_with_kpath",
    "read_chgcar",
    "read_doscar",
    "read_eigenval",
    "read_hopping_list",
    "read_kpoints",
    "read_poscar",
    "read_procar",
    "read_wannier90_hr",
    "read_wannier90_tb",
    "reduce_orbitals",
    "save_certificate_json",
    "save_to_h5",
    "select_atoms",
    "index_wavecar",
    "load_wavecar_records",
    "wavecar_header",
]
