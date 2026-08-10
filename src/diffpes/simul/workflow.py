"""Load and prepare VASP projection data for coherent ARPES workflows.

Extended Summary
----------------
The module combines the retained input-boundary tasks: loading VASP outputs,
selecting atoms, and attaching OAM channels. Physical spectral assembly uses
the coherent APIs in :mod:`diffpes.simul.spectral`. Plan 08a defines the
canonical detector/count driver after completing its effects chain.

Routine Listings
----------------
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`prepare_projection`
    Prepare orbital projections for simulation.
"""

from pathlib import Path

from beartype import beartype
from beartype.typing import Literal, Optional, cast
from jaxtyping import Array, Float64, jaxtyped

from diffpes.inout import (
    check_consistency,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_procar,
    select_atoms,
)
from diffpes.types import (
    BandStructure,
    DosType,
    KPathInfo,
    ProjectionType,
    ScalarFloat,
    SpinOrbitalProjection,
    WorkflowContext,
    make_orbital_projection,
    make_spin_orbital_projection,
    make_workflow_context,
)

from .oam import compute_oam


@jaxtyped(typechecker=beartype)
def load_vasp_context(
    directory: str = ".",
    eigenval_file: str = "EIGENVAL",
    procar_file: str = "PROCAR",
    doscar_file: Optional[str] = "DOSCAR",
    kpoints_file: Optional[str] = "KPOINTS",
    fermi_energy: Optional[ScalarFloat] = None,
    procar_mode: Literal["legacy", "full"] = "full",
    doscar_mode: Literal["legacy", "full"] = "legacy",
    check_dimensions: bool = True,
) -> WorkflowContext:
    """Load a simulation-ready context from VASP output files.

    Parses the required band and projection files. It also loads optional
    density-of-states and k-path data when those files are available.

    :see: :class:`~.test_workflow.TestLoadVaspContext`

    Implementation Logic
    --------------------
    1. **Resolve the input directory**::

           root = Path(directory)

       All requested VASP filenames are relative to this path.

    2. **Resolve the Fermi energy**::

           resolved_fermi = dos.fermi_energy
           resolved_fermi = fermi_energy

       An explicit value takes priority. Otherwise, DOSCAR supplies the value.

    3. **Parse the required carriers**::

           bands = read_eigenval(...)
           orb_proj = read_procar(...)

       EIGENVAL and PROCAR provide the arrays required by every workflow.

    4. **Load optional path metadata**::

           kpath = read_kpoints(str(kpoints_path))

       The parser runs only when the optional KPOINTS file exists.

    5. **Validate and construct the context**::

           check_consistency(bands, orb_proj, kpath)
           context = make_workflow_context(...)

       The check rejects incompatible files before the factory builds a PyTree.

    Parameters
    ----------
    directory : str, optional
        Directory containing VASP files. Default is current directory.
    eigenval_file : str, optional
        EIGENVAL filename. Default is ``"EIGENVAL"``.
    procar_file : str, optional
        PROCAR filename. Default is ``"PROCAR"``.
    doscar_file : Optional[str], optional
        DOSCAR filename used to infer Fermi energy when
        ``fermi_energy`` is not provided. Use ``None`` to skip DOSCAR.
    kpoints_file : Optional[str], optional
        KPOINTS filename for optional path metadata. Use ``None`` to
        skip KPOINTS parsing.
    fermi_energy : Optional[ScalarFloat], optional
        Manual Fermi energy in eV. If ``None``, the function reads DOSCAR when
        available. Otherwise, it uses 0.0.
    procar_mode : Literal["legacy", "full"], optional
        PROCAR return mode. ``"full"`` preserves spin data when present.
    doscar_mode : Literal["legacy", "full"], optional
        DOSCAR return mode when DOSCAR is read.
    check_dimensions : bool, optional
        If True, run cross-file consistency checks.

    Returns
    -------
    context : WorkflowContext
        Loaded VASP data bundled for downstream workflow calls.

    Raises
    ------
    FileNotFoundError
        If the function needs DOSCAR to infer the Fermi energy but cannot find
        the file.
    """
    root: Path = Path(directory)

    dos: Optional[DosType] = None
    if fermi_energy is None:
        if doscar_file is None:
            resolved_fermi: ScalarFloat = 0.0
        else:
            dos_path_req: Path = root / doscar_file
            if not dos_path_req.exists():
                msg: str = (
                    "DOSCAR is required to infer fermi_energy but was "
                    f"not found: {dos_path_req}"
                )
                raise FileNotFoundError(msg)
            dos = read_doscar(str(dos_path_req), return_mode=doscar_mode)
            resolved_fermi = dos.fermi_energy
    else:
        resolved_fermi = fermi_energy
        if doscar_file is not None:
            dos_path_opt: Path = root / doscar_file
            if dos_path_opt.exists():
                dos = read_doscar(str(dos_path_opt), return_mode=doscar_mode)

    bands: BandStructure = cast(
        BandStructure,
        read_eigenval(
            str(root / eigenval_file),
            fermi_energy=resolved_fermi,
            return_mode="legacy",
        ),
    )
    orb_proj: ProjectionType = read_procar(
        str(root / procar_file),
        return_mode=procar_mode,
    )

    kpath: Optional[KPathInfo] = None
    if kpoints_file is not None:
        kpoints_path: Path = root / kpoints_file
        if kpoints_path.exists():
            kpath = read_kpoints(str(kpoints_path))

    if check_dimensions:
        check_consistency(bands, orb_proj, kpath)

    context: WorkflowContext = make_workflow_context(
        bands=bands,
        orb_proj=orb_proj,
        kpath=kpath,
        dos=dos,
    )
    return context


@jaxtyped(typechecker=beartype)
def prepare_projection(
    orb_proj: ProjectionType,
    atom_indices: Optional[list[int]] = None,
    attach_oam: bool = False,
) -> ProjectionType:
    """Prepare orbital projections for simulation.

    Applies common pre-processing steps used in MATLAB-like workflows:
    selecting atom subsets and attaching OAM channels derived from
    orbital projections.

    :see: :class:`~.test_workflow.TestPrepareProjection`

    Implementation Logic
    --------------------
    1. **Select atoms**::

           prepared = select_atoms(prepared, atom_indices)

       The optional selection keeps only the requested zero-based atom axes.
    2. **Attach OAM channels**::

           oam = compute_oam(prepared.projections)

       The function computes missing channels and rebuilds the same projection
       carrier type with the new data.

    Parameters
    ----------
    orb_proj : ProjectionType
        Input projection object.
    atom_indices : Optional[list[int]], optional
        Optional 0-based atom indices to keep.
    attach_oam : bool, optional
        If True and OAM is absent, compute OAM and attach it.

    Returns
    -------
    prepared : ProjectionType
        Prepared projection object, preserving spin-aware type.
    """
    prepared: ProjectionType = orb_proj
    if atom_indices is not None:
        prepared = select_atoms(prepared, atom_indices)

    if attach_oam and prepared.oam is None:
        oam: Float64[Array, "K B A 3"] = compute_oam(prepared.projections)
        if isinstance(prepared, SpinOrbitalProjection):
            prepared = make_spin_orbital_projection(
                projections=prepared.projections,
                spin=prepared.spin,
                oam=oam,
            )
        else:
            prepared = make_orbital_projection(
                projections=prepared.projections,
                spin=prepared.spin,
                oam=oam,
            )
    return prepared


__all__: list[str] = [
    "load_vasp_context",
    "prepare_projection",
]
