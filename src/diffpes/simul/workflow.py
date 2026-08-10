"""Load VASP metadata and compose the explicit-H coherent cut workflow.

Extended Summary
----------------
The module owns the file-system boundary for VASP outputs. It also provides
one honest compatibility workflow: EIGENVAL and PROCAR supply metadata, while
an explicit Hermitian Hamiltonian raster supplies all resolvent values and
inversion derivatives. PROCAR has no complex phases, so its manufactured band
vectors never become Hamiltonian or inversion authority.

Routine Listings
----------------
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`prepare_projection`
    Prepare orbital projections for simulation.
:func:`run_vasp_workflow`
    Run the explicit-H coherent cut workflow with VASP metadata.
"""

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, cast
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.inout import (
    check_consistency,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_procar,
    select_atoms,
)
from diffpes.tightb import vasp_to_diagonalized
from diffpes.types import (
    BandStructure,
    CrystalGeometry,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    DosType,
    ExperimentGeometry,
    FinalStateSpec,
    KPath,
    KPathInfo,
    MatrixElementParams,
    OrbitalBasis,
    OrbitalProjection,
    ProjectionType,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SpinOrbitalProjection,
    WorkflowContext,
    make_kpath,
    make_orbital_projection,
    make_spin_orbital_projection,
    make_workflow_context,
)

from .oam import compute_oam
from .spectrum import simulate_arpes_cut


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


def _context_kpath(
    context: WorkflowContext,
    crystal_geometry: CrystalGeometry,
) -> KPath:
    """PRIVATE: Build a fixed-kz path from parsed VASP coordinates.

    Parameters
    ----------
    context : WorkflowContext
        Parsed EIGENVAL coordinates and optional KPOINTS metadata.
    crystal_geometry : CrystalGeometry
        Crystal reciprocal basis for the explicit Cartesian ``kz`` value.

    Returns
    -------
    kpath : KPath
        Parsed fractional points with verified labels or no labels.

    Notes
    -----
    KPOINTS labels survive only for reciprocal line-mode metadata whose point
    count, nonempty labels, indices, and anchor coordinates match EIGENVAL.
    Invalid or incomplete plotting metadata does not invalidate the physical
    EIGENVAL path; the path remains available without labels.
    """
    kpoints: Float64[Array, "n_k 3"] = context.bands.kpoints
    kpoints_cart: Float64[Array, "n_k 3"] = (
        kpoints @ crystal_geometry.reciprocal
    )
    labels: Tuple[str, ...] = ()
    label_indices: Tuple[int, ...] = ()
    n_per_segment: int = 1
    metadata: Optional[KPathInfo] = context.kpath
    if metadata is not None and metadata.mode == "Line-mode":
        candidate_labels: Tuple[str, ...] = tuple(
            label.strip() for label in metadata.labels
        )
        candidate_indices: Tuple[int, ...] = tuple(
            int(index) for index in metadata.label_indices.tolist()
        )
        points_per_segment: int = int(metadata.points_per_segment)
        coordinate_mode: str = metadata.coordinate_mode.casefold()
        endpoints_match: bool = False
        if (
            metadata.kpoints is not None
            and metadata.kpoints.shape == (len(candidate_indices), 3)
            and len(candidate_indices) > 0
            and all(
                0 <= index < kpoints.shape[0] for index in candidate_indices
            )
        ):
            endpoints_match = bool(
                jnp.allclose(
                    metadata.kpoints,
                    kpoints[jnp.asarray(candidate_indices)],
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            )
        metadata_valid: bool = (
            int(metadata.num_kpoints) == kpoints.shape[0]
            and points_per_segment > 0
            and coordinate_mode.startswith(("reciprocal", "direct"))
            and len(candidate_labels) == len(candidate_indices)
            and bool(candidate_labels)
            and all(candidate_labels)
            and all(
                next_index > index
                for index, next_index in zip(
                    candidate_indices,
                    candidate_indices[1:],
                    strict=False,
                )
            )
            and endpoints_match
        )
        if metadata_valid:
            labels = candidate_labels
            label_indices = candidate_indices
            n_per_segment = points_per_segment

    kpath: KPath = make_kpath(
        kpoints=kpoints,
        labels=labels,
        label_indices=label_indices,
        n_per_segment=n_per_segment,
        kz=kpoints_cart[0, 2],
    )
    return kpath


@jaxtyped(typechecker=beartype)
def run_vasp_workflow(  # noqa: DOC502, DOC503, PLR0913
    hamiltonians_ev: Complex128[Array, "n_k n_orb n_orb"],
    *,
    crystal_geometry: CrystalGeometry,
    orbital_basis: OrbitalBasis,
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    experiment_geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    energy_axis: Float64[Array, " n_e"],
    detector_calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    directory: str = ".",
    eigenval_file: str = "EIGENVAL",
    procar_file: str = "PROCAR",
    doscar_file: Optional[str] = "DOSCAR",
    kpoints_file: Optional[str] = "KPOINTS",
    fermi_energy: Optional[ScalarFloat] = None,
    phase_loss: Literal["warn", "ignore", "error"] = "warn",
    check_dimensions: bool = True,
    eta: ScalarFloat = 1.0e-4,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
) -> DetectorRaster:
    """Run the explicit-H coherent cut workflow with VASP metadata.

    This compatibility boundary loads EIGENVAL, PROCAR, and optional path and
    Fermi-level metadata. It invokes :func:`vasp_to_diagonalized` to attach the
    requested crystal and orbital metadata, then calls
    :func:`simulate_arpes_cut` with every Plan-06/07/08 carrier explicit.

    PROCAR stores orbital weights, not complex coefficients. The adapter's
    positive square roots are therefore phase-dead and cannot support
    interference claims. They remain metadata in this workflow. Only
    ``hamiltonians_ev`` supplies the coherent resolvent and its inversion
    derivatives; this function never reconstructs it from VASP eigenpairs.

    :see: :class:`~.test_workflow.TestRunVaspWorkflow`

    Implementation Logic
    --------------------
    1. **Load weight-only VASP metadata**::

           context = load_vasp_context(..., procar_mode="legacy")

       The legacy parser shape is the explicit phase-dead PROCAR carrier.
    2. **Bind the explicit Hamiltonian axes**::

           hamiltonians_ev.shape == (n_k, n_orb, n_orb)

       A mismatch is rejected before any spectral or detector computation.
    3. **Construct the metadata carrier and path**::

           bands = vasp_to_diagonalized(..., phase_loss=phase_loss)
           kpath = _context_kpath(context, crystal_geometry)

       Valid KPOINTS labels survive; incomplete labels become an unlabeled
       self-describing path.
    4. **Execute the one coherent effects chain**::

           raster = simulate_arpes_cut(...)

       The caller-owned Hamiltonian and all physical carriers reach the
       canonical driver unchanged.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Explicit absolute-energy Hermitian Hamiltonians in eV. This tensor is
        the sole Hamiltonian and inversion authority.
    crystal_geometry : CrystalGeometry
        Crystal geometry associated with the VASP calculation.
    orbital_basis : OrbitalBasis
        Static atom and orbital registration for PROCAR projection metadata.
    radial_spec : RadialSpec
        Shell-shared radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and coherent channel phases.
    radial_quadrature : RadialQuadratureSpec
        Fixed radial quadrature contract.
    final_state : FinalStateSpec
        Explicit photoelectron final-state model.
    experiment_geometry : ExperimentGeometry
        Traced beam, sample, and detector geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy model.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned sampled energies relative to the Fermi level in eV.
    detector_calibration : DetectorCalibration
        Native detector bins, point-spread widths, and transmission domain.
    detector_effects : DetectorEffects
        Domain, transmission, background, sensitivity, and exposure state.
    directory : str, optional
        Directory containing VASP files. Default is current directory.
    eigenval_file : str, optional
        EIGENVAL filename relative to ``directory``.
    procar_file : str, optional
        PROCAR filename relative to ``directory``.
    doscar_file : Optional[str], optional
        DOSCAR filename for Fermi-level inference, or ``None``.
    kpoints_file : Optional[str], optional
        KPOINTS filename for plotting metadata, or ``None``.
    fermi_energy : Optional[ScalarFloat], optional
        Explicit Fermi energy in eV, or ``None`` to infer it.
    phase_loss : Literal["warn", "ignore", "error"], optional
        Policy passed to the phase-lossy PROCAR adapter. Default is ``"warn"``.
    check_dimensions : bool, optional
        Whether to validate shared VASP file axes. Default is ``True``.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static momentum chunk size. Default is 32.
    energy_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize live chunks in reverse mode. Default is ``True``.

    Returns
    -------
    raster : DetectorRaster
        Native-coordinate expected detector counts.

    Raises
    ------
    ValueError
        If VASP inputs disagree, the explicit Hamiltonian axes mismatch the
        parsed path or orbital basis, or the phase-loss policy requests an
        error.
    EquinoxRuntimeError
        If a traced Hamiltonian, physical carrier, or detector contract fails.

    Notes
    -----
    This wrapper has no tier, fidelity, hidden energy-axis construction,
    normalization, or momentum-broadening selector. PROCAR-derived
    coefficients have no D4/D5/S4 or inversion authority.
    """
    context: WorkflowContext = load_vasp_context(
        directory=directory,
        eigenval_file=eigenval_file,
        procar_file=procar_file,
        doscar_file=doscar_file,
        kpoints_file=kpoints_file,
        fermi_energy=fermi_energy,
        procar_mode="legacy",
        doscar_mode="legacy",
        check_dimensions=check_dimensions,
    )
    n_k: int = context.bands.kpoints.shape[0]
    n_orb: int = len(orbital_basis.n)
    if hamiltonians_ev.shape != (n_k, n_orb, n_orb):
        raise ValueError(
            "run_vasp_workflow: hamiltonians_ev must have shape "
            "(VASP n_k, basis n_orb, basis n_orb)"
        )
    if not isinstance(context.orb_proj, OrbitalProjection):
        raise TypeError(
            "run_vasp_workflow requires the weight-only PROCAR carrier"
        )
    bands: DiagonalizedBands = vasp_to_diagonalized(
        context.bands,
        context.orb_proj,
        crystal_geometry,
        orbital_basis,
        phase_loss=phase_loss,
    )
    kpath: KPath = _context_kpath(context, crystal_geometry)
    raster: DetectorRaster = simulate_arpes_cut(
        hamiltonians_by_domain=(hamiltonians_ev,),
        bands_by_domain=(bands,),
        radial_spec=radial_spec,
        matrix_element_params=matrix_element_params,
        radial_quadrature=radial_quadrature,
        final_state=final_state,
        geometry=experiment_geometry,
        self_energy=self_energy,
        kpath=kpath,
        energy_axis=energy_axis,
        detector_calibration=detector_calibration,
        detector_effects=detector_effects,
        eta=eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    return raster


__all__: list[str] = [
    "load_vasp_context",
    "prepare_projection",
    "run_vasp_workflow",
]
