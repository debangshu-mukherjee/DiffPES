"""Run high-level workflows for VASP-to-ARPES simulation.

Extended Summary
----------------
The module combines common workflow tasks. These tasks load VASP outputs,
select atoms, attach OAM channels, and run an ARPES simulation. Optional steps
apply momentum broadening and z-score normalization.

Routine Listings
----------------
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`prepare_projection`
    Prepare orbital projections for simulation.
:func:`run_vasp_workflow`
    Run an end-to-end VASP-to-ARPES workflow in one call.
:func:`simulate_context`
    Run a level-dispatched simulation from a loaded workflow context.
"""

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal, Optional, cast
from jaxtyping import Array, Float, jaxtyped

from diffpes.inout import (
    check_consistency,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_procar,
    select_atoms,
)
from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    DosType,
    KPathInfo,
    OrbitalBasis,
    ProjectionType,
    ScalarFloat,
    SpinOrbitalProjection,
    WorkflowContext,
    make_arpes_spectrum,
    make_orbital_projection,
    make_spin_orbital_projection,
    make_workflow_context,
)
from diffpes.utils import zscore_normalize

from .expanded import simulate_expanded
from .oam import compute_oam
from .resolution import apply_momentum_broadening


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
        oam: Float[Array, "K B A 3"] = compute_oam(prepared.projections)
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


@beartype
def _kpath_distances(
    kpoints: Float[Array, "K 3"],
) -> Float[Array, " K"]:
    """Compute cumulative k-path distances from k-point coordinates."""
    dk_vecs: Float[Array, "Km1 3"] = jnp.diff(kpoints, axis=0)
    dk_norms: Float[Array, " Km1"] = jnp.linalg.norm(dk_vecs, axis=1)
    distances: Float[Array, " K"] = jnp.concatenate(
        [jnp.zeros(1, dtype=kpoints.dtype), jnp.cumsum(dk_norms)]
    )
    return distances


@jaxtyped(typechecker=beartype)
def simulate_context(  # noqa: PLR0913
    context: WorkflowContext,
    level: str = "novice",
    atom_indices: Optional[list[int]] = None,
    attach_oam: bool = False,
    normalize: bool = False,
    dk: Optional[ScalarFloat] = None,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    fidelity: int = 25000,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 21.2,
    basis: Optional[OrbitalBasis] = None,
    atomic_numbers: Optional[tuple[int, ...]] = None,
) -> ArpesSpectrum:
    """Run a level-dispatched simulation from a loaded workflow context.

    Uses a parsed workflow context as the single input carrier. Optional
    post-processing preserves its energy axis and returns a new spectrum.

    :see: :class:`~.test_workflow.TestSimulateContext`

    Implementation Logic
    --------------------
    1. **Prepare the projection carrier**::

           prepared = prepare_projection(...)

       This step applies the requested atom selection and OAM construction.

    2. **Run the selected simulation**::

           spectrum = simulate_expanded(...)

       The level selects the forward model while JAX traces continuous inputs.

    3. **Apply optional post-processing**::

           intensity = apply_momentum_broadening(intensity, k_dist, dk)
           intensity = zscore_normalize(intensity)

       Each enabled operation transforms the intensity without changing the
       energy axis.

    4. **Construct the result**::

           result = make_arpes_spectrum(...)

       The factory validates the final spectrum carrier.

    Parameters
    ----------
    context : WorkflowContext
        Parsed VASP context from :func:`load_vasp_context`.
    level : str, optional
        Simulation level for :func:`diffpes.simul.simulate_expanded`.
    atom_indices : Optional[list[int]], optional
        Optional atom subset used before simulation.
    attach_oam : bool, optional
        If True and OAM is absent, compute OAM before simulation.
    normalize : bool, optional
        If True, apply global z-score normalization to output intensity.
    dk : Optional[ScalarFloat], optional
        If provided, apply momentum broadening along the k-axis.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV.
    gamma : ScalarFloat, optional
        Lorentzian broadening width in eV.
    fidelity : int, optional
        Number of points on the energy axis.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV.
    basis : Optional[OrbitalBasis], optional
        Atom-major subshell basis required by the basic tier.
    atomic_numbers : Optional[tuple[int, ...]], optional
        Atomic numbers required by the basic tier.

    Returns
    -------
    result : ArpesSpectrum
        Simulated ARPES spectrum after optional post-processing.
    """
    prepared: ProjectionType = prepare_projection(
        context.orb_proj,
        atom_indices=atom_indices,
        attach_oam=attach_oam,
    )

    spectrum: ArpesSpectrum = simulate_expanded(
        level=level,
        eigenbands=context.bands.eigenvalues,
        surface_orb=prepared.projections,
        ef=context.bands.fermi_energy,
        sigma=sigma,
        gamma=gamma,
        fidelity=fidelity,
        temperature=temperature,
        photon_energy=photon_energy,
        basis=basis,
        atomic_numbers=atomic_numbers,
    )

    intensity: Float[Array, "K E"] = spectrum.intensity
    if dk is not None:
        k_dist: Float[Array, " K"] = _kpath_distances(context.bands.kpoints)
        intensity = apply_momentum_broadening(intensity, k_dist, dk)

    if normalize:
        intensity = zscore_normalize(intensity)

    result: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=spectrum.energy_axis,
    )
    return result


@jaxtyped(typechecker=beartype)
def run_vasp_workflow(  # noqa: PLR0913
    level: str = "novice",
    directory: str = ".",
    eigenval_file: str = "EIGENVAL",
    procar_file: str = "PROCAR",
    doscar_file: Optional[str] = "DOSCAR",
    kpoints_file: Optional[str] = "KPOINTS",
    fermi_energy: Optional[ScalarFloat] = None,
    atom_indices: Optional[list[int]] = None,
    attach_oam: bool = False,
    normalize: bool = False,
    dk: Optional[ScalarFloat] = None,
    procar_mode: Literal["legacy", "full"] = "full",
    doscar_mode: Literal["legacy", "full"] = "legacy",
    check_dimensions: bool = True,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    fidelity: int = 25000,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 21.2,
    basis: Optional[OrbitalBasis] = None,
    atomic_numbers: Optional[tuple[int, ...]] = None,
) -> ArpesSpectrum:
    """Run an end-to-end VASP-to-ARPES workflow in one call.

    This helper loads VASP files into a :class:`WorkflowContext` and
    immediately delegates to :func:`simulate_context`.

    :see: :class:`~.test_workflow.TestRunVaspWorkflow`

    Implementation Logic
    --------------------
    1. **Load the workflow context**::

           context = load_vasp_context(...)

       The loader parses the files and checks their shared dimensions.
    2. **Run the configured simulation**::

           spectrum = simulate_context(...)

       The dispatcher applies the requested physics and post-processing steps.

    Parameters
    ----------
    level : str, optional
        Simulation complexity level (**static**; changing it retraces).
    directory : str, optional
        Directory containing the VASP files.
    eigenval_file : str, optional
        EIGENVAL filename relative to ``directory``.
    procar_file : str, optional
        PROCAR filename relative to ``directory``.
    doscar_file : Optional[str], optional
        Optional DOSCAR filename used to infer the Fermi energy.
    kpoints_file : Optional[str], optional
        Optional KPOINTS filename used for path metadata.
    fermi_energy : Optional[ScalarFloat], optional
        Explicit Fermi energy in eV, or ``None`` to infer it.
    atom_indices : Optional[list[int]], optional
        Optional zero-based atom subset.
    attach_oam : bool, optional
        Whether to derive OAM channels before simulation.
    normalize : bool, optional
        Whether to z-score normalize the final intensity.
    dk : Optional[ScalarFloat], optional
        Momentum broadening width in 1/Angstrom, or ``None``.
    procar_mode : Literal["legacy", "full"], optional
        PROCAR parsing mode.
    doscar_mode : Literal["legacy", "full"], optional
        DOSCAR parsing mode.
    check_dimensions : bool, optional
        Whether to validate dimensions across input files.
    sigma : ScalarFloat, optional
        Gaussian energy broadening width in eV.
    gamma : ScalarFloat, optional
        Lorentzian energy broadening width in eV.
    fidelity : int, optional
        Number of energy samples (**static**; changing it retraces).
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV.
    basis : Optional[OrbitalBasis], optional
        Atom-major subshell basis required by the basic tier.
    atomic_numbers : Optional[tuple[int, ...]], optional
        Atomic numbers required by the basic tier.

    Returns
    -------
    spectrum : ArpesSpectrum
        Final simulated spectrum.
    """
    context: WorkflowContext = load_vasp_context(
        directory=directory,
        eigenval_file=eigenval_file,
        procar_file=procar_file,
        doscar_file=doscar_file,
        kpoints_file=kpoints_file,
        fermi_energy=fermi_energy,
        procar_mode=procar_mode,
        doscar_mode=doscar_mode,
        check_dimensions=check_dimensions,
    )
    spectrum: ArpesSpectrum = simulate_context(
        context=context,
        level=level,
        atom_indices=atom_indices,
        attach_oam=attach_oam,
        normalize=normalize,
        dk=dk,
        sigma=sigma,
        gamma=gamma,
        fidelity=fidelity,
        temperature=temperature,
        photon_energy=photon_energy,
        basis=basis,
        atomic_numbers=atomic_numbers,
    )
    return spectrum


__all__: list[str] = [
    "load_vasp_context",
    "prepare_projection",
    "run_vasp_workflow",
    "simulate_context",
]
