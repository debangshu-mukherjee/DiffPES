"""High-level workflow helpers for VASP-to-ARPES simulation.

Extended Summary
----------------
Provides convenience functions that combine common multi-step tasks:
loading VASP outputs, selecting atom subsets, attaching OAM channels,
running ARPES simulations by level, and optional post-processing
(momentum broadening and z-score normalization).

Routine Listings
----------------
:class:`WorkflowContext`
    Bundle of parsed VASP inputs for simulation workflows.
:func:`load_vasp_context`
    Read EIGENVAL/PROCAR and optional DOSCAR/KPOINTS into a context.
:func:`prepare_projection`
    Select atoms and optionally attach OAM to orbital projections.
:func:`simulate_context`
    Run a level-dispatched ARPES simulation from a workflow context.
:func:`run_vasp_workflow`
    End-to-end convenience wrapper: load, prepare, simulate.
"""

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal, NamedTuple, Optional, Union
from jaxtyping import Array, Float

from arpyes.inout.doscar import read_doscar
from arpyes.inout.eigenval import read_eigenval
from arpyes.inout.helpers import check_consistency, select_atoms
from arpyes.inout.kpoints import read_kpoints
from arpyes.inout.procar import read_procar
from arpyes.types import (
    ArpesSpectrum,
    BandStructure,
    DensityOfStates,
    FullDensityOfStates,
    KPathInfo,
    OrbitalProjection,
    SpinOrbitalProjection,
    make_arpes_spectrum,
    make_orbital_projection,
    make_spin_orbital_projection,
)
from arpyes.types.aliases import ScalarFloat
from arpyes.utils import zscore_normalize

from .expanded import simulate_expanded
from .oam import compute_oam
from .resolution import apply_momentum_broadening

ProjectionType = Union[OrbitalProjection, SpinOrbitalProjection]
DosType = Union[DensityOfStates, FullDensityOfStates]


class WorkflowContext(NamedTuple):
    """Container for parsed VASP inputs used by workflow helpers.

    Attributes
    ----------
    bands : BandStructure
        Band eigenvalues and k-point coordinates from EIGENVAL.
    orb_proj : OrbitalProjection or SpinOrbitalProjection
        Orbital projections from PROCAR, optionally including spin.
    kpath : Optional[KPathInfo]
        Parsed KPOINTS metadata when available.
    dos : Optional[DensityOfStates or FullDensityOfStates]
        Parsed DOSCAR data when available.
    """

    bands: BandStructure
    orb_proj: ProjectionType
    kpath: Optional[KPathInfo]
    dos: Optional[DosType]


@beartype
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

    Implementation Logic
    --------------------
    1. Resolve the root directory and optional DOSCAR/KPOINTS paths.
    2. Determine Fermi energy from explicit input or DOSCAR.
    3. Parse EIGENVAL in legacy mode into a ``BandStructure``.
    4. Parse PROCAR (legacy or full) into an orbital projection object.
    5. Parse optional KPOINTS metadata when the file exists.
    6. Optionally validate dimensional consistency across parsed files.

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
        Manual Fermi energy in eV. If ``None``, DOSCAR is used when
        available, otherwise 0.0.
    procar_mode : {"legacy", "full"}, optional
        PROCAR return mode. ``"full"`` preserves spin data when present.
    doscar_mode : {"legacy", "full"}, optional
        DOSCAR return mode when DOSCAR is read.
    check_dimensions : bool, optional
        If True, run cross-file consistency checks.

    Returns
    -------
    WorkflowContext
        Loaded VASP data bundled for downstream workflow calls.

    Raises
    ------
    FileNotFoundError
        If DOSCAR is required to infer Fermi energy but is missing.
    """
    root: Path = Path(directory)

    dos: Optional[DosType] = None
    if fermi_energy is None:
        if doscar_file is None:
            resolved_fermi: float = 0.0
        else:
            dos_path_req: Path = root / doscar_file
            if not dos_path_req.exists():
                msg: str = (
                    "DOSCAR is required to infer fermi_energy but was "
                    f"not found: {dos_path_req}"
                )
                raise FileNotFoundError(msg)
            dos = read_doscar(str(dos_path_req), return_mode=doscar_mode)
            resolved_fermi = float(dos.fermi_energy)
    else:
        resolved_fermi = float(fermi_energy)
        if doscar_file is not None:
            dos_path_opt: Path = root / doscar_file
            if dos_path_opt.exists():
                dos = read_doscar(str(dos_path_opt), return_mode=doscar_mode)

    bands: BandStructure = read_eigenval(
        str(root / eigenval_file),
        fermi_energy=resolved_fermi,
        return_mode="legacy",
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

    context: WorkflowContext = WorkflowContext(
        bands=bands,
        orb_proj=orb_proj,
        kpath=kpath,
        dos=dos,
    )
    return context


@beartype
def prepare_projection(
    orb_proj: ProjectionType,
    atom_indices: Optional[list[int]] = None,
    attach_oam: bool = False,
) -> ProjectionType:
    """Prepare orbital projections for simulation.

    Applies common pre-processing steps used in MATLAB-like workflows:
    selecting atom subsets and attaching OAM channels derived from
    orbital projections.

    Parameters
    ----------
    orb_proj : OrbitalProjection or SpinOrbitalProjection
        Input projection object.
    atom_indices : Optional[list[int]], optional
        Optional 0-based atom indices to keep.
    attach_oam : bool, optional
        If True and OAM is absent, compute OAM and attach it.

    Returns
    -------
    OrbitalProjection or SpinOrbitalProjection
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


@beartype
def simulate_context(  # noqa: PLR0913
    context: WorkflowContext,
    level: str = "advanced",
    atom_indices: Optional[list[int]] = None,
    attach_oam: bool = False,
    normalize: bool = False,
    dk: Optional[ScalarFloat] = None,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    fidelity: int = 25000,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Run a level-dispatched simulation from a loaded workflow context.

    Parameters
    ----------
    context : WorkflowContext
        Parsed VASP context from :func:`load_vasp_context`.
    level : str, optional
        Simulation level for :func:`arpyes.simul.simulate_expanded`.
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
    polarization : str, optional
        Polarization type (e.g. ``"unpolarized"``, ``"LHP"``).
    incident_theta : ScalarFloat, optional
        Incident polar angle in degrees.
    incident_phi : ScalarFloat, optional
        Incident azimuth angle in degrees.
    polarization_angle : ScalarFloat, optional
        Linear polarization angle in radians.
    ls_scale : ScalarFloat, optional
        SOC scale used only when ``level="soc"``.

    Returns
    -------
    ArpesSpectrum
        Simulated ARPES spectrum after optional post-processing.
    """
    prepared: ProjectionType = prepare_projection(
        context.orb_proj,
        atom_indices=atom_indices,
        attach_oam=attach_oam,
    )

    surface_spin: Optional[Float[Array, "K B A 6"]] = prepared.spin
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
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
        surface_spin=surface_spin,
        ls_scale=ls_scale,
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


@beartype
def run_vasp_workflow(  # noqa: PLR0913
    level: str = "advanced",
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
    photon_energy: ScalarFloat = 11.0,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Run an end-to-end VASP-to-ARPES workflow in one call.

    This helper loads VASP files into a :class:`WorkflowContext` and
    immediately delegates to :func:`simulate_context`.

    Returns
    -------
    ArpesSpectrum
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
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
        ls_scale=ls_scale,
    )
    return spectrum


__all__: list[str] = [
    "WorkflowContext",
    "load_vasp_context",
    "prepare_projection",
    "run_vasp_workflow",
    "simulate_context",
]
