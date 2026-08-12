"""PRIVATE: Stream bounded coherent source-intensity chunks.

Extended Summary
----------------
This private module bounds live matrix-element and spectral source storage.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.constants import CARTESIAN_COMPONENTS
from diffpes.matrixel import resolve_orbital_positions_cart
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SurfaceCell,
    TransitionSourceSchedule,
    make_transition_source_schedule,
)

from ._spectrum_validation import _checked_source_axes
from .kinematics import final_state_k_inv_ang, kinetic_energy_ev
from .polarization import lab_polarization_to_sample, sample_azimuth_rotation
from .spectral import _stream_spectral_intensity


def _vacuum_final_momentum_schedule(
    energy_axis: Float64[Array, " n_e"],
    geometry: ExperimentGeometry,
) -> Tuple[
    Float64[Array, " n_e"],
    Bool[Array, " n_e"],
]:
    """PRIVATE: Build the compact vacuum final-momentum schedule.

    Parameters
    ----------
    energy_axis : Float64[Array, " n_e"]
        Sampled energy relative to the Fermi level.
    geometry : ExperimentGeometry
        Supplies photon energy and work function.

    Returns
    -------
    final_norm : Float64[Array, " n_e"]
        Vacuum final-momentum magnitude for each sampled energy.
    emission_energy_valid : Bool[Array, " n_e"]
        Positive-energy and valid-final-state mask.

    Notes
    -----
    The streamed spectral block combines these one-dimensional values with
    the live initial-momentum chunk. It selects the positive detector-normal
    branch and applies the in-plane aperture condition there, so no complete
    ``(K, E, 3)`` carrier exists.
    """
    kinetic_energy: Float64[Array, " n_e"]
    energy_valid: Bool[Array, " n_e"]
    kinetic_energy, energy_valid = kinetic_energy_ev(
        geometry.photon_energy_ev,
        geometry.work_function_ev,
        energy_axis,
    )
    final_norm: Float64[Array, " n_e"]
    momentum_valid: Bool[Array, " n_e"]
    final_norm, momentum_valid = final_state_k_inv_ang(kinetic_energy)
    emission_energy_valid: Bool[Array, " n_e"] = energy_valid & momentum_valid
    result: Tuple[Float64[Array, " n_e"], Bool[Array, " n_e"]] = (
        final_norm,
        emission_energy_valid,
    )
    return result


def _padded_extent(size: int, chunk: int) -> int:
    """PRIVATE: Return the smallest chunk multiple containing ``size``.

    Parameters
    ----------
    size : int
        Positive physical axis size.
    chunk : int
        Positive static chunk size.

    Returns
    -------
    extent : int
        Smallest multiple of ``chunk`` not less than ``size``.
    """
    extent: int = ((size + chunk - 1) // chunk) * chunk
    return extent


def _stream_cartesian_intensity(  # noqa: DOC503, PLR0913, PLR0917
    hamiltonians_ev: Complex128[Array, "n_k n_orb n_orb"],
    k_cart: Float64[Array, "n_k 3"],
    basis: OrbitalBasis,
    positions_cart: Float64[Array, "n_orb 3"],
    depths: Float64[Array, " n_orb"],
    fermi_energy_ev: Float64[Array, ""],
    energy_axis: Float64[Array, " n_e"],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    eta: ScalarFloat,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
    use_inner_potential: bool,
) -> Float64[Array, "n_k n_e"]:
    """PRIVATE: Stream one Cartesian source through the resolvent scan.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Explicit absolute-energy orbital Hamiltonians.
    k_cart : Float64[Array, "n_k 3"]
        Initial crystal momenta in the registered sample Cartesian frame.
    basis : OrbitalBasis
        Static orbital basis shared by all physical carriers.
    positions_cart : Float64[Array, "n_orb 3"]
        Orbital centres in sample-frame Cartesian Angstrom coordinates.
    depths : Float64[Array, " n_orb"]
        Orbital depths for coherent attenuation; exact zeros in bulk modes.
    fermi_energy_ev : Float64[Array, ""]
        Absolute Fermi energy in eV.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned relative-energy samples.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and phase coordinates.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit final-state selection.
    geometry : ExperimentGeometry
        Experiment and optical geometry.
    self_energy : SelfEnergyModel
        Causal self-energy model.
    eta : ScalarFloat
        Positive resolvent regulator in eV.
    k_chunk : int
        Static k chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Whether to rematerialize live chunks in reverse mode.
    use_inner_potential : bool
        Whether final momenta use exact finite-energy inner-potential kz.

    Returns
    -------
    intensity : Float64[Array, "n_k n_e"]
        Intrinsic physical intensity on the caller-owned source grid.

    Raises
    ------
    ValueError
        If the caller-owned energy axis is empty or static axes disagree.

    Notes
    -----
    Padding values stay finite and inside the sampled self-energy interval;
    masks remove them exactly from the physical result.
    """
    n_k: int = k_cart.shape[0]
    n_energy: int = energy_axis.shape[0]
    if n_energy < 1:
        raise ValueError("energy_axis must contain at least one sample")
    n_orb: int = len(basis.n)
    if (
        hamiltonians_ev.shape != (n_k, n_orb, n_orb)
        or positions_cart.shape != (n_orb, CARTESIAN_COMPONENTS)
        or depths.shape != (n_orb,)
        or type(use_inner_potential) is not bool
    ):
        raise ValueError(
            "Cartesian source, Hamiltonian, and orbital axes disagree"
        )
    checked_energy_axis: Float64[Array, " n_e"] = eqx.error_if(
        energy_axis,
        ~jnp.all(jnp.isfinite(energy_axis))
        | jnp.any(jnp.diff(energy_axis) <= 0.0),
        "energy_axis must be finite and strictly increasing",
    )
    checked_k_cart: Float64[Array, "n_k 3"] = eqx.error_if(
        k_cart,
        ~jnp.all(jnp.isfinite(k_cart)),
        "initial Cartesian momenta must be finite",
    )
    final_norm: Float64[Array, " n_e"]
    emission_energy_valid: Bool[Array, " n_e"]
    final_norm, emission_energy_valid = _vacuum_final_momentum_schedule(
        checked_energy_axis, geometry
    )
    padded_k: int = _padded_extent(n_k, k_chunk)
    padded_energy: int = _padded_extent(n_energy, energy_chunk)
    pad_k: int = padded_k - n_k
    pad_energy: int = padded_energy - n_energy
    padded_hamiltonians: Complex128[Array, "n_k_padded n_orb n_orb"] = jnp.pad(
        hamiltonians_ev, ((0, pad_k), (0, 0), (0, 0))
    )
    padded_k_cart: Float64[Array, "n_k_padded 3"] = jnp.pad(
        checked_k_cart, ((0, pad_k), (0, 0))
    )
    padded_final_norm: Float64[Array, " n_e_padded"] = jnp.pad(
        final_norm,
        (0, pad_energy),
        constant_values=final_norm[-1],
    )
    padded_emission_energy_valid: Bool[Array, " n_e_padded"] = jnp.pad(
        emission_energy_valid,
        (0, pad_energy),
        constant_values=False,
    )
    padded_energy_axis: Float64[Array, " n_e_padded"] = jnp.pad(
        checked_energy_axis,
        (0, pad_energy),
        constant_values=checked_energy_axis[-1],
    )
    k_valid: Bool[Array, " n_k_padded"] = jnp.arange(padded_k) < n_k
    energy_valid: Bool[Array, " n_e_padded"] = (
        jnp.arange(padded_energy) < n_energy
    )
    sample_orientation: Float64[Array, "3 3"] = sample_azimuth_rotation(
        geometry.sample_azimuth
    )
    polarization_sample: Complex128[Array, " 3"] = lab_polarization_to_sample(
        geometry.polarization,
        sample_orientation,
    )
    schedule: TransitionSourceSchedule = make_transition_source_schedule(
        k_i_cart=padded_k_cart,
        final_norm=padded_final_norm,
        emission_energy_valid=padded_emission_energy_valid,
        positions_cart=positions_cart,
        depths=depths,
        polarization_sample_cart=polarization_sample,
        mean_free_path_ang=geometry.mean_free_path_ang,
        radial=radial_spec,
        matrix_element=matrix_element_params,
        quadrature=radial_quadrature,
        final_state=final_state,
        inner_potential_geometry=(geometry if use_inner_potential else None),
    )
    padded_intensity: Float64[Array, "n_k_padded n_e_padded"] = (
        _stream_spectral_intensity(
            padded_hamiltonians,
            padded_energy_axis,
            k_valid,
            energy_valid,
            schedule,
            self_energy,
            fermi_energy_ev,
            geometry.temperature_k,
            eta,
            k_chunk=k_chunk,
            omega_chunk=energy_chunk,
            checkpoint=checkpoint,
        )
    )
    intensity: Float64[Array, "n_k n_e"] = padded_intensity[:n_k, :n_energy]
    return intensity


def _checked_coherent_slab_bands(  # noqa: DOC502, DOC503
    bands: DiagonalizedBands,
    surface_cell: SurfaceCell,
) -> DiagonalizedBands:
    """PRIVATE: Bind one slab eigensystem to its surface frame.

    Parameters
    ----------
    bands : DiagonalizedBands
        Depth-bearing slab data whose geometry is already in the surface
        frame.
    surface_cell : SurfaceCell
        Surface carrier returned by the same slab construction.

    Returns
    -------
    checked : DiagonalizedBands
        The unchanged bands with the frame guard attached to the reciprocal
        lattice consumed by source-coordinate conversion.

    Raises
    ------
    EquinoxRuntimeError
        If the slab in-plane lattice rows differ from the surface cell, or
        the slab lattice is not aligned with positive surface-frame z.

    Notes
    -----
    The slab construction guarantees only that the slab lattice begins
    with ``surface_cell.in_plane_vectors`` and ends with
    ``(0, 0, height > 0)``. ``DiagonalizedBands`` does not retain enough bulk
    provenance to reconstruct or compare Miller coefficients or rotations.
    """
    slab_lattice: Float64[Array, "3 3"] = bands.geometry.lattice
    surface_scale: Float64[Array, ""] = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(slab_lattice)),
    )
    frame_tolerance: Float64[Array, ""] = 1.0e-10 * surface_scale
    in_plane_matches: Bool[Array, ""] = jnp.allclose(
        slab_lattice[:2],
        surface_cell.in_plane_vectors,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    surface_aligned: Bool[Array, ""] = (
        jnp.all(jnp.abs(slab_lattice[:2, 2]) <= frame_tolerance)
        & jnp.all(jnp.abs(slab_lattice[2, :2]) <= frame_tolerance)
        & (slab_lattice[2, 2] > 0.0)
    )
    checked_reciprocal: Float64[Array, "3 3"] = eqx.error_if(
        bands.geometry.reciprocal,
        ~(in_plane_matches & surface_aligned),
        "coherent_slab SurfaceCell must match the DiagonalizedBands "
        "surface frame",
    )
    checked_geometry: CrystalGeometry = eqx.tree_at(
        lambda item: item.reciprocal,
        bands.geometry,
        checked_reciprocal,
    )
    checked: DiagonalizedBands = eqx.tree_at(
        lambda item: item.geometry,
        bands,
        checked_geometry,
    )
    return checked


def _stream_domain_intensity(  # noqa: DOC503, PLR0913, PLR0917
    hamiltonians_ev: Complex128[Array, "n_k n_orb n_orb"],
    bands: DiagonalizedBands,
    source_kpoints: Float64[Array, "n_k 3"],
    energy_axis: Float64[Array, " n_e"],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    eta: ScalarFloat,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
    use_inner_potential: bool = False,
    surface_cell: SurfaceCell | None = None,
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Float64[Array, "n_k 3"],
]:
    """PRIVATE: Resolve one domain and stream its physical intensity.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Explicit absolute-energy Hamiltonians.
    bands : DiagonalizedBands
        Geometry, basis, positions, depths, and Fermi metadata.
    source_kpoints : Float64[Array, "n_k 3"]
        Fractional source points required to match ``bands``.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing relative-energy samples.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and phase coordinates.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final state.
    geometry : ExperimentGeometry
        Traced experiment geometry.
    self_energy : SelfEnergyModel
        Causal self-energy model.
    eta : ScalarFloat
        Positive resolvent regulator in eV.
    k_chunk : int
        Static k-point chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Reverse-mode rematerialization selector.
    use_inner_potential : bool, optional
        Use exact finite-energy internal final kz. Default is ``False``.
    surface_cell : SurfaceCell | None, optional
        Surface frame required by coherent-slab mode. Default is
        ``None``.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]]
        Intrinsic intensity and complete Cartesian source points.

    Raises
    ------
    ValueError
        If the public sampled-energy axis contains fewer than two points.
        If the coherent-slab route lacks its surface cell.
    EquinoxRuntimeError
        If a coherent slab geometry disagrees with its surface frame.

    Notes
    -----
    Native mode retains the 08a vacuum branch. Coherent-slab mode selects the
    exact finite-energy internal branch without adding bulk Lorentzian nodes.
    """
    minimum_points: int = 2
    if energy_axis.shape[0] < minimum_points:
        raise ValueError("energy_axis must contain at least two samples")
    checked_bands: DiagonalizedBands = bands
    if use_inner_potential:
        if surface_cell is None:
            raise ValueError("coherent_slab requires its surface cell")
        checked_bands = _checked_coherent_slab_bands(bands, surface_cell)
    k_cart: Float64[Array, "n_k 3"] = _checked_source_axes(
        checked_bands, source_kpoints
    )
    n_orb: int = len(checked_bands.basis.n)
    depths: Float64[Array, " n_orb"] = (
        jnp.zeros((n_orb,), dtype=jnp.float64)
        if checked_bands.depths is None
        else checked_bands.depths
    )
    intensity: Float64[Array, "n_k n_e"] = _stream_cartesian_intensity(
        hamiltonians_ev,
        k_cart,
        checked_bands.basis,
        resolve_orbital_positions_cart(checked_bands),
        depths,
        checked_bands.fermi_energy,
        energy_axis,
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        geometry,
        self_energy,
        eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
        use_inner_potential=use_inner_potential,
    )
    result: Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]] = (
        intensity,
        k_cart,
    )
    return result


__all__: list[str] = []
