r"""Compose the coherent ARPES forward and photon-energy-scan drivers.

Extended Summary
----------------
This module composes the intrinsic Plan-06/07 observable with the Plan-08a
detector chain and the Plan-08b bulk-:math:`k_z` integral.  It builds
matrix-element source kets only inside each live energy/k-point/node chunk.
It solves the explicit orbital Hamiltonian through the degeneracy-safe
resolvent and multiplies by sampled Fermi occupation.  It materializes a
self-describing source carrier only after any node-local reduction.  The
common detector operator then handles domain mapping, mixing, transmission,
resolution, backgrounds, sensitivity, and expected counts.

The Hamiltonian is an explicit input.  ``DiagonalizedBands`` supplies orbital,
crystal, Fermi-level, and source-coordinate metadata, but this module never
reconstructs a Hamiltonian from its eigenvectors.  Such a value-only
reconstruction silently replaces the native Hamiltonian derivative with an
eigensystem derivative.  The complete Plan-06 parameter
surface is likewise explicit: ``RadialSpec``, ``MatrixElementParams``,
``RadialQuadratureSpec``, and ``FinalStateSpec`` all cross the driver boundary.

The deterministic chain is::

    node-local source -> A(k,w) f_FD -> optional kz reduction
        -> source carrier -> detector effects

There is one canonical driver surface with four mutually exclusive
out-of-plane routes: retained native direct, exact bulk direct, wrapped
finite-width bulk :math:`k_z`, and coherent slab.  The mode string selects
physical carrier ownership rather than a fidelity tier, and no tier dispatcher
exists.  The caller owns the sampled energy and photon-energy axes.  Display
normalization is an explicit helper and is never called by a physical driver.

Routine Listings
----------------
:func:`hv_map_at_energy`
    Interpolate a photon-energy scan at one sampled binding energy.
:func:`normalize_intensity`
    Return an explicit display-only normalization of carrier values.
:func:`simulate_arpes`
    Simulate the canonical detector raster.
:func:`simulate_arpes_cut`
    Simulate the canonical path-cut detector raster.
:func:`simulate_hv_scan`
    Simulate a single-domain pre-detector photon-energy scan.

Notes
-----
The source scan pads arbitrary caller shapes to static chunk multiples and
masks the padding exactly.  It never materializes a complete ``(K, E, B)`` or
``(K, E, n_out, n_orb)`` source tensor.  Every two-dimensional ``KGrid`` must
be separable in the registered sample Cartesian frame.  The validator rejects
rotated or otherwise nonseparable rasters instead of assigning false
one-dimensional ``kx`` and ``ky`` axes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Tuple, Union
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.tightb import bloch_hamiltonian_batch
from diffpes.types import (
    CARTESIAN_COMPONENTS,
    ArpesCube,
    ArpesSpectrum,
    CrystalGeometry,
    DetectorCalibration,
    DetectorEffects,
    DetectorRaster,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    KPath,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SurfaceCell,
    TBModel,
    make_arpes_cube,
    make_arpes_spectrum,
)

from . import effects as _effects
from . import spectral as _spectral
from .kinematics import (
    final_state_k_inv_ang,
    kinetic_energy_ev,
    kz_from_inner_potential,
)
from .matrixel import resolve_orbital_positions_cart
from .polarization import (
    lab_polarization_to_sample,
    sample_azimuth_rotation,
)


def _basis_key(basis: OrbitalBasis) -> Tuple[Tuple[object, ...], ...]:
    """PRIVATE: Return the exact static identity of an orbital basis.

    Parameters
    ----------
    basis : OrbitalBasis
        Static orbital metadata.

    Returns
    -------
    key : Tuple[Tuple[object, ...], ...]
        Hashable field-wise identity.
    """
    key: Tuple[Tuple[object, ...], ...] = (
        basis.atom_indices,
        basis.n,
        basis.l,
        basis.m,
        basis.spin,
        basis.labels,
    )
    return key


def _validate_static_inputs(  # noqa: DOC105
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
) -> None:
    """PRIVATE: Validate static domain, basis, and chunk invariants.

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
        Explicit absolute-energy Hamiltonians for every domain.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers.
    radial_spec : RadialSpec
        Shared radial carrier.
    matrix_element_params : MatrixElementParams
        Shared matrix-element carrier.
    k_chunk : int
        Static k-point chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Static rematerialization selector.

    Raises
    ------
    ValueError
        If a static domain, basis, shape, or chunk contract disagrees.
    """
    if not bands_by_domain:
        raise ValueError("simulate_arpes requires at least one source domain")
    if len(hamiltonians_by_domain) != len(bands_by_domain):
        raise ValueError(
            "hamiltonians_by_domain and bands_by_domain must have equal length"
        )
    if type(k_chunk) is not int or k_chunk <= 0:
        raise ValueError("k_chunk must be a positive integer")
    if type(energy_chunk) is not int or energy_chunk <= 0:
        raise ValueError("energy_chunk must be a positive integer")
    if type(checkpoint) is not bool:
        raise ValueError("checkpoint must be a boolean")
    radial_key: Tuple[Tuple[object, ...], ...] = _basis_key(radial_spec.basis)
    matrix_key: Tuple[Tuple[object, ...], ...] = _basis_key(
        matrix_element_params.basis
    )
    if radial_key != matrix_key or (
        radial_spec.radial_shell_index
        != matrix_element_params.radial_shell_index
    ):
        raise ValueError(
            "radial_spec and matrix_element_params must share one basis "
            "and shell partition"
        )
    domain: DiagonalizedBands
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"]
    for domain, hamiltonians in zip(
        bands_by_domain, hamiltonians_by_domain, strict=True
    ):
        n_k: int = domain.kpoints.shape[0]
        n_orb: int = len(domain.basis.n)
        if _basis_key(domain.basis) != radial_key:
            raise ValueError(
                "every domain must share the explicit radial orbital basis"
            )
        if hamiltonians.shape != (n_k, n_orb, n_orb):
            raise ValueError(
                "each Hamiltonian array must have shape (n_k, n_orb, n_orb)"
            )


def _validate_kz_mode_inputs(  # noqa: PLR0912, PLR0913
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    bulk_models_by_domain: Tuple[TBModel, ...] | None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None,
    kz_mode: str,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
) -> None:
    """PRIVATE: Validate one mutually exclusive out-of-plane driver mode.

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
        Native or coherent-slab Hamiltonians.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Native or coherent-slab metadata.
    radial_spec : RadialSpec
        Shared radial carrier.
    matrix_element_params : MatrixElementParams
        Shared matrix-element carrier.
    bulk_models_by_domain : Tuple[TBModel, ...] | None
        Bulk models for direct or finite-width integration.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None
        Exact surface frames for bulk or coherent-slab modes.
    kz_nodes_frac : Float64[Array, " n_kz"] | None
        Static uniform fractional nodes for finite-width integration.
    kz_mode : str
        One of the four registered mode names.
    k_chunk : int
        Static k-point chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Static rematerialization selector.

    Raises
    ------
    ValueError
        If carriers, nodes, or static controls do not match the selected mode.
    """
    modes: Tuple[str, ...] = (
        "native_direct",
        "bulk_direct",
        "bulk_kz",
        "coherent_slab",
    )
    if kz_mode not in modes:
        raise ValueError(
            "kz_mode must be 'native_direct', 'bulk_direct', 'bulk_kz', or "
            "'coherent_slab'"
        )
    if kz_mode == "native_direct":
        if (
            bulk_models_by_domain is not None
            or surface_cells_by_domain is not None
            or kz_nodes_frac is not None
        ):
            raise ValueError(
                "native_direct rejects bulk models, surface cells, and kz "
                "nodes"
            )
        _validate_static_inputs(
            hamiltonians_by_domain,
            bands_by_domain,
            radial_spec,
            matrix_element_params,
            k_chunk=k_chunk,
            energy_chunk=energy_chunk,
            checkpoint=checkpoint,
        )
        return
    if kz_mode == "coherent_slab":
        if bulk_models_by_domain is not None or kz_nodes_frac is not None:
            raise ValueError(
                "coherent_slab rejects bulk models and finite-kz nodes"
            )
        if surface_cells_by_domain is None or len(
            surface_cells_by_domain
        ) != len(bands_by_domain):
            raise ValueError(
                "coherent_slab requires one surface cell per source domain"
            )
        _validate_static_inputs(
            hamiltonians_by_domain,
            bands_by_domain,
            radial_spec,
            matrix_element_params,
            k_chunk=k_chunk,
            energy_chunk=energy_chunk,
            checkpoint=checkpoint,
        )
        if any(domain.depths is None for domain in bands_by_domain):
            raise ValueError(
                "coherent_slab requires depth-bearing diagonalized bands"
            )
        return
    if hamiltonians_by_domain or bands_by_domain:
        raise ValueError(
            "bulk_direct and bulk_kz require empty native Hamiltonian and "
            "band tuples"
        )
    if (
        bulk_models_by_domain is None
        or surface_cells_by_domain is None
        or not bulk_models_by_domain
        or len(bulk_models_by_domain) != len(surface_cells_by_domain)
    ):
        raise ValueError(
            "bulk modes require equal nonempty bulk-model and surface-cell "
            "tuples"
        )
    if type(k_chunk) is not int or k_chunk <= 0:
        raise ValueError("k_chunk must be a positive integer")
    if type(energy_chunk) is not int or energy_chunk <= 0:
        raise ValueError("energy_chunk must be a positive integer")
    if type(checkpoint) is not bool:
        raise ValueError("checkpoint must be a boolean")
    radial_key: Tuple[Tuple[object, ...], ...] = _basis_key(radial_spec.basis)
    matrix_key: Tuple[Tuple[object, ...], ...] = _basis_key(
        matrix_element_params.basis
    )
    if radial_key != matrix_key or (
        radial_spec.radial_shell_index
        != matrix_element_params.radial_shell_index
    ):
        raise ValueError(
            "radial_spec and matrix_element_params must share one basis and "
            "shell partition"
        )
    model: TBModel
    for model in bulk_models_by_domain:
        if _basis_key(model.basis) != radial_key:
            raise ValueError(
                "every bulk model must share the radial orbital basis"
            )
        if model.depths is not None:
            raise ValueError("bulk modes require models without slab depths")
    if kz_mode == "bulk_direct":
        if kz_nodes_frac is not None:
            raise ValueError("bulk_direct rejects finite-width kz nodes")
        return
    minimum_nodes: int = 2
    if kz_nodes_frac is None or kz_nodes_frac.ndim != 1:
        raise ValueError("bulk_kz requires a one-dimensional kz node array")
    if kz_nodes_frac.shape[0] < minimum_nodes:
        raise ValueError("bulk_kz requires at least two kz nodes")


def _checked_source_axes(  # noqa: DOC503 -- traced guards raise indirectly.
    bands: DiagonalizedBands,
    source_kpoints: Float64[Array, "n_k 3"],
) -> Float64[Array, "n_k 3"]:
    """PRIVATE: Bind the declared source grid to one domain carrier.

    Parameters
    ----------
    bands : DiagonalizedBands
        Domain whose k-points must match the declared source grid.
    source_kpoints : Float64[Array, "n_k 3"]
        Fractional grid or path points.

    Returns
    -------
    cartesian : Float64[Array, "n_k 3"]
        Domain points in the registered sample Cartesian frame.

    Raises
    ------
    ValueError
        If the static point axes disagree.
    EquinoxRuntimeError
        If the traced fractional points do not match.
    """
    if bands.kpoints.shape != source_kpoints.shape:
        raise ValueError("source and band k-point axes must agree")
    checked_kpoints: Float64[Array, "n_k 3"] = eqx.error_if(
        bands.kpoints,
        ~jnp.allclose(
            bands.kpoints,
            source_kpoints,
            rtol=1.0e-12,
            atol=1.0e-13,
        ),
        "band k-points must match the declared source grid",
    )
    cartesian: Float64[Array, "n_k 3"] = (
        checked_kpoints @ bands.geometry.reciprocal
    )
    return cartesian


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
    schedule: _spectral._TransitionSourceSchedule = (
        _spectral._TransitionSourceSchedule(
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
            inner_potential_geometry=(
                geometry if use_inner_potential else None
            ),
        )
    )
    padded_intensity: Float64[Array, "n_k_padded n_e_padded"] = (
        _spectral._stream_spectral_intensity(
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
    """PRIVATE: Bind one slab eigensystem to its Plan-05 surface frame.

    Parameters
    ----------
    bands : DiagonalizedBands
        Depth-bearing slab data whose geometry is already in the surface
        frame.
    surface_cell : SurfaceCell
        Surface carrier returned by the same Plan-05 slab construction.

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
    Plan 05 reconstructibly guarantees only that the slab lattice begins
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
        Plan-05 surface frame required by coherent-slab mode. Default is
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
        If a coherent slab geometry disagrees with its Plan-05 surface frame.

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
            raise ValueError("coherent_slab requires its Plan-05 surface cell")
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


def _bulk_source_parallel_cartesian(  # noqa: DOC502, DOC503
    source_kpoints: Float64[Array, "n_k 3"],
    model: TBModel,
    surface_cell: SurfaceCell,
) -> Float64[Array, "n_k 3"]:
    """PRIVATE: Resolve bulk-fractional points onto the surface plane.

    Parameters
    ----------
    source_kpoints : Float64[Array, "n_k 3"]
        Caller-owned fractional points in ``model.geometry``.
    model : TBModel
        Bulk tight-binding model defining the reciprocal conversion.
    surface_cell : SurfaceCell
        Exact bulk-to-surface frame and primitive stacking metadata.

    Returns
    -------
    k_parallel : Float64[Array, "n_k 3"]
        Physical surface-plane momenta in inverse Angstroms.

    Raises
    ------
    ValueError
        If the source does not have one trailing Cartesian axis.
    EquinoxRuntimeError
        If the source or surface/bulk frame is invalid.

    Notes
    -----
    Retain only the physical surface projection. Exact finite-energy kz
    replaces the input normal coordinate in both bulk modes.
    """
    if source_kpoints.ndim != 2 or source_kpoints.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError("bulk source points must have shape (n_k, 3)")
    bulk_cartesian: Float64[Array, "n_k 3"] = (
        source_kpoints @ model.geometry.reciprocal
    )
    surface_cartesian: Float64[Array, "n_k 3"] = (
        bulk_cartesian @ surface_cell.rotation.T
    )
    normal_hat: Float64[Array, " 3"] = _effects._surface_kz_frame(
        surface_cell, model.geometry
    )[2]
    normal_component: Float64[Array, " n_k"] = jnp.einsum(
        "ki,i->k", surface_cartesian, normal_hat
    )
    k_parallel: Float64[Array, "n_k 3"] = (
        surface_cartesian - normal_component[:, None] * normal_hat
    )
    checked_parallel: Float64[Array, "n_k 3"] = eqx.error_if(
        k_parallel,
        ~jnp.all(jnp.isfinite(source_kpoints))
        | ~jnp.all(jnp.isfinite(k_parallel)),
        "bulk source points and their surface projection must be finite",
    )
    return checked_parallel


def _bulk_orbital_positions_surface_cartesian(
    model: TBModel,
    surface_cell: SurfaceCell,
) -> Float64[Array, "n_orb 3"]:
    """PRIVATE: Resolve bulk orbital centres into the surface frame.

    Parameters
    ----------
    model : TBModel
        Bulk tight-binding model with fractional orbital provenance.
    surface_cell : SurfaceCell
        Active bulk-to-surface rotation.

    Returns
    -------
    positions_surface : Float64[Array, "n_orb 3"]
        Orbital centres in surface-frame Cartesian Angstrom coordinates.
    """
    positions_fractional: Float64[Array, "n_orb 3"]
    if model.orbital_positions is None:
        atom_indices: Array = jnp.asarray(
            model.basis.atom_indices,
            dtype=jnp.int32,
        )
        positions_fractional = model.geometry.positions[atom_indices]
    else:
        positions_fractional = model.orbital_positions
    positions_bulk: Float64[Array, "n_orb 3"] = (
        positions_fractional @ model.geometry.lattice
    )
    positions_surface: Float64[Array, "n_orb 3"] = (
        positions_bulk @ surface_cell.rotation.T
    )
    return positions_surface


def _exact_folded_center_and_mask(  # noqa: DOC502, DOC503
    k_parallel_cart: Float64[Array, "n_k 3"],
    energy_axis: Float64[Array, " n_e"],
    geometry: ExperimentGeometry,
    direct_surface: Float64[Array, "3 3"],
    normal_hat: Float64[Array, " 3"],
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Bool[Array, "n_k n_e"],
]:
    """PRIVATE: Compute exact folded centres in a validated surface frame.

    Parameters
    ----------
    k_parallel_cart : Float64[Array, "n_k 3"]
        Physical surface-plane momenta.
    energy_axis : Float64[Array, " n_e"]
        Relative-energy samples.
    geometry : ExperimentGeometry
        Photon energy, work function, and inner potential.
    direct_surface : Float64[Array, "3 3"]
        Validated direct surface frame.
    normal_hat : Float64[Array, " 3"]
        Oriented unit surface normal.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Bool[Array, "n_k n_e"]]
        Folded fractional centres and their propagation mask.

    Notes
    -----
    The lateral component of ``direct_surface[2]`` contributes to the complete
    fractional centre before wrapping onto ``[-1/2, 1/2)``.
    """
    k_parallel_norm: Float64[Array, " n_k"] = jnp.linalg.norm(
        k_parallel_cart, axis=-1
    )
    kz_complex: Complex128[Array, "n_k n_e"]
    propagating: Bool[Array, "n_k n_e"]
    kz_complex, propagating = kz_from_inner_potential(
        geometry.photon_energy_ev,
        geometry.work_function_ev,
        geometry.inner_potential_ev,
        energy_axis[None, :],
        k_parallel_norm[:, None],
    )
    safe_kz: Float64[Array, "n_k n_e"] = jnp.where(
        propagating,
        jnp.real(kz_complex),
        0.0,
    )
    center_cartesian: Float64[Array, "n_k n_e 3"] = (
        k_parallel_cart[:, None, :] + safe_kz[..., None] * normal_hat
    )
    center_unfolded: Float64[Array, "n_k n_e"] = jnp.einsum(
        "kei,i->ke", center_cartesian, direct_surface[2]
    ) / (2.0 * jnp.pi)
    center_folded: Float64[Array, "n_k n_e"] = (
        jnp.mod(center_unfolded + 0.5, 1.0) - 0.5
    )
    result: Tuple[
        Float64[Array, "n_k n_e"],
        Bool[Array, "n_k n_e"],
    ] = (center_folded, propagating)
    return result


def _exact_folded_surface_center(  # noqa: DOC502, DOC503
    k_parallel_cart: Float64[Array, "n_k 3"],
    energy_axis: Float64[Array, " n_e"],
    geometry: ExperimentGeometry,
    surface_cell: SurfaceCell,
    bulk_geometry: CrystalGeometry,
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Bool[Array, "n_k n_e"],
    Float64[Array, "n_k n_e 3"],
    Float64[Array, "n_k n_e 3"],
]:
    """PRIVATE: Compute exact finite-energy kz centres in the folded bulk BZ.

    Parameters
    ----------
    k_parallel_cart : Float64[Array, "n_k 3"]
        Physical surface-plane momenta.
    energy_axis : Float64[Array, " n_e"]
        Relative-energy samples.
    geometry : ExperimentGeometry
        Photon energy, work function, and inner potential.
    surface_cell : SurfaceCell
        Exact surface reciprocal frame.
    bulk_geometry : CrystalGeometry
        Bulk crystal geometry consumed by the reciprocal mapper.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Bool[Array, "n_k n_e"], \
Float64[Array, "n_k n_e 3"], Float64[Array, "n_k n_e 3"]]
        Folded fractional centres, propagation mask, folded surface Cartesian
        momenta, and folded bulk-fractional momenta.

    Notes
    -----
    The lateral component of ``surface_cell.stacking_vector`` contributes to
    the complete fractional centre before wrapping onto ``[-1/2, 1/2)``.
    """
    direct_surface: Float64[Array, "3 3"]
    normal_hat: Float64[Array, " 3"]
    direct_surface, _, normal_hat, _ = _effects._surface_kz_frame(
        surface_cell, bulk_geometry
    )
    center_folded: Float64[Array, "n_k n_e"]
    propagating: Bool[Array, "n_k n_e"]
    center_folded, propagating = _exact_folded_center_and_mask(
        k_parallel_cart,
        energy_axis,
        geometry,
        direct_surface,
        normal_hat,
    )
    surface_folded: Float64[Array, "n_k n_e 3"]
    bulk_folded: Float64[Array, "n_k n_e 3"]
    surface_folded, bulk_folded = _effects._map_surface_fractional_to_bulk(
        k_parallel_cart,
        center_folded,
        surface_cell,
        bulk_geometry,
    )
    result: Tuple[
        Float64[Array, "n_k n_e"],
        Bool[Array, "n_k n_e"],
        Float64[Array, "n_k n_e 3"],
        Float64[Array, "n_k n_e 3"],
    ] = (center_folded, propagating, surface_folded, bulk_folded)
    return result


def _blockwise_exact_folded_center_and_mask(  # noqa: DOC502, DOC503
    k_parallel_blocks: Float64[Array, "n_k_block k_chunk 3"],
    n_k: int,
    energy_axis: Float64[Array, " n_e"],
    geometry: ExperimentGeometry,
    direct_surface: Float64[Array, "3 3"],
    normal_hat: Float64[Array, " 3"],
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Bool[Array, "n_k n_e"],
]:
    """PRIVATE: Stream exact finite-width centres over fixed K blocks.

    Parameters
    ----------
    k_parallel_blocks : Float64[Array, "n_k_block k_chunk 3"]
        Padded physical surface-plane momenta grouped into static blocks.
    n_k : int
        Unpadded caller-owned momentum count.
    energy_axis : Float64[Array, " n_e"]
        Relative-energy samples.
    geometry : ExperimentGeometry
        Photon energy, work function, and inner potential.
    direct_surface : Float64[Array, "3 3"]
        Validated direct surface frame, hoisted outside the block map.
    normal_hat : Float64[Array, " 3"]
        Oriented unit surface normal, hoisted outside the block map.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Bool[Array, "n_k n_e"]]
        Cropped folded centres and propagation mask.

    Notes
    -----
    The block map returns only the two finite-width carriers. Full mapped
    surface and bulk point arrays remain exclusive to ``bulk_direct``.
    """

    def exact_center_block(
        k_parallel_block: Float64[Array, "k_chunk 3"],
    ) -> Tuple[
        Float64[Array, "k_chunk n_e"],
        Bool[Array, "k_chunk n_e"],
    ]:
        """Evaluate exact kinematics for one fixed-size momentum block."""
        result: Tuple[
            Float64[Array, "k_chunk n_e"],
            Bool[Array, "k_chunk n_e"],
        ] = _exact_folded_center_and_mask(
            k_parallel_block,
            energy_axis,
            geometry,
            direct_surface,
            normal_hat,
        )
        return result

    center_blocks: Float64[Array, "n_k_block k_chunk n_e"]
    propagating_blocks: Bool[Array, "n_k_block k_chunk n_e"]
    center_blocks, propagating_blocks = jax.lax.map(
        exact_center_block,
        k_parallel_blocks,
    )
    padded_k: int = k_parallel_blocks.shape[0] * k_parallel_blocks.shape[1]
    center_padded: Float64[Array, "n_k_padded n_e"] = jnp.reshape(
        center_blocks,
        (padded_k, energy_axis.shape[0]),
    )
    propagating_padded: Bool[Array, "n_k_padded n_e"] = jnp.reshape(
        propagating_blocks,
        (padded_k, energy_axis.shape[0]),
    )
    center_folded: Float64[Array, "n_k n_e"] = center_padded[:n_k]
    propagating: Bool[Array, "n_k n_e"] = propagating_padded[:n_k]
    result: Tuple[
        Float64[Array, "n_k n_e"],
        Bool[Array, "n_k n_e"],
    ] = (center_folded, propagating)
    return result


def _bulk_domain_intensity(  # noqa: DOC502, DOC503, PLR0913, PLR0915, PLR0917
    model: TBModel,
    surface_cell: SurfaceCell,
    source_kpoints: Float64[Array, "n_k 3"],
    energy_axis: Float64[Array, " n_e"],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    eta: ScalarFloat,
    kz_nodes_frac: Float64[Array, " n_kz"] | None,
    kz_mode: str,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Float64[Array, "n_k 3"],
]:
    """PRIVATE: Stream one bulk-direct or finite-width bulk-kz domain.

    Parameters
    ----------
    model : TBModel
        Bulk tight-binding model evaluated at folded fractional points.
    surface_cell : SurfaceCell
        Exact primitive surface frame.
    source_kpoints : Float64[Array, "n_k 3"]
        Caller-owned bulk-fractional source points; their surface projection
        defines the physical parallel momenta.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing relative-energy samples.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and phase coordinates.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final-state model.
    geometry : ExperimentGeometry
        Traced photon, optical, thermal, and escape-depth geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy.
    eta : ScalarFloat
        Positive resolvent regulator in eV.
    kz_nodes_frac : Float64[Array, " n_kz"] | None
        Registered finite-width node centres, or ``None`` in direct mode.
    kz_mode : str
        ``"bulk_direct"`` or ``"bulk_kz"``.
    k_chunk : int
        Static k-point chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Rematerialize node/energy scan bodies in reverse mode.

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]]
        Intrinsic intensity and physical surface-plane source points.

    Raises
    ------
    ValueError
        If the selected mode and node carrier disagree.
    EquinoxRuntimeError
        If exact finite-energy kinematics or reciprocal mapping fails.

    Notes
    -----
    ``bulk_kz`` scans nodes and keeps one ``K x E`` accumulator. It constructs
    no complete all-node band, source, kinematics, or weight carrier. The
    direct route instead scans sampled energy because its exact folded TB
    Hamiltonian changes with omega.
    """
    if kz_mode not in {"bulk_direct", "bulk_kz"}:
        raise ValueError("bulk domain mode must be 'bulk_direct' or 'bulk_kz'")
    if energy_axis.shape[0] < 2:  # noqa: PLR2004
        raise ValueError("bulk energy_axis must contain at least two samples")
    k_parallel: Float64[Array, "n_k 3"] = _bulk_source_parallel_cartesian(
        source_kpoints,
        model,
        surface_cell,
    )
    positions_surface: Float64[Array, "n_orb 3"] = (
        _bulk_orbital_positions_surface_cartesian(model, surface_cell)
    )
    n_orb: int = len(model.basis.n)
    zero_depths: Float64[Array, " n_orb"] = jnp.zeros(
        (n_orb,), dtype=jnp.float64
    )
    bulk_fermi_energy: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    if kz_mode == "bulk_direct":
        if kz_nodes_frac is not None:
            raise ValueError("bulk_direct rejects finite-width kz nodes")
        propagating: Bool[Array, "n_k n_e"]
        direct_surface_points: Float64[Array, "n_k n_e 3"]
        direct_bulk_points: Float64[Array, "n_k n_e 3"]
        (
            _,
            propagating,
            direct_surface_points,
            direct_bulk_points,
        ) = _exact_folded_surface_center(
            k_parallel,
            energy_axis,
            geometry,
            surface_cell,
            model.geometry,
        )

        def direct_energy(
            carry: None,
            arguments: Tuple[
                Float64[Array, ""],
                Bool[Array, " n_k"],
                Float64[Array, "n_k 3"],
                Float64[Array, "n_k 3"],
            ],
        ) -> Tuple[None, Float64[Array, " n_k"]]:
            """Evaluate one exact finite-energy folded bulk Hamiltonian."""
            omega: Float64[Array, ""]
            valid: Bool[Array, " n_k"]
            surface_points: Float64[Array, "n_k 3"]
            bulk_points: Float64[Array, "n_k 3"]
            omega, valid, surface_points, bulk_points = arguments
            hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = (
                bloch_hamiltonian_batch(model, bulk_points)
            )
            one_energy: Float64[Array, "n_k 1"] = _stream_cartesian_intensity(
                hamiltonians,
                surface_points,
                model.basis,
                positions_surface,
                zero_depths,
                bulk_fermi_energy,
                omega[None],
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                geometry,
                self_energy,
                eta,
                k_chunk=k_chunk,
                energy_chunk=1,
                checkpoint=checkpoint,
                use_inner_potential=True,
            )
            values: Float64[Array, " n_k"] = jnp.where(
                valid,
                one_energy[:, 0],
                0.0,
            )
            result: Tuple[None, Float64[Array, " n_k"]] = (carry, values)
            return result

        direct_step: Any = (
            jax.checkpoint(direct_energy) if checkpoint else direct_energy
        )
        energy_values: Float64[Array, "n_e n_k"]
        _, energy_values = jax.lax.scan(
            direct_step,
            None,
            (
                energy_axis,
                jnp.swapaxes(propagating, 0, 1),
                jnp.swapaxes(direct_surface_points, 0, 1),
                jnp.swapaxes(direct_bulk_points, 0, 1),
            ),
        )
        direct_intensity: Float64[Array, "n_k n_e"] = jnp.swapaxes(
            energy_values, 0, 1
        )
        direct_result: Tuple[
            Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]
        ] = (direct_intensity, k_parallel)
        return direct_result
    if kz_nodes_frac is None:
        raise ValueError("bulk_kz requires registered finite-width nodes")
    expected_nodes: Float64[Array, " n_kz"] = _effects.kz_fractional_nodes(
        kz_nodes_frac.shape[0]
    )
    checked_kz_nodes: Float64[Array, " n_kz"] = eqx.error_if(
        kz_nodes_frac,
        ~jnp.allclose(kz_nodes_frac, expected_nodes, rtol=0.0, atol=1.0e-14),
        "bulk_kz nodes must equal the registered uniform fractional centers",
    )
    direct_surface: Float64[Array, "3 3"]
    normal_hat: Float64[Array, " 3"]
    period_inv_ang: Float64[Array, ""]
    direct_surface, _, normal_hat, period_inv_ang = _effects._surface_kz_frame(
        surface_cell,
        model.geometry,
    )
    n_kz: int = checked_kz_nodes.shape[0]
    edges: Float64[Array, " n_kz_plus_one"] = jnp.linspace(
        -0.5,
        0.5,
        n_kz + 1,
        dtype=jnp.float64,
    )
    n_k: int = source_kpoints.shape[0]
    padded_k: int = _padded_extent(n_k, k_chunk)
    pad_k: int = padded_k - n_k
    padded_k_parallel: Float64[Array, "n_k_padded 3"] = jnp.pad(
        k_parallel,
        ((0, pad_k), (0, 0)),
    )
    k_parallel_blocks: Float64[Array, "n_k_block k_chunk 3"] = jnp.reshape(
        padded_k_parallel,
        (-1, k_chunk, CARTESIAN_COMPONENTS),
    )
    center_folded: Float64[Array, "n_k n_e"]
    propagating: Bool[Array, "n_k n_e"]
    center_folded, propagating = _blockwise_exact_folded_center_and_mask(
        k_parallel_blocks,
        n_k,
        energy_axis,
        geometry,
        direct_surface,
        normal_hat,
    )

    def integrate_node(
        accumulated: Float64[Array, "n_k n_e"],
        arguments: Tuple[
            Float64[Array, ""],
            Float64[Array, ""],
            Float64[Array, ""],
        ],
    ) -> Tuple[
        Float64[Array, "n_k n_e"],
        None,
    ]:
        """Evaluate and accumulate one finite-width bulk node."""
        node: Float64[Array, ""]
        lower_edge: Float64[Array, ""]
        upper_edge: Float64[Array, ""]
        node, lower_edge, upper_edge = arguments

        def stream_k_block(
            k_parallel_block: Float64[Array, "k_chunk 3"],
        ) -> Float64[Array, "k_chunk n_e"]:
            """Stream one fixed-size k block at the current bulk node."""
            folded_block_nodes: Float64[Array, " k_chunk"] = jnp.broadcast_to(
                node,
                (k_chunk,),
            )
            surface_block: Float64[Array, "k_chunk 3"]
            bulk_block: Float64[Array, "k_chunk 3"]
            surface_block, bulk_block = (
                _effects._map_surface_fractional_to_bulk(
                    k_parallel_block,
                    folded_block_nodes,
                    surface_cell,
                    model.geometry,
                )
            )
            block_hamiltonians: Complex128[Array, "k_chunk n_orb n_orb"] = (
                bloch_hamiltonian_batch(model, bulk_block)
            )
            block_intensity: Float64[Array, "k_chunk n_e"] = (
                _stream_cartesian_intensity(
                    block_hamiltonians,
                    surface_block,
                    model.basis,
                    positions_surface,
                    zero_depths,
                    bulk_fermi_energy,
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
                    use_inner_potential=True,
                )
            )
            return block_intensity

        block_intensities: Float64[Array, "n_k_block k_chunk n_e"] = (
            jax.lax.map(stream_k_block, k_parallel_blocks)
        )
        padded_node_intensity: Float64[Array, "n_k_padded n_e"] = jnp.reshape(
            block_intensities,
            (padded_k, energy_axis.shape[0]),
        )
        node_intensity: Float64[Array, "n_k n_e"] = padded_node_intensity[:n_k]
        weight: Float64[Array, "n_k n_e"] = (
            _effects._kz_wrapped_lorentzian_bin_weight(
                lower_edge,
                upper_edge,
                center_folded,
                geometry.mean_free_path_ang,
                period_inv_ang,
            )
        )
        contribution: Float64[Array, "n_k n_e"] = jnp.where(
            propagating,
            node_intensity * weight,
            0.0,
        )
        next_accumulated: Float64[Array, "n_k n_e"] = (
            accumulated + contribution
        )
        result: Tuple[Float64[Array, "n_k n_e"], None] = (
            next_accumulated,
            None,
        )
        return result

    node_step: Any = (
        jax.checkpoint(integrate_node) if checkpoint else integrate_node
    )
    initial_intensity: Float64[Array, "n_k n_e"] = jnp.zeros(
        (source_kpoints.shape[0], energy_axis.shape[0]), dtype=jnp.float64
    )
    integrated: Float64[Array, "n_k n_e"]
    integrated, _ = jax.lax.scan(
        node_step,
        initial_intensity,
        (
            checked_kz_nodes,
            edges[:-1],
            edges[1:],
        ),
    )
    bulk_result: Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]] = (
        integrated,
        k_parallel,
    )
    return bulk_result


def _separable_grid_axes(  # noqa: DOC503 -- traced guards raise indirectly.
    kpoints_cart: Float64[Array, "n_k 3"],
    mesh_shape: Tuple[int, int],
    expected_kz: Float64[Array, ""],
) -> Tuple[Float64[Array, " n_kx"], Float64[Array, " n_ky"]]:
    """PRIVATE: Extract and validate sample-Cartesian raster axes.

    Parameters
    ----------
    kpoints_cart : Float64[Array, "n_k 3"]
        Flattened Cartesian points in row-major ``(ky, kx)`` order.
    mesh_shape : Tuple[int, int]
        Static ``(n_ky, n_kx)`` raster shape.
    expected_kz : Float64[Array, ""]
        Explicit fixed Cartesian out-of-plane source momentum.

    Returns
    -------
    axes : Tuple[Float64[Array, " n_kx"], Float64[Array, " n_ky"]]
        Strict sample-Cartesian source axes.

    Raises
    ------
    ValueError
        If either source interpolation axis has fewer than two points.
    EquinoxRuntimeError
        If the flattened grid is nonseparable, varies in kz, or has a
        non-increasing Cartesian axis.
    """
    n_ky: int
    n_kx: int
    n_ky, n_kx = mesh_shape
    minimum_points: int = 2
    if n_kx < minimum_points or n_ky < minimum_points:
        raise ValueError(
            "simulate_arpes requires at least two kx and two ky source points"
        )
    cartesian_grid: Float64[Array, "n_ky n_kx 3"] = jnp.reshape(
        kpoints_cart, (n_ky, n_kx, CARTESIAN_COMPONENTS)
    )
    kx_axis: Float64[Array, " n_kx"] = cartesian_grid[0, :, 0]
    ky_axis: Float64[Array, " n_ky"] = cartesian_grid[:, 0, 1]
    expected_kx: Float64[Array, "n_ky n_kx"] = jnp.broadcast_to(
        kx_axis[None, :], (n_ky, n_kx)
    )
    expected_ky: Float64[Array, "n_ky n_kx"] = jnp.broadcast_to(
        ky_axis[:, None], (n_ky, n_kx)
    )
    reference_kz: Float64[Array, ""] = cartesian_grid[0, 0, 2]
    separable: Bool[Array, ""] = (
        jnp.allclose(
            cartesian_grid[:, :, 0],
            expected_kx,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        & jnp.allclose(
            cartesian_grid[:, :, 1],
            expected_ky,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        & jnp.allclose(
            cartesian_grid[:, :, 2],
            reference_kz,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        & jnp.isclose(
            reference_kz,
            expected_kz,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        & jnp.all(jnp.diff(kx_axis) > 0.0)
        & jnp.all(jnp.diff(ky_axis) > 0.0)
    )
    checked_kx: Float64[Array, " n_kx"] = eqx.error_if(
        kx_axis,
        ~separable,
        "KGrid must be a strictly increasing separable "
        "sample-Cartesian raster",
    )
    axes: Tuple[Float64[Array, " n_kx"], Float64[Array, " n_ky"]] = (
        checked_kx,
        ky_axis,
    )
    return axes


def _physical_cubes(  # noqa: DOC105, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kgrid: KGrid,
    energy_axis: Float64[Array, " n_e"],
    eta: ScalarFloat,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> Tuple[ArpesCube, ...]:
    """PRIVATE: Materialize every domain as an explicit physical cube.

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy Hamiltonians.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and phase coordinates.
    radial_quadrature : RadialQuadratureSpec
        Certified radial quadrature.
    final_state : FinalStateSpec
        Final-state model.
    geometry : ExperimentGeometry
        Experiment geometry.
    self_energy : SelfEnergyModel
        Causal self-energy model.
    kgrid : KGrid
        Declared separable source raster.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned relative-energy axis.
    eta : ScalarFloat
        Positive resolvent regulator.
    k_chunk : int
        Static k chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Reverse-mode rematerialization selector.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Bulk models for the two bulk modes. Default is ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Exact surface frame per bulk or coherent domain. Default is ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Registered fractional nodes for ``bulk_kz``. Default is ``None``.
    kz_mode : str, optional
        Registered mutually exclusive driver mode. Default is
        ``"native_direct"``.

    Returns
    -------
    cubes : Tuple[ArpesCube, ...]
        Self-describing physical source cubes.

    Raises
    ------
    ValueError
        If the grid is not an explicit single-kz raster.
    """
    if kgrid.kz is None or kgrid.photon_energy_axis_ev is not None:
        raise ValueError(
            "simulate_arpes requires an explicit fixed-kz grid without a "
            "photon-energy axis"
        )
    cubes: list[ArpesCube] = []
    reference_axes: (
        Tuple[Float64[Array, " n_kx"], Float64[Array, " n_ky"]] | None
    ) = None
    bulk_mode: bool = kz_mode in {"bulk_direct", "bulk_kz"}
    n_domains: int = (
        len(bulk_models_by_domain)
        if bulk_mode and bulk_models_by_domain is not None
        else len(bands_by_domain)
    )
    domain_index: int
    for domain_index in range(n_domains):
        intensity_flat: Float64[Array, "n_k n_e"]
        kpoints_cart: Float64[Array, "n_k 3"]
        if bulk_mode:
            if (
                bulk_models_by_domain is None
                or surface_cells_by_domain is None
            ):
                raise ValueError(
                    "bulk cube mode requires model and surface tuples"
                )
            intensity_flat, kpoints_cart = _bulk_domain_intensity(
                bulk_models_by_domain[domain_index],
                surface_cells_by_domain[domain_index],
                kgrid.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                geometry,
                self_energy,
                eta,
                kz_nodes_frac,
                kz_mode,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
            )
        else:
            bands: DiagonalizedBands = bands_by_domain[domain_index]
            hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = (
                hamiltonians_by_domain[domain_index]
            )
            intensity_flat, kpoints_cart = _stream_domain_intensity(
                hamiltonians,
                bands,
                kgrid.kpoints,
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
                use_inner_potential=kz_mode == "coherent_slab",
                surface_cell=(
                    surface_cells_by_domain[domain_index]
                    if kz_mode == "coherent_slab"
                    and surface_cells_by_domain is not None
                    else None
                ),
            )
        expected_source_kz: Float64[Array, ""] = (
            jnp.asarray(0.0, dtype=jnp.float64) if bulk_mode else kgrid.kz
        )
        kx_axis: Float64[Array, " n_kx"]
        ky_axis: Float64[Array, " n_ky"]
        kx_axis, ky_axis = _separable_grid_axes(
            kpoints_cart,
            kgrid.mesh_shape,
            expected_source_kz,
        )
        if reference_axes is not None:
            kx_axis = eqx.error_if(
                kx_axis,
                ~jnp.allclose(
                    kx_axis,
                    reference_axes[0],
                    rtol=1.0e-12,
                    atol=1.0e-13,
                )
                | ~jnp.allclose(
                    ky_axis,
                    reference_axes[1],
                    rtol=1.0e-12,
                    atol=1.0e-13,
                ),
                "all domains must share one source Cartesian raster",
            )
        else:
            reference_axes = (kx_axis, ky_axis)
        n_ky: int
        n_kx: int
        n_ky, n_kx = kgrid.mesh_shape
        intensity_cube: Float64[Array, "n_kx n_ky n_e"] = jnp.transpose(
            jnp.reshape(
                intensity_flat,
                (n_ky, n_kx, energy_axis.shape[0]),
            ),
            (1, 0, 2),
        )
        cube: ArpesCube = make_arpes_cube(
            intensity_cube,
            kx_axis,
            ky_axis,
            energy_axis,
            cartesian_frame_id="org.diffpes.frame.sample_cartesian",
            provenance=(
                f"simulate_arpes/domain={domain_index}/single-kz"
                if kz_mode == "native_direct"
                else f"simulate_arpes/domain={domain_index}/{kz_mode}"
            ),
        )
        cubes.append(cube)
    result: Tuple[ArpesCube, ...] = tuple(cubes)
    return result


def _physical_spectra(  # noqa: DOC105, DOC503, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kpath: KPath,
    energy_axis: Float64[Array, " n_e"],
    eta: ScalarFloat,
    *,
    k_chunk: int,
    energy_chunk: int,
    checkpoint: bool,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> Tuple[ArpesSpectrum, ...]:
    """PRIVATE: Materialize every domain as a self-describing path cut.

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy Hamiltonians.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and phase coordinates.
    radial_quadrature : RadialQuadratureSpec
        Certified radial quadrature.
    final_state : FinalStateSpec
        Final-state model.
    geometry : ExperimentGeometry
        Experiment geometry.
    self_energy : SelfEnergyModel
        Causal self-energy model.
    kpath : KPath
        Declared fractional source path.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned relative-energy axis.
    eta : ScalarFloat
        Positive resolvent regulator.
    k_chunk : int
        Static k chunk size.
    energy_chunk : int
        Static energy chunk size.
    checkpoint : bool
        Reverse-mode rematerialization selector.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Bulk models for direct or finite-width integration. Default is
        ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Exact surface frames for bulk or coherent routes. Default is ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Registered fractional nodes in ``bulk_kz``. Default is ``None``.
    kz_mode : str, optional
        Registered mutually exclusive driver mode. Default is
        ``"native_direct"``.

    Returns
    -------
    spectra : Tuple[ArpesSpectrum, ...]
        Self-describing physical source path spectra.

    Raises
    ------
    ValueError
        If the path has fewer than two source nodes or no explicit fixed kz.
    EquinoxRuntimeError
        If its Cartesian points disagree with the explicit fixed kz.
    """
    minimum_points: int = 2
    if kpath.kpoints.shape[0] < minimum_points:
        raise ValueError(
            "simulate_arpes_cut requires at least two path points"
        )
    if kpath.kz is None:
        raise ValueError("simulate_arpes_cut requires an explicit fixed kz")
    spectra: list[ArpesSpectrum] = []
    reference_points: Float64[Array, "n_k 3"] | None = None
    bulk_mode: bool = kz_mode in {"bulk_direct", "bulk_kz"}
    n_domains: int = (
        len(bulk_models_by_domain)
        if bulk_mode and bulk_models_by_domain is not None
        else len(bands_by_domain)
    )
    domain_index: int
    for domain_index in range(n_domains):
        intensity: Float64[Array, "n_k n_e"]
        kpoints_cart: Float64[Array, "n_k 3"]
        if bulk_mode:
            if (
                bulk_models_by_domain is None
                or surface_cells_by_domain is None
            ):
                raise ValueError(
                    "bulk cut mode requires model and surface tuples"
                )
            intensity, kpoints_cart = _bulk_domain_intensity(
                bulk_models_by_domain[domain_index],
                surface_cells_by_domain[domain_index],
                kpath.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                geometry,
                self_energy,
                eta,
                kz_nodes_frac,
                kz_mode,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
            )
        else:
            bands: DiagonalizedBands = bands_by_domain[domain_index]
            hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = (
                hamiltonians_by_domain[domain_index]
            )
            intensity, kpoints_cart = _stream_domain_intensity(
                hamiltonians,
                bands,
                kpath.kpoints,
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
                use_inner_potential=kz_mode == "coherent_slab",
                surface_cell=(
                    surface_cells_by_domain[domain_index]
                    if kz_mode == "coherent_slab"
                    and surface_cells_by_domain is not None
                    else None
                ),
            )
        expected_source_kz: Float64[Array, ""] = (
            jnp.asarray(0.0, dtype=jnp.float64) if bulk_mode else kpath.kz
        )
        kpoints_cart = eqx.error_if(
            kpoints_cart,
            ~jnp.allclose(
                kpoints_cart[:, 2],
                expected_source_kz,
                rtol=1.0e-12,
                atol=1.0e-13,
            ),
            "KPath Cartesian points must match its explicit fixed kz",
        )
        if reference_points is not None:
            kpoints_cart = eqx.error_if(
                kpoints_cart,
                ~jnp.allclose(
                    kpoints_cart,
                    reference_points,
                    rtol=1.0e-12,
                    atol=1.0e-13,
                ),
                "all domains must share one source Cartesian path",
            )
        else:
            reference_points = kpoints_cart
        step_lengths: Float64[Array, " n_step"] = jnp.linalg.norm(
            jnp.diff(kpoints_cart, axis=0), axis=-1
        )
        k_axis: Float64[Array, " n_k"] = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.float64),
                jnp.cumsum(step_lengths),
            )
        )
        spectra.append(
            make_arpes_spectrum(
                intensity,
                energy_axis,
                k_axis,
                kpoints_cart,
                cartesian_frame_id="org.diffpes.frame.sample_cartesian",
            )
        )
    result: Tuple[ArpesSpectrum, ...] = tuple(spectra)
    return result


@jaxtyped(typechecker=beartype)
def simulate_arpes(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kgrid: KGrid,
    energy_axis: Float64[Array, " n_e"],
    detector_calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> DetectorRaster:
    """Simulate the canonical detector raster.

    The driver constructs one physical source cube per static domain through
    the degeneracy-safe resolvent.  It then invokes the single shared detector
    chain.  No normalization, random sampling, fidelity construction, or
    approximation-tier dispatch occurs here.

    :see: :class:`~.test_spectrum.TestSimulateArpes`

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy orbital Hamiltonians in eV, one per domain.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers on exactly ``kgrid.kpoints``.
    radial_spec : RadialSpec
        Shell-shared radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit plane-wave or Coulomb radial final state.
    geometry : ExperimentGeometry
        Traced single-acquisition geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy model.
    kgrid : KGrid
        Fixed-kz separable sample-Cartesian source raster.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned ``E - E_F`` samples in eV.
    detector_calibration : DetectorCalibration
        Explicit target bins, PSF widths, and transmission domain.
    detector_effects : DetectorEffects
        Complete deterministic detector and nuisance parameters.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize live chunks in reverse mode. Default is ``True``.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Per-domain bulk models for ``bulk_direct`` and ``bulk_kz``. Default is
        ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Per-domain exact surface frames for bulk/coherent modes. Default is
        ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The library G6
        profile certifies 2048 nodes. Every other explicit uniform count is a
        caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        ``"native_direct"``, ``"bulk_direct"``, ``"bulk_kz"``, or
        ``"coherent_slab"``. Default is ``"native_direct"``.

    Returns
    -------
    raster : DetectorRaster
        Native-axis expected detector counts.

    Raises
    ------
    ValueError
        If domain/static shapes disagree or the source grid lacks separable
        sample-Cartesian axes.
    EquinoxRuntimeError
        If a traced physical, kinematic, spectral, or detector contract fails.

    Notes
    -----
    ``DiagonalizedBands`` is metadata at this seam.  The explicit Hamiltonian
    owns resolvent values and derivatives; it is never reconstructed from the
    carrier's eigensystem.
    """
    _validate_kz_mode_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        bulk_models_by_domain,
        surface_cells_by_domain,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    physical_by_domain: Tuple[ArpesCube, ...] = _physical_cubes(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        geometry,
        self_energy,
        kgrid,
        energy_axis,
        eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        kz_nodes_frac=kz_nodes_frac,
        kz_mode=kz_mode,
    )
    raster: DetectorRaster = _effects.apply_detector_effects(
        physical_by_domain,
        geometry,
        detector_calibration,
        detector_effects,
    )
    return raster


@jaxtyped(typechecker=beartype)
def simulate_arpes_cut(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kpath: KPath,
    energy_axis: Float64[Array, " n_e"],
    detector_calibration: DetectorCalibration,
    detector_effects: DetectorEffects,
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_models_by_domain: Tuple[TBModel, ...] | None = None,
    surface_cells_by_domain: Tuple[SurfaceCell, ...] | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> DetectorRaster:
    """Simulate the canonical path-cut detector raster.

    Every domain becomes an ``ArpesSpectrum`` carrying cumulative distance,
    the complete sample-Cartesian path, and its registered frame identity.
    The slit spans one bin in native detector ``v`` coordinates.  The shared
    detector chain applies all resolution after mapping.

    :see: :class:`~.test_spectrum.TestSimulateArpesCut`

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
        Explicit absolute-energy orbital Hamiltonians in eV, one per domain.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Domain metadata carriers on exactly ``kpath.kpoints``.
    radial_spec : RadialSpec
        Shell-shared radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit plane-wave or Coulomb radial final state.
    geometry : ExperimentGeometry
        Traced single-acquisition geometry.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy model.
    kpath : KPath
        Fractional source path retaining complete Cartesian vectors.
    energy_axis : Float64[Array, " n_e"]
        Caller-owned ``E - E_F`` samples in eV.
    detector_calibration : DetectorCalibration
        Explicit slit target bins, PSF widths, and transmission domain.
    detector_effects : DetectorEffects
        Complete deterministic detector and nuisance parameters.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Positive static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Positive static energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize live chunks in reverse mode. Default is ``True``.
    bulk_models_by_domain : Tuple[TBModel, ...] | None, optional
        Per-domain bulk models for ``bulk_direct`` and ``bulk_kz``. Default is
        ``None``.
    surface_cells_by_domain : Tuple[SurfaceCell, ...] | None, optional
        Per-domain exact surface frames for bulk/coherent modes. Default is
        ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The library G6
        profile certifies 2048 nodes. Every other explicit uniform count is a
        caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        ``"native_direct"``, ``"bulk_direct"``, ``"bulk_kz"``, or
        ``"coherent_slab"``. Default is ``"native_direct"``.

    Returns
    -------
    raster : DetectorRaster
        Native slit-axis expected detector counts.

    Raises
    ------
    ValueError
        If domain/static shapes disagree or the path has fewer than two nodes.
    EquinoxRuntimeError
        If a traced physical, kinematic, spectral, or detector contract fails.

    Notes
    -----
    ``DiagonalizedBands`` remains metadata.  The explicit Hamiltonian owns
    resolvent values and derivatives through the complete cut path.
    """
    _validate_kz_mode_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        bulk_models_by_domain,
        surface_cells_by_domain,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    physical_by_domain: Tuple[ArpesSpectrum, ...] = _physical_spectra(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        geometry,
        self_energy,
        kpath,
        energy_axis,
        eta,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        kz_nodes_frac=kz_nodes_frac,
        kz_mode=kz_mode,
    )
    raster: DetectorRaster = _effects.apply_detector_effects(
        physical_by_domain,
        geometry,
        detector_calibration,
        detector_effects,
    )
    return raster


@jaxtyped(typechecker=beartype)
def simulate_hv_scan(  # noqa: DOC105, DOC502, DOC503, PLR0913, PLR0917
    hamiltonian: Complex128[Array, "n_k n_orb n_orb"] | None,
    bands: DiagonalizedBands | None,
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    geometry: ExperimentGeometry,
    self_energy: SelfEnergyModel,
    kpath: KPath,
    energy_axis: Float64[Array, " n_e"],
    photon_energies_ev: Float64[Array, " n_hv"],
    eta: ScalarFloat = 1.0e-4,
    *,
    k_chunk: int = 32,
    energy_chunk: int = 32,
    checkpoint: bool = True,
    bulk_model: TBModel | None = None,
    surface_cell: SurfaceCell | None = None,
    kz_nodes_frac: Float64[Array, " n_kz"] | None = None,
    kz_mode: str = "native_direct",
) -> Float64[Array, "n_hv n_k n_e"]:
    """Simulate a single-domain pre-detector photon-energy scan.

    The scan keeps the photon-energy axis explicit and re-evaluates exact
    finite-energy kinematics and matrix elements at every row. It carries no
    detector response, transmission, sampling, or display normalization.

    :see: :class:`~.test_spectrum.TestSimulateHvScan`

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_k n_orb n_orb"] | None
        Explicit Hamiltonian for native/coherent modes; ``None`` in bulk
        modes.
    bands : DiagonalizedBands | None
        Metadata paired with explicit H; ``None`` in bulk modes.
    radial_spec : RadialSpec
        Shell-shared radial parameters.
    matrix_element_params : MatrixElementParams
        Shell scales and physical channel phases.
    radial_quadrature : RadialQuadratureSpec
        Certified fixed radial quadrature.
    final_state : FinalStateSpec
        Explicit radial final-state model.
    geometry : ExperimentGeometry
        Base experiment geometry; each row replaces only photon energy.
    self_energy : SelfEnergyModel
        Causal intrinsic self-energy.
    kpath : KPath
        Fixed-shape source path.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing relative-energy samples.
    photon_energies_ev : Float64[Array, " n_hv"]
        Positive finite photon energies.
    eta : ScalarFloat, optional
        Positive resolvent regulator in eV. Default is ``1e-4``.
    k_chunk : int, optional
        Static k-point chunk size. Default is 32.
    energy_chunk : int, optional
        Static sampled-energy chunk size. Default is 32.
    checkpoint : bool, optional
        Rematerialize scan bodies in reverse mode. Default is ``True``.
    bulk_model : TBModel | None, optional
        Single bulk model for either bulk mode. Default is ``None``.
    surface_cell : SurfaceCell | None, optional
        Exact surface frame for bulk/coherent mode. Default is ``None``.
    kz_nodes_frac : Float64[Array, " n_kz"] | None, optional
        Explicit registered midpoint nodes for ``bulk_kz``. The library G6
        profile certifies 2048 nodes. Every other explicit uniform count is a
        caller-owned recalibration or reduced diagnostic and carries no
        library accuracy claim. Default is ``None``.
    kz_mode : str, optional
        Registered mutually exclusive mode. Default is ``"native_direct"``.

    Returns
    -------
    scan : Float64[Array, "n_hv n_k n_e"]
        Intrinsic single-domain intensity for every photon energy.

    Raises
    ------
    ValueError
        If the mode/carrier surface is invalid or an axis is empty.
    EquinoxRuntimeError
        If traced photon energies, kinematics, or physics leave their domain.

    Notes
    -----
    A :func:`jax.lax.scan` owns the photon-energy loop. Node count and all
    chunk choices remain static; photon-energy values remain differentiable.
    """
    if photon_energies_ev.ndim != 1 or photon_energies_ev.shape[0] < 1:
        raise ValueError("photon_energies_ev must be a nonempty vector")
    if kpath.kz is None or kpath.kpoints.shape[0] < 2:  # noqa: PLR2004
        raise ValueError(
            "simulate_hv_scan requires a fixed-kz path with two points"
        )
    checked_photon_energies: Float64[Array, " n_hv"] = eqx.error_if(
        photon_energies_ev,
        ~jnp.all(jnp.isfinite(photon_energies_ev))
        | jnp.any(photon_energies_ev <= 0.0),
        "photon energies must be finite and positive",
    )
    hamiltonian_tuple: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...] = (
        () if hamiltonian is None else (hamiltonian,)
    )
    bands_tuple: Tuple[DiagonalizedBands, ...] = (
        () if bands is None else (bands,)
    )
    bulk_tuple: Tuple[TBModel, ...] | None = (
        None if bulk_model is None else (bulk_model,)
    )
    surface_tuple: Tuple[SurfaceCell, ...] | None = (
        None if surface_cell is None else (surface_cell,)
    )
    _validate_kz_mode_inputs(
        hamiltonian_tuple,
        bands_tuple,
        radial_spec,
        matrix_element_params,
        bulk_tuple,
        surface_tuple,
        kz_nodes_frac,
        kz_mode,
        k_chunk=k_chunk,
        energy_chunk=energy_chunk,
        checkpoint=checkpoint,
    )
    bulk_mode: bool = kz_mode in {"bulk_direct", "bulk_kz"}

    def one_photon_energy(
        carry: None,
        photon_energy: Float64[Array, ""],
    ) -> Tuple[None, Float64[Array, "n_k n_e"]]:
        """Evaluate one physical scan row with an updated geometry leaf."""
        row_geometry: ExperimentGeometry = eqx.tree_at(
            lambda item: item.photon_energy_ev,
            geometry,
            photon_energy,
        )
        row_intensity: Float64[Array, "n_k n_e"]
        if bulk_mode:
            if bulk_model is None or surface_cell is None:
                raise ValueError("bulk scan requires a model and surface cell")
            row_intensity, _ = _bulk_domain_intensity(
                bulk_model,
                surface_cell,
                kpath.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                row_geometry,
                self_energy,
                eta,
                kz_nodes_frac,
                kz_mode,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
            )
        else:
            if hamiltonian is None or bands is None:
                raise ValueError("native/coherent scan requires H and bands")
            row_intensity, _ = _stream_domain_intensity(
                hamiltonian,
                bands,
                kpath.kpoints,
                energy_axis,
                radial_spec,
                matrix_element_params,
                radial_quadrature,
                final_state,
                row_geometry,
                self_energy,
                eta,
                k_chunk=k_chunk,
                energy_chunk=energy_chunk,
                checkpoint=checkpoint,
                use_inner_potential=kz_mode == "coherent_slab",
                surface_cell=(
                    surface_cell if kz_mode == "coherent_slab" else None
                ),
            )
        result: Tuple[None, Float64[Array, "n_k n_e"]] = (
            carry,
            row_intensity,
        )
        return result

    scan_step: Any = (
        jax.checkpoint(one_photon_energy) if checkpoint else one_photon_energy
    )
    scan: Float64[Array, "n_hv n_k n_e"]
    _, scan = jax.lax.scan(scan_step, None, checked_photon_energies)
    return scan


@jaxtyped(typechecker=beartype)
def hv_map_at_energy(  # noqa: DOC503
    scan: Float64[Array, "n_hv n_k n_e"],
    energy_axis: Float64[Array, " n_e"],
    energy_ev: ScalarFloat,
) -> Float64[Array, "n_k n_hv"]:
    """Interpolate a photon-energy scan at one sampled binding energy.

    The helper applies piecewise-linear interpolation on the caller-owned
    sampled-energy axis. It then returns momentum as the leading plotting axis.

    :see: :class:`~.test_spectrum.TestHvMapAtEnergy`

    Parameters
    ----------
    scan : Float64[Array, "n_hv n_k n_e"]
        Single-domain pre-detector photon-energy scan.
    energy_axis : Float64[Array, " n_e"]
        Strictly increasing sampled relative-energy axis.
    energy_ev : ScalarFloat
        Requested in-domain relative energy in eV.

    Returns
    -------
    hv_map : Float64[Array, "n_k n_hv"]
        Linearly interpolated path-by-photon-energy map.

    Raises
    ------
    ValueError
        If array axes disagree or the energy axis contains fewer than two
        nodes.
    EquinoxRuntimeError
        If the axis/query is non-finite, non-increasing, or out of domain.

    Notes
    -----
    The output orientation puts path momentum first for direct plotting.
    Query derivatives are piecewise linear away from sampled knots.
    """
    minimum_points: int = 2
    if (
        scan.ndim != 3  # noqa: PLR2004
        or energy_axis.ndim != 1
        or energy_axis.shape[0] < minimum_points
        or scan.shape[-1] != energy_axis.shape[0]
    ):
        raise ValueError(
            "scan and energy axis must have compatible sampled axes"
        )
    query: Float64[Array, ""] = jnp.asarray(energy_ev, dtype=jnp.float64)
    checked_axis: Float64[Array, " n_e"] = eqx.error_if(
        energy_axis,
        ~jnp.all(jnp.isfinite(energy_axis))
        | jnp.any(jnp.diff(energy_axis) <= 0.0)
        | ~jnp.isfinite(query)
        | (query < energy_axis[0])
        | (query > energy_axis[-1]),
        "energy axis must increase and the query must lie in its domain",
    )
    upper: Array = jnp.clip(
        jnp.searchsorted(checked_axis, query, side="right"),
        1,
        checked_axis.shape[0] - 1,
    )
    lower: Array = upper - 1
    fraction: Float64[Array, ""] = (query - checked_axis[lower]) / (
        checked_axis[upper] - checked_axis[lower]
    )
    values: Float64[Array, "n_hv n_k"] = (1.0 - fraction) * scan[
        :, :, lower
    ] + fraction * scan[:, :, upper]
    hv_map: Float64[Array, "n_k n_hv"] = jnp.swapaxes(values, 0, 1)
    return hv_map


@jaxtyped(typechecker=beartype)
def normalize_intensity(  # noqa: DOC105, DOC503
    carrier: Union[ArpesCube, ArpesSpectrum, DetectorRaster],
    mode: str = "none",
) -> Float64[Array, " ..."]:
    """Return an explicit display-only normalization of carrier values.

    The function returns a plain array rather than relabeling normalized or
    z-scored display values as physical intensity or expected counts.  Neither
    canonical driver calls this helper.

    :see: :class:`~.test_spectrum.TestNormalizeIntensity`

    Parameters
    ----------
    carrier : ArpesCube | ArpesSpectrum | DetectorRaster
        Physical source intensity or native expected counts.
    mode : str, optional
        ``"none"``, ``"sum"``, or ``"zscore"``. Default is ``"none"``.

    Returns
    -------
    normalized : Float64[Array, " ..."]
        Plain display array with the carrier's original shape.

    Raises
    ------
    ValueError
        If ``mode`` has an unsupported value.
    EquinoxRuntimeError
        If a requested sum or standard deviation is zero.

    Notes
    -----
    ``"sum"`` divides by the complete-array sum.  ``"zscore"`` subtracts the
    complete-array mean and divides by the population standard deviation.
    These crop-dependent transforms are unsuitable for a physical likelihood.
    """
    if mode not in {"none", "sum", "zscore"}:
        raise ValueError("mode must be 'none', 'sum', or 'zscore'")
    values: Float64[Array, " ..."] = (
        carrier.expected_counts
        if isinstance(carrier, DetectorRaster)
        else carrier.intensity
    )
    if mode == "none":
        return values
    if mode == "sum":
        total: Float64[Array, ""] = jnp.sum(values)
        checked_total: Float64[Array, ""] = eqx.error_if(
            total,
            total == 0.0,
            "sum normalization requires nonzero total intensity",
        )
        normalized_sum: Float64[Array, " ..."] = values / checked_total
        return normalized_sum
    mean: Float64[Array, ""] = jnp.mean(values)
    standard_deviation: Float64[Array, ""] = jnp.std(values)
    checked_deviation: Float64[Array, ""] = eqx.error_if(
        standard_deviation,
        standard_deviation == 0.0,
        "zscore normalization requires nonzero standard deviation",
    )
    normalized_zscore: Float64[Array, " ..."] = (
        values - mean
    ) / checked_deviation
    return normalized_zscore


__all__: list[str] = [
    "hv_map_at_energy",
    "normalize_intensity",
    "simulate_arpes",
    "simulate_arpes_cut",
    "simulate_hv_scan",
]
