r"""Compose the coherent single-:math:`k_z` ARPES forward driver.

Extended Summary
----------------
This module composes the intrinsic Plan-06/07 observable with the Plan-08a
detector chain.  It builds matrix-element source kets only inside each live
energy/k-point chunk.  It solves the explicit orbital Hamiltonian through the
degeneracy-safe resolvent.  It multiplies by sampled Fermi occupation and
materializes a self-describing source carrier.  The common detector operator
then handles domain mapping, mixing, transmission, resolution, backgrounds,
sensitivity, and expected counts.

The Hamiltonian is an explicit input.  ``DiagonalizedBands`` supplies orbital,
crystal, Fermi-level, and source-coordinate metadata, but this module never
reconstructs a Hamiltonian from its eigenvectors.  Such a value-only
reconstruction silently replaces the native Hamiltonian derivative with an
eigensystem derivative.  The complete Plan-06 parameter
surface is likewise explicit: ``RadialSpec``, ``MatrixElementParams``,
``RadialQuadratureSpec``, and ``FinalStateSpec`` all cross the driver boundary.

The deterministic chain is

``orbital source -> A(k,w) f_FD -> source carrier -> detector effects``.

There is one coherent route and no fidelity or string tier dispatcher.  The
caller owns the sampled energy axis.  Display normalization is an explicit
helper and is never called by either physical driver.

Routine Listings
----------------
:func:`normalize_intensity`
    Return an explicit display-only normalization of carrier values.
:func:`simulate_arpes`
    Simulate the canonical coherent single-kz detector raster.
:func:`simulate_arpes_cut`
    Simulate the canonical coherent single-kz path-cut detector raster.

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
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from diffpes.types import (
    CARTESIAN_COMPONENTS,
    ArpesCube,
    ArpesSpectrum,
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
    make_arpes_cube,
    make_arpes_spectrum,
)

from . import effects as _effects
from . import spectral as _spectral
from .kinematics import final_state_k_inv_ang, kinetic_energy_ev
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
    hamiltonians_by_domain : Tuple[Complex128[Array, "K O O"], ...]
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
) -> Tuple[
    Float64[Array, "n_k n_e"],
    Float64[Array, "n_k 3"],
]:
    """PRIVATE: Assemble one domain through the padded resolvent scan.

    Parameters
    ----------
    hamiltonians_ev : Complex128[Array, "n_k n_orb n_orb"]
        Explicit absolute-energy orbital Hamiltonians.
    bands : DiagonalizedBands
        Domain metadata and Fermi energy.
    source_kpoints : Float64[Array, "n_k 3"]
        Declared fractional source grid.
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

    Returns
    -------
    result : Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]]
        Intrinsic intensity and complete Cartesian source points.

    Raises
    ------
    ValueError
        If the caller-owned energy axis has fewer than two samples.

    Notes
    -----
    Padding values stay finite and inside the sampled self-energy interval;
    masks remove them exactly from the physical result.
    """
    n_k: int = source_kpoints.shape[0]
    n_energy: int = energy_axis.shape[0]
    minimum_points: int = 2
    if n_energy < minimum_points:
        raise ValueError("energy_axis must contain at least two samples")
    checked_energy_axis: Float64[Array, " n_e"] = eqx.error_if(
        energy_axis,
        ~jnp.all(jnp.isfinite(energy_axis))
        | ~jnp.all(jnp.diff(energy_axis) > 0.0),
        "energy_axis must be finite and strictly increasing",
    )
    k_cart: Float64[Array, "n_k 3"] = _checked_source_axes(
        bands, source_kpoints
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
        k_cart, ((0, pad_k), (0, 0))
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
    n_orb: int = len(bands.basis.n)
    depths: Float64[Array, " n_orb"] = (
        jnp.zeros((n_orb,), dtype=jnp.float64)
        if bands.depths is None
        else bands.depths
    )
    schedule: _spectral._TransitionSourceSchedule = (
        _spectral._TransitionSourceSchedule(
            k_i_cart=padded_k_cart,
            final_norm=padded_final_norm,
            emission_energy_valid=padded_emission_energy_valid,
            positions_cart=resolve_orbital_positions_cart(bands),
            depths=depths,
            polarization_sample_cart=polarization_sample,
            mean_free_path_ang=geometry.mean_free_path_ang,
            radial=radial_spec,
            matrix_element=matrix_element_params,
            quadrature=radial_quadrature,
            final_state=final_state,
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
            bands.fermi_energy,
            geometry.temperature_k,
            eta,
            k_chunk=k_chunk,
            omega_chunk=energy_chunk,
            checkpoint=checkpoint,
        )
    )
    intensity: Float64[Array, "n_k n_e"] = padded_intensity[:n_k, :n_energy]
    result: Tuple[Float64[Array, "n_k n_e"], Float64[Array, "n_k 3"]] = (
        intensity,
        k_cart,
    )
    return result


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
    domain_index: int
    bands: DiagonalizedBands
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"]
    for domain_index, (bands, hamiltonians) in enumerate(
        zip(bands_by_domain, hamiltonians_by_domain, strict=True)
    ):
        intensity_flat: Float64[Array, "n_k n_e"]
        kpoints_cart: Float64[Array, "n_k 3"]
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
        )
        kx_axis: Float64[Array, " n_kx"]
        ky_axis: Float64[Array, " n_ky"]
        kx_axis, ky_axis = _separable_grid_axes(
            kpoints_cart,
            kgrid.mesh_shape,
            kgrid.kz,
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
            provenance=f"simulate_arpes/domain={domain_index}/single-kz",
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
    bands: DiagonalizedBands
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"]
    for bands, hamiltonians in zip(
        bands_by_domain, hamiltonians_by_domain, strict=True
    ):
        intensity: Float64[Array, "n_k n_e"]
        kpoints_cart: Float64[Array, "n_k 3"]
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
        )
        kpoints_cart = eqx.error_if(
            kpoints_cart,
            ~jnp.allclose(
                kpoints_cart[:, 2],
                kpath.kz,
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
) -> DetectorRaster:
    """Simulate the canonical coherent single-kz detector raster.

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
    _validate_static_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
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
) -> DetectorRaster:
    """Simulate the canonical coherent single-kz path-cut detector raster.

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
    _validate_static_inputs(
        hamiltonians_by_domain,
        bands_by_domain,
        radial_spec,
        matrix_element_params,
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
    )
    raster: DetectorRaster = _effects.apply_detector_effects(
        physical_by_domain,
        geometry,
        detector_calibration,
        detector_effects,
    )
    return raster


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
    "normalize_intensity",
    "simulate_arpes",
    "simulate_arpes_cut",
]
