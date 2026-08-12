"""PRIVATE: Construct physical ARPES source carriers.

Extended Summary
----------------
This private module creates self-describing source cubes and path spectra.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import List, Tuple
from jaxtyping import Array, Complex128, Float64

from diffpes.types import (
    ArpesCube,
    ArpesSpectrum,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    KGrid,
    KPath,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SurfaceCell,
    TBModel,
    make_arpes_cube,
    make_arpes_spectrum,
)

from ._kz_spectrum import _bulk_domain_intensity
from ._spectrum_stream import _stream_domain_intensity
from ._spectrum_validation import _separable_grid_axes


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
    cubes: List[ArpesCube] = []
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
    spectra: List[ArpesSpectrum] = []
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


__all__: list[str] = []
