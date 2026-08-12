"""PRIVATE: Validate coherent ARPES driver structure.

Extended Summary
----------------
This private module validates static ownership and source axes.
It also validates separable grids.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.constants import CARTESIAN_COMPONENTS
from diffpes.types import (
    DiagonalizedBands,
    MatrixElementParams,
    OrbitalBasis,
    RadialSpec,
    SurfaceCell,
    TBModel,
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


__all__: list[str] = []
