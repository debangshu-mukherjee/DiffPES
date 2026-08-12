"""PRIVATE: Assemble exact and broadened bulk out-of-plane spectra.

Extended Summary
----------------
This private module maps and folds bulk momentum before the detector boundary.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int32

from diffpes.constants import CARTESIAN_COMPONENTS
from diffpes.tightb import bloch_hamiltonian_batch
from diffpes.types import (
    CrystalGeometry,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    RadialQuadratureSpec,
    RadialSpec,
    ScalarFloat,
    SelfEnergyModel,
    SurfaceCell,
    TBModel,
)

from ._spectrum_stream import _padded_extent, _stream_cartesian_intensity
from .kinematics import kz_from_inner_potential
from .kz_broadening import (
    _kz_wrapped_lorentzian_bin_weight,
    _map_surface_fractional_to_bulk,
    _surface_kz_frame,
    kz_fractional_nodes,
)


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
    normal_hat: Float64[Array, " 3"] = _surface_kz_frame(
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
        atom_indices: Int32[Array, " n_orb"] = jnp.asarray(
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
    direct_surface, _, normal_hat, _ = _surface_kz_frame(
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
    surface_folded, bulk_folded = _map_surface_fractional_to_bulk(
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
    expected_nodes: Float64[Array, " n_kz"] = kz_fractional_nodes(
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
    direct_surface, _, normal_hat, period_inv_ang = _surface_kz_frame(
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
            surface_block, bulk_block = _map_surface_fractional_to_bulk(
                k_parallel_block,
                folded_block_nodes,
                surface_cell,
                model.geometry,
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
        weight: Float64[Array, "n_k n_e"] = _kz_wrapped_lorentzian_bin_weight(
            lower_edge,
            upper_edge,
            center_folded,
            geometry.mean_free_path_ang,
            period_inv_ang,
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


__all__: list[str] = []
