r"""Build differentiable paths and fixed-shape rasters in k-space.

Extended Summary
----------------
This module converts fractional and Cartesian k-points through one reciprocal
lattice contract. It also builds paths, reciprocal-space meshes, and ARPES
rasters without data-dependent output shapes.

Routine Listings
----------------
:func:`kpoints_frac_to_cart`
    Convert fractional k-points to Cartesian momenta.
:func:`kpoints_cart_to_frac`
    Convert Cartesian momenta to fractional k-points.
:func:`build_kpath`
    Build a labeled path between k-space anchors.
:func:`kpath_arc_length`
    Compute cumulative Cartesian distance along a k-path.
:func:`first_bz_mask`
    Mark Cartesian points inside the first Brillouin zone.
:func:`build_bz_mesh`
    Build a fixed-shape reciprocal mesh and its first-zone mask.
:func:`build_arpes_kmesh`
    Build a fixed-kz ARPES raster in fractional coordinates.
:func:`build_kmesh_hv_at_fermi`
    Build an at-Fermi photon-energy raster in fractional coordinates.

Notes
-----
Every conversion uses reciprocal vectors as matrix rows. The first-zone
operation returns a mask because a gather creates a data-dependent shape.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int64, jaxtyped

from diffpes.maths import safe_norm
from diffpes.simul import kz_from_inner_potential_at_fermi
from diffpes.types import (
    CrystalGeometry,
    KGrid,
    KPath,
    ScalarFloat,
    make_kgrid,
    make_kpath,
)


@jaxtyped(typechecker=beartype)
def kpoints_frac_to_cart(
    kpoints_frac: Float64[Array, "n_k 3"],
    geometry: CrystalGeometry,
) -> Float64[Array, "n_k 3"]:
    r"""Convert fractional k-points to Cartesian momenta.

    The row-vector convention applies the reciprocal lattice once. The
    reciprocal lattice includes the factor of :math:`2\pi`.

    :see: :class:`~.test_kspace.TestKpointsFracToCart`

    Parameters
    ----------
    kpoints_frac : Float64[Array, "n_k 3"]
        Fractional k-points.
    geometry : CrystalGeometry
        Crystal geometry with reciprocal vectors in 1/Angstrom.

    Returns
    -------
    kpoints_cart : Float64[Array, "n_k 3"]
        Cartesian momenta in 1/Angstrom.

    Notes
    -----
    The function computes ``kpoints_frac @ geometry.reciprocal``. Gradients
    flow through the coordinates and all applicable lattice entries.
    """
    kpoints_cart: Float64[Array, "n_k 3"] = kpoints_frac @ geometry.reciprocal
    return kpoints_cart


@jaxtyped(typechecker=beartype)
def kpoints_cart_to_frac(
    kpoints_cart: Float64[Array, "n_k 3"],
    geometry: CrystalGeometry,
) -> Float64[Array, "n_k 3"]:
    r"""Convert Cartesian momenta to fractional k-points.

    The function uses the closed reciprocal identity instead of a numerical
    linear solve.

    :see: :class:`~.test_kspace.TestKpointsCartToFrac`

    Parameters
    ----------
    kpoints_cart : Float64[Array, "n_k 3"]
        Cartesian momenta in 1/Angstrom.
    geometry : CrystalGeometry
        Crystal geometry with real-space vectors in Angstrom.

    Returns
    -------
    kpoints_frac : Float64[Array, "n_k 3"]
        Fractional k-points.

    Notes
    -----
    Reciprocal and real-space lattice rows satisfy
    :math:`B A^T=2\pi I`. Thus, the exact expression is
    ``kpoints_cart @ geometry.lattice.T / (2 * pi)``.
    """
    kpoints_frac: Float64[Array, "n_k 3"] = (
        kpoints_cart @ geometry.lattice.T / (2.0 * jnp.pi)
    )
    return kpoints_frac


@jaxtyped(typechecker=beartype)
def build_kpath(  # noqa: DOC503, PLR2004
    anchors: Float64[Array, "n_anchor 3"],
    geometry: CrystalGeometry,
    n_per_segment: int,
    labels: Tuple[str, ...],
    anchor_units: str = "fractional",
) -> KPath:
    """Build a labeled path between k-space anchors.

    Each segment contains both endpoints. Adjacent segments therefore repeat
    their shared anchor, which matches the Chinook ``klib.kpath`` convention.

    :see: :class:`~.test_kspace.TestBuildKpath`

    Implementation Logic
    --------------------
    1. **Convert the anchors**::

           fractional_anchors = kpoints_cart_to_frac(anchors, geometry)

       This branch applies only to anchors with absolute units.

    2. **Interpolate all segments**::

           segments = starts + fractions * (ends - starts)

       A broadcast creates every segment with a fixed output shape.

    3. **Create the path carrier**::

           return make_kpath(...)

       Static indices identify each anchor in the flattened point array.

    Parameters
    ----------
    anchors : Float64[Array, "n_anchor 3"]
        Path anchors in the coordinates selected by ``anchor_units``.
    geometry : CrystalGeometry
        Crystal geometry for absolute-to-fractional conversion.
    n_per_segment : int
        Points in each segment, including both endpoints. This value is
        **static**. A change causes retracing.
    labels : Tuple[str, ...]
        One label for each anchor. This value is **static**. A change causes
        retracing.
    anchor_units : str, optional
        Static coordinate selector. Use ``"fractional"`` or ``"absolute"``.
        A change causes retracing. Default is ``"fractional"``.

    Returns
    -------
    kpath : KPath
        Fractional path with ``(n_anchor - 1) * n_per_segment`` points.

    Raises
    ------
    ValueError
        If the path has fewer than two anchors or invalid static settings.
    EquinoxRuntimeError
        If an anchor is non-finite.

    Notes
    -----
    ``"absolute"`` means Cartesian momentum in 1/Angstrom. The interpolation
    remains differentiable with respect to anchors and the lattice.
    """
    if anchors.shape[0] < 2:  # noqa: PLR2004
        message: str = "build_kpath requires at least two anchors"
        raise ValueError(message)
    if n_per_segment < 2:  # noqa: PLR2004
        message = "n_per_segment must be at least two"
        raise ValueError(message)
    if len(labels) != anchors.shape[0]:
        message = "labels must contain one entry for each anchor"
        raise ValueError(message)
    if anchor_units not in ("fractional", "absolute"):
        message = "anchor_units must be 'fractional' or 'absolute'"
        raise ValueError(message)

    checked_anchors: Float64[Array, "n_anchor 3"] = eqx.error_if(
        anchors,
        ~jnp.all(jnp.isfinite(anchors)),
        "anchors must be finite",
    )
    fractional_anchors: Float64[Array, "n_anchor 3"]
    if anchor_units == "absolute":
        fractional_anchors = kpoints_cart_to_frac(checked_anchors, geometry)
    else:
        fractional_anchors = checked_anchors
    fractions: Float64[Array, " n_per_segment"] = jnp.linspace(
        0.0, 1.0, n_per_segment
    )
    starts: Float64[Array, "n_segment 1 3"] = fractional_anchors[:-1, None, :]
    changes: Float64[Array, "n_segment 1 3"] = (
        fractional_anchors[1:, None, :] - starts
    )
    segments: Float64[Array, "n_segment n_per_segment 3"] = (
        starts + fractions[None, :, None] * changes
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.reshape(segments, (-1, 3))
    last_index: int = kpoints.shape[0] - 1
    label_indices: Tuple[int, ...] = tuple(
        anchor_index * n_per_segment
        if anchor_index < anchors.shape[0] - 1
        else last_index
        for anchor_index in range(anchors.shape[0])
    )
    kpath: KPath = make_kpath(
        kpoints=kpoints,
        labels=labels,
        label_indices=label_indices,
        n_per_segment=n_per_segment,
    )
    return kpath


@jaxtyped(typechecker=beartype)
def kpath_arc_length(
    kpath: KPath,
    geometry: CrystalGeometry,
) -> Float64[Array, " n_k"]:
    """Compute cumulative Cartesian distance along a k-path.

    The plotting coordinate measures each segment after reciprocal-lattice
    conversion. Repeated junction points add zero distance.

    :see: :class:`~.test_kspace.TestKpathArcLength`

    Parameters
    ----------
    kpath : KPath
        Fractional path and its static plotting metadata.
    geometry : CrystalGeometry
        Crystal geometry with reciprocal vectors in 1/Angstrom.

    Returns
    -------
    arc_length : Float64[Array, " n_k"]
        Cumulative Cartesian path distance in 1/Angstrom.

    Notes
    -----
    :func:`diffpes.maths.safe_norm` gives a zero gradient for a repeated
    junction. All nonzero segment lengths retain their usual gradients.
    """
    cartesian: Float64[Array, "n_k 3"] = kpoints_frac_to_cart(
        kpath.kpoints, geometry
    )
    changes: Float64[Array, "n_delta 3"] = jnp.diff(cartesian, axis=0)
    segment_lengths: Float64[Array, " n_delta"] = safe_norm(changes)
    arc_length: Float64[Array, " n_k"] = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=segment_lengths.dtype),
            jnp.cumsum(segment_lengths),
        )
    )
    return arc_length


@jaxtyped(typechecker=beartype)
def first_bz_mask(  # noqa: DOC503, PLR2004
    kpoints_cart: Float64[Array, "n_k 3"],
    geometry: CrystalGeometry,
    shell_radius: int = 2,
) -> Bool[Array, " n_k"]:
    """Mark Cartesian points inside the first Brillouin zone.

    The Wigner-Seitz test compares the origin distance with reciprocal-lattice
    points in a static integer shell. A singular-value bound proves that no
    point outside the shell can change a retained membership result.

    :see: :class:`~.test_kspace.TestFirstBzMask`

    Parameters
    ----------
    kpoints_cart : Float64[Array, "n_k 3"]
        Cartesian momenta in 1/Angstrom.
    geometry : CrystalGeometry
        Crystal geometry with reciprocal vectors in 1/Angstrom.
    shell_radius : int, optional
        Static maximum absolute reciprocal coefficient. Increase this value
        for a skew, anisotropic, or unreduced basis. Default is 2.

    Returns
    -------
    mask : Bool[Array, " n_k"]
        True for points inside or on the first-zone boundary.

    Raises
    ------
    ValueError
        If ``shell_radius`` is less than one.
    EquinoxRuntimeError
        If the static shell is not provably complete for the supplied points
        and reciprocal basis.

    Notes
    -----
    The comparisons use squared distances and include ties. The Boolean mask
    has no boundary derivative. Consumers must not differentiate through its
    discrete membership changes. Treating its current values as fixed does
    not supply a lattice-shape derivative for a mask-weighted observable.
    Such an observable needs a preregistered fixed mask or a separate smooth
    shape-derivative scheme.

    For any unseen reciprocal vector ``G = n @ B``, its norm is at least
    ``sigma_min(B) * norm(n)``. A vector can beat the origin only when
    ``norm(G) <= 2 * norm(k)``. The function checks this sufficient bound for
    every point that survives the requested shell. It raises instead of
    returning an uncertified false positive when the bound is inconclusive.
    """
    if shell_radius < 1:
        message: str = "shell_radius must be at least one"
        raise ValueError(message)
    if kpoints_cart.shape[0] == 0:
        empty_mask: Bool[Array, " 0"] = jnp.ones((0,), dtype=bool)
        return empty_mask

    shell_axis: Float64[Array, " shell_size"] = jnp.arange(
        -shell_radius,
        shell_radius + 1,
        dtype=jnp.float64,
    )
    shell_x: Float64[Array, "shell_size shell_size shell_size"]
    shell_y: Float64[Array, "shell_size shell_size shell_size"]
    shell_z: Float64[Array, "shell_size shell_size shell_size"]
    shell_x, shell_y, shell_z = jnp.meshgrid(
        shell_axis, shell_axis, shell_axis, indexing="ij"
    )
    shell: Float64[Array, "n_shell 3"] = jnp.stack(
        (shell_x, shell_y, shell_z), axis=-1
    ).reshape((-1, 3))
    reciprocal_points: Float64[Array, "n_shell 3"] = (
        shell @ geometry.reciprocal
    )
    origin_distances: Float64[Array, " n_k"] = jnp.sum(
        kpoints_cart * kpoints_cart, axis=-1
    )

    def compare_reciprocal_point(
        index: Int64[Array, ""],
        current_mask: Bool[Array, " n_k"],
    ) -> Bool[Array, " n_k"]:
        """Update membership against one reciprocal-lattice point."""
        differences: Float64[Array, "n_k 3"] = (
            kpoints_cart - reciprocal_points[index]
        )
        reciprocal_distances: Float64[Array, " n_k"] = jnp.sum(
            differences * differences,
            axis=-1,
        )
        distance_scale: Float64[Array, " n_k"] = jnp.maximum(
            1.0,
            jnp.maximum(origin_distances, reciprocal_distances),
        )
        distance_tolerance: Float64[Array, " n_k"] = (
            64.0 * jnp.finfo(kpoints_cart.dtype).eps * distance_scale
        )
        updated_mask: Bool[Array, " n_k"] = current_mask & (
            origin_distances <= reciprocal_distances + distance_tolerance
        )
        return updated_mask

    initial_mask: Bool[Array, " n_k"] = jnp.ones(
        (kpoints_cart.shape[0],),
        dtype=bool,
    )
    mask: Bool[Array, " n_k"] = jax.lax.fori_loop(
        0,
        reciprocal_points.shape[0],
        compare_reciprocal_point,
        initial_mask,
    )
    retained_norms: Float64[Array, " n_k"] = jnp.where(
        mask,
        safe_norm(kpoints_cart),
        0.0,
    )
    maximum_retained_norm: Float64[Array, ""] = jnp.max(retained_norms)
    singular_values: Float64[Array, " 3"] = jnp.linalg.svdvals(
        geometry.reciprocal
    )
    singular_tolerance: Float64[Array, ""] = (
        64.0 * jnp.finfo(geometry.reciprocal.dtype).eps * singular_values[0]
    )
    smallest_singular_value: Float64[Array, ""] = jnp.maximum(
        singular_values[-1] - singular_tolerance,
        0.0,
    )
    sufficient: Bool[Array, ""] = (
        2.0 * maximum_retained_norm <= smallest_singular_value * shell_radius
    )
    certified_mask: Bool[Array, " n_k"] = eqx.error_if(
        mask,
        ~sufficient,
        "first_bz_mask: shell_radius is not provably sufficient",
    )
    return certified_mask


@jaxtyped(typechecker=beartype)
def build_bz_mesh(  # noqa: DOC503, PLR2004
    geometry: CrystalGeometry,
    n_per_axis: int,
    shell_radius: int = 2,
) -> Tuple[KGrid, Bool[Array, " n_k"]]:
    """Build a fixed-shape reciprocal mesh and its first-zone mask.

    The mesh samples the fractional cube from -1 to 1. A basis-derived bound
    must prove that this cube contains the complete first zone.

    :see: :class:`~.test_kspace.TestBuildBzMesh`

    Implementation Logic
    --------------------
    1. **Build the fractional cube**::

           mesh_x, mesh_y, mesh_z = jnp.meshgrid(axis, axis, axis)

       Static axis lengths give the flattened point array a fixed shape.

    2. **Compute zone membership**::

           mask = first_bz_mask(kpoints_cart, geometry)

       The mask preserves the fixed point count during compiled execution.

    3. **Return the grid and mask**::

           return kgrid, mask

       The grid stores the three-dimensional mesh as rows of z coordinates.

    Parameters
    ----------
    geometry : CrystalGeometry
        Crystal geometry for reciprocal conversion.
    n_per_axis : int
        Number of samples on each fractional axis. This value is **static**.
        A change causes retracing.
    shell_radius : int, optional
        Static reciprocal-coefficient radius for the first-zone test. Increase
        this value for a skew, anisotropic, or unreduced basis. Default is 2.

    Returns
    -------
    kgrid : KGrid
        Fractional mesh with ``n_per_axis**3`` flattened points.
    mask : Bool[Array, " n_k"]
        First-zone membership for every mesh point.

    Raises
    ------
    ValueError
        If ``n_per_axis`` is less than two or ``shell_radius`` is less than
        one.
    EquinoxRuntimeError
        If the reciprocal basis does not prove that the fractional cube
        contains the first zone. The function also raises if the shell is not
        provably complete for the provisional first-zone points.

    Notes
    -----
    The static mesh shape is ``(n_per_axis**2, n_per_axis)``. Each row follows
    the z axis for one pair of x and y coordinates.

    The coverage guard uses the inequalities
    ``abs(k dot b_i) <= norm(b_i)**2 / 2`` that every first-zone point obeys.
    It maps their coordinate-wise bounds through the reciprocal Gram matrix.
    A failed conservative bound asks the caller for a reduced reciprocal basis
    instead of returning a partial zone.
    """
    if n_per_axis < 2:  # noqa: PLR2004
        message: str = "n_per_axis must be at least two"
        raise ValueError(message)
    reciprocal_gram: Float64[Array, "3 3"] = (
        geometry.reciprocal @ geometry.reciprocal.T
    )
    half_squared_lengths: Float64[Array, " 3"] = 0.5 * jnp.diag(
        reciprocal_gram
    )
    inverse_gram: Float64[Array, "3 3"] = jnp.linalg.inv(reciprocal_gram)
    required_fractional_extents: Float64[Array, " 3"] = (
        half_squared_lengths @ jnp.abs(inverse_gram)
    )
    coverage_tolerance: Float64[Array, " 3"] = (
        64.0
        * jnp.finfo(geometry.reciprocal.dtype).eps
        * jnp.maximum(1.0, required_fractional_extents)
    )
    coverage_sufficient: Bool[Array, ""] = jnp.all(
        required_fractional_extents <= 1.0 + coverage_tolerance
    )
    unchecked_axis: Float64[Array, " n_per_axis"] = jnp.linspace(
        -1.0,
        1.0,
        n_per_axis,
    )
    axis: Float64[Array, " n_per_axis"] = eqx.error_if(
        unchecked_axis,
        ~coverage_sufficient,
        "build_bz_mesh: reciprocal basis does not prove cube coverage",
    )
    mesh_x: Float64[Array, "n_per_axis n_per_axis n_per_axis"]
    mesh_y: Float64[Array, "n_per_axis n_per_axis n_per_axis"]
    mesh_z: Float64[Array, "n_per_axis n_per_axis n_per_axis"]
    mesh_x, mesh_y, mesh_z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    kpoints: Float64[Array, "n_k 3"] = jnp.stack(
        (mesh_x, mesh_y, mesh_z), axis=-1
    ).reshape((-1, 3))
    kgrid: KGrid = make_kgrid(
        kpoints=kpoints,
        mesh_shape=(n_per_axis * n_per_axis, n_per_axis),
    )
    cartesian: Float64[Array, "n_k 3"] = kpoints_frac_to_cart(
        kpoints, geometry
    )
    mask: Bool[Array, " n_k"] = first_bz_mask(
        cartesian,
        geometry,
        shell_radius=shell_radius,
    )
    result: Tuple[KGrid, Bool[Array, " n_k"]] = (kgrid, mask)
    return result


@jaxtyped(typechecker=beartype)
def build_arpes_kmesh(
    kx_axis_inv_ang: Float64[Array, " n_kx"],
    ky_axis_inv_ang: Float64[Array, " n_ky"],
    kz_inv_ang: ScalarFloat,
    sample_azimuth: ScalarFloat,
    geometry: CrystalGeometry,
) -> KGrid:
    """Build a fixed-kz ARPES raster in fractional coordinates.

    The input axes describe the laboratory frame. A negative sample azimuth
    rotates that raster into the crystal frame before fractional conversion.

    :see: :class:`~.test_kspace.TestBuildArpesKmesh`

    Implementation Logic
    --------------------
    1. **Build the laboratory raster**::

           lab_x, lab_y = jnp.meshgrid(kx_axis, ky_axis)

       The result uses rows for y and columns for x.

    2. **Apply the sample rotation**::

           sample_x = cosine * lab_x + sine * lab_y

       This expression applies an active rotation by negative azimuth.

    3. **Convert and return the grid**::

           return make_kgrid(kpoints_frac, mesh_shape, kz)

       The fixed ``kz`` value remains a traced carrier leaf.

    Parameters
    ----------
    kx_axis_inv_ang : Float64[Array, " n_kx"]
        Laboratory x momenta in 1/Angstrom.
    ky_axis_inv_ang : Float64[Array, " n_ky"]
        Laboratory y momenta in 1/Angstrom.
    kz_inv_ang : ScalarFloat
        Fixed out-of-plane momentum in 1/Angstrom.
    sample_azimuth : ScalarFloat
        Sample rotation about the surface normal in radians.
    geometry : CrystalGeometry
        Crystal geometry for Cartesian-to-fractional conversion.

    Returns
    -------
    kgrid : KGrid
        Fractional raster with static shape ``(n_ky, n_kx)``.

    Notes
    -----
    Gradients flow through both axes, ``kz``, the azimuth, and the lattice.
    The raster shape depends only on the static input shapes.
    """
    lab_x: Float64[Array, "n_ky n_kx"]
    lab_y: Float64[Array, "n_ky n_kx"]
    lab_x, lab_y = jnp.meshgrid(
        kx_axis_inv_ang, ky_axis_inv_ang, indexing="xy"
    )
    azimuth: Float64[Array, ""] = jnp.asarray(
        sample_azimuth, dtype=jnp.float64
    )
    cosine: Float64[Array, ""] = jnp.cos(azimuth)
    sine: Float64[Array, ""] = jnp.sin(azimuth)
    sample_x: Float64[Array, "n_ky n_kx"] = cosine * lab_x + sine * lab_y
    sample_y: Float64[Array, "n_ky n_kx"] = -sine * lab_x + cosine * lab_y
    kz_array: Float64[Array, ""] = jnp.asarray(kz_inv_ang, dtype=jnp.float64)
    sample_z: Float64[Array, "n_ky n_kx"] = jnp.broadcast_to(
        kz_array, sample_x.shape
    )
    cartesian: Float64[Array, "n_k 3"] = jnp.stack(
        (sample_x, sample_y, sample_z), axis=-1
    ).reshape((-1, 3))
    fractional: Float64[Array, "n_k 3"] = kpoints_cart_to_frac(
        cartesian, geometry
    )
    kgrid: KGrid = make_kgrid(
        kpoints=fractional,
        mesh_shape=(ky_axis_inv_ang.shape[0], kx_axis_inv_ang.shape[0]),
        kz=kz_array,
    )
    return kgrid


@jaxtyped(typechecker=beartype)
def build_kmesh_hv_at_fermi(  # noqa: DOC502, PLR2004
    kpar_axis_inv_ang: Float64[Array, " n_kpar"],
    photon_energies_ev: Float64[Array, " n_hv"],
    work_function_ev: ScalarFloat,
    inner_potential_ev: ScalarFloat,
    sample_azimuth: ScalarFloat,
    kpar_direction: Float64[Array, "2"],
    geometry: CrystalGeometry,
) -> KGrid:
    """Build an at-Fermi photon-energy raster in fractional coordinates.

    Each row contains one photon energy and all requested parallel momenta.
    The free-electron final-state model supplies row-dependent ``kz`` at
    ``omega_rel_fermi_ev = 0``. Use exact energy-dependent kinematics for
    finite-energy spectra.

    :see: :class:`~.test_kspace.TestBuildKmeshHvAtFermi`

    Implementation Logic
    --------------------
    1. **Compute each out-of-plane row**::

           kz_rows = jax.vmap(kz_for_energy)(photon_energies_ev)

       One vectorized map preserves the dense photon-energy axis.

    2. **Build and rotate the Cartesian raster**::

           sample_x = cosine * lab_x + sine * lab_y

       The traced azimuth maps the laboratory direction into the sample.

    3. **Convert and return the grid**::

           return make_kgrid(..., photon_energy_axis_ev=photon_energies_ev)

       The carrier records one photon energy for each raster row.

    Parameters
    ----------
    kpar_axis_inv_ang : Float64[Array, " n_kpar"]
        Signed parallel momenta in 1/Angstrom.
    photon_energies_ev : Float64[Array, " n_hv"]
        Photon energies in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    inner_potential_ev : ScalarFloat
        Inner potential in eV.
    sample_azimuth : ScalarFloat
        Sample rotation about the surface normal in radians.
    kpar_direction : Float64[Array, "2"]
        Unit direction in the laboratory surface plane.
    geometry : CrystalGeometry
        Crystal geometry for Cartesian-to-fractional conversion.

    Returns
    -------
    kgrid : KGrid
        Fractional raster with static shape ``(n_hv, n_kpar)``.

    Raises
    ------
    EquinoxRuntimeError
        If the direction is not finite and unit length. The function also
        rejects a raster that contains an evanescent channel.

    Notes
    -----
    The ``KGrid`` carrier stores no scalar ``kz`` because each row has a
    different value. The third coordinate of each point stores that value.
    """
    direction_norm: Float64[Array, ""] = safe_norm(kpar_direction)
    checked_direction: Float64[Array, "2"] = eqx.error_if(
        kpar_direction,
        ~jnp.all(jnp.isfinite(kpar_direction))
        | (jnp.abs(direction_norm - 1.0) > 1e-12),  # noqa: PLR2004
        "kpar_direction must be finite and have unit length",
    )

    def kz_for_energy(
        photon_energy: Float64[Array, ""],
    ) -> Tuple[Complex128[Array, " n_kpar"], Bool[Array, " n_kpar"]]:
        """Compute one out-of-plane row for a photon energy."""
        kz_row: Complex128[Array, " n_kpar"]
        propagating_row: Bool[Array, " n_kpar"]
        kz_row, propagating_row = kz_from_inner_potential_at_fermi(
            photon_energy,
            work_function_ev,
            inner_potential_ev,
            jnp.abs(kpar_axis_inv_ang),
        )
        result: Tuple[Complex128[Array, " n_kpar"], Bool[Array, " n_kpar"]] = (
            kz_row,
            propagating_row,
        )
        return result

    kz_rows: Complex128[Array, "n_hv n_kpar"]
    propagating: Bool[Array, "n_hv n_kpar"]
    kz_rows, propagating = jax.vmap(kz_for_energy)(photon_energies_ev)
    checked_kz_rows: Complex128[Array, "n_hv n_kpar"] = eqx.error_if(
        kz_rows,
        ~jnp.all(propagating),
        "build_kmesh_hv_at_fermi requires propagating channels",
    )
    real_kz_rows: Float64[Array, "n_hv n_kpar"] = jnp.real(checked_kz_rows)
    lab_x: Float64[Array, "n_hv n_kpar"] = jnp.broadcast_to(
        kpar_axis_inv_ang[None, :] * checked_direction[0], real_kz_rows.shape
    )
    lab_y: Float64[Array, "n_hv n_kpar"] = jnp.broadcast_to(
        kpar_axis_inv_ang[None, :] * checked_direction[1], real_kz_rows.shape
    )
    azimuth: Float64[Array, ""] = jnp.asarray(
        sample_azimuth, dtype=jnp.float64
    )
    cosine: Float64[Array, ""] = jnp.cos(azimuth)
    sine: Float64[Array, ""] = jnp.sin(azimuth)
    sample_x: Float64[Array, "n_hv n_kpar"] = cosine * lab_x + sine * lab_y
    sample_y: Float64[Array, "n_hv n_kpar"] = -sine * lab_x + cosine * lab_y
    cartesian: Float64[Array, "n_k 3"] = jnp.stack(
        (sample_x, sample_y, real_kz_rows), axis=-1
    ).reshape((-1, 3))
    fractional: Float64[Array, "n_k 3"] = kpoints_cart_to_frac(
        cartesian, geometry
    )
    kgrid: KGrid = make_kgrid(
        kpoints=fractional,
        mesh_shape=(photon_energies_ev.shape[0], kpar_axis_inv_ang.shape[0]),
        photon_energy_axis_ev=photon_energies_ev,
    )
    return kgrid


__all__: list[str] = [
    "build_arpes_kmesh",
    "build_bz_mesh",
    "build_kmesh_hv_at_fermi",
    "build_kpath",
    "first_bz_mask",
    "kpath_arc_length",
    "kpoints_cart_to_frac",
    "kpoints_frac_to_cart",
]
