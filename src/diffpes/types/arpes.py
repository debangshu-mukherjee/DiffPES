"""Define ARPES data carriers and coordinate slices.

Extended Summary
----------------
This module stores source-coordinate ARPES intensity. It also
provides differentiable slices and energy-window maps.

Routine Listings
----------------
:class:`ArpesCube`
    Store source-coordinate ARPES intensity on a Cartesian momentum raster.
:class:`ArpesSpectrum`
    Store self-describing ARPES path intensity in a JAX PyTree.
:func:`constant_energy_map`
    Compute an ARPES map inside an explicit energy window.
:func:`fermi_surface_map`
    Compute an ARPES map around the Fermi level.
:func:`make_arpes_cube`
    Create a validated ``ArpesCube`` instance.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`slice_edc`
    Interpolate an energy-distribution curve from an ARPES cube.
:func:`slice_mdc`
    Interpolate a momentum-distribution map from an ARPES cube.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, Int32, jaxtyped

from diffpes.constants import (
    MIN_INTERPOLATION_AXIS_POINTS,
    PATH_STEP_ATOL_INV_ANG,
    PATH_STEP_RTOL,
    SAMPLE_CARTESIAN_FRAME_ID,
)

from .aliases import ScalarFloat


class ArpesCube(eqx.Module):
    """Store source-coordinate ARPES intensity on a Cartesian momentum raster.

    The carrier binds a three-dimensional intensity field to explicit
    Cartesian momentum and relative-energy coordinates. Static metadata names
    the registered frame and records human-readable provenance.

    :see: :class:`~.test_arpes.TestArpesCube`

    Attributes
    ----------
    intensity : Float64[Array, "n_kx n_ky n_e"]
        Physical source intensity on the Cartesian momentum and energy grid.
    kx_axis : Float64[Array, " n_kx"]
        Cartesian sample-frame momentum axis in inverse angstroms.
    ky_axis : Float64[Array, " n_ky"]
        Cartesian sample-frame momentum axis in inverse angstroms.
    energy_axis : Float64[Array, " n_e"]
        Energy relative to the Fermi level in eV.
    provenance : str
        **Static.** Human-readable source description. Changing it triggers
        retracing. Machine-verifiable provenance belongs in a certificate.
    cartesian_frame_id : str
        **Static.** Registered Cartesian sample-frame identifier.
        Changing it triggers retracing.

    Notes
    -----
    This pre-detector carrier is an immutable :class:`equinox.Module`. Its
    numerical leaves remain differentiable. It is not a detector raster:
    nonlinear detector coordinates require an explicit calibrated mapping.

    See Also
    --------
    make_arpes_cube : Validated factory for this type.
    """

    intensity: Float64[Array, "n_kx n_ky n_e"]
    kx_axis: Float64[Array, " n_kx"]
    ky_axis: Float64[Array, " n_ky"]
    energy_axis: Float64[Array, " n_e"]
    provenance: str = eqx.field(static=True)
    cartesian_frame_id: str = eqx.field(static=True)


class ArpesSpectrum(eqx.Module):
    """Store self-describing ARPES path intensity in a JAX PyTree.

    The carrier keeps every Cartesian path vector alongside cumulative path
    distance. This contract prevents downstream code from treating a
    one-dimensional plotting coordinate as complete momentum geometry.

    :see: :class:`~.test_arpes.TestArpesSpectrum`

    Attributes
    ----------
    intensity : Float64[Array, "n_k n_e"]
        Physical source intensity along a momentum path.
    energy_axis : Float64[Array, " n_e"]
        Energy relative to the Fermi level in eV.
    k_axis : Float64[Array, " n_k"]
        Cumulative Cartesian path distance in inverse angstroms.
    kpoints_cart_inv_ang : Float64[Array, "n_k 3"]
        Full Cartesian path in the registered sample frame, in inverse
        angstroms.
    cartesian_frame_id : str
        **Static.** Registered Cartesian sample-frame identifier.
        Changing it triggers retracing.

    Notes
    -----
    Cumulative distance alone cannot distinguish paths with equal lengths but
    different directions. The full Cartesian path and its static frame
    identity therefore remain attached to the intensity through detector
    mapping and inversion.

    See Also
    --------
    make_arpes_spectrum : Validated factory for this type.
    """

    intensity: Float64[Array, "n_k n_e"]
    energy_axis: Float64[Array, " n_e"]
    k_axis: Float64[Array, " n_k"]
    kpoints_cart_inv_ang: Float64[Array, "n_k 3"]
    cartesian_frame_id: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_arpes_cube(  # noqa: DOC503
    intensity: Float64[Array, "Kx Ky Ei"],
    kx_axis: Float64[Array, " Kxa"],
    ky_axis: Float64[Array, " Kya"],
    energy_axis: Float64[Array, " Ea"],
    cartesian_frame_id: str = SAMPLE_CARTESIAN_FRAME_ID,
    provenance: str = "",
) -> ArpesCube:
    """Create a validated ``ArpesCube`` instance.

    The factory normalizes source arrays to float64 and binds finite,
    nonnegative, and monotone-axis checks to the returned carrier.

    :see: :class:`~.test_arpes.TestMakeArpesCube`

    Parameters
    ----------
    intensity : Float64[Array, "Kx Ky Ei"]
        Source-coordinate physical intensity.
    kx_axis : Float64[Array, " Kxa"]
        Cartesian ``k_x`` axis in inverse angstroms.
    ky_axis : Float64[Array, " Kya"]
        Cartesian ``k_y`` axis in inverse angstroms.
    energy_axis : Float64[Array, " Ea"]
        Energy relative to the Fermi level in eV.
    cartesian_frame_id : str, optional
        **Static.** Registered Cartesian sample frame. Changing it triggers
        retracing.
    provenance : str, optional
        **Static.** Human-readable source description. Changing it triggers
        retracing.

    Returns
    -------
    cube : ArpesCube
        Validated source-coordinate cube with float64 leaves.

    Raises
    ------
    ValueError
        If dimensions disagree, an interpolation axis has fewer than two
        points, or the frame identifier is not registered.
    EquinoxRuntimeError
        If numerical values are non-finite, intensity is negative, or an axis
        is not strictly increasing.

    Notes
    -----
    Value-threaded Equinox checks preserve the same numerical validation in
    eager and compiled execution.
    """
    intensity_arr: Float64[Array, "Kx Ky E"] = jnp.asarray(
        intensity, dtype=jnp.float64
    )
    kx_arr: Float64[Array, " Kx"] = jnp.asarray(kx_axis, dtype=jnp.float64)
    ky_arr: Float64[Array, " Ky"] = jnp.asarray(ky_axis, dtype=jnp.float64)
    energy_arr: Float64[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )
    if intensity_arr.shape != (
        kx_arr.shape[0],
        ky_arr.shape[0],
        energy_arr.shape[0],
    ):
        raise ValueError("make_arpes_cube: intensity and axes disagree")
    if (
        min(kx_arr.shape[0], ky_arr.shape[0], energy_arr.shape[0])
        < MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_arpes_cube: each axis requires at least two points"
        )
    if cartesian_frame_id != SAMPLE_CARTESIAN_FRAME_ID:
        raise ValueError("make_arpes_cube: unknown Cartesian frame identifier")

    def validate_and_create() -> ArpesCube:
        """Validate traced leaves and construct the source cube.

        Returns
        -------
        cube : ArpesCube
            Validated Cartesian source-density cube.
        """
        nonlocal energy_arr, intensity_arr, kx_arr, ky_arr
        intensity_arr = eqx.error_if(
            intensity_arr,
            ~jnp.all(jnp.isfinite(intensity_arr)),
            "make_arpes_cube: intensity finite",
        )
        intensity_arr = eqx.error_if(
            intensity_arr,
            ~jnp.all(intensity_arr >= 0.0),
            "make_arpes_cube: intensity non negative",
        )
        kx_arr = eqx.error_if(
            kx_arr,
            ~jnp.all(jnp.isfinite(kx_arr)) | ~jnp.all(jnp.diff(kx_arr) > 0.0),
            "make_arpes_cube: kx axis finite and strictly increasing",
        )
        ky_arr = eqx.error_if(
            ky_arr,
            ~jnp.all(jnp.isfinite(ky_arr)) | ~jnp.all(jnp.diff(ky_arr) > 0.0),
            "make_arpes_cube: ky axis finite and strictly increasing",
        )
        energy_arr = eqx.error_if(
            energy_arr,
            ~jnp.all(jnp.isfinite(energy_arr))
            | ~jnp.all(jnp.diff(energy_arr) > 0.0),
            "make_arpes_cube: energy axis finite and strictly increasing",
        )
        validated_cube: ArpesCube = ArpesCube(
            intensity=intensity_arr,
            kx_axis=kx_arr,
            ky_axis=ky_arr,
            energy_axis=energy_arr,
            provenance=provenance,
            cartesian_frame_id=cartesian_frame_id,
        )
        return validated_cube

    cube: ArpesCube = validate_and_create()
    return cube


@jaxtyped(typechecker=beartype)
def make_arpes_spectrum(  # noqa: DOC503
    intensity: Float64[Array, "K Ei"],
    energy_axis: Float64[Array, " Ea"],
    k_axis: Float64[Array, " Ka"],
    kpoints_cart_inv_ang: Float64[Array, "Kc 3"],
    cartesian_frame_id: str = SAMPLE_CARTESIAN_FRAME_ID,
) -> ArpesSpectrum:
    """Create a validated ``ArpesSpectrum`` instance.

    The factory verifies array dimensions and checks cumulative distance
    against the complete Cartesian path. It preserves the registered sample
    frame as static metadata.

    :see: :class:`~.test_arpes.TestMakeArpesSpectrum`

    Parameters
    ----------
    intensity : Float64[Array, "K Ei"]
        Source-coordinate physical intensity along a path.
    energy_axis : Float64[Array, " Ea"]
        Energy relative to the Fermi level in eV.
    k_axis : Float64[Array, " Ka"]
        Cumulative Cartesian path distance in inverse angstroms.
    kpoints_cart_inv_ang : Float64[Array, "Kc 3"]
        Full Cartesian path in inverse angstroms.
    cartesian_frame_id : str, optional
        **Static.** Registered Cartesian sample frame. Changing it triggers
        retracing.

    Returns
    -------
    spectrum : ArpesSpectrum
        Validated self-describing spectrum with float64 leaves.

    Raises
    ------
    ValueError
        If dimensions disagree, the path is empty, the energy axis has fewer
        than two points, or the
        frame identifier is not registered.
    EquinoxRuntimeError
        If values are non-finite, intensity is negative, axes are not strictly
        increasing, or Cartesian step lengths disagree with ``diff(k_axis)``.

    Notes
    -----
    Value-threaded Equinox checks keep geometry and numerical validation alive
    in eager and compiled execution.
    """
    intensity_arr: Float64[Array, "K E"] = jnp.asarray(
        intensity, dtype=jnp.float64
    )
    energy_arr: Float64[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )
    k_axis_arr: Float64[Array, " K"] = jnp.asarray(k_axis, dtype=jnp.float64)
    kpoints_arr: Float64[Array, "K 3"] = jnp.asarray(
        kpoints_cart_inv_ang, dtype=jnp.float64
    )
    if intensity_arr.shape != (k_axis_arr.shape[0], energy_arr.shape[0]):
        raise ValueError("make_arpes_spectrum: intensity and axes disagree")
    if kpoints_arr.shape[0] != k_axis_arr.shape[0]:
        raise ValueError(
            "make_arpes_spectrum: Cartesian points and k_axis disagree"
        )
    if k_axis_arr.shape[0] < 1 or (
        energy_arr.shape[0] < MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_arpes_spectrum: path cannot be empty and energy requires "
            "two points"
        )
    if cartesian_frame_id != SAMPLE_CARTESIAN_FRAME_ID:
        raise ValueError(
            "make_arpes_spectrum: unknown Cartesian frame identifier"
        )

    def validate_and_create() -> ArpesSpectrum:
        """Validate traced leaves and construct the source spectrum.

        Returns
        -------
        spectrum : ArpesSpectrum
            Validated self-describing path spectrum.
        """
        nonlocal energy_arr, intensity_arr, k_axis_arr, kpoints_arr
        intensity_arr = eqx.error_if(
            intensity_arr,
            ~jnp.all(jnp.isfinite(intensity_arr)),
            "make_arpes_spectrum: intensity finite",
        )
        intensity_arr = eqx.error_if(
            intensity_arr,
            ~jnp.all(intensity_arr >= 0.0),
            "make_arpes_spectrum: intensity non negative",
        )
        energy_arr = eqx.error_if(
            energy_arr,
            ~jnp.all(jnp.isfinite(energy_arr))
            | ~jnp.all(jnp.diff(energy_arr) > 0.0),
            "make_arpes_spectrum: energy axis strictly increasing and finite",
        )
        k_axis_arr = eqx.error_if(
            k_axis_arr,
            ~jnp.all(jnp.isfinite(k_axis_arr))
            | ~jnp.all(jnp.diff(k_axis_arr) > 0.0),
            "make_arpes_spectrum: k axis finite and strictly increasing",
        )
        kpoints_arr = eqx.error_if(
            kpoints_arr,
            ~jnp.all(jnp.isfinite(kpoints_arr)),
            "make_arpes_spectrum: Cartesian points finite",
        )
        cartesian_steps: Float64[Array, " Km1"] = jnp.linalg.norm(
            jnp.diff(kpoints_arr, axis=0), axis=1
        )
        k_axis_arr = eqx.error_if(
            k_axis_arr,
            ~jnp.allclose(
                cartesian_steps,
                jnp.diff(k_axis_arr),
                rtol=PATH_STEP_RTOL,
                atol=PATH_STEP_ATOL_INV_ANG,
            ),
            "make_arpes_spectrum: Cartesian path steps disagree with k_axis",
        )
        validated_spectrum: ArpesSpectrum = ArpesSpectrum(
            intensity=intensity_arr,
            energy_axis=energy_arr,
            k_axis=k_axis_arr,
            kpoints_cart_inv_ang=kpoints_arr,
            cartesian_frame_id=cartesian_frame_id,
        )
        return validated_spectrum

    spectrum: ArpesSpectrum = validate_and_create()
    return spectrum


def _linear_bracket(
    axis: Float64[Array, " N"],
    query: Float64[Array, ""],
) -> Tuple[Int32[Array, ""], Int32[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Return adjacent indices and a guarded linear weight.

    Parameters
    ----------
    axis : Float64[Array, " N"]
        Strictly increasing source coordinate.
    query : Float64[Array, ""]
        In-domain scalar query coordinate.

    Returns
    -------
    bracket : Tuple[Int32[Array, ""], Int32[Array, ""], Float64[Array, ""]]
        Lower index, upper index, and piecewise-linear upper weight.

    Notes
    -----
    Two ``where`` guards keep a finite denominator and explicitly set the
    weight to zero if malformed repeated coordinates reach this private seam.
    Public factories prevent that case.
    """
    upper: Int32[Array, ""] = jnp.clip(
        jnp.searchsorted(axis, query, side="right"),
        1,
        axis.size - 1,
    )
    lower: Int32[Array, ""] = upper - 1
    lower_value: Float64[Array, ""] = axis[lower]
    upper_value: Float64[Array, ""] = axis[upper]
    denominator: Float64[Array, ""] = upper_value - lower_value
    safe_denominator: Float64[Array, ""] = jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    weight: Float64[Array, ""] = (query - lower_value) / safe_denominator
    weight = jnp.where(denominator > 0.0, weight, 0.0)
    bracket: Tuple[Int32[Array, ""], Int32[Array, ""], Float64[Array, ""]] = (
        lower,
        upper,
        weight,
    )
    return bracket


def _validated_query(
    query: ScalarFloat,
    axis: Float64[Array, " N"],
    name: str,
) -> Float64[Array, ""]:
    """PRIVATE: Cast one query and reject non-finite or exterior values.

    Parameters
    ----------
    query : ScalarFloat
        Scalar query accepted by a public slicer.
    axis : Float64[Array, " N"]
        Source axis defining the closed interpolation domain.
    name : str
        Public operation and coordinate name used in diagnostics.

    Returns
    -------
    validated : Float64[Array, ""]
        Float64 scalar whose eager and compiled validation is value-threaded.

    Notes
    -----
    The returned scalar carries the finite and closed-domain checks through
    compiled slicer calls.
    """
    query_arr: Float64[Array, ""] = jnp.asarray(query, dtype=jnp.float64)
    validated: Float64[Array, ""] = eqx.error_if(
        query_arr,
        ~jnp.isfinite(query_arr)
        | (query_arr < axis[0])
        | (query_arr > axis[-1]),
        f"{name}: query lies outside the source axis",
    )
    return validated


@jaxtyped(typechecker=beartype)
def slice_edc(
    cube: ArpesCube,
    kx_inv_ang: ScalarFloat,
    ky_inv_ang: ScalarFloat,
) -> Float64[Array, " n_e"]:
    """Interpolate an energy-distribution curve from an ARPES cube.

    The slicer performs bilinear interpolation over the two Cartesian
    momentum axes while retaining every sampled energy value.

    :see: :class:`~.test_arpes.TestSliceEdc`

    Parameters
    ----------
    cube : ArpesCube
        Source-coordinate physical intensity cube.
    kx_inv_ang : ScalarFloat
        Cartesian ``k_x`` query in inverse angstroms.
    ky_inv_ang : ScalarFloat
        Cartesian ``k_y`` query in inverse angstroms.

    Returns
    -------
    edc : Float64[Array, " n_e"]
        Bilinearly interpolated energy-distribution curve.

    Notes
    -----
    Integer bracket indices have zero derivatives with respect to stored axes.
    Within a grid cell, interpolation weights give exact piecewise-linear
    derivatives with respect to both query coordinates and cube intensity.
    """
    kx_query: Float64[Array, ""] = _validated_query(
        kx_inv_ang, cube.kx_axis, "slice_edc kx"
    )
    ky_query: Float64[Array, ""] = _validated_query(
        ky_inv_ang, cube.ky_axis, "slice_edc ky"
    )
    ix0: Int32[Array, ""]
    ix1: Int32[Array, ""]
    wx: Float64[Array, ""]
    iy0: Int32[Array, ""]
    iy1: Int32[Array, ""]
    wy: Float64[Array, ""]
    ix0, ix1, wx = _linear_bracket(cube.kx_axis, kx_query)
    iy0, iy1, wy = _linear_bracket(cube.ky_axis, ky_query)
    lower_y: Float64[Array, " n_e"] = (1.0 - wx) * cube.intensity[
        ix0, iy0
    ] + wx * cube.intensity[ix1, iy0]
    upper_y: Float64[Array, " n_e"] = (1.0 - wx) * cube.intensity[
        ix0, iy1
    ] + wx * cube.intensity[ix1, iy1]
    edc: Float64[Array, " n_e"] = (1.0 - wy) * lower_y + wy * upper_y
    return edc


@jaxtyped(typechecker=beartype)
def slice_mdc(
    cube: ArpesCube,
    energy_ev: ScalarFloat,
) -> Float64[Array, "n_kx n_ky"]:
    """Interpolate a momentum-distribution map from an ARPES cube.

    The slicer linearly interpolates between adjacent energy planes and
    preserves the complete Cartesian momentum raster.

    :see: :class:`~.test_arpes.TestSliceMdc`

    Parameters
    ----------
    cube : ArpesCube
        Source-coordinate physical intensity cube.
    energy_ev : ScalarFloat
        Energy query relative to the Fermi level in eV.

    Returns
    -------
    mdc : Float64[Array, "n_kx n_ky"]
        Linearly interpolated momentum-distribution map.

    Notes
    -----
    The integer energy bracket has zero derivative with respect to the stored
    axis. The interpolation weight is piecewise linear in ``energy_ev``.
    """
    energy_query: Float64[Array, ""] = _validated_query(
        energy_ev, cube.energy_axis, "slice_mdc energy"
    )
    ie0: Int32[Array, ""]
    ie1: Int32[Array, ""]
    weight: Float64[Array, ""]
    ie0, ie1, weight = _linear_bracket(cube.energy_axis, energy_query)
    mdc: Float64[Array, "n_kx n_ky"] = (1.0 - weight) * cube.intensity[
        ..., ie0
    ] + weight * cube.intensity[..., ie1]
    return mdc


@jaxtyped(typechecker=beartype)
def constant_energy_map(
    cube: ArpesCube,
    energy_ev: ScalarFloat,
    tol_ev: ScalarFloat,
) -> Float64[Array, "n_kx n_ky"]:
    """Compute an ARPES map inside an explicit energy window.

    The display helper averages every sampled plane inside a closed top-hat
    window. It rejects a window that selects no energy sample.

    :see: :class:`~.test_arpes.TestConstantEnergyMap`

    Parameters
    ----------
    cube : ArpesCube
        Source-coordinate physical intensity cube.
    energy_ev : ScalarFloat
        Centre of the display window relative to the Fermi level in eV.
    tol_ev : ScalarFloat
        Nonnegative top-hat half-width in eV.

    Returns
    -------
    energy_map : Float64[Array, "n_kx n_ky"]
        Mean of all sampled energy planes inside the closed window.

    Notes
    -----
    Membership uses an explicit top-hat. Its derivative with respect to
    ``tol_ev`` and ``energy_ev`` is zero almost everywhere by design; these
    are display parameters. Choose :func:`slice_mdc` for an energy-query
    derivative.
    """
    centre: Float64[Array, ""] = jnp.asarray(energy_ev, dtype=jnp.float64)
    tolerance: Float64[Array, ""] = jnp.asarray(tol_ev, dtype=jnp.float64)
    tolerance = eqx.error_if(
        tolerance,
        ~jnp.isfinite(tolerance) | (tolerance < 0.0),
        "constant_energy_map: tolerance finite and non negative",
    )
    centre = eqx.error_if(
        centre,
        ~jnp.isfinite(centre),
        "constant_energy_map: energy finite",
    )
    weights: Float64[Array, " n_e"] = (
        jnp.abs(cube.energy_axis - centre) <= tolerance
    ).astype(jnp.float64)
    count: Float64[Array, ""] = jnp.sum(weights)
    safe_count: Float64[Array, ""] = jnp.where(count > 0.0, count, 1.0)
    energy_map: Float64[Array, "n_kx n_ky"] = (
        jnp.tensordot(cube.intensity, weights, axes=((-1,), (0,))) / safe_count
    )
    validated_map: Float64[Array, "n_kx n_ky"] = eqx.error_if(
        energy_map,
        count <= 0.0,
        "constant_energy_map: energy window selects no samples",
    )
    return validated_map


@jaxtyped(typechecker=beartype)
def fermi_surface_map(
    cube: ArpesCube,
    tol_ev: ScalarFloat,
) -> Float64[Array, "n_kx n_ky"]:
    """Compute an ARPES map around the Fermi level.

    The display helper fixes the top-hat centre at zero relative energy and
    delegates the sampled-plane average to :func:`constant_energy_map`.

    :see: :class:`~.test_arpes.TestFermiSurfaceMap`

    Parameters
    ----------
    cube : ArpesCube
        Source-coordinate physical intensity cube.
    tol_ev : ScalarFloat
        Nonnegative top-hat half-width around zero energy in eV.

    Returns
    -------
    fermi_map : Float64[Array, "n_kx n_ky"]
        Constant-energy map centred exactly at the Fermi level.

    Notes
    -----
    This function is exactly ``constant_energy_map(cube, 0.0, tol_ev)`` and
    inherits its documented zero derivative with respect to the display
    tolerance.
    """
    fermi_map: Float64[Array, "n_kx n_ky"] = constant_energy_map(
        cube,
        jnp.asarray(0.0, dtype=jnp.float64),
        tol_ev,
    )
    return fermi_map


__all__: list[str] = [
    "ArpesCube",
    "ArpesSpectrum",
    "constant_energy_map",
    "fermi_surface_map",
    "make_arpes_cube",
    "make_arpes_spectrum",
    "slice_edc",
    "slice_mdc",
]
