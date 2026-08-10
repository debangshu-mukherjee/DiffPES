"""Define band-structure and orbital-projection data structures.

Extended Summary
----------------
This module defines PyTree types for electronic band-structure data and
orbital-resolved projections from VASP calculations. These are
the primary inputs to all ARPES simulation functions.

Routine Listings
----------------
:class:`ArpesCube`
    Store source-coordinate ARPES intensity on a Cartesian momentum raster.
:class:`ArpesSpectrum`
    Store self-describing ARPES path intensity in a JAX PyTree.
:class:`BandStructure`
    Store electronic band-structure data in a JAX PyTree.
:class:`DetectorCalibration`
    Store native detector-bin and point-spread calibration.
:class:`DetectorRaster`
    Store expected detector counts on native recorded coordinates.
:class:`OrbitalProjection`
    Store orbital-resolved band projections in a JAX PyTree.
:class:`SpinBandStructure`
    Store spin-resolved electronic band-structure data in a JAX PyTree.
:class:`SpinOrbitalProjection`
    Store orbital projections with spin data in a JAX PyTree.
:func:`constant_energy_map`
    Compute an ARPES map inside an explicit energy window.
:func:`fermi_surface_map`
    Compute an ARPES map around the Fermi level.
:func:`make_arpes_cube`
    Create a validated ``ArpesCube`` instance.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`make_band_structure`
    Create a validated ``BandStructure`` instance.
:func:`make_detector_calibration`
    Create a validated ``DetectorCalibration`` instance.
:func:`make_detector_raster`
    Create a validated ``DetectorRaster`` instance.
:func:`make_orbital_projection`
    Create a validated ``OrbitalProjection`` instance.
:func:`make_spin_band_structure`
    Create a validated ``SpinBandStructure`` instance.
:func:`make_spin_orbital_projection`
    Create a validated ``SpinOrbitalProjection`` instance.
:func:`slice_edc`
    Interpolate an energy-distribution curve from an ARPES cube.
:func:`slice_mdc`
    Interpolate a momentum-distribution map from an ARPES cube.

Notes
-----
Orbital indexing convention (9 orbitals):
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``
matching VASP PROCAR output ordering.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric
from .constants import N_ORBITALS, N_SPIN_COMPONENTS

_SAMPLE_CARTESIAN_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"
_DETECTOR_COORDINATE_SYSTEM: str = "hemispherical_angles"
_DETECTOR_BOUNDARY_POLICY: str = "loss"
_PATH_STEP_RTOL: float = 1.0e-12
_PATH_STEP_ATOL_INV_ANG: float = 1.0e-13
_MIN_INTERPOLATION_AXIS_POINTS: int = 2


class BandStructure(eqx.Module):
    """Store electronic band-structure data in a JAX PyTree.

    This type stores the core outputs of a DFT band-structure calculation.
    The outputs include E_n(k), the reciprocal-space k-point mesh, the
    k-point integration weights, and the Fermi energy. These fields describe
    the single-particle electronic structure for ARPES simulations.

    This JAX-compatible PyTree passes through ``jax.jit``, ``jax.vmap``, and
    ``jax.grad`` without manual flattening. All four fields contain JAX-traced
    arrays and no static auxiliary data. JAX can differentiate the object
    with respect to each field.


    :see: :class:`~.test_bands.TestBandStructure`

    Attributes
    ----------
    eigenvalues : Float64[Array, "K B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float64[Array, "K 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Float64[Array, " K"]
        Integration weights for each k-point.
    fermi_energy : Float64[Array, " "]
        Fermi level energy in eV.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    Equinox derives the tree structure from the annotated fields; all
    fields are differentiable leaves and no static metadata is present.
    """

    eigenvalues: Float64[Array, "K B"]
    kpoints: Float64[Array, "K 3"]
    kpoint_weights: Float64[Array, " K"]
    fermi_energy: Float64[Array, " "]


class OrbitalProjection(eqx.Module):
    """Store orbital-resolved band projections in a JAX PyTree.

    All array fields are differentiable PyTree leaves. Optional fields are
    empty subtrees when ``None``; changing their presence changes the tree
    structure and may trigger recompilation.


    :see: :class:`~.test_bands.TestOrbitalProjection`

    Attributes
    ----------
    projections : Float64[Array, "K B A 9"]
        Orbital projection weights.
    spin : Optional[Float64[Array, "K B A 6"]]
        Optional spin projections.
    oam : Optional[Float64[Array, "K B A 3"]]
        Optional orbital-angular-momentum projections.
    """

    projections: Float64[Array, "K B A 9"]
    spin: Optional[Float64[Array, "K B A 6"]]
    oam: Optional[Float64[Array, "K B A 3"]]


class SpinOrbitalProjection(eqx.Module):
    """Store orbital projections with spin data in a JAX PyTree.

    All present arrays are differentiable PyTree leaves. ``oam=None`` is an
    empty subtree; changing its presence changes the tree structure.


    :see: :class:`~.test_bands.TestSpinOrbitalProjection`

    Attributes
    ----------
    projections : Float64[Array, "K B A 9"]
        Orbital projection weights.
    spin : Float64[Array, "K B A 6"]
        Mandatory spin projections.
    oam : Optional[Float64[Array, "K B A 3"]]
        Optional orbital-angular-momentum projections.
    """

    projections: Float64[Array, "K B A 9"]
    spin: Float64[Array, "K B A 6"]
    oam: Optional[Float64[Array, "K B A 3"]]


@jaxtyped(typechecker=beartype)
def make_spin_orbital_projection(  # noqa: DOC503
    projections: Float64[Array, "Kp Bp Ap Op"],
    spin: Float64[Array, "Ks Bs As Ss"],
    oam: Optional[Float64[Array, "Ko Bo Ao 3"]] = None,
) -> SpinOrbitalProjection:
    """Create a validated ``SpinOrbitalProjection`` instance.

    The factory validates and normalizes orbital projection
    data with mandatory spin before constructing a
    ``SpinOrbitalProjection`` PyTree. This factory supports spin-orbit
    coupling. Unlike :func:`make_orbital_projection`, it requires the ``spin``
    field. Therefore, downstream SOC simulation kernels receive complete spin
    data without runtime checks.

    The factory casts all present arrays to ``float64`` for numerical
    stability. ``@jaxtyped(typechecker=beartype)`` checks the shape
    constraints at call time. The K, B, and A dimensions must agree across
    all arrays.

    :see: :class:`~.test_bands.TestMakeSpinOrbitalProjection`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           proj_arr = jnp.asarray(projections, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           proj_arr.shape[:3] != spin_arr.shape[:3]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(proj_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return soc_proj

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    projections : Float64[Array, "Kp Bp Ap Op"]
        Orbital projection weights ``|<psi|Y_{lm}>|^2`` following VASP
        ordering. Must share the K, B, A dimensions with ``spin``.
    spin : Float64[Array, "Ks Bs As Ss"]
        Spin projections ``[Sx_up, Sx_dn, Sy_up, Sy_dn, Sz_up,
        Sz_dn]``. Required (non-optional).
    oam : Optional[Float64[Array, "Ko Bo Ao 3"]], optional
        Orbital angular momentum ``[L_p, L_d, L_total]``.
        Default is None.

    Returns
    -------
    soc_proj : SpinOrbitalProjection
        Validated instance with all non-None arrays in ``float64``.

    Raises
    ------
    ValueError
        If the projection, spin, and optional OAM axes disagree. The function
        also rejects an orbital axis without 9 columns or a spin axis without
        6 columns.
    EquinoxRuntimeError
        If projection values are non-finite or negative, or spin values are
        non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the projection, spin, or OAM shapes violate their structural contract.
    Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for non-finite arrays or negative projections.

    See Also
    --------
    make_orbital_projection : Factory for the optional-spin variant.
    SpinOrbitalProjection : The PyTree class constructed by this
        factory.
    """
    proj_arr: Float64[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Float64[Array, "K B A 6"] = jnp.asarray(spin, dtype=jnp.float64)
    oam_arr: Optional[Float64[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)

    if proj_arr.shape[:3] != spin_arr.shape[:3]:
        raise ValueError(
            "make_spin_orbital_projection: projections and spin axes disagree"
        )
    if proj_arr.shape[3] != N_ORBITALS:
        raise ValueError(
            "make_spin_orbital_projection: projections must have "
            f"{N_ORBITALS} orbital columns"
        )
    if spin_arr.shape[3] != N_SPIN_COMPONENTS:
        raise ValueError(
            "make_spin_orbital_projection: spin must have 6 component columns"
        )
    if oam_arr is not None and proj_arr.shape[:3] != oam_arr.shape[:3]:
        raise ValueError(
            "make_spin_orbital_projection: projections and oam axes disagree"
        )

    def validate_and_create() -> SpinOrbitalProjection:
        nonlocal proj_arr, spin_arr
        proj_arr = eqx.error_if(
            proj_arr,
            ~(jnp.all(jnp.isfinite(proj_arr))),
            "make_spin_orbital_projection: projections finite",
        )
        proj_arr = eqx.error_if(
            proj_arr,
            ~(jnp.all(proj_arr >= 0.0)),
            "make_spin_orbital_projection: projections non negative",
        )
        spin_arr = eqx.error_if(
            spin_arr,
            ~(jnp.all(jnp.isfinite(spin_arr))),
            "make_spin_orbital_projection: spin finite",
        )
        validated_projection: SpinOrbitalProjection = SpinOrbitalProjection(
            projections=proj_arr,
            spin=spin_arr,
            oam=oam_arr,
        )
        return validated_projection

    soc_proj: SpinOrbitalProjection = validate_and_create()
    return soc_proj


class SpinBandStructure(eqx.Module):
    """Store spin-resolved electronic band-structure data in a JAX PyTree.

    This type stores eigenvalues for both spin channels from an ISPIN=2 VASP
    calculation. The two spin channels share the same k-point mesh
    and weights. ``read_eigenval`` returns this type when
    ``return_mode="full"`` and the EIGENVAL file contains spin-polarized data.

    This class is an immutable :class:`equinox.Module` PyTree. JAX stores all
    five dense array fields as children and uses no auxiliary data. JAX can
    differentiate the complete object with respect to each field.


    :see: :class:`~.test_bands.TestSpinBandStructure`

    Attributes
    ----------
    eigenvalues_up : Float64[Array, "K B"]
        Spin-up (majority) band energies in eV for K k-points and
        B bands. JAX-traced (differentiable).
    eigenvalues_down : Float64[Array, "K B"]
        Spin-down (minority) band energies in eV for K k-points
        and B bands. JAX-traced (differentiable).
    kpoints : Float64[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space, shared
        by both spin channels. JAX-traced (differentiable).
    kpoint_weights : Float64[Array, " K"]
        Integration weights for each k-point, used for Brillouin-zone
        averaging. Uniform weights (all ones) are the norm for band
        structure paths. JAX-traced (differentiable).
    fermi_energy : Float64[Array, " "]
        Fermi level energy in eV. A 0-D scalar array.
        JAX-traced (differentiable).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    Equinox derives the tree structure from the annotated fields; all
    fields are differentiable leaves and no static metadata is present.

    See Also
    --------
    BandStructure : Single-spin-channel variant.
    make_spin_band_structure : Factory function with validation and
        float64 casting.
    """

    eigenvalues_up: Float64[Array, "K B"]
    eigenvalues_down: Float64[Array, "K B"]
    kpoints: Float64[Array, "K 3"]
    kpoint_weights: Float64[Array, " K"]
    fermi_energy: Float64[Array, " "]


@jaxtyped(typechecker=beartype)
def make_spin_band_structure(  # noqa: DOC503
    eigenvalues_up: Float64[Array, "Ku Bu"],
    eigenvalues_down: Float64[Array, "Kd Bd"],
    kpoints: Float64[Array, "Kk 3"],
    kpoint_weights: Union[Float64[Array, " Kw"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> SpinBandStructure:
    """Create a validated ``SpinBandStructure`` instance.

    The factory validates and normalizes raw spin-resolved
    band structure data before constructing a ``SpinBandStructure``
    PyTree. This is the spin-polarized (ISPIN=2) counterpart to
    :func:`make_band_structure`. The factory casts all input arrays to
    ``float64`` for numerical stability. It replaces missing k-point weights
    with uniform weights. Callers therefore do not handle the common
    equal-weight case explicitly.

    ``@jaxtyped(typechecker=beartype)`` checks the input shapes and dtypes at
    call time. This check finds different K or B dimensions before the
    simulation uses them.

    :see: :class:`~.test_bands.TestMakeSpinBandStructure`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           up_arr = jnp.asarray(eigenvalues_up, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           up_arr.shape != down_arr.shape

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(up_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return bands

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    eigenvalues_up : Float64[Array, "Ku Bu"]
        Spin-up band energies in eV for K k-points and B bands.
    eigenvalues_down : Float64[Array, "Kd Bd"]
        Spin-down band energies in eV. Must share the same (K, B)
        shape as ``eigenvalues_up``.
    kpoints : Float64[Array, "Kk 3"]
        k-point coordinates in reciprocal (fractional) space.
    kpoint_weights : Union[Float64[Array, " Kw"], None], optional
        Integration weights per k-point. Defaults to uniform weights
        ``jnp.ones(K)``.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : SpinBandStructure
        Validated spin-resolved band structure with all arrays in
        ``float64``.

    Raises
    ------
    ValueError
        If the spin channels disagree on their k-point or band counts, or
        the k-point and weight counts disagree with the eigenvalues.
    EquinoxRuntimeError
        If eigenvalues or k-points are non-finite, or weights are non-finite
        or negative.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the spin, k-point, band, or weight dimensions disagree. Traced validation
    uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for non-finite
    arrays or negative weights.

    See Also
    --------
    make_band_structure : Factory for single-spin-channel data.
    SpinBandStructure : The PyTree class constructed by this factory.
    """
    up_arr: Float64[Array, "K B"] = jnp.asarray(
        eigenvalues_up, dtype=jnp.float64
    )
    down_arr: Float64[Array, "K B"] = jnp.asarray(
        eigenvalues_down, dtype=jnp.float64
    )
    kpts_arr: Float64[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    nkpts: int = up_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float64[Array, " K"] = jnp.ones(nkpts, dtype=jnp.float64)
    else:
        weights_arr = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    fermi_arr: Float64[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )

    if up_arr.shape != down_arr.shape:
        raise ValueError(
            "make_spin_band_structure: spin channels disagree on K/B axes"
        )
    if kpts_arr.shape[0] != nkpts:
        raise ValueError(
            "make_spin_band_structure: eigenvalues and kpoints disagree "
            "on K axis"
        )
    if weights_arr.shape[0] != nkpts:
        raise ValueError(
            "make_spin_band_structure: eigenvalues and weights disagree "
            "on K axis"
        )

    def validate_and_create() -> SpinBandStructure:
        nonlocal down_arr, kpts_arr, up_arr, weights_arr
        up_arr = eqx.error_if(
            up_arr,
            ~(jnp.all(jnp.isfinite(up_arr))),
            "make_spin_band_structure: eigenvalues up finite",
        )
        down_arr = eqx.error_if(
            down_arr,
            ~(jnp.all(jnp.isfinite(down_arr))),
            "make_spin_band_structure: eigenvalues down finite",
        )
        kpts_arr = eqx.error_if(
            kpts_arr,
            ~(jnp.all(jnp.isfinite(kpts_arr))),
            "make_spin_band_structure: kpoints finite",
        )
        weights_arr = eqx.error_if(
            weights_arr,
            ~(jnp.all(jnp.isfinite(weights_arr))),
            "make_spin_band_structure: weights finite",
        )
        weights_arr = eqx.error_if(
            weights_arr,
            ~(jnp.all(weights_arr >= 0.0)),
            "make_spin_band_structure: weights non negative",
        )
        validated_bands: SpinBandStructure = SpinBandStructure(
            eigenvalues_up=up_arr,
            eigenvalues_down=down_arr,
            kpoints=kpts_arr,
            kpoint_weights=weights_arr,
            fermi_energy=fermi_arr,
        )
        return validated_bands

    bands: SpinBandStructure = validate_and_create()
    return bands


class ArpesCube(eqx.Module):
    """Store source-coordinate ARPES intensity on a Cartesian momentum raster.

    The carrier binds a three-dimensional intensity field to explicit
    Cartesian momentum and relative-energy coordinates. Static metadata names
    the registered frame and records human-readable provenance.

    :see: :class:`~.test_bands.TestArpesCube`

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
        **Static.** Registered Plan-03 Cartesian sample-frame identifier.
        Changing it triggers retracing.

    Notes
    -----
    This pre-detector carrier is an immutable :class:`equinox.Module`. Its
    numerical leaves remain differentiable. It is not a detector raster:
    nonlinear detector coordinates require an explicit calibrated mapping.
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

    :see: :class:`~.test_bands.TestArpesSpectrum`

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
        **Static.** Registered Plan-03 Cartesian sample-frame identifier.
        Changing it triggers retracing.

    Notes
    -----
    Cumulative distance alone cannot distinguish paths with equal lengths but
    different directions. The full Cartesian path and its static frame
    identity therefore remain attached to the intensity through detector
    mapping and inversion.
    """

    intensity: Float64[Array, "n_k n_e"]
    energy_axis: Float64[Array, " n_e"]
    k_axis: Float64[Array, " n_k"]
    kpoints_cart_inv_ang: Float64[Array, "n_k 3"]
    cartesian_frame_id: str = eqx.field(static=True)


class DetectorCalibration(eqx.Module):
    """Store native detector-bin and point-spread calibration.

    The carrier defines recorded-coordinate edges, point-spread widths, and a
    fixed transmission domain. Static selectors make the coordinate and
    boundary conventions explicit at tracing time.

    :see: :class:`~.test_bands.TestDetectorCalibration`

    Attributes
    ----------
    u_bin_edges : Float64[Array, " n_u_plus_1"]
        Native detector ``T_x`` bin edges in radians.
    v_bin_edges : Float64[Array, " n_v_plus_1"]
        Native detector ``T_y`` bin edges in radians. A slit has two edges.
    energy_bin_edges_ev : Float64[Array, " n_e_plus_1"]
        Recorded energy-bin edges relative to the Fermi level in eV.
    psf_fwhm_u : Float64[Array, ""]
        Positive point-spread FWHM in native ``u`` radians.
    psf_fwhm_v : Float64[Array, ""]
        Positive point-spread FWHM in native ``v`` radians.
    psf_fwhm_energy_ev : Float64[Array, ""]
        Positive energy point-spread FWHM in eV.
    transmission_reference_domain_ev : Float64[Array, " 2"]
        Fixed true-kinetic-energy calibration domain in eV.
    coordinate_system : str
        **Static.** V1 is exactly ``"hemispherical_angles"``. Changing it
        triggers retracing.
    transmission_monotonic_sign : int
        **Static.** Declared transmission direction, exactly ``-1`` or ``1``.
        Changing it triggers retracing.
    boundary_policy : str
        **Static.** V1 is exactly ``"loss"``. Changing it triggers retracing.

    Notes
    -----
    This carrier is the sole authority for target bins and native detector
    PSF widths. The source mesh never infers or replaces these values.
    """

    u_bin_edges: Float64[Array, " n_u_plus_1"]
    v_bin_edges: Float64[Array, " n_v_plus_1"]
    energy_bin_edges_ev: Float64[Array, " n_e_plus_1"]
    psf_fwhm_u: Float64[Array, ""]
    psf_fwhm_v: Float64[Array, ""]
    psf_fwhm_energy_ev: Float64[Array, ""]
    transmission_reference_domain_ev: Float64[Array, " 2"]
    coordinate_system: str = eqx.field(static=True)
    transmission_monotonic_sign: int = eqx.field(static=True)
    boundary_policy: str = eqx.field(static=True)


class DetectorRaster(eqx.Module):
    """Store expected detector counts on native recorded coordinates.

    The carrier associates a nonempty channel axis with native detector and
    energy coordinates. Optional quantization vectors remain attached to
    their detector pixels without relabeling them as momentum coordinates.

    :see: :class:`~.test_bands.TestDetectorRaster`

    Attributes
    ----------
    expected_counts : Float64[Array, "n_channel n_u n_v n_e"]
        Nonnegative expected counts per native recorded bin.
    detector_u_axis : Float64[Array, " n_u"]
        Native detector ``T_x`` bin centres in radians.
    detector_v_axis : Float64[Array, " n_v"]
        Native detector ``T_y`` bin centres in radians. A slit has length one.
    energy_axis : Float64[Array, " n_e"]
        Recorded energy-bin centres relative to the Fermi level in eV.
    quantization_axis : Optional[Float64[Array, "n_u n_v 3"]]
        Optional per-pixel laboratory spin-quantization axes. A constant axis
        is explicitly broadcast over native detector pixels.
    channel_labels : Tuple[str, ...]
        **Static.** One unique nonempty label per leading channel. Changing
        the labels triggers retracing.
    coordinate_system : str
        **Static.** V1 is exactly ``"hemispherical_angles"``. Changing it
        triggers retracing.

    Notes
    -----
    Detector arrays are not relabeled Cartesian momentum grids. This carrier
    keeps native coordinates and expected-count units explicit.
    """

    expected_counts: Float64[Array, "n_channel n_u n_v n_e"]
    detector_u_axis: Float64[Array, " n_u"]
    detector_v_axis: Float64[Array, " n_v"]
    energy_axis: Float64[Array, " n_e"]
    quantization_axis: Optional[Float64[Array, "n_u n_v 3"]]
    channel_labels: Tuple[str, ...] = eqx.field(static=True)
    coordinate_system: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_band_structure(  # noqa: DOC503
    eigenvalues: Float64[Array, "Ke B"],
    kpoints: Float64[Array, "Kk 3"],
    kpoint_weights: Union[Float64[Array, " Kw"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> BandStructure:
    """Create a validated ``BandStructure`` instance.

    The factory validates and normalizes raw band-structure
    data before it constructs a ``BandStructure`` PyTree. The factory casts
    all input arrays to ``float64`` for numerical stability. Energy
    differences and Lorentzian broadening depend on precision. The factory
    replaces missing k-point
    weights with uniform weights. Callers therefore do not handle the common
    equal-weight case explicitly.

    ``@jaxtyped(typechecker=beartype)`` checks input shapes and dtypes at call
    time. This check finds different dimensions before the simulation uses
    them.

    :see: :class:`~.test_bands.TestMakeBandStructure`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           eigenvalues_arr = jnp.asarray(eigenvalues, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           kpoints_arr.shape[0] != nkpts

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(eigenvalues_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return bands

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    eigenvalues : Float64[Array, "Ke B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float64[Array, "Kk 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Union[Float64[Array, " Kw"], None], optional
        Integration weights. Defaults to uniform weights.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : BandStructure
        Validated band structure instance with all arrays in
        ``float64``.

    Raises
    ------
    ValueError
        If eigenvalues, k-points, and weights disagree on their k-point
        count.
    EquinoxRuntimeError
        If eigenvalues or k-points are non-finite, or weights are non-finite
        or negative.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    k-point or weight counts disagree with the eigenvalues. Traced validation
    uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for non-finite
    arrays or negative weights.
    """
    eigenvalues_arr: Float64[Array, "K B"] = jnp.asarray(
        eigenvalues, dtype=jnp.float64
    )
    kpoints_arr: Float64[Array, "K 3"] = jnp.asarray(
        kpoints, dtype=jnp.float64
    )
    nkpts: int = eigenvalues_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float64[Array, " K"] = jnp.ones(nkpts, dtype=jnp.float64)
    else:
        weights_arr = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    fermi_arr: Float64[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )

    if kpoints_arr.shape[0] != nkpts:
        raise ValueError(
            "make_band_structure: eigenvalues and kpoints disagree on K axis"
        )
    if weights_arr.shape[0] != nkpts:
        raise ValueError(
            "make_band_structure: eigenvalues and weights disagree on K axis"
        )

    def validate_and_create() -> BandStructure:
        nonlocal eigenvalues_arr, kpoints_arr, weights_arr
        eigenvalues_arr = eqx.error_if(
            eigenvalues_arr,
            ~(jnp.all(jnp.isfinite(eigenvalues_arr))),
            "make_band_structure: eigenvalues finite",
        )
        kpoints_arr = eqx.error_if(
            kpoints_arr,
            ~(jnp.all(jnp.isfinite(kpoints_arr))),
            "make_band_structure: kpoints finite",
        )
        weights_arr = eqx.error_if(
            weights_arr,
            ~(jnp.all(jnp.isfinite(weights_arr))),
            "make_band_structure: weights finite",
        )
        weights_arr = eqx.error_if(
            weights_arr,
            ~(jnp.all(weights_arr >= 0.0)),
            "make_band_structure: weights non negative",
        )
        validated_bands: BandStructure = BandStructure(
            eigenvalues=eigenvalues_arr,
            kpoints=kpoints_arr,
            kpoint_weights=weights_arr,
            fermi_energy=fermi_arr,
        )
        return validated_bands

    bands: BandStructure = validate_and_create()
    return bands


@jaxtyped(typechecker=beartype)
def make_orbital_projection(  # noqa: DOC503
    projections: Float64[Array, "Kp Bp Ap Op"],
    spin: Optional[Float64[Array, "Ks Bs As 6"]] = None,
    oam: Optional[Float64[Array, "Ko Bo Ao 3"]] = None,
) -> OrbitalProjection:
    """Create a validated ``OrbitalProjection`` instance.

    The factory validates and normalizes raw orbital
    projection data before constructing an ``OrbitalProjection``
    PyTree. The factory casts the mandatory ``projections`` array to
    ``float64``. It casts the optional ``spin`` and ``oam`` arrays only when
    they are present. Thus, ``None`` continues to identify calculations
    without spin-orbit coupling.

    ``@jaxtyped(typechecker=beartype)`` checks the shape constraints at call
    time. The K, B, and A dimensions must agree across all arrays.

    :see: :class:`~.test_bands.TestMakeOrbitalProjection`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           proj_arr = jnp.asarray(projections, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           proj_arr.shape[3] != N_ORBITALS

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(proj_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return orb_proj

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    projections : Float64[Array, "Kp Bp Ap Op"]
        Orbital projection weights.
    spin : Optional[Float64[Array, "Ks Bs As 6"]], optional
        Spin projections. Default is None.
    oam : Optional[Float64[Array, "Ko Bo Ao 3"]], optional
        Orbital angular momentum. Default is None.

    Returns
    -------
    orb_proj : OrbitalProjection
        Validated orbital projection instance with all non-None
        arrays in ``float64``.

    Raises
    ------
    ValueError
        If optional channel axes disagree with the projection axes or the
        projection orbital axis does not contain 9 columns.
    EquinoxRuntimeError
        If projections are non-finite or negative, or a present spin channel
        is non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the projection, spin, or OAM shapes violate their structural contract.
    Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for non-finite arrays or negative projections.
    """
    proj_arr: Float64[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Optional[Float64[Array, "K B A 6"]] = None
    if spin is not None:
        spin_arr = jnp.asarray(spin, dtype=jnp.float64)
    oam_arr: Optional[Float64[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)

    if proj_arr.shape[3] != N_ORBITALS:
        raise ValueError(
            "make_orbital_projection: projections must have "
            f"{N_ORBITALS} orbital columns"
        )
    if spin_arr is not None and proj_arr.shape[:3] != spin_arr.shape[:3]:
        raise ValueError(
            "make_orbital_projection: projections and spin axes disagree"
        )
    if oam_arr is not None and proj_arr.shape[:3] != oam_arr.shape[:3]:
        raise ValueError(
            "make_orbital_projection: projections and oam axes disagree"
        )

    def validate_and_create() -> OrbitalProjection:
        nonlocal proj_arr, spin_arr
        proj_arr = eqx.error_if(
            proj_arr,
            ~(jnp.all(jnp.isfinite(proj_arr))),
            "make_orbital_projection: projections finite",
        )
        proj_arr = eqx.error_if(
            proj_arr,
            ~(jnp.all(proj_arr >= 0.0)),
            "make_orbital_projection: projections non negative",
        )
        if spin_arr is not None:
            spin_arr = eqx.error_if(
                spin_arr,
                ~(jnp.all(jnp.isfinite(spin_arr))),
                "make_orbital_projection: spin finite",
            )
        validated_projection: OrbitalProjection = OrbitalProjection(
            projections=proj_arr,
            spin=spin_arr,
            oam=oam_arr,
        )
        return validated_projection

    orb_proj: OrbitalProjection = validate_and_create()
    return orb_proj


@jaxtyped(typechecker=beartype)
def make_arpes_cube(  # noqa: DOC503
    intensity: Float64[Array, "Kx Ky Ei"],
    kx_axis: Float64[Array, " Kxa"],
    ky_axis: Float64[Array, " Kya"],
    energy_axis: Float64[Array, " Ea"],
    cartesian_frame_id: str = _SAMPLE_CARTESIAN_FRAME_ID,
    provenance: str = "",
) -> ArpesCube:
    """Create a validated ``ArpesCube`` instance.

    The factory normalizes source arrays to float64 and binds finite,
    nonnegative, and monotone-axis checks to the returned carrier.

    :see: :class:`~.test_bands.TestMakeArpesCube`

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
        **Static.** Registered Plan-03 sample frame. Changing it triggers
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
        < _MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_arpes_cube: each axis requires at least two points"
        )
    if cartesian_frame_id != _SAMPLE_CARTESIAN_FRAME_ID:
        raise ValueError("make_arpes_cube: unknown Cartesian frame identifier")

    def validate_and_create() -> ArpesCube:
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
    cartesian_frame_id: str = _SAMPLE_CARTESIAN_FRAME_ID,
) -> ArpesSpectrum:
    """Create a validated ``ArpesSpectrum`` instance.

    The factory verifies array dimensions and checks cumulative distance
    against the complete Cartesian path. It preserves the registered sample
    frame as static metadata.

    :see: :class:`~.test_bands.TestMakeArpesSpectrum`

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
        **Static.** Registered Plan-03 sample frame. Changing it triggers
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
        energy_arr.shape[0] < _MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_arpes_spectrum: path cannot be empty and energy requires "
            "two points"
        )
    if cartesian_frame_id != _SAMPLE_CARTESIAN_FRAME_ID:
        raise ValueError(
            "make_arpes_spectrum: unknown Cartesian frame identifier"
        )

    def validate_and_create() -> ArpesSpectrum:
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
                rtol=_PATH_STEP_RTOL,
                atol=_PATH_STEP_ATOL_INV_ANG,
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


@jaxtyped(typechecker=beartype)
def make_detector_raster(  # noqa: DOC503
    expected_counts: Float64[Array, "C U V Ei"],
    detector_u_axis: Float64[Array, " Ua"],
    detector_v_axis: Float64[Array, " Va"],
    energy_axis: Float64[Array, " Ea"],
    channel_labels: Tuple[str, ...],
    coordinate_system: str,
    quantization_axis: Optional[Float64[Array, "Uq Vq 3"]] = None,
) -> DetectorRaster:
    """Create a validated ``DetectorRaster`` instance.

    The factory aligns expected counts with native coordinate axes and static
    channel labels. It also validates optional per-pixel quantization vectors.

    :see: :class:`~.test_bands.TestMakeDetectorRaster`

    Parameters
    ----------
    expected_counts : Float64[Array, "C U V Ei"]
        Nonnegative expected counts in native detector bins.
    detector_u_axis : Float64[Array, " Ua"]
        Native detector ``T_x`` bin centres in radians.
    detector_v_axis : Float64[Array, " Va"]
        Native detector ``T_y`` bin centres in radians.
    energy_axis : Float64[Array, " Ea"]
        Recorded energy-bin centres relative to the Fermi level in eV.
    channel_labels : Tuple[str, ...]
        **Static.** One unique nonempty label per channel. Changing it triggers
        retracing.
    coordinate_system : str
        **Static.** Must be ``"hemispherical_angles"``. Changing it triggers
        retracing.
    quantization_axis : Optional[Float64[Array, "Uq Vq 3"]], optional
        Per-pixel laboratory spin-quantization axes.

    Returns
    -------
    raster : DetectorRaster
        Validated native-coordinate expected-count raster.

    Raises
    ------
    ValueError
        If dimensions or labels disagree, the channel axis is empty, or the
        coordinate system is invalid.
    EquinoxRuntimeError
        If values are non-finite, counts are negative, or a multi-point axis
        is not strictly increasing.

    Notes
    -----
    Static shape checks reject an empty channel axis before numerical
    validation runs in eager or compiled execution.
    """
    counts_arr: Float64[Array, "C U V E"] = jnp.asarray(
        expected_counts, dtype=jnp.float64
    )
    u_arr: Float64[Array, " U"] = jnp.asarray(
        detector_u_axis, dtype=jnp.float64
    )
    v_arr: Float64[Array, " V"] = jnp.asarray(
        detector_v_axis, dtype=jnp.float64
    )
    energy_arr: Float64[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )
    quantization_arr: Optional[Float64[Array, "U V 3"]] = None
    if quantization_axis is not None:
        quantization_arr = jnp.asarray(quantization_axis, dtype=jnp.float64)
    if counts_arr.shape[1:] != (
        u_arr.shape[0],
        v_arr.shape[0],
        energy_arr.shape[0],
    ):
        raise ValueError("make_detector_raster: counts and axes disagree")
    if min(u_arr.shape[0], v_arr.shape[0], energy_arr.shape[0]) < 1:
        raise ValueError("make_detector_raster: axes cannot be empty")
    if counts_arr.shape[0] < 1:
        raise ValueError(
            "make_detector_raster: channel axis requires at least one channel"
        )
    if len(channel_labels) != counts_arr.shape[0]:
        raise ValueError("make_detector_raster: channel labels disagree")
    labels_invalid: bool = any(not label for label in channel_labels) or len(
        set(channel_labels)
    ) != len(channel_labels)
    if labels_invalid:
        raise ValueError(
            "make_detector_raster: channel labels must be unique and nonempty"
        )
    if (
        quantization_arr is not None
        and quantization_arr.shape[:2] != counts_arr.shape[1:3]
    ):
        raise ValueError(
            "make_detector_raster: quantization and detector axes disagree"
        )
    if coordinate_system != _DETECTOR_COORDINATE_SYSTEM:
        raise ValueError("make_detector_raster: unknown coordinate system")

    def validate_and_create() -> DetectorRaster:
        nonlocal counts_arr, energy_arr, quantization_arr, u_arr, v_arr
        counts_arr = eqx.error_if(
            counts_arr,
            ~jnp.all(jnp.isfinite(counts_arr)),
            "make_detector_raster: expected counts finite",
        )
        counts_arr = eqx.error_if(
            counts_arr,
            ~jnp.all(counts_arr >= 0.0),
            "make_detector_raster: expected counts non negative",
        )
        u_arr = eqx.error_if(
            u_arr,
            ~jnp.all(jnp.isfinite(u_arr)) | ~jnp.all(jnp.diff(u_arr) > 0.0),
            "make_detector_raster: u axis finite and strictly increasing",
        )
        v_arr = eqx.error_if(
            v_arr,
            ~jnp.all(jnp.isfinite(v_arr)) | ~jnp.all(jnp.diff(v_arr) > 0.0),
            "make_detector_raster: v axis finite and strictly increasing",
        )
        energy_arr = eqx.error_if(
            energy_arr,
            ~jnp.all(jnp.isfinite(energy_arr))
            | ~jnp.all(jnp.diff(energy_arr) > 0.0),
            "make_detector_raster: energy axis finite and strictly increasing",
        )
        if quantization_arr is not None:
            quantization_arr = eqx.error_if(
                quantization_arr,
                ~jnp.all(jnp.isfinite(quantization_arr)),
                "make_detector_raster: quantization axis finite",
            )
        validated_raster: DetectorRaster = DetectorRaster(
            expected_counts=counts_arr,
            detector_u_axis=u_arr,
            detector_v_axis=v_arr,
            energy_axis=energy_arr,
            quantization_axis=quantization_arr,
            channel_labels=channel_labels,
            coordinate_system=coordinate_system,
        )
        return validated_raster

    raster: DetectorRaster = validate_and_create()
    return raster


@jaxtyped(typechecker=beartype)
def make_detector_calibration(  # noqa: DOC503
    u_bin_edges: Float64[Array, " U"],
    v_bin_edges: Float64[Array, " V"],
    energy_bin_edges_ev: Float64[Array, " E"],
    psf_fwhm_u: ScalarFloat,
    psf_fwhm_v: ScalarFloat,
    psf_fwhm_energy_ev: ScalarFloat,
    transmission_reference_domain_ev: Float64[Array, " 2"],
    coordinate_system: str = _DETECTOR_COORDINATE_SYSTEM,
    transmission_monotonic_sign: int = 1,
    boundary_policy: str = _DETECTOR_BOUNDARY_POLICY,
) -> DetectorCalibration:
    """Create a validated ``DetectorCalibration`` instance.

    The factory normalizes detector edges and point-spread widths to float64.
    It binds monotonicity, positivity, and fixed-policy checks to one carrier.

    :see: :class:`~.test_bands.TestMakeDetectorCalibration`

    Parameters
    ----------
    u_bin_edges : Float64[Array, " U"]
        Native detector ``T_x`` bin edges in radians.
    v_bin_edges : Float64[Array, " V"]
        Native detector ``T_y`` bin edges in radians.
    energy_bin_edges_ev : Float64[Array, " E"]
        Recorded energy-bin edges relative to the Fermi level in eV.
    psf_fwhm_u : ScalarFloat
        Positive native ``u`` point-spread FWHM in radians.
    psf_fwhm_v : ScalarFloat
        Positive native ``v`` point-spread FWHM in radians.
    psf_fwhm_energy_ev : ScalarFloat
        Positive energy point-spread FWHM in eV.
    transmission_reference_domain_ev : Float64[Array, " 2"]
        Fixed true-kinetic-energy calibration domain in eV.
    coordinate_system : str, optional
        **Static.** Must be ``"hemispherical_angles"``. Changing it triggers
        retracing.
    transmission_monotonic_sign : int, optional
        **Static.** Exactly ``-1`` or ``1``. Changing it triggers retracing.
    boundary_policy : str, optional
        **Static.** Must be ``"loss"``. Changing it triggers retracing.

    Returns
    -------
    calibration : DetectorCalibration
        Validated detector calibration with float64 numerical leaves.

    Raises
    ------
    ValueError
        If an edge axis has fewer than two entries or static metadata is not
        supported.
    EquinoxRuntimeError
        If values are non-finite, edges or the transmission domain are not
        strictly increasing, or a point-spread width is not positive.

    Notes
    -----
    Static policy checks run before value-threaded Equinox validation of the
    numerical calibration leaves.
    """
    u_edges: Float64[Array, " U"] = jnp.asarray(u_bin_edges, dtype=jnp.float64)
    v_edges: Float64[Array, " V"] = jnp.asarray(v_bin_edges, dtype=jnp.float64)
    energy_edges: Float64[Array, " E"] = jnp.asarray(
        energy_bin_edges_ev, dtype=jnp.float64
    )
    fwhm_u: Float64[Array, ""] = jnp.asarray(psf_fwhm_u, dtype=jnp.float64)
    fwhm_v: Float64[Array, ""] = jnp.asarray(psf_fwhm_v, dtype=jnp.float64)
    fwhm_energy: Float64[Array, ""] = jnp.asarray(
        psf_fwhm_energy_ev, dtype=jnp.float64
    )
    transmission_domain: Float64[Array, " 2"] = jnp.asarray(
        transmission_reference_domain_ev, dtype=jnp.float64
    )
    if (
        min(u_edges.shape[0], v_edges.shape[0], energy_edges.shape[0])
        < _MIN_INTERPOLATION_AXIS_POINTS
    ):
        raise ValueError(
            "make_detector_calibration: bin edges require two points"
        )
    if coordinate_system != _DETECTOR_COORDINATE_SYSTEM:
        raise ValueError(
            "make_detector_calibration: unknown coordinate system"
        )
    if transmission_monotonic_sign not in (-1, 1):
        raise ValueError(
            "make_detector_calibration: transmission sign must be -1 or 1"
        )
    if boundary_policy != _DETECTOR_BOUNDARY_POLICY:
        raise ValueError(
            "make_detector_calibration: boundary policy must be loss"
        )

    def validate_and_create() -> DetectorCalibration:
        nonlocal energy_edges, fwhm_energy, fwhm_u, fwhm_v
        nonlocal transmission_domain, u_edges, v_edges
        u_edges = eqx.error_if(
            u_edges,
            ~jnp.all(jnp.isfinite(u_edges))
            | ~jnp.all(jnp.diff(u_edges) > 0.0),
            "make_detector_calibration: u edges must be finite/increasing",
        )
        v_edges = eqx.error_if(
            v_edges,
            ~jnp.all(jnp.isfinite(v_edges))
            | ~jnp.all(jnp.diff(v_edges) > 0.0),
            "make_detector_calibration: v edges must be finite/increasing",
        )
        energy_edges = eqx.error_if(
            energy_edges,
            ~jnp.all(jnp.isfinite(energy_edges))
            | ~jnp.all(jnp.diff(energy_edges) > 0.0),
            "make_detector_calibration: energy edges finite/increasing",
        )
        fwhm_u = eqx.error_if(
            fwhm_u,
            ~jnp.isfinite(fwhm_u) | (fwhm_u <= 0.0),
            "make_detector_calibration: u FWHM finite and positive",
        )
        fwhm_v = eqx.error_if(
            fwhm_v,
            ~jnp.isfinite(fwhm_v) | (fwhm_v <= 0.0),
            "make_detector_calibration: v FWHM finite and positive",
        )
        fwhm_energy = eqx.error_if(
            fwhm_energy,
            ~jnp.isfinite(fwhm_energy) | (fwhm_energy <= 0.0),
            "make_detector_calibration: energy FWHM finite and positive",
        )
        transmission_domain = eqx.error_if(
            transmission_domain,
            ~jnp.all(jnp.isfinite(transmission_domain))
            | ~(transmission_domain[1] > transmission_domain[0]),
            "make_detector_calibration: transmission domain increasing",
        )
        validated_calibration: DetectorCalibration = DetectorCalibration(
            u_bin_edges=u_edges,
            v_bin_edges=v_edges,
            energy_bin_edges_ev=energy_edges,
            psf_fwhm_u=fwhm_u,
            psf_fwhm_v=fwhm_v,
            psf_fwhm_energy_ev=fwhm_energy,
            transmission_reference_domain_ev=transmission_domain,
            coordinate_system=coordinate_system,
            transmission_monotonic_sign=transmission_monotonic_sign,
            boundary_policy=boundary_policy,
        )
        return validated_calibration

    calibration: DetectorCalibration = validate_and_create()
    return calibration


def _linear_bracket(
    axis: Float64[Array, " N"],
    query: Float64[Array, ""],
) -> Tuple[Array, Array, Float64[Array, ""]]:
    """PRIVATE: Return adjacent indices and a guarded linear weight.

    Parameters
    ----------
    axis : Float64[Array, " N"]
        Strictly increasing source coordinate.
    query : Float64[Array, ""]
        In-domain scalar query coordinate.

    Returns
    -------
    bracket : Tuple[Array, Array, Float64[Array, ""]]
        Lower index, upper index, and piecewise-linear upper weight.

    Notes
    -----
    Two ``where`` guards keep a finite denominator and explicitly set the
    weight to zero if malformed repeated coordinates reach this private seam.
    Public factories prevent that case.
    """
    upper: Array = jnp.clip(
        jnp.searchsorted(axis, query, side="right"),
        1,
        axis.size - 1,
    )
    lower: Array = upper - 1
    lower_value: Float64[Array, ""] = axis[lower]
    upper_value: Float64[Array, ""] = axis[upper]
    denominator: Float64[Array, ""] = upper_value - lower_value
    safe_denominator: Float64[Array, ""] = jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    weight: Float64[Array, ""] = (query - lower_value) / safe_denominator
    weight = jnp.where(denominator > 0.0, weight, 0.0)
    bracket: Tuple[Array, Array, Float64[Array, ""]] = (lower, upper, weight)
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

    :see: :class:`~.test_bands.TestSliceEdc`

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
    ix0: Array
    ix1: Array
    wx: Float64[Array, ""]
    iy0: Array
    iy1: Array
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

    :see: :class:`~.test_bands.TestSliceMdc`

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
    ie0: Array
    ie1: Array
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

    :see: :class:`~.test_bands.TestConstantEnergyMap`

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

    :see: :class:`~.test_bands.TestFermiSurfaceMap`

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
    "BandStructure",
    "DetectorCalibration",
    "DetectorRaster",
    "OrbitalProjection",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "constant_energy_map",
    "fermi_surface_map",
    "make_arpes_cube",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_detector_calibration",
    "make_detector_raster",
    "make_orbital_projection",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
    "slice_edc",
    "slice_mdc",
]
