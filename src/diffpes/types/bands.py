"""Define band-structure and orbital-projection data structures.

Extended Summary
----------------
This module defines PyTree types for electronic band-structure data and
orbital-resolved projections from VASP calculations. These are
the primary inputs to all ARPES simulation functions.

Routine Listings
----------------
:class:`ArpesSpectrum`
    Store ARPES simulation output in a JAX PyTree.
:class:`BandStructure`
    Store electronic band-structure data in a JAX PyTree.
:class:`OrbitalProjection`
    Store orbital-resolved band projections in a JAX PyTree.
:class:`SpinBandStructure`
    Store spin-resolved electronic band-structure data in a JAX PyTree.
:class:`SpinOrbitalProjection`
    Store orbital projections with spin data in a JAX PyTree.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`make_band_structure`
    Create a validated ``BandStructure`` instance.
:func:`make_orbital_projection`
    Create a validated ``OrbitalProjection`` instance.
:func:`make_spin_band_structure`
    Create a validated ``SpinBandStructure`` instance.
:func:`make_spin_orbital_projection`
    Create a validated ``SpinOrbitalProjection`` instance.

Notes
-----
Orbital indexing convention (9 orbitals):
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``
matching VASP PROCAR output ordering.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarNumeric
from .constants import N_ORBITALS, N_SPIN_COMPONENTS


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


class ArpesSpectrum(eqx.Module):
    """Store ARPES simulation output in a JAX PyTree.

    This type stores an ARPES simulation result. The result contains a
    two-dimensional photoemission intensity map I(k, E) and its energy axis.
    The k-point dimension indexes the momentum-resolved detector
    channels, and the energy dimension indexes the binding-energy grid
    for the spectral function.

    This JAX-compatible PyTree lets ``jit``-compiled loss functions compare
    simulated spectra with experimental data. ``grad`` can differentiate
    these functions during parameter fitting. Both fields contain dense JAX
    arrays and no static auxiliary data. JAX can differentiate the complete
    object.


    :see: :class:`~.test_bands.TestArpesSpectrum`

    Attributes
    ----------
    intensity : Float64[Array, "K E"]
        Photoemission intensity for K k-points and E energies.
    energy_axis : Float64[Array, " E"]
        Energy axis values in eV.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree. Equinox
    derives the tree structure from the annotated fields; both fields
    are differentiable leaves.
    """

    intensity: Float64[Array, "K E"]
    energy_axis: Float64[Array, " E"]


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
def make_arpes_spectrum(  # noqa: DOC503
    intensity: Float64[Array, "K Ei"],
    energy_axis: Float64[Array, " Ea"],
) -> ArpesSpectrum:
    """Create a validated ``ArpesSpectrum`` instance.

    The factory validates and normalizes simulated ARPES
    data before it constructs an ``ArpesSpectrum`` PyTree. The factory casts
    both input arrays to ``float64``. Downstream loss functions therefore
    maintain double-precision accuracy.

    ``@jaxtyped(typechecker=beartype)`` checks the energy dimension *E* at call
    time. This dimension must agree between ``intensity`` and ``energy_axis``.

    :see: :class:`~.test_bands.TestMakeArpesSpectrum`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           intensity_arr = jnp.asarray(intensity, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           intensity_arr.shape[1] != energy_arr.shape[0]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(intensity_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return spectrum

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    intensity : Float64[Array, "K Ei"]
        Photoemission intensity map.
    energy_axis : Float64[Array, " Ea"]
        Energy axis values in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        Validated ARPES spectrum instance with all arrays in
        ``float64``.

    Raises
    ------
    ValueError
        If the intensity energy-axis length does not match ``energy_axis``.
    EquinoxRuntimeError
        If intensity is non-finite or the energy axis is not strictly
        increasing.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the two energy-axis lengths disagree. Traced validation uses
    ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for non-finite
    intensity or an energy axis that is not strictly increasing.
    """
    intensity_arr: Float64[Array, "K E"] = jnp.asarray(
        intensity, dtype=jnp.float64
    )
    energy_arr: Float64[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )

    if intensity_arr.shape[1] != energy_arr.shape[0]:
        raise ValueError(
            "make_arpes_spectrum: intensity and energy_axis disagree on E axis"
        )

    def validate_and_create() -> ArpesSpectrum:
        nonlocal energy_arr, intensity_arr
        intensity_arr = eqx.error_if(
            intensity_arr,
            ~(jnp.all(jnp.isfinite(intensity_arr))),
            "make_arpes_spectrum: intensity finite",
        )
        energy_arr = eqx.error_if(
            energy_arr,
            ~(jnp.all(jnp.diff(energy_arr) > 0.0)),
            "make_arpes_spectrum: energy axis strictly increasing",
        )
        validated_spectrum: ArpesSpectrum = ArpesSpectrum(
            intensity=intensity_arr,
            energy_axis=energy_arr,
        )
        return validated_spectrum

    spectrum: ArpesSpectrum = validate_and_create()
    return spectrum


__all__: list[str] = [
    "ArpesSpectrum",
    "BandStructure",
    "OrbitalProjection",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_orbital_projection",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
]
