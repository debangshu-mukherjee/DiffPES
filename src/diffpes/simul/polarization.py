"""Compute photon polarization and explicit frame transformations.

Extended Summary
----------------
The module computes complex polarization vectors from photon geometry and
converts them to the spherical basis. A fixed laboratory photon field maps
to sample coordinates through the inverse sample orientation only. Detector
rotations instead map emission directions and detector-fixed spin axes.

Routine Listings
----------------
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`detector_rotation`
    Build the detector-frame rotation.
:func:`detector_axis_to_sample`
    Convert a detector-fixed axis to sample coordinates.
:func:`lab_polarization_to_sample`
    Convert fixed laboratory polarization to sample coordinates.
:func:`photon_wavevector`
    Build the unit photon wavevector from incidence angles.
:func:`polarization_from_angles`
    Construct polarization from incidence angles.
:func:`polarization_to_spherical`
    Convert Cartesian polarization to spherical components.
:func:`rotate_frame_vectors`
    Rotate a detector-fixed real axis across a detector-angle grid.
:func:`sample_azimuth_rotation`
    Build the active sample-to-laboratory azimuth rotation.

Notes
-----
The horizontal detector frame uses ``Rx(ty) @ Ry(tx)``. DiffPES maps the
Chinook ``tilt.k_mesh`` angles as ``T=-tx, P=ty``. The vertical frame uses
``Rx(tx) @ Ry(ty)`` and maps ``T=-ty, P=tx``. These detector rotations do
not act on the fixed photon beam. If ``S`` maps sample components into
laboratory components and ``D`` maps detector components into laboratory
components, the binding frame equations are ``epsilon_sample = S.T @
epsilon_lab`` and ``axis_sample = S.T @ D @ axis_detector``.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.maths import rodrigues_rotation
from diffpes.types import ScalarFloat


@jaxtyped(typechecker=beartype)
def build_polarization_vectors(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """Construct s- and p-polarization basis vectors.

    The function constructs an orthonormal pair of polarization vectors from
    the photon incidence angles. The s-polarization is perpendicular to the
    incidence plane. The p-polarization is in the incidence plane and
    perpendicular to the wavevector.

    :see: :class:`~.test_polarization.TestBuildPolarizationVectors`

    Implementation Logic
    --------------------
    1. **Construct the s-polarization vector**::

           e_s = [sin(phi), -cos(phi), 0]

       This closed form is perpendicular to the incidence plane. It is the
       normalized ``k cross z`` convention continued to normal incidence.

    2. **Construct the p-polarization vector**::

           e_p = [-cos(theta) cos(phi),
                  -cos(theta) sin(phi),
                   sin(theta)]

       This vector equals ``k_photon cross e_s`` for the incoming propagation
       direction. It completes a right-handed transverse basis with
       ``e_s cross e_p = k_photon``.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    e_s : Float[Array, " 3"]
        s-polarization unit vector (perpendicular to
        incidence plane).
    e_p : Float[Array, " 3"]
        p-polarization unit vector (in incidence plane,
        perpendicular to photon wavevector).

    Notes
    -----
    The direct trigonometric form has no artificial collinearity threshold.
    At normal incidence, ``phi`` fixes the otherwise free transverse-frame
    gauge. The basis is smooth in both input angles for that gauge choice.
    """
    e_s: Float[Array, " 3"] = jnp.array(
        [
            jnp.sin(phi),
            -jnp.cos(phi),
            jnp.zeros_like(jnp.asarray(phi)),
        ],
        dtype=jnp.float64,
    )
    e_p: Float[Array, " 3"] = jnp.array(
        [
            -jnp.cos(theta) * jnp.cos(phi),
            -jnp.cos(theta) * jnp.sin(phi),
            jnp.sin(theta),
        ],
        dtype=jnp.float64,
    )
    polarization_vectors: Tuple[Float[Array, " 3"], Float[Array, " 3"]] = (
        e_s,
        e_p,
    )
    return polarization_vectors


@jaxtyped(typechecker=beartype)
def photon_wavevector(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Float[Array, " 3"]:
    """Build the unit photon wavevector from incidence angles.

    The function constructs the incoming unit photon propagation vector from
    spherical incidence angles. Theta starts at the outward surface normal,
    and phi is the azimuthal angle. The propagation vector points toward the
    sample. Spin-orbit ARPES simulations use it in the S·k_photon correction
    for circular dichroism.

    :see: :class:`~.test_polarization.TestPhotonWavevector`

    Notes
    -----
    Form the source-pointing spherical vector and negate it to obtain incoming
    propagation. Normalize the result and return it as ``k_hat``.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    k_photon : Float[Array, " 3"]
        Incoming unit wavevector in Cartesian coordinates.

    See Also
    --------
    build_polarization_vectors : Build the same k for the s-polarization and
        p-polarization basis. Use this function only for the propagation
        direction.
    """
    k: Float[Array, " 3"] = -jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        dtype=jnp.float64,
    )
    k_hat: Float[Array, " 3"] = k / jnp.linalg.norm(k)
    return k_hat


@jaxtyped(typechecker=beartype)
def polarization_from_angles(
    incidence_theta: ScalarFloat,
    incidence_phi: ScalarFloat,
    kind: str,
    polarization_angle: ScalarFloat = 0.0,
) -> Complex[Array, " 3"]:
    """Construct polarization from incidence angles.

    The function returns an explicit complex Cartesian vector for a standard
    polarization state. The incidence angles use the laboratory frame.

    :see: :class:`~.test_polarization.TestPolarizationFromAngles`

    Implementation Logic
    --------------------
    1. **Construct the transverse basis**::

           e_s, e_p = build_polarization_vectors(theta, phi)

       The basis is orthonormal and perpendicular to the photon direction.

    2. **Select the requested state**::

           polarization = coefficients[0] * e_s + coefficients[1] * e_p

       The static selector chooses s, p, circular, or linear coefficients.

    3. **Return the complex vector**::

           return polarization

       The result retains phase information for later coherent contraction.

    Parameters
    ----------
    incidence_theta : ScalarFloat
        Photon angle from the surface normal in radians.
    incidence_phi : ScalarFloat
        Photon azimuth in radians.
    kind : str
        Polarization kind (**static**). Use ``"s"``, ``"p"``, ``"c+"``,
        ``"c-"``, or ``"linear"``.
    polarization_angle : ScalarFloat, optional
        Linear-basis angle in radians. Default is 0.0.

    Returns
    -------
    polarization : Complex[Array, " 3"]
        Unit polarization vector in the laboratory frame.

    Raises
    ------
    ValueError
        If ``kind`` is not a supported polarization kind.

    Notes
    -----
    The ``kind`` value is static and selects a Python branch before tracing.
    JAX differentiates the result with respect to all angle arguments.
    With incoming propagation ``q`` and the right-handed basis
    ``e_s cross e_p = q``, ``"c+"`` has eigenvalue ``+1`` under
    ``i q cross``. The ``"c-"`` state has eigenvalue ``-1``. These algebraic
    labels avoid observer-dependent circular-polarization names.

    See Also
    --------
    build_polarization_vectors : Construct the transverse real basis.
    polarization_to_spherical : Convert the result to spherical components.
    """
    if kind not in ("s", "p", "c+", "c-", "linear"):
        msg: str = (
            "polarization_from_angles: kind must be one of "
            "('s', 'p', 'c+', 'c-', 'linear')"
        )
        raise ValueError(msg)

    e_s: Float[Array, " 3"]
    e_p: Float[Array, " 3"]
    e_s, e_p = build_polarization_vectors(incidence_theta, incidence_phi)
    e_s_complex: Complex[Array, " 3"] = e_s.astype(jnp.complex128)
    e_p_complex: Complex[Array, " 3"] = e_p.astype(jnp.complex128)
    if kind == "s":
        polarization: Complex[Array, " 3"] = e_s_complex
    elif kind == "p":
        polarization = e_p_complex
    elif kind == "c+":
        polarization = (e_s_complex + 1j * e_p_complex) / jnp.sqrt(2.0)
    elif kind == "c-":
        polarization = (e_s_complex - 1j * e_p_complex) / jnp.sqrt(2.0)
    else:
        angle: Float[Array, " "] = jnp.asarray(
            polarization_angle,
            dtype=jnp.float64,
        )
        polarization = (
            jnp.cos(angle) * e_s_complex + jnp.sin(angle) * e_p_complex
        )
    return polarization


@jaxtyped(typechecker=beartype)
def polarization_to_spherical(
    polarization: Complex[Array, " 3"],
) -> Complex[Array, " 3"]:
    """Convert Cartesian polarization to spherical components.

    The result uses component order ``(q=-1, q=0, q=+1)`` and the
    Condon-Shortley phase convention.

    :see: :class:`~.test_polarization.TestPolarizationToSpherical`

    Implementation Logic
    --------------------
    1. **Read Cartesian components**::

           epsilon_x, epsilon_y, epsilon_z = polarization

       The input remains complex so the operation preserves optical phase.

    2. **Apply the spherical-basis transform**::

           epsilon_minus = (epsilon_x - 1j * epsilon_y) / sqrt(2)

       The transform follows the registered Condon-Shortley convention.

    3. **Stack the ordered result**::

           spherical = stack((epsilon_minus, epsilon_z, epsilon_plus))

       This order matches the transitions ``q = (-1, 0, +1)``.

    Parameters
    ----------
    polarization : Complex[Array, " 3"]
        Cartesian polarization vector.

    Returns
    -------
    spherical : Complex[Array, " 3"]
        Spherical components in ``(q=-1, q=0, q=+1)`` order.

    Notes
    -----
    The transform is complex-linear. It preserves the squared vector norm and
    supports JVP, VJP, and complex-step checks without a conjugation.
    """
    epsilon_x: Complex[Array, " "] = polarization[0]
    epsilon_y: Complex[Array, " "] = polarization[1]
    epsilon_z: Complex[Array, " "] = polarization[2]
    root_two: Float[Array, " "] = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    epsilon_minus: Complex[Array, " "] = (
        epsilon_x - 1j * epsilon_y
    ) / root_two
    epsilon_plus: Complex[Array, " "] = (
        -(epsilon_x + 1j * epsilon_y) / root_two
    )
    spherical: Complex[Array, " 3"] = jnp.stack(
        (epsilon_minus, epsilon_z, epsilon_plus)
    )
    return spherical


@jaxtyped(typechecker=beartype)
def detector_rotation(
    tx: ScalarFloat,
    ty: ScalarFloat,
    slit: str,
) -> Float[Array, "3 3"]:
    """Build the detector-frame rotation.

    The function composes the two analyzer-angle rotations in the order set
    by the static slit orientation.

    :see: :class:`~.test_polarization.TestDetectorRotation`

    Implementation Logic
    --------------------
    1. **Build the Cartesian axis rotations**::

           rotation_x = rodrigues_rotation(x_axis, ty)

       Rodrigues matrices retain derivatives with respect to both angles.

    2. **Compose the slit convention**::

           horizontal = rotation_x_ty @ rotation_y_tx
           vertical = rotation_x_tx @ rotation_y_ty

       Horizontal and vertical slits use the registered composition orders.

    Parameters
    ----------
    tx : ScalarFloat
        First detector angle in radians.
    ty : ScalarFloat
        Second detector angle in radians.
    slit : str
        Slit orientation (**static**). Use ``"H"`` or ``"V"``.

    Returns
    -------
    rotation : Float[Array, "3 3"]
        Proper rotation from the reference detector frame.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    ``"H"`` uses ``R_x(ty) R_y(tx)``. ``"V"`` uses
    ``R_x(tx) R_y(ty)``. The matrix rotates the emitted direction and
    detector-fixed axes into the laboratory frame. Do not apply it to a fixed
    laboratory photon polarization.
    """
    if slit not in ("H", "V"):
        msg: str = "detector_rotation: slit must be 'H' or 'V'"
        raise ValueError(msg)
    x_axis: Float[Array, " 3"] = jnp.asarray(
        [1.0, 0.0, 0.0],
        dtype=jnp.float64,
    )
    y_axis: Float[Array, " 3"] = jnp.asarray(
        [0.0, 1.0, 0.0],
        dtype=jnp.float64,
    )
    if slit == "H":
        rotation_y_tx: Float[Array, "3 3"] = rodrigues_rotation(y_axis, tx)
        rotation_x_ty: Float[Array, "3 3"] = rodrigues_rotation(x_axis, ty)
        rotation: Float[Array, "3 3"] = rotation_x_ty @ rotation_y_tx
    else:
        rotation_x_tx: Float[Array, "3 3"] = rodrigues_rotation(x_axis, tx)
        rotation_y_ty: Float[Array, "3 3"] = rodrigues_rotation(y_axis, ty)
        rotation = rotation_x_tx @ rotation_y_ty
    return rotation


@jaxtyped(typechecker=beartype)
def sample_azimuth_rotation(
    sample_azimuth: ScalarFloat,
) -> Float[Array, "3 3"]:
    """Build the active sample-to-laboratory azimuth rotation.

    The sample orientation is a right-handed rotation about the common
    surface normal. Its transpose maps laboratory components into sample
    components.

    :see: :class:`~.test_polarization.TestSampleAzimuthRotation`

    Parameters
    ----------
    sample_azimuth : ScalarFloat
        Sample azimuth in radians.

    Returns
    -------
    sample_orientation : Float[Array, "3 3"]
        Proper rotation mapping sample components to laboratory components.

    Notes
    -----
    A laboratory vector ``v_lab`` therefore has sample components
    ``sample_azimuth_rotation(phi).T @ v_lab``. This matches the negative
    azimuth used by the ARPES k-mesh builders.
    """
    azimuth: Float[Array, ""] = jnp.asarray(
        sample_azimuth,
        dtype=jnp.float64,
    )
    cosine: Float[Array, ""] = jnp.cos(azimuth)
    sine: Float[Array, ""] = jnp.sin(azimuth)
    zero: Float[Array, ""] = jnp.zeros_like(azimuth)
    one: Float[Array, ""] = jnp.ones_like(azimuth)
    sample_orientation: Float[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((cosine, -sine, zero)),
            jnp.stack((sine, cosine, zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    return sample_orientation


@jaxtyped(typechecker=beartype)
def lab_polarization_to_sample(
    polarization_lab: Complex[Array, " 3"],
    sample_orientation: Float[Array, "3 3"],
) -> Complex[Array, " 3"]:
    """Convert fixed laboratory polarization to sample coordinates.

    The function applies only the inverse sample orientation. It leaves the
    physical beam independent of detector coordinates.

    :see: :class:`~.test_polarization.TestLabPolarizationToSample`

    Parameters
    ----------
    polarization_lab : Complex[Array, " 3"]
        Complex photon polarization in laboratory coordinates.
    sample_orientation : Float[Array, "3 3"]
        Active rotation mapping sample components to laboratory components.

    Returns
    -------
    polarization_sample : Complex[Array, " 3"]
        The same physical photon field in sample coordinates.

    Notes
    -----
    The mapping is ``sample_orientation.T @ polarization_lab``. No detector
    angle enters. The incident beam stays fixed across detector pixels. The
    operation is complex-linear and preserves optical phase.
    """
    polarization_sample: Complex[Array, " 3"] = (
        sample_orientation.T @ polarization_lab
    )
    return polarization_sample


@jaxtyped(typechecker=beartype)
def detector_axis_to_sample(
    axis_detector: Float[Array, " 3"],
    detector_orientation: Float[Array, "3 3"],
    sample_orientation: Float[Array, "3 3"],
) -> Float[Array, " 3"]:
    """Convert a detector-fixed axis to sample coordinates.

    The function composes detector-to-laboratory orientation with the inverse
    sample orientation.

    :see: :class:`~.test_polarization.TestDetectorAxisToSample`

    Parameters
    ----------
    axis_detector : Float[Array, " 3"]
        Real axis expressed in detector coordinates.
    detector_orientation : Float[Array, "3 3"]
        Active rotation mapping detector components to laboratory components.
    sample_orientation : Float[Array, "3 3"]
        Active rotation mapping sample components to laboratory components.

    Returns
    -------
    axis_sample : Float[Array, " 3"]
        Detector-fixed axis expressed in sample coordinates.

    Notes
    -----
    The binding composition is ``sample_orientation.T @
    detector_orientation @ axis_detector``. This composition is appropriate
    for analyzer spin axes, not for the fixed photon polarization.
    """
    axis_sample: Float[Array, " 3"] = (
        sample_orientation.T @ detector_orientation @ axis_detector
    )
    return axis_sample


@jaxtyped(typechecker=beartype)
def rotate_frame_vectors(
    vector: Float[Array, " 3"],
    tx: Float[Array, " n_tx"],
    ty: Float[Array, " n_ty"],
    slit: str,
    sample_azimuth: ScalarFloat = 0.0,
) -> Float[Array, "n_tx n_ty 3"]:
    """Rotate a detector-fixed real axis across a detector-angle grid.

    The function composes each detector orientation with the inverse sample
    orientation. It preserves both detector axes in the output.

    :see: :class:`~.test_polarization.TestRotateFrameVectors`

    Implementation Logic
    --------------------
    1. **Map over both angle axes**::

           rotations = vmap(vmap(detector_rotation))(tx, ty)

       Nested mapping builds one rotation for every detector coordinate.

    2. **Apply the detector/sample composition**::

           rotated = sample_orientation.T @ rotations @ vector

       Matrix multiplication preserves the vector norm.

    Parameters
    ----------
    vector : Float[Array, " 3"]
        Real axis fixed in the detector frame.
    tx : Float[Array, " n_tx"]
        First detector-angle axis in radians.
    ty : Float[Array, " n_ty"]
        Second detector-angle axis in radians.
    slit : str
        Slit orientation (**static**). Use ``"H"`` or ``"V"``.
    sample_azimuth : ScalarFloat, optional
        Sample azimuth in radians. Default is 0.0.

    Returns
    -------
    rotated : Float[Array, "n_tx n_ty 3"]
        Detector-fixed axis in sample coordinates at each detector point.

    Notes
    -----
    The output has fixed shape for fixed angle-axis lengths. JAX can compile
    and differentiate the two mapped detector axes and sample azimuth without
    Python data loops. Transform laboratory-fixed vectors directly with
    :func:`lab_polarization_to_sample`, not with this grid function.
    """
    sample_orientation: Float[Array, "3 3"] = sample_azimuth_rotation(
        sample_azimuth
    )

    def rotate_one_tx(
        tx_value: Float[Array, " "],
    ) -> Float[Array, "n_ty 3"]:
        """Rotate one vector across the second angle axis.

        Parameters
        ----------
        tx_value : Float[Array, " "]
            Fixed first detector angle in radians.

        Returns
        -------
        rotated_row : Float[Array, "n_ty 3"]
            Rotated vectors for the second angle axis.
        """

        def rotate_one_ty(
            ty_value: Float[Array, " "],
        ) -> Float[Array, " 3"]:
            """Rotate one vector at one detector coordinate.

            Parameters
            ----------
            ty_value : Float[Array, " "]
                Second detector angle in radians.

            Returns
            -------
            rotated_vector : Float[Array, " 3"]
                Rotated vector at the detector coordinate.
            """
            rotation: Float[Array, "3 3"] = detector_rotation(
                tx_value,
                ty_value,
                slit,
            )
            rotated_vector: Float[Array, " 3"] = detector_axis_to_sample(
                vector,
                rotation,
                sample_orientation,
            )
            return rotated_vector

        rotated_row: Float[Array, "n_ty 3"] = jax.vmap(rotate_one_ty)(ty)
        return rotated_row

    rotated: Float[Array, "n_tx n_ty 3"] = jax.vmap(rotate_one_tx)(tx)
    return rotated


__all__: list[str] = [
    "build_polarization_vectors",
    "detector_axis_to_sample",
    "detector_rotation",
    "lab_polarization_to_sample",
    "photon_wavevector",
    "polarization_from_angles",
    "polarization_to_spherical",
    "rotate_frame_vectors",
    "sample_azimuth_rotation",
]
