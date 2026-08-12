"""Construct real-harmonic Slater--Koster hopping blocks.

Extended Summary
----------------
This module rotates Cartesian vector and tensor representations.
It does not use singular Euler angles.

Routine Listings
----------------
:func:`sk_block`
    Construct a real-harmonic Slater--Koster hopping block.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Bool, Float64, jaxtyped

from diffpes.constants import CARTESIAN_COMPONENTS, MAX_SK_ANGULAR_MOMENTUM


def _validate_angular_momentum(l: int, name: str) -> None:  # noqa: E741
    """PRIVATE: Validate one static Slater--Koster shell angular momentum.

    Parameters
    ----------
    l : int
        Candidate static shell angular momentum.
    name : str
        Argument name used in the error message.

    Raises
    ------
    ValueError
        If ``l`` is not an integer from zero through
        ``MAX_SK_ANGULAR_MOMENTUM``.

    Notes
    -----
    The check reads the static Python value, so an unsupported shell
    fails before any block construction. The implemented two-center
    kernel covers s, p, and d shells only.
    """
    if type(l) is not int or not 0 <= l <= MAX_SK_ANGULAR_MOMENTUM:
        message: str = (
            f"{name} must be an integer from 0 through "
            f"{MAX_SK_ANGULAR_MOMENTUM}"
        )
        raise ValueError(message)


def _rotation_z_to_direction(
    direction: Float64[Array, " 3"],
) -> Float64[Array, "3 3"]:
    """PRIVATE: Construct a proper rotation taking positive z onto the bond.

    Parameters
    ----------
    direction : Float64[Array, " 3"]
        Cartesian unit vector along the bond.

    Returns
    -------
    rotation : Float64[Array, "3 3"]
        Proper rotation matrix that maps ``(0, 0, 1)`` onto
        ``direction``.

    Notes
    -----
    Two smooth Rodrigues charts cover the bond sphere. The north chart
    rotates ``+z`` directly onto the direction and is singular only at
    the south pole. The south chart composes a half turn about x with a
    rotation from ``-z`` and is singular only at the north pole.
    ``jnp.where`` selects the chart by the sign of the z component and
    sanitizes the unused denominator. The value and the transverse
    position gradients therefore stay finite at both poles. The charts
    differ by a residual rotation about the bond axis. The axial
    degeneracy of the pi and delta channels makes the composed
    Slater--Koster block independent of that residual gauge.
    """
    dtype: jnp.dtype = direction.dtype
    identity: Float64[Array, "3 3"] = jnp.eye(
        CARTESIAN_COMPONENTS,
        dtype=dtype,
    )
    direction_x: Float64[Array, ""] = direction[0]
    direction_y: Float64[Array, ""] = direction[1]
    direction_z: Float64[Array, ""] = direction[2]
    use_north_chart: Bool[Array, ""] = direction_z >= 0.0

    north_denominator: Float64[Array, ""] = jnp.where(
        use_north_chart,
        1.0 + direction_z,
        1.0,
    )
    north_cross: Float64[Array, " 3"] = jnp.stack(
        (-direction_y, direction_x, jnp.zeros_like(direction_z))
    )
    north_skew: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -north_cross[2], north_cross[1]),
            (north_cross[2], 0.0, -north_cross[0]),
            (-north_cross[1], north_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    north_rotation: Float64[Array, "3 3"] = (
        identity + north_skew + north_skew @ north_skew / north_denominator
    )

    south_denominator: Float64[Array, ""] = jnp.where(
        use_north_chart,
        1.0,
        1.0 - direction_z,
    )
    south_cross: Float64[Array, " 3"] = jnp.stack(
        (direction_y, -direction_x, jnp.zeros_like(direction_z))
    )
    south_skew: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, -south_cross[2], south_cross[1]),
            (south_cross[2], 0.0, -south_cross[0]),
            (-south_cross[1], south_cross[0], 0.0),
        ),
        dtype=dtype,
    )
    half_turn_x: Float64[Array, "3 3"] = jnp.diag(
        jnp.asarray((1.0, -1.0, -1.0), dtype=dtype)
    )
    from_negative_z: Float64[Array, "3 3"] = (
        identity + south_skew + south_skew @ south_skew / south_denominator
    )
    south_rotation: Float64[Array, "3 3"] = from_negative_z @ half_turn_x
    rotation: Float64[Array, "3 3"] = jnp.where(
        use_north_chart,
        north_rotation,
        south_rotation,
    )
    return rotation


def _p_axes(dtype: jnp.dtype) -> Float64[Array, "3 3"]:
    """PRIVATE: Return Cartesian unit vectors in real-harmonic p-shell order.

    Parameters
    ----------
    dtype : jnp.dtype
        Floating dtype for the returned constant.

    Returns
    -------
    axes : Float64[Array, "3 3"]
        Rows ``(y, z, x)`` matching the package p-orbital order
        ``(p_y, p_z, p_x)``.

    Notes
    -----
    The rows are the change of basis between the Cartesian vector
    representation and the real p orbitals. A Cartesian rotation ``R``
    therefore acts on the shell as ``axes @ R @ axes.T``.
    """
    axes: Float64[Array, "3 3"] = jnp.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=dtype,
    )
    return axes


def _d_tensors(dtype: jnp.dtype) -> Float64[Array, "5 3 3"]:
    """PRIVATE: Return orthonormal traceless tensors in real d-shell order.

    Parameters
    ----------
    dtype : jnp.dtype
        Floating dtype for the returned constant.

    Returns
    -------
    tensors : Float64[Array, "5 3 3"]
        Symmetric traceless tensors for the package d-orbital order
        ``(d_xy, d_yz, d_z2, d_xz, d_x2-y2)``, orthonormal under the
        Frobenius inner product.

    Notes
    -----
    The ``1/sqrt(2)`` and ``1/sqrt(6)`` factors normalize every tensor
    to unit Frobenius norm. A Cartesian rotation acts on the d shell
    through the congruence ``R T R.T`` of each tensor followed by
    projection back onto this basis.
    """
    inverse_sqrt_two: float = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six: float = 1.0 / np.sqrt(6.0)
    tensors: Float64[Array, "5 3 3"] = jnp.asarray(
        (
            (
                (0.0, inverse_sqrt_two, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, inverse_sqrt_two, 0.0),
            ),
            (
                (-inverse_sqrt_six, 0.0, 0.0),
                (0.0, -inverse_sqrt_six, 0.0),
                (0.0, 0.0, 2.0 * inverse_sqrt_six),
            ),
            (
                (0.0, 0.0, inverse_sqrt_two),
                (0.0, 0.0, 0.0),
                (inverse_sqrt_two, 0.0, 0.0),
            ),
            (
                (inverse_sqrt_two, 0.0, 0.0),
                (0.0, -inverse_sqrt_two, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
        dtype=dtype,
    )
    return tensors


def _orbital_rotation(
    l: int,  # noqa: E741
    rotation: Float64[Array, "3 3"],
) -> Float64[Array, "m1 m2"]:
    """PRIVATE: Represent a Cartesian rotation in a real s, p, or d shell.

    Parameters
    ----------
    l : int
        Static shell angular momentum: ``0``, ``1``, or ``2``.
    rotation : Float64[Array, "3 3"]
        Proper Cartesian rotation matrix.

    Returns
    -------
    representation : Float64[Array, "m1 m2"]
        Orthogonal ``(2*l + 1)``-dimensional representation of the
        rotation in the package real-harmonic order.

    Notes
    -----
    The s shell returns the trivial one-by-one identity. The p shell
    conjugates the rotation with the real-harmonic axis rows. The d
    shell rotates every orthonormal traceless tensor by congruence and
    projects the result back onto the tensor basis. Rotating vector and
    tensor representations avoids differentiating singular Euler angles.
    """
    if l == 0:
        scalar: Float64[Array, "1 1"] = jnp.ones(
            (1, 1),
            dtype=rotation.dtype,
        )
        return scalar
    if l == 1:
        axes: Float64[Array, "3 3"] = _p_axes(rotation.dtype)
        vector_representation: Float64[Array, "3 3"] = axes @ rotation @ axes.T
        return vector_representation

    tensors: Float64[Array, "5 3 3"] = _d_tensors(rotation.dtype)
    rotated_tensors: Float64[Array, "5 3 3"] = jnp.einsum(
        "ik,akl,jl->aij",
        rotation,
        tensors,
        rotation,
    )
    tensor_representation: Float64[Array, "5 5"] = jnp.einsum(
        "bij,aij->ba",
        tensors,
        rotated_tensors,
    )
    return tensor_representation


@jaxtyped(typechecker=beartype)
def sk_block(  # noqa: DOC502, DOC503
    l1: int,
    l2: int,
    v_llm: Float64[Array, " n_m"],
    bond_cart: Float64[Array, " 3"],
) -> Float64[Array, "m1 m2"]:
    r"""Construct a real-harmonic Slater--Koster hopping block.

    Rotate the bond-axis sigma, pi, and delta channels into the declared
    laboratory-frame real-harmonic order.

    :see: :class:`~.test_slaterkoster.TestSkBlock`

    Parameters
    ----------
    l1 : int
        Angular momentum of the row shell: ``0`` (s), ``1`` (p), or ``2``
        (d).
    l2 : int
        Angular momentum of the column shell, with the same range.
    v_llm : Float64[Array, " n_m"]
        Fundamental integrals ordered by ``abs(m)``: sigma, pi, then delta.
        Its length must be ``min(l1, l2) + 1``.
    bond_cart : Float64[Array, " 3"]
        Nonzero Cartesian displacement from the row-shell atom to the
        column-shell atom.

    Returns
    -------
    block : Float64[Array, "m1 m2"]
        Hopping matrix with shape ``(2*l1 + 1, 2*l2 + 1)`` in the declared
        real-harmonic order.

    Raises
    ------
    ValueError
        If an angular momentum lies outside the implemented s/p/d range or
        the integral vector has the wrong rank or length.
    EquinoxRuntimeError
        If the bond is zero/non-finite or an integral is non-finite.

    Notes
    -----
    A bond-axis matrix couples equal magnetic numbers with
    ``v_llm[abs(m)]``. Cartesian vector/tensor representations rotate it into
    the laboratory real-harmonic basis. For a swapped shell order, the
    radial parity convention
    :math:`V_{l_2l_1m}=(-1)^{l_1+l_2}V_{l_1l_2m}` is applied automatically
    when ``l1 > l2``.
    """
    _validate_angular_momentum(l1, "l1")
    _validate_angular_momentum(l2, "l2")
    if v_llm.ndim != 1:
        message: str = "v_llm must be one-dimensional"
        raise ValueError(message)
    expected_integrals: int = min(l1, l2) + 1
    if v_llm.shape[0] != expected_integrals:
        message = (
            f"v_llm length must equal min(l1, l2) + 1 ({expected_integrals})"
        )
        raise ValueError(message)

    values: Float64[Array, " n_m"] = eqx.error_if(
        v_llm,
        ~jnp.all(jnp.isfinite(v_llm)),
        "sk_block: integrals finite",
    )
    bond: Float64[Array, " 3"] = eqx.error_if(
        bond_cart,
        ~jnp.all(jnp.isfinite(bond_cart)),
        "sk_block: bond finite",
    )
    bond_norm: Float64[Array, ""] = jnp.linalg.norm(bond)
    bond = eqx.error_if(
        bond,
        ~(bond_norm > 0.0),
        "sk_block: bond nonzero",
    )
    direction: Float64[Array, " 3"] = bond / bond_norm
    rotation: Float64[Array, "3 3"] = _rotation_z_to_direction(direction)
    left_rotation: Float64[Array, "m1 m1"] = _orbital_rotation(l1, rotation)
    right_rotation: Float64[Array, "m2 m2"] = _orbital_rotation(l2, rotation)

    bond_axis: Float64[Array, "m1 m2"] = jnp.zeros(
        (2 * l1 + 1, 2 * l2 + 1),
        dtype=values.dtype,
    )
    magnetic_number: int
    for magnetic_number in range(-min(l1, l2), min(l1, l2) + 1):
        bond_axis = bond_axis.at[
            magnetic_number + l1,
            magnetic_number + l2,
        ].set(values[abs(magnetic_number)])

    radial_parity: int = (-1) ** (l1 + l2) if l1 > l2 else 1
    block: Float64[Array, "m1 m2"] = (
        radial_parity * left_rotation @ bond_axis @ right_rotation.T
    )
    return block


__all__: list[str] = [
    "sk_block",
]
