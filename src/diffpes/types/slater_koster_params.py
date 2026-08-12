"""Define Slater--Koster two-center parameters.

Extended Summary
----------------
This module stores differentiable two-center integrals with their
static material and channel identifiers.

Routine Listings
----------------
:class:`SlaterKosterParams`
    Store differentiable Slater--Koster two-center integrals.
:func:`make_slater_koster_params`
    Create validated Slater--Koster two-center parameters.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, Float64, jaxtyped


def _validate_slater_koster_structure(
    values: Float64[Array, " n_sk"],
    keys: Tuple[str, ...],
) -> None:
    """PRIVATE: Validate Slater--Koster parameter axes and identifiers.

    Implementation Logic
    --------------------
    Check the traced axis only through ``ndim`` and ``shape`` so that no
    numerical value leaves the traced domain. Compare the key-set size
    against the tuple length to reject duplicates.

    Parameters
    ----------
    values : Float64[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV.
    keys : Tuple[str, ...]
        Static material/channel identifiers, one per value.

    Raises
    ------
    ValueError
        If ``values`` is not one-dimensional. If ``keys`` disagrees
        with ``values`` on length or contains invalid or duplicate
        strings. This is the static construction-time contract.
    """
    if values.ndim != 1:
        message: str = "SlaterKosterParams values must be one-dimensional"
        raise ValueError(message)
    if type(keys) is not tuple:
        message = "SlaterKosterParams keys must be a tuple"
        raise ValueError(message)
    if len(keys) != values.shape[0]:
        message = (
            "SlaterKosterParams values and keys must have the same length"
        )
        raise ValueError(message)
    if any(type(key) is not str or not key for key in keys):
        message = "SlaterKosterParams keys must contain non-empty strings"
        raise ValueError(message)
    if len(set(keys)) != len(keys):
        message = "SlaterKosterParams keys must be unique"
        raise ValueError(message)


class SlaterKosterParams(eqx.Module):
    """Store differentiable Slater--Koster two-center integrals.

    The numerical values are the flat-real optimization coordinates for a
    Slater--Koster material model. Their keys are static identifiers such as
    ``"C-C:pp_sigma"`` or ``"Ru-O:pd_pi"``. A key change alters the material
    topology and therefore triggers JAX retracing.

    :see: :class:`~.test_slater_koster_params.TestSlaterKosterParams`

    Attributes
    ----------
    values : Float64[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV. These values remain
        differentiable JAX leaves.
    keys : Tuple[str, ...]
        Unique material/channel identifiers (**static** -- changing them
        triggers retracing).

    Notes
    -----
    The carrier deliberately does not prescribe distance scaling. The
    Slater--Koster builder interprets the identifiers and assigns them to
    frozen neighbor shells.

    See Also
    --------
    make_slater_koster_params : Validating factory for this carrier.
    """

    values: Float64[Array, " n_sk"]
    keys: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the traced axis against the static key tuple."""
        _validate_slater_koster_structure(self.values, self.keys)


@jaxtyped(typechecker=beartype)
def make_slater_koster_params(  # noqa: DOC502, DOC503
    values: Float[Array, " n_sk"],
    keys: Tuple[str, ...],
) -> SlaterKosterParams:
    """Create validated Slater--Koster two-center parameters.

    The factory normalizes numerical values and validates every static channel
    identifier before constructing the carrier.

    :see: :class:`~.test_slater_koster_params.TestMakeSlaterKosterParams`

    Parameters
    ----------
    values : Float[Array, " n_sk"]
        Fundamental two-center hopping integrals in eV.
    keys : Tuple[str, ...]
        Unique static identifiers, one for every value. Material builders use
        identifiers such as ``"C-C:pp_sigma"``.

    Returns
    -------
    params : SlaterKosterParams
        Parameter carrier with float64 differentiable values and static keys.

    Raises
    ------
    ValueError
        If values are not one-dimensional, keys are not a tuple, lengths
        differ, or a key is empty or duplicated.
    EquinoxRuntimeError
        If any value is non-finite, in eager or compiled execution.

    Notes
    -----
    Values may have either sign and may be zero. Only finiteness is a
    numerical invariant; channel and material semantics belong to the
    Slater--Koster model builder.

    See Also
    --------
    SlaterKosterParams : Carrier constructed by this factory.
    """
    value_array: Float64[Array, " n_sk"] = jnp.asarray(
        values,
        dtype=jnp.float64,
    )
    _validate_slater_koster_structure(value_array, keys)
    value_array = eqx.error_if(
        value_array,
        ~jnp.all(jnp.isfinite(value_array)),
        "make_slater_koster_params: values finite",
    )
    params: SlaterKosterParams = SlaterKosterParams(
        values=value_array,
        keys=keys,
    )
    return params


__all__: list[str] = [
    "SlaterKosterParams",
    "make_slater_koster_params",
]
