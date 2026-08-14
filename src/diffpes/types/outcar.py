"""Define the VASP OUTCAR summary data structure.

Extended Summary
----------------
This module defines the carrier for scalar summary values from a VASP
OUTCAR file. The carrier keeps the Fermi energy and the electron count.
Parsers hand these values to band readers and charge-integration checks.

Routine Listings
----------------
:class:`OutcarData`
    Store scalar VASP OUTCAR summary values in a JAX PyTree.
:func:`make_outcar_data`
    Create a validated ``OutcarData`` instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarFloat


class OutcarData(eqx.Module):
    """Store scalar VASP OUTCAR summary values in a JAX PyTree.

    This type stores the self-consistent Fermi energy and the electron
    count from one OUTCAR file. Band readers consume the Fermi energy.
    Charge-integration checks consume the electron count.

    :see: :class:`~.test_outcar.TestOutcarData`

    Attributes
    ----------
    fermi_energy : Float64[Array, ""]
        Fermi energy in eV from the final electronic step.
        JAX-traced (differentiable).
    nelect : Float64[Array, ""]
        Number of valence electrons in the cell (``NELECT``).
        JAX-traced (differentiable).

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields. Equinox
    stores both fields as traced scalar leaves.

    See Also
    --------
    make_outcar_data : Factory function with validation and float64
        casting.
    """

    fermi_energy: Float64[Array, ""]
    nelect: Float64[Array, ""]


@jaxtyped(typechecker=beartype)
def make_outcar_data(  # noqa: DOC502, DOC503 -- traced Equinox guards.
    fermi_energy: ScalarFloat,
    nelect: ScalarFloat,
) -> OutcarData:
    """Create a validated ``OutcarData`` instance.

    The factory casts both scalars to ``float64``. It binds finiteness
    checks to the returned carrier. It rejects a nonpositive electron
    count.

    :see: :class:`~.test_outcar.TestMakeOutcarData`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           fermi_arr = jnp.asarray(fermi_energy, dtype=jnp.float64)

       This expression gives the validation steps a stable dtype.

    2. **Apply traced validation**::

           ~jnp.isfinite(fermi_arr)

       This predicate remains active during eager and compiled
       execution.

    3. **Return the named instance**::

           return summary

       The explicit name keeps the implementation and the Returns
       section synchronized.

    Parameters
    ----------
    fermi_energy : ScalarFloat
        Fermi energy in eV.
    nelect : ScalarFloat
        Number of valence electrons in the cell.

    Returns
    -------
    summary : OutcarData
        Validated OUTCAR summary with ``float64`` scalar leaves.

    Raises
    ------
    EquinoxRuntimeError
        If a value is non-finite or the electron count is not positive.

    Notes
    -----
    Value-threaded Equinox checks preserve the same numerical validation
    in eager and compiled execution.

    See Also
    --------
    OutcarData : The PyTree class constructed by this factory.
    """
    fermi_arr: Float64[Array, ""] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )
    nelect_arr: Float64[Array, ""] = jnp.asarray(nelect, dtype=jnp.float64)
    fermi_checked: Float64[Array, ""] = eqx.error_if(
        fermi_arr,
        ~jnp.isfinite(fermi_arr),
        "fermi_energy must be finite",
    )
    nelect_checked: Float64[Array, ""] = eqx.error_if(
        nelect_arr,
        ~jnp.isfinite(nelect_arr) | (nelect_arr <= 0.0),
        "nelect must be finite and positive",
    )
    summary: OutcarData = OutcarData(
        fermi_energy=fermi_checked,
        nelect=nelect_checked,
    )
    return summary


__all__: list[str] = [
    "OutcarData",
    "make_outcar_data",
]
