"""Define certified radial quadrature and final-state profiles.

Extended Summary
----------------
This module stores immutable radial quadrature selections and the
certified final-state mode used by matrix-element calculations.

Routine Listings
----------------
:class:`FinalStateSpec`
    Store a certified radial final-state selection.
:class:`RadialQuadratureSpec`
    Store one immutable certified radial-quadrature profile.
:func:`make_final_state_spec`
    Create a validated radial final-state selection.
:func:`make_radial_quadrature_spec`
    Select one immutable certified quadrature profile.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped

from diffpes.constants import (
    CERTIFIED_RADIAL_PROFILES,
    FINAL_STATE_MODES,
    HERMITE_TABLE_POINTS,
    RADIAL_ACCELERATORS,
)


class RadialQuadratureSpec(eqx.Module):
    """Store one immutable certified radial-quadrature profile.

    Callers select a registered identity. They cannot self-assert numerical
    tolerances or enlarge its domain.

    :see: :class:`~.test_radial_profiles.TestRadialQuadratureSpec`

    Attributes
    ----------
    profile_id : str
        Registered profile identity (**static**).
    n_nodes : int
        Gauss--Legendre node count (**static**).
    r_max_bohr : float
        Certified radial cutoff in Bohr (**static**).
    k_max_bohr_inv : float
        Certified momentum limit in inverse Bohr (**static**).
    l_prime_max : int
        Certified final angular-momentum limit (**static**).
    value_rtol : float
        Registered value tolerance (**static**).
    gradient_rtol : float
        Registered derivative tolerance (**static**).
    tail_bound_method_id : str
        Registered tail-bound method (**static**).
    coefficient_condition_max : float
        Maximum certified normalized-contraction condition (**static**).
    min_decay_parameter : float
        Minimum certified exponential decay in inverse Bohr (**static**).
    max_decay_parameter : float
        Maximum certified exponential decay in inverse Bohr (**static**).

    See Also
    --------
    make_radial_quadrature_spec : Validated factory for this type.
    """

    profile_id: str = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    r_max_bohr: float = eqx.field(static=True)
    k_max_bohr_inv: float = eqx.field(static=True)
    l_prime_max: int = eqx.field(static=True)
    value_rtol: float = eqx.field(static=True)
    gradient_rtol: float = eqx.field(static=True)
    tail_bound_method_id: str = eqx.field(static=True)
    coefficient_condition_max: float = eqx.field(static=True)
    min_decay_parameter: float = eqx.field(static=True)
    max_decay_parameter: float = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Require exact agreement with the selected certified profile."""
        expected: (
            Tuple[
                int,
                float,
                float,
                int,
                float,
                float,
                str,
                float,
                float,
                float,
            ]
            | None
        ) = CERTIFIED_RADIAL_PROFILES.get(self.profile_id)
        actual: Tuple[
            int,
            float,
            float,
            int,
            float,
            float,
            str,
            float,
            float,
            float,
        ] = (
            self.n_nodes,
            self.r_max_bohr,
            self.k_max_bohr_inv,
            self.l_prime_max,
            self.value_rtol,
            self.gradient_rtol,
            self.tail_bound_method_id,
            self.coefficient_condition_max,
            self.min_decay_parameter,
            self.max_decay_parameter,
        )
        if expected is None or actual != expected:
            message: str = (
                "quadrature properties must match a certified profile"
            )
            raise ValueError(message)


class FinalStateSpec(eqx.Module):
    """Store a certified radial final-state selection.

    The numerical effective charge remains differentiable. Static mode and
    accelerator choices determine the compiled radial kernel.

    :see: :class:`~.test_radial_profiles.TestFinalStateSpec`

    Attributes
    ----------
    effective_charge : Float64[Array, ""]
        Coulomb effective charge in elementary-charge units.
    mode : str
        ``"plane_wave"`` or ``"coulomb"`` (**static**).
    radial_accelerator : str
        ``"direct"`` (**static**). The schema retains ``"hermite"`` for
        validation and raises because the frozen radial accelerator fails.
    table_n_points : int
        Registered Hermite table size (**static**).

    See Also
    --------
    make_final_state_spec : Validated factory for this type.
    """

    effective_charge: Float64[Array, ""]
    mode: str = eqx.field(static=True)
    radial_accelerator: str = eqx.field(static=True)
    table_n_points: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static final-state choices."""
        if self.mode not in FINAL_STATE_MODES:
            message: str = f"mode must be one of {FINAL_STATE_MODES}"
            raise ValueError(message)
        if self.radial_accelerator not in RADIAL_ACCELERATORS:
            message = (
                f"radial_accelerator must be one of {RADIAL_ACCELERATORS}"
            )
            raise ValueError(message)
        if self.radial_accelerator == "hermite":
            message = (
                "Hermite mode failed the frozen radial accelerator "
                "1025-to-2049 next-rung certification"
            )
            raise ValueError(message)
        if self.table_n_points not in HERMITE_TABLE_POINTS:
            message = f"table_n_points must be one of {HERMITE_TABLE_POINTS}"
            raise ValueError(message)
        if self.mode == "coulomb" and self.radial_accelerator != "direct":
            message = "coulomb final states require direct radial evaluation"
            raise ValueError(message)


@jaxtyped(typechecker=beartype)
def make_radial_quadrature_spec(
    profile_id: str = "gl1024-r120-k4-l9-v1",
) -> RadialQuadratureSpec:
    """Select one immutable certified quadrature profile.

    The profile identity resolves every numerical property. Callers cannot
    override tolerances or domain limits.

    :see: :class:`~.test_radial_profiles.TestMakeRadialQuadratureSpec`

    Notes
    -----
    Resolve every domain and tolerance field from the immutable profile map.

    Parameters
    ----------
    profile_id : str, optional
        Registered profile identity.

    Returns
    -------
    spec : RadialQuadratureSpec
        Immutable certified profile.

    Raises
    ------
    ValueError
        If ``profile_id`` is not registered.
    """
    profile: (
        Tuple[
            int,
            float,
            float,
            int,
            float,
            float,
            str,
            float,
            float,
            float,
        ]
        | None
    ) = CERTIFIED_RADIAL_PROFILES.get(profile_id)
    if profile is None:
        message: str = "unknown certified radial quadrature profile"
        raise ValueError(message)
    spec: RadialQuadratureSpec = RadialQuadratureSpec(
        profile_id=profile_id,
        n_nodes=profile[0],
        r_max_bohr=profile[1],
        k_max_bohr_inv=profile[2],
        l_prime_max=profile[3],
        value_rtol=profile[4],
        gradient_rtol=profile[5],
        tail_bound_method_id=profile[6],
        coefficient_condition_max=profile[7],
        min_decay_parameter=profile[8],
        max_decay_parameter=profile[9],
    )
    return spec


@jaxtyped(typechecker=beartype)
def make_final_state_spec(  # noqa: DOC503
    mode: str = "plane_wave",
    effective_charge: float | Float64[Array, ""] = 0.0,
    radial_accelerator: str = "direct",
    table_n_points: int = 257,
) -> FinalStateSpec:
    """Create a validated radial final-state selection.

    Plane waves require zero charge. All final states require direct radial
    evaluation because the frozen Hermite convergence criterion failed.

    :see: :class:`~.test_radial_profiles.TestMakeFinalStateSpec`

    Notes
    -----
    Validate static mode compatibility before checking the traced charge.

    Parameters
    ----------
    mode : str, optional
        ``"plane_wave"`` or ``"coulomb"``.
    effective_charge : float | Float64[Array, ""], optional
        Final-state effective charge.
    radial_accelerator : str, optional
        ``"direct"``. The factory recognizes ``"hermite"`` but raises.
    table_n_points : int, optional
        Registered Hermite table size.

    Returns
    -------
    spec : FinalStateSpec
        Validated final-state carrier.

    Raises
    ------
    ValueError
        When a static choice falls outside the registered options.
    EquinoxRuntimeError
        If the charge is non-finite or nonzero for a plane wave.
    """
    if mode not in FINAL_STATE_MODES:
        message: str = f"mode must be one of {FINAL_STATE_MODES}"
        raise ValueError(message)
    if radial_accelerator not in RADIAL_ACCELERATORS:
        message = f"radial_accelerator must be one of {RADIAL_ACCELERATORS}"
        raise ValueError(message)
    if radial_accelerator == "hermite":
        message = (
            "Hermite mode failed the frozen radial accelerator "
            "1025-to-2049 next-rung certification"
        )
        raise ValueError(message)
    if table_n_points not in HERMITE_TABLE_POINTS:
        message = f"table_n_points must be one of {HERMITE_TABLE_POINTS}"
        raise ValueError(message)
    if mode == "coulomb" and radial_accelerator != "direct":
        message = "coulomb final states require direct radial evaluation"
        raise ValueError(message)
    charge: Float64[Array, ""] = jnp.asarray(
        effective_charge,
        dtype=jnp.float64,
    )
    charge = eqx.error_if(
        charge,
        ~jnp.isfinite(charge),
        "effective_charge must be finite",
    )
    if mode == "plane_wave":
        charge = eqx.error_if(
            charge,
            charge != 0.0,
            "plane-wave final states require zero effective charge",
        )
    spec: FinalStateSpec = FinalStateSpec(
        effective_charge=charge,
        mode=mode,
        radial_accelerator=radial_accelerator,
        table_n_points=table_n_points,
    )
    return spec


__all__: list[str] = [
    "FinalStateSpec",
    "RadialQuadratureSpec",
    "make_final_state_spec",
    "make_radial_quadrature_spec",
]
