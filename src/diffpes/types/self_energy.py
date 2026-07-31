"""Define the causal self-energy model carrier.

Routine Listings
----------------
:class:`SelfEnergyModel`
    Store a causal self-energy parameterization as a JAX PyTree.
:func:`make_self_energy_model`
    Create a validated self-energy model.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat

_MODES: tuple[str, ...] = (
    "constant",
    "poly",
    "grid",
    "fermi_liquid",
    "bosonic_kink",
)
_MIN_GRID_NODES: int = 2
_TAIL_COORDINATE_BOUND: float = 30.0


class SelfEnergyModel(eqx.Module):
    """Store a causal self-energy parameterization as a JAX PyTree.

    The carrier stores unconstrained coordinates as differentiable leaves. It
    stores the selectors for the model and the tail as static fields.

    :see: :class:`~.test_self_energy.TestSelfEnergyModel`

    Attributes
    ----------
    coefficients : Float[Array, " n_coef"]
        Unconstrained raw coordinates. Physical positive parameters use a
        smooth softplus map without clipping.
    energy_nodes_rel_fermi_ev : Optional[Float[Array, " n_nodes"]]
        Strictly increasing grid nodes on the :math:`E-E_F` axis in eV.
    kk_domain_rel_fermi_ev : Optional[Float[Array, " 2"]]
        Finite numerical Kramers--Kronig domain ``[a, b]`` in eV.
    tail_coefficients : Optional[Float[Array, " n_tail"]]
        For ``power2``, the two raw delta-beta coordinates in ``[-30, 30]``.
    subtraction_point_rel_fermi_ev : Float[Array, ""]
        Fixed subtraction point on the relative-energy axis in eV.
    mode : str
        Model family. **Static** -- changing it triggers retracing.
    kk_consistent : bool
        Whether the model is causal. **Static** -- changing it retraces.
    tail_mode : str
        Tail contract. **Static** -- changing it triggers retracing.

    Notes
    -----
    Numerical fields are PyTree leaves. State selectors are static fields.
    Structural errors raise :class:`ValueError`; numerical predicates use
    :func:`equinox.error_if` and remain active in compiled code.
    """

    coefficients: Float[Array, "..."]
    energy_nodes_rel_fermi_ev: Optional[Float[Array, "..."]]
    kk_domain_rel_fermi_ev: Optional[Float[Array, "..."]]
    tail_coefficients: Optional[Float[Array, "..."]]
    subtraction_point_rel_fermi_ev: Float[Array, ""]
    mode: str = eqx.field(static=True)
    kk_consistent: bool = eqx.field(static=True)
    tail_mode: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate structural and traced carrier invariants."""
        self._check_structure()
        self._check_values()

    def _check_structure(self) -> None:
        """Reject invalid static carrier structure."""
        if self.mode not in _MODES:
            raise ValueError(
                f"mode must be one of {_MODES}, got {self.mode!r}"
            )
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must be a one-dimensional array")
        expected: dict[str, int | None] = {
            "constant": 1,
            "poly": None,
            "grid": None,
            "fermi_liquid": 3,
            "bosonic_kink": 4,
        }
        count: int | None = expected[self.mode]
        if self.coefficients.shape[0] < 1 or (
            count is not None and self.coefficients.shape[0] != count
        ):
            raise ValueError(
                f"coefficients have invalid length for mode={self.mode!r}"
            )

        self._check_node_structure()
        numerical: bool = self.mode in ("poly", "grid", "fermi_liquid")
        domain: Optional[Float[Array, "..."]] = self.kk_domain_rel_fermi_ev
        tail: Optional[Float[Array, "..."]] = self.tail_coefficients
        if numerical:
            if not self.kk_consistent or self.tail_mode != "power2":
                raise ValueError(
                    "energy-dependent numerical modes require "
                    "kk_consistent=True and tail_mode='power2'"
                )
            if domain is None or domain.ndim != 1 or domain.shape != (2,):
                raise ValueError("kk_domain_rel_fermi_ev must have shape (2,)")
            if tail is None or tail.ndim != 1 or tail.shape != (2,):
                raise ValueError(
                    "tail_coefficients must have exactly two entries"
                )
        elif self.mode == "constant":
            if (
                domain is not None
                or tail is not None
                or self.tail_mode != "none"
            ):
                raise ValueError(
                    "constant mode rejects KK domain and tail fields"
                )
        else:
            if not self.kk_consistent or self.tail_mode != "analytic":
                raise ValueError(
                    "bosonic_kink requires kk_consistent=True and "
                    "tail_mode='analytic'"
                )
            if domain is not None or tail is not None:
                raise ValueError(
                    "bosonic_kink rejects numerical KK domain and tail fields"
                )

    def _check_node_structure(self) -> None:
        """Reject node shapes incompatible with the selected mode."""
        nodes: Optional[Float[Array, "..."]] = self.energy_nodes_rel_fermi_ev
        if self.mode == "grid":
            if (
                nodes is None
                or nodes.ndim != 1
                or nodes.shape[0] < _MIN_GRID_NODES
            ):
                raise ValueError(
                    "energy_nodes_rel_fermi_ev must be a vector of length >= 2"
                )
            if nodes.shape[0] != self.coefficients.shape[0]:
                raise ValueError(
                    "energy_nodes_rel_fermi_ev and coefficients must have "
                    "the same length"
                )
        elif nodes is not None:
            raise ValueError(
                "energy_nodes_rel_fermi_ev is valid only for grid mode"
            )

    def _check_values(self) -> None:
        """Reject invalid numerical values eagerly and while traced."""
        nodes: Optional[Float[Array, "..."]] = self.energy_nodes_rel_fermi_ev
        domain: Optional[Float[Array, "..."]] = self.kk_domain_rel_fermi_ev
        tail: Optional[Float[Array, "..."]] = self.tail_coefficients
        object.__setattr__(
            self,
            "coefficients",
            eqx.error_if(
                self.coefficients,
                ~jnp.all(jnp.isfinite(self.coefficients)),
                "coefficients must be finite",
            ),
        )
        if nodes is not None:
            checked_nodes: Float[Array, "..."] = eqx.error_if(
                nodes,
                ~jnp.all(jnp.isfinite(nodes)),
                "energy nodes must be finite",
            )
            checked_nodes = eqx.error_if(
                checked_nodes,
                ~jnp.all(jnp.diff(checked_nodes) > 0.0),
                "energy nodes must be strictly increasing",
            )
            object.__setattr__(
                self, "energy_nodes_rel_fermi_ev", checked_nodes
            )
        if domain is not None:
            checked_domain: Float[Array, "..."] = eqx.error_if(
                domain,
                ~jnp.all(jnp.isfinite(domain)),
                "kk_domain_rel_fermi_ev must be finite",
            )
            checked_domain = eqx.error_if(
                checked_domain,
                ~(checked_domain[0] < checked_domain[1]),
                "kk_domain_rel_fermi_ev must be increasing",
            )
            if nodes is not None:
                checked_domain = eqx.error_if(
                    checked_domain,
                    ~jnp.all(
                        nodes[jnp.array([0, nodes.shape[0] - 1])]
                        == checked_domain
                    ),
                    "grid endpoints must exactly equal the KK domain",
                )
            subtraction: Float[Array, ""] = eqx.error_if(
                self.subtraction_point_rel_fermi_ev,
                ~jnp.isfinite(self.subtraction_point_rel_fermi_ev),
                "subtraction point must be finite",
            )
            subtraction = eqx.error_if(
                subtraction,
                (subtraction < checked_domain[0])
                | (subtraction > checked_domain[1]),
                "subtraction point must lie inside the KK domain",
            )
            object.__setattr__(self, "kk_domain_rel_fermi_ev", checked_domain)
            object.__setattr__(
                self, "subtraction_point_rel_fermi_ev", subtraction
            )
        else:
            object.__setattr__(
                self,
                "subtraction_point_rel_fermi_ev",
                eqx.error_if(
                    self.subtraction_point_rel_fermi_ev,
                    ~jnp.isfinite(self.subtraction_point_rel_fermi_ev),
                    "subtraction point must be finite",
                ),
            )
        if tail is not None:
            checked_tail: Float[Array, "..."] = eqx.error_if(
                tail,
                ~jnp.all(
                    jnp.isfinite(tail)
                    & (tail >= -_TAIL_COORDINATE_BOUND)
                    & (tail <= _TAIL_COORDINATE_BOUND)
                ),
                "tail_coefficients must be finite and inside [-30, 30]",
            )
            object.__setattr__(self, "tail_coefficients", checked_tail)


@jaxtyped(typechecker=beartype)
def make_self_energy_model(
    coefficients: Optional[Float[Array, "..."]] = None,
    gamma: ScalarFloat = 0.1,
    mode: str = "constant",
    energy_nodes_rel_fermi_ev: Optional[Float[Array, "..."]] = None,
    kk_consistent: bool = True,
    kk_domain_rel_fermi_ev: Optional[Float[Array, "..."]] = None,
    tail_coefficients: Optional[Float[Array, "..."]] = None,
    subtraction_point_rel_fermi_ev: ScalarFloat = 0.0,
    tail_mode: str = "none",
) -> SelfEnergyModel:
    """Create a validated self-energy model.

    The factory converts all numerical inputs to float64 arrays. It delegates
    the common structural and numerical checks to the carrier initializer.

    :see: :class:`~.test_self_energy.TestMakeSelfEnergyModel`

    Implementation Logic
    --------------------
    1. **Validate the static inputs**::

           if mode not in _MODES:
               raise ValueError(msg)

       The factory also validates the exclusive ``gamma`` shortcut.
    2. **Convert the numerical inputs**::

           coefficients_array = jnp.asarray(coefficients, dtype=jnp.float64)

       Float64 arrays preserve the numerical contract of the model.
    3. **Construct the carrier**::

           model = SelfEnergyModel(coefficients=coefficients_array, ...)

       The initializer applies the common structural and traced validation.

    Parameters
    ----------
    coefficients : Optional[Float[Array, " n_coef"]], optional
        Explicit unconstrained raw model coordinates.
    gamma : ScalarFloat, optional
        Positive finite constant linewidth shortcut in eV. Used only when
        ``coefficients`` is absent and ``mode='constant'``.
    mode : str, optional
        One of ``constant``, ``poly``, ``grid``, ``fermi_liquid``, or
        ``bosonic_kink``. **Static** -- changing it triggers retracing.
    energy_nodes_rel_fermi_ev : Optional[Float[Array, " n_nodes"]], optional
        Grid nodes on the :math:`E-E_F` axis in eV.
    kk_consistent : bool, optional
        Causality selector. **Static** -- changing it triggers retracing.
    kk_domain_rel_fermi_ev : Optional[Float[Array, " 2"]], optional
        Numerical KK domain ``[a, b]`` in relative-energy eV.
    tail_coefficients : Optional[Float[Array, " n_tail"]], optional
        Two bounded raw power-tail coordinates.
    subtraction_point_rel_fermi_ev : ScalarFloat, optional
        Subtraction point on the relative-energy axis in eV.
    tail_mode : str, optional
        ``none``, ``power2``, or ``analytic``. **Static** -- changing it
        triggers retracing.

    Returns
    -------
    model : SelfEnergyModel
        Validated self-energy model.

    Raises
    ------
    ValueError
        If static structure does not match the complete carrier state table.
    EquinoxRuntimeError
        If a traced numerical value violates its finite, ordering, domain, or
        tail-bound constraint.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if coefficients is None:
        if mode != "constant":
            raise ValueError(
                "coefficients are required unless mode='constant'"
            )
        gamma_array: Float[Array, ""] = jnp.asarray(gamma, dtype=jnp.float64)
        gamma_array = eqx.error_if(
            gamma_array,
            ~jnp.isfinite(gamma_array) | (gamma_array <= 0.0),
            "gamma must be finite and strictly positive",
        )
        # log(1-exp(-g)) + g is stable at both small and large positive g.
        coefficients_array: Float[Array, "..."] = jnp.atleast_1d(
            gamma_array + jnp.log(-jnp.expm1(-gamma_array))
        )
    else:
        coefficients_array = jnp.asarray(coefficients, dtype=jnp.float64)
    model: SelfEnergyModel = SelfEnergyModel(
        coefficients=coefficients_array,
        energy_nodes_rel_fermi_ev=(
            None
            if energy_nodes_rel_fermi_ev is None
            else jnp.asarray(energy_nodes_rel_fermi_ev, dtype=jnp.float64)
        ),
        kk_domain_rel_fermi_ev=(
            None
            if kk_domain_rel_fermi_ev is None
            else jnp.asarray(kk_domain_rel_fermi_ev, dtype=jnp.float64)
        ),
        tail_coefficients=(
            None
            if tail_coefficients is None
            else jnp.asarray(tail_coefficients, dtype=jnp.float64)
        ),
        subtraction_point_rel_fermi_ev=jnp.asarray(
            subtraction_point_rel_fermi_ev, dtype=jnp.float64
        ),
        mode=mode,
        kk_consistent=kk_consistent,
        tail_mode=tail_mode,
    )
    return model


__all__: list[str] = ["SelfEnergyModel", "make_self_energy_model"]
