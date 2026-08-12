"""Define the typed factorized-photocurrent model boundary.

Extended Summary
----------------
The factorized model binds radial, final-state, self-energy, and static
execution choices while leaving electronic-state data behind a capability
protocol.  Its sole observable is intrinsic scalar intensity; detector
response remains a later operation.

Routine Listings
----------------
:class:`FactorizedArpesModel`
    Bind factorized photocurrent physics to a stable model identity.
:func:`make_factorized_arpes_model`
    Construct a validated factorized-photocurrent model.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped

from .radial_params import MatrixElementParams, RadialSpec
from .radial_profiles import FinalStateSpec, RadialQuadratureSpec
from .self_energy import SelfEnergyModel


class FactorizedArpesModel(eqx.Module):
    """Bind factorized photocurrent physics to a stable model identity.

    Attributes
    ----------
    radial_spec : RadialSpec
        Differentiable radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Differentiable dipole matrix-element parameters.
    radial_quadrature : RadialQuadratureSpec
        Static certified radial integration profile.
    final_state : FinalStateSpec
        Static final-state model selection.
    self_energy : SelfEnergyModel
        Causal electronic self-energy parameterization.
    eta_ev : Float64[Array, ""]
        Positive retarded regulator in eV.
    kz_nodes_frac : Optional[Float64[Array, " n_kz"]]
        Optional explicit out-of-plane quadrature nodes.
    kz_mode : str
        Static out-of-plane modelling declaration.
    required_capabilities : Tuple[str, ...]
        Electronic-state capabilities required before evaluation.
    intrinsic_payload_kind : str
        Static intrinsic observable discriminator.
    model_ref : str
        Stable factorized-model identity.

    See Also
    --------
    diffpes.simul.evaluate_spectral_projection
        Evaluate this model through the electronic-state protocol.
    """

    radial_spec: RadialSpec
    matrix_element_params: MatrixElementParams
    radial_quadrature: RadialQuadratureSpec
    final_state: FinalStateSpec
    self_energy: SelfEnergyModel
    eta_ev: Float64[Array, ""]
    kz_nodes_frac: Optional[Float64[Array, " n_kz"]]
    kz_mode: str = eqx.field(static=True)
    required_capabilities: Tuple[str, ...] = eqx.field(static=True)
    intrinsic_payload_kind: str = eqx.field(static=True)
    model_ref: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static model declarations and finite traced regulators."""
        if self.kz_mode not in (
            "native_direct",
            "bulk_direct",
            "bulk_kz",
            "coherent_slab",
        ):
            raise ValueError("unknown factorized out-of-plane mode")
        if self.intrinsic_payload_kind != "scalar_intensity":
            raise ValueError("factorized models must produce scalar intensity")
        if not self.model_ref or not self.required_capabilities:
            raise ValueError(
                "factorized model identity and capabilities are required"
            )
        if self.kz_nodes_frac is not None and self.kz_nodes_frac.ndim != 1:
            raise ValueError("out-of-plane nodes must have one dimension")


@jaxtyped(typechecker=beartype)
def make_factorized_arpes_model(
    radial_spec: RadialSpec,
    matrix_element_params: MatrixElementParams,
    radial_quadrature: RadialQuadratureSpec,
    final_state: FinalStateSpec,
    self_energy: SelfEnergyModel,
    eta_ev: Optional[Float64[Array, ""]] = None,
    *,
    kz_nodes_frac: Optional[Float64[Array, " n_kz"]] = None,
    kz_mode: str = "native_direct",
) -> FactorizedArpesModel:
    """Construct a validated factorized-photocurrent model.

    Parameters
    ----------
    radial_spec : RadialSpec
        Differentiable radial-wavefunction parameters.
    matrix_element_params : MatrixElementParams
        Differentiable dipole matrix-element parameters.
    radial_quadrature : RadialQuadratureSpec
        Static certified radial integration profile.
    final_state : FinalStateSpec
        Static final-state model selection.
    self_energy : SelfEnergyModel
        Causal self-energy parameterization.
    eta_ev : Optional[Float64[Array, ""]]
        Positive retarded regulator in eV.
    kz_nodes_frac : Optional[Float64[Array, " n_kz"]]
        Optional explicit fractional out-of-plane quadrature nodes.
    kz_mode : str
        Static out-of-plane modelling declaration.

    Returns
    -------
    FactorizedArpesModel
        Immutable typed factorized-photocurrent configuration.
    """
    raw_regulator = (
        jnp.asarray(1.0e-4, dtype=jnp.float64) if eta_ev is None else eta_ev
    )
    regulator = eqx.error_if(
        raw_regulator,
        raw_regulator <= 0.0,
        "factorized retarded regulator must be positive",
    )
    nodes = (
        None
        if kz_nodes_frac is None
        else kz_nodes_frac.astype(regulator.dtype)
    )
    return FactorizedArpesModel(
        radial_spec,
        matrix_element_params,
        radial_quadrature,
        final_state,
        self_energy,
        regulator,
        nodes,
        kz_mode,
        ("hamiltonian", "overlap"),
        "scalar_intensity",
        "org.diffpes.photocurrent.spectral_projection@0.1.0",
    )


__all__: list[str] = [
    "FactorizedArpesModel",
    "make_factorized_arpes_model",
]
