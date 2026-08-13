"""Define the typed factorized-photocurrent model boundary.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`FactorizedArpesModel`
    Define the ``FactorizedArpesModel`` public contract.
:func:`make_factorized_arpes_model`
    Compute the ``make_factorized_arpes_model`` public contract.
"""

import importlib

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped

from .coordinates import MeasurementCoordinates
from .electronic_state import HamiltonianOverlapSource
from .radial_params import MatrixElementParams, RadialSpec
from .radial_profiles import FinalStateSpec, RadialQuadratureSpec
from .result import IntrinsicPhotocurrent
from .self_energy import SelfEnergyModel


class FactorizedArpesModel(eqx.Module):
    """Define the ``FactorizedArpesModel`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_photocurrent.TestFactorizedarpesmodel`

    Attributes
    ----------
    radial_spec : RadialSpec
        Store radial parameters.
    matrix_element_params : MatrixElementParams
        Store matrix-element parameters.
    radial_quadrature : RadialQuadratureSpec
        Store radial quadrature.
    final_state : FinalStateSpec
        Store the final-state model.
    self_energy : SelfEnergyModel
        Store the self-energy model.
    eta_ev : Float64[Array, ""]
        Store the retarded regulator.
    kz_nodes_frac : Optional[Float64[Array, " n_kz"]]
        Store out-of-plane nodes.
    kz_mode : str
        Store the out-of-plane mode.
    required_capabilities : Tuple[str, ...]
        Store required capabilities.
    intrinsic_payload_kind : str
        Store the payload kind.
    model_ref : str
        Store the model identity.

    See Also
    --------
    make_factorized_arpes_model
        Construct a validated factorized model.
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
        if self.required_capabilities != ("hamiltonian", "overlap"):
            raise ValueError(
                "spectral projection requires Hamiltonian and overlap"
            )

    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        electronic_state: HamiltonianOverlapSource,
        coordinates: MeasurementCoordinates,
        omega_rel_fermi_ev: Float64[Array, " n_omega"],
        temperature_k: Float64[Array, " n_temperature"],
    ) -> IntrinsicPhotocurrent:
        """Evaluate this model through the spectral-projection entry point."""
        evaluator: Callable[..., IntrinsicPhotocurrent] = (
            importlib.import_module(
                "diffpes.simul"
            ).evaluate_spectral_projection
        )
        photocurrent: IntrinsicPhotocurrent = evaluator(
            self,
            electronic_state,
            coordinates,
            omega_rel_fermi_ev,
            temperature_k,
        )
        return photocurrent


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
    """Compute the ``make_factorized_arpes_model`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_photocurrent.TestMakeFactorizedArpesModel`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    radial_spec : RadialSpec
        Input value for this operation.
    matrix_element_params : MatrixElementParams
        Input value for this operation.
    radial_quadrature : RadialQuadratureSpec
        Input value for this operation.
    final_state : FinalStateSpec
        Input value for this operation.
    self_energy : SelfEnergyModel
        Input value for this operation.
    eta_ev : Optional[Float64[Array, '']]
        Input value for this operation.
    kz_nodes_frac : Optional[Float64[Array, ' n_kz']]
        Input value for this operation.
    kz_mode : str
        Input value for this operation.

    Returns
    -------
    result : FactorizedArpesModel
        Validated operation result.
    """
    raw_regulator: Float64[Array, ""] = (
        jnp.asarray(1.0e-4, dtype=jnp.float64) if eta_ev is None else eta_ev
    )
    regulator: Float64[Array, ""] = jnp.asarray(
        raw_regulator, dtype=jnp.float64
    )
    regulator = eqx.error_if(
        regulator,
        ~jnp.isfinite(regulator) | (regulator <= 0.0),
        "factorized retarded regulator must be finite and positive",
    )
    nodes: Optional[Float64[Array, " n_kz"]] = (
        None
        if kz_nodes_frac is None
        else jnp.asarray(kz_nodes_frac, dtype=regulator.dtype)
    )
    if nodes is not None:
        nodes = eqx.error_if(
            nodes,
            ~jnp.all(jnp.isfinite(nodes)),
            "out-of-plane nodes must be finite",
        )
    result: FactorizedArpesModel = FactorizedArpesModel(
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
    return result


__all__: list[str] = [
    "FactorizedArpesModel",
    "make_factorized_arpes_model",
]
