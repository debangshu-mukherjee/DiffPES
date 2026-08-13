"""Verify factorized-photocurrent model invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict
from jaxtyping import TypeCheckError

from diffpes.types import (
    FactorizedArpesModel,
    make_factorized_arpes_model,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
)


def _model(**overrides: object) -> FactorizedArpesModel:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    basis: Any = make_orbital_basis(
        atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
    )
    values: Dict[str, object] = {
        "radial_spec": make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
        ),
        "matrix_element_params": make_matrix_element_params(
            basis,
            (0,),
            sigma_shell=jnp.asarray([1.0]),
            phase_shift_angles_shell=jnp.asarray([0.0]),
        ),
        "radial_quadrature": make_radial_quadrature_spec(),
        "final_state": make_final_state_spec(),
        "self_energy": make_self_energy_model(gamma=0.05),
        "eta_ev": jnp.asarray(0.01),
    }
    values.update(overrides)
    result: Any = make_factorized_arpes_model(**values)
    return result


class TestFactorizedarpesmodel:
    """Verify ``diffpes.types.FactorizedArpesModel`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_factory_declares_exact_evaluator_capabilities(self) -> None:
        """Declare only Hamiltonian and overlap capabilities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect the immutable tuple and scalar-payload identity.
        """
        model: Any = _model(kz_nodes_frac=jnp.asarray([-0.2, 0.2]))
        assert model.required_capabilities == ("hamiltonian", "overlap")
        assert model.intrinsic_payload_kind == "scalar_intensity"
        assert jnp.array_equal(model.kz_nodes_frac, jnp.asarray([-0.2, 0.2]))

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"kz_mode": "bad"}, "unknown factorized", ValueError),
            (
                {"kz_nodes_frac": jnp.zeros((1, 1))},
                "kz_nodes_frac",
                TypeCheckError,
            ),
            (
                {"eta_ev": jnp.asarray(0.0)},
                "regulator must be finite and positive",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"kz_nodes_frac": jnp.asarray([jnp.nan])},
                "nodes must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_public_model_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject bad mode, node shape, regulator, and node values.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one field in the otherwise valid one-orbital model.
        """
        with pytest.raises(error, match=message):
            _model(**overrides)


class TestMakeFactorizedArpesModel:
    """Verify ``diffpes.types.make_factorized_arpes_model``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
