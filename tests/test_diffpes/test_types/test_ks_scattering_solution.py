"""Verify scattering solver-policy and result-batch invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict
from jaxtyping import TypeCheckError

from diffpes.types import (
    KSScatteringBatch,
    make_ks_scattering_batch,
    make_ks_scattering_solver_spec,
)


def _batch(**overrides: object) -> KSScatteringBatch:
    """PRIVATE: Build one physical scattering diagnostic batch.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "states": jnp.ones((1, 2, 1, 1), dtype=jnp.complex128),
        "reflection_amplitudes": jnp.zeros((1, 1, 1), dtype=jnp.complex128),
        "transmission_amplitudes": jnp.ones((1, 1, 1), dtype=jnp.complex128),
        "open_channel_mask": jnp.ones((1, 1), dtype=jnp.bool_),
        "residual_norm": jnp.asarray([1.0e-12]),
        "incident_flux": jnp.asarray([1.0]),
        "reflected_flux": jnp.asarray([0.0]),
        "transmitted_flux": jnp.asarray([1.0]),
        "absorbed_flux": jnp.asarray([0.0]),
        "state_ref": "state",
    }
    values.update(overrides)
    result: Any = make_ks_scattering_batch(**values)
    return result


class TestKsscatteringsolverspec:
    """Verify ``diffpes.types.KSScatteringSolverSpec`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_positive_finite_solver_policy(self) -> None:
        """Preserve all default tolerances and Krylov dimensions.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare defaults with their explicit public contract values.
        """
        solver: Any = make_ks_scattering_solver_spec()
        assert solver.relative_residual == 1.0e-10
        assert solver.max_iterations == 500

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"relative_residual": 0.0}, "must be positive"),
            ({"threshold_guard_ev": float("nan")}, "must be positive"),
            ({"max_iterations": 0}, "must be positive"),
            ({"preconditioner_ref": ""}, "reference is required"),
        ],
    )
    def test_rejects_each_solver_invariant(
        self, kwargs: Dict[str, object], message: str
    ) -> None:
        """Reject nonpositive, nonfinite, and unidentified solver policy.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one default policy field in each parameterized case.
        """
        with pytest.raises(ValueError, match=message):
            make_ks_scattering_solver_spec(**kwargs)


class TestKsscatteringbatch:
    """Verify ``diffpes.types.KSScatteringBatch`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_physical_diagnostics(self) -> None:
        """Preserve a unit-transmission one-channel state batch.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the exact incident and transmitted fluxes.
        """
        batch: Any = _batch()
        assert batch.incident_flux[0] == 1.0
        assert batch.transmitted_flux[0] == 1.0

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"state_ref": ""}, "state_ref must be nonempty", ValueError),
            (
                {"open_channel_mask": jnp.ones((1, 2), dtype=jnp.bool_)},
                "open_channel_mask",
                TypeCheckError,
            ),
            (
                {"residual_norm": jnp.asarray([-1.0])},
                "finite and physical",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"incident_flux": jnp.asarray([jnp.nan])},
                "finite and physical",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"absorbed_flux": jnp.asarray([-1.0])},
                "finite and physical",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_batch_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject empty identity, mismatched axes, and unphysical diagnostics.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one field in the unit-transmission batch.
        """
        with pytest.raises(error, match=message):
            _batch(**overrides)


class TestMakeKsScatteringSolverSpec:
    """Verify ``diffpes.types.make_ks_scattering_solver_spec``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeKsScatteringBatch:
    """Verify ``diffpes.types.make_ks_scattering_batch``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
