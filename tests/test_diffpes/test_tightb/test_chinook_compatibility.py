"""Compare Plan 04 bands with frozen offline Chinook artifacts.

These K-type tests establish behavioral compatibility only after the
Slater--Koster, Hamiltonian, and spin--orbit C gates have independently
established correctness. Chinook is never imported or executed by pytest.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diffpes.tightb import eigvalsh_bands
from diffpes.types import TBModel
from tests._factories import (
    make_graphene_model,
    make_rashba_model,
    make_t2g_soc_model,
)

_ARTIFACT_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "plan04_chinook_tightb_reference.json"
)
_ARTIFACT_SHA256: str = (
    "db52d72562f2efb49d25f9ce2b9affefed1af6f6fac927d1e20f9bb96f1510dc"
)
_COMPATIBILITY_RTOL: float = 1e-8
_COMPATIBILITY_ATOL: float = 2e-12


def _reference() -> dict[str, Any]:
    """Load and authenticate the inert numeric compatibility artifact."""
    encoded: bytes = _ARTIFACT_PATH.read_bytes()
    digest: str = hashlib.sha256(encoded).hexdigest()
    if digest != _ARTIFACT_SHA256:
        message: str = (
            "Plan 04 Chinook artifact checksum differs from its pinned digest"
        )
        raise ValueError(message)
    payload: dict[str, Any] = json.loads(encoded)
    if (
        payload["metadata"]["gate"] != "04.G6"
        or payload["metadata"]["classification"]
        != "K-type behavioral compatibility"
    ):
        message = "Plan 04 Chinook artifact metadata is invalid"
        raise ValueError(message)
    return payload


class TestChinookCompatibility:
    """Resolve the three-model Plan 04 K-type compatibility battery."""

    def test_graphene_bands_agree_after_the_c_gates(self) -> None:
        """Match frozen Chinook nearest-neighbor graphene eigenvalues.

        The case uses an authenticated inert compatibility artifact.

        Notes
        -----
        Compare the complete registered k-point path after native construction.
        """
        reference: dict[str, Any] = _reference()["graphene"]
        kpoints: Array = jnp.asarray(reference["kpoints_fractional"])
        model: TBModel = make_graphene_model(
            t=reference["conventions"]["hopping_ev"]
        )
        actual: Array = eigvalsh_bands(model, kpoints)

        np.testing.assert_allclose(
            actual,
            np.asarray(reference["eigenvalues_ev"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=_COMPATIBILITY_ATOL,
        )

    def test_rashba_bands_agree_after_the_c_gates(self) -> None:
        """Match frozen Chinook square-lattice Rashba eigenvalues.

        The case checks the spinful square-lattice convention independently.

        Notes
        -----
        Compare every frozen band after native model construction.
        """
        reference: dict[str, Any] = _reference()["rashba"]
        conventions: dict[str, Any] = reference["conventions"]
        model: TBModel = make_rashba_model(
            hopping=conventions["kinetic_ev"],
            rashba=conventions["rashba_ev"],
        )
        actual: Array = eigvalsh_bands(
            model,
            jnp.asarray(reference["kpoints_fractional"]),
        )

        np.testing.assert_allclose(
            actual,
            np.asarray(reference["eigenvalues_ev"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=_COMPATIBILITY_ATOL,
        )

    def test_t2g_soc_bands_agree_after_the_c_gates(self) -> None:
        """Match frozen Chinook projected atomic t2g SOC multiplets.

        The case checks the registered real-cubic basis convention.

        Notes
        -----
        Compare the complete frozen multiplet after native SOC construction.
        """
        reference: dict[str, Any] = _reference()["t2g_soc"]
        model: TBModel = make_t2g_soc_model(
            coupling=reference["conventions"]["lambda_ev"],
        )
        actual: Array = eigvalsh_bands(
            model,
            jnp.asarray(reference["kpoints_fractional"]),
        )

        np.testing.assert_allclose(
            actual,
            np.asarray(reference["eigenvalues_ev"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=_COMPATIBILITY_ATOL,
        )


__all__: list[str] = []
