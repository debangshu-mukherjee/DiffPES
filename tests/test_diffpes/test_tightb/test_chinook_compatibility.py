"""Compare tight-binding bands with frozen offline Chinook artifacts.

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
from beartype.typing import Dict
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
    / "chinook_tightb_reference.json"
)
_ARTIFACT_SHA256: str = (
    "86e92af3d455bf744521c993f838b397f7ef38163ad92bd4230b0b7d009ef7fa"
)
_COMPATIBILITY_RTOL: float = 1e-8
_COMPATIBILITY_ATOL: float = 2e-12


def _reference() -> Dict[str, Any]:
    """PRIVATE: Load and authenticate the inert numeric compatibility artifact.

    Returns
    -------
    payload : dict[str, Any]
        Parsed JSON content of ``chinook_tightb_reference.json`` with
        per-model k-points, eigenvalues in eV, and conventions.

    Raises
    ------
    ValueError
        If the SHA-256 digest differs from the pinned constant, or if
        the artifact metadata does not declare the K-type
        tight-binding-parity requirement.

    Notes
    -----
    Reads the artifact bytes, checks them against ``_ARTIFACT_SHA256``,
    and validates the recorded requirement and classification before any
    numeric comparison uses the payload. Chinook itself never runs.
    """
    encoded: bytes = _ARTIFACT_PATH.read_bytes()
    digest: str = hashlib.sha256(encoded).hexdigest()
    if digest != _ARTIFACT_SHA256:
        message: str = (
            "Chinook artifact checksum differs from its pinned digest"
        )
        raise ValueError(message)
    payload: Dict[str, Any] = json.loads(encoded)
    if (
        payload["metadata"]["requirement"] != "chinook-tightbinding-parity"
        or payload["metadata"]["classification"]
        != "K-type behavioral compatibility"
    ):
        message = "Chinook artifact metadata is invalid"
        raise ValueError(message)
    return payload


class TestChinookCompatibility:
    """Resolve the three-model Chinook K-type compatibility battery."""

    def test_graphene_bands_agree_after_the_c_gates(self) -> None:
        """Match frozen Chinook nearest-neighbor graphene eigenvalues.

        The case uses an authenticated inert compatibility artifact.

        Notes
        -----
        Compare the complete registered k-point path after native construction.
        """
        reference: Dict[str, Any] = _reference()["graphene"]
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
        reference: Dict[str, Any] = _reference()["rashba"]
        conventions: Dict[str, Any] = reference["conventions"]
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
        reference: Dict[str, Any] = _reference()["t2g_soc"]
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
