"""Validate retained expanded-input incoherent wrappers.

The tests cover both wrappers, dispatch, and retired-level rejection.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

import diffpes
from diffpes.simul import (
    simulate_basic_expanded,
    simulate_expanded,
    simulate_novice_expanded,
)
from diffpes.types import make_orbital_basis


def _arrays() -> tuple[jax.Array, jax.Array]:
    """Return compact band and projection arrays."""
    eigenvalues: jax.Array = jnp.asarray([[-0.4, 0.1], [-0.2, 0.3]])
    projections: jax.Array = jnp.full((2, 2, 1, 9), 0.1)
    return eigenvalues, projections


def _basis() -> diffpes.types.OrbitalBasis:
    """Return atom-major Cu subshell metadata."""
    basis: diffpes.types.OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 9,
        n=(3,) * 9,
        l=(0, 1, 1, 1, 2, 2, 2, 2, 2),
        m=(0, -1, 0, 1, -2, -1, 0, 1, 2),
    )
    return basis


class TestSimulateNoviceExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_novice_expanded`."""

    def test_returns_requested_grid(self) -> None:
        """Return finite novice intensity on the requested grid.

        The wrapper must preserve the k axis and requested energy fidelity.

        Notes
        -----
        Evaluate fixed raw arrays and inspect the resulting shape and finite
        values.
        """
        eigenvalues: jax.Array
        projections: jax.Array
        eigenvalues, projections = _arrays()
        spectrum: diffpes.types.ArpesSpectrum = simulate_novice_expanded(
            eigenvalues,
            projections,
            0.0,
            0.04,
            0.1,
            48,
            15.0,
            200.0,
        )
        chex.assert_shape(spectrum.intensity, (2, 48))
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateBasicExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_basic_expanded`."""

    def test_returns_cross_section_weighted_grid(self) -> None:
        """Return finite basic intensity with atomic cross sections.

        The wrapper must consume explicit Cu subshell metadata at 200 eV.

        Notes
        -----
        Evaluate fixed raw arrays and inspect the resulting shape and finite
        values.
        """
        eigenvalues: jax.Array
        projections: jax.Array
        eigenvalues, projections = _arrays()
        spectrum: diffpes.types.ArpesSpectrum = simulate_basic_expanded(
            eigenvalues,
            projections,
            0.0,
            0.04,
            48,
            15.0,
            200.0,
            _basis(),
            (29,),
        )
        chex.assert_shape(spectrum.intensity, (2, 48))
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_expanded`."""

    def test_dispatches_retained_tiers(self) -> None:
        """Dispatch both retained incoherent tiers.

        Novice and basic selections must produce the requested output shape.

        Notes
        -----
        Route the same raw arrays through each static selector and compare
        their shape contracts.
        """
        eigenvalues: jax.Array
        projections: jax.Array
        eigenvalues, projections = _arrays()
        novice: diffpes.types.ArpesSpectrum = simulate_expanded(
            "novice",
            eigenvalues,
            projections,
            fidelity=32,
        )
        basic: diffpes.types.ArpesSpectrum = simulate_expanded(
            "basic",
            eigenvalues,
            projections,
            fidelity=32,
            photon_energy=200.0,
            basis=_basis(),
            atomic_numbers=(29,),
        )
        chex.assert_shape(novice.intensity, (2, 32))
        chex.assert_shape(basic.intensity, (2, 32))

    def test_rejects_deleted_level(self) -> None:
        """Reject former heuristic dispatcher entries.

        The static dispatcher must not route the retired advanced level.

        Notes
        -----
        Pass the deleted selector and match the two-level error message.
        """
        eigenvalues: jax.Array
        projections: jax.Array
        eigenvalues, projections = _arrays()
        with pytest.raises(ValueError, match="novice.*basic"):
            simulate_expanded("advanced", eigenvalues, projections)
