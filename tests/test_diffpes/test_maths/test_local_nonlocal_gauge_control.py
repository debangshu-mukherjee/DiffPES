"""Execute the frozen Plan 06 local/nonlocal gauge negative control.

The test checks the local commutator identity and the registered nonlocal
projector disagreement through both public Cartesian gauge APIs.
"""

import hashlib
import math
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray

from diffpes.maths import (
    dipole_length_cartesian,
    dipole_momentum_cartesian,
)

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "_reference_data"
    / "plan06_g12_reference.npz"
)
_REFERENCE_SHA256 = (
    "e136dfd8214cd4e1e83d11b1d20d87a8597c66e61f54636b949d3c159fc579f0"
)


def _derivative_sixth(
    values: Float[NDArray, " n_node"], spacing: float
) -> Float[NDArray, " n_node"]:
    """Differentiate with the frozen seven-point sixth-order stencils."""
    derivative: Float[NDArray, " n_node"] = np.empty_like(values)
    index: int
    for index in range(values.size):
        start: int = min(max(index - 3, 0), values.size - 7)
        stencil_indices: Int[NDArray, " 7"] = np.arange(start, start + 7)
        offsets: Float[NDArray, " 7"] = (stencil_indices - index) * spacing
        moments: Float[NDArray, "7 7"] = np.vander(
            offsets, 7, increasing=True
        ).T
        target: Float[NDArray, " 7"] = np.zeros(7)
        target[1] = 1.0
        derivative[index] = (
            np.linalg.solve(moments, target) @ values[stencil_indices]
        )
    return derivative


def _public_reduced_gauges(
    radial_grid: Float[NDArray, " n_node"],
    radial_weights: Float[NDArray, " n_node"],
    states: Float[NDArray, "2 n_node"],
) -> tuple[Array, Array]:
    """Pass the exact radial angular reduction through both public APIs."""
    state_s: Float[NDArray, " n_node"] = states[0]
    state_p: Float[NDArray, " n_node"] = states[1]
    radial_initial: Float[NDArray, " n_node"] = np.divide(
        state_s,
        radial_grid,
        out=np.zeros_like(state_s),
        where=radial_grid > 0.0,
    )
    radial_final: Float[NDArray, " n_node"] = np.divide(
        state_p,
        radial_grid,
        out=np.zeros_like(state_p),
        where=radial_grid > 0.0,
    )
    derivative_s: Float[NDArray, " n_node"] = _derivative_sixth(
        state_s,
        radial_grid[1] - radial_grid[0],
    )
    radial_initial_derivative: Float[NDArray, " n_node"] = np.divide(
        derivative_s * radial_grid - state_s,
        radial_grid**2,
        out=np.zeros_like(state_s),
        where=radial_grid > 0.0,
    )
    positions: Array = jnp.stack(
        (
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.asarray(radial_grid) / math.sqrt(3.0),
        ),
        axis=-1,
    )
    gradient: Array = jnp.stack(
        (
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.asarray(radial_initial_derivative) / math.sqrt(3.0),
        ),
        axis=-1,
    ).astype(jnp.complex128)
    weights: Array = jnp.asarray(radial_weights * radial_grid**2)
    polarization: Array = jnp.asarray(
        (0.0, 0.0, 1.0),
        dtype=jnp.complex128,
    )
    length: Array = dipole_length_cartesian(
        jnp.asarray(radial_final, dtype=jnp.complex128),
        jnp.asarray(radial_initial, dtype=jnp.complex128),
        positions,
        weights,
        polarization,
    )
    momentum: Array = dipole_momentum_cartesian(
        jnp.asarray(radial_final, dtype=jnp.complex128),
        gradient,
        weights,
        polarization,
    )
    return length, momentum


def test_g12_d12_local_passes_and_nonlocal_projector_must_disagree() -> None:
    """Assert the public local identity and both nonlocal controls.

    The test compares local and nonlocal fixtures through both public gauges.

    Notes
    -----
    It loads frozen radial states and evaluates the independent reduction.
    """
    digest: str = hashlib.sha256(_REFERENCE_PATH.read_bytes()).hexdigest()
    assert digest == _REFERENCE_SHA256
    reference: np.lib.npyio.NpzFile
    with np.load(_REFERENCE_PATH) as reference:
        radial_grid: Float[NDArray, " n_node"] = reference["local_r_coarse"]
        radial_weights: Float[NDArray, " n_node"] = reference["local_w_coarse"]
        local_states: Float[NDArray, "2 n_node"] = reference[
            "local_states_coarse"
        ]
        local_energies: Float[NDArray, " 2"] = reference[
            "local_energies_coarse"
        ]
        nonlocal_states: Float[NDArray, "2 n_node"] = reference[
            "nonlocal_states"
        ]
        nonlocal_energies: Float[NDArray, " 2"] = reference[
            "nonlocal_energies"
        ]
        nonlocal_strength_derivative: complex = complex(
            reference["nonlocal_strength_derivative"]
        )

    local_length: Array
    local_momentum: Array
    local_length, local_momentum = _public_reduced_gauges(
        radial_grid,
        radial_weights,
        local_states,
    )
    local_gap: float = float(local_energies[1] - local_energies[0])
    local_residual: Array = local_momentum - 1j * local_gap * local_length
    chex.assert_trees_all_close(
        local_residual,
        jnp.asarray(0.0j),
        rtol=0.0,
        atol=1.0e-10,
    )

    nonlocal_length: Array
    nonlocal_momentum: Array
    nonlocal_length, nonlocal_momentum = _public_reduced_gauges(
        radial_grid,
        radial_weights,
        nonlocal_states,
    )
    nonlocal_gap: float = float(nonlocal_energies[1] - nonlocal_energies[0])
    nonlocal_residual: Array = (
        nonlocal_momentum - 1j * nonlocal_gap * nonlocal_length
    )
    assert float(jnp.abs(nonlocal_residual)) >= 1.0e-5
    chex.assert_trees_all_close(
        jnp.abs(nonlocal_residual),
        jnp.asarray(0.5666054758006736),
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert math.isfinite(nonlocal_strength_derivative.imag)
    assert abs(nonlocal_strength_derivative) >= 1.0e-5
    assert abs(nonlocal_strength_derivative.imag) > 0.8
    assert nonlocal_strength_derivative.real == 0.0
