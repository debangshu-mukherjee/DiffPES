"""Execute the frozen local/nonlocal gauge negative control.

Extended Summary
----------------
The test checks the local commutator identity and the registered nonlocal
projector disagreement through both public Cartesian gauge APIs.
"""

import hashlib
import math
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from diffpes.maths import (
    dipole_length_cartesian,
    dipole_momentum_cartesian,
)

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "_reference_data"
    / "local_nonlocal_gauge_reference.npz"
)
_REFERENCE_SHA256 = (
    "e136dfd8214cd4e1e83d11b1d20d87a8597c66e61f54636b949d3c159fc579f0"
)


@jaxtyped(typechecker=beartype)
def _derivative_sixth(
    values: Float64[NDArray, " n_node"], spacing: float
) -> Float64[NDArray, " n_node"]:
    """PRIVATE: Differentiate with the frozen seven-point sixth-order stencils.

    Parameters
    ----------
    values : Float64[NDArray, " n_node"]
        Samples on a uniform grid.
    spacing : float
        Uniform grid spacing in Bohr.

    Returns
    -------
    derivative : Float64[NDArray, " n_node"]
        First-derivative samples at every node.

    Notes
    -----
    Clamps one seven-point window inside the grid for each node. Solves
    its Vandermonde moment system. Applies the resulting derivative
    weights to the windowed samples.
    """
    derivative: Float64[NDArray, " n_node"] = np.empty_like(values)
    index: int
    for index in range(values.size):
        start: int = min(max(index - 3, 0), values.size - 7)
        stencil_indices: Int64[NDArray, " 7"] = np.arange(start, start + 7)
        offsets: Float64[NDArray, " 7"] = (stencil_indices - index) * spacing
        moments: Float64[NDArray, "7 7"] = np.vander(
            offsets, 7, increasing=True
        ).T
        target: Float64[NDArray, " 7"] = np.zeros(7)
        target[1] = 1.0
        derivative[index] = (
            np.linalg.solve(moments, target) @ values[stencil_indices]
        )
    return derivative


@jaxtyped(typechecker=beartype)
def _public_reduced_gauges(
    radial_grid: Float64[NDArray, " n_node"],
    radial_weights: Float64[NDArray, " n_node"],
    states: Float64[NDArray, "2 n_node"],
) -> Tuple[Complex128[Array, ""], Complex128[Array, ""]]:
    """PRIVATE: Pass the exact radial angular reduction through public APIs.

    Implementation Logic
    --------------------
    1. **Recover the radial functions**::

           radial_initial = np.divide(state_s, radial_grid, ...)

       A zero output value handles the origin of each quotient.

    2. **Construct the Cartesian reduction**::

           positions = jnp.stack((zero, zero, radial_grid / sqrt(3)), axis=-1)

       The same angular factor scales the derivative component.

    3. **Evaluate both public gauges**::

           length = dipole_length_cartesian(...)

       Radial :math:`r^2` weights supply the sampled volume measure.

    Parameters
    ----------
    radial_grid : Float64[NDArray, " n_node"]
        Uniform radial nodes in Bohr.
    radial_weights : Float64[NDArray, " n_node"]
        Radial quadrature weights in Bohr.
    states : Float64[NDArray, "2 n_node"]
        Reduced s and p radial states in rows 0 and 1.

    Returns
    -------
    length : Complex128[Array, ""]
        Length-gauge s-to-p amplitude for z polarization.
    momentum : Complex128[Array, ""]
        Momentum-gauge s-to-p amplitude for z polarization.
    """
    state_s: Float64[NDArray, " n_node"] = states[0]
    state_p: Float64[NDArray, " n_node"] = states[1]
    radial_initial: Float64[NDArray, " n_node"] = np.divide(
        state_s,
        radial_grid,
        out=np.zeros_like(state_s),
        where=radial_grid > 0.0,
    )
    radial_final: Float64[NDArray, " n_node"] = np.divide(
        state_p,
        radial_grid,
        out=np.zeros_like(state_p),
        where=radial_grid > 0.0,
    )
    derivative_s: Float64[NDArray, " n_node"] = _derivative_sixth(
        state_s,
        radial_grid[1] - radial_grid[0],
    )
    radial_initial_derivative: Float64[NDArray, " n_node"] = np.divide(
        derivative_s * radial_grid - state_s,
        radial_grid**2,
        out=np.zeros_like(state_s),
        where=radial_grid > 0.0,
    )
    positions: Float64[Array, "n_node 3"] = jnp.stack(
        (
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.asarray(radial_grid) / math.sqrt(3.0),
        ),
        axis=-1,
    )
    gradient: Complex128[Array, "n_node 3"] = jnp.stack(
        (
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.zeros_like(jnp.asarray(radial_grid)),
            jnp.asarray(radial_initial_derivative) / math.sqrt(3.0),
        ),
        axis=-1,
    ).astype(jnp.complex128)
    weights: Float64[Array, " n_node"] = jnp.asarray(
        radial_weights * radial_grid**2
    )
    polarization: Complex128[Array, " 3"] = jnp.asarray(
        (0.0, 0.0, 1.0),
        dtype=jnp.complex128,
    )
    length: Complex128[Array, ""] = dipole_length_cartesian(
        jnp.asarray(radial_final, dtype=jnp.complex128),
        jnp.asarray(radial_initial, dtype=jnp.complex128),
        positions,
        weights,
        polarization,
    )
    momentum: Complex128[Array, ""] = dipole_momentum_cartesian(
        jnp.asarray(radial_final, dtype=jnp.complex128),
        gradient,
        weights,
        polarization,
    )
    result: Tuple[Complex128[Array, ""], Complex128[Array, ""]] = (
        length,
        momentum,
    )
    return result


def test_local_identity_passes_and_nonlocal_projector_disagrees() -> None:
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
        radial_grid: Float64[NDArray, " n_node"] = reference["local_r_coarse"]
        radial_weights: Float64[NDArray, " n_node"] = reference[
            "local_w_coarse"
        ]
        local_states: Float64[NDArray, "2 n_node"] = reference[
            "local_states_coarse"
        ]
        local_energies: Float64[NDArray, " 2"] = reference[
            "local_energies_coarse"
        ]
        nonlocal_states: Float64[NDArray, "2 n_node"] = reference[
            "nonlocal_states"
        ]
        nonlocal_energies: Float64[NDArray, " 2"] = reference[
            "nonlocal_energies"
        ]
        nonlocal_strength_derivative: complex = complex(
            reference["nonlocal_strength_derivative"]
        )

    local_length: Complex128[Array, ""]
    local_momentum: Complex128[Array, ""]
    local_length, local_momentum = _public_reduced_gauges(
        radial_grid,
        radial_weights,
        local_states,
    )
    local_gap: float = float(local_energies[1] - local_energies[0])
    local_residual: Complex128[Array, ""] = (
        local_momentum - 1j * local_gap * local_length
    )
    chex.assert_trees_all_close(
        local_residual,
        jnp.asarray(0.0j),
        rtol=0.0,
        atol=1.0e-10,
    )

    nonlocal_length: Complex128[Array, ""]
    nonlocal_momentum: Complex128[Array, ""]
    nonlocal_length, nonlocal_momentum = _public_reduced_gauges(
        radial_grid,
        radial_weights,
        nonlocal_states,
    )
    nonlocal_gap: float = float(nonlocal_energies[1] - nonlocal_energies[0])
    nonlocal_residual: Complex128[Array, ""] = (
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
