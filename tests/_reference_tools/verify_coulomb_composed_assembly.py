"""Verify the composed Coulomb assembly derivatives.

Differentiate effective charge and photon energy through the complete
matrix-element assembly. Compare automatic derivatives with finite-difference
ladders and require nonzero sensitivities.
"""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.simul import (
    assemble_orbital_transition_channels,
    contract_experiment_polarization,
    final_state_k_inv_ang,
    kinetic_energy_ev,
    matrix_element_intensity,
    project_band_channels,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    ExperimentGeometry,
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
)

_EFFECTIVE_CHARGE_DERIVATIVE_FLOOR: float = 1.0e-5
_PHOTON_ENERGY_DERIVATIVE_FLOOR: float = 1.0e-6
_FD_STEP_LADDERS: Tuple[Tuple[float, ...], ...] = (
    (4.0e-3, 2.0e-3, 1.0e-3),
    (4.0e-2, 2.0e-2, 1.0e-2),
)


def _fixture() -> Tuple[
    DiagonalizedBands,
    RadialSpec,
    MatrixElementParams,
    RadialQuadratureSpec,
    ExperimentGeometry,
]:
    """PRIVATE: Build a compact supported-radial full-assembly fixture.

    Returns
    -------
    fixture : Tuple[DiagonalizedBands, RadialSpec, MatrixElementParams,
        RadialQuadratureSpec, ExperimentGeometry]
        One-orbital, one-band, one-k fixture. It contains diagonalized bands
        and a compact radial spec on ``[0, 8]`` Bohr. Its radial edge vanishes
        smoothly. It also contains matrix-element parameters, default
        quadrature, and a 30 eV x-polarized experiment.

    Notes
    -----
    The cubic lattice constant ``2 pi`` Angstrom makes reciprocal and
    fractional coordinates coincide; the radial row forces an exact
    zero at the support edge.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        2.0 * math.pi * jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        ("X",),
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=jnp.zeros((1, 1), dtype=jnp.float64),
        eigenvectors=jnp.ones((1, 1, 1), dtype=jnp.complex128),
        kpoints=jnp.asarray([[0.35, 0.0, 0.0]], dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
    )
    radius: Float64[Array, " 65"] = jnp.linspace(
        0.0,
        8.0,
        65,
        dtype=jnp.float64,
    )
    radial_row: Float64[Array, " 65"] = (
        jnp.exp(-radius) * (1.0 - radius / 8.0) ** 2
    )
    radial_row = radial_row.at[-1].set(0.0)
    radial: RadialSpec = make_radial_spec(
        basis,
        (0,),
        mode="grid",
        r_grid=radius,
        grid_values_shell=radial_row[None, :],
    )
    matrix_params: MatrixElementParams = make_matrix_element_params(
        basis,
        (0,),
    )
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    experiment: ExperimentGeometry = make_experiment_geometry(
        30.0,
        jnp.asarray(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            dtype=jnp.complex128,
        ),
    )
    fixture: Tuple[
        DiagonalizedBands,
        RadialSpec,
        MatrixElementParams,
        RadialQuadratureSpec,
        ExperimentGeometry,
    ] = (
        bands,
        radial,
        matrix_params,
        quadrature,
        experiment,
    )
    return fixture


def _intensity(
    parameters: Float64[Array, " 2"],
    bands: DiagonalizedBands,
    radial: RadialSpec,
    matrix_params: MatrixElementParams,
    quadrature: RadialQuadratureSpec,
    experiment: ExperimentGeometry,
) -> Float64[Array, ""]:
    """PRIVATE: Compose charge and photon energy through the assembly.

    Parameters
    ----------
    parameters : Float64[Array, " 2"]
        Effective charge (dimensionless) and photon energy in eV.
    bands : DiagonalizedBands
        Frozen one-band electronic structure.
    radial : RadialSpec
        Frozen compact radial-wavefunction specification.
    matrix_params : MatrixElementParams
        Frozen orbital-to-shell matrix-element parameters.
    quadrature : RadialQuadratureSpec
        Frozen radial quadrature configuration.
    experiment : ExperimentGeometry
        Frozen photon and polarization geometry.

    Returns
    -------
    intensity : Float64[Array, ""]
        Scalar matrix-element intensity for the single band and
        k-point.

    Implementation Logic
    --------------------
    Run kinematics from photon energy. Build the Coulomb final state at the
    effective charge. Compute orbital channels, band projection, polarization
    contraction, and the intensity map. Keep both inputs differentiable end
    to end.
    """
    effective_charge: Float64[Array, ""] = parameters[0]
    photon_energy: Float64[Array, ""] = parameters[1]
    kinetic_energy: Float64[Array, " 1"]
    energy_valid: Bool[Array, " 1"]
    kinetic_energy, energy_valid = kinetic_energy_ev(
        photon_energy,
        experiment.work_function_ev,
        jnp.zeros((1,), dtype=jnp.float64),
    )
    momentum_magnitude: Float64[Array, " 1"]
    momentum_valid: Bool[Array, " 1"]
    momentum_magnitude, momentum_valid = final_state_k_inv_ang(kinetic_energy)
    parallel_momentum: Float64[Array, " 1"] = jnp.full_like(
        momentum_magnitude,
        0.35,
    )
    final_momentum: Float64[Array, "1 3"] = jnp.stack(
        (
            parallel_momentum,
            jnp.zeros_like(parallel_momentum),
            jnp.sqrt(momentum_magnitude**2 - parallel_momentum**2),
        ),
        axis=-1,
    )
    final_state: FinalStateSpec = make_final_state_spec(
        mode="coulomb",
        effective_charge=effective_charge,
    )
    orbital_channels: Complex128[Array, "1 1 1 3"] = (
        assemble_orbital_transition_channels(
            bands,
            radial,
            matrix_params,
            quadrature,
            final_state,
            experiment,
            final_momentum,
            energy_valid & momentum_valid,
        )
    )
    band_channels: Complex128[Array, "1 1 1 3"] = project_band_channels(
        orbital_channels,
        bands.eigenvectors,
    )
    amplitudes: Complex128[Array, "1 1 1"] = contract_experiment_polarization(
        band_channels,
        experiment,
    )
    intensity: Float64[Array, ""] = matrix_element_intensity(amplitudes)[0, 0]
    return intensity


def _five_point_derivative(
    parameters: Float64[Array, " 2"],
    coordinate: int,
    step: float,
    bands: DiagonalizedBands,
    radial: RadialSpec,
    matrix_params: MatrixElementParams,
    quadrature: RadialQuadratureSpec,
    experiment: ExperimentGeometry,
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate one five-point central derivative.

    Parameters
    ----------
    parameters : Float64[Array, " 2"]
        Expansion point: effective charge and photon energy in eV.
    coordinate : int
        Perturbed coordinate index (0 or 1).
    step : float
        Positive stencil step in the coordinate's units.
    bands : DiagonalizedBands
        Frozen one-band electronic structure.
    radial : RadialSpec
        Frozen compact radial-wavefunction specification.
    matrix_params : MatrixElementParams
        Frozen orbital-to-shell matrix-element parameters.
    quadrature : RadialQuadratureSpec
        Frozen radial quadrature configuration.
    experiment : ExperimentGeometry
        Frozen photon and polarization geometry.

    Returns
    -------
    derivative : Float64[Array, ""]
        Central-difference derivative with truncation order
        ``step**4``.

    Notes
    -----
    The stencil is ``(-f(+2s) + 8 f(+s) - 8 f(-s) + f(-2s)) /
    (12 s)`` along one coordinate of :func:`_intensity`.
    """
    direction: Float64[Array, " 2"] = (
        jnp.zeros_like(parameters).at[coordinate].set(step)
    )
    derivative: Float64[Array, ""] = (
        -_intensity(
            parameters + 2.0 * direction,
            bands,
            radial,
            matrix_params,
            quadrature,
            experiment,
        )
        + 8.0
        * _intensity(
            parameters + direction,
            bands,
            radial,
            matrix_params,
            quadrature,
            experiment,
        )
        - 8.0
        * _intensity(
            parameters - direction,
            bands,
            radial,
            matrix_params,
            quadrature,
            experiment,
        )
        + _intensity(
            parameters - 2.0 * direction,
            bands,
            radial,
            matrix_params,
            quadrature,
            experiment,
        )
    ) / (12.0 * step)
    return derivative


def main() -> None:
    """Compare forward and reverse autodiff with a finite-difference plateau.

    Raises
    ------
    AssertionError
        If either sensitivity tripwire or the finite-difference plateau fails.

    Notes
    -----
    The command checks the complete composed Coulomb transition route. It
    prints the value and derivative record to standard output.
    """
    bands: DiagonalizedBands
    radial: RadialSpec
    matrix_params: MatrixElementParams
    quadrature: RadialQuadratureSpec
    experiment: ExperimentGeometry
    bands, radial, matrix_params, quadrature, experiment = _fixture()
    parameters: Float64[Array, " 2"] = jnp.asarray(
        [0.4, 30.0],
        dtype=jnp.float64,
    )

    def objective(values: Float64[Array, " 2"]) -> Float64[Array, ""]:
        """Compute the composed intensity for one parameter pair.

        Parameters
        ----------
        values : Float64[Array, " 2"]
            Effective charge and photon energy in eV.

        Returns
        -------
        intensity : Float64[Array, ""]
            Scalar matrix-element intensity.

        Notes
        -----
        The closure fixes every non-differentiated carrier.
        """
        intensity: Float64[Array, ""] = _intensity(
            values,
            bands,
            radial,
            matrix_params,
            quadrature,
            experiment,
        )
        return intensity

    forward: Float64[Array, " 2"] = jax.jacfwd(objective)(parameters)
    reverse: Float64[Array, " 2"] = jax.jacrev(objective)(parameters)
    finite_difference: Float64[Array, "3 2"] = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _five_point_derivative(
                        parameters,
                        coordinate,
                        step,
                        bands,
                        radial,
                        matrix_params,
                        quadrature,
                        experiment,
                    )
                    for coordinate, step in enumerate(rung)
                )
            )
            for rung in zip(*_FD_STEP_LADDERS, strict=True)
        )
    )
    np.testing.assert_allclose(
        forward,
        reverse,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        jnp.broadcast_to(forward, finite_difference.shape),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-10,
    )
    if float(jnp.abs(forward[0])) <= _EFFECTIVE_CHARGE_DERIVATIVE_FLOOR:
        message: str = "effective-charge derivative tripwire failed"
        raise AssertionError(message)
    if float(jnp.abs(forward[1])) <= _PHOTON_ENERGY_DERIVATIVE_FLOOR:
        message: str = "photon-energy derivative tripwire failed"
        raise AssertionError(message)
    value: Float64[Array, ""] = objective(parameters)
    np.testing.assert_allclose(
        value,
        2.0058154144143075e-4,
        rtol=1.0e-10,
        atol=1.0e-14,
    )
    fd_budget_ratios: Float64[Array, "3 2"] = jnp.abs(
        finite_difference - forward[None, :]
    ) / (1.0e-10 + 1.0e-6 * jnp.abs(forward[None, :]))
    plateau_spread: Float64[Array, " 2"] = jnp.abs(
        finite_difference[-1] - finite_difference[-2]
    ) / (1.0e-10 + 1.0e-6 * jnp.abs(forward))
    if bool(jnp.any(plateau_spread > 1.0)):
        message: str = f"composed Coulomb FD plateau failed: {plateau_spread}"
        raise AssertionError(message)
    metrics: Dict[str, Any] = {
        "forward_derivative": [float(value) for value in forward],
        "reverse_derivative": [float(value) for value in reverse],
        "fd_step_ladders": [list(values) for values in _FD_STEP_LADDERS],
        "fd_values": [
            [float(value) for value in row] for row in finite_difference
        ],
        "fd_budget_ratios": [
            [float(value) for value in row] for row in fd_budget_ratios
        ],
        "plateau_spread_budget_ratios": [
            float(value) for value in plateau_spread
        ],
        "value": float(value),
    }
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
