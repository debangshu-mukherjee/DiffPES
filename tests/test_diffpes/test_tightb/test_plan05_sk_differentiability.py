"""Certify Plan-05 slab gradients from fundamental Slater--Koster values.

The tests exercise Plan-05 numerical and structural contracts.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from diffpes.tightb import (
    bloch_hamiltonian,
    build_sk_model,
    diagonalize_tb,
    freeze_slab_topology,
    layer_resolved_group_traces,
    rebuild_slab,
    sk_model_parameter_view,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    SlabTopology,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_slater_koster_params,
)
from tests._gradients import gradient_gate


def _graphene_context() -> tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Array,
]:
    """Return a complete-p-shell nearest-neighbor graphene SK context."""
    lattice_constant: float = 2.46
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(
            (
                (lattice_constant, 0.0, 0.0),
                (
                    lattice_constant / 2.0,
                    lattice_constant * np.sqrt(3.0) / 2.0,
                    0.0,
                ),
                (0.0, 0.0, 10.0),
            )
        ),
        positions=jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0 / 3.0, 1.0 / 3.0, 0.0),
            )
        ),
        species=("C", "C"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 1, 1, 1),
        n=(2,) * 6,
        l=(1,) * 6,
        m=(-1, 0, 1, -1, 0, 1),
        labels=("A_py", "A_pz", "A_px", "B_py", "B_pz", "B_px"),
    )
    sk_params: SlaterKosterParams = make_slater_koster_params(
        values=jnp.asarray((1.1, -2.7)),
        keys=("C-C:pp_sigma", "C-C:pp_pi"),
    )
    onsite: Array = jnp.asarray((0.15, -0.1, 0.37, 0.11, -0.08, 0.41))
    return geometry, basis, sk_params, onsite


def _complete_p_soc_context() -> tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Array,
    Array,
    tuple[int, ...],
]:
    """Return a two-site complete-p spinor SK model with atomic SOC."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.diag(jnp.asarray((2.2, 2.5, 4.0))),
        positions=jnp.asarray(
            ((0.0, 0.0, 0.0), (0.2, 0.15, 0.18)),
            dtype=jnp.float64,
        ),
        species=("X", "Y"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 6 + (1,) * 6,
        n=(2,) * 12,
        l=(1,) * 12,
        m=(-1, 0, 1, -1, 0, 1) * 2,
        spin=(-1, -1, -1, 1, 1, 1) * 2,
        labels=(
            "X_py_down",
            "X_pz_down",
            "X_px_down",
            "X_py_up",
            "X_pz_up",
            "X_px_up",
            "Y_py_down",
            "Y_pz_down",
            "Y_px_down",
            "Y_py_up",
            "Y_pz_up",
            "Y_px_up",
        ),
    )
    sk_params: SlaterKosterParams = make_slater_koster_params(
        values=jnp.asarray((-1.17, 0.43)),
        keys=("X-Y:pp_sigma", "X-Y:pp_pi"),
    )
    onsite: Array = jnp.asarray(
        (0.13, -0.21, 0.47) * 2 + (-0.34, 0.26, 0.71) * 2
    )
    soc: Array = jnp.asarray((0.29, 0.17))
    shell_index: tuple[int, ...] = (0,) * 6 + (1,) * 6
    return geometry, basis, sk_params, onsite, soc, shell_index


def _broadened_spectral_moment(bands: DiagonalizedBands) -> Array:
    """Evaluate a smooth registered spectral invariant."""
    probes: Array = jnp.asarray((-1.31, -0.44, 0.18, 0.76, 1.42))
    coefficients: Array = jnp.asarray((0.7, -0.3, 0.9, 0.2, -0.5))
    width: float = 0.41
    profiles: Array = jnp.exp(
        -((bands.eigenvalues[:, :, None] - probes[None, None, :]) ** 2)
        / (2.0 * width**2)
    )
    return jnp.sum(profiles * coefficients[None, None, :])


def _graphene_sk_gate() -> tuple[Array, Callable[[Array], Array]]:
    """Create the true-SK graphene slab loss and its active coordinate."""
    geometry: CrystalGeometry
    basis: OrbitalBasis
    sk_params: SlaterKosterParams
    onsite: Array
    geometry, basis, sk_params, onsite = _graphene_context()
    direct: TBModel = build_sk_model(
        geometry=geometry,
        basis=basis,
        sk_params=sk_params,
        onsite_energies=onsite,
        soc_lambdas=jnp.zeros((0,)),
        shell_index=(-1,) * 6,
        cutoff=1.5,
    )
    parameters: Array
    rebuild_sk: Callable[[Array], TBModel]
    parameters, rebuild_sk = sk_model_parameter_view(
        geometry=geometry,
        basis=basis,
        sk_params=sk_params,
        onsite_energies=onsite,
        soc_lambdas=jnp.zeros((0,)),
        shell_index=(-1,) * 6,
        cutoff=1.5,
    )
    rebuilt: TBModel = rebuild_sk(parameters)
    assert jnp.array_equal(
        rebuilt.hopping_amplitudes,
        direct.hopping_amplitudes,
    )
    topology: SlabTopology = freeze_slab_topology(
        direct,
        miller=(0, 0, 1),
        thickness_ang=10.0,
        vacuum_ang=5.0,
    )
    k_x: Array = jnp.asarray((-0.43, -0.21, 0.06, 0.28, 0.41))
    kpoints: Array = jnp.stack(
        (k_x, 0.17 * k_x + 0.09, jnp.zeros_like(k_x)),
        axis=-1,
    )
    active: Array = parameters[: sk_params.values.size]

    def loss(candidate: Array) -> Array:
        vector: Array = parameters.at[: active.size].set(candidate)
        bulk: TBModel = rebuild_sk(vector)
        slab: TBModel
        slab, _ = rebuild_slab(bulk, topology)
        bands: DiagonalizedBands = diagonalize_tb(slab, kpoints)
        return _broadened_spectral_moment(bands)

    return active, loss


def _soc_sk_gate() -> tuple[
    Array,
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Create the complete-shell SOC slab SK and group-trace losses."""
    geometry: CrystalGeometry
    basis: OrbitalBasis
    sk_params: SlaterKosterParams
    onsite: Array
    soc: Array
    shell_index: tuple[int, ...]
    geometry, basis, sk_params, onsite, soc, shell_index = (
        _complete_p_soc_context()
    )
    direct: TBModel = build_sk_model(
        geometry=geometry,
        basis=basis,
        sk_params=sk_params,
        onsite_energies=onsite,
        soc_lambdas=soc,
        shell_index=shell_index,
        cutoff=2.6,
        spinor=True,
    )
    parameters: Array
    rebuild_sk: Callable[[Array], TBModel]
    parameters, rebuild_sk = sk_model_parameter_view(
        geometry=geometry,
        basis=basis,
        sk_params=sk_params,
        onsite_energies=onsite,
        soc_lambdas=soc,
        shell_index=shell_index,
        cutoff=2.6,
    )
    rebuilt: TBModel = rebuild_sk(parameters)
    assert jnp.array_equal(
        rebuilt.hopping_amplitudes,
        direct.hopping_amplitudes,
    )
    topology: SlabTopology = freeze_slab_topology(
        direct,
        miller=(0, 0, 1),
        thickness_ang=3.9,
        vacuum_ang=4.0,
    )
    k_x: Array = jnp.asarray((-0.39, -0.16, 0.08, 0.27, 0.46))
    kpoints: Array = jnp.stack(
        (k_x, -0.19 * k_x + 0.07, jnp.zeros_like(k_x)),
        axis=-1,
    )
    active: Array = parameters[: sk_params.values.size]

    def bands_for(candidate: Array) -> DiagonalizedBands:
        vector: Array = parameters.at[: active.size].set(candidate)
        bulk: TBModel = rebuild_sk(vector)
        slab: TBModel
        slab, _ = rebuild_slab(bulk, topology)
        return diagonalize_tb(slab, kpoints)

    def group_loss(candidate: Array) -> Array:
        bands: DiagonalizedBands = bands_for(candidate)
        return jnp.sum(
            layer_resolved_group_traces(
                bands,
                ((0, 1),),
                2.2,
            )
        )

    def combined_loss(candidate: Array) -> Array:
        bands: DiagonalizedBands = bands_for(candidate)
        group_trace: Array = layer_resolved_group_traces(
            bands,
            ((0, 1),),
            2.2,
        )
        return _broadened_spectral_moment(bands) + 0.23 * jnp.sum(group_trace)

    def imaginary_norm(candidate: Array) -> Array:
        vector: Array = parameters.at[: active.size].set(candidate)
        bulk: TBModel = rebuild_sk(vector)
        slab: TBModel
        slab, _ = rebuild_slab(bulk, topology)
        hamiltonian: Array = bloch_hamiltonian(slab, kpoints[2])
        return jnp.linalg.norm(jnp.imag(hamiltonian))

    return active, combined_loss, group_loss, imaginary_norm


class TestPlan05FundamentalSKGradients:
    """Certify D1 from independent SK integrals rather than derived hoppings."""

    @pytest.mark.rss_limit_mb(1536)
    def test_graphene_sk_parameter_gradient_gate(self) -> None:
        """Match fwd/rev/FD and retain nonzero graphene pp-pi sensitivity.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        active: Array
        loss: Callable[[Array], Array]
        active, loss = _graphene_sk_gate()

        gradient_gate(
            loss,
            active,
            regime="smooth",
            elementwise=True,
            directional_atol=2e-8,
        )

    @pytest.mark.rss_limit_mb(1800)
    def test_complete_shell_soc_sk_parameter_gradient_gate(self) -> None:
        """Match every pp SK row through a generic-complex SOC slab.

        Exercise this Plan-05 condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        active: Array
        loss: Callable[[Array], Array]
        group_loss: Callable[[Array], Array]
        imaginary_norm: Callable[[Array], Array]
        active, loss, group_loss, imaginary_norm = _soc_sk_gate()

        gradient_gate(
            loss,
            active,
            regime="smooth",
            elementwise=True,
            directional_atol=3e-8,
        )
        gradient_gate(
            group_loss,
            active,
            regime="smooth",
            elementwise=True,
            directional_atol=3e-8,
        )
        assert imaginary_norm(active) > 1e-6
