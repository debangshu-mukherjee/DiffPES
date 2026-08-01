"""Exercise novice-chain differentiability and scalability properties.

Extended Summary
----------------
Validates the migrated carrier seams of the novice forward chain, JIT cache
behavior across dynamic and static fields, and vectorization over a batched
simulation-parameter leaf.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, Float64

import diffpes
from diffpes.simul import simulate_novice
from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    SimulationParams,
    make_band_structure,
    make_orbital_projection,
    make_simulation_params,
)
from tests._gradients import assert_grad_matches_fd, assert_nonzero_grad


def _novice_fixture(
    fidelity: int = 16,
) -> tuple[BandStructure, OrbitalProjection, SimulationParams]:
    """Build the pinned two-band audit fixture."""
    bands: BandStructure = make_band_structure(
        eigenvalues=jnp.array([[-1.0, 1.0]], dtype=jnp.float64),
        kpoints=jnp.zeros((1, 3), dtype=jnp.float64),
    )
    projections: Float64[Array, "1 2 1 9"] = jnp.ones(
        (1, 2, 1, 9), dtype=jnp.float64
    )
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections
    )
    params: SimulationParams = make_simulation_params(
        energy_min=-2.0,
        energy_max=2.0,
        fidelity=fidelity,
        sigma=0.08,
        gamma=0.12,
    )
    result: tuple[BandStructure, OrbitalProjection, SimulationParams] = (
        bands,
        orbital_projection,
        params,
    )
    return result


class TestNoviceCarrierGradients:
    """Validate autodiff across each migrated novice-chain carrier seam.

    :see: :func:`diffpes.simul.simulate_novice`
    """

    def test_gradients_match_finite_differences(self) -> None:
        """Match central differences for bands, projections, and parameters.

        Extended Summary
        ----------------
        The test differentiates summed spectral intensity through each complete
        carrier on the pinned ``[-1, +1]`` eV two-band fixture. Automatic
        reverse-mode derivatives must agree with the shared central-FD gate.

        Notes
        -----
        The test uses a sixteen-point energy grid to keep the full leafwise FD census
        small while retaining both occupied and underflowed unoccupied bands.
        """
        bands: diffpes.types.BandStructure
        projections: Array
        params: diffpes.types.SimulationParams

        bands, projections, params = _novice_fixture()

        def bands_loss(candidate: BandStructure) -> Float64[Array, ""]:
            spectrum: ArpesSpectrum = simulate_novice(
                candidate, projections, params, 15.0
            )
            loss: Float64[Array, ""] = jnp.sum(spectrum.intensity)
            return loss

        def projection_loss(
            candidate: OrbitalProjection,
        ) -> Float64[Array, ""]:
            spectrum: ArpesSpectrum = simulate_novice(
                bands, candidate, params, 15.0
            )
            loss: Float64[Array, ""] = jnp.sum(spectrum.intensity)
            return loss

        def params_loss(candidate: SimulationParams) -> Float64[Array, ""]:
            spectrum: ArpesSpectrum = simulate_novice(
                bands, projections, candidate, 15.0
            )
            loss: Float64[Array, ""] = jnp.sum(spectrum.intensity)
            return loss

        assert_grad_matches_fd(bands_loss, bands, modes=("rev",))
        assert_grad_matches_fd(projection_loss, projections, modes=("rev",))
        assert_grad_matches_fd(params_loss, params, modes=("rev",))
        assert_nonzero_grad(
            bands_loss,
            bands,
            sensitive_paths=(".eigenvalues",),
        )
        assert_nonzero_grad(
            projection_loss,
            projections,
            sensitive_paths=(".projections",),
        )


class TestNoviceScalability:
    """Validate JIT-cache and batched-leaf scalability floors.

    :see: :func:`diffpes.simul.simulate_novice`
    """

    def test_compile_count_tracks_only_static_changes(self) -> None:
        """Compile once for data changes and once per fidelity change.

        Extended Summary
        ----------------
        Same-shaped carrier leaf changes reuse one trace. Changing the static
        ``SimulationParams.fidelity`` field triggers exactly one additional
        trace because it changes the output energy-axis shape.

        Notes
        -----
        A Python counter records traces around the JIT simulation.
        The test blocks completed outputs so asynchronous dispatch cannot hide
        additional traces.
        """
        bands: diffpes.types.BandStructure
        projections: Array
        params: diffpes.types.SimulationParams
        compiled: Callable[..., Any]

        bands, projections, params = _novice_fixture()
        trace_count: list[int] = [0]

        def counted(
            dynamic_bands: BandStructure,
            dynamic_projections: OrbitalProjection,
            dynamic_params: SimulationParams,
        ) -> Array:
            trace_count[0] += 1
            spectrum: ArpesSpectrum = simulate_novice(
                dynamic_bands, dynamic_projections, dynamic_params, 15.0
            )
            result: Array = spectrum.intensity
            return result

        compiled = eqx.filter_jit(counted)
        compiled(bands, projections, params).block_until_ready()
        changed_bands: BandStructure = eqx.tree_at(
            lambda carrier: carrier.eigenvalues,
            bands,
            bands.eigenvalues + 0.1,
        )
        compiled(changed_bands, projections, params).block_until_ready()
        chex.assert_equal(trace_count[0], 1)

        changed_fidelity: SimulationParams = make_simulation_params(
            energy_min=-2.0,
            energy_max=2.0,
            fidelity=20,
            sigma=0.08,
            gamma=0.12,
        )
        compiled(bands, projections, changed_fidelity).block_until_ready()
        chex.assert_equal(trace_count[0], 2)

    def test_vmap_over_sigma_leaf(self) -> None:
        """Vectorize a parameter leaf into ``(P, K, E)`` intensity.

        Extended Summary
        ----------------
        The test maps three Gaussian widths over shared carriers. The output
        retains the parameter, k-point, and energy axes.

        Notes
        -----
        The test builds each mapped parameter carrier with :func:`equinox.tree_at` and
        evaluates the production forward function under :func:`jax.vmap`.
        """
        bands: diffpes.types.BandStructure
        projections: Array
        params: diffpes.types.SimulationParams

        bands, projections, params = _novice_fixture()
        sigmas: Float64[Array, " 3"] = jnp.array([0.04, 0.08, 0.16])

        def simulate_sigma(
            sigma: Float64[Array, ""],
        ) -> Float64[Array, "1 16"]:
            mapped_params: SimulationParams = eqx.tree_at(
                lambda carrier: carrier.sigma, params, sigma
            )
            spectrum: ArpesSpectrum = simulate_novice(
                bands, projections, mapped_params, 15.0
            )
            result: Float64[Array, "1 16"] = spectrum.intensity
            return result

        intensities: Float64[Array, "3 1 16"] = jax.vmap(simulate_sigma)(
            sigmas
        )
        chex.assert_shape(intensities, (3, 1, 16))
        chex.assert_tree_all_finite(intensities)
