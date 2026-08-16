"""Validate compact tight-binding reference models for executable experiments.

The tests cover graphene degeneracy, chain bandwidth, carrier structure, and
first-order derivatives away from a degeneracy.
"""

import jax.numpy as jnp
import pytest
from jax import test_util
from jaxtyping import Array, Float64

from diffpes.harness import (
    graphene_pz_model,
    linear_chain_model,
    two_orbital_dirac_model,
)
from diffpes.tightb import diagonalize_tb
from diffpes.types import TBModel


class TestGraphenePzModel:
    """Validate :func:`~diffpes.harness.graphene_pz_model` output.

    The case scope covers the honeycomb carrier and K-point degeneracy.
    """

    def test_builds_two_bands_degenerate_at_the_honeycomb_k_point(
        self,
    ) -> None:
        """Build two graphene bands degenerate at the honeycomb K point.

        The nearest-neighbor model must have zero energy at fractional K.

        Notes
        -----
        Diagonalize the default model at ``K=(2/3, 1/3, 0)``.
        """
        model: TBModel = graphene_pz_model()
        k_point: Float64[Array, "1 3"] = jnp.asarray(
            [[2.0 / 3.0, 1.0 / 3.0, 0.0]],
            dtype=jnp.float64,
        )
        bands: Float64[Array, "1 2"] = diagonalize_tb(
            model,
            k_point,
        ).eigenvalues

        assert len(model.basis.labels) == 2
        assert jnp.allclose(bands[0], 0.0, rtol=0.0, atol=1.0e-9)

    @pytest.mark.rss_limit_mb(800)
    def test_retains_first_order_hopping_and_lattice_derivatives(self) -> None:
        """Retain first-order hopping and lattice derivatives away from K.

        A fixed Cartesian momentum must expose both physical input
        sensitivities.

        Notes
        -----
        Convert one fixed Cartesian momentum to each model fractional frame.
        """
        cartesian: Float64[Array, "3"] = jnp.asarray(
            [0.31, 0.17, 0.0],
            dtype=jnp.float64,
        )

        def energy(
            hopping_ev: Float64[Array, ""],
            lattice_a_ang: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Return one upper graphene band at fixed Cartesian momentum."""
            hopping_value: Float64[Array, ""] = jnp.asarray(
                hopping_ev,
                dtype=jnp.float64,
            )
            lattice_value: Float64[Array, ""] = jnp.asarray(
                lattice_a_ang,
                dtype=jnp.float64,
            )
            model: TBModel = graphene_pz_model(
                hopping_value,
                lattice_value,
            )
            fractional: Float64[Array, "3"] = cartesian @ jnp.linalg.inv(
                model.geometry.reciprocal
            )
            values: Float64[Array, "1 2"] = diagonalize_tb(
                model,
                fractional[None, :],
            ).eigenvalues
            output: Float64[Array, ""] = values[0, 1]
            return output

        test_util.check_grads(
            energy,
            (jnp.asarray(-2.7), jnp.asarray(2.46)),
            order=1,
            modes=("fwd", "rev"),
            eps=1.0e-5,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


class TestLinearChainModel:
    """Validate :func:`~diffpes.harness.linear_chain_model` output.

    The case scope covers the one-orbital carrier and analytic bandwidth.
    """

    def test_builds_one_band_with_the_analytic_bandwidth(self) -> None:
        """Build one band with the analytic nearest-neighbor bandwidth.

        A chain with hopping ``t`` must span ``4 * abs(t)`` eV.

        Notes
        -----
        Diagonalize the chain at fractional zero and one-half momenta.
        """
        hopping_ev: float = -1.25
        model: TBModel = linear_chain_model(hopping_ev=hopping_ev)
        kpoints: Float64[Array, "2 3"] = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=jnp.float64,
        )
        values: Float64[Array, "2 1"] = diagonalize_tb(
            model,
            kpoints,
        ).eigenvalues
        bandwidth: float = float(jnp.max(values) - jnp.min(values))

        assert len(model.basis.labels) == 1
        assert bandwidth == 4.0 * abs(hopping_ev)


class TestTwoOrbitalDiracModel:
    """Validate :func:`~diffpes.harness.two_orbital_dirac_model` output.

    The case scope covers the compact two-orbital Dirac-like carrier.
    """

    def test_builds_a_two_orbital_model_with_hermitian_hoppings(self) -> None:
        """Build a two-orbital model with Hermitian hopping records.

        The reference model must expose two labels and four hopping amplitudes.

        Notes
        -----
        Build the default model and inspect static carrier dimensions.
        """
        model: TBModel = two_orbital_dirac_model()

        assert len(model.basis.labels) == 2
        assert model.hopping_amplitudes.shape == (4,)
        assert jnp.allclose(
            model.hopping_amplitudes[0],
            jnp.conj(model.hopping_amplitudes[1]),
        )
