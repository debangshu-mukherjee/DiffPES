"""Validate the slaterkoster model module.

The cases use analytic values, invariants, and finite differences.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64

from diffpes.tightb import (
    bloch_hamiltonian,
    build_sk_model,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_slater_koster_params,
)
from tests._gradients import assert_grad_matches_fd, assert_nonzero_grad

from ._slaterkoster_helpers import (
    _ALL_SK_KEYS,
    _compact_spd_basis,
    _graphene_geometry,
)


class TestBuildSkModel:
    """Validate :func:`diffpes.tightb.build_sk_model`.

    The cases check shell keys, Hermitian closure, exact cells, traced
    geometry, and spectral gradients.
    """

    def test_distance_shell_keys_select_distinct_integrals(self) -> None:
        """Verify independent first- and second-neighbor chain hoppings.

        Distinct distance-shell keys must select their matching amplitudes.

        Notes
        -----
        Pin the one-based ``@N`` key grammar and setup-time distance binning.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((1.0, 10.0, 10.0), dtype=jnp.float64)),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0,),
            (1,),
            (0,),
            (0,),
            labels=("X_s",),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-1.0, -0.3), dtype=jnp.float64),
            ("X-X@1:ss_sigma", "X-X@2:ss_sigma"),
        )
        model: TBModel = build_sk_model(
            geometry,
            basis,
            params,
            jnp.zeros((1,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            (-1,),
            2.1,
        )
        by_cell: Dict[Tuple[int, int, int], complex] = dict(
            zip(
                model.hopping_cells,
                np.asarray(model.hopping_amplitudes),
                strict=True,
            )
        )

        assert by_cell[(-1, 0, 0)] == -1.0
        assert by_cell[(1, 0, 0)] == -1.0
        assert by_cell[(-2, 0, 0)] == -0.3
        assert by_cell[(2, 0, 0)] == -0.3

    def test_graphene_model_is_closed_and_uses_exact_cells(self) -> None:
        """Build the three-bond pz graphene model from one pi integral.

        The builder must emit both orientations for every nearest-neighbor
        bond.

        Notes
        -----
        Require the six directed records and their exact conjugate closure.
        """
        geometry: CrystalGeometry = _graphene_geometry()
        basis: OrbitalBasis = make_orbital_basis(
            (0, 1),
            (2, 2),
            (1, 1),
            (0, 0),
            labels=("A_pz", "B_pz"),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-2.7,), dtype=jnp.float64),
            ("C-C:pp_pi",),
        )
        model: TBModel = build_sk_model(
            geometry,
            basis,
            params,
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )

        assert model.hopping_pairs == (
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
        )
        assert model.hopping_cells == (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        np.testing.assert_allclose(
            model.hopping_amplitudes,
            -2.7,
            rtol=0.0,
            atol=1e-14,
        )

    def test_jit_rejects_uncertified_traced_geometry(self) -> None:
        """Reject neighbor discovery when compilation hides its geometry.

        The compiled lattice lacks a concrete primal, so the host cannot
        freeze a singular-value topology certificate.

        Notes
        -----
        Static-geometry compiled rebuilds remain supported because host setup
        selects their topology before tracing.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.eye(3, dtype=jnp.float64),
            jnp.zeros((1, 3), dtype=jnp.float64),
            ("X",),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0,),
            (1,),
            (0,),
            (0,),
            labels=("X_s",),
        )
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((-1.0,), dtype=jnp.float64),
            ("X-X:ss_sigma",),
        )

        def hopping(
            lattice: Float64[Array, "3 3"],
        ) -> Float64[Array, " n_hop"]:
            """Build hopping amplitudes from a traced lattice."""
            candidate: CrystalGeometry = eqx.tree_at(
                lambda item: item.lattice,
                geometry,
                lattice,
            )
            model: TBModel = build_sk_model(
                candidate,
                basis,
                params,
                jnp.zeros((1,), dtype=jnp.float64),
                jnp.zeros((0,), dtype=jnp.float64),
                (-1,),
                1.1,
            )
            amplitudes: Float64[Array, " n_hop"] = jnp.real(
                model.hopping_amplitudes
            )
            return amplitudes

        with pytest.raises(
            ValueError,
            match=(
                "cannot certify neighbor topology from fully traced geometry"
            ),
        ):
            jax.jit(hopping)(geometry.lattice)

    @pytest.mark.parametrize("pole", [1.0, -1.0])
    def test_position_gradient_flows_through_frozen_topology(
        self,
        pole: float,
    ) -> None:
        """Differentiate an assembled s--px hopping at both bond poles.

        The setup selects the neighbor tuple from the concrete AD primal while
        the bond direction remains traced.

        Notes
        -----
        Compare the fractional-position derivative with the analytic Cartesian
        derivative times the lattice-vector length.
        """
        lattice: Float64[Array, "3 3"] = jnp.diag(
            jnp.asarray((10.0, 10.0, 10.0), dtype=jnp.float64)
        )
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice,
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.2 * pole)),
                dtype=jnp.float64,
            ),
            ("A", "B"),
        )
        basis: OrbitalBasis = make_orbital_basis(
            (0, 1),
            (1, 2),
            (0, 1),
            (0, 1),
            labels=("A_s", "B_px"),
        )
        integral: float = 1.4
        params: SlaterKosterParams = make_slater_koster_params(
            jnp.asarray((integral,), dtype=jnp.float64),
            ("A-B:sp_sigma",),
        )

        def hopping(
            positions: Float64[Array, "2 3"],
        ) -> Float64[Array, ""]:
            """Return the forward s--px hopping on frozen topology."""
            candidate: CrystalGeometry = eqx.tree_at(
                lambda item: item.positions,
                geometry,
                positions,
            )
            model: TBModel = build_sk_model(
                candidate,
                basis,
                params,
                jnp.zeros((2,), dtype=jnp.float64),
                jnp.zeros((0,), dtype=jnp.float64),
                (-1, -1),
                3.0,
            )
            value: Float64[Array, ""] = jnp.real(model.hopping_amplitudes[0])
            return value

        gradient: Float64[Array, "2 3"] = jax.grad(hopping)(geometry.positions)

        np.testing.assert_allclose(
            gradient[:, 0],
            jnp.asarray((-5.0 * integral, 5.0 * integral)),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            gradient[:, 1:],
            0.0,
            rtol=0.0,
            atol=1e-13,
        )

    @pytest.mark.rss_limit_mb(900)
    def test_every_integral_has_fd_correct_band_spectral_gradient(
        self,
    ) -> None:
        """Differentiate a band spectral loss with respect to all ten values.

        Generic s, px, and dxy orbitals on two atoms exercise every
        fundamental integral without a needlessly large eigensystem. Squared
        eigenvalues form a gauge-invariant band loss with nonzero sensitivity
        to every channel.

        Notes
        -----
        Apply the program-wide f64 finite-difference harness in forward and
        reverse mode, then enforce the zero-gradient tripwire.
        """
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.asarray((5.0, 6.0, 7.0), dtype=jnp.float64)),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.21, 0.17, 0.13)),
                dtype=jnp.float64,
            ),
            ("X", "X"),
        )
        basis: OrbitalBasis = _compact_spd_basis()
        initial: Float64[Array, " 10"] = jnp.asarray(
            (-0.8, 1.1, -0.7, 1.5, -0.4, 0.9, -0.3, 1.2, -0.6, 0.2),
            dtype=jnp.float64,
        )
        onsite: Float64[Array, " 6"] = jnp.linspace(
            -0.35,
            0.42,
            6,
            dtype=jnp.float64,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.137, -0.219, 0.083),
            dtype=jnp.float64,
        )

        def spectral_loss(values: Float64[Array, " 10"]) -> Float64[Array, ""]:
            """Return the sum of squared tight-binding band energies."""
            params: SlaterKosterParams = make_slater_koster_params(
                values,
                _ALL_SK_KEYS,
            )
            model: TBModel = build_sk_model(
                geometry,
                basis,
                params,
                onsite,
                jnp.zeros((0,), dtype=jnp.float64),
                (-1,) * 6,
                2.0,
            )
            eigenvalues: Float64[Array, " 6"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            loss: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return loss

        assert_grad_matches_fd(spectral_loss, initial)
        assert_nonzero_grad(spectral_loss, initial, elementwise=True)
