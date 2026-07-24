"""Validate lossless real optimizer views of tight-binding parameters.

The tests pin independent conjugate-pair coordinates, exact reconstruction,
SK-fundamental rebuilding, optional geometry leaves, JIT behavior, and
gradient equivalence with direct parameterizations.
"""

import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Complex, Float

from diffpes.tightb import (
    bloch_hamiltonian,
    build_sk_model,
    sk_model_parameter_view,
    tb_parameter_view,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_slater_koster_params,
    make_tb_model,
)
from diffpes.utils import unpack_complex


def _materialized_model(
    reverse_residual: complex = 0.0j,
) -> TBModel:
    """Construct one exact or tolerance-close complex chain model."""
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.diag(jnp.asarray((2.0, 5.0, 6.0), dtype=jnp.float64)),
        jnp.asarray(((0.13, 0.07, 0.19),), dtype=jnp.float64),
        ("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        (0,),
        (1,),
        (0,),
        (0,),
        labels=("X_s",),
    )
    forward: complex = complex(1.2, -0.35)
    self_reverse: complex = complex(0.7, -0.0)
    amplitudes: Complex[Array, " 3"] = jnp.asarray(
        (
            forward,
            np.conj(forward) + reverse_residual,
            self_reverse,
        ),
        dtype=jnp.complex128,
    )
    model: TBModel = make_tb_model(
        amplitudes,
        jnp.asarray((-0.25,), dtype=jnp.float64),
        jnp.asarray((0.0,), dtype=jnp.float64),
        geometry,
        basis,
        ((0, 0), (0, 0), (0, 0)),
        ((1, 0, 0), (-1, 0, 0), (0, 0, 0)),
        (0,),
    )
    return model


def _graphene_context() -> tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Float[Array, " 2"],
]:
    """Construct the minimal pz Slater--Koster graphene context."""
    lattice_constant: float = 2.46
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.asarray(
            (
                (lattice_constant, 0.0, 0.0),
                (
                    lattice_constant / 2.0,
                    lattice_constant * np.sqrt(3.0) / 2.0,
                    0.0,
                ),
                (0.0, 0.0, 10.0),
            ),
            dtype=jnp.float64,
        ),
        jnp.asarray(
            ((0.0, 0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0, 0.0)),
            dtype=jnp.float64,
        ),
        ("C", "C"),
    )
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
    onsite: Float[Array, " 2"] = jnp.asarray(
        (0.11, -0.09),
        dtype=jnp.float64,
    )
    return geometry, basis, params, onsite


def _sp_context() -> tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Float[Array, " 2"],
]:
    """Construct an oblique isolated s--px bond with lattice sensitivity."""
    geometry: CrystalGeometry = make_crystal_geometry(
        10.0 * jnp.eye(3, dtype=jnp.float64),
        jnp.asarray(
            ((0.0, 0.0, 0.0), (0.12, 0.08, 0.05)),
            dtype=jnp.float64,
        ),
        ("X", "X"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        (0, 1),
        (1, 2),
        (0, 1),
        (0, 1),
        labels=("X_s", "X_px"),
    )
    params: SlaterKosterParams = make_slater_koster_params(
        jnp.asarray((1.1,), dtype=jnp.float64),
        ("X-X:sp_sigma",),
    )
    onsite: Float[Array, " 2"] = jnp.asarray(
        (0.2, -0.1),
        dtype=jnp.float64,
    )
    return geometry, basis, params, onsite


def _assert_models_bitwise(actual: TBModel, expected: TBModel) -> None:
    """Compare all traced leaves bitwise and all static fields exactly."""
    actual_leaves: list[jax.Array] = jax.tree.leaves(actual)
    expected_leaves: list[jax.Array] = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    actual_leaf: jax.Array
    expected_leaf: jax.Array
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    assert actual.basis == expected.basis
    assert actual.hopping_pairs == expected.hopping_pairs
    assert actual.hopping_cells == expected.hopping_cells
    assert actual.shell_index == expected.shell_index
    assert actual.spinor is expected.spinor
    assert actual.geometry.species == expected.geometry.species


class TestTBParameterView:
    """Validate :func:`~diffpes.tightb.tb_parameter_view`."""

    def test_round_trip_is_bitwise_and_hoppings_are_independent(self) -> None:
        """Pack one complex pair and one self-reverse record without redundancy.

        The case pins every coordinate in the compact real vector.

        Notes
        -----
        Require real/imaginary ordering, exact static metadata, and bitwise
        reconstruction of all numerical leaves.
        """
        model: TBModel = _materialized_model()
        parameters: Float[Array, " 5"]
        rebuild: Callable[[Float[Array, " 5"]], TBModel]
        parameters, rebuild = tb_parameter_view(model)

        np.testing.assert_array_equal(
            parameters,
            jnp.asarray(
                (1.2, -0.35, 0.7, -0.25, 0.0),
                dtype=jnp.float64,
            ),
        )
        rebuilt: TBModel = rebuild(parameters)

        _assert_models_bitwise(rebuilt, model)
        assert np.signbit(
            np.asarray(rebuilt.hopping_amplitudes[2]).imag
        ) == np.signbit(np.asarray(model.hopping_amplitudes[2]).imag)

    def test_optional_geometry_and_jit_rebuild(self) -> None:
        """Append fractional positions and lattice rows through JIT.

        The case checks geometry reconstruction alongside unchanged hoppings.

        Notes
        -----
        Perturb one position and one lattice coordinate while preserving the
        static hopping topology. Require reciprocal-lattice recomputation.
        """
        model: TBModel = _materialized_model()
        parameters: Float[Array, " 17"]
        rebuild: Callable[[Float[Array, " 17"]], TBModel]
        parameters, rebuild = tb_parameter_view(
            model,
            include_positions=True,
            include_lattice=True,
        )
        perturbed: Float[Array, " 17"] = parameters.at[5].add(0.02)
        perturbed = perturbed.at[8].add(0.1)
        rebuilt: TBModel = jax.jit(rebuild)(perturbed)

        assert parameters.shape == (17,)
        np.testing.assert_array_equal(
            rebuilt.geometry.positions,
            model.geometry.positions.at[0, 0].add(0.02),
        )
        np.testing.assert_array_equal(
            rebuilt.geometry.lattice,
            model.geometry.lattice.at[0, 0].add(0.1),
        )
        np.testing.assert_allclose(
            rebuilt.geometry.reciprocal,
            2.0 * jnp.pi * jnp.linalg.inv(rebuilt.geometry.lattice).T,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            rebuilt.hopping_amplitudes,
            model.hopping_amplitudes,
        )

    def test_view_gradient_matches_direct_complex_coordinate(self) -> None:
        """Match gradients through the view with a direct analytic chain.

        The case differentiates the same scalar through two parameterizations.

        Notes
        -----
        Compare the two stacked-real hopping derivatives at ``1e-12``.
        """
        model: TBModel = _materialized_model()
        parameters: Float[Array, " 5"]
        rebuild: Callable[[Float[Array, " 5"]], TBModel]
        parameters, rebuild = tb_parameter_view(model)
        kpoint: float = 0.231
        phase: complex = np.exp(2.0j * np.pi * kpoint)

        def view_loss(packed: Float[Array, " 2"]) -> jax.Array:
            """Evaluate the scalar band loss through the inverse view."""
            vector: Float[Array, " 5"] = parameters.at[:2].set(packed)
            candidate: TBModel = rebuild(vector)
            hamiltonian: Complex[Array, "1 1"] = bloch_hamiltonian(
                candidate,
                jnp.asarray((kpoint, 0.0, 0.0), dtype=jnp.float64),
            )
            return jnp.real(hamiltonian[0, 0]) ** 2

        def direct_loss(packed: Float[Array, " 2"]) -> jax.Array:
            """Evaluate the same scalar band loss from the closed form."""
            amplitude: Complex[Array, ""] = unpack_complex(packed)
            energy: Complex[Array, ""] = (
                model.onsite_energies[0]
                + jnp.real(model.hopping_amplitudes[2])
                + amplitude * phase
                + jnp.conj(amplitude) * np.conj(phase)
            )
            return jnp.real(energy) ** 2

        actual: Float[Array, " 2"] = jax.grad(view_loss)(parameters[:2])
        expected: Float[Array, " 2"] = jax.grad(direct_loss)(parameters[:2])

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_rejects_tolerance_close_projection_and_invalid_vectors(
        self,
    ) -> None:
        """Reject lossy near-closure and malformed rebuilding coordinates.

        The case distinguishes carrier tolerance from optimizer exactness.

        Notes
        -----
        The carrier tolerance admits the constructed residual, but the
        optimizer view refuses to project it onto exact Hermiticity.
        """
        tolerance_close: TBModel = _materialized_model(
            reverse_residual=1.0e-13j
        )
        with pytest.raises(ValueError, match="exactly conjugate-closed"):
            tb_parameter_view(tolerance_close)

        parameters: Float[Array, " 5"]
        rebuild: Callable[[Float[Array, " 5"]], TBModel]
        parameters, rebuild = tb_parameter_view(_materialized_model())
        with pytest.raises(ValueError, match="must have shape"):
            rebuild(jnp.zeros((parameters.size + 1,), dtype=jnp.float64))
        with pytest.raises(Exception, match="parameters finite"):
            rebuild(parameters.at[0].set(jnp.nan))


class TestSKModelParameterView:
    """Validate :func:`~diffpes.tightb.sk_model_parameter_view`."""

    def test_round_trip_and_position_layout(self) -> None:
        """Return the initial graphene SK model and append positions last.

        The case pins the optional position and lattice coordinate layout.

        Notes
        -----
        Compare the non-position view bitwise and pin the optional flat layout.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float[Array, " 2"]
        geometry, basis, params, onsite = _graphene_context()
        expected: TBModel = build_sk_model(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )
        parameters: Float[Array, " 3"]
        rebuild: Callable[[Float[Array, " 3"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )
        _assert_models_bitwise(rebuild(parameters), expected)

        positioned: Float[Array, " 9"]
        rebuild_positioned: Callable[[Float[Array, " 9"]], TBModel]
        positioned, rebuild_positioned = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
            include_positions=True,
        )
        assert parameters.shape == (3,)
        assert positioned.shape == (9,)
        np.testing.assert_array_equal(positioned[:3], parameters)
        shifted: Float[Array, " 9"] = positioned.at[-3].add(1e-4)
        shifted_model: TBModel = rebuild_positioned(shifted)
        np.testing.assert_array_equal(
            shifted_model.geometry.positions,
            geometry.positions.at[1, 0].add(1e-4),
        )

        geometric: Float[Array, " 18"]
        rebuild_geometric: Callable[[Float[Array, " 18"]], TBModel]
        geometric, rebuild_geometric = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
            include_positions=True,
            include_lattice=True,
        )
        assert geometric.shape == (18,)
        np.testing.assert_array_equal(geometric[:9], positioned)
        strained: Float[Array, " 18"] = geometric.at[9].add(1e-4)
        strained_model: TBModel = rebuild_geometric(strained)
        np.testing.assert_array_equal(
            strained_model.geometry.lattice,
            geometry.lattice.at[0, 0].add(1e-4),
        )
        np.testing.assert_allclose(
            strained_model.geometry.reciprocal,
            2.0 * jnp.pi * jnp.linalg.inv(strained_model.geometry.lattice).T,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_gradient_through_view_matches_direct_sk_value(self) -> None:
        """Match the SK-integral band gradient through the rebuilding view.

        The case compares an optimizer view against direct model construction.

        Notes
        -----
        Hold onsite coordinates fixed and compare the fundamental pp-pi
        derivative at ``1e-12``.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float[Array, " 2"]
        geometry, basis, params, onsite = _graphene_context()
        parameters: Float[Array, " 3"]
        rebuild: Callable[[Float[Array, " 3"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )
        kpoint: Float[Array, " 3"] = jnp.asarray(
            (0.173, -0.081, 0.0),
            dtype=jnp.float64,
        )

        def band_loss(model: TBModel) -> jax.Array:
            """Return a gauge-invariant spectral polynomial."""
            eigenvalues: Float[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            result: jax.Array = jnp.sum(eigenvalues**2)
            return result

        def through_view(value: Float[Array, ""]) -> jax.Array:
            """Return bands after rebuilding from optimizer coordinates."""
            vector: Float[Array, " 3"] = parameters.at[0].set(value)
            result: jax.Array = band_loss(rebuild(vector))
            return result

        def direct(value: Float[Array, ""]) -> jax.Array:
            """Return bands rebuilt from the fundamental SK value."""
            direct_params: SlaterKosterParams = SlaterKosterParams(
                values=jnp.reshape(value, (1,)),
                keys=params.keys,
            )
            model: TBModel = build_sk_model(
                geometry,
                basis,
                direct_params,
                onsite,
                jnp.zeros((0,), dtype=jnp.float64),
                (-1, -1),
                1.5,
            )
            result: jax.Array = band_loss(model)
            return result

        actual: Float[Array, ""] = jax.grad(through_view)(parameters[0])
        expected: Float[Array, ""] = jax.grad(direct)(parameters[0])

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_lattice_gradient_through_view_matches_direct_builder(
        self,
    ) -> None:
        """Match a nonzero strain derivative through the SK view.

        The case compares flat lattice coordinates with direct geometry input.

        Notes
        -----
        An oblique s--px bond makes its direction cosine sensitive to the
        first lattice row. Compare the flat-view derivative with direct
        reconstruction and reject a silently zero strain channel.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float[Array, " 2"]
        geometry, basis, params, onsite = _sp_context()
        parameters: Float[Array, " 12"]
        rebuild: Callable[[Float[Array, " 12"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            2.0,
            include_lattice=True,
        )
        lattice_offset: int = 3
        kpoint: Float[Array, " 3"] = jnp.asarray(
            (0.17, -0.09, 0.03),
            dtype=jnp.float64,
        )

        def band_loss(model: TBModel) -> jax.Array:
            """Return a spectral invariant with bond-direction sensitivity."""
            eigenvalues: Float[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            result: jax.Array = jnp.sum(eigenvalues**2)
            return result

        def through_view(value: Float[Array, ""]) -> jax.Array:
            """Replace one lattice coordinate in the flat view."""
            vector: Float[Array, " 12"] = parameters.at[lattice_offset].set(
                value
            )
            result: jax.Array = band_loss(rebuild(vector))
            return result

        def direct(value: Float[Array, ""]) -> jax.Array:
            """Return bands rebuilt from the same lattice coordinate."""
            direct_geometry: CrystalGeometry = make_crystal_geometry(
                geometry.lattice.at[0, 0].set(value),
                geometry.positions,
                geometry.species,
            )
            model: TBModel = build_sk_model(
                direct_geometry,
                basis,
                params,
                onsite,
                jnp.zeros((0,), dtype=jnp.float64),
                (-1, -1),
                2.0,
            )
            result: jax.Array = band_loss(model)
            return result

        initial: Float[Array, ""] = parameters[lattice_offset]
        actual: Float[Array, ""] = jax.grad(through_view)(initial)
        expected: Float[Array, ""] = jax.grad(direct)(initial)

        assert jnp.abs(actual) > 1e-8
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_jit_and_static_vector_validation(self) -> None:
        """Compile the rebuilding closure and reject a wrong vector length.

        The case confirms one captured topology supports compiled rebuilding.

        Notes
        -----
        The captured geometry and cutoff fix the static topology.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float[Array, " 2"]
        geometry, basis, params, onsite = _graphene_context()
        parameters: Float[Array, " 3"]
        rebuild: Callable[[Float[Array, " 3"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )
        rebuilt: TBModel = jax.jit(rebuild)(parameters)

        np.testing.assert_array_equal(
            rebuilt.hopping_amplitudes,
            -2.7,
        )
        with pytest.raises(ValueError, match="must have shape"):
            jax.jit(rebuild)(
                jnp.zeros((parameters.size + 1,), dtype=jnp.float64)
            )

    def test_docstrings_register_the_energy_zero_gauge(self) -> None:
        """Keep the identifiability warning visible on both public views.

        The case protects the inversion-facing gauge documentation.

        Notes
        -----
        Inspect both public docstrings for the registered warning phrase.
        """
        tb_doc: str = inspect.getdoc(tb_parameter_view)
        sk_doc: str = inspect.getdoc(sk_model_parameter_view)

        assert "band-energy-zero gauge" in tb_doc
        assert "band-energy-zero gauge" in sk_doc
