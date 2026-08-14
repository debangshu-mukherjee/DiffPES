"""Validate lossless real optimizer views of tight-binding parameters.

The tests pin independent conjugate-pair coordinates, exact reconstruction,
SK-fundamental rebuilding, optional geometry leaves, JIT behavior, and
gradient equivalence with direct parameterizations.
"""

import inspect
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List, Tuple, Union
from jaxtyping import Array, Complex128, Float64

from diffpes.maths import unpack_complex
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


def _materialized_model(
    reverse_residual: complex = 0.0j,
) -> TBModel:
    """PRIVATE: Construct one exact or tolerance-close complex chain model.

    Parameters
    ----------
    reverse_residual : complex
        Deviation in eV added to the conjugate reverse hopping; zero
        yields an exactly Hermitian pair.

    Returns
    -------
    model : TBModel
        One-orbital chain with a complex forward hopping
        ``1.2 - 0.35j`` eV, its perturbed conjugate reverse partner,
        and one home-cell self-reverse record of ``0.7`` eV.

    Notes
    -----
    The parameter-view tests use ``reverse_residual`` to probe the
    conjugate-pair matching tolerance. A zero residual must round-trip
    bitwise. A small residual must follow the documented policy.
    """
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
    amplitudes: Complex128[Array, " 3"] = jnp.asarray(
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


def _graphene_context() -> Tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Float64[Array, " 2"],
]:
    """PRIVATE: Construct the minimal pz Slater--Koster graphene context.

    Returns
    -------
    geometry : CrystalGeometry
        Hexagonal two-carbon cell with lattice constant 2.46 Angstrom.
    basis : OrbitalBasis
        Two-orbital pz basis on the A and B sublattices.
    params : SlaterKosterParams
        The single ``C-C:pp_pi`` integral of -2.7 eV.
    onsite : Float64[Array, " 2"]
        The two distinct onsite energies ``(0.11, -0.09)`` eV.

    Notes
    -----
    This is the smallest SK-fundamental context: one hopping channel
    and two onsite coordinates, so parameter-view round trips stay
    hand-checkable.
    """
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
    onsite: Float64[Array, " 2"] = jnp.asarray(
        (0.11, -0.09),
        dtype=jnp.float64,
    )
    context: Tuple[
        CrystalGeometry,
        OrbitalBasis,
        SlaterKosterParams,
        Float64[Array, " 2"],
    ] = (geometry, basis, params, onsite)
    return context


def _sp_context() -> Tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Float64[Array, " 2"],
]:
    """PRIVATE: Construct an oblique isolated s--px bond with lattice
    sensitivity.

    Returns
    -------
    geometry : CrystalGeometry
        A 10 Angstrom cubic cell with two sites on an oblique bond.
    basis : OrbitalBasis
        One s orbital and one px orbital.
    params : SlaterKosterParams
        The single ``X-X:sp_sigma`` integral of 1.1 eV.
    onsite : Float64[Array, " 2"]
        Onsite energies ``(0.2, -0.1)`` eV.

    Notes
    -----
    The bond direction has nonzero x, y, and z components. The
    direction cosines, and with them the rebuilt hoppings, therefore
    respond to lattice and position perturbations. This makes the
    context suitable for optional-geometry-leaf gradient checks.
    """
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
    onsite: Float64[Array, " 2"] = jnp.asarray(
        (0.2, -0.1),
        dtype=jnp.float64,
    )
    context: Tuple[
        CrystalGeometry,
        OrbitalBasis,
        SlaterKosterParams,
        Float64[Array, " 2"],
    ] = (geometry, basis, params, onsite)
    return context


def _assert_models_bitwise(actual: TBModel, expected: TBModel) -> None:
    """PRIVATE: Compare all traced leaves bitwise and all static fields
    exactly.

    Parameters
    ----------
    actual : TBModel
        Reconstructed model under test.
    expected : TBModel
        Source model that defines the required content.

    Notes
    -----
    Flattens both models with ``jax.tree.leaves``, requires equal leaf
    counts, and asserts exact array equality leaf by leaf. Then checks
    the static basis, hopping pairs, hopping cells, shell index, spinor
    flag, and species tuples directly. Bitwise equality certifies that
    a parameter-view round trip is lossless, not merely close.
    """
    actual_leaves: List[
        Union[Float64[Array, "..."], Complex128[Array, "..."]]
    ] = jax.tree.leaves(actual)
    expected_leaves: List[
        Union[Float64[Array, "..."], Complex128[Array, "..."]]
    ] = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    actual_leaf: Union[Float64[Array, "..."], Complex128[Array, "..."]]
    expected_leaf: Union[Float64[Array, "..."], Complex128[Array, "..."]]
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
    """Validate :func:`~diffpes.tightb.tb_parameter_view`.

    The cases check exact rebuilds, geometry structure, gradients, compilation,
    and invalid parameter vectors.
    """

    def test_round_trip_is_bitwise_and_hoppings_are_independent(self) -> None:
        """Pack one complex pair and one self-reverse record without
        redundancy.

        The case pins every coordinate in the compact real vector.

        Notes
        -----
        Require real/imaginary ordering, exact static metadata, and bitwise
        reconstruction of all numerical leaves.
        """
        model: TBModel = _materialized_model()
        parameters: Float64[Array, " 5"]
        rebuild: Callable[[Float64[Array, " 5"]], TBModel]
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
        parameters: Float64[Array, " 17"]
        rebuild: Callable[[Float64[Array, " 17"]], TBModel]
        parameters, rebuild = tb_parameter_view(
            model,
            include_positions=True,
            include_lattice=True,
        )
        perturbed: Float64[Array, " 17"] = parameters.at[5].add(0.02)
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

    def test_materialized_geometry_is_spectrally_structural(self) -> None:
        """Expose geometry metadata without inventing hopping strain physics.

        At a fixed fractional k-point the materialized hopping values and
        integer cells are held fixed. Basis positions change only the Bloch
        gauge, while lattice rows do not enter the Hamiltonian.

        Notes
        -----
        Differentiate a band-energy invariant with respect to every optional
        geometry coordinate and require the documented structural zero.
        """
        model: TBModel = _materialized_model()
        parameters: Float64[Array, " 17"]
        rebuild: Callable[[Float64[Array, " 17"]], TBModel]
        parameters, rebuild = tb_parameter_view(
            model,
            include_positions=True,
            include_lattice=True,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.231, -0.117, 0.083),
            dtype=jnp.float64,
        )

        def loss(vector: Float64[Array, " 12"]) -> Float64[Array, ""]:
            """Return a spectral invariant with hoppings and k fixed."""
            candidate: TBModel = rebuild(parameters.at[5:].set(vector))
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(candidate, kpoint)
            )
            value: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return value

        derivative: Float64[Array, " 12"] = jax.grad(loss)(parameters[5:])

        np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=1e-13)

    def test_view_gradient_matches_direct_complex_coordinate(self) -> None:
        """Match gradients through the view with a direct analytic chain.

        The case differentiates the same scalar through two parameterizations.

        Notes
        -----
        Compare the two stacked-real hopping derivatives at ``1e-12``.
        """
        model: TBModel = _materialized_model()
        parameters: Float64[Array, " 5"]
        rebuild: Callable[[Float64[Array, " 5"]], TBModel]
        parameters, rebuild = tb_parameter_view(model)
        kpoint: float = 0.231
        phase: complex = np.exp(2.0j * np.pi * kpoint)

        def view_loss(packed: Float64[Array, " 2"]) -> Float64[Array, ""]:
            """Evaluate the scalar band loss through the inverse view."""
            vector: Float64[Array, " 5"] = parameters.at[:2].set(packed)
            candidate: TBModel = rebuild(vector)
            hamiltonian: Complex128[Array, "1 1"] = bloch_hamiltonian(
                candidate,
                jnp.asarray((kpoint, 0.0, 0.0), dtype=jnp.float64),
            )
            value: Float64[Array, ""] = jnp.real(hamiltonian[0, 0]) ** 2
            return value

        def direct_loss(packed: Float64[Array, " 2"]) -> Float64[Array, ""]:
            """Evaluate the same scalar band loss from the closed form."""
            amplitude: Complex128[Array, ""] = unpack_complex(packed)
            energy: Complex128[Array, ""] = (
                model.onsite_energies[0]
                + jnp.real(model.hopping_amplitudes[2])
                + amplitude * phase
                + jnp.conj(amplitude) * np.conj(phase)
            )
            value: Float64[Array, ""] = jnp.real(energy) ** 2
            return value

        actual: Float64[Array, " 2"] = jax.grad(view_loss)(parameters[:2])
        expected: Float64[Array, " 2"] = jax.grad(direct_loss)(parameters[:2])

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

        parameters: Float64[Array, " 5"]
        rebuild: Callable[[Float64[Array, " 5"]], TBModel]
        parameters, rebuild = tb_parameter_view(_materialized_model())
        with pytest.raises(ValueError, match="must have shape"):
            rebuild(jnp.zeros((parameters.size + 1,), dtype=jnp.float64))
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="parameters finite",
        ):
            rebuild(parameters.at[0].set(jnp.nan))


class TestSKModelParameterView:
    """Validate :func:`~diffpes.tightb.sk_model_parameter_view`.

    The cases check Slater--Koster rebuilds, gradients, topology capture,
    static validation, and gauge documentation.
    """

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
        onsite: Float64[Array, " 2"]
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
        parameters: Float64[Array, " 3"]
        rebuild: Callable[[Float64[Array, " 3"]], TBModel]
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

        positioned: Float64[Array, " 9"]
        rebuild_positioned: Callable[[Float64[Array, " 9"]], TBModel]
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
        shifted: Float64[Array, " 9"] = positioned.at[-3].add(1e-4)
        shifted_model: TBModel = rebuild_positioned(shifted)
        np.testing.assert_array_equal(
            shifted_model.geometry.positions,
            geometry.positions.at[1, 0].add(1e-4),
        )

        geometric: Float64[Array, " 18"]
        rebuild_geometric: Callable[[Float64[Array, " 18"]], TBModel]
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
        strained: Float64[Array, " 18"] = geometric.at[9].add(1e-4)
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
        onsite: Float64[Array, " 2"]
        geometry, basis, params, onsite = _graphene_context()
        parameters: Float64[Array, " 3"]
        rebuild: Callable[[Float64[Array, " 3"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            1.5,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.173, -0.081, 0.0),
            dtype=jnp.float64,
        )

        def band_loss(model: TBModel) -> Float64[Array, ""]:
            """Return a gauge-invariant spectral polynomial."""
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            result: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return result

        def through_view(value: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return bands after rebuilding from optimizer coordinates."""
            vector: Float64[Array, " 3"] = parameters.at[0].set(value)
            result: Float64[Array, ""] = band_loss(rebuild(vector))
            return result

        def direct(value: Float64[Array, ""]) -> Float64[Array, ""]:
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
            result: Float64[Array, ""] = band_loss(model)
            return result

        actual: Float64[Array, ""] = jax.grad(through_view)(parameters[0])
        expected: Float64[Array, ""] = jax.grad(direct)(parameters[0])

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
        onsite: Float64[Array, " 2"]
        geometry, basis, params, onsite = _sp_context()
        parameters: Float64[Array, " 12"]
        rebuild: Callable[[Float64[Array, " 12"]], TBModel]
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
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.17, -0.09, 0.03),
            dtype=jnp.float64,
        )

        def band_loss(model: TBModel) -> Float64[Array, ""]:
            """Return a spectral invariant with bond-direction sensitivity."""
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            result: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return result

        def through_view(value: Float64[Array, ""]) -> Float64[Array, ""]:
            """Replace one lattice coordinate in the flat view."""
            vector: Float64[Array, " 12"] = parameters.at[lattice_offset].set(
                value
            )
            result: Float64[Array, ""] = band_loss(rebuild(vector))
            return result

        def direct(value: Float64[Array, ""]) -> Float64[Array, ""]:
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
            result: Float64[Array, ""] = band_loss(model)
            return result

        initial: Float64[Array, ""] = parameters[lattice_offset]
        actual: Float64[Array, ""] = jax.grad(through_view)(initial)
        expected: Float64[Array, ""] = jax.grad(direct)(initial)

        assert jnp.abs(actual) > 1e-8
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_geometry_null_directions_with_distance_independent_sk(
        self,
    ) -> None:
        """Pin translation and isotropic-scale structural null directions.

        The fixed neighbor topology, cutoff, fractional k-point, SK values,
        onsite values, and SOC values are held fixed. Because the built-in SK
        law depends on bond direction but not bond length, a uniform
        translation or dilation cannot change the band spectrum.

        Notes
        -----
        The neighboring shear test supplies the retained nonzero angular
        sensitivity. Here autodiff must return zero for the two absent
        physical channels instead of implying radial strain sensitivity.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float64[Array, " 2"]
        geometry, basis, params, onsite = _sp_context()
        parameters: Float64[Array, " 18"]
        rebuild: Callable[[Float64[Array, " 18"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            2.0,
            include_positions=True,
            include_lattice=True,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.17, -0.09, 0.03),
            dtype=jnp.float64,
        )
        position_slice: slice = slice(3, 9)
        lattice_slice: slice = slice(9, 18)

        def band_loss(
            vector: Float64[Array, " 18"],
        ) -> Float64[Array, ""]:
            """Return a spectral invariant at fixed fractional k."""
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(rebuild(vector), kpoint)
            )
            value: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return value

        def translated(
            shift: Float64[Array, " 3"],
        ) -> Float64[Array, ""]:
            """Return equally translated fractional basis positions."""
            positions: Float64[Array, "2 3"] = jnp.reshape(
                parameters[position_slice],
                (2, 3),
            )
            vector: Float64[Array, " 18"] = parameters.at[position_slice].set(
                jnp.ravel(positions + shift[None, :])
            )
            value: Float64[Array, ""] = band_loss(vector)
            return value

        def dilated(scale: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return lattice rows dilated by one scale."""
            vector: Float64[Array, " 18"] = parameters.at[lattice_slice].set(
                scale * parameters[lattice_slice]
            )
            value: Float64[Array, ""] = band_loss(vector)
            return value

        translation_gradient: Float64[Array, " 3"] = jax.grad(translated)(
            jnp.zeros((3,), dtype=jnp.float64)
        )
        dilation_gradient: Float64[Array, ""] = jax.grad(dilated)(
            jnp.asarray(1.0, dtype=jnp.float64)
        )

        np.testing.assert_allclose(
            translation_gradient,
            0.0,
            rtol=0.0,
            atol=1e-13,
        )
        assert float(dilation_gradient) == pytest.approx(0.0, abs=1e-13)

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
        onsite: Float64[Array, " 2"]
        geometry, basis, params, onsite = _graphene_context()
        parameters: Float64[Array, " 3"]
        rebuild: Callable[[Float64[Array, " 3"]], TBModel]
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

    @pytest.mark.rss_limit_mb(700)
    def test_jit_gradient_uses_captured_geometry_topology(self) -> None:
        """Compile geometry-sensitive rebuilding on frozen neighbor cells.

        ``jit`` traces every position and lattice coordinate.
        Setup must capture certified pairs, cells, and shell numbers first.

        Notes
        -----
        This is the optimizer counterexample to rediscovering neighbors from
        traced geometry. Require an identical compiled model and finite,
        nonzero position and lattice derivatives that match eager automatic
        differentiation.
        """
        geometry: CrystalGeometry
        basis: OrbitalBasis
        params: SlaterKosterParams
        onsite: Float64[Array, " 2"]
        geometry, basis, params, onsite = _sp_context()
        parameters: Float64[Array, " 18"]
        rebuild: Callable[[Float64[Array, " 18"]], TBModel]
        parameters, rebuild = sk_model_parameter_view(
            geometry,
            basis,
            params,
            onsite,
            jnp.zeros((0,), dtype=jnp.float64),
            (-1, -1),
            2.0,
            include_positions=True,
            include_lattice=True,
        )
        kpoint: Float64[Array, " 3"] = jnp.asarray(
            (0.17, -0.09, 0.03),
            dtype=jnp.float64,
        )

        def loss(vector: Float64[Array, " 18"]) -> Float64[Array, ""]:
            """Return a spectral invariant from the frozen-topology view."""
            model: TBModel = rebuild(vector)
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            value: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            return value

        def loss_with_model(
            vector: Float64[Array, " 18"],
        ) -> Tuple[Float64[Array, ""], TBModel]:
            """Return the spectral invariant and rebuilt auxiliary model."""
            model: TBModel = rebuild(vector)
            eigenvalues: Float64[Array, " n_orb"] = jnp.linalg.eigvalsh(
                bloch_hamiltonian(model, kpoint)
            )
            value: Float64[Array, ""] = jnp.sum(eigenvalues**2)
            result: Tuple[Float64[Array, ""], TBModel] = (value, model)
            return result

        expected_value: Float64[Array, ""]
        expected_gradient: Float64[Array, " 18"]
        expected_value, expected_gradient = jax.value_and_grad(loss)(
            parameters
        )
        expected_model: TBModel = rebuild(parameters)
        actual_value: Float64[Array, ""]
        actual_model: TBModel
        actual_gradient: Float64[Array, " 18"]
        (actual_value, actual_model), actual_gradient = jax.jit(
            jax.value_and_grad(loss_with_model, has_aux=True)
        )(parameters)

        _assert_models_bitwise(actual_model, expected_model)
        assert jnp.all(jnp.isfinite(actual_gradient))
        assert jnp.abs(actual_gradient[6]) > 1e-8
        assert jnp.abs(actual_gradient[9]) > 1e-8
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-12,
            atol=1e-13,
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
