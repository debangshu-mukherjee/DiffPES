"""Validate coherent orbital and band photoemission matrix elements.

Extended Summary
----------------
The tests use analytic centre phases, attenuation ratios, pole values, and
generic-complex projections to expose phase, conjugation, spin-reduction, and
coordinate-frame mistakes.  They also exercise the complete fixed-radial
assembler and its explicit vacuum-momentum boundary.
"""

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Bool, Complex, Float

from diffpes.maths import real_spherical_harmonics_all
from diffpes.radial import evaluate_radial
from diffpes.simul import (
    assemble_orbital_transition_channels,
    band_group_weight_sensitivity,
    contract_experiment_polarization,
    contract_polarization,
    log_band_group_weight_sensitivity,
    matrix_element_intensity,
    matrix_element_phase_gauge_direction,
    orbital_transition_channels,
    pack_matrixel_params,
    project_band_channels,
    radial_coefficient_scale_gauge_directions,
    real_spherical_harmonics_cartesian_all,
    resolve_orbital_positions_cart,
    transition_source,
    unpack_matrixel_params,
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

type MatrixFixture = tuple[
    DiagonalizedBands,
    RadialSpec,
    MatrixElementParams,
    RadialQuadratureSpec,
    FinalStateSpec,
    ExperimentGeometry,
]


def _s_basis(
    atom_indices: tuple[int, ...],
    spin: tuple[int, ...] = (),
) -> OrbitalBasis:
    """Return a real s-orbital basis for analytic fixtures."""
    n_orbitals: int = len(atom_indices)
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=atom_indices,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        spin=spin,
    )
    return basis


def _matrix_params(
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
) -> MatrixElementParams:
    """Return unit-scale, zero-phase shell parameters."""
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        shell_index,
    )
    return params


def _packing_fixture() -> tuple[
    RadialSpec,
    MatrixElementParams,
    Float[Array, ""],
]:
    """Return a two-shell Slater packing fixture."""
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 2, 2),
        l=(0, 1, 1, 1),
        m=(0, -1, 0, 1),
    )
    shell_index: tuple[int, ...] = (0, 1, 1, 1)
    radial: RadialSpec = make_radial_spec(
        basis,
        shell_index,
        zeta_shell=jnp.array([[1.2, 2.1], [0.9, 1.7]]),
        coefficients_shell=jnp.array([[0.8, -0.3], [0.6, 0.4]]),
    )
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        shell_index,
        sigma_shell=jnp.array([1.3, 0.7]),
        phase_shift_angles_shell=jnp.array([0.2, -0.4, 0.6]),
    )
    mean_free_path: Float[Array, ""] = jnp.array(8.5)
    fixture: tuple[
        RadialSpec,
        MatrixElementParams,
        Float[Array, ""],
    ] = (radial, params, mean_free_path)
    return fixture


def _isolated_group_bands(group_size: int) -> DiagonalizedBands:
    """Return one isolated degenerate group and one complement band."""
    n_bands: int = group_size + 1
    basis: OrbitalBasis = _s_basis((0,) * n_bands)
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.eye(3),
        jnp.zeros((1, 3)),
        ("X",),
    )
    eigenvalues: Float[Array, "1 n_bands"] = jnp.concatenate(
        (
            jnp.zeros((1, group_size)),
            2.0 * jnp.ones((1, 1)),
        ),
        axis=1,
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues,
        jnp.eye(n_bands, dtype=jnp.complex128)[None, :, :],
        jnp.zeros((1, 3)),
        geometry,
        basis,
    )
    return bands


def _sensitivity_experiment() -> ExperimentGeometry:
    """Return a transverse experiment carrier for sensitivity callbacks."""
    experiment: ExperimentGeometry = make_experiment_geometry(
        21.2,
        jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
    )
    return experiment


def _simple_bands(
    basis: OrbitalBasis,
    atom_positions: Float[Array, "n_atom 3"],
    *,
    orbital_positions: Float[Array, "n_orb 3"] | None = None,
    depths: Float[Array, " n_orb"] | None = None,
) -> DiagonalizedBands:
    """Return a one-k-point carrier on a unit real-space lattice."""
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.eye(3),
        atom_positions,
        tuple("X" for _ in range(atom_positions.shape[0])),
    )
    n_orbitals: int = len(basis.n)
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=jnp.zeros((1, n_orbitals)),
        eigenvectors=jnp.eye(n_orbitals, dtype=jnp.complex128)[None, :, :],
        kpoints=jnp.zeros((1, 3)),
        geometry=geometry,
        basis=basis,
        orbital_positions=orbital_positions,
        depths=depths,
    )
    return bands


class TestPackMatrixelParams:
    """Validate :func:`diffpes.simul.pack_matrixel_params`.

    :see: :func:`diffpes.simul.pack_matrixel_params`
    """

    def test_packs_only_active_physical_coordinates(self) -> None:
        """Verify mode-aware packing and compact physical phase coordinates.

        The exact coordinate count exposes accidental calibration or padding entries.

        Notes
        -----
        Pack a two-shell Slater fixture and count every documented coordinate.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: tuple[tuple[tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        chex.assert_shape(flat, (14,))
        assert tree_definition.num_leaves == len(metadata)
        assert flat.dtype == jnp.float64
        assert params.phase_channel_keys == ((0, 1), (1, 0), (1, 2))
        chex.assert_shape(params.phase_shift_angles_shell, (3,))

    def test_fixed_channel_ratios_do_not_enter_the_flat_view(self) -> None:
        """Verify exclusion of calibrated fixed-radial channel ratios.

        Two different normalized calibration shapes must produce one flat view.

        Notes
        -----
        Hold shell phases and scales fixed while changing both fixed rows.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        first: RadialSpec = make_radial_spec(
            radial.basis,
            radial.radial_shell_index,
            mode="fixed",
            fixed_integrals_shell=jnp.array([[0.0, 1.0], [0.6, 0.8]]),
        )
        second: RadialSpec = make_radial_spec(
            radial.basis,
            radial.radial_shell_index,
            mode="fixed",
            fixed_integrals_shell=jnp.array([[0.0, 2.0], [0.8, 0.6]]),
        )
        first_flat: Float[Array, " n_theta"] = pack_matrixel_params(
            first,
            params,
            mean_free_path,
        )[0]
        second_flat: Float[Array, " n_theta"] = pack_matrixel_params(
            second,
            params,
            mean_free_path,
        )[0]
        chex.assert_trees_all_close(
            first_flat,
            second_flat,
            rtol=0.0,
            atol=0.0,
        )


class TestUnpackMatrixelParams:
    """Validate :func:`diffpes.simul.unpack_matrixel_params`.

    :see: :func:`diffpes.simul.unpack_matrixel_params`
    """

    def test_round_trip_is_bit_exact(self) -> None:
        """Verify exact reconstruction of every packed carrier field.

        The check also preserves the excluded and static template metadata.

        Notes
        -----
        Pack one Slater fixture and reconstruct it with the returned metadata.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: tuple[tuple[tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        rebuilt_radial: RadialSpec
        rebuilt_params: MatrixElementParams
        rebuilt_mfp: Float[Array, ""]
        rebuilt_radial, rebuilt_params, rebuilt_mfp = unpack_matrixel_params(
            flat,
            tree_definition,
            metadata,
            radial,
            params,
        )
        chex.assert_trees_all_equal(rebuilt_radial, radial)
        chex.assert_trees_all_equal(rebuilt_params, params)
        chex.assert_trees_all_equal(rebuilt_mfp, mean_free_path)


class TestMatrixElementPhaseGaugeDirection:
    """Validate :func:`diffpes.simul.matrix_element_phase_gauge_direction`.

    :see: :func:`diffpes.simul.matrix_element_phase_gauge_direction`
    """

    def test_phase_sweep_and_directional_derivative_are_null(self) -> None:
        """Verify intensity invariance along the packed overall-phase tangent.

        Generic complex radial rows make an omitted branch shift observable.

        Notes
        -----
        Sweep the unit tangent and compare its JVP with an exact zero.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: tuple[tuple[tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        direction: Float[Array, " n_theta"] = (
            matrix_element_phase_gauge_direction(
                radial,
                params,
                mean_free_path,
            )
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(direction),
            jnp.array(1.0),
            rtol=1e-14,
            atol=1e-14,
        )

        def intensity(candidate: Float[Array, " n_theta"]) -> Float[Array, ""]:
            """Return one coherent generic-complex orbital intensity."""
            rebuilt_radial: RadialSpec
            rebuilt_params: MatrixElementParams
            rebuilt_mfp: Float[Array, ""]
            rebuilt_radial, rebuilt_params, rebuilt_mfp = (
                unpack_matrixel_params(
                    candidate,
                    tree_definition,
                    metadata,
                    radial,
                    params,
                )
            )
            del rebuilt_radial
            bvals: Complex[Array, "1 4 2"] = jnp.array(
                [
                    [
                        [0.0 + 0.0j, 0.7 + 0.2j],
                        [0.4 - 0.3j, -0.2 + 0.8j],
                        [-0.1 + 0.5j, 0.6 + 0.1j],
                        [0.3 + 0.2j, -0.4 + 0.7j],
                    ]
                ]
            )
            channels: Complex[Array, "1 1 4 3"] = orbital_transition_channels(
                jnp.array([[0.1, -0.2, 0.0]]),
                jnp.array([[0.3, 0.2, 1.1]]),
                jnp.zeros((4, 3)),
                jnp.zeros(4),
                bvals,
                rebuilt_params,
                rebuilt_mfp,
                params.basis,
            )
            polarized: Complex[Array, "1 1 4"] = contract_polarization(
                channels,
                jnp.array([0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j]),
            )
            result: Float[Array, ""] = jnp.abs(jnp.sum(polarized)) ** 2
            return result

        reference: Float[Array, ""] = intensity(flat)
        alpha: float
        for alpha in (-2.0, -0.3, 0.8, 2.4):
            chex.assert_trees_all_close(
                intensity(flat + alpha * direction),
                reference,
                rtol=1e-13,
                atol=1e-13,
            )
        derivative: Float[Array, ""] = jax.jvp(
            intensity,
            (flat,),
            (direction,),
        )[1]
        chex.assert_trees_all_close(
            derivative,
            jnp.array(0.0),
            rtol=0.0,
            atol=1e-12,
        )


class TestRadialCoefficientScaleGaugeDirections:
    """Validate radial coefficient-scale gauge tangent construction.

    :see: :func:`diffpes.simul.radial_coefficient_scale_gauge_directions`
    """

    def test_each_normalized_shell_direction_is_null(self) -> None:
        """Verify one unit null tangent for each Slater contraction shell.

        Independent shell rows expose missing or mixed coefficient blocks.

        Notes
        -----
        Apply every packed JVP to normalized radial values on a fixed grid.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: tuple[tuple[tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        directions: Float[Array, "n_gauge n_theta"] = (
            radial_coefficient_scale_gauge_directions(
                radial,
                params,
                mean_free_path,
            )
        )
        chex.assert_shape(directions, (2, flat.shape[0]))
        chex.assert_trees_all_close(
            jnp.linalg.norm(directions, axis=-1),
            jnp.ones(2),
            rtol=1e-14,
            atol=1e-14,
        )

        def radial_values(
            candidate: Float[Array, " n_theta"],
        ) -> Float[Array, "n_orb n_r"]:
            """Return normalized orbital radial rows."""
            rebuilt: RadialSpec = unpack_matrixel_params(
                candidate,
                tree_definition,
                metadata,
                radial,
                params,
            )[0]
            values: Float[Array, "n_orb n_r"] = evaluate_radial(
                rebuilt,
                jnp.linspace(0.01, 5.0, 41),
            )
            return values

        direction: Float[Array, " n_theta"]
        for direction in directions:
            derivative: Float[Array, "n_orb n_r"] = jax.jvp(
                radial_values,
                (flat,),
                (direction,),
            )[1]
            chex.assert_trees_all_close(
                derivative,
                jnp.zeros_like(derivative),
                rtol=0.0,
                atol=1e-12,
            )


class TestBandGroupWeightSensitivity:
    """Validate :func:`diffpes.simul.band_group_weight_sensitivity`.

    :see: :func:`diffpes.simul.band_group_weight_sensitivity`
    """

    @staticmethod
    def _rebuild(
        candidate: Float[Array, " n_theta"],
        bands: DiagonalizedBands,
        experiment: ExperimentGeometry,
    ) -> Complex[Array, "n_k n_bands 2"]:
        """Build generic complex two-spin amplitudes from orbital rows."""
        del experiment
        n_orbitals: int = bands.eigenvectors.shape[-1]
        index: Float[Array, " n_orb"] = jnp.arange(
            1,
            n_orbitals + 1,
            dtype=jnp.float64,
        )
        base: Complex[Array, " n_orb"] = index * (0.3 + 0.2j) + (0.1 - 0.4j)
        orbital_rows: Complex[Array, "n_orb 2"] = jnp.stack(
            (
                (1.0 + candidate[0]) * base + candidate[1] * jnp.conj(base),
                (0.4 - 0.3 * candidate[0]) * jnp.conj(base)
                + 0.2j * candidate[1] * base,
            ),
            axis=-1,
        )
        amplitudes: Complex[Array, "n_k n_bands 2"] = jnp.einsum(
            "kbo,os->kbs",
            bands.eigenvectors,
            orbital_rows,
        )
        return amplitudes

    @pytest.mark.parametrize("group_size", [2, 3])
    def test_u_group_rotation_preserves_weights_and_jacobian(
        self,
        group_size: int,
    ) -> None:
        """Verify U(2) and U(3) invariance of complete-group sensitivities.

        Each rotation changes member weights while preserving the group norm.

        Notes
        -----
        Rotate only the degenerate eigenvector rows and compare full outputs.
        """
        bands: DiagonalizedBands = _isolated_group_bands(group_size)
        experiment: ExperimentGeometry = _sensitivity_experiment()
        flat: Float[Array, " 2"] = jnp.array([0.2, -0.15])
        group: tuple[int, ...] = tuple(range(group_size))
        weights: Float[Array, "1 1"]
        jacobian: Float[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            bands,
            experiment,
            (group,),
        )
        unitary: Complex[Array, "n_group n_group"]
        if group_size == 2:
            angle: float = 0.61
            unitary = jnp.array(
                [
                    [math.cos(angle), 1j * math.sin(angle)],
                    [1j * math.sin(angle), math.cos(angle)],
                ],
                dtype=jnp.complex128,
            )
        else:
            root: Complex[Array, ""] = jnp.exp(2.0j * jnp.pi / 3.0)
            unitary = jnp.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, root, root**2],
                    [1.0, root**2, root],
                ],
                dtype=jnp.complex128,
            ) / math.sqrt(3.0)
        rotated_eigenvectors: Complex[Array, "1 n_bands n_orb"] = (
            bands.eigenvectors.at[0, :group_size].set(
                unitary @ bands.eigenvectors[0, :group_size]
            )
        )
        rotated_bands: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            bands,
            rotated_eigenvectors,
        )
        rotated_weights: Float[Array, "1 1"]
        rotated_jacobian: Float[Array, "2 1 1"]
        rotated_weights, rotated_jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            rotated_bands,
            experiment,
            (group,),
        )
        original_members: Float[Array, "1 n_group"] = matrix_element_intensity(
            self._rebuild(flat, bands, experiment)[:, :group_size]
        )
        rotated_members: Float[Array, "1 n_group"] = matrix_element_intensity(
            self._rebuild(flat, rotated_bands, experiment)[:, :group_size]
        )
        assert not jnp.allclose(original_members, rotated_members)
        chex.assert_trees_all_close(
            rotated_weights,
            weights,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            rotated_jacobian,
            jacobian,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_jacobian_matches_fd_and_rejects_partial_groups(self) -> None:
        """Verify finite differences and exact incomplete-group rejection.

        A degenerate partner in the complement makes the singleton group partial.

        Notes
        -----
        Compare both parameter columns and plant partial and nonisolated groups.
        """
        bands: DiagonalizedBands = _isolated_group_bands(2)
        experiment: ExperimentGeometry = _sensitivity_experiment()
        flat: Float[Array, " 2"] = jnp.array([0.13, -0.21])
        weights: Float[Array, "1 1"]
        jacobian: Float[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            bands,
            experiment,
            ((0, 1),),
        )
        del weights
        step: float = 1.0e-6
        parameter: int
        for parameter in range(2):
            direction: Float[Array, " 2"] = jnp.zeros(2).at[parameter].set(1.0)
            plus: Float[Array, "1 1"] = band_group_weight_sensitivity(
                flat + step * direction,
                self._rebuild,
                bands,
                experiment,
                ((0, 1),),
            )[0]
            minus: Float[Array, "1 1"] = band_group_weight_sensitivity(
                flat - step * direction,
                self._rebuild,
                bands,
                experiment,
                ((0, 1),),
            )[0]
            finite_difference: Float[Array, "1 1"] = (plus - minus) / (
                2.0 * step
            )
            chex.assert_trees_all_close(
                jacobian[parameter],
                finite_difference,
                rtol=1e-6,
                atol=1e-8,
            )
        with pytest.raises(ValueError, match="cuts a degeneracy"):
            band_group_weight_sensitivity(
                flat,
                self._rebuild,
                bands,
                experiment,
                ((0,),),
            )
        close_eigenvalues: Float[Array, "1 3"] = bands.eigenvalues.at[
            0, 2
        ].set(0.5e-6)
        nonisolated: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvalues,
            bands,
            close_eigenvalues,
        )
        with pytest.raises(ValueError, match="lacks complement isolation"):
            band_group_weight_sensitivity(
                flat,
                self._rebuild,
                nonisolated,
                experiment,
                ((0, 1),),
            )

    def test_dark_weight_has_finite_zero_derivative(self) -> None:
        """Verify a smooth exact dark point without dividing by its weight.

        A linear amplitude crossing produces a quadratic weight and zero slope.

        Notes
        -----
        Evaluate the group helper at the registered zero-amplitude coordinate.
        """
        bands: DiagonalizedBands = _isolated_group_bands(1)
        experiment: ExperimentGeometry = _sensitivity_experiment()

        def dark_rebuild(
            candidate: Float[Array, " 2"],
            candidate_bands: DiagonalizedBands,
            candidate_experiment: ExperimentGeometry,
        ) -> Complex[Array, "1 2 1"]:
            """Build one dark band and one inert complement amplitude."""
            del candidate_bands, candidate_experiment
            amplitudes: Complex[Array, "1 2 1"] = jnp.array(
                [[[candidate[0] + 1j * candidate[1]], [0.0 + 0.0j]]]
            )
            return amplitudes

        weights: Float[Array, "1 1"]
        jacobian: Float[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            jnp.zeros(2),
            dark_rebuild,
            bands,
            experiment,
            ((0,),),
        )
        chex.assert_trees_all_close(weights, jnp.zeros((1, 1)))
        chex.assert_trees_all_close(jacobian, jnp.zeros((2, 1, 1)))
        chex.assert_tree_all_finite(jacobian)


class TestLogBandGroupWeightSensitivity:
    """Validate logarithmic complete-group sensitivity conversion.

    :see: :func:`diffpes.simul.log_band_group_weight_sensitivity`
    """

    def test_dark_mask_and_positive_log_identity(self) -> None:
        """Verify the dark sentinel and positive-domain ``dw/w`` identity.

        Adjacent zero and positive weights exercise both validity branches.

        Notes
        -----
        Supply analytic weights and Jacobians and compare every returned entry.
        """
        weights: Float[Array, " 2"] = jnp.array([0.0, 0.05])
        jacobian: Float[Array, "2 2"] = jnp.array([[0.0, 0.4], [0.0, -0.2]])
        log_jacobian: Float[Array, "2 2"]
        valid: Bool[Array, " 2"]
        log_jacobian, valid = log_band_group_weight_sensitivity(
            weights,
            jacobian,
            1.0e-8,
        )
        chex.assert_trees_all_equal(valid, jnp.array([False, True]))
        chex.assert_trees_all_close(log_jacobian[:, 0], jnp.zeros(2))
        chex.assert_trees_all_close(
            log_jacobian[:, 1],
            jacobian[:, 1] / weights[1],
            rtol=1e-14,
            atol=1e-14,
        )
        chex.assert_tree_all_finite(log_jacobian)


class TestResolveOrbitalPositionsCart:
    """Validate :func:`diffpes.simul.resolve_orbital_positions_cart`.

    :see: :func:`diffpes.simul.resolve_orbital_positions_cart`
    """

    def test_explicit_wannier_centres_precede_atom_fallback(self) -> None:
        """Use explicit fractional centres and apply the lattice once.

        The displaced centres differ from their host atoms and alter both rows.

        Notes
        -----
        Compare the result with the direct explicit-centre matrix product.
        """
        basis: OrbitalBasis = _s_basis((0, 1))
        geometry: CrystalGeometry = make_crystal_geometry(
            jnp.diag(jnp.array([2.0, 3.0, 4.0])),
            jnp.array([[0.1, 0.2, 0.3], [0.6, 0.4, 0.2]]),
            ("A", "B"),
        )
        explicit: Float[Array, "2 3"] = jnp.array(
            [[0.17, 0.11, 0.07], [0.55, 0.45, 0.25]]
        )
        bands: DiagonalizedBands = make_diagonalized_bands(
            jnp.zeros((1, 2)),
            jnp.eye(2, dtype=jnp.complex128)[None, :, :],
            jnp.zeros((1, 3)),
            geometry,
            basis,
            orbital_positions=explicit,
        )
        actual: Float[Array, "2 3"] = resolve_orbital_positions_cart(bands)
        chex.assert_trees_all_close(
            actual,
            explicit @ geometry.lattice,
            rtol=0.0,
            atol=0.0,
        )
        assert not jnp.allclose(
            actual,
            geometry.positions @ geometry.lattice,
        )

    def test_none_matches_atom_derived_path(self) -> None:
        """Verify host-atom gathering before the fractional-to-Cartesian map.

        The reversed basis assignment exposes an omitted orbital-to-atom gather.

        Notes
        -----
        Compare two resolved rows against the reversed atomic-position rows.
        """
        basis: OrbitalBasis = _s_basis((1, 0))
        bands: DiagonalizedBands = _simple_bands(
            basis,
            jnp.array([[0.1, 0.2, 0.3], [0.7, 0.5, 0.4]]),
        )
        expected: Float[Array, "2 3"] = bands.geometry.positions[
            jnp.array([1, 0])
        ]
        chex.assert_trees_all_close(
            resolve_orbital_positions_cart(bands),
            expected,
            rtol=0.0,
            atol=0.0,
        )


class TestRealSphericalHarmonicsCartesianAll:
    """Validate Cartesian solid harmonics through the final-state limit.

    :see: :func:`diffpes.simul.real_spherical_harmonics_cartesian_all`
    """

    def test_matches_angular_reference_through_l_five(self) -> None:
        """Match the existing real-harmonic convention away from poles.

        A generic direction exercises every nontrivial real-harmonic branch.

        Notes
        -----
        Convert the direction to angles only inside the independent reference.
        """
        direction: Float[Array, " 3"] = jnp.array([0.3, -0.4, 0.8])
        direction = direction / jnp.linalg.norm(direction)
        theta: Float[Array, ""] = jnp.arccos(direction[2])
        phi: Float[Array, ""] = jnp.arctan2(direction[1], direction[0])
        expected: Float[Array, " 36"] = real_spherical_harmonics_all(
            5,
            theta,
            phi,
        )
        actual: Float[Array, " 36"] = real_spherical_harmonics_cartesian_all(
            direction, 5
        )
        chex.assert_trees_all_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_exact_poles_and_transverse_derivatives(self) -> None:
        """Recover pole values without an arbitrary azimuthal gauge.

        The transverse Jacobian also checks smooth directional information.

        Notes
        -----
        Compare analytic parity values and the real p-orbital Cartesian slopes.
        """
        poles: Float[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        actual: Float[Array, "2 36"] = real_spherical_harmonics_cartesian_all(
            poles, 5
        )
        degree: int
        for degree in range(6):
            centre_index: int = degree * degree + degree
            magnitude: float = math.sqrt((2 * degree + 1) / (4.0 * math.pi))
            assert actual[0, centre_index] == pytest.approx(magnitude)
            assert actual[1, centre_index] == pytest.approx(
                ((-1) ** degree) * magnitude
            )
        off_axis: Array = jnp.delete(
            actual,
            jnp.asarray([degree * degree + degree for degree in range(6)]),
            axis=-1,
        )
        chex.assert_trees_all_close(off_axis, jnp.zeros_like(off_axis))

        def p_harmonics(transverse: Float[Array, " 2"]) -> Float[Array, " 2"]:
            """Return the real p_y and p_x rows near the north pole."""
            vector: Float[Array, " 3"] = jnp.array(
                [transverse[0], transverse[1], 1.0]
            )
            values: Float[Array, " 4"] = (
                real_spherical_harmonics_cartesian_all(vector, 1)
            )
            result: Float[Array, " 2"] = values[jnp.asarray([1, 3])]
            return result

        jacobian: Float[Array, "2 2"] = jax.jacfwd(p_harmonics)(jnp.zeros(2))
        normalization: float = math.sqrt(3.0 / (4.0 * math.pi))
        expected_jacobian: Float[Array, "2 2"] = jnp.array(
            [[0.0, normalization], [normalization, 0.0]]
        )
        chex.assert_trees_all_close(
            jacobian,
            expected_jacobian,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_jit_and_zero_rejection(self) -> None:
        """Compile a batch and reject an undefined zero direction.

        The check covers the transformed path and its physical domain guard.

        Notes
        -----
        Compile two nonzero rows and call the eager boundary with zero.
        """
        directions: Float[Array, "2 3"] = jnp.array(
            [[1.0, 0.0, 1.0], [0.0, -2.0, 1.0]]
        )
        actual: Float[Array, "2 16"] = jax.jit(
            lambda values: real_spherical_harmonics_cartesian_all(values, 3)
        )(directions)
        chex.assert_shape(actual, (2, 16))
        chex.assert_tree_all_finite(actual)
        with pytest.raises(eqx.EquinoxRuntimeError):
            real_spherical_harmonics_cartesian_all(jnp.zeros(3), 1)


class TestOrbitalTransitionChannels:
    """Validate coherent orbital assembly and attenuation.

    :see: :func:`diffpes.simul.orbital_transition_channels`
    """

    def test_two_centre_interference_and_translation_covariance(self) -> None:
        """Match the analytic two-centre cosine and common-translation phase.

        The fixture exposes a missing centre phase or the opposite phase sign.

        Notes
        -----
        Compare the coherent sum with its cosine structure factor and phase.
        """
        basis: OrbitalBasis = _s_basis((0, 1))
        params: MatrixElementParams = _matrix_params(basis, (0, 1))
        separation: float = 1.7
        positions: Float[Array, "2 3"] = jnp.array(
            [[-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0]]
        )
        initial: Float[Array, "1 3"] = jnp.array([[0.63, 0.0, 0.0]])
        final: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 1.2]])
        bvals: Complex[Array, "1 2 2"] = jnp.array(
            [[[0.0 + 0.0j, 0.2 + 1.1j], [0.0 + 0.0j, 0.2 + 1.1j]]]
        )
        channels: Complex[Array, "1 1 2 3"] = orbital_transition_channels(
            initial,
            final,
            positions,
            jnp.zeros(2),
            bvals,
            params,
            jnp.array(9.0),
            basis,
        )
        amplitude: Complex[Array, ""] = jnp.sum(
            contract_polarization(
                channels,
                jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
            )
        )
        atomic_amplitude: Complex[Array, ""] = contract_polarization(
            channels[:, :, :1, :],
            jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
        )[0, 0, 0]
        phase_argument: float = 0.63 * separation / 2.0
        expected_intensity: Float[Array, ""] = (
            4.0
            * math.cos(phase_argument) ** 2
            * jnp.abs(atomic_amplitude) ** 2
        )
        chex.assert_trees_all_close(
            jnp.abs(amplitude) ** 2,
            expected_intensity,
            rtol=1e-13,
            atol=1e-14,
        )

        translation: Float[Array, " 3"] = jnp.array([0.21, -0.14, 0.08])
        translated: Complex[Array, "1 1 2 3"] = orbital_transition_channels(
            initial,
            final,
            positions + translation,
            jnp.zeros(2),
            bvals,
            params,
            jnp.array(9.0),
            basis,
        )
        expected_phase: Complex[Array, ""] = jnp.exp(
            1j * jnp.dot(initial[0] - final[0], translation)
        )
        chex.assert_trees_all_close(
            translated,
            expected_phase * channels,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_depth_ratio_clamp_and_mfp_gradient(self) -> None:
        """Use the half exponent, clamp tolerance noise, and differentiate mfp.

        The isolated-layer ratio distinguishes amplitude and intensity lengths.

        Notes
        -----
        Compare the ratio and reverse-mode derivative with closed forms.
        """
        basis: OrbitalBasis = _s_basis((0, 1))
        params: MatrixElementParams = _matrix_params(basis, (0, 1))
        depth: float = 4.5
        mean_free_path: Float[Array, ""] = jnp.array(8.0)
        common_arguments: tuple[Array, ...] = (
            jnp.zeros((1, 3)),
            jnp.array([[0.0, 0.0, 1.0]]),
            jnp.zeros((2, 3)),
        )
        radial: Complex[Array, "1 2 2"] = jnp.array(
            [[[0.0 + 0.0j, 1.0j], [0.0 + 0.0j, 1.0j]]]
        )

        def layer_intensity(mfp: Float[Array, ""]) -> Float[Array, ""]:
            """Return the isolated deep-layer intensity."""
            transition: Complex[Array, "1 1 2 3"] = (
                orbital_transition_channels(
                    *common_arguments,
                    jnp.array([-0.5e-12, depth]),
                    radial,
                    params,
                    mfp,
                    basis,
                )
            )
            result: Float[Array, ""] = jnp.sum(
                jnp.abs(transition[0, 0, 1]) ** 2
            )
            return result

        channels: Complex[Array, "1 1 2 3"] = orbital_transition_channels(
            *common_arguments,
            jnp.array([-0.5e-12, depth]),
            radial,
            params,
            mean_free_path,
            basis,
        )
        intensities: Float[Array, " 2"] = jnp.sum(
            jnp.abs(channels[0, 0]) ** 2,
            axis=-1,
        )
        chex.assert_trees_all_close(
            intensities[1] / intensities[0],
            jnp.exp(-depth / mean_free_path),
            rtol=1e-14,
            atol=1e-14,
        )
        actual_gradient: Float[Array, ""] = jax.grad(layer_intensity)(
            mean_free_path
        )
        expected_gradient: Float[Array, ""] = (
            layer_intensity(mean_free_path) * depth / mean_free_path**2
        )
        chex.assert_trees_all_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-13,
            atol=1e-14,
        )


class TestContractPolarization:
    """Validate :func:`diffpes.simul.contract_polarization`.

    :see: :func:`diffpes.simul.contract_polarization`
    """

    def test_uses_cartesian_to_real_permutation_without_conjugation(
        self,
    ) -> None:
        """Verify ``(y,z,x)`` contraction with complex Cartesian light.

        Generic phases expose a conjugated or unpermuted polarization vector.

        Notes
        -----
        Compute the expected direct matrix-vector product in real-channel order.
        """
        channels: Complex[Array, "2 3"] = jnp.array(
            [[1.0 + 2.0j, -0.3 + 0.4j, 0.7 - 0.2j], [2.0j, 3.0, -1.0j]]
        )
        polarization: Complex[Array, " 3"] = jnp.array(
            [0.2 + 0.1j, -0.4 + 0.3j, 0.5 - 0.7j]
        )
        expected: Complex[Array, " 2"] = (
            channels @ polarization[jnp.asarray([1, 2, 0])]
        )
        actual: Complex[Array, " 2"] = contract_polarization(
            channels,
            polarization,
        )
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)


class TestContractExperimentPolarization:
    """Validate the fixed laboratory-to-sample polarization seam.

    :see: :func:`diffpes.simul.contract_experiment_polarization`
    """

    def test_rotates_lab_polarization_once_before_contraction(self) -> None:
        """Match the analytic inverse sample-azimuth rotation.

        A nonzero azimuth distinguishes laboratory and sample coordinates.

        Notes
        -----
        Build the analytic sample vector and compare its late contraction.
        """
        azimuth: float = 0.37
        experiment: ExperimentGeometry = make_experiment_geometry(
            21.2,
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            sample_azimuth=azimuth,
        )
        channels: Complex[Array, " 3"] = jnp.array(
            [0.7 + 0.2j, -0.1j, 1.3 - 0.4j]
        )
        polarization_sample: Complex[Array, " 3"] = jnp.array(
            [math.cos(azimuth), -math.sin(azimuth), 0.0],
            dtype=jnp.complex128,
        )
        expected: Complex[Array, ""] = contract_polarization(
            channels,
            polarization_sample,
        )
        actual: Complex[Array, ""] = contract_experiment_polarization(
            channels,
            experiment,
        )
        chex.assert_trees_all_close(actual, expected, rtol=1e-14, atol=1e-14)


class TestTransitionSource:
    """Validate :func:`diffpes.simul.transition_source`.

    :see: :func:`diffpes.simul.transition_source`
    """

    def test_conjugates_once_and_embeds_opposite_block_zeros(self) -> None:
        """Build two outgoing-spin source kets from generic complex rows.

        Distinct phases expose missing or repeated source conjugation.

        Notes
        -----
        Compare every embedded entry and each exact opposite-block zero.
        """
        rows: Complex[Array, "2 2"] = jnp.array(
            [[1.0 + 2.0j, -0.3j], [0.7 - 0.2j, -1.1 + 0.4j]]
        )
        actual: Complex[Array, "2 4"] = transition_source(rows)
        expected: Complex[Array, "2 4"] = jnp.array(
            [
                [1.0 - 2.0j, 0.3j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.7 + 0.2j, -1.1 - 0.4j],
            ]
        )
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    def test_dense_resolvent_matches_spectral_rows_and_parameter_fd(
        self,
    ) -> None:
        """Match spin-block resolvents and the dual-convention derivative.

        Generic complex SOC mixing exposes coherent source sums and wrong bras.

        Notes
        -----
        Build an independent NumPy resolvent and compare its spectral expansion.
        Differentiate the same source convention and check a centered quotient.
        """
        hamiltonian: np.ndarray = np.asarray(
            [
                [0.2, 0.13 + 0.07j, 0.04j, -0.03],
                [0.13 - 0.07j, -0.4, 0.05 + 0.02j, 0.01j],
                [-0.04j, 0.05 - 0.02j, 0.6, -0.11 + 0.06j],
                [-0.03, -0.01j, -0.11 - 0.06j, -0.1],
            ],
            dtype=np.complex128,
        )
        energy: complex = 0.31 + 0.27j
        rows_numpy: np.ndarray = np.asarray(
            [[0.7 + 0.2j, -0.1 + 0.4j], [0.3 - 0.5j, -0.6 + 0.1j]],
            dtype=np.complex128,
        )
        source_numpy: np.ndarray = np.asarray(
            transition_source(jnp.asarray(rows_numpy))
        )
        resolvent: np.ndarray = np.linalg.inv(
            energy * np.eye(4, dtype=np.complex128) - hamiltonian
        )
        direct: complex = sum(
            source.conj() @ resolvent @ source for source in source_numpy
        )
        eigenvalues: np.ndarray
        eigenvectors: np.ndarray
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        spectral: complex = 0.0 + 0.0j
        outgoing_spin: int
        band: int
        for outgoing_spin in range(2):
            for band in range(4):
                spin_block: np.ndarray = eigenvectors[
                    2 * outgoing_spin : 2 * (outgoing_spin + 1),
                    band,
                ]
                amplitude: complex = rows_numpy[outgoing_spin] @ spin_block
                spectral += abs(amplitude) ** 2 / (energy - eigenvalues[band])
        np.testing.assert_allclose(
            direct, spectral, rtol=1.0e-12, atol=1.0e-12
        )
        coherent_source: np.ndarray = np.sum(source_numpy, axis=0)
        coherent_control: complex = (
            coherent_source.conj() @ resolvent @ coherent_source
        )
        assert not np.isclose(
            coherent_control,
            direct,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

        hamiltonian_jax: Complex[Array, "4 4"] = jnp.asarray(hamiltonian)
        resolvent_jax: Complex[Array, "4 4"] = jnp.linalg.inv(
            energy * jnp.eye(4, dtype=jnp.complex128) - hamiltonian_jax
        )
        direction: Complex[Array, "2 2"] = jnp.asarray(
            [[0.2 - 0.3j, 0.1 + 0.05j], [-0.08j, 0.17 + 0.11j]]
        )

        def response(parameter: Float[Array, ""]) -> Complex[Array, ""]:
            """Return the dense response along one real row direction."""
            rows: Complex[Array, "2 2"] = (
                jnp.asarray(rows_numpy) + parameter * direction
            )
            sources: Complex[Array, "2 4"] = transition_source(rows)
            values: Complex[Array, " 2"] = jnp.einsum(
                "si,ij,sj->s",
                jnp.conj(sources),
                resolvent_jax,
                sources,
            )
            result: Complex[Array, ""] = jnp.sum(values)
            return result

        derivative: Complex[Array, ""] = jax.jacfwd(response)(jnp.asarray(0.0))
        step: float = 1.0e-5
        finite_difference: Complex[Array, ""] = (
            response(jnp.asarray(step)) - response(jnp.asarray(-step))
        ) / (2.0 * step)
        chex.assert_trees_all_close(
            derivative,
            finite_difference,
            rtol=1.0e-10,
            atol=1.0e-12,
        )


class TestProjectBandChannels:
    """Validate :func:`diffpes.simul.project_band_channels`.

    :see: :func:`diffpes.simul.project_band_channels`
    """

    def test_generic_complex_projection_has_no_conjugate(self) -> None:
        """Match a direct complex sum and reject the planted bra contraction.

        Complex coefficients distinguish the ket convention from a bra sum.

        Notes
        -----
        Compare with direct and deliberately conjugated coefficient contractions.
        """
        transition: Complex[Array, "1 1 2 3"] = jnp.array(
            [[[[1.0 + 0.3j, 0.2j, -0.4], [0.7j, 1.2, 0.5 - 0.1j]]]]
        )
        eigenvectors: Complex[Array, "1 1 2"] = jnp.array(
            [[[0.6 + 0.2j, -0.3 + 0.7j]]]
        )
        actual: Complex[Array, "1 1 1 3"] = project_band_channels(
            transition,
            eigenvectors,
        )
        expected: Complex[Array, " 3"] = jnp.sum(
            transition[0, 0] * eigenvectors[0, 0, :, None],
            axis=0,
        )
        planted_wrong: Complex[Array, " 3"] = jnp.sum(
            transition[0, 0] * jnp.conj(eigenvectors[0, 0, :, None]),
            axis=0,
        )
        chex.assert_trees_all_close(
            actual[0, 0, 0],
            expected,
            rtol=0.0,
            atol=0.0,
        )
        assert not jnp.allclose(actual[0, 0, 0], planted_wrong)

    def test_band_gauge_phase_leaves_intensity_invariant(self) -> None:
        """Verify band-phase cancellation after the late modulus square.

        The complex band gauge changes amplitude phase but not physical intensity.

        Notes
        -----
        Multiply every coefficient by one phase and compare reduced intensities.
        """
        transition: Complex[Array, "1 1 2 3"] = jnp.array(
            [[[[1.0j, 0.4, -0.2j], [0.3 + 0.7j, -0.1j, 0.8]]]]
        )
        eigenvectors: Complex[Array, "1 1 2"] = jnp.array(
            [[[0.5 + 0.4j, -0.2 + 0.7j]]]
        )
        phase: Complex[Array, ""] = jnp.exp(0.83j)
        first: Complex[Array, "1 1 1"] = contract_polarization(
            project_band_channels(transition, eigenvectors),
            jnp.array([0.2 + 0.1j, 0.4 - 0.3j, -0.5j]),
        )
        second: Complex[Array, "1 1 1"] = contract_polarization(
            project_band_channels(transition, phase * eigenvectors),
            jnp.array([0.2 + 0.1j, 0.4 - 0.3j, -0.5j]),
        )
        chex.assert_trees_all_close(
            matrix_element_intensity(first),
            matrix_element_intensity(second),
            rtol=1e-14,
            atol=1e-14,
        )


class TestMatrixElementIntensity:
    """Validate the only unresolved-spin modulus-square reduction.

    :see: :func:`diffpes.simul.matrix_element_intensity`
    """

    def test_relative_spin_phase_is_unobservable(self) -> None:
        """Keep incoherent spin intensity fixed while a coherent sum oscillates.

        Several relative phases expose an unphysical amplitude-level spin sum.

        Notes
        -----
        Compare the production reduction with a deliberately coherent control.
        """
        phases: Float[Array, " 4"] = jnp.array([0.0, 0.4, 1.7, math.pi])
        amplitudes: Complex[Array, "4 2"] = jnp.stack(
            (
                jnp.ones_like(phases, dtype=jnp.complex128),
                jnp.exp(1j * phases),
            ),
            axis=-1,
        ) / math.sqrt(2.0)
        actual: Float[Array, " 4"] = matrix_element_intensity(amplitudes)
        planted_coherent: Float[Array, " 4"] = (
            jnp.abs(jnp.sum(amplitudes, axis=-1)) ** 2
        )
        chex.assert_trees_all_close(
            actual, jnp.ones(4), rtol=1e-14, atol=1e-14
        )
        assert float(jnp.ptp(planted_coherent)) > 1.0


class TestAssembleOrbitalTransitionChannels:
    """Validate the complete fixed-radial matrix-element assembler.

    :see: :func:`diffpes.simul.assemble_orbital_transition_channels`
    """

    @staticmethod
    def _fixture() -> MatrixFixture:
        """Build one s-orbital zero-umklapp fixture."""
        basis: OrbitalBasis = _s_basis((0,))
        bands: DiagonalizedBands = _simple_bands(
            basis,
            jnp.array([[0.17, 0.11, 0.07]]),
        )
        radial: RadialSpec = make_radial_spec(
            basis,
            (0,),
            mode="fixed",
            fixed_integrals_shell=jnp.array([[0.0, 1.0]]),
        )
        params: MatrixElementParams = _matrix_params(basis, (0,))
        quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
        final_state: FinalStateSpec = make_final_state_spec()
        experiment: ExperimentGeometry = make_experiment_geometry(
            21.2,
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        fixture: MatrixFixture = (
            bands,
            radial,
            params,
            quadrature,
            final_state,
            experiment,
        )
        return fixture

    def test_consumes_explicit_vacuum_momentum_and_ignores_v0(self) -> None:
        """Verify fixed vacuum momentum across an inner-potential change.

        This false control catches accidental use of the internal final-state kz.

        Notes
        -----
        Replace only the inner potential and compare the complete transition tensor.
        """
        fixture: MatrixFixture = self._fixture()
        final_momentum: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 1.0]])
        first: Complex[Array, "1 1 1 3"] = (
            assemble_orbital_transition_channels(
                *fixture,
                final_momentum,
                jnp.array([True]),
            )
        )
        changed_experiment: ExperimentGeometry = eqx.tree_at(
            lambda item: item.inner_potential_ev,
            fixture[-1],
            jnp.array(77.0),
        )
        second_fixture: MatrixFixture = fixture[:-1] + (changed_experiment,)
        second: Complex[Array, "1 1 1 3"] = (
            assemble_orbital_transition_channels(
                *second_fixture,
                final_momentum,
                jnp.array([True]),
            )
        )
        chex.assert_trees_all_close(first, second, rtol=0.0, atol=0.0)
        chex.assert_tree_all_finite(first)
        assert float(jnp.linalg.norm(first)) > 0.0

    @pytest.mark.parametrize(
        ("final_momentum", "validity"),
        [
            (jnp.array([[2.0e-12, 0.0, 1.0]]), jnp.array([True])),
            (jnp.array([[0.0, 0.0, 1.0]]), jnp.array([False])),
            (jnp.array([[0.0, 0.0, 0.0]]), jnp.array([True])),
        ],
    )
    def test_rejects_nonzero_gparallel_invalidity_and_zero_momentum(
        self,
        final_momentum: Float[Array, "1 3"],
        validity: Array,
    ) -> None:
        """Reject every forbidden explicit vacuum-momentum boundary.

        The cases cover reciprocal shift, false validity, and zero direction.

        Notes
        -----
        Call the same assembler fixture with each planted invalid input.
        """
        with pytest.raises(eqx.EquinoxRuntimeError):
            assemble_orbital_transition_channels(
                *self._fixture(),
                final_momentum,
                validity,
            )
