"""Validate matrix-element parameters, gauges, and sensitivities.

Extended Summary
----------------
The tests exercise real optimization coordinates and their inverse mapping.
They also verify gauge tangents and complete band-group derivatives with
analytic values and finite differences.
"""

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int64

from diffpes.matrixel import (
    band_group_weight_sensitivity,
    contract_polarization,
    log_band_group_weight_sensitivity,
    matrix_element_intensity,
    matrix_element_phase_gauge_direction,
    orbital_transition_channels,
    pack_matrixel_params,
    radial_coefficient_scale_gauge_directions,
    unpack_matrixel_params,
)
from diffpes.radial import evaluate_radial
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    ExperimentGeometry,
    MatrixElementParams,
    OrbitalBasis,
    PyTreeDef,
    RadialSpec,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_experiment_geometry,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_spec,
)
from tests._gradients import (
    assert_grad_matches_fd,
    assert_gradients_match_finite_differences,
)


def _packing_fixture() -> Tuple[
    RadialSpec,
    MatrixElementParams,
    Float64[Array, ""],
]:
    """PRIVATE: Return a two-shell Slater packing fixture.

    Returns
    -------
    radial : RadialSpec
        Two-term contracted Slater spec for the s and p shells.
    params : MatrixElementParams
        Carrier with distinct sigma scales and three branch phase-shift
        angles.
    mean_free_path : Float64[Array, ""]
        Mean free path of 8.5 Angstrom.

    Notes
    -----
    Builds the mixed 1s plus n=2 p basis with shell map (0, 1, 1, 1)
    and generic zeta pairs and contraction coefficients.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0, 0),
        n=(1, 2, 2, 2),
        l=(0, 1, 1, 1),
        m=(0, -1, 0, 1),
    )
    shell_index: Tuple[int, ...] = (0, 1, 1, 1)
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
    mean_free_path: Float64[Array, ""] = jnp.array(8.5)
    fixture: Tuple[
        RadialSpec,
        MatrixElementParams,
        Float64[Array, ""],
    ] = (radial, params, mean_free_path)
    return fixture


def _isolated_group_bands(group_size: int) -> DiagonalizedBands:
    """PRIVATE: Return one isolated degenerate group and one complement band.

    Parameters
    ----------
    group_size : int
        Number of degenerate bands in the zero-energy group.

    Returns
    -------
    bands : DiagonalizedBands
        Identity-eigenvector bands at one zone-center k-point with the
        group at 0 eV and one complement band at 2 eV.

    Notes
    -----
    The 2 eV gap isolates the degenerate group; every orbital is an s
    orbital on the single atom.
    """
    n_bands: int = group_size + 1
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_bands,
        n=(1,) * n_bands,
        l=(0,) * n_bands,
        m=(0,) * n_bands,
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.eye(3),
        jnp.zeros((1, 3)),
        ("X",),
    )
    eigenvalues: Float64[Array, "1 n_bands"] = jnp.concatenate(
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
    """PRIVATE: Return a transverse experiment for sensitivity callbacks.

    Returns
    -------
    experiment : ExperimentGeometry
        Carrier with 21.2 eV photon energy and x-polarized light.

    Notes
    -----
    Keeps every remaining geometry field at its factory default.
    """
    experiment: ExperimentGeometry = make_experiment_geometry(
        21.2,
        jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
    )
    return experiment


class TestPackMatrixelParams:
    """Validate :func:`diffpes.matrixel.pack_matrixel_params`.

    :see: :func:`diffpes.matrixel.pack_matrixel_params`
    """

    def test_packs_only_active_physical_coordinates(self) -> None:
        """Verify mode-aware packing and compact physical phase coordinates.

        The exact coordinate count exposes accidental calibration or padding
        entries.

        Notes
        -----
        Pack a two-shell Slater fixture and count every documented coordinate.
        """
        radial: RadialSpec
        params: MatrixElementParams
        mean_free_path: Float64[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float64[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: Tuple[Tuple[Tuple[int, ...], bool], ...]
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
        mean_free_path: Float64[Array, ""]
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
        first_flat: Float64[Array, " n_theta"] = pack_matrixel_params(
            first,
            params,
            mean_free_path,
        )[0]
        second_flat: Float64[Array, " n_theta"] = pack_matrixel_params(
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
    """Validate :func:`diffpes.matrixel.unpack_matrixel_params`.

    :see: :func:`diffpes.matrixel.unpack_matrixel_params`
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
        mean_free_path: Float64[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float64[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: Tuple[Tuple[Tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        rebuilt_radial: RadialSpec
        rebuilt_params: MatrixElementParams
        rebuilt_mfp: Float64[Array, ""]
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
    """Validate :func:`diffpes.matrixel.matrix_element_phase_gauge_direction`.

    :see: :func:`diffpes.matrixel.matrix_element_phase_gauge_direction`
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
        mean_free_path: Float64[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float64[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: Tuple[Tuple[Tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        direction: Float64[Array, " n_theta"] = (
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

        def intensity(
            candidate: Float64[Array, " n_theta"],
        ) -> Float64[Array, ""]:
            """Return one coherent generic-complex orbital intensity."""
            rebuilt_radial: RadialSpec
            rebuilt_params: MatrixElementParams
            rebuilt_mfp: Float64[Array, ""]
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
            bvals: Complex128[Array, "1 4 2"] = jnp.array(
                [
                    [
                        [0.0 + 0.0j, 0.7 + 0.2j],
                        [0.4 - 0.3j, -0.2 + 0.8j],
                        [-0.1 + 0.5j, 0.6 + 0.1j],
                        [0.3 + 0.2j, -0.4 + 0.7j],
                    ]
                ]
            )
            channels: Complex128[Array, "1 1 4 3"] = (
                orbital_transition_channels(
                    jnp.array([[0.1, -0.2, 0.0]]),
                    jnp.array([[0.3, 0.2, 1.1]]),
                    jnp.zeros((4, 3)),
                    jnp.zeros(4),
                    bvals,
                    rebuilt_params,
                    rebuilt_mfp,
                    params.basis,
                )
            )
            polarized: Complex128[Array, "1 1 4"] = contract_polarization(
                channels,
                jnp.array([0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j]),
            )
            result: Float64[Array, ""] = jnp.abs(jnp.sum(polarized)) ** 2
            return result

        reference: Float64[Array, ""] = intensity(flat)
        alpha: float
        for alpha in (-2.0, -0.3, 0.8, 2.4):
            chex.assert_trees_all_close(
                intensity(flat + alpha * direction),
                reference,
                rtol=1e-13,
                atol=1e-13,
            )
        derivative: Float64[Array, ""] = jax.jvp(
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

    :see: :func:`diffpes.matrixel.radial_coefficient_scale_gauge_directions`
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
        mean_free_path: Float64[Array, ""]
        radial, params, mean_free_path = _packing_fixture()
        flat: Float64[Array, " n_theta"]
        tree_definition: PyTreeDef
        metadata: Tuple[Tuple[Tuple[int, ...], bool], ...]
        flat, tree_definition, metadata = pack_matrixel_params(
            radial,
            params,
            mean_free_path,
        )
        directions: Float64[Array, "n_gauge n_theta"] = (
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
            candidate: Float64[Array, " n_theta"],
        ) -> Float64[Array, "n_orb n_r"]:
            """Return normalized orbital radial rows."""
            rebuilt: RadialSpec = unpack_matrixel_params(
                candidate,
                tree_definition,
                metadata,
                radial,
                params,
            )[0]
            values: Float64[Array, "n_orb n_r"] = evaluate_radial(
                rebuilt,
                jnp.linspace(0.01, 5.0, 41),
            )
            return values

        direction: Float64[Array, " n_theta"]
        for direction in directions:
            derivative: Float64[Array, "n_orb n_r"] = jax.jvp(
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
    """Validate :func:`diffpes.matrixel.band_group_weight_sensitivity`.

    :see: :func:`diffpes.matrixel.band_group_weight_sensitivity`
    """

    @staticmethod
    def _rebuild(
        candidate: Float64[Array, " n_theta"],
        bands: DiagonalizedBands,
        experiment: ExperimentGeometry,
    ) -> Complex128[Array, "n_k n_bands 2"]:
        """PRIVATE: Build generic two-spin amplitudes from orbital rows.

        Parameters
        ----------
        candidate : Float64[Array, " n_theta"]
            Two real parameters that mix the base row and its
            conjugate.
        bands : DiagonalizedBands
            Carrier whose eigenvectors project the orbital rows.
        experiment : ExperimentGeometry
            Unused registered argument; deleted immediately.

        Returns
        -------
        amplitudes : Complex128[Array, "n_k n_bands 2"]
            Per-band two-spin amplitudes.

        Implementation Logic
        --------------------
        Forms a generic complex base vector from each 1-based orbital
        index. Builds two parameter-dependent spin rows from the base
        and its conjugate. Contracts eigenvectors with those rows
        through einsum kbo,os->kbs.
        """
        del experiment
        n_orbitals: int = bands.eigenvectors.shape[-1]
        index: Int64[Array, " n_orb"] = jnp.arange(
            1,
            n_orbitals + 1,
            dtype=jnp.float64,
        )
        base: Complex128[Array, " n_orb"] = index * (0.3 + 0.2j) + (0.1 - 0.4j)
        orbital_rows: Complex128[Array, "n_orb 2"] = jnp.stack(
            (
                (1.0 + candidate[0]) * base + candidate[1] * jnp.conj(base),
                (0.4 - 0.3 * candidate[0]) * jnp.conj(base)
                + 0.2j * candidate[1] * base,
            ),
            axis=-1,
        )
        amplitudes: Complex128[Array, "n_k n_bands 2"] = jnp.einsum(
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
        flat: Float64[Array, " 2"] = jnp.array([0.2, -0.15])
        group: Tuple[int, ...] = tuple(range(group_size))
        weights: Float64[Array, "1 1"]
        jacobian: Float64[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            bands,
            experiment,
            (group,),
        )
        unitary: Complex128[Array, "n_group n_group"]
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
            root: Complex128[Array, ""] = jnp.exp(2.0j * jnp.pi / 3.0)
            unitary = jnp.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, root, root**2],
                    [1.0, root**2, root],
                ],
                dtype=jnp.complex128,
            ) / math.sqrt(3.0)
        rotated_eigenvectors: Complex128[Array, "1 n_bands n_orb"] = (
            bands.eigenvectors.at[0, :group_size].set(
                unitary @ bands.eigenvectors[0, :group_size]
            )
        )
        rotated_bands: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            bands,
            rotated_eigenvectors,
        )
        rotated_weights: Float64[Array, "1 1"]
        rotated_jacobian: Float64[Array, "2 1 1"]
        rotated_weights, rotated_jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            rotated_bands,
            experiment,
            (group,),
        )
        original_members: Float64[Array, "1 n_group"] = (
            matrix_element_intensity(
                self._rebuild(flat, bands, experiment)[:, :group_size]
            )
        )
        rotated_members: Float64[Array, "1 n_group"] = (
            matrix_element_intensity(
                self._rebuild(flat, rotated_bands, experiment)[:, :group_size]
            )
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
        """Verify the shared gradient check and incomplete-group rejection.

        A degenerate partner in the complement makes the singleton group
        partial.

        Notes
        -----
        Run both AD modes and the registered scale-aware FD ladder.  Compare
        both parameter columns and plant partial and nonisolated groups.
        """
        bands: DiagonalizedBands = _isolated_group_bands(2)
        experiment: ExperimentGeometry = _sensitivity_experiment()
        flat: Float64[Array, " 2"] = jnp.array([0.13, -0.21])

        def group_weight(
            candidate: Float64[Array, " 2"],
        ) -> Float64[Array, ""]:
            """Return the complete isolated-group weight."""
            candidate_weights: Float64[Array, "1 1"] = (
                band_group_weight_sensitivity(
                    candidate,
                    self._rebuild,
                    bands,
                    experiment,
                    ((0, 1),),
                )[0]
            )
            returned: Float64[Array, ""] = candidate_weights[0, 0]
            return returned

        assert_gradients_match_finite_differences(
            group_weight,
            flat,
            regime="smooth",
            modes=("fwd", "rev"),
            elementwise=True,
        )
        weights: Float64[Array, "1 1"]
        jacobian: Float64[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            flat,
            self._rebuild,
            bands,
            experiment,
            ((0, 1),),
        )
        chex.assert_trees_all_close(
            weights[0, 0],
            group_weight(flat),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        chex.assert_trees_all_close(
            jacobian[:, 0, 0],
            jax.grad(group_weight)(flat),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        with pytest.raises(ValueError, match="cuts a degeneracy"):
            band_group_weight_sensitivity(
                flat,
                self._rebuild,
                bands,
                experiment,
                ((0,),),
            )
        close_eigenvalues: Float64[Array, "1 3"] = bands.eigenvalues.at[
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
        """Verify exact dark behavior and its positive-domain neighbor.

        A linear amplitude crossing produces a quadratic weight and zero slope.

        Notes
        -----
        Use the shared two-mode FD harness at the dark point and its neighbor.
        The neighbor also pins the positive-domain logarithmic derivative.
        """
        bands: DiagonalizedBands = _isolated_group_bands(1)
        experiment: ExperimentGeometry = _sensitivity_experiment()

        def dark_rebuild(
            candidate: Float64[Array, " 2"],
            candidate_bands: DiagonalizedBands,
            candidate_experiment: ExperimentGeometry,
        ) -> Complex128[Array, "1 2 1"]:
            """Build one dark band and one inert complement amplitude."""
            del candidate_bands, candidate_experiment
            amplitudes: Complex128[Array, "1 2 1"] = jnp.array(
                [[[candidate[0] + 1j * candidate[1]], [0.0 + 0.0j]]]
            )
            return amplitudes

        def dark_weight(candidate: Float64[Array, " 2"]) -> Float64[Array, ""]:
            """Return the manufactured complete-group corridor weight."""
            candidate_weights: Float64[Array, "1 1"] = (
                band_group_weight_sensitivity(
                    candidate,
                    dark_rebuild,
                    bands,
                    experiment,
                    ((0,),),
                )[0]
            )
            returned: Float64[Array, ""] = candidate_weights[0, 0]
            return returned

        dark: Float64[Array, " 2"] = jnp.zeros(2)
        assert_grad_matches_fd(
            dark_weight,
            dark,
            regime="smooth",
            modes=("fwd", "rev"),
        )
        weights: Float64[Array, "1 1"]
        jacobian: Float64[Array, "2 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            dark,
            dark_rebuild,
            bands,
            experiment,
            ((0,),),
        )
        chex.assert_trees_all_close(weights, jnp.zeros((1, 1)))
        chex.assert_trees_all_close(jacobian, jnp.zeros((2, 1, 1)))
        chex.assert_tree_all_finite(jacobian)
        dark_log_jacobian: Float64[Array, "2 1 1"]
        dark_valid: Bool[Array, "1 1"]
        dark_log_jacobian, dark_valid = log_band_group_weight_sensitivity(
            weights,
            jacobian,
            1.0e-8,
        )
        chex.assert_trees_all_equal(dark_valid, jnp.array([[False]]))
        chex.assert_trees_all_equal(
            dark_log_jacobian,
            jnp.zeros((2, 1, 1)),
        )

        positive: Float64[Array, " 2"] = jnp.array([0.2, -0.15])
        assert_gradients_match_finite_differences(
            dark_weight,
            positive,
            regime="smooth",
            modes=("fwd", "rev"),
            elementwise=True,
        )
        positive_weights: Float64[Array, "1 1"]
        positive_jacobian: Float64[Array, "2 1 1"]
        positive_weights, positive_jacobian = band_group_weight_sensitivity(
            positive,
            dark_rebuild,
            bands,
            experiment,
            ((0,),),
        )
        positive_log_jacobian: Float64[Array, "2 1 1"]
        positive_valid: Bool[Array, "1 1"]
        positive_log_jacobian, positive_valid = (
            log_band_group_weight_sensitivity(
                positive_weights,
                positive_jacobian,
                1.0e-8,
            )
        )
        chex.assert_trees_all_equal(positive_valid, jnp.array([[True]]))
        chex.assert_trees_all_close(
            positive_log_jacobian[:, 0, 0],
            jax.grad(lambda candidate: jnp.log(dark_weight(candidate)))(
                positive
            ),
            rtol=1.0e-10,
            atol=1.0e-12,
        )


class TestLogBandGroupWeightSensitivity:
    """Validate logarithmic complete-group sensitivity conversion.

    :see: :func:`diffpes.matrixel.log_band_group_weight_sensitivity`
    """

    def test_dark_mask_and_positive_log_identity(self) -> None:
        """Verify the dark sentinel and positive-domain ``dw/w`` identity.

        Adjacent zero and positive weights exercise both validity branches.

        Notes
        -----
        Supply analytic weights and Jacobians and compare every returned entry.
        """
        weights: Float64[Array, " 2"] = jnp.array([0.0, 0.05])
        jacobian: Float64[Array, "2 2"] = jnp.array([[0.0, 0.4], [0.0, -0.2]])
        log_jacobian: Float64[Array, "2 2"]
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
