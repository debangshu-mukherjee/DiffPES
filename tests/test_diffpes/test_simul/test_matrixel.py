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
from beartype.typing import Tuple
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Bool, Complex128, Float64
from numpy.typing import NDArray

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
from tests._gradients import assert_grad_matches_fd, gradient_gate

type MatrixFixture = Tuple[
    DiagonalizedBands,
    RadialSpec,
    MatrixElementParams,
    RadialQuadratureSpec,
    FinalStateSpec,
    ExperimentGeometry,
]


def _s_basis(
    atom_indices: Tuple[int, ...],
    spin: Tuple[int, ...] = (),
) -> OrbitalBasis:
    """PRIVATE: Return a real s-orbital basis for analytic fixtures.

    Parameters
    ----------
    atom_indices : Tuple[int, ...]
        Host atom index for each orbital.
    spin : Tuple[int, ...]
        Optional per-orbital spin labels; empty keeps the basis
        spinless.

    Returns
    -------
    basis : OrbitalBasis
        One 1s orbital (n=1, l=0, m=0) per atom index.

    Notes
    -----
    Repeats the same quantum numbers so only atom index and spin
    distinguish the orbitals.
    """
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
    shell_index: Tuple[int, ...],
) -> MatrixElementParams:
    """PRIVATE: Return unit-scale, zero-phase shell parameters.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata for the fixture.
    shell_index : Tuple[int, ...]
        Shell assignment for each orbital.

    Returns
    -------
    params : MatrixElementParams
        Default parameter carrier from the public factory.

    Notes
    -----
    Passes only the basis and the shell map so every optional scale and
    phase keeps its default value.
    """
    params: MatrixElementParams = make_matrix_element_params(
        basis,
        shell_index,
    )
    return params


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
    basis: OrbitalBasis = _s_basis((0,) * n_bands)
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


def _simple_bands(
    basis: OrbitalBasis,
    atom_positions: Float64[Array, "n_atom 3"],
    *,
    orbital_positions: Float64[Array, "n_orb 3"] | None = None,
    depths: Float64[Array, " n_orb"] | None = None,
) -> DiagonalizedBands:
    """PRIVATE: Return a one-k-point carrier on a unit real-space lattice.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital metadata for the carrier.
    atom_positions : Float64[Array, "n_atom 3"]
        Fractional atom positions on the identity lattice.
    orbital_positions : Float64[Array, "n_orb 3"] | None
        Fractional orbital centres; None keeps the host atom sites.
    depths : Float64[Array, " n_orb"] | None
        Orbital depths in Angstrom; None drops the depth carrier.

    Returns
    -------
    bands : DiagonalizedBands
        Zero-energy identity-eigenvector bands at one zone-center
        k-point.

    Notes
    -----
    Assigns the species X to every atom on the identity lattice in
    Angstrom.
    """
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
    """Validate :func:`diffpes.simul.band_group_weight_sensitivity`.

    :see: :func:`diffpes.simul.band_group_weight_sensitivity`
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
        index: Float64[Array, " n_orb"] = jnp.arange(
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
        """Verify the shared gradient gate and incomplete-group rejection.

        A degenerate partner in the complement makes the singleton group partial.

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
            return candidate_weights[0, 0]

        gradient_gate(
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
            return candidate_weights[0, 0]

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
        gradient_gate(
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

    :see: :func:`diffpes.simul.log_band_group_weight_sensitivity`
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
        explicit: Float64[Array, "2 3"] = jnp.array(
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
        actual: Float64[Array, "2 3"] = resolve_orbital_positions_cart(bands)
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
        expected: Float64[Array, "2 3"] = bands.geometry.positions[
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
        direction: Float64[Array, " 3"] = jnp.array([0.3, -0.4, 0.8])
        direction = direction / jnp.linalg.norm(direction)
        theta: Float64[Array, ""] = jnp.arccos(direction[2])
        phi: Float64[Array, ""] = jnp.arctan2(direction[1], direction[0])
        expected: Float64[Array, " 36"] = real_spherical_harmonics_all(
            5,
            theta,
            phi,
        )
        actual: Float64[Array, " 36"] = real_spherical_harmonics_cartesian_all(
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
        poles: Float64[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        actual: Float64[Array, "2 36"] = (
            real_spherical_harmonics_cartesian_all(poles, 5)
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

        def p_harmonics(
            transverse: Float64[Array, " 2"],
        ) -> Float64[Array, " 2"]:
            """Return the real p_y and p_x rows near the north pole."""
            vector: Float64[Array, " 3"] = jnp.array(
                [transverse[0], transverse[1], 1.0]
            )
            values: Float64[Array, " 4"] = (
                real_spherical_harmonics_cartesian_all(vector, 1)
            )
            result: Float64[Array, " 2"] = values[jnp.asarray([1, 3])]
            return result

        jacobian: Float64[Array, "2 2"] = jax.jacfwd(p_harmonics)(jnp.zeros(2))
        normalization: float = math.sqrt(3.0 / (4.0 * math.pi))
        expected_jacobian: Float64[Array, "2 2"] = jnp.array(
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
        directions: Float64[Array, "2 3"] = jnp.array(
            [[1.0, 0.0, 1.0], [0.0, -2.0, 1.0]]
        )
        actual: Float64[Array, "2 16"] = jax.jit(
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
        positions: Float64[Array, "2 3"] = jnp.array(
            [[-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0]]
        )
        initial: Float64[Array, "1 3"] = jnp.array([[0.63, 0.0, 0.0]])
        final: Float64[Array, "1 3"] = jnp.array([[0.0, 0.0, 1.2]])
        bvals: Complex128[Array, "1 2 2"] = jnp.array(
            [[[0.0 + 0.0j, 0.2 + 1.1j], [0.0 + 0.0j, 0.2 + 1.1j]]]
        )
        channels: Complex128[Array, "1 1 2 3"] = orbital_transition_channels(
            initial,
            final,
            positions,
            jnp.zeros(2),
            bvals,
            params,
            jnp.array(9.0),
            basis,
        )
        amplitude: Complex128[Array, ""] = jnp.sum(
            contract_polarization(
                channels,
                jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
            )
        )
        atomic_amplitude: Complex128[Array, ""] = contract_polarization(
            channels[:, :, :1, :],
            jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
        )[0, 0, 0]
        phase_argument: float = 0.63 * separation / 2.0
        expected_intensity: Float64[Array, ""] = (
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

        translation: Float64[Array, " 3"] = jnp.array([0.21, -0.14, 0.08])
        translated: Complex128[Array, "1 1 2 3"] = orbital_transition_channels(
            initial,
            final,
            positions + translation,
            jnp.zeros(2),
            bvals,
            params,
            jnp.array(9.0),
            basis,
        )
        expected_phase: Complex128[Array, ""] = jnp.exp(
            1j * jnp.dot(initial[0] - final[0], translation)
        )
        chex.assert_trees_all_close(
            translated,
            expected_phase * channels,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_g6_analytic_graphene_structure_factor(self) -> None:
        """Match analytic zeros, maxima, and opposite-valley orientations.

        Equal two-sublattice atomic rows isolate the complete centre-phase
        structure factor without a behavioral comparator.

        Notes
        -----
        Check Gamma, an exact destructive phase, and conjugate plus/minus-K
        dark-state orientations against closed-form complex phases.
        """
        basis: OrbitalBasis = _s_basis((0, 1))
        params: MatrixElementParams = _matrix_params(basis, (0, 1))
        separation: float = 1.7
        valley_phase: float = 2.0 * math.pi / 3.0
        valley_momentum: float = valley_phase / separation
        positions: Float64[Array, "2 3"] = jnp.asarray(
            ((0.0, 0.0, 0.0), (separation, 0.0, 0.0))
        )
        initial_momenta: Float64[Array, "4 3"] = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (math.pi / separation, 0.0, 0.0),
                (valley_momentum, 0.0, 0.0),
                (-valley_momentum, 0.0, 0.0),
            )
        )
        final_momenta: Float64[Array, "4 3"] = jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.2)),
            (4, 3),
        )
        radial_values: Complex128[Array, "4 2 2"] = jnp.broadcast_to(
            jnp.asarray(((0.0j, 1.0j), (0.0j, 1.0j))),
            (4, 2, 2),
        )
        channels: Complex128[Array, "4 1 2 3"] = orbital_transition_channels(
            initial_momenta,
            final_momenta,
            positions,
            jnp.zeros((2,)),
            radial_values,
            params,
            jnp.asarray(9.0),
            basis,
        )
        polarization: Complex128[Array, " 3"] = jnp.asarray(
            (0.0j, 0.0j, 1.0 + 0.0j)
        )
        orbital_amplitudes: Complex128[Array, "4 2"] = contract_polarization(
            channels,
            polarization,
        )[:, 0, :]
        atomic_intensity: Float64[Array, ""] = (
            jnp.abs(orbital_amplitudes[0, 0]) ** 2
        )
        gamma_amplitude: Complex128[Array, ""] = jnp.sum(orbital_amplitudes[0])
        destructive_amplitude: Complex128[Array, ""] = jnp.sum(
            orbital_amplitudes[1]
        )
        chex.assert_trees_all_close(
            jnp.abs(gamma_amplitude) ** 2,
            4.0 * atomic_intensity,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            destructive_amplitude,
            jnp.asarray(0.0j),
            rtol=0.0,
            atol=1.0e-14,
        )

        plus_phase: Complex128[Array, ""] = jnp.exp(
            1j * jnp.asarray(valley_phase)
        )
        minus_phase: Complex128[Array, ""] = jnp.conj(plus_phase)
        chex.assert_trees_all_close(
            orbital_amplitudes[2, 1] / orbital_amplitudes[2, 0],
            plus_phase,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            orbital_amplitudes[3, 1] / orbital_amplitudes[3, 0],
            minus_phase,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        valley_eigenvectors: Complex128[Array, "2 2 2"] = jnp.asarray(
            (
                (
                    (1.0, -jnp.conj(plus_phase)),
                    (1.0, jnp.conj(plus_phase)),
                ),
                (
                    (1.0, -jnp.conj(minus_phase)),
                    (1.0, jnp.conj(minus_phase)),
                ),
            )
        ) / math.sqrt(2.0)
        valley_channels: Complex128[Array, "2 2 1 3"] = project_band_channels(
            channels[2:],
            valley_eigenvectors,
        )
        valley_amplitudes: Complex128[Array, "2 2"] = contract_polarization(
            valley_channels,
            polarization,
        )[:, :, 0]
        chex.assert_trees_all_close(
            valley_amplitudes[:, 0],
            jnp.zeros((2,), dtype=jnp.complex128),
            rtol=0.0,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            jnp.abs(valley_amplitudes[:, 1]) ** 2,
            2.0 * jnp.broadcast_to(atomic_intensity, (2,)),
            rtol=1.0e-14,
            atol=1.0e-14,
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
        mean_free_path: Float64[Array, ""] = jnp.array(8.0)
        common_arguments: Tuple[Array, ...] = (
            jnp.zeros((1, 3)),
            jnp.array([[0.0, 0.0, 1.0]]),
            jnp.zeros((2, 3)),
        )
        radial: Complex128[Array, "1 2 2"] = jnp.array(
            [[[0.0 + 0.0j, 1.0j], [0.0 + 0.0j, 1.0j]]]
        )

        def layer_intensity(mfp: Float64[Array, ""]) -> Float64[Array, ""]:
            """Return the isolated deep-layer intensity."""
            transition: Complex128[Array, "1 1 2 3"] = (
                orbital_transition_channels(
                    *common_arguments,
                    jnp.array([-0.5e-12, depth]),
                    radial,
                    params,
                    mfp,
                    basis,
                )
            )
            result: Float64[Array, ""] = jnp.sum(
                jnp.abs(transition[0, 0, 1]) ** 2
            )
            return result

        channels: Complex128[Array, "1 1 2 3"] = orbital_transition_channels(
            *common_arguments,
            jnp.array([-0.5e-12, depth]),
            radial,
            params,
            mean_free_path,
            basis,
        )
        intensities: Float64[Array, " 2"] = jnp.sum(
            jnp.abs(channels[0, 0]) ** 2,
            axis=-1,
        )
        chex.assert_trees_all_close(
            intensities[1] / intensities[0],
            jnp.exp(-depth / mean_free_path),
            rtol=1e-14,
            atol=1e-14,
        )
        actual_gradient: Float64[Array, ""] = jax.grad(layer_intensity)(
            mean_free_path
        )
        expected_gradient: Float64[Array, ""] = (
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
        channels: Complex128[Array, "2 3"] = jnp.array(
            [[1.0 + 2.0j, -0.3 + 0.4j, 0.7 - 0.2j], [2.0j, 3.0, -1.0j]]
        )
        polarization: Complex128[Array, " 3"] = jnp.array(
            [0.2 + 0.1j, -0.4 + 0.3j, 0.5 - 0.7j]
        )
        expected: Complex128[Array, " 2"] = (
            channels @ polarization[jnp.asarray([1, 2, 0])]
        )
        actual: Complex128[Array, " 2"] = contract_polarization(
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
        channels: Complex128[Array, " 3"] = jnp.array(
            [0.7 + 0.2j, -0.1j, 1.3 - 0.4j]
        )
        polarization_sample: Complex128[Array, " 3"] = jnp.array(
            [math.cos(azimuth), -math.sin(azimuth), 0.0],
            dtype=jnp.complex128,
        )
        expected: Complex128[Array, ""] = contract_polarization(
            channels,
            polarization_sample,
        )
        actual: Complex128[Array, ""] = contract_experiment_polarization(
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
        rows: Complex128[Array, "2 2"] = jnp.array(
            [[1.0 + 2.0j, -0.3j], [0.7 - 0.2j, -1.1 + 0.4j]]
        )
        actual: Complex128[Array, "2 4"] = transition_source(rows)
        expected: Complex128[Array, "2 4"] = jnp.array(
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
        hamiltonian: Complex128[NDArray, "4 4"] = np.asarray(
            [
                [0.2, 0.13 + 0.07j, 0.04j, -0.03],
                [0.13 - 0.07j, -0.4, 0.05 + 0.02j, 0.01j],
                [-0.04j, 0.05 - 0.02j, 0.6, -0.11 + 0.06j],
                [-0.03, -0.01j, -0.11 - 0.06j, -0.1],
            ],
            dtype=np.complex128,
        )
        energy: complex = 0.31 + 0.27j
        rows_numpy: Complex128[NDArray, "2 2"] = np.asarray(
            [[0.7 + 0.2j, -0.1 + 0.4j], [0.3 - 0.5j, -0.6 + 0.1j]],
            dtype=np.complex128,
        )
        source_numpy: Complex128[NDArray, "2 4"] = np.asarray(
            transition_source(jnp.asarray(rows_numpy))
        )
        resolvent: Complex128[NDArray, "4 4"] = np.linalg.inv(
            energy * np.eye(4, dtype=np.complex128) - hamiltonian
        )
        direct: complex = sum(
            source.conj() @ resolvent @ source for source in source_numpy
        )
        eigenvalues: Float64[NDArray, " 4"]
        eigenvectors: Complex128[NDArray, "4 4"]
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        spectral: complex = 0.0 + 0.0j
        outgoing_spin: int
        band: int
        for outgoing_spin in range(2):
            for band in range(4):
                spin_block: Complex128[NDArray, " 2"] = eigenvectors[
                    2 * outgoing_spin : 2 * (outgoing_spin + 1),
                    band,
                ]
                amplitude: complex = rows_numpy[outgoing_spin] @ spin_block
                spectral += abs(amplitude) ** 2 / (energy - eigenvalues[band])
        np.testing.assert_allclose(
            direct, spectral, rtol=1.0e-12, atol=1.0e-12
        )
        coherent_source: Complex128[NDArray, " 4"] = np.sum(
            source_numpy, axis=0
        )
        coherent_control: complex = (
            coherent_source.conj() @ resolvent @ coherent_source
        )
        assert not np.isclose(
            coherent_control,
            direct,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

        hamiltonian_jax: Complex128[Array, "4 4"] = jnp.asarray(hamiltonian)
        resolvent_jax: Complex128[Array, "4 4"] = jnp.linalg.inv(
            energy * jnp.eye(4, dtype=jnp.complex128) - hamiltonian_jax
        )
        direction: Complex128[Array, "2 2"] = jnp.asarray(
            [[0.2 - 0.3j, 0.1 + 0.05j], [-0.08j, 0.17 + 0.11j]]
        )

        def response(parameter: Float64[Array, ""]) -> Complex128[Array, ""]:
            """Return the dense response along one real row direction."""
            rows: Complex128[Array, "2 2"] = (
                jnp.asarray(rows_numpy) + parameter * direction
            )
            sources: Complex128[Array, "2 4"] = transition_source(rows)
            values: Complex128[Array, " 2"] = jnp.einsum(
                "si,ij,sj->s",
                jnp.conj(sources),
                resolvent_jax,
                sources,
            )
            result: Complex128[Array, ""] = jnp.sum(values)
            return result

        derivative: Complex128[Array, ""] = jax.jacfwd(response)(
            jnp.asarray(0.0)
        )
        step: float = 1.0e-5
        finite_difference: Complex128[Array, ""] = (
            response(jnp.asarray(step)) - response(jnp.asarray(-step))
        ) / (2.0 * step)
        chex.assert_trees_all_close(
            derivative,
            finite_difference,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_g10_spinless_generic_complex_dense_resolvent(self) -> None:
        """Match a spinless generic-complex dense resolvent and its slope.

        A three-orbital Hermitian fixture independently exercises the one-spin
        source convention without relying on the SOC block embedding.

        Notes
        -----
        Compare a direct NumPy inverse with its spectral expansion. Reject a
        planted bra row and check the JAX directional derivative by FD.
        """
        hamiltonian: Complex128[NDArray, "3 3"] = np.asarray(
            (
                (0.17, 0.21 + 0.09j, -0.04j),
                (0.21 - 0.09j, -0.38, 0.13 + 0.06j),
                (0.04j, 0.13 - 0.06j, 0.52),
            ),
            dtype=np.complex128,
        )
        energy: complex = 0.29 + 0.23j
        row_numpy: Complex128[NDArray, " 3"] = np.asarray(
            (0.61 + 0.17j, -0.32 + 0.49j, 0.28 - 0.37j),
            dtype=np.complex128,
        )
        source_numpy: Complex128[NDArray, " 3"] = np.asarray(
            transition_source(jnp.asarray(row_numpy[None, :]))
        )[0]
        resolvent_numpy: Complex128[NDArray, "3 3"] = np.linalg.inv(
            energy * np.eye(3, dtype=np.complex128) - hamiltonian
        )
        direct: complex = source_numpy.conj() @ resolvent_numpy @ source_numpy
        eigenvalues: Float64[NDArray, " 3"]
        eigenvectors: Complex128[NDArray, "3 3"]
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        band_amplitudes: Complex128[NDArray, " 3"] = row_numpy @ eigenvectors
        spectral: complex = complex(
            np.sum(np.abs(band_amplitudes) ** 2 / (energy - eigenvalues))
        )
        np.testing.assert_allclose(
            direct,
            spectral,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        planted_bra: Complex128[NDArray, " 3"] = (
            np.conj(row_numpy) @ eigenvectors
        )
        planted_response: complex = complex(
            np.sum(np.abs(planted_bra) ** 2 / (energy - eigenvalues))
        )
        assert not np.isclose(
            planted_response,
            direct,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

        hamiltonian_jax: Complex128[Array, "3 3"] = jnp.asarray(hamiltonian)
        resolvent_jax: Complex128[Array, "3 3"] = jnp.linalg.inv(
            energy * jnp.eye(3, dtype=jnp.complex128) - hamiltonian_jax
        )
        direction: Complex128[Array, " 3"] = jnp.asarray(
            (0.19 - 0.11j, -0.07 + 0.16j, 0.13 + 0.05j)
        )

        def response(parameter: Float64[Array, ""]) -> Complex128[Array, ""]:
            """Return the spinless response along one real row direction."""
            row: Complex128[Array, " 3"] = (
                jnp.asarray(row_numpy) + parameter * direction
            )
            source: Complex128[Array, " 3"] = transition_source(row[None, :])[
                0
            ]
            value: Complex128[Array, ""] = (
                jnp.conj(source) @ resolvent_jax @ source
            )
            return value

        derivative: Complex128[Array, ""] = jax.jacfwd(response)(
            jnp.asarray(0.0)
        )
        step: float = 1.0e-5
        finite_difference: Complex128[Array, ""] = (
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
        transition: Complex128[Array, "1 1 2 3"] = jnp.array(
            [[[[1.0 + 0.3j, 0.2j, -0.4], [0.7j, 1.2, 0.5 - 0.1j]]]]
        )
        eigenvectors: Complex128[Array, "1 1 2"] = jnp.array(
            [[[0.6 + 0.2j, -0.3 + 0.7j]]]
        )
        actual: Complex128[Array, "1 1 1 3"] = project_band_channels(
            transition,
            eigenvectors,
        )
        expected: Complex128[Array, " 3"] = jnp.sum(
            transition[0, 0] * eigenvectors[0, 0, :, None],
            axis=0,
        )
        planted_wrong: Complex128[Array, " 3"] = jnp.sum(
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
        transition: Complex128[Array, "1 1 2 3"] = jnp.array(
            [[[[1.0j, 0.4, -0.2j], [0.3 + 0.7j, -0.1j, 0.8]]]]
        )
        eigenvectors: Complex128[Array, "1 1 2"] = jnp.array(
            [[[0.5 + 0.4j, -0.2 + 0.7j]]]
        )
        phase: Complex128[Array, ""] = jnp.exp(0.83j)
        first: Complex128[Array, "1 1 1"] = contract_polarization(
            project_band_channels(transition, eigenvectors),
            jnp.array([0.2 + 0.1j, 0.4 - 0.3j, -0.5j]),
        )
        second: Complex128[Array, "1 1 1"] = contract_polarization(
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
        phases: Float64[Array, " 4"] = jnp.array([0.0, 0.4, 1.7, math.pi])
        amplitudes: Complex128[Array, "4 2"] = jnp.stack(
            (
                jnp.ones_like(phases, dtype=jnp.complex128),
                jnp.exp(1j * phases),
            ),
            axis=-1,
        ) / math.sqrt(2.0)
        actual: Float64[Array, " 4"] = matrix_element_intensity(amplitudes)
        planted_coherent: Float64[Array, " 4"] = (
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
        """PRIVATE: Build one s-orbital zero-umklapp fixture.

        Returns
        -------
        fixture : MatrixFixture
            Bands, radial spec, parameters, quadrature, final state,
            and experiment for one s orbital.

        Notes
        -----
        Uses fixed radial integrals (0, 1) on both branches. Uses the
        21.2 eV x-polarized experiment and one zone-center k-point. The
        parallel momentum transfer therefore stays zero.
        """
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
        final_momentum: Float64[Array, "1 3"] = jnp.array([[0.0, 0.0, 1.0]])
        first: Complex128[Array, "1 1 1 3"] = (
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
        second: Complex128[Array, "1 1 1 3"] = (
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
        final_momentum: Float64[Array, "1 3"],
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
