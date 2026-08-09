"""Certify end-to-end Slater--Koster depth sensitivity.

Extended Summary
----------------
The test freezes discrete slab topology outside all JAX transformations. One
fundamental Slater--Koster parameter then flows through continuous slab
rebuilding, diagonalization, depth attenuation, coherent matrix elements,
and a complete isolated band-group weight.
"""

from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64

from diffpes.simul import (
    assemble_orbital_transition_channels,
    band_group_weight_sensitivity,
    contract_experiment_polarization,
    matrix_element_intensity,
    polarization_from_angles,
    project_band_channels,
)
from diffpes.tightb import (
    build_sk_model,
    diagonalize_tb,
    freeze_slab_topology,
    rebuild_slab,
    sk_model_parameter_view,
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
    SlabTopology,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_experiment_geometry,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_slater_koster_params,
)
from tests._gradients import gradient_gate

type Capstone = Tuple[
    Float64[Array, " 1"],
    Callable[[Float64[Array, " 1"]], Float64[Array, ""]],
    Callable[
        [Float64[Array, " 1"], DiagonalizedBands, ExperimentGeometry],
        Complex128[Array, "1 n_bands 1"],
    ],
    Callable[
        [Float64[Array, " 1"], Float64[Array, ""]],
        Float64[Array, ""],
    ],
    DiagonalizedBands,
    ExperimentGeometry,
]


def _bulk_context() -> Tuple[
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    Float64[Array, " 2"],
]:
    """PRIVATE: Build a depth-asymmetric two-site s-orbital SK context.

    Returns
    -------
    geometry : CrystalGeometry
        Tetragonal cell in Angstrom with two sites split along z.
    basis : OrbitalBasis
        One 1s orbital on each site.
    sk_params : SlaterKosterParams
        Single fundamental X-Y ss_sigma integral of -1.1 eV.
    onsite : Float64[Array, " 2"]
        Distinct onsite energies in eV that break site symmetry.

    Notes
    -----
    The 0.35 fractional z offset of the second site makes the slab
    depths asymmetric after cleaving.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        jnp.diag(jnp.asarray([3.0, 3.0, 4.0])),
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.35]]),
        ("X", "Y"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
    )
    sk_params: SlaterKosterParams = make_slater_koster_params(
        jnp.asarray([-1.1]),
        ("X-Y:ss_sigma",),
    )
    onsite: Float64[Array, " 2"] = jnp.asarray([0.2, -0.35])
    context: Tuple[
        CrystalGeometry,
        OrbitalBasis,
        SlaterKosterParams,
        Float64[Array, " 2"],
    ] = (geometry, basis, sk_params, onsite)
    return context


def _capstone() -> Capstone:
    """PRIVATE: Build the frozen-topology loss and registered group callback.

    Returns
    -------
    capstone : Capstone
        Active ss_sigma parameter slice, scalar group-weight loss,
        band-amplitude callback, explicit depth-scale loss, baseline
        diagonalized bands, and the registered experiment geometry.

    Implementation Logic
    --------------------
    Freezes the (0, 0, 1) slab topology from the direct bulk model and
    closes over a parameter view that maps the single active ss_sigma
    value back into the full vector. The closures rebuild and
    diagonalize the slab, scale the depth carrier, assemble
    fixed-integral matrix-element channels, project them onto bands,
    contract the p-polarized experiment, and sum the two lowest band
    intensities into the isolated group weight.
    """
    geometry: CrystalGeometry
    basis: OrbitalBasis
    sk_params: SlaterKosterParams
    onsite: Float64[Array, " 2"]
    geometry, basis, sk_params, onsite = _bulk_context()
    direct_bulk: TBModel = build_sk_model(
        geometry,
        basis,
        sk_params,
        onsite,
        jnp.zeros((0,), dtype=jnp.float64),
        (-1, -1),
        1.6,
    )
    parameters: Float64[Array, " n_parameter"]
    rebuild_sk: Callable[[Float64[Array, " n_parameter"]], TBModel]
    parameters, rebuild_sk = sk_model_parameter_view(
        geometry,
        basis,
        sk_params,
        onsite,
        jnp.zeros((0,), dtype=jnp.float64),
        (-1, -1),
        1.6,
    )
    topology: SlabTopology = freeze_slab_topology(
        direct_bulk,
        miller=(0, 0, 1),
        thickness_ang=1.5,
        vacuum_ang=4.0,
    )
    baseline_bulk: TBModel = rebuild_sk(parameters)
    baseline_slab: TBModel
    baseline_slab, _ = rebuild_slab(baseline_bulk, topology)
    n_orbitals: int = len(baseline_slab.basis.n)
    shell_map: Tuple[int, ...] = tuple(range(n_orbitals))
    radial: RadialSpec = make_radial_spec(
        baseline_slab.basis,
        shell_map,
        mode="fixed",
        fixed_integrals_shell=jnp.tile(
            jnp.asarray([[0.0, 1.0]]),
            (n_orbitals, 1),
        ),
    )
    matrix_params: MatrixElementParams = make_matrix_element_params(
        baseline_slab.basis,
        shell_map,
        sigma_shell=jnp.asarray([1.0, 0.81, 1.17, 0.93]),
        phase_shift_angles_shell=jnp.asarray([0.13, -0.29, 0.47, -0.61]),
    )
    quadrature: RadialQuadratureSpec = make_radial_quadrature_spec()
    final_state: FinalStateSpec = make_final_state_spec()
    polarization: Complex128[Array, " 3"] = polarization_from_angles(
        jnp.asarray(0.6),
        jnp.asarray(0.2),
        "p",
    )
    experiment: ExperimentGeometry = make_experiment_geometry(
        24.0,
        polarization,
        incidence_theta=0.6,
        incidence_phi=0.2,
        sample_azimuth=0.31,
        work_function_ev=4.5,
        mean_free_path_ang=2.7,
    )
    kpoints: Float64[Array, "1 3"] = jnp.zeros((1, 3), dtype=jnp.float64)
    final_momentum: Float64[Array, "1 3"] = jnp.asarray([[0.0, 0.0, 1.7]])
    validity: Bool[Array, " 1"] = jnp.asarray([True])
    active: Float64[Array, " 1"] = parameters[:1]

    def bands_for(candidate: Float64[Array, " 1"]) -> DiagonalizedBands:
        """Build and diagonalize the depth-bearing slab."""
        vector: Float64[Array, " n_parameter"] = parameters.at[:1].set(
            candidate
        )
        bulk: TBModel = rebuild_sk(vector)
        slab: TBModel
        slab, _ = rebuild_slab(bulk, topology)
        bands: DiagonalizedBands = diagonalize_tb(slab, kpoints)
        return bands

    def amplitudes_with_depth_scale(
        candidate: Float64[Array, " 1"],
        registered_experiment: ExperimentGeometry,
        depth_scale: Float64[Array, ""],
    ) -> Complex128[Array, "1 n_bands 1"]:
        """Return amplitudes after scaling only the slab depth carrier."""
        bands: DiagonalizedBands = bands_for(candidate)
        if bands.depths is None:
            message: str = "depth sensitivity requires the slab depth carrier"
            raise AssertionError(message)
        scaled_bands: DiagonalizedBands = eqx.tree_at(
            lambda carrier: carrier.depths,
            bands,
            depth_scale * bands.depths,
        )
        channels: Complex128[Array, "1 1 n_orb 3"] = (
            assemble_orbital_transition_channels(
                scaled_bands,
                radial,
                matrix_params,
                quadrature,
                final_state,
                registered_experiment,
                final_momentum,
                validity,
            )
        )
        band_channels: Complex128[Array, "1 n_bands 1 3"] = (
            project_band_channels(
                channels,
                scaled_bands.eigenvectors,
            )
        )
        amplitudes: Complex128[Array, "1 n_bands 1"] = (
            contract_experiment_polarization(
                band_channels,
                registered_experiment,
            )
        )
        return amplitudes

    def amplitudes_for(
        candidate: Float64[Array, " 1"],
        _registered_bands: DiagonalizedBands,
        registered_experiment: ExperimentGeometry,
    ) -> Complex128[Array, "1 n_bands 1"]:
        """Return late-contracted amplitudes for every slab band."""
        amplitudes: Complex128[Array, "1 n_bands 1"] = (
            amplitudes_with_depth_scale(
                candidate,
                registered_experiment,
                jnp.asarray(1.0),
            )
        )
        return amplitudes

    baseline_bands: DiagonalizedBands = bands_for(active)

    def depth_loss(
        candidate: Float64[Array, " 1"],
        depth_scale: Float64[Array, ""],
    ) -> Float64[Array, ""]:
        """Return the group weight at one explicit depth-carrier scale."""
        amplitudes: Complex128[Array, "1 n_bands 1"] = (
            amplitudes_with_depth_scale(
                candidate,
                experiment,
                depth_scale,
            )
        )
        band_weights: Float64[Array, "1 n_bands"] = matrix_element_intensity(
            amplitudes
        )
        group_weight: Float64[Array, ""] = jnp.sum(band_weights[:, :2])
        return group_weight

    def loss(candidate: Float64[Array, " 1"]) -> Float64[Array, ""]:
        """Return the attenuated isolated lower-doublet group weight."""
        group_weight: Float64[Array, ""] = depth_loss(
            candidate,
            jnp.asarray(1.0),
        )
        return group_weight

    capstone: Capstone = (
        active,
        loss,
        amplitudes_for,
        depth_loss,
        baseline_bands,
        experiment,
    )
    return capstone


class TestSlaterKosterDepthGradient:
    """Certify the SK-to-depth-to-group-weight derivative capstone."""

    @pytest.mark.rss_limit_mb(1800)
    def test_sk_parameter_reaches_complete_group_weight(self) -> None:
        """Match both AD modes and retain nonzero sensitivity.

        The lower two bands form one exact degenerate doublet isolated from
        the upper doublet. Depth asymmetry makes its coherent weight sensitive
        to the fundamental ss-sigma integral.

        Notes
        -----
        Run the stiff finite-difference ladder and compare the registered group helper.
        """
        active: Float64[Array, " 1"]
        loss: Callable[[Float64[Array, " 1"]], Float64[Array, ""]]
        amplitudes_for: Callable[
            [Float64[Array, " 1"], DiagonalizedBands, ExperimentGeometry],
            Complex128[Array, "1 n_bands 1"],
        ]
        depth_loss: Callable[
            [Float64[Array, " 1"], Float64[Array, ""]],
            Float64[Array, ""],
        ]
        baseline_bands: DiagonalizedBands
        experiment: ExperimentGeometry
        (
            active,
            loss,
            amplitudes_for,
            depth_loss,
            baseline_bands,
            experiment,
        ) = _capstone()

        gradient_gate(
            loss,
            active,
            regime="stiff",
            modes=("fwd", "rev"),
            elementwise=True,
        )
        weights: Float64[Array, "1 1"]
        jacobian: Float64[Array, "1 1 1"]
        weights, jacobian = band_group_weight_sensitivity(
            active,
            amplitudes_for,
            baseline_bands,
            experiment,
            ((0, 1),),
        )
        expected_gradient: Float64[Array, " 1"] = jax.grad(loss)(active)
        chex.assert_trees_all_close(
            weights[0, 0],
            loss(active),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        chex.assert_trees_all_close(
            jacobian[:, 0, 0],
            expected_gradient,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        assert float(jnp.abs(expected_gradient[0])) > 1.0e-8
        zero_depth_gradient: Float64[Array, " 1"] = jax.grad(
            lambda candidate: depth_loss(candidate, jnp.asarray(0.0))
        )(active)
        depth_contribution: Float64[Array, " 1"] = (
            expected_gradient - zero_depth_gradient
        )
        assert float(jnp.abs(depth_contribution[0])) > 1.0e-8
        depth_scale_derivative: Float64[Array, ""] = jax.grad(
            lambda scale: depth_loss(active, scale)
        )(jnp.asarray(1.0))
        assert float(jnp.abs(depth_scale_derivative)) > 1.0e-8
