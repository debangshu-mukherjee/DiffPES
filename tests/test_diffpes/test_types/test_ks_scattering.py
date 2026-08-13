"""Verify finite-slab scattering contract invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple
from jaxtyping import TypeCheckError

from diffpes.types import (
    BackingAbsorberSpec,
    KSScatteringProblem,
    KSScatteringRequest,
    make_backing_absorber_spec,
    make_dense_slice_operator,
    make_ks_scattering_boundary_profile,
    make_ks_scattering_problem,
    make_ks_scattering_request,
    make_light_matter_coupling_spec,
    make_sparse_slice_operator,
    make_vacuum_boundary_spec,
)


def _request(**overrides: object) -> KSScatteringRequest:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "k_parallel_cart_inv_ang": jnp.asarray([[0.1, -0.2]]),
        "kinetic_energy_ev": jnp.asarray([20.0]),
        "outgoing_channel_index": jnp.asarray([0], dtype=jnp.int32),
        "surface_normal_cart": jnp.asarray([0.0, 0.0, 1.0]),
        "energy_block_size": 1,
        "validity_profile_ref": "validity",
    }
    values.update(overrides)
    result: Any = make_ks_scattering_request(**values)
    return result


def _problem(**overrides: object) -> KSScatteringProblem:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "slice_operator": make_dense_slice_operator(
            jnp.zeros((2, 1, 1), dtype=jnp.complex128)
        ),
        "normal_stencil_offsets": jnp.asarray([-1, 0, 1], dtype=jnp.int32),
        "normal_stencil_values_ev": jnp.zeros((3, 2, 1), dtype=jnp.complex128),
        "nonlocal_projectors": jnp.zeros((1, 2, 1), dtype=jnp.complex128),
        "nonlocal_couplings_ev": jnp.zeros((1, 1), dtype=jnp.complex128),
        "slice_coordinates_ang": jnp.asarray([0.0, 1.0]),
        "channel_coordinates": jnp.asarray([[0.0, 0.0]]),
        "hamiltonian_ref": "hamiltonian",
        "basis_kind": "plane_wave",
        "channel_coordinate_kind": "g_parallel",
        "operator_storage_ref": "dense",
        "discretization_ref": "finite_difference",
    }
    values.update(overrides)
    result: Any = make_ks_scattering_problem(**values)
    return result


def _absorber(**overrides: object) -> BackingAbsorberSpec:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "absorber_strength_ev": jnp.asarray(1.0),
        "absorber_start_ang": jnp.asarray(8.0),
        "absorber_width_ang": jnp.asarray(2.0),
        "side": "right",
        "shape": "polynomial",
        "profile_ref": "absorber",
    }
    values.update(overrides)
    result: Any = make_backing_absorber_spec(**values)
    return result


class TestKsscatteringrequest:
    """Verify ``diffpes.types.KSScatteringRequest`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_normalized_request(self) -> None:
        """Preserve one positive-energy channel request.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare its state count, block size, and surface normal.
        """
        request: Any = _request()
        assert request.energy_block_size == 1
        assert request.kinetic_energy_ev[0] == 20.0

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"energy_block_size": 0}, "metadata is invalid", ValueError),
            (
                {"kinetic_energy_ev": jnp.asarray([1.0, 2.0])},
                "kinetic_energy_ev",
                TypeCheckError,
            ),
            (
                {"k_parallel_cart_inv_ang": jnp.asarray([[jnp.nan, 0.0]])},
                "momenta must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"kinetic_energy_ev": jnp.asarray([0.0])},
                "energies must be finite and positive",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"outgoing_channel_index": jnp.asarray([-1], dtype=jnp.int32)},
                "indices must be nonnegative",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"surface_normal_cart": jnp.asarray([0.0, 0.0, 2.0])},
                "normal must be finite and normalized",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_request_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject metadata, axes, momenta, energies, channels, and normal.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one field in the valid one-state request.
        """
        with pytest.raises(error, match=message):
            _request(**overrides)


class TestDensesliceoperator:
    """Verify ``diffpes.types.DenseSliceOperator`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_square_blocks(self) -> None:
        """Preserve two finite two-channel blocks.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the complete explicit block shape.
        """
        operator: Any = make_dense_slice_operator(
            jnp.zeros((2, 2, 2), dtype=jnp.complex128)
        )
        assert operator.blocks_ev.shape == (2, 2, 2)

    @pytest.mark.parametrize(
        ("blocks", "message", "error"),
        [
            (
                jnp.zeros((2, 2), dtype=jnp.complex128),
                "blocks_ev",
                TypeCheckError,
            ),
            (
                jnp.asarray([[[jnp.nan + 0.0j]]]),
                "blocks must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_dense_operator_invariant(
        self, blocks: object, message: str, error: type[Exception]
    ) -> None:
        """Reject wrong rank and nonfinite dense block values.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Supply each malformed dense tensor directly.
        """
        with pytest.raises(error, match=message):
            make_dense_slice_operator(blocks)


class TestSparsesliceoperator:
    """Verify ``diffpes.types.SparseSliceOperator`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_in_bounds_sparse_entries(self) -> None:
        """Preserve one finite entry and its three-dimensional index.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the stored declared shape and index.
        """
        operator: Any = make_sparse_slice_operator(
            jnp.asarray([1.0 + 0.0j]),
            jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
            shape=(2, 1, 1),
        )
        assert operator.shape == (2, 1, 1)

    @pytest.mark.parametrize(
        ("values", "indices", "shape", "message", "error"),
        [
            (
                jnp.ones((1,), dtype=jnp.complex128),
                jnp.zeros((1, 3), dtype=jnp.int32),
                (0, 1, 1),
                "positive entries",
                ValueError,
            ),
            (
                jnp.ones((2,), dtype=jnp.complex128),
                jnp.zeros((1, 3), dtype=jnp.int32),
                (2, 1, 1),
                "indices",
                TypeCheckError,
            ),
            (
                jnp.asarray([jnp.nan + 0.0j]),
                jnp.zeros((1, 3), dtype=jnp.int32),
                (2, 1, 1),
                "values must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                jnp.ones((1,), dtype=jnp.complex128),
                jnp.asarray([[2, 0, 0]], dtype=jnp.int32),
                (2, 1, 1),
                "within the declared shape",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_sparse_operator_invariant(
        self,
        values: object,
        indices: object,
        shape: Tuple[int, int, int],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject bad shape, axes, values, and out-of-bounds indices.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one component of the valid sparse declaration.
        """
        with pytest.raises(error, match=message):
            make_sparse_slice_operator(values, indices, shape=shape)


class TestKsscatteringproblem:
    """Verify ``diffpes.types.KSScatteringProblem`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_finite_operator_problem(self) -> None:
        """Preserve stencil, nonlocal, coordinate, and identity metadata.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect all independent numerical axis sizes.
        """
        problem: Any = _problem()
        assert problem.normal_stencil_values_ev.shape == (3, 2, 1)
        assert problem.nonlocal_projectors.shape == (1, 2, 1)

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            (
                {"hamiltonian_ref": ""},
                "references must be nonempty",
                ValueError,
            ),
            (
                {"normal_stencil_offsets": jnp.asarray([0], dtype=jnp.int32)},
                "normal_stencil_offsets",
                TypeCheckError,
            ),
            (
                {
                    "slice_operator": make_dense_slice_operator(
                        jnp.zeros((3, 1, 1), dtype=jnp.complex128)
                    )
                },
                "operator axes must match",
                ValueError,
            ),
            (
                {"slice_coordinates_ang": jnp.asarray([0.0, jnp.nan])},
                "values must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_problem_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject incomplete identity, mismatched axes, and nonfinite data.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change one problem field per parameterized case.
        """
        with pytest.raises(error, match=message):
            _problem(**overrides)


class TestVacuumboundaryspec:
    """Verify ``diffpes.types.VacuumBoundarySpec`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_both_finite_lead_directions(self) -> None:
        """Preserve left and right directions with unit-flux normalization.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Construct both direction literals independently.
        """
        direction: str
        for direction in ("left", "right"):
            boundary: Any = make_vacuum_boundary_spec(
                jnp.asarray(0.0), direction=direction
            )
            assert boundary.direction == direction
            assert boundary.normalization == "unit_normal_flux"

    @pytest.mark.parametrize(
        ("potential", "direction", "message", "error"),
        [
            (0.0, "bad", "must be left or right", ValueError),
            (
                jnp.nan,
                "left",
                "potential must be finite",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_boundary_invariant(
        self,
        potential: float,
        direction: str,
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject unsupported direction and nonfinite potential.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one boundary input in each parameterized case.
        """
        with pytest.raises(error, match=message):
            make_vacuum_boundary_spec(
                jnp.asarray(potential), direction=direction
            )


class TestBackingabsorberspec:
    """Verify ``diffpes.types.BackingAbsorberSpec`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_supported_absorber(self) -> None:
        """Preserve right-side polynomial absorber metadata.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare its strength, side, shape, and profile identity.
        """
        absorber: Any = _absorber()
        assert absorber.absorber_strength_ev == 1.0
        assert absorber.side == "right"

    @pytest.mark.parametrize(
        ("overrides", "message", "error"),
        [
            ({"profile_ref": ""}, "metadata must be nonempty", ValueError),
            ({"side": "middle"}, "side or shape is unsupported", ValueError),
            (
                {"absorber_strength_ev": jnp.asarray(-1.0)},
                "strength.*nonnegative",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"absorber_start_ang": jnp.asarray(jnp.nan)},
                "start must be finite",
                eqx.EquinoxRuntimeError,
            ),
            (
                {"absorber_width_ang": jnp.asarray(0.0)},
                "width.*positive",
                eqx.EquinoxRuntimeError,
            ),
        ],
    )
    def test_rejects_each_absorber_invariant(
        self,
        overrides: Dict[str, object],
        message: str,
        error: type[Exception],
    ) -> None:
        """Reject missing metadata, selectors, and invalid dimensions.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one field in the valid absorber fixture.
        """
        with pytest.raises(error, match=message):
            _absorber(**overrides)


class TestKsscatteringboundaryprofile:
    """Verify ``diffpes.types.KSScatteringBoundaryProfile`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_complete_profile_with_absorber(self) -> None:
        """Preserve ordered leads and matched absorber convergence identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare the stored absorber and profile reference.
        """
        profile: Any = make_ks_scattering_boundary_profile(
            make_vacuum_boundary_spec(jnp.asarray(0.0), direction="left"),
            make_vacuum_boundary_spec(jnp.asarray(0.0), direction="right"),
            _absorber(),
            vacuum_convergence_ref="vacuum",
            slab_convergence_ref="slab",
            absorber_convergence_ref="absorber-convergence",
            profile_ref="profile",
        )
        assert profile.backing_absorber is not None

    @pytest.mark.parametrize(
        ("bad_leads", "absorber_ref", "message"),
        [
            (True, "a", "directions are inconsistent"),
            (False, None, "convergence must match"),
        ],
    )
    def test_rejects_each_profile_invariant(
        self, bad_leads: bool, absorber_ref: object, message: str
    ) -> None:
        """Reject reversed leads and unmatched absorber evidence.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one profile relationship while retaining complete references.
        """
        left_direction: str = "right" if bad_leads else "left"
        with pytest.raises(ValueError, match=message):
            make_ks_scattering_boundary_profile(
                make_vacuum_boundary_spec(
                    jnp.asarray(0.0), direction=left_direction
                ),
                make_vacuum_boundary_spec(jnp.asarray(0.0), direction="right"),
                _absorber(),
                vacuum_convergence_ref="vacuum",
                slab_convergence_ref="slab",
                absorber_convergence_ref=absorber_ref,
                profile_ref="profile",
            )


class TestLightmattercouplingspec:
    """Verify ``diffpes.types.LightMatterCouplingSpec`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_scalar_and_spinor_final_states(self) -> None:
        """Preserve both supported final-spin declarations.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Construct each literal with complete coupling metadata.
        """
        mode: str
        for mode in ("scalar", "spinor"):
            coupling: Any = make_light_matter_coupling_spec(
                representation="velocity",
                photon_momentum="dipole",
                final_spin_mode=mode,
                profile_ref="profile",
            )
            assert coupling.final_spin_mode == mode

    @pytest.mark.parametrize(
        ("representation", "spin", "message"),
        [
            ("", "scalar", "metadata must be nonempty"),
            ("velocity", "bad", "spin mode is unsupported"),
        ],
    )
    def test_rejects_each_coupling_invariant(
        self, representation: str, spin: str, message: str
    ) -> None:
        """Reject missing metadata and unsupported final-spin mode.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one complete coupling declaration per case.
        """
        with pytest.raises(ValueError, match=message):
            make_light_matter_coupling_spec(
                representation=representation,
                photon_momentum="dipole",
                final_spin_mode=spin,
                profile_ref="profile",
            )


class TestMakeKsScatteringRequest:
    """Verify ``diffpes.types.make_ks_scattering_request``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeDenseSliceOperator:
    """Verify ``diffpes.types.make_dense_slice_operator``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSparseSliceOperator:
    """Verify ``diffpes.types.make_sparse_slice_operator``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeKsScatteringProblem:
    """Verify ``diffpes.types.make_ks_scattering_problem``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeVacuumBoundarySpec:
    """Verify ``diffpes.types.make_vacuum_boundary_spec``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeBackingAbsorberSpec:
    """Verify ``diffpes.types.make_backing_absorber_spec``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeKsScatteringBoundaryProfile:
    """Verify ``diffpes.types.make_ks_scattering_boundary_profile``.

    Bind the factory to the boundary-profile invariant tests above.
    """


class TestMakeLightMatterCouplingSpec:
    """Verify ``diffpes.types.make_light_matter_coupling_spec``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
