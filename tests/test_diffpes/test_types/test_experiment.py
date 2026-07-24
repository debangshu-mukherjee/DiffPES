"""Validate the carrier for the geometry of an ARPES experiment.

The tests cover the static slit field, traced validation, polarization
normalization, PyTree reconstruction, and sensitivity of every numerical
field.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies
from jaxtyping import Array, Complex, Float

from diffpes.types import ExperimentGeometry, make_experiment_geometry
from tests._assertions import assert_rejects
from tests._gradients import gradient_gate


class TestExperimentGeometry:
    """Validate :class:`~diffpes.types.ExperimentGeometry` as a JAX PyTree.

    The cases cover reconstruction and the exact division between numerical
    leaves and the static slit selector.

    :see: :class:`~diffpes.types.ExperimentGeometry`
    """

    def test_preserves_the_carrier_through_a_tree_round_trip(self) -> None:
        """Preserve all fields through a JAX tree round trip.

        The restored carrier must retain equal numerical leaves and the same
        static slit selector.

        Notes
        -----
        The test flattens one geometry with JAX and reconstructs it from the
        resulting leaves and tree definition.
        """
        polarization: Complex[Array, "3"] = jnp.array(
            [1.0 + 0.0j, 0.0 + 2.0j, 0.0 + 0.0j]
        )
        geometry: ExperimentGeometry = make_experiment_geometry(
            21.2, polarization, sample_azimuth=0.2, slit="V"
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree.flatten(geometry)
        restored: ExperimentGeometry = jax.tree.unflatten(tree, leaves)

        chex.assert_trees_all_close(restored, geometry)
        chex.assert_equal(restored.slit, "V")

    def test_keeps_only_the_slit_out_of_the_numerical_leaves(self) -> None:
        """Keep all 11 numerical fields in the traced partition.

        The static slit must not appear among the leaves that JAX transforms.

        Notes
        -----
        The test partitions the carrier with :func:`equinox.is_array` and
        counts the numerical leaves in the dynamic result.
        """
        geometry: ExperimentGeometry = make_experiment_geometry(
            21.2, jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]), slit="V"
        )
        dynamic: ExperimentGeometry
        static: ExperimentGeometry
        dynamic, static = eqx.partition(geometry, eqx.is_array)
        leaves: list[Array] = jax.tree.leaves(dynamic)

        chex.assert_equal(len(leaves), 11)
        chex.assert_equal(dynamic.slit, "V")
        chex.assert_equal(static.slit, "V")

    def test_crosses_a_compiled_boundary_as_one_carrier(self) -> None:
        """Pass the complete carrier through compiled execution.

        The compiled function must read traced energy and polarization fields
        while it preserves the static slit selection.

        Notes
        -----
        The test compiles a scalar reduction over two numerical fields and
        compares it with the same closed-form sum.
        """
        geometry: ExperimentGeometry = make_experiment_geometry(
            30.0, jnp.array([1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j])
        )

        def reduce_carrier(candidate: ExperimentGeometry) -> Float[Array, ""]:
            """Reduce two traced fields to one scalar."""
            result: Float[Array, ""] = candidate.photon_energy_ev + jnp.real(
                candidate.polarization[0]
            )
            return result

        actual: Float[Array, ""] = eqx.filter_jit(reduce_carrier)(geometry)
        expected: Float[Array, ""] = 30.0 + 1.0 / jnp.sqrt(2.0)
        chex.assert_trees_all_close(actual, expected, atol=1e-15, rtol=1e-15)


class TestMakeExperimentGeometry:
    """Validate :func:`~diffpes.types.make_experiment_geometry`.

    The cases cover every range check under JIT, the polarization gauge, and
    gradients through all numerical experiment fields.

    :see: :func:`~diffpes.types.make_experiment_geometry`
    """

    @given(
        strategies.tuples(
            *[
                strategies.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
                for _ in range(4)
            ]
        ).filter(lambda values: sum(value * value for value in values) > 1e-4)
    )
    @settings(max_examples=20, deadline=None)
    def test_normalizes_each_nonzero_complex_polarization(
        self, components: tuple[float, float, float, float]
    ) -> None:
        """Normalize each accepted complex polarization to unit norm.

        The norm must equal one within ``1e-15`` for generic complex vectors.

        Notes
        -----
        Hypothesis supplies four bounded transverse components. The test
        combines them into two complex values and checks the Hermitian vector
        norm.
        """
        polarization: Complex[Array, "3"] = jnp.array(
            [
                components[0] + 1j * components[1],
                components[2] + 1j * components[3],
                0.0 + 0.0j,
            ]
        )
        geometry: ExperimentGeometry = make_experiment_geometry(
            21.2, polarization
        )
        norm: Float[Array, ""] = jnp.linalg.norm(geometry.polarization)

        chex.assert_trees_all_close(norm, 1.0, atol=1e-15, rtol=1e-15)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("photon_energy_ev", 0.0, "photon_energy_ev"),
            ("photon_energy_ev", jnp.nan, "photon_energy_ev"),
            ("incidence_theta", jnp.nan, "incidence_theta"),
            ("incidence_phi", jnp.nan, "incidence_phi"),
            ("sample_azimuth", jnp.nan, "sample_azimuth"),
            ("work_function_ev", -1.0, "work_function_ev"),
            ("work_function_ev", 21.2, "work_function_ev"),
            ("inner_potential_ev", -1.0, "inner_potential_ev"),
            ("temperature_k", -1.0, "temperature_k"),
            ("temperature_k", 0.0, "temperature_k"),
            ("energy_resolution_ev", -1.0, "energy_resolution_ev"),
            (
                "momentum_resolution_inv_ang",
                -1.0,
                "momentum_resolution_inv_ang",
            ),
            ("mean_free_path_ang", 0.0, "mean_free_path_ang"),
        ],
    )
    def test_rejects_each_invalid_scalar_under_jit(
        self, field: str, value: object, match: str
    ) -> None:
        """Reject each scalar that violates the traced physical domain.

        Eager and compiled execution must report the same field-specific
        validation failure.

        Notes
        -----
        The test changes one default input and delegates both execution modes
        to the shared rejection assertion.
        """
        arguments: dict[str, object] = {
            "photon_energy_ev": 21.2,
            "polarization": jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        }
        arguments[field] = value
        assert_rejects(make_experiment_geometry, **arguments, match=match)

    @pytest.mark.parametrize(
        "polarization",
        [
            jnp.zeros((3,), dtype=jnp.complex128),
            jnp.array([jnp.nan + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        ],
    )
    def test_rejects_invalid_polarization_under_jit(
        self, polarization: Complex[Array, "3"]
    ) -> None:
        """Reject a zero or non-finite polarization under JIT.

        The factory must keep both invalid branches finite before it raises
        the applicable runtime error.

        Notes
        -----
        The shared assertion calls the factory eagerly and through
        :func:`equinox.filter_jit`.
        """
        assert_rejects(
            make_experiment_geometry,
            21.2,
            polarization,
            match="polarization",
        )

    @pytest.mark.parametrize(
        ("theta", "phi", "polarization"),
        [
            (
                0.0,
                0.0,
                jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
            ),
            (
                jnp.pi / 2.0,
                0.0,
                jnp.array([1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            ),
        ],
    )
    def test_rejects_longitudinal_polarization_under_jit(
        self,
        theta: object,
        phi: object,
        polarization: Complex[Array, "3"],
    ) -> None:
        """Reject electric fields with a longitudinal photon component.

        Real and complex counterexamples must fail with the same
        transversality diagnostic in eager and compiled execution.

        Notes
        -----
        The shared rejection assertion evaluates each longitudinal field
        directly and through :func:`equinox.filter_jit`.
        """
        assert_rejects(
            make_experiment_geometry,
            21.2,
            polarization,
            incidence_theta=theta,
            incidence_phi=phi,
            match="transverse",
        )

    def test_preserves_transversality_under_incidence_azimuth_rotation(
        self,
    ) -> None:
        """Rotate photon direction and polarization covariantly about z.

        Each azimuth reconstructs the p-polarization basis. Every resulting
        carrier must retain unit norm and zero longitudinal amplitude.

        Notes
        -----
        Vectorize four azimuths, reconstruct the matching photon directions,
        and compare their complex longitudinal amplitudes with zero.
        """
        theta: Float[Array, ""] = jnp.asarray(0.43)
        phis: Float[Array, " 4"] = jnp.asarray([-1.1, -0.2, 0.7, 2.2])

        def one_geometry(phi: Float[Array, ""]) -> ExperimentGeometry:
            """Construct a covariantly rotated p-polarized geometry."""
            polarization: Complex[Array, "3"] = jnp.asarray(
                [
                    jnp.cos(theta) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.sin(phi),
                    -jnp.sin(theta),
                ],
                dtype=jnp.complex128,
            )
            return make_experiment_geometry(
                21.2,
                polarization,
                incidence_theta=theta,
                incidence_phi=phi,
            )

        geometries: ExperimentGeometry = eqx.filter_vmap(one_geometry)(phis)
        directions: Float[Array, "4 3"] = jnp.stack(
            (
                jnp.sin(theta) * jnp.cos(phis),
                jnp.sin(theta) * jnp.sin(phis),
                jnp.full_like(phis, jnp.cos(theta)),
            ),
            axis=-1,
        )
        longitudinal: Complex[Array, " 4"] = jnp.sum(
            directions * geometries.polarization,
            axis=-1,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(geometries.polarization, axis=-1),
            jnp.ones_like(phis),
            rtol=1e-15,
            atol=1e-15,
        )
        chex.assert_trees_all_close(
            longitudinal,
            jnp.zeros_like(longitudinal),
            rtol=0.0,
            atol=2e-16,
        )

    def test_rejects_an_unknown_static_slit(self) -> None:
        """Reject a slit selector outside the two detector conventions.

        The static check must fail before JAX traces numerical validation.

        Notes
        -----
        The test supplies ``"X"`` and matches the slit diagnostic in eager
        and compiled execution.
        """
        assert_rejects(
            make_experiment_geometry,
            21.2,
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            slit="X",
            match="slit",
        )

    def test_preserves_sensitivity_of_each_scalar_field(self) -> None:
        """Preserve finite and nonzero gradients through every scalar field.

        Automatic derivatives must agree with central differences at the
        smooth registered geometry.

        Notes
        -----
        A weighted scalar loss reads all ten scalar fields. The shared gate
        checks reverse mode, finite differences, and every gradient entry.
        """
        values: Float[Array, "10"] = jnp.array(
            [30.0, 0.2, -0.3, 0.1, 4.5, 12.0, 20.0, 0.03, 0.02, 8.0]
        )
        weights: Float[Array, "10"] = jnp.arange(1.0, 11.0)

        def loss(candidate: Float[Array, "10"]) -> Float[Array, ""]:
            """Read all scalar fields from one constructed carrier."""
            polarization: Complex[Array, "3"] = jnp.asarray(
                [
                    -jnp.sin(candidate[2]),
                    jnp.cos(candidate[2]),
                    0.0,
                ],
                dtype=jnp.complex128,
            )
            geometry: ExperimentGeometry = make_experiment_geometry(
                candidate[0],
                polarization,
                incidence_theta=candidate[1],
                incidence_phi=candidate[2],
                sample_azimuth=candidate[3],
                work_function_ev=candidate[4],
                inner_potential_ev=candidate[5],
                temperature_k=candidate[6],
                energy_resolution_ev=candidate[7],
                momentum_resolution_inv_ang=candidate[8],
                mean_free_path_ang=candidate[9],
            )
            fields: Float[Array, "10"] = jnp.stack(
                (
                    geometry.photon_energy_ev,
                    geometry.incidence_theta,
                    geometry.incidence_phi,
                    geometry.sample_azimuth,
                    geometry.work_function_ev,
                    geometry.inner_potential_ev,
                    geometry.temperature_k,
                    geometry.energy_resolution_ev,
                    geometry.momentum_resolution_inv_ang,
                    geometry.mean_free_path_ang,
                )
            )
            result: Float[Array, ""] = jnp.sum(weights * fields)
            return result

        gradient_gate(loss, values, modes=("rev",))
