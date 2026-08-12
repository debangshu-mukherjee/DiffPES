"""Validate ARPES polarization and detector-frame functions.

Extended Summary
----------------
Exercise explicit polarization-vector construction and frame transforms.
Compare the detector frame with an offline table from the pinned Chinook
source.

"""

import json
import math
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List, Tuple
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Complex128, Float64

from diffpes.maths import rodrigues_rotation
from diffpes.matrixel import contract_polarization
from diffpes.simul import (
    build_polarization_vectors,
    contract_experiment_polarization,
    detector_angles_to_kpar,
    detector_axis_to_sample,
    detector_rotation,
    final_state_k_inv_ang,
    lab_polarization_to_sample,
    photon_wavevector,
    polarization_from_angles,
    polarization_to_spherical,
    rotate_frame_vectors,
    sample_azimuth_rotation,
)
from diffpes.types import ExperimentGeometry, make_experiment_geometry
from tests._gradients import (
    assert_gradients_match_finite_differences,
    complex_step_derivative,
)


class TestBuildPolarizationVectors(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.build_polarization_vectors`.

    Verifies the geometric properties of the s- and p-polarization basis
    vectors, including mutual orthogonality, unit norm, and correct output
    shape for various incidence angle combinations.

    :see: :func:`~diffpes.simul.build_polarization_vectors`
    """

    def test_orthogonality(self) -> None:
        """Verify that e_s and e_p are mutually orthogonal.

        The test establishes the orthogonality contract for polarization
        vectors with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Calls ``build_polarization_vectors`` with theta=pi/4, phi=0
           to produce s- and p-polarization unit vectors.

        2. **Compute dot product**:
           Takes the inner product of e_s and e_p.

        **Expected assertions**

        The dot product of e_s and e_p is zero (within tolerance 1e-10),
        confirming the two vectors are perpendicular.
        """
        theta: Float64[Array, "..."]
        phi: float
        e_s: Float64[Array, "..."]
        e_p: Float64[Array, "..."]
        dot_product: Float64[Array, "..."]

        theta = jnp.pi / 4.0
        phi = 0.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        dot_product = jnp.dot(e_s, e_p)
        chex.assert_trees_all_close(dot_product, jnp.float64(0.0), atol=1e-10)

    def test_unit_vectors(self) -> None:
        """Verify that both e_s and e_p have unit norm.

        The test establishes the unit-vector contract for polarization vectors
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Call ``build_polarization_vectors`` with theta=pi/3 and
           phi=pi/6.

        2. **Compute norms**:
           Calculates the Euclidean norm of each vector.

        **Expected assertions**

        Both vectors have unit norm within tolerance 1e-10.
        """
        theta: Float64[Array, "..."]
        phi: Float64[Array, "..."]
        e_s: Float64[Array, "..."]
        e_p: Float64[Array, "..."]

        theta = jnp.pi / 3.0
        phi = jnp.pi / 6.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_s),
            jnp.float64(1.0),
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_p),
            jnp.float64(1.0),
            atol=1e-10,
        )

    def test_shape(self) -> None:
        """Verify that the output vectors have the correct 3D shape.

        The test establishes the shape contract for polarization vectors with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Calls ``build_polarization_vectors`` with theta=0.5, phi=0.0.

        2. **Check shapes**:
           Confirms both returned arrays have shape ``(3,)``.

        **Expected assertions**

        Both e_s and e_p have shape ``(3,)``, matching the 3D Cartesian
        coordinate system.
        """
        e_s: Float64[Array, "..."]
        e_p: Float64[Array, "..."]

        e_s, e_p = build_polarization_vectors(0.5, 0.0)
        chex.assert_shape(e_s, (3,))
        chex.assert_shape(e_p, (3,))

    def test_is_smooth_across_the_old_collinearity_threshold(self) -> None:
        """Keep one transverse frame across near-normal incidence angles.

        The basis must not jump when the photon direction crosses the former
        reference-axis threshold. Its angle Jacobian must match the analytic
        trigonometric basis on both sides.

        Notes
        -----
        Evaluate two angles around ``arccos(0.99)`` at a fixed azimuth. Compare
        both bases and both theta derivatives with their closed forms.
        """
        phi: Float64[Array, ""] = jnp.asarray(0.37)
        threshold: Float64[Array, "..."] = jnp.arccos(jnp.asarray(0.99))
        theta: Float64[Array, "..."] = threshold + jnp.asarray([-1e-6, 1e-6])
        e_s: Float64[Array, "..."]
        e_p: Float64[Array, "..."]
        e_s, e_p = jax.vmap(
            lambda value: build_polarization_vectors(value, phi)
        )(theta)
        expected_s: Float64[Array, "..."] = jnp.asarray(
            [jnp.sin(phi), -jnp.cos(phi), 0.0]
        )
        expected_p: Float64[Array, "..."] = jnp.stack(
            (
                -jnp.cos(theta) * jnp.cos(phi),
                -jnp.cos(theta) * jnp.sin(phi),
                jnp.sin(theta),
            ),
            axis=-1,
        )
        derivative: Float64[Array, "..."] = jax.vmap(
            jax.jacfwd(lambda value: build_polarization_vectors(value, phi)[1])
        )(theta)
        expected_derivative: Float64[Array, "..."] = jnp.stack(
            (
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ),
            axis=-1,
        )

        chex.assert_trees_all_close(
            e_s,
            jnp.broadcast_to(expected_s, (2, 3)),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(e_p, expected_p, rtol=0.0, atol=1e-14)
        chex.assert_trees_all_close(
            derivative,
            expected_derivative,
            rtol=0.0,
            atol=1e-14,
        )


class TestPhotonWavevector(chex.TestCase):
    """Validate :func:`~diffpes.simul.photon_wavevector`.

    Covers the spherical-angle convention and unit normalization for photon
    propagation along the surface normal and in the surface plane.

    :see: :func:`~diffpes.simul.photon_wavevector`
    """

    def test_matches_cardinal_incidence_directions(self) -> None:
        """Match normal and grazing incidence to Cartesian unit vectors.

        Map zero polar angle to negative z. Map a right-angle polar angle
        at zero azimuth to negative x within float64 tolerance.

        Notes
        -----
        Evaluate two scalar angle pairs. Compare both vectors with analytic
        Cartesian directions at ``atol=1e-12``. Check their unit norms.
        """
        normal: Float64[Array, "..."]
        grazing: Float64[Array, "..."]

        normal = photon_wavevector(0.0, 0.0)
        grazing = photon_wavevector(jnp.pi / 2.0, 0.0)
        chex.assert_trees_all_close(
            normal,
            jnp.array([0.0, 0.0, -1.0]),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            grazing,
            jnp.array([-1.0, 0.0, 0.0]),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(normal),
            jnp.float64(1.0),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(grazing),
            jnp.float64(1.0),
            rtol=0.0,
            atol=1e-12,
        )


class TestPolarizationFromAngles(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.polarization_from_angles`.

    The tests verify each static polarization selector and the traced linear
    angle against the transverse basis.

    :see: :func:`~diffpes.simul.polarization_from_angles`
    """

    def test_constructs_standard_states(self) -> None:
        """Verify standard states against the transverse basis.

        The test constructs all static states at one incidence geometry and
        compares each vector with its closed-form basis combination.

        Notes
        -----
        Evaluate one generic incidence geometry. Compare four polarization
        kinds with the analytic transverse-basis combinations at 1e-14.
        """
        theta: Float64[Array, ""] = jnp.asarray(0.7)
        phi: Float64[Array, ""] = jnp.asarray(-0.2)
        e_s: Float64[Array, "..."]
        e_p: Float64[Array, "..."]
        e_s, e_p = build_polarization_vectors(theta, phi)
        circular: Complex128[Array, "..."] = polarization_from_angles(
            theta, phi, "c+"
        )
        linear: Complex128[Array, "..."] = polarization_from_angles(
            theta,
            phi,
            "linear",
            polarization_angle=jnp.pi / 4.0,
        )
        chex.assert_trees_all_close(
            polarization_from_angles(theta, phi, "s"),
            e_s.astype(jnp.complex128),
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            polarization_from_angles(theta, phi, "p"),
            e_p.astype(jnp.complex128),
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            circular,
            (e_s + 1j * e_p) / jnp.sqrt(2.0),
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            linear,
            (e_s + e_p) / jnp.sqrt(2.0),
            atol=1e-14,
        )

    def test_rejects_unknown_kind(self) -> None:
        """Verify rejection of an unknown polarization kind.

        The test calls the constructor with an unregistered static selector
        and checks the specific validation error.

        Notes
        -----
        Pass ``"unknown"`` as the static selector. Require ``ValueError``
        with the polarization-kind message.
        """
        with pytest.raises(ValueError, match="kind must be one of"):
            polarization_from_angles(0.5, 0.0, "unknown")

    def test_circular_labels_match_incoming_helicity_operator(self) -> None:
        """Match both circular labels to their photon-helicity eigenvalues.

        The incoming propagation direction and transverse basis define the
        operator ``i q cross`` without observer-dependent naming.

        Notes
        -----
        Evaluate a generic incidence direction. Apply the Cartesian cross
        product and compare with eigenvalues plus and minus one.
        """
        theta: Float64[Array, ""] = jnp.asarray(0.61)
        phi: Float64[Array, ""] = jnp.asarray(-0.37)
        direction: Float64[Array, "3"] = photon_wavevector(theta, phi)
        plus: Complex128[Array, "3"] = polarization_from_angles(
            theta,
            phi,
            "c+",
        )
        minus: Complex128[Array, "3"] = polarization_from_angles(
            theta,
            phi,
            "c-",
        )
        plus_action: Complex128[Array, "3"] = 1j * jnp.cross(direction, plus)
        minus_action: Complex128[Array, "3"] = 1j * jnp.cross(direction, minus)
        chex.assert_trees_all_close(
            plus_action,
            plus,
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            minus_action,
            -minus,
            rtol=0.0,
            atol=1e-14,
        )


class TestPolarizationToSpherical(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.polarization_to_spherical`.

    The tests pin the Condon-Shortley convention, norm preservation, and the
    complex-linear derivative.

    :see: :func:`~diffpes.simul.polarization_to_spherical`
    """

    def test_matches_closed_form_states(self) -> None:
        """Verify circular and linear closed-form states.

        The test transforms both helicities and Cartesian x polarization and
        compares their spherical components with analytic values.

        Notes
        -----
        Build two circular states and one linear state. Compare their ordered
        spherical components with the Condon-Shortley values at 1e-15.
        """
        root_two: Float64[Array, "..."] = jnp.sqrt(jnp.asarray(2.0))
        sigma_plus: Complex128[Array, "..."] = (
            jnp.asarray(
                [1.0, 1j, 0.0],
                dtype=jnp.complex128,
            )
            / root_two
        )
        sigma_minus: Complex128[Array, "..."] = (
            jnp.asarray(
                [1.0, -1j, 0.0],
                dtype=jnp.complex128,
            )
            / root_two
        )
        x_linear: Complex128[Array, "3"] = jnp.asarray(
            [1.0, 0.0, 0.0],
            dtype=jnp.complex128,
        )
        chex.assert_trees_all_close(
            polarization_to_spherical(sigma_plus),
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.complex128),
            atol=1e-15,
        )
        chex.assert_trees_all_close(
            polarization_to_spherical(sigma_minus),
            jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.complex128),
            atol=1e-15,
        )
        expected_x: Complex128[Array, "..."] = jnp.asarray(
            [1.0 / root_two, 0.0, -1.0 / root_two],
            dtype=jnp.complex128,
        )
        chex.assert_trees_all_close(
            polarization_to_spherical(x_linear),
            expected_x,
            atol=1e-15,
        )

    def test_preserves_norm_and_jvp(self) -> None:
        """Verify norm preservation and the complex-linear JVP.

        The test transforms a generic complex vector and tangent, then checks
        the norm identity and the exact transformed tangent.

        Notes
        -----
        Apply a JAX JVP to a generic complex vector. Compare the norm and
        tangent identities at 1e-14.
        """
        polarization: Complex128[Array, "3"] = jnp.asarray(
            [0.3 + 0.2j, -0.4 + 0.1j, 0.5 - 0.7j],
            dtype=jnp.complex128,
        )
        tangent: Complex128[Array, "3"] = jnp.asarray(
            [-0.2 + 0.8j, 0.6 - 0.3j, 0.1 + 0.4j],
            dtype=jnp.complex128,
        )
        spherical: Complex128[Array, "..."]
        spherical_tangent: Complex128[Array, "..."]
        spherical, spherical_tangent = jax.jvp(
            polarization_to_spherical,
            (polarization,),
            (tangent,),
        )
        chex.assert_trees_all_close(
            jnp.vdot(spherical, spherical).real,
            jnp.vdot(polarization, polarization).real,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            spherical_tangent,
            polarization_to_spherical(tangent),
            atol=1e-14,
        )

    @given(
        components=st.lists(
            st.complex_numbers(
                min_magnitude=1e-3,
                max_magnitude=4.0,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=3,
            max_size=3,
        )
    )
    @settings(max_examples=24, deadline=None)
    def test_preserves_norm_for_generic_complex_vectors(
        self,
        components: List[complex],
    ) -> None:
        r"""Preserve :math:`\sum_q|\epsilon_q|^2=|\epsilon|^2`.

        The spherical transform must be unitary within ``1e-15`` for every
        generic complex Cartesian polarization generated by Hypothesis.

        Notes
        -----
        Each of the three Cartesian components has independently generated
        real and imaginary parts and a finite, nonzero magnitude.
        """
        polarization: Complex128[Array, "..."] = jnp.asarray(
            components,
            dtype=jnp.complex128,
        )
        spherical: Complex128[Array, "..."] = polarization_to_spherical(
            polarization
        )
        actual_norm_squared: Float64[Array, "..."] = jnp.sum(
            jnp.abs(spherical) ** 2
        )
        expected_norm_squared: Float64[Array, "..."] = jnp.sum(
            jnp.abs(polarization) ** 2
        )
        chex.assert_trees_all_close(
            actual_norm_squared,
            expected_norm_squared,
            rtol=1e-15,
            atol=1e-15,
        )

    def test_holomorphic_channels_match_complex_step_and_jvp(self) -> None:
        """Match the spherical-basis derivative by complex step and JVP.

        The repository complex-step harness uses ``imag(f(x+i*h))/h``.
        Phase adjustment makes channel probes real on real inputs. The test
        checks all five nonzero coefficients at ``h=1e-20`` without baseline
        imaginary contamination.

        Notes
        -----
        Compare the complex-step values with both a JAX JVP and the analytic
        Condon--Shortley coefficients at absolute tolerance ``1e-14``.
        """

        def holomorphic_channels(
            values: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Return real-phased probes of every nonzero transform entry."""
            zero: Float64[Array, ""] = jnp.zeros((), dtype=values.dtype)
            x_minus: Complex128[Array, "..."] = polarization_to_spherical(
                jnp.stack((values[0], zero, zero))
            )[0]
            y_minus: Complex128[Array, "..."] = (
                1j
                * polarization_to_spherical(
                    jnp.stack((zero, values[1], zero))
                )[0]
            )
            z_zero: Complex128[Array, "..."] = polarization_to_spherical(
                jnp.stack((zero, zero, values[2]))
            )[1]
            x_plus: Complex128[Array, "..."] = -polarization_to_spherical(
                jnp.stack((values[3], zero, zero))
            )[2]
            y_plus: Complex128[Array, "..."] = (
                1j
                * polarization_to_spherical(
                    jnp.stack((zero, values[4], zero))
                )[2]
            )
            channels: Float64[Array, "..."] = jnp.stack(
                (x_minus, y_minus, z_zero, x_plus, y_plus)
            )
            return channels

        inputs: Float64[Array, "5"] = jnp.asarray([0.3, -0.4, 0.7, -0.2, 0.5])
        tangent: Float64[Array, "..."] = jnp.ones_like(inputs)
        complex_step: Float64[Array, "..."] = complex_step_derivative(
            holomorphic_channels,
            inputs,
            h=1e-20,
        )
        _: Float64[Array, "..."]
        jvp: Float64[Array, "..."]
        _, jvp = jax.jvp(
            holomorphic_channels,
            (inputs.astype(jnp.complex128),),
            (tangent.astype(jnp.complex128),),
        )
        inverse_root_two: Float64[Array, "..."] = 1.0 / jnp.sqrt(2.0)
        expected: Float64[Array, "..."] = jnp.asarray(
            [
                inverse_root_two,
                inverse_root_two,
                1.0,
                inverse_root_two,
                inverse_root_two,
            ]
        )
        chex.assert_trees_all_close(
            complex_step,
            expected,
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            jvp,
            expected.astype(jnp.complex128),
            rtol=0.0,
            atol=1e-14,
        )


class TestDetectorRotation(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.detector_rotation`.

    The tests compare both slit conventions with closed forms and exercise
    their traced angle derivatives.

    :see: :func:`~diffpes.simul.detector_rotation`
    """

    def test_rotates_reference_direction_for_both_slits(self) -> None:
        """Verify both slit conventions against closed forms.

        The test rotates the reference z direction and compares both results
        with their analytic trigonometric expressions.

        Notes
        -----
        Use two nonzero detector angles. Compare the horizontal and vertical
        directions with their closed forms at 1e-14.
        """
        tx: Float64[Array, ""] = jnp.asarray(0.23)
        ty: Float64[Array, ""] = jnp.asarray(-0.17)
        z_axis: Float64[Array, "3"] = jnp.asarray([0.0, 0.0, 1.0])
        expected_h: Float64[Array, "..."] = jnp.asarray(
            [
                jnp.sin(tx),
                -jnp.cos(tx) * jnp.sin(ty),
                jnp.cos(tx) * jnp.cos(ty),
            ]
        )
        expected_v: Float64[Array, "..."] = jnp.asarray(
            [
                jnp.sin(ty),
                -jnp.sin(tx) * jnp.cos(ty),
                jnp.cos(tx) * jnp.cos(ty),
            ]
        )
        chex.assert_trees_all_close(
            detector_rotation(tx, ty, "H") @ z_axis,
            expected_h,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            detector_rotation(tx, ty, "V") @ z_axis,
            expected_v,
            atol=1e-14,
        )

    def test_is_proper_and_differentiable(self) -> None:
        """Verify proper rotation and a nonzero angle derivative.

        The test checks orthogonality, determinant, and the derivative of one
        emitted-direction component with respect to the first angle.

        Notes
        -----
        Compute one horizontal rotation. Differentiate its x direction and
        compare the result with the analytic cosine at 1e-14.
        """
        angle: Float64[Array, ""] = jnp.asarray(0.31)
        rotation: Float64[Array, "..."] = detector_rotation(angle, -0.19, "H")
        chex.assert_trees_all_close(
            rotation @ rotation.T,
            jnp.eye(3),
            atol=1e-14,
        )
        chex.assert_trees_all_close(jnp.linalg.det(rotation), 1.0, atol=1e-14)
        derivative: Float64[Array, "..."] = jax.grad(
            lambda value: detector_rotation(value, -0.19, "H")[0, 2]
        )(angle)
        chex.assert_trees_all_close(
            derivative,
            jnp.cos(angle),
            atol=1e-14,
        )

    def test_matches_pinned_chinook_detector_artifact(self) -> None:
        """Match the pinned Chinook detector-direction table.

        The artifact contains the full detector rotation and direction. The
        test intentionally excludes its legacy pixel-rotated polarization
        records because a fixed beam is not an analyzer-frame vector.

        Notes
        -----
        Load the offline artifact without a Chinook import. Compare the
        detector-only records on both 5 by 5 angle grids at the recorded
        relative tolerance.
        """
        tests_root: Path = Path(__file__).resolve().parents[2]
        artifact_path: Path = (
            tests_root / "data" / "kspace" / "tilt_polarization_reference.json"
        )
        reference: Dict[str, Any] = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
        expected_mapping: Dict[str, Dict[str, str]] = {
            "H": {
                "active_rotation": "Rx(diffpes_ty) @ Ry(diffpes_tx)",
                "tilt_k_mesh": (
                    "chinook_Tx=-diffpes_tx, chinook_Ty=diffpes_ty"
                ),
            },
            "V": {
                "active_rotation": "Rx(diffpes_tx) @ Ry(diffpes_ty)",
                "tilt_k_mesh": (
                    "chinook_Tx=-diffpes_ty, chinook_Ty=diffpes_tx"
                ),
            },
        }
        self.assertEqual(
            reference["requirement"], "tilt-polarization-reference"
        )
        tolerance: float = float(reference["rtol"])
        slit: str
        for slit in ("H", "V"):
            self.assertEqual(
                reference["mapping"][slit]["active_rotation"],
                expected_mapping[slit]["active_rotation"],
            )
            self.assertEqual(
                reference["mapping"][slit]["tilt_k_mesh"],
                expected_mapping[slit]["tilt_k_mesh"],
            )
            records: List[Dict[str, Any]] = [
                record
                for record in reference["records"]
                if record["slit"] == slit
            ]
            axis_size: int = int(len(records) ** 0.5)
            tx: Float64[Array, "..."] = jnp.asarray(
                [
                    records[index * axis_size]["tx_rad"]
                    for index in range(axis_size)
                ]
            )
            ty: Float64[Array, "..."] = jnp.asarray(
                [records[index]["ty_rad"] for index in range(axis_size)]
            )
            expected_rotations: Float64[Array, "..."] = jnp.asarray(
                [record["rotation_matrix"] for record in records]
            ).reshape(axis_size, axis_size, 3, 3)

            def rotations_for_tx(
                tx_value: Float64[Array, ""],
                ty_axis: Float64[Array, " axis"] = ty,
                slit_value: str = slit,
            ) -> Float64[Array, "axis 3 3"]:
                """Return all rotations for one detector-x tilt."""

                def rotation_for_ty(
                    ty_value: Float64[Array, ""],
                ) -> Float64[Array, "3 3"]:
                    """Return one detector rotation."""
                    rotation: Float64[Array, "3 3"] = detector_rotation(
                        tx_value,
                        ty_value,
                        slit_value,
                    )
                    return rotation

                rotations: Float64[Array, "axis 3 3"] = jax.vmap(
                    rotation_for_ty
                )(ty_axis)
                return rotations

            actual_rotations: Float64[Array, "..."] = jax.vmap(
                rotations_for_tx
            )(tx)
            chex.assert_trees_all_close(
                actual_rotations,
                expected_rotations,
                rtol=tolerance,
                atol=1e-12,
            )
            expected_directions: Float64[Array, "..."] = jnp.asarray(
                [record["detector_direction"] for record in records]
            ).reshape(axis_size, axis_size, 3)
            energy: Float64[Array, ""] = jnp.asarray(35.0)
            momentum: Float64[Array, "..."] = final_state_k_inv_ang(energy)[0]
            actual_k_parallel: Float64[Array, "..."] = detector_angles_to_kpar(
                tx[:, None],
                ty[None, :],
                energy,
                slit,
            )
            chex.assert_trees_all_close(
                actual_k_parallel / momentum,
                expected_directions[..., :2],
                rtol=tolerance,
                atol=1e-12,
            )

    def test_rejects_unknown_slit(self) -> None:
        """Verify rejection of an unknown slit orientation.

        The test calls the frame constructor with an unsupported static slit
        value and checks the validation error.

        Notes
        -----
        Pass ``"X"`` as the slit. Require ``ValueError`` with the slit
        validation message.
        """
        with pytest.raises(ValueError, match="slit must be"):
            detector_rotation(0.0, 0.0, "X")


class TestRotateFrameVectors(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.rotate_frame_vectors`.

    The tests check the detector-grid shape, vector norms, and explicit
    detector/sample composition for detector-fixed axes.

    :see: :func:`~diffpes.simul.rotate_frame_vectors`
    """

    def test_maps_real_vector_over_grid(self) -> None:
        """Verify mapped real-vector values and shape.

        The test rotates one normalized vector over two angle axes and checks
        each result against direct detector-frame multiplication.

        Notes
        -----
        Use a 2 by 3 horizontal angle grid and nonzero sample azimuth. Check
        its shape, norms, and one direct composition at 1e-14.
        """
        vector: Float64[Array, "3"] = jnp.asarray([0.0, 0.0, 1.0])
        tx: Float64[Array, "2"] = jnp.asarray([-0.2, 0.1])
        ty: Float64[Array, "3"] = jnp.asarray([-0.1, 0.0, 0.3])
        sample_azimuth: Float64[Array, ""] = jnp.asarray(0.27)
        rotated: Float64[Array, "..."] = rotate_frame_vectors(
            vector,
            tx,
            ty,
            "H",
            sample_azimuth,
        )
        chex.assert_shape(rotated, (2, 3, 3))
        chex.assert_trees_all_close(
            jnp.linalg.norm(rotated, axis=-1),
            jnp.ones((2, 3)),
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            rotated[1, 2],
            sample_azimuth_rotation(sample_azimuth).T
            @ detector_rotation(tx[1], ty[2], "H")
            @ vector,
            atol=1e-14,
        )

    def test_detector_axis_fixture(self) -> None:
        """Pin detector/sample axis orientation at cardinal rotations.

        The fixture verifies the explicit detector and sample composition at
        a cardinal orientation.

        Notes
        -----
        At the detector origin, a detector x axis is a laboratory x axis.
        A positive quarter-turn sample azimuth maps it to negative sample y.
        """
        detector_x: Float64[Array, "3"] = jnp.asarray([1.0, 0.0, 0.0])
        mapped: Float64[Array, "..."] = rotate_frame_vectors(
            detector_x,
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            "H",
            jnp.pi / 2.0,
        )
        chex.assert_trees_all_close(
            mapped[0, 0],
            jnp.asarray([0.0, -1.0, 0.0]),
            rtol=0.0,
            atol=1e-14,
        )


class TestSampleAzimuthRotation:
    """Validate :func:`~diffpes.simul.sample_azimuth_rotation`.

    The case applies zero degrees and compares the returned rotation with the
    three-dimensional identity matrix.
    """

    def test_zero_azimuth_is_the_identity(self) -> None:
        """Return the proper identity orientation at zero sample azimuth.

        The case evaluates the typesafe public frame constructor.

        Notes
        -----
        Compare the full matrix with the exact Cartesian identity.
        """
        orientation: Float64[Array, "..."] = sample_azimuth_rotation(0.0)

        chex.assert_trees_all_close(
            orientation,
            jnp.eye(3, dtype=jnp.float64),
            rtol=0.0,
            atol=0.0,
        )


class TestLabPolarizationToSample:
    """Validate :func:`~diffpes.simul.lab_polarization_to_sample`.

    The case rotates a laboratory polarization through the inverse sample
    orientation and compares it with a direct matrix reference.
    """

    def test_applies_inverse_sample_orientation(self) -> None:
        """Verify one laboratory field uses the inverse orientation.

        The case uses a generic field and nonzero sample azimuth.

        Notes
        -----
        Compare the public result with direct transpose multiplication.
        """
        polarization: Complex128[Array, "3"] = jnp.asarray(
            (0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j),
            dtype=jnp.complex128,
        )
        orientation: Float64[Array, "..."] = sample_azimuth_rotation(0.37)
        actual: Complex128[Array, "..."] = lab_polarization_to_sample(
            polarization,
            orientation,
        )

        chex.assert_trees_all_close(
            actual,
            orientation.T @ polarization,
            rtol=0.0,
            atol=0.0,
        )


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


class TestDetectorAxisToSample:
    """Validate :func:`~diffpes.simul.detector_axis_to_sample`.

    The case composes detector and sample rotations for one detector axis and
    compares the result with the explicit matrix product.
    """

    def test_composes_detector_and_sample_orientations(self) -> None:
        """Verify one detector-fixed axis uses both active orientations.

        The case uses generic detector angles and sample azimuth.

        Notes
        -----
        Compare the public result with the declared matrix composition.
        """
        axis: Float64[Array, "3"] = jnp.asarray(
            (0.2, -0.7, 0.5), dtype=jnp.float64
        )
        detector_orientation: Float64[Array, "..."] = detector_rotation(
            0.19, -0.31, "V"
        )
        sample_orientation: Float64[Array, "..."] = sample_azimuth_rotation(
            -0.23
        )
        actual: Float64[Array, "..."] = detector_axis_to_sample(
            axis,
            detector_orientation,
            sample_orientation,
        )

        chex.assert_trees_all_close(
            actual,
            sample_orientation.T @ detector_orientation @ axis,
            rtol=0.0,
            atol=0.0,
        )


class TestFrameSemantics(chex.TestCase):
    """Validate fixed-beam and detector-axis frame semantics.

    :see: :func:`~diffpes.simul.lab_polarization_to_sample`
    :see: :func:`~diffpes.simul.detector_axis_to_sample`
    """

    def test_fixed_beam_is_pixel_independent(self) -> None:
        """Verify a laboratory beam does not rotate with detector pixels.

        A sample-frame polarization has one value for the full detector map.
        Detector rotations vary across the same pixels, providing a direct
        counterexample to the removed analyzer-rotation semantics.

        Notes
        -----
        Map one generic complex field through a nonzero sample azimuth and
        broadcast it over a 2 by 3 detector grid. Require exact pixel
        independence and confirm detector orientations are nonconstant.
        """
        polarization_lab: Complex128[Array, "3"] = jnp.asarray(
            [0.2 + 0.5j, -0.3 + 0.1j, 0.7 - 0.2j],
            dtype=jnp.complex128,
        )
        tx: Float64[Array, "2"] = jnp.asarray([-0.2, 0.1])
        ty: Float64[Array, "3"] = jnp.asarray([-0.1, 0.0, 0.3])
        sample_orientation: Float64[Array, "..."] = sample_azimuth_rotation(
            0.37
        )
        polarization_sample: Complex128[Array, "..."] = (
            lab_polarization_to_sample(
                polarization_lab,
                sample_orientation,
            )
        )
        polarization_grid: Complex128[Array, "..."] = jnp.broadcast_to(
            polarization_sample,
            (tx.shape[0], ty.shape[0], 3),
        )
        chex.assert_shape(polarization_grid, (2, 3, 3))
        chex.assert_trees_all_close(
            polarization_grid,
            jnp.broadcast_to(polarization_grid[0, 0], polarization_grid.shape),
            rtol=0.0,
            atol=0.0,
        )
        detector_directions: Float64[Array, "..."] = rotate_frame_vectors(
            jnp.asarray([0.0, 0.0, 1.0]),
            tx,
            ty,
            "H",
            0.37,
        )
        assert not bool(
            jnp.allclose(
                detector_directions,
                detector_directions[0, 0],
                rtol=0.0,
                atol=1e-14,
            )
        )

    def test_full_frame_covariance(self) -> None:
        """Verify covariance under a generic laboratory-frame rotation.

        The test checks both fixed photon polarization and a detector-fixed
        analyzer axis under one common frame change.

        Notes
        -----
        Left-multiply sample and detector orientations and rotate the
        laboratory polarization by the same proper matrix. Both resulting
        sample-frame vectors must remain invariant.
        """
        covariance_rotation: Float64[Array, "..."] = rodrigues_rotation(
            jnp.asarray([0.3, -0.4, 0.8]),
            0.41,
        )
        sample_orientation: Float64[Array, "..."] = sample_azimuth_rotation(
            -0.23
        )
        detector_orientation: Float64[Array, "..."] = detector_rotation(
            0.19, -0.31, "V"
        )
        polarization_lab: Complex128[Array, "3"] = jnp.asarray(
            [0.3 + 0.2j, -0.4 + 0.6j, 0.7 - 0.1j],
            dtype=jnp.complex128,
        )
        detector_axis: Float64[Array, "3"] = jnp.asarray([0.2, -0.7, 0.5])
        detector_axis = detector_axis / jnp.linalg.norm(detector_axis)

        reference_polarization: Complex128[Array, "..."] = (
            lab_polarization_to_sample(
                polarization_lab,
                sample_orientation,
            )
        )
        transformed_polarization: Complex128[Array, "..."] = (
            lab_polarization_to_sample(
                covariance_rotation @ polarization_lab,
                covariance_rotation @ sample_orientation,
            )
        )
        reference_axis: Float64[Array, "..."] = detector_axis_to_sample(
            detector_axis,
            detector_orientation,
            sample_orientation,
        )
        transformed_axis: Float64[Array, "..."] = detector_axis_to_sample(
            detector_axis,
            covariance_rotation @ detector_orientation,
            covariance_rotation @ sample_orientation,
        )
        chex.assert_trees_all_close(
            transformed_polarization,
            reference_polarization,
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            transformed_axis,
            reference_axis,
            rtol=0.0,
            atol=1e-14,
        )

    def test_frame_maps_match_finite_differences(self) -> None:
        """Verify gradients through sample and detector orientations.

        The test checks the sensitivity of both frame maps against numerical
        derivatives.

        Notes
        -----
        Reduce generic fixed-beam and detector-axis outputs with generic
        weights. Compare autodiff with the shared finite-difference check for
        polarization, detector angles, and sample azimuth.
        """
        polarization: Complex128[Array, "3"] = jnp.asarray(
            [0.2 + 0.5j, -0.3 + 0.1j, 0.7 - 0.2j],
            dtype=jnp.complex128,
        )
        detector_axis: Float64[Array, "3"] = jnp.asarray([0.3, -0.5, 0.8])
        tx: Float64[Array, "2"] = jnp.asarray([-0.2, 0.1])
        ty: Float64[Array, "2"] = jnp.asarray([0.0, 0.3])
        weights: Float64[Array, "..."] = jnp.asarray(
            [
                0.7,
                -0.4,
                0.3,
                -0.5,
                0.9,
                -0.6,
                0.2,
                0.8,
                -0.7,
                0.4,
                -0.3,
                0.6,
            ],
            dtype=jnp.float64,
        ).reshape((2, 2, 3))

        def loss(
            arguments: Tuple[
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
                Float64[Array, "..."],
            ],
        ) -> Float64[Array, "..."]:
            """Reduce fixed-beam and detector-axis frame outputs."""
            candidate_polarization: Float64[Array, "..."]
            candidate_tx: Float64[Array, "..."]
            candidate_ty: Float64[Array, "..."]
            candidate_azimuth: Float64[Array, "..."]
            (
                candidate_polarization,
                candidate_tx,
                candidate_ty,
                candidate_azimuth,
            ) = arguments
            candidate_sample_orientation: Float64[Array, "..."] = (
                sample_azimuth_rotation(candidate_azimuth)
            )
            candidate_beam: Complex128[Array, "..."] = (
                lab_polarization_to_sample(
                    candidate_polarization,
                    candidate_sample_orientation,
                )
            )
            candidate_axes: Float64[Array, "..."] = rotate_frame_vectors(
                detector_axis,
                candidate_tx,
                candidate_ty,
                "V",
                candidate_azimuth,
            )
            beam_weights: Complex128[Array, "3"] = jnp.asarray(
                [0.4 - 0.2j, -0.1 + 0.6j, 0.8 + 0.3j]
            )
            result: Float64[Array, "..."] = jnp.real(
                jnp.vdot(beam_weights, candidate_beam)
            ) + jnp.sum(weights * candidate_axes)
            return result

        assert_gradients_match_finite_differences(
            loss, (polarization, tx, ty, jnp.asarray(0.23))
        )
