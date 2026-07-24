"""Validate ARPES polarization and detector-frame functions.

Extended Summary
----------------
Exercise build_polarization_vectors, build_efield, and
dipole_matrix_elements. Verify the polarization basis and each field
mode. Check dipole matrix element shape and sign. Compare the detector frame
with an offline table from the pinned Chinook source.

"""

import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array

import diffpes
from diffpes.simul import (
    build_efield,
    build_polarization_vectors,
    detector_angles_to_kpar,
    detector_rotation,
    dipole_matrix_elements,
    final_state_k_inv_ang,
    photon_wavevector,
    polarization_from_angles,
    polarization_to_spherical,
    rotate_frame_vectors,
    rotate_polarization_grid,
)
from diffpes.types import make_polarization_config
from tests._gradients import complex_step_derivative, gradient_gate


class TestBuildPolarizationVectors(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.build_polarization_vectors`.

    Verifies the geometric properties of the s- and p-polarization basis
    vectors, including mutual orthogonality, unit norm, and correct output
    shape for various incidence angle combinations.

    :see: :func:`~diffpes.simul.build_polarization_vectors`
    """

    def test_orthogonality(self) -> None:
        """Verify that e_s and e_p are mutually orthogonal.

        The test establishes the orthogonality contract for build polarization vectors
        with the concrete values and array shapes described below.

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
        theta: Array
        phi: float
        e_s: Array
        e_p: Array
        dot_product: Array

        theta = jnp.pi / 4.0
        phi = 0.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        dot_product = jnp.dot(e_s, e_p)
        chex.assert_trees_all_close(dot_product, jnp.float64(0.0), atol=1e-10)

    def test_unit_vectors(self) -> None:
        """Verify that both e_s and e_p have unit norm.

        The test establishes the unit vectors contract for build polarization vectors
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
        theta: Array
        phi: Array
        e_s: Array
        e_p: Array

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

        The test establishes the shape contract for build polarization vectors with the
        concrete values and array shapes described below.

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
        e_s: Array
        e_p: Array

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
        phi: Array = jnp.asarray(0.37)
        threshold: Array = jnp.arccos(jnp.asarray(0.99))
        theta: Array = threshold + jnp.asarray([-1e-6, 1e-6])
        e_s: Array
        e_p: Array
        e_s, e_p = jax.vmap(
            lambda value: build_polarization_vectors(value, phi)
        )(theta)
        expected_s: Array = jnp.asarray([jnp.sin(phi), -jnp.cos(phi), 0.0])
        expected_p: Array = jnp.stack(
            (
                -jnp.cos(theta) * jnp.cos(phi),
                -jnp.cos(theta) * jnp.sin(phi),
                jnp.sin(theta),
            ),
            axis=-1,
        )
        derivative: Array = jax.vmap(
            jax.jacfwd(lambda value: build_polarization_vectors(value, phi)[1])
        )(theta)
        expected_derivative: Array = jnp.stack(
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

        Map zero polar angle to positive z. Map a right-angle polar angle
        at zero azimuth to positive x within float64 tolerance.

        Notes
        -----
        Evaluate two scalar angle pairs. Compare both vectors with analytic
        Cartesian directions at ``atol=1e-12``. Check their unit norms.
        """
        normal: Array
        grazing: Array

        normal = photon_wavevector(0.0, 0.0)
        grazing = photon_wavevector(jnp.pi / 2.0, 0.0)
        chex.assert_trees_all_close(
            normal,
            jnp.array([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            grazing,
            jnp.array([1.0, 0.0, 0.0]),
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


class TestBuildEfield(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.build_efield`.

    Verify the electric field for each polarization type. Check the LVP,
    LHP, RCP, and LCP relations.

    :see: :func:`~diffpes.simul.build_efield`
    """

    def test_lvp_equals_s(self) -> None:
        """Verify that LVP (linear vertical polarization) yields the s-polarization vector.

        The test establishes the lvp equals s contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LVP config**:
           Create an LVP configuration at theta=pi/4 and phi=0. Compute
           its electric field vector.

        2. **Build reference s-polarization vector**:
           Independently computes e_s using ``build_polarization_vectors``
           with the same angles.

        3. **Compare real parts**:
           Checks that the real part of the E-field matches e_s.

        **Expected assertions**

        The real part of the LVP E-field equals the s-polarization vector
        within tolerance 1e-10.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LVP",
        )
        efield = build_efield(config)
        e_s, _ = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_s.astype(jnp.float64),
            atol=1e-10,
        )

    def test_lhp_equals_p(self) -> None:
        """Verify that LHP (linear horizontal polarization) yields the p-polarization vector.

        The test establishes the lhp equals p contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LHP config**:
           Create an LHP configuration at theta=pi/4 and phi=0. Compute
           its electric field vector.

        2. **Build reference p-polarization vector**:
           Independently computes e_p using ``build_polarization_vectors``
           with the same angles.

        3. **Compare real parts**:
           Checks that the real part of the E-field matches e_p.

        **Expected assertions**

        The real part of the LHP E-field equals the p-polarization vector
        within tolerance 1e-10.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_p: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LHP",
        )
        efield = build_efield(config)
        _, e_p = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_p.astype(jnp.float64),
            atol=1e-10,
        )

    def test_rcp_lcp_conjugate(self) -> None:
        """Verify that RCP and LCP E-fields share the same real part.

        The test establishes the rcp lcp conjugate contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build RCP and LCP E-fields**:
           Create RCP and LCP configurations at theta=pi/4 and phi=0.
           Compute both E-field vectors.

        2. **Compare real parts**:
           Use the circular-polarization definitions. Compare their common
           real part, e_s/sqrt(2).

        **Expected assertions**

        The real parts of the RCP and LCP E-fields are equal within
        tolerance 1e-10, confirming the conjugate symmetry relationship.
        """
        config_r: diffpes.types.PolarizationConfig
        config_l: diffpes.types.PolarizationConfig
        e_rcp: Array
        e_lcp: Array

        config_r = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="RCP",
        )
        config_l = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LCP",
        )
        e_rcp = build_efield(config_r)
        e_lcp = build_efield(config_l)
        chex.assert_trees_all_close(
            jnp.real(e_rcp),
            jnp.real(e_lcp),
            atol=1e-10,
        )

    def test_lap_linear_combination(self) -> None:
        """Verify that LAP (linear arbitrary) yields cos(angle)*e_s + sin(angle)*e_p.

        The test establishes the lap linear combination contract for build efield with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LAP config**:
           Create an LAP configuration with polarization_angle=0.3.
           Compute its electric field.

        2. **Build e_s and e_p** with the same angles. Form the expected
           combination cos(0.3)*e_s + sin(0.3)*e_p.

        3. **Compare**: E-field matches the expected combination.

        **Expected assertions**

        LAP E-field equals the linear combination of s- and p-vectors
        at the given polarization angle.
        """
        angle: float
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array
        e_p: Array
        e_s_c: Array
        e_p_c: Array
        expected: Array

        angle = 0.3
        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LAP",
            polarization_angle=angle,
        )
        efield = build_efield(config)
        e_s, e_p = build_polarization_vectors(config.theta, config.phi)
        e_s_c = e_s.astype(jnp.complex128)
        e_p_c = e_p.astype(jnp.complex128)
        expected = jnp.cos(angle) * e_s_c + jnp.sin(angle) * e_p_c
        chex.assert_trees_all_close(efield, expected, atol=1e-10)

    def test_unknown_pol_type_fallback_to_s(self) -> None:
        """Verify that unknown polarization type falls back to e_s.

        The test establishes the unknown pol type fallback to s contract for build
        efield with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with an unsupported type** (e.g. "unpolarized"):
           build_efield treats it as the default branch and returns e_s.

        2. **Compare** real part of E-field to e_s from
           build_polarization_vectors.

        **Expected assertions**

        The default/else branch returns the s-polarization vector.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="unpolarized",
        )
        efield = build_efield(config)
        e_s, _ = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_s.astype(jnp.float64),
            atol=1e-10,
        )


class TestDipoleMatrixElements(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.dipole_matrix_elements`.

    Verify the output shape and the zero s-orbital element. Check px
    coupling and non-negative elements for arbitrary polarization.

    :see: :func:`~diffpes.simul.dipole_matrix_elements`
    """

    def test_shape(self) -> None:
        """Verify that the output has shape ``(9,)`` for the 9-orbital basis.

        The test establishes the shape contract for dipole matrix elements with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and checks the output shape.

        **Expected assertions**

        The output array has shape ``(9,)``, one element per orbital in
        the [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2] basis.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_shape(m, (9,))

    def test_s_orbital_zero(self) -> None:
        """Verify that the s-orbital dipole matrix element is always zero.

        The test establishes the s orbital zero contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and inspects the s-orbital
           entry (index 0).

        **Expected assertions**

        The s-orbital direction vector is [0, 0, 0]. Its matrix element
        is zero within tolerance 1e-10 for any electric field.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_trees_all_close(m[0], jnp.float64(0.0), atol=1e-10)

    def test_px_with_x_field(self) -> None:
        """Verify that an x-polarized field produces a positive px matrix element.

        The test establishes the px with x field contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector purely along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and inspects the px-orbital
           entry (index 3), whose direction vector is [1, 0, 0].

        **Expected assertions**

        The px-orbital matrix element (index 3) is strictly positive,
        confirming that the x-polarized field couples to the px-orbital
        via the dipole selection rule.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_scalar_positive(float(m[3]))

    def test_all_nonnegative(self) -> None:
        """Verify that all dipole matrix elements are non-negative for arbitrary polarization.

        The test establishes the all nonnegative contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create arbitrary normalized E-field**:
           Create a complex field with components [0.5, 0.3, 0.8].
           Normalize it to unit length.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` on the normalized field.

        3. **Check non-negativity of all elements**:
           Iterates over all 9 entries and asserts each is >= 0.

        **Expected assertions**

        Every element of the 9-element matrix element vector is non-negative,
        as expected from the squared-modulus definition
        ``|e . d|^2 >= 0``.
        """
        i: int

        efield: Array
        m: Array

        efield = jnp.array([0.5, 0.3, 0.8], dtype=jnp.complex128)
        efield = efield / jnp.linalg.norm(efield)
        m = dipole_matrix_elements(efield)
        for i in range(9):
            assert float(m[i]) >= 0.0


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
        theta: Array = jnp.asarray(0.7)
        phi: Array = jnp.asarray(-0.2)
        e_s: Array
        e_p: Array
        e_s, e_p = build_polarization_vectors(theta, phi)
        circular: Array = polarization_from_angles(theta, phi, "c+")
        linear: Array = polarization_from_angles(
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
        root_two: Array = jnp.sqrt(jnp.asarray(2.0))
        sigma_plus: Array = (
            jnp.asarray(
                [1.0, 1j, 0.0],
                dtype=jnp.complex128,
            )
            / root_two
        )
        sigma_minus: Array = (
            jnp.asarray(
                [1.0, -1j, 0.0],
                dtype=jnp.complex128,
            )
            / root_two
        )
        x_linear: Array = jnp.asarray(
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
        expected_x: Array = jnp.asarray(
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
        polarization: Array = jnp.asarray(
            [0.3 + 0.2j, -0.4 + 0.1j, 0.5 - 0.7j],
            dtype=jnp.complex128,
        )
        tangent: Array = jnp.asarray(
            [-0.2 + 0.8j, 0.6 - 0.3j, 0.1 + 0.4j],
            dtype=jnp.complex128,
        )
        spherical: Array
        spherical_tangent: Array
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
        components: list[complex],
    ) -> None:
        r"""Preserve :math:`\sum_q|\epsilon_q|^2=|\epsilon|^2`.

        The spherical transform must be unitary within ``1e-15`` for every
        generic complex Cartesian polarization generated by Hypothesis.

        Notes
        -----
        Each of the three Cartesian components has independently generated
        real and imaginary parts and a finite, nonzero magnitude.
        """
        polarization: Array = jnp.asarray(
            components,
            dtype=jnp.complex128,
        )
        spherical: Array = polarization_to_spherical(polarization)
        actual_norm_squared: Array = jnp.sum(jnp.abs(spherical) ** 2)
        expected_norm_squared: Array = jnp.sum(jnp.abs(polarization) ** 2)
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

        def holomorphic_channels(values: Array) -> Array:
            """Return real-phased probes of every nonzero transform entry."""
            zero: Array = jnp.zeros((), dtype=values.dtype)
            x_minus: Array = polarization_to_spherical(
                jnp.stack((values[0], zero, zero))
            )[0]
            y_minus: Array = (
                1j
                * polarization_to_spherical(
                    jnp.stack((zero, values[1], zero))
                )[0]
            )
            z_zero: Array = polarization_to_spherical(
                jnp.stack((zero, zero, values[2]))
            )[1]
            x_plus: Array = -polarization_to_spherical(
                jnp.stack((values[3], zero, zero))
            )[2]
            y_plus: Array = (
                1j
                * polarization_to_spherical(
                    jnp.stack((zero, values[4], zero))
                )[2]
            )
            channels: Array = jnp.stack(
                (x_minus, y_minus, z_zero, x_plus, y_plus)
            )
            return channels

        inputs: Array = jnp.asarray([0.3, -0.4, 0.7, -0.2, 0.5])
        tangent: Array = jnp.ones_like(inputs)
        complex_step: Array = complex_step_derivative(
            holomorphic_channels,
            inputs,
            h=1e-20,
        )
        _: Array
        jvp: Array
        _, jvp = jax.jvp(
            holomorphic_channels,
            (inputs.astype(jnp.complex128),),
            (tangent.astype(jnp.complex128),),
        )
        inverse_root_two: Array = 1.0 / jnp.sqrt(2.0)
        expected: Array = jnp.asarray(
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
        tx: Array = jnp.asarray(0.23)
        ty: Array = jnp.asarray(-0.17)
        z_axis: Array = jnp.asarray([0.0, 0.0, 1.0])
        expected_h: Array = jnp.asarray(
            [
                jnp.sin(tx),
                -jnp.cos(tx) * jnp.sin(ty),
                jnp.cos(tx) * jnp.cos(ty),
            ]
        )
        expected_v: Array = jnp.asarray(
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
        angle: Array = jnp.asarray(0.31)
        rotation: Array = detector_rotation(angle, -0.19, "H")
        chex.assert_trees_all_close(
            rotation @ rotation.T,
            jnp.eye(3),
            atol=1e-14,
        )
        chex.assert_trees_all_close(jnp.linalg.det(rotation), 1.0, atol=1e-14)
        derivative: Array = jax.grad(
            lambda value: detector_rotation(value, -0.19, "H")[0, 2]
        )(angle)
        chex.assert_trees_all_close(
            derivative,
            jnp.cos(angle),
            atol=1e-14,
        )

    def test_matches_pinned_chinook_artifact(self) -> None:
        """Match the pinned Chinook direction and polarization table.

        The artifact records both declared coordinate remaps. It contains the
        full rotation, direction, and three complex polarization states.

        Notes
        -----
        Load gate 03.G4 without a Chinook import. Compare all points on the
        two 5 by 5 angle grids at the recorded relative tolerance.
        """
        tests_root: Path = Path(__file__).resolve().parents[2]
        artifact_path: Path = (
            tests_root / "data" / "kspace" / "tilt_polarization_reference.json"
        )
        reference: dict[str, Any] = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
        expected_mapping: dict[str, dict[str, str]] = {
            "H": {
                "active_rotation": "Rx(diffpes_ty) @ Ry(diffpes_tx)",
                "gen_all_pol": (
                    "chinook_theta=-diffpes_tx, chinook_phi=-diffpes_ty"
                ),
                "tilt_k_mesh": (
                    "chinook_Tx=-diffpes_tx, chinook_Ty=diffpes_ty"
                ),
            },
            "V": {
                "active_rotation": "Rx(diffpes_tx) @ Ry(diffpes_ty)",
                "gen_all_pol": (
                    "chinook_theta=-diffpes_ty, chinook_phi=-diffpes_tx"
                ),
                "tilt_k_mesh": (
                    "chinook_Tx=-diffpes_ty, chinook_Ty=diffpes_tx"
                ),
            },
        }
        self.assertEqual(reference["gate"], "03.G4")
        self.assertEqual(reference["mapping"], expected_mapping)
        tolerance: float = float(reference["rtol"])
        polarization_records: dict[str, Any] = reference["polarization_inputs"]
        slit: str
        for slit in ("H", "V"):
            records: list[dict[str, Any]] = [
                record
                for record in reference["records"]
                if record["slit"] == slit
            ]
            axis_size: int = int(len(records) ** 0.5)
            tx: Array = jnp.asarray(
                [
                    records[index * axis_size]["tx_rad"]
                    for index in range(axis_size)
                ]
            )
            ty: Array = jnp.asarray(
                [records[index]["ty_rad"] for index in range(axis_size)]
            )
            expected_rotations: Array = jnp.asarray(
                [record["rotation_matrix"] for record in records]
            ).reshape(axis_size, axis_size, 3, 3)
            actual_rotations: Array = jax.vmap(
                lambda tx_value: jax.vmap(
                    lambda ty_value: detector_rotation(
                        tx_value, ty_value, slit
                    )
                )(ty)
            )(tx)
            chex.assert_trees_all_close(
                actual_rotations,
                expected_rotations,
                rtol=tolerance,
                atol=1e-12,
            )
            expected_directions: Array = jnp.asarray(
                [record["detector_direction"] for record in records]
            ).reshape(axis_size, axis_size, 3)
            energy: Array = jnp.asarray(35.0)
            momentum: Array = final_state_k_inv_ang(energy)
            actual_k_parallel: Array = detector_angles_to_kpar(
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
            polarization_name: str
            for polarization_name in ("s", "p", "c_plus"):
                input_parts: dict[str, list[float]] = polarization_records[
                    polarization_name
                ]
                polarization: Array = jnp.asarray(input_parts["real"]) + (
                    1j * jnp.asarray(input_parts["imag"])
                )
                expected_parts: list[dict[str, list[float]]] = [
                    record["rotated_polarizations"][polarization_name]
                    for record in records
                ]
                expected_polarization: Array = (
                    jnp.asarray([part["real"] for part in expected_parts])
                    + 1j
                    * jnp.asarray([part["imag"] for part in expected_parts])
                ).reshape(axis_size, axis_size, 3)
                actual_polarization: Array = rotate_polarization_grid(
                    polarization,
                    tx,
                    ty,
                    slit,
                )
                chex.assert_trees_all_close(
                    actual_polarization,
                    expected_polarization,
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

    The test checks the detector-grid shape, vector norms, and mapped frame
    values for a real vector.

    :see: :func:`~diffpes.simul.rotate_frame_vectors`
    """

    def test_maps_real_vector_over_grid(self) -> None:
        """Verify mapped real-vector values and shape.

        The test rotates one normalized vector over two angle axes and checks
        each result against direct detector-frame multiplication.

        Notes
        -----
        Use a 2 by 3 horizontal angle grid. Check its shape, norms, and one
        direct matrix product at 1e-14.
        """
        vector: Array = jnp.asarray([0.0, 0.0, 1.0])
        tx: Array = jnp.asarray([-0.2, 0.1])
        ty: Array = jnp.asarray([-0.1, 0.0, 0.3])
        rotated: Array = rotate_frame_vectors(vector, tx, ty, "H")
        chex.assert_shape(rotated, (2, 3, 3))
        chex.assert_trees_all_close(
            jnp.linalg.norm(rotated, axis=-1),
            jnp.ones((2, 3)),
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            rotated[1, 2],
            detector_rotation(tx[1], ty[2], "H") @ vector,
            atol=1e-14,
        )


class TestRotatePolarizationGrid(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.rotate_polarization_grid`.

    The test checks complex phase preservation, detector-grid shape, and
    differentiation through the mapped detector angles.

    :see: :func:`~diffpes.simul.rotate_polarization_grid`
    """

    def test_maps_complex_polarization_and_gradients(self) -> None:
        """Verify complex grid values and angle gradients.

        The test rotates a generic complex vector and compares one cell with
        direct multiplication. It also checks a nonzero mapped derivative.

        Notes
        -----
        Use a 2 by 2 vertical angle grid. Compare one cell and differentiate
        one real component with respect to the first angle.
        """
        polarization: Array = jnp.asarray(
            [0.2 + 0.5j, -0.3 + 0.1j, 0.7 - 0.2j],
            dtype=jnp.complex128,
        )
        tx: Array = jnp.asarray([-0.2, 0.1])
        ty: Array = jnp.asarray([0.0, 0.3])
        rotated: Array = rotate_polarization_grid(
            polarization,
            tx,
            ty,
            "V",
        )
        chex.assert_shape(rotated, (2, 2, 3))
        chex.assert_trees_all_close(
            rotated[0, 1],
            detector_rotation(tx[0], ty[1], "V") @ polarization,
            atol=1e-14,
        )
        weights: Array = jnp.asarray(
            [
                0.7 - 0.2j,
                -0.4 + 0.6j,
                0.3 + 0.8j,
                -0.5 - 0.1j,
                0.9 + 0.4j,
                -0.6 + 0.7j,
                0.2 - 0.9j,
                0.8 + 0.3j,
                -0.7 + 0.5j,
                0.4 + 0.2j,
                -0.3 - 0.8j,
                0.6 - 0.4j,
            ],
            dtype=jnp.complex128,
        ).reshape((2, 2, 3))

        def loss(arguments: tuple[Array, Array, Array]) -> Array:
            """Reduce a rotated grid with generic complex weights."""
            candidate_polarization: Array
            candidate_tx: Array
            candidate_ty: Array
            candidate_polarization, candidate_tx, candidate_ty = arguments
            candidate_grid: Array = rotate_polarization_grid(
                candidate_polarization,
                candidate_tx,
                candidate_ty,
                "V",
            )
            result: Array = jnp.real(jnp.vdot(weights, candidate_grid))
            return result

        gradient_gate(loss, (polarization, tx, ty))
