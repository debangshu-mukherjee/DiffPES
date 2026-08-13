"""Check pseudo-wave point-detector amplitude kernels.

Use independent inputs to define the covered behavior.
"""

import jax.numpy as jnp
from beartype.typing import Any
from jax.test_util import check_grads
from jaxtyping import Array, Complex128, Float64, Int32

from diffpes.simul import (
    plane_wave_mask,
    plane_wave_pseudo_amplitude,
    surface_window_transform,
)
from diffpes.types import (
    PlaneWaveBatch,
    make_crystal_geometry,
    make_plane_wave_batch,
)

_TOLERANCE = 1.0e-12


def _batch(
    coefficients: Complex128[Array, "n_state n_pw n_spinor"],
    g_vectors: Int32[Array, "n_state n_pw 3"],
) -> PlaneWaveBatch:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    geometry: Any
    batch: Any
    geometry = make_crystal_geometry(
        jnp.eye(3) * (2.0 * jnp.pi),
        jnp.zeros((1, 3)),
        ("X",),
    )
    batch = make_plane_wave_batch(
        coefficients,
        g_vectors,
        jnp.asarray([coefficients.shape[1]], dtype=jnp.int32),
        jnp.zeros((1, 3)),
        jnp.ones((1,)),
        jnp.zeros((1,)),
        jnp.ones((1,)),
        jnp.zeros((1, 3), dtype=jnp.int32),
        geometry,
        jnp.asarray(0.0),
        spin_mode="scalar",
        source_ref="fixture",
        gauge_ref="velocity",
    )
    return batch


class TestPlaneWaveMask:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.plane_wave_mask``
    """

    def test_marks_only_stored_coefficients(self) -> None:
        """Mark the declared prefix and zero every padded coefficient lane.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        batch: Any
        batch = _batch(
            jnp.ones((1, 2, 1), dtype=jnp.complex128),
            jnp.zeros((1, 2, 3), dtype=jnp.int32),
        )
        assert jnp.array_equal(plane_wave_mask(batch), jnp.ones((1, 2)))


class TestSurfaceWindowTransform:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.surface_window_transform``
    """

    def test_matches_gaussian_half_space_transform(self) -> None:
        """Match the hand-derived lateral Gaussian and normal pole factors.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        actual: Any
        expected: Any
        delta: Float64[Array, "1 3"] = jnp.asarray([[0.2, -0.3, 0.4]])
        coherence: Float64[Array, ""] = jnp.asarray(2.0)
        path: Float64[Array, ""] = jnp.asarray(5.0)
        actual = surface_window_transform(delta, coherence, path)
        expected = jnp.exp(-0.5 * coherence**2 * (0.2**2 + 0.3**2)) / (
            1.0 / path - 0.4j
        )
        assert jnp.allclose(actual[0], expected, rtol=_TOLERANCE)


class TestPlaneWavePseudoAmplitude:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.plane_wave_pseudo_amplitude``
    """

    def test_matches_single_g_closed_form(self) -> None:
        """Match one coefficient times its dipole and window factors.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Evaluate the scalar expression directly for one reciprocal vector.
        """
        coefficient: complex = 0.7 - 0.25j
        batch: PlaneWaveBatch = _batch(
            jnp.asarray([[[coefficient]]]),
            jnp.asarray([[[1, 0, 0]]], dtype=jnp.int32),
        )
        final_k: Float64[Array, "1 3"] = jnp.asarray([[0.3, -0.2, 0.4]])
        polarization: Complex128[Array, " 3"] = jnp.asarray(
            [0.6 + 0.1j, -0.2j, 0.3]
        )
        coherence: Float64[Array, ""] = jnp.asarray(1.7)
        path: Float64[Array, ""] = jnp.asarray(3.2)
        actual: Any = plane_wave_pseudo_amplitude(
            batch, final_k, polarization, coherence, path
        )
        delta: Float64[Array, " 3"] = final_k[0] - jnp.asarray([1.0, 0.0, 0.0])
        window: Any = jnp.exp(
            -0.5 * coherence**2 * jnp.sum(delta[:2] ** 2)
        ) / (1.0 / path - 1.0j * delta[2])
        expected: Any = coefficient * polarization[0] * window
        assert jnp.allclose(actual[0, 0, 0], expected, rtol=_TOLERANCE)

    def test_matches_single_and_two_g_interference(self) -> None:
        """Match a hand sum of two interfering reciprocal components.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        coefficients: Any
        vectors: Any
        batch: Any
        final_k: Any
        polarization: Any
        actual: Any
        wavevectors: Any
        delta: Any
        window: Any
        expected: Any
        coefficients = jnp.asarray([[[1.0 + 2.0j], [0.5 - 0.25j]]])
        vectors = jnp.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=jnp.int32)
        batch = _batch(coefficients, vectors)
        final_k = jnp.asarray([[0.1, 0.2, 0.3]])
        polarization = jnp.asarray([0.6 + 0.1j, -0.2j, 0.4 + 0.0j])
        actual = plane_wave_pseudo_amplitude(
            batch, final_k, polarization, jnp.asarray(3.0), jnp.asarray(4.0)
        )
        wavevectors = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        delta = final_k[0, None, :] - wavevectors
        window = jnp.exp(-4.5 * jnp.sum(delta[:, :2] ** 2, axis=1)) / (
            0.25 - 1.0j * delta[:, 2]
        )
        expected = jnp.sum(
            coefficients[0, :, 0] * (wavevectors @ polarization) * window
        )
        assert jnp.allclose(actual[0, 0, 0], expected, rtol=_TOLERANCE)

    def test_polarization_node_is_exact(self) -> None:
        """Check a zero single-G component when polarization is perpendicular.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        batch: Any
        actual: Any
        batch = _batch(
            jnp.ones((1, 1, 1), dtype=jnp.complex128),
            jnp.asarray([[[1, 0, 0]]], dtype=jnp.int32),
        )
        actual = plane_wave_pseudo_amplitude(
            batch,
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.asarray(10.0),
            jnp.asarray(5.0),
        )
        assert actual[0, 0, 0] == 0.0

    def test_large_lateral_window_suppresses_mismatch(self) -> None:
        """Check lateral momentum conservation as coherence grows.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        transformed: Any
        delta: Float64[Array, "2 3"] = jnp.asarray(
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]]
        )
        transformed = surface_window_transform(
            delta, jnp.asarray(100.0), jnp.asarray(3.0)
        )
        assert jnp.abs(transformed[1]) < 1.0e-20 * jnp.abs(transformed[0])

    def test_gradients_cover_detector_window_parameters(self) -> None:
        """Match both gradient modes for four differentiable inputs.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        batch: Any
        arguments: Any
        batch = _batch(
            jnp.asarray([[[0.7 + 0.2j]]]),
            jnp.asarray([[[1, 1, 1]]], dtype=jnp.int32),
        )

        def observable(
            polarization: Complex128[Array, " 3"],
            final_k: Float64[Array, "1 3"],
            path: Float64[Array, ""],
            coherence: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """Check the private helper behavior."""
            amplitude: Any
            amplitude = plane_wave_pseudo_amplitude(
                batch,
                jnp.asarray(final_k),
                jnp.asarray(polarization),
                jnp.asarray(coherence),
                jnp.asarray(path),
            )
            result: Float64[Array, ""] = jnp.real(
                jnp.sum(amplitude.conj() * amplitude)
            )
            return result

        arguments = (
            jnp.asarray([0.3 + 0.1j, -0.2 + 0.4j, 0.5 - 0.1j]),
            jnp.asarray([[0.2, 0.3, 0.4]]),
            jnp.asarray(2.7),
            jnp.asarray(1.8),
        )
        check_grads(observable, arguments, order=1, modes=("fwd", "rev"))
