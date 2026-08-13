"""Validate the ARPES mathematical utilities.

Extended Summary
----------------
The tests exercise the Faddeeva function, z-score normalization, and the
complex packing boundary. They cover JIT, vectorization, precision, round
trips, and gradients.
"""

import hashlib
import json
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from beartype.typing import Any, Callable, Dict, List, Tuple, Union
from jax.test_util import check_grads
from jaxtyping import Array, Complex128, Float64, Shaped, jaxtyped
from numpy.typing import NDArray
from scipy.special import wofz

from diffpes.utils import (
    faddeeva,
    pack_complex,
    unpack_complex,
    zscore_normalize,
)


@jaxtyped(typechecker=beartype)
def _faddeeva_reference() -> Dict[str, Shaped[NDArray, "..."]]:
    """PRIVATE: Load the frozen arbitrary-precision Faddeeva reference.

    Returns
    -------
    result : Dict[str, Shaped[NDArray, "..."]]
        Every array member of the frozen 100-digit mpmath reference
        archive, keyed by its stored name.

    Notes
    -----
    Opens the committed ``.npz`` archive under ``_reference_data``
    with ``allow_pickle=False`` and copies each member into a plain
    dictionary before the file closes.
    """
    path: Path = (
        Path(__file__).parents[1]
        / "_reference_data"
        / "faddeeva_mpmath_100digit_reference.npz"
    )
    archive: np.lib.npyio.NpzFile
    with np.load(path, allow_pickle=False) as archive:
        result: Dict[str, Shaped[NDArray, "..."]] = {
            name: archive[name] for name in archive.files
        }
    return result


@jaxtyped(typechecker=beartype)
def _packed_norm_squared(
    packed: Float64[Array, " ... 2"],
) -> Float64[Array, ""]:
    """PRIVATE: Compute squared complex magnitude from packed real coordinates.

    Parameters
    ----------
    packed : Float64[Array, " ... 2"]
        Packed real and imaginary coordinates.

    Returns
    -------
    loss : Float64[Array, ""]
        Sum of the squared complex magnitudes.

    Notes
    -----
    The helper unpacks the coordinates and sums ``abs(z)**2`` for the
    gradient test.
    """
    unpacked: Complex128[Array, " ..."] = unpack_complex(packed)
    loss: Float64[Array, ""] = jnp.sum(jnp.abs(unpacked) ** 2)
    return loss


@jaxtyped(typechecker=beartype)
def _complex_abs_squared(z: Complex128[Array, ""]) -> Float64[Array, ""]:
    """PRIVATE: Compute squared magnitude for the Wirtinger convention test.

    Parameters
    ----------
    z : Complex128[Array, ""]
        Complex128 scalar under test.

    Returns
    -------
    loss : Float64[Array, ""]
        Squared magnitude of ``z``.

    Notes
    -----
    The helper supplies a real scalar loss to ``jax.grad``.
    """
    loss: Float64[Array, ""] = jnp.abs(z) ** 2
    return loss


class TestFaddeeva(chex.TestCase):
    """Validate :func:`diffpes.utils.math.faddeeva`.

    The tests cover both coordinate axes and the analytic value at the origin.
    Each test uses compiled and uncompiled execution.

    :see: :func:`~diffpes.utils.faddeeva`
    """

    def test_reference_provenance_and_scipy_crosscheck(self) -> None:
        """Bind the frozen artifact to its generator and SciPy comparator.

        The test verifies immutable digests, the preregistered grid, and an
        independent double-precision cross-check.

        Notes
        -----
        It recomputes both SHA-256 values and compares every rounded value with
        ``scipy.special.wofz`` at a mixed double-precision tolerance.
        """
        root: Path = Path(__file__).parents[3]
        data_directory: Path = (
            root / "tests" / "test_diffpes" / "_reference_data"
        )
        manifest_path: Path = data_directory / "faddeeva_mpmath_manifest.json"
        manifest: Dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        archive_path: Path = data_directory / manifest["archive"]
        generator_path: Path = root / manifest["generator"]
        assert manifest["schema"] == "diffpes.faddeeva-mpmath-reference.v1"
        assert manifest["reference_engine"]["decimal_digits"] == 100
        assert (
            hashlib.sha256(archive_path.read_bytes()).hexdigest()
            == manifest["archive_sha256"]
        )
        assert (
            hashlib.sha256(generator_path.read_bytes()).hexdigest()
            == manifest["generator_sha256"]
        )
        reference: Dict[str, Shaped[NDArray, "..."]] = _faddeeva_reference()
        scipy_values: Complex128[NDArray, " n"] = wofz(reference["points"])
        np.testing.assert_allclose(
            scipy_values,
            reference["values"],
            rtol=2.0e-14,
            atol=2.0e-15,
        )

    def test_primal_full_envelope(self) -> None:
        """Match both Faddeeva components over the complete public envelope.

        The test includes logarithmic radii through ``1e8``, both axes, all
        upper-half-plane angles, and the known approximation failure cases.

        Notes
        -----
        It applies the frozen componentwise mixed bound independently to the
        real and imaginary output rows.
        """
        reference: Dict[str, Shaped[NDArray, "..."]] = _faddeeva_reference()
        points: Complex128[Array, " n"] = jnp.asarray(reference["points"])
        expected: Complex128[NDArray, " n"] = reference["values"]
        actual: Complex128[NDArray, " n"] = np.asarray(
            jax.jit(faddeeva)(points)
        )
        real_bound: Float64[NDArray, " n"] = 2.0e-15 + 2.0e-12 * np.abs(
            expected.real
        )
        imag_bound: Float64[NDArray, " n"] = 2.0e-15 + 2.0e-12 * np.abs(
            expected.imag
        )
        np.testing.assert_array_less(
            np.abs(actual.real - expected.real),
            real_bound,
        )
        np.testing.assert_array_less(
            np.abs(actual.imag - expected.imag),
            imag_bound,
        )

    def test_reference_derivative_full_envelope(self) -> None:
        """Match native JVPs with arbitrary-precision ODE derivatives.

        The test differentiates every frozen point in the three registered
        complex directions over the complete upper-half-plane envelope.

        Notes
        -----
        It forms the cancellation-sensitive ODE truth in mpmath before the
        artifact rounds it to complex128.
        """
        reference: Dict[str, Shaped[NDArray, "..."]] = _faddeeva_reference()
        points: Complex128[Array, " n"] = jnp.asarray(reference["points"])
        derivatives: Complex128[NDArray, " n"] = reference["derivatives"]
        direction: complex
        for direction in reference["directions"]:
            tangents: Complex128[Array, " n"] = jnp.full_like(
                points, direction
            )
            actual: Complex128[NDArray, " n"] = np.asarray(
                jax.jit(
                    lambda arguments, vectors: jax.jvp(
                        faddeeva,
                        (arguments,),
                        (vectors,),
                    )[1]
                )(points, tangents)
            )
            expected: Complex128[NDArray, " n"] = derivatives * direction
            bound: Float64[NDArray, " n"] = 2.0e-14 / (
                1.0 + np.abs(reference["points"])
            ) ** 2 + 2.0e-11 * np.abs(expected)
            np.testing.assert_array_less(np.abs(actual - expected), bound)

    def test_directional_jvps_match_finite_differences(self) -> None:
        """Compare native JVPs with multistep central finite differences.

        The test probes small, intermediate, and asymptotic interior points in
        generic real and complex directions.

        Notes
        -----
        It uses three relative steps. The two smallest step sizes must agree
        with the analytic rational derivative inside the mixed error bound.
        """
        points: Tuple[complex, ...] = (
            0.3 + 0.2j,
            3.0 + 1.0j,
            25.0 + 0.5j,
            1.0e4 + 100.0j,
        )
        directions: Tuple[complex, ...] = (1.0 + 0.0j, 0.6 + 0.8j)
        point: complex
        direction: complex
        exponent: int
        for point in points:
            argument: Complex128[Array, ""] = jnp.asarray(point)
            scale: float = max(1.0, abs(point))
            for direction in directions:
                tangent: Complex128[Array, ""] = jnp.asarray(direction)
                exact: complex = complex(
                    jax.jvp(faddeeva, (argument,), (tangent,))[1]
                )
                estimates: List[complex] = []
                for exponent in (12, 14, 16):
                    step: float = scale * 2.0**-exponent
                    estimate: complex = complex(
                        (
                            faddeeva(argument + step * tangent)
                            - faddeeva(argument - step * tangent)
                        )
                        / (2.0 * step)
                    )
                    estimates.append(estimate)
                tolerance: float = 1.0e-9 + 1.0e-6 * abs(exact)
                assert abs(estimates[-1] - exact) < tolerance
                assert abs(estimates[-2] - exact) < tolerance

    def test_forward_and_reverse_modes(self) -> None:
        """Check forward and reverse derivatives at generic complex points.

        The test projects each complex result to an asymmetric real loss so
        both output components participate.

        Notes
        -----
        It invokes JAX's independent randomized directional checker in both AD
        modes away from any physical-domain boundary.
        """
        points: Tuple[complex, ...] = (
            0.2 + 0.4j,
            2.3 + 0.7j,
            17.0 + 3.0j,
        )
        point: complex
        for point in points:

            @jaxtyped(typechecker=beartype)
            def _loss(
                argument: Union[
                    Complex128[Array, ""],
                    Complex128[NDArray, ""],
                ],
            ) -> Float64[Array, ""]:
                """PRIVATE: Check the private helper behavior.

                Parameters
                ----------
                argument : Union[Complex128[Array, ""],Complex128[NDArray, ""]]
                    Complex upper-half-plane argument from JAX or NumPy.

                Returns
                -------
                result : Float64[Array, ""]
                    Asymmetric real projection of the Faddeeva value.

                Notes
                -----
                Weights the imaginary component by 0.37 so both complex
                components contribute to the derivative check.
                """
                normalized: Complex128[Array, ""] = jnp.asarray(
                    argument,
                    dtype=jnp.complex128,
                )
                value: Complex128[Array, ""] = faddeeva(normalized)
                result: Float64[Array, ""] = jnp.real(value) + 0.37 * jnp.imag(
                    value
                )
                return result

            check_grads(
                _loss,
                (jnp.asarray(point),),
                order=1,
                modes=("fwd", "rev"),
                atol=1.0e-9,
                rtol=1.0e-6,
            )

    def test_domain_rejections(self) -> None:
        """Reject nonfinite and out-of-envelope complex arguments.

        The test covers eager and compiled validation for every public domain
        boundary.

        Notes
        -----
        It requires the same physical-domain message from direct and JIT calls.
        """
        arguments: Tuple[complex, ...] = (
            np.nan + 0.0j,
            np.inf + 0.0j,
            1.0 - 1.0e-12j,
            1.00000001e8 + 0.0j,
        )
        argument: complex
        for argument in arguments:
            with pytest.raises(
                (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
                match="finite",
            ):
                faddeeva(jnp.asarray(argument))
            with pytest.raises(
                (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
                match="finite",
            ):
                jax.jit(faddeeva)(jnp.asarray(argument))

    def test_real_axis_identity_and_vmap(self) -> None:
        """Verify the Gaussian real-axis identity under vectorization.

        The test covers signed real arguments and compares mapped evaluation
        with direct array evaluation.

        Notes
        -----
        It checks ``Re(w(x))=exp(-x**2)`` and exact agreement between the two
        public JAX transformation patterns.
        """
        coordinates: Float64[Array, " n"] = jnp.linspace(-10.0, 10.0, 201)
        points: Complex128[Array, " n"] = coordinates.astype(jnp.complex128)
        direct: Complex128[Array, " n"] = faddeeva(points)
        mapped: Complex128[Array, " n"] = jax.vmap(faddeeva)(points)
        chex.assert_trees_all_equal(mapped, direct)
        np.testing.assert_allclose(
            np.asarray(jnp.real(direct)),
            np.asarray(jnp.exp(-(coordinates**2))),
            rtol=2.0e-12,
            atol=2.0e-15,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_real_axis(self) -> None:
        """Verify ``faddeeva`` returns finite values on the real axis.

        The function preserves the input shape and produces finite real
        components for the specified interval.

        Notes
        -----
        The test uses 100 complex points across ``[-3, 3]``. It checks shape
        ``(100,)`` and the finite real components under both JAX variants.
        """
        x: Float64[Array, " 100"]
        z: Complex128[Array, " 100"]
        var_fn: Callable[..., Any]
        w: Complex128[Array, " 100"]

        x = jnp.linspace(-3.0, 3.0, 100)
        z = x + 0j
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (100,))
        chex.assert_tree_all_finite(jnp.real(w))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero(self) -> None:
        """Verify ``faddeeva(0)`` returns approximately 1.0.

        The Faddeeva function satisfies ``w(0) = erfc(0) = 1``.

        Notes
        -----
        The test supplies the complex scalar zero. It compares the real
        component with 1.0 at absolute tolerance 0.05 under both variants.
        """
        z: Complex128[Array, ""]
        var_fn: Callable[..., Any]
        w: Complex128[Array, ""]

        z = jnp.array(0.0 + 0j)
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_trees_all_close(jnp.real(w), jnp.float64(1.0), atol=0.05)

    @chex.variants(with_jit=True, without_jit=True)
    def test_imaginary_axis(self) -> None:
        """Verify ``faddeeva`` returns finite values on the imaginary axis.

        The function preserves the input shape and produces finite real
        components for three imaginary inputs.

        Notes
        -----
        The test uses ``i*[0.5, 1.0, 2.0]``. It checks shape ``(3,)`` and
        the finite real components under both JAX variants.
        """
        y: Float64[Array, " 3"]
        z: Complex128[Array, " 3"]
        var_fn: Callable[..., Any]
        w: Complex128[Array, " 3"]

        y = jnp.array([0.5, 1.0, 2.0])
        z = 1j * y
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (3,))
        chex.assert_tree_all_finite(jnp.real(w))


class TestZscoreNormalize(chex.TestCase):
    """Validate :func:`diffpes.utils.math.zscore_normalize`.

    The tests cover standard data, constant data, and two-dimensional data.
    They verify the global normalization and the zero-variance guard.

    :see: :func:`~diffpes.utils.zscore_normalize`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalized_stats(self) -> None:
        """Verify ``zscore_normalize`` produces zero mean and unit deviation.

        Standard z-score normalization produces a mean of zero and a standard
        deviation of one.

        Notes
        -----
        The test normalizes ``[1, 2, 3, 4, 5]``. It compares both statistics
        with their analytic values at an absolute tolerance of 1e-10.
        """
        data: Float64[Array, " 5"]
        var_fn: Callable[..., Any]
        result: Float64[Array, " 5"]

        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.std(result), jnp.float64(1.0), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_constant_input(self) -> None:
        """Verify ``zscore_normalize`` returns zeros for a constant array.

        The zero-variance guard produces a finite zero array.

        Notes
        -----
        The test normalizes an array of ten ones under both JAX variants.
        It compares the result with ten zeros at absolute tolerance 1e-10.
        """
        data: Float64[Array, " 10"]
        var_fn: Callable[..., Any]
        result: Float64[Array, " 10"]

        data = jnp.ones(10, dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            result, jnp.zeros(10, dtype=jnp.float64), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_input(self) -> None:
        """Verify ``zscore_normalize`` handles arrays with two dimensions.

        The function preserves the two-dimensional shape and uses global
        statistics across both axes.

        Notes
        -----
        The test normalizes ``arange(12)`` with shape ``(3, 4)``. It checks the
        shape and compares the global mean with zero at a tolerance of 1e-10.
        """
        data: Float64[Array, "3 4"]
        var_fn: Callable[..., Any]
        result: Float64[Array, "3 4"]

        data = jnp.arange(12.0).reshape(3, 4)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_shape(result, (3, 4))
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )


class TestPackComplex(chex.TestCase):
    """Validate :func:`~diffpes.utils.math.pack_complex`.

    The tests cover exact round trips, dtype preservation, JIT, and
    vectorization. Asymmetric values reveal an incorrect coordinate order.

    :see: :func:`~diffpes.utils.pack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve generic complex128 values exactly through a JIT round trip.

        The input uses unequal real and imaginary components. These values
        reveal a component swap or a real-symmetric implementation.

        Notes
        -----
        The test packs and unpacks a ``(2, 2)`` array with ``jax.jit``. It
        checks the packed shape, both dtypes, and exact value equality.
        """
        complex_values: Complex128[Array, "2 2"] = jnp.array(
            [[1.0 + 2.0j, -3.0 + 0.5j], [7.0 - 4.0j, 0.25 + 9.0j]],
            dtype=jnp.complex128,
        )
        packed: Float64[Array, "2 2 2"] = jax.jit(pack_complex)(complex_values)
        round_tripped: Complex128[Array, "2 2"] = jax.jit(unpack_complex)(
            packed
        )

        chex.assert_shape(packed, (2, 2, 2))
        chex.assert_equal(packed.dtype, jnp.dtype("float64"))
        chex.assert_trees_all_equal(round_tripped, complex_values)

    def test_vmap(self) -> None:
        """Vectorize packing independently over a leading parameter batch.

        The result retains the batch and parameter axes. It appends only the
        two-component packing axis.

        Notes
        -----
        The test applies ``jax.vmap`` to three complex parameter vectors. It
        compares the result with direct packing and checks the exact round
        trip.
        """
        complex_values: Complex128[Array, "3 2"] = jnp.array(
            [
                [1.0 + 4.0j, 2.0 - 3.0j],
                [-5.0 + 0.25j, 7.0 + 8.0j],
                [9.0 - 2.0j, -1.5 + 6.0j],
            ],
            dtype=jnp.complex128,
        )
        vmapped: Float64[Array, "3 2 2"] = jax.vmap(pack_complex)(
            complex_values
        )
        direct: Float64[Array, "3 2 2"] = pack_complex(complex_values)
        unpacked: Complex128[Array, "3 2"] = jax.vmap(unpack_complex)(vmapped)

        chex.assert_shape(vmapped, (3, 2, 2))
        chex.assert_trees_all_equal(vmapped, direct)
        chex.assert_trees_all_equal(unpacked, complex_values)


class TestUnpackComplex(chex.TestCase):
    """Validate :func:`~diffpes.utils.math.unpack_complex`.

    The tests cover exact round trips and dtype preservation. They also compare
    complex magnitude gradients with gradients in real coordinates.

    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve stacked float64 values through a JIT round trip.

        The final axis contains asymmetric real and imaginary coordinates.
        These values reveal an incorrect coordinate order.

        Notes
        -----
        The test unpacks and packs a ``(2, 3, 2)`` array with ``jax.jit``.
        It checks the shape, both dtypes, and exact coordinate equality.
        """
        packed_values: Float64[Array, "2 3 2"] = jnp.array(
            [
                [[1.0, 2.0], [-3.0, 0.5], [7.0, -4.0]],
                [[0.25, 9.0], [6.0, -8.0], [-2.5, 11.0]],
            ],
            dtype=jnp.float64,
        )
        unpacked: Complex128[Array, "2 3"] = jax.jit(unpack_complex)(
            packed_values
        )
        round_tripped: Float64[Array, "2 3 2"] = jax.jit(pack_complex)(
            unpacked
        )

        chex.assert_shape(unpacked, (2, 3))
        chex.assert_equal(unpacked.dtype, jnp.dtype("complex128"))
        chex.assert_trees_all_equal(round_tripped, packed_values)

    def test_gradient_equivalence(self) -> None:
        """Match complex-magnitude gradients to packed real coordinates.

        For ``p = stack([x, y])``, the real gradient equals
        ``stack([2x, 2y])``.

        Notes
        -----
        The test differentiates a compiled loss on generic float64 coordinates.
        It compares the gradient with twice the input by exact equality.
        """
        packed_values: Float64[Array, "3 2"] = jnp.array(
            [[1.0, 2.0], [-3.0, 0.5], [7.0, -4.0]], dtype=jnp.float64
        )
        gradient: Float64[Array, "3 2"] = jax.jit(
            jax.grad(_packed_norm_squared)
        )(packed_values)
        expected: Float64[Array, "3 2"] = 2.0 * packed_values

        chex.assert_trees_all_equal(gradient, expected)


class TestComplexAutodiffConvention(chex.TestCase):
    """Pin JAX's complex-gradient convention at the packing boundary.

    The tests fix the conjugated Wirtinger convention at the boundary between
    complex physics values and real optimizer coordinates.

    :see: :func:`~diffpes.utils.pack_complex`
    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_wirtinger_convention(self) -> None:
        """Pin ``grad(abs(z)**2)`` at ``1+1j`` to exactly ``2-2j``.

        The convention determines the relation between complex gradients and
        gradients of stacked real coordinates.

        Notes
        -----
        The test applies reverse-mode autodiff at ``1+1j``. It checks exact
        equality with ``2-2j`` to detect a change in the JAX convention.
        """
        z: Complex128[Array, ""] = jnp.asarray(
            1.0 + 1.0j, dtype=jnp.complex128
        )
        gradient: Complex128[Array, ""] = jax.grad(_complex_abs_squared)(z)
        expected: Complex128[Array, ""] = jnp.asarray(
            2.0 - 2.0j, dtype=jnp.complex128
        )

        chex.assert_trees_all_equal(gradient, expected)
