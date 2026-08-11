"""Validate the shared gradient harness against external analytic truths.

Extended Summary
----------------
Exercises scale-aware finite differences, JAX's complex-to-real convention,
complex-step differentiation, and planted wrong and zero gradients. These
self-tests establish gradient checks before physics code uses this harness.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Tuple
from jax import test_util
from jaxtyping import Array, Complex128, Float64

from tests._gradients import (
    RTOL_LADDER,
    assert_grad_matches_fd,
    assert_gradients_match_finite_differences,
    assert_nonzero_grad,
    central_fd_grad,
    complex_step_derivative,
    fd_step,
    random_generic_complex,
)
from tests._types import GradRegime


@jax.custom_jvp
def _wrong_sine(x: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """PRIVATE: Return sine with a deliberately incorrect ten-percent tangent.

    Parameters
    ----------
    x : Float64[Array, "..."]
        Dimensionless input values in radians.

    Returns
    -------
    result : Float64[Array, "..."]
        The elementwise sine of ``x``.

    Notes
    -----
    The primal is the exact sine. The paired ``_wrong_sine_jvp`` rule
    plants a tangent scaled by 1.1. The harness self-tests verify that
    the gradient check rejects this ten-percent derivative defect.
    """
    result: Float64[Array, "..."] = jnp.sin(x)
    return result


@_wrong_sine.defjvp
def _wrong_sine_jvp(
    primals: Tuple[Float64[Array, "..."], ...],
    tangents: Tuple[Float64[Array, "..."], ...],
) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """PRIVATE: Plant a tangent scaled by 1.1 for harness-defect detection.

    Parameters
    ----------
    primals : Tuple[Float64[Array, "..."], ...]
        One-element tuple that holds the input array ``x``.
    tangents : Tuple[Float64[Array, "..."], ...]
        One-element tuple that holds the input tangent of ``x``.

    Returns
    -------
    result : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        The exact sine primal and the corrupted tangent.

    Notes
    -----
    Computes ``1.1 * cos(x) * x_tangent``, ten percent above the true
    derivative. Every tolerance rung of ``assert_grad_matches_fd`` must
    reject this planted defect while the primal stays exact.
    """
    x: Float64[Array, "..."]
    x_tangent: Float64[Array, "..."]
    (x,) = primals
    (x_tangent,) = tangents
    primal: Float64[Array, "..."] = _wrong_sine(x)
    tangent: Float64[Array, "..."] = 1.1 * jnp.cos(x) * x_tangent
    result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        primal,
        tangent,
    )
    return result


@jax.custom_jvp
def _near_wrong_sine(x: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """PRIVATE: Return sine with a deliberately incorrect five-digit tangent.

    Parameters
    ----------
    x : Float64[Array, "..."]
        Dimensionless input values in radians.

    Returns
    -------
    result : Float64[Array, "..."]
        The elementwise sine of ``x``.

    Notes
    -----
    The primal is the exact sine. The paired ``_near_wrong_sine_jvp``
    rule plants a tangent scaled by 1.00001, so the detection-floor test
    can pin the strictest relative tolerance rung of the harness.
    """
    result: Float64[Array, "..."] = jnp.sin(x)
    return result


@_near_wrong_sine.defjvp
def _near_wrong_sine_jvp(
    primals: Tuple[Float64[Array, "..."], ...],
    tangents: Tuple[Float64[Array, "..."], ...],
) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """PRIVATE: Plant a tangent scaled by 1.00001 to pin the detection floor.

    Parameters
    ----------
    primals : Tuple[Float64[Array, "..."], ...]
        One-element tuple that holds the input array ``x``.
    tangents : Tuple[Float64[Array, "..."], ...]
        One-element tuple that holds the input tangent of ``x``.

    Returns
    -------
    result : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        The exact sine primal and the slightly corrupted tangent.

    Notes
    -----
    Computes ``1.00001 * cos(x) * x_tangent``, one part in 100000 above
    the true derivative. The smooth 1e-6 rung must still detect this
    defect, which documents the sensitivity floor of the gradient check.
    """
    x: Float64[Array, "..."]
    x_tangent: Float64[Array, "..."]
    (x,) = primals
    (x_tangent,) = tangents
    primal: Float64[Array, "..."] = _near_wrong_sine(x)
    tangent: Float64[Array, "..."] = 1.00001 * jnp.cos(x) * x_tangent
    result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        primal,
        tangent,
    )
    return result


@jax.custom_jvp
def _tiny_linear(x: Float64[Array, ""]) -> Float64[Array, ""]:
    """PRIVATE: Return an order-one loss with a resolvable ``1e-7`` derivative.

    Parameters
    ----------
    x : Float64[Array, ""]
        Dimensionless scalar parameter.

    Returns
    -------
    result : Float64[Array, ""]
        The affine loss ``1.0 + 1e-7 * x``.

    Notes
    -----
    The true slope ``1e-7`` sits above the central-difference roundoff
    floor for an order-one loss. The paired ``_tiny_linear_jvp`` rule
    plants a zero tangent, so the finite-difference absolute-tolerance
    tests can verify that the missing slope is not hidden.
    """
    result: Float64[Array, ""] = 1.0 + 1e-7 * x
    return result


@_tiny_linear.defjvp
def _tiny_linear_jvp(
    primals: Tuple[Float64[Array, ""], ...],
    tangents: Tuple[Float64[Array, ""], ...],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Plant an exactly zero tangent for the nonzero linear primal.

    Parameters
    ----------
    primals : Tuple[Float64[Array, ""], ...]
        One-element tuple that holds the scalar input ``x``.
    tangents : Tuple[Float64[Array, ""], ...]
        One-element tuple that holds the input tangent of ``x``.

    Returns
    -------
    result : Tuple[Float64[Array, ""], Float64[Array, ""]]
        The exact affine primal and a zero tangent.

    Notes
    -----
    Discards the input tangent and returns ``zeros_like(x)``. This is a
    one-hundred-percent-wrong derivative that the elementwise and
    directional finite-difference comparisons must both reject.
    """
    x: Float64[Array, ""]
    (x,) = primals
    primal: Float64[Array, ""] = _tiny_linear(x)
    tangent: Float64[Array, ""] = jnp.zeros_like(x)
    result: Tuple[Float64[Array, ""], Float64[Array, ""]] = primal, tangent
    return result


@jax.custom_jvp
def _mixed_scale_linear(x: Float64[Array, "2"]) -> Float64[Array, ""]:
    """PRIVATE: Return a loss whose large parameter has a tiny sensitivity.

    Parameters
    ----------
    x : Float64[Array, "2"]
        Dimensionless parameter pair; the test drives it at scales
        ``1e-3`` and ``1e3``.

    Returns
    -------
    result : Float64[Array, ""]
        The affine loss ``1.0 + 1e-4 * x[0] + 1e-10 * x[1]``.

    Notes
    -----
    The second slope ``1e-10`` stays resolvable only under the large
    per-parameter finite-difference step of ``x[1]``. The paired
    ``_mixed_scale_linear_jvp`` rule suppresses that slope, so the test
    can require a per-coordinate roundoff floor.
    """
    result: Float64[Array, ""] = 1.0 + 1e-4 * x[0] + 1e-10 * x[1]
    return result


@_mixed_scale_linear.defjvp
def _mixed_scale_linear_jvp(
    primals: Tuple[Float64[Array, "2"], ...],
    tangents: Tuple[Float64[Array, "2"], ...],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Plant zero for only the large parameter's resolvable tangent.

    Parameters
    ----------
    primals : Tuple[Float64[Array, "2"], ...]
        One-element tuple that holds the parameter pair ``x``.
    tangents : Tuple[Float64[Array, "2"], ...]
        One-element tuple that holds the input tangent of ``x``.

    Returns
    -------
    result : Tuple[Float64[Array, ""], Float64[Array, ""]]
        The exact affine primal and the truncated tangent.

    Notes
    -----
    Computes ``1e-4 * x_tangent[0]`` and drops the ``1e-10`` sensitivity
    of ``x[1]``. The per-coordinate finite-difference comparison must
    expose the suppressed second coordinate despite its million-fold
    larger step.
    """
    x: Float64[Array, "2"]
    x_tangent: Float64[Array, "2"]
    (x,) = primals
    (x_tangent,) = tangents
    primal: Float64[Array, ""] = _mixed_scale_linear(x)
    tangent: Float64[Array, ""] = 1e-4 * x_tangent[0]
    result: Tuple[Float64[Array, ""], Float64[Array, ""]] = primal, tangent
    return result


class TestGradientHarness(chex.TestCase):
    """Validate the shared finite-difference and gradient harness.

    Covers analytic real and complex derivatives, scale-aware steps, planted
    tangent defects, zero-gradient tripwires, and complex-step restrictions.

    :see: :func:`~tests._gradients.assert_grad_matches_fd`
    :see: :func:`~tests._gradients.assert_nonzero_grad`
    :see: :func:`~tests._gradients.central_fd_grad`
    :see: :func:`~tests._gradients.complex_step_derivative`
    :see: :func:`~tests._gradients.fd_step`
    :see: :func:`~tests._gradients.assert_gradients_match_finite_differences`
    """

    def test_closed_form_truths(self) -> None:
        """Verify analytic smooth gradients pass at relative tolerance 1e-6.

        The shared check must accept closed-form derivatives across smooth and
        stiff regimes and across parameter scales from ``1e-3`` to ``5``.

        Notes
        -----
        The test checks sine, a Gaussian sum, and the two-dimensional
        Rosenbrock function. It also checks a mixed-unit rational monomial.
        Each check uses both autodiff modes and elementwise finite differences
        for the finite-difference requirement.
        """
        sine_input: Float64[Array, "3"] = jnp.array([-0.7, 0.2, 1.1])
        assert_grad_matches_fd(lambda x: jnp.sum(jnp.sin(x)), sine_input)
        gaussian_input: Float64[Array, "3"] = jnp.array([-1.2, 0.3, 0.9])
        assert_grad_matches_fd(
            lambda x: jnp.sum(jnp.exp(-(x**2))), gaussian_input
        )
        rosenbrock_input: Float64[Array, "2"] = jnp.array([-0.8, 1.4])
        assert_grad_matches_fd(
            lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2,
            rosenbrock_input,
            regime="stiff",
        )
        mixed_scale: Float64[Array, "3"] = jnp.array([1e-3, 5.0, 3.0])
        assert_gradients_match_finite_differences(
            lambda x: x[0] ** 2 * x[1] / x[2],
            mixed_scale,
            regime="smooth",
        )

    def test_fd_step_scales_per_element(self) -> None:
        """Verify finite-difference steps scale with each parameter magnitude.

        The step policy must retain numerical resolution for small parameters
        without applying one global perturbation to mixed-unit inputs.

        Notes
        -----
        The test compares a mixed-unit vector against the exact
        ``eps**(1/3) * max(abs(theta), 1e-3)`` prescription.
        """
        theta: Float64[Array, "3"] = jnp.array([1e-4, 5.0, -3.0])
        actual: Float64[Array, "3"] = fd_step(theta)
        ratio: Float64[Array, "3"] = actual / actual[0]
        expected_ratio: Float64[Array, "3"] = jnp.array([1.0, 5000.0, 3000.0])
        chex.assert_trees_all_close(ratio, expected_ratio)

    def test_wirtinger_convention(self) -> None:
        """Pin JAX's C-to-R gradient as d/dRe minus i times d/dIm.

        The finite-difference harness must reproduce JAX's complex gradient
        convention for a real-valued modulus-squared loss.

        Notes
        -----
        The test checks the exact gradient ``2-2j`` at ``1+1j``.
        It also checks generic asymmetric complex data at relative tolerance
        ``1e-8`` for the finite-difference requirement.
        """
        exact: Complex128[Array, ""] = jax.grad(lambda z: jnp.abs(z) ** 2)(
            jnp.asarray(1.0 + 1.0j)
        )
        chex.assert_trees_all_equal(exact, jnp.asarray(2.0 - 2.0j))
        values: Complex128[Array, "4"] = random_generic_complex(
            jax.random.key(20260713), (4,)
        )
        automatic: Complex128[Array, "4"] = jax.grad(
            lambda z: jnp.sum(jnp.abs(z) ** 2)
        )(values)
        finite_difference: Complex128[Array, "4"] = central_fd_grad(
            lambda z: jnp.sum(jnp.abs(z) ** 2), values
        )
        chex.assert_trees_all_close(
            automatic, finite_difference, rtol=1e-8, atol=1e-10
        )

    def test_planted_wrong_gradient(self) -> None:
        """Verify a ten-percent tangent defect fails every tolerance rung.

        No configured smooth, stiff, or singular tolerance may accept the
        deliberately corrupted derivative of an otherwise correct primal.

        Notes
        -----
        A ``custom_jvp`` retains the correct sine primal. It scales the
        derivative by 1.1. The shared check must raise for this requirement.
        """
        theta: Float64[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        regime: GradRegime
        for regime in RTOL_LADDER:
            with (
                self.subTest(regime=regime),
                pytest.raises(
                    AssertionError,
                    match="Not equal to tolerance|gradient mismatch",
                ),
            ):
                assert_grad_matches_fd(
                    lambda x: jnp.sum(_wrong_sine(x)), theta, regime=regime
                )

    def test_detection_floor(self) -> None:
        """Verify a one-part-in-100000 defect fails the smooth tolerance.

        The strict smooth regime must detect an error at its sensitivity floor.
        The forward values remain exact.

        Notes
        -----
        The test uses a planted ``1.00001*cos(x)`` tangent. It requires
        detection at the strictest 1e-6 relative rung.
        """
        theta: Float64[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        with pytest.raises(
            AssertionError,
            match="Not equal to tolerance|gradient mismatch",
        ):
            assert_grad_matches_fd(
                lambda x: jnp.sum(_near_wrong_sine(x)), theta
            )

    def test_tiny_zeroed_gradient_is_not_hidden_by_fd_atol(self) -> None:
        """Reject a missing ``1e-7`` derivative beside an order-one loss.

        The central-FD roundoff bound has gradient units
        ``eps * max(1, abs(loss)) / h``. The historical extra division by
        ``eps**(1/3)`` inflated this bound to roughly ``6e-6`` and admitted
        this 100%-wrong tangent.

        Notes
        -----
        The test evaluates the planted zero tangent without directional
        checks. It requires the elementwise finite-difference comparison to
        raise.
        """
        theta: Float64[Array, ""] = jnp.asarray(1.0)
        with pytest.raises(AssertionError, match="gradient mismatch"):
            assert_grad_matches_fd(_tiny_linear, theta, modes=())

    def test_mixed_scale_uses_per_parameter_fd_atol(self) -> None:
        """Reject a tiny defect on a parameter with a large FD step.

        The second parameter's step is one million times the first one's.
        Its resolvable ``1e-10`` derivative must use its own roundoff floor,
        not a global tolerance derived from the median step.

        Notes
        -----
        The test evaluates a two-coordinate custom tangent. It requires the
        per-coordinate finite-difference comparison to expose the suppressed
        second sensitivity.
        """
        theta: Float64[Array, "2"] = jnp.array([1e-3, 1e3])
        with pytest.raises(AssertionError, match="gradient mismatch"):
            assert_grad_matches_fd(_mixed_scale_linear, theta, modes=())

    def test_fd_atol_tracks_loss_rescaling(self) -> None:
        """Reject the same missing relative sensitivity after loss rescaling.

        Scale an order-one loss and its derivative equally. The absolute
        tolerance must retain the defect at each scale.

        Notes
        -----
        The test repeats the finite-difference check at unit and million-fold
        scales. It requires both comparisons to reject the planted tangent.
        """
        theta: Float64[Array, ""] = jnp.asarray(1.0)

        def _scaled_loss(
            candidate: Float64[Array, ""],
            *,
            scale: float,
        ) -> Float64[Array, ""]:
            """PRIVATE: Scale the planted tiny linear objective.

            Parameters
            ----------
            candidate : Float64[Array, ""]
                Scalar coordinate under differentiation.
            scale : float
                Multiplicative objective scale.

            Returns
            -------
            value : Float64[Array, ""]
                Scaled planted objective value.

            Notes
            -----
            Keeps the scale static while the gradient check varies the input.
            """
            value: Float64[Array, ""] = scale * _tiny_linear(candidate)
            return value

        loss_scale: float
        for loss_scale in (1.0, 1e6):
            with (
                self.subTest(loss_scale=loss_scale),
                pytest.raises(AssertionError, match="gradient mismatch"),
            ):
                assert_grad_matches_fd(
                    partial(_scaled_loss, scale=loss_scale),
                    theta,
                    modes=(),
                )

    def test_directional_tolerance_is_independent(self) -> None:
        """Reject a tiny defect through the randomized directional anchor.

        ``check_grads`` has one scalar perturbation and therefore uses a
        separately derived directional roundoff tolerance. It must not reuse
        an elementwise or historically inflated absolute tolerance.

        Notes
        -----
        The test enables only forward-mode directional checking. It requires
        that independent check to reject the planted zero tangent.
        """
        theta: Float64[Array, ""] = jnp.asarray(1.0)
        with pytest.raises(
            AssertionError,
            match="Not equal to tolerance|gradient mismatch",
        ):
            assert_grad_matches_fd(_tiny_linear, theta, modes=("fwd",))

    def test_planted_zero_gradient(self) -> None:
        """Verify finite-but-zero stopped gradients fail both tripwires.

        A finite primal and finite automatic derivative are insufficient when
        the physical loss retains nonzero finite-difference sensitivity.

        Notes
        -----
        The primal remains ``sum(x**2)`` while ``stop_gradient`` removes all
        autodiff sensitivity. Finite differences and the independent norm
        check must each raise for the gradient requirement.
        """
        theta: Float64[Array, "2"] = jnp.array([1.0, -2.0])

        def stopped_loss(x: Float64[Array, "2"]) -> Float64[Array, ""]:
            result: Float64[Array, ""] = jnp.sum(jax.lax.stop_gradient(x) ** 2)
            return result

        with pytest.raises(
            AssertionError,
            match="Not equal to tolerance|gradient mismatch",
        ):
            assert_grad_matches_fd(stopped_loss, theta)
        with pytest.raises(AssertionError, match="gradient at .* has norm"):
            assert_nonzero_grad(stopped_loss, theta)

    def test_elementwise_tripwire_rejects_masked_zero_coordinate(self) -> None:
        """Prevent one sensitive coordinate from masking a zero coordinate.

        Leaf-level sensitivity remains the default because structural zeros
        can be physical. A check that registers every coordinate must opt in
        to the elementwise tripwire.

        Notes
        -----
        The test first accepts the partially sensitive leaf under the default
        norm check. It then requires the elementwise mode to identify index
        one.
        """
        theta: Float64[Array, "3"] = jnp.asarray((0.4, -0.2, 0.7))

        def partially_sensitive(x: Float64[Array, "3"]) -> Float64[Array, ""]:
            """Return a loss independent of the middle coordinate."""
            result: Float64[Array, ""] = x[0] + x[2] ** 2
            return result

        assert_nonzero_grad(partially_sensitive, theta)
        with pytest.raises(
            AssertionError,
            match=r"coordinate 1 has magnitude 0\.000000e\+00",
        ):
            assert_nonzero_grad(
                partially_sensitive,
                theta,
                elementwise=True,
            )

    def test_complex_step_matches_directional_truth(self) -> None:
        """Compare complex step with an analytic directional derivative.

        Complex step retains machine precision for a real-on-real holomorphic
        function. Correctness comes from the independent cosine expression,
        not from inspecting whether the estimate is nonzero.

        Notes
        -----
        The test compares explicit, default, and compiled directions against
        the cosine derivative. It uses a nonuniform direction to expose
        accidental scalar treatment.
        """
        x: Float64[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        direction: Float64[Array, "3"] = jnp.array([0.5, -2.0, 1.25])
        derivative: Float64[Array, "3"] = complex_step_derivative(
            jnp.sin,
            x,
            direction=direction,
        )
        chex.assert_trees_all_close(
            derivative,
            jnp.cos(x) * direction,
            rtol=1e-15,
            atol=0.0,
        )
        default_direction: Float64[Array, "3"] = complex_step_derivative(
            jnp.sin,
            x,
        )
        chex.assert_trees_all_close(
            default_direction,
            jnp.cos(x),
            rtol=1e-15,
            atol=0.0,
        )
        compiled: Float64[Array, "3"] = jax.jit(
            lambda value: complex_step_derivative(
                jnp.sin,
                value,
                direction=direction,
            )
        )(x)
        chex.assert_trees_all_close(
            compiled,
            jnp.cos(x) * direction,
            rtol=1e-15,
            atol=0.0,
        )

    def test_complex_step_accepts_valid_zero_derivatives(self) -> None:
        """Keep stationary and constant holomorphic derivatives equal to zero.

        A zero imaginary response is not evidence of non-holomorphy. Constants
        and ``z**2`` at the origin are explicit counterexamples to the former
        value-based guard.

        Notes
        -----
        The test applies complex step to a constant and a stationary
        quadratic. It compares both estimates with exact zero arrays.
        """
        x: Float64[Array, "3"] = jnp.zeros((3,))
        constant: Float64[Array, "3"] = complex_step_derivative(
            jnp.ones_like,
            x,
        )
        stationary: Float64[Array, "3"] = complex_step_derivative(
            lambda value: value**2,
            x,
        )
        chex.assert_trees_all_equal(constant, jnp.zeros_like(x))
        chex.assert_trees_all_equal(stationary, jnp.zeros_like(x))

    def test_complex_step_requires_independent_holomorphy_truth(self) -> None:
        """Expose conjugation through an independent analytic comparison.

        At real inputs complex step returns ``-1`` for conjugation, although
        the real-direction derivative is ``+1``. A nonzero estimate therefore
        cannot certify holomorphy.

        Notes
        -----
        The test records the misleading conjugation estimate. It then compares
        that estimate with the independent positive derivative and requires
        failure.
        """
        x: Float64[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        estimate: Float64[Array, "3"] = complex_step_derivative(jnp.conj, x)
        chex.assert_trees_all_equal(estimate, -jnp.ones_like(x))
        with pytest.raises(
            AssertionError,
            match="Values not approximately equal",
        ):
            chex.assert_trees_all_close(
                estimate,
                jnp.ones_like(x),
                rtol=1e-15,
                atol=0.0,
            )

    def test_nonholomorphic_maps_use_stacked_real_fd(self) -> None:
        """Validate a complex-to-real map with stacked-real differences.

        Modulus-squared is intentionally outside complex-step's domain. The
        The central-FD check handles its real and imaginary directions.

        Notes
        -----
        The test supplies asymmetric complex values to the shared gradient
        check. The check compares autodiff with separate real and imaginary
        perturbations.
        """
        values: Complex128[Array, "3"] = jnp.array(
            [0.2 + 0.3j, -0.7 + 0.5j, 1.1 - 0.4j]
        )
        assert_grad_matches_fd(
            lambda value: jnp.sum(jnp.abs(value) ** 2),
            values,
        )

    def test_check_grads_semantics_anchor(self) -> None:
        """Pin JAX check_grads behavior on truth and a planted tangent defect.

        The upstream JAX checker must accept the analytical sine derivative.
        It must reject the corrupted tangent from the harness test.

        Notes
        -----
        The test calls the JAX directional-gradient checker on sine and the
        scaled ``custom_jvp``. This direct check establishes the independent
        semantic reference for the finite-difference requirement.
        """
        theta: Float64[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        test_util.check_grads(
            lambda x: jnp.sum(jnp.sin(x)),
            (theta,),
            order=1,
            modes=("fwd", "rev"),
            eps=1e-5,
        )
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            test_util.check_grads(
                lambda x: jnp.sum(_wrong_sine(x)),
                (theta,),
                order=1,
                modes=("fwd", "rev"),
                eps=1e-5,
            )
