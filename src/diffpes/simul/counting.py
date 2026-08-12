"""Sample detector counts from expected rates.

Extended Summary
----------------
This module keeps explicit random keys outside the expected-rate graph.
The expected-rate graph stays differentiable.

Routine Listings
----------------
:func:`fixed_total_probabilities`
    Normalize all detector rates to one event-probability tensor.
:func:`sample_fixed_total_counts`
    Generate one fixed-total multinomial count tensor.
:func:`sample_poisson_counts`
    Generate independent Poisson counts for a rate tensor.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float64, Int64, PRNGKeyArray, jaxtyped


@jaxtyped(typechecker=beartype)
def fixed_total_probabilities(  # noqa: DOC503
    rates: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Normalize all detector rates to one event-probability tensor.

    The operation treats every array entry as one category in a single
    fixed-total acquisition.

    :see: :class:`~.test_counting.TestFixedTotalProbabilities`

    Parameters
    ----------
    rates : Float64[Array, "..."]
        Finite nonnegative rates with a positive global sum.

    Returns
    -------
    probabilities : Float64[Array, "..."]
        Same-shaped probabilities whose global sum is one.

    Raises
    ------
    ValueError
        If rates are scalar or empty.
    EquinoxRuntimeError
        If a rate is non-finite or negative, or their global sum is not
        positive.

    Notes
    -----
    Global normalization preserves the tensor shape and removes the overall
    exposure scale from multinomial probabilities.
    """
    rate_array: Float64[Array, "..."] = jnp.asarray(rates, dtype=jnp.float64)
    if rate_array.ndim < 1 or rate_array.size < 1:
        raise ValueError("fixed-total rates must be a nonempty array")
    rate_array = eqx.error_if(
        rate_array,
        ~jnp.all(jnp.isfinite(rate_array)) | ~jnp.all(rate_array >= 0.0),
        "fixed-total rates must be finite and nonnegative",
    )
    total_rate: Float64[Array, ""] = jnp.sum(rate_array)
    total_rate = eqx.error_if(
        total_rate,
        ~(total_rate > 0.0),
        "fixed-total rates must have a positive sum",
    )
    probabilities: Float64[Array, "..."] = rate_array / total_rate
    return probabilities


@jaxtyped(typechecker=beartype)
def sample_poisson_counts(  # noqa: DOC503
    key: PRNGKeyArray,
    rates: Float64[Array, "..."],
) -> Int64[Array, "..."]:
    """Generate independent Poisson counts for a rate tensor.

    The sampler maps each expected rate to an independent integer variate
    using one explicit JAX key.

    :see: :class:`~.test_counting.TestSamplePoissonCounts`

    Parameters
    ----------
    key : PRNGKeyArray
        Explicit JAX random key.
    rates : Float64[Array, "..."]
        Finite nonnegative Poisson means.

    Returns
    -------
    counts : Int64[Array, "..."]
        Integer sample with the same shape as ``rates``.

    Raises
    ------
    ValueError
        If rates are scalar or empty.
    EquinoxRuntimeError
        If a rate is non-finite or negative.

    Notes
    -----
    Integer draws are intentionally outside the differentiable graph.
    """
    rate_array: Float64[Array, "..."] = jnp.asarray(rates, dtype=jnp.float64)
    if rate_array.ndim < 1 or rate_array.size < 1:
        raise ValueError("Poisson rates must be a nonempty array")
    rate_array = eqx.error_if(
        rate_array,
        ~jnp.all(jnp.isfinite(rate_array)) | ~jnp.all(rate_array >= 0.0),
        "Poisson rates must be finite and nonnegative",
    )
    counts: Int64[Array, "..."] = jax.random.poisson(
        key, rate_array, dtype=jnp.int64
    )
    return counts


@jaxtyped(typechecker=beartype)
def sample_fixed_total_counts(  # noqa: DOC503
    key: PRNGKeyArray,
    rates: Float64[Array, "..."],
    total_count: int,
) -> Int64[Array, "..."]:
    """Generate one fixed-total multinomial count tensor.

    The sampler normalizes all rates and returns one integer realization with
    an exact declared total.

    :see: :class:`~.test_counting.TestSampleFixedTotalCounts`

    Parameters
    ----------
    key : PRNGKeyArray
        Explicit JAX random key.
    rates : Float64[Array, "..."]
        Finite nonnegative rates with positive global sum.
    total_count : int
        Positive static number of acquired events.

    Returns
    -------
    counts : Int64[Array, "..."]
        One multinomial count tensor summing exactly to ``total_count``.

    Raises
    ------
    ValueError
        If ``total_count`` is not a positive integer or rates are empty.
    EquinoxRuntimeError
        If rates are non-finite, negative, or have a nonpositive sum.

    Notes
    -----
    This is one multinomial draw over all bins, not independent Poisson
    sampling. Integer draws are intentionally outside the derivative graph.
    """
    if type(total_count) is not int or total_count <= 0:
        raise ValueError("total_count must be a positive integer")
    probabilities: Float64[Array, "..."] = fixed_total_probabilities(rates)
    flat_counts: Int64[Array, " N"] = jax.random.multinomial(
        key,
        total_count,
        probabilities.reshape((-1,)),
        dtype=jnp.float64,
    ).astype(jnp.int64)
    counts: Int64[Array, "..."] = flat_counts.reshape(probabilities.shape)
    return counts


__all__: list[str] = [
    "fixed_total_probabilities",
    "sample_fixed_total_counts",
    "sample_poisson_counts",
]
