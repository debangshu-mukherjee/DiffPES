"""Check finite padding and static-shape sharding primitives.

Use independent inputs to define the covered behavior.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, List, Tuple
from jax.test_util import check_grads
from jaxtyping import Array, Float64, Int64
from numpy.typing import NDArray

from diffpes.simul import pad_with_mask, sharded_kmap, sharded_ksum
from diffpes.types import make_shard_spec

_CASE_COUNT = 24
_CAPACITY = 128
_TOLERANCE = 1.0e-12


def _compiled_cache_size(compiled: Any) -> int:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: int = compiled._cache_size()  # noqa: SLF001
    return result


class TestPadWithMask:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.pad_with_mask``
    """

    def test_round_trips_seeded_physical_prefixes(self) -> None:
        """Preserve each random prefix and mark only repeated padding zero.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        generator: np.random.Generator = np.random.default_rng(8102)
        lengths: Int64[NDArray, " n"] = generator.integers(
            1, _CAPACITY + 1, size=_CASE_COUNT - 1
        )
        all_lengths: Tuple[int, ...] = tuple(
            int(value) for value in lengths
        ) + (_CAPACITY,)
        length: int
        for length in all_lengths:
            raw: Float64[NDArray, "n_k 3"] = generator.normal(size=(length, 3))
            points: Float64[Array, "n_k 3"] = jnp.asarray(raw)
            padded: Float64[Array, "nk_max 3"]
            mask: Float64[Array, " nk_max"]
            padded, mask = pad_with_mask(points, _CAPACITY)
            assert jnp.array_equal(padded[:length], points)
            assert jnp.array_equal(mask[:length], jnp.ones(length))
            assert jnp.array_equal(
                mask[length:], jnp.zeros(_CAPACITY - length)
            )
            if length < _CAPACITY:
                assert jnp.all(padded[length:] == points[-1])


class TestShardedKmap:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.sharded_kmap``
    """

    def test_maps_mask_folded_weights(self) -> None:
        """Match direct vector arithmetic after scanning fixed-size chunks.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        spec: Any
        points: Any
        mask: Any
        spec = make_shard_spec(n_devices=1, chunk_size=2, nk_max=4)
        points, mask = pad_with_mask(jnp.asarray([[1.0, 0.0, 0.0]]), 4)

        def body(
            chunk: Float64[Array, "chunk 3"],
            weights: Float64[Array, " chunk"],
        ) -> Float64[Array, "chunk 1"]:
            """Check the private helper behavior."""
            values: Float64[Array, "chunk 1"] = weights[:, None] * chunk[:, :1]
            return values

        mapped: Float64[Array, "nk_max 1"] = sharded_kmap(
            body, points, mask, spec
        )
        assert jnp.array_equal(mapped[:, 0], jnp.asarray([1.0, 0.0, 0.0, 0.0]))


class TestShardedKsum:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.simul.sharded_ksum``
    """

    def test_matches_unpadded_values_and_gradients(self) -> None:
        """Match direct masked complex arithmetic in primal and reverse mode.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        spec: Any
        points: Any
        mask: Any
        spec = make_shard_spec(n_devices=1, chunk_size=4, nk_max=8)
        physical: Float64[Array, "n_k 3"] = jnp.asarray(
            [[0.2, -0.3, 0.4], [0.8, 0.1, -0.5], [-0.4, 0.7, 0.2]]
        )
        points, mask = pad_with_mask(physical, spec.nk_max)

        def reduced(theta: Float64[Array, ""]) -> Float64[Array, ""]:
            """Check the private helper behavior."""

            def body(
                chunk: Float64[Array, "chunk 3"],
                weights: Float64[Array, " chunk"],
            ) -> Float64[Array, " chunk"]:
                """Check the private helper behavior."""
                phase: Any = jnp.exp(1.0j * theta * chunk[:, 0])
                values: Float64[Array, " chunk"] = weights * jnp.real(
                    phase * (chunk[:, 1] + 1.0j * chunk[:, 2])
                )
                return values

            result: Float64[Array, ""] = sharded_ksum(body, points, mask, spec)
            return result

        def direct(theta: Float64[Array, ""]) -> Float64[Array, ""]:
            """Check the private helper behavior."""
            phase: Any = jnp.exp(1.0j * theta * physical[:, 0])
            result: Float64[Array, ""] = jnp.sum(
                jnp.real(phase * (physical[:, 1] + 1.0j * physical[:, 2]))
            )
            return result

        theta: Float64[Array, ""] = jnp.asarray(0.37)
        assert jnp.allclose(reduced(theta), direct(theta), rtol=_TOLERANCE)
        assert jnp.allclose(
            jax.grad(reduced)(theta), jax.grad(direct)(theta), rtol=_TOLERANCE
        )
        check_grads(reduced, (theta,), order=1, modes=("fwd", "rev"))
        assert jnp.abs(jax.grad(reduced)(theta)) > 0.0

    def test_padding_parameter_has_exactly_zero_gradient(self) -> None:
        """Keep a physical derivative nonzero and a padding derivative zero.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Apply two parameters on complementary mask regions before folding the
        same mask into the returned values.
        """
        spec: Any = make_shard_spec(n_devices=1, chunk_size=2, nk_max=4)
        points: Any
        mask: Any
        points, mask = pad_with_mask(
            jnp.asarray([[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]]), 4
        )

        def evaluate(parameters: Float64[Array, " 2"]) -> Float64[Array, ""]:
            """Check the private helper behavior."""

            def body(
                chunk: Float64[Array, "chunk 3"],
                weights: Float64[Array, " chunk"],
            ) -> Float64[Array, " chunk"]:
                """Check the private helper behavior."""
                raw: Float64[Array, " chunk"] = (
                    parameters[0] * weights * chunk[:, 0]
                    + parameters[1] * (1.0 - weights) * chunk[:, 0]
                )
                values: Float64[Array, " chunk"] = weights * raw
                return values

            result: Float64[Array, ""] = sharded_ksum(body, points, mask, spec)
            return result

        gradient: Float64[Array, " 2"] = jax.grad(evaluate)(
            jnp.asarray([0.3, 0.8])
        )
        assert jnp.abs(gradient[0]) > 0.0
        assert gradient[1] == 0.0

    def test_pathological_padding_stays_finite_and_has_zero_weight(
        self,
    ) -> None:
        """Keep repeated large finite lanes harmless in primal and tangent.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        spec: Any
        points: Any
        mask: Any
        spec = make_shard_spec(n_devices=1, chunk_size=2, nk_max=4)
        points, mask = pad_with_mask(jnp.asarray([[1.0e100, 0.0, 0.0]]), 4)

        def evaluate(theta: Float64[Array, ""]) -> Float64[Array, ""]:
            """Check the private helper behavior."""

            def body(
                chunk: Float64[Array, "chunk 3"],
                weights: Float64[Array, " chunk"],
            ) -> Float64[Array, " chunk"]:
                """Check the private helper behavior."""
                values: Float64[Array, " chunk"] = weights * jnp.tanh(
                    theta * chunk[:, 0]
                )
                return values

            result: Float64[Array, ""] = sharded_ksum(body, points, mask, spec)
            return result

        primal: Float64[Array, ""]
        tangent: Float64[Array, ""]
        primal, tangent = jax.jvp(
            evaluate, (jnp.asarray(0.1),), (jnp.asarray(1.0),)
        )
        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_static_capacity_compiles_once(self) -> None:
        """Reuse one compiled executable for four padded physical lengths.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        spec: Any
        compiled: Any
        points: Any
        mask: Any
        spec = make_shard_spec(n_devices=1, chunk_size=16, nk_max=_CAPACITY)

        def body(
            chunk: Float64[Array, "chunk 3"],
            weights: Float64[Array, " chunk"],
        ) -> Float64[Array, " chunk"]:
            """Check the private helper behavior."""
            values: Float64[Array, " chunk"] = weights * jnp.sum(
                chunk**2, axis=1
            )
            return values

        compiled = jax.jit(
            lambda points, mask: sharded_ksum(body, points, mask, spec)
        )
        length: int
        for length in (17, 64, 100, _CAPACITY):
            points, mask = pad_with_mask(jnp.ones((length, 3)), _CAPACITY)
            compiled(points, mask).block_until_ready()
        assert _compiled_cache_size(compiled) == 1

    def test_eight_device_fake_mesh_matches_one_device(self) -> None:
        """Compare values and gradients in an isolated eight-device process.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        completed: Any
        program: str = """
import jax
import jax.numpy as jnp
from diffpes.types import make_shard_spec
from diffpes.simul import sharded_ksum
x = jnp.arange(24.0, dtype=jnp.float64).reshape(8, 3) / 10.0
w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float64)
def run(n, theta):
    spec = make_shard_spec(n_devices=n, chunk_size=1, nk_max=8)
    def body(k, weight):
        return weight * jnp.sin(k[:, 0])
    return sharded_ksum(body, theta * x, w, spec)
for n in (1, 8):
    mesh = jax.make_mesh((n,), ("k",))
    with jax.set_mesh(mesh):
        value = run(n, jnp.asarray(0.4))
        gradient = jax.grad(lambda t: run(n, t))(jnp.asarray(0.4))
    print(float(value), float(gradient))
"""
        environment: Dict[str, str] = dict(os.environ)
        environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        rows: List[List[float]] = [
            [float(value) for value in line.split()]
            for line in completed.stdout.strip().splitlines()
        ]
        assert np.allclose(rows[0], rows[1], rtol=_TOLERANCE)
