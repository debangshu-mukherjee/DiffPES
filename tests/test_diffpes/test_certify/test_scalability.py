"""Measure independent certification identity and resource behavior.

The tests isolate process identity, streaming memory, and orchestration time.
They are calibration fixtures and do not claim the full-cube release limits.
"""

import subprocess
import sys
import time
import tracemalloc
import uuid

import jax.numpy as jnp
from beartype.typing import Any, Iterator

from diffpes.certify import (
    certify_forward,
    checksum_chunks,
    prepare_certification,
    register_model,
)
from diffpes.inout import certificate_identity
from diffpes.types import make_execution_manifest, make_forward_model_spec
from tests.test_diffpes.test_inout.test_certificate import sample_certificate

_MEBIBYTE = 1024 * 1024


class TestCertificateProcessFixture:
    """Measure canonical certificate identity in an independent process.

    The case compares the same scientific certificate across process limits.
    """

    def test_identity_matches_child_process(self) -> None:
        """Compute the same canonical identity in a fresh Python process.

        Process initialization must not change the scientific identity.

        Notes
        -----
        The child imports the shared deterministic certificate fixture.
        """
        expected: str = certificate_identity(sample_certificate())
        program: str = (
            "from diffpes.inout import certificate_identity; "
            "from tests.test_diffpes.test_inout.test_certificate "
            "import sample_certificate; "
            "print(certificate_identity(sample_certificate()))"
        )
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().splitlines()[-1] == expected


class TestChecksumMemoryFixture:
    """Measure bounded allocation during streamed SHA-256 computation.

    The case streams 1 GiB through one reusable 1 MiB memory view.
    """

    def test_stream_does_not_retain_all_chunks(self) -> None:
        """Keep added allocation below ten percent of one streamed chunk.

        The checksum operation must not accumulate the 1 GiB logical stream.

        Notes
        -----
        The test starts allocation tracing after it creates the source chunk.
        """
        chunk: bytes = b"x" * _MEBIBYTE

        def chunks() -> Iterator[memoryview]:
            for _ in range(1024):
                yield memoryview(chunk)

        tracemalloc.start()
        checksum: str = checksum_chunks(chunks(), record_kind="memory-fixture")
        current: int
        peak: int
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del current
        assert checksum.startswith("sha256:1:memory-fixture:")
        assert peak < _MEBIBYTE // 10


class TestCertificationOverheadFixture:
    """Measure model runtime and certification orchestration separately.

    The case records warm CPU timings for a closed-form scalar model.
    """

    def test_records_separate_warm_timings(self, record_property: Any) -> None:
        """Record finite model and certification durations independently.

        The two properties must keep model time separate from orchestration time.

        Notes
        -----
        The fixture excludes compilation by warming both calls before timing.
        """
        suffix: str = uuid.uuid4().hex
        model_id: str = f"org.diffpes.model.overhead_test.{suffix}"

        def executor(value: Any) -> Any:
            result: Any = value**2 + 1.0
            return result

        spec: Any = make_forward_model_spec(
            model_id=model_id,
            model_version="1.0.0",
            observable_id="org.diffpes.observable.test.scalar",
            implementation_ref="tests.overhead",
            differentiable_paths=("x",),
        )
        register_model(spec, executor)
        manifest: Any = make_execution_manifest(
            execution_id=f"execution-{suffix}",
            model_ref=f"{model_id}@1.0.0",
            schema_version="1.0.0",
            package_version="test",
            source_checksum="source",
            environment_checksum="environment",
            backend="cpu",
            precision_policy="float64",
            deterministic=True,
            started_at_utc="2026-07-21T00:00:00Z",
        )
        context: Any = prepare_certification(
            model_id,
            "1.0.0",
            manifest,
            policy_id="org.diffpes.policy.exploratory.v1",
        )
        inputs: Any = jnp.array([2.0])
        executor(inputs).block_until_ready()
        certify_forward(
            context, inputs, spectrum_rank=1
        ).value.block_until_ready()

        model_start: float = time.perf_counter()
        executor(inputs).block_until_ready()
        model_seconds: float = time.perf_counter() - model_start

        certification_start: float = time.perf_counter()
        certified: Any = certify_forward(context, inputs, spectrum_rank=1)
        certified.value.block_until_ready()
        certification_seconds: float = (
            time.perf_counter() - certification_start
        )

        record_property("model_seconds", model_seconds)
        record_property("certification_seconds", certification_seconds)
        assert model_seconds >= 0.0
        assert certification_seconds >= model_seconds
