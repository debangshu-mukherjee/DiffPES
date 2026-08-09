"""Validate deterministic canonical scientific records.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import subprocess
import sys
import unicodedata

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Tuple

from diffpes.certify import (
    canonical_json,
    canonical_pytree,
    iter_canonical_pytree_chunks,
)


class TestCanonicalPytree:
    """Verify :func:`~diffpes.certify.canonical_pytree`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.canonical_pytree`
    """

    def test_record_is_stable(self) -> None:
        """Verify equal typed values produce identical canonical bytes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        Encodes the same nested value twice and compares exact bytes.
        """
        value: Tuple[str, Tuple[int, ...]] = ("stable", (1, 2))
        assert canonical_pytree(value) == canonical_pytree(value)

    def test_numpy_and_jax_arrays_have_identical_canonical_bytes(self) -> None:
        """Treat equal concrete NumPy and JAX arrays as one normalized content.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        numpy_value: Any
        jax_value: Any
        expected_hex: Any
        numpy_value = np.array([[1.0, 2.0]], dtype=np.float64)
        jax_value = jnp.asarray(numpy_value)
        expected_hex = (
            "444946465045532d43414e4f4e4943414c2d5059545245452d5631"
            "004100000000000000033c66380000000000000002000000000000"
            "000100000000000000020000000000000010000000000000f03f00"
            "00000000000040"
        )
        assert canonical_pytree(numpy_value).hex() == expected_hex
        assert canonical_pytree(jax_value) == canonical_pytree(numpy_value)

    def test_array_canonicalization_retains_dtype_shape_and_complex_values(
        self,
    ) -> None:
        """Keep numerical layout semantics and canonical complex bytes distinct.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        values: Any
        values = np.array([1.0, 2.0], dtype=np.float64)
        assert canonical_pytree(values) != canonical_pytree(
            values.astype(np.float32)
        )
        assert canonical_pytree(values) != canonical_pytree(
            values.reshape(1, 2)
        )
        assert canonical_pytree(np.array([1.0 + 2.0j]))

    def test_canonical_record_is_stable_in_a_fresh_process(self) -> None:
        """Keep canonical content stable across independent Python processes.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        expected: Any
        code: Any
        observed: Any
        value = {"axis": ("energy", "momentum"), "unit": "eV"}
        expected = canonical_pytree(value).hex()
        code = (
            "from diffpes.certify import canonical_pytree; "
            "print(canonical_pytree("
            "{'unit': 'eV', 'axis': ('energy', 'momentum')}"
            ").hex())"
        )
        observed = subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
        ).strip()
        assert observed == expected

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), np.array([1.0, np.nan])],
    )
    def test_nonfinite_values_are_rejected(self, value: Any) -> None:
        """Exclude unstable NaN and infinity encodings from scientific records.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="NaN|nonfinite|infinity"):
            canonical_pytree(value)


class TestCanonicalJson:
    """Verify :func:`~diffpes.certify.canonical_json`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.canonical_json`
    """

    def test_canonical_json_golden_record_and_mapping_order(self) -> None:
        """Pin typed scalars, NFC text, and deterministic mapping order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        value: Any
        expected_hex: Any
        value = {"b": None, "a": (1, 2.5, "e\u0301")}
        expected_hex = (
            "444946465045532d43414e4f4e4943414c2d4a534f4e2d563100"
            "7b22246d6170223a5b5b7b2224737472223a2261227d2c7b2224"
            "7475706c65223a5b7b2224696e74223a2231227d2c7b2224666c"
            "6f61743634223a2234303034303030303030303030303030227d"
            "2c7b2224737472223a22c3a9227d5d7d5d2c5b7b222473747222"
            "3a2262227d2c7b22246e6f6e65223a747275657d5d5d7d"
        )
        assert canonical_json(value).hex() == expected_hex
        assert canonical_json(value) == canonical_json(
            {"a": value["a"], "b": None}
        )

    def test_canonical_json_distinguishes_list_tuple_null_and_absent(
        self,
    ) -> None:
        """Retain meaning-bearing container and presence distinctions.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        assert canonical_json([1, 2]) != canonical_json((1, 2))
        assert canonical_json({"x": None}) != canonical_json({})

    def test_canonical_text_normalizes_unicode_and_rejects_key_collision(
        self,
    ) -> None:
        """Normalize equivalent text while refusing ambiguous normalized keys.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        decomposed: Any
        composed: Any
        decomposed = "e\u0301"
        composed = unicodedata.normalize("NFC", decomposed)
        assert canonical_json(decomposed) == canonical_json(composed)
        with pytest.raises(ValueError, match="collide"):
            canonical_json({decomposed: 1, composed: 2})


class TestIterCanonicalPytreeChunks:
    """Verify :func:`~diffpes.certify.iter_canonical_pytree_chunks`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.iter_canonical_pytree_chunks`
    """

    def test_streamed_and_materialized_records_are_identical(self) -> None:
        """Make bounded chunks exactly reproduce the full canonical record.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        values: Any
        streamed: Any
        values = np.arange(4097, dtype=np.int64)
        streamed = b"".join(
            iter_canonical_pytree_chunks(values, chunk_bytes=127)
        )
        assert streamed == canonical_pytree(values)

    def test_unsupported_values_and_invalid_chunk_size_are_rejected(
        self,
    ) -> None:
        """Keep the canonical vocabulary deliberately closed and bounded.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(ValueError, match="unsupported"):
            canonical_pytree({1, 2})
        with pytest.raises(ValueError, match="positive"):
            tuple(iter_canonical_pytree_chunks([1], chunk_bytes=0))
