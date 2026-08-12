"""Specify bounded host-only WAVECAR indexing contracts.

Extended Summary
----------------
These tests use minimal synthetic direct-access files.  They verify that the
reader rejects unsupported metadata before coefficient loading and retains a
typed, non-materializing source descriptor.

Routine Listings
----------------
:class:`TestWavecarHeader`
    Specify WAVECAR-header validation.
:class:`TestWavecarIndex`
    Specify host-only WAVECAR indexing.
"""

import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

from diffpes.inout import index_wavecar, wavecar_header
from diffpes.types import make_vasp_wavefunction_source


class TestWavecarHeader:
    """Specify WAVECAR-header validation.

    See Also
    --------
    diffpes.inout.wavecar_header
        Production parser covered by these contract tests.
    """

    def test_rejects_unsupported_layout(self, tmp_path: Path) -> None:
        """Reject unknown precision tags before requesting coefficient data."""
        path = tmp_path / "WAVECAR"
        path.write_bytes(struct.pack("<3d", 24.0, 1.0, 99999.0) + b"\0" * 32)
        with pytest.raises(ValueError, match="unsupported"):
            wavecar_header(path)


class TestWavecarIndex:
    """Specify host-only WAVECAR indexing.

    See Also
    --------
    diffpes.inout.index_wavecar
        Production indexer covered by these contract tests.
    """

    def test_is_descriptor_only(self, tmp_path: Path) -> None:
        """Retain source identity without materializing coefficient tensors."""
        wavecar = tmp_path / "WAVECAR"
        wavecar.write_bytes(
            struct.pack("<3d", 24.0, 1.0, 45200.0) + b"\0" * 72
        )
        source = make_vasp_wavefunction_source(
            wavecar,
            jnp.asarray([1.0]),
            jnp.asarray(0.0),
            spin_mode="scalar",
            source_ref="org.diffpes.fixture.wavecar@1",
            potcar_sha256=("fixture-hash",),
        )
        dataset = index_wavecar(source)
        assert dataset.source.wavecar_path == wavecar
        assert dataset.header.precision_tag == 45200
