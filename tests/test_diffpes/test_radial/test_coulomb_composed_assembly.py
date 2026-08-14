"""Run the isolated complete Coulomb-assembly gradient witness.

Extended Summary
----------------
The test delegates the expensive charge and photon-energy derivative set
to a process that releases compiled Coulomb executables on exit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestCoulombComposedAssembly:
    """Certify charge and photon-energy derivatives through full assembly.

    The case covers forward, reverse, and finite-difference sensitivities in
    the complete supported-radial computation. It runs the committed verifier
    in an isolated process and requires a successful exit with diagnostics.
    """

    @pytest.mark.slow
    def test_forward_reverse_and_fd_witnesses_in_isolated_process(
        self,
    ) -> None:
        """Require the compact supported-radial derivative capstone to pass.

        The test checks forward, reverse, and finite-difference witnesses.

        Notes
        -----
        It runs the committed verifier and reports captured process output.
        """
        root: Path = Path(__file__).parents[3]
        script: Path = (
            root
            / "tests"
            / "_reference_tools"
            / "verify_coulomb_composed_assembly.py"
        )
        completed: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
