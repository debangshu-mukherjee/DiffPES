"""Run the isolated complete-assembly Plan 06 D11 witness.

The test delegates the expensive charge and photon-energy derivative battery
to a process that releases compiled Coulomb executables on exit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class TestPlan06CoulombComposedAssembly:
    """Certify charge and photon-energy derivatives through full assembly."""

    def test_forward_reverse_and_fd_witnesses_in_isolated_process(
        self,
    ) -> None:
        """Require the compact supported-radial D11 capstone to pass.

        The test checks forward, reverse, and finite-difference witnesses.

        Notes
        -----
        It runs the committed verifier and reports captured process output.
        """
        root: Path = Path(__file__).parents[3]
        script: Path = root / "scripts" / "verify_coulomb_composed_assembly.py"
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
