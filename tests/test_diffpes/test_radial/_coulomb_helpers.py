"""Provide shared fixtures for the mirrored split test modules.

The helpers preserve the inputs and independent reference calculations.
"""

from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import Dict
from jaxtyping import Shaped, jaxtyped
from numpy.typing import NDArray


@jaxtyped(typechecker=beartype)
def _reference() -> Dict[str, Shaped[NDArray, "..."]]:
    """PRIVATE: Load the frozen 80-digit Coulomb value and derivative artifact.

    Returns
    -------
    result : Dict[str, Shaped[NDArray, "..."]]
        Every array in the archive keyed by its stored name. The
        content includes the dimensionless ``etas`` grid and the frozen
        ``phase`` and ``phase_eta`` rows for orders zero to four.

    Notes
    -----
    Opens ``coulomb_mpmath_80digit.npz`` beside this module and copies
    each member of the archive into a plain dictionary. The values come
    from an offline 80-digit mpmath computation.
    """
    path: Path = (
        Path(__file__).with_name("data") / "coulomb_mpmath_80digit.npz"
    )
    archive: np.lib.npyio.NpzFile = np.load(path)
    result: Dict[str, Shaped[NDArray, "..."]] = {
        name: archive[name] for name in archive.files
    }
    return result
