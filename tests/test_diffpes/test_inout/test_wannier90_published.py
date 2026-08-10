"""Exercise the published-input companion benchmark for Wannier90 WSe2 parity.

The test authenticates a compressed WSe2 Wannier input and frozen eigenvalues.
"""

import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Float64
from numpy.typing import NDArray

from diffpes.inout import read_wannier90_hr
from diffpes.tightb import eigvalsh_bands
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    WannierOperatorData,
    make_crystal_geometry,
    make_orbital_basis,
)

_REFERENCE_DIRECTORY = Path(__file__).parents[1] / "_reference_data"
_COMPRESSED_INPUT = _REFERENCE_DIRECTORY / "wannier90_wse2_soc_11bnd_hr.dat.xz"
_FROZEN_REFERENCE = _REFERENCE_DIRECTORY / "wannier90_wse2_reference.json"
_COMPRESSED_SHA256 = (
    "756fdcf2541aa75dad69ae172327fd5cdf6ba044812c918efb9c62a690ece9d4"
)
_REFERENCE_SHA256 = (
    "0a9acf21d86167b7f3a9533b87139e4383981f65bed9acad7f951a275a18b411"
)
_SOURCE_SHA256 = (
    "8ea8140e4fb3d1e56c188d5d680ab077b9ad57070f9205c7365cbb24a7c40dd1"
)


def _sha256(payload: bytes) -> str:
    """PRIVATE: Return the hexadecimal SHA-256 digest of a byte payload.

    Parameters
    ----------
    payload : bytes
        Raw bytes to digest.

    Returns
    -------
    digest : str
        Lowercase 64-character hexadecimal SHA-256 digest.

    Notes
    -----
    Wraps ``hashlib.sha256`` so the authentication assertions stay on
    one line.
    """
    return hashlib.sha256(payload).hexdigest()


def _wannier_context(
    n_wannier: int,
) -> Tuple[CrystalGeometry, OrbitalBasis]:
    """PRIVATE: Build a neutral hr-gauge carrier context for eigenvalue comparison.

    Parameters
    ----------
    n_wannier : int
        Number of Wannier functions in the published model.

    Returns
    -------
    geometry : CrystalGeometry
        Identity cubic lattice in Angstrom with one placeholder site.
    basis : OrbitalBasis
        ``n_wannier`` nominal s orbitals on the one site, labeled
        ``wannier_<index>``.

    Notes
    -----
    Places every Wannier function at the origin so the Bloch phases
    reproduce the hr gauge, which keeps the eigenvalues comparable
    with the frozen reference.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("WSe2 Wannier cell",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_wannier,
        n=(1,) * n_wannier,
        l=(0,) * n_wannier,
        m=(0,) * n_wannier,
        labels=tuple(f"wannier_{index}" for index in range(n_wannier)),
    )
    return geometry, basis


def test_published_wse2_hr_gamma_x_eigenvalues(tmp_path: Path) -> None:
    """Parse the authenticated input and reproduce frozen Γ/X eigenvalues.

    The case verifies both source bytes and the independent reference payload.

    Notes
    -----
    Compare all 22 bands at both registered fractional momenta.
    """
    compressed: bytes = _COMPRESSED_INPUT.read_bytes()
    reference_payload: bytes = _FROZEN_REFERENCE.read_bytes()
    assert _sha256(compressed) == _COMPRESSED_SHA256
    assert _sha256(reference_payload) == _REFERENCE_SHA256

    source: bytes = lzma.decompress(compressed)
    assert _sha256(source) == _SOURCE_SHA256
    hr_path: Path = tmp_path / "wse2_soc_11bnd_hr.dat"
    hr_path.write_bytes(source)

    reference: Dict[str, Any] = json.loads(reference_payload)
    assert reference["metadata"]["requirement"] == "wannier90-wse2-parity"
    assert reference["metadata"]["source_sha256"] == _SOURCE_SHA256
    n_wannier: int = int(reference["num_wann"])
    geometry: CrystalGeometry
    basis: OrbitalBasis
    geometry, basis = _wannier_context(n_wannier)

    model: TBModel
    operator_data: WannierOperatorData
    model, operator_data = read_wannier90_hr(
        str(hr_path),
        geometry,
        basis,
        jnp.zeros((n_wannier, 3), dtype=jnp.float64),
    )
    assert operator_data.source_format == "hr"
    assert operator_data.position_matrices is None
    assert len(operator_data.cells) == int(reference["num_cells"])

    labels: Tuple[str, ...] = ("Gamma", "X")
    kpoints: Float64[Array, "2 3"] = jnp.asarray(
        [reference["kpoints_fractional"][label] for label in labels],
        dtype=jnp.float64,
    )
    expected: Float64[NDArray, "n_label nband"] = np.asarray(
        [reference["eigenvalues_ev"][label] for label in labels],
        dtype=np.float64,
    )
    actual: Float64[NDArray, "n_label nband"] = np.asarray(
        eigvalsh_bands(model, kpoints)
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-10,
    )
