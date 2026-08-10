r"""Generate the frozen degenerate-k gradient witness fixture.

Extended Summary
----------------
The degenerate-k gradient witness differentiates a gauge-invariant
resolvent intensity at exact band degeneracies: (i) the Dirac point of the
registered graphene test model and (ii) a Kramers-degenerate k-point of
the registered projected-t2g spin--orbit test model.  This generator
registers the two perturbation coordinates the witness test varies and
freezes independent numerical evidence that both witnesses are
analytically sensitive:

- The graphene witness perturbs exactly one nearest-neighbor bond,
  ``t1 -> t1 + theta`` on the registered ``(0, 1)`` pair in cell
  ``(0, 0, 0)``, through the public :func:`tb_parameter_view` coordinate.
  Its ``dH(K)/dtheta`` is nonzero at K even though ``H(K)`` vanishes.
- The common nearest-neighbor scale is measured as the documented negative
  control: at K the assembled ``H(K)`` is identically zero, so the scale
  coordinate has zero ``dH/ds`` and zero gauge-invariant intensity
  sensitivity.  That structural zero is why the witness registry forbids
  it as a witness.
- The Kramers witness perturbs the registered spin-symmetric tetragonal
  crystal-field direction (both ``dxy`` onsite energies) of the
  registered t2g spin--orbit fixture.  Time reversal survives the crystal
  field, so every level stays exactly doubly degenerate while the
  projected intensity keeps a nonzero central-finite-difference
  sensitivity.

The script uses only public DiffPES interfaces plus the registered test
factories, is fully deterministic (fixed constants, no random state), and
writes
``tests/test_diffpes/_reference_data/degenerate_k_gradient_witness.json``.

Routine Listings
----------------
:func:`main`
    Measure both witnesses and write the frozen JSON fixture.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, Tuple
from jaxtyping import Array, Complex128, Float64

DIRAC_K: Tuple[float, float, float] = (2.0 / 3.0, 1.0 / 3.0, 0.0)
KRAMERS_K: Tuple[float, float, float] = (0.173, -0.219, 0.083)
GRAPHENE_HOPPING_EV: float = -2.7
ONE_BOND_VIEW_INDEX: int = 0
SCALE_VIEW_INDICES: Tuple[int, int, int] = (0, 2, 4)
GRAPHENE_SOURCE: Tuple[complex, complex] = (0.6 + 0.35j, -0.45 + 0.8j)
GRAPHENE_OMEGA_EV: float = 0.15
GRAPHENE_ETA_EV: float = 0.05
T2G_COUPLING_EV: float = 0.4
CRYSTAL_FIELD_VIEW_INDICES: Tuple[int, int] = (0, 3)
CRYSTAL_FIELD_CHECK_DELTA_EV: float = 0.3
T2G_SOURCE: Tuple[complex, ...] = (
    0.31 - 0.44j,
    -0.52 + 0.17j,
    0.23 + 0.61j,
    -0.37 - 0.28j,
    0.55 + 0.09j,
    0.12 - 0.73j,
)
T2G_OMEGA_EV: float = 0.1
T2G_GAMMA_EV: float = 0.06
FD_STEPS: Tuple[float, float, float] = (2.0**-12, 2.0**-14, 2.0**-16)


def _sha256(path: Path) -> str:
    """PRIVATE: Return the SHA-256 hex digest of one file.

    Parameters
    ----------
    path : Path
        File whose bytes are hashed.

    Returns
    -------
    digest : str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures() -> Any:
    """PRIVATE: Import the registered fixture factory module.

    Returns
    -------
    factories : Any
        The ``tests._factories`` module with the registered
        ``make_graphene_model`` and ``make_t2g_soc_model`` fixtures.
    """
    root: Path = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests import _factories

    return _factories


def _resolvent_intensity(
    hamiltonian: Complex128[Array, "n_orb n_orb"],
    source: Complex128[Array, " n_orb"],
    omega: float,
    eta: float,
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the gauge-invariant retarded resolvent intensity.

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_orb n_orb"]
        Hermitian Bloch Hamiltonian in eV.
    source : Complex128[Array, " n_orb"]
        Fixed generic complex source vector ``M``.
    omega : float
        Probe frequency in eV.
    eta : float
        Positive retarded regulator in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        ``-Im[M^H ((omega + i eta) I - H)^{-1} M] / pi``.
    """
    dimension: int = hamiltonian.shape[0]
    operator: Complex128[Array, "n_orb n_orb"] = (omega + 1j * eta) * jnp.eye(
        dimension, dtype=jnp.complex128
    ) - hamiltonian
    solved: Complex128[Array, " n_orb"] = jnp.linalg.solve(operator, source)
    return -jnp.imag(jnp.vdot(source, solved)) / jnp.pi


def _projected_intensity(
    hamiltonian: Complex128[Array, "n_orb n_orb"],
    source: Complex128[Array, " n_orb"],
    omega: float,
    gamma: float,
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the mock gauge-invariant projected observable.

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n_orb n_orb"]
        Hermitian Bloch Hamiltonian in eV.
    source : Complex128[Array, " n_orb"]
        Fixed generic complex projection vector ``M``.
    omega : float
        Fixed probe frequency in eV.
    gamma : float
        Lorentzian half-width in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        ``sum_n |<M|n>|^2 (gamma/pi) / ((omega - E_n)^2 + gamma^2)``.

    Notes
    -----
    Summing full degenerate multiplets keeps the observable invariant
    under intra-multiplet basis rotations, so central finite differences
    of this value are well defined at exact degeneracies even though the
    eigenvector autodiff path is not.
    """
    eigenvalues: Float64[Array, " n_orb"]
    eigenvectors: Complex128[Array, "n_orb n_orb"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
    weights: Float64[Array, " n_orb"] = (
        jnp.abs(eigenvectors.conj().T @ source) ** 2
    )
    lorentzian: Float64[Array, " n_orb"] = (gamma / jnp.pi) / (
        (omega - eigenvalues) ** 2 + gamma**2
    )
    return jnp.sum(weights * lorentzian)


def _central_fd_ladder(
    function: Any,
    center: float,
) -> list[float]:
    """PRIVATE: Evaluate the three registered central finite-difference rungs.

    Parameters
    ----------
    function : Any
        Scalar callable of one float coordinate.
    center : float
        Expansion point of the central stencil.

    Returns
    -------
    derivatives : list[float]
        Central differences at steps ``2**-12``, ``2**-14``, ``2**-16``.
    """
    derivatives: list[float] = []
    step: float
    for step in FD_STEPS:
        upper: float = float(function(center + step))
        lower: float = float(function(center - step))
        derivatives.append((upper - lower) / (2.0 * step))
    return derivatives


def _graphene_payload(factories: Any) -> Dict[str, Any]:
    """PRIVATE: Measure the graphene one-bond witness and its negative control.

    Parameters
    ----------
    factories : Any
        Imported ``tests._factories`` module.

    Returns
    -------
    payload : dict[str, Any]
        Frozen graphene witness section of the fixture.
    """
    from diffpes.tightb import (
        bloch_hamiltonian,
        eigvalsh_bands,
        tb_parameter_view,
    )

    model: Any = factories.make_graphene_model(t=GRAPHENE_HOPPING_EV)
    parameters: Float64[Array, " n_par"]
    parameters, rebuild = tb_parameter_view(model)
    dirac_k: Float64[Array, " 3"] = jnp.asarray(DIRAC_K, dtype=jnp.float64)
    source: Complex128[Array, " 2"] = jnp.asarray(
        GRAPHENE_SOURCE, dtype=jnp.complex128
    )
    scale_mask: Float64[Array, " n_par"] = (
        jnp.zeros_like(parameters).at[jnp.asarray(SCALE_VIEW_INDICES)].set(1.0)
    )

    def bond_hamiltonian(
        theta: Float64[Array, ""],
    ) -> Complex128[Array, "2 2"]:
        """Assemble H(K) with one bond perturbed as ``t1 + theta``."""
        perturbed: Float64[Array, " n_par"] = parameters.at[
            ONE_BOND_VIEW_INDEX
        ].add(theta)
        return bloch_hamiltonian(model=rebuild(perturbed), k=dirac_k)

    def scale_hamiltonian(
        scale: Float64[Array, ""],
    ) -> Complex128[Array, "2 2"]:
        """Assemble H(K) with every bond scaled by the common factor."""
        scaled: Float64[Array, " n_par"] = jnp.where(
            scale_mask == 1.0, parameters * scale, parameters
        )
        return bloch_hamiltonian(model=rebuild(scaled), k=dirac_k)

    def bond_intensity(theta: Float64[Array, ""]) -> Float64[Array, ""]:
        """Evaluate the resolvent intensity along the one-bond witness."""
        return _resolvent_intensity(
            bond_hamiltonian(theta),
            source,
            GRAPHENE_OMEGA_EV,
            GRAPHENE_ETA_EV,
        )

    def scale_intensity(scale: Float64[Array, ""]) -> Float64[Array, ""]:
        """Evaluate the resolvent intensity along the forbidden scale."""
        return _resolvent_intensity(
            scale_hamiltonian(scale),
            source,
            GRAPHENE_OMEGA_EV,
            GRAPHENE_ETA_EV,
        )

    bands: Float64[Array, " 2"] = eigvalsh_bands(model, dirac_k[None, :])[0]
    hamiltonian: Complex128[Array, "2 2"] = bloch_hamiltonian(model, dirac_k)
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    one: Float64[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
    bond_jacobian: Complex128[Array, "2 2"] = jax.jacfwd(bond_hamiltonian)(
        zero
    )
    scale_jacobian: Complex128[Array, "2 2"] = jax.jacfwd(scale_hamiltonian)(
        one
    )
    grad_rev: float = float(jax.grad(bond_intensity)(zero))
    grad_fwd: float = float(jax.jacfwd(bond_intensity)(zero))
    fd_bond: list[float] = _central_fd_ladder(bond_intensity, 0.0)
    grad_scale_rev: float = float(jax.grad(scale_intensity)(one))
    fd_scale: list[float] = _central_fd_ladder(scale_intensity, 1.0)

    def omega_zero_intensity(
        theta: Float64[Array, ""],
    ) -> Float64[Array, ""]:
        """Evaluate the documented omega=0 symmetric-zero intensity."""
        return _resolvent_intensity(
            bond_hamiltonian(theta), source, 0.0, GRAPHENE_ETA_EV
        )

    grad_omega_zero: float = float(jax.grad(omega_zero_intensity)(zero))
    return {
        "model": {
            "factory": "tests._factories.make_graphene_model",
            "factory_kwargs": {"t": GRAPHENE_HOPPING_EV},
            "lattice_constant_angstrom": 2.46,
            "lattice_rows_angstrom": [
                [2.46, 0.0, 0.0],
                [1.23, 2.46 * float(np.sqrt(3.0)) / 2.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            "fractional_positions": [
                [0.0, 0.0, 0.0],
                [1.0 / 3.0, 1.0 / 3.0, 0.0],
            ],
            "species": ["C", "C"],
            "basis": {
                "atom_indices": [0, 1],
                "n": [2, 2],
                "l": [1, 1],
                "m": [0, 0],
                "labels": ["A_pz", "B_pz"],
            },
            "hopping_pairs": [[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]],
            "hopping_cells": [
                [0, 0, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            "onsite_energies_ev": [0.0, 0.0],
            "shell_index": [-1, -1],
            "spinor": False,
        },
        "dirac_k_fractional": list(DIRAC_K),
        "parameter_view": {
            "packing": (
                "tb_parameter_view: [Re t(0,1|000), Im t(0,1|000), "
                "Re t(0,1|-100), Im t(0,1|-100), Re t(0,1|0-10), "
                "Im t(0,1|0-10), onsite_A, onsite_B]; reverse hoppings "
                "are conjugate-closed by rebuild"
            ),
            "values": [float(value) for value in np.asarray(parameters)],
        },
        "registered_one_bond_coordinate": {
            "description": (
                "Single nearest-neighbor bond perturbation t1 -> t1 + "
                "theta on pair (0, 1) in cell (0, 0, 0); the other two "
                "bonds stay fixed (Kekule-type one-bond coordinate)"
            ),
            "view_index": ONE_BOND_VIEW_INDEX,
            "theta0": 0.0,
        },
        "forbidden_common_scale_coordinate": {
            "description": (
                "Common nearest-neighbor scale s multiplying every bond "
                "representative; forbidden as the degeneracy witness "
                "because H(K) = 0 makes it a structural zero at K"
            ),
            "view_indices": list(SCALE_VIEW_INDICES),
            "scale0": 1.0,
        },
        "intensity": {
            "definition": (
                "I = -Im[M^H ((omega + i eta) I - H(K))^{-1} M] / pi"
            ),
            "omega_ev": GRAPHENE_OMEGA_EV,
            "eta_ev": GRAPHENE_ETA_EV,
            "source_real": [value.real for value in GRAPHENE_SOURCE],
            "source_imag": [value.imag for value in GRAPHENE_SOURCE],
        },
        "measurements": {
            "bands_at_k_ev": [float(value) for value in np.asarray(bands)],
            "degeneracy_gap_ev": float(bands[1] - bands[0]),
            "hamiltonian_at_k_fro_norm_ev": float(
                jnp.linalg.norm(hamiltonian)
            ),
            "one_bond_dh_dtheta_fro_norm": float(
                jnp.linalg.norm(bond_jacobian)
            ),
            "one_bond_dh_dtheta_offdiag": {
                "real": float(jnp.real(bond_jacobian[0, 1])),
                "imag": float(jnp.imag(bond_jacobian[0, 1])),
            },
            "common_scale_dh_ds_fro_norm": float(
                jnp.linalg.norm(scale_jacobian)
            ),
            "one_bond_grad_reverse": grad_rev,
            "one_bond_grad_forward": grad_fwd,
            "one_bond_fd_central": fd_bond,
            "one_bond_fd_relative_error_vs_grad": [
                abs(value - grad_rev) / abs(grad_rev) for value in fd_bond
            ],
            "common_scale_grad_reverse": grad_scale_rev,
            "common_scale_fd_central": fd_scale,
            "omega_zero_symmetric_zero_grad": grad_omega_zero,
        },
        "notes": (
            "H(K) vanishes identically, so G = I/(omega + i eta) and the "
            "omega = 0 first-order response -Im[M^H dH M / (i eta)^2]/pi "
            "is exactly zero for every Hermitian dH; the registered probe "
            "frequency is therefore off the degenerate energy"
        ),
    }


def _t2g_payload(factories: Any) -> Dict[str, Any]:
    """PRIVATE: Measure the t2g spin--orbit Kramers crystal-field witness.

    Parameters
    ----------
    factories : Any
        Imported ``tests._factories`` module.

    Returns
    -------
    payload : dict[str, Any]
        Frozen Kramers witness section of the fixture.
    """
    from diffpes.tightb import (
        bloch_hamiltonian,
        eigvalsh_bands,
        tb_parameter_view,
    )

    model: Any = factories.make_t2g_soc_model(coupling=T2G_COUPLING_EV)
    parameters: Float64[Array, " n_par"]
    parameters, rebuild = tb_parameter_view(model)
    kramers_k: Float64[Array, " 3"] = jnp.asarray(KRAMERS_K, dtype=jnp.float64)
    source: Complex128[Array, " 6"] = jnp.asarray(
        T2G_SOURCE, dtype=jnp.complex128
    )
    direction: Float64[Array, " n_par"] = (
        jnp.zeros_like(parameters)
        .at[jnp.asarray(CRYSTAL_FIELD_VIEW_INDICES)]
        .set(1.0)
    )

    def field_intensity(delta: Float64[Array, ""]) -> Float64[Array, ""]:
        """Evaluate the projected intensity along the crystal field."""
        candidate: Any = rebuild(parameters + delta * direction)
        return _projected_intensity(
            bloch_hamiltonian(candidate, kramers_k),
            source,
            T2G_OMEGA_EV,
            T2G_GAMMA_EV,
        )

    bands: Float64[Array, " 6"] = eigvalsh_bands(model, kramers_k[None, :])[0]
    pair_gaps: Float64[Array, " 3"] = bands[1::2] - bands[0::2]
    perturbed: Any = rebuild(
        parameters + CRYSTAL_FIELD_CHECK_DELTA_EV * direction
    )
    perturbed_bands: Float64[Array, " 6"] = eigvalsh_bands(
        perturbed, kramers_k[None, :]
    )[0]
    perturbed_gaps: Float64[Array, " 3"] = (
        perturbed_bands[1::2] - perturbed_bands[0::2]
    )
    fd_field: list[float] = _central_fd_ladder(field_intensity, 0.0)
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    eigen_path_grad: float = float(jax.grad(field_intensity)(zero))
    return {
        "model": {
            "factory": "tests._factories.make_t2g_soc_model",
            "factory_kwargs": {"coupling": T2G_COUPLING_EV},
            "lattice_rows_angstrom": [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            "fractional_positions": [[0.0, 0.0, 0.0]],
            "species": ["Ti"],
            "basis": {
                "atom_indices": [0, 0, 0, 0, 0, 0],
                "n": [3, 3, 3, 3, 3, 3],
                "l": [2, 2, 2, 2, 2, 2],
                "m": [-2, -1, 1, -2, -1, 1],
                "spin": [-1, -1, -1, 1, 1, 1],
                "labels": [
                    "dxy_down",
                    "dyz_down",
                    "dxz_down",
                    "dxy_up",
                    "dyz_up",
                    "dxz_up",
                ],
            },
            "onsite_energies_ev": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "soc_lambdas_ev": [T2G_COUPLING_EV],
            "shell_index": [0, 0, 0, 0, 0, 0],
            "spinor": True,
        },
        "kramers_k_fractional": list(KRAMERS_K),
        "parameter_view": {
            "packing": (
                "tb_parameter_view: [onsite dxy_down, dyz_down, dxz_down, "
                "dxy_up, dyz_up, dxz_up, soc_lambda]"
            ),
            "values": [float(value) for value in np.asarray(parameters)],
        },
        "registered_crystal_field_coordinate": {
            "description": (
                "Spin-symmetric tetragonal crystal-field direction: "
                "onsite(dxy_down) and onsite(dxy_up) shift together as "
                "delta; time reversal is preserved, so Kramers pairs stay "
                "exactly degenerate at every delta"
            ),
            "view_indices": list(CRYSTAL_FIELD_VIEW_INDICES),
            "direction": [float(value) for value in np.asarray(direction)],
            "delta0": 0.0,
        },
        "intensity": {
            "definition": (
                "I = sum_n |<M|n>|^2 (gamma/pi) / ((omega - E_n)^2 + gamma^2)"
            ),
            "omega_ev": T2G_OMEGA_EV,
            "gamma_ev": T2G_GAMMA_EV,
            "source_real": [value.real for value in T2G_SOURCE],
            "source_imag": [value.imag for value in T2G_SOURCE],
        },
        "measurements": {
            "bands_at_k_ev": [float(value) for value in np.asarray(bands)],
            "kramers_pair_gaps_ev": [
                float(value) for value in np.asarray(pair_gaps)
            ],
            "max_kramers_pair_gap_ev": float(jnp.max(jnp.abs(pair_gaps))),
            "perturbed_delta_ev": CRYSTAL_FIELD_CHECK_DELTA_EV,
            "perturbed_bands_ev": [
                float(value) for value in np.asarray(perturbed_bands)
            ],
            "perturbed_kramers_pair_gaps_ev": [
                float(value) for value in np.asarray(perturbed_gaps)
            ],
            "crystal_field_fd_central": fd_field,
            "eigen_path_grad_at_degeneracy": eigen_path_grad,
            "eigen_path_grad_minus_fd": [
                eigen_path_grad - value for value in fd_field
            ],
        },
        "notes": (
            "The atomic model is k-independent, so the registered generic "
            "k inherits the exact j_eff = 3/2 quartet at -lambda/2 and "
            "j_eff = 1/2 doublet at +lambda; the eigen autodiff path "
            "returns a finite but wrong derivative at this degeneracy. "
            "That route is outside the declared nondegenerate domain, so "
            "no eigen-path derivative claim is made; finite-difference "
            "truth comes from the invariant multiplet-summed observable "
            "instead"
        ),
    }


def main() -> None:
    """Measure both witnesses and write the frozen JSON fixture."""
    factories: Any = _fixtures()
    import diffpes

    root: Path = Path(__file__).resolve().parents[2]
    data_directory: Path = root / "tests" / "test_diffpes" / "_reference_data"
    data_directory.mkdir(parents=True, exist_ok=True)
    fixture_path: Path = data_directory / "degenerate_k_gradient_witness.json"
    payload: Dict[str, Any] = {
        "fixture": "degenerate_k_gradient_witness",
        "requirement": "degenerate-k-gradient-witness",
        "witness_registry": (
            "The one-bond graphene hopping with nonzero dH/dtheta at K is "
            "the registered witness; the common nearest-neighbor scale is "
            "forbidden; the Kramers witness uses the registered t2g "
            "spin-orbit test model and a registered crystal-field "
            "direction with nonzero projected-intensity sensitivity"
        ),
        "fd_steps": list(FD_STEPS),
        "graphene_one_bond_witness": _graphene_payload(factories),
        "t2g_soc_kramers_witness": _t2g_payload(factories),
        "environment": {
            "diffpes": diffpes.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "generator": (
            "tests/_reference_tools/generate_degenerate_k_gradient_witness.py"
        ),
    }
    fixture_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {fixture_path}")  # noqa: T201
    print(f"sha256 {_sha256(fixture_path)}")  # noqa: T201


if __name__ == "__main__":
    main()
