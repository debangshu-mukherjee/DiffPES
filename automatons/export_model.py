# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Export a symbolic intrinsic spectral cut and compare two dynamic sizes.

The automaton sets Equinox error behavior before importing DiffPES. It exports
a symbolic band-centre and energy-axis forward calculation through JAX export.
It records StableHLO bytes and compares exported calls with the in-process
calculation for two shapes. Smoke mode keeps both shapes compact.
"""

from __future__ import annotations

import os

os.environ["EQX_ON_ERROR"] = "nan"

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


@jaxtyped(typechecker=beartype)
def _spectral_cut(
    band_centers: Float64[Array, " n_band"],
    energy_axis: Float64[Array, " n_energy"],
) -> Float64[Array, " n_energy"]:
    """PRIVATE: Assemble one compact occupied intrinsic spectral cut.

    Parameters
    ----------
    band_centers : Float64[Array, " n_band"]
        Band-centre energies in eV.
    energy_axis : Float64[Array, " n_energy"]
        Relative-energy samples in eV.

    Returns
    -------
    intensity : Float64[Array, " n_energy"]
        Summed occupied Gaussian spectral intensity in inverse eV.

    Notes
    -----
    The function uses the public Gaussian and Fermi-Dirac expressions. It
    keeps all band contributions until the final physically required sum.
    """
    profiles: Float64[Array, "n_band n_energy"] = jax.vmap(
        lambda center: dp.simul.gaussian(energy_axis, center, 0.05)
    )(band_centers)
    occupation: Float64[Array, " n_energy"] = jax.vmap(
        lambda energy: dp.simul.fermi_dirac(energy, 0.0, 30.0)
    )(energy_axis)
    intensity: Float64[Array, " n_energy"] = jnp.sum(
        profiles * occupation[None, :],
        axis=0,
    )
    return intensity


@jaxtyped(typechecker=beartype)
def _sample_inputs(
    n_bands: int,
    n_energy: int,
) -> Tuple[Float64[Array, " n_band"], Float64[Array, " n_energy"]]:
    """PRIVATE: Build deterministic band and energy inputs for one shape.

    Parameters
    ----------
    n_bands : int
        Number of compact band centres.
    n_energy : int
        Number of relative-energy samples.

    Returns
    -------
    inputs : Tuple[Float64[Array, " n_band"], Float64[Array, " n_energy"]]
        Band-centre and relative-energy arrays in eV.

    Notes
    -----
    The two linearly spaced arrays define a bounded intrinsic spectral cut.
    """
    band_centers: Float64[Array, " n_band"] = jnp.linspace(
        -0.32,
        -0.08,
        n_bands,
        dtype=jnp.float64,
    )
    energy_axis: Float64[Array, " n_energy"] = jnp.linspace(
        -0.5,
        0.1,
        n_energy,
        dtype=jnp.float64,
    )
    inputs: Tuple[Float64[Array, " n_band"], Float64[Array, " n_energy"]] = (
        band_centers,
        energy_axis,
    )
    return inputs


def _subprocess_program(portable: bool) -> str:
    """PRIVATE: Build the isolated artifact inspection program text.

    Parameters
    ----------
    portable : bool
        Whether JAX export serialization is available in this environment.

    Returns
    -------
    program : str
        Source text for an isolated virtual-environment interpreter.

    Notes
    -----
    The portable branch deserializes the JAX export payload. The fallback only
    checks StableHLO bytes because this environment lacks the serializer extra.
    """
    lines: Tuple[str, ...]
    if portable:
        lines = (
            "import os",
            "os.environ['EQX_ON_ERROR'] = 'nan'",
            "import diffpes as dp",
            "import jax",
            "import jax.numpy as jnp",
            "import pathlib",
            "import sys",
            "payload = pathlib.Path(sys.argv[1]).read_bytes()",
            "exported = jax.export.deserialize(payload)",
            "n_bands = int(sys.argv[2])",
            "n_energy = int(sys.argv[3])",
            "centers = jnp.linspace(-0.32, -0.08, n_bands, dtype=jnp.float64)",
            "energy = jnp.linspace(-0.5, 0.1, n_energy, dtype=jnp.float64)",
            "profiles = jax.vmap("
            "lambda center: dp.simul.gaussian(energy, center, 0.05)"
            ")(centers)",
            "occupation = jax.vmap("
            "lambda value: dp.simul.fermi_dirac(value, 0.0, 30.0)"
            ")(energy)",
            "reference = jnp.sum(profiles * occupation[None, :], axis=0)",
            "error = float(jnp.max(jnp.abs("
            "exported.call(centers, energy) - reference)))",
            "print(error)",
        )
    else:
        lines = (
            "import hashlib",
            "import pathlib",
            "import sys",
            "payload = pathlib.Path(sys.argv[1]).read_bytes()",
            "print(hashlib.sha256(payload).hexdigest())",
        )
    program: str = "\n".join(lines)
    return program


@dp.harness.experiment(
    name="export-model",
    params=(
        dp.types.make_automaton_param(
            "n_bands",
            int,
            default=2,
            help="Number of symbolic compact band centres.",
            bounds=(1.0, 8.0),
            example=2,
        ),
        dp.types.make_automaton_param(
            "n_energy",
            int,
            default=12,
            help="Number of symbolic relative-energy samples.",
            bounds=(4.0, 128.0),
            example=12,
        ),
    ),
    returns={
        "metrics": {
            "artifact_bytes": {"type": "integer"},
            "separate_process_ok": {"type": "boolean"},
            "same_result_max_abs_error": {"type": "number"},
            "sizes_verified": {"type": "array"},
        },
        "artifacts": {
            "roles": [
                "stablehlo_artifact",
                "export_manifest",
                "export_arrays",
                "metrics",
            ]
        },
    },
)
def main(  # noqa: PLR0915
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Export a symbolic spectral cut and return cross-shape comparisons.

    The body exports a JIT expression with two symbolic physical sizes.
    It writes StableHLO bytes and attempts portable serializer validation.
    """
    n_bands: int = min(args.n_bands, 2) if args.smoke else args.n_bands
    n_energy: int = min(args.n_energy, 12) if args.smoke else args.n_energy
    scope: Any = jax.export.SymbolicScope()
    symbolic_dimensions: Tuple[Any, Any] = jax.export.symbolic_shape(
        "n_bands, n_energy",
        scope=scope,
    )
    band_specification: Any = jax.ShapeDtypeStruct(
        (symbolic_dimensions[0],),
        jnp.float64,
    )
    energy_specification: Any = jax.ShapeDtypeStruct(
        (symbolic_dimensions[1],),
        jnp.float64,
    )
    exported: Any = jax.export.export(jax.jit(_spectral_cut))(
        band_specification,
        energy_specification,
    )
    portable: bool = True
    serialization_error: str = ""
    try:
        artifact_bytes: bytes = exported.serialize()
        callable_export: Any = jax.export.deserialize(artifact_bytes)
    except ImportError as error:
        portable = False
        serialization_error = str(error)
        artifact_bytes = exported.mlir_module_serialized
        callable_export = exported
    stablehlo_path: Path = dp.harness.artifact_path(
        ctx,
        "spectral_cut.stablehlo",
    )
    stablehlo_path.write_bytes(artifact_bytes)
    first_bands: Float64[Array, " n_band"]
    first_energy: Float64[Array, " n_energy"]
    first_bands, first_energy = _sample_inputs(n_bands, n_energy)
    second_band_count: int = n_bands + 1
    second_energy_count: int = n_energy + 3
    second_bands: Float64[Array, " n_band"]
    second_energy: Float64[Array, " n_energy"]
    second_bands, second_energy = _sample_inputs(
        second_band_count,
        second_energy_count,
    )
    first_reference: Float64[Array, " n_energy"] = _spectral_cut(
        first_bands,
        first_energy,
    )
    second_reference: Float64[Array, " n_energy"] = _spectral_cut(
        second_bands,
        second_energy,
    )
    first_exported: Float64[Array, " n_energy"] = callable_export.call(
        first_bands,
        first_energy,
    )
    second_exported: Float64[Array, " n_energy"] = callable_export.call(
        second_bands,
        second_energy,
    )
    first_error: Float64[Array, ""] = jnp.max(
        jnp.abs(first_reference - first_exported)
    )
    second_error: Float64[Array, ""] = jnp.max(
        jnp.abs(second_reference - second_exported)
    )
    maximum_error: Float64[Array, ""] = jnp.maximum(first_error, second_error)
    environment: Dict[str, str] = os.environ.copy()
    subprocess_result: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            _subprocess_program(portable),
            str(stablehlo_path),
            str(second_band_count),
            str(second_energy_count),
        ],
        capture_output=True,
        check=False,
        env={
            **environment,
            "JAX_PLATFORMS": "cpu",
            "MPLCONFIGDIR": "/tmp/dp-mpl",  # noqa: S108
        },
        text=True,
    )
    separate_process_error: float = float("inf")
    subprocess_digest: str = ""
    if portable and subprocess_result.returncode == 0:
        separate_process_error = float(subprocess_result.stdout.strip())
    elif not portable and subprocess_result.returncode == 0:
        subprocess_digest = subprocess_result.stdout.strip()
    stablehlo_sha256: str = hashlib.sha256(artifact_bytes).hexdigest()
    separate_process_tolerance: float = 1.0e-10
    separate_process_ok: bool = (
        portable
        and subprocess_result.returncode == 0
        and separate_process_error < separate_process_tolerance
    )
    sizes_verified: List[List[int]] = [
        [n_bands, n_energy],
        [second_band_count, second_energy_count],
    ]
    export_manifest: Dict[str, Any] = {
        "portable_serialization": portable,
        "serialization_error": serialization_error,
        "stablehlo_sha256": stablehlo_sha256,
        "subprocess_returncode": subprocess_result.returncode,
        "subprocess_stablehlo_sha256": subprocess_digest,
        "subprocess_max_abs_error": separate_process_error
        if portable
        else None,
        "sizes_verified": sizes_verified,
    }
    metrics: Dict[str, Any] = {
        "artifact_bytes": len(artifact_bytes),
        "separate_process_ok": separate_process_ok,
        "same_result_max_abs_error": float(maximum_error),
        "sizes_verified": sizes_verified,
        "portable_serialization": portable,
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.record_artifact(
            ctx,
            stablehlo_path,
            role="stablehlo_artifact",
            mime="application/vnd.stablehlo",
            preview=True,
        ),
        dp.harness.save_json_artifact(
            ctx,
            "export_manifest.json",
            export_manifest,
            role="export_manifest",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "export_arrays.npz",
            {
                "first_reference": first_reference,
                "first_exported": first_exported,
                "second_reference": second_reference,
                "second_exported": second_exported,
            },
            role="export_arrays",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
