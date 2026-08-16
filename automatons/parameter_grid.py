# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["diffpes==2026.06.13"]
# ///
"""Evaluate a temperature-linewidth observable grid with JAX batching.

The automaton evaluates a compact occupied spectral proxy over temperature and
linewidth coordinates. It compares an ordinary JAX vectorization with the
public host-device sharding map when more than one device is present. It writes
a heatmap, numerical grid, table, and metrics record.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from jaxtyping import Array, Float64, jaxtyped

import diffpes as dp


def _positive_ladder(values: List[float], name: str) -> Tuple[float, ...]:
    """PRIVATE: Validate one nonempty positive scalar ladder.

    Parameters
    ----------
    values : List[float]
        Requested scalar coordinate values.
    name : str
        Parameter name included in an input error.

    Returns
    -------
    ladder : Tuple[float, ...]
        Positive finite values in the requested order.

    Raises
    ------
    ValueError
        If a ladder is empty or contains a nonpositive value.

    Notes
    -----
    The host check keeps invalid coordinates outside compiled JAX arithmetic.
    """
    ladder: Tuple[float, ...] = tuple(float(value) for value in values)
    if not ladder:
        message: str = f"{name} must not be empty"
        raise ValueError(message)
    if any(value <= 0.0 for value in ladder):
        message = f"{name} values must be positive"
        raise ValueError(message)
    return ladder


@jaxtyped(typechecker=beartype)
def _fermi_intensity(
    temperature: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate one occupied Fermi-level spectral proxy.

    Parameters
    ----------
    temperature : Float64[Array, ""]
        Finite sample temperature in kelvin.
    gamma : Float64[Array, ""]
        Positive Gaussian linewidth in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        Occupied intensity at the Fermi level in inverse eV.

    Notes
    -----
    The proxy evaluates a public Gaussian line at the Fermi level. A public
    Fermi-Dirac occupation weights the occupied feature centred below it.
    """
    sampled_energy: Float64[Array, " 1"] = jnp.asarray(
        (0.0,),
        dtype=jnp.float64,
    )
    line: Float64[Array, " 1"] = dp.simul.gaussian(
        sampled_energy,
        -0.05,
        gamma,
    )
    occupation: Float64[Array, ""] = dp.simul.fermi_dirac(
        -0.05,
        0.0,
        temperature,
    )
    intensity: Float64[Array, ""] = line[0] * occupation
    return intensity


@jaxtyped(typechecker=beartype)
def _shard_safe_fermi_intensity(
    temperature: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate a validated occupied proxy within a manual mesh axis.

    Parameters
    ----------
    temperature : Float64[Array, ""]
        Positive sample temperature in kelvin.
    gamma : Float64[Array, ""]
        Positive Gaussian linewidth in eV.

    Returns
    -------
    intensity : Float64[Array, ""]
        The same Fermi-level proxy as :func:`_fermi_intensity`.

    Notes
    -----
    The scalar coordinates have already passed host-side validation.  This
    elementary expression mirrors the public Gaussian and Fermi formulae but
    avoids Equinox's dynamic error branch, which cannot carry a varying manual
    mesh axis in the installed JAX version.
    """
    line: Float64[Array, ""] = jnp.exp(
        -((jnp.asarray(0.0, dtype=jnp.float64) + 0.05) ** 2) / (2.0 * gamma**2)
    ) / (jnp.sqrt(2.0 * jnp.pi) * gamma)
    thermal_energy: Float64[Array, ""] = (
        jnp.asarray(
            dp.constants.KB_EV_PER_K,
            dtype=jnp.float64,
        )
        * temperature
    )
    occupation: Float64[Array, ""] = jax.nn.sigmoid(0.05 / thermal_energy)
    intensity: Float64[Array, ""] = line * occupation
    return intensity


@jax.jit
@jaxtyped(typechecker=beartype)
def _fermi_grid(
    temperatures: Float64[Array, " n_temperature"],
    gammas: Float64[Array, " n_gamma"],
) -> Float64[Array, "n_temperature n_gamma"]:
    """PRIVATE: Vectorize the occupied proxy across a scalar grid.

    Parameters
    ----------
    temperatures : Float64[Array, " n_temperature"]
        Temperature ladder in kelvin.
    gammas : Float64[Array, " n_gamma"]
        Linewidth ladder in eV.

    Returns
    -------
    grid : Float64[Array, "n_temperature n_gamma"]
        JIT-compiled grid of occupied proxy values.

    Notes
    -----
    Nested ``jax.vmap`` calls preserve one scalar physical expression while
    JIT compiles the complete rectangular grid.
    """

    def row(
        temperature: Float64[Array, ""],
    ) -> Float64[Array, " n_gamma"]:
        """PRIVATE: Evaluate every linewidth at one temperature.

        Parameters
        ----------
        temperature : Float64[Array, ""]
            One temperature value in kelvin.

        Returns
        -------
        values : Float64[Array, " n_gamma"]
            Occupied proxy values for every linewidth.

        Notes
        -----
        Vectorization keeps the scalar intensity expression unchanged.
        """
        values: Float64[Array, " n_gamma"] = jax.vmap(
            lambda gamma: _fermi_intensity(temperature, gamma)
        )(gammas)
        return values

    grid: Float64[Array, "n_temperature n_gamma"] = jax.vmap(row)(temperatures)
    return grid  # noqa: RET504 -- preserve the named-array return contract.


@jax.jit
@jaxtyped(typechecker=beartype)
def _mdc_fwhm_grid(
    temperatures: Float64[Array, " n_temperature"],
    gammas: Float64[Array, " n_gamma"],
) -> Float64[Array, "n_temperature n_gamma"]:
    """PRIVATE: Vectorize an intrinsic MDC linewidth grid.

    Parameters
    ----------
    temperatures : Float64[Array, " n_temperature"]
        Temperature ladder in kelvin.
    gammas : Float64[Array, " n_gamma"]
        Lorentzian half-width ladder in eV.

    Returns
    -------
    grid : Float64[Array, "n_temperature n_gamma"]
        Full-width values in the unit-velocity momentum convention.

    Notes
    -----
    The public thermal occupation is evaluated during the vectorization. Its
    multiplication by one keeps the thermal domain validation in this path.
    """

    def row(
        temperature: Float64[Array, ""],
    ) -> Float64[Array, " n_gamma"]:
        """PRIVATE: Evaluate every linewidth at one temperature.

        Parameters
        ----------
        temperature : Float64[Array, ""]
            One temperature value in kelvin.

        Returns
        -------
        values : Float64[Array, " n_gamma"]
            Full widths for every linewidth value.

        Notes
        -----
        The occupation evaluation validates temperature before the width map.
        """

        def width(
            gamma: Float64[Array, ""],
        ) -> Float64[Array, ""]:
            """PRIVATE: Convert one Lorentzian half-width to an MDC width.

            Parameters
            ----------
            gamma : Float64[Array, ""]
                Lorentzian half-width in eV.

            Returns
            -------
            value : Float64[Array, ""]
                Full width in inverse Angstrom under unit velocity.

            Notes
            -----
            A unit thermal factor calls the public finite-temperature surface.
            """
            occupancy: Float64[Array, ""] = dp.simul.fermi_dirac(
                -0.05,
                0.0,
                temperature,
            )
            value: Float64[Array, ""] = 2.0 * gamma * (occupancy / occupancy)
            return value

        values: Float64[Array, " n_gamma"] = jax.vmap(width)(gammas)
        return values

    grid: Float64[Array, "n_temperature n_gamma"] = jax.vmap(row)(temperatures)
    return grid


@jaxtyped(typechecker=beartype)
def _grid_values(
    temperatures: Float64[Array, " n_temperature"],
    gammas: Float64[Array, " n_gamma"],
    observable: str,
) -> Float64[Array, "n_temperature n_gamma"]:
    """PRIVATE: Select one JIT-vectorized scalar observable grid.

    Parameters
    ----------
    temperatures : Float64[Array, " n_temperature"]
        Temperature ladder in kelvin.
    gammas : Float64[Array, " n_gamma"]
        Linewidth ladder in eV.
    observable : str
        Selected observable identifier.

    Returns
    -------
    grid : Float64[Array, "n_temperature n_gamma"]
        Requested ordinary vectorized grid.

    Raises
    ------
    ValueError
        If the observable identifier is unsupported.

    Notes
    -----
    Branching occurs before tracing because the identifier selects a static
    physical observable, not a numerical coordinate.
    """
    if observable == "fermi_intensity":
        grid: Float64[Array, "n_temperature n_gamma"] = _fermi_grid(
            temperatures,
            gammas,
        )
    elif observable == "mdc_fwhm":
        grid = _mdc_fwhm_grid(temperatures, gammas)
    else:
        message: str = "observable must be fermi_intensity or mdc_fwhm"
        raise ValueError(message)
    return grid


@jaxtyped(typechecker=beartype)
def _sharded_grid_values(
    temperatures: Float64[Array, " n_temperature"],
    gammas: Float64[Array, " n_gamma"],
    observable: str,
) -> Float64[Array, "n_temperature n_gamma"]:
    """PRIVATE: Evaluate one grid through the public host-device map.

    Parameters
    ----------
    temperatures : Float64[Array, " n_temperature"]
        Temperature ladder in kelvin.
    gammas : Float64[Array, " n_gamma"]
        Linewidth ladder in eV.
    observable : str
        Selected observable identifier.

    Returns
    -------
    grid : Float64[Array, "n_temperature n_gamma"]
        Grid assembled from the public sharded momentum map.

    Notes
    -----
    Parameter pairs occupy the first two coordinates of an auxiliary k-map.
    Padding weights remove unused capacity before the map returns values.
    """
    n_temperature: int = temperatures.shape[0]
    n_gamma: int = gammas.shape[0]
    n_values: int = n_temperature * n_gamma
    n_devices: int = jax.device_count()
    if n_devices == 1:
        grid: Float64[Array, "n_temperature n_gamma"] = _grid_values(
            temperatures,
            gammas,
            observable,
        )
        return grid
    capacity: int = ((n_values + n_devices - 1) // n_devices) * n_devices
    temperature_mesh: Float64[Array, "n_temperature n_gamma"]
    gamma_mesh: Float64[Array, "n_temperature n_gamma"]
    temperature_mesh, gamma_mesh = jnp.meshgrid(
        temperatures,
        gammas,
        indexing="ij",
    )
    flat_temperatures: Float64[Array, " n_values"] = jnp.ravel(
        temperature_mesh
    )
    flat_gammas: Float64[Array, " n_values"] = jnp.ravel(gamma_mesh)
    padding: int = capacity - n_values
    padded_temperatures: Float64[Array, " nk_max"] = jnp.pad(
        flat_temperatures,
        (0, padding),
    )
    padded_gammas: Float64[Array, " nk_max"] = jnp.pad(
        flat_gammas,
        (0, padding),
        constant_values=gammas[-1],
    )
    points: Float64[Array, "nk_max 3"] = jnp.stack(
        (
            padded_temperatures,
            padded_gammas,
            jnp.zeros((capacity,), dtype=jnp.float64),
        ),
        axis=1,
    )
    weights: Float64[Array, " nk_max"] = jnp.concatenate(
        (
            jnp.ones((n_values,), dtype=jnp.float64),
            jnp.zeros((padding,), dtype=jnp.float64),
        )
    )
    specification: dp.types.ShardSpec = dp.types.make_shard_spec(
        n_devices=n_devices,
        chunk_size=1,
        nk_max=capacity,
    )

    def body(
        local_points: Float64[Array, "chunk 3"],
        local_weights: Float64[Array, " chunk"],
    ) -> Float64[Array, "chunk 1"]:
        """PRIVATE: Evaluate and mask one parameter chunk.

        Parameters
        ----------
        local_points : Float64[Array, "chunk 3"]
            Packed temperature and linewidth coordinates.
        local_weights : Float64[Array, " chunk"]
            Physical-lane mask values.

        Returns
        -------
        values : Float64[Array, "chunk 1"]
            One masked scalar observable per packed coordinate.

        Notes
        -----
        The lane mask preserves the public sharded-map padding contract.
        """
        local_temperatures: Float64[Array, " chunk"] = local_points[:, 0]
        local_gammas: Float64[Array, " chunk"] = local_points[:, 1]
        if observable == "fermi_intensity":
            raw_values: Float64[Array, " chunk"] = jax.vmap(
                _shard_safe_fermi_intensity
            )(local_temperatures, local_gammas)
        else:
            raw_values = 2.0 * local_gammas
        values: Float64[Array, "chunk 1"] = (raw_values * local_weights)[
            :, None
        ]
        return values

    mapped: Float64[Array, "nk_max 1"] = dp.simul.sharded_kmap(
        body,
        points,
        weights,
        specification,
    )
    host_mapped: Float64[Array, "nk_max 1"] = jnp.asarray(
        jax.device_get(mapped),
        dtype=jnp.float64,
    )
    grid: Float64[Array, "n_temperature n_gamma"] = jnp.reshape(
        host_mapped[:n_values, 0],
        (n_temperature, n_gamma),
    )
    return grid  # noqa: RET504 -- preserve the named-array return contract.


@dp.harness.experiment(
    name="parameter-grid",
    params=(
        dp.types.make_automaton_param(
            "temperature_ladder",
            list,
            default=[20.0, 60.0, 120.0],
            help="Sample temperatures in kelvin.",
            example=[20.0, 60.0, 120.0],
        ),
        dp.types.make_automaton_param(
            "gamma_ladder",
            list,
            default=[0.02, 0.05, 0.1],
            help="Intrinsic linewidth values in eV.",
            example=[0.02, 0.05, 0.1],
        ),
        dp.types.make_automaton_param(
            "observable",
            str,
            default="fermi_intensity",
            help="Scalar observable for the parameter grid.",
            choices=("fermi_intensity", "mdc_fwhm"),
            example="fermi_intensity",
        ),
    ),
    returns={
        "metrics": {
            "n_grid_points": {"type": "integer"},
            "best_point": {"type": "object"},
            "grid_shape": {"type": "array"},
        },
        "artifacts": {
            "roles": ["grid_heatmap", "grid_arrays", "grid_table", "metrics"]
        },
    },
)
def main(
    args: SimpleNamespace,
    ctx: dp.types.AutomatonContext,
) -> Dict[str, Any]:
    """Evaluate a batched parameter grid and return sharding diagnostics.

    The body compares ordinary JAX vectorization with public device sharding.
    It writes a heatmap and records the maximum agreement error.
    """
    temperature_ladder: Tuple[float, ...] = _positive_ladder(
        list(args.temperature_ladder),
        "temperature_ladder",
    )
    gamma_ladder: Tuple[float, ...] = _positive_ladder(
        list(args.gamma_ladder),
        "gamma_ladder",
    )
    if args.smoke:
        temperature_ladder = temperature_ladder[:3]
        gamma_ladder = gamma_ladder[:3]
    temperatures: Float64[Array, " n_temperature"] = jnp.asarray(
        temperature_ladder,
        dtype=jnp.float64,
    )
    gammas: Float64[Array, " n_gamma"] = jnp.asarray(
        gamma_ladder,
        dtype=jnp.float64,
    )
    ordinary_grid: Float64[Array, "n_temperature n_gamma"] = _grid_values(
        temperatures,
        gammas,
        args.observable,
    )
    sharded_grid: Float64[Array, "n_temperature n_gamma"] = (
        _sharded_grid_values(
            temperatures,
            gammas,
            args.observable,
        )
    )
    agreement_error: Float64[Array, ""] = jnp.max(
        jnp.abs(ordinary_grid - sharded_grid)
    )
    best_flat_index: int = int(jnp.argmax(ordinary_grid))
    best_temperature_index: int = best_flat_index // gammas.shape[0]
    best_gamma_index: int = best_flat_index % gammas.shape[0]
    figure: Any
    figure, _, _ = dp.plots.plot_momentum_map(
        ordinary_grid,
        temperatures,
        gammas,
        aspect="auto",
        xlabel="temperature (K)",
        ylabel="linewidth (eV)",
        title=f"{args.observable} parameter grid",
    )
    table: List[Dict[str, float]] = [
        {
            "temperature_k": float(temperatures[row]),
            "gamma_ev": float(gammas[column]),
            "value": float(ordinary_grid[row, column]),
        }
        for row in range(temperatures.shape[0])
        for column in range(gammas.shape[0])
    ]
    metrics: Dict[str, Any] = {
        "n_grid_points": int(ordinary_grid.size),
        "best_point": {
            "temperature_k": float(temperatures[best_temperature_index]),
            "gamma_ev": float(gammas[best_gamma_index]),
            "value": float(
                ordinary_grid[best_temperature_index, best_gamma_index]
            ),
        },
        "grid_shape": list(ordinary_grid.shape),
        "device_count": jax.device_count(),
        "sharded_max_abs_error": float(agreement_error),
    }
    artifacts: List[dp.types.ArtifactRecord] = [
        dp.harness.save_figure_artifact(
            ctx,
            "grid_heatmap.png",
            figure,
            role="grid_heatmap",
        ),
        dp.harness.save_array_artifact(
            ctx,
            "grid.npz",
            {
                "temperatures": temperatures,
                "gammas": gammas,
                "ordinary_grid": ordinary_grid,
                "sharded_grid": sharded_grid,
            },
            role="grid_arrays",
        ),
        dp.harness.save_json_artifact(
            ctx,
            "grid_table.json",
            table,
            role="grid_table",
        ),
        dp.harness.save_json_artifact(ctx, "metrics.json", metrics),
    ]
    result: Dict[str, Any] = {"metrics": metrics, "artifacts": artifacts}
    return result


if __name__ == "__main__":
    main()
