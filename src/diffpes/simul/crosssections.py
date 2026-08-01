"""Interpolate authenticated Yeh--Lindau photoionization cross sections.

Extended Summary
----------------
This module exposes the element- and subshell-resolved Yeh--Lindau (1985)
tables in megabarn. The generator derives the packaged data from the CC BY 4.0
Regoutz-group digitisation. Interpolation uses shape-preserving cubic Hermite
in log cross section versus log photon energy. Missing and published-zero
entries remain missing. The interpolator never adds a floor or extrapolates.

Routine Listings
----------------
:func:`yeh_lindau_cross_section_table`
    Return one raw Yeh--Lindau subshell row.
:func:`yeh_lindau_cross_section`
    Interpolate an atomic subshell photoionization cross section.
:func:`yeh_lindau_orbital_weights`
    Return Yeh--Lindau weights for every orbital in a basis.

Notes
-----
The tables describe isolated-atom cross sections.  They provide the grounded
weighting for the deliberately incoherent ``simulate_basic`` tier; coherent
matrix-element calculations use :mod:`diffpes.simul.matrixel` instead.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Bool, Float64, Int16, Int32, jaxtyped
from numpy.typing import NDArray

from diffpes.types import OrbitalBasis, ScalarFloat


@cache
def _load_data() -> tuple[
    Int32[NDArray, " n_row_plus_one"],
    Float64[NDArray, " n_packed"],
    Float64[NDArray, " n_packed"],
    Float64[NDArray, " n_packed"],
    dict[tuple[int, int, int], int],
]:
    """Load and cache the immutable packed table arrays."""
    data_resource: Traversable = resources.files("diffpes.simul").joinpath(
        "data",
        "yeh_lindau_1985.npz",
    )
    archive: np.lib.npyio.NpzFile
    with np.load(data_resource) as archive:
        keys: Int16[NDArray, "n_row 3"] = np.asarray(
            archive["keys"], dtype=np.int16
        )
        offsets: Int32[NDArray, " n_row_plus_one"] = np.asarray(
            archive["offsets"], dtype=np.int32
        )
        energy_nodes: Float64[NDArray, " n_packed"] = np.asarray(
            archive["photon_energy_ev"],
            dtype=np.float64,
        )
        sigma_nodes: Float64[NDArray, " n_packed"] = np.asarray(
            archive["sigma_megabarn"],
            dtype=np.float64,
        )
        log_slopes: Float64[NDArray, " n_packed"] = np.asarray(
            archive["log_slopes"],
            dtype=np.float64,
        )
    key_to_row: dict[tuple[int, int, int], int] = {
        tuple(int(value) for value in key): index
        for index, key in enumerate(keys)
    }
    loaded: tuple[
        Int32[NDArray, " n_row_plus_one"],
        Float64[NDArray, " n_packed"],
        Float64[NDArray, " n_packed"],
        Float64[NDArray, " n_packed"],
        dict[tuple[int, int, int], int],
    ] = (
        offsets,
        energy_nodes,
        sigma_nodes,
        log_slopes,
        key_to_row,
    )
    return loaded


def _table_slice(
    atomic_number: int,
    principal_quantum_number: int,
    angular_momentum: int,
) -> slice:
    """Resolve a static subshell key to its packed-data slice."""
    key: tuple[int, int, int] = (
        atomic_number,
        principal_quantum_number,
        angular_momentum,
    )
    offsets: Int32[NDArray, " n_row_plus_one"]
    key_to_row: dict[tuple[int, int, int], int]
    offsets, _, _, _, key_to_row = _load_data()
    if key not in key_to_row:
        message: str = (
            "unsupported Yeh--Lindau subshell "
            f"(Z, n, l)={key}; consult yeh_lindau_1985.json"
        )
        raise ValueError(message)
    row: int = key_to_row[key]
    table_slice: slice = slice(
        int(offsets[row]),
        int(offsets[row + 1]),
    )
    return table_slice


@jaxtyped(typechecker=beartype)
def yeh_lindau_cross_section_table(  # noqa: DOC502
    atomic_number: int,
    n: int,
    l: int,  # noqa: E741
) -> tuple[
    Float64[NDArray, " node"],
    Float64[NDArray, " node"],
    Float64[NDArray, " node"],
]:
    """Return one raw Yeh--Lindau subshell row.

    The accessor returns missing entries as ``NaN`` and preserves published
    zeros.
    A finite ``log_slopes`` entry marks a node belonging to a positive
    interpolation interval.

    :see: :class:`~.test_crosssections.TestYehLindauCrossSectionTable`

    Notes
    -----
    Resolve the static ``(Z,n,l)`` key into the packed archive. Copy the raw
    energy, cross-section, and slope rows so callers cannot mutate the cache.

    Parameters
    ----------
    atomic_number : int
        Nuclear charge ``Z`` in the tabulation.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum, with ``0=s``, ``1=p``, ``2=d``, and
        ``3=f``.

    Returns
    -------
    photon_energy_ev : Float64[NDArray, " node"]
        Published photon-energy nodes in eV.
    sigma_megabarn : Float64[NDArray, " node"]
        Published cross sections in megabarn, preserving missing values.
    log_slopes : Float64[NDArray, " node"]
        Precomputed PCHIP derivatives of ``log(sigma)`` with respect to
        ``log(photon_energy)``. Unsupported nodes contain ``NaN``.

    Raises
    ------
    ValueError
        If the element/subshell key is not included in the manifest.
    """
    table_slice: slice = _table_slice(atomic_number, n, l)
    energy_nodes: Float64[NDArray, " n_packed"]
    sigma_nodes: Float64[NDArray, " n_packed"]
    slope_nodes: Float64[NDArray, " n_packed"]
    _, energy_nodes, sigma_nodes, slope_nodes, _ = _load_data()
    photon_energy_ev: Float64[NDArray, " node"] = energy_nodes[
        table_slice
    ].copy()
    sigma_megabarn: Float64[NDArray, " node"] = sigma_nodes[table_slice].copy()
    log_slopes: Float64[NDArray, " node"] = slope_nodes[table_slice].copy()
    table: tuple[
        Float64[NDArray, " node"],
        Float64[NDArray, " node"],
        Float64[NDArray, " node"],
    ] = (
        photon_energy_ev,
        sigma_megabarn,
        log_slopes,
    )
    return table


def _interval_index(
    photon_energy_ev: Float64[Array, ""],
    energy_nodes: Float64[Array, " node"],
    sigma_nodes: Float64[Array, " node"],
    slope_nodes: Float64[Array, " node"],
) -> tuple[Int32[Array, ""], Float64[Array, ""]]:
    """Select a positive interval, including either exact endpoint."""
    count: int = energy_nodes.shape[0]
    right_index: Int32[Array, ""] = jnp.clip(
        jnp.searchsorted(energy_nodes, photon_energy_ev, side="right") - 1,
        0,
        count - 2,
    )
    left_index: Int32[Array, ""] = jnp.clip(right_index - 1, 0, count - 2)

    def interval_valid(index: Array) -> Bool[Array, ""]:
        is_valid: Bool[Array, ""] = (
            jnp.isfinite(sigma_nodes[index])
            & jnp.isfinite(sigma_nodes[index + 1])
            & (sigma_nodes[index] > 0.0)
            & (sigma_nodes[index + 1] > 0.0)
            & jnp.isfinite(slope_nodes[index])
            & jnp.isfinite(slope_nodes[index + 1])
            & (photon_energy_ev >= energy_nodes[index])
            & (photon_energy_ev <= energy_nodes[index + 1])
        )
        return is_valid

    right_valid: Bool[Array, ""] = interval_valid(right_index)
    left_valid: Bool[Array, ""] = interval_valid(left_index)
    selected_index: Int32[Array, ""] = jnp.where(
        right_valid,
        right_index,
        left_index,
    )
    valid: Bool[Array, ""] = right_valid | left_valid
    checked_energy: Float64[Array, ""] = eqx.error_if(
        photon_energy_ev,
        ~valid,
        "photon energy lies outside the positive Yeh--Lindau intervals",
    )
    selected: tuple[Int32[Array, ""], Float64[Array, ""]] = (
        selected_index,
        checked_energy,
    )
    return selected


@jaxtyped(typechecker=beartype)
def yeh_lindau_cross_section(  # noqa: DOC502
    photon_energy_ev: ScalarFloat,
    atomic_number: int,
    n: int,
    l: int,  # noqa: E741
) -> Float64[Array, ""]:
    """Interpolate an atomic subshell photoionization cross section.

    The interpolation is a monotone PCHIP-type cubic in
    ``log(sigma_megabarn)`` versus ``log(photon_energy_ev)``. Queries never
    cross a missing or zero table entry and never extrapolate.

    :see: :class:`~.test_crosssections.TestYehLindauCrossSection`

    Notes
    -----
    Select one contiguous positive interval with static table nodes. Evaluate
    its log-log cubic Hermite polynomial and exponentiate exactly once.

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    atomic_number : int
        Nuclear charge ``Z``.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum.

    Returns
    -------
    sigma_megabarn : Float64[Array, ""]
        Interpolated subshell cross section in megabarn.

    Raises
    ------
    ValueError
        When the query selects an absent subshell or leaves every positive
        interval.
    """
    table_slice: slice = _table_slice(atomic_number, n, l)
    energy_data: Float64[NDArray, " n_packed"]
    sigma_data: Float64[NDArray, " n_packed"]
    slope_data: Float64[NDArray, " n_packed"]
    _, energy_data, sigma_data, slope_data, _ = _load_data()
    energy_nodes: Float64[Array, " node"] = jnp.asarray(
        energy_data[table_slice],
        dtype=jnp.float64,
    )
    sigma_nodes: Float64[Array, " node"] = jnp.asarray(
        sigma_data[table_slice],
        dtype=jnp.float64,
    )
    slope_nodes: Float64[Array, " node"] = jnp.asarray(
        slope_data[table_slice],
        dtype=jnp.float64,
    )
    energy: Float64[Array, ""] = jnp.asarray(
        photon_energy_ev,
        dtype=jnp.float64,
    )
    interval_index: Int32[Array, ""]
    checked_energy: Float64[Array, ""]
    interval_index, checked_energy = _interval_index(
        energy,
        energy_nodes,
        sigma_nodes,
        slope_nodes,
    )
    x_value: Float64[Array, ""] = jnp.log(checked_energy)
    x_left: Float64[Array, ""] = jnp.log(energy_nodes[interval_index])
    x_right: Float64[Array, ""] = jnp.log(energy_nodes[interval_index + 1])
    width: Float64[Array, ""] = x_right - x_left
    fraction: Float64[Array, ""] = (x_value - x_left) / width
    fraction_squared: Float64[Array, ""] = fraction * fraction
    fraction_cubed: Float64[Array, ""] = fraction_squared * fraction
    value_left: Float64[Array, ""] = jnp.log(sigma_nodes[interval_index])
    value_right: Float64[Array, ""] = jnp.log(sigma_nodes[interval_index + 1])
    slope_left: Float64[Array, ""] = slope_nodes[interval_index]
    slope_right: Float64[Array, ""] = slope_nodes[interval_index + 1]
    log_sigma: Float64[Array, ""] = (
        (2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0) * value_left
        + (fraction_cubed - 2.0 * fraction_squared + fraction)
        * width
        * slope_left
        + (-2.0 * fraction_cubed + 3.0 * fraction_squared) * value_right
        + (fraction_cubed - fraction_squared) * width * slope_right
    )
    sigma_megabarn: Float64[Array, ""] = jnp.exp(log_sigma)
    return sigma_megabarn


@jaxtyped(typechecker=beartype)
def yeh_lindau_orbital_weights(
    photon_energy_ev: ScalarFloat,
    basis: OrbitalBasis,
    atomic_numbers: tuple[int, ...],
) -> Float64[Array, " n_orb"]:
    """Return Yeh--Lindau weights for every orbital in a basis.

    The static basis supplies each element and subshell identity.

    :see: :class:`~.test_crosssections.TestYehLindauOrbitalWeights`

    Notes
    -----
    Gather each orbital's atomic number through ``basis.atom_indices``. Apply
    :func:`yeh_lindau_cross_section` with the orbital's static ``(n,l)`` pair.

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    basis : OrbitalBasis
        Static orbital-to-atom mapping and ``(n,l)`` quantum numbers.
    atomic_numbers : tuple[int, ...]
        Atomic number for every atom row referenced by
        ``basis.atom_indices``.

    Returns
    -------
    weights : Float64[Array, " n_orb"]
        Per-orbital cross sections in megabarn.

    Raises
    ------
    ValueError
        When an atom index exceeds the supplied elements or a table query
        leaves its supported domain.
    """
    if any(
        atom_index >= len(atomic_numbers) for atom_index in basis.atom_indices
    ):
        message: str = "atomic_numbers does not cover every basis atom index"
        raise ValueError(message)
    values: tuple[Float64[Array, ""], ...] = tuple(
        yeh_lindau_cross_section(
            photon_energy_ev,
            atomic_numbers[atom_index],
            principal,
            angular,
        )
        for atom_index, principal, angular in zip(
            basis.atom_indices,
            basis.n,
            basis.l,
            strict=True,
        )
    )
    weights: Float64[Array, " n_orb"] = jnp.stack(values)
    return weights


__all__: list[str] = [
    "yeh_lindau_cross_section",
    "yeh_lindau_cross_section_table",
    "yeh_lindau_orbital_weights",
]
