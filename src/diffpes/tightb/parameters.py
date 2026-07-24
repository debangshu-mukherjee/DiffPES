"""Expose independent real optimizer coordinates for tight-binding models.

Extended Summary
----------------
Materialized tight-binding models contain a directed, Hermitian-closed
hopping list. Optimizers must not vary both members of a conjugate pair
independently. This module packs one representative of each pair, followed by
onsite energies, spin--orbit strengths, and optional fractional positions and
lattice vectors.
Complex representatives cross the optimizer boundary through
``diffpes.utils.pack_complex``.

Slater--Koster models have a separate view. Their primary coordinates are the
fundamental two-center integrals rather than the redundant materialized
hoppings. Its inverse closure rebuilds the model through
``diffpes.tightb.build_sk_model``.

Routine Listings
----------------
:func:`tb_parameter_view`
    Pack a materialized tight-binding model into independent coordinates.
:func:`sk_model_parameter_view`
    Pack Slater--Koster fundamentals and return a rebuilding closure.

Notes
-----
Both views retain the band-energy-zero gauge. A uniform onsite shift matches
the same Fermi-energy or energy-offset shift. Downstream inversion must
constrain that direction because these lossless views retain it.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlaterKosterParams,
    TBModel,
    make_crystal_geometry,
    make_tb_model,
)
from diffpes.utils import pack_complex, unpack_complex

from .slaterkoster import build_sk_model


def _reverse_indices(model: TBModel) -> tuple[int, ...]:
    """Derive the exact reverse-hopping permutation."""
    records: tuple[tuple[int, int, tuple[int, int, int]], ...] = tuple(
        (pair[0], pair[1], cell)
        for pair, cell in zip(
            model.hopping_pairs,
            model.hopping_cells,
            strict=True,
        )
    )
    lookup: dict[tuple[int, int, tuple[int, int, int]], int] = {
        record: index for index, record in enumerate(records)
    }
    reverse: tuple[int, ...] = tuple(
        lookup[(orbital_j, orbital_i, (-cell[0], -cell[1], -cell[2]))]
        for orbital_i, orbital_j, cell in records
    )
    return reverse


def _require_exact_closure(
    model: TBModel,
    reverse_indices: tuple[int, ...],
) -> np.ndarray:
    """Require lossless rather than tolerance-projected Hermitian closure."""
    host_amplitudes: np.ndarray = np.asarray(model.hopping_amplitudes)
    reversed_amplitudes: np.ndarray = host_amplitudes[
        np.asarray(reverse_indices, dtype=np.int64)
    ]
    if not np.array_equal(reversed_amplitudes, np.conj(host_amplitudes)):
        message: str = (
            "tb_parameter_view requires exactly conjugate-closed hopping "
            "amplitudes; tolerance-close inputs cannot be packed losslessly"
        )
        raise ValueError(message)
    return host_amplitudes


def _checked_vector(
    vector: Float[Array, " n_par"],
    expected_size: int,
    context: str,
) -> Float[Array, " n_par"]:
    """Normalize and validate one flat optimizer vector."""
    array: Float[Array, " n_par"] = jnp.asarray(
        vector,
        dtype=jnp.float64,
    )
    if array.ndim != 1 or array.shape[0] != expected_size:
        message: str = (
            f"{context}: parameter vector must have shape ({expected_size},)"
        )
        raise ValueError(message)
    checked: Float[Array, " n_par"] = eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array)),
        f"{context}: parameters finite",
    )
    return checked


@jaxtyped(typechecker=beartype)
def tb_parameter_view(  # noqa: DOC503, PLR0915
    model: TBModel,
    include_positions: bool = False,
    include_lattice: bool = False,
) -> tuple[
    Float[Array, " n_par"],
    Callable[[Float[Array, " n_par"]], TBModel],
]:
    """Pack a materialized tight-binding model into independent coordinates.

    The view removes redundant Hermitian partners while preserving every
    independent real model coordinate.

    :see: :class:`~.test_parameters.TestTBParameterView`

    Parameters
    ----------
    model : TBModel
        Exactly Hermitian-closed materialized tight-binding model.
    include_positions : bool, optional
        Append flattened fractional atomic positions when ``True``. Default is
        ``False``.
    include_lattice : bool, optional
        Append flattened real-space lattice rows when ``True``. Default is
        ``False``. The rebuilding closure recomputes the reciprocal lattice.

    Returns
    -------
    parameters : Float[Array, " n_par"]
        Flat float64 optimizer coordinates. Each conjugate pair contributes
        one representative. Complex representatives contribute real and
        imaginary parts. Self-reverse records contribute one real value.
        Onsite, SOC, position, and lattice values follow.
    rebuild : Callable[[Float[Array, " n_par"]], TBModel]
        Pure inverse closure that reconstructs an exactly Hermitian-closed
        model.

    Raises
    ------
    ValueError
        If an inclusion selector is not Boolean or the input hopping values
        are only tolerance-close, rather than exactly, conjugate-closed. The
        rebuilding closure also rejects a vector with the wrong static shape.
    EquinoxRuntimeError
        If a rebuilding vector contains a non-finite value.

    Notes
    -----
    This view exposes materialized complex hoppings and cannot recover
    Slater--Koster fundamentals that the model does not retain. Use
    :func:`sk_model_parameter_view` for that parameterization.

    The onsite block contains a band-energy-zero gauge direction: a uniform
    onsite shift is degenerate with the Fermi-energy/energy-offset nuisance.
    Inversion code must constrain or quotient that direction rather than
    expecting this lossless packing layer to discard it.
    """
    if type(include_positions) is not bool:
        message: str = "include_positions must be a bool"
        raise ValueError(message)
    if type(include_lattice) is not bool:
        message = "include_lattice must be a bool"
        raise ValueError(message)
    reverse_indices: tuple[int, ...] = _reverse_indices(model)
    host_amplitudes: np.ndarray = _require_exact_closure(
        model,
        reverse_indices,
    )
    representatives: tuple[int, ...] = tuple(
        index
        for index, reverse in enumerate(reverse_indices)
        if index <= reverse
    )
    self_reverse: tuple[bool, ...] = tuple(
        representative == reverse_indices[representative]
        for representative in representatives
    )

    parts: list[Float[Array, " n_part"]] = []
    representative: int
    is_self_reverse: bool
    for representative, is_self_reverse in zip(
        representatives,
        self_reverse,
        strict=True,
    ):
        amplitude: Complex[Array, ""] = model.hopping_amplitudes[
            representative
        ]
        if is_self_reverse:
            parts.append(jnp.reshape(jnp.real(amplitude), (1,)))
        else:
            parts.append(jnp.ravel(pack_complex(amplitude)))
    parts.extend(
        (
            jnp.ravel(model.onsite_energies),
            jnp.ravel(model.soc_lambdas),
        )
    )
    if include_positions:
        parts.append(jnp.ravel(model.geometry.positions))
    if include_lattice:
        parts.append(jnp.ravel(model.geometry.lattice))
    parameters: Float[Array, " n_par"] = (
        jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.float64)
    )
    expected_size: int = parameters.shape[0]

    def rebuild(vector: Float[Array, " n_par"]) -> TBModel:
        """Return the model reconstructed from independent coordinates."""
        checked: Float[Array, " n_par"] = _checked_vector(
            vector,
            expected_size,
            "tb_parameter_view rebuild",
        )
        hopping: Complex[Array, " n_hop"] = jnp.zeros_like(
            model.hopping_amplitudes
        )
        offset: int = 0
        representative: int
        is_self_reverse: bool
        for representative, is_self_reverse in zip(
            representatives,
            self_reverse,
            strict=True,
        ):
            reverse: int = reverse_indices[representative]
            amplitude: Complex[Array, ""]
            if is_self_reverse:
                preserved_zero: Float[Array, ""] = jnp.asarray(
                    np.imag(host_amplitudes[representative]),
                    dtype=jnp.float64,
                )
                amplitude = jax.lax.complex(
                    checked[offset],
                    preserved_zero,
                )
                offset += 1
                hopping = hopping.at[representative].set(amplitude)
            else:
                packed: Float[Array, " 2"] = checked[offset : offset + 2]
                amplitude = unpack_complex(packed)
                offset += 2
                hopping = hopping.at[representative].set(amplitude)
                hopping = hopping.at[reverse].set(jnp.conj(amplitude))

        n_onsite: int = model.onsite_energies.size
        onsite: Float[Array, " n_orb"] = jnp.reshape(
            checked[offset : offset + n_onsite],
            model.onsite_energies.shape,
        )
        offset += n_onsite
        n_soc: int = model.soc_lambdas.size
        soc: Float[Array, " n_shells"] = jnp.reshape(
            checked[offset : offset + n_soc],
            model.soc_lambdas.shape,
        )
        offset += n_soc
        geometry: CrystalGeometry = model.geometry
        positions: Float[Array, "n_atoms 3"] = model.geometry.positions
        if include_positions:
            n_positions: int = model.geometry.positions.size
            positions = jnp.reshape(
                checked[offset : offset + n_positions],
                model.geometry.positions.shape,
            )
            offset += n_positions
            geometry = eqx.tree_at(
                lambda item: item.positions,
                model.geometry,
                positions,
            )
        if include_lattice:
            lattice: Float[Array, "3 3"] = jnp.reshape(
                checked[offset:],
                model.geometry.lattice.shape,
            )
            geometry = make_crystal_geometry(
                lattice,
                positions,
                model.geometry.species,
            )
        rebuilt: TBModel = make_tb_model(
            hopping,
            onsite,
            soc,
            geometry,
            model.basis,
            model.hopping_pairs,
            model.hopping_cells,
            model.shell_index,
            model.spinor,
        )
        return rebuilt

    result: tuple[
        Float[Array, " n_par"],
        Callable[[Float[Array, " n_par"]], TBModel],
    ] = (parameters, rebuild)
    return result


@jaxtyped(typechecker=beartype)
def sk_model_parameter_view(  # noqa: DOC502, DOC503
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    sk_params: SlaterKosterParams,
    onsite_energies: Float[Array, " n_orb"],
    soc_lambdas: Float[Array, " n_shells"],
    shell_index: tuple[int, ...],
    cutoff: float,
    include_positions: bool = False,
    include_lattice: bool = False,
) -> tuple[
    Float[Array, " n_par"],
    Callable[[Float[Array, " n_par"]], TBModel],
]:
    """Pack Slater--Koster fundamentals and return a rebuilding closure.

    The closure rebuilds all derived hoppings from independent two-center
    integrals and optional geometry coordinates.

    :see: :class:`~.test_parameters.TestSKModelParameterView`

    Parameters
    ----------
    geometry : CrystalGeometry
        Crystal geometry captured by the builder context.
    basis : OrbitalBasis
        Static orbital metadata captured by the builder context.
    sk_params : SlaterKosterParams
        Differentiable fundamental two-center integrals.
    onsite_energies : Float[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float[Array, " n_shells"]
        Atomic spin--orbit strengths in eV.
    shell_index : tuple[int, ...]
        Static orbital-to-SOC-shell mapping.
    cutoff : float
        Static neighbor cutoff in angstroms.
    include_positions : bool, optional
        Append flattened fractional atomic positions when ``True``. Default is
        ``False``.
    include_lattice : bool, optional
        Append flattened real-space lattice rows when ``True``. Default is
        ``False``. The rebuilding closure recomputes the reciprocal lattice.

    Returns
    -------
    parameters : Float[Array, " n_par"]
        Flat float64 vector ordered as SK values, onsite energies, SOC
        strengths, optional positions, and optional lattice rows.
    rebuild : Callable[[Float[Array, " n_par"]], TBModel]
        Pure closure that reconstructs through :func:`build_sk_model`.

    Raises
    ------
    ValueError
        If an inclusion selector is not Boolean, the captured SK model context
        is structurally invalid, or a rebuilding vector has the wrong shape.
    EquinoxRuntimeError
        If the captured model or a rebuilding vector contains non-finite
        numerical values.

    Notes
    -----
    This view is intentionally disjoint from :func:`tb_parameter_view`: it
    exposes fundamental SK values and never additionally packs the derived
    materialized hoppings.

    The onsite block retains the band-energy-zero gauge. A uniform onsite
    shift matches the fitted Fermi-energy or energy-offset shift. Downstream
    inversion must constrain that nuisance direction.
    """
    if type(include_positions) is not bool:
        message: str = "include_positions must be a bool"
        raise ValueError(message)
    if type(include_lattice) is not bool:
        message = "include_lattice must be a bool"
        raise ValueError(message)
    spinor: bool = bool(basis.spin)
    template: TBModel = build_sk_model(
        geometry,
        basis,
        sk_params,
        onsite_energies,
        soc_lambdas,
        shell_index,
        cutoff,
        spinor=spinor,
    )
    parts: list[Float[Array, " n_part"]] = [
        jnp.ravel(sk_params.values),
        jnp.ravel(template.onsite_energies),
        jnp.ravel(template.soc_lambdas),
    ]
    if include_positions:
        parts.append(jnp.ravel(template.geometry.positions))
    if include_lattice:
        parts.append(jnp.ravel(template.geometry.lattice))
    parameters: Float[Array, " n_par"] = jnp.concatenate(parts)
    expected_size: int = parameters.shape[0]
    n_sk: int = sk_params.values.size
    n_onsite: int = template.onsite_energies.size
    n_soc: int = template.soc_lambdas.size

    def rebuild(vector: Float[Array, " n_par"]) -> TBModel:
        """Return the Slater--Koster model from optimizer coordinates."""
        checked: Float[Array, " n_par"] = _checked_vector(
            vector,
            expected_size,
            "sk_model_parameter_view rebuild",
        )
        offset: int = 0
        values: Float[Array, " n_sk"] = checked[offset : offset + n_sk]
        offset += n_sk
        onsite: Float[Array, " n_orb"] = jnp.reshape(
            checked[offset : offset + n_onsite],
            template.onsite_energies.shape,
        )
        offset += n_onsite
        soc: Float[Array, " n_shells"] = jnp.reshape(
            checked[offset : offset + n_soc],
            template.soc_lambdas.shape,
        )
        offset += n_soc
        rebuilt_geometry: CrystalGeometry = template.geometry
        positions: Float[Array, "n_atoms 3"] = template.geometry.positions
        if include_positions:
            n_positions: int = template.geometry.positions.size
            positions = jnp.reshape(
                checked[offset : offset + n_positions],
                template.geometry.positions.shape,
            )
            offset += n_positions
            rebuilt_geometry = eqx.tree_at(
                lambda item: item.positions,
                template.geometry,
                positions,
            )
        if include_lattice:
            lattice: Float[Array, "3 3"] = jnp.reshape(
                checked[offset:],
                template.geometry.lattice.shape,
            )
            rebuilt_geometry = make_crystal_geometry(
                lattice,
                positions,
                template.geometry.species,
            )
        rebuilt_params: SlaterKosterParams = SlaterKosterParams(
            values=values,
            keys=sk_params.keys,
        )
        rebuilt: TBModel = build_sk_model(
            rebuilt_geometry,
            basis,
            rebuilt_params,
            onsite,
            soc,
            shell_index,
            cutoff,
            spinor=spinor,
        )
        return rebuilt

    result: tuple[
        Float[Array, " n_par"],
        Callable[[Float[Array, " n_par"]], TBModel],
    ] = (parameters, rebuild)
    return result


__all__: list[str] = [
    "sk_model_parameter_view",
    "tb_parameter_view",
]
