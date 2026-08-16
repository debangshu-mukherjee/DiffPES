"""Build compact tight-binding reference models for executable experiments.

Extended Summary
----------------
This module provides deterministic analytic models that experiment scripts can
reuse without duplicating lattice and hopping construction. The models use the
public tight-binding carrier factories and remain differentiable in their
physical input parameters.

Routine Listings
----------------
:func:`graphene_pz_model`
    Build the nearest-neighbor graphene pz reference model.
:func:`linear_chain_model`
    Build the nearest-neighbor one-dimensional chain reference model.
:func:`two_orbital_dirac_model`
    Build a two-orbital lattice Dirac reference model.
"""

from __future__ import annotations

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    ScalarFloat,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)


@jaxtyped(typechecker=beartype)
def graphene_pz_model(
    hopping_ev: ScalarFloat = -2.7,
    lattice_a_ang: ScalarFloat = 2.46,
) -> TBModel:
    """Build the nearest-neighbor graphene pz reference model.

    The model has two carbon pz orbitals and six Hermitian nearest-neighbor
    hopping records. It gives two degenerate bands at a honeycomb K point.

    :see: :class:`~.test_reference_models.TestGraphenePzModel`

    Implementation Logic
    --------------------
    1. **Build the honeycomb geometry**::

           lattice = [[a, 0, 0], [a / 2, sqrt(3) * a / 2, 0], [0, 0, 10]]

       The lattice gives a two-dimensional honeycomb cell with vacuum spacing.

    2. **Add Hermitian nearest neighbors**::

           hopping_pairs = ((0, 1), ..., (1, 0))

       The records close under reversing orbital indices and lattice cells.

    Parameters
    ----------
    hopping_ev : ScalarFloat, optional
        Carbon pz nearest-neighbor hopping in eV. Default is ``-2.7``.
    lattice_a_ang : ScalarFloat, optional
        In-plane lattice constant in Angstrom. Default is ``2.46``.

    Returns
    -------
    model : TBModel
        Validated graphene pz tight-binding model.
    """
    lattice_constant: Float64[Array, ""] = jnp.asarray(
        lattice_a_ang,
        dtype=jnp.float64,
    )
    lattice: Float64[Array, "3 3"] = jnp.stack(
        (
            jnp.array([lattice_constant, 0.0, 0.0]),
            jnp.array(
                [
                    lattice_constant / 2.0,
                    lattice_constant * jnp.sqrt(3.0) / 2.0,
                    0.0,
                ]
            ),
            jnp.array([0.0, 0.0, 10.0]),
        )
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
        positions=jnp.asarray(
            [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]],
            dtype=jnp.float64,
        ),
        species=("C", "C"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(2, 2),
        l=(1, 1),
        m=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(
        hopping_ev,
        dtype=jnp.complex128,
    )
    hopping: Complex128[Array, "6"] = jnp.stack((hopping_value,) * 6)
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping,
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
        hopping_cells=(
            (0, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        ),
        shell_index=(-1, -1),
    )
    return model


@jaxtyped(typechecker=beartype)
def linear_chain_model(
    hopping_ev: ScalarFloat = -1.0,
    lattice_a_ang: ScalarFloat = 1.0,
) -> TBModel:
    r"""Build the nearest-neighbor one-dimensional chain reference model.

    The one-orbital model gives the analytic dispersion
    :math:`E(k)=2t\\cos(2\\pi k)`. Its band width equals ``4 * abs(t)``.

    :see: :class:`~.test_reference_models.TestLinearChainModel`

    Implementation Logic
    --------------------
    1. **Build one orbital and lattice cell**::

           geometry = make_crystal_geometry(lattice, positions, species)

       The cell has one active x-direction lattice vector.

    2. **Add opposite neighbor records**::

           hopping_cells = ((1, 0, 0), (-1, 0, 0))

       The pair gives an exact Hermitian nearest-neighbor chain.

    Parameters
    ----------
    hopping_ev : ScalarFloat, optional
        Nearest-neighbor hopping in eV. Default is ``-1.0``.
    lattice_a_ang : ScalarFloat, optional
        Chain lattice constant in Angstrom. Default is ``1.0``.

    Returns
    -------
    model : TBModel
        Validated one-orbital chain tight-binding model.
    """
    lattice_constant: Float64[Array, ""] = jnp.asarray(
        lattice_a_ang,
        dtype=jnp.float64,
    )
    lattice: Float64[Array, "3 3"] = jnp.diag(
        jnp.array([lattice_constant, 10.0, 10.0])
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(
        hopping_ev,
        dtype=jnp.complex128,
    )
    hopping: Complex128[Array, "2"] = jnp.stack((hopping_value, hopping_value))
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping,
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0)),
        shell_index=(-1,),
    )
    return model


@jaxtyped(typechecker=beartype)
def two_orbital_dirac_model(
    velocity_ev_ang: ScalarFloat = 1.0,
    lattice_a_ang: ScalarFloat = 1.0,
) -> TBModel:
    """Build a two-orbital lattice Dirac reference model.

    The model uses opposite imaginary inter-orbital hoppings. Its low-momentum
    spectrum supplies a compact phase-complete Dirac-like test surface.

    :see: :class:`~.test_reference_models.TestTwoOrbitalDiracModel`

    Implementation Logic
    --------------------
    1. **Build the two-orbital cell**::

           basis = make_orbital_basis(atom_indices=(0, 0), ...)

       The two labels represent the components of the lattice Dirac spinor.

    2. **Set antisymmetric inter-orbital hoppings**::

           forward = 0.5j * velocity / lattice_constant

       Opposite cells and orbital pairs provide exact Hermitian closure.

    Parameters
    ----------
    velocity_ev_ang : ScalarFloat, optional
        Dirac velocity scale in eV Angstrom. Default is ``1.0``.
    lattice_a_ang : ScalarFloat, optional
        Lattice constant in Angstrom. Default is ``1.0``.

    Returns
    -------
    model : TBModel
        Validated two-orbital lattice Dirac tight-binding model.
    """
    lattice_constant: Float64[Array, ""] = jnp.asarray(
        lattice_a_ang,
        dtype=jnp.float64,
    )
    velocity: Float64[Array, ""] = jnp.asarray(
        velocity_ev_ang,
        dtype=jnp.float64,
    )
    lattice: Float64[Array, "3 3"] = jnp.diag(
        jnp.array([lattice_constant, lattice_constant, 10.0])
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        labels=("upper", "lower"),
    )
    forward: Complex128[Array, ""] = jnp.asarray(
        0.5j * velocity / lattice_constant,
        dtype=jnp.complex128,
    )
    hopping: Complex128[Array, "4"] = jnp.stack(
        (forward, -forward, -forward, forward)
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping,
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1), (1, 0), (0, 1), (1, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0), (-1, 0, 0), (1, 0, 0)),
        shell_index=(-1, -1),
    )
    return model


__all__: list[str] = [
    "graphene_pz_model",
    "linear_chain_model",
    "two_orbital_dirac_model",
]
