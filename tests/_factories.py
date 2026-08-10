"""Build deterministic toy carriers for tests.

Extended Summary
----------------
Provides small, fixed-policy inputs for forward, tight-binding, and radial
tests. Random factories are deterministic for a supplied JAX key; analytic
factories use fixed grids and physical parameters. Each factory checks every
returned traced leaf for finiteness.
"""

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, PRNGKeyArray, jaxtyped

from diffpes.tightb import (
    diagonalize_tb,
)
from diffpes.types import (
    BandStructure,
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    OrbitalProjection,
    TBModel,
    make_band_structure,
    make_crystal_geometry,
    make_orbital_basis,
    make_orbital_projection,
    make_tb_model,
)
from diffpes.types.aliases import ScalarFloat


def _assert_finite(tree: object) -> None:
    """PRIVATE: Require every numerical leaf in a toy carrier to be finite.

    Parameters
    ----------
    tree : object
        Toy carrier or any other PyTree of numerical leaves.

    Raises
    ------
    AssertionError
        If any leaf holds a non-finite value, from
        ``chex.assert_tree_all_finite``.

    Notes
    -----
    ``jax.tree.leaves`` collects the leaves first, so every factory
    output passes through one uniform finiteness check.
    """
    leaves: Tuple[object, ...] = tuple(jax.tree.leaves(tree))
    chex.assert_tree_all_finite(leaves)


@jaxtyped(typechecker=beartype)
def make_1d_chain_model(t: ScalarFloat = -1.0) -> TBModel:
    r"""Build the closed nearest-neighbor one-dimensional chain fixture.

    The single-orbital model is an external-truth fixture for
    :math:`E(k)=2t\cos(2\pi k)`. It uses exact integer cells and explicit
    reverse hoppings under the basis-position gauge.

    Parameters
    ----------
    t : ScalarFloat, optional
        Nearest-neighbor hopping in eV. Default is ``-1.0`` eV.

    Returns
    -------
    model : TBModel
        Validated one-orbital chain model.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
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
    hopping_value: Complex128[Array, ""] = jnp.asarray(t, dtype=jnp.complex128)
    hopping: Complex128[Array, " 2"] = jnp.stack(
        (hopping_value, hopping_value)
    )
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
def make_graphene_model(t: ScalarFloat = -2.7) -> TBModel:
    """Build the closed nearest-neighbor graphene fixture.

    Parameters
    ----------
    t : ScalarFloat, optional
        Carbon pz nearest-neighbor hopping in eV. Default is ``-2.7`` eV.

    Returns
    -------
    model : TBModel
        Validated two-orbital honeycomb model in the basis-position gauge.
    """
    lattice_constant: float = 2.46
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        [
            [lattice_constant, 0.0, 0.0],
            [
                lattice_constant / 2.0,
                lattice_constant * jnp.sqrt(3.0) / 2.0,
                0.0,
            ],
            [0.0, 0.0, 10.0],
        ],
        dtype=jnp.float64,
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
    hopping_value: Complex128[Array, ""] = jnp.asarray(t, dtype=jnp.complex128)
    hopping: Complex128[Array, " 6"] = jnp.stack((hopping_value,) * 6)
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
def make_rashba_model(
    hopping: ScalarFloat = -0.63,
    rashba: ScalarFloat = 0.27,
) -> TBModel:
    """Build a closed square-lattice Rashba spinor fixture.

    Parameters
    ----------
    hopping : ScalarFloat, optional
        Spin-independent nearest-neighbor hopping in eV.
    rashba : ScalarFloat, optional
        Rashba coupling in eV.

    Returns
    -------
    model : TBModel
        Validated two-state spinor model in down--up block order.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.diag(jnp.asarray([3.2, 3.2, 12.0], dtype=jnp.float64)),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        spin=(-1, 1),
        labels=("s_down", "s_up"),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(
        hopping,
        dtype=jnp.complex128,
    )
    rashba_value: Complex128[Array, ""] = jnp.asarray(
        rashba,
        dtype=jnp.complex128,
    )
    amplitudes: list[Complex128[Array, ""]] = []
    pairs: list[Tuple[int, int]] = []
    cells: list[Tuple[int, int, int]] = []
    cell: Tuple[int, int, int]
    spin: int
    nearest_cells: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    )
    for spin in (0, 1):
        for cell in nearest_cells:
            amplitudes.append(hopping_value)
            pairs.append((spin, spin))
            cells.append(cell)
    forward_amplitudes: Tuple[Complex128[Array, ""], ...] = (
        -0.5 * rashba_value,
        0.5 * rashba_value,
        -0.5j * rashba_value,
        0.5j * rashba_value,
    )
    amplitude: Complex128[Array, ""]
    for cell, amplitude in zip(
        nearest_cells,
        forward_amplitudes,
        strict=True,
    ):
        amplitudes.append(amplitude)
        pairs.append((0, 1))
        cells.append(cell)
        amplitudes.append(jnp.conj(amplitude))
        pairs.append((1, 0))
        cells.append(tuple(-component for component in cell))
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.stack(amplitudes),
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(pairs),
        hopping_cells=tuple(cells),
        shell_index=(-1, -1),
        spinor=True,
    )
    _assert_finite(model)
    return model


@jaxtyped(typechecker=beartype)
def make_t2g_soc_model(coupling: ScalarFloat = 0.4) -> TBModel:
    """Build an isolated projected-t2g spin--orbit fixture.

    Parameters
    ----------
    coupling : ScalarFloat, optional
        Atomic spin--orbit coupling in eV.

    Returns
    -------
    model : TBModel
        Validated six-state t2g model in down--up block order.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=4.0 * jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("Ti",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 6,
        n=(3,) * 6,
        l=(2,) * 6,
        m=(-2, -1, 1, -2, -1, 1),
        spin=(-1, -1, -1, 1, 1, 1),
        labels=(
            "dxy_down",
            "dyz_down",
            "dxz_down",
            "dxy_up",
            "dyz_up",
            "dxz_up",
        ),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((6,), dtype=jnp.float64),
        soc_lambdas=jnp.asarray([coupling], dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(0,) * 6,
        spinor=True,
    )
    _assert_finite(model)
    return model


@jaxtyped(typechecker=beartype)
def toy_band_structure(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
) -> BandStructure:
    """Build a reproducible occupied-state toy band structure.

    The factory samples eigenvalues in [-2.5, 0.25] eV, safely below
    ``E_F + 0.5`` eV. This intentionally avoids the known upper-state
    ``fermi_dirac`` gradient defect in upper states. The supplied key
    is the entire seed policy and is never mutated.
    """
    energy_key: PRNGKeyArray
    kpoint_key: PRNGKeyArray
    energy_key, kpoint_key = jax.random.split(key)
    eigenvalues: Float64[Array, "n_k n_bands"] = jax.random.uniform(
        energy_key,
        (n_k, n_bands),
        minval=-2.5,
        maxval=0.25,
        dtype=jnp.float64,
    )
    eigenvalues = jnp.sort(eigenvalues, axis=-1)
    kpoints: Float64[Array, "n_k 3"] = jax.random.uniform(
        kpoint_key,
        (n_k, 3),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float64,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        kpoint_weights=jnp.full(n_k, 1.0 / n_k, dtype=jnp.float64),
        fermi_energy=0.0,
    )
    _assert_finite(bands)
    return bands


@jaxtyped(typechecker=beartype)
def toy_orbital_projection(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
    n_atoms: int = 2,
) -> OrbitalProjection:
    """Build reproducible normalized orbital weights.

    Positive weights are drawn from a uniform distribution using only the
    supplied key, then normalized over atom and orbital axes for each state.
    Spin and orbital-angular-momentum fields remain absent.
    """
    raw: Float64[Array, "n_k n_bands n_atoms 9"] = jax.random.uniform(
        key,
        (n_k, n_bands, n_atoms, 9),
        minval=0.1,
        maxval=1.0,
        dtype=jnp.float64,
    )
    normalization: Float64[Array, "n_k n_bands 1 1"] = jnp.sum(
        raw, axis=(-2, -1), keepdims=True
    )
    projections: Float64[Array, "n_k n_bands n_atoms 9"] = raw / normalization
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections
    )
    _assert_finite(orbital_projection)
    return orbital_projection


@jaxtyped(typechecker=beartype)
def toy_graphene_diagonalized(
    n_k: int = 12,
) -> Tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native graphene model on a fixed Gamma-to-K path.

    Uses the production -2.7 eV nearest-neighbor model and an
    endpoint-inclusive fractional path from Gamma to K = (1/3, 1/3, 0).
    The factory uses no random seed.
    """
    model: TBModel = make_graphene_model()
    path_coordinate: Float64[Array, " n_k"] = jnp.linspace(
        0.0, 1.0, n_k, dtype=jnp.float64
    )
    kpoints: Float64[Array, "n_k 3"] = path_coordinate[:, None] * jnp.array(
        [1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=jnp.float64
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: Tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result


@jaxtyped(typechecker=beartype)
def toy_chain_diagonalized(
    n_k: int = 16,
) -> Tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native one-dimensional chain on a fixed k-path.

    Uses the production -1 eV hopping and an endpoint-inclusive fractional
    path from -1/2 to 1/2 along kx. The factory uses no random seed.
    """
    model: TBModel = make_1d_chain_model()
    kx: Float64[Array, " n_k"] = jnp.linspace(
        -0.5, 0.5, n_k, dtype=jnp.float64
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.stack(
        (kx, jnp.zeros_like(kx), jnp.zeros_like(kx)), axis=-1
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: Tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result
