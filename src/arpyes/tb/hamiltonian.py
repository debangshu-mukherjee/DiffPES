"""Tight-binding Hamiltonian builder in JAX.

Extended Summary
----------------
Provides a minimal Slater-Koster Hamiltonian builder and convenience
functions for creating test models (graphene, 1D chain). The
Hamiltonian is fully JAX-traceable so that ``jax.grad`` can
differentiate eigenvalues with respect to hopping parameters.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from arpyes.types import OrbitalBasis, TBModel, make_orbital_basis, make_tb_model
from arpyes.types.aliases import ScalarFloat


@jaxtyped(typechecker=beartype)
def build_hamiltonian_k(
    k: Float[Array, " 3"],
    hopping_params: Float[Array, " H"],
    hopping_indices: tuple,
    n_orbitals: int,
    lattice_vectors: Float[Array, "3 3"],
) -> Complex[Array, "O O"]:
    r"""Build the Bloch Hamiltonian H(k) at a single k-point.

    .. math::

        H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} t_{ij,\mathbf{R}}
            \exp(i \mathbf{k} \cdot \mathbf{R})

    where the sum runs over lattice vectors R defined by the
    hopping indices. The result is Hermitianized:
    ``H = (H_raw + H_raw^dag) / 2``.

    Parameters
    ----------
    k : Float[Array, " 3"]
        k-point in fractional coordinates.
    hopping_params : Float[Array, " H"]
        Hopping amplitudes (differentiable).
    hopping_indices : tuple
        ``(orb_i, orb_j, (R_x, R_y, R_z))`` per hopping.
    n_orbitals : int
        Number of orbitals in the unit cell.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors (rows).

    Returns
    -------
    H_k : Complex[Array, "O O"]
        Hermitian Hamiltonian matrix.
    """
    H = jnp.zeros((n_orbitals, n_orbitals), dtype=jnp.complex128)

    for h_idx, (orb_i, orb_j, R_ijk) in enumerate(hopping_indices):
        t = hopping_params[h_idx]
        R_frac = jnp.array(R_ijk, dtype=jnp.float64)
        # Bloch phase with fractional coordinates:
        # exp(2πi k_frac · R_frac), since k·R = 2π k_frac·R_frac
        phase = jnp.exp(2j * jnp.pi * jnp.dot(k, R_frac))
        H = H.at[orb_i, orb_j].add(t * phase)

    # Hermitianize
    H = (H + H.conj().T) / 2.0
    return H


def make_1d_chain_model(
    t: ScalarFloat = -1.0,
) -> TBModel:
    """Create a 1D chain tight-binding model.

    Single orbital per unit cell with nearest-neighbor hopping t.

    Parameters
    ----------
    t : ScalarFloat
        Hopping amplitude. Default -1.0.

    Returns
    -------
    model : TBModel
        1D chain model.
    """
    basis = make_orbital_basis(
        n_values=(1,),
        l_values=(0,),
        m_values=(0,),
        labels=("s",),
    )
    hopping_indices = (
        (0, 0, (1, 0, 0)),   # +R hop
        (0, 0, (-1, 0, 0)),  # -R hop
    )
    lattice = jnp.eye(3, dtype=jnp.float64)
    return make_tb_model(
        hopping_params=jnp.array([t, t], dtype=jnp.float64),
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=1,
        orbital_basis=basis,
    )


def make_graphene_model(
    t: ScalarFloat = -2.7,
) -> TBModel:
    """Create a graphene pz tight-binding model.

    Two-orbital (A/B sublattice) model on a honeycomb lattice
    with nearest-neighbor hopping t.

    Parameters
    ----------
    t : ScalarFloat
        Nearest-neighbor hopping. Default -2.7 eV.

    Returns
    -------
    model : TBModel
        Graphene model.
    """
    basis = make_orbital_basis(
        n_values=(2, 2),
        l_values=(1, 1),
        m_values=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    # Honeycomb lattice vectors (Angstrom)
    a = 2.46
    a1 = jnp.array([a, 0.0, 0.0], dtype=jnp.float64)
    a2 = jnp.array([a / 2.0, a * jnp.sqrt(3.0) / 2.0, 0.0], dtype=jnp.float64)
    a3 = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)
    lattice = jnp.stack([a1, a2, a3])

    # Three nearest-neighbor hoppings A->B
    hopping_indices = (
        (0, 1, (0, 0, 0)),    # same cell
        (0, 1, (-1, 0, 0)),   # -a1
        (0, 1, (0, -1, 0)),   # -a2
        # Hermitian conjugates (B->A)
        (1, 0, (0, 0, 0)),    # same cell
        (1, 0, (1, 0, 0)),    # +a1
        (1, 0, (0, 1, 0)),    # +a2
    )
    t_val = jnp.asarray(t, dtype=jnp.float64)
    hopping_params = jnp.array(
        [t_val, t_val, t_val, t_val, t_val, t_val], dtype=jnp.float64
    )
    return make_tb_model(
        hopping_params=hopping_params,
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=2,
        orbital_basis=basis,
    )


__all__: list[str] = [
    "build_hamiltonian_k",
    "make_1d_chain_model",
    "make_graphene_model",
]
