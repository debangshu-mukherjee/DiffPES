"""Tight-binding Hamiltonian builder in JAX.

Extended Summary
----------------
Provides a minimal Slater-Koster Hamiltonian builder and convenience
functions for creating test models (graphene, 1D chain). The
Hamiltonian is fully JAX-traceable so that ``jax.grad`` can
differentiate eigenvalues with respect to hopping parameters.

Routine Listings
----------------
:func:`build_hamiltonian_k`
    Build the Bloch Hamiltonian H(k) at a single k-point.
:func:`make_1d_chain_model`
    Create a 1D chain tight-binding model.
:func:`make_graphene_model`
    Create a graphene pz tight-binding model.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import (
    OrbitalBasis,
    TBModel,
    make_orbital_basis,
    make_tb_model,
)
from diffpes.types.aliases import ScalarFloat


@jaxtyped(typechecker=beartype)
def build_hamiltonian_k(
    k: Float[Array, " 3"],
    hopping_params: Float[Array, " H"],
    hopping_indices: tuple,
    n_orbitals: int,
    lattice_vectors: Float[Array, "3 3"],  # noqa: ARG001
) -> Complex[Array, "O O"]:
    r"""Build the Bloch Hamiltonian H(k) at a single k-point.

    .. math::

        H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} t_{ij,\mathbf{R}}
            \exp(i \mathbf{k} \cdot \mathbf{R})

    where the sum runs over lattice vectors R defined by the
    hopping indices. The result is Hermitianized:
    ``H = (H_raw + H_raw^dag) / 2``.

    Extended Summary
    ----------------
    The Bloch sum is evaluated entirely in **fractional coordinates**.
    Because ``k`` is given in units of the reciprocal lattice vectors
    and ``R`` (encoded in each hopping triple) is given in units of the
    direct lattice vectors, the dot product ``k_frac . R_frac`` already
    absorbs the metric: the Cartesian phase would be
    ``exp(i k_cart . R_cart) = exp(2 pi i k_frac . R_frac)``, so the
    lattice-vector matrix itself is never needed at this stage (it is
    stored in the model for downstream Cartesian conversions).

    The function iterates over every hopping entry in Python (not a
    JAX scan) because the number of hoppings is typically small and
    known at trace time.  For each hopping ``(orb_i, orb_j, R_ijk)``
    the Bloch phase ``exp(2 pi i k . R)`` is computed and the
    corresponding hopping amplitude is accumulated into entry
    ``H[orb_i, orb_j]``.

    After all hoppings are accumulated the matrix is explicitly
    Hermitianized via ``(H + H^dag) / 2``.  This is necessary because
    the user-supplied hopping list may only contain the upper triangle
    (e.g. only A -> B hops); the Hermitianization symmetrically fills
    the lower triangle and also corrects floating-point round-off that
    would otherwise break ``jnp.linalg.eigh``.

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

    Notes
    -----
    The hopping amplitudes ``hopping_params`` are plain real floats.
    Complex (spin-orbit) hoppings would require promoting them to
    ``complex128`` and removing the Hermitianization short-cut.

    Because the function is decorated with ``@jaxtyped`` /
    ``@beartype``, shape and dtype errors are caught at call time
    rather than deep inside the JAX trace.
    """
    H: Complex[Array, "O O"] = jnp.zeros(
        (n_orbitals, n_orbitals), dtype=jnp.complex128
    )

    for h_idx, (orb_i, orb_j, R_ijk) in enumerate(hopping_indices):
        t: Float[Array, " "] = hopping_params[h_idx]
        R_frac: Float[Array, " 3"] = jnp.array(R_ijk, dtype=jnp.float64)
        # Bloch phase with fractional coordinates:
        # exp(2πi k_frac · R_frac), since k·R = 2π k_frac·R_frac
        phase: Complex[Array, " "] = jnp.exp(2j * jnp.pi * jnp.dot(k, R_frac))
        H = H.at[orb_i, orb_j].add(t * phase)

    # Hermitianize
    H = (H + H.conj().T) / 2.0
    H_k: Complex[Array, "O O"] = H
    return H_k


def make_1d_chain_model(
    t: ScalarFloat = -1.0,
) -> TBModel:
    r"""Create a 1D chain tight-binding model.

    Single orbital per unit cell with nearest-neighbor hopping t.

    Extended Summary
    ----------------
    The 1D chain is the simplest possible tight-binding model: one
    s-orbital per unit cell with hopping only to the two nearest
    neighbors at lattice vectors ``+a1`` and ``-a1``.  The lattice is
    set to an identity matrix (``a1 = [1, 0, 0]``, etc.) so that
    fractional and Cartesian coordinates coincide with a lattice
    constant of 1 (arbitrary units).

    The resulting band dispersion is the textbook cosine band:

    .. math::

        E(k) = 2t \cos(2 \pi k)

    with bandwidth ``|4t|``.  This model is useful as a minimal
    smoke-test for the Hamiltonian builder, diagonalizer, and
    gradient machinery.

    The hopping list contains two entries -- ``(0, 0, (+1,0,0))`` and
    ``(0, 0, (-1,0,0))`` -- which are the forward and backward
    nearest-neighbor hops of the single orbital to itself in adjacent
    unit cells.  After Hermitianization in ``build_hamiltonian_k``
    these are redundant (each is its own conjugate), so the
    on-diagonal entry receives ``2t cos(2 pi k)`` as expected.

    Parameters
    ----------
    t : ScalarFloat
        Hopping amplitude. Default -1.0.

    Returns
    -------
    model : TBModel
        1D chain model.
    """
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(1,),
        l_values=(0,),
        m_values=(0,),
        labels=("s",),
    )
    hopping_indices: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
        (0, 0, (1, 0, 0)),  # +R hop
        (0, 0, (-1, 0, 0)),  # -R hop
    )
    lattice: Float[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)
    model: TBModel = make_tb_model(
        hopping_params=jnp.array([t, t], dtype=jnp.float64),
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=1,
        orbital_basis=basis,
    )
    return model


def make_graphene_model(
    t: ScalarFloat = -2.7,
) -> TBModel:
    """Create a graphene pz tight-binding model.

    Two-orbital (A/B sublattice) model on a honeycomb lattice
    with nearest-neighbor hopping t.

    Extended Summary
    ----------------
    Graphene's honeycomb lattice has two atoms (sublattices A and B)
    per primitive cell.  The lattice vectors used here are:

    * ``a1 = (a, 0, 0)``
    * ``a2 = (a/2, a*sqrt(3)/2, 0)``
    * ``a3 = (0, 0, 10)``  (vacuum slab for 2-D periodicity)

    with ``a = 2.46`` Angstrom (the experimental graphene lattice
    constant).  The two orbitals are labeled ``A_pz`` and ``B_pz``
    with quantum numbers ``(n=2, l=1, m=0)``, representing carbon
    p_z orbitals on each sublattice.

    Each A-site atom has three nearest-neighbor B-site atoms.  In
    fractional coordinates the three A -> B hoppings connect to cells
    ``(0,0,0)``, ``(-1,0,0)``, and ``(0,-1,0)``.  The reverse B -> A
    hoppings at ``(0,0,0)``, ``(+1,0,0)``, and ``(0,+1,0)`` are
    listed explicitly so that the raw Hamiltonian matrix is already
    nearly Hermitian before the Hermitianization step in
    ``build_hamiltonian_k``.

    The resulting 2x2 Hamiltonian produces the classic Dirac-cone
    band structure with linear dispersion near the K and K' points
    and a bandwidth of ``|6t|``.

    Parameters
    ----------
    t : ScalarFloat
        Nearest-neighbor hopping. Default -2.7 eV.

    Returns
    -------
    model : TBModel
        Graphene model.

    Notes
    -----
    The default hopping value of -2.7 eV reproduces the standard
    nearest-neighbor graphene band structure commonly used in the
    literature (e.g. Castro Neto et al., Rev. Mod. Phys. 81, 109).
    The negative sign follows the convention that bonding states
    are lower in energy.
    """
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(2, 2),
        l_values=(1, 1),
        m_values=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    # Honeycomb lattice vectors (Angstrom)
    a: float = 2.46
    a1: Float[Array, " 3"] = jnp.array([a, 0.0, 0.0], dtype=jnp.float64)
    a2: Float[Array, " 3"] = jnp.array(
        [a / 2.0, a * jnp.sqrt(3.0) / 2.0, 0.0], dtype=jnp.float64
    )
    a3: Float[Array, " 3"] = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)
    lattice: Float[Array, "3 3"] = jnp.stack([a1, a2, a3])

    # Three nearest-neighbor hoppings A->B
    hopping_indices: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
        (0, 1, (0, 0, 0)),  # same cell
        (0, 1, (-1, 0, 0)),  # -a1
        (0, 1, (0, -1, 0)),  # -a2
        # Hermitian conjugates (B->A)
        (1, 0, (0, 0, 0)),  # same cell
        (1, 0, (1, 0, 0)),  # +a1
        (1, 0, (0, 1, 0)),  # +a2
    )
    t_val: Float[Array, " "] = jnp.asarray(t, dtype=jnp.float64)
    hopping_params: Float[Array, " H"] = jnp.array(
        [t_val, t_val, t_val, t_val, t_val, t_val], dtype=jnp.float64
    )
    model: TBModel = make_tb_model(
        hopping_params=hopping_params,
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=2,
        orbital_basis=basis,
    )
    return model


__all__: list[str] = [
    "build_hamiltonian_k",
    "make_1d_chain_model",
    "make_graphene_model",
]
