"""Tight-binding model and diagonalized band data structures.

Extended Summary
----------------
Defines PyTree types for tight-binding model parameters and
diagonalized electronic structure. ``DiagonalizedBands`` is the
common interface between TB-derived and VASP-derived inputs for
the differentiable forward simulator.

Routine Listings
----------------
:class:`DiagonalizedBands`
    PyTree for diagonalized electronic structure.
:class:`TBModel`
    PyTree for tight-binding model parameters.
:func:`make_diagonalized_bands`
    Factory for DiagonalizedBands.
:func:`make_tb_model`
    Factory for TBModel.

Notes
-----
``DiagonalizedBands`` has all-array children (fully differentiable).
``TBModel`` separates differentiable hopping amplitudes (children)
from static structural metadata (auxiliary data).
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Complex, Float, jaxtyped

from .aliases import ScalarNumeric
from .radial_params import OrbitalBasis


@register_pytree_node_class
class DiagonalizedBands(NamedTuple):
    """PyTree for diagonalized electronic structure.

    The common interface between VASP-derived and TB-derived inputs
    for the forward simulator ``simulate_tb_radial``. The companion
    TB-DFT project produces this PyTree; so does the VASP adapter
    ``vasp_to_diagonalized``.

    Attributes
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV.
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb}.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal space.
    fermi_energy : Float[Array, " "]
        Fermi level in eV.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    All four fields are JAX array children (fully differentiable).
    """

    eigenvalues: Float[Array, "K B"]
    eigenvectors: Complex[Array, "K B O"]
    kpoints: Float[Array, "K 3"]
    fermi_energy: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "K B"],
            Complex[Array, "K B O"],
            Float[Array, "K 3"],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten into JAX leaf arrays and auxiliary data.

        Returns
        -------
        children : tuple of jax.Array
            All four fields.
        aux_data : None
        """
        return (
            (
                self.eigenvalues,
                self.eigenvectors,
                self.kpoints,
                self.fermi_energy,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, "K B"],
            Complex[Array, "K B O"],
            Float[Array, "K 3"],
            Float[Array, " "],
        ],
    ) -> "DiagonalizedBands":
        """Reconstruct from flattened components.

        Parameters
        ----------
        _aux_data : None
        children : tuple of jax.Array

        Returns
        -------
        DiagonalizedBands
        """
        return cls(*children)


@register_pytree_node_class
class TBModel(NamedTuple):
    """PyTree for tight-binding model parameters.

    Minimal Slater-Koster tight-binding model for testing the
    differentiable forward simulator. The hopping amplitudes are
    differentiable (children) while the structural information
    (which orbitals connect to which, lattice vectors, orbital
    basis) is static (auxiliary data).

    Attributes
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes. Differentiable (child).
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors (rows). Differentiable (child).
    hopping_indices : tuple[tuple[int, int, tuple[int, int, int]], ...]
        ``(orb_i, orb_j, (R_x, R_y, R_z))`` per hopping. Static.
    n_orbitals : int
        Number of orbitals in the unit cell. Static.
    orbital_basis : OrbitalBasis
        Quantum numbers for each orbital. Static.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    ``hopping_params`` and ``lattice_vectors`` are children;
    ``hopping_indices``, ``n_orbitals``, and ``orbital_basis`` are
    auxiliary data.
    """

    hopping_params: Float[Array, " H"]
    lattice_vectors: Float[Array, "3 3"]
    hopping_indices: tuple
    n_orbitals: int
    orbital_basis: OrbitalBasis

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, " H"], Float[Array, "3 3"]],
        Tuple[tuple, int, OrbitalBasis],
    ]:
        """Flatten into JAX children and auxiliary data.

        Returns
        -------
        children : tuple of Array
            ``(hopping_params, lattice_vectors)``.
        aux_data : tuple
            ``(hopping_indices, n_orbitals, orbital_basis)``.
        """
        return (
            (self.hopping_params, self.lattice_vectors),
            (self.hopping_indices, self.n_orbitals, self.orbital_basis),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[tuple, int, OrbitalBasis],
        children: Tuple[Float[Array, " H"], Float[Array, "3 3"]],
    ) -> "TBModel":
        """Reconstruct a TBModel from flattened components.

        Parameters
        ----------
        aux_data : tuple
            ``(hopping_indices, n_orbitals, orbital_basis)``.
        children : tuple of Array
            ``(hopping_params, lattice_vectors)``.

        Returns
        -------
        model : TBModel
            Reconstructed instance.
        """
        hopping_params, lattice_vectors = children
        hopping_indices, n_orbitals, orbital_basis = aux_data
        return cls(
            hopping_params=hopping_params,
            lattice_vectors=lattice_vectors,
            hopping_indices=hopping_indices,
            n_orbitals=n_orbitals,
            orbital_basis=orbital_basis,
        )


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(
    eigenvalues: Float[Array, "K B"],
    eigenvectors: Complex[Array, "K B O"],
    kpoints: Float[Array, "K 3"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DiagonalizedBands:
    """Create a validated DiagonalizedBands instance.

    Parameters
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV.
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients.
    kpoints : Float[Array, "K 3"]
        k-point coordinates.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : DiagonalizedBands
        Validated instance with float64/complex128 arrays.
    """
    eig_arr = jnp.asarray(eigenvalues, dtype=jnp.float64)
    vec_arr = jnp.asarray(eigenvectors, dtype=jnp.complex128)
    kpt_arr = jnp.asarray(kpoints, dtype=jnp.float64)
    ef_arr = jnp.asarray(fermi_energy, dtype=jnp.float64)
    return DiagonalizedBands(
        eigenvalues=eig_arr,
        eigenvectors=vec_arr,
        kpoints=kpt_arr,
        fermi_energy=ef_arr,
    )


@jaxtyped(typechecker=beartype)
def make_tb_model(
    hopping_params: Float[Array, " H"],
    lattice_vectors: Float[Array, "3 3"],
    hopping_indices: tuple,
    n_orbitals: int,
    orbital_basis: OrbitalBasis,
) -> TBModel:
    """Create a validated TBModel instance.

    Parameters
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors.
    hopping_indices : tuple
        ``(orb_i, orb_j, (R_x, R_y, R_z))`` per hopping.
    n_orbitals : int
        Number of orbitals in the unit cell.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata.

    Returns
    -------
    model : TBModel
        Validated tight-binding model.
    """
    hop_arr = jnp.asarray(hopping_params, dtype=jnp.float64)
    lat_arr = jnp.asarray(lattice_vectors, dtype=jnp.float64)
    return TBModel(
        hopping_params=hop_arr,
        lattice_vectors=lat_arr,
        hopping_indices=hopping_indices,
        n_orbitals=n_orbitals,
        orbital_basis=orbital_basis,
    )


__all__: list[str] = [
    "DiagonalizedBands",
    "TBModel",
    "make_diagonalized_bands",
    "make_tb_model",
]
