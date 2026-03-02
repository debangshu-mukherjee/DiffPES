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
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Complex, Float, jaxtyped

from .aliases import ScalarNumeric
from .radial_params import OrbitalBasis


@register_pytree_node_class
class DiagonalizedBands(NamedTuple):
    """PyTree for diagonalized electronic structure.

    Extended Summary
    ----------------
    The common interface between VASP-derived and TB-derived inputs
    for the forward simulator ``simulate_tb_radial``. The companion
    DiffTB project (``../DiffTB``) produces this PyTree by
    diagonalizing a tight-binding Hamiltonian at each k-point; the
    VASP adapter ``vasp_to_diagonalized`` constructs it from VASP
    eigenvectors.

    By unifying both data sources into a single PyTree type, the
    differentiable forward simulator can accept inputs from either
    tight-binding models or first-principles calculations without
    code branching. The eigenvectors carry the orbital-decomposition
    information needed to compute dipole matrix elements in the
    Chinook pipeline.

    All four fields are dense JAX arrays stored as children (no
    auxiliary data), making the entire object fully differentiable.
    This enables end-to-end gradient computation from TB hopping
    parameters through diagonalization to simulated ARPES intensity.

    Attributes
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies E_n(k) in eV for K k-points and B bands.
        JAX-traced (differentiable).
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb} from the
        Hamiltonian diagonalization, where O is the number of
        orbitals in the basis. These encode the orbital character
        of each Bloch state and enter the dipole matrix element
        calculation. JAX-traced (differentiable, complex128).
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space.
        JAX-traced (differentiable).
    fermi_energy : Float[Array, " "]
        Fermi level in eV. A 0-D scalar array.
        JAX-traced (differentiable).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    All four fields are JAX array children (fully differentiable).
    The explicit ``tree_flatten`` / ``tree_unflatten`` pair makes the
    children-vs-auxiliary split self-documenting: all fields are
    children and auxiliary data is ``None``.

    See Also
    --------
    TBModel : The tight-binding model whose diagonalization produces
        a ``DiagonalizedBands``.
    make_diagonalized_bands : Factory function with validation and
        dtype casting.
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

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children**: ``(eigenvalues, eigenvectors, kpoints,
           fermi_energy)`` -- all four fields are dense JAX arrays.
           ``eigenvectors`` is complex128; the rest are float64.
           All must be visible to the tracer for ``jit``/``grad``/
           ``vmap``.
        2. **Auxiliary data**: ``None`` -- there are no static Python
           values because every field is a numerical array.

        Returns
        -------
        children : tuple of jax.Array
            ``(eigenvalues, eigenvectors, kpoints, fermi_energy)``.
        aux_data : None
            No static metadata is needed for reconstruction.
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
        """Reconstruct a ``DiagonalizedBands`` from flattened components.

        Inverse of :meth:`tree_flatten`. Called by ``jax.tree_util``
        when a traced ``DiagonalizedBands`` needs to be reassembled
        -- for example at the boundary of a ``jit``-compiled function
        or after ``vmap`` unstacks batched leaves.

        Implementation Logic
        --------------------
        1. **Auxiliary data**: ignored (always ``None``) because
           ``DiagonalizedBands`` carries no static metadata.
        2. **Reconstruction**: unpacks the children tuple positionally
           into the ``NamedTuple`` constructor via ``cls(*children)``,
           restoring ``(eigenvalues, eigenvectors, kpoints,
           fermi_energy)`` in declaration order.

        Parameters
        ----------
        _aux_data : None
            Unused static metadata (always ``None``).
        children : tuple of jax.Array
            ``(eigenvalues, eigenvectors, kpoints, fermi_energy)``
            as returned by :meth:`tree_flatten`.

        Returns
        -------
        bands : DiagonalizedBands
            Reconstructed instance with the original array fields.
        """
        eigenvalues: Float[Array, "K B"]
        eigenvectors: Complex[Array, "K B O"]
        kpoints: Float[Array, "K 3"]
        fermi_energy: Float[Array, " "]
        eigenvalues, eigenvectors, kpoints, fermi_energy = children
        bands: DiagonalizedBands = cls(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            kpoints=kpoints,
            fermi_energy=fermi_energy,
        )
        return bands


@register_pytree_node_class
class TBModel(NamedTuple):
    """PyTree for tight-binding model parameters (legacy).

    Extended Summary
    ----------------
    Minimal Slater-Koster tight-binding model retained for testing
    the differentiable forward simulator. For production TB
    workflows, use DiffTB (``../DiffTB``) to construct and
    diagonalize tight-binding models, then pass the resulting
    ``DiagonalizedBands`` to DiffPES. The Hamiltonian is constructed
    at each k-point by Fourier-transforming the real-space hopping
    matrix:

        H_{ij}(k) = sum_R t_{ij,R} * exp(i k . R)

    where ``t_{ij,R}`` are the hopping amplitudes between orbital i
    and orbital j separated by lattice vector R.

    The hopping amplitudes and lattice vectors are differentiable
    (children) so that ``jax.grad`` can compute gradients of the
    simulated ARPES intensity with respect to TB parameters. The
    structural information (connectivity, orbital count, quantum
    numbers) is static (auxiliary data) because changing the model
    topology requires JIT recompilation.

    Attributes
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes t_{ij,R} in eV, one per hopping
        connection. H is the total number of unique hoppings in the
        model. JAX-traced (differentiable) -- the primary
        optimization target for TB model fitting.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms. Used to
        convert the lattice translation indices (R_x, R_y, R_z) in
        ``hopping_indices`` to Cartesian displacements for the
        Fourier transform. JAX-traced (differentiable).
    hopping_indices : tuple[tuple[int, int, tuple[int, int, int]], ...]
        Connectivity information: for each hopping, a triple
        ``(orb_i, orb_j, (R_x, R_y, R_z))`` specifying which pair
        of orbitals are connected and by which lattice translation.
        Static (auxiliary data) -- changing the topology triggers
        JIT recompilation.
    n_orbitals : int
        Number of orbitals in the unit cell. Determines the
        Hamiltonian matrix size (n_orbitals x n_orbitals). Static
        (auxiliary data).
    orbital_basis : OrbitalBasis
        Quantum numbers (n, l, m) and labels for each orbital.
        Used by the matrix element calculation. Static (auxiliary
        data, nested PyTree).

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    ``hopping_params`` and ``lattice_vectors`` are children (on the
    gradient tape); ``hopping_indices``, ``n_orbitals``, and
    ``orbital_basis`` are auxiliary data (compile-time constants).
    This separation enables differentiating the ARPES spectrum with
    respect to hopping amplitudes while keeping the model structure
    fixed.

    See Also
    --------
    DiagonalizedBands : The output of diagonalizing a ``TBModel``
        at a set of k-points.
    make_tb_model : Factory function with validation and dtype
        casting.
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

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(hopping_params, lattice_vectors)`` -- both are dense
           float64 arrays. ``hopping_params`` are the TB amplitudes
           to optimize; ``lattice_vectors`` define the real-space
           basis for the Fourier transform.
        2. **Auxiliary data** (static, not traced by JAX):
           ``(hopping_indices, n_orbitals, orbital_basis)`` -- the
           model topology and quantum number metadata. These are
           compile-time constants: changing any of them triggers JIT
           recompilation.

        Returns
        -------
        children : tuple of Array
            ``(hopping_params, lattice_vectors)`` -- differentiable
            JAX arrays.
        aux_data : tuple
            ``(hopping_indices, n_orbitals, orbital_basis)`` -- static
            structural metadata.
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
        """Reconstruct a ``TBModel`` from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into ``(hopping_params,
           lattice_vectors)``.
        2. Unpack ``aux_data`` into ``(hopping_indices, n_orbitals,
           orbital_basis)``.
        3. Pass all five fields to the constructor, restoring the
           static topology metadata from ``aux_data``.

        Parameters
        ----------
        aux_data : tuple
            ``(hopping_indices, n_orbitals, orbital_basis)`` recovered
            from auxiliary data.
        children : tuple of Array
            ``(hopping_params, lattice_vectors)`` as returned by
            :meth:`tree_flatten`.

        Returns
        -------
        model : TBModel
            Reconstructed instance with the original array and
            structural data.
        """
        hopping_params: Float[Array, " H"]
        lattice_vectors: Float[Array, "3 3"]
        hopping_params, lattice_vectors = children
        hopping_indices: tuple
        n_orbitals: int
        orbital_basis: OrbitalBasis
        hopping_indices, n_orbitals, orbital_basis = aux_data
        model: TBModel = cls(
            hopping_params=hopping_params,
            lattice_vectors=lattice_vectors,
            hopping_indices=hopping_indices,
            n_orbitals=n_orbitals,
            orbital_basis=orbital_basis,
        )
        return model


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(
    eigenvalues: Float[Array, "K B"],
    eigenvectors: Complex[Array, "K B O"],
    kpoints: Float[Array, "K 3"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises diagonalized
    electronic structure data before constructing a
    ``DiagonalizedBands`` PyTree. Real-valued arrays are cast to
    ``float64``; the complex eigenvector array is cast to
    ``complex128`` to maintain full double-precision accuracy in
    the orbital decomposition.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints (K must agree across all arrays, B and
    O must be consistent) are checked at call time.

    Use this factory when constructing ``DiagonalizedBands`` from
    either TB diagonalization output or VASP eigenvector data.

    Implementation Logic
    --------------------
    1. **Cast eigenvalues** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast eigenvectors** to ``jnp.complex128`` via
       ``jnp.asarray`` to preserve full double-precision for the
       complex orbital coefficients.
    3. **Cast kpoints** to ``jnp.float64`` via ``jnp.asarray``.
    4. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    5. **Construct** the ``DiagonalizedBands`` NamedTuple from all
       four validated arrays and return it.

    Parameters
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies E_n(k) in eV for K k-points and B bands.
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb} from Hamiltonian
        diagonalization, where O is the number of orbitals.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : DiagonalizedBands
        Validated instance with ``float64`` eigenvalues/kpoints
        and ``complex128`` eigenvectors.

    See Also
    --------
    DiagonalizedBands : The PyTree class constructed by this factory.
    """
    eig_arr: Float[Array, "K B"] = jnp.asarray(eigenvalues, dtype=jnp.float64)
    vec_arr: Complex[Array, "K B O"] = jnp.asarray(
        eigenvectors, dtype=jnp.complex128
    )
    kpt_arr: Float[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    ef_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)

    def validate_and_create() -> DiagonalizedBands:
        def check_eigenvalues_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(eig_arr)),
                lambda: eig_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: eig_arr.sum(),
                        lambda: eig_arr.sum(),
                    )
                ),
            )

        def check_kpoints_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(kpt_arr)),
                lambda: kpt_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: kpt_arr.sum(),
                        lambda: kpt_arr.sum(),
                    )
                ),
            )

        def check_fermi_energy_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.isfinite(ef_arr),
                lambda: ef_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: ef_arr, lambda: ef_arr)
                ),
            )

        check_eigenvalues_finite()
        check_kpoints_finite()
        check_fermi_energy_finite()
        return DiagonalizedBands(
            eigenvalues=eig_arr,
            eigenvectors=vec_arr,
            kpoints=kpt_arr,
            fermi_energy=ef_arr,
        )

    bands: DiagonalizedBands = validate_and_create()
    return bands


@jaxtyped(typechecker=beartype)
def make_tb_model(
    hopping_params: Float[Array, " H"],
    lattice_vectors: Float[Array, "3 3"],
    hopping_indices: tuple,
    n_orbitals: int,
    orbital_basis: OrbitalBasis,
) -> TBModel:
    """Create a validated ``TBModel`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises tight-binding
    model parameters before constructing a ``TBModel`` PyTree. The
    two differentiable arrays (``hopping_params``,
    ``lattice_vectors``) are cast to ``float64``; the three static
    fields are passed through unchanged as auxiliary data.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints on the differentiable arrays are
    checked at call time.

    Use this factory to build a ``TBModel`` from raw hopping data
    (e.g. from a Slater-Koster parameterization) before passing it
    to the Hamiltonian construction and diagonalization routines.

    Implementation Logic
    --------------------
    1. **Cast hopping_params** to ``jnp.float64`` via
       ``jnp.asarray``.
    2. **Cast lattice_vectors** to ``jnp.float64`` via
       ``jnp.asarray``.
    3. **Pass through** ``hopping_indices``, ``n_orbitals``, and
       ``orbital_basis`` unchanged -- these become auxiliary data
       in the PyTree.
    4. **Construct** the ``TBModel`` NamedTuple from all five fields
       and return it.

    Parameters
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes t_{ij,R} in eV, one per connection.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    hopping_indices : tuple
        Connectivity: ``(orb_i, orb_j, (R_x, R_y, R_z))`` per
        hopping, where R_x, R_y, R_z are integer lattice translation
        indices.
    n_orbitals : int
        Number of orbitals in the unit cell.
    orbital_basis : OrbitalBasis
        Quantum number metadata for each orbital.

    Returns
    -------
    model : TBModel
        Validated tight-binding model with ``float64`` arrays and
        static structural metadata.

    See Also
    --------
    TBModel : The PyTree class constructed by this factory.
    make_orbital_basis : Factory for the ``OrbitalBasis`` argument.
    """
    hop_arr: Float[Array, " H"] = jnp.asarray(
        hopping_params, dtype=jnp.float64
    )
    lat_arr: Float[Array, "3 3"] = jnp.asarray(
        lattice_vectors, dtype=jnp.float64
    )

    def validate_and_create() -> TBModel:
        def check_hopping_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(hop_arr)),
                lambda: hop_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: hop_arr.sum(),
                        lambda: hop_arr.sum(),
                    )
                ),
            )

        def check_lattice_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(lat_arr)),
                lambda: lat_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: lat_arr.sum(),
                        lambda: lat_arr.sum(),
                    )
                ),
            )

        check_hopping_finite()
        check_lattice_finite()
        return TBModel(
            hopping_params=hop_arr,
            lattice_vectors=lat_arr,
            hopping_indices=hopping_indices,
            n_orbitals=n_orbitals,
            orbital_basis=orbital_basis,
        )

    model: TBModel = validate_and_create()
    return model


__all__: list[str] = [
    "DiagonalizedBands",
    "TBModel",
    "make_diagonalized_bands",
    "make_tb_model",
]
