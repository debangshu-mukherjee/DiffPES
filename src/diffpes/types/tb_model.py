"""Define tight-binding model and diagonalized-band data structures.

Extended Summary
----------------
This module defines the native tight-binding carrier and the diagonalized
electronic-structure interface consumed by later ARPES stages. Tight-binding
connectivity is exact static metadata, while energies, complex amplitudes,
geometry, and eigensystems remain differentiable JAX leaves.

Routine Listings
----------------
:class:`DiagonalizedBands`
    Store diagonalized electronic-structure data in a JAX PyTree.
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.

Notes
-----
The tight-binding phase convention is the basis-position gauge. Each physical
fractional bond displacement follows ``R + tau_j - tau_i`` from exact
integer-cell metadata and either explicit orbital centres or atomic positions.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from .aliases import ScalarNumeric
from .geometry import CrystalGeometry
from .radial_params import OrbitalBasis

_HERMITICITY_TOLERANCE: float = 1e-12
_PAIR_LENGTH: int = 2
_CELL_COMPONENTS: int = 3
_EIGENVALUE_NDIM: int = 2
_EIGENVECTOR_NDIM: int = 3
_ORBITAL_POSITION_NDIM: int = 2


def _validate_basis_geometry(
    basis: OrbitalBasis,
    geometry: CrystalGeometry,
) -> None:
    """Validate the orbital-to-atom mapping against a geometry."""
    n_atoms: int = geometry.positions.shape[0]
    if any(index >= n_atoms for index in basis.atom_indices):
        message: str = (
            "basis atom_indices must refer to geometry.positions rows"
        )
        raise ValueError(message)


def _validate_hopping_metadata(
    hopping_pairs: tuple[tuple[int, int], ...],
    hopping_cells: tuple[tuple[int, int, int], ...],
    n_orbitals: int,
) -> tuple[int, ...]:
    """Validate exact connectivity and derive its closure permutation."""
    keys: list[tuple[int, int, tuple[int, int, int]]] = []
    pair: tuple[int, int]
    cell: tuple[int, int, int]
    for pair, cell in zip(hopping_pairs, hopping_cells, strict=True):
        if (
            type(pair) is not tuple
            or len(pair) != _PAIR_LENGTH
            or any(type(index) is not int for index in pair)
        ):
            message: str = "hopping_pairs must contain pairs of integers"
            raise ValueError(message)
        if (
            type(cell) is not tuple
            or len(cell) != _CELL_COMPONENTS
            or any(type(component) is not int for component in cell)
        ):
            message = "hopping_cells must contain integer triples"
            raise ValueError(message)
        if any(index < 0 or index >= n_orbitals for index in pair):
            message = "hopping pair indices must be in [0, n_orbitals)"
            raise ValueError(message)
        keys.append((pair[0], pair[1], cell))

    if len(set(keys)) != len(keys):
        message = "duplicate (i, j, R) hopping records are not allowed"
        raise ValueError(message)

    buckets: dict[
        tuple[int, int, tuple[int, int, int]],
        list[int],
    ] = {}
    index: int
    key: tuple[int, int, tuple[int, int, int]]
    for index, key in enumerate(keys):
        buckets.setdefault(key, []).append(index)

    closure: list[int] = [-1] * len(keys)
    indices: list[int]
    for key, indices in buckets.items():
        orbital_i: int
        orbital_j: int
        orbital_i, orbital_j, cell = key
        reverse_key: tuple[int, int, tuple[int, int, int]] = (
            orbital_j,
            orbital_i,
            (-cell[0], -cell[1], -cell[2]),
        )
        reverse_indices: list[int] | None = buckets.get(reverse_key)
        if reverse_indices is None or len(reverse_indices) != len(indices):
            message = (
                "hopping metadata must be Hermitian-closed with one "
                "(j, i, -R) partner per (i, j, R) entry"
            )
            raise ValueError(message)
        occurrence: int
        for occurrence, index in enumerate(indices):
            closure[index] = reverse_indices[occurrence]
    closure_permutation: tuple[int, ...] = tuple(closure)
    return closure_permutation


def _validate_shell_metadata(
    soc_lambdas: Float[Array, " n_shells"],
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
) -> None:
    """Validate contiguous atomic-shell identifiers and their groups."""
    if any(type(index) is not int or index < -1 for index in shell_index):
        message: str = (
            "shell_index entries must be integers greater than or equal to -1"
        )
        raise ValueError(message)
    expected_shells: int = max(shell_index, default=-1) + 1
    if soc_lambdas.shape[0] != expected_shells:
        message = (
            "soc_lambdas length must equal max(shell_index) + 1, with -1 "
            "denoting no shell"
        )
        raise ValueError(message)
    active_shells: set[int] = {index for index in shell_index if index >= 0}
    if active_shells != set(range(expected_shells)):
        message = "nonnegative shell_index IDs must be contiguous from 0"
        raise ValueError(message)

    shell_groups: dict[int, tuple[int, int, int]] = {}
    group_shells: dict[tuple[int, int, int], int] = {}
    orbital: int
    shell: int
    for orbital, shell in enumerate(shell_index):
        if shell < 0:
            continue
        group: tuple[int, int, int] = (
            basis.atom_indices[orbital],
            basis.n[orbital],
            basis.l[orbital],
        )
        existing_group: tuple[int, int, int] | None = shell_groups.get(shell)
        if existing_group is not None and existing_group != group:
            message = "each shell_index ID must map to one (atom, n, l) group"
            raise ValueError(message)
        existing_shell: int | None = group_shells.get(group)
        if existing_shell is not None and existing_shell != shell:
            message = "each (atom, n, l) group must map to one shell_index ID"
            raise ValueError(message)
        shell_groups[shell] = group
        group_shells[group] = shell


def _validate_tb_structure(
    hopping_amplitudes: Complex[Array, " n_hop"],
    onsite_energies: Float[Array, " n_orb"],
    soc_lambdas: Float[Array, " n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: tuple[tuple[int, int], ...],
    hopping_cells: tuple[tuple[int, int, int], ...],
    shell_index: tuple[int, ...],
    spinor: bool,
    orbital_positions: Optional[Float[Array, "n_orb 3"]],
) -> tuple[int, ...]:
    """Validate static tight-binding structure and return reverse indices."""
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    if not isinstance(basis, OrbitalBasis):
        message = "basis must be an OrbitalBasis"
        raise ValueError(message)
    if any(
        type(values) is not tuple
        for values in (hopping_pairs, hopping_cells, shell_index)
    ):
        message = (
            "hopping_pairs, hopping_cells, and shell_index must be tuples"
        )
        raise ValueError(message)
    if hopping_amplitudes.ndim != 1:
        message = "hopping_amplitudes must be one-dimensional"
        raise ValueError(message)
    if onsite_energies.ndim != 1:
        message = "onsite_energies must be one-dimensional"
        raise ValueError(message)
    if soc_lambdas.ndim != 1:
        message = "soc_lambdas must be one-dimensional"
        raise ValueError(message)

    n_hoppings: int = hopping_amplitudes.shape[0]
    n_orbitals: int = onsite_energies.shape[0]
    if len(hopping_pairs) != n_hoppings or len(hopping_cells) != n_hoppings:
        message = (
            "hopping_amplitudes, hopping_pairs, and hopping_cells must have "
            "the same length"
        )
        raise ValueError(message)
    if len(basis.n) != n_orbitals or len(shell_index) != n_orbitals:
        message = (
            "onsite_energies, basis, and shell_index must have the same "
            "orbital count"
        )
        raise ValueError(message)
    _validate_basis_geometry(basis, geometry)
    if orbital_positions is not None and (
        orbital_positions.ndim != _ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (n_orbitals, _CELL_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)

    _validate_shell_metadata(soc_lambdas, basis, shell_index)
    if type(spinor) is not bool:
        message = "spinor must be a bool"
        raise ValueError(message)
    if spinor and (
        len(basis.spin) != n_orbitals
        or any(channel not in (-1, 1) for channel in basis.spin)
    ):
        message = "spinor models require one +1 or -1 basis spin per orbital"
        raise ValueError(message)
    if not spinor and basis.spin:
        message = "spinless models require an empty basis spin tuple"
        raise ValueError(message)

    closure: tuple[int, ...] = _validate_hopping_metadata(
        hopping_pairs,
        hopping_cells,
        n_orbitals,
    )
    return closure


def _checked_geometry(
    geometry: CrystalGeometry, context: str
) -> CrystalGeometry:
    """Attach finite-value runtime checks to every geometry array leaf."""
    lattice: Float[Array, "3 3"] = eqx.error_if(
        geometry.lattice,
        ~jnp.all(jnp.isfinite(geometry.lattice)),
        f"{context}: geometry lattice finite",
    )
    reciprocal: Float[Array, "3 3"] = eqx.error_if(
        geometry.reciprocal,
        ~jnp.all(jnp.isfinite(geometry.reciprocal)),
        f"{context}: geometry reciprocal finite",
    )
    positions: Float[Array, "n_atoms 3"] = eqx.error_if(
        geometry.positions,
        ~jnp.all(jnp.isfinite(geometry.positions)),
        f"{context}: geometry positions finite",
    )
    checked: CrystalGeometry = eqx.tree_at(
        lambda item: (item.lattice, item.reciprocal, item.positions),
        geometry,
        (lattice, reciprocal, positions),
    )
    return checked


class DiagonalizedBands(eqx.Module):
    """Store diagonalized electronic-structure data in a JAX PyTree.

    The carrier is the tight-binding-to-ARPES interface. Geometry and orbital
    metadata travel with each eigensystem so later matrix-element stages can
    form Cartesian momenta, atomic interference phases, and orbital operators.

    :see: :class:`~.test_tb_model.TestDiagonalizedBands`

    Attributes
    ----------
    eigenvalues : Float[Array, "n_k n_bands"]
        Band energies in eV.
    eigenvectors : Complex[Array, "n_k n_bands n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float[Array, "n_k 3"]
        Fractional reciprocal-space coordinates.
    fermi_energy : Float[Array, ""]
        Fermi energy in eV.
    geometry : CrystalGeometry
        Crystal geometry. Its numerical fields are differentiable children.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    orbital_positions : Optional[Float[Array, "n_orb 3"]]
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` ties centres to atoms.

    Notes
    -----
    The numerical eigensystem, geometry, and optional orbital-position fields
    remain JAX leaves. ``basis`` is static because its quantum numbers and
    atom mapping shape compiled operator construction.

    See Also
    --------
    TBModel : Tight-binding carrier whose diagonalization produces bands.
    make_diagonalized_bands : Validating carrier factory.
    """

    eigenvalues: Float[Array, "n_k n_bands"]
    eigenvectors: Complex[Array, "n_k n_bands n_orb"]
    kpoints: Float[Array, "n_k 3"]
    fermi_energy: Float[Array, ""]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    orbital_positions: Optional[Float[Array, "n_orb 3"]] = None

    def __check_init__(self) -> None:
        """Validate the static eigensystem invariants again."""
        _validate_diagonalized_structure(
            self.eigenvalues,
            self.eigenvectors,
            self.kpoints,
            self.fermi_energy,
            self.geometry,
            self.basis,
            self.orbital_positions,
        )


class TBModel(eqx.Module):
    r"""Store tight-binding parameters in a JAX PyTree.

    Each hopping record is ``(i, j, R, t)`` with exact integer lattice
    translation ``R``. In the pinned basis-position gauge, Hamiltonian phases
    use the physical fractional displacement
    :math:`R + \tau_j - \tau_i`. Explicit ``orbital_positions`` provide one
    :math:`\tau` per Wannier orbital. Otherwise they are derived from
    ``geometry.positions`` and ``basis.atom_indices``. Physical displacements
    are never stored in place of ``R`` or rounded back into connectivity.

    :see: :class:`~.test_tb_model.TestTBModel`

    Attributes
    ----------
    hopping_amplitudes : Complex[Array, "n_hop"]
        Complex hopping amplitudes in eV. These differentiable values support
        spin-orbit and other intrinsically complex couplings.
    onsite_energies : Float[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float[Array, "n_shells"]
        Atomic spin-orbit couplings in eV, one per ``(atom, n, l)`` shell.
    geometry : CrystalGeometry
        Differentiable lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : tuple[tuple[int, int], ...]
        Directed orbital pairs ``(i, j)`` (**static** -- changing them triggers
        retracing).
    hopping_cells : tuple[tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : tuple[int, ...]
        Orbital-to-SOC-shell mapping; ``-1`` means no shell. Nonnegative IDs
        are contiguous and map one-to-one to ``(atom, n, l)`` groups, with
        spin copies sharing an ID (**static** -- changing it triggers
        retracing).
    spinor : bool
        Whether the basis carries explicit spin channels (**static** --
        changing it triggers retracing).
    orbital_positions : Optional[Float[Array, "n_orb 3"]]
        Explicit fractional orbital or Wannier centres. ``None`` ties every
        orbital centre to its assigned atomic position. Explicit centres are
        differentiable independently of the atomic geometry.

    Notes
    -----
    Hopping metadata excludes duplicate ``(i, j, R)`` records and includes a
    ``(j, i, -R)`` partner for every entry. The factory checks corresponding
    amplitudes elementwise against their complex conjugates. Hamiltonian
    assembly therefore needs no Hermitianization repair.

    See Also
    --------
    DiagonalizedBands : Eigensystem carrier produced from this model.
    make_tb_model : Validating carrier factory.
    """

    hopping_amplitudes: Complex[Array, " n_hop"]
    onsite_energies: Float[Array, " n_orb"]
    soc_lambdas: Float[Array, " n_shells"]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    hopping_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    hopping_cells: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    shell_index: tuple[int, ...] = eqx.field(static=True)
    spinor: bool = eqx.field(static=True)
    orbital_positions: Optional[Float[Array, "n_orb 3"]] = None

    def __check_init__(self) -> None:
        """Validate the static tight-binding invariants again."""
        _validate_tb_structure(
            self.hopping_amplitudes,
            self.onsite_energies,
            self.soc_lambdas,
            self.geometry,
            self.basis,
            self.hopping_pairs,
            self.hopping_cells,
            self.shell_index,
            self.spinor,
            self.orbital_positions,
        )


def _validate_diagonalized_structure(
    eigenvalues: Float[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float[Array, "n_k_p 3"],
    fermi_energy: Float[Array, ""],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    orbital_positions: Optional[Float[Array, "n_orb 3"]],
) -> None:
    """Validate static eigensystem shapes and context."""
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    if not isinstance(basis, OrbitalBasis):
        message = "basis must be an OrbitalBasis"
        raise ValueError(message)
    if eigenvalues.ndim != _EIGENVALUE_NDIM:
        message = "eigenvalues must be two-dimensional"
        raise ValueError(message)
    if eigenvectors.ndim != _EIGENVECTOR_NDIM:
        message = "eigenvectors must be three-dimensional"
        raise ValueError(message)
    if (
        kpoints.ndim != _EIGENVALUE_NDIM
        or kpoints.shape[1] != _CELL_COMPONENTS
    ):
        message = "kpoints must have shape (n_k, 3)"
        raise ValueError(message)
    if fermi_energy.ndim != 0:
        message = "fermi_energy must be scalar"
        raise ValueError(message)
    if eigenvalues.shape != eigenvectors.shape[:2]:
        message = "eigenvalues and eigenvectors must agree on n_k and n_bands"
        raise ValueError(message)
    if eigenvalues.shape[0] != kpoints.shape[0]:
        message = "eigenvalues and kpoints must agree on n_k"
        raise ValueError(message)
    if eigenvectors.shape[2] != len(basis.n):
        message = "eigenvector orbital axis must match basis"
        raise ValueError(message)
    if orbital_positions is not None and (
        orbital_positions.ndim != _ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (len(basis.n), _CELL_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)
    _validate_basis_geometry(basis, geometry)


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(  # noqa: DOC502, DOC503
    eigenvalues: Float[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float[Array, "n_k_p 3"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    fermi_energy: ScalarNumeric = 0.0,
    orbital_positions: Optional[Float[Array, "n_orb 3"]] = None,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    The factory normalizes every numerical array and validates its axes
    against the supplied geometry and orbital basis.

    :see: :class:`~.test_tb_model.TestMakeDiagonalizedBands`

    Parameters
    ----------
    eigenvalues : Float[Array, "n_k_e n_bands_e"]
        Band energies in eV.
    eigenvectors : Complex[Array, "n_k_v n_bands_v n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float[Array, "n_k_p 3"]
        Fractional reciprocal-space coordinates.
    geometry : CrystalGeometry
        Crystal geometry whose numerical leaves remain differentiable.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    fermi_energy : ScalarNumeric, optional
        Fermi energy in eV. Default is 0.0.
    orbital_positions : Optional[Float[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` derives centres from
        atom assignments. Default is ``None``.

    Returns
    -------
    bands : DiagonalizedBands
        Validated double-precision eigensystem and its structural context.

    Raises
    ------
    ValueError
        If eigensystem axes, basis size, or atom assignments disagree.
    EquinoxRuntimeError
        If any numerical leaf is non-finite.

    Notes
    -----
    Static validation checks array axes and context before tracing. Runtime
    validation uses :func:`equinox.error_if` for every numerical leaf, so the
    same rejection behavior remains active under JIT.

    See Also
    --------
    DiagonalizedBands : Carrier constructed by this factory.
    make_tb_model : Construct the model diagonalized by native TB producers.
    """
    eigenvalue_array: Float[Array, "n_k n_bands"] = jnp.asarray(
        eigenvalues,
        dtype=jnp.float64,
    )
    eigenvector_array: Complex[Array, "n_k n_bands n_orb"] = jnp.asarray(
        eigenvectors,
        dtype=jnp.complex128,
    )
    kpoint_array: Float[Array, "n_k 3"] = jnp.asarray(
        kpoints,
        dtype=jnp.float64,
    )
    fermi_array: Float[Array, ""] = jnp.asarray(
        fermi_energy,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    _validate_diagonalized_structure(
        eigenvalue_array,
        eigenvector_array,
        kpoint_array,
        fermi_array,
        geometry,
        basis,
        orbital_position_array,
    )

    eigenvalue_array = eqx.error_if(
        eigenvalue_array,
        ~jnp.all(jnp.isfinite(eigenvalue_array)),
        "make_diagonalized_bands: eigenvalues finite",
    )
    eigenvector_array = eqx.error_if(
        eigenvector_array,
        ~jnp.all(jnp.isfinite(eigenvector_array)),
        "make_diagonalized_bands: eigenvectors finite",
    )
    kpoint_array = eqx.error_if(
        kpoint_array,
        ~jnp.all(jnp.isfinite(kpoint_array)),
        "make_diagonalized_bands: kpoints finite",
    )
    fermi_array = eqx.error_if(
        fermi_array,
        ~jnp.isfinite(fermi_array),
        "make_diagonalized_bands: fermi energy finite",
    )
    if orbital_position_array is not None:
        orbital_position_array = eqx.error_if(
            orbital_position_array,
            ~jnp.all(jnp.isfinite(orbital_position_array)),
            "make_diagonalized_bands: orbital positions finite",
        )
    checked_geometry: CrystalGeometry = _checked_geometry(
        geometry,
        "make_diagonalized_bands",
    )
    bands: DiagonalizedBands = DiagonalizedBands(
        eigenvalues=eigenvalue_array,
        eigenvectors=eigenvector_array,
        kpoints=kpoint_array,
        fermi_energy=fermi_array,
        geometry=checked_geometry,
        basis=basis,
        orbital_positions=orbital_position_array,
    )
    return bands


@jaxtyped(typechecker=beartype)
def make_tb_model(  # noqa: DOC502, DOC503
    hopping_amplitudes: Complex[Array, " n_hop"],
    onsite_energies: Float[Array, " n_orb"],
    soc_lambdas: Float[Array, " n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: tuple[tuple[int, int], ...],
    hopping_cells: tuple[tuple[int, int, int], ...],
    shell_index: tuple[int, ...],
    spinor: bool = False,
    orbital_positions: Optional[Float[Array, "n_orb 3"]] = None,
) -> TBModel:
    r"""Create a validated ``TBModel`` instance.

    The factory normalizes numerical leaves, validates exact connectivity,
    and enforces complex-conjugate hopping closure under JAX transformations.

    :see: :class:`~.test_tb_model.TestMakeTBModel`

    Parameters
    ----------
    hopping_amplitudes : Complex[Array, "n_hop"]
        Directed hopping amplitudes in eV.
    onsite_energies : Float[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float[Array, "n_shells"]
        Spin-orbit coupling energies in eV, one per atomic shell.
    geometry : CrystalGeometry
        Crystal lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : tuple[tuple[int, int], ...]
        Directed ``(i, j)`` orbital pairs (**static** -- changing them
        triggers retracing).
    hopping_cells : tuple[tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : tuple[int, ...]
        Orbital-to-SOC-shell map with ``-1`` denoting no shell (**static** --
        changing it triggers retracing).
    spinor : bool, optional
        Whether the basis has explicit spin channels (**static** -- changing
        it triggers retracing). Default is ``False``.
    orbital_positions : Optional[Float[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres for the basis-position gauge.
        ``None`` derives centres from atom assignments. Default is ``None``.

    Returns
    -------
    model : TBModel
        Validated complex tight-binding model.

    Raises
    ------
    ValueError
        If structural lengths, indices, spin metadata, shell metadata, or
        Hermiticity closure metadata are inconsistent.
    EquinoxRuntimeError
        If a numerical leaf is non-finite or reverse hopping amplitudes are
        not complex conjugates to absolute tolerance ``1e-12`` eV.

    Notes
    -----
    The algorithm is:

    1. Check static dimensions, unique exact integer metadata, contiguous
       atomic-shell IDs, spin semantics, and closure under
       ``(i, j, R) -> (j, i, -R)``.
    2. Derive a static reverse-entry permutation from exact metadata.
    3. Use :func:`equinox.error_if` to reject non-finite leaves and enforce
       :math:`t_{ji}(-R) = t_{ij}(R)^*` under eager and compiled execution.

    The physical displacement is not stored. Hamiltonian consumers derive
    ``R + tau_j - tau_i`` from explicit orbital positions when present and
    otherwise from assigned atomic positions.

    See Also
    --------
    TBModel : Carrier constructed by this factory.
    make_diagonalized_bands : Construct the downstream eigensystem carrier.
    """
    hopping_array: Complex[Array, " n_hop"] = jnp.asarray(
        hopping_amplitudes,
        dtype=jnp.complex128,
    )
    onsite_array: Float[Array, " n_orb"] = jnp.asarray(
        onsite_energies,
        dtype=jnp.float64,
    )
    soc_array: Float[Array, " n_shells"] = jnp.asarray(
        soc_lambdas,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    closure: tuple[int, ...] = _validate_tb_structure(
        hopping_array,
        onsite_array,
        soc_array,
        geometry,
        basis,
        hopping_pairs,
        hopping_cells,
        shell_index,
        spinor,
        orbital_position_array,
    )

    hopping_array = eqx.error_if(
        hopping_array,
        ~jnp.all(jnp.isfinite(hopping_array)),
        "make_tb_model: hopping amplitudes finite",
    )
    onsite_array = eqx.error_if(
        onsite_array,
        ~jnp.all(jnp.isfinite(onsite_array)),
        "make_tb_model: onsite energies finite",
    )
    soc_array = eqx.error_if(
        soc_array,
        ~jnp.all(jnp.isfinite(soc_array)),
        "make_tb_model: soc lambdas finite",
    )
    if orbital_position_array is not None:
        orbital_position_array = eqx.error_if(
            orbital_position_array,
            ~jnp.all(jnp.isfinite(orbital_position_array)),
            "make_tb_model: orbital positions finite",
        )
    closure_indices: Int[Array, " n_hop"] = jnp.asarray(
        closure,
        dtype=jnp.int32,
    )
    reverse_amplitudes: Complex[Array, " n_hop"] = hopping_array[
        closure_indices
    ]
    closure_error: Float[Array, " n_hop"] = jnp.abs(
        reverse_amplitudes - jnp.conj(hopping_array)
    )
    hopping_array = eqx.error_if(
        hopping_array,
        ~jnp.all(closure_error <= _HERMITICITY_TOLERANCE),
        "make_tb_model: reverse hopping amplitudes must be complex conjugates",
    )
    checked_geometry: CrystalGeometry = _checked_geometry(
        geometry,
        "make_tb_model",
    )
    model: TBModel = TBModel(
        hopping_amplitudes=hopping_array,
        onsite_energies=onsite_array,
        soc_lambdas=soc_array,
        geometry=checked_geometry,
        basis=basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=shell_index,
        spinor=spinor,
        orbital_positions=orbital_position_array,
    )
    return model


__all__: list[str] = [
    "DiagonalizedBands",
    "TBModel",
    "make_diagonalized_bands",
    "make_tb_model",
]
