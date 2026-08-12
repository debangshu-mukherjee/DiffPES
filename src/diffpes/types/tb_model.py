"""Define differentiable tight-binding model parameters.

Extended Summary
----------------
This module stores tight-binding amplitudes, onsite energies, spin-
orbit parameters, and exact connectivity metadata.

Routine Listings
----------------
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    DEPTH_TOLERANCE_ANG,
    ORBITAL_POSITION_NDIM,
    TB_HERMITICITY_TOLERANCE,
    TB_PAIR_LENGTH,
)

from .electronic_structure_validation import (
    _checked_geometry,
    _validate_basis_geometry,
    _validate_depths_shape,
)
from .geometry import CrystalGeometry
from .orbital_basis import OrbitalBasis


def _validate_hopping_metadata(
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    n_orbitals: int,
) -> Tuple[int, ...]:
    """PRIVATE: Validate exact connectivity and derive its closure permutation.

    Implementation Logic
    --------------------
    Build the ``(i, j, R)`` key of every record, then bucket record
    indices by key. For every key require a reversed-key bucket of
    equal size and map matching occurrences positionally. The resulting
    permutation lets the factory compare each amplitude with the
    conjugate of its Hermitian partner.

    Parameters
    ----------
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)``, one per hopping record.
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations ``R``, one per record.
    n_orbitals : int
        Number of orbitals that bounds the pair indices.

    Returns
    -------
    closure_permutation : Tuple[int, ...]
        Index of the ``(j, i, -R)`` partner record for every
        ``(i, j, R)`` record.

    Raises
    ------
    ValueError
        If a pair or cell has invalid integer metadata. If a pair index
        lies outside ``[0, n_orbitals)``. If duplicate records exist.
        If any record lacks a ``(j, i, -R)`` partner. This is the static
        construction-time contract.
    """
    keys: List[Tuple[int, int, Tuple[int, int, int]]] = []
    pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for pair, cell in zip(hopping_pairs, hopping_cells, strict=True):
        if (
            type(pair) is not tuple
            or len(pair) != TB_PAIR_LENGTH
            or any(type(index) is not int for index in pair)
        ):
            message: str = "hopping_pairs must contain pairs of integers"
            raise ValueError(message)
        if (
            type(cell) is not tuple
            or len(cell) != CARTESIAN_COMPONENTS
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

    buckets: Dict[
        Tuple[int, int, Tuple[int, int, int]],
        List[int],
    ] = {}
    index: int
    key: Tuple[int, int, Tuple[int, int, int]]
    for index, key in enumerate(keys):
        buckets.setdefault(key, []).append(index)

    closure: List[int] = [-1] * len(keys)
    indices: List[int]
    for key, indices in buckets.items():
        orbital_i: int
        orbital_j: int
        orbital_i, orbital_j, cell = key
        reverse_key: Tuple[int, int, Tuple[int, int, int]] = (
            orbital_j,
            orbital_i,
            (-cell[0], -cell[1], -cell[2]),
        )
        reverse_indices: List[int] | None = buckets.get(reverse_key)
        if reverse_indices is None or len(reverse_indices) != len(indices):
            message = (
                "hopping metadata must be Hermitian-closed with one "
                "(j, i, -R) partner per (i, j, R) entry"
            )
            raise ValueError(message)
        occurrence: int
        for occurrence, index in enumerate(indices):
            closure[index] = reverse_indices[occurrence]
    closure_permutation: Tuple[int, ...] = tuple(closure)
    return closure_permutation


def _validate_shell_metadata(
    soc_lambdas: Float64[Array, " n_shells"],
    basis: OrbitalBasis,
    shell_index: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate contiguous atomic-shell identifiers and their groups.

    Implementation Logic
    --------------------
    Walk every orbital with a nonnegative shell and record its
    ``(atom_index, n, l)`` group in two dictionaries. Reject a shell
    that mixes groups and a group that is split across shells. Spin
    copies of one group therefore share one ``soc_lambdas`` entry.

    Parameters
    ----------
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin-orbit couplings in eV, one per shell.
    basis : OrbitalBasis
        Orbital metadata that provides ``(atom, n, l)`` per orbital.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map; ``-1`` denotes no shell.

    Raises
    ------
    ValueError
        If ``shell_index`` contains an integer below ``-1``. If
        ``soc_lambdas`` has the wrong length. If nonnegative identifiers
        are not contiguous from zero. If shells and ``(atom, n, l)``
        groups lack a one-to-one map. This is the static
        construction-time contract.
    """
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

    shell_groups: Dict[int, Tuple[int, int, int]] = {}
    group_shells: Dict[Tuple[int, int, int], int] = {}
    orbital: int
    shell: int
    for orbital, shell in enumerate(shell_index):
        if shell < 0:
            continue
        group: Tuple[int, int, int] = (
            basis.atom_indices[orbital],
            basis.n[orbital],
            basis.l[orbital],
        )
        existing_group: Tuple[int, int, int] | None = shell_groups.get(shell)
        if existing_group is not None and existing_group != group:
            message = "each shell_index ID must map to one (atom, n, l) group"
            raise ValueError(message)
        existing_shell: int | None = group_shells.get(group)
        if existing_shell is not None and existing_shell != shell:
            message = "each (atom, n, l) group must map to one shell_index ID"
            raise ValueError(message)
        shell_groups[shell] = group
        group_shells[group] = shell


def _validate_tb_structure(  # noqa: PLR0913, PLR0917
    hopping_amplitudes: Complex128[Array, " n_hop"],
    onsite_energies: Float64[Array, " n_orb"],
    soc_lambdas: Float64[Array, " n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    shell_index: Tuple[int, ...],
    spinor: bool,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]],
    depths: Optional[Float64[Array, " n_depth"]],
) -> Tuple[int, ...]:
    """PRIVATE: Validate tight-binding structure and return its closure.

    Implementation Logic
    --------------------
    Check types, ranks, and axis agreements first. Then delegate to
    ``_validate_basis_geometry``, ``_validate_depths_shape``,
    ``_validate_shell_metadata``, and ``_validate_hopping_metadata``,
    and return the closure permutation of the last one.

    Parameters
    ----------
    hopping_amplitudes : Complex128[Array, " n_hop"]
        Complex hopping amplitudes in eV, one per hopping record.
    onsite_energies : Float64[Array, " n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, " n_shells"]
        Atomic spin-orbit couplings in eV, one per shell.
    geometry : CrystalGeometry
        Differentiable lattice and atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata.
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)``, one per record.
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer lattice translations ``R``, one per record.
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map; ``-1`` denotes no shell.
    spinor : bool
        Whether the basis carries explicit spin channels.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres, or ``None``.
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom, or ``None`` for a bulk model.

    Returns
    -------
    closure : Tuple[int, ...]
        Hermitian-closure permutation: the index of the ``(j, i, -R)``
        partner record for every hopping record.

    Raises
    ------
    ValueError
        If geometry, basis, or connectivity metadata has the wrong
        type. If a numerical array has the wrong rank or length. If
        ``orbital_positions`` has the wrong shape. If ``spinor`` is not
        Boolean or contradicts the spin channels. The delegated
        validators raise ``ValueError`` for their own static contracts.
    """
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
        orbital_positions.ndim != ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (n_orbitals, CARTESIAN_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)
    _validate_depths_shape(depths, n_orbitals)

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

    closure: Tuple[int, ...] = _validate_hopping_metadata(
        hopping_pairs,
        hopping_cells,
        n_orbitals,
    )
    return closure


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
    hopping_amplitudes : Complex128[Array, "n_hop"]
        Complex hopping amplitudes in eV. These differentiable values support
        spin-orbit and other intrinsically complex couplings.
    onsite_energies : Float64[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, "n_shells"]
        Atomic spin-orbit couplings in eV, one per ``(atom, n, l)`` shell.
    geometry : CrystalGeometry
        Differentiable lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed orbital pairs ``(i, j)`` (**static** -- changing them triggers
        retracing).
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell mapping; ``-1`` means no shell. Nonnegative IDs
        are contiguous and map one-to-one to ``(atom, n, l)`` groups, with
        spin copies sharing an ID (**static** -- changing it triggers
        retracing).
    spinor : bool
        Whether the basis carries explicit spin channels (**static** --
        changing it triggers retracing).
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital or Wannier centres. ``None`` ties every
        orbital centre to its assigned atomic position. Explicit centres are
        differentiable independently of the atomic geometry.
    depths : Optional[Float64[Array, "n_orb"]]
        Orbital depths in Angstrom below the top surface. Values are finite
        and nonnegative up to the numerical boundary tolerance. ``None``
        denotes a bulk model.

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

    hopping_amplitudes: Complex128[Array, " n_hop"]
    onsite_energies: Float64[Array, " n_orb"]
    soc_lambdas: Float64[Array, " n_shells"]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    hopping_pairs: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    hopping_cells: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    shell_index: Tuple[int, ...] = eqx.field(static=True)
    spinor: bool = eqx.field(static=True)
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None
    depths: Optional[Float64[Array, " n_orb"]] = None

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
            self.depths,
        )


@jaxtyped(typechecker=beartype)
def make_tb_model(  # noqa: DOC502, DOC503, PLR0913, PLR0917
    hopping_amplitudes: Complex128[Array, "n_hop"],
    onsite_energies: Float64[Array, "n_orb"],
    soc_lambdas: Float64[Array, "n_shells"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    hopping_pairs: Tuple[Tuple[int, int], ...],
    hopping_cells: Tuple[Tuple[int, int, int], ...],
    shell_index: Tuple[int, ...],
    spinor: bool = False,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
    depths: Optional[Float64[Array, " n_depth"]] = None,
) -> TBModel:
    r"""Create a validated ``TBModel`` instance.

    The factory normalizes numerical leaves, validates exact connectivity,
    and enforces complex-conjugate hopping closure under JAX transformations.

    :see: :class:`~.test_tb_model.TestMakeTBModel`

    Parameters
    ----------
    hopping_amplitudes : Complex128[Array, "n_hop"]
        Directed hopping amplitudes in eV.
    onsite_energies : Float64[Array, "n_orb"]
        Onsite orbital energies in eV.
    soc_lambdas : Float64[Array, "n_shells"]
        Spin-orbit coupling energies in eV, one per atomic shell.
    geometry : CrystalGeometry
        Crystal lattice and fractional atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata (**static** -- changing it
        triggers retracing).
    hopping_pairs : Tuple[Tuple[int, int], ...]
        Directed ``(i, j)`` orbital pairs (**static** -- changing them
        triggers retracing).
    hopping_cells : Tuple[Tuple[int, int, int], ...]
        Exact integer translations ``R`` (**static** -- changing them triggers
        retracing).
    shell_index : Tuple[int, ...]
        Orbital-to-SOC-shell map with ``-1`` denoting no shell (**static** --
        changing it triggers retracing).
    spinor : bool, optional
        Whether the basis has explicit spin channels (**static** -- changing
        it triggers retracing). Default is ``False``.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres for the basis-position gauge.
        ``None`` derives centres from atom assignments. Default is ``None``.
    depths : Optional[Float64[Array, "n_depth"]], optional
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Default is ``None``.

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
    hopping_array: Complex128[Array, " n_hop"] = jnp.asarray(
        hopping_amplitudes,
        dtype=jnp.complex128,
    )
    onsite_array: Float64[Array, " n_orb"] = jnp.asarray(
        onsite_energies,
        dtype=jnp.float64,
    )
    soc_array: Float64[Array, " n_shells"] = jnp.asarray(
        soc_lambdas,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float64[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    depth_array: Optional[Float64[Array, " n_depth"]] = None
    if depths is not None:
        depth_array = jnp.asarray(depths, dtype=jnp.float64)
    closure: Tuple[int, ...] = _validate_tb_structure(
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
        depth_array,
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
    if depth_array is not None:
        depth_array = eqx.error_if(
            depth_array,
            ~jnp.all(jnp.isfinite(depth_array)),
            "make_tb_model: depths must be finite",
        )
        depth_array = eqx.error_if(
            depth_array,
            jnp.any(depth_array < -DEPTH_TOLERANCE_ANG),
            "make_tb_model: depths must be nonnegative",
        )
    closure_indices: Int32[Array, " n_hop"] = jnp.asarray(
        closure,
        dtype=jnp.int32,
    )
    reverse_amplitudes: Complex128[Array, " n_hop"] = hopping_array[
        closure_indices
    ]
    closure_error: Float64[Array, " n_hop"] = jnp.abs(
        reverse_amplitudes - jnp.conj(hopping_array)
    )
    hopping_array = eqx.error_if(
        hopping_array,
        ~jnp.all(closure_error <= TB_HERMITICITY_TOLERANCE),
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
        depths=depth_array,
    )
    return model


__all__: list[str] = [
    "TBModel",
    "make_tb_model",
]
