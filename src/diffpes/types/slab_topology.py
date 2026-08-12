"""Define host-selected slab topology metadata.

Extended Summary
----------------
This module stores discrete slab topology for pure-JAX rebuilding
after host-side surface selection.

Routine Listings
----------------
:class:`SlabTopology`
    Store host-selected discrete slab topology for pure-JAX rebuilding.
:func:`make_slab_topology`
    Create validated host-selected slab topology metadata.
"""

import math

import equinox as eqx
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped

from diffpes.constants import SURFACE_VECTOR_COUNT

from .slab_geometry import (
    _integer_determinant,
    _integer_dot,
    _validate_integer_triple,
)


def _validate_slab_topology_structure(  # noqa: DOC503, PLR0912, PLR0913, PLR0915
    *,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
    atom_shifts: Tuple[Tuple[int, int, int], ...],
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
    termination: Tuple[str, str],
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    n_layers: int,
    bulk_atom_count: int,
    basis_atom_indices: Tuple[int, ...],
) -> None:
    """PRIVATE: Validate static topology selected for a slab rebuild.

    Parameters
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced surface Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact in-plane lattice-coefficient rows.
    stacking_coeffs : Tuple[int, int, int]
        Exact one-plane stacking coefficient row.
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom.
    termination : Tuple[str, str]
        Top and bottom termination labels.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    n_layers : int
        Positive number of selected layers.
    bulk_atom_count : int
        Positive atom count of the source bulk geometry.
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source basis orbital.

    Raises
    ------
    ValueError
        If exact surface coefficients, atom provenance, layer bounds, lengths,
        termination metadata, or finite slab dimensions are inconsistent.

    Implementation Logic
    --------------------
    Validate the exact surface basis first. Then require one coordinate shift
    per frozen bulk atom and paired provenance for every selected slab atom.
    Finally, check all bulk and layer indices against their frozen bounds.
    """
    _validate_integer_triple(miller, "miller")
    if (
        type(in_plane_coeffs) is not tuple
        or len(in_plane_coeffs) != SURFACE_VECTOR_COUNT
    ):
        message: str = "in_plane_coeffs must contain two integer triples"
        raise ValueError(message)
    _validate_integer_triple(in_plane_coeffs[0], "in_plane_coeffs[0]")
    _validate_integer_triple(in_plane_coeffs[1], "in_plane_coeffs[1]")
    _validate_integer_triple(stacking_coeffs, "stacking_coeffs")
    if (
        _integer_dot(miller, in_plane_coeffs[0]) != 0
        or _integer_dot(miller, in_plane_coeffs[1]) != 0
    ):
        message = "in_plane_coeffs must lie in the Miller plane"
        raise ValueError(message)
    if _integer_dot(miller, stacking_coeffs) != 1:
        message = "miller dotted with stacking_coeffs must equal one"
        raise ValueError(message)
    coefficient_rows: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = (in_plane_coeffs[0], in_plane_coeffs[1], stacking_coeffs)
    if _integer_determinant(coefficient_rows) == 0:
        message = "slab topology coefficient rows must be independent"
        raise ValueError(message)
    if type(atom_shifts) is not tuple:
        message = "atom_shifts must be a tuple of integer triples"
        raise ValueError(message)
    atom_shift: Tuple[int, int, int]
    for atom_shift in atom_shifts:
        _validate_integer_triple(atom_shift, "atom_shifts entry")
    if any(
        type(values) is not tuple
        for values in (
            bulk_atom_of_slab_atom,
            layer_of_slab_atom,
            basis_atom_indices,
        )
    ):
        message = "slab topology provenance maps must be tuples"
        raise ValueError(message)
    if len(bulk_atom_of_slab_atom) != len(layer_of_slab_atom):
        message = "slab atom provenance maps must agree in length"
        raise ValueError(message)
    if (
        type(termination) is not tuple
        or len(termination) != SURFACE_VECTOR_COUNT
        or any(type(species) is not str for species in termination)
    ):
        message = "termination must contain two species labels"
        raise ValueError(message)
    if (
        type(thickness_ang) is not float
        or type(vacuum_ang) is not float
        or not math.isfinite(thickness_ang)
        or not math.isfinite(vacuum_ang)
        or thickness_ang < 0.0
        or vacuum_ang < 0.0
    ):
        message = "thickness_ang and vacuum_ang must be finite and nonnegative"
        raise ValueError(message)
    if (
        type(fine) is not tuple
        or len(fine) != SURFACE_VECTOR_COUNT
        or any(type(value) is not float for value in fine)
        or not all(math.isfinite(value) for value in fine)
    ):
        message = "fine must contain two finite floats"
        raise ValueError(message)
    if type(n_layers) is not int or n_layers <= 0:
        message = "n_layers must be a positive integer"
        raise ValueError(message)
    if type(bulk_atom_count) is not int or bulk_atom_count <= 0:
        message = "bulk_atom_count must be a positive integer"
        raise ValueError(message)
    if len(atom_shifts) != bulk_atom_count:
        message = "atom_shifts must contain one entry per frozen bulk atom"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 or index >= bulk_atom_count
        for index in bulk_atom_of_slab_atom
    ):
        message = "bulk atom provenance must lie in the frozen bulk geometry"
        raise ValueError(message)
    if any(
        type(layer) is not int or layer < 0 or layer >= n_layers
        for layer in layer_of_slab_atom
    ):
        message = "layer provenance must lie in the frozen layer range"
        raise ValueError(message)
    if any(
        type(index) is not int or index < 0 or index >= bulk_atom_count
        for index in basis_atom_indices
    ):
        message = "basis atom indices must lie in the frozen bulk geometry"
        raise ValueError(message)


class SlabTopology(eqx.Module):
    """Store host-selected discrete slab topology for pure-JAX rebuilding.

    The carrier contains only static integer choices, endpoint metadata, and
    design values selected before a JAX transformation.

    :see: :class:`~.test_slab_topology.TestSlabTopology`

    Attributes
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced Miller tuple (**static**).
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact coefficient rows for the surface plane (**static**).
    stacking_coeffs : Tuple[int, int, int]
        Exact coefficient row for one stacking period (**static**).
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom (**static**).
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom (**static**).
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom (**static**).
    termination : Tuple[str, str]
        Top and bottom species labels (**static**).
    thickness_ang : float
        Requested slab thickness in Angstrom (**static**).
    vacuum_ang : float
        Vacuum padding in Angstrom (**static**).
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom (**static**).
    n_layers : int
        Number of selected layers (**static**).
    bulk_atom_count : int
        Atom count of the source bulk geometry (**static**).
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source orbital (**static**).

    See Also
    --------
    make_slab_topology : Validated factory for this type.
    """

    miller: Tuple[int, int, int] = eqx.field(static=True)
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ] = eqx.field(static=True)
    stacking_coeffs: Tuple[int, int, int] = eqx.field(static=True)
    atom_shifts: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)
    bulk_atom_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    layer_of_slab_atom: Tuple[int, ...] = eqx.field(static=True)
    termination: Tuple[str, str] = eqx.field(static=True)
    thickness_ang: float = eqx.field(static=True)
    vacuum_ang: float = eqx.field(static=True)
    fine: Tuple[float, float] = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    bulk_atom_count: int = eqx.field(static=True)
    basis_atom_indices: Tuple[int, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate frozen slab topology on direct construction."""
        _validate_slab_topology_structure(
            miller=self.miller,
            in_plane_coeffs=self.in_plane_coeffs,
            stacking_coeffs=self.stacking_coeffs,
            atom_shifts=self.atom_shifts,
            bulk_atom_of_slab_atom=self.bulk_atom_of_slab_atom,
            layer_of_slab_atom=self.layer_of_slab_atom,
            termination=self.termination,
            thickness_ang=self.thickness_ang,
            vacuum_ang=self.vacuum_ang,
            fine=self.fine,
            n_layers=self.n_layers,
            bulk_atom_count=self.bulk_atom_count,
            basis_atom_indices=self.basis_atom_indices,
        )


@jaxtyped(typechecker=beartype)
def make_slab_topology(  # noqa: DOC502, PLR0913
    *,
    miller: Tuple[int, int, int],
    in_plane_coeffs: Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
    ],
    stacking_coeffs: Tuple[int, int, int],
    atom_shifts: Tuple[Tuple[int, int, int], ...],
    bulk_atom_of_slab_atom: Tuple[int, ...],
    layer_of_slab_atom: Tuple[int, ...],
    termination: Tuple[str, str],
    thickness_ang: float,
    vacuum_ang: float,
    fine: Tuple[float, float],
    n_layers: int,
    bulk_atom_count: int,
    basis_atom_indices: Tuple[int, ...],
) -> SlabTopology:
    """Create validated host-selected slab topology metadata.

    The factory binds one exact surface basis to atom, layer, termination,
    and source-model provenance. All returned fields are static PyTree
    metadata because topology selection occurs before JAX transformation.

    :see: :class:`~.test_slab_topology.TestMakeSlabTopology`

    Parameters
    ----------
    miller : Tuple[int, int, int]
        GCD-reduced surface Miller tuple.
    in_plane_coeffs : Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Exact in-plane lattice-coefficient rows.
    stacking_coeffs : Tuple[int, int, int]
        Exact one-plane stacking coefficient row.
    atom_shifts : Tuple[Tuple[int, int, int], ...]
        Surface-cell shift for every source bulk atom.
    bulk_atom_of_slab_atom : Tuple[int, ...]
        Bulk-atom provenance for every slab atom.
    layer_of_slab_atom : Tuple[int, ...]
        Layer index for every slab atom.
    termination : Tuple[str, str]
        Top and bottom termination labels.
    thickness_ang : float
        Requested slab thickness in Angstrom.
    vacuum_ang : float
        Vacuum padding in Angstrom.
    fine : Tuple[float, float]
        Top and bottom cut shifts in Angstrom.
    n_layers : int
        Positive number of selected layers.
    bulk_atom_count : int
        Positive atom count of the source bulk geometry.
    basis_atom_indices : Tuple[int, ...]
        Bulk-atom assignment for every source basis orbital.

    Returns
    -------
    topology : SlabTopology
        Validated static topology for pure-JAX slab rebuilding.

    Raises
    ------
    ValueError
        If exact surface coefficients, atom provenance, layer bounds, lengths,
        termination metadata, or finite slab dimensions are inconsistent.

    Notes
    -----
    Every check is static because all topology fields are immutable host
    choices. Numerical lattice and Hamiltonian leaves remain outside this
    carrier and retain their existing differentiable rebuild path.
    """
    _validate_slab_topology_structure(
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
        atom_shifts=atom_shifts,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
        termination=termination,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        n_layers=n_layers,
        bulk_atom_count=bulk_atom_count,
        basis_atom_indices=basis_atom_indices,
    )
    topology: SlabTopology = SlabTopology(
        miller=miller,
        in_plane_coeffs=in_plane_coeffs,
        stacking_coeffs=stacking_coeffs,
        atom_shifts=atom_shifts,
        bulk_atom_of_slab_atom=bulk_atom_of_slab_atom,
        layer_of_slab_atom=layer_of_slab_atom,
        termination=termination,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        fine=fine,
        n_layers=n_layers,
        bulk_atom_count=bulk_atom_count,
        basis_atom_indices=basis_atom_indices,
    )
    return topology


__all__: list[str] = [
    "SlabTopology",
    "make_slab_topology",
]
