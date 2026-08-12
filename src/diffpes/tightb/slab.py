"""Construct finite Miller-index slabs.

Extended Summary
----------------
This module exposes slab construction through one frozen-topology composition.

Routine Listings
----------------
:func:`gen_slab`
    Construct a finite Miller-index slab with exact open-normal topology.
"""

from __future__ import annotations

from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped

from diffpes.types import SlabSpec, SlabTopology, TBModel

from .slab_assembly import rebuild_slab
from .slab_topology import freeze_slab_topology


@jaxtyped(typechecker=beartype)
def gen_slab(  # noqa: DOC105, DOC502, PLR0913, PLR0915
    bulk_model: TBModel,
    miller: Tuple[int, int, int],
    thickness_ang: float,
    vacuum_ang: float,
    termination: Tuple[str, str] | None = None,
    fine: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[TBModel, SlabSpec]:
    """Construct a finite Miller-index slab with exact open-normal topology.

    Eager input selects discrete atoms, orbitals, and hoppings. Every assembled
    geometry, centre, depth, amplitude, onsite, and SOC value remains a JAX
    array. Re-run the factory after a continuous perturbation crosses a layer,
    termination, or metric-reduction boundary.

    Parameters
    ----------
    bulk_model : TBModel
        Validated bulk tight-binding model.
    miller : Tuple[int, int, int]
        Primitive Miller indices (**static**).
    thickness_ang : float
        Nonnegative minimum material span in Angstrom (**static**). Zero
        requests the one-plane limiting slab.
    vacuum_ang : float
        Nonnegative vacuum padding in Angstrom (**static**).
    termination : Tuple[str, str] or None, optional
        Requested ``(top, bottom)`` species. ``None`` retains a natural
        complete stack. Default is ``None``.
    fine : Tuple[float, float], optional
        Static ``(top, bottom)`` inward cut shifts in Angstrom.

    Returns
    -------
    result : Tuple[TBModel, SlabSpec]
        Slab model and its immutable construction provenance.

    Raises
    ------
    ValueError
        If geometry choices fail, no requested termination exists, or the
        propagated graph retains a normal image.

    Examples
    --------
    Build a nearest-neighbour graphene model and cut a finite zigzag ribbon
    normal to the first reciprocal-lattice vector. The third bulk vector
    provides a noninteracting embedding direction. Exact cells certify slab
    openness independently of its length.

    >>> import jax.numpy as jnp
    >>> from diffpes.types import (
    ...     make_crystal_geometry,
    ...     make_orbital_basis,
    ...     make_tb_model,
    ... )
    >>> bond = 1.42
    >>> root_three = jnp.sqrt(3.0)
    >>> geometry = make_crystal_geometry(
    ...     lattice=jnp.asarray(
    ...         (
    ...             (root_three * bond, 0.0, 0.0),
    ...             (root_three * bond / 2.0, 1.5 * bond, 0.0),
    ...             (0.0, 0.0, 20.0),
    ...         )
    ...     ),
    ...     positions=jnp.asarray(((0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0))),
    ...     species=("C", "C"),
    ... )
    >>> basis = make_orbital_basis(
    ...     atom_indices=(0, 1),
    ...     n=(2, 2),
    ...     l=(1, 1),
    ...     m=(0, 0),
    ...     labels=("pz_A", "pz_B"),
    ... )
    >>> graphene = make_tb_model(
    ...     hopping_amplitudes=-2.7 * jnp.ones(6, dtype=jnp.complex128),
    ...     onsite_energies=jnp.zeros(2),
    ...     soc_lambdas=jnp.zeros(0),
    ...     geometry=geometry,
    ...     basis=basis,
    ...     hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
    ...     hopping_cells=(
    ...         (0, 0, 0),
    ...         (-1, 0, 0),
    ...         (0, -1, 0),
    ...         (0, 0, 0),
    ...         (1, 0, 0),
    ...         (0, 1, 0),
    ...     ),
    ...     shell_index=(-1, -1),
    ... )
    >>> ribbon, ribbon_spec = gen_slab(
    ...     graphene,
    ...     miller=(1, 0, 0),
    ...     thickness_ang=15.0,
    ...     vacuum_ang=12.0,
    ... )
    >>> ribbon_spec.n_layers > 1
    True
    >>> all(cell[2] == 0 for cell in ribbon.hopping_cells)
    True

    Notes
    -----
    This convenience factory performs host-only topology selection before
    calling :func:`rebuild_slab`; it is not itself a ``jit``/``grad``/``vmap``
    target. For transformed calculations, call :func:`freeze_slab_topology`
    eagerly once and transform :func:`rebuild_slab`.

    :see: :class:`~.test_slab.TestGenSlab`
    """
    topology: SlabTopology = freeze_slab_topology(
        bulk_model=bulk_model,
        miller=miller,
        thickness_ang=thickness_ang,
        vacuum_ang=vacuum_ang,
        termination=termination,
        fine=fine,
    )
    result: Tuple[TBModel, SlabSpec] = rebuild_slab(bulk_model, topology)
    return result


__all__: list[str] = [
    "gen_slab",
]
