"""Define structured inputs for high-level VASP simulation workflows.

Extended Summary
----------------
This module defines the Equinox container that bundles parsed VASP outputs for
the high-level simulation workflow. Its location in :mod:`diffpes.types`
provides one canonical import surface for every diffpes PyTree.

Routine Listings
----------------
:class:`WorkflowContext`
    Store parsed VASP inputs for high-level workflow helpers.
:func:`make_workflow_context`
    Create a workflow context from parsed VASP inputs.
:obj:`DosType`
    Supported density-of-states containers.
:obj:`ProjectionType`
    Supported orbital-projection containers.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional, TypeAlias, Union
from jaxtyping import jaxtyped

from .bands import BandStructure, OrbitalProjection, SpinOrbitalProjection
from .dos import DensityOfStates, FullDensityOfStates
from .kpath import KPathInfo

ProjectionType: TypeAlias = Union[OrbitalProjection, SpinOrbitalProjection]
DosType: TypeAlias = Union[DensityOfStates, FullDensityOfStates]


class WorkflowContext(eqx.Module):
    """Store parsed VASP inputs for high-level workflow helpers.

    Bundles the electronic bands and orbital projections required by an
    ARPES simulation with optional k-path and density-of-states metadata.
    Every non-``None`` field is itself an Equinox PyTree, so the complete
    context can pass through JAX transformations as one immutable object.

    :see: :class:`~.test_context.TestWorkflowContext`

    Attributes
    ----------
    bands : BandStructure
        Band eigenvalues and k-point coordinates from EIGENVAL.
    orb_proj : ProjectionType
        Orbital projections from PROCAR, optionally including spin.
    kpath : Optional[KPathInfo]
        Parsed KPOINTS metadata when available.
    dos : Optional[DosType]
        Parsed DOSCAR data when available.

    See Also
    --------
    make_workflow_context : Validated factory for this type.
    """

    bands: BandStructure
    orb_proj: ProjectionType
    kpath: Optional[KPathInfo]
    dos: Optional[DosType]


@jaxtyped(typechecker=beartype)
def make_workflow_context(
    bands: BandStructure,
    orb_proj: ProjectionType,
    kpath: Optional[KPathInfo] = None,
    dos: Optional[DosType] = None,
) -> WorkflowContext:
    """Create a workflow context from parsed VASP inputs.

    This type collects parsed electronic bands and orbital projections with
    optional k-path and density-of-states data. The result is one immutable
    Equinox PyTree for high-level simulation workflows.

    :see: :class:`~.test_context.TestMakeWorkflowContext`

    Implementation Logic
    --------------------
    1. **Check the shared axes**::

           bands.eigenvalues.shape[:2] != orb_proj.projections.shape[:2]

       This static shape check raises ``ValueError`` before construction.
    2. **Construct the context**::

           context = WorkflowContext(...)

       The Equinox module keeps all supplied carriers in one PyTree.

    Parameters
    ----------
    bands : BandStructure
        Band eigenvalues and k-point coordinates.
    orb_proj : ProjectionType
        Orbital projections, optionally including spin channels.
    kpath : Optional[KPathInfo], optional
        Parsed k-path metadata. Default is ``None``.
    dos : Optional[DosType], optional
        Parsed density-of-states data. Default is ``None``.

    Returns
    -------
    context : WorkflowContext
        Immutable Equinox container holding the supplied inputs.

    Raises
    ------
    ValueError
        If the k-point or band dimensions disagree between ``bands`` and
        ``orb_proj``.
    """
    if bands.eigenvalues.shape[:2] != orb_proj.projections.shape[:2]:
        msg: str = (
            "bands and orb_proj must agree on k-point and band dimensions"
        )
        raise ValueError(msg)
    context: WorkflowContext = WorkflowContext(
        bands=bands,
        orb_proj=orb_proj,
        kpath=kpath,
        dos=dos,
    )
    return context


__all__: list[str] = [
    "WorkflowContext",
    "make_workflow_context",
    "DosType",
    "ProjectionType",
]
