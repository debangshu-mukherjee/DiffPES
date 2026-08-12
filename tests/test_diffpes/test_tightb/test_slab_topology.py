"""Validate the slab topology module.

The cases use analytic values, invariants, and finite differences.
"""


class TestFreezeSlabTopology:
    """Mirror coverage for :func:`diffpes.tightb.freeze_slab_topology`.

    The detailed slab cases check deterministic discrete topology and exact
    atom provenance.
    """
