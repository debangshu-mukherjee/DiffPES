"""Validate the slab assembly module.

The cases use analytic values, invariants, and finite differences.
"""


class TestRebuildSlab:
    """Mirror coverage for :func:`diffpes.tightb.rebuild_slab`.

    The detailed slab cases rebuild traced geometry from frozen topology and
    compare derivatives.
    """


class TestValidateOpenSurfaceAdjacency:
    """Mirror coverage for the exact open-normal graph validator.

    :see: :func:`diffpes.tightb.validate_open_surface_adjacency`
    """
