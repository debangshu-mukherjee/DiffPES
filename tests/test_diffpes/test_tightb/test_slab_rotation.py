"""Validate the slab rotation module.

The cases use analytic values, invariants, and finite differences.
"""


class TestRotateTbModel:
    """Mirror coverage for :func:`diffpes.tightb.rotate_tb_model`.

    The detailed slab cases check active frame rotation for geometry and
    hopping cells.
    """
