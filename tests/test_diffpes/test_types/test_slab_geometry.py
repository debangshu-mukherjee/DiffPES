"""Validate the slab geometry contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""


class TestSurfaceCell:
    """Mirror coverage for :class:`diffpes.types.SurfaceCell`.

    The detailed carrier cases check traced frame data, exact coefficients, and
    factory rejection.
    """


class TestSlabSpec:
    """Mirror coverage for :class:`diffpes.types.SlabSpec`.

    The detailed carrier cases check slab choices, atom provenance, and
    persistence behavior.
    """


class TestMakeSurfaceCell:
    """Mirror coverage for :func:`diffpes.types.make_surface_cell`.

    The detailed factory cases construct a surface frame and reject invalid
    exact geometry metadata.
    """


class TestMakeSlabSpec:
    """Mirror coverage for :func:`diffpes.types.make_slab_spec`.

    The detailed factory cases construct slab metadata and reject inconsistent
    species or provenance.
    """
