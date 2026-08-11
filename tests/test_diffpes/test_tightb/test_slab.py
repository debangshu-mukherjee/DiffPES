"""Mirror the public slab API onto symbol-owned test classes.

The specialized slab suites hold the physics fixtures; these aliases keep
the repository's one-public-symbol/one-test-class navigation contract exact.
"""


class TestFindSurfaceCell:
    """Mirror coverage for :func:`diffpes.tightb.find_surface_cell`.

    The detailed slab cases check primitive surface frames for several Miller
    directions.
    """


class TestFreezeSlabTopology:
    """Mirror coverage for :func:`diffpes.tightb.freeze_slab_topology`.

    The detailed slab cases check deterministic discrete topology and exact
    atom provenance.
    """


class TestRebuildSlab:
    """Mirror coverage for :func:`diffpes.tightb.rebuild_slab`.

    The detailed slab cases rebuild traced geometry from frozen topology and
    compare derivatives.
    """


class TestGenSlab:
    """Mirror coverage for :func:`diffpes.tightb.gen_slab`.

    The detailed slab cases check open-boundary construction, layer metadata,
    and model closure.
    """


class TestGenSlabWithOperators:
    """Mirror coverage for :func:`diffpes.tightb.gen_slab_with_operators`.

    The detailed operator cases check slab construction with transformed
    position matrices.
    """


class TestRotateTbModel:
    """Mirror coverage for :func:`diffpes.tightb.rotate_tb_model`.

    The detailed slab cases check active frame rotation for geometry and
    hopping cells.
    """


class TestValidateOpenSurfaceAdjacency:
    """Mirror coverage for the exact open-normal graph validator.

    :see: :func:`diffpes.tightb.validate_open_surface_adjacency`
    """
