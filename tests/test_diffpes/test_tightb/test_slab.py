"""Mirror the public slab API onto symbol-owned test classes.

The specialized slab suites hold the physics fixtures; these aliases keep
the repository's one-public-symbol/one-test-class navigation contract exact.
"""


class TestFindSurfaceCell:
    """Mirror coverage for :func:`diffpes.tightb.find_surface_cell`."""


class TestFreezeSlabTopology:
    """Mirror coverage for :func:`diffpes.tightb.freeze_slab_topology`."""


class TestRebuildSlab:
    """Mirror coverage for :func:`diffpes.tightb.rebuild_slab`."""


class TestGenSlab:
    """Mirror coverage for :func:`diffpes.tightb.gen_slab`."""


class TestGenSlabWithOperators:
    """Mirror coverage for :func:`diffpes.tightb.gen_slab_with_operators`."""


class TestRotateTbModel:
    """Mirror coverage for :func:`diffpes.tightb.rotate_tb_model`."""


class TestValidateOpenSurfaceAdjacency:
    """Mirror coverage for the exact open-normal graph validator.

    :see: :func:`diffpes.tightb.validate_open_surface_adjacency`
    """


__all__: list[str] = []
