"""Specify static sharding-policy validation.

Extended Summary
----------------
These tests exercise static factory validation without requiring a physical
mesh or a compiled scientific model.

Routine Listings
----------------
:class:`TestMakeShardSpec`
    Specify static sharding-policy factory behavior.
"""

import pytest

from diffpes.types import make_shard_spec


class TestMakeShardSpec:
    """Specify static sharding-policy factory behavior.

    See Also
    --------
    diffpes.types.make_shard_spec
        Production factory covered by these contract tests.
    """

    def test_rejects_nondivisible_capacity(self) -> None:
        """Reject a capacity that cannot form device-local chunks."""
        with pytest.raises(ValueError, match="invalid"):
            make_shard_spec(n_devices=2, chunk_size=4, nk_max=9)

    def test_declares_all_static_execution_fields(self) -> None:
        """Preserve the same execution-policy surface on one device."""
        spec = make_shard_spec(n_devices=1, chunk_size=4, nk_max=8)
        assert spec.n_devices == 1
        assert spec.nk_max == 8
