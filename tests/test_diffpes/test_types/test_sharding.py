"""Check static sharding-policy validation.

Use independent inputs to define the covered behavior.
"""

import pytest
from beartype.typing import Any, Dict

from diffpes.types import make_shard_spec


class TestShardSpec:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.types.ShardSpec``
    """

    def test_exposes_documented_defaults(self) -> None:
        """Resolve default mesh, chunk, checkpoint, and precision policy.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        spec: Any
        spec = make_shard_spec(n_devices=1)
        assert spec.device_axis == "k"
        assert spec.chunk_size == 256
        assert spec.checkpoint_policy == "everything"
        assert spec.demote_accumulation is False


class TestMakeShardSpec:
    """Check the public symbol contract.

    Use independent inputs to define the covered behavior.

    ``diffpes.types.make_shard_spec``
    """

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"device_axis": ""}, "device axis must be nonempty"),
            ({"n_devices": 0}, "device count must be positive"),
            ({"chunk_size": 0}, "chunk size must be positive"),
            ({"nk_max": 0}, "capacity must be positive"),
            (
                {"n_devices": 2, "chunk_size": 4, "nk_max": 9},
                "capacity must divide into local chunks",
            ),
            ({"checkpoint_policy": "unknown"}, "policy is unsupported"),
            (
                {"demote_accumulation": True},
                "f32 accumulation is not implemented",
            ),
        ],
    )
    def test_rejects_each_invalid_field(
        self, kwargs: Dict[str, object], message: str
    ) -> None:
        """Reject each malformed field with its own exact diagnostic.

        Compare the result with an independent expected property.

        Notes
        -----
        Use direct arithmetic or explicit rejection messages as the oracle.
        """
        with pytest.raises(ValueError, match=message):
            make_shard_spec(**kwargs)
