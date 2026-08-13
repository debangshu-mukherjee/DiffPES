"""Verify measurement-coordinate carrier invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict

from diffpes.types import MeasurementCoordinates, make_measurement_coordinates


def _make(**overrides: object) -> MeasurementCoordinates:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    values: Dict[str, object] = {
        "coordinate_arrays": (jnp.asarray([[0.0, 0.1, 0.2]]),),
        "coordinate_names": ("k_points_frac",),
        "coordinate_units": ("1",),
        "coordinate_dimensions": (("k", "cart"),),
        "dimension_names": ("k", "cart"),
        "coordinate_system": "fractional",
        "frame_id": "sample",
    }
    values.update(overrides)
    result: Any = make_measurement_coordinates(**values)
    return result


class TestMeasurementCoordinates:
    """Verify ``diffpes.types.MeasurementCoordinates`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_factory_accepts_complete_finite_metadata(self) -> None:
        """Preserve one finite coordinate and all declared identities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare each stored field with the explicit factory inputs.
        """
        coordinates: Any = _make()
        assert coordinates.coordinate_names == ("k_points_frac",)
        assert coordinates.coordinate_units == ("1",)
        assert coordinates.frame_id == "sample"
        assert jnp.array_equal(
            coordinates.coordinate_arrays[0],
            jnp.asarray([[0.0, 0.1, 0.2]]),
        )


class TestMakeMeasurementCoordinates:
    """Verify ``diffpes.types.make_measurement_coordinates`` validation.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    @pytest.mark.parametrize(
        "overrides",
        [
            {"coordinate_arrays": ()},
            {"coordinate_system": ""},
            {"frame_id": ""},
            {"coordinate_names": ()},
            {"coordinate_units": ()},
            {"coordinate_dimensions": ()},
            {
                "coordinate_arrays": (jnp.asarray([0.0]), jnp.asarray([1.0])),
                "coordinate_names": ("x", "x"),
                "coordinate_units": ("1", "1"),
                "coordinate_dimensions": (("x",), ("x",)),
                "dimension_names": ("x",),
            },
            {"dimension_names": ("k", "k")},
        ],
    )
    def test_rejects_inconsistent_metadata(
        self, overrides: Dict[str, object]
    ) -> None:
        """Reject each missing, misaligned, or duplicate metadata identity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one component of the valid fixture for each parameterized case.
        """
        with pytest.raises(ValueError, match="metadata is inconsistent"):
            _make(**overrides)

    def test_rejects_coordinate_rank_mismatch(self) -> None:
        """Reject a vector assigned two named dimensions.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Keep the coordinate values finite while changing only their rank.
        """
        with pytest.raises(ValueError, match="rank disagrees"):
            _make(coordinate_arrays=(jnp.asarray([0.0, 0.1, 0.2]),))

    def test_rejects_undeclared_coordinate_dimension(self) -> None:
        """Reject a coordinate dimension absent from the dimension registry.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace the declared Cartesian dimension with an unknown name.
        """
        with pytest.raises(ValueError, match="dimensions must be declared"):
            _make(coordinate_dimensions=(("k", "component"),))

    def test_rejects_nonfinite_values(self) -> None:
        """Reject nonfinite coordinate values through the traced guard.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Insert one NaN in an otherwise valid coordinate matrix.
        """
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="coordinates must be finite"
        ):
            _make(coordinate_arrays=(jnp.asarray([[jnp.nan, 0.0, 0.0]]),))
