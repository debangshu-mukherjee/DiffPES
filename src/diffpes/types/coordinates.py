"""Define typed measurement coordinates.

Extended Summary
----------------
This module stores axes and auxiliary coordinates without a dense coordinate
mesh. The carrier keeps coordinate names, units, dimensions, and frame
identity next to the JAX numerical leaves.

Routine Listings
----------------
:class:`MeasurementCoordinates`
    Store typed axes and auxiliary measurement coordinates.
:func:`make_measurement_coordinates`
    Create validated typed measurement coordinates.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped


class MeasurementCoordinates(eqx.Module):
    """Store typed axes and auxiliary measurement coordinates.

    The carrier stores each coordinate as a separate leaf. It does not create
    a dense mesh. Static metadata identifies the coordinate system and frame.

    :see: :class:`~.test_coordinates.TestMeasurementCoordinates`

    Attributes
    ----------
    coordinate_arrays : Tuple[Float64[Array, "..."], ...]
        Numeric coordinates in ``coordinate_names`` order.
    coordinate_names : Tuple[str, ...]
        **Static.** Unique coordinate labels. Changing a label triggers
        retracing.
    coordinate_units : Tuple[str, ...]
        **Static.** Units for every coordinate.
    coordinate_dimensions : Tuple[Tuple[str, ...], ...]
        **Static.** Named dimensions for every coordinate leaf.
    dimension_names : Tuple[str, ...]
        **Static.** Unique dimensions of the measurement representation.
    coordinate_system : str
        **Static.** Registered coordinate-system identity.
    frame_id : str
        **Static.** Registered coordinate-frame identity.

    See Also
    --------
    make_measurement_coordinates : Create validated typed measurement
        coordinates.
    """

    coordinate_arrays: Tuple[Float64[Array, "..."], ...]
    coordinate_names: Tuple[str, ...] = eqx.field(static=True)
    coordinate_units: Tuple[str, ...] = eqx.field(static=True)
    coordinate_dimensions: Tuple[Tuple[str, ...], ...] = eqx.field(static=True)
    dimension_names: Tuple[str, ...] = eqx.field(static=True)
    coordinate_system: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """PRIVATE: Validate the coordinate topology and traced values.

        Raises
        ------
        ValueError
            If metadata is incomplete, non-unique, or has incompatible ranks.
        """
        count: int = len(self.coordinate_arrays)
        if (
            count == 0
            or not self.coordinate_system
            or not self.frame_id
            or len(self.coordinate_names) != count
            or len(self.coordinate_units) != count
            or len(self.coordinate_dimensions) != count
            or len(set(self.coordinate_names)) != count
            or len(set(self.dimension_names)) != len(self.dimension_names)
        ):
            raise ValueError("measurement-coordinate metadata is inconsistent")
        checked_arrays: Tuple[Float64[Array, "..."], ...] = tuple(
            eqx.error_if(
                array,
                ~jnp.all(jnp.isfinite(array)),
                "measurement coordinates must be finite",
            )
            for array in self.coordinate_arrays
        )
        for array, dimensions in zip(
            checked_arrays, self.coordinate_dimensions, strict=True
        ):
            if array.ndim != len(dimensions):
                raise ValueError(
                    "coordinate array rank disagrees with dimensions"
                )
            if any(name not in self.dimension_names for name in dimensions):
                raise ValueError("coordinate dimensions must be declared")
        object.__setattr__(self, "coordinate_arrays", checked_arrays)


@jaxtyped(typechecker=beartype)
def make_measurement_coordinates(
    coordinate_arrays: Tuple[Float64[Array, "..."], ...],
    *,
    coordinate_names: Tuple[str, ...],
    coordinate_units: Tuple[str, ...],
    coordinate_dimensions: Tuple[Tuple[str, ...], ...],
    dimension_names: Tuple[str, ...],
    coordinate_system: str,
    frame_id: str,
) -> MeasurementCoordinates:
    """Create validated typed measurement coordinates.

    The factory converts numeric leaves to float64. It validates static
    topology eagerly. The carrier validates finite values under tracing.

    :see: :class:`~.test_coordinates.TestMakeMeasurementCoordinates`

    Parameters
    ----------
    coordinate_arrays : Tuple[Float64[Array, "..."], ...]
        Numeric coordinate leaves.
    coordinate_names : Tuple[str, ...]
        **Static.** Labels for the coordinate leaves.
    coordinate_units : Tuple[str, ...]
        **Static.** Units for the coordinate leaves.
    coordinate_dimensions : Tuple[Tuple[str, ...], ...]
        **Static.** Dimensions for the coordinate leaves.
    dimension_names : Tuple[str, ...]
        **Static.** All declared dimensions.
    coordinate_system : str
        **Static.** Coordinate-system identity.
    frame_id : str
        **Static.** Coordinate-frame identity.

    Returns
    -------
    coordinates : MeasurementCoordinates
        Validated measurement coordinates with float64 leaves.

    """
    arrays: Tuple[Float64[Array, "..."], ...] = tuple(
        jnp.asarray(array, dtype=jnp.float64) for array in coordinate_arrays
    )
    coordinates: MeasurementCoordinates = MeasurementCoordinates(
        coordinate_arrays=arrays,
        coordinate_names=coordinate_names,
        coordinate_units=coordinate_units,
        coordinate_dimensions=coordinate_dimensions,
        dimension_names=dimension_names,
        coordinate_system=coordinate_system,
        frame_id=frame_id,
    )
    return coordinates


__all__: list[str] = [
    "MeasurementCoordinates",
    "make_measurement_coordinates",
]
