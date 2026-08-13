"""Define typed measurement coordinates.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`MeasurementCoordinates`
    Define the ``MeasurementCoordinates`` public contract.
:func:`make_measurement_coordinates`
    Compute the ``make_measurement_coordinates`` public contract.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, jaxtyped


class MeasurementCoordinates(eqx.Module):
    """Define the ``MeasurementCoordinates`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_coordinates.TestMeasurementCoordinates`

    Attributes
    ----------
    coordinate_arrays : Tuple[Float64[Array, "..."], ...]
        Store coordinate arrays.
    coordinate_names : Tuple[str, ...]
        Store coordinate names.
    coordinate_units : Tuple[str, ...]
        Store coordinate units.
    coordinate_dimensions : Tuple[Tuple[str, ...], ...]
        Store dimensions for each coordinate.
    dimension_names : Tuple[str, ...]
        Store dimension names.
    coordinate_system : str
        Store the coordinate-system identity.
    frame_id : str
        Store the frame identity.

    See Also
    --------
    make_measurement_coordinates
        Construct validated measurement coordinates.
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
        array: Float64[Array, "..."]
        dimensions: Tuple[str, ...]
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
    """Compute the ``make_measurement_coordinates`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_coordinates.TestMakeMeasurementCoordinates`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    coordinate_arrays : Tuple[Float64[Array, '...'], ...]
        Input value for this operation.
    coordinate_names : Tuple[str, ...]
        Input value for this operation.
    coordinate_units : Tuple[str, ...]
        Input value for this operation.
    coordinate_dimensions : Tuple[Tuple[str, ...], ...]
        Input value for this operation.
    dimension_names : Tuple[str, ...]
        Input value for this operation.
    coordinate_system : str
        Input value for this operation.
    frame_id : str
        Input value for this operation.

    Returns
    -------
    result : MeasurementCoordinates
        Validated operation result.
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
