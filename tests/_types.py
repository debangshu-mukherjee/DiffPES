"""Define shared test-side type aliases.

Extended Summary
----------------
This module centralizes annotations for the differentiability test harness.
It imports numerical aliases from :mod:`diffpes.types` or :mod:`jaxtyping`.
It does not duplicate production type definitions.
"""

from beartype.typing import Callable, Literal, TypeAlias
from jaxtyping import Array, Float64

GradRegime: TypeAlias = Literal["smooth", "stiff", "singular"]
ScalarLoss: TypeAlias = Callable[..., Float64[Array, ""]]
