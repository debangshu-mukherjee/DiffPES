"""Verify the detector-effects carrier and its validated factory.

The tests pin traced leaves, static acquisition metadata, kernel
normalization, and eager plus compiled rejection behavior.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Dict

from diffpes.types import DetectorEffects, make_detector_effects
from tests._assertions import assert_rejects

_FRAME_ID: str = "org.diffpes.frame.sample_cartesian"
_DETERMINISTIC_RTOL: float = 1.0e-10


def _effects(**overrides: object) -> DetectorEffects:
    """PRIVATE: Build valid one-domain detector effects.

    Keyword overrides select the factory contract under test.

    Parameters
    ----------
    **overrides : object
        Values that replace valid flat-background defaults.

    Returns
    -------
    effects : DetectorEffects
        Validated detector-effects carrier.
    """
    parameters: Dict[str, object] = {
        "domain_logits": jnp.array([0.2]),
        "domain_euler_angles_rad": jnp.array([[0.1, -0.2, 0.3]]),
        "transmission_raw_slopes": jnp.array([0.15, -0.25]),
        "background_coefficients": jnp.array([0.1]),
        "sensitivity_coefficients": jnp.array([]),
        "exposure": 2.5,
        "background_mode": "flat",
        "sensitivity_mode": "constant",
        "domain_frame_ids": (_FRAME_ID,),
    }
    parameters.update(overrides)
    effects: DetectorEffects = make_detector_effects(**parameters)
    return effects


class TestDetectorEffects:
    """Verify :class:`diffpes.types.DetectorEffects`.

    The class owns PyTree leaves and static v1 acquisition metadata.
    """

    def test_preserves_complete_pytree_and_static_metadata(self) -> None:
        """Preserve all seven numerical leaves and static selectors.

        The case also checks normalized calibrated response state.

        Notes
        -----
        The test builds fixed-total effects through the public factory. It
        inspects the JAX leaves and compares the normalized kernel directly.
        """
        effects: DetectorEffects = _effects(
            post_count_mode="calibrated",
            post_count_kernel=jnp.array([1.0, 3.0, 2.0]),
            acquisition_mode="fixed_total",
            fixed_total_count=137,
        )
        leaves: list[object] = jax.tree.leaves(effects)

        chex.assert_trees_all_close(
            effects.post_count_kernel,
            jnp.array([1.0, 3.0, 2.0]) / 6.0,
            rtol=_DETERMINISTIC_RTOL,
            atol=0.0,
        )
        assert len(leaves) == 7
        assert effects.coordinate_density == "per_native_volume"
        assert effects.acquisition_mode == "fixed_total"
        assert effects.fixed_total_count == 137
        assert effects.domain_frame_ids == (_FRAME_ID,)


class TestMakeDetectorEffects:
    """Verify :func:`diffpes.types.make_detector_effects`.

    The class owns structural and traced factory rejection contracts.
    """

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            (
                {"domain_euler_angles_rad": jnp.zeros((2, 3))},
                "rotations and logits disagree",
            ),
            (
                {"transmission_raw_slopes": jnp.zeros(4)},
                "transmission length",
            ),
            (
                {
                    "background_mode": "smooth",
                    "background_coefficients": jnp.zeros(6),
                },
                "background coefficient length",
            ),
            (
                {
                    "post_count_mode": "calibrated",
                    "post_count_kernel": jnp.ones(2),
                },
                "kernel length must be odd",
            ),
            (
                {"domain_frame_ids": ("org.example.unregistered",)},
                "unregistered domain frame",
            ),
            (
                {
                    "acquisition_mode": "fixed_total",
                    "fixed_total_count": 0,
                },
                "positive integer count",
            ),
        ],
    )
    def test_rejects_invalid_static_contracts(
        self, overrides: Dict[str, object], match: str
    ) -> None:
        """Reject invalid shapes, modes, frames, and acquisition totals.

        The parameterized rows cover each independent structural family.

        Notes
        -----
        The test replaces one valid factory argument per row and matches the
        dedicated error message.
        """
        with pytest.raises(ValueError, match=match):
            _effects(**overrides)

    def test_rejects_invalid_traced_values_eager_and_jit(self) -> None:
        """Reject non-finite leaves and nonpositive exposure under JIT.

        The case pins traced validation to compiled execution.

        Notes
        -----
        The shared rejection helper calls the same local factory eagerly and
        through Equinox filtered JIT for both numerical failures.
        """
        parameters: Dict[str, jax.Array] = {
            "logits": jnp.array([jnp.nan]),
            "rotations": jnp.zeros((1, 3)),
            "slopes": jnp.zeros(2),
            "background": jnp.zeros(1),
            "sensitivity": jnp.zeros(0),
            "exposure": jnp.array(1.0),
        }

        def build(
            logits: jax.Array,
            rotations: jax.Array,
            slopes: jax.Array,
            background: jax.Array,
            sensitivity: jax.Array,
            exposure: jax.Array,
        ) -> DetectorEffects:
            effects: DetectorEffects = make_detector_effects(
                logits,
                rotations,
                slopes,
                background,
                sensitivity,
                exposure,
                background_mode="flat",
                sensitivity_mode="constant",
                domain_frame_ids=(_FRAME_ID,),
            )
            return effects

        assert_rejects(
            build,
            *parameters.values(),
            match="domain logits finite",
        )
        assert_rejects(
            build,
            jnp.zeros(1),
            jnp.zeros((1, 3)),
            jnp.zeros(2),
            jnp.zeros(1),
            jnp.zeros(0),
            jnp.array(0.0),
            match="exposure finite and positive",
        )
