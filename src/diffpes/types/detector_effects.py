"""Define detector-coordinate nuisance and acquisition state.

Routine Listings
----------------
:class:`DetectorEffects`
    Store the complete v1 detector-effects PyTree.
:func:`make_detector_effects`
    Create validated detector-effects state.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float64, jaxtyped

from .aliases import ScalarFloat

_ACQUISITION_MODES: Tuple[str, ...] = ("poisson", "fixed_total")
_BACKGROUND_MODES: Tuple[str, ...] = ("flat", "shirley", "smooth")
_COORDINATE_DENSITY: str = "per_native_volume"
_POST_COUNT_MODES: Tuple[str, ...] = ("none", "calibrated")
_REGISTERED_DOMAIN_FRAME_IDS: Tuple[str, ...] = (
    "org.diffpes.frame.sample_cartesian",
)
_SENSITIVITY_MODES: Tuple[str, ...] = ("constant", "smooth")

__all__: list[str] = ["DetectorEffects", "make_detector_effects"]


class DetectorEffects(eqx.Module):
    """Store the complete v1 detector-effects PyTree.

    The carrier separates differentiable numerical leaves from static mode,
    frame, coordinate-density, and acquisition metadata.

    :see: :class:`~.test_detector_effects.TestDetectorEffects`

    Attributes
    ----------
    domain_logits : Float64[Array, " n_domain"]
        Unconstrained logits for detector-space domain mixing.
    domain_euler_angles_rad : Float64[Array, "n_domain 3"]
        Active Cartesian rotations for the source domains, in radians.
    transmission_raw_slopes : Float64[Array, " q"]
        Raw analyser-transmission coordinates; v1 permits two or three.
    background_coefficients : Float64[Array, " n_background"]
        Raw detector-density background coordinates.
    sensitivity_coefficients : Float64[Array, " n_sensitivity"]
        Smooth multiplicative sensitivity coordinates.
    exposure : Float64[Array, ""]
        Positive exposure in acquisition units.
    post_count_kernel : Optional[Float64[Array, " n_kernel"]]
        Optional normalized calibrated energy-response kernel.
    background_mode : str
        **Static.** ``"flat"``, ``"shirley"``, or ``"smooth"``.
    sensitivity_mode : str
        **Static.** ``"constant"`` or ``"smooth"``.
    post_count_mode : str
        **Static.** ``"none"`` or ``"calibrated"``.
    coordinate_density : str
        **Static.** V1 is exactly ``"per_native_volume"``.
    acquisition_mode : str
        **Static.** ``"poisson"`` or ``"fixed_total"``.
    fixed_total_count : Optional[int]
        **Static.** Positive event count for fixed-total acquisition.
    domain_frame_ids : Tuple[str, ...]
        **Static.** Registered Cartesian frame for every source domain.

    Notes
    -----
    Numerical fields are differentiable leaves. Selectors and discrete
    acquisition metadata are static. The factory normalizes a calibrated
    response kernel exactly once.
    """

    domain_logits: Float64[Array, " n_domain"]
    domain_euler_angles_rad: Float64[Array, "n_domain 3"]
    transmission_raw_slopes: Float64[Array, " q"]
    background_coefficients: Float64[Array, " n_background"]
    sensitivity_coefficients: Float64[Array, " n_sensitivity"]
    exposure: Float64[Array, ""]
    post_count_kernel: Optional[Float64[Array, " n_kernel"]]
    background_mode: str = eqx.field(static=True)
    sensitivity_mode: str = eqx.field(static=True)
    post_count_mode: str = eqx.field(static=True)
    coordinate_density: str = eqx.field(static=True)
    acquisition_mode: str = eqx.field(static=True)
    fixed_total_count: Optional[int] = eqx.field(static=True)
    domain_frame_ids: Tuple[str, ...] = eqx.field(static=True)


def _require_mode(value: str, allowed: Tuple[str, ...], name: str) -> None:
    """PRIVATE: Reject an unsupported static selector.

    The helper applies one consistent diagnostic to every discrete effects
    mode.

    Parameters
    ----------
    value : str
        Candidate selector.
    allowed : Tuple[str, ...]
        Exact accepted selector values.
    name : str
        Reader-facing selector name.

    Raises
    ------
    ValueError
        If ``value`` does not appear in ``allowed``.
    """
    if value not in allowed:
        raise ValueError(f"make_detector_effects: unsupported {name}")


def _finite(  # noqa: DOC502
    values: Float64[Array, "..."], name: str
) -> Float64[Array, "..."]:
    """PRIVATE: Bind a compiled finite-value check to one traced leaf.

    The returned value carries the Equinox error dependency through compiled
    execution.

    Parameters
    ----------
    values : Float64[Array, "..."]
        Candidate traced numerical leaf.
    name : str
        Reader-facing leaf name for diagnostics.

    Returns
    -------
    validated_values : Float64[Array, "..."]
        Input leaf with the compiled finite check attached.

    Raises
    ------
    EquinoxRuntimeError
        If any value is non-finite.
    """
    validated_values: Float64[Array, "..."] = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        f"make_detector_effects: {name} finite",
    )
    return validated_values


@jaxtyped(typechecker=beartype)
def make_detector_effects(  # noqa: DOC503, PLR0912, PLR0913, PLR0915, PLR0917
    domain_logits: Float64[Array, " D"],
    domain_euler_angles_rad: Float64[Array, "Dr 3"],
    transmission_raw_slopes: Float64[Array, " Q"],
    background_coefficients: Float64[Array, " B"],
    sensitivity_coefficients: Float64[Array, " S"],
    exposure: ScalarFloat,
    *,
    background_mode: str,
    sensitivity_mode: str,
    post_count_mode: str = "none",
    post_count_kernel: Optional[Float64[Array, " K"]] = None,
    coordinate_density: str = _COORDINATE_DENSITY,
    acquisition_mode: str = "poisson",
    fixed_total_count: Optional[int] = None,
    domain_frame_ids: Tuple[str, ...],
) -> DetectorEffects:
    """Create validated detector-effects state.

    The factory fixes every v1 structural choice before it binds numerical
    validation to the traced leaves.

    :see: :class:`~.test_detector_effects.TestMakeDetectorEffects`

    Parameters
    ----------
    domain_logits : Float64[Array, " D"]
        One unconstrained mixture logit per source domain.
    domain_euler_angles_rad : Float64[Array, "Dr 3"]
        One active Euler-angle triple per source domain, in radians.
    transmission_raw_slopes : Float64[Array, " Q"]
        Two or three analyser-transmission coordinates.
    background_coefficients : Float64[Array, " B"]
        Mode-specific raw background coordinates. Smooth mode accepts the
        map and slit lengths here; calibration selects the exact length.
    sensitivity_coefficients : Float64[Array, " S"]
        Mode-specific raw sensitivity coordinates. Smooth mode accepts the
        map and slit lengths here; calibration selects the exact length.
    exposure : ScalarFloat
        Positive acquisition exposure.
    background_mode : str
        ``"flat"``, ``"shirley"``, or ``"smooth"``.
    sensitivity_mode : str
        ``"constant"`` or ``"smooth"``.
    post_count_mode : str, optional
        ``"none"`` or ``"calibrated"``. Default is ``"none"``.
    post_count_kernel : Optional[Float64[Array, " K"]], optional
        Odd, nonnegative calibrated energy-response kernel. This factory
        normalizes its positive sum. Default is ``None``.
    coordinate_density : str, optional
        Must be ``"per_native_volume"``. Default is that value.
    acquisition_mode : str, optional
        ``"poisson"`` or ``"fixed_total"``. Default is ``"poisson"``.
    fixed_total_count : Optional[int], optional
        Positive static event count for fixed-total acquisition. Default is
        ``None``.
    domain_frame_ids : Tuple[str, ...]
        One registered Cartesian source-frame identifier per domain.

    Returns
    -------
    effects : DetectorEffects
        Validated immutable carrier with float64 numerical leaves.

    Raises
    ------
    ValueError
        If a static selector, shape, coefficient length, frame identifier,
        kernel structure, or acquisition contract is invalid.
    EquinoxRuntimeError
        If a traced leaf is non-finite, exposure is not positive, or a
        calibrated kernel is negative or has a nonpositive sum.

    Notes
    -----
    Smooth coefficient lengths are calibration-dependent. The factory
    permits exactly the v1 slit/map alternatives; the deterministic effects
    functions enforce the one selected by a concrete calibration.
    """
    logits: Float64[Array, " D"] = jnp.asarray(
        domain_logits, dtype=jnp.float64
    )
    rotations: Float64[Array, "D 3"] = jnp.asarray(
        domain_euler_angles_rad, dtype=jnp.float64
    )
    transmission: Float64[Array, " Q"] = jnp.asarray(
        transmission_raw_slopes, dtype=jnp.float64
    )
    background: Float64[Array, " B"] = jnp.asarray(
        background_coefficients, dtype=jnp.float64
    )
    sensitivity: Float64[Array, " S"] = jnp.asarray(
        sensitivity_coefficients, dtype=jnp.float64
    )
    exposure_arr: Float64[Array, ""] = jnp.asarray(exposure, dtype=jnp.float64)

    if logits.ndim != 1 or logits.shape[0] < 1:
        raise ValueError(
            "make_detector_effects: domain logits must be nonempty and 1D"
        )
    if rotations.shape != (logits.shape[0], 3):
        raise ValueError(
            "make_detector_effects: domain rotations and logits disagree"
        )
    if transmission.ndim != 1 or transmission.shape[0] not in (2, 3):
        raise ValueError(
            "make_detector_effects: transmission length must be 2 or 3"
        )
    if background.ndim != 1 or sensitivity.ndim != 1:
        raise ValueError(
            "make_detector_effects: coefficient arrays must be 1D"
        )
    if exposure_arr.ndim != 0:
        raise ValueError("make_detector_effects: exposure must be scalar")

    _require_mode(background_mode, _BACKGROUND_MODES, "background mode")
    _require_mode(sensitivity_mode, _SENSITIVITY_MODES, "sensitivity mode")
    _require_mode(post_count_mode, _POST_COUNT_MODES, "post-count mode")
    _require_mode(acquisition_mode, _ACQUISITION_MODES, "acquisition mode")
    if coordinate_density != _COORDINATE_DENSITY:
        raise ValueError(
            "make_detector_effects: coordinate density must be "
            "per_native_volume"
        )

    expected_background_lengths: Tuple[int, ...]
    if background_mode == "flat":
        expected_background_lengths = (1,)
    elif background_mode == "shirley":
        expected_background_lengths = (2,)
    else:
        expected_background_lengths = (5, 7)
    if background.shape[0] not in expected_background_lengths:
        raise ValueError(
            "make_detector_effects: background coefficient length disagrees "
            "with mode"
        )

    expected_sensitivity_lengths: Tuple[int, ...]
    if sensitivity_mode == "constant":
        expected_sensitivity_lengths = (0,)
    else:
        expected_sensitivity_lengths = (4, 6)
    if sensitivity.shape[0] not in expected_sensitivity_lengths:
        raise ValueError(
            "make_detector_effects: sensitivity coefficient length "
            "disagrees with mode"
        )
    if (
        background_mode == "smooth"
        and sensitivity_mode == "smooth"
        and background.shape[0] != sensitivity.shape[0] + 1
    ):
        raise ValueError(
            "make_detector_effects: smooth background and sensitivity axes "
            "disagree"
        )

    if len(domain_frame_ids) != logits.shape[0]:
        raise ValueError(
            "make_detector_effects: domain frame identifiers and logits "
            "disagree"
        )
    if any(
        frame_id not in _REGISTERED_DOMAIN_FRAME_IDS
        for frame_id in domain_frame_ids
    ):
        raise ValueError(
            "make_detector_effects: unregistered domain frame identifier"
        )

    kernel: Optional[Float64[Array, " K"]] = None
    if post_count_mode == "none":
        if post_count_kernel is not None:
            raise ValueError(
                "make_detector_effects: none post-count mode requires no "
                "kernel"
            )
    else:
        if post_count_kernel is None:
            raise ValueError(
                "make_detector_effects: calibrated post-count mode requires "
                "a kernel"
            )
        kernel = jnp.asarray(post_count_kernel, dtype=jnp.float64)
        if kernel.ndim != 1 or kernel.shape[0] < 1:
            raise ValueError(
                "make_detector_effects: post-count kernel must be nonempty "
                "and 1D"
            )
        if kernel.shape[0] % 2 != 1:
            raise ValueError(
                "make_detector_effects: post-count kernel length must be odd"
            )

    if acquisition_mode == "poisson":
        if fixed_total_count is not None:
            raise ValueError(
                "make_detector_effects: poisson mode requires no fixed total"
            )
    elif type(fixed_total_count) is not int or fixed_total_count <= 0:
        raise ValueError(
            "make_detector_effects: fixed-total mode requires a positive "
            "integer count"
        )

    logits = _finite(logits, "domain logits")
    rotations = _finite(rotations, "domain rotations")
    transmission = _finite(transmission, "transmission slopes")
    background = _finite(background, "background coefficients")
    sensitivity = _finite(sensitivity, "sensitivity coefficients")
    exposure_arr = eqx.error_if(
        exposure_arr,
        ~jnp.isfinite(exposure_arr) | (exposure_arr <= 0.0),
        "make_detector_effects: exposure finite and positive",
    )
    if kernel is not None:
        kernel = eqx.error_if(
            kernel,
            ~jnp.all(jnp.isfinite(kernel)) | ~jnp.all(kernel >= 0.0),
            "make_detector_effects: post-count kernel finite and nonnegative",
        )
        kernel_sum: Float64[Array, ""] = jnp.sum(kernel)
        kernel_sum = eqx.error_if(
            kernel_sum,
            ~(kernel_sum > 0.0),
            "make_detector_effects: post-count kernel sum positive",
        )
        kernel = kernel / kernel_sum

    effects: DetectorEffects = DetectorEffects(
        domain_logits=logits,
        domain_euler_angles_rad=rotations,
        transmission_raw_slopes=transmission,
        background_coefficients=background,
        sensitivity_coefficients=sensitivity,
        exposure=exposure_arr,
        post_count_kernel=kernel,
        background_mode=background_mode,
        sensitivity_mode=sensitivity_mode,
        post_count_mode=post_count_mode,
        coordinate_density=coordinate_density,
        acquisition_mode=acquisition_mode,
        fixed_total_count=fixed_total_count,
        domain_frame_ids=domain_frame_ids,
    )
    return effects
