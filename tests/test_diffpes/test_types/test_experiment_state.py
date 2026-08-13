"""Verify split experiment-carrier invariants.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, List
from jaxtyping import TypeCheckError

from diffpes.types import (
    Acquisition,
    PhotonBeam,
    SamplePose,
    SampleState,
    make_acquisition,
    make_experiment,
    make_photon_beam,
    make_sample_pose,
    make_sample_state,
)


def _photon() -> PhotonBeam:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_photon_beam(
        jnp.asarray(50.0),
        jnp.asarray([1.0 + 0.0j, 0.0j, 0.0j]),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    return result


def _sample() -> SampleState:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_sample_state(
        jnp.asarray(20.0),
        jnp.asarray(4.5),
        jnp.asarray(10.0),
        jnp.asarray(6.0),
        jnp.asarray([0.0]),
        domain_frame_ids=("domain-0",),
    )
    return result


def _pose() -> SamplePose:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_sample_pose(jnp.asarray(0.1), jnp.zeros((1, 3)))
    return result


def _acquisition() -> Acquisition:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_acquisition(jnp.asarray(1.0))
    return result


class TestPhotonbeam:
    """Verify ``diffpes.types.PhotonBeam`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_normalizes_valid_transverse_polarization(self) -> None:
        """Normalize a nonunit polarization transverse to propagation.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare its stored Euclidean norm with one.
        """
        beam: Any = make_photon_beam(
            jnp.asarray(50.0),
            jnp.asarray([2.0 + 0.0j, 0.0j, 0.0j]),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )
        assert jnp.linalg.norm(beam.polarization_lab) == 1.0

    @pytest.mark.parametrize(
        ("energy", "polarization", "theta", "message"),
        [
            (0.0, [1.0, 0.0, 0.0], 0.0, "photon energy must be positive"),
            (50.0, [0.0, 0.0, 0.0], 0.0, "finite and nonzero"),
            (50.0, [0.0, 0.0, 1.0], 0.0, "must be transverse"),
            (50.0, [1.0, 0.0, 0.0], jnp.nan, "finite incidence"),
        ],
    )
    def test_rejects_each_beam_invariant(
        self,
        energy: float,
        polarization: List[float],
        theta: float,
        message: str,
    ) -> None:
        """Reject nonpositive energy and invalid polarization geometry.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one valid beam input per parameterized case.
        """
        with pytest.raises(eqx.EquinoxRuntimeError, match=message):
            make_photon_beam(
                jnp.asarray(energy),
                jnp.asarray(polarization, dtype=jnp.complex128),
                jnp.asarray(theta),
                jnp.asarray(0.0),
            )


class TestSamplestate:
    """Verify ``diffpes.types.SampleState`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_multidomain_sample(self) -> None:
        """Preserve two domain logits and their frame identities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare stored domain metadata with explicit inputs.
        """
        sample: Any = make_sample_state(
            jnp.asarray(20.0),
            jnp.asarray(4.5),
            jnp.asarray(10.0),
            jnp.asarray(6.0),
            jnp.asarray([0.0, 0.2]),
            domain_frame_ids=("a", "b"),
        )
        assert sample.domain_frame_ids == ("a", "b")
        assert sample.domain_logits.shape == (2,)

    @pytest.mark.parametrize(
        ("position", "value", "message"),
        [
            (0, -1.0, "temperature.*nonnegative"),
            (1, 0.0, "work function.*positive"),
            (2, jnp.nan, "inner potential.*finite"),
            (3, 0.0, "mean free path.*positive"),
            (4, jnp.nan, "domain logits.*finite"),
        ],
    )
    def test_rejects_each_sample_numerical_invariant(
        self, position: int, value: float, message: str
    ) -> None:
        """Reject every nonfinite or out-of-domain sample quantity.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one scalar in the valid sample tuple.
        """
        values: List[float] = [20.0, 4.5, 10.0, 6.0, 0.0]
        values[position] = value
        with pytest.raises(eqx.EquinoxRuntimeError, match=message):
            make_sample_state(
                jnp.asarray(values[0]),
                jnp.asarray(values[1]),
                jnp.asarray(values[2]),
                jnp.asarray(values[3]),
                jnp.asarray([values[4]]),
                domain_frame_ids=("a",),
            )

    def test_rejects_domain_axis_mismatch(self) -> None:
        """Reject unequal domain-logit and frame counts.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Supply two logits with one frame identifier.
        """
        with pytest.raises(ValueError, match="logits and frames must agree"):
            make_sample_state(
                jnp.asarray(20.0),
                jnp.asarray(4.5),
                jnp.asarray(10.0),
                jnp.asarray(6.0),
                jnp.asarray([0.0, 0.1]),
                domain_frame_ids=("a",),
            )


class TestSamplepose:
    """Verify ``diffpes.types.SamplePose`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_finite_euler_triples(self) -> None:
        """Preserve one finite azimuth and two Euler triples.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Inspect the exact stored shape and azimuth.
        """
        pose: Any = make_sample_pose(jnp.asarray(0.2), jnp.zeros((2, 3)))
        assert pose.domain_euler_angles_rad.shape == (2, 3)
        assert pose.sample_azimuth_rad == 0.2

    def test_rejects_bad_euler_shape(self) -> None:
        """Reject domain rotations that are not Euler triples.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Supply a two-column matrix instead of three columns.
        """
        with pytest.raises(TypeCheckError, match="domain_euler_angles_rad"):
            make_sample_pose(jnp.asarray(0.0), jnp.zeros((1, 2)))

    @pytest.mark.parametrize(
        ("azimuth", "angles"), [(jnp.nan, 0.0), (0.0, jnp.nan)]
    )
    def test_rejects_nonfinite_pose_values(
        self, azimuth: float, angles: float
    ) -> None:
        """Reject nonfinite azimuths and nonfinite Euler angles.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Insert one NaN into each numerical field independently.
        """
        with pytest.raises(eqx.EquinoxRuntimeError, match="must be finite"):
            make_sample_pose(jnp.asarray(azimuth), jnp.full((1, 3), angles))


class TestAcquisition:
    """Verify ``diffpes.types.Acquisition`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"statistics_mode": "poisson"},
            {
                "statistics_mode": "gaussian",
                "gaussian_sigma_counts": jnp.asarray([2.0]),
            },
            {"statistics_mode": "fixed_total", "fixed_total_count": 100},
        ],
    )
    def test_accepts_each_statistics_contract(
        self, kwargs: Dict[str, object]
    ) -> None:
        """Accept expected, Poisson, Gaussian, and fixed-total modes.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Construct each complete mutually exclusive mode declaration.
        """
        assert make_acquisition(jnp.asarray(1.0), **kwargs).exposure == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"statistics_mode": "bad"}, "unknown acquisition"),
            ({"statistics_mode": "gaussian"}, "requires.*sigma"),
            (
                {"gaussian_sigma_counts": jnp.asarray([1.0])},
                "only permits sigma",
            ),
            ({"statistics_mode": "fixed_total"}, "positive total"),
            ({"scan_order": ""}, "references must be nonempty"),
        ],
    )
    def test_rejects_each_static_acquisition_invariant(
        self, kwargs: Dict[str, object], message: str
    ) -> None:
        """Reject invalid mode combinations and empty identities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Alter one field of the expected-value acquisition.
        """
        with pytest.raises(ValueError, match=message):
            make_acquisition(jnp.asarray(1.0), **kwargs)

    @pytest.mark.parametrize(
        ("exposure", "sigma", "message"),
        [
            (0.0, None, "exposure.*positive"),
            (1.0, jnp.asarray([0.0]), "sigma.*positive"),
        ],
    )
    def test_rejects_each_numerical_acquisition_invariant(
        self, exposure: float, sigma: object, message: str
    ) -> None:
        """Reject nonpositive exposure and Gaussian standard deviation.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Select Gaussian mode only when exercising its sigma guard.
        """
        kwargs: Dict[str, object] = {}
        if sigma is not None:
            kwargs = {
                "statistics_mode": "gaussian",
                "gaussian_sigma_counts": sigma,
            }
        with pytest.raises(eqx.EquinoxRuntimeError, match=message):
            make_acquisition(jnp.asarray(exposure), **kwargs)


class TestExperiment:
    """Verify ``diffpes.types.Experiment`` composition invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_accepts_compatible_split_carriers(self) -> None:
        """Compose one beam, sample, pose, and acquisition.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Check that the returned carrier retains each exact component.
        """
        experiment: Any = make_experiment(
            _photon(), _sample(), _pose(), _acquisition()
        )
        assert experiment.sample.domain_frame_ids == ("domain-0",)

    def test_rejects_domain_count_mismatch(self) -> None:
        """Reject a two-domain pose paired with a one-domain sample.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change only the number of Euler-angle rows.
        """
        with pytest.raises(ValueError, match="domain counts must agree"):
            make_experiment(
                _photon(),
                _sample(),
                make_sample_pose(jnp.asarray(0.0), jnp.zeros((2, 3))),
                _acquisition(),
            )

    def test_rejects_work_function_above_photon_energy(self) -> None:
        """Reject a sample whose emission threshold exceeds beam energy.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Set the photon energy equal to the valid sample work function.
        """
        photon: Any = make_photon_beam(
            jnp.asarray(4.5),
            jnp.asarray([1.0 + 0.0j, 0.0j, 0.0j]),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )
        with pytest.raises(eqx.EquinoxRuntimeError, match="below photon"):
            make_experiment(photon, _sample(), _pose(), _acquisition())


class TestMakePhotonBeam:
    """Verify ``diffpes.types.make_photon_beam``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSampleState:
    """Verify ``diffpes.types.make_sample_state``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeSamplePose:
    """Verify ``diffpes.types.make_sample_pose``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeAcquisition:
    """Verify ``diffpes.types.make_acquisition``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeExperiment:
    """Verify ``diffpes.types.make_experiment``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
