"""Validate static transformation-contract carriers and their factories.

The cases cover immutable field storage, validity consistency, and the
identity and semantic-term rejection rules owned by ``diffpes.types``.
"""

import pytest
from beartype.typing import Dict

from diffpes.types import (
    CompositionReport,
    TransformationContract,
    make_composition_report,
    make_transformation_contract,
)


class TestTransformationContract:
    """Validate :class:`~diffpes.types.TransformationContract` storage.

    The carrier must preserve the normalized static semantic contract returned
    by its public factory.

    :see: :class:`~diffpes.types.TransformationContract`
    """

    def test_stores_static_semantics(self) -> None:
        """Preserve every declared semantic term without reordering it.

        The check verifies the carrier fields against explicitly supplied
        semantic labels and the declared JAX-purity flag.

        Notes
        -----
        The test constructs one contract through the validated factory and
        compares its
        immutable tuples and Boolean flag with the independent input values.
        """
        contract: TransformationContract = make_transformation_contract(
            "org.diffpes.test.contract",
            "1.2.3",
            requires=("bands",),
            produces=("intensity",),
            preserves=("energy-axis",),
            jax_pure=True,
        )

        assert contract.requires == ("bands",)
        assert contract.produces == ("intensity",)
        assert contract.preserves == ("energy-axis",)
        assert contract.jax_pure is True


class TestCompositionReport:
    """Validate :class:`~diffpes.types.CompositionReport` storage.

    The carrier must expose the cumulative semantics and transformation order
    recorded by its public factory.

    :see: :class:`~diffpes.types.CompositionReport`
    """

    def test_stores_composition_state(self) -> None:
        """Preserve successful composition state and ordered references.

        The check verifies that an error-free report retains its final
        semantic set and transformation order exactly.

        Notes
        -----
        The test builds a valid report with two transformation references and
        compares
        each static field with the independently specified tuples.
        """
        report: CompositionReport = make_composition_report(
            valid=True,
            errors=(),
            available_semantics=("bands", "intensity"),
            destroyed_information=("phase",),
            invalidated_claims=("coherent-amplitude",),
            transformation_refs=("a@1.0.0", "b@1.0.0"),
        )

        assert report.valid is True
        assert report.available_semantics == ("bands", "intensity")
        assert report.transformation_refs == ("a@1.0.0", "b@1.0.0")


class TestMakeTransformationContract:
    """Validate :func:`~diffpes.types.make_transformation_contract`.

    The factory must reject malformed identities, duplicate terms, and
    contradictory preservation and destruction declarations.

    :see: :func:`~diffpes.types.make_transformation_contract`
    """

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {
                    "transformation_id": "invalid",
                    "transformation_version": "1.0.0",
                },
                "reverse-DNS",
            ),
            (
                {
                    "transformation_id": "org.diffpes.test",
                    "transformation_version": "one",
                },
                "semantic version",
            ),
            (
                {
                    "transformation_id": "org.diffpes.test",
                    "transformation_version": "1.0.0",
                    "requires": ("bands", "bands"),
                },
                "unique",
            ),
            (
                {
                    "transformation_id": "org.diffpes.test",
                    "transformation_version": "1.0.0",
                    "preserves": ("phase",),
                    "destroys": ("phase",),
                },
                "preserve/introduce and destroy",
            ),
        ],
    )
    def test_rejects_invalid_contracts(
        self,
        kwargs: Dict[str, object],
        message: str,
    ) -> None:
        """Reject every malformed static contract declaration.

        The parameter table covers invalid identities and versions, duplicate
        terms, and mutually contradictory information-flow declarations.

        Notes
        -----
        The test calls the factory once for each invalid declaration.
        It matches the diagnostic text and identifies each static branch.
        """
        with pytest.raises(ValueError, match=message):
            make_transformation_contract(**kwargs)


class TestMakeCompositionReport:
    """Validate :func:`~diffpes.types.make_composition_report`.

    The factory must keep its validity flag equivalent to an empty error set.

    :see: :func:`~diffpes.types.make_composition_report`
    """

    def test_rejects_inconsistent_validity(self) -> None:
        """Reject a successful flag paired with composition errors.

        The check isolates the factory invariant that ``valid`` is true if and
        only if the error sequence is empty.

        Notes
        -----
        The test supplies one explicit error with ``valid=True``.
        It matches the consistency diagnostic before report construction.
        """
        with pytest.raises(ValueError, match="valid must be true"):
            make_composition_report(
                valid=True,
                errors=("missing semantic",),
                available_semantics=(),
                destroyed_information=(),
                invalidated_claims=(),
                transformation_refs=(),
            )
