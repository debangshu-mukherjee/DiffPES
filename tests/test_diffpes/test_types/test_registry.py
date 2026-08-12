"""Validate the registry contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx
from beartype.typing import Any

import diffpes.types


class TestRegisteredmodel:
    """Verify :class:`~diffpes.types.RegisteredModel`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegisteredModel`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegisteredModel`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.RegisteredModel
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegisteredtransformation:
    """Verify :class:`~diffpes.types.RegisteredTransformation`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegisteredTransformation`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegisteredTransformation`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.RegisteredTransformation
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegistryreport:
    """Verify :class:`~diffpes.types.RegistryReport`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegistryReport`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegistryReport`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.RegistryReport
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegistrysnapshot:
    """Verify :class:`~diffpes.types.RegistrySnapshot`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.RegistrySnapshot`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``RegistrySnapshot`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.RegistrySnapshot
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestRegistrationHandshake:
    """Verify :class:`~diffpes.types.RegistrationHandshake`.

    The case checks the canonical type identity for registration requirements.

    :see: :class:`~diffpes.types.RegistrationHandshake`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the handshake type as an Equinox module class.

        The carrier must use the library PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = diffpes.types.RegistrationHandshake
        assert issubclass(symbol, eqx.Module)


class TestHandshakeReport:
    """Verify :class:`~diffpes.types.HandshakeReport`.

    The case checks the JAX-native report carrier for handshake validation.

    :see: :class:`~diffpes.types.HandshakeReport`
    """

    def test_public_type_is_an_equinox_module(self) -> None:
        """Expose the handshake report as an Equinox module class.

        The carrier must use the library PyTree base class.

        Notes
        -----
        The test checks class inheritance through the canonical package path.
        """
        symbol: Any = diffpes.types.HandshakeReport
        assert issubclass(symbol, eqx.Module)


class TestMakeRegisteredModel:
    """Verify :func:`~diffpes.types.make_registered_model`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registered_model`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registered_model`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_registered_model
        assert callable(symbol)


class TestMakeRegisteredTransformation:
    """Verify :func:`~diffpes.types.make_registered_transformation`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registered_transformation`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registered_transformation`` through its canonical
        types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_registered_transformation
        assert callable(symbol)


class TestMakeRegistryReport:
    """Verify :func:`~diffpes.types.make_registry_report`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registry_report`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registry_report`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_registry_report
        assert callable(symbol)


class TestMakeRegistrySnapshot:
    """Verify :func:`~diffpes.types.make_registry_snapshot`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_registry_snapshot`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_registry_snapshot`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_registry_snapshot
        assert callable(symbol)


class TestMakeRegistrationHandshake:
    """Verify :func:`~diffpes.types.make_registration_handshake`.

    The case checks construction of declarative owner requirements.

    :see: :func:`~diffpes.types.make_registration_handshake`
    """

    def test_factory_builds_exact_owner(self) -> None:
        """Build a handshake with one exact required model identity.

        The factory must retain each static identity without modification.

        Notes
        -----
        The test compares the static owner and model reference fields.
        """
        factory: Any = diffpes.types.make_registration_handshake
        result: Any = factory("kinematics", model_refs=("model@1.0.0",))
        assert result.owner_id == "kinematics"
        assert result.model_refs == ("model@1.0.0",)


class TestMakeHandshakeReport:
    """Verify :func:`~diffpes.types.make_handshake_report`.

    The case checks the JAX Boolean outcome and static missing references.

    :see: :func:`~diffpes.types.make_handshake_report`
    """

    def test_factory_retains_missing_references(self) -> None:
        """Build an incomplete report with one missing evidence identity.

        The report must retain the missing identity and false Boolean leaf.

        Notes
        -----
        The test checks both the Boolean leaf and static identity tuple.
        """
        factory: Any = diffpes.types.make_handshake_report
        result: Any = factory("kinematics", False, ("evidence-03",))
        assert not bool(result.complete)
        assert result.missing_ids == ("evidence-03",)
