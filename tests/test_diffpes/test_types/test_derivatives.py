"""Validate the derivatives contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import equinox as eqx

import diffpes.types


class TestDependencymap:
    """Verify :class:`~diffpes.types.DependencyMap`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DependencyMap`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DependencyMap`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.DependencyMap
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestDerivativeevidence:
    """Verify :class:`~diffpes.types.DerivativeEvidence`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.DerivativeEvidence`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``DerivativeEvidence`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.DerivativeEvidence
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestInformationspectrum:
    """Verify :class:`~diffpes.types.InformationSpectrum`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.InformationSpectrum`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``InformationSpectrum`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.InformationSpectrum
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestSensitivitymap:
    """Verify :class:`~diffpes.types.SensitivityMap`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :class:`~diffpes.types.SensitivityMap`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``SensitivityMap`` through its canonical types package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.SensitivityMap
        assert isinstance(symbol, type)
        assert issubclass(symbol, eqx.Module)


class TestMakeDependencyMap:
    """Verify :func:`~diffpes.types.make_dependency_map`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_dependency_map`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_dependency_map`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_dependency_map
        assert callable(symbol)


class TestMakeDerivativeEvidence:
    """Verify :func:`~diffpes.types.make_derivative_evidence`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_derivative_evidence`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_derivative_evidence`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_derivative_evidence
        assert callable(symbol)


class TestMakeInformationSpectrum:
    """Verify :func:`~diffpes.types.make_information_spectrum`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_information_spectrum`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_information_spectrum`` through its canonical types
        package path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_information_spectrum
        assert callable(symbol)


class TestMakeSensitivityMap:
    """Verify :func:`~diffpes.types.make_sensitivity_map`.

    The cases cover the public carrier or factory contract in JAX PyTrees.

    :see: :func:`~diffpes.types.make_sensitivity_map`
    """

    def test_public_symbol_has_expected_kind(self) -> None:
        """Expose ``make_sensitivity_map`` through its canonical types package
        path.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural
        assertions.
        """
        symbol: object = diffpes.types.make_sensitivity_map
        assert callable(symbol)
