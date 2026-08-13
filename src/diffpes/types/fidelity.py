"""Define immutable scientific-fidelity declarations.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`DerivativeCapability`
    Define the ``DerivativeCapability`` public contract.
:class:`FidelityManifest`
    Define the ``FidelityManifest`` public contract.
:func:`make_derivative_capability`
    Compute the ``make_derivative_capability`` public contract.
:func:`make_fidelity_manifest`
    Compute the ``make_fidelity_manifest`` public contract.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import jaxtyped

from diffpes.constants import DERIVATIVE_CAPABILITY_MODES


class DerivativeCapability(eqx.Module):
    """Define the ``DerivativeCapability`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_fidelity.TestDerivativecapability`

    Attributes
    ----------
    input_path : str
        Store the differentiable input path.
    mode : str
        Store the derivative mode.
    policy_ref : str
        Store the derivative-policy identity.

    See Also
    --------
    make_derivative_capability
        Construct a validated derivative capability.
    """

    input_path: str = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    policy_ref: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static derivative-boundary metadata.

        Raises
        ------
        ValueError
            When a reference is empty or the mode has no support.

        Notes
        -----
        This private Equinox hook is intentionally side-effect free.
        """
        if not self.input_path or not self.policy_ref:
            raise ValueError(
                "derivative capability references must be nonempty"
            )
        if self.mode not in DERIVATIVE_CAPABILITY_MODES:
            raise ValueError("unknown derivative capability mode")


class FidelityManifest(eqx.Module):
    """Define the ``FidelityManifest`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_fidelity.TestFidelitymanifest`

    Attributes
    ----------
    schema_version : str
        Store the schema version.
    model_ref : str
        Store the model identity.
    instrument_ref : str
        Store the instrument identity.
    acquisition_ref : str
        Store the acquisition identity.
    initial_state : str
        Store the initial-state declaration.
    spectral_physics : str
        Store the spectral-physics declaration.
    photocurrent : str
        Store the photocurrent declaration.
    light_interaction : str
        Store the light-interaction declaration.
    instrument : str
        Store the instrument declaration.
    derivative_capabilities : Tuple[DerivativeCapability, ...]
        Store derivative capabilities.
    validation_refs : Tuple[str, ...]
        Store validation identities.
    validity_domain_refs : Tuple[str, ...]
        Store validity-domain identities.
    discrepancy_ref : Optional[str]
        Store the discrepancy-model identity.

    See Also
    --------
    make_fidelity_manifest
        Construct a validated fidelity manifest.
    """

    schema_version: str = eqx.field(static=True)
    model_ref: str = eqx.field(static=True)
    instrument_ref: str = eqx.field(static=True)
    acquisition_ref: str = eqx.field(static=True)
    initial_state: str = eqx.field(static=True)
    spectral_physics: str = eqx.field(static=True)
    photocurrent: str = eqx.field(static=True)
    light_interaction: str = eqx.field(static=True)
    instrument: str = eqx.field(static=True)
    derivative_capabilities: Tuple[DerivativeCapability, ...] = eqx.field(
        static=True
    )
    validation_refs: Tuple[str, ...] = eqx.field(static=True)
    validity_domain_refs: Tuple[str, ...] = eqx.field(static=True)
    discrepancy_ref: Optional[str] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate the complete static scientific identity.

        Raises
        ------
        ValueError
            If a required declaration or reference is empty.
        """
        required: Tuple[str, ...] = (
            self.schema_version,
            self.model_ref,
            self.instrument_ref,
            self.acquisition_ref,
            self.initial_state,
            self.spectral_physics,
            self.photocurrent,
            self.light_interaction,
            self.instrument,
        )
        if not all(required):
            raise ValueError("fidelity manifest fields must be nonempty")
        if any(not reference for reference in self.validation_refs):
            raise ValueError("validation references must be nonempty")
        if any(not reference for reference in self.validity_domain_refs):
            raise ValueError("validity-domain references must be nonempty")
        input_paths: Tuple[str, ...] = tuple(
            capability.input_path
            for capability in self.derivative_capabilities
        )
        if len(set(input_paths)) != len(input_paths):
            raise ValueError("derivative capability paths must be unique")
        if self.discrepancy_ref == "":
            raise ValueError("discrepancy reference must be nonempty when set")


@jaxtyped(typechecker=beartype)
def make_derivative_capability(
    input_path: str,
    mode: str,
    policy_ref: str,
) -> DerivativeCapability:
    """Compute the ``make_derivative_capability`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_fidelity.TestMakeDerivativeCapability`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    input_path : str
        Input value for this operation.
    mode : str
        Input value for this operation.
    policy_ref : str
        Input value for this operation.

    Returns
    -------
    result : DerivativeCapability
        Validated operation result.
    """
    result: DerivativeCapability = DerivativeCapability(
        input_path, mode, policy_ref
    )
    return result


@jaxtyped(typechecker=beartype)  # noqa: PLR0913
def make_fidelity_manifest(  # noqa: PLR0913
    *,
    schema_version: str,
    model_ref: str,
    instrument_ref: str,
    acquisition_ref: str,
    initial_state: str,
    spectral_physics: str,
    photocurrent: str,
    light_interaction: str,
    instrument: str,
    derivative_capabilities: Tuple[DerivativeCapability, ...] = (),
    validation_refs: Tuple[str, ...] = (),
    validity_domain_refs: Tuple[str, ...] = (),
    discrepancy_ref: Optional[str] = None,
) -> FidelityManifest:
    """Compute the ``make_fidelity_manifest`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_fidelity.TestMakeFidelityManifest`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    schema_version : str
        Input value for this operation.
    model_ref : str
        Input value for this operation.
    instrument_ref : str
        Input value for this operation.
    acquisition_ref : str
        Input value for this operation.
    initial_state : str
        Input value for this operation.
    spectral_physics : str
        Input value for this operation.
    photocurrent : str
        Input value for this operation.
    light_interaction : str
        Input value for this operation.
    instrument : str
        Input value for this operation.
    derivative_capabilities : Tuple[DerivativeCapability, ...]
        Input value for this operation.
    validation_refs : Tuple[str, ...]
        Input value for this operation.
    validity_domain_refs : Tuple[str, ...]
        Input value for this operation.
    discrepancy_ref : Optional[str]
        Input value for this operation.

    Returns
    -------
    result : FidelityManifest
        Validated operation result.
    """
    result: FidelityManifest = FidelityManifest(
        schema_version,
        model_ref,
        instrument_ref,
        acquisition_ref,
        initial_state,
        spectral_physics,
        photocurrent,
        light_interaction,
        instrument,
        derivative_capabilities,
        validation_refs,
        validity_domain_refs,
        discrepancy_ref,
    )
    return result


__all__: list[str] = [
    "DerivativeCapability",
    "FidelityManifest",
    "make_derivative_capability",
    "make_fidelity_manifest",
]
