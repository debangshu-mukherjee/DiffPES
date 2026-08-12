"""Define immutable scientific-fidelity declarations.

Extended Summary
----------------
Fidelity manifests record the non-numerical scientific identity resolved for
one forward result.  They deliberately keep references and derivative
boundaries static so that changing a declaration changes the PyTree identity
without introducing differentiable leaves.

Routine Listings
----------------
:class:`DerivativeCapability`
    Declare the derivative treatment of one stable input path.
:class:`FidelityManifest`
    Declare the resolved scientific identity of a simulation result.
:func:`make_derivative_capability`
    Construct a validated derivative-capability declaration.
:func:`make_fidelity_manifest`
    Construct a validated immutable fidelity manifest.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import jaxtyped

from diffpes.constants import DERIVATIVE_CAPABILITY_MODES


class DerivativeCapability(eqx.Module):
    """Declare derivative treatment for one stable parameter path.

    Attributes
    ----------
    input_path : str
        Stable dotted path naming the differentiated input.
    mode : str
        One of :obj:`diffpes.constants.DERIVATIVE_CAPABILITY_MODES`.
    policy_ref : str
        Stable reference describing the derivative policy.

    See Also
    --------
    FidelityManifest
        Collects declarations for one resolved scientific calculation.
    """

    input_path: str = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    policy_ref: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate static derivative-boundary metadata.

        Raises
        ------
        ValueError
            If a reference is empty or the mode is unsupported.

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
    """Store the resolved scientific identity of a simulation result.

    Attributes
    ----------
    schema_version : str
        Version of the manifest schema.
    model_ref : str
        Registered forward-model identity.
    instrument_ref : str
        Registered instrument identity.
    acquisition_ref : str
        Registered acquisition-model identity.
    initial_state : str
        Initial-state modelling declaration.
    spectral_physics : str
        Spectral-function modelling declaration.
    photocurrent : str
        Intrinsic-photocurrent modelling declaration.
    light_interaction : str
        Light--matter interaction declaration.
    instrument : str
        Detector and instrument-response declaration.
    derivative_capabilities : Tuple[DerivativeCapability, ...]
        Derivative boundaries for exposed parameter paths.
    validation_refs : Tuple[str, ...]
        References to validation evidence.
    validity_domain_refs : Tuple[str, ...]
        References to declared validity domains.
    discrepancy_ref : Optional[str]
        Optional model-discrepancy declaration.

    See Also
    --------
    DerivativeCapability
        Defines each element of ``derivative_capabilities``.
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
        required = (
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


@jaxtyped(typechecker=beartype)
def make_derivative_capability(
    input_path: str,
    mode: str,
    policy_ref: str,
) -> DerivativeCapability:
    """Construct a validated derivative-capability declaration.

    Parameters
    ----------
    input_path : str
        Stable dotted path naming the differentiated input.
    mode : str
        Supported derivative-boundary mode.
    policy_ref : str
        Stable reference documenting the policy.

    Returns
    -------
    DerivativeCapability
        Immutable declaration with static metadata.
    """
    return DerivativeCapability(input_path, mode, policy_ref)


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
    """Construct a validated immutable fidelity manifest.

    Parameters
    ----------
    schema_version : str
        Version of the manifest schema.
    model_ref : str
        Registered forward-model identity.
    instrument_ref : str
        Registered instrument identity.
    acquisition_ref : str
        Registered acquisition-model identity.
    initial_state : str
        Electronic initial-state declaration.
    spectral_physics : str
        Spectral-physics declaration.
    photocurrent : str
        Photocurrent approximation declaration.
    light_interaction : str
        Light-interaction declaration.
    instrument : str
        Instrument declaration.
    derivative_capabilities : Tuple[DerivativeCapability, ...]
        Static derivative-boundary declarations.
    validation_refs : Tuple[str, ...]
        Validation-evidence references.
    validity_domain_refs : Tuple[str, ...]
        Validity-domain references.
    discrepancy_ref : Optional[str]
        Optional model-discrepancy declaration.

    Returns
    -------
    FidelityManifest
        Immutable manifest carrying no numerical leaves.
    """
    return FidelityManifest(
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


__all__: list[str] = [
    "DerivativeCapability",
    "FidelityManifest",
    "make_derivative_capability",
    "make_fidelity_manifest",
]
