"""Define solver-neutral electronic-state capabilities and a native source."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Protocol, Tuple, runtime_checkable
from jaxtyping import Array, Complex128, jaxtyped

from .coordinates import MeasurementCoordinates
from .diagonalized_bands import DiagonalizedBands
from .plane_wave import PlaneWaveStateSource
from .slab_geometry import SurfaceCell
from .tb_model import TBModel


@runtime_checkable
class ElectronicStateSource(Protocol):
    """Protocol satisfied by a state provider with declared capabilities."""

    capabilities: Tuple[str, ...]
    state_ref: str
    derivative_mode: str


@runtime_checkable
class WavefunctionSource(ElectronicStateSource, Protocol):
    """Protocol for sources exposing a bounded wavefunction representation.

    See Also
    --------
    PlaneWaveStateSource
        Specializes this protocol for plane-wave coefficient batches.
    """


class ElectronicStateArchive(eqx.Module):
    """Store solver-neutral static electronic-state provenance and
    capabilities.

    Attributes
    ----------
    archive_ref : str
        Stable identity for the serialized archive.
    geometry_ref : str
        Stable geometry and unit-convention identity.
    gauge_ref : str
        Stable wavefunction-gauge identity.
    spin_layout : str
        Declared scalar, collinear, or spinor layout.
    capabilities : Tuple[str, ...]
        Capability identifiers represented by the archive.
    source_hashes : Tuple[str, ...]
        Immutable hashes of contributing external artifacts.

    See Also
    --------
    ElectronicStateSource
        Runtime capability protocol served by archive adapters.
    """

    archive_ref: str = eqx.field(static=True)
    geometry_ref: str = eqx.field(static=True)
    gauge_ref: str = eqx.field(static=True)
    spin_layout: str = eqx.field(static=True)
    capabilities: Tuple[str, ...] = eqx.field(static=True)
    source_hashes: Tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate immutable archive identity and capability declarations."""
        if not all((self.archive_ref, self.geometry_ref, self.gauge_ref)):
            raise ValueError(
                "electronic-state archive references must be nonempty"
            )
        if self.spin_layout not in ("scalar", "collinear", "spinor"):
            raise ValueError(
                "electronic-state archive spin layout is unsupported"
            )
        if not self.capabilities or any(
            not item for item in self.capabilities
        ):
            raise ValueError(
                "electronic-state archive capabilities must be nonempty"
            )
        if any(not digest for digest in self.source_hashes):
            raise ValueError(
                "electronic-state archive hashes must be nonempty"
            )


@runtime_checkable
class HamiltonianSource(ElectronicStateSource, Protocol):
    """Capability protocol for localized Hamiltonian providers."""

    def hamiltonian(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return Fermi-referenced Hamiltonians at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class EigensystemSource(ElectronicStateSource, Protocol):
    """Capability protocol for generalized eigensystem providers."""

    def eigensystem(
        self, coordinates: MeasurementCoordinates
    ) -> DiagonalizedBands:
        """Return the eigensystem at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class OverlapSource(ElectronicStateSource, Protocol):
    """Capability protocol for nonorthogonal localized bases."""

    def overlap(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return the basis overlap at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class HamiltonianOverlapSource(HamiltonianSource, OverlapSource, Protocol):
    """Protocol for localized sources exposing Hamiltonian and overlap data."""


@runtime_checkable
class RetardedGreenFunctionSource(ElectronicStateSource, Protocol):
    """Capability protocol for a state source exposing retarded Green data."""

    def retarded_green_function(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_energy n_orb n_orb"]:
        """Return direct retarded Green functions at requested coordinates."""
        ...  # noqa: PIE790


class TightBindingStateSource(eqx.Module):
    """Provide native tight-binding Hamiltonian and eigensystem
    capabilities.
    """

    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
    bands_by_domain: Tuple[DiagonalizedBands, ...]
    overlaps_by_domain: Optional[
        Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
    ]
    bulk_models_by_domain: Optional[Tuple[TBModel, ...]]
    surface_cells_by_domain: Optional[Tuple[SurfaceCell, ...]]
    capabilities: Tuple[str, ...] = eqx.field(static=True)
    state_ref: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate domain-local state arrays and static identity."""
        domains: int = len(self.hamiltonians_by_domain)
        if domains == 0 or len(self.bands_by_domain) != domains:
            raise ValueError(
                "tight-binding state domains must agree and be nonempty"
            )
        if (
            self.overlaps_by_domain is not None
            and len(self.overlaps_by_domain) != domains
        ):
            raise ValueError("overlap domains must match Hamiltonian domains")
        if not self.state_ref or self.derivative_mode != "exact_ad":
            raise ValueError(
                "native state source requires exact_ad and a state_ref"
            )
        for index, hamiltonian in enumerate(self.hamiltonians_by_domain):
            if (
                hamiltonian.ndim != 3  # noqa: PLR2004
                or hamiltonian.shape[-1] != hamiltonian.shape[-2]
            ):
                raise ValueError(
                    "Hamiltonians must have square orbital matrices"
                )
            if (
                hamiltonian.shape[0]
                != self.bands_by_domain[index].eigenvalues.shape[0]
            ):
                raise ValueError(
                    "Hamiltonian and eigensystem k axes must agree"
                )

    @jaxtyped(typechecker=beartype)
    def hamiltonian(
        self,
        coordinates: MeasurementCoordinates,
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return the sole-domain Hamiltonian for a factorized evaluation."""
        if len(self.hamiltonians_by_domain) != 1:
            raise ValueError(
                "select one domain before requesting a Hamiltonian"
            )
        del coordinates
        return self.hamiltonians_by_domain[0]

    @jaxtyped(typechecker=beartype)
    def eigensystem(
        self, coordinates: MeasurementCoordinates
    ) -> DiagonalizedBands:
        """Return the sole-domain eigensystem for a factorized evaluation."""
        if len(self.bands_by_domain) != 1:
            raise ValueError(
                "select one domain before requesting an eigensystem"
            )
        del coordinates
        return self.bands_by_domain[0]

    @jaxtyped(typechecker=beartype)
    def overlap(
        self,
        coordinates: MeasurementCoordinates,
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return a declared overlap for a declared orthonormal state."""
        del coordinates
        if self.overlaps_by_domain is not None:
            if len(self.overlaps_by_domain) != 1:
                raise ValueError(
                    "select one domain before requesting an overlap"
                )
            return self.overlaps_by_domain[0]
        if "orthonormal_basis" not in self.capabilities:
            raise ValueError(
                "nonorthogonal states require an overlap capability"
            )
        hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = (
            self.hamiltonians_by_domain[0]
        )
        return jnp.broadcast_to(
            jnp.eye(hamiltonian.shape[-1], dtype=jnp.complex128),
            hamiltonian.shape,
        )


@jaxtyped(typechecker=beartype)
def make_tight_binding_state_source(
    hamiltonians_by_domain: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...],
    bands_by_domain: Tuple[DiagonalizedBands, ...],
    *,
    overlaps_by_domain: Optional[
        Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
    ] = None,
    bulk_models_by_domain: Optional[Tuple[TBModel, ...]] = None,
    surface_cells_by_domain: Optional[Tuple[SurfaceCell, ...]] = None,
    state_ref: str = "org.diffpes.electronic_state.tb@0.1.0",
) -> TightBindingStateSource:
    """Create the native exact-AD tight-binding state provider."""
    hamiltonians: Tuple[Complex128[Array, "n_k n_orb n_orb"], ...] = tuple(
        jnp.asarray(value, dtype=jnp.complex128)
        for value in hamiltonians_by_domain
    )
    overlaps: Optional[Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]] = (
        None
        if overlaps_by_domain is None
        else tuple(
            jnp.asarray(value, dtype=jnp.complex128)
            for value in overlaps_by_domain
        )
    )
    capabilities: Tuple[str, ...] = (
        "hamiltonian",
        "eigensystem",
    ) + (("overlap",) if overlaps is not None else ("orthonormal_basis",))
    return TightBindingStateSource(
        hamiltonians_by_domain=hamiltonians,
        bands_by_domain=bands_by_domain,
        overlaps_by_domain=overlaps,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        capabilities=capabilities,
        state_ref=state_ref,
        derivative_mode="exact_ad",
    )


@jaxtyped(typechecker=beartype)
def make_electronic_state_archive(
    *,
    archive_ref: str,
    geometry_ref: str,
    gauge_ref: str,
    spin_layout: str,
    capabilities: Tuple[str, ...],
    source_hashes: Tuple[str, ...],
) -> ElectronicStateArchive:
    """Construct a solver-neutral electronic-state archive declaration.

    Parameters
    ----------
    archive_ref : str
        Stable serialized-archive identity.
    geometry_ref : str
        Stable geometry and unit-convention identity.
    gauge_ref : str
        Stable wavefunction-gauge identity.
    spin_layout : str
        Declared source spin layout.
    capabilities : Tuple[str, ...]
        Capability identifiers represented by the archive.
    source_hashes : Tuple[str, ...]
        Immutable hashes of contributing source artifacts.

    Returns
    -------
    ElectronicStateArchive
        Validated immutable archive metadata.
    """
    return ElectronicStateArchive(
        archive_ref,
        geometry_ref,
        gauge_ref,
        spin_layout,
        capabilities,
        source_hashes,
    )


__all__: list[str] = [
    "ElectronicStateSource",
    "ElectronicStateArchive",
    "EigensystemSource",
    "HamiltonianSource",
    "HamiltonianOverlapSource",
    "OverlapSource",
    "PlaneWaveStateSource",
    "RetardedGreenFunctionSource",
    "TightBindingStateSource",
    "WavefunctionSource",
    "make_electronic_state_archive",
    "make_tight_binding_state_source",
]
