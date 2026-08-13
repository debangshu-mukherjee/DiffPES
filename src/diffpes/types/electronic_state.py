"""Define solver-neutral electronic-state capabilities and a native source.

Extended Summary
----------------
Use this module for its validated public contracts and operations.

Routine Listings
----------------
:class:`ElectronicStateSource`
    Define the ``ElectronicStateSource`` public contract.
:class:`ElectronicStateArchive`
    Define the ``ElectronicStateArchive`` public contract.
:class:`EigensystemSource`
    Define the ``EigensystemSource`` public contract.
:class:`HamiltonianSource`
    Define the ``HamiltonianSource`` public contract.
:class:`HamiltonianOverlapSource`
    Define the ``HamiltonianOverlapSource`` public contract.
:class:`OverlapSource`
    Define the ``OverlapSource`` public contract.
:obj:`PlaneWaveStateSource`
    Expose the PlaneWaveStateSource public contract.
:class:`RetardedGreenFunctionSource`
    Define the ``RetardedGreenFunctionSource`` public contract.
:class:`TightBindingStateSource`
    Define the ``TightBindingStateSource`` public contract.
:class:`WavefunctionSource`
    Define the ``WavefunctionSource`` public contract.
:func:`make_electronic_state_archive`
    Compute the ``make_electronic_state_archive`` public contract.
:func:`make_tight_binding_state_source`
    Compute the ``make_tight_binding_state_source`` public contract.
"""

# Exact pydoclint attribute types cannot split across physical lines.
# ruff: noqa: E501

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Protocol, Tuple, runtime_checkable
from jaxtyping import Array, Bool, Complex128, Float64, jaxtyped

from .coordinates import MeasurementCoordinates
from .diagonalized_bands import DiagonalizedBands
from .plane_wave import PlaneWaveStateSource
from .slab_geometry import SurfaceCell
from .tb_model import TBModel


def _coordinate_k_points(
    coordinates: MeasurementCoordinates,
) -> Float64[Array, "n_k 3"]:
    """PRIVATE: Return the required fractional k-point coordinate.

    Parameters
    ----------
    coordinates : MeasurementCoordinates
        Coordinate carrier to query.

    Returns
    -------
    k_points : Float64[Array, "n_k 3"]
        Fractional momenta.

    Raises
    ------
    ValueError
        If the coordinate carrier omits fractional momenta.
    """
    if "k_points_frac" not in coordinates.coordinate_names:
        raise ValueError("electronic-state coordinates require k_points_frac")
    k_points: Float64[Array, "n_k 3"] = coordinates.coordinate_arrays[
        coordinates.coordinate_names.index("k_points_frac")
    ]
    return k_points


@runtime_checkable
class ElectronicStateSource(Protocol):
    """Define the ``ElectronicStateSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestElectronicstatesource`
    """

    capabilities: Tuple[str, ...]
    state_ref: str
    derivative_mode: str


@runtime_checkable
class WavefunctionSource(ElectronicStateSource, Protocol):
    """Define the ``WavefunctionSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestWavefunctionsource`
    """


class ElectronicStateArchive(eqx.Module):
    """Define the ``ElectronicStateArchive`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestElectronicstatearchive`

    Attributes
    ----------
    archive_ref : str
        Store the archive identity.
    geometry_ref : str
        Store the geometry identity.
    gauge_ref : str
        Store the gauge identity.
    spin_layout : str
        Store the spin layout.
    capabilities : Tuple[str, ...]
        Store supported capabilities.
    source_hashes : Tuple[str, ...]
        Store source hashes.

    See Also
    --------
    make_electronic_state_archive
        Construct a validated archive.
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
    """Define the ``HamiltonianSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestHamiltoniansource`
    """

    def hamiltonian(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return Fermi-referenced Hamiltonians at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class EigensystemSource(ElectronicStateSource, Protocol):
    """Define the ``EigensystemSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestEigensystemsource`
    """

    def eigensystem(
        self, coordinates: MeasurementCoordinates
    ) -> DiagonalizedBands:
        """Return the eigensystem at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class OverlapSource(ElectronicStateSource, Protocol):
    """Define the ``OverlapSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestOverlapsource`
    """

    def overlap(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return the basis overlap at requested coordinates."""
        ...  # noqa: PIE790


@runtime_checkable
class HamiltonianOverlapSource(HamiltonianSource, OverlapSource, Protocol):
    """Define the ``HamiltonianOverlapSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestHamiltonianoverlapsource`
    """


@runtime_checkable
class RetardedGreenFunctionSource(ElectronicStateSource, Protocol):
    """Define the ``RetardedGreenFunctionSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestRetardedgreenfunctionsource`
    """

    def retarded_green_function(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "n_k n_energy n_orb n_orb"]:
        """Return direct retarded Green functions at requested coordinates."""
        ...  # noqa: PIE790


class TightBindingStateSource(eqx.Module):
    """Define the ``TightBindingStateSource`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestTightbindingstatesource`

    Attributes
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]
        Store domain Hamiltonians.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Store domain bands.
    overlaps_by_domain : Optional[Tuple[Complex128[Array, "n_k n_orb n_orb"], ...]]
        Store domain overlaps.
    bulk_models_by_domain : Optional[Tuple[TBModel, ...]]
        Store bulk models.
    surface_cells_by_domain : Optional[Tuple[SurfaceCell, ...]]
        Store surface cells.
    capabilities : Tuple[str, ...]
        Store supported capabilities.
    state_ref : str
        Store the state identity.
    derivative_mode : str
        Store the derivative mode.

    See Also
    --------
    make_tight_binding_state_source
        Construct a validated tight-binding source.
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
        index: int
        hamiltonian: Complex128[Array, "n_k n_orb n_orb"]
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
        k_points: Float64[Array, "n_k 3"] = _coordinate_k_points(coordinates)
        values: Complex128[Array, "n_k n_orb n_orb"] = eqx.error_if(
            self.hamiltonians_by_domain[0],
            ~jnp.array_equal(k_points, self.bands_by_domain[0].kpoints),
            "Hamiltonian coordinates must match exact k points",
        )
        return values

    @jaxtyped(typechecker=beartype)
    def eigensystem(
        self, coordinates: MeasurementCoordinates
    ) -> DiagonalizedBands:
        """Return the sole-domain eigensystem for a factorized evaluation."""
        if len(self.bands_by_domain) != 1:
            raise ValueError(
                "select one domain before requesting an eigensystem"
            )
        k_points: Float64[Array, "n_k 3"] = _coordinate_k_points(coordinates)
        eigenvalues: Float64[Array, "n_k n_band"] = eqx.error_if(
            self.bands_by_domain[0].eigenvalues,
            ~jnp.array_equal(k_points, self.bands_by_domain[0].kpoints),
            "eigensystem coordinates must match exact k points",
        )
        bands: DiagonalizedBands = eqx.tree_at(
            lambda value: value.eigenvalues,
            self.bands_by_domain[0],
            eigenvalues,
        )
        return bands

    @jaxtyped(typechecker=beartype)
    def overlap(
        self,
        coordinates: MeasurementCoordinates,
    ) -> Complex128[Array, "n_k n_orb n_orb"]:
        """Return a declared overlap for a declared orthonormal state."""
        k_points: Float64[Array, "n_k 3"] = _coordinate_k_points(coordinates)
        coordinate_matches: Bool[Array, ""] = jnp.array_equal(
            k_points, self.bands_by_domain[0].kpoints
        )
        if self.overlaps_by_domain is not None:
            if len(self.overlaps_by_domain) != 1:
                raise ValueError(
                    "select one domain before requesting an overlap"
                )
            overlap: Complex128[Array, "n_k n_orb n_orb"] = eqx.error_if(
                self.overlaps_by_domain[0],
                ~coordinate_matches,
                "overlap coordinates must match exact k points",
            )
            return overlap
        if "orthonormal_basis" not in self.capabilities:
            raise ValueError(
                "nonorthogonal states require an overlap capability"
            )
        hamiltonian: Complex128[Array, "n_k n_orb n_orb"] = (
            self.hamiltonians_by_domain[0]
        )
        identity_overlap: Complex128[Array, "n_k n_orb n_orb"] = (
            jnp.broadcast_to(
                jnp.eye(hamiltonian.shape[-1], dtype=jnp.complex128),
                hamiltonian.shape,
            )
        )
        identity_overlap = eqx.error_if(
            identity_overlap,
            ~coordinate_matches,
            "overlap coordinates must match exact k points",
        )
        return identity_overlap  # noqa: RET504


@jaxtyped(typechecker=beartype)
def make_tight_binding_state_source(  # noqa: DOC105
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
    """Compute the ``make_tight_binding_state_source`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestMakeTightBindingStateSource`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    hamiltonians_by_domain : Tuple[Complex128[Array, 'n_k n_orb n_orb'], ...]
        Input value for this operation.
    bands_by_domain : Tuple[DiagonalizedBands, ...]
        Input value for this operation.
    overlaps_by_domain : object
        Input value for this operation.
    bulk_models_by_domain : Optional[Tuple[TBModel, ...]]
        Input value for this operation.
    surface_cells_by_domain : Optional[Tuple[SurfaceCell, ...]]
        Input value for this operation.
    state_ref : str
        Input value for this operation.

    Returns
    -------
    result : TightBindingStateSource
        Validated operation result.
    """
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
    result: TightBindingStateSource = TightBindingStateSource(
        hamiltonians_by_domain=hamiltonians,
        bands_by_domain=bands_by_domain,
        overlaps_by_domain=overlaps,
        bulk_models_by_domain=bulk_models_by_domain,
        surface_cells_by_domain=surface_cells_by_domain,
        capabilities=capabilities,
        state_ref=state_ref,
        derivative_mode="exact_ad",
    )
    return result


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
    """Compute the ``make_electronic_state_archive`` public contract.

    Validate documented inputs and preserve the declared scientific identity.

    :see: :class:`~.test_electronic_state.TestMakeElectronicStateArchive`

    Notes
    -----
    Validate inputs before returning the named result.

    Parameters
    ----------
    archive_ref : str
        Input value for this operation.
    geometry_ref : str
        Input value for this operation.
    gauge_ref : str
        Input value for this operation.
    spin_layout : str
        Input value for this operation.
    capabilities : Tuple[str, ...]
        Input value for this operation.
    source_hashes : Tuple[str, ...]
        Input value for this operation.

    Returns
    -------
    result : ElectronicStateArchive
        Validated operation result.
    """
    result: ElectronicStateArchive = ElectronicStateArchive(
        archive_ref,
        geometry_ref,
        gauge_ref,
        spin_layout,
        capabilities,
        source_hashes,
    )
    return result


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
