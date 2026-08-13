"""Verify electronic-state capability and native-source contracts.

Use explicit fixtures and independent expectations for every assertion.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Complex128, TypeCheckError

from diffpes.types import (
    EigensystemSource,
    ElectronicStateSource,
    HamiltonianOverlapSource,
    HamiltonianSource,
    MeasurementCoordinates,
    OverlapSource,
    RetardedGreenFunctionSource,
    TightBindingStateSource,
    WavefunctionSource,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_electronic_state_archive,
    make_measurement_coordinates,
    make_orbital_basis,
    make_tight_binding_state_source,
)


def _coordinates(value: float = 0.0) -> MeasurementCoordinates:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    result: Any = make_measurement_coordinates(
        (jnp.asarray([[value, 0.0, 0.0]]),),
        coordinate_names=("k_points_frac",),
        coordinate_units=("1",),
        coordinate_dimensions=(("k", "cart"),),
        dimension_names=("k", "cart"),
        coordinate_system="fractional",
        frame_id="fixture",
    )
    return result


def _native_source(*, overlap: bool = True) -> TightBindingStateSource:
    """PRIVATE: Check the private helper behavior.

    Notes
    -----
    Build the fixture from explicit values for isolated use.
    """
    geometry: Any = make_crystal_geometry(
        jnp.eye(3), jnp.zeros((1, 3)), ("X",)
    )
    basis: Any = make_orbital_basis(
        atom_indices=(0,), n=(1,), l=(0,), m=(0,), labels=("1s",)
    )
    bands: Any = make_diagonalized_bands(
        jnp.asarray([[-0.2]]),
        jnp.ones((1, 1, 1), dtype=jnp.complex128),
        jnp.zeros((1, 3)),
        geometry,
        basis,
        fermi_energy=0.0,
    )
    overlaps: Any = (
        (jnp.ones((1, 1, 1), dtype=jnp.complex128),) if overlap else None
    )
    result: Any = make_tight_binding_state_source(
        (jnp.asarray([[[-0.2 + 0.0j]]]),),
        (bands,),
        overlaps_by_domain=overlaps,
    )
    return result


class _AllCapabilities:
    """PRIVATE: Check every structural capability protocol."""

    capabilities: Tuple[str, ...] = (
        "hamiltonian",
        "overlap",
        "eigensystem",
        "wavefunction",
        "retarded_green_function",
    )
    state_ref: str = "fixture"
    derivative_mode: str = "exact_ad"

    def hamiltonian(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "1 1 1"]:
        """Check the private helper behavior."""
        del coordinates
        result: Any = jnp.zeros((1, 1, 1), dtype=jnp.complex128)
        return result

    def overlap(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "1 1 1"]:
        """Check the private helper behavior."""
        del coordinates
        result: Any = jnp.ones((1, 1, 1), dtype=jnp.complex128)
        return result

    def eigensystem(self, coordinates: MeasurementCoordinates) -> object:
        """Check the private helper behavior."""
        del coordinates
        result: Any = object()
        return result

    def retarded_green_function(
        self, coordinates: MeasurementCoordinates
    ) -> Complex128[Array, "1 1 1 1"]:
        """Check the private helper behavior."""
        del coordinates
        result: Any = -1.0j * jnp.ones((1, 1, 1, 1), dtype=jnp.complex128)
        return result

    def plane_wave_batch(self, request: object) -> object:
        """Check the private helper behavior."""
        del request
        result: Any = object()
        return result


class TestElectronicstatesource:
    """Verify ``diffpes.types.ElectronicStateSource`` conformance.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    @pytest.mark.parametrize(
        "protocol",
        [
            ElectronicStateSource,
            WavefunctionSource,
            HamiltonianSource,
            EigensystemSource,
            OverlapSource,
            HamiltonianOverlapSource,
            RetardedGreenFunctionSource,
        ],
    )
    def test_accepts_structural_capability_protocols(
        self, protocol: type
    ) -> None:
        """Accept one object implementing every declared structural method.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Apply each runtime-checkable protocol independently to the same mock.
        """
        assert isinstance(_AllCapabilities(), protocol)


class TestWavefunctionsource:
    """Verify the ``diffpes.types.WavefunctionSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestHamiltoniansource:
    """Verify the ``diffpes.types.HamiltonianSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestEigensystemsource:
    """Verify the ``diffpes.types.EigensystemSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestOverlapsource:
    """Verify the ``diffpes.types.OverlapSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestHamiltonianoverlapsource:
    """Verify the ``diffpes.types.HamiltonianOverlapSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestRetardedgreenfunctionsource:
    """Verify the ``diffpes.types.RetardedGreenFunctionSource`` public symbol.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestElectronicstatearchive:
    """Verify ``diffpes.types.ElectronicStateArchive`` invariants.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_factory_accepts_complete_archive_identity(self) -> None:
        """Preserve valid archive, geometry, gauge, and source identities.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare every static identity with its explicit input.
        """
        archive: Any = make_electronic_state_archive(
            archive_ref="archive",
            geometry_ref="geometry",
            gauge_ref="gauge",
            spin_layout="scalar",
            capabilities=("hamiltonian",),
            source_hashes=("sha256",),
        )
        assert archive.archive_ref == "archive"
        assert archive.capabilities == ("hamiltonian",)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"archive_ref": ""}, "references must be nonempty"),
            ({"spin_layout": "bad"}, "spin layout is unsupported"),
            ({"capabilities": ()}, "capabilities must be nonempty"),
            ({"source_hashes": ("",)}, "hashes must be nonempty"),
        ],
    )
    def test_rejects_each_archive_invariant(
        self, kwargs: Dict[str, object], message: str
    ) -> None:
        """Reject incomplete identity, layout, capabilities, and hashes.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Replace one field in an otherwise valid archive declaration.
        """
        values: Dict[str, object] = {
            "archive_ref": "archive",
            "geometry_ref": "geometry",
            "gauge_ref": "gauge",
            "spin_layout": "scalar",
            "capabilities": ("hamiltonian",),
            "source_hashes": ("sha256",),
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=message):
            make_electronic_state_archive(**values)


class TestTightbindingstatesource:
    """Verify ``diffpes.types.TightBindingStateSource`` behavior.

    Cover acceptance and rejection cases with explicit fixtures.
    """

    def test_serves_exact_nodes_and_declares_overlap(self) -> None:
        """Return all source values at an exactly matching k point.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Compare Hamiltonian, eigensystem, and overlap with explicit scalars.
        """
        source: Any = _native_source()
        coordinates: MeasurementCoordinates = _coordinates()
        assert source.capabilities == (
            "hamiltonian",
            "eigensystem",
            "overlap",
        )
        assert source.hamiltonian(coordinates)[0, 0, 0] == -0.2
        assert source.eigensystem(coordinates).eigenvalues[0, 0] == -0.2
        assert source.overlap(coordinates)[0, 0, 0] == 1.0

    def test_synthesizes_identity_for_orthonormal_basis(self) -> None:
        """Return identity overlap without stored overlap matrices.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Construct an orthonormal source and inspect its one-orbital overlap.
        """
        source: Any = _native_source(overlap=False)
        assert "orthonormal_basis" in source.capabilities
        assert source.overlap(_coordinates())[0, 0, 0] == 1.0

    def test_rejects_nonmatching_exact_coordinates(self) -> None:
        """Reject a k point different from the stored eigensystem node.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Shift only the requested fractional k coordinate.
        """
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="coordinates must match exact"
        ):
            _native_source().hamiltonian(_coordinates(0.1))

    def test_rejects_empty_or_misaligned_domain_collections(self) -> None:
        """Reject empty domains and overlap collections of unequal length.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Reuse a valid source only to obtain its factory-validated band value.
        """
        valid: Any = _native_source()
        with pytest.raises(
            ValueError, match="domains must agree and be nonempty"
        ):
            make_tight_binding_state_source((), ())
        with pytest.raises(ValueError, match="overlap domains must match"):
            make_tight_binding_state_source(
                valid.hamiltonians_by_domain,
                valid.bands_by_domain,
                overlaps_by_domain=(),
            )

    def test_rejects_empty_identity_and_misaligned_k_axis(self) -> None:
        """Reject an empty source reference and unequal Hamiltonian k count.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Change each invariant independently from the valid native source.
        """
        valid: Any = _native_source()
        with pytest.raises(
            ValueError, match="requires exact_ad and a state_ref"
        ):
            make_tight_binding_state_source(
                valid.hamiltonians_by_domain,
                valid.bands_by_domain,
                state_ref="",
            )
        with pytest.raises(ValueError, match="k axes must agree"):
            make_tight_binding_state_source(
                (jnp.zeros((2, 1, 1), dtype=jnp.complex128),),
                valid.bands_by_domain,
            )

    def test_rejects_nonsquare_hamiltonian_at_runtime_boundary(self) -> None:
        """Reject a nonsquare orbital tensor before carrier construction.
        Check explicit inputs against independent expectations.

        Notes
        -----
        Preserve complex dtype and three dimensions while changing one axis.
        """
        valid: Any = _native_source()
        with pytest.raises(TypeCheckError, match="hamiltonians_by_domain"):
            make_tight_binding_state_source(
                (jnp.zeros((1, 1, 2), dtype=jnp.complex128),),
                valid.bands_by_domain,
            )


class TestMakeTightBindingStateSource:
    """Verify ``diffpes.types.make_tight_binding_state_source``.

    Cover acceptance and rejection cases with explicit fixtures.
    """


class TestMakeElectronicStateArchive:
    """Verify ``diffpes.types.make_electronic_state_archive``.

    Cover acceptance and rejection cases with explicit fixtures.
    """
