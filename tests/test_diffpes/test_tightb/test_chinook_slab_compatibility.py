"""Compare tight-binding slabs with frozen offline Chinook artifacts.

The tests exercise slab numerical and structural contracts.
"""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray

from diffpes.tightb import (
    diagonalize_tb,
    gen_slab,
    layer_resolved_weights,
    surface_projector,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    SlabSpec,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
)

_ARTIFACT_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "chinook_slab_reference.json"
)
_ARTIFACT_SHA256: str = (
    "8be4ff280d627dac3bdfce7b56251cfd218a8f173621fa7eafd7ebb914e6eaec"
)
_COMPATIBILITY_RTOL: float = 1e-8
_COMPATIBILITY_ATOL_EV: float = 1e-8


def _reference() -> Dict[str, Any]:
    """PRIVATE: Load and authenticate the inert numeric compatibility artifact.

    Returns
    -------
    payload : Dict[str, Any]
        Parsed JSON content of ``chinook_slab_reference.json`` with the
        neutral model specification and the frozen Chinook slab
        reference in eV and Angstrom.

    Raises
    ------
    ValueError
        If the SHA-256 digest differs from the pinned constant, or if
        the metadata does not declare both K-type slab-parity
        requirements.

    Notes
    -----
    Reads the artifact bytes, checks them against ``_ARTIFACT_SHA256``,
    and validates the recorded requirements and classification before
    any comparison uses the payload. Chinook itself never runs.
    """
    encoded: bytes = _ARTIFACT_PATH.read_bytes()
    digest: str = hashlib.sha256(encoded).hexdigest()
    if digest != _ARTIFACT_SHA256:
        message: str = (
            "Chinook slab artifact checksum differs from its pinned digest"
        )
        raise ValueError(message)
    payload: Dict[str, Any] = json.loads(encoded)
    metadata: Dict[str, Any] = payload["metadata"]
    if (
        metadata["requirements"]
        != ["chinook-slab-band-parity", "chinook-surface-state-parity"]
        or metadata["classification"] != "K-type behavioral compatibility"
    ):
        message = "Chinook slab artifact metadata is invalid"
        raise ValueError(message)
    return payload


def _bulk_model(specification: Dict[str, Any]) -> TBModel:
    """PRIVATE: Build the native side of the implementation-neutral model.

    Parameters
    ----------
    specification : Dict[str, Any]
        Frozen ``model_specification`` block with the lattice in
        Angstrom, fractional positions, species, onsite energy in eV,
        and anisotropic nearest-neighbor hoppings in eV.

    Returns
    -------
    model : TBModel
        One-orbital cubic bulk model with six nearest-neighbor hopping
        records that mirror the specification exactly.

    Notes
    -----
    Duplicates each of the x, y, and z hopping values onto the forward
    and reverse cells. The model is therefore Hermitian by
    construction. It matches the model that Chinook builds from the
    same neutral numbers.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(
            specification["lattice_angstrom"],
            dtype=jnp.float64,
        ),
        positions=jnp.asarray(
            specification["positions_fractional"],
            dtype=jnp.float64,
        ),
        species=tuple(specification["species"]),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=tuple(specification["basis"]),
    )
    hopping: Dict[str, float] = specification["nearest_neighbor_hopping_ev"]
    amplitudes: Complex128[Array, " 6"] = jnp.asarray(
        (
            hopping["x"],
            hopping["x"],
            hopping["y"],
            hopping["y"],
            hopping["z"],
            hopping["z"],
        ),
        dtype=jnp.complex128,
    )
    cells: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=amplitudes,
        onsite_energies=jnp.asarray(
            [specification["onsite_ev"]],
            dtype=jnp.float64,
        ),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0),) * len(cells),
        hopping_cells=cells,
        shell_index=(-1,),
    )
    return model


def _native_slab(
    payload: Dict[str, Any],
) -> Tuple[TBModel, SlabSpec]:
    """PRIVATE: Construct the native slab from the frozen specification.

    Parameters
    ----------
    payload : Dict[str, Any]
        Authenticated artifact dictionary from :func:`_reference`.

    Returns
    -------
    slab_and_spec : Tuple[TBModel, SlabSpec]
        The extruded slab model and its specification carrier.

    Notes
    -----
    Feeds the frozen Miller index, thickness and vacuum in Angstrom,
    termination species, and fine offsets into the public
    :func:`gen_slab` on top of :func:`_bulk_model`, so the native slab
    derives only from implementation-neutral numbers.
    """
    specification: Dict[str, Any] = payload["model_specification"]
    slab_and_spec: Tuple[TBModel, SlabSpec] = gen_slab(
        bulk_model=_bulk_model(specification),
        miller=tuple(specification["miller"]),
        thickness_ang=specification["thickness_angstrom"],
        vacuum_ang=specification["vacuum_angstrom"],
        termination=tuple(specification["termination_species_top_bottom"]),
        fine=tuple(specification["fine_top_bottom_angstrom"]),
    )
    return slab_and_spec


def _gauss_reduced_metric(
    vectors: Float64[NDArray, "2 3"],
) -> Float64[NDArray, "2 2"]:
    """PRIVATE: Return a deterministic reduced two-dimensional lattice metric.

    Parameters
    ----------
    vectors : Float64[NDArray, "2 3"]
        Two in-plane surface lattice vectors in Angstrom as rows.

    Returns
    -------
    metric : Float64[NDArray, "2 2"]
        The Gram matrix ``reduced @ reduced.T`` of the Gauss-reduced
        vectors in Angstrom squared.

    Raises
    ------
    RuntimeError
        If the reduction loop does not converge within 32 sweeps.

    Notes
    -----
    Applies classical Gauss lattice reduction. The loop orders the two
    vectors by squared length. It then subtracts the nearest-integer
    multiple of the shorter vector from the longer vector. The loop
    stops when the pair reaches reduced form. The resulting metric is a
    unimodular invariant, so it permits a direct comparison between two
    differently chosen surface cells.
    """
    reduced: Float64[NDArray, "2 3"] = np.asarray(
        vectors, dtype=np.float64
    ).copy()
    for _ in range(32):
        first_norm: float = float(reduced[0] @ reduced[0])
        second_norm: float = float(reduced[1] @ reduced[1])
        if second_norm < first_norm:
            reduced[[0, 1]] = reduced[[1, 0]]
            reduced[1] *= -1.0
            continue
        nearest: int = int(
            np.rint(float(reduced[0] @ reduced[1]) / first_norm)
        )
        if nearest == 0:
            break
        reduced[1] -= nearest * reduced[0]
    else:
        raise RuntimeError("reference surface metric did not reduce")
    metric: Float64[NDArray, "2 2"] = reduced @ reduced.T
    return metric


class TestChinookSlabCompatibility:
    """Resolve Chinook slab band and surface-state compatibility.

    The cases compare surface cells, slab spectra, and nondegenerate surface
    projections with pinned external data.
    """

    def test_surface_cell_is_unimodularly_equivalent(self) -> None:
        """Match Chinook surface area and the reduced in-plane metric.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        payload: Dict[str, Any] = _reference()
        reference: Dict[str, Any] = payload["chinook_reference"]
        slab, _ = _native_slab(payload)
        native_vectors: Float64[NDArray, "2 3"] = np.asarray(
            slab.geometry.lattice[:2],
            dtype=np.float64,
        )
        chinook_vectors: Float64[NDArray, "2 3"] = np.asarray(
            reference["realization"]["slab_lattice_angstrom"],
            dtype=np.float64,
        )[:2]
        native_area: float = float(
            np.linalg.norm(np.cross(native_vectors[0], native_vectors[1]))
        )
        chinook_area: float = float(
            np.linalg.norm(np.cross(chinook_vectors[0], chinook_vectors[1]))
        )

        np.testing.assert_allclose(
            native_area,
            chinook_area,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _gauss_reduced_metric(native_vectors),
            _gauss_reduced_metric(chinook_vectors),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_slab_spectrum_agrees_after_independent_checks(self) -> None:
        """Match the frozen nondegenerate Chinook slab spectrum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: Any
        payload: Dict[str, Any] = _reference()
        specification: Dict[str, Any] = payload["model_specification"]
        reference: Dict[str, Any] = payload["chinook_reference"]
        slab: TBModel
        slab_spec: SlabSpec
        slab, slab_spec = _native_slab(payload)

        assert len(slab.basis.n) == reference["realization"]["n_orbitals"]
        assert slab_spec.n_layers == reference["realization"]["n_orbitals"]
        np.testing.assert_allclose(
            slab.depths,
            np.asarray(reference["realization"]["depths_angstrom"]),
            rtol=0.0,
            atol=1e-12,
        )
        bands = diagonalize_tb(
            slab,
            jnp.asarray(
                specification["kpoints_fractional_bulk"],
                dtype=jnp.float64,
            ),
        )
        np.testing.assert_allclose(
            bands.eigenvalues,
            np.asarray(reference["eigenvalues_ev"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=_COMPATIBILITY_ATOL_EV,
        )

    def test_surface_projection_agrees_off_degeneracy(self) -> None:
        """Match Chinook's depth law and per-band surface expectations.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        bands: Any
        slab: Any
        payload: Dict[str, Any] = _reference()
        specification: Dict[str, Any] = payload["model_specification"]
        reference: Dict[str, Any] = payload["chinook_reference"]
        slab, _ = _native_slab(payload)
        escape_length: float = specification[
            "intensity_escape_length_angstrom"
        ]
        bands = diagonalize_tb(
            slab,
            jnp.asarray(
                specification["kpoints_fractional_bulk"],
                dtype=jnp.float64,
            ),
        )

        np.testing.assert_allclose(
            surface_projector(slab.depths, escape_length),
            np.asarray(reference["surface_projector_diagonal"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            layer_resolved_weights(bands, escape_length),
            np.asarray(reference["surface_weight_expectations"]),
            rtol=_COMPATIBILITY_RTOL,
            atol=1e-12,
        )
