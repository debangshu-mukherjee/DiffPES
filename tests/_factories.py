"""Build deterministic toy carriers for tests.

Extended Summary
----------------
Provides small, fixed-policy inputs for forward, tight-binding, and radial
tests. Random factories are deterministic for a supplied JAX key; analytic
factories use fixed grids and physical parameters. Each factory checks every
returned traced leaf for finiteness.
"""

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, List, Tuple
from jaxtyping import Array, Complex128, Float64, PRNGKeyArray, jaxtyped

from diffpes.tightb import (
    diagonalize_tb,
)
from diffpes.types import (
    BandStructure,
    CrystalGeometry,
    DiagonalizedBands,
    ForwardCertificate,
    ForwardModelSpec,
    OrbitalBasis,
    OrbitalProjection,
    ScalarFloat,
    TBModel,
    make_artifact_ref,
    make_band_structure,
    make_certification_claim,
    make_convention_ref,
    make_crystal_geometry,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_human_attestation_ref,
    make_information_spectrum,
    make_orbital_basis,
    make_orbital_projection,
    make_policy_report,
    make_sensitivity_map,
    make_tb_model,
    make_transformation_record,
)


def _assert_finite(tree: object) -> None:
    """PRIVATE: Require every numerical leaf in a toy carrier to be finite.

    Parameters
    ----------
    tree : object
        Toy carrier or any other PyTree of numerical leaves.

    Notes
    -----
    ``jax.tree.leaves`` collects the leaves first, so every factory
    output passes through one uniform finiteness check. The Chex assertion
    raises ``AssertionError`` when a leaf contains a non-finite value.
    """
    leaves: Tuple[object, ...] = tuple(jax.tree.leaves(tree))
    chex.assert_tree_all_finite(leaves)


@jaxtyped(typechecker=beartype)
def make_minimal_crystal_geometry(
    n_atoms: int = 1,
) -> CrystalGeometry:
    """Build a right-handed cubic geometry for carrier tests.

    The fixture uses an identity lattice and places each species-``X`` atom
    at the fractional origin. It supplies the common geometry for tests that
    isolate electronic-structure carrier behavior.

    Parameters
    ----------
    n_atoms : int, optional
        Number of coincident atoms. Default 1.

    Returns
    -------
    geometry : CrystalGeometry
        Validated identity-cell geometry with ``n_atoms`` species entries.

    Notes
    -----
    The positive unit-cell determinant satisfies the handedness contract.
    """
    positions: Float64[Array, "n_atoms 3"] = jnp.zeros(
        (n_atoms, 3), dtype=jnp.float64
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=positions,
        species=tuple("X" for _ in range(n_atoms)),
    )
    return geometry


@jaxtyped(typechecker=beartype)
def make_minimal_orbital_basis(
    atom_indices: Tuple[int, ...] = (0,),
    spin: Tuple[int, ...] = (),
) -> OrbitalBasis:
    """Build one valid s orbital for each supplied atom index.

    The fixture gives every orbital the same hydrogenic quantum numbers and
    assigns deterministic labels in input order.

    Parameters
    ----------
    atom_indices : Tuple[int, ...], optional
        Atom index for each orbital. Default ``(0,)``.
    spin : Tuple[int, ...], optional
        Spin channel for each orbital, or an empty tuple for spinless data.
        Default ``()``.

    Returns
    -------
    basis : OrbitalBasis
        Validated static orbital metadata with one s orbital per index.

    Notes
    -----
    The helper labels orbitals ``s0``, ``s1``, and so on. The factory validates
    all tuple lengths and spin values.
    """
    n_orbitals: int = len(atom_indices)
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=atom_indices,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        spin=spin,
        labels=tuple(f"s{index}" for index in range(n_orbitals)),
    )
    return basis


@jaxtyped(typechecker=beartype)
def make_1d_chain_model(t: ScalarFloat = -1.0) -> TBModel:
    r"""Build the closed nearest-neighbor one-dimensional chain fixture.

    The single-orbital model is an external-truth fixture for
    :math:`E(k)=2t\cos(2\pi k)`. It uses exact integer cells and explicit
    reverse hoppings under the basis-position gauge.

    Parameters
    ----------
    t : ScalarFloat, optional
        Nearest-neighbor hopping in eV. Default is ``-1.0`` eV.

    Returns
    -------
    model : TBModel
        Validated one-orbital chain model.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(t, dtype=jnp.complex128)
    hopping: Complex128[Array, " 2"] = jnp.stack(
        (hopping_value, hopping_value)
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping,
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((1, 0, 0), (-1, 0, 0)),
        shell_index=(-1,),
    )
    return model


@jaxtyped(typechecker=beartype)
def make_graphene_model(t: ScalarFloat = -2.7) -> TBModel:
    """Build the closed nearest-neighbor graphene fixture.

    Parameters
    ----------
    t : ScalarFloat, optional
        Carbon pz nearest-neighbor hopping in eV. Default is ``-2.7`` eV.

    Returns
    -------
    model : TBModel
        Validated two-orbital honeycomb model in the basis-position gauge.
    """
    lattice_constant: float = 2.46
    lattice: Float64[Array, "3 3"] = jnp.asarray(
        [
            [lattice_constant, 0.0, 0.0],
            [
                lattice_constant / 2.0,
                lattice_constant * jnp.sqrt(3.0) / 2.0,
                0.0,
            ],
            [0.0, 0.0, 10.0],
        ],
        dtype=jnp.float64,
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
        positions=jnp.asarray(
            [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]],
            dtype=jnp.float64,
        ),
        species=("C", "C"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(2, 2),
        l=(1, 1),
        m=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(t, dtype=jnp.complex128)
    hopping: Complex128[Array, " 6"] = jnp.stack((hopping_value,) * 6)
    model: TBModel = make_tb_model(
        hopping_amplitudes=hopping,
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
        hopping_cells=(
            (0, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        ),
        shell_index=(-1, -1),
    )
    return model


@jaxtyped(typechecker=beartype)
def make_rashba_model(
    hopping: ScalarFloat = -0.63,
    rashba: ScalarFloat = 0.27,
) -> TBModel:
    """Build a closed square-lattice Rashba spinor fixture.

    Parameters
    ----------
    hopping : ScalarFloat, optional
        Spin-independent nearest-neighbor hopping in eV.
    rashba : ScalarFloat, optional
        Rashba coupling in eV.

    Returns
    -------
    model : TBModel
        Validated two-state spinor model in down--up block order.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.diag(jnp.asarray([3.2, 3.2, 12.0], dtype=jnp.float64)),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        spin=(-1, 1),
        labels=("s_down", "s_up"),
    )
    hopping_value: Complex128[Array, ""] = jnp.asarray(
        hopping,
        dtype=jnp.complex128,
    )
    rashba_value: Complex128[Array, ""] = jnp.asarray(
        rashba,
        dtype=jnp.complex128,
    )
    amplitudes: List[Complex128[Array, ""]] = []
    pairs: List[Tuple[int, int]] = []
    cells: List[Tuple[int, int, int]] = []
    cell: Tuple[int, int, int]
    spin: int
    nearest_cells: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    )
    for spin in (0, 1):
        for cell in nearest_cells:
            amplitudes.append(hopping_value)
            pairs.append((spin, spin))
            cells.append(cell)
    forward_amplitudes: Tuple[Complex128[Array, ""], ...] = (
        -0.5 * rashba_value,
        0.5 * rashba_value,
        -0.5j * rashba_value,
        0.5j * rashba_value,
    )
    amplitude: Complex128[Array, ""]
    for cell, amplitude in zip(
        nearest_cells,
        forward_amplitudes,
        strict=True,
    ):
        amplitudes.append(amplitude)
        pairs.append((0, 1))
        cells.append(cell)
        amplitudes.append(jnp.conj(amplitude))
        pairs.append((1, 0))
        cells.append(tuple(-component for component in cell))
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.stack(amplitudes),
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=tuple(pairs),
        hopping_cells=tuple(cells),
        shell_index=(-1, -1),
        spinor=True,
    )
    _assert_finite(model)
    return model


@jaxtyped(typechecker=beartype)
def make_t2g_soc_model(coupling: ScalarFloat = 0.4) -> TBModel:
    """Build an isolated projected-t2g spin--orbit fixture.

    Parameters
    ----------
    coupling : ScalarFloat, optional
        Atomic spin--orbit coupling in eV.

    Returns
    -------
    model : TBModel
        Validated six-state t2g model in down--up block order.
    """
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=4.0 * jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("Ti",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * 6,
        n=(3,) * 6,
        l=(2,) * 6,
        m=(-2, -1, 1, -2, -1, 1),
        spin=(-1, -1, -1, 1, 1, 1),
        labels=(
            "dxy_down",
            "dyz_down",
            "dxz_down",
            "dxy_up",
            "dyz_up",
            "dxz_up",
        ),
    )
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((6,), dtype=jnp.float64),
        soc_lambdas=jnp.asarray([coupling], dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(0,) * 6,
        spinor=True,
    )
    _assert_finite(model)
    return model


@jaxtyped(typechecker=beartype)
def toy_band_structure(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
) -> BandStructure:
    """Build a reproducible occupied-state toy band structure.

    The factory samples eigenvalues in [-2.5, 0.25] eV, safely below
    ``E_F + 0.5`` eV. This intentionally avoids the known upper-state
    ``fermi_dirac`` gradient defect in upper states. The supplied key
    is the entire seed policy and is never mutated.
    """
    energy_key: PRNGKeyArray
    kpoint_key: PRNGKeyArray
    energy_key, kpoint_key = jax.random.split(key)
    eigenvalues: Float64[Array, "n_k n_bands"] = jax.random.uniform(
        energy_key,
        (n_k, n_bands),
        minval=-2.5,
        maxval=0.25,
        dtype=jnp.float64,
    )
    eigenvalues = jnp.sort(eigenvalues, axis=-1)
    kpoints: Float64[Array, "n_k 3"] = jax.random.uniform(
        kpoint_key,
        (n_k, 3),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float64,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        kpoint_weights=jnp.full(n_k, 1.0 / n_k, dtype=jnp.float64),
        fermi_energy=0.0,
    )
    _assert_finite(bands)
    return bands


@jaxtyped(typechecker=beartype)
def toy_orbital_projection(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
    n_atoms: int = 2,
) -> OrbitalProjection:
    """Build reproducible normalized orbital weights.

    Positive weights are drawn from a uniform distribution using only the
    supplied key, then normalized over atom and orbital axes for each state.
    Spin and orbital-angular-momentum fields remain absent.
    """
    raw: Float64[Array, "n_k n_bands n_atoms 9"] = jax.random.uniform(
        key,
        (n_k, n_bands, n_atoms, 9),
        minval=0.1,
        maxval=1.0,
        dtype=jnp.float64,
    )
    normalization: Float64[Array, "n_k n_bands 1 1"] = jnp.sum(
        raw, axis=(-2, -1), keepdims=True
    )
    projections: Float64[Array, "n_k n_bands n_atoms 9"] = raw / normalization
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections
    )
    _assert_finite(orbital_projection)
    return orbital_projection


@jaxtyped(typechecker=beartype)
def toy_graphene_diagonalized(
    n_k: int = 12,
) -> Tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native graphene model on a fixed Gamma-to-K path.

    Uses the production -2.7 eV nearest-neighbor model and an
    endpoint-inclusive fractional path from Gamma to K = (1/3, 1/3, 0).
    The factory uses no random seed.
    """
    model: TBModel = make_graphene_model()
    path_coordinate: Float64[Array, " n_k"] = jnp.linspace(
        0.0, 1.0, n_k, dtype=jnp.float64
    )
    kpoints: Float64[Array, "n_k 3"] = path_coordinate[:, None] * jnp.array(
        [1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=jnp.float64
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: Tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result


def registry_model_spec(name: str) -> ForwardModelSpec:
    """PRIVATE: Build one registry-test forward-model spec from a name.

    Parameters
    ----------
    name : str
        Short name that sets the model identity and the implementation
        reference.

    Returns
    -------
    spec : ForwardModelSpec
        Forward-model spec at version 1.0.0 for the ARPES intensity
        observable with one differentiable scale path.

    Notes
    -----
    Embeds the name in the identity
    ``org.diffpes.model.registry_test.<name>`` and in the implementation
    reference ``tests.registry:<name>``.
    """
    spec: ForwardModelSpec = make_forward_model_spec(
        model_id=f"org.diffpes.model.registry_test.{name}",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref=f"tests.registry:{name}",
        differentiable_paths=("parameters.scale",),
    )
    return spec


def sample_forward_certificate(
    *,
    execution_id: str = "run-001",
    started_at_utc: str = "2026-07-21T12:00:00Z",
    model_version: str = "1.0.0",
    environment_checksum: str = (
        "sha256:1:environment:"
        "89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef89abcdef"
    ),
    extensions_json: str = '{"project":"demo","unicode":"Å"}',
) -> ForwardCertificate:
    """Build one small, fully populated forward certificate.

    Parameters
    ----------
    execution_id : str, optional
        Stable execution identifier. Default ``"run-001"``.
    started_at_utc : str, optional
        Absolute UTC start time. Default ``"2026-07-21T12:00:00Z"``.
    model_version : str, optional
        Semantic version of the test model. Default ``"1.0.0"``.
    environment_checksum : str, optional
        Canonical environment checksum used by the execution manifest.
    extensions_json : str, optional
        Canonical JSON object retained in the certificate extensions.

    Returns
    -------
    certificate : ForwardCertificate
        Validated certificate with representative evidence, derivative,
        dependency, sensitivity, information, and policy records.

    Notes
    -----
    The values are deterministic and intentionally cover every persisted
    certification carrier used by the JSON and HDF5 storage tests.
    """
    convention: Any
    predicate: Any
    model: Any
    manifest: Any
    artifact: Any
    transformation: Any
    attestation: Any
    evidence: Any
    claim: Any
    domain: Any
    derivatives: Any
    dependencies: Any
    sensitivities: Any
    information: Any
    policy: Any
    convention = make_convention_ref(
        "org.diffpes.convention.energy.fermi_referenced_ev",
        "1.0.0",
    )
    predicate = make_domain_predicate(
        "org.diffpes.domain.photon_energy.positive",
        "photon_energy_ev > 0",
        units="eV",
    )
    model = make_forward_model_spec(
        model_id="org.diffpes.model.arpes.test",
        model_version=model_version,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="tests.forward",
        assumptions=("dipole_approximation",),
        conventions=(convention,),
        domain=(predicate,),
        differentiable_paths=("params.sigma", "params.temperature"),
    )
    manifest = make_execution_manifest(
        execution_id=execution_id,
        model_ref=f"{model.model_id}@{model.model_version}",
        schema_version="1.0.0",
        package_version="2026.06.02",
        source_checksum=(
            "sha256:1:source:"
            "0123456701234567012345670123456701234567012345670123456701234567"
        ),
        environment_checksum=environment_checksum,
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc=started_at_utc,
    )
    artifact = make_artifact_ref(
        artifact_id="bands",
        media_type="application/x-vasp-eigenval",
        byte_checksum=(
            "sha256:1:artifact-bytes:"
            "1020304010203040102030401020304010203040102030401020304010203040"
        ),
        content_checksum=(
            "sha256:1:normalized-content:"
            "2030405020304050203040502030405020304050203040502030405020304050"
        ),
        semantic_checksum=(
            "sha256:1:semantic:"
            "3040506030405060304050603040506030405060304050603040506030405060"
        ),
        locator="/private/data/EIGENVAL",
        role="initial_state",
    )
    transformation = make_transformation_record(
        transformation_id="org.diffpes.transform.amplitude.intensity",
        transformation_version="1.0.0",
        parent_ids=("amplitude",),
        output_ids=("intensity",),
        preserves=("energy_reference",),
        destroys=("overall_phase",),
        invalidates_claims=("claim.phase",),
        parameters_checksum=(
            "sha256:1:parameters:"
            "4050607040506070405060704050607040506070405060704050607040506070"
        ),
    )
    attestation = make_human_attestation_ref(
        attestation_id="attestation.reference-review",
        reviewer_ref="reviewer.example",
        scope_ids=("reference-spectrum",),
        statement="Reviewed the named evidence lineage.",
        recorded_at_utc="2026-07-24T12:00:00Z",
    )
    evidence = make_evidence_ref(
        evidence_id="reference-spectrum",
        method_id="org.diffpes.method.reference",
        source_type="analytic_reference",
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1e-8),
        lineage=make_evidence_lineage(
            implementation_refs=("reference.impl",),
            generator_refs=("reference.generator",),
            artifact_refs=("bands",),
            derivation_refs=("reference.derivation",),
            relationship_ids=(
                "independent-derivation:reference.derivation",
                "resolves-node:reference.impl",
                "resolves-node:reference.generator",
                "resolves-node:reference.derivation",
            ),
        ),
        human_attestation_refs=(attestation.attestation_id,),
    )
    claim = make_certification_claim(
        claim_id="claim.output.finite",
        subject_id=model.observable_id,
        predicate_id="output.finite",
        evidence_ids=(evidence.evidence_id,),
        measured=jnp.zeros(1),
        reference=jnp.zeros(1),
        residual=jnp.zeros(1),
        tolerance=jnp.zeros(1),
        passed=True,
        checked=True,
        in_domain=True,
        margin=0.5,
        severity_code=1,
    )
    domain = make_domain_result(
        predicate_id=predicate.predicate_id,
        measured=21.2,
        reference=20.0,
        residual=1.2,
        tolerance=20.0,
        margin=18.8,
        passed=True,
        checked=True,
        in_domain=True,
        severity_code=1,
    )
    derivatives = make_derivative_evidence(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        method="jax.linearize+jvp+vjp+central_fd",
        scales=jnp.array([0.05, 30.0]),
        jvp_probes=jnp.array([[1.0], [0.5]]),
        vjp_probes=jnp.array([[1.0, 0.5], [0.2, 0.1]]),
        reference_derivatives=jnp.array([[1.0], [0.5]]),
        derivative_residuals=jnp.zeros((2, 1)),
        singular_values=jnp.array([2.0, 0.25]),
        effective_rank=2,
        condition_estimate=8.0,
        finite=True,
        fd_correct=True,
    )
    dependencies = make_dependency_map(
        model_id=model.model_id,
        input_paths=("params.sigma", "params.temperature"),
        output_paths=("spectrum.intensity",),
        structural=jnp.array([[True, True]]),
        traced=jnp.array([[True, True]]),
    )
    sensitivities = make_sensitivity_map(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        scales=jnp.array([0.05, 30.0]),
        sensitivities=jnp.array([[1.0, 0.5]]),
        threshold=1e-12,
        active=jnp.array([[True, True]]),
    )
    information = make_information_spectrum(
        input_paths=("params.sigma", "params.temperature"),
        singular_values=jnp.array([2.0, 0.25]),
        right_singular_vectors=jnp.eye(2),
        effective_rank=2,
        condition_estimate=8.0,
        threshold=1e-10,
    )
    policy = make_policy_report(
        policy_id="org.diffpes.policy.research.v1",
        level_ids=(
            "identified",
            "validated",
            "differentiable",
            "verified",
            "benchmarked",
            "reproducible",
        ),
        required_claim_ids=(claim.claim_id,),
        claim_passed=jnp.array([True]),
        claim_checked=jnp.array([True]),
        claim_in_domain=jnp.array([True]),
        achieved=jnp.array([True, True, True, True, False, False]),
    )
    certificate: ForwardCertificate = make_forward_certificate(
        manifest=manifest,
        model=model,
        artifacts=(artifact,),
        transformations=(transformation,),
        evidence=(evidence,),
        attestations=(attestation,),
        claims=(claim,),
        domains=(domain,),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy,
        policy_id=policy.policy_id,
        certificate_checksum=(
            "sha256:1:certificate:"
            "5060708050607080506070805060708050607080506070805060708050607080"
        ),
        extensions_json=extensions_json,
    )
    return certificate


@jaxtyped(typechecker=beartype)
def toy_chain_diagonalized(
    n_k: int = 16,
) -> Tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native one-dimensional chain on a fixed k-path.

    Uses the production -1 eV hopping and an endpoint-inclusive fractional
    path from -1/2 to 1/2 along kx. The factory uses no random seed.
    """
    model: TBModel = make_1d_chain_model()
    kx: Float64[Array, " n_k"] = jnp.linspace(
        -0.5, 0.5, n_k, dtype=jnp.float64
    )
    kpoints: Float64[Array, "n_k 3"] = jnp.stack(
        (kx, jnp.zeros_like(kx), jnp.zeros_like(kx)), axis=-1
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: Tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result
