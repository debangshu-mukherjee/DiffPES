"""Define types and factory functions for diffpes.

Extended Summary
----------------
This package provides PyTree-compatible data structures and their factory
functions for ARPES simulation data. The data includes crystal geometry,
band structures, orbital projections, experiment geometry, and detector
calibration. JAX stores fields that participate in autodiff as array children
and static topology or selector values as auxiliary data.

The package contains these submodules:

- :mod:`aliases`
    Define scalar type aliases for JAX-compatible numeric types.
- :mod:`bands`
    Define band-structure and orbital-projection data structures.
- :mod:`certification`
    Store JAX-native carriers for certified forward-model executions.
- :mod:`certification_constants`
    Define static identifiers and schema constants for forward certification.
- :mod:`constants`
    Define numerical, physical, orbital, and VASP-format constants for diffpes.
- :mod:`context`
    Define structured inputs for high-level VASP simulation workflows.
- :mod:`contracts`
    Define static carriers for certified transformation contracts.
- :mod:`detector_effects`
    Define detector-coordinate nuisance and acquisition state.
- :mod:`dos`
    Define density-of-states data structures.
- :mod:`experiment`
    Define the geometry of an ARPES experiment.
- :mod:`geometry`
    Define crystal-geometry data structures for VASP crystal structures.
- :mod:`inspection`
    Store types-owned records from certificate inspection.
- :mod:`kpath`
    Define k-space path and grid data structures.
- :mod:`provenance`
    Store types-owned carriers for artifact provenance and information flow.
- :mod:`radial_params`
    Define radial-wavefunction parameter structures.
- :mod:`runtime`
    Store mutable host-side state for certification services.
- :mod:`self_energy`
    Define the causal self-energy model carrier.
- :mod:`spectral`
    Define spectral-tail and streamed-source data structures.
- :mod:`tb_model`
    Define tight-binding model and diagonalized-band data structures.
- :mod:`volumetric`
    Define volumetric data structures for VASP CHGCAR files.
- :mod:`wannier`
    Define operator metadata carried alongside an ingested Wannier model.

Routine Listings
----------------
:class:`ArpesCube`
    Store source-coordinate ARPES intensity on a Cartesian momentum raster.
:class:`ArpesSpectrum`
    Store self-describing ARPES path intensity in a JAX PyTree.
:class:`ArtifactRef`
    Store static identity and role for one source or derived artifact.
:class:`BandStructure`
    Store electronic band-structure data in a JAX PyTree.
:class:`CertificateDiff`
    Store categorized differences between two forward certificates.
:class:`CertificationClaim`
    Store a named claim and its continuous numerical evidence.
:class:`CertificationContext`
    Store prepared selections and references for compiled certification.
:class:`CertificationRegistryState`
    Store mutable entries for the process-local certification registry.
:class:`CertifiedResult`
    Store a numerical result paired with its differentiable certificate.
:class:`CompositionReport`
    Store a conservative transformation-composition result.
:class:`ConventionRef`
    Store a versioned semantic convention used by a scientific model.
:class:`CrystalGeometry`
    Store VASP POSCAR crystal geometry in a JAX PyTree.
:class:`DensityOfStates`
    Store density-of-states data in a JAX PyTree.
:class:`DependencyAnalysisCache`
    Store cached structural dependency analyses and access counters.
:class:`DependencyMap`
    Store declared and JAXPR-observed dependency relations.
:class:`DerivativeEvidence`
    Store JVP, VJP, reference, and information-spectrum evidence.
:class:`DetectorCalibration`
    Store native detector-bin and point-spread calibration.
:class:`DetectorEffects`
    Store the complete v1 detector-effects PyTree.
:class:`DetectorRaster`
    Store expected detector counts on native recorded coordinates.
:class:`DiagonalizedBands`
    Store diagonalized electronic-structure data in a JAX PyTree.
:class:`DomainPredicate`
    Store a static declaration of one model-domain predicate.
:class:`DomainResult`
    Store the traced evaluation of one declared domain predicate.
:class:`EvidenceLineage`
    Store named implementation, generator, artifact, and derivation lineage.
:class:`EvidenceRef`
    Store numerical evidence with static method and source identity.
:class:`EvidenceReport`
    Store the offline consistency outcome for one evidence record.
:class:`ExecutionManifest`
    Store software and execution identity prepared at the I/O boundary.
:class:`ExperimentGeometry`
    Store the geometry of an ARPES experiment.
:class:`FinalStateSpec`
    Store a certified radial final-state selection.
:class:`ForwardCertificate`
    Store the complete assurance record for one forward execution.
:class:`ForwardModelSpec`
    Store the identity of a differentiable forward model.
:class:`FullDensityOfStates`
    Store spin-resolved total and projected DOS data in a JAX PyTree.
:class:`HamiltonianBlocks`
    Store normalized Hamiltonian matrices with exact block metadata.
:class:`HandshakeReport`
    Store the validation outcome for one registration handshake.
:class:`HoppingRecord`
    Store one parsed non-onsite hopping and its source line.
:class:`HumanAttestationRef`
    Record a human review separately from computational evidence.
:class:`InformationSpectrum`
    Store a matrix-free information spectrum in input coordinates.
:class:`InformationState`
    Store effective semantic state for one artifact or result node.
:class:`KGrid`
    Store a fixed-shape raster in fractional k-space.
:class:`KPath`
    Store a generated path through fractional k-space.
:class:`KPathInfo`
    Store k-point path metadata in a JAX PyTree.
:class:`MatrixElementParams`
    Store shell-shared matrix-element scales and channel phases.
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:class:`OrbitalProjection`
    Store orbital-resolved band projections in a JAX PyTree.
:class:`PolicyReport`
    Store a traced policy truth table for derived certification levels.
:class:`Power2TailSpec`
    Store the six derived coefficients for two causal power-law tails.
:class:`ProvenanceAnalysis`
    Store the complete result of one provenance-graph analysis.
:class:`ProvenanceGraph`
    Store a validated lineage graph and its propagated semantics.
:class:`ProvenanceReport`
    Store a structural and semantic provenance-validation report.
:class:`RadialQuadratureSpec`
    Store one immutable certified radial-quadrature profile.
:class:`RadialSpec`
    Store shell-shared radial-wavefunction parameters.
:class:`RegisteredModel`
    Store a frozen binding between a model spec and its executor.
:class:`RegisteredTransformation`
    Store a frozen transformation and its consistency checksum.
:class:`RegistrationHandshake`
    Store registration requirements for one certification owner.
:class:`RegistryReport`
    Store the structural validation result for one registry snapshot.
:class:`RegistrySnapshot`
    Store an immutable deterministic snapshot of registry entries.
:class:`ReproductionReport`
    Store a numerical comparison from deliberate forward re-execution.
:class:`SelfEnergyModel`
    Store a causal self-energy parameterization as a JAX PyTree.
:class:`SensitivityMap`
    Store scaled sensitivities from inputs to output projections.
:class:`SlabSpec`
    Store static slab construction choices and provenance.
:class:`SlabTopology`
    Store host-selected discrete slab topology for pure-JAX rebuilding.
:class:`SlaterKosterParams`
    Store differentiable Slater--Koster two-center integrals.
:class:`SOCVolumetricData`
    Store SOC CHGCAR volumetric-grid data in a JAX PyTree.
:class:`SpinBandStructure`
    Store spin-resolved electronic band-structure data in a JAX PyTree.
:class:`SpinOrbitalProjection`
    Store orbital projections with spin data in a JAX PyTree.
:class:`SurfaceCell`
    Store a validated Cartesian surface-cell frame.
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
:class:`TextLineCursor`
    Record strict line-numbered parsing for one text file.
:class:`TransformationContract`
    Store the static semantic contract for one registered transformation.
:class:`TransformationRecord`
    Store one transformation and its semantic information effects.
:class:`TransitionSourceSchedule`
    Store compact inputs for block-local transition-source assembly.
:class:`VerificationReport`
    Store an offline certificate-verification outcome.
:class:`VolumetricData`
    Store CHGCAR volumetric-grid data in a JAX PyTree.
:class:`WaiverRecord`
    Store a bounded policy-waiver declaration without changing claim status.
:class:`WaiverReport`
    Store the temporal validation outcome for one waiver.
:class:`WannierOperatorData`
    Store operator metadata for a parsed Wannier tight-binding model.
:class:`WorkflowContext`
    Store parsed VASP inputs for high-level workflow helpers.
:func:`constant_energy_map`
    Compute an ARPES map inside an explicit energy window.
:func:`fermi_surface_map`
    Compute an ARPES map around the Fermi level.
:func:`make_arpes_cube`
    Create a validated ``ArpesCube`` instance.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`make_artifact_ref`
    Create a validated artifact reference.
:func:`make_band_structure`
    Create a validated ``BandStructure`` instance.
:func:`make_certificate_diff`
    Construct a validated certificate-difference record.
:func:`make_certification_claim`
    Create a claim retaining both continuous and discrete evidence.
:func:`make_certification_context`
    Create a prepared certification context.
:func:`make_certification_registry_state`
    Create empty mutable state for the certification registry.
:func:`make_certified_result`
    Pair any JAX-compatible result value with a forward certificate.
:func:`make_composition_report`
    Create a validated immutable transformation-composition report.
:func:`make_convention_ref`
    Create a validated convention reference.
:func:`make_crystal_geometry`
    Create a validated CrystalGeometry instance.
:func:`make_density_of_states`
    Create a validated DensityOfStates instance.
:func:`make_dependency_analysis_cache`
    Create an empty mutable cache for dependency analyses.
:func:`make_dependency_map`
    Create a structural dependency map.
:func:`make_derivative_evidence`
    Create validated derivative and local-information evidence.
:func:`make_detector_calibration`
    Create a validated ``DetectorCalibration`` instance.
:func:`make_detector_effects`
    Create validated detector-effects state.
:func:`make_detector_raster`
    Create a validated ``DetectorRaster`` instance.
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
:func:`make_domain_predicate`
    Create a validated domain-predicate declaration.
:func:`make_domain_result`
    Create one traced domain evaluation.
:func:`make_evidence_lineage`
    Create named evidence lineage without asserting independence.
:func:`make_evidence_ref`
    Create validated vector-valued numerical evidence.
:func:`make_evidence_report`
    Create an offline evidence-verification report.
:func:`make_execution_manifest`
    Create a validated execution manifest.
:func:`make_experiment_geometry`
    Create a validated geometry for an ARPES experiment.
:func:`make_final_state_spec`
    Create a validated radial final-state selection.
:func:`make_forward_certificate`
    Create and cross-validate a complete forward certificate.
:func:`make_forward_model_spec`
    Create a validated stable forward-model specification.
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.
:func:`make_hamiltonian_blocks`
    Create normalized Hamiltonian blocks without changing parsed values.
:func:`make_handshake_report`
    Create a report for one registration handshake.
:func:`make_hopping_record`
    Create one parsed hopping record without changing its values.
:func:`make_human_attestation_ref`
    Create a named human-review record.
:func:`make_information_spectrum`
    Create a validated local information spectrum.
:func:`make_information_state`
    Create a validated semantic-information state for one graph node.
:func:`make_kgrid`
    Create a validated fixed-shape k-space raster.
:func:`make_kpath`
    Create a validated path through fractional k-space.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
:func:`make_matrix_element_params`
    Create validated shell-shared matrix-element parameters.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_orbital_projection`
    Create a validated ``OrbitalProjection`` instance.
:func:`make_policy_report`
    Create a validated policy truth table.
:func:`make_power2_tail_spec`
    Create a scalar-valued causal-tail carrier.
:func:`make_provenance_analysis`
    Create an immutable provenance-analysis carrier.
:func:`make_provenance_graph`
    Create a validated immutable provenance graph carrier.
:func:`make_provenance_report`
    Create a validated structural and semantic provenance report.
:func:`make_radial_quadrature_spec`
    Select one immutable certified quadrature profile.
:func:`make_radial_spec`
    Create a validated shell-shared radial specification.
:func:`make_registered_model`
    Create a validated model-registry binding.
:func:`make_registered_transformation`
    Create a validated transformation-registry binding.
:func:`make_registration_handshake`
    Create registration requirements for one certification owner.
:func:`make_registry_report`
    Create a validated structural registry report.
:func:`make_registry_snapshot`
    Create an immutable registry snapshot.
:func:`make_reproduction_report`
    Create a report comparing a result with its re-execution.
:func:`make_self_energy_model`
    Create a validated self-energy model.
:func:`make_sensitivity_map`
    Create a named, scaled local-sensitivity map.
:func:`make_slab_spec`
    Create a validated slab-construction sidecar.
:func:`make_slab_topology`
    Create validated host-selected slab topology metadata.
:func:`make_slater_koster_params`
    Create validated Slater--Koster two-center parameters.
:func:`make_soc_volumetric_data`
    Create a validated ``SOCVolumetricData`` instance.
:func:`make_spin_band_structure`
    Create a validated ``SpinBandStructure`` instance.
:func:`make_spin_orbital_projection`
    Create a validated ``SpinOrbitalProjection`` instance.
:func:`make_surface_cell`
    Create a validated Cartesian surface-cell carrier.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.
:func:`make_text_line_cursor`
    Create a line cursor from one UTF-8 text file.
:func:`make_transformation_contract`
    Create a validated immutable transformation contract.
:func:`make_transformation_record`
    Create a validated information-aware transformation record.
:func:`make_transition_source_schedule`
    Create a shape-consistent streamed transition-source schedule.
:func:`make_verification_report`
    Create an offline certificate-verification report.
:func:`make_volumetric_data`
    Create a validated ``VolumetricData`` instance.
:func:`make_waiver_record`
    Create a bounded policy-waiver declaration.
:func:`make_waiver_report`
    Create a temporal waiver-validation report.
:func:`make_wannier_operator_data`
    Create validated Wannier operator metadata.
:func:`make_workflow_context`
    Create a workflow context from parsed VASP inputs.
:func:`slice_edc`
    Interpolate an energy-distribution curve from an ARPES cube.
:func:`slice_mdc`
    Interpolate a momentum-distribution map from an ARPES cube.
:obj:`ArtifactResolver`
    Resolve an artifact to normalized content and optional source bytes.
:obj:`ATTR_AUX`
    HDF5 attribute name storing auxiliary PyTree data as JSON.
:obj:`ATTR_NONE`
    HDF5 attribute name listing PyTree fields stored as None.
:obj:`ATTR_TYPE`
    HDF5 attribute name storing the PyTree type name.
:obj:`BAND_GROUP_COMPLEMENT_GAP_MIN_EV`
    Minimum isolation required for a complete static band group.
:obj:`BAND_LINE_MIN_VALUES`
    Minimum tokens on an EIGENVAL band line.
:obj:`BAND_LINE_SPIN_VALUES`
    Tokens on a spin-polarized EIGENVAL band line.
:obj:`BAND_NDIM`
    Expected dimensionality of band-energy arrays.
:obj:`BOHR_TO_ANGSTROM`
    Bohr radius in Angstrom.
:obj:`CANONICAL_ARRAY_CHUNK_BYTES`
    Array chunk size used by canonical PyTree encoding in bytes.
:obj:`CANONICAL_JSON_PREFIX`
    Domain prefix for canonical JSON identities.
:obj:`CANONICAL_JSON_VERSION`
    Version of the canonical JSON representation.
:obj:`CANONICAL_PYTREE_PREFIX`
    Domain prefix for canonical PyTree identities.
:obj:`CANONICAL_PYTREE_VERSION`
    Version of the canonical PyTree representation.
:obj:`CANONICAL_SUPPORTED_ARRAY_KINDS`
    NumPy dtype kinds accepted by canonical array encoding.
:obj:`CARTESIAN_COMPONENTS`
    Number of Cartesian components used by tight-binding bond vectors.
:obj:`CERTIFICATE_ARRAY_KINDS`
    NumPy dtype kinds accepted in persisted certificates.
:obj:`CERTIFICATE_ARRAY_PREVIEW_ITEMS`
    Maximum array elements shown by certificate inspection.
:obj:`CERTIFICATE_DOCUMENT_KEYS`
    Required top-level keys in a certificate document.
:obj:`CERTIFICATE_FORMAT`
    Stable identifier for the forward-certificate document format.
:obj:`CERTIFICATE_H5_GROUP`
    Reserved HDF5 group containing attached certificates.
:obj:`CERTIFICATE_SCHEMA_MAJOR`
    Supported major version of the certificate schema.
:obj:`CERTIFICATE_SCHEMA_MINOR`
    Supported minor version of the certificate schema.
:obj:`CERTIFICATE_SCHEMA_PATTERN`
    Pattern matching supported certificate schema versions.
:obj:`CERTIFICATION_IDENTIFIER_PATTERN`
    Pattern matching permanent certification identifiers.
:obj:`CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES`
    Claim prefixes requiring independent evidence.
:obj:`CERTIFICATION_LEVEL_IDS`
    Ordered cumulative scientific-certification level identifiers.
:obj:`CERTIFICATION_LEVEL_PREFIXES`
    Evidence prefixes required by each certification level.
:obj:`CERTIFICATION_LINEAGE_RELATIONSHIPS`
    Typed lineage relationships accepted by certification policy.
:obj:`CERTIFICATION_POLICY_IDS`
    Stable identifiers of built-in cumulative policies.
:obj:`CERTIFICATION_POLICY_LEVEL_COUNT`
    Number of required levels for each built-in policy.
:obj:`CERTIFICATION_SEMVER_PATTERN`
    Pattern matching certification semantic versions.
:obj:`CERTIFICATION_SHARED_RELATIONSHIPS`
    Lineage relationships that share implementation ancestry.
:obj:`CHANNELS_BY_PAIR`
    Slater--Koster channels supported for each angular-momentum pair.
:obj:`CheckFunction`
    Callable signature for a pure JAX certification check.
:obj:`CHECKSUM_ALGORITHM`
    Name of the scientific-identity digest algorithm.
:obj:`CHECKSUM_FILE_CHUNK_BYTES`
    File chunk size used by streaming scientific identities in bytes.
:obj:`CHECKSUM_FORMAT_VERSION`
    Version of the scientific-identity text format.
:obj:`CHECKSUM_PATTERN`
    Pattern matching formatted scientific-identity records.
:obj:`CHECKSUM_RECORD_KIND_PATTERN`
    Pattern matching scientific-identity record-kind identifiers.
:obj:`COORDINATE_MODE_TOKENS`
    Recognized KPOINTS coordinate-mode tokens.
:obj:`D_ORBITAL_SLICE`
    Slice selecting the five d orbitals.
:obj:`DEGENERACY_GROUP_TOL_EV`
    Maximum group-to-complement gap treated as a cut degeneracy.
:obj:`DosType`
    Supported density-of-states containers.
:obj:`EIG_DOWN_INDEX`
    Column index of spin-down eigenvalues in EIGENVAL.
:obj:`EIG_UP_INDEX`
    Column index of spin-up eigenvalues in EIGENVAL.
:obj:`ENERGY_AXIS_NDIM`
    Expected dimensionality of energy-axis arrays.
:obj:`EPS`
    Epsilon floor guarding divisions and norms.
:obj:`EPS_DEG`
    Lorentzian width regularizing degenerate eigenvector derivatives.
:obj:`FLOAT_TOKEN_RE`
    Compiled regex matching floating-point tokens.
:obj:`G_PARALLEL_ATOL_INV_ANG`
    Surface parallel-momentum conservation tolerance in inverse Angstrom.
:obj:`GAUNT_IMAG_TOL`
    Tolerance for discarding imaginary Gaunt residues.
:obj:`GROUP_COMPLEMENT_GAP_MIN_EV`
    Minimum spectral isolation required for a registered band group.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`HBAR_SQ_OVER_2ME_EV_ANG2`
    Store the free-electron dispersion constant in eV Angstrom squared.
:obj:`HOPPING_LIST_COMPLEX_FIELDS`
    Number of fields in a complex Cartesian hopping-list row.
:obj:`HOPPING_LIST_REAL_FIELDS`
    Number of fields in a real Cartesian hopping-list row.
:obj:`INTENSITY_NDIM`
    Expected dimensionality of intensity arrays.
:obj:`ISPIN2_BLOCKS`
    PROCAR block count for ISPIN=2 calculations.
:obj:`ISPIN_SPIN_POLARIZED`
    ISPIN value marking spin-polarized VASP runs.
:obj:`K_PREFACTOR_INV_ANG_SQRT_EV`
    Store the momentum prefactor in inverse Angstrom per square-root eV.
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:obj:`KNOWN_CHANNELS`
    Complete set of supported Slater--Koster channel names.
:obj:`KPATH_AUX_WITH_COMMENT_LEN`
    KPathInfo auxiliary-data length including a comment.
:obj:`KPATH_AUX_WITH_COORD_MODE_LEN`
    KPathInfo auxiliary-data length including a coordinate mode.
:obj:`KPOINT_LINE_VALUES`
    Tokens on an EIGENVAL k-point line.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`LATTICE_ROWS`
    Number of lattice-vector rows in POSCAR/CHGCAR headers.
:obj:`M_D`
    Magnetic quantum numbers of the d orbitals.
:obj:`M_P`
    Magnetic quantum numbers of the p orbitals.
:obj:`MATRIX_NDIM`
    Expected dimensionality of tight-binding operator matrices.
:obj:`MAX_SK_ANGULAR_MOMENTUM`
    Maximum angular momentum supported by Slater--Koster construction.
:obj:`ME_EV`
    Electron rest energy in eV.
:obj:`MIN_BOND_DISTANCE`
    Minimum nonzero distance accepted by neighbor discovery.
:obj:`MIN_SUM`
    Minimum-sum floor guarding normalizations.
:obj:`MINIMUM_AXIS_POINTS`
    Minimum number of points accepted on a sampled DOS energy axis.
:obj:`N_ORBITALS`
    Number of orbitals in the VASP projection basis.
:obj:`N_SOC_MAG_BLOCKS`
    Magnetization block count in SOC CHGCAR files.
:obj:`N_SPIN_COMPONENTS`
    Spin-projection component count in PROCAR.
:obj:`NON_S_ORBITAL_SLICE`
    Slice selecting all non-s orbitals.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:obj:`NONSPIN_COLS`
    DOSCAR column count without spin polarization.
:obj:`NORM_EPS`
    Epsilon floor guarding eigenvector normalization.
:obj:`ORBITAL_INDEX`
    Mapping from orbital name to VASP orbital index.
:obj:`P_ORBITAL_SLICE`
    Slice selecting the three p orbitals.
:obj:`PARAMETER_KEY_PARTS`
    Number of colon-delimited parts in a qualified Slater--Koster key.
:obj:`PHASE_LOSS_MESSAGE`
    Warning text for PROCAR magnitude-only eigenvectors.
:obj:`PRESET_NAMES`
    Recognized band-scatter plotting preset names.
:obj:`ProjectionType`
    Supported orbital-projection containers.
:obj:`PyTreeDef`
    Runtime pytree definition with a typed static-analysis stand-in.
:obj:`S_IDX`
    Index of the s orbital.
:obj:`SCALAR_LINE_COMPONENTS`
    Tokens on a scalar CHGCAR header line.
:obj:`ScalarBool`
    Union of ``bool`` and ``Bool[Array, " "]``.
:obj:`ScalarComplex`
    Union of ``complex`` and ``Complex[Array, " "]``.
:obj:`ScalarFloat`
    Union of ``float`` and ``Float[Array, " "]``.
:obj:`ScalarInteger`
    Union of ``int`` and ``Int[Array, " "]``.
:obj:`ScalarNumeric`
    Union of ``int``, ``float``, ``complex``, and ``Num[Array, " "]``.
:obj:`SHELL_ATOLERANCE`
    Absolute tolerance for grouping equal-distance neighbor shells.
:obj:`SHELL_RTOLERANCE`
    Relative tolerance for grouping equal-distance neighbor shells.
:obj:`SMALL_ARGUMENT`
    Small-argument cutoff for spherical Bessel seeds.
:obj:`SOC_BLOCKS`
    PROCAR block count for SOC calculations.
:obj:`SPECIES_PAIR_PARTS`
    Number of species labels in a Slater--Koster material-pair key.
:obj:`SPECTRUM_NDIM`
    Expected dimensionality of tight-binding eigenspectra.
:obj:`SPIN_COLS`
    DOSCAR column count with spin polarization.
:obj:`TB_RADIAL_INPUT_COUNT`
    Number of positional inputs accepted by the radial ARPES model adapter.
:obj:`TB_RADIAL_MODEL_ID`
    Permanent identifier of the radial ARPES forward model.
:obj:`TB_RADIAL_MODEL_VERSION`
    Semantic version of the radial ARPES forward model.
:obj:`TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2`
    Store the inverse free-electron dispersion constant.
:obj:`WANNIER_CELL_FIELDS`
    Number of integer components in a Wannier translation.
:obj:`WANNIER_CENTRE_CONSISTENCY_TOLERANCE`
    Cartesian tolerance for centres assigned to one atom.
:obj:`WANNIER_DEGENERACIES_PER_LINE`
    Maximum degeneracy weights on one Wannier90 line.
:obj:`WANNIER_HERMITICITY_TOLERANCE`
    Absolute tolerance for real-space Hermitian closure.
:obj:`WANNIER_HR_HAMILTONIAN_FIELDS`
    Number of fields in a Wannier90 HR Hamiltonian row.
:obj:`WANNIER_HR_SUFFIX`
    Required suffix for a Wannier90 HR file.
:obj:`WANNIER_INTEGER_RECOVERY_TOLERANCE`
    Fractional tolerance for recovering an exact translation.
:obj:`WANNIER_TB_HAMILTONIAN_FIELDS`
    Number of fields in a Wannier90 TB Hamiltonian row.
:obj:`WANNIER_TB_POSITION_FIELDS`
    Number of fields in a Wannier90 TB position row.
:obj:`WANNIER_TB_SUFFIX`
    Required suffix for a Wannier90 TB file.
:obj:`WEIGHT_COMPONENT_COUNT`
    Tokens on a weighted k-point line.
:obj:`WEIGHT_COMPONENT_INDEX`
    Index of the weight token on a k-point line.
:obj:`XYZ_COMPONENTS`
    Number of Cartesian vector components.

Notes
-----
All structured carriers are immutable :class:`equinox.Module` PyTrees.
Array fields remain differentiable leaves, while shape and control-flow
metadata use ``equinox.field(static=True)``.
"""

from .aliases import (
    NonJaxNumber,
    PyTreeDef,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
)
from .bands import (
    ArpesCube,
    ArpesSpectrum,
    BandStructure,
    DetectorCalibration,
    DetectorRaster,
    OrbitalProjection,
    SpinBandStructure,
    SpinOrbitalProjection,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_arpes_spectrum,
    make_band_structure,
    make_detector_calibration,
    make_detector_raster,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
    slice_edc,
    slice_mdc,
)
from .certification import (
    ArtifactRef,
    ArtifactResolver,
    CertificationClaim,
    CertificationContext,
    CertifiedResult,
    CheckFunction,
    ConventionRef,
    DependencyMap,
    DerivativeEvidence,
    DomainPredicate,
    DomainResult,
    EvidenceLineage,
    EvidenceRef,
    EvidenceReport,
    ExecutionManifest,
    ForwardCertificate,
    ForwardModelSpec,
    HandshakeReport,
    HumanAttestationRef,
    InformationSpectrum,
    PolicyReport,
    RegisteredModel,
    RegisteredTransformation,
    RegistrationHandshake,
    RegistryReport,
    RegistrySnapshot,
    ReproductionReport,
    SensitivityMap,
    TransformationRecord,
    VerificationReport,
    WaiverRecord,
    WaiverReport,
    make_artifact_ref,
    make_certification_claim,
    make_certification_context,
    make_certified_result,
    make_convention_ref,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_lineage,
    make_evidence_ref,
    make_evidence_report,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_handshake_report,
    make_human_attestation_ref,
    make_information_spectrum,
    make_policy_report,
    make_registered_model,
    make_registered_transformation,
    make_registration_handshake,
    make_registry_report,
    make_registry_snapshot,
    make_reproduction_report,
    make_sensitivity_map,
    make_transformation_record,
    make_verification_report,
    make_waiver_record,
    make_waiver_report,
)
from .certification_constants import (
    CANONICAL_ARRAY_CHUNK_BYTES,
    CANONICAL_JSON_PREFIX,
    CANONICAL_JSON_VERSION,
    CANONICAL_PYTREE_PREFIX,
    CANONICAL_PYTREE_VERSION,
    CANONICAL_SUPPORTED_ARRAY_KINDS,
    CERTIFICATE_ARRAY_KINDS,
    CERTIFICATE_ARRAY_PREVIEW_ITEMS,
    CERTIFICATE_DOCUMENT_KEYS,
    CERTIFICATE_FORMAT,
    CERTIFICATE_H5_GROUP,
    CERTIFICATE_SCHEMA_MAJOR,
    CERTIFICATE_SCHEMA_MINOR,
    CERTIFICATE_SCHEMA_PATTERN,
    CERTIFICATION_IDENTIFIER_PATTERN,
    CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES,
    CERTIFICATION_LEVEL_IDS,
    CERTIFICATION_LEVEL_PREFIXES,
    CERTIFICATION_LINEAGE_RELATIONSHIPS,
    CERTIFICATION_POLICY_IDS,
    CERTIFICATION_POLICY_LEVEL_COUNT,
    CERTIFICATION_SEMVER_PATTERN,
    CERTIFICATION_SHARED_RELATIONSHIPS,
    CHECKSUM_ALGORITHM,
    CHECKSUM_FILE_CHUNK_BYTES,
    CHECKSUM_FORMAT_VERSION,
    CHECKSUM_PATTERN,
    CHECKSUM_RECORD_KIND_PATTERN,
    TB_RADIAL_INPUT_COUNT,
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
)
from .constants import (
    ATTR_AUX,
    ATTR_NONE,
    ATTR_TYPE,
    BAND_GROUP_COMPLEMENT_GAP_MIN_EV,
    BAND_LINE_MIN_VALUES,
    BAND_LINE_SPIN_VALUES,
    BAND_NDIM,
    BOHR_TO_ANGSTROM,
    CARTESIAN_COMPONENTS,
    CHANNELS_BY_PAIR,
    COORDINATE_MODE_TOKENS,
    D_ORBITAL_SLICE,
    DEGENERACY_GROUP_TOL_EV,
    EIG_DOWN_INDEX,
    EIG_UP_INDEX,
    ENERGY_AXIS_NDIM,
    EPS,
    EPS_DEG,
    FLOAT_TOKEN_RE,
    G_PARALLEL_ATOL_INV_ANG,
    GAUNT_IMAG_TOL,
    GROUP_COMPLEMENT_GAP_MIN_EV,
    HBAR_C_EV_A,
    HBAR_EV_S,
    HBAR_SQ_OVER_2ME_EV_ANG2,
    INTENSITY_NDIM,
    ISPIN2_BLOCKS,
    ISPIN_SPIN_POLARIZED,
    K_PREFACTOR_INV_ANG_SQRT_EV,
    KB_EV_PER_K,
    KNOWN_CHANNELS,
    KPATH_AUX_WITH_COMMENT_LEN,
    KPATH_AUX_WITH_COORD_MODE_LEN,
    KPOINT_LINE_VALUES,
    L_MAX,
    LATTICE_ROWS,
    M_D,
    M_P,
    MATRIX_NDIM,
    MAX_SK_ANGULAR_MOMENTUM,
    ME_EV,
    MIN_BOND_DISTANCE,
    MIN_SUM,
    MINIMUM_AXIS_POINTS,
    N_ORBITALS,
    N_SOC_MAG_BLOCKS,
    N_SPIN_COMPONENTS,
    NON_S_ORBITAL_SLICE,
    NONSPIN_COLS,
    NORM_EPS,
    ORBITAL_INDEX,
    P_ORBITAL_SLICE,
    PARAMETER_KEY_PARTS,
    PHASE_LOSS_MESSAGE,
    PRESET_NAMES,
    S_IDX,
    SCALAR_LINE_COMPONENTS,
    SHELL_ATOLERANCE,
    SHELL_RTOLERANCE,
    SMALL_ARGUMENT,
    SOC_BLOCKS,
    SPECIES_PAIR_PARTS,
    SPECTRUM_NDIM,
    SPIN_COLS,
    TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2,
    WEIGHT_COMPONENT_COUNT,
    WEIGHT_COMPONENT_INDEX,
    XYZ_COMPONENTS,
)
from .context import (
    DosType,
    ProjectionType,
    WorkflowContext,
    make_workflow_context,
)
from .contracts import (
    CompositionReport,
    TransformationContract,
    make_composition_report,
    make_transformation_contract,
)
from .detector_effects import DetectorEffects, make_detector_effects
from .dos import (
    DensityOfStates,
    FullDensityOfStates,
    make_density_of_states,
    make_full_density_of_states,
)
from .experiment import ExperimentGeometry, make_experiment_geometry
from .geometry import (
    CrystalGeometry,
    make_crystal_geometry,
)
from .inspection import CertificateDiff, make_certificate_diff
from .kpath import (
    KGrid,
    KPath,
    KPathInfo,
    make_kgrid,
    make_kpath,
    make_kpath_info,
)
from .provenance import (
    InformationState,
    ProvenanceAnalysis,
    ProvenanceGraph,
    ProvenanceReport,
    make_information_state,
    make_provenance_analysis,
    make_provenance_graph,
    make_provenance_report,
)
from .radial_params import (
    FinalStateSpec,
    MatrixElementParams,
    OrbitalBasis,
    RadialQuadratureSpec,
    RadialSpec,
    SlaterKosterParams,
    make_final_state_spec,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_slater_koster_params,
)
from .runtime import (
    CertificationRegistryState,
    DependencyAnalysisCache,
    make_certification_registry_state,
    make_dependency_analysis_cache,
)
from .self_energy import (
    SelfEnergyModel,
    make_self_energy_model,
)
from .spectral import (
    Power2TailSpec,
    TransitionSourceSchedule,
    make_power2_tail_spec,
    make_transition_source_schedule,
)
from .tb_model import (
    DiagonalizedBands,
    SlabSpec,
    SlabTopology,
    SurfaceCell,
    TBModel,
    make_diagonalized_bands,
    make_slab_spec,
    make_slab_topology,
    make_surface_cell,
    make_tb_model,
)
from .volumetric import (
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)
from .wannier import (
    HOPPING_LIST_COMPLEX_FIELDS,
    HOPPING_LIST_REAL_FIELDS,
    WANNIER_CELL_FIELDS,
    WANNIER_CENTRE_CONSISTENCY_TOLERANCE,
    WANNIER_DEGENERACIES_PER_LINE,
    WANNIER_HERMITICITY_TOLERANCE,
    WANNIER_HR_HAMILTONIAN_FIELDS,
    WANNIER_HR_SUFFIX,
    WANNIER_INTEGER_RECOVERY_TOLERANCE,
    WANNIER_TB_HAMILTONIAN_FIELDS,
    WANNIER_TB_POSITION_FIELDS,
    WANNIER_TB_SUFFIX,
    HamiltonianBlocks,
    HoppingRecord,
    TextLineCursor,
    WannierOperatorData,
    make_hamiltonian_blocks,
    make_hopping_record,
    make_text_line_cursor,
    make_wannier_operator_data,
)

__all__: list[str] = [
    "ArpesCube",
    "ArpesSpectrum",
    "ArtifactRef",
    "BandStructure",
    "CertificateDiff",
    "CertificationClaim",
    "CertificationContext",
    "CertificationRegistryState",
    "CertifiedResult",
    "CompositionReport",
    "ConventionRef",
    "CrystalGeometry",
    "DensityOfStates",
    "DependencyAnalysisCache",
    "DependencyMap",
    "DerivativeEvidence",
    "DetectorCalibration",
    "DetectorEffects",
    "DetectorRaster",
    "DiagonalizedBands",
    "DomainPredicate",
    "DomainResult",
    "EvidenceLineage",
    "EvidenceRef",
    "EvidenceReport",
    "ExecutionManifest",
    "ExperimentGeometry",
    "FinalStateSpec",
    "ForwardCertificate",
    "ForwardModelSpec",
    "FullDensityOfStates",
    "HamiltonianBlocks",
    "HandshakeReport",
    "HoppingRecord",
    "HumanAttestationRef",
    "InformationSpectrum",
    "InformationState",
    "KGrid",
    "KPath",
    "KPathInfo",
    "MatrixElementParams",
    "OrbitalBasis",
    "OrbitalProjection",
    "PolicyReport",
    "Power2TailSpec",
    "ProvenanceAnalysis",
    "ProvenanceGraph",
    "ProvenanceReport",
    "RadialQuadratureSpec",
    "RadialSpec",
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistrationHandshake",
    "RegistryReport",
    "RegistrySnapshot",
    "ReproductionReport",
    "SelfEnergyModel",
    "SensitivityMap",
    "SlabSpec",
    "SlabTopology",
    "SlaterKosterParams",
    "SOCVolumetricData",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "SurfaceCell",
    "TBModel",
    "TextLineCursor",
    "TransformationContract",
    "TransformationRecord",
    "TransitionSourceSchedule",
    "VerificationReport",
    "VolumetricData",
    "WaiverRecord",
    "WaiverReport",
    "WannierOperatorData",
    "WorkflowContext",
    "constant_energy_map",
    "fermi_surface_map",
    "make_arpes_cube",
    "make_arpes_spectrum",
    "make_artifact_ref",
    "make_band_structure",
    "make_certificate_diff",
    "make_certification_claim",
    "make_certification_context",
    "make_certification_registry_state",
    "make_certified_result",
    "make_composition_report",
    "make_convention_ref",
    "make_crystal_geometry",
    "make_density_of_states",
    "make_dependency_analysis_cache",
    "make_dependency_map",
    "make_derivative_evidence",
    "make_detector_calibration",
    "make_detector_effects",
    "make_detector_raster",
    "make_diagonalized_bands",
    "make_domain_predicate",
    "make_domain_result",
    "make_evidence_lineage",
    "make_evidence_ref",
    "make_evidence_report",
    "make_execution_manifest",
    "make_experiment_geometry",
    "make_final_state_spec",
    "make_forward_certificate",
    "make_forward_model_spec",
    "make_full_density_of_states",
    "make_hamiltonian_blocks",
    "make_handshake_report",
    "make_hopping_record",
    "make_human_attestation_ref",
    "make_information_spectrum",
    "make_information_state",
    "make_kgrid",
    "make_kpath",
    "make_kpath_info",
    "make_matrix_element_params",
    "make_orbital_basis",
    "make_orbital_projection",
    "make_policy_report",
    "make_power2_tail_spec",
    "make_provenance_analysis",
    "make_provenance_graph",
    "make_provenance_report",
    "make_radial_quadrature_spec",
    "make_radial_spec",
    "make_registered_model",
    "make_registered_transformation",
    "make_registration_handshake",
    "make_registry_report",
    "make_registry_snapshot",
    "make_reproduction_report",
    "make_self_energy_model",
    "make_sensitivity_map",
    "make_slab_spec",
    "make_slab_topology",
    "make_slater_koster_params",
    "make_soc_volumetric_data",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
    "make_surface_cell",
    "make_tb_model",
    "make_text_line_cursor",
    "make_transformation_contract",
    "make_transformation_record",
    "make_transition_source_schedule",
    "make_verification_report",
    "make_volumetric_data",
    "make_waiver_record",
    "make_waiver_report",
    "make_wannier_operator_data",
    "make_workflow_context",
    "slice_edc",
    "slice_mdc",
    "ArtifactResolver",
    "ATTR_AUX",
    "ATTR_NONE",
    "ATTR_TYPE",
    "BAND_GROUP_COMPLEMENT_GAP_MIN_EV",
    "BAND_LINE_MIN_VALUES",
    "BAND_LINE_SPIN_VALUES",
    "BAND_NDIM",
    "BOHR_TO_ANGSTROM",
    "CANONICAL_ARRAY_CHUNK_BYTES",
    "CANONICAL_JSON_PREFIX",
    "CANONICAL_JSON_VERSION",
    "CANONICAL_PYTREE_PREFIX",
    "CANONICAL_PYTREE_VERSION",
    "CANONICAL_SUPPORTED_ARRAY_KINDS",
    "CARTESIAN_COMPONENTS",
    "CERTIFICATE_ARRAY_KINDS",
    "CERTIFICATE_ARRAY_PREVIEW_ITEMS",
    "CERTIFICATE_DOCUMENT_KEYS",
    "CERTIFICATE_FORMAT",
    "CERTIFICATE_H5_GROUP",
    "CERTIFICATE_SCHEMA_MAJOR",
    "CERTIFICATE_SCHEMA_MINOR",
    "CERTIFICATE_SCHEMA_PATTERN",
    "CERTIFICATION_IDENTIFIER_PATTERN",
    "CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES",
    "CERTIFICATION_LEVEL_IDS",
    "CERTIFICATION_LEVEL_PREFIXES",
    "CERTIFICATION_LINEAGE_RELATIONSHIPS",
    "CERTIFICATION_POLICY_IDS",
    "CERTIFICATION_POLICY_LEVEL_COUNT",
    "CERTIFICATION_SEMVER_PATTERN",
    "CERTIFICATION_SHARED_RELATIONSHIPS",
    "CHANNELS_BY_PAIR",
    "CheckFunction",
    "CHECKSUM_ALGORITHM",
    "CHECKSUM_FILE_CHUNK_BYTES",
    "CHECKSUM_FORMAT_VERSION",
    "CHECKSUM_PATTERN",
    "CHECKSUM_RECORD_KIND_PATTERN",
    "COORDINATE_MODE_TOKENS",
    "D_ORBITAL_SLICE",
    "DEGENERACY_GROUP_TOL_EV",
    "DosType",
    "EIG_DOWN_INDEX",
    "EIG_UP_INDEX",
    "ENERGY_AXIS_NDIM",
    "EPS",
    "EPS_DEG",
    "FLOAT_TOKEN_RE",
    "G_PARALLEL_ATOL_INV_ANG",
    "GAUNT_IMAG_TOL",
    "GROUP_COMPLEMENT_GAP_MIN_EV",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "HBAR_SQ_OVER_2ME_EV_ANG2",
    "HOPPING_LIST_COMPLEX_FIELDS",
    "HOPPING_LIST_REAL_FIELDS",
    "INTENSITY_NDIM",
    "ISPIN2_BLOCKS",
    "ISPIN_SPIN_POLARIZED",
    "K_PREFACTOR_INV_ANG_SQRT_EV",
    "KB_EV_PER_K",
    "KNOWN_CHANNELS",
    "KPATH_AUX_WITH_COMMENT_LEN",
    "KPATH_AUX_WITH_COORD_MODE_LEN",
    "KPOINT_LINE_VALUES",
    "L_MAX",
    "LATTICE_ROWS",
    "M_D",
    "M_P",
    "MATRIX_NDIM",
    "MAX_SK_ANGULAR_MOMENTUM",
    "ME_EV",
    "MIN_BOND_DISTANCE",
    "MIN_SUM",
    "MINIMUM_AXIS_POINTS",
    "N_ORBITALS",
    "N_SOC_MAG_BLOCKS",
    "N_SPIN_COMPONENTS",
    "NON_S_ORBITAL_SLICE",
    "NonJaxNumber",
    "NONSPIN_COLS",
    "NORM_EPS",
    "ORBITAL_INDEX",
    "P_ORBITAL_SLICE",
    "PARAMETER_KEY_PARTS",
    "PHASE_LOSS_MESSAGE",
    "PRESET_NAMES",
    "ProjectionType",
    "PyTreeDef",
    "S_IDX",
    "SCALAR_LINE_COMPONENTS",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SHELL_ATOLERANCE",
    "SHELL_RTOLERANCE",
    "SMALL_ARGUMENT",
    "SOC_BLOCKS",
    "SPECIES_PAIR_PARTS",
    "SPECTRUM_NDIM",
    "SPIN_COLS",
    "TB_RADIAL_INPUT_COUNT",
    "TB_RADIAL_MODEL_ID",
    "TB_RADIAL_MODEL_VERSION",
    "TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2",
    "WANNIER_CELL_FIELDS",
    "WANNIER_CENTRE_CONSISTENCY_TOLERANCE",
    "WANNIER_DEGENERACIES_PER_LINE",
    "WANNIER_HERMITICITY_TOLERANCE",
    "WANNIER_HR_HAMILTONIAN_FIELDS",
    "WANNIER_HR_SUFFIX",
    "WANNIER_INTEGER_RECOVERY_TOLERANCE",
    "WANNIER_TB_HAMILTONIAN_FIELDS",
    "WANNIER_TB_POSITION_FIELDS",
    "WANNIER_TB_SUFFIX",
    "WEIGHT_COMPONENT_COUNT",
    "WEIGHT_COMPONENT_INDEX",
    "XYZ_COMPONENTS",
]
