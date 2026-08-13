"""Expose the public :mod:`diffpes.types` surface.

Extended Summary
----------------
- :mod:`aliases`
    Define scalar type aliases for JAX-compatible numeric types.
- :mod:`arpes`
    Define ARPES data carriers and coordinate slices.
- :mod:`bands`
    Define band-structure and orbital-projection data structures.
- :mod:`certification`
    Store aggregate carriers for certified forward executions.
- :mod:`certification_validation`
    Validate shared certification values.
- :mod:`context`
    Define structured inputs for high-level VASP simulation workflows.
- :mod:`contracts`
    Define static carriers for certified transformation contracts.
- :mod:`coordinates`
    Define typed measurement coordinates.
- :mod:`derivatives`
    Define derivative and information evidence records.
- :mod:`detector_data`
    Define detector calibration and raster data structures.
- :mod:`detector_effects`
    Define detector-coordinate nuisance and acquisition state.
- :mod:`diagonalized_bands`
    Define diagonalized electronic-structure data.
- :mod:`dos`
    Define density-of-states data structures.
- :mod:`electronic_state`
    Define solver-neutral electronic-state capabilities and a native source.
- :mod:`electronic_structure_validation`
    Validate shared electronic-structure geometry.
- :mod:`evidence`
    Define certification evidence and lineage records.
- :mod:`experiment`
    Define the geometry of an ARPES experiment.
- :mod:`experiment_state`
    Define split experiment carriers without hidden workflow state.
- :mod:`fidelity`
    Define immutable scientific-fidelity declarations.
- :mod:`generalized_spectral`
    Define typed sources and evaluated batches for generalized spectra.
- :mod:`geometry`
    Define crystal-geometry data structures for VASP crystal structures.
- :mod:`inspection`
    Store types-owned records from certificate inspection.
- :mod:`kpath`
    Define k-space path and grid data structures.
- :mod:`ks_scattering`
    Define native finite-slab Kohn--Sham scattering contracts.
- :mod:`ks_scattering_solution`
    Define scattering solver policies and evaluated result batches.
- :mod:`orbital_basis`
    Define orbital-basis metadata for radial models.
- :mod:`photocurrent`
    Define the typed factorized-photocurrent model boundary.
- :mod:`plane_wave`
    Define bounded plane-wave and PAW carriers for solver-neutral ARPES.
- :mod:`provenance`
    Store types-owned carriers for artifact provenance and information flow.
- :mod:`radial_params`
    Define radial-wavefunction and matrix-element parameters.
- :mod:`radial_profiles`
    Define certified radial quadrature and final-state profiles.
- :mod:`registry`
    Define immutable certification registry records.
- :mod:`reports`
    Define certification policy and verification reports.
- :mod:`retarded_validation`
    Validate matrix-valued retarded spectral evidence.
- :mod:`result`
    Define intrinsic and observed ARPES result carriers.
- :mod:`runtime`
    Store mutable host-side state for certification services.
- :mod:`self_energy`
    Define the causal self-energy model carrier.
- :mod:`sharding`
    Define static execution policies for JAX sharding.
- :mod:`slab_geometry`
    Define exact surface-cell and slab geometry metadata.
- :mod:`slab_topology`
    Define host-selected slab topology metadata.
- :mod:`slater_koster_params`
    Define Slater--Koster two-center parameters.
- :mod:`specification`
    Define certified forward-model specifications.
- :mod:`spectral`
    Define spectral-tail and streamed-source data structures.
- :mod:`tb_model`
    Define differentiable tight-binding model parameters.
- :mod:`volumetric`
    Define volumetric data structures for VASP CHGCAR files.
- :mod:`wannier`
    Define operator metadata carried alongside an ingested Wannier model.

Routine Listings
----------------
:class:`Acquisition`
    Define the ``Acquisition`` public contract.
:class:`ArpesCube`
    Store source-coordinate ARPES intensity on a Cartesian momentum raster.
:class:`ArpesSpectrum`
    Store self-describing ARPES path intensity in a JAX PyTree.
:class:`ArtifactRef`
    Store static identity and role for one source or derived artifact.
:class:`BackingAbsorberSpec`
    Define the ``BackingAbsorberSpec`` public contract.
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
:class:`DerivativeCapability`
    Define the ``DerivativeCapability`` public contract.
:class:`DysonSpectralSource`
    Define the ``DysonSpectralSource`` public contract.
:class:`DependencyAnalysisCache`
    Store cached structural dependency analyses and access counters.
:class:`DependencyMap`
    Store declared and JAXPR-observed dependency relations.
:class:`DerivativeEvidence`
    Store JVP, VJP, reference, and information-spectrum evidence.
:class:`DenseSliceOperator`
    Define the ``DenseSliceOperator`` public contract.
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
:class:`Experiment`
    Define the ``Experiment`` public contract.
:class:`EvidenceRef`
    Store numerical evidence with static method and source identity.
:class:`EvidenceReport`
    Store the offline consistency outcome for one evidence record.
:class:`ExecutionManifest`
    Store software and execution identity prepared at the I/O boundary.
:class:`ExperimentGeometry`
    Store the geometry of an ARPES experiment.
:class:`ElectronicStateSource`
    Define the ``ElectronicStateSource`` public contract.
:class:`ElectronicStateArchive`
    Define the ``ElectronicStateArchive`` public contract.
:class:`EigensystemSource`
    Define the ``EigensystemSource`` public contract.
:class:`FidelityManifest`
    Define the ``FidelityManifest`` public contract.
:class:`FinalStateSpec`
    Store a certified radial final-state selection.
:class:`FactorizedArpesModel`
    Define the ``FactorizedArpesModel`` public contract.
:class:`ForwardCertificate`
    Store the complete assurance record for one forward execution.
:class:`ForwardModelSpec`
    Store the identity of a differentiable forward model.
:class:`FullDensityOfStates`
    Store spin-resolved total and projected DOS data in a JAX PyTree.
:class:`HamiltonianBlocks`
    Store normalized Hamiltonian matrices with exact block metadata.
:class:`HamiltonianSource`
    Define the ``HamiltonianSource`` public contract.
:class:`HamiltonianOverlapSource`
    Define the ``HamiltonianOverlapSource`` public contract.
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
:class:`IntrinsicPhotocurrent`
    Define the ``IntrinsicPhotocurrent`` public contract.
:class:`InMemoryPlaneWaveSource`
    Define the ``InMemoryPlaneWaveSource`` public contract.
:class:`KGrid`
    Store a fixed-shape raster in fractional k-space.
:class:`KPath`
    Store a generated path through fractional k-space.
:class:`KPathInfo`
    Store k-point path metadata in a JAX PyTree.
:class:`KSScatteringBatch`
    Define the ``KSScatteringBatch`` public contract.
:class:`KSScatteringBoundaryProfile`
    Define the ``KSScatteringBoundaryProfile`` public contract.
:class:`KSScatteringProblem`
    Define the ``KSScatteringProblem`` public contract.
:class:`KSScatteringRequest`
    Define the ``KSScatteringRequest`` public contract.
:class:`KSScatteringSolverSpec`
    Define the ``KSScatteringSolverSpec`` public contract.
:class:`LightMatterCouplingSpec`
    Define the ``LightMatterCouplingSpec`` public contract.
:class:`MatrixElementParams`
    Store shell-shared matrix-element scales and channel phases.
:class:`MeasurementCoordinates`
    Define the ``MeasurementCoordinates`` public contract.
:class:`OverlapSource`
    Define the ``OverlapSource`` public contract.
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:class:`OrbitalProjection`
    Store orbital-resolved band projections in a JAX PyTree.
:class:`PhotonBeam`
    Define the ``PhotonBeam`` public contract.
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
:obj:`PyTreeDef`
    Runtime pytree definition with a typed static-analysis stand-in.
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
:class:`SOCVolumetricData`
    Store SOC CHGCAR volumetric-grid data in a JAX PyTree.
:class:`SamplePose`
    Define the ``SamplePose`` public contract.
:class:`SampleState`
    Define the ``SampleState`` public contract.
:class:`SelfEnergyModel`
    Store a causal self-energy parameterization as a JAX PyTree.
:class:`ShardSpec`
    Define the ``ShardSpec`` public contract.
:class:`RetardedGreenFunctionSource`
    Define the ``RetardedGreenFunctionSource`` public contract.
:class:`RetardedGreenBatch`
    Define the ``RetardedGreenBatch`` public contract.
:obj:`RetardedSelfEnergySource`
    Union of parametric and tabulated retarded self-energy sources.
:class:`RetardedValidationReport`
    Define the ``RetardedValidationReport`` public contract.
:class:`SelfEnergyBatch`
    Define the ``SelfEnergyBatch`` public contract.
:class:`SimulationResult`
    Define the ``SimulationResult`` public contract.
:class:`SpectralEvaluationRequest`
    Define the ``SpectralEvaluationRequest`` public contract.
:class:`StateBatchRequest`
    Define the ``StateBatchRequest`` public contract.
:class:`VaspWavefunctionSource`
    Define the ``VaspWavefunctionSource`` public contract.
:class:`WavecarDataset`
    Define the ``WavecarDataset`` public contract.
:class:`WavecarHeader`
    Define the ``WavecarHeader`` public contract.
:class:`ParametricSelfEnergy`
    Define the ``ParametricSelfEnergy`` public contract.
:class:`PlaneWaveBatch`
    Define the ``PlaneWaveBatch`` public contract.
:class:`PlaneWaveStateSource`
    Define the ``PlaneWaveStateSource`` public contract.
:class:`TabulatedMatrixSelfEnergy`
    Define the ``TabulatedMatrixSelfEnergy`` public contract.
:class:`TabulatedRetardedGreenFunctionSource`
    Define the ``TabulatedRetardedGreenFunctionSource`` public contract.
:class:`SensitivityMap`
    Store scaled sensitivities from inputs to output projections.
:class:`SlabSpec`
    Store static slab construction choices and provenance.
:class:`SlabTopology`
    Store host-selected discrete slab topology for pure-JAX rebuilding.
:class:`SlaterKosterParams`
    Store differentiable Slater--Koster two-center integrals.
:obj:`SliceOperator`
    Union of dense and sparse finite-slice operators.
:class:`SpinBandStructure`
    Store spin-resolved electronic band-structure data in a JAX PyTree.
:class:`SpinOrbitalProjection`
    Store orbital projections with spin data in a JAX PyTree.
:class:`SurfaceCell`
    Store a validated Cartesian surface-cell frame.
:class:`SparseSliceOperator`
    Define the ``SparseSliceOperator`` public contract.
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
:class:`TightBindingStateSource`
    Define the ``TightBindingStateSource`` public contract.
:class:`WavefunctionSource`
    Define the ``WavefunctionSource`` public contract.
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
:class:`VacuumBoundarySpec`
    Define the ``VacuumBoundarySpec`` public contract.
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
:func:`make_acquisition`
    Compute the ``make_acquisition`` public contract.
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
:func:`make_dyson_spectral_source`
    Compute the ``make_dyson_spectral_source`` public contract.
:func:`make_derivative_capability`
    Compute the ``make_derivative_capability`` public contract.
:func:`make_fidelity_manifest`
    Compute the ``make_fidelity_manifest`` public contract.
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
:func:`make_experiment`
    Compute the ``make_experiment`` public contract.
:func:`make_experiment_geometry`
    Create a validated geometry for an ARPES experiment.
:func:`make_ks_scattering_solver_spec`
    Compute the ``make_ks_scattering_solver_spec`` public contract.
:func:`make_backing_absorber_spec`
    Compute the ``make_backing_absorber_spec`` public contract.
:func:`make_dense_slice_operator`
    Compute the ``make_dense_slice_operator`` public contract.
:func:`make_ks_scattering_batch`
    Compute the ``make_ks_scattering_batch`` public contract.
:func:`make_ks_scattering_boundary_profile`
    Compute the ``make_ks_scattering_boundary_profile`` public contract.
:func:`make_ks_scattering_problem`
    Compute the ``make_ks_scattering_problem`` public contract.
:func:`make_ks_scattering_request`
    Compute the ``make_ks_scattering_request`` public contract.
:func:`make_light_matter_coupling_spec`
    Compute the ``make_light_matter_coupling_spec`` public contract.
:func:`make_sparse_slice_operator`
    Compute the ``make_sparse_slice_operator`` public contract.
:func:`make_electronic_state_archive`
    Compute the ``make_electronic_state_archive`` public contract.
:func:`make_final_state_spec`
    Create a validated radial final-state selection.
:func:`make_factorized_arpes_model`
    Compute the ``make_factorized_arpes_model`` public contract.
:func:`make_forward_certificate`
    Create and cross-validate a complete forward certificate.
:func:`make_forward_model_spec`
    Create a validated stable forward-model specification.
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.
:func:`make_hamiltonian_blocks`
    Create normalized Hamiltonian blocks without changing parsed values.
:func:`make_measurement_coordinates`
    Compute the ``make_measurement_coordinates`` public contract.
:func:`make_parametric_self_energy`
    Compute the ``make_parametric_self_energy`` public contract.
:func:`make_plane_wave_batch`
    Compute the ``make_plane_wave_batch`` public contract.
:func:`make_in_memory_plane_wave_source`
    Compute the ``make_in_memory_plane_wave_source`` public contract.
:func:`make_retarded_green_batch`
    Compute the ``make_retarded_green_batch`` public contract.
:func:`make_retarded_validation_report`
    Compute the ``make_retarded_validation_report`` public contract.
:func:`make_self_energy_batch`
    Compute the ``make_self_energy_batch`` public contract.
:func:`make_spectral_evaluation_request`
    Compute the ``make_spectral_evaluation_request`` public contract.
:func:`make_tabulated_matrix_self_energy`
    Compute the ``make_tabulated_matrix_self_energy`` public contract.
:func:`make_tabulated_retarded_green_function_source`
    Create a tabulated retarded Green-function source.
:func:`make_state_batch_request`
    Compute the ``make_state_batch_request`` public contract.
:func:`make_vasp_wavefunction_source`
    Compute the ``make_vasp_wavefunction_source`` public contract.
:func:`make_wavecar_dataset`
    Compute the ``make_wavecar_dataset`` public contract.
:func:`make_wavecar_header`
    Compute the ``make_wavecar_header`` public contract.
:func:`make_vacuum_boundary_spec`
    Compute the ``make_vacuum_boundary_spec`` public contract.
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
:func:`make_intrinsic_photocurrent`
    Compute the ``make_intrinsic_photocurrent`` public contract.
:func:`make_kgrid`
    Create a validated fixed-shape k-space raster.
:func:`make_kpath`
    Create a validated path through fractional k-space.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
:func:`make_matrix_element_params`
    Create validated shell-shared matrix-element parameters.
:func:`make_photon_beam`
    Compute the ``make_photon_beam`` public contract.
:func:`make_sample_pose`
    Compute the ``make_sample_pose`` public contract.
:func:`make_sample_state`
    Compute the ``make_sample_state`` public contract.
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
:func:`make_shard_spec`
    Compute the ``make_shard_spec`` public contract.
:func:`make_sensitivity_map`
    Create a named, scaled local-sensitivity map.
:func:`make_simulation_result`
    Compute the ``make_simulation_result`` public contract.
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
:func:`make_tight_binding_state_source`
    Compute the ``make_tight_binding_state_source`` public contract.
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
:obj:`CheckFunction`
    Callable signature for a pure JAX certification check.
:obj:`DosType`
    Supported density-of-states containers.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:obj:`ProjectionType`
    Supported orbital-projection containers.
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
"""

from .aliases import (
    NonJaxNumber,
    PyTreeDef,
    RetardedSelfEnergySource,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    SliceOperator,
)
from .arpes import (
    ArpesCube,
    ArpesSpectrum,
    constant_energy_map,
    fermi_surface_map,
    make_arpes_cube,
    make_arpes_spectrum,
    slice_edc,
    slice_mdc,
)
from .bands import (
    BandStructure,
    OrbitalProjection,
    SpinBandStructure,
    SpinOrbitalProjection,
    make_band_structure,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
)
from .certification import (
    CertificationContext,
    CertifiedResult,
    ExecutionManifest,
    ForwardCertificate,
    make_certification_context,
    make_certified_result,
    make_execution_manifest,
    make_forward_certificate,
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
from .coordinates import MeasurementCoordinates, make_measurement_coordinates
from .derivatives import (
    DependencyMap,
    DerivativeEvidence,
    InformationSpectrum,
    SensitivityMap,
    make_dependency_map,
    make_derivative_evidence,
    make_information_spectrum,
    make_sensitivity_map,
)
from .detector_data import (
    DetectorCalibration,
    DetectorRaster,
    make_detector_calibration,
    make_detector_raster,
)
from .detector_effects import DetectorEffects, make_detector_effects
from .diagonalized_bands import DiagonalizedBands, make_diagonalized_bands
from .dos import (
    DensityOfStates,
    FullDensityOfStates,
    make_density_of_states,
    make_full_density_of_states,
)
from .electronic_state import (
    EigensystemSource,
    ElectronicStateArchive,
    ElectronicStateSource,
    HamiltonianOverlapSource,
    HamiltonianSource,
    OverlapSource,
    RetardedGreenFunctionSource,
    TightBindingStateSource,
    WavefunctionSource,
    make_electronic_state_archive,
    make_tight_binding_state_source,
)
from .evidence import (
    CertificationClaim,
    EvidenceLineage,
    EvidenceRef,
    HumanAttestationRef,
    TransformationRecord,
    make_certification_claim,
    make_evidence_lineage,
    make_evidence_ref,
    make_human_attestation_ref,
    make_transformation_record,
)
from .experiment import ExperimentGeometry, make_experiment_geometry
from .experiment_state import (
    Acquisition,
    Experiment,
    PhotonBeam,
    SamplePose,
    SampleState,
    make_acquisition,
    make_experiment,
    make_photon_beam,
    make_sample_pose,
    make_sample_state,
)
from .fidelity import (
    DerivativeCapability,
    FidelityManifest,
    make_derivative_capability,
    make_fidelity_manifest,
)
from .generalized_spectral import (
    DysonSpectralSource,
    ParametricSelfEnergy,
    RetardedGreenBatch,
    SelfEnergyBatch,
    SpectralEvaluationRequest,
    TabulatedMatrixSelfEnergy,
    TabulatedRetardedGreenFunctionSource,
    make_dyson_spectral_source,
    make_parametric_self_energy,
    make_retarded_green_batch,
    make_self_energy_batch,
    make_spectral_evaluation_request,
    make_tabulated_matrix_self_energy,
    make_tabulated_retarded_green_function_source,
)
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
from .ks_scattering import (
    BackingAbsorberSpec,
    DenseSliceOperator,
    KSScatteringBoundaryProfile,
    KSScatteringProblem,
    KSScatteringRequest,
    LightMatterCouplingSpec,
    SparseSliceOperator,
    VacuumBoundarySpec,
    make_backing_absorber_spec,
    make_dense_slice_operator,
    make_ks_scattering_boundary_profile,
    make_ks_scattering_problem,
    make_ks_scattering_request,
    make_light_matter_coupling_spec,
    make_sparse_slice_operator,
    make_vacuum_boundary_spec,
)
from .ks_scattering_solution import (
    KSScatteringBatch,
    KSScatteringSolverSpec,
    make_ks_scattering_batch,
    make_ks_scattering_solver_spec,
)
from .orbital_basis import OrbitalBasis, make_orbital_basis
from .photocurrent import FactorizedArpesModel, make_factorized_arpes_model
from .plane_wave import (
    InMemoryPlaneWaveSource,
    PlaneWaveBatch,
    PlaneWaveStateSource,
    StateBatchRequest,
    VaspWavefunctionSource,
    WavecarDataset,
    WavecarHeader,
    make_in_memory_plane_wave_source,
    make_plane_wave_batch,
    make_state_batch_request,
    make_vasp_wavefunction_source,
    make_wavecar_dataset,
    make_wavecar_header,
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
    MatrixElementParams,
    RadialSpec,
    make_matrix_element_params,
    make_radial_spec,
)
from .radial_profiles import (
    FinalStateSpec,
    RadialQuadratureSpec,
    make_final_state_spec,
    make_radial_quadrature_spec,
)
from .registry import (
    HandshakeReport,
    RegisteredModel,
    RegisteredTransformation,
    RegistrationHandshake,
    RegistryReport,
    RegistrySnapshot,
    make_handshake_report,
    make_registered_model,
    make_registered_transformation,
    make_registration_handshake,
    make_registry_report,
    make_registry_snapshot,
)
from .reports import (
    EvidenceReport,
    PolicyReport,
    ReproductionReport,
    VerificationReport,
    WaiverRecord,
    WaiverReport,
    make_evidence_report,
    make_policy_report,
    make_reproduction_report,
    make_verification_report,
    make_waiver_record,
    make_waiver_report,
)
from .result import (
    IntrinsicPhotocurrent,
    SimulationResult,
    make_intrinsic_photocurrent,
    make_simulation_result,
)
from .retarded_validation import (
    RetardedValidationReport,
    make_retarded_validation_report,
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
from .sharding import ShardSpec, make_shard_spec
from .slab_geometry import (
    SlabSpec,
    SurfaceCell,
    make_slab_spec,
    make_surface_cell,
)
from .slab_topology import SlabTopology, make_slab_topology
from .slater_koster_params import (
    SlaterKosterParams,
    make_slater_koster_params,
)
from .specification import (
    ArtifactRef,
    ArtifactResolver,
    CheckFunction,
    ConventionRef,
    DomainPredicate,
    DomainResult,
    ForwardModelSpec,
    make_artifact_ref,
    make_convention_ref,
    make_domain_predicate,
    make_domain_result,
    make_forward_model_spec,
)
from .spectral import (
    Power2TailSpec,
    TransitionSourceSchedule,
    make_power2_tail_spec,
    make_transition_source_schedule,
)
from .tb_model import (
    TBModel,
    make_tb_model,
)
from .volumetric import (
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)
from .wannier import (
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
    "Acquisition",
    "ArpesCube",
    "ArpesSpectrum",
    "ArtifactRef",
    "BackingAbsorberSpec",
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
    "DerivativeCapability",
    "DysonSpectralSource",
    "DependencyAnalysisCache",
    "DependencyMap",
    "DerivativeEvidence",
    "DenseSliceOperator",
    "DetectorCalibration",
    "DetectorEffects",
    "DetectorRaster",
    "DiagonalizedBands",
    "DomainPredicate",
    "DomainResult",
    "EvidenceLineage",
    "Experiment",
    "EvidenceRef",
    "EvidenceReport",
    "ExecutionManifest",
    "ExperimentGeometry",
    "ElectronicStateSource",
    "ElectronicStateArchive",
    "EigensystemSource",
    "FidelityManifest",
    "FinalStateSpec",
    "FactorizedArpesModel",
    "ForwardCertificate",
    "ForwardModelSpec",
    "FullDensityOfStates",
    "HamiltonianBlocks",
    "HamiltonianSource",
    "HamiltonianOverlapSource",
    "HandshakeReport",
    "HoppingRecord",
    "HumanAttestationRef",
    "InformationSpectrum",
    "InformationState",
    "IntrinsicPhotocurrent",
    "InMemoryPlaneWaveSource",
    "KGrid",
    "KPath",
    "KPathInfo",
    "KSScatteringBatch",
    "KSScatteringBoundaryProfile",
    "KSScatteringProblem",
    "KSScatteringRequest",
    "KSScatteringSolverSpec",
    "LightMatterCouplingSpec",
    "MatrixElementParams",
    "MeasurementCoordinates",
    "OverlapSource",
    "OrbitalBasis",
    "OrbitalProjection",
    "PhotonBeam",
    "PolicyReport",
    "Power2TailSpec",
    "ProvenanceAnalysis",
    "ProvenanceGraph",
    "ProvenanceReport",
    "PyTreeDef",
    "RadialQuadratureSpec",
    "RadialSpec",
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistrationHandshake",
    "RegistryReport",
    "RegistrySnapshot",
    "ReproductionReport",
    "SOCVolumetricData",
    "SamplePose",
    "SampleState",
    "SelfEnergyModel",
    "ShardSpec",
    "RetardedGreenFunctionSource",
    "RetardedGreenBatch",
    "RetardedSelfEnergySource",
    "RetardedValidationReport",
    "SelfEnergyBatch",
    "SimulationResult",
    "SpectralEvaluationRequest",
    "StateBatchRequest",
    "VaspWavefunctionSource",
    "WavecarDataset",
    "WavecarHeader",
    "ParametricSelfEnergy",
    "PlaneWaveBatch",
    "PlaneWaveStateSource",
    "TabulatedMatrixSelfEnergy",
    "TabulatedRetardedGreenFunctionSource",
    "SensitivityMap",
    "SlabSpec",
    "SlabTopology",
    "SlaterKosterParams",
    "SliceOperator",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "SurfaceCell",
    "SparseSliceOperator",
    "TBModel",
    "TightBindingStateSource",
    "WavefunctionSource",
    "TextLineCursor",
    "TransformationContract",
    "TransformationRecord",
    "TransitionSourceSchedule",
    "VerificationReport",
    "VolumetricData",
    "VacuumBoundarySpec",
    "WaiverRecord",
    "WaiverReport",
    "WannierOperatorData",
    "WorkflowContext",
    "constant_energy_map",
    "fermi_surface_map",
    "make_acquisition",
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
    "make_dyson_spectral_source",
    "make_derivative_capability",
    "make_fidelity_manifest",
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
    "make_experiment",
    "make_experiment_geometry",
    "make_ks_scattering_solver_spec",
    "make_backing_absorber_spec",
    "make_dense_slice_operator",
    "make_ks_scattering_batch",
    "make_ks_scattering_boundary_profile",
    "make_ks_scattering_problem",
    "make_ks_scattering_request",
    "make_light_matter_coupling_spec",
    "make_sparse_slice_operator",
    "make_electronic_state_archive",
    "make_final_state_spec",
    "make_factorized_arpes_model",
    "make_forward_certificate",
    "make_forward_model_spec",
    "make_full_density_of_states",
    "make_hamiltonian_blocks",
    "make_measurement_coordinates",
    "make_parametric_self_energy",
    "make_plane_wave_batch",
    "make_in_memory_plane_wave_source",
    "make_retarded_green_batch",
    "make_retarded_validation_report",
    "make_self_energy_batch",
    "make_spectral_evaluation_request",
    "make_tabulated_matrix_self_energy",
    "make_tabulated_retarded_green_function_source",
    "make_state_batch_request",
    "make_vasp_wavefunction_source",
    "make_wavecar_dataset",
    "make_wavecar_header",
    "make_vacuum_boundary_spec",
    "make_handshake_report",
    "make_hopping_record",
    "make_human_attestation_ref",
    "make_information_spectrum",
    "make_information_state",
    "make_intrinsic_photocurrent",
    "make_kgrid",
    "make_kpath",
    "make_kpath_info",
    "make_matrix_element_params",
    "make_photon_beam",
    "make_sample_pose",
    "make_sample_state",
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
    "make_shard_spec",
    "make_sensitivity_map",
    "make_simulation_result",
    "make_slab_spec",
    "make_slab_topology",
    "make_slater_koster_params",
    "make_soc_volumetric_data",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
    "make_surface_cell",
    "make_tb_model",
    "make_tight_binding_state_source",
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
    "CheckFunction",
    "DosType",
    "NonJaxNumber",
    "ProjectionType",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
]
