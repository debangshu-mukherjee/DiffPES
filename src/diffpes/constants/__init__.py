"""Expose the declarative constants of diffpes.

Extended Summary
----------------
This package provides one import surface for immutable physical values,
selector vocabularies, schema identifiers, parser tokens, and validation
tolerances. Source modules import constants from :mod:`diffpes.constants`.

The following submodules organize the constants:

- :mod:`carriers`
    Define validation constants for diffpes carrier factories.
- :mod:`certification`
    Define static identifiers and schema constants for forward certification.
- :mod:`numerical`
    Define frozen generated coefficient tables for numerical kernels.
- :mod:`shared`
    Define shared physical, numerical, orbital, and parser constants.
- :mod:`wannier`
    Define constants for Wannier90 and Cartesian hopping-list formats.

Routine Listings
----------------
:obj:`ACQUISITION_MODES`
    Acquisition modes accepted by detector-effects carriers.
:obj:`ARRAY_MATRIX_NDIM`
    Expected dimensionality of radial parameter matrices.
:obj:`ATTR_AUX`
    HDF5 attribute name storing auxiliary PyTree data as JSON.
:obj:`ATTR_NONE`
    HDF5 attribute name listing PyTree fields stored as None.
:obj:`ATTR_TYPE`
    HDF5 attribute name storing the PyTree type name.
:obj:`BACKGROUND_MODES`
    Background modes accepted by detector-effects carriers.
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
:obj:`CERTIFIED_RADIAL_PROFILES`
    Registered immutable radial-quadrature profile specifications.
:obj:`CERTIFIED_R_MAX_BOHR`
    Certified maximum radial coordinate in Bohr.
:obj:`CERTIFIED_TAIL_ENVELOPE_ID`
    Stable identifier of the certified radial tail envelope.
:obj:`CHANNELS_BY_PAIR`
    Slater--Koster channels supported for each angular-momentum pair.
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
:obj:`COORDINATE_DENSITY`
    Detector coordinate-density convention.
:obj:`COORDINATE_MODE_TOKENS`
    Recognized KPOINTS coordinate-mode tokens.
:obj:`DEGENERACY_GROUP_TOL_EV`
    Maximum group-to-complement gap treated as a cut degeneracy.
:obj:`DEPTH_TOLERANCE_ANG`
    Nonnegative orbital-depth tolerance in Angstrom.
:obj:`DERIVATIVE_CAPABILITY_MODES`
    Derivative-boundary modes accepted by fidelity manifests.
:obj:`DETECTOR_BOUNDARY_POLICY`
    Detector boundary-loss convention.
:obj:`DETECTOR_COORDINATE_SYSTEM`
    Detector coordinate-system identifier.
:obj:`D_ORBITAL_SLICE`
    Slice selecting the five d orbitals.
:obj:`EIGENVALUE_NDIM`
    Expected dimensionality of tight-binding eigenvalues.
:obj:`EIGENVECTOR_NDIM`
    Expected dimensionality of tight-binding eigenvectors.
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
:obj:`FADDEEVA_WEIDEMAN_COEFFICIENTS`
    Frozen order-40 Weideman rational coefficients.
:obj:`FINAL_STATE_MODES`
    Final-state modes accepted by radial carriers.
:obj:`FLOAT_TOKEN_RE`
    Compiled regex matching floating-point tokens.
:obj:`GAUNT_IMAG_TOL`
    Tolerance for discarding imaginary Gaunt residues.
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:obj:`GROUP_COMPLEMENT_GAP_MIN_EV`
    Minimum spectral isolation required for a registered band group.
:obj:`G_PARALLEL_ATOL_INV_ANG`
    Surface parallel-momentum conservation tolerance in inverse Angstrom.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`HBAR_SQ_OVER_2ME_EV_ANG2`
    Store the free-electron dispersion constant in eV Angstrom squared.
:obj:`HERMITE_TABLE_POINTS`
    Supported sizes of certified Hermite tables.
:obj:`HERMITICITY_RELATIVE_TOLERANCE`
    Relative tolerance for eager matrix-Hermiticity validation.
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
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:obj:`KNOWN_CHANNELS`
    Complete set of supported Slater--Koster channel names.
:obj:`KPATH_AUX_WITH_COMMENT_LEN`
    KPathInfo auxiliary-data length including a comment.
:obj:`KPATH_AUX_WITH_COORD_MODE_LEN`
    KPathInfo auxiliary-data length including a coordinate mode.
:obj:`KPATH_MODES`
    K-point path modes accepted by VASP metadata carriers.
:obj:`KPOINT_LINE_VALUES`
    Tokens on an EIGENVAL k-point line.
:obj:`K_PREFACTOR_INV_ANG_SQRT_EV`
    Store the momentum prefactor in inverse Angstrom per square-root eV.
:obj:`LATTICE_ROWS`
    Number of lattice-vector rows in POSCAR/CHGCAR headers.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`MATRIX_NDIM`
    Expected dimensionality of tight-binding operator matrices.
:obj:`MAX_COEFFICIENT_CONDITION`
    Maximum certified radial coefficient condition number.
:obj:`MAX_DECAY_PARAMETER`
    Maximum certified radial decay parameter.
:obj:`MAX_EFFECTIVE_PRINCIPAL`
    Maximum certified effective principal quantum number.
:obj:`MAX_HYDROGENIC_PRINCIPAL`
    Maximum supported hydrogenic principal quantum number.
:obj:`MAX_LATTICE_CONDITION_NUMBER`
    Maximum accepted lattice condition number.
:obj:`MAX_MATRIXEL_L`
    Maximum angular momentum accepted by matrix-element carriers.
:obj:`MAX_SK_ANGULAR_MOMENTUM`
    Maximum angular momentum supported by Slater--Koster construction.
:obj:`ME_EV`
    Electron rest energy in eV.
:obj:`MINIMUM_AXIS_POINTS`
    Minimum number of points accepted on a sampled DOS energy axis.
:obj:`MIN_BOND_DISTANCE`
    Minimum nonzero distance accepted by neighbor discovery.
:obj:`MIN_COMPACT_GRID_POINTS`
    Minimum number of points in a compact radial grid.
:obj:`MIN_DECAY_PARAMETER`
    Minimum certified radial decay parameter.
:obj:`MIN_GRID_NODES`
    Minimum node count for a gridded self-energy.
:obj:`MIN_INTERPOLATION_AXIS_POINTS`
    Minimum point count on a carrier interpolation axis.
:obj:`MIN_SCALED_SINGULAR_VALUE`
    Minimum accepted scaled lattice singular value.
:obj:`MIN_SUM`
    Minimum-sum floor guarding normalizations.
:obj:`M_D`
    Magnetic quantum numbers of the d orbitals.
:obj:`M_P`
    Magnetic quantum numbers of the p orbitals.
:obj:`NONSPIN_COLS`
    DOSCAR column count without spin polarization.
:obj:`NON_S_ORBITAL_SLICE`
    Slice selecting all non-s orbitals.
:obj:`NORM_EPS`
    Epsilon floor guarding eigenvector normalization.
:obj:`N_ORBITALS`
    Number of orbitals in the VASP projection basis.
:obj:`N_SOC_MAG_BLOCKS`
    Magnetization block count in SOC CHGCAR files.
:obj:`N_SPIN_COMPONENTS`
    Spin-projection component count in PROCAR.
:obj:`ORBITAL_INDEX`
    Mapping from orbital name to VASP orbital index.
:obj:`ORBITAL_POSITION_NDIM`
    Expected dimensionality of orbital-position arrays.
:obj:`PARAMETER_KEY_PARTS`
    Number of colon-delimited parts in a qualified Slater--Koster key.
:obj:`PATH_STEP_ATOL_INV_ANG`
    Absolute path-step tolerance in inverse Angstrom.
:obj:`PATH_STEP_RTOL`
    Relative path-step tolerance.
:obj:`PHASE_LOSS_MESSAGE`
    Warning text for PROCAR magnitude-only eigenvectors.
:obj:`POST_COUNT_MODES`
    Post-count modes accepted by detector-effects carriers.
:obj:`PRESET_NAMES`
    Recognized band-scatter plotting preset names.
:obj:`P_ORBITAL_SLICE`
    Slice selecting the three p orbitals.
:obj:`RADIAL_ACCELERATORS`
    Radial accelerator modes accepted by radial carriers.
:obj:`RADIAL_MODES`
    Radial-wavefunction modes accepted by radial carriers.
:obj:`REGISTERED_DOMAIN_FRAME_IDS`
    Registered frame identifiers for detector domains.
:obj:`ROTATION_ORTHOGONALITY_TOLERANCE`
    Orthogonality tolerance for surface rotations.
:obj:`SAMPLE_CARTESIAN_FRAME_ID`
    Stable identifier of the sample Cartesian frame.
:obj:`SCALAR_LINE_COMPONENTS`
    Tokens on a scalar CHGCAR header line.
:obj:`SELF_ENERGY_MODES`
    Model modes accepted by self-energy carriers.
:obj:`SHARD_CHECKPOINT_POLICIES`
    Rematerialization policies accepted by static sharding carriers.
:obj:`SENSITIVITY_MODES`
    Sensitivity modes accepted by detector-effects carriers.
:obj:`SHELL_ATOLERANCE`
    Absolute tolerance for grouping equal-distance neighbor shells.
:obj:`SHELL_RTOLERANCE`
    Relative tolerance for grouping equal-distance neighbor shells.
:obj:`SLIT_ORIENTATIONS`
    Detector slit orientations accepted by experiment geometry.
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
:obj:`SURFACE_VECTOR_COUNT`
    Number of in-plane vectors in a surface-cell basis.
:obj:`S_IDX`
    Index of the s orbital.
:obj:`TAIL_COORDINATE_BOUND`
    Absolute bound for self-energy tail coordinates.
:obj:`TB_HERMITICITY_TOLERANCE`
    Hermiticity tolerance for tight-binding carriers.
:obj:`TB_PAIR_LENGTH`
    Number of indices in a tight-binding orbital pair.
:obj:`TB_RADIAL_INPUT_COUNT`
    Number of positional inputs accepted by the radial ARPES model adapter.
:obj:`TB_RADIAL_MODEL_ID`
    Permanent identifier of the radial ARPES forward model.
:obj:`TB_RADIAL_MODEL_VERSION`
    Semantic version of the radial ARPES forward model.
:obj:`TRANSVERSALITY_ATOL`
    Absolute transversality tolerance for polarization vectors.
:obj:`TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2`
    Store the inverse free-electron dispersion constant.
:obj:`WANNIER_CELL_FIELDS`
    Number of integer components in a Wannier translation.
:obj:`WANNIER_CENTRE_CONSISTENCY_TOLERANCE`
    Cartesian tolerance for centres assigned to one atom.
:obj:`WANNIER_CENTRE_NDIM`
    Expected dimensionality of Wannier centre arrays.
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
:obj:`WANNIER_POSITION_NDIM`
    Expected dimensionality of Wannier position-operator arrays.
:obj:`WANNIER_SOURCE_FORMATS`
    Source formats accepted by Wannier operator carriers.
:obj:`WANNIER_SPIN_LAYOUTS`
    Spin layouts accepted by Wannier operator carriers.
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
"""

from .carriers import (
    ACQUISITION_MODES,
    ARRAY_MATRIX_NDIM,
    BACKGROUND_MODES,
    CERTIFIED_R_MAX_BOHR,
    CERTIFIED_RADIAL_PROFILES,
    CERTIFIED_TAIL_ENVELOPE_ID,
    COORDINATE_DENSITY,
    DEPTH_TOLERANCE_ANG,
    DERIVATIVE_CAPABILITY_MODES,
    DETECTOR_BOUNDARY_POLICY,
    DETECTOR_COORDINATE_SYSTEM,
    EIGENVALUE_NDIM,
    EIGENVECTOR_NDIM,
    FINAL_STATE_MODES,
    HERMITE_TABLE_POINTS,
    HERMITICITY_RELATIVE_TOLERANCE,
    KPATH_MODES,
    MAX_COEFFICIENT_CONDITION,
    MAX_DECAY_PARAMETER,
    MAX_EFFECTIVE_PRINCIPAL,
    MAX_HYDROGENIC_PRINCIPAL,
    MAX_LATTICE_CONDITION_NUMBER,
    MAX_MATRIXEL_L,
    MIN_COMPACT_GRID_POINTS,
    MIN_DECAY_PARAMETER,
    MIN_GRID_NODES,
    MIN_INTERPOLATION_AXIS_POINTS,
    MIN_SCALED_SINGULAR_VALUE,
    ORBITAL_POSITION_NDIM,
    PATH_STEP_ATOL_INV_ANG,
    PATH_STEP_RTOL,
    POST_COUNT_MODES,
    RADIAL_ACCELERATORS,
    RADIAL_MODES,
    REGISTERED_DOMAIN_FRAME_IDS,
    ROTATION_ORTHOGONALITY_TOLERANCE,
    SAMPLE_CARTESIAN_FRAME_ID,
    SELF_ENERGY_MODES,
    SENSITIVITY_MODES,
    SHARD_CHECKPOINT_POLICIES,
    SLIT_ORIENTATIONS,
    SURFACE_VECTOR_COUNT,
    TAIL_COORDINATE_BOUND,
    TB_HERMITICITY_TOLERANCE,
    TB_PAIR_LENGTH,
    TRANSVERSALITY_ATOL,
)
from .certification import (
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
from .numerical import FADDEEVA_WEIDEMAN_COEFFICIENTS, GAUNT_TABLE
from .shared import (
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
from .wannier import (
    HOPPING_LIST_COMPLEX_FIELDS,
    HOPPING_LIST_REAL_FIELDS,
    WANNIER_CELL_FIELDS,
    WANNIER_CENTRE_CONSISTENCY_TOLERANCE,
    WANNIER_CENTRE_NDIM,
    WANNIER_DEGENERACIES_PER_LINE,
    WANNIER_HERMITICITY_TOLERANCE,
    WANNIER_HR_HAMILTONIAN_FIELDS,
    WANNIER_HR_SUFFIX,
    WANNIER_INTEGER_RECOVERY_TOLERANCE,
    WANNIER_POSITION_NDIM,
    WANNIER_SOURCE_FORMATS,
    WANNIER_SPIN_LAYOUTS,
    WANNIER_TB_HAMILTONIAN_FIELDS,
    WANNIER_TB_POSITION_FIELDS,
    WANNIER_TB_SUFFIX,
)

__all__: list[str] = [
    "ACQUISITION_MODES",
    "ARRAY_MATRIX_NDIM",
    "ATTR_AUX",
    "ATTR_NONE",
    "ATTR_TYPE",
    "BACKGROUND_MODES",
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
    "CERTIFIED_RADIAL_PROFILES",
    "CERTIFIED_R_MAX_BOHR",
    "CERTIFIED_TAIL_ENVELOPE_ID",
    "CHANNELS_BY_PAIR",
    "CHECKSUM_ALGORITHM",
    "CHECKSUM_FILE_CHUNK_BYTES",
    "CHECKSUM_FORMAT_VERSION",
    "CHECKSUM_PATTERN",
    "CHECKSUM_RECORD_KIND_PATTERN",
    "COORDINATE_DENSITY",
    "COORDINATE_MODE_TOKENS",
    "DEGENERACY_GROUP_TOL_EV",
    "DEPTH_TOLERANCE_ANG",
    "DERIVATIVE_CAPABILITY_MODES",
    "DETECTOR_BOUNDARY_POLICY",
    "DETECTOR_COORDINATE_SYSTEM",
    "D_ORBITAL_SLICE",
    "EIGENVALUE_NDIM",
    "EIGENVECTOR_NDIM",
    "EIG_DOWN_INDEX",
    "EIG_UP_INDEX",
    "ENERGY_AXIS_NDIM",
    "EPS",
    "EPS_DEG",
    "FADDEEVA_WEIDEMAN_COEFFICIENTS",
    "FINAL_STATE_MODES",
    "FLOAT_TOKEN_RE",
    "GAUNT_IMAG_TOL",
    "GAUNT_TABLE",
    "GROUP_COMPLEMENT_GAP_MIN_EV",
    "G_PARALLEL_ATOL_INV_ANG",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "HBAR_SQ_OVER_2ME_EV_ANG2",
    "HERMITE_TABLE_POINTS",
    "HERMITICITY_RELATIVE_TOLERANCE",
    "HOPPING_LIST_COMPLEX_FIELDS",
    "HOPPING_LIST_REAL_FIELDS",
    "INTENSITY_NDIM",
    "ISPIN2_BLOCKS",
    "ISPIN_SPIN_POLARIZED",
    "KB_EV_PER_K",
    "KNOWN_CHANNELS",
    "KPATH_AUX_WITH_COMMENT_LEN",
    "KPATH_AUX_WITH_COORD_MODE_LEN",
    "KPATH_MODES",
    "KPOINT_LINE_VALUES",
    "K_PREFACTOR_INV_ANG_SQRT_EV",
    "LATTICE_ROWS",
    "L_MAX",
    "MATRIX_NDIM",
    "MAX_COEFFICIENT_CONDITION",
    "MAX_DECAY_PARAMETER",
    "MAX_EFFECTIVE_PRINCIPAL",
    "MAX_HYDROGENIC_PRINCIPAL",
    "MAX_LATTICE_CONDITION_NUMBER",
    "MAX_MATRIXEL_L",
    "MAX_SK_ANGULAR_MOMENTUM",
    "ME_EV",
    "MINIMUM_AXIS_POINTS",
    "MIN_BOND_DISTANCE",
    "MIN_COMPACT_GRID_POINTS",
    "MIN_DECAY_PARAMETER",
    "MIN_GRID_NODES",
    "MIN_INTERPOLATION_AXIS_POINTS",
    "MIN_SCALED_SINGULAR_VALUE",
    "MIN_SUM",
    "M_D",
    "M_P",
    "NONSPIN_COLS",
    "NON_S_ORBITAL_SLICE",
    "NORM_EPS",
    "N_ORBITALS",
    "N_SOC_MAG_BLOCKS",
    "N_SPIN_COMPONENTS",
    "ORBITAL_INDEX",
    "ORBITAL_POSITION_NDIM",
    "PARAMETER_KEY_PARTS",
    "PATH_STEP_ATOL_INV_ANG",
    "PATH_STEP_RTOL",
    "PHASE_LOSS_MESSAGE",
    "POST_COUNT_MODES",
    "PRESET_NAMES",
    "P_ORBITAL_SLICE",
    "RADIAL_ACCELERATORS",
    "RADIAL_MODES",
    "REGISTERED_DOMAIN_FRAME_IDS",
    "ROTATION_ORTHOGONALITY_TOLERANCE",
    "SAMPLE_CARTESIAN_FRAME_ID",
    "SCALAR_LINE_COMPONENTS",
    "SELF_ENERGY_MODES",
    "SHARD_CHECKPOINT_POLICIES",
    "SENSITIVITY_MODES",
    "SHELL_ATOLERANCE",
    "SHELL_RTOLERANCE",
    "SLIT_ORIENTATIONS",
    "SMALL_ARGUMENT",
    "SOC_BLOCKS",
    "SPECIES_PAIR_PARTS",
    "SPECTRUM_NDIM",
    "SPIN_COLS",
    "SURFACE_VECTOR_COUNT",
    "S_IDX",
    "TAIL_COORDINATE_BOUND",
    "TB_HERMITICITY_TOLERANCE",
    "TB_PAIR_LENGTH",
    "TB_RADIAL_INPUT_COUNT",
    "TB_RADIAL_MODEL_ID",
    "TB_RADIAL_MODEL_VERSION",
    "TRANSVERSALITY_ATOL",
    "TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2",
    "WANNIER_CELL_FIELDS",
    "WANNIER_CENTRE_CONSISTENCY_TOLERANCE",
    "WANNIER_CENTRE_NDIM",
    "WANNIER_DEGENERACIES_PER_LINE",
    "WANNIER_HERMITICITY_TOLERANCE",
    "WANNIER_HR_HAMILTONIAN_FIELDS",
    "WANNIER_HR_SUFFIX",
    "WANNIER_INTEGER_RECOVERY_TOLERANCE",
    "WANNIER_POSITION_NDIM",
    "WANNIER_SOURCE_FORMATS",
    "WANNIER_SPIN_LAYOUTS",
    "WANNIER_TB_HAMILTONIAN_FIELDS",
    "WANNIER_TB_POSITION_FIELDS",
    "WANNIER_TB_SUFFIX",
    "WEIGHT_COMPONENT_COUNT",
    "WEIGHT_COMPONENT_INDEX",
    "XYZ_COMPONENTS",
]
