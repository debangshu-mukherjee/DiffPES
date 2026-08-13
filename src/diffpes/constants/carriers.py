"""Define validation constants for diffpes carrier factories.

Extended Summary
----------------
This module owns immutable selectors, dimensions, and tolerances used by
carrier validation. The constants keep each carrier module free of local
declarative values.

Routine Listings
----------------
:obj:`ACQUISITION_MODES`
    Acquisition modes accepted by detector-effects carriers.
:obj:`ARRAY_MATRIX_NDIM`
    Expected dimensionality of radial parameter matrices.
:obj:`BACKGROUND_MODES`
    Background modes accepted by detector-effects carriers.
:obj:`CERTIFIED_RADIAL_PROFILES`
    Registered immutable radial-quadrature profile specifications.
:obj:`CERTIFIED_R_MAX_BOHR`
    Certified maximum radial coordinate in Bohr.
:obj:`CERTIFIED_TAIL_ENVELOPE_ID`
    Stable identifier of the certified radial tail envelope.
:obj:`COORDINATE_DENSITY`
    Detector coordinate-density convention.
:obj:`DEPTH_TOLERANCE_ANG`
    Nonnegative orbital-depth tolerance in Angstrom.
:obj:`DERIVATIVE_CAPABILITY_MODES`
    Derivative-boundary modes accepted by fidelity manifests.
:obj:`DETECTOR_BOUNDARY_POLICY`
    Detector boundary-loss convention.
:obj:`DETECTOR_COORDINATE_SYSTEM`
    Detector coordinate-system identifier.
:obj:`EIGENVALUE_NDIM`
    Expected dimensionality of tight-binding eigenvalues.
:obj:`EIGENVECTOR_NDIM`
    Expected dimensionality of tight-binding eigenvectors.
:obj:`FINAL_STATE_MODES`
    Final-state modes accepted by radial carriers.
:obj:`HERMITE_TABLE_POINTS`
    Supported sizes of certified Hermite tables.
:obj:`HERMITICITY_RELATIVE_TOLERANCE`
    Relative tolerance for eager matrix-Hermiticity validation.
:obj:`KPATH_MODES`
    K-point path modes accepted by VASP metadata carriers.
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
:obj:`ORBITAL_POSITION_NDIM`
    Expected dimensionality of orbital-position arrays.
:obj:`PATH_STEP_ATOL_INV_ANG`
    Absolute path-step tolerance in inverse Angstrom.
:obj:`PATH_STEP_RTOL`
    Relative path-step tolerance.
:obj:`POST_COUNT_MODES`
    Post-count modes accepted by detector-effects carriers.
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
:obj:`SELF_ENERGY_MODES`
    Model modes accepted by self-energy carriers.
:obj:`SHARD_CHECKPOINT_POLICIES`
    Rematerialization policies accepted by static sharding carriers.
:obj:`SENSITIVITY_MODES`
    Sensitivity modes accepted by detector-effects carriers.
:obj:`SLIT_ORIENTATIONS`
    Detector slit orientations accepted by experiment geometry.
:obj:`SURFACE_VECTOR_COUNT`
    Number of in-plane vectors in a surface-cell basis.
:obj:`TAIL_COORDINATE_BOUND`
    Absolute bound for self-energy tail coordinates.
:obj:`TB_HERMITICITY_TOLERANCE`
    Hermiticity tolerance for tight-binding carriers.
:obj:`TB_PAIR_LENGTH`
    Number of indices in a tight-binding orbital pair.
:obj:`TRANSVERSALITY_ATOL`
    Absolute transversality tolerance for polarization vectors.
:obj:`WAVECAR_SECOND_RECORD_VALUES`
    Number of float64 values in the supported WAVECAR metadata record.
:obj:`WAVECAR_SINGLE_PRECISION_TAG`
    VASP WAVECAR tag for complex64 coefficient records.

Notes
-----
A selector change modifies the static carrier contract. A tolerance change
modifies accepted scientific input. Review each change as an API change.
"""

from collections.abc import Mapping
from types import MappingProxyType

from beartype.typing import Final, Tuple

ACQUISITION_MODES: Final[Tuple[str, ...]] = ("poisson", "fixed_total")
ARRAY_MATRIX_NDIM: Final[int] = 2
BACKGROUND_MODES: Final[Tuple[str, ...]] = ("flat", "shirley", "smooth")
CERTIFIED_RADIAL_PROFILES: Final[
    Mapping[
        str,
        Tuple[
            int,
            float,
            float,
            int,
            float,
            float,
            str,
            float,
            float,
            float,
        ],
    ]
] = MappingProxyType(
    {
        "gl1024-r120-k4-l9-v1": (
            1024,
            120.0,
            4.0,
            9,
            1.0e-10,
            1.0e-8,
            "analytic-exp-r120-or-compact-v1",
            32.0,
            0.5,
            4.0,
        ),
        "gl2048-r120-k4-l9-reference-v1": (
            2048,
            120.0,
            4.0,
            9,
            5.0e-11,
            5.0e-9,
            "analytic-exp-r120-or-compact-v1",
            32.0,
            0.5,
            4.0,
        ),
    }
)
CERTIFIED_R_MAX_BOHR: Final[float] = 120.0
CERTIFIED_TAIL_ENVELOPE_ID: Final[str] = "r120-zeta0p5-to4-v1"
COORDINATE_DENSITY: Final[str] = "per_native_volume"
DEPTH_TOLERANCE_ANG: Final[float] = 1e-12
DERIVATIVE_CAPABILITY_MODES: Final[Tuple[str, ...]] = (
    "exact_ad",
    "implicit_ad",
    "frozen_upstream",
    "finite_difference",
    "surrogate",
    "none",
)
DETECTOR_BOUNDARY_POLICY: Final[str] = "loss"
DETECTOR_COORDINATE_SYSTEM: Final[str] = "hemispherical_angles"
EIGENVALUE_NDIM: Final[int] = 2
EIGENVECTOR_NDIM: Final[int] = 3
FINAL_STATE_MODES: Final[Tuple[str, ...]] = ("plane_wave", "coulomb")
HERMITE_TABLE_POINTS: Final[Tuple[int, ...]] = (257, 513, 1025, 2049)
HERMITICITY_RELATIVE_TOLERANCE: Final[float] = 1.0e-12
KPATH_MODES: Final[Tuple[str, ...]] = (
    "Automatic",
    "Line-mode",
    "Explicit",
)
MAX_COEFFICIENT_CONDITION: Final[float] = 32.0
MAX_DECAY_PARAMETER: Final[float] = 4.0
MAX_EFFECTIVE_PRINCIPAL: Final[float] = 4.2
MAX_HYDROGENIC_PRINCIPAL: Final[int] = 7
MAX_LATTICE_CONDITION_NUMBER: Final[float] = 1e12
MAX_MATRIXEL_L: Final[int] = 4
MIN_COMPACT_GRID_POINTS: Final[int] = 3
MIN_DECAY_PARAMETER: Final[float] = 0.5
MIN_GRID_NODES: Final[int] = 2
MIN_INTERPOLATION_AXIS_POINTS: Final[int] = 2
MIN_SCALED_SINGULAR_VALUE: Final[float] = 1e-12
ORBITAL_POSITION_NDIM: Final[int] = 2
PATH_STEP_ATOL_INV_ANG: Final[float] = 1.0e-13
PATH_STEP_RTOL: Final[float] = 1.0e-12
POST_COUNT_MODES: Final[Tuple[str, ...]] = ("none", "calibrated")
RADIAL_ACCELERATORS: Final[Tuple[str, ...]] = ("direct", "hermite")
RADIAL_MODES: Final[Tuple[str, ...]] = (
    "slater",
    "hydrogenic",
    "grid",
    "fixed",
)
REGISTERED_DOMAIN_FRAME_IDS: Final[Tuple[str, ...]] = (
    "org.diffpes.frame.sample_cartesian",
)
ROTATION_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-10
SAMPLE_CARTESIAN_FRAME_ID: Final[str] = "org.diffpes.frame.sample_cartesian"
SELF_ENERGY_MODES: Final[Tuple[str, ...]] = (
    "constant",
    "poly",
    "grid",
    "fermi_liquid",
    "bosonic_kink",
)
SHARD_CHECKPOINT_POLICIES: Final[Tuple[str, ...]] = (
    "everything",
    "dots_saveable",
)
SENSITIVITY_MODES: Final[Tuple[str, ...]] = ("constant", "smooth")
SLIT_ORIENTATIONS: Final[Tuple[str, ...]] = ("H", "V")
SURFACE_VECTOR_COUNT: Final[int] = 2
TAIL_COORDINATE_BOUND: Final[float] = 30.0
TB_HERMITICITY_TOLERANCE: Final[float] = 1e-12
TB_PAIR_LENGTH: Final[int] = 2
TRANSVERSALITY_ATOL: Final[float] = 1.0e-10
WAVECAR_SECOND_RECORD_VALUES: Final[int] = 13
WAVECAR_SINGLE_PRECISION_TAG: Final[int] = 45200

__all__: list[str] = [
    "ACQUISITION_MODES",
    "ARRAY_MATRIX_NDIM",
    "BACKGROUND_MODES",
    "CERTIFIED_RADIAL_PROFILES",
    "CERTIFIED_R_MAX_BOHR",
    "CERTIFIED_TAIL_ENVELOPE_ID",
    "COORDINATE_DENSITY",
    "DEPTH_TOLERANCE_ANG",
    "DERIVATIVE_CAPABILITY_MODES",
    "DETECTOR_BOUNDARY_POLICY",
    "DETECTOR_COORDINATE_SYSTEM",
    "EIGENVALUE_NDIM",
    "EIGENVECTOR_NDIM",
    "FINAL_STATE_MODES",
    "HERMITE_TABLE_POINTS",
    "HERMITICITY_RELATIVE_TOLERANCE",
    "KPATH_MODES",
    "MAX_COEFFICIENT_CONDITION",
    "MAX_DECAY_PARAMETER",
    "MAX_EFFECTIVE_PRINCIPAL",
    "MAX_HYDROGENIC_PRINCIPAL",
    "MAX_LATTICE_CONDITION_NUMBER",
    "MAX_MATRIXEL_L",
    "MIN_COMPACT_GRID_POINTS",
    "MIN_DECAY_PARAMETER",
    "MIN_GRID_NODES",
    "MIN_INTERPOLATION_AXIS_POINTS",
    "MIN_SCALED_SINGULAR_VALUE",
    "ORBITAL_POSITION_NDIM",
    "PATH_STEP_ATOL_INV_ANG",
    "PATH_STEP_RTOL",
    "POST_COUNT_MODES",
    "RADIAL_ACCELERATORS",
    "RADIAL_MODES",
    "REGISTERED_DOMAIN_FRAME_IDS",
    "ROTATION_ORTHOGONALITY_TOLERANCE",
    "SAMPLE_CARTESIAN_FRAME_ID",
    "SELF_ENERGY_MODES",
    "SHARD_CHECKPOINT_POLICIES",
    "SENSITIVITY_MODES",
    "SLIT_ORIENTATIONS",
    "SURFACE_VECTOR_COUNT",
    "TAIL_COORDINATE_BOUND",
    "TB_HERMITICITY_TOLERANCE",
    "TB_PAIR_LENGTH",
    "TRANSVERSALITY_ATOL",
    "WAVECAR_SECOND_RECORD_VALUES",
    "WAVECAR_SINGLE_PRECISION_TAG",
]
