r"""Define numerical, physical, orbital, and VASP-format constants for diffpes.

Extended Summary
----------------
This module owns the scalar constants and small-array constants that the
subpackages share. CODATA physical constants fix the unit system from the
conventions charter. The system uses eV for energies and Angstrom for lengths.
These constants include the Bohr-to-Angstrom conversion, :math:`\hbar c`,
:math:`\hbar`, :math:`k_B`, and the electron rest energy.

Numerical tolerances keep kernels finite and differentiable near singular
points. Orbital conventions fix the shared projection basis. VASP-format
constants define the file formats that ``diffpes.inout`` reads and writes.
The module exports every external constant through :mod:`diffpes.types`.
Private names identify only module intermediates.

Routine Listings
----------------
:obj:`ATTR_AUX`
    HDF5 attribute name storing auxiliary PyTree data as JSON.
:obj:`ATTR_NONE`
    HDF5 attribute name listing PyTree fields stored as None.
:obj:`ATTR_TYPE`
    HDF5 attribute name storing the PyTree type name.
:obj:`BAND_LINE_MIN_VALUES`
    Minimum tokens on an EIGENVAL band line.
:obj:`BAND_LINE_SPIN_VALUES`
    Tokens on a spin-polarized EIGENVAL band line.
:obj:`BAND_NDIM`
    Expected dimensionality of band-energy arrays.
:obj:`BOHR_TO_ANGSTROM`
    Bohr radius in Angstrom.
:obj:`CARTESIAN_COMPONENTS`
    Number of Cartesian components used by tight-binding bond vectors.
:obj:`CHANNELS_BY_PAIR`
    Slater--Koster channels supported for each angular-momentum pair.
:obj:`COORDINATE_MODE_TOKENS`
    Recognized KPOINTS coordinate-mode tokens.
:obj:`D_ORBITAL_SLICE`
    Slice selecting the five d orbitals.
:obj:`EIG_DOWN_INDEX`
    Column index of spin-down eigenvalues in EIGENVAL.
:obj:`EIG_UP_INDEX`
    Column index of spin-up eigenvalues in EIGENVAL.
:obj:`ENERGY_AXIS_NDIM`
    Expected dimensionality of energy-axis arrays.
:obj:`EKIN_FLOOR_EV`
    Set the physical kinetic-energy floor in eV.
:obj:`EPS`
    Epsilon floor guarding divisions and norms.
:obj:`EPS_DEG`
    Lorentzian width regularizing degenerate eigenvector derivatives.
:obj:`FLOAT_TOKEN_RE`
    Compiled regex matching floating-point tokens.
:obj:`GAUNT_IMAG_TOL`
    Tolerance for discarding imaginary Gaunt residues.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`HBAR_SQ_OVER_2ME_EV_ANG2`
    Store the free-electron dispersion constant in eV Angstrom squared.
:obj:`INTENSITY_NDIM`
    Expected dimensionality of intensity arrays.
:obj:`ISPIN2_BLOCKS`
    PROCAR block count for ISPIN=2 calculations.
:obj:`ISPIN_SPIN_POLARIZED`
    ISPIN value marking spin-polarized VASP runs.
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:obj:`KPATH_AUX_WITH_COMMENT_LEN`
    KPathInfo auxiliary-data length including a comment.
:obj:`KPATH_AUX_WITH_COORD_MODE_LEN`
    KPathInfo auxiliary-data length including a coordinate mode.
:obj:`KPOINT_LINE_VALUES`
    Tokens on an EIGENVAL k-point line.
:obj:`K_PREFACTOR_INV_ANG_SQRT_EV`
    Store the momentum prefactor in inverse Angstrom per square-root eV.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`LATTICE_ROWS`
    Number of lattice-vector rows in POSCAR/CHGCAR headers.
:obj:`DEFAULT_SUPERCELL_RADIUS`
    Default translation radius for tight-binding neighbor discovery.
:obj:`M_D`
    Magnetic quantum numbers of the d orbitals.
:obj:`M_P`
    Magnetic quantum numbers of the p orbitals.
:obj:`ME_EV`
    Electron rest energy in eV.
:obj:`MIN_SUM`
    Minimum-sum floor guarding normalizations.
:obj:`MIN_BOND_DISTANCE`
    Minimum nonzero distance accepted by neighbor discovery.
:obj:`MINIMUM_AXIS_POINTS`
    Minimum number of points accepted on a sampled DOS energy axis.
:obj:`N_ORBITALS`
    Number of orbitals in the VASP projection basis.
:obj:`N_SOC_MAG_BLOCKS`
    Magnetization block count in SOC CHGCAR files.
:obj:`N_SPIN_COMPONENTS`
    Spin-projection component count in PROCAR.
:obj:`N_TAYLOR`
    Taylor-series order for the Faddeeva evaluation.
:obj:`NON_S_ORBITAL_SLICE`
    Slice selecting all non-s orbitals.
:obj:`NONSPIN_COLS`
    DOSCAR column count without spin polarization.
:obj:`NORM_EPS`
    Epsilon floor guarding eigenvector normalization.
:obj:`KNOWN_CHANNELS`
    Complete set of supported Slater--Koster channel names.
:obj:`MATRIX_NDIM`
    Expected dimensionality of tight-binding operator matrices.
:obj:`MAX_SK_ANGULAR_MOMENTUM`
    Maximum angular momentum supported by Slater--Koster construction.
:obj:`ORBITAL_DIRS_NORMALIZED`
    Unit-normalized orbital directions in VASP orbital order.
:obj:`ORBITAL_INDEX`
    Mapping from orbital name to VASP orbital index.
:obj:`P_ORBITAL_SLICE`
    Slice selecting the three p orbitals.
:obj:`PHASE_LOSS_MESSAGE`
    Warning text for PROCAR magnitude-only eigenvectors.
:obj:`PARAMETER_KEY_PARTS`
    Number of colon-delimited parts in a qualified Slater--Koster key.
:obj:`PRESET_NAMES`
    Recognized band-scatter plotting preset names.
:obj:`S_IDX`
    Index of the s orbital.
:obj:`SCALAR_LINE_COMPONENTS`
    Tokens on a scalar CHGCAR header line.
:obj:`SMALL_ARGUMENT`
    Small-argument cutoff for spherical Bessel seeds.
:obj:`SHELL_ATOLERANCE`
    Absolute tolerance for grouping equal-distance neighbor shells.
:obj:`SHELL_RTOLERANCE`
    Relative tolerance for grouping equal-distance neighbor shells.
:obj:`SOC_BLOCKS`
    PROCAR block count for SOC calculations.
:obj:`SPECIES_PAIR_PARTS`
    Number of species labels in a Slater--Koster material-pair key.
:obj:`SPIN_COLS`
    DOSCAR column count with spin polarization.
:obj:`SPECTRUM_NDIM`
    Expected dimensionality of tight-binding eigenspectra.
:obj:`TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2`
    Store the inverse free-electron dispersion constant.
:obj:`WEIGHT_COMPONENT_COUNT`
    Tokens on a weighted k-point line.
:obj:`WEIGHT_COMPONENT_INDEX`
    Index of the weight token on a k-point line.
:obj:`XYZ_COMPONENTS`
    Number of Cartesian vector components.

Notes
-----
A tolerance change modifies the numerical behavior of every consumer. Treat
each edit as a physics change. Rerun the gradient and finite-difference gates.
This module imports JAX because the orbital tables contain device arrays.
Place constants that must load without JAX in a different module.
"""

import re
from types import MappingProxyType

import jax.numpy as jnp
from beartype.typing import Final
from jaxtyping import Array, Float

ATTR_AUX: Final[str] = "_aux_data_json"
ATTR_NONE: Final[str] = "_none_fields"
ATTR_TYPE: Final[str] = "_pytree_type"
BAND_LINE_MIN_VALUES: Final[int] = 2
BAND_LINE_SPIN_VALUES: Final[int] = 3
BAND_NDIM: Final[int] = 2
BOHR_TO_ANGSTROM: Final[float] = 0.529177
CARTESIAN_COMPONENTS: Final[int] = 3
CHANNELS_BY_PAIR: Final[MappingProxyType] = MappingProxyType(
    {
        (0, 0): ("ss_sigma",),
        (0, 1): ("sp_sigma",),
        (0, 2): ("sd_sigma",),
        (1, 1): ("pp_sigma", "pp_pi"),
        (1, 2): ("pd_sigma", "pd_pi"),
        (2, 2): ("dd_sigma", "dd_pi", "dd_delta"),
    }
)
COORDINATE_MODE_TOKENS: Final[frozenset[str]] = frozenset(
    {"cartesian", "reciprocal", "direct", "fractional"}
)
DEFAULT_SUPERCELL_RADIUS: Final[int] = 2
D_ORBITAL_SLICE: Final[slice] = slice(4, 9)
EIG_DOWN_INDEX: Final[int] = 2
EIG_UP_INDEX: Final[int] = 1
ENERGY_AXIS_NDIM: Final[int] = 1
EKIN_FLOOR_EV: Final[float] = 1e-2
EPS: Final[float] = 1e-12
EPS_DEG: Final[float] = 1e-9
FLOAT_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
GAUNT_IMAG_TOL: Final[float] = 1e-12
HBAR_C_EV_A: Final[float] = 1973.269804
HBAR_EV_S: Final[float] = 6.582119569e-16
HBAR_SQ_OVER_2ME_EV_ANG2: Final[float] = 3.8099821
TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2: Final[float] = 1.0 / HBAR_SQ_OVER_2ME_EV_ANG2
K_PREFACTOR_INV_ANG_SQRT_EV: Final[float] = (
    TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2**0.5
)
INTENSITY_NDIM: Final[int] = 2
ISPIN2_BLOCKS: Final[int] = 2
ISPIN_SPIN_POLARIZED: Final[int] = 2
KB_EV_PER_K: Final[float] = 8.617333e-5
KPATH_AUX_WITH_COMMENT_LEN: Final[int] = 3
KPATH_AUX_WITH_COORD_MODE_LEN: Final[int] = 4
KPOINT_LINE_VALUES: Final[int] = 4
L_MAX: Final[int] = 4
LATTICE_ROWS: Final[int] = 3
KNOWN_CHANNELS: Final[frozenset[str]] = frozenset(
    channel for channels in CHANNELS_BY_PAIR.values() for channel in channels
)
MATRIX_NDIM: Final[int] = 2
MAX_SK_ANGULAR_MOMENTUM: Final[int] = 2
M_D: Float[Array, " 5"] = jnp.array(
    [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float64
)
M_P: Float[Array, " 3"] = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float64)
ME_EV: Final[float] = 510998.95
MIN_SUM: Final[float] = 1e-30
MIN_BOND_DISTANCE: Final[float] = 1e-12
MINIMUM_AXIS_POINTS: Final[int] = 2
N_ORBITALS: Final[int] = 9
N_SOC_MAG_BLOCKS: Final[int] = 3
N_SPIN_COMPONENTS: Final[int] = 6
N_TAYLOR: Final[int] = 64
NON_S_ORBITAL_SLICE: Final[slice] = slice(1, 9)
NONSPIN_COLS: Final[int] = 3
NORM_EPS: Final[float] = 1e-12
ORBITAL_INDEX: Final[MappingProxyType] = MappingProxyType(
    {
        "s": 0,
        "py": 1,
        "pz": 2,
        "px": 3,
        "dxy": 4,
        "dyz": 5,
        "dz2": 6,
        "dxz": 7,
        "dx2y2": 8,
    }
)
P_ORBITAL_SLICE: Final[slice] = slice(1, 4)
PHASE_LOSS_MESSAGE: Final[str] = (
    "vasp_to_diagonalized uses sqrt(|c|^2) magnitudes from PROCAR and cannot "
    "recover complex eigenvector phases. Matrix elements that depend on phase "
    "interference are approximate."
)
PARAMETER_KEY_PARTS: Final[int] = 2
PRESET_NAMES: Final[tuple[str, ...]] = (
    "s",
    "py",
    "pz",
    "px",
    "p",
    "dxy",
    "dyz",
    "dz2",
    "dxz",
    "dx2y2",
    "d",
    "non_s",
    "total",
    "spin_x_up",
    "spin_x_down",
    "spin_y_up",
    "spin_y_down",
    "spin_z_up",
    "spin_z_down",
    "spin_x",
    "spin_y",
    "spin_z",
    "oam_p",
    "oam_d",
    "oam_total",
    "oam_abs_total",
)
S_IDX: Final[int] = 0
SCALAR_LINE_COMPONENTS: Final[int] = 3
SMALL_ARGUMENT: Final[float] = 1e-8
SHELL_ATOLERANCE: Final[float] = 1e-8
SHELL_RTOLERANCE: Final[float] = 1e-8
SOC_BLOCKS: Final[int] = 4
SPECIES_PAIR_PARTS: Final[int] = 2
SPIN_COLS: Final[int] = 5
SPECTRUM_NDIM: Final[int] = 2
WEIGHT_COMPONENT_COUNT: Final[int] = 4
WEIGHT_COMPONENT_INDEX: Final[int] = 3
XYZ_COMPONENTS: Final[int] = 3

_ORBITAL_DIRS: Float[Array, "9 3"] = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=jnp.float64,
)
_ORBITAL_NORMS: Float[Array, " 9"] = jnp.where(
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1) > 0.0,
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1),
    1.0,
)
ORBITAL_DIRS_NORMALIZED: Float[Array, "9 3"] = (
    _ORBITAL_DIRS / _ORBITAL_NORMS[:, jnp.newaxis]
)

__all__: list[str] = [
    "ATTR_AUX",
    "ATTR_NONE",
    "ATTR_TYPE",
    "BAND_LINE_MIN_VALUES",
    "BAND_LINE_SPIN_VALUES",
    "BAND_NDIM",
    "BOHR_TO_ANGSTROM",
    "CARTESIAN_COMPONENTS",
    "CHANNELS_BY_PAIR",
    "COORDINATE_MODE_TOKENS",
    "D_ORBITAL_SLICE",
    "DEFAULT_SUPERCELL_RADIUS",
    "EIG_DOWN_INDEX",
    "EIG_UP_INDEX",
    "ENERGY_AXIS_NDIM",
    "EKIN_FLOOR_EV",
    "EPS",
    "EPS_DEG",
    "FLOAT_TOKEN_RE",
    "GAUNT_IMAG_TOL",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "HBAR_SQ_OVER_2ME_EV_ANG2",
    "INTENSITY_NDIM",
    "ISPIN2_BLOCKS",
    "ISPIN_SPIN_POLARIZED",
    "KB_EV_PER_K",
    "KPATH_AUX_WITH_COMMENT_LEN",
    "KPATH_AUX_WITH_COORD_MODE_LEN",
    "KPOINT_LINE_VALUES",
    "K_PREFACTOR_INV_ANG_SQRT_EV",
    "L_MAX",
    "LATTICE_ROWS",
    "KNOWN_CHANNELS",
    "M_D",
    "M_P",
    "MATRIX_NDIM",
    "MAX_SK_ANGULAR_MOMENTUM",
    "ME_EV",
    "MIN_BOND_DISTANCE",
    "MINIMUM_AXIS_POINTS",
    "MIN_SUM",
    "N_ORBITALS",
    "N_SOC_MAG_BLOCKS",
    "N_SPIN_COMPONENTS",
    "N_TAYLOR",
    "NON_S_ORBITAL_SLICE",
    "NONSPIN_COLS",
    "NORM_EPS",
    "ORBITAL_DIRS_NORMALIZED",
    "ORBITAL_INDEX",
    "P_ORBITAL_SLICE",
    "PARAMETER_KEY_PARTS",
    "PHASE_LOSS_MESSAGE",
    "PRESET_NAMES",
    "S_IDX",
    "SCALAR_LINE_COMPONENTS",
    "SHELL_ATOLERANCE",
    "SHELL_RTOLERANCE",
    "SMALL_ARGUMENT",
    "SOC_BLOCKS",
    "SPECIES_PAIR_PARTS",
    "SPIN_COLS",
    "SPECTRUM_NDIM",
    "TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2",
    "WEIGHT_COMPONENT_COUNT",
    "WEIGHT_COMPONENT_INDEX",
    "XYZ_COMPONENTS",
]
