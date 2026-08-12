"""Define diagonalized electronic-structure data.

Extended Summary
----------------
This module stores eigensystems with their k-points, geometry,
orbital basis, surface depths, and Fermi energy.

Routine Listings
----------------
:class:`DiagonalizedBands`
    Store diagonalized electronic-structure data in a JAX PyTree.
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.constants import (
    CARTESIAN_COMPONENTS,
    DEPTH_TOLERANCE_ANG,
    EIGENVALUE_NDIM,
    EIGENVECTOR_NDIM,
    ORBITAL_POSITION_NDIM,
)

from .aliases import ScalarNumeric
from .electronic_structure_validation import (
    _checked_geometry,
    _validate_basis_geometry,
    _validate_depths_shape,
)
from .geometry import CrystalGeometry
from .orbital_basis import OrbitalBasis


class DiagonalizedBands(eqx.Module):
    """Store diagonalized electronic-structure data in a JAX PyTree.

    The carrier is the tight-binding-to-ARPES interface. Geometry and orbital
    metadata travel with each eigensystem so later matrix-element stages can
    form Cartesian momenta, atomic interference phases, and orbital operators.

    :see: :class:`~.test_diagonalized_bands.TestDiagonalizedBands`

    Attributes
    ----------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float64[Array, "n_k 3"]
        Fractional reciprocal-space coordinates.
    fermi_energy : Float64[Array, ""]
        Fermi energy in eV.
    geometry : CrystalGeometry
        Crystal geometry. Its numerical fields are differentiable children.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` ties centres to atoms.
    depths : Optional[Float64[Array, "n_orb"]]
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Native tight-binding diagonalization propagates this
        differentiable leaf without transformation.

    Notes
    -----
    The numerical eigensystem, geometry, and optional orbital-position fields
    remain JAX leaves. ``basis`` is static because its quantum numbers and
    atom mapping shape compiled operator construction.

    See Also
    --------
    TBModel : Tight-binding carrier whose diagonalization produces bands.
    make_diagonalized_bands : Validating carrier factory.
    """

    eigenvalues: Float64[Array, "n_k n_bands"]
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"]
    kpoints: Float64[Array, "n_k 3"]
    fermi_energy: Float64[Array, ""]
    geometry: CrystalGeometry
    basis: OrbitalBasis = eqx.field(static=True)
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None
    depths: Optional[Float64[Array, " n_orb"]] = None

    def __check_init__(self) -> None:
        """Validate the static eigensystem invariants again."""
        _validate_diagonalized_structure(
            self.eigenvalues,
            self.eigenvectors,
            self.kpoints,
            self.fermi_energy,
            self.geometry,
            self.basis,
            self.orbital_positions,
            self.depths,
        )


def _validate_diagonalized_structure(
    eigenvalues: Float64[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex128[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float64[Array, "n_k_p 3"],
    fermi_energy: Float64[Array, ""],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]],
    depths: Optional[Float64[Array, " n_depth"]],
) -> None:
    """PRIVATE: Validate static eigensystem shapes and context.

    Implementation Logic
    --------------------
    Check types, ranks, and axis agreements through static shape
    metadata only. Then delegate the depth axis to
    ``_validate_depths_shape`` and the orbital-to-atom map to
    ``_validate_basis_geometry``.

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k_e n_bands_e"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k_v n_bands_v n_orb"]
        Complex orbital coefficients per k-point and band.
    kpoints : Float64[Array, "n_k_p 3"]
        Fractional reciprocal-space coordinates.
    fermi_energy : Float64[Array, ""]
        Fermi energy in eV.
    geometry : CrystalGeometry
        Differentiable lattice and atomic positions.
    basis : OrbitalBasis
        Orbital-to-atom and quantum-number metadata.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]]
        Explicit fractional orbital centres, or ``None``.
    depths : Optional[Float64[Array, " n_depth"]]
        Orbital depths in Angstrom, or ``None`` for a bulk model.

    Raises
    ------
    ValueError
        If geometry or basis has the wrong type. If an eigensystem
        array has the wrong rank or shape. If k-point, band, or orbital
        axes disagree. If ``orbital_positions`` has the wrong shape.
        The delegated validators raise ``ValueError`` for their own
        static contracts.
    """
    if not isinstance(geometry, CrystalGeometry):
        message: str = "geometry must be a CrystalGeometry"
        raise ValueError(message)
    if not isinstance(basis, OrbitalBasis):
        message = "basis must be an OrbitalBasis"
        raise ValueError(message)
    if eigenvalues.ndim != EIGENVALUE_NDIM:
        message = "eigenvalues must be two-dimensional"
        raise ValueError(message)
    if eigenvectors.ndim != EIGENVECTOR_NDIM:
        message = "eigenvectors must be three-dimensional"
        raise ValueError(message)
    if (
        kpoints.ndim != EIGENVALUE_NDIM
        or kpoints.shape[1] != CARTESIAN_COMPONENTS
    ):
        message = "kpoints must have shape (n_k, 3)"
        raise ValueError(message)
    if fermi_energy.ndim != 0:
        message = "fermi_energy must be scalar"
        raise ValueError(message)
    if eigenvalues.shape != eigenvectors.shape[:2]:
        message = "eigenvalues and eigenvectors must agree on n_k and n_bands"
        raise ValueError(message)
    if eigenvalues.shape[0] != kpoints.shape[0]:
        message = "eigenvalues and kpoints must agree on n_k"
        raise ValueError(message)
    if eigenvectors.shape[2] != len(basis.n):
        message = "eigenvector orbital axis must match basis"
        raise ValueError(message)
    if orbital_positions is not None and (
        orbital_positions.ndim != ORBITAL_POSITION_NDIM
        or orbital_positions.shape != (len(basis.n), CARTESIAN_COMPONENTS)
    ):
        message = "orbital_positions must have shape (n_orbitals, 3)"
        raise ValueError(message)
    _validate_depths_shape(depths, len(basis.n))
    _validate_basis_geometry(basis, geometry)


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(  # noqa: DOC502, DOC503
    eigenvalues: Float64[Array, "n_k_e n_bands_e"],
    eigenvectors: Complex128[Array, "n_k_v n_bands_v n_orb"],
    kpoints: Float64[Array, "n_k_p 3"],
    geometry: CrystalGeometry,
    basis: OrbitalBasis,
    fermi_energy: ScalarNumeric = 0.0,
    orbital_positions: Optional[Float64[Array, "n_orb 3"]] = None,
    depths: Optional[Float64[Array, " n_depth"]] = None,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    The factory normalizes every numerical array and validates its axes
    against the supplied geometry and orbital basis.

    :see: :class:`~.test_diagonalized_bands.TestMakeDiagonalizedBands`

    Parameters
    ----------
    eigenvalues : Float64[Array, "n_k_e n_bands_e"]
        Band energies in eV.
    eigenvectors : Complex128[Array, "n_k_v n_bands_v n_orb"]
        Complex orbital coefficients in the basis-position gauge.
    kpoints : Float64[Array, "n_k_p 3"]
        Fractional reciprocal-space coordinates.
    geometry : CrystalGeometry
        Crystal geometry whose numerical leaves remain differentiable.
    basis : OrbitalBasis
        Orbital and atom metadata (**static** -- changing it triggers
        retracing).
    fermi_energy : ScalarNumeric, optional
        Fermi energy in eV. Default is 0.0.
    orbital_positions : Optional[Float64[Array, "n_orb 3"]], optional
        Explicit fractional orbital centres associated with the
        basis-position-gauge coefficients. ``None`` derives centres from
        atom assignments. Default is ``None``.
    depths : Optional[Float64[Array, "n_depth"]], optional
        Orbital depths in Angstrom below the top surface. ``None`` denotes a
        bulk model. Default is ``None``.

    Returns
    -------
    bands : DiagonalizedBands
        Validated double-precision eigensystem and its structural context.

    Raises
    ------
    ValueError
        If eigensystem axes, basis size, or atom assignments disagree.
    EquinoxRuntimeError
        If any numerical leaf is non-finite.

    Notes
    -----
    Static validation checks array axes and context before tracing. Runtime
    validation uses :func:`equinox.error_if` for every numerical leaf, so the
    same rejection behavior remains active under JIT.

    See Also
    --------
    DiagonalizedBands : Carrier constructed by this factory.
    make_tb_model : Construct the model diagonalized by native TB producers.
    """
    eigenvalue_array: Float64[Array, "n_k n_bands"] = jnp.asarray(
        eigenvalues,
        dtype=jnp.float64,
    )
    eigenvector_array: Complex128[Array, "n_k n_bands n_orb"] = jnp.asarray(
        eigenvectors,
        dtype=jnp.complex128,
    )
    kpoint_array: Float64[Array, "n_k 3"] = jnp.asarray(
        kpoints,
        dtype=jnp.float64,
    )
    fermi_array: Float64[Array, ""] = jnp.asarray(
        fermi_energy,
        dtype=jnp.float64,
    )
    orbital_position_array: Optional[Float64[Array, "n_orb 3"]] = None
    if orbital_positions is not None:
        orbital_position_array = jnp.asarray(
            orbital_positions,
            dtype=jnp.float64,
        )
    depth_array: Optional[Float64[Array, " n_depth"]] = None
    if depths is not None:
        depth_array = jnp.asarray(depths, dtype=jnp.float64)
    _validate_diagonalized_structure(
        eigenvalue_array,
        eigenvector_array,
        kpoint_array,
        fermi_array,
        geometry,
        basis,
        orbital_position_array,
        depth_array,
    )

    eigenvalue_array = eqx.error_if(
        eigenvalue_array,
        ~jnp.all(jnp.isfinite(eigenvalue_array)),
        "make_diagonalized_bands: eigenvalues finite",
    )
    eigenvector_array = eqx.error_if(
        eigenvector_array,
        ~jnp.all(jnp.isfinite(eigenvector_array)),
        "make_diagonalized_bands: eigenvectors finite",
    )
    kpoint_array = eqx.error_if(
        kpoint_array,
        ~jnp.all(jnp.isfinite(kpoint_array)),
        "make_diagonalized_bands: kpoints finite",
    )
    fermi_array = eqx.error_if(
        fermi_array,
        ~jnp.isfinite(fermi_array),
        "make_diagonalized_bands: fermi energy finite",
    )
    if orbital_position_array is not None:
        orbital_position_array = eqx.error_if(
            orbital_position_array,
            ~jnp.all(jnp.isfinite(orbital_position_array)),
            "make_diagonalized_bands: orbital positions finite",
        )
    if depth_array is not None:
        depth_array = eqx.error_if(
            depth_array,
            ~jnp.all(jnp.isfinite(depth_array)),
            "make_diagonalized_bands: depths must be finite",
        )
        depth_array = eqx.error_if(
            depth_array,
            jnp.any(depth_array < -DEPTH_TOLERANCE_ANG),
            "make_diagonalized_bands: depths must be nonnegative",
        )
    checked_geometry: CrystalGeometry = _checked_geometry(
        geometry,
        "make_diagonalized_bands",
    )
    bands: DiagonalizedBands = DiagonalizedBands(
        eigenvalues=eigenvalue_array,
        eigenvectors=eigenvector_array,
        kpoints=kpoint_array,
        fermi_energy=fermi_array,
        geometry=checked_geometry,
        basis=basis,
        orbital_positions=orbital_position_array,
        depths=depth_array,
    )
    return bands


__all__: list[str] = [
    "DiagonalizedBands",
    "make_diagonalized_bands",
]
