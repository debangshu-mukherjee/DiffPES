"""Band structure and orbital projection data structures.

Extended Summary
----------------
Defines PyTree types for electronic band structure data and
orbital-resolved projections from VASP calculations. These are
the primary inputs to all ARPES simulation functions.

Routine Listings
----------------
:class:`BandStructure`
    PyTree for eigenvalues and k-point data.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:func:`make_band_structure`
    Factory for BandStructure with validation.
:func:`make_orbital_projection`
    Factory for OrbitalProjection with validation.
:func:`make_arpes_spectrum`
    Factory for ArpesSpectrum.

Notes
-----
Orbital indexing convention (9 orbitals):
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``
matching VASP PROCAR output ordering.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarNumeric


@register_pytree_node_class
class BandStructure(NamedTuple):
    """PyTree for electronic band structure.

    Attributes
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Float[Array, " K"]
        Integration weights for each k-point.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.
    """

    eigenvalues: Float[Array, "K B"]
    kpoints: Float[Array, "K 3"]
    kpoint_weights: Float[Array, " K"]
    fermi_energy: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "K B"],
            Float[Array, "K 3"],
            Float[Array, " K"],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (
                self.eigenvalues,
                self.kpoints,
                self.kpoint_weights,
                self.fermi_energy,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, "K B"],
            Float[Array, "K 3"],
            Float[Array, " K"],
            Float[Array, " "],
        ],
    ) -> "BandStructure":
        """Reconstruct from flattened components."""
        return cls(*children)


@register_pytree_node_class
class OrbitalProjection(NamedTuple):
    """PyTree for orbital-resolved band projections.

    Attributes
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights for K k-points, B bands,
        A atoms, and 9 orbitals.
    spin : Optional[Float[Array, "K B A 6"]]
        Spin projections (up/down for x, y, z) or None.
    oam : Optional[Float[Array, "K B A 3"]]
        Orbital angular momentum (p, d, total) or None.
    """

    projections: Float[Array, "K B A 9"]
    spin: Optional[Float[Array, "K B A 6"]]
    oam: Optional[Float[Array, "K B A 3"]]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "K B A 9"],
            Optional[Float[Array, "K B A 6"]],
            Optional[Float[Array, "K B A 3"]],
        ],
        None,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return (
            (self.projections, self.spin, self.oam),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, "K B A 9"],
            Optional[Float[Array, "K B A 6"]],
            Optional[Float[Array, "K B A 3"]],
        ],
    ) -> "OrbitalProjection":
        """Reconstruct from flattened components."""
        return cls(*children)


@register_pytree_node_class
class ArpesSpectrum(NamedTuple):
    """PyTree for ARPES simulation output.

    Attributes
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity for K k-points and E energies.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.
    """

    intensity: Float[Array, "K E"]
    energy_axis: Float[Array, " E"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, "K E"], Float[Array, " E"]],
        None,
    ]:
        """Flatten into JAX children and auxiliary data."""
        return ((self.intensity, self.energy_axis), None)

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, "K E"],
            Float[Array, " E"],
        ],
    ) -> "ArpesSpectrum":
        """Reconstruct from flattened components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_band_structure(
    eigenvalues: Float[Array, "K B"],
    kpoints: Float[Array, "K 3"],
    kpoint_weights: Union[Float[Array, " K"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> BandStructure:
    """Create a validated BandStructure instance.

    Parameters
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Union[Float[Array, " K"], None], optional
        Integration weights. Defaults to uniform weights.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : BandStructure
        Validated band structure instance.
    """
    eigenvalues_arr: Float[Array, "K B"] = jnp.asarray(
        eigenvalues, dtype=jnp.float64
    )
    kpoints_arr: Float[Array, "K 3"] = jnp.asarray(
        kpoints, dtype=jnp.float64
    )
    nkpts: int = eigenvalues_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float[Array, " K"] = jnp.ones(
            nkpts, dtype=jnp.float64
        )
    else:
        weights_arr = jnp.asarray(
            kpoint_weights, dtype=jnp.float64
        )
    fermi_arr: Float[Array, " "] = jnp.asarray(
        fermi_energy, dtype=jnp.float64
    )
    bands: BandStructure = BandStructure(
        eigenvalues=eigenvalues_arr,
        kpoints=kpoints_arr,
        kpoint_weights=weights_arr,
        fermi_energy=fermi_arr,
    )
    return bands


@jaxtyped(typechecker=beartype)
def make_orbital_projection(
    projections: Float[Array, "K B A 9"],
    spin: Optional[Float[Array, "K B A 6"]] = None,
    oam: Optional[Float[Array, "K B A 3"]] = None,
) -> OrbitalProjection:
    """Create a validated OrbitalProjection instance.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights.
    spin : Optional[Float[Array, "K B A 6"]], optional
        Spin projections. Default is None.
    oam : Optional[Float[Array, "K B A 3"]], optional
        Orbital angular momentum. Default is None.

    Returns
    -------
    orb_proj : OrbitalProjection
        Validated orbital projection instance.
    """
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Optional[Float[Array, "K B A 6"]] = None
    if spin is not None:
        spin_arr = jnp.asarray(spin, dtype=jnp.float64)
    oam_arr: Optional[Float[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)
    orb_proj: OrbitalProjection = OrbitalProjection(
        projections=proj_arr,
        spin=spin_arr,
        oam=oam_arr,
    )
    return orb_proj


@jaxtyped(typechecker=beartype)
def make_arpes_spectrum(
    intensity: Float[Array, "K E"],
    energy_axis: Float[Array, " E"],
) -> ArpesSpectrum:
    """Create an ArpesSpectrum instance.

    Parameters
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity map.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        ARPES spectrum instance.
    """
    intensity_arr: Float[Array, "K E"] = jnp.asarray(
        intensity, dtype=jnp.float64
    )
    energy_arr: Float[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )
    spectrum: ArpesSpectrum = ArpesSpectrum(
        intensity=intensity_arr,
        energy_axis=energy_arr,
    )
    return spectrum


__all__: list[str] = [
    "ArpesSpectrum",
    "BandStructure",
    "OrbitalProjection",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_orbital_projection",
]
