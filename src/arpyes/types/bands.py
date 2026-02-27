"""Band structure and orbital projection data structures.

Extended Summary
----------------
Defines PyTree types for electronic band structure data and
orbital-resolved projections from VASP calculations. These are
the primary inputs to all ARPES simulation functions.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:class:`BandStructure`
    PyTree for eigenvalues and k-point data.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:func:`make_arpes_spectrum`
    Factory for ArpesSpectrum.
:func:`make_band_structure`
    Factory for BandStructure with validation.
:func:`make_orbital_projection`
    Factory for OrbitalProjection with validation.

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

    Extended Summary
    ----------------
    Stores the core outputs of a DFT band structure calculation: the
    eigenvalue spectrum E_n(k), the k-point mesh in reciprocal space, the
    integration weights for each k-point, and the Fermi energy. Together
    these fields fully describe the single-particle electronic structure
    needed to simulate angle-resolved photoemission spectra.

    This type exists as a JAX-compatible PyTree so that band structure data
    can flow through ``jax.jit``, ``jax.vmap``, and ``jax.grad``
    transformations without manual flattening. All four fields are
    JAX-traced arrays (no static auxiliary data), which means the entire
    object is differentiable with respect to any of its fields.

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

    Notes
    -----
    Registered as a JAX PyTree via ``@register_pytree_node_class``.
    Because ``BandStructure`` is a ``NamedTuple``, JAX would normally
    flatten it by treating each field as a leaf in declaration order.
    The explicit ``tree_flatten`` / ``tree_unflatten`` pair overrides
    this default to make the children-vs-auxiliary split explicit and
    self-documenting, even though in this case all fields are children
    and the auxiliary data is ``None``.
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
        """Flatten into JAX leaf arrays and auxiliary data.

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children**: ``(eigenvalues, kpoints, kpoint_weights,
           fermi_energy)`` -- all four fields are dense JAX arrays that
           must be visible to the tracer so that ``jit``, ``grad``, and
           ``vmap`` can differentiate through or batch over them.
        2. **Auxiliary data**: ``None`` -- there are no static Python
           values (no strings, ints, or flags) because every field in a
           ``BandStructure`` is a numerical array.

        Returns
        -------
        children : tuple of jax.Array
            ``(eigenvalues, kpoints, kpoint_weights, fermi_energy)``.
        aux_data : None
            No static metadata is needed for reconstruction.
        """
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
        """Reconstruct a ``BandStructure`` from flattened components.

        Inverse of :meth:`tree_flatten`. Called by ``jax.tree_util`` when
        a traced ``BandStructure`` needs to be reassembled -- for example
        at the boundary of a ``jit``-compiled function or after a
        ``vmap`` transformation unstacks its batched leaves.

        Implementation Logic
        --------------------
        1. **Auxiliary data**: ignored (always ``None``) because
           ``BandStructure`` carries no static metadata.
        2. **Reconstruction**: unpacks the children tuple positionally
           into the ``NamedTuple`` constructor via ``cls(*children)``,
           restoring ``(eigenvalues, kpoints, kpoint_weights,
           fermi_energy)`` in declaration order.

        Parameters
        ----------
        _aux_data : None
            Unused static metadata (always ``None``).
        children : tuple of jax.Array
            ``(eigenvalues, kpoints, kpoint_weights, fermi_energy)``
            as returned by :meth:`tree_flatten`.

        Returns
        -------
        band_structure : BandStructure
            Reconstructed instance with the original array fields.
        """
        return cls(*children)


@register_pytree_node_class
class OrbitalProjection(NamedTuple):
    """PyTree for orbital-resolved band projections.

    Extended Summary
    ----------------
    Stores the orbital character of each Bloch state as extracted from
    a VASP PROCAR file: the squared projection of each eigenstate onto
    atom-centred spherical harmonics. The nine orbital channels follow
    the VASP ordering ``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``.

    Optional spin-resolved and orbital-angular-momentum (OAM) projections
    are included when the DFT calculation was spin-orbit coupled. These
    additional fields enable simulation of spin-ARPES and circular-
    dichroism ARPES experiments.

    This type exists as a JAX-compatible PyTree so that orbital weights
    can participate in JAX transformations (``jit``, ``vmap``, ``grad``)
    alongside the band structure data. All fields are JAX arrays (or
    ``None``), so there is no static auxiliary data.

    Attributes
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights for K k-points, B bands,
        A atoms, and 9 orbitals.
    spin : Optional[Float[Array, "K B A 6"]]
        Spin projections (up/down for x, y, z) or None.
    oam : Optional[Float[Array, "K B A 3"]]
        Orbital angular momentum (p, d, total) or None.

    Notes
    -----
    Registered as a JAX PyTree via ``@register_pytree_node_class``.
    The ``spin`` and ``oam`` fields may be ``None`` when the underlying
    DFT calculation did not include spin-orbit coupling. JAX's tree
    utilities treat ``None`` leaves correctly (they are simply skipped
    during tracing), so the PyTree structure adapts automatically to
    both spin-polarised and non-spin-polarised inputs.
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
        """Flatten into JAX leaf arrays and auxiliary data.

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children**: ``(projections, spin, oam)`` -- all three
           fields are either dense JAX arrays or ``None``. They are
           placed in the children tuple so that the JAX tracer can see
           (and differentiate through) the numerical data. When
           ``spin`` or ``oam`` is ``None``, JAX treats that leaf as an
           empty subtree and skips it during tracing.
        2. **Auxiliary data**: ``None`` -- there are no static Python
           values. The presence or absence of ``spin`` / ``oam`` is
           encoded implicitly by the ``None`` leaves inside the
           children tuple, not by a separate flag in auxiliary data.

        Returns
        -------
        children : tuple of (jax.Array or None)
            ``(projections, spin, oam)``.
        aux_data : None
            No static metadata is needed for reconstruction.
        """
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
        """Reconstruct an ``OrbitalProjection`` from flattened components.

        Inverse of :meth:`tree_flatten`. Called by ``jax.tree_util`` when
        a traced ``OrbitalProjection`` needs to be reassembled -- for
        example at the boundary of a ``jit``-compiled function or after
        ``vmap`` unstacks batched leaves.

        Implementation Logic
        --------------------
        1. **Auxiliary data**: ignored (always ``None``) because
           ``OrbitalProjection`` carries no static metadata.
        2. **Reconstruction**: unpacks the children tuple positionally
           into the ``NamedTuple`` constructor via ``cls(*children)``,
           restoring ``(projections, spin, oam)`` in declaration order.
           If ``spin`` or ``oam`` was ``None`` before flattening, it
           remains ``None`` after reconstruction.

        Parameters
        ----------
        _aux_data : None
            Unused static metadata (always ``None``).
        children : tuple of (jax.Array or None)
            ``(projections, spin, oam)`` as returned by
            :meth:`tree_flatten`.

        Returns
        -------
        orbital_projection : OrbitalProjection
            Reconstructed instance with the original array fields.
        """
        return cls(*children)


@register_pytree_node_class
class SpinOrbitalProjection(NamedTuple):
    """PyTree for orbital projections with mandatory spin data.

    Variant of :class:`OrbitalProjection` where the ``spin`` field
    is required (not Optional). Used by spin-orbit coupling
    simulations (:func:`~arpyes.simul.simulate_soc`) that need
    guaranteed access to spin projection data.

    Hardening (validation that spin data exists) belongs at the I/O
    boundary — the factory :func:`make_spin_orbital_projection`
    enforces the contract so the simulation kernel stays pure.

    Attributes
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights for K k-points, B bands,
        A atoms, and 9 orbitals.
    spin : Float[Array, "K B A 6"]
        Spin projections (up/down for x, y, z). Required.
    oam : Optional[Float[Array, "K B A 3"]]
        Orbital angular momentum (p, d, total) or None.

    Notes
    -----
    Registered as a JAX PyTree via ``@register_pytree_node_class``.
    """

    projections: Float[Array, "K B A 9"]
    spin: Float[Array, "K B A 6"]
    oam: Optional[Float[Array, "K B A 3"]]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "K B A 9"],
            Float[Array, "K B A 6"],
            Optional[Float[Array, "K B A 3"]],
        ],
        None,
    ]:
        """Flatten into JAX leaf arrays and auxiliary data.

        Returns
        -------
        children : tuple of (jax.Array or None)
            ``(projections, spin, oam)``.
        aux_data : None
            No static metadata.
        """
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
            Float[Array, "K B A 6"],
            Optional[Float[Array, "K B A 3"]],
        ],
    ) -> "SpinOrbitalProjection":
        """Reconstruct a ``SpinOrbitalProjection`` from flattened components.

        Parameters
        ----------
        _aux_data : None
            Unused static metadata (always ``None``).
        children : tuple of (jax.Array or None)
            ``(projections, spin, oam)``.

        Returns
        -------
        spin_orbital_projection : SpinOrbitalProjection
            Reconstructed instance.
        """
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_spin_orbital_projection(
    projections: Float[Array, "K B A 9"],
    spin: Float[Array, "K B A 6"],
    oam: Optional[Float[Array, "K B A 3"]] = None,
) -> SpinOrbitalProjection:
    """Create a validated ``SpinOrbitalProjection`` instance.

    Factory function that validates and normalises orbital projection
    data with mandatory spin. All non-None arrays are cast to
    ``float64``.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights.
    spin : Float[Array, "K B A 6"]
        Spin projections. Required.
    oam : Optional[Float[Array, "K B A 3"]], optional
        Orbital angular momentum. Default is None.

    Returns
    -------
    soc_proj : SpinOrbitalProjection
        Validated instance with all non-None arrays in ``float64``.
    """
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Float[Array, "K B A 6"] = jnp.asarray(
        spin, dtype=jnp.float64
    )
    oam_arr: Optional[Float[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)
    return SpinOrbitalProjection(
        projections=proj_arr,
        spin=spin_arr,
        oam=oam_arr,
    )


@register_pytree_node_class
class ArpesSpectrum(NamedTuple):
    """PyTree for ARPES simulation output.

    Extended Summary
    ----------------
    Stores the result of an ARPES simulation: a two-dimensional
    photoemission intensity map I(k, E) together with its energy axis.
    The k-point dimension indexes the momentum-resolved detector
    channels, and the energy dimension indexes the binding-energy grid
    on which the spectral function was evaluated.

    This type exists as a JAX-compatible PyTree so that simulated
    spectra can be compared to experimental data inside ``jit``-compiled
    loss functions and differentiated with ``grad`` for parameter
    fitting. Both fields are dense JAX arrays with no static auxiliary
    data, making the entire object fully differentiable.

    Attributes
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity for K k-points and E energies.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.

    Notes
    -----
    Registered as a JAX PyTree via ``@register_pytree_node_class``.
    The explicit ``tree_flatten`` / ``tree_unflatten`` pair overrides
    the default ``NamedTuple`` flattening to make the children-vs-
    auxiliary split self-documenting. In this case all fields are
    children and the auxiliary data is ``None``.
    """

    intensity: Float[Array, "K E"]
    energy_axis: Float[Array, " E"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, "K E"], Float[Array, " E"]],
        None,
    ]:
        """Flatten into JAX leaf arrays and auxiliary data.

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children**: ``(intensity, energy_axis)`` -- both fields are
           dense JAX arrays. ``intensity`` is the 2-D photoemission map
           and ``energy_axis`` is the 1-D energy grid. Both must be
           visible to the tracer for ``jit``/``grad``/``vmap``.
        2. **Auxiliary data**: ``None`` -- there are no static Python
           values because every field in an ``ArpesSpectrum`` is a
           numerical array.

        Returns
        -------
        children : tuple of jax.Array
            ``(intensity, energy_axis)``.
        aux_data : None
            No static metadata is needed for reconstruction.
        """
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
        """Reconstruct an ``ArpesSpectrum`` from flattened components.

        Inverse of :meth:`tree_flatten`. Called by ``jax.tree_util`` when
        a traced ``ArpesSpectrum`` needs to be reassembled -- for example
        at the boundary of a ``jit``-compiled function or after ``vmap``
        unstacks batched leaves.

        Implementation Logic
        --------------------
        1. **Auxiliary data**: ignored (always ``None``) because
           ``ArpesSpectrum`` carries no static metadata.
        2. **Reconstruction**: unpacks the children tuple positionally
           into the ``NamedTuple`` constructor via ``cls(*children)``,
           restoring ``(intensity, energy_axis)`` in declaration order.

        Parameters
        ----------
        _aux_data : None
            Unused static metadata (always ``None``).
        children : tuple of jax.Array
            ``(intensity, energy_axis)`` as returned by
            :meth:`tree_flatten`.

        Returns
        -------
        spectrum : ArpesSpectrum
            Reconstructed instance with the original array fields.
        """
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_band_structure(
    eigenvalues: Float[Array, "K B"],
    kpoints: Float[Array, "K 3"],
    kpoint_weights: Union[Float[Array, " K"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> BandStructure:
    """Create a validated ``BandStructure`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw band structure
    data before constructing a ``BandStructure`` PyTree. All input
    arrays are cast to ``float64`` for numerical stability in
    downstream ARPES simulations (energy differences and Lorentzian
    broadening are sensitive to precision). Missing k-point weights
    are replaced by uniform weights so that callers do not need to
    handle the common equal-weight case explicitly.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape and dtype constraints on the inputs are checked at
    call time, catching mismatched dimensions before they propagate
    into the simulation pipeline.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenvalues`` and ``kpoints`` are
       converted via ``jnp.asarray(..., dtype=jnp.float64)`` to
       guarantee double-precision arithmetic in all subsequent
       computations.
    2. **Default handling**: if ``kpoint_weights`` is ``None``, a
       uniform weight vector ``jnp.ones(K)`` is created (where *K* is
       inferred from ``eigenvalues.shape[0]``). This is the standard
       assumption for band structure paths (as opposed to Brillouin-
       zone integrations where weights vary).
    3. **Construction**: the four validated arrays are passed to the
       ``BandStructure`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

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
        Validated band structure instance with all arrays in
        ``float64``.
    """
    eigenvalues_arr: Float[Array, "K B"] = jnp.asarray(
        eigenvalues, dtype=jnp.float64
    )
    kpoints_arr: Float[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    nkpts: int = eigenvalues_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float[Array, " K"] = jnp.ones(nkpts, dtype=jnp.float64)
    else:
        weights_arr = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)
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
    """Create a validated ``OrbitalProjection`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw orbital
    projection data before constructing an ``OrbitalProjection``
    PyTree. The mandatory ``projections`` array is cast to ``float64``;
    the optional ``spin`` and ``oam`` arrays are cast only when they
    are not ``None``, preserving the sentinel meaning of ``None`` for
    calculations without spin-orbit coupling.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints (K, B, A dimensions must agree across
    all provided arrays) are checked at call time.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``projections`` is converted via
       ``jnp.asarray(..., dtype=jnp.float64)``. ``spin`` and ``oam``
       are likewise converted when they are not ``None``.
    2. **Default handling**: ``spin`` and ``oam`` default to ``None``,
       which signals that the DFT calculation did not include spin-
       orbit coupling. No placeholder arrays are created -- the
       ``None`` propagates into the PyTree leaf, keeping the tree
       structure minimal.
    3. **Construction**: the three validated fields are passed to the
       ``OrbitalProjection`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

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
        Validated orbital projection instance with all non-None
        arrays in ``float64``.
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
    """Create a validated ``ArpesSpectrum`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises simulated ARPES
    data before constructing an ``ArpesSpectrum`` PyTree. Both input
    arrays are cast to ``float64`` so that downstream loss functions
    (e.g. mean-squared error against experimental data) maintain full
    double-precision accuracy.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that the energy dimension *E* is checked for consistency
    between ``intensity`` and ``energy_axis`` at call time.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``intensity`` and ``energy_axis`` are
       converted via ``jnp.asarray(..., dtype=jnp.float64)`` to
       guarantee double-precision arithmetic.
    2. **Default handling**: none -- both fields are mandatory and
       have no sentinel defaults.
    3. **Construction**: the two validated arrays are passed to the
       ``ArpesSpectrum`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

    Parameters
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity map.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        Validated ARPES spectrum instance with all arrays in
        ``float64``.
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
    "SpinOrbitalProjection",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_orbital_projection",
    "make_spin_orbital_projection",
]
