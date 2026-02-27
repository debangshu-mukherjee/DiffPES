"""Simulation parameter data structures.

Extended Summary
----------------
Defines PyTree types for ARPES simulation parameters including
energy resolution, broadening widths, temperature, photon energy,
and light polarization configuration.

Routine Listings
----------------
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:class:`SimulationParams`
    PyTree for core simulation parameters.
:func:`make_polarization_config`
    Factory for PolarizationConfig.
:func:`make_simulation_params`
    Factory for SimulationParams.

Notes
-----
Polarization types follow standard optics conventions:
``"LVP"`` (s-pol), ``"LHP"`` (p-pol), ``"RCP"``, ``"LCP"``,
``"LAP"`` (linear arbitrary), ``"unpolarized"``.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric


@register_pytree_node_class
class SimulationParams(NamedTuple):
    """PyTree for ARPES simulation parameters.

    Collects all scalar physical parameters that control an ARPES
    simulation: the energy window and its discretisation, Gaussian
    and Lorentzian broadening widths, sample temperature, and photon
    energy. These parameters are consumed by the spectral-function
    convolution and Fermi-Dirac weighting routines.

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``. All float-valued fields are
    stored as JAX array children (visible to autodiff), while the
    ``fidelity`` field is a plain Python ``int`` stored as auxiliary
    data because it controls array shapes and must be known at
    compile time.

    Attributes
    ----------
    energy_min : Float[Array, " "]
        Lower bound of energy window in eV.
    energy_max : Float[Array, " "]
        Upper bound of energy window in eV.
    fidelity : int
        Number of points along the energy axis.
    sigma : Float[Array, " "]
        Gaussian instrumental broadening width in eV.
    gamma : Float[Array, " "]
        Lorentzian lifetime broadening half-width in eV.
    temperature : Float[Array, " "]
        Sample temperature in Kelvin.
    photon_energy : Float[Array, " "]
        Incident photon energy in eV.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    The ``fidelity`` field is a Python ``int`` (not a JAX array) and
    is therefore stored as auxiliary data rather than as a child. JAX
    treats auxiliary data as a compile-time constant: changing
    ``fidelity`` triggers recompilation of any ``jit``-compiled
    function that receives this PyTree. This is intentional because
    ``fidelity`` determines the length of the energy axis array and
    JAX requires static shapes.

    See Also
    --------
    make_simulation_params : Factory function with validation and
        float64 casting.
    """

    energy_min: Float[Array, " "]
    energy_max: Float[Array, " "]
    fidelity: int
    sigma: Float[Array, " "]
    gamma: Float[Array, " "]
    temperature: Float[Array, " "]
    photon_energy: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        int,
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For SimulationParams,
        the six float-valued scalar fields are children and the
        ``fidelity`` Python int is auxiliary data.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(energy_min, energy_max, sigma, gamma, temperature,
           photon_energy)`` -- six 0-D float64 arrays.
        2. **Auxiliary data** (static, not traced by JAX):
           ``fidelity`` -- a Python ``int`` that controls energy-axis
           length. Stored as aux_data because JAX cannot trace Python
           ints and because shape-determining values must be static.

        Returns
        -------
        children : tuple of Array
            Tuple of six 0-D float64 JAX arrays.
        aux_data : int
            The ``fidelity`` value, stored outside JAX tracing.
        """
        return (
            (
                self.energy_min,
                self.energy_max,
                self.sigma,
                self.gamma,
                self.temperature,
                self.photon_energy,
            ),
            self.fidelity,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: int,
        children: Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "SimulationParams":
        """Reconstruct a SimulationParams from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into six 0-D float64 JAX arrays:
           ``(energy_min, energy_max, sigma, gamma, temperature,
           photon_energy)``.
        2. Receive ``aux_data`` as the ``fidelity`` Python int.
        3. Pass all seven fields to the constructor, restoring
           ``fidelity`` from ``aux_data`` into its correct position.

        Parameters
        ----------
        aux_data : int
            The ``fidelity`` value recovered from auxiliary data.
        children : tuple of Array
            Tuple of six 0-D float64 JAX arrays.

        Returns
        -------
        params : SimulationParams
            Reconstructed instance with identical data.
        """
        (
            energy_min,
            energy_max,
            sigma,
            gamma,
            temperature,
            photon_energy,
        ) = children
        return cls(
            energy_min=energy_min,
            energy_max=energy_max,
            fidelity=aux_data,
            sigma=sigma,
            gamma=gamma,
            temperature=temperature,
            photon_energy=photon_energy,
        )


@register_pytree_node_class
class PolarizationConfig(NamedTuple):
    """PyTree for photon polarization geometry.

    Describes the photon polarization state and incidence geometry
    for an ARPES experiment. The three angular fields define the
    light direction and polarization orientation, while the string
    ``polarization_type`` selects among standard optics conventions
    (s-pol, p-pol, circular, arbitrary linear, or unpolarized).

    This class is registered as a JAX PyTree via
    ``@register_pytree_node_class``. The three float-valued angular
    fields are stored as JAX array children (visible to autodiff),
    while ``polarization_type`` is a Python string stored as
    auxiliary data because JAX cannot trace strings.

    Attributes
    ----------
    theta : Float[Array, " "]
        Incident angle from surface normal in radians.
    phi : Float[Array, " "]
        In-plane azimuthal angle in radians.
    polarization_angle : Float[Array, " "]
        Arbitrary linear polarization angle in radians.
    polarization_type : str
        One of LVP, LHP, RCP, LCP, LAP, unpolarized.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    The ``polarization_type`` field is a Python string and is
    therefore stored as auxiliary data rather than as a child. JAX
    treats auxiliary data as a compile-time constant: changing
    ``polarization_type`` triggers recompilation of any
    ``jit``-compiled function that receives this PyTree. This is
    intentional because the polarization type selects different
    code branches in the matrix-element calculation.

    See Also
    --------
    make_polarization_config : Factory function with validation and
        float64 casting.
    """

    theta: Float[Array, " "]
    phi: Float[Array, " "]
    polarization_angle: Float[Array, " "]
    polarization_type: str

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        str,
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates the PyTree into children (JAX-traced arrays) and
        auxiliary data (static Python values). For PolarizationConfig,
        the three angular fields are children and the
        ``polarization_type`` string is auxiliary data.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(theta, phi, polarization_angle)`` -- three 0-D float64
           arrays.
        2. **Auxiliary data** (static, not traced by JAX):
           ``polarization_type`` -- a Python string selecting the
           polarization mode. Stored as aux_data because JAX cannot
           trace strings and because the polarization type controls
           code-path selection at compile time.

        Returns
        -------
        children : tuple of Array
            Tuple of three 0-D float64 JAX arrays.
        aux_data : str
            The ``polarization_type`` string, stored outside JAX
            tracing.
        """
        return (
            (self.theta, self.phi, self.polarization_angle),
            self.polarization_type,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: str,
        children: Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "PolarizationConfig":
        """Reconstruct a PolarizationConfig from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into three 0-D float64 JAX arrays:
           ``(theta, phi, polarization_angle)``.
        2. Receive ``aux_data`` as the ``polarization_type`` string.
        3. Pass all four fields to the constructor, restoring
           ``polarization_type`` from ``aux_data`` into its correct
           position.

        Parameters
        ----------
        aux_data : str
            The ``polarization_type`` string recovered from auxiliary
            data.
        children : tuple of Array
            Tuple of three 0-D float64 JAX arrays.

        Returns
        -------
        config : PolarizationConfig
            Reconstructed instance with identical data.
        """
        theta, phi, polarization_angle = children
        return cls(
            theta=theta,
            phi=phi,
            polarization_angle=polarization_angle,
            polarization_type=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_simulation_params(
    energy_min: ScalarNumeric = -3.0,
    energy_max: ScalarNumeric = 1.0,
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
) -> SimulationParams:
    """Create a validated SimulationParams instance.

    Validates and normalises the inputs before constructing the
    SimulationParams PyTree. All float-valued parameters are cast to
    0-D float64 JAX arrays, while ``fidelity`` is kept as a plain
    Python int (it becomes auxiliary data in the PyTree). Sensible
    defaults are provided for a typical low-temperature ARPES
    experiment: the energy window defaults to [-3.0, 1.0] eV, which
    is wide enough for most valence-band spectra near the Fermi
    level. In higher-level simulation drivers the ``energy_min`` and
    ``energy_max`` defaults may be overridden based on the actual
    eigenband energy range supplied via an ``eigenbands`` parameter.

    Implementation Logic
    --------------------
    1. **Cast energy_min** to 0-D ``jnp.float64`` via
       ``jnp.asarray``.
    2. **Cast energy_max** to 0-D ``jnp.float64`` via
       ``jnp.asarray``.
    3. **Keep fidelity** as a Python ``int`` (not cast). It is
       stored as auxiliary data in the PyTree because it determines
       array shapes.
    4. **Cast sigma, gamma, temperature, photon_energy** each to
       0-D ``jnp.float64`` arrays via ``jnp.asarray``.
    5. **Construct** the ``SimulationParams`` NamedTuple from all
       seven fields and return it.

    Parameters
    ----------
    energy_min : ScalarNumeric, optional
        Lower energy bound in eV. Default is -3.0.
    energy_max : ScalarNumeric, optional
        Upper energy bound in eV. Default is 1.0.
    fidelity : int, optional
        Energy axis resolution. Default is 25000.
    sigma : ScalarFloat, optional
        Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening in eV. Default is 0.1.
    temperature : ScalarFloat, optional
        Temperature in Kelvin. Default is 15.0.
    photon_energy : ScalarFloat, optional
        Photon energy in eV. Default is 11.0.

    Returns
    -------
    params : SimulationParams
        Validated simulation parameters.

    See Also
    --------
    SimulationParams : The PyTree class constructed by this factory.
    """
    params: SimulationParams = SimulationParams(
        energy_min=jnp.asarray(energy_min, dtype=jnp.float64),
        energy_max=jnp.asarray(energy_max, dtype=jnp.float64),
        fidelity=fidelity,
        sigma=jnp.asarray(sigma, dtype=jnp.float64),
        gamma=jnp.asarray(gamma, dtype=jnp.float64),
        temperature=jnp.asarray(
            temperature, dtype=jnp.float64
        ),
        photon_energy=jnp.asarray(
            photon_energy, dtype=jnp.float64
        ),
    )
    return params


@jaxtyped(typechecker=beartype)
def make_polarization_config(
    theta: ScalarFloat = 0.7854,
    phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    polarization_type: str = "unpolarized",
) -> PolarizationConfig:
    """Create a validated PolarizationConfig instance.

    Validates and normalises the inputs before constructing the
    PolarizationConfig PyTree. All float-valued angular parameters
    are cast to 0-D float64 JAX arrays, while
    ``polarization_type`` is kept as a plain Python string (it
    becomes auxiliary data in the PyTree).

    The default incidence angle ``theta = 0.7854`` rad corresponds
    to pi/4 (approximately 45 degrees), a common experimental
    geometry that provides balanced sensitivity to both in-plane
    and out-of-plane orbital components.

    Implementation Logic
    --------------------
    1. **Cast theta** to 0-D ``jnp.float64`` via ``jnp.asarray``.
       Default is ``0.7854`` rad = pi/4 ~ 45 degrees.
    2. **Cast phi** to 0-D ``jnp.float64`` via ``jnp.asarray``.
       Default is ``0.0`` rad.
    3. **Cast polarization_angle** to 0-D ``jnp.float64`` via
       ``jnp.asarray``. Default is ``0.0`` rad.
    4. **Keep polarization_type** as a Python string (not cast).
       It is stored as auxiliary data in the PyTree because it
       selects code branches at compile time.
    5. **Construct** the ``PolarizationConfig`` NamedTuple from all
       four fields and return it.

    Parameters
    ----------
    theta : ScalarFloat, optional
        Incident angle in radians. Default is pi/4 ~ 0.7854 rad
        (45 degrees).
    phi : ScalarFloat, optional
        Azimuthal angle in radians. Default is 0.
    polarization_angle : ScalarFloat, optional
        Linear polarization angle in radians. Default is 0.
    polarization_type : str, optional
        Polarization type. Default is ``"unpolarized"``.

    Returns
    -------
    config : PolarizationConfig
        Validated polarization configuration.

    See Also
    --------
    PolarizationConfig : The PyTree class constructed by this
        factory.
    """
    config: PolarizationConfig = PolarizationConfig(
        theta=jnp.asarray(theta, dtype=jnp.float64),
        phi=jnp.asarray(phi, dtype=jnp.float64),
        polarization_angle=jnp.asarray(
            polarization_angle, dtype=jnp.float64
        ),
        polarization_type=polarization_type,
    )
    return config


__all__: list[str] = [
    "PolarizationConfig",
    "SimulationParams",
    "make_polarization_config",
    "make_simulation_params",
]
