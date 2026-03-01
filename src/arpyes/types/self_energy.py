"""Self-energy configuration data structures.

Extended Summary
----------------
Defines the PyTree type for energy-dependent self-energy
(lifetime broadening) used by the differentiable forward simulator.

Routine Listings
----------------
:class:`SelfEnergyConfig`
    PyTree for energy-dependent self-energy.
:func:`make_self_energy_config`
    Factory for SelfEnergyConfig.

Notes
-----
The self-energy coefficients are differentiable (JAX children)
while the mode string is static (auxiliary data) because it
selects different code branches.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat


@register_pytree_node_class
class SelfEnergyConfig(NamedTuple):
    """PyTree for energy-dependent self-energy (lifetime broadening).

    Extended Summary
    ----------------
    Models the imaginary part of the electronic self-energy
    Im[Sigma(E)] as a function of binding energy. In the forward
    ARPES simulator this replaces the scalar Lorentzian half-width
    ``gamma`` with an energy-dependent broadening that captures
    quasiparticle lifetime effects more realistically.

    Three modes are supported, selected by the ``mode`` string:

    - **constant**: a single scalar broadening ``gamma`` applied
      uniformly at all energies. Equivalent to the standard
      ``SimulationParams.gamma``.
    - **polynomial**: Im[Sigma(E)] = a0 + a1*E + a2*E^2 + ...,
      with coefficients stored in ``coefficients``.
    - **tabulated**: Im[Sigma] is specified at discrete energy nodes
      and interpolated between them. Requires ``energy_nodes``.

    The ``coefficients`` array is the primary differentiable
    quantity: ``jax.grad`` with respect to these coefficients gives
    the sensitivity of the simulated ARPES spectrum to the
    self-energy shape, enabling inverse fitting of lifetime
    broadening from experimental data.

    Attributes
    ----------
    coefficients : Float[Array, " P"]
        Parameters for the self-energy model. JAX-traced
        (differentiable).

        - mode="constant": P=1, ``[gamma]`` in eV.
        - mode="polynomial": P=degree+1, ``[a0, a1, ...]`` where
          Im[Sigma(E)] = sum_i a_i * E^i.
        - mode="tabulated": P=N, ``[gamma_1, ..., gamma_N]`` in eV
          at the corresponding ``energy_nodes``.
    energy_nodes : Optional[Float[Array, " P"]]
        Energy grid (eV) for tabulated mode. Must have the same
        length P as ``coefficients`` in tabulated mode. ``None``
        for constant and polynomial modes. JAX-traced when present.
    mode : str
        One of ``"constant"``, ``"polynomial"``, ``"tabulated"``.
        Stored as auxiliary data (static) because it selects
        different code branches in the forward simulator.

    Notes
    -----
    Registered as a JAX PyTree with ``@register_pytree_node_class``.
    ``coefficients`` and ``energy_nodes`` are children (on the
    gradient tape); ``mode`` is auxiliary data (compile-time
    constant). Changing ``mode`` triggers JIT recompilation because
    it alters the computation graph.

    See Also
    --------
    make_self_energy_config : Factory function with mode validation
        and default coefficient generation.
    """

    coefficients: Float[Array, " P"]
    energy_nodes: Optional[Float[Array, " P"]]
    mode: str

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, " P"], Optional[Float[Array, " P"]]],
        str,
    ]:
        """Flatten into JAX children and auxiliary data.

        Separates JAX-traced arrays (children) from static Python
        values (auxiliary data) for ``jax.tree_util`` compatibility.

        Implementation Logic
        --------------------
        1. **Children** (JAX arrays, participate in autodiff):
           ``(coefficients, energy_nodes)`` -- both are either dense
           JAX arrays or ``None``. ``coefficients`` is always present;
           ``energy_nodes`` is ``None`` for constant and polynomial
           modes. JAX treats ``None`` leaves as empty subtrees.
        2. **Auxiliary data** (static, not traced by JAX):
           ``mode`` -- a Python string selecting the self-energy
           model. Stored as aux_data because it controls code-path
           selection at compile time.

        Returns
        -------
        children : tuple of (jax.Array or None)
            ``(coefficients, energy_nodes)``.
        aux_data : str
            The ``mode`` string, stored outside JAX tracing.
        """
        return ((self.coefficients, self.energy_nodes), self.mode)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: str,
        children: Tuple[Float[Array, " P"], Optional[Float[Array, " P"]]],
    ) -> "SelfEnergyConfig":
        """Reconstruct a ``SelfEnergyConfig`` from flattened components.

        Inverse of :meth:`tree_flatten`. JAX calls this method
        automatically when unflattening a PyTree after a
        transformation (e.g., inside ``jax.jit`` or ``jax.grad``).

        Implementation Logic
        --------------------
        1. Unpack ``children`` into ``(coefficients, energy_nodes)``.
        2. Receive ``aux_data`` as the ``mode`` string.
        3. Pass all three fields to the constructor, restoring
           ``mode`` from ``aux_data``.

        Parameters
        ----------
        aux_data : str
            The ``mode`` string recovered from auxiliary data.
        children : tuple of (jax.Array or None)
            ``(coefficients, energy_nodes)`` as returned by
            :meth:`tree_flatten`.

        Returns
        -------
        config : SelfEnergyConfig
            Reconstructed instance with identical data.
        """
        coefficients, energy_nodes = children
        return cls(
            coefficients=coefficients,
            energy_nodes=energy_nodes,
            mode=aux_data,
        )


@jaxtyped(typechecker=beartype)
def make_self_energy_config(
    gamma: ScalarFloat = 0.1,
    mode: str = "constant",
    coefficients: Optional[Float[Array, " P"]] = None,
    energy_nodes: Optional[Float[Array, " P"]] = None,
) -> SelfEnergyConfig:
    """Create a validated ``SelfEnergyConfig`` instance.

    Extended Summary
    ----------------
    Factory function that validates inputs and constructs a
    ``SelfEnergyConfig`` PyTree. Performs mode validation (only
    ``"constant"``, ``"polynomial"``, ``"tabulated"`` are accepted)
    and ensures that ``energy_nodes`` is provided when required by
    the tabulated mode.

    The convenience parameter ``gamma`` provides a shortcut for the
    common constant-broadening case: when ``coefficients`` is
    ``None``, a single-element array ``[gamma]`` is created
    automatically.

    Implementation Logic
    --------------------
    1. **Validate mode**: raise ``ValueError`` if ``mode`` is not one
       of the three supported strings.
    2. **Default coefficients**: if ``coefficients`` is ``None``,
       create ``jnp.asarray([gamma], dtype=jnp.float64)`` -- a
       single-element array holding the constant broadening.
       Otherwise cast the provided array to ``jnp.float64``.
    3. **Cast energy_nodes**: if provided, cast to ``jnp.float64``;
       otherwise leave as ``None``.
    4. **Validate tabulated mode**: raise ``ValueError`` if
       ``mode == "tabulated"`` but ``energy_nodes`` is ``None``,
       because interpolation requires an energy grid.
    5. **Construct** the ``SelfEnergyConfig`` NamedTuple and return.

    Parameters
    ----------
    gamma : ScalarFloat, optional
        Constant broadening in eV. Used as the sole coefficient when
        ``coefficients`` is ``None`` and mode is ``"constant"``.
        Default is 0.1.
    mode : str, optional
        Self-energy model. One of ``"constant"``, ``"polynomial"``,
        ``"tabulated"``. Default is ``"constant"``.
    coefficients : Float[Array, " P"], optional
        Explicit model coefficients. If ``None``, defaults to
        ``[gamma]``. For polynomial mode, these are the polynomial
        coefficients ``[a0, a1, ...]``. For tabulated mode, these
        are the broadening values at each energy node.
    energy_nodes : Float[Array, " P"], optional
        Energy grid (eV) for tabulated mode. Must have the same
        length as ``coefficients`` in tabulated mode. Ignored for
        other modes. Default is ``None``.

    Returns
    -------
    config : SelfEnergyConfig
        Validated self-energy configuration with ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three supported strings, or if
        ``mode == "tabulated"`` and ``energy_nodes`` is ``None``.

    See Also
    --------
    SelfEnergyConfig : The PyTree class constructed by this factory.
    """
    if mode not in ("constant", "polynomial", "tabulated"):
        msg = (
            "mode must be 'constant', 'polynomial',"
            f" or 'tabulated', got '{mode}'"
        )
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr = jnp.asarray([gamma], dtype=jnp.float64)
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    nodes_arr: Optional[Float[Array, " P"]] = None
    if energy_nodes is not None:
        nodes_arr = jnp.asarray(energy_nodes, dtype=jnp.float64)
    if mode == "tabulated" and nodes_arr is None:
        msg = "energy_nodes required for tabulated mode"
        raise ValueError(msg)
    return SelfEnergyConfig(
        coefficients=coeff_arr,
        energy_nodes=nodes_arr,
        mode=mode,
    )


__all__: list[str] = [
    "SelfEnergyConfig",
    "make_self_energy_config",
]
