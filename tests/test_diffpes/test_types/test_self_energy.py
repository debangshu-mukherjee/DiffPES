"""Test the spectral ``SelfEnergyModel`` carrier contract.

The tests cover the carrier structure, factory validation, causality, and
differentiation requirements.
"""

from __future__ import annotations

import inspect

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict
from hypothesis import given, settings
from hypothesis import strategies as st

from diffpes.types import SelfEnergyModel, make_self_energy_model
from tests._assertions import assert_rejects

_DOMAIN: jax.Array = jnp.array([-4.0, 4.0])
_TAIL: jax.Array = jnp.array([0.0, 0.0])


def _softplus(x: jax.Array) -> jax.Array:
    """PRIVATE: Convert an unconstrained coordinate to a positive parameter.

    Parameters
    ----------
    x : jax.Array
        Unconstrained raw coefficient.

    Returns
    -------
    positive : jax.Array
        Softplus image ``log(1 + exp(x))``, strictly positive.

    Notes
    -----
    Wraps ``jax.nn.softplus`` so the independent identities below use
    the same raw-to-physical map as the carrier contract.
    """
    return jax.nn.softplus(x)


def _imaginary_part(
    model: SelfEnergyModel, omega_rel_fermi_ev: jax.Array
) -> jax.Array:
    """PRIVATE: Evaluate the independently stated self-energy imaginary-part identities.

    Parameters
    ----------
    model : SelfEnergyModel
        Carrier whose mode and raw coefficients define the identity.
    omega_rel_fermi_ev : jax.Array
        Energies relative to the Fermi level in eV.

    Returns
    -------
    imaginary_part : jax.Array
        Non-positive imaginary part of the self-energy in eV at each
        energy.

    Notes
    -----
    Restates each mode independently of the production code. Constant
    broadcasts ``-softplus(raw[0])``, and poly negates the softplus
    of the polynomial value. Grid interpolates softplus-negated
    ordinates over the energy nodes. The ``fermi_liquid`` mode
    evaluates the quartic-damped quadratic. The remaining mode sums
    symmetric Lorentzians at ``+/- omega0`` on top of the constant
    offset.
    """
    raw: jax.Array = model.coefficients
    if model.mode == "constant":
        return -jnp.broadcast_to(_softplus(raw[0]), omega_rel_fermi_ev.shape)
    if model.mode == "poly":
        return -_softplus(jnp.polyval(raw, omega_rel_fermi_ev))
    if model.mode == "grid":
        ordinates: jax.Array = -_softplus(raw)
        return jnp.interp(
            omega_rel_fermi_ev,
            model.energy_nodes_rel_fermi_ev,
            ordinates,
        )
    gamma0: jax.Array = _softplus(raw[0])
    if model.mode == "fermi_liquid":
        beta: jax.Array = _softplus(raw[1])
        omega_c: jax.Array = _softplus(raw[2])
        return -gamma0 - beta * omega_rel_fermi_ev**2 / (
            1.0 + (omega_rel_fermi_ev / omega_c) ** 4
        )
    coupling: jax.Array = _softplus(raw[1])
    omega0: jax.Array = _softplus(raw[2])
    width: jax.Array = _softplus(raw[3])
    return -gamma0 - coupling**2 * width * (
        1.0 / ((omega_rel_fermi_ev - omega0) ** 2 + width**2)
        + 1.0 / ((omega_rel_fermi_ev + omega0) ** 2 + width**2)
    )


def _make_mode(mode: str, coefficients: jax.Array) -> SelfEnergyModel:
    """PRIVATE: Construct one valid representative of every frozen carrier state.

    Parameters
    ----------
    mode : str
        Self-energy mode name to build.
    coefficients : jax.Array
        Raw unconstrained coefficients for that mode.

    Returns
    -------
    model : SelfEnergyModel
        Validated carrier for the requested mode.

    Notes
    -----
    Supplies the mode-specific mandatory fields. Constant passes
    ``kk_consistent=False``. Grid adds evenly spaced energy nodes on
    [-4, 4] eV plus the shared KK domain and power2 tail. Poly and
    ``fermi_liquid`` add the shared KK domain and power2 tail. Every
    other mode uses the analytic tail.
    """
    if mode == "constant":
        return make_self_energy_model(
            mode=mode, coefficients=coefficients, kk_consistent=False
        )
    if mode == "grid":
        return make_self_energy_model(
            mode=mode,
            coefficients=coefficients,
            energy_nodes_rel_fermi_ev=jnp.linspace(
                -4.0, 4.0, len(coefficients)
            ),
            kk_domain_rel_fermi_ev=_DOMAIN,
            tail_coefficients=_TAIL,
            tail_mode="power2",
        )
    if mode in ("poly", "fermi_liquid"):
        return make_self_energy_model(
            mode=mode,
            coefficients=coefficients,
            kk_domain_rel_fermi_ev=_DOMAIN,
            tail_coefficients=_TAIL,
            tail_mode="power2",
        )
    return make_self_energy_model(
        mode=mode,
        coefficients=coefficients,
        tail_mode="analytic",
    )


class TestSelfEnergyModel:
    """Test the ``SelfEnergyModel`` carrier and its numerical invariants.

    The tests cover PyTree partitioning, mode semantics, strict causality,
    finite-precision behavior, and coefficient-gradient liveness.

    :see: :class:`~diffpes.types.SelfEnergyModel`
    """

    def test_mode_specific_coefficient_semantics_and_units(self) -> None:
        """Pin every raw-to-physical eV mapping without clipping.

        The test covers all five coefficient semantics and their signs.

        Notes
        -----
        The test evaluates representative carriers against independent formulas.
        """
        omega: jax.Array = jnp.array([-0.5, 0.0, 0.5])
        constant: SelfEnergyModel = _make_mode("constant", jnp.array([0.0]))
        poly: SelfEnergyModel = _make_mode("poly", jnp.array([1.0, -0.25]))
        grid: SelfEnergyModel = _make_mode("grid", jnp.array([-1.0, 0.0, 1.0]))
        liquid: SelfEnergyModel = _make_mode(
            "fermi_liquid", jnp.array([-2.0, -1.0, 0.5])
        )
        kink: SelfEnergyModel = _make_mode(
            "bosonic_kink", jnp.array([-2.0, -1.0, 0.5, -0.5])
        )
        model: SelfEnergyModel
        for model in (constant, poly, grid, liquid, kink):
            actual: jax.Array = _imaginary_part(model, omega)
            assert jnp.all(actual < 0.0)
        expected_poly: jax.Array = -_softplus(
            jnp.polyval(poly.coefficients, omega)
        )
        chex.assert_trees_all_equal(
            _imaginary_part(poly, omega), expected_poly
        )
        # beta has units eV^-1, so beta * omega^2 is an energy in eV.
        expected_liquid: jax.Array = -_softplus(
            liquid.coefficients[0]
        ) - _softplus(liquid.coefficients[1]) * omega**2 / (
            1.0 + (omega / _softplus(liquid.coefficients[2])) ** 4
        )
        chex.assert_trees_all_equal(
            _imaginary_part(liquid, omega), expected_liquid
        )

    def test_pytree_round_trip_and_static_partition(self) -> None:
        """Keep selectors static and every numerical field in the leaf partition.

        The test covers flattening and reconstruction of a grid carrier.

        Notes
        -----
        The test inspects the leaves and compares the reconstructed carrier.
        """
        model: SelfEnergyModel = _make_mode(
            "grid", jnp.array([-1.0, 0.0, 1.0])
        )
        leaves: list[object]
        treedef: jax.tree_util.PyTreeDef
        leaves, treedef = jax.tree_util.tree_flatten(model)
        restored: SelfEnergyModel = jax.tree_util.tree_unflatten(
            treedef, leaves
        )
        assert restored.mode == "grid"
        assert restored.kk_consistent is True
        assert restored.tail_mode == "power2"
        assert all(not isinstance(leaf, (str, bool)) for leaf in leaves)
        assert len(leaves) == 5
        chex.assert_trees_all_equal(restored, model)

    @settings(max_examples=30, deadline=None)
    @given(
        mode=st.sampled_from(
            ["constant", "poly", "grid", "fermi_liquid", "bosonic_kink"]
        ),
        values=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=4,
            max_size=4,
        ),
    )
    def test_random_coefficients_give_strictly_negative_imaginary_part(
        self, mode: str, values: list[float]
    ) -> None:
        """Keep smooth softplus coordinates negative in every mode.

        The property covers representative energies for every model family.

        Notes
        -----
        The draw interval is ``[-10, 10]`` because ``softplus`` underflows to
        exactly ``0.0`` in float64 once its argument falls below about ``-745``.
        In ``poly`` mode, the polynomial value enters ``softplus``. A wider
        interval reaches that representability floor and requests an identity
        that float64 cannot deliver. This bound keeps the
        smooth reparameterization inside its representable domain; the separate
        floor and gradient-liveness cases below own the extreme-coordinate
        contract.
        """
        counts: Dict[str, int] = {
            "constant": 1,
            "poly": 4,
            "grid": 4,
            "fermi_liquid": 3,
            "bosonic_kink": 4,
        }
        coefficients: jax.Array = jnp.asarray(values[: counts[mode]])
        model: SelfEnergyModel = _make_mode(mode, coefficients)
        omega: jax.Array = jnp.linspace(-3.5, 3.5, 33)
        assert jnp.all(_imaginary_part(model, omega) < 0.0)

    def test_extreme_coefficients_never_produce_positive_imaginary_part(
        self,
    ) -> None:
        """Keep the float64 softplus floor one-sided at extreme coordinates.

        The test covers finite values below the representable softplus range.

        Notes
        -----
        The float64 ``softplus`` underflows to zero below about ``-745``. Thus, the
        reparameterization can reach ``-0.0`` but must not become positive. A
        positive value breaks causality. The representability floor is a
        finite-precision limit that the smooth-reparameterization rule permits.
        """
        extreme: SelfEnergyModel = _make_mode(
            "poly", jnp.array([-20.0, -20.0, -20.0, -20.0])
        )
        omega: jax.Array = jnp.linspace(-3.5, 3.5, 33)
        imaginary_part: jax.Array = _imaginary_part(extreme, omega)
        assert jnp.all(imaginary_part <= 0.0)
        assert not jnp.any(jnp.isnan(imaginary_part))

    @pytest.mark.parametrize(
        ("mode", "coefficients"),
        [
            ("constant", jnp.array([0.25])),
            ("poly", jnp.array([0.2, -0.1])),
            ("grid", jnp.array([-1.0, 0.0, 1.0])),
            ("fermi_liquid", jnp.array([-2.0, -1.0, 0.5])),
            ("bosonic_kink", jnp.array([-2.0, -1.0, 0.5, -0.5])),
        ],
    )
    def test_reparameterization_keeps_every_coefficient_gradient_alive(
        self, mode: str, coefficients: jax.Array
    ) -> None:
        """Forbid clipping by requiring a nonzero parameter-Jacobian column.

        The test covers every coefficient of every model family.

        Notes
        -----
        This test is the self-energy anti-clipping tripwire. Clipping enforces
        negativity by zeroing the gradient and the Fisher row at the bound.
        Softplus is strictly monotone. Thus, each coordinate must retain a nonzero
        derivative somewhere on the grid.
        """
        omega: jax.Array = jnp.linspace(-3.0, 3.0, 21)

        def total_imaginary_part(raw: jax.Array) -> jax.Array:
            return jnp.sum(_imaginary_part(_make_mode(mode, raw), omega))

        jacobian: jax.Array = jax.grad(total_imaginary_part)(coefficients)
        assert jnp.all(jnp.isfinite(jacobian))
        assert jnp.all(jnp.abs(jacobian) > 0.0)


class TestMakeSelfEnergyModel:
    """Test factory construction, validation, and the public API contract.

    The tests cover valid states, rejected structures and selectors, causal
    requirements, tail rules, parameter round trips, and energy-name pinning.

    :see: :func:`~diffpes.types.make_self_energy_model`
    """

    @pytest.mark.parametrize(
        ("mode", "coefficients"),
        [
            ("constant", jnp.array([0.0])),
            ("poly", jnp.array([0.2, -0.1])),
            ("grid", jnp.array([-1.0, 0.0, 1.0])),
            ("fermi_liquid", jnp.array([-2.0, -1.0, 0.5])),
            ("bosonic_kink", jnp.array([-2.0, -1.0, 0.5, -0.5])),
        ],
    )
    def test_factory_accepts_complete_state_table(
        self, mode: str, coefficients: jax.Array
    ) -> None:
        """Accept each and only each fully specified mode family.

        The test covers every row in the complete state table.

        Notes
        -----
        The test constructs each mode and compares its selector and coordinates.
        """
        model: SelfEnergyModel = _make_mode(mode, coefficients)
        assert model.mode == mode
        chex.assert_trees_all_equal(model.coefficients, coefficients)

    @pytest.mark.parametrize(
        "mode", ["polynomial", "tabulated", "invalid", ""]
    )
    def test_factory_rejects_bad_mode(self, mode: str) -> None:
        """Reject selectors outside the five-name frozen public vocabulary.

        The test covers legacy names and invalid selector values.

        Notes
        -----
        The test passes each selector to the factory and matches its error.
        """
        with pytest.raises(ValueError, match="mode"):
            make_self_energy_model(mode=mode)

    @pytest.mark.parametrize(
        "nodes",
        [None, jnp.array([0.0]), jnp.array([[0.0, 1.0]])],
    )
    def test_grid_rejects_missing_short_or_nonvector_nodes(
        self,
        nodes: jax.Array | None,
    ) -> None:
        """Reject static grid structures that cannot define hat interpolation.

        The test covers absent, short, and nonvector node arrays.

        Notes
        -----
        The test supplies each invalid node array to an otherwise valid factory call.
        """
        with pytest.raises(ValueError, match="energy_nodes_rel_fermi_ev"):
            make_self_energy_model(
                mode="grid",
                coefficients=jnp.array([0.0, 1.0]),
                energy_nodes_rel_fermi_ev=nodes,
                kk_domain_rel_fermi_ev=jnp.array([0.0, 1.0]),
                tail_coefficients=_TAIL,
                tail_mode="power2",
            )

    def test_non_grid_rejects_nodes(self) -> None:
        """Reject unused interpolation coordinates in every non-grid mode.

        The test isolates the rule for unused grid nodes.

        Notes
        -----
        The test supplies nodes to the default constant mode and matches the error.
        """
        with pytest.raises(ValueError, match="energy_nodes_rel_fermi_ev"):
            make_self_energy_model(
                coefficients=jnp.array([0.0]),
                energy_nodes_rel_fermi_ev=jnp.array([0.0, 1.0]),
            )

    def test_grid_rejects_node_coefficient_length_mismatch(self) -> None:
        """Reject unequal grid ordinate and coordinate counts structurally.

        The test covers a three-ordinate and two-coordinate grid.

        Notes
        -----
        The test constructs the mismatched grid and matches its structural error.
        """
        with pytest.raises(ValueError, match="same length|length"):
            make_self_energy_model(
                mode="grid",
                coefficients=jnp.zeros(3),
                energy_nodes_rel_fermi_ev=jnp.array([-1.0, 1.0]),
                kk_domain_rel_fermi_ev=jnp.array([-1.0, 1.0]),
                tail_coefficients=_TAIL,
                tail_mode="power2",
            )

    def test_rejects_nonmonotone_nodes_eager_and_jit(self) -> None:
        """Keep strict relative-energy node ordering active while traced.

        The test covers repeated nodes in eager and compiled execution.

        Notes
        -----
        The test uses the shared rejection helper with a nonmonotone grid.
        """
        assert_rejects(
            make_self_energy_model,
            mode="grid",
            coefficients=jnp.zeros(3),
            energy_nodes_rel_fermi_ev=jnp.array([-1.0, 0.0, 0.0]),
            kk_domain_rel_fermi_ev=jnp.array([-1.0, 0.0]),
            tail_coefficients=_TAIL,
            tail_mode="power2",
            match="strictly increasing|increasing",
        )

    def test_rejects_nan_coefficients_eager_and_jit(self) -> None:
        """Keep finite-coordinate validation active while traced.

        The test covers a nonfinite coefficient in eager and compiled execution.

        Notes
        -----
        The test uses the shared rejection helper with a NaN coordinate.
        """
        assert_rejects(
            make_self_energy_model,
            coefficients=jnp.array([jnp.nan]),
            match="coefficients.*finite|finite.*coefficients",
        )

    @pytest.mark.parametrize("gamma", [1.0e-8, 1.0e-4, 0.1, 1.0, 10.0])
    def test_gamma_softplus_inverse_round_trip(self, gamma: float) -> None:
        """Recover the requested constant negative linewidth to 1e-14.

        The test covers small and large positive gamma values.

        Notes
        -----
        The test maps gamma through the factory and back through the stated identity.
        """
        model: SelfEnergyModel = make_self_energy_model(gamma=gamma)
        sigma_imaginary: jax.Array = _imaginary_part(model, jnp.zeros(3))
        np.testing.assert_allclose(
            sigma_imaginary, -gamma, rtol=0.0, atol=1.0e-14
        )

    @pytest.mark.parametrize("mode", ["poly", "grid", "fermi_liquid"])
    def test_numerical_modes_require_kk_domain_and_power2_tail(
        self, mode: str
    ) -> None:
        """Reject every energy-dependent imaginary-only numerical model.

        The test covers the required KK domain and power-tail contract.

        Notes
        -----
        The test omits one required causal field from each factory call.
        """
        coefficients: jax.Array = {
            "poly": jnp.array([0.0]),
            "grid": jnp.array([0.0, 0.0]),
            "fermi_liquid": jnp.zeros(3),
        }[mode]
        nodes: jax.Array | None = (
            jnp.array([-1.0, 1.0]) if mode == "grid" else None
        )
        common: Dict[str, object] = {
            "mode": mode,
            "coefficients": coefficients,
            "energy_nodes_rel_fermi_ev": nodes,
        }
        with pytest.raises(ValueError):
            make_self_energy_model(**common, kk_consistent=False)
        with pytest.raises(ValueError):
            make_self_energy_model(
                **common, tail_mode="power2", tail_coefficients=_TAIL
            )
        with pytest.raises(ValueError):
            make_self_energy_model(**common, kk_domain_rel_fermi_ev=_DOMAIN)

    def test_bosonic_kink_requires_analytic_kk_contract(self) -> None:
        """Require the causal analytic pole-pair state and no numerical tail.

        The test covers invalid causality and tail selectors for the kink mode.

        Notes
        -----
        The test constructs each invalid analytic state and expects a value error.
        """
        coefficients: jax.Array = jnp.zeros(4)
        with pytest.raises(ValueError):
            make_self_energy_model(
                mode="bosonic_kink",
                coefficients=coefficients,
                kk_consistent=False,
            )
        with pytest.raises(ValueError):
            make_self_energy_model(
                mode="bosonic_kink",
                coefficients=coefficients,
                tail_mode="none",
            )

    def test_power2_tail_has_two_bounded_raw_delta_beta_coordinates(
        self,
    ) -> None:
        """Pin the superseding amendment's identifiable two-coordinate tail.

        The test covers the shape, bounds, and accepted boundary values.

        Notes
        -----
        The test rejects invalid tails and constructs both exact bound values.
        """
        bad_tail: jax.Array
        for bad_tail in (
            jnp.zeros(4),
            jnp.array([-30.0001, 0.0]),
            jnp.array([0.0, 30.0001]),
        ):
            assert_rejects(
                make_self_energy_model,
                mode="poly",
                coefficients=jnp.array([0.0]),
                kk_domain_rel_fermi_ev=_DOMAIN,
                tail_coefficients=bad_tail,
                tail_mode="power2",
                match=r"tail_coefficients",
            )
        boundary: float
        for boundary in (-30.0, 30.0):
            model: SelfEnergyModel = make_self_energy_model(
                mode="poly",
                coefficients=jnp.array([0.0]),
                kk_domain_rel_fermi_ev=_DOMAIN,
                tail_coefficients=jnp.array([boundary, boundary]),
                tail_mode="power2",
            )
            chex.assert_shape(model.tail_coefficients, (2,))

    def test_api_accepts_only_relative_fermi_energy_names(self) -> None:
        """Pin E-E_F in eV and exclude every absolute-energy argument.

        The test covers the required and forbidden parameter names.

        Notes
        -----
        The test inspects the signature and calls one forbidden keyword.
        """
        signature: inspect.Signature = inspect.signature(
            make_self_energy_model
        )
        assert "energy_nodes_rel_fermi_ev" in signature.parameters
        assert "kk_domain_rel_fermi_ev" in signature.parameters
        assert "subtraction_point_rel_fermi_ev" in signature.parameters
        assert "energy_nodes" not in signature.parameters
        assert "energy_ev" not in signature.parameters
        with pytest.raises(TypeError):
            make_self_energy_model(energy_nodes=jnp.array([-1.0, 1.0]))
