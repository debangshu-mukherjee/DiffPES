"""Validate the VASP OUTCAR summary carrier and its factory.

Covers field storage, float64 casting, and rejection of non-finite or
nonpositive values.
"""

import chex
import equinox as eqx
import jax.numpy as jnp
import pytest

from diffpes.types import (
    OutcarData,
    make_outcar_data,
)


class TestOutcarData(chex.TestCase):
    """Validate :class:`diffpes.types.OutcarData`.

    Verifies that the carrier stores the Fermi energy and the electron
    count as traced float64 scalars.

    :see: :class:`~diffpes.types.OutcarData`
    """

    def test_stores_scalar_fields(self) -> None:
        """Store both summary scalars on the carrier.

        The test builds a carrier through the factory and checks that
        both fields keep their values and the float64 dtype.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        summary: OutcarData

        summary = make_outcar_data(fermi_energy=2.39, nelect=84.0)
        chex.assert_shape(summary.fermi_energy, ())
        chex.assert_shape(summary.nelect, ())
        assert summary.fermi_energy.dtype == jnp.float64
        assert summary.nelect.dtype == jnp.float64
        chex.assert_trees_all_close(
            summary.fermi_energy, jnp.float64(2.39), atol=1e-12
        )
        chex.assert_trees_all_close(
            summary.nelect, jnp.float64(84.0), atol=1e-12
        )


class TestMakeOutcarData(chex.TestCase):
    """Validate :func:`diffpes.types.make_outcar_data`.

    :see: :func:`~diffpes.types.make_outcar_data`
    """

    def test_negative_fermi_energy_is_valid(self) -> None:
        """Accept a negative Fermi energy.

        VASP reports negative Fermi energies for some reference choices.
        The factory keeps the sign.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        summary: OutcarData

        summary = make_outcar_data(fermi_energy=-1.5, nelect=10.0)
        chex.assert_trees_all_close(
            summary.fermi_energy, jnp.float64(-1.5), atol=1e-12
        )

    def test_nonpositive_nelect_raises(self) -> None:
        """Reject a nonpositive electron count.

        The factory binds a traced guard to the electron count. A zero
        count trips the guard.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="nelect must be finite and positive",
        ):
            make_outcar_data(fermi_energy=2.0, nelect=0.0)

    def test_non_finite_fermi_energy_raises(self) -> None:
        """Reject a non-finite Fermi energy.

        The factory binds a finiteness guard to the Fermi energy. A NaN
        value trips the guard.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated
        property with the documented numerical or structural assertions.
        """
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="fermi_energy must be finite",
        ):
            make_outcar_data(fermi_energy=float("nan"), nelect=8.0)
