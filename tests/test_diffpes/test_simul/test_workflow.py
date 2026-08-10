"""Validate high-level workflow helpers in :mod:`diffpes.simul.workflow`.

Extended Summary
----------------
Validates VASP context loading and projection preparation with controlled
temporary inputs.
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import diffpes
from diffpes.simul import (
    load_vasp_context,
    prepare_projection,
)
from diffpes.types import (
    SpinOrbitalProjection,
    make_orbital_projection,
    make_spin_orbital_projection,
)

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parents[1] / "test_inout" / "fixtures"
)


class TestLoadVaspContextEdgeCases(chex.TestCase):
    """Validate additional paths in :func:`diffpes.simul.load_vasp_context`.

    :see: :func:`~diffpes.simul.load_vasp_context`
    """

    def test_no_doscar_fermi_defaults_to_zero(self) -> None:
        """Verify Fermi energy is 0.0 when doscar_file=None and fermi_energy=None.

        The test passes ``doscar_file=None`` and ``fermi_energy=None``, exercising
        workflow.py line 142 (``resolved_fermi = 0.0``). Asserts the
        returned band structure has fermi_energy == 0.0.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file=None,
            kpoints_file=None,
            fermi_energy=None,
            check_dimensions=True,
        )
        chex.assert_trees_all_close(
            context.bands.fermi_energy, jnp.float64(0.0), atol=1e-12
        )
        assert context.dos is None

    def test_missing_doscar_raises(self) -> None:
        """Verify that a missing required DOSCAR raises FileNotFoundError.

        The test passes a non-existent ``doscar_file`` with ``fermi_energy=None``,
        exercising workflow.py lines 146-150 (FileNotFoundError path).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        with pytest.raises(FileNotFoundError):
            load_vasp_context(
                directory=str(_FIXTURES_DIR),
                eigenval_file="EIGENVAL_spin",
                procar_file="PROCAR_spin",
                doscar_file="DOES_NOT_EXIST",
                kpoints_file=None,
                fermi_energy=None,
                check_dimensions=False,
            )

    def test_explicit_fermi_reads_doscar_optionally(self) -> None:
        """Verify optional DOSCAR loading with an explicit Fermi energy.

        The test passes ``fermi_energy=1.5`` and a valid ``doscar_file``, exercising
        workflow.py lines 154-158 (optional DOSCAR read). The explicit
        Fermi energy controls the bands, and the file supplies ``dos``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file="DOSCAR",
            kpoints_file=None,
            fermi_energy=1.5,
            check_dimensions=True,
        )
        chex.assert_trees_all_close(
            context.bands.fermi_energy, jnp.float64(1.5), atol=1e-12
        )
        assert context.dos is not None


class TestLoadVaspContext(chex.TestCase):
    """Validate :func:`diffpes.simul.load_vasp_context`.

    :see: :func:`~diffpes.simul.load_vasp_context`
    """

    def test_loads_context_with_optional_dos_and_kpath(self) -> None:
        """Verify context loading with inferred Fermi level and checks.

        The test establishes the loads context with optional dos and kpath contract for
        load vasp context with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        context: diffpes.types.WorkflowContext

        context = load_vasp_context(
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR_spin",
            doscar_file="DOSCAR",
            kpoints_file="KPOINTS_line_fallback",
            procar_mode="full",
            check_dimensions=True,
        )
        chex.assert_shape(context.bands.eigenvalues, (2, 2))
        chex.assert_shape(context.orb_proj.projections, (2, 2, 1, 9))
        assert context.orb_proj.spin is not None
        assert context.kpath is not None
        assert context.dos is not None
        chex.assert_trees_all_close(
            context.bands.fermi_energy,
            jnp.float64(0.5),
            atol=1e-12,
        )


class TestPrepareProjection(chex.TestCase):
    """Validate :func:`diffpes.simul.prepare_projection`.

    :see: :func:`~diffpes.simul.prepare_projection`
    """

    def test_spin_orbital_projection_attaches_oam(self) -> None:
        """Verify OAM attachment works for SpinOrbitalProjection input.

        The test constructs a SpinOrbitalProjection and calls ``prepare_projection``
        with ``attach_oam=True``. Asserts the returned object is still a
        SpinOrbitalProjection with OAM attached, covering workflow.py
        line 224 (make_spin_orbital_projection with oam).

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        projections: Array
        spin: Array
        orb: diffpes.types.SpinOrbitalProjection
        prepared: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        projections = jnp.ones((2, 2, 2, 9), dtype=jnp.float64)
        spin = jnp.ones((2, 2, 2, 6), dtype=jnp.float64)
        orb = make_spin_orbital_projection(projections=projections, spin=spin)
        prepared = prepare_projection(orb_proj=orb, attach_oam=True)
        assert isinstance(prepared, SpinOrbitalProjection)
        assert prepared.oam is not None
        chex.assert_shape(prepared.oam, (2, 2, 2, 3))

    def test_selects_atoms_and_attaches_oam(self) -> None:
        """Verify atom sub-selection and OAM attachment in one call.

        The test establishes the selects atoms and attaches oam contract for prepare
        projection with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        projections: Array
        orb: diffpes.types.OrbitalProjection
        prepared: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        projections = jnp.ones((2, 2, 3, 9), dtype=jnp.float64)
        orb = make_orbital_projection(projections=projections)
        prepared = prepare_projection(
            orb_proj=orb,
            atom_indices=[0, 2],
            attach_oam=True,
        )
        chex.assert_shape(prepared.projections, (2, 2, 2, 9))
        assert prepared.oam is not None
        chex.assert_shape(prepared.oam, (2, 2, 2, 3))
