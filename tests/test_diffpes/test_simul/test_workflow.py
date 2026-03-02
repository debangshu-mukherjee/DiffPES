"""Tests for high-level workflow helpers in :mod:`diffpes.simul.workflow`."""

from pathlib import Path

import chex
import jax.numpy as jnp

from diffpes.simul import (
    WorkflowContext,
    load_vasp_context,
    prepare_projection,
    run_vasp_workflow,
    simulate_context,
)
from diffpes.types import (
    make_band_structure,
    make_orbital_projection,
)

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parents[1] / "test_inout" / "fixtures"
)


class TestLoadVaspContext(chex.TestCase):
    """Tests for :func:`diffpes.simul.load_vasp_context`."""

    def test_loads_context_with_optional_dos_and_kpath(self):
        """Verify context loading with inferred Fermi level and checks."""
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
    """Tests for :func:`diffpes.simul.prepare_projection`."""

    def test_selects_atoms_and_attaches_oam(self):
        """Verify atom sub-selection and OAM attachment in one call."""
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


class TestSimulateContext(chex.TestCase):
    """Tests for :func:`diffpes.simul.simulate_context`."""

    def test_momentum_broadening_changes_output(self):
        """Verify nonzero dk changes simulated intensity."""
        nk: int = 12
        nb: int = 4
        na: int = 2
        eigenbands = jnp.linspace(
            -2.0, 0.6, nk * nb, dtype=jnp.float64
        ).reshape(nk, nb)
        kx = jnp.linspace(0.0, 1.0, nk, dtype=jnp.float64)
        kpoints = jnp.stack(
            [kx, jnp.zeros_like(kx), jnp.zeros_like(kx)],
            axis=1,
        )
        projections = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        projections = projections.at[:, :, :, 4:9].set(0.3)

        bands = make_band_structure(
            eigenvalues=eigenbands,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        orb = make_orbital_projection(projections=projections)
        context = WorkflowContext(
            bands=bands,
            orb_proj=orb,
            kpath=None,
            dos=None,
        )

        base = simulate_context(
            context=context,
            level="basic",
            fidelity=320,
            sigma=0.05,
            temperature=20.0,
            photon_energy=35.0,
            normalize=False,
        )
        broadened = simulate_context(
            context=context,
            level="basic",
            fidelity=320,
            sigma=0.05,
            temperature=20.0,
            photon_energy=35.0,
            dk=0.06,
            normalize=False,
        )

        chex.assert_shape(base.intensity, (nk, 320))
        chex.assert_shape(broadened.intensity, (nk, 320))
        assert not jnp.allclose(base.intensity, broadened.intensity)


class TestRunVaspWorkflow(chex.TestCase):
    """Tests for :func:`diffpes.simul.run_vasp_workflow`."""

    def test_runs_end_to_end_with_normalization(self):
        """Verify one-call workflow runs and returns normalized spectrum."""
        spectrum = run_vasp_workflow(
            level="basic",
            directory=str(_FIXTURES_DIR),
            eigenval_file="EIGENVAL_spin",
            procar_file="PROCAR",
            doscar_file="DOSCAR",
            kpoints_file="KPOINTS_line_fallback",
            fidelity=180,
            sigma=0.05,
            temperature=15.0,
            photon_energy=11.0,
            normalize=True,
            check_dimensions=True,
        )
        chex.assert_shape(spectrum.intensity, (2, 180))
        chex.assert_shape(spectrum.energy_axis, (180,))
        mean_val = jnp.mean(spectrum.intensity)
        chex.assert_trees_all_close(mean_val, jnp.float64(0.0), atol=1e-8)
