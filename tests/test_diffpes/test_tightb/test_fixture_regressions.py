r"""Validate analytic Rashba and projected-t2g fixture regressions.

The tests independently pin the closed square-lattice Rashba spectrum and
the isolated projected-t2g spin--orbit multiplets.  They also diagnose the
atomic :math:`\langle\mathbf L\cdot\mathbf S\rangle` values with fixed
degenerate groups in the declared down--up spin convention.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diffpes.tightb import (
    diagonalize_tb,
    eigvalsh_bands,
    expectation_path,
    group_trace,
    ls_operator,
)
from diffpes.types import DiagonalizedBands, TBModel
from tests._factories import make_rashba_model, make_t2g_soc_model


class TestRashbaFixtureRegression:
    """Validate the analytic square-lattice Rashba fixture."""

    def test_eigenvalues_match_closed_form_on_generic_and_symmetry_points(
        self,
    ) -> None:
        r"""Recover the two closed-form Rashba branches.

        The down--up Hamiltonian has scalar dispersion
        :math:`2t(\cos q_x+\cos q_y)` and spin splitting
        :math:`\lambda\sqrt{\sin^2q_x+\sin^2q_y}`, where
        :math:`q_i=2\pi k_i`.

        Notes
        -----
        Include Gamma, X, and M, where the Rashba splitting vanishes, plus
        generic points that exercise both in-plane spin components.
        """
        hopping: float = -0.63
        rashba: float = 0.27
        kpoints: Array = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (0.125, 0.25, 0.0),
                (-0.18, 0.31, 0.17),
            ),
            dtype=jnp.float64,
        )
        model: TBModel = make_rashba_model(hopping, rashba)
        actual: Array = eigvalsh_bands(model, kpoints)

        fractional: np.ndarray = np.asarray(kpoints)
        qx: np.ndarray = 2.0 * np.pi * fractional[:, 0]
        qy: np.ndarray = 2.0 * np.pi * fractional[:, 1]
        center: np.ndarray = 2.0 * hopping * (np.cos(qx) + np.cos(qy))
        splitting: np.ndarray = abs(rashba) * np.sqrt(
            np.sin(qx) ** 2 + np.sin(qy) ** 2
        )
        expected: np.ndarray = np.stack(
            (center - splitting, center + splitting),
            axis=-1,
        )

        assert model.basis.spin == (-1, 1)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=2e-14,
        )


class TestT2gSocFixtureRegression:
    """Validate the analytic projected-t2g spin--orbit fixture."""

    def test_projected_t2g_multiplet_matches_effective_l_one_truth(
        self,
    ) -> None:
        r"""Recover the fourfold and twofold projected-t2g levels.

        Projecting the physical d-shell angular momentum into t2g gives the
        negative effective-l-one representation.  Positive lambda therefore
        places the j-effective three-halves quartet at ``-lambda/2`` and the
        one-half doublet at ``+lambda``.

        Notes
        -----
        Several k-points confirm that the isolated atomic fixture has no
        accidental dispersion.
        """
        coupling: float = 0.37
        kpoints: Array = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.23, -0.19, 0.41),
            ),
            dtype=jnp.float64,
        )
        model: TBModel = make_t2g_soc_model(coupling)
        actual: Array = eigvalsh_bands(model, kpoints)
        expected_row: np.ndarray = np.asarray(
            (-0.5 * coupling,) * 4 + (coupling,) * 2,
        )
        expected: np.ndarray = np.broadcast_to(
            expected_row,
            actual.shape,
        )

        assert model.basis.spin == (-1, -1, -1, 1, 1, 1)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=2e-14,
        )

    def test_ls_diagnostic_uses_fixed_degenerate_group_traces(self) -> None:
        r"""Recover analytic t2g :math:`\langle\mathbf L\cdot\mathbf S\rangle`.

        In explicit down--up block order, the lower quartet has expectation
        ``-1/2`` per state and the upper doublet has expectation ``+1`` per
        state.  Their fixed-group traces are consequently ``-2`` and ``+2``.

        Notes
        -----
        Fixed band groups make the traces independent of the arbitrary
        eigensolver basis within either exactly degenerate multiplet.
        Diagnostic per-band expectations must equal the corresponding
        fixed-group trace divided by the registered multiplicity.
        """
        model: TBModel = make_t2g_soc_model(coupling=0.4)
        kpoints: Array = jnp.asarray(
            ((0.0, 0.0, 0.0), (0.17, -0.29, 0.11)),
            dtype=jnp.float64,
        )
        bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
        operator: Array = ls_operator(model.basis, model.shell_index)
        lower_trace: Array = group_trace(bands, operator, (0, 1, 2, 3))
        upper_trace: Array = group_trace(bands, operator, (4, 5))
        expectations: Array = expectation_path(bands, operator)

        assert model.basis.spin == (-1, -1, -1, 1, 1, 1)
        np.testing.assert_allclose(
            lower_trace,
            -2.0,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            upper_trace,
            2.0,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            expectations,
            np.broadcast_to(
                np.asarray((-0.5,) * 4 + (1.0,) * 2),
                expectations.shape,
            ),
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            expectations[:, 0],
            lower_trace / 4.0,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            expectations[:, -1],
            upper_trace / 2.0,
            rtol=0.0,
            atol=2e-14,
        )


__all__: list[str] = []
