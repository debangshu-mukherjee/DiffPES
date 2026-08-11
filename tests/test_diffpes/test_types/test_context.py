"""Validate workflow-context storage and cross-carrier consistency.

The cases cover the immutable context boundary and rejection of band and
projection dimensions that cannot describe the same simulation.
"""

import jax.numpy as jnp

from diffpes.types import (
    BandStructure,
    OrbitalProjection,
    WorkflowContext,
    make_band_structure,
    make_orbital_projection,
    make_workflow_context,
)
from tests._assertions import assert_rejects


class TestWorkflowContext:
    """Validate :class:`~diffpes.types.WorkflowContext` field storage.

    The carrier must keep compatible bands and projections together while
    retaining absent optional metadata as ``None``.

    :see: :class:`~diffpes.types.WorkflowContext`
    """

    def test_stores_compatible_members(self) -> None:
        """Preserve compatible workflow members and optional defaults.

        The check verifies carrier identity and the two absent optional fields
        for a one-k-point, one-band workflow.

        Notes
        -----
        The test builds both members with their public factories, constructs
        the
        context, and compares the stored objects and ``None`` sentinels.
        """
        bands: BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)), kpoints=jnp.zeros((1, 3))
        )
        projection: OrbitalProjection = make_orbital_projection(
            jnp.zeros((1, 1, 1, 9))
        )
        context: WorkflowContext = make_workflow_context(bands, projection)

        assert context.bands is bands
        assert context.orb_proj is projection
        assert context.kpath is None
        assert context.dos is None


class TestMakeWorkflowContext:
    """Validate :func:`~diffpes.types.make_workflow_context`.

    The factory must reject band and projection carriers with inconsistent
    k-point or band dimensions.

    :see: :func:`~diffpes.types.make_workflow_context`
    """

    def test_rejects_band_projection_mismatch(self) -> None:
        """Reject workflow members with inconsistent K/B dimensions.

        The check covers the static leading-dimension contract between band
        eigenvalues and orbital projection weights.

        Notes
        -----
        The test constructs independently valid carriers with incompatible
        k-point
        counts and matches the factory's dimension diagnostic.
        """
        bands: BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((2, 1)), kpoints=jnp.zeros((2, 3))
        )
        projection: OrbitalProjection = make_orbital_projection(
            jnp.zeros((1, 1, 1, 9))
        )

        assert_rejects(
            make_workflow_context,
            bands=bands,
            orb_proj=projection,
            match="bands and orb_proj must agree",
        )
