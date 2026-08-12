"""Validate the slab topology contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import jax
import pytest

from diffpes.types import (
    SlabTopology,
    make_slab_topology,
)


class TestSlabTopology:
    """Validate :class:`~diffpes.types.SlabTopology`.

    The cases bind frozen surface and atom provenance to static PyTree fields.
    """

    def test_fields_are_static_pytree_metadata(self) -> None:
        """Keep every frozen topology choice outside differentiable leaves.

        The carrier stores one primitive single-atom slab selection.

        Notes
        -----
        Flatten the carrier and compare representative integer metadata.
        """
        topology: SlabTopology = make_slab_topology(
            miller=(0, 0, 1),
            in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
            stacking_coeffs=(0, 0, 1),
            atom_shifts=((0, 0, 0),),
            bulk_atom_of_slab_atom=(0,),
            layer_of_slab_atom=(0,),
            termination=("X", "X"),
            thickness_ang=0.0,
            vacuum_ang=3.0,
            fine=(0.0, 0.0),
            n_layers=1,
            bulk_atom_count=1,
            basis_atom_indices=(0,),
        )

        assert jax.tree_util.tree_leaves(topology) == []
        assert topology.miller == (0, 0, 1)
        assert topology.n_layers == 1


class TestMakeSlabTopology:
    """Validate :func:`~diffpes.types.make_slab_topology`.

    The cases cover valid static metadata and mismatched atom provenance.
    """

    @staticmethod
    def _valid_topology() -> SlabTopology:
        """PRIVATE: Build one primitive single-atom topology.

        Returns
        -------
        topology : SlabTopology
            Validated topology for one layer and one bulk atom.

        Notes
        -----
        Keep all exact coefficient rows in the cubic identity frame.
        """
        topology: SlabTopology = make_slab_topology(
            miller=(0, 0, 1),
            in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
            stacking_coeffs=(0, 0, 1),
            atom_shifts=((0, 0, 0),),
            bulk_atom_of_slab_atom=(0,),
            layer_of_slab_atom=(0,),
            termination=("X", "X"),
            thickness_ang=0.0,
            vacuum_ang=3.0,
            fine=(0.0, 0.0),
            n_layers=1,
            bulk_atom_count=1,
            basis_atom_indices=(0,),
        )
        return topology

    def test_builds_validated_static_metadata(self) -> None:
        """Construct one topology through its public factory.

        The case compares exact surface coefficients and atom provenance.

        Notes
        -----
        Flatten the result to confirm that no differentiable leaves appear.
        """
        topology: SlabTopology = self._valid_topology()

        assert jax.tree_util.tree_leaves(topology) == []
        assert topology.stacking_coeffs == (0, 0, 1)
        assert topology.bulk_atom_of_slab_atom == (0,)

    def test_rejects_mismatched_atom_provenance(self) -> None:
        """Reject topology records that omit one selected atom mapping.

        The case plants two atom shifts for one frozen bulk atom.

        Notes
        -----
        Match the factory's explicit static length-validation message.
        """
        with pytest.raises(
            ValueError,
            match="atom_shifts must contain one entry per frozen bulk atom",
        ):
            make_slab_topology(
                miller=(0, 0, 1),
                in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
                stacking_coeffs=(0, 0, 1),
                atom_shifts=((0, 0, 0), (0, 0, 1)),
                bulk_atom_of_slab_atom=(0,),
                layer_of_slab_atom=(0,),
                termination=("X", "X"),
                thickness_ang=0.0,
                vacuum_ang=3.0,
                fine=(0.0, 0.0),
                n_layers=1,
                bulk_atom_count=1,
                basis_atom_indices=(0,),
            )
