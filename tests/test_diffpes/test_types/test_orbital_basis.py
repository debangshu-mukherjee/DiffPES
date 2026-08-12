"""Validate the orbital basis contracts.

The tests cover public carrier behavior, validation, and JAX
transformations for this implementation module.
"""

import chex
import jax
import pytest
from absl.testing import parameterized
from beartype.typing import Dict, List

from diffpes.types import (
    OrbitalBasis,
    PyTreeDef,
    make_orbital_basis,
)
from tests._assertions import assert_rejects


def _basis() -> OrbitalBasis:
    """PRIVATE: Create a two-orbital, two-atom spinless test basis.

    Returns
    -------
    basis : OrbitalBasis
        A 1s orbital on atom 0 and a 2pz orbital on atom 1, with no
        spin channel.

    Notes
    -----
    Uses the public factory so the radial-parameter carriers under
    test receive a validated basis with two distinct (n, l) shells.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 2),
        l=(0, 1),
        m=(0, 0),
        labels=("1s", "2pz"),
    )
    return basis


class TestOrbitalBasis(chex.TestCase):
    """Validate :class:`~diffpes.types.OrbitalBasis`.

    The case round-trips the PyTree and compares every static orbital field.
    """

    def test_pytree_round_trip_preserves_all_static_fields(self) -> None:
        """Preserve atom, quantum-number, spin, and label tuples exactly.

        The case flattens and rebuilds a spinful two-atom orbital basis.

        Notes
        -----
        Compare every static tuple after reconstruction and require no leaves.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0, 1, 1),
            n=(2, 2, 2, 2),
            l=(1, 1, 1, 1),
            m=(-1, 0, -1, 0),
            spin=(1, 1, -1, -1),
            labels=("a_px_up", "a_pz_up", "b_px_dn", "b_pz_dn"),
        )
        leaves: List[object]
        tree: PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(basis)
        restored: OrbitalBasis = jax.tree_util.tree_unflatten(tree, leaves)

        assert leaves == []
        assert restored.atom_indices == basis.atom_indices
        assert restored.n == basis.n
        assert restored.l == basis.l
        assert restored.m == basis.m
        assert restored.spin == basis.spin
        assert restored.labels == basis.labels


class TestMakeOrbitalBasis(chex.TestCase):
    """Validate :func:`~diffpes.types.make_orbital_basis`.

    The cases check label generation, spin defaults, static rejection, and
    direct-constructor invariants.
    """

    def test_generates_labels_and_spinless_default(self) -> None:
        """Generate stable labels and an empty spin tuple by default.

        The case omits optional metadata for a two-orbital basis.

        Notes
        -----
        Compare generated labels by position and require an empty spin tuple.
        """
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(1, 2),
            l=(0, 1),
            m=(0, 0),
        )

        assert basis.labels == ("orb_0", "orb_1")
        assert basis.spin == ()

    @parameterized.named_parameters(
        (
            "length_mismatch",
            "length",
            "must have the same length",
        ),
        (
            "negative_atom_index",
            "atom",
            "atom_indices must contain non-negative integers",
        ),
        (
            "invalid_principal_quantum_number",
            "principal",
            "n must contain integers of at least 1",
        ),
        (
            "invalid_angular_quantum_number",
            "angular",
            "l must contain integers satisfying",
        ),
        (
            "invalid_spin_length",
            "spin_length",
            "spin must be empty or have one entry per orbital",
        ),
        (
            "invalid_spin_channel",
            "spin_channel",
            r"spin entries must be \+1 or -1",
        ),
    )
    def test_rejects_invalid_static_metadata_eager_and_jit(
        self,
        defect: str,
        match: str,
    ) -> None:
        """Reject malformed atom, quantum-number, and spin tuples.

        Parameterized cases isolate one structural metadata defect at a time.

        Notes
        -----
        Route every case through the shared eager and compiled rejection check.
        """
        arguments: Dict[str, object] = {
            "atom_indices": (0,),
            "n": (1,),
            "l": (0,),
            "m": (0,),
            "spin": (),
        }
        if defect == "length":
            arguments["m"] = (0, 0)
        elif defect == "atom":
            arguments["atom_indices"] = (-1,)
        elif defect == "principal":
            arguments["n"] = (0,)
        elif defect == "angular":
            arguments["l"] = (1,)
        elif defect == "spin_length":
            arguments["atom_indices"] = (0, 0)
            arguments["n"] = (1, 1)
            arguments["l"] = (0, 0)
            arguments["m"] = (0, 0)
            arguments["spin"] = (1,)
        else:
            arguments["spin"] = (0,)

        assert_rejects(make_orbital_basis, match=match, **arguments)

    def test_raw_constructor_reasserts_static_invariants(self) -> None:
        """Prevent direct construction from bypassing spin validation.

        The case supplies an invalid spin channel to the raw module
        constructor.

        Notes
        -----
        Require the same validation error that the public factory emits.
        """
        with pytest.raises(ValueError, match="spin entries"):
            OrbitalBasis(
                atom_indices=(0,),
                n=(1,),
                l=(0,),
                m=(0,),
                spin=(0,),
                labels=("s",),
            )
