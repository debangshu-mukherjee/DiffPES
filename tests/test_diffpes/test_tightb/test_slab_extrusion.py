"""Verify slab extrusion and exact hopping propagation.

The tests exercise slab numerical and structural contracts.
"""

import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.tightb import bloch_hamiltonian
from diffpes.tightb.slab import (
    _propagate_hoppings,
    gen_slab,
    gen_slab_with_operators,
    rotate_tb_model,
    validate_open_surface_adjacency,
)
from diffpes.types import (
    CrystalGeometry,
    OrbitalBasis,
    TBModel,
    make_crystal_geometry,
    make_orbital_basis,
    make_tb_model,
    make_wannier_operator_data,
)


def _z_chain_model(hopping: jax.Array | float = -1.0) -> TBModel:
    """Build a one-orbital chain whose periodic direction is bulk z."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,),
        n=(1,),
        l=(0,),
        m=(0,),
        labels=("s",),
    )
    value: jax.Array = jnp.asarray(hopping, dtype=jnp.complex128)
    model: TBModel = make_tb_model(
        hopping_amplitudes=jnp.stack((value, jnp.conj(value))),
        onsite_energies=jnp.zeros((1,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )
    return model


def _alternating_species_model() -> TBModel:
    """Build an inert X/Y stack with two planes per stacking period."""
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.eye(3, dtype=jnp.float64),
        positions=jnp.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
            dtype=jnp.float64,
        ),
        species=("X", "Y"),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 1),
        n=(1, 1),
        l=(0, 0),
        m=(0, 0),
        labels=("X-s", "Y-s"),
    )
    return make_tb_model(
        hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((2,), dtype=jnp.float64),
        soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
        geometry=geometry,
        basis=basis,
        hopping_pairs=(),
        hopping_cells=(),
        shell_index=(-1, -1),
    )


class TestGenSlab:
    """Verify :func:`diffpes.tightb.gen_slab`.

    :see: :func:`~diffpes.tightb.gen_slab`
    """

    @pytest.mark.parametrize("n_layers", (1, 2, 5, 20))
    def test_finite_chain_closed_form(self, n_layers: int) -> None:
        """Match every open-chain eigenvalue to the textbook spectrum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        spec: Any
        hopping: float = -0.73
        slab: TBModel
        slab, spec = gen_slab(
            _z_chain_model(hopping),
            miller=(0, 0, 1),
            thickness_ang=float(n_layers - 1),
            vacuum_ang=6.0,
        )
        hamiltonian: jax.Array = bloch_hamiltonian(
            slab,
            jnp.zeros((3,), dtype=jnp.float64),
        )
        actual: jax.Array = jnp.linalg.eigvalsh(hamiltonian)
        modes: jax.Array = jnp.arange(1, n_layers + 1)
        expected: jax.Array = (
            2.0 * hopping * jnp.cos(modes * jnp.pi / (n_layers + 1))
        )

        assert spec.n_layers == n_layers
        assert jnp.allclose(
            actual,
            jnp.sort(expected),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_depths_and_open_normal_cells_are_exact(self) -> None:
        """Produce integral layer depths and no periodic normal hopping.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        spec: Any
        slab, spec = gen_slab(
            _z_chain_model(),
            miller=(0, 0, 1),
            thickness_ang=6.0,
            vacuum_ang=9.0,
        )

        assert slab.depths is not None
        assert jnp.array_equal(
            slab.depths,
            jnp.arange(6, -1, -1, dtype=jnp.float64),
        )
        assert all(cell[2] == 0 for cell in slab.hopping_cells)
        assert spec.surface_cell.stacking_coeffs == (0, 0, 1)
        assert (
            sum(
                miller * coefficient
                for miller, coefficient in zip(
                    spec.surface_cell.miller,
                    spec.surface_cell.stacking_coeffs,
                    strict=True,
                )
            )
            == 1
        )

    def test_hopping_gather_preserves_parameter_gradient(self) -> None:
        """Differentiate a slab Hamiltonian norm through the exact gather.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """

        def loss(hopping: jax.Array) -> jax.Array:
            slab: Any
            slab, _ = gen_slab(
                _z_chain_model(hopping),
                miller=(0, 0, 1),
                thickness_ang=4.0,
                vacuum_ang=5.0,
            )
            hamiltonian: jax.Array = bloch_hamiltonian(
                slab,
                jnp.zeros((3,), dtype=jnp.float64),
            )
            return jnp.real(jnp.vdot(hamiltonian, hamiltonian))

        value: jax.Array = jnp.asarray(-0.61, dtype=jnp.float64)
        derivative: jax.Array = jax.grad(loss)(value)
        step: float = 1e-5
        finite_difference: jax.Array = (
            loss(value + step) - loss(value - step)
        ) / (2.0 * step)

        assert jnp.isfinite(derivative)
        assert jnp.abs(derivative) > 0.0
        assert jnp.allclose(derivative, finite_difference, rtol=1e-8)

    def test_internal_gather_is_exact(self) -> None:
        """Expose the static source-hop gather for a six-layer chain.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        amplitudes: Any
        cells: Any
        gather: Any
        pairs: Any
        slab: Any
        spec: Any
        bulk: TBModel = _z_chain_model(-0.4)
        slab, spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=6.0,
            vacuum_ang=4.0,
        )
        rotated: TBModel = rotate_tb_model(
            bulk,
            spec.surface_cell.rotation,
        )
        amplitudes, pairs, cells, gather = _propagate_hoppings(rotated, spec)

        assert len(gather) == len(slab.hopping_cells)
        assert len(pairs) == len(cells) == amplitudes.shape[0]
        assert set(gather) == {0, 1, 2}
        assert jnp.array_equal(
            amplitudes,
            rotated.hopping_amplitudes[jnp.asarray(gather)],
        )
        assert slab.hopping_cells == cells

    def test_rejects_nonprimitive_miller_tuple(self) -> None:
        """Reject a doubled Miller normal rather than changing layer period.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        with pytest.raises(ValueError, match="gcd-reduced"):
            gen_slab(
                _z_chain_model(),
                miller=(0, 0, 2),
                thickness_ang=4.0,
                vacuum_ang=4.0,
            )

    @pytest.mark.parametrize(
        "fine",
        ((0.2, 0.0), (0.6, 0.6)),
    )
    def test_post_fine_natural_span_remains_a_minimum(
        self,
        fine: tuple[float, float],
    ) -> None:
        """Expand a natural stack before its fine-shifted planes are cut.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        requested_span: float = 3.0
        slab, _ = gen_slab(
            _z_chain_model(),
            miller=(0, 0, 1),
            thickness_ang=requested_span,
            vacuum_ang=4.0,
            fine=fine,
        )
        cartesian: jax.Array = slab.geometry.positions @ slab.geometry.lattice
        realized_span: jax.Array = jnp.max(cartesian[:, 2]) - jnp.min(
            cartesian[:, 2]
        )

        assert realized_span >= requested_span - 1e-12

    def test_post_fine_explicit_endpoints_match_provenance(self) -> None:
        """Verify inward snapping reaches the requested species.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        spec: Any
        requested_span: float = 2.0
        slab, spec = gen_slab(
            _alternating_species_model(),
            miller=(0, 0, 1),
            thickness_ang=requested_span,
            vacuum_ang=4.0,
            termination=("X", "X"),
            fine=(0.1, 0.1),
        )
        cartesian: jax.Array = slab.geometry.positions @ slab.geometry.lattice
        bottom: int = int(jnp.argmin(cartesian[:, 2]))
        top: int = int(jnp.argmax(cartesian[:, 2]))

        assert spec.termination == ("X", "X")
        assert slab.geometry.species[top] == "X"
        assert slab.geometry.species[bottom] == "X"
        assert (
            cartesian[top, 2] - cartesian[bottom, 2] >= requested_span - 1e-12
        )

    def test_zero_thickness_survives_symmetric_fine_cut(self) -> None:
        """Keep the T=0 one-plane limit well-defined after fine expansion.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        spec: Any
        slab, spec = gen_slab(
            _z_chain_model(),
            miller=(0, 0, 1),
            thickness_ang=0.0,
            vacuum_ang=4.0,
            fine=(0.6, 0.6),
        )

        assert spec.n_layers == 1
        assert len(slab.basis.n) == 1
        assert slab.depths is not None
        assert jnp.array_equal(slab.depths, jnp.zeros((1,)))


class TestValidateOpenSurfaceAdjacency:
    """Verify the exact normal-image invariant.

    :see: :func:`~diffpes.tightb.validate_open_surface_adjacency`
    """

    def test_rejects_image_edge_even_with_large_vacuum_proxy(self) -> None:
        """Reject a normal image from exact cells without using a distance.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        with pytest.raises(ValueError, match="normal-image hopping"):
            validate_open_surface_adjacency(_z_chain_model())


class TestGenSlabWithOperators:
    """Verify the bulk-to-slab Wannier-operator seam.

    :see: :func:`~diffpes.tightb.gen_slab_with_operators`
    """

    def test_noncoincident_centres_and_matrix_are_preserved(self) -> None:
        """Propagate distinct same-atom centres and an off-diagonal operator.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        operator_data: Any
        propagated: Any
        slab: Any
        spec: Any
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice=jnp.eye(3, dtype=jnp.float64),
            positions=jnp.zeros((1, 3), dtype=jnp.float64),
            species=("X",),
        )
        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 0),
            n=(1, 2),
            l=(0, 0),
            m=(0, 0),
            labels=("s1", "s2"),
        )
        centres: jax.Array = jnp.asarray(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.37]],
            dtype=jnp.float64,
        )
        bulk: TBModel = make_tb_model(
            hopping_amplitudes=jnp.zeros((0,), dtype=jnp.complex128),
            onsite_energies=jnp.asarray([0.2, -0.1], dtype=jnp.float64),
            soc_lambdas=jnp.zeros((0,), dtype=jnp.float64),
            geometry=geometry,
            basis=basis,
            hopping_pairs=(),
            hopping_cells=(),
            shell_index=(-1, -1),
            orbital_positions=centres,
        )
        position: jax.Array = jnp.zeros(
            (1, 2, 2, 3),
            dtype=jnp.complex128,
        )
        position = position.at[0, 0, 0].set(centres[0])
        position = position.at[0, 1, 1].set(centres[1])
        position = position.at[0, 0, 1].set(
            jnp.asarray([0.2 + 0.1j, -0.3j, 0.4])
        )
        position = position.at[0, 1, 0].set(jnp.conj(position[0, 0, 1]))
        operator_data = make_wannier_operator_data(
            position_matrices=position,
            centres_cart=centres,
            cells=((0, 0, 0),),
            degeneracies=(1,),
            spin_layout="block_down_up",
            source_format="tb",
        )

        slab, spec, propagated = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(0, 0, 1),
            thickness_ang=1.0,
            vacuum_ang=4.0,
        )

        assert slab.orbital_positions is not None
        assert propagated.position_matrices is not None
        assert spec.n_layers == 2
        assert propagated.centres_cart.shape == (4, 3)
        assert jnp.allclose(
            propagated.centres_cart[:2, 2],
            centres[:, 2],
            atol=1e-13,
        )
        assert jnp.allclose(
            propagated.centres_cart[2:, 2],
            centres[:, 2] + 1.0,
            atol=1e-13,
        )
        zero_index: int = propagated.cells.index((0, 0, 0))
        diagonal: jax.Array = propagated.position_matrices[
            zero_index,
            jnp.arange(4),
            jnp.arange(4),
        ]
        assert jnp.allclose(
            diagonal,
            propagated.centres_cart,
            atol=1e-13,
        )
        assert jnp.allclose(
            propagated.position_matrices[zero_index, 0, 1],
            position[0, 0, 1],
            atol=1e-13,
        )
        assert jnp.allclose(
            propagated.position_matrices[zero_index, 2, 3],
            position[0, 0, 1],
            atol=1e-13,
        )

    def test_absent_position_matrix_remains_absent(self) -> None:
        """Keep an hr-format sidecar explicitly matrix-free.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        operator_data: Any
        propagated: Any
        bulk: TBModel = _z_chain_model()
        operator_data = make_wannier_operator_data(
            position_matrices=None,
            centres_cart=jnp.zeros((1, 3), dtype=jnp.float64),
            cells=((0, 0, 1), (0, 0, -1)),
            degeneracies=(1, 1),
            spin_layout="block_down_up",
            source_format="hr",
        )

        _, _, propagated = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(0, 0, 1),
            thickness_ang=2.0,
            vacuum_ang=4.0,
        )

        assert propagated.position_matrices is None
        assert propagated.source_format == "hr"
        assert propagated.cells == ((0, 0, 0),)
        assert propagated.degeneracies == (1,)

    def test_matrix_free_sidecar_preserves_in_plane_cells(self) -> None:
        """Verify nonzero hr-cell remapping preserves provenance.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        operator_data: Any
        propagated: Any
        slab: Any
        bulk: TBModel = _z_chain_model()
        bulk = make_tb_model(
            hopping_amplitudes=bulk.hopping_amplitudes,
            onsite_energies=bulk.onsite_energies,
            soc_lambdas=bulk.soc_lambdas,
            geometry=bulk.geometry,
            basis=bulk.basis,
            hopping_pairs=bulk.hopping_pairs,
            hopping_cells=((1, 0, 0), (-1, 0, 0)),
            shell_index=bulk.shell_index,
        )
        operator_data = make_wannier_operator_data(
            position_matrices=None,
            centres_cart=jnp.zeros((1, 3), dtype=jnp.float64),
            cells=((1, 0, 0), (-1, 0, 0)),
            degeneracies=(2, 3),
            spin_layout="block_down_up",
            source_format="hr",
        )

        slab, _, propagated = gen_slab_with_operators(
            bulk,
            operator_data,
            miller=(0, 0, 1),
            thickness_ang=3.0,
            vacuum_ang=4.0,
        )

        assert propagated.position_matrices is None
        assert propagated.cells == ((-1, 0, 0), (1, 0, 0))
        assert propagated.degeneracies == (1, 1)
        assert all(cell[2] == 0 for cell in slab.hopping_cells)

    def test_natural_slab_accepts_unknown_species(self) -> None:
        """Use the explicit unknown-species sentinel only for natural cuts.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        slab: Any
        spec: Any
        unknown_geometry: Any
        bulk: TBModel = _z_chain_model()
        unknown_geometry = make_crystal_geometry(
            lattice=bulk.geometry.lattice,
            positions=bulk.geometry.positions,
            species=(),
        )
        bulk = make_tb_model(
            hopping_amplitudes=bulk.hopping_amplitudes,
            onsite_energies=bulk.onsite_energies,
            soc_lambdas=bulk.soc_lambdas,
            geometry=unknown_geometry,
            basis=bulk.basis,
            hopping_pairs=bulk.hopping_pairs,
            hopping_cells=bulk.hopping_cells,
            shell_index=bulk.shell_index,
        )

        slab, spec = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=2.0,
            vacuum_ang=4.0,
        )

        assert slab.geometry.species == ("X", "X", "X")
        assert spec.termination == ("X", "X")
