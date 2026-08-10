"""Certify slab differentiation, gauge invariance, and scale bounds.

The tests exercise slab numerical and structural contracts.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Tuple
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal
from jaxtyping import Array

from diffpes.tightb import (
    diagonalize_tb,
    eigvalsh_bands,
    eigvalsh_bands_chunked,
    freeze_slab_topology,
    gen_slab,
    layer_resolved_group_traces,
    layer_resolved_weights,
    rebuild_slab,
    spin_double_model,
)
from diffpes.types import (
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    SlabTopology,
    TBModel,
    make_crystal_geometry,
    make_diagonalized_bands,
    make_orbital_basis,
    make_tb_model,
)
from tests._gradients import gradient_gate


def _geometry() -> CrystalGeometry:
    """PRIVATE: Build the anisotropic one-site geometry for the gradient
    gates.

    Returns
    -------
    geometry : CrystalGeometry
        One-site orthorhombic cell with lattice constants 2.2, 2.5, and
        1.3 Angstrom.

    Notes
    -----
    The three distinct lattice constants remove accidental cubic
    symmetry, so gradient and gauge checks see generic geometry.
    """
    return make_crystal_geometry(
        lattice=jnp.diag(jnp.asarray((2.2, 2.5, 1.3))),
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species=("X",),
    )


def _complex_soc_bulk(parameters: Array) -> TBModel:
    """PRIVATE: Build a complete p-shell spinor model from five coordinates.

    Parameters
    ----------
    parameters : Array
        Five active coordinates ``(p0, p1, p2, p3, soc)``; the first
        four scale hopping entries in eV and the last is the atomic
        spin--orbit coupling in eV.

    Returns
    -------
    model : TBModel
        Spin-doubled complete-p model with dense complex x- and
        z-direction hopping blocks, fixed onsite energies, three
        distinct z-offset orbital positions, and the traced SOC
        strength.

    Notes
    -----
    Every entry of the two dense three-by-three blocks carries one of
    the four hopping coordinates with a distinct complex coefficient.
    The conjugate-transpose blocks sit on the reverse cells. The Bloch
    Hamiltonian therefore stays Hermitian for any parameter value
    while each coordinate keeps a nonzero, generic sensitivity. The
    final :func:`spin_double_model` call doubles the basis and applies
    the SOC coordinate.
    """
    p0: Any
    p1: Any
    p2: Any
    p3: Any
    soc: Any
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 0),
        n=(2, 2, 2),
        l=(1, 1, 1),
        m=(-1, 0, 1),
        labels=("px", "py", "pz"),
    )
    p0, p1, p2, p3, soc = parameters
    x_block: Array = jnp.asarray(
        (
            (0.7 * p0, (0.21 + 0.13j) * p3, (-0.08 + 0.17j) * p2),
            ((-0.06 + 0.11j) * p3, -0.4 * p1, (0.09 - 0.07j) * p0),
            ((0.14 + 0.05j) * p2, (-0.12 + 0.19j) * p0, 0.5 * p2),
        ),
        dtype=jnp.complex128,
    )
    z_block: Array = jnp.asarray(
        (
            (-0.31 * p1, (0.04 + 0.16j) * p2, (0.12 - 0.03j) * p3),
            ((-0.09 + 0.08j) * p2, 0.27 * p0, (0.15 + 0.06j) * p1),
            ((0.05 - 0.18j) * p3, (0.02 + 0.09j) * p1, -0.44 * p3),
        ),
        dtype=jnp.complex128,
    )
    blocks: Tuple[Array, ...] = (
        x_block,
        x_block.conj().T,
        z_block,
        z_block.conj().T,
    )
    cells: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    pairs_one_block: Tuple[Tuple[int, int], ...] = tuple(
        (row, column) for row in range(3) for column in range(3)
    )
    spinless: TBModel = make_tb_model(
        hopping_amplitudes=jnp.concatenate(
            tuple(block.reshape(-1) for block in blocks)
        ),
        onsite_energies=jnp.asarray((-0.37, 0.11, 0.63)),
        soc_lambdas=jnp.reshape(soc, (1,)),
        geometry=_geometry(),
        basis=basis,
        hopping_pairs=pairs_one_block * len(blocks),
        hopping_cells=tuple(
            cell for cell in cells for _ in range(len(pairs_one_block))
        ),
        shell_index=(0, 0, 0),
        orbital_positions=jnp.asarray(
            (
                (0.0, 0.0, 0.05),
                (0.0, 0.0, 0.27),
                (0.0, 0.0, 0.49),
            )
        ),
    )
    return spin_double_model(spinless)


def _canonical_topology() -> SlabTopology:
    """PRIVATE: Return the registered thin (001) slab outside traced paths.

    Returns
    -------
    topology : SlabTopology
        Frozen static two-layer slab topology of the nominal-parameter
        bulk with Miller index (001), thickness 1.3 Angstrom, and
        vacuum 4.0 Angstrom.

    Notes
    -----
    Freezing at the fixed nominal coordinates happens once, outside
    any traced function. Gradient losses can then rebuild the slab
    under autodiff with :func:`rebuild_slab` while the integer topology
    stays static.
    """
    nominal: Array = jnp.asarray((0.53, -0.47, 0.61, -0.39, 0.22))
    return freeze_slab_topology(
        _complex_soc_bulk(nominal),
        miller=(0, 0, 1),
        thickness_ang=1.3,
        vacuum_ang=4.0,
    )


def _oblique_chain_model(shape_parameter: Array | float) -> TBModel:
    """PRIVATE: Build a one-orbital chain in a continuously deformable
    oblique cell.

    Parameters
    ----------
    shape_parameter : Array | float
        Dimensionless deformation coordinate; three off-diagonal
        lattice entries in Angstrom vary linearly with it.

    Returns
    -------
    model : TBModel
        One-orbital model with a Hermitian ``-0.4`` eV hopping pair
        along the third lattice vector inside the deformed cell.

    Notes
    -----
    Differentiating through ``shape_parameter`` exercises the lattice
    path of slab construction: the surface cell, layer spacing, and
    depths all respond to the continuous cell deformation.
    """
    parameter: Array = jnp.asarray(shape_parameter, dtype=jnp.float64)
    lattice: Array = jnp.asarray(
        (
            (2.1, 0.23 + 0.11 * parameter, 0.31),
            (0.17, 1.8, 0.14 - 0.07 * parameter),
            (0.09 + 0.13 * parameter, 0.28, 2.3),
        ),
        dtype=jnp.float64,
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=lattice,
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
    return make_tb_model(
        hopping_amplitudes=jnp.asarray((-0.4, -0.4), dtype=jnp.complex128),
        onsite_energies=jnp.zeros((1,)),
        soc_lambdas=jnp.zeros((0,)),
        geometry=geometry,
        basis=basis,
        hopping_pairs=((0, 0), (0, 0)),
        hopping_cells=((0, 0, 1), (0, 0, -1)),
        shell_index=(-1,),
    )


def _spectral_loss(parameters: Array, topology: SlabTopology) -> Array:
    """PRIVATE: Return a smooth broadened spectrum plus an isolated-group
    trace.

    Parameters
    ----------
    parameters : Array
        Five active bulk coordinates for :func:`_complex_soc_bulk`.
    topology : SlabTopology
        Frozen static slab topology from :func:`_canonical_topology`.

    Returns
    -------
    loss : Array
        Scalar sum of two gauge-invariant terms. The first is a
        Gaussian-broadened spectral moment over four probe energies in
        eV with width 0.37 eV. The second is 0.17 times the summed
        layer-resolved trace of the isolated two-band group.

    Notes
    -----
    Rebuilds the slab from the traced parameters, diagonalizes it on
    five generic in-plane k-points, and contracts the eigenvalues with
    fixed probe weights. Both terms are gauge invariant, so the loss
    gives a degeneracy-safe gradient path from the eigensolver back to
    every hopping and SOC coordinate.
    """
    slab: TBModel
    slab, _ = rebuild_slab(_complex_soc_bulk(parameters), topology)
    k_x: Array = jnp.asarray((-0.41, -0.19, 0.07, 0.23, 0.44))
    kpoints: Array = jnp.stack(
        (k_x, 0.13 * k_x + 0.04, jnp.zeros_like(k_x)),
        axis=-1,
    )
    bands: DiagonalizedBands = diagonalize_tb(slab, kpoints)
    probe_energies: Array = jnp.asarray((-0.83, -0.21, 0.34, 0.91))
    sigma: float = 0.37
    broadened: Array = jnp.exp(
        -((bands.eigenvalues[:, :, None] - probe_energies[None, None, :]) ** 2)
        / (2.0 * sigma**2)
    )
    spectral_moment: Array = jnp.sum(
        broadened * jnp.asarray((0.7, -0.4, 0.9, 0.3))[None, None, :]
    )
    group_trace: Array = layer_resolved_group_traces(
        bands,
        ((0, 1),),
        2.1,
    )
    return spectral_moment + 0.17 * jnp.sum(group_trace)


def _group_trace_loss(parameters: Array, topology: SlabTopology) -> Array:
    """PRIVATE: Return the isolated fixed-group component of the
    depth-gradient loss.

    Parameters
    ----------
    parameters : Array
        Five active bulk coordinates for :func:`_complex_soc_bulk`.
    topology : SlabTopology
        Frozen static slab topology from :func:`_canonical_topology`.

    Returns
    -------
    loss : Array
        Scalar sum of the layer-resolved trace of the fixed band group
        ``(0, 1)`` at an intensity escape length of 2.1 Angstrom.

    Notes
    -----
    Repeats the slab rebuild and diagonalization of
    :func:`_spectral_loss` on the same five k-points but keeps only the
    group-trace term, so tests can attribute a gradient defect to the
    depth-weighted projector path alone.
    """
    slab: TBModel
    slab, _ = rebuild_slab(_complex_soc_bulk(parameters), topology)
    k_x: Array = jnp.asarray((-0.41, -0.19, 0.07, 0.23, 0.44))
    kpoints: Array = jnp.stack(
        (k_x, 0.13 * k_x + 0.04, jnp.zeros_like(k_x)),
        axis=-1,
    )
    bands: DiagonalizedBands = diagonalize_tb(slab, kpoints)
    return jnp.sum(layer_resolved_group_traces(bands, ((0, 1),), 2.1))


def _bands(
    eigenvalues: Array,
    eigenvectors: Array,
    depths: Array,
) -> DiagonalizedBands:
    """PRIVATE: Build a small surface-bearing eigensystem.

    Parameters
    ----------
    eigenvalues : Array
        Synthetic band energies in eV.
    eigenvectors : Array
        Band-major eigenvector rows for each k-point.
    depths : Array
        Per-orbital depths below the surface in Angstrom.

    Returns
    -------
    bands : DiagonalizedBands
        Carrier with the supplied eigensystem and depths, zero
        k-points, Fermi energy zero eV, and a one-site placeholder
        geometry with an all-s basis.

    Notes
    -----
    The attached depths make the carrier acceptable to the surface and
    layer-resolved operators; the placeholder geometry carries no
    physics.
    """
    n_orbitals: int = eigenvalues.shape[-1]
    return make_diagonalized_bands(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=jnp.zeros((eigenvalues.shape[0], 3)),
        fermi_energy=0.0,
        geometry=make_crystal_geometry(
            lattice=jnp.eye(3),
            positions=jnp.zeros((1, 3)),
            species=("X",),
        ),
        basis=make_orbital_basis(
            atom_indices=(0,) * n_orbitals,
            n=(1,) * n_orbitals,
            l=(0,) * n_orbitals,
            m=(0,) * n_orbitals,
            labels=tuple(f"o{index}" for index in range(n_orbitals)),
        ),
        depths=depths,
    )


def _random_unitary(key: Array, size: int) -> Array:
    """PRIVATE: Return a deterministic Haar-like complex unitary from a
    complex QR.

    Parameters
    ----------
    key : Array
        JAX PRNG key that fixes the draw.
    size : int
        Matrix dimension.

    Returns
    -------
    unitary : Array
        A ``size`` by ``size`` complex unitary matrix.

    Notes
    -----
    Draws a complex Gaussian matrix and takes its QR decomposition.
    Multiplies each column of Q by the conjugate unit phase of the
    matching R diagonal entry. The phase fix removes the sign ambiguity
    of QR, so the same key always yields the same unitary for the
    gauge-invariance checks.
    """
    real_key: Array
    imaginary_key: Array
    real_key, imaginary_key = jax.random.split(key)
    matrix: Array = jax.random.normal(real_key, (size, size)) + 1j * (
        jax.random.normal(imaginary_key, (size, size))
    )
    q_matrix: Array
    diagonal: Array
    q_matrix, diagonal = jnp.linalg.qr(matrix)
    phases: Array = jnp.diag(diagonal)
    return q_matrix * (phases / jnp.abs(phases)).conj()[None, :]


def _diagonal_model(n_orbitals: int) -> TBModel:
    """PRIVATE: Build a bounded comparator for low-cost eigensolver contract
    tests.

    Parameters
    ----------
    n_orbitals : int
        Number of independent scalar bands.

    Returns
    -------
    model : TBModel
        Model whose hoppings are all orbital-diagonal conjugate pairs
        along x with forward amplitudes spread over ``[-0.31, -0.07]``
        eV and onsite energies spread over ``[-1.3, 1.7]`` eV.

    Notes
    -----
    ``H(k)`` stays diagonal with closed cosine entries, so eigensolver
    contract checks scale to any size without meaningful linear-algebra
    cost.
    """
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0,) * n_orbitals,
        n=(1,) * n_orbitals,
        l=(0,) * n_orbitals,
        m=(0,) * n_orbitals,
        labels=tuple(f"s{index}" for index in range(n_orbitals)),
    )
    pairs: Tuple[Tuple[int, int], ...] = tuple(
        (index, index) for index in range(n_orbitals)
    )
    forward: Array = jnp.linspace(-0.31, -0.07, n_orbitals).astype(
        jnp.complex128
    )
    return make_tb_model(
        hopping_amplitudes=jnp.concatenate((forward, forward)),
        onsite_energies=jnp.linspace(-1.3, 1.7, n_orbitals),
        soc_lambdas=jnp.zeros((0,)),
        geometry=_geometry(),
        basis=basis,
        hopping_pairs=pairs + pairs,
        hopping_cells=((1, 0, 0),) * n_orbitals + ((-1, 0, 0),) * n_orbitals,
        shell_index=(-1,) * n_orbitals,
    )


def _scaling_bulk_model(*, alternating_species: bool = False) -> TBModel:
    """PRIVATE: Build a dense four-orbital bulk for real slab scaling tests.

    Parameters
    ----------
    alternating_species : bool
        If true, split the four orbitals over an X site and a Y site
        at fractional z one half. Otherwise place all four orbitals on
        one X site.

    Returns
    -------
    model : TBModel
        Four-orbital model with dense complex sixteen-entry hopping
        blocks on the x and z conjugate cell pairs. A deterministic
        sine--cosine seed at scale 0.13 eV generates the blocks.
        Onsite energies are ``(-1.2, -0.3, 0.4, 1.5)`` eV.

    Notes
    -----
    The z blocks carry factor 1.3 relative to the x blocks. Each
    reverse cell holds the conjugate transpose, so the bulk is
    Hermitian, dense, and generic. Slab scaling tests extrude it layer
    by layer with four orbitals per layer.
    """
    positions: Array = (
        jnp.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)))
        if alternating_species
        else jnp.zeros((1, 3))
    )
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=_geometry().lattice,
        positions=positions,
        species=("X", "Y") if alternating_species else ("X",),
    )
    basis: OrbitalBasis = make_orbital_basis(
        atom_indices=(0, 0, 1, 1) if alternating_species else (0,) * 4,
        n=(1, 2, 1, 2) if alternating_species else (1, 2, 3, 4),
        l=(0,) * 4,
        m=(0,) * 4,
        labels=("s1", "s2", "s3", "s4"),
    )
    pairs: Tuple[Tuple[int, int], ...] = tuple(
        (row, column) for row in range(4) for column in range(4)
    )
    seed: Array = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    forward: Array = 0.13 * (
        jnp.sin(seed + 0.2) + 1j * jnp.cos(0.7 * seed + 0.1)
    )
    cells: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    blocks: Tuple[Array, ...] = (
        forward,
        forward.conj().T,
        1.3 * forward,
        1.3 * forward.conj().T,
    )
    return make_tb_model(
        hopping_amplitudes=jnp.concatenate(
            tuple(block.reshape(-1) for block in blocks)
        ),
        onsite_energies=jnp.asarray((-1.2, -0.3, 0.4, 1.5)),
        soc_lambdas=jnp.zeros((0,)),
        geometry=geometry,
        basis=basis,
        hopping_pairs=pairs * len(cells),
        hopping_cells=tuple(cell for cell in cells for _ in range(len(pairs))),
        shell_index=(-1,) * 4,
    )


def _scaling_slab(n_layers: int) -> TBModel:
    """PRIVATE: Build a four-orbital-per-layer slab through the public
    generator.

    Parameters
    ----------
    n_layers : int
        Number of extruded layers.

    Returns
    -------
    slab : TBModel
        The (001) slab of :func:`_scaling_bulk_model` with thickness
        ``(n_layers - 1) * 1.3`` Angstrom and 4.0 Angstrom of vacuum.

    Notes
    -----
    Asserts that the generator reports exactly ``n_layers`` layers and
    ``4 * n_layers`` orbitals before returning, so the scaling sweeps
    measure the size they claim.
    """
    specification: Any
    slab: TBModel
    slab, specification = gen_slab(
        _scaling_bulk_model(),
        miller=(0, 0, 1),
        thickness_ang=(n_layers - 1) * 1.3,
        vacuum_ang=4.0,
    )
    assert specification.n_layers == n_layers
    assert slab.onsite_energies.shape == (4 * n_layers,)
    return slab


def _collect_shapes(value: object, shapes: list[Tuple[int, ...]]) -> None:
    """PRIVATE: Collect shaped variables recursively from one JAXPR.

    Parameters
    ----------
    value : object
        A ``ClosedJaxpr``, ``Jaxpr``, container, or any other object
        found inside JAXPR equation parameters.
    shapes : list[tuple[int, ...]]
        Mutable accumulator that receives one shape tuple per shaped
        variable.

    Notes
    -----
    Walks constant, input, output, and equation variables, skips
    literals, and appends the ``aval`` shape of every variable that has
    one. Recurses into equation parameters, tuples, lists, and
    dictionaries, so shapes hidden behind call primitives also appear.
    The traversal mutates ``shapes`` in place and returns ``None``.
    """
    equation: Any
    item: Any
    parameter: Any
    shape: Any
    variable: Any
    if isinstance(value, ClosedJaxpr):
        _collect_shapes(value.jaxpr, shapes)
        return
    if isinstance(value, Jaxpr):
        variables: list[object] = [
            *value.constvars,
            *value.invars,
            *value.outvars,
        ]
        for equation in value.eqns:
            variables.extend(equation.invars)
            variables.extend(equation.outvars)
            for parameter in equation.params.values():
                _collect_shapes(parameter, shapes)
        for variable in variables:
            if isinstance(variable, Literal):
                continue
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                shapes.append(tuple(int(axis) for axis in shape))
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_shapes(item, shapes)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_shapes(item, shapes)


class TestSlabDifferentiability:
    """Certify depth and structural gradients with finite differences and nonzero tripwires."""

    @pytest.mark.rss_limit_mb(1536)
    def test_generic_complex_soc_slab_gradient_gate(self) -> None:
        """Match fwd/rev/FD for every active hopping and SOC coordinate.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        parameters: Array = jnp.asarray(
            (0.53, -0.47, 0.61, -0.39, 0.22),
            dtype=jnp.float64,
        )
        topology: SlabTopology = _canonical_topology()
        group_gradient: Array = jax.grad(
            lambda candidate: _group_trace_loss(candidate, topology)
        )(parameters)

        gradient_gate(
            lambda candidate: _spectral_loss(candidate, topology),
            parameters,
            regime="smooth",
            elementwise=True,
            directional_atol=2e-8,
        )
        assert jnp.linalg.norm(group_gradient) > 1e-9

    @pytest.mark.rss_limit_mb(1024)
    @pytest.mark.parametrize("observable", ("spacing", "rotation", "depths"))
    def test_oblique_frozen_surface_geometry_gradient(
        self,
        observable: str,
    ) -> None:
        """Match forward, reverse, and finite-difference slab rebuild gradients.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        nominal: Array = jnp.asarray(0.37)
        topology: SlabTopology = freeze_slab_topology(
            _oblique_chain_model(nominal),
            miller=(1, 1, 1),
            thickness_ang=5.0,
            vacuum_ang=4.0,
        )
        rotation_weights: Array = jnp.asarray(
            (
                (0.7, -0.2, 0.5),
                (-0.3, 0.9, 0.4),
                (0.6, -0.8, 0.1),
            )
        )

        def loss(parameter: Array) -> Array:
            spec: Any
            slab: TBModel
            slab, spec = rebuild_slab(
                _oblique_chain_model(parameter),
                topology,
            )
            if observable == "spacing":
                return spec.surface_cell.interlayer_spacing_ang
            if observable == "rotation":
                return jnp.sum(spec.surface_cell.rotation * rotation_weights)
            assert slab.depths is not None
            depth_weights: Array = jnp.linspace(
                0.4,
                1.3,
                slab.depths.shape[0],
            )
            return jnp.sum(depth_weights * slab.depths)

        gradient_gate(
            loss,
            nominal,
            regime="smooth",
            directional_atol=2e-9,
        )

    def test_group_trace_probe_depth_gradient_and_small_guard(self) -> None:
        """Match structural finite-difference evidence and keep the 1e-8-A probe finite.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        weights: Array = jnp.asarray((0.18, 0.77, 0.42))
        depths: Array = -jnp.log(weights)
        theta: float = 0.37
        rotation: Array = jnp.asarray(
            (
                (jnp.cos(theta), jnp.sin(theta), 0.0),
                (-jnp.sin(theta), jnp.cos(theta), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=jnp.complex128,
        )
        bands: DiagonalizedBands = _bands(
            jnp.asarray(((0.0, 0.0, 2e-3),)),
            rotation[None, :, :],
            depths,
        )

        def loss(length: Array) -> Array:
            return jnp.sum(
                layer_resolved_group_traces(
                    bands,
                    ((0, 1),),
                    length,
                )
            )

        gradient_gate(
            loss,
            jnp.asarray(1.7),
            regime="smooth",
        )
        small_length: Array = jnp.asarray(1e-8)
        small_value: Array = loss(small_length)
        small_gradient: Array = jax.grad(loss)(small_length)
        assert jnp.isfinite(small_value)
        assert jnp.isfinite(small_gradient)


class TestSlabGaugeInvariance:
    """Certify gauge invariance under random phases and complete-group unitaries."""

    @pytest.mark.parametrize("seed", (7, 19, 43))
    def test_random_phases_and_u2_preserve_complete_trace(
        self,
        seed: int,
    ) -> None:
        """Verify per-vector changes preserve an isolated U(2) sum.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = -jnp.log(jnp.asarray((0.2, 0.8, 0.4)))
        canonical: Array = jnp.eye(3, dtype=jnp.complex128)
        unitary: Array = _random_unitary(jax.random.key(seed), 2)
        phases: Array = jnp.exp(
            1j
            * jax.random.uniform(
                jax.random.key(seed + 100),
                (3,),
                minval=-jnp.pi,
                maxval=jnp.pi,
            )
        )
        transformed: Array = canonical.at[:2, :].set(
            unitary @ canonical[:2, :]
        )
        transformed = phases[:, None] * transformed
        original: DiagonalizedBands = _bands(
            jnp.asarray(((0.0, 0.0, 2e-3),)),
            canonical[None, :, :],
            depths,
        )
        gauged: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            original,
            transformed[None, :, :],
        )
        original_trace: Array = layer_resolved_group_traces(
            original,
            ((0, 1),),
            1.0,
        )
        gauged_trace: Array = layer_resolved_group_traces(
            gauged,
            ((0, 1),),
            1.0,
        )
        individual_change: Array = jnp.max(
            jnp.abs(
                layer_resolved_weights(gauged, 1.0)
                - layer_resolved_weights(original, 1.0)
            )
        )

        assert individual_change > 1e-3
        assert jnp.allclose(
            gauged_trace,
            original_trace,
            rtol=5e-13,
            atol=5e-13,
        )
        np.testing.assert_array_max_ulp(
            np.asarray(gauged_trace),
            np.asarray(original_trace),
            maxulp=32,
        )

    @pytest.mark.parametrize("seed", (3, 29))
    def test_random_u3_preserves_full_trace(self, seed: int) -> None:
        """Preserve the complete U(3) trace while rejecting a partial group.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        depths: Array = -jnp.log(jnp.asarray((0.1, 0.4, 0.9)))
        canonical: Array = jnp.eye(3, dtype=jnp.complex128)
        transformed: Array = _random_unitary(jax.random.key(seed), 3)
        original: DiagonalizedBands = _bands(
            jnp.zeros((1, 3)),
            canonical[None, :, :],
            depths,
        )
        gauged: DiagonalizedBands = eqx.tree_at(
            lambda item: item.eigenvectors,
            original,
            transformed[None, :, :],
        )

        with pytest.raises(RuntimeError, match="cuts a degenerate"):
            layer_resolved_group_traces(gauged, ((0, 1),), 1.0)
        expected: Array = layer_resolved_group_traces(
            original,
            ((0, 1, 2),),
            1.0,
        )
        actual: Array = layer_resolved_group_traces(
            gauged,
            ((0, 1, 2),),
            1.0,
        )
        assert jnp.allclose(actual, expected, rtol=5e-13, atol=5e-13)
        np.testing.assert_array_max_ulp(
            np.asarray(actual),
            np.asarray(expected),
            maxulp=32,
        )


class TestSlabScaling:
    """Certify bounded execution and static-shape evidence."""

    @pytest.mark.rss_limit_mb(1024)
    def test_chunked_values_and_remat_grad_match_nonchunked(self) -> None:
        """Match values and gradients on a generated three-layer slab.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        model: TBModel = _scaling_slab(3)
        kpoints: Array = jnp.stack(
            (
                jnp.linspace(-0.47, 0.43, 16),
                jnp.linspace(0.03, 0.19, 16),
                jnp.zeros((16,)),
            ),
            axis=-1,
        )
        expected: Array = eigvalsh_bands(model, kpoints)
        actual: Array = eigvalsh_bands_chunked(model, kpoints, 4)

        def loss(scale: Array, chunked: bool) -> Array:
            changed: TBModel = eqx.tree_at(
                lambda item: item.hopping_amplitudes,
                model,
                scale * model.hopping_amplitudes,
            )
            values: Array = (
                eigvalsh_bands_chunked(changed, kpoints, 4)
                if chunked
                else eigvalsh_bands(changed, kpoints)
            )
            return jnp.sum(jnp.sin(0.7 * values) + 0.13 * values**2)

        chunked_gradient: Array = jax.grad(loss, argnums=0)(1.1, True)
        ordinary_gradient: Array = jax.grad(loss, argnums=0)(1.1, False)
        assert jnp.allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert jnp.allclose(
            chunked_gradient,
            ordinary_gradient,
            rtol=1e-12,
            atol=1e-12,
        )
        assert jnp.abs(ordinary_gradient) > 1e-9

    def test_jaxpr_has_no_full_k_hamiltonian(self) -> None:
        """Keep the dense Hamiltonian live axis bounded by the chunk size.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        n_kpoints: int = 32
        n_orbitals: int = 16
        chunk_size: int = 4
        model: TBModel = _scaling_slab(4)
        assert model.onsite_energies.shape == (n_orbitals,)
        kpoints: Array = jnp.zeros((n_kpoints, 3), dtype=jnp.float64)
        jaxpr: ClosedJaxpr = jax.make_jaxpr(
            lambda points: eigvalsh_bands_chunked(
                model,
                points,
                chunk_size,
            )
        )(kpoints)
        shapes: list[Tuple[int, ...]] = []
        _collect_shapes(jaxpr, shapes)

        assert (chunk_size, n_orbitals, n_orbitals) in shapes
        assert (n_kpoints, n_orbitals, n_orbitals) not in shapes

    def test_one_compile_for_padded_path_sweep_and_static_retrace(
        self,
    ) -> None:
        """Reuse one padded shape and retrace exactly once for a new design.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        active_length: Any
        fixed_model: Any
        fixed_specification: Any
        termination_model: Any
        termination_specification: Any
        thickness_model: Any
        thickness_specification: Any
        bulk: TBModel = _scaling_bulk_model(alternating_species=True)
        fixed_model, fixed_specification = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=3 * 1.3,
            vacuum_ang=4.0,
            termination=("X", "Y"),
        )
        thickness_model, thickness_specification = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=4 * 1.3,
            vacuum_ang=4.0,
            termination=("X", "Y"),
        )
        termination_model, termination_specification = gen_slab(
            bulk,
            miller=(0, 0, 1),
            thickness_ang=4 * 1.3,
            vacuum_ang=4.0,
            termination=("Y", "X"),
        )
        kpoints: Array = jnp.stack(
            (
                jnp.linspace(-0.4, 0.4, 16),
                jnp.zeros((16,)),
                jnp.zeros((16,)),
            ),
            axis=-1,
        )
        trace_count: list[int] = [0]

        def counted(
            candidate: TBModel,
            points: Array,
            active_mask: Array,
        ) -> Array:
            trace_count[0] += 1
            values: Array = eigvalsh_bands_chunked(candidate, points, 4)
            return jnp.sum(values * active_mask[:, None])

        compiled: Callable[..., Array] = eqx.filter_jit(counted)
        for active_length in (5, 9, 16, 7):
            mask: Array = jnp.arange(16) < active_length
            compiled(
                fixed_model,
                kpoints,
                mask,
            ).block_until_ready()
            assert trace_count[0] == 1

        compiled(
            thickness_model,
            kpoints,
            jnp.ones((16,), dtype=bool),
        ).block_until_ready()
        assert trace_count[0] == 2
        compiled(
            termination_model,
            kpoints,
            jnp.ones((16,), dtype=bool),
        ).block_until_ready()
        assert trace_count[0] == 3
        assert fixed_specification.termination == ("X", "Y")
        assert thickness_specification.termination == ("X", "Y")
        assert termination_specification.termination == ("Y", "X")
        assert (
            fixed_model.onsite_energies.shape
            != thickness_model.onsite_energies.shape
        )
        assert (
            thickness_specification.bulk_atom_of_slab_atom
            != termination_specification.bulk_atom_of_slab_atom
        )
        assert thickness_model.hopping_pairs != termination_model.hopping_pairs

    @pytest.mark.parametrize("chunk_size", (0, -1, 3))
    def test_rejects_invalid_chunk_contract(self, chunk_size: int) -> None:
        """Reject nonpositive or non-dividing static chunk sizes.

        Exercise this slab condition with fixed fixtures.

        Notes
        -----
        Compare outputs with declared numerical or structural references.
        """
        model: TBModel = _diagonal_model(4)
        with pytest.raises(ValueError, match="chunk_size|divisible"):
            eigvalsh_bands_chunked(model, jnp.zeros((8, 3)), chunk_size)
