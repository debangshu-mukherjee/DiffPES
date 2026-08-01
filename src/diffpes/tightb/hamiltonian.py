r"""Assemble native tight-binding Bloch Hamiltonians.

Extended Summary
----------------
The module assembles a validated :class:`~diffpes.types.TBModel` in the
basis-position Bloch gauge. Exact integer cells remain static connectivity;
the model geometry determines fractional bond displacements at runtime.
One vectorized scatter operation accumulates all hoppings.

Routine Listings
----------------
:func:`bloch_hamiltonian`
    Assemble one basis-position-gauge Bloch Hamiltonian.
:func:`bloch_hamiltonian_batch`
    Assemble Bloch Hamiltonians for a batch of fractional k-points.

Notes
-----
The model factory requires an explicitly Hermitian-closed hopping list. This
module rechecks the numerical closure at evaluation time so differentiable
updates cannot bypass that invariant. It never repairs input with
``(H + H.conj().T) / 2``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import (
    Array,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int32,
    jaxtyped,
)

from diffpes.types import EPS, TBModel

from .soc import soc_matrix


def _reverse_hopping_indices(model: TBModel) -> Int[Array, " n_hop"]:
    """Derive the reverse-entry permutation from exact static metadata."""
    records: tuple[tuple[int, int, tuple[int, int, int]], ...] = tuple(
        (pair[0], pair[1], cell)
        for pair, cell in zip(
            model.hopping_pairs,
            model.hopping_cells,
            strict=True,
        )
    )
    lookup: dict[tuple[int, int, tuple[int, int, int]], int] = {
        record: index for index, record in enumerate(records)
    }
    reverse: tuple[int, ...] = tuple(
        lookup[(orbital_j, orbital_i, (-cell[0], -cell[1], -cell[2]))]
        for orbital_i, orbital_j, cell in records
    )
    indices: Int32[Array, " n_hop"] = jnp.asarray(reverse, dtype=jnp.int32)
    return indices


def _validated_hopping_amplitudes(
    model: TBModel,
) -> Complex[Array, " n_hop"]:
    """Validate traced hopping invariants again."""
    amplitudes: Complex[Array, " n_hop"] = eqx.error_if(
        model.hopping_amplitudes,
        ~jnp.all(jnp.isfinite(model.hopping_amplitudes)),
        "bloch_hamiltonian: hopping amplitudes finite",
    )
    reverse_indices: Int[Array, " n_hop"] = _reverse_hopping_indices(model)
    reverse_amplitudes: Complex[Array, " n_hop"] = amplitudes[reverse_indices]
    amplitudes = eqx.error_if(
        amplitudes,
        ~jnp.all(jnp.abs(reverse_amplitudes - jnp.conj(amplitudes)) <= EPS),
        "bloch_hamiltonian: hopping amplitudes must remain Hermitian-closed",
    )
    return amplitudes  # noqa: RET504 -- assign-before-return is required.


def _assemble_bloch_hamiltonian(
    model: TBModel,
    k: Float[Array, " 3"],
    amplitudes: Complex[Array, " n_hop"],
) -> Complex[Array, "n_orb n_orb"]:
    """Assemble one Hamiltonian from already validated amplitudes."""
    n_orbitals: int = model.onsite_energies.shape[0]
    if model.orbital_positions is None:
        atom_indices: Int32[Array, " n_orb"] = jnp.asarray(
            model.basis.atom_indices,
            dtype=jnp.int32,
        )
        orbital_positions: Float[Array, "n_orb 3"] = model.geometry.positions[
            atom_indices
        ]
    else:
        orbital_positions = eqx.error_if(
            model.orbital_positions,
            ~jnp.all(jnp.isfinite(model.orbital_positions)),
            "bloch_hamiltonian: orbital positions finite",
        )
    pairs: Int32[Array, "n_hop 2"] = jnp.asarray(
        model.hopping_pairs,
        dtype=jnp.int32,
    ).reshape((-1, 2))
    cells: Float64[Array, "n_hop 3"] = jnp.asarray(
        model.hopping_cells,
        dtype=jnp.float64,
    ).reshape((-1, 3))
    source: Int32[Array, " n_hop"] = pairs[:, 0]
    target: Int32[Array, " n_hop"] = pairs[:, 1]
    displacements: Float[Array, "n_hop 3"] = (
        cells + orbital_positions[target] - orbital_positions[source]
    )
    phases: Complex[Array, " n_hop"] = jnp.exp(
        2j * jnp.pi * (displacements @ k)
    )
    flat_indices: Int32[Array, " n_hop"] = source * n_orbitals + target
    flattened: Complex128[Array, " n_flat"] = jnp.zeros(
        (n_orbitals * n_orbitals,),
        dtype=jnp.complex128,
    )
    flattened = flattened.at[flat_indices].add(amplitudes * phases)
    hamiltonian: Complex128[Array, "n_orb n_orb"] = flattened.reshape(
        (n_orbitals, n_orbitals)
    )
    diagonal_indices: Int32[Array, " n_orb"] = jnp.arange(
        n_orbitals,
        dtype=jnp.int32,
    )
    hamiltonian = hamiltonian.at[
        diagonal_indices,
        diagonal_indices,
    ].add(model.onsite_energies)
    if model.spinor:
        hamiltonian = hamiltonian + soc_matrix(
            model.basis,
            model.shell_index,
            model.soc_lambdas,
        )
    else:
        hamiltonian = eqx.error_if(
            hamiltonian,
            ~jnp.all(model.soc_lambdas == 0.0),
            "bloch_hamiltonian: nonzero SOC requires spin doubling",
        )
    hamiltonian = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.isfinite(hamiltonian)),
        "bloch_hamiltonian: assembled Hamiltonian finite",
    )
    hamiltonian = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.abs(hamiltonian - hamiltonian.conj().T) <= EPS),
        "bloch_hamiltonian: assembled Hamiltonian Hermitian",
    )
    return hamiltonian  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def bloch_hamiltonian(  # noqa: DOC502
    model: TBModel,
    k: Float[Array, " 3"],
) -> Complex[Array, "n_orb n_orb"]:
    r"""Assemble one basis-position-gauge Bloch Hamiltonian.

    The phase of hopping record ``(i, j, R)`` is
    :math:`\exp[2\pi i k\cdot(R+\tau_j-\tau_i)]`. Here ``R`` is the exact
    model cell. Explicit ``model.orbital_positions`` supply each
    :math:`\tau` when present. Otherwise ``basis.atom_indices`` selects them
    from ``geometry.positions``.

    :see: :class:`~.test_hamiltonian.TestBlochHamiltonian`

    Parameters
    ----------
    model : TBModel
        Validated tight-binding model. Static connectivity changes trigger
        retracing.
    k : Float[Array, " 3"]
        Fractional reciprocal-space k-point.

    Returns
    -------
    hamiltonian : Complex[Array, "n_orb n_orb"]
        Complex Hermitian Bloch Hamiltonian in eV.

    Raises
    ------
    EquinoxRuntimeError
        If hopping amplitudes or SOC couplings are non-finite, hopping
        amplitudes are no longer conjugate-closed, or the assembled
        Hamiltonian is non-finite or non-Hermitian.

    Notes
    -----
    The algorithm derives all bond displacements at once, evaluates their
    phases, and performs one flattened scatter-add. It then adds onsite
    energies to the diagonal and, for a spinor model, adds the shell-resolved
    atomic SOC matrix exactly once. The onsite and basis lengths already
    represent the complete Hamiltonian dimension; ``spinor=True`` never
    doubles it again.
    """
    amplitudes: Complex[Array, " n_hop"] = _validated_hopping_amplitudes(model)
    hamiltonian: Complex[Array, "n_orb n_orb"] = _assemble_bloch_hamiltonian(
        model, k, amplitudes
    )
    return hamiltonian  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def bloch_hamiltonian_batch(  # noqa: DOC502
    model: TBModel,
    kpoints: Float[Array, "n_k 3"],
) -> Complex[Array, "n_k n_orb n_orb"]:
    """Assemble Bloch Hamiltonians for a batch of fractional k-points.

    The function maps the single-point assembler over the leading k-point
    axis while retaining one shared model structure.

    :see: :class:`~.test_hamiltonian.TestBlochHamiltonianBatch`

    Parameters
    ----------
    model : TBModel
        Validated tight-binding model. Static connectivity changes trigger
        retracing.
    kpoints : Float[Array, "n_k 3"]
        Fractional reciprocal-space k-points.

    Returns
    -------
    hamiltonians : Complex[Array, "n_k n_orb n_orb"]
        Complex Hermitian Bloch Hamiltonians in eV.

    Raises
    ------
    EquinoxRuntimeError
        If hopping amplitudes or SOC couplings are non-finite, hopping
        amplitudes are no longer conjugate-closed, or an assembled
        Hamiltonian is non-finite or non-Hermitian.

    Notes
    -----
    :func:`jax.vmap` traces one assembler and broadcasts the shared model over
    all supplied fractional k-points.
    """
    amplitudes: Complex[Array, " n_hop"] = _validated_hopping_amplitudes(model)
    hamiltonians: Complex[Array, "n_k n_orb n_orb"] = jax.vmap(
        lambda point: _assemble_bloch_hamiltonian(model, point, amplitudes)
    )(kpoints)
    return hamiltonians


__all__: list[str] = [
    "bloch_hamiltonian",
    "bloch_hamiltonian_batch",
]
