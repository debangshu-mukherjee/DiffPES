"""Rotate complete-shell tight-binding models.

Extended Summary
----------------
This module applies real-harmonic and spin Wigner rotations to complete shells.

Routine Listings
----------------
:func:`rotate_tb_model`
    Construct a rotated complete-shell tight-binding model.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.maths import (
    real_harmonic_unitary,
    safe_arccos,
    safe_arctan2,
    wigner_d,
)
from diffpes.types import (
    CrystalGeometry,
    TBModel,
    make_crystal_geometry,
    make_tb_model,
)


def _shell_groups(
    model: TBModel,
) -> Dict[Tuple[int, int, int, int], List[int]]:
    """PRIVATE: Compute orbital groups by site, principal shell, l, and spin.

    Parameters
    ----------
    model : TBModel
        Model that supplies the static basis metadata.

    Returns
    -------
    groups : Dict[Tuple[int, int, int, int], List[int]]
        Map from ``(atom, n, l, spin)`` to the basis positions of that
        shell's orbitals.

    Notes
    -----
    A spinless model uses the placeholder spin ``0`` for every orbital,
    so each spatial shell forms one group. The groups are the units on
    which Wigner rotation blocks act.
    """
    groups: Dict[Tuple[int, int, int, int], List[int]] = {}
    orbital: int
    atom: int
    principal: int
    angular: int
    spin: int
    for orbital, (
        atom,
        principal,
        angular,
        spin,
    ) in enumerate(
        zip(
            model.basis.atom_indices,
            model.basis.n,
            model.basis.l,
            model.basis.spin if model.spinor else (0,) * len(model.basis.n),
            strict=True,
        )
    ):
        groups.setdefault((atom, principal, angular, spin), []).append(orbital)
    return groups


def _missing_magnetic_numbers(
    model: TBModel,
) -> Dict[
    Tuple[int, int, int, int],
    Tuple[int, ...],
]:
    """PRIVATE: Return missing m values for every incomplete shell.

    Parameters
    ----------
    model : TBModel
        Model carrying the registered shells to audit.

    Returns
    -------
    missing : Dict[Tuple[int, int, int, int], Tuple[int, ...]]
        Map from every incomplete ``(atom, n, l, spin)`` shell to its
        absent magnetic numbers. Complete shells do not appear.

    Notes
    -----
    A shell is incomplete when an ``m`` value from ``-l..+l`` is absent,
    an ``m`` value repeats, or the orbital count differs from
    ``2*l + 1``. A duplicated shell can report an empty absent tuple, so
    callers must test dictionary membership, not tuple truth. Only
    complete shells support a covariant Wigner rotation.
    """
    missing: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}
    key: Tuple[int, int, int, int]
    indices: List[int]
    for key, indices in _shell_groups(model).items():
        angular: int = key[2]
        present: List[int] = [model.basis.m[index] for index in indices]
        expected: set[int] = set(range(-angular, angular + 1))
        absent: Tuple[int, ...] = tuple(sorted(expected - set(present)))
        duplicated: bool = len(present) != len(set(present))
        if absent or duplicated or len(present) != 2 * angular + 1:
            missing[key] = absent
    return missing


def _rotation_euler_zyz(
    rotation: Float64[Array, "3 3"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Convert one active Cartesian rotation to guarded z-y-z angles.

    Parameters
    ----------
    rotation : Float64[Array, "3 3"]
        Proper active rotation matrix.

    Returns
    -------
    result : tuple
        Euler angles ``(alpha, beta, gamma)`` in radians for the z-y-z
        convention, each a scalar array.

    Notes
    -----
    ``beta`` comes from :func:`diffpes.maths.safe_arccos` of the lower
    right element. Away from the poles, generic ``arctan2`` expressions
    read the third row and column. Within ``1e-12`` of
    ``sin(beta) = 0`` the two z rotations degenerate. The guarded branch
    assigns the whole in-plane angle to ``alpha`` from the upper block
    and sets ``gamma`` to zero. The sign branch follows the pole. All
    selections use ``jnp.where``, so the angles stay traced.
    """
    beta: Float64[Array, ""] = safe_arccos(rotation[2, 2])
    sine_beta: Float64[Array, ""] = jnp.sin(beta)
    generic_alpha: Float64[Array, ""] = safe_arctan2(
        rotation[1, 2],
        rotation[0, 2],
    )
    generic_gamma: Float64[Array, ""] = safe_arctan2(
        rotation[2, 1],
        -rotation[2, 0],
    )
    positive_alpha: Float64[Array, ""] = safe_arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )
    negative_alpha: Float64[Array, ""] = safe_arctan2(
        -rotation[1, 0],
        -rotation[0, 0],
    )
    pole_alpha: Float64[Array, ""] = jnp.where(
        rotation[2, 2] >= 0.0,
        positive_alpha,
        negative_alpha,
    )
    alpha: Float64[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_alpha,
        pole_alpha,
    )
    gamma: Float64[Array, ""] = jnp.where(
        jnp.abs(sine_beta) > 1e-12,  # noqa: PLR2004
        generic_gamma,
        0.0,
    )
    result: Tuple[
        Float64[Array, ""],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (alpha, beta, gamma)
    return result


def _real_wigner(
    angular: int,
    alpha: Float64[Array, ""],
    beta: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Complex128[Array, "m1 m2"]:
    """PRIVATE: Convert the complex Wigner representation to real harmonics.

    Parameters
    ----------
    angular : int
        Static shell angular momentum.
    alpha : Float64[Array, ""]
        First z-y-z Euler angle in radians.
    beta : Float64[Array, ""]
        Second z-y-z Euler angle in radians.
    gamma : Float64[Array, ""]
        Third z-y-z Euler angle in radians.

    Returns
    -------
    real_matrix : Complex128[Array, "m1 m2"]
        Wigner rotation of the shell in the package real-harmonic basis.

    Notes
    -----
    The transform is the package operator convention
    ``U.conj() @ D @ U.T``, with :func:`diffpes.maths.wigner_d`
    supplying the complex-harmonic matrix and
    :func:`diffpes.maths.real_harmonic_unitary` supplying ``U``.
    """
    complex_matrix: Complex128[Array, "m1 m2"] = wigner_d(
        angular,
        alpha,
        beta,
        gamma,
    )
    unitary: Complex128[Array, "m1 m2"] = real_harmonic_unitary(angular)
    real_matrix: Complex128[Array, "m1 m2"] = (
        unitary.conj() @ complex_matrix @ unitary.T
    )
    return real_matrix


def _spin_half_wigner(
    alpha: Float64[Array, ""],
    beta: Float64[Array, ""],
    gamma: Float64[Array, ""],
) -> Complex128[Array, "2 2"]:
    """PRIVATE: Construct the spin-half Wigner matrix in (-1, +1) order.

    Parameters
    ----------
    alpha : Float64[Array, ""]
        First z-y-z Euler angle in radians.
    beta : Float64[Array, ""]
        Second z-y-z Euler angle in radians.
    gamma : Float64[Array, ""]
        Third z-y-z Euler angle in radians.

    Returns
    -------
    matrix : Complex128[Array, "2 2"]
        Spinor rotation with rows and columns ordered as spin down then
        spin up.

    Notes
    -----
    The small-d factor uses ``cos(beta/2)`` and ``sin(beta/2)``. The
    phases ``exp(-i m alpha)`` and ``exp(-i m gamma)`` multiply from the
    left and right with ``m = (-1/2, +1/2)``. The down--up ordering
    matches the package C8 spin storage convention.
    """
    cosine: Float64[Array, ""] = jnp.cos(0.5 * beta)
    sine: Float64[Array, ""] = jnp.sin(0.5 * beta)
    small: Float64[Array, "2 2"] = jnp.stack(
        (
            jnp.stack((cosine, sine)),
            jnp.stack((-sine, cosine)),
        )
    )
    magnetic: Float64[Array, " 2"] = jnp.asarray(
        (-0.5, 0.5),
        dtype=beta.dtype,
    )
    alpha_phase: Complex128[Array, " 2"] = jnp.exp(-1j * magnetic * alpha)
    gamma_phase: Complex128[Array, " 2"] = jnp.exp(-1j * magnetic * gamma)
    matrix: Complex128[Array, "2 2"] = (
        alpha_phase[:, None] * small * gamma_phase[None, :]
    )
    return matrix


def _orbital_rotation(
    model: TBModel,
    rotation: Float64[Array, "3 3"],
) -> Complex128[Array, "n_orb n_orb"]:
    """PRIVATE: Assemble the block-diagonal orbital and spin representation.

    Parameters
    ----------
    model : TBModel
        Model whose static basis metadata defines the shell blocks.
    rotation : Float64[Array, "3 3"]
        Proper active Cartesian rotation.

    Returns
    -------
    representation : Complex128[Array, "n_orb n_orb"]
        Unitary rotation representation in the model basis order.

    Notes
    -----
    Guarded z-y-z Euler angles feed one real-harmonic Wigner matrix per
    distinct ``l`` and, for a spinor model, one spin-half matrix. Every
    entry couples two orbitals of the same ``(atom, n, l)`` shell. It
    multiplies the angular factor at their magnetic numbers by the
    spin-half factor at their spin labels; a spinless model uses a unit
    spin factor. Orbitals of different shells never mix, so the matrix
    is block diagonal over complete shells in any basis order.
    """
    alpha: Float64[Array, ""]
    beta: Float64[Array, ""]
    gamma: Float64[Array, ""]
    alpha, beta, gamma = _rotation_euler_zyz(rotation)
    n_orbitals: int = len(model.basis.n)
    representation: Complex128[Array, "n_orb n_orb"] = jnp.zeros(
        (n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    angular_matrices: Dict[int, Complex128[Array, "m1 m2"]] = {
        angular: _real_wigner(angular, alpha, beta, gamma)
        for angular in set(model.basis.l)
    }
    spin_matrix: Complex128[Array, "2 2"] | None = (
        _spin_half_wigner(alpha, beta, gamma) if model.spinor else None
    )
    row: int
    column: int
    for row in range(n_orbitals):
        for column in range(n_orbitals):
            same_shell: bool = (
                model.basis.atom_indices[row]
                == model.basis.atom_indices[column]
                and model.basis.n[row] == model.basis.n[column]
                and model.basis.l[row] == model.basis.l[column]
            )
            if not same_shell:
                continue
            angular: int = model.basis.l[row]
            angular_factor: Complex128[Array, ""] = angular_matrices[angular][
                model.basis.m[row] + angular,
                model.basis.m[column] + angular,
            ]
            if spin_matrix is None:
                spin_factor: complex | Complex128[Array, ""] = (
                    1.0 if row == column or same_shell else 0.0
                )
            else:
                row_spin: int = 0 if model.basis.spin[row] == -1 else 1
                column_spin: int = 0 if model.basis.spin[column] == -1 else 1
                spin_factor = spin_matrix[row_spin, column_spin]
            representation = representation.at[row, column].set(
                angular_factor * spin_factor
            )
    return representation


def _translation_blocks(
    model: TBModel,
) -> Tuple[Tuple[Tuple[int, int, int], ...], Complex128[Array, "n_r n_o n_o"]]:
    """PRIVATE: Materialize translation blocks with diagonal onsite terms.

    Parameters
    ----------
    model : TBModel
        Model supplying the sparse hopping records to densify.

    Returns
    -------
    result : tuple
        Sorted static cell tuples and one dense complex block per cell.
        Each block holds the hoppings in eV scattered by pair; the
        zero-cell block also carries the onsite energies on its
        diagonal.

    Notes
    -----
    The zero cell is always present even without home-cell hoppings, so
    the onsite diagonal has a destination. Dense blocks let one unitary
    conjugation rotate arbitrary onsite and hopping structure without
    assuming shell degeneracy.
    """
    zero_cell: Tuple[int, int, int] = (0, 0, 0)
    cells: Tuple[Tuple[int, int, int], ...] = tuple(
        sorted(set(model.hopping_cells) | {zero_cell})
    )
    n_orbitals: int = len(model.basis.n)
    blocks: Complex128[Array, "n_r n_o n_o"] = jnp.zeros(
        (len(cells), n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    cell_lookup: Dict[Tuple[int, int, int], int] = {
        cell: index for index, cell in enumerate(cells)
    }
    hopping: int
    pair: Tuple[int, int]
    cell: Tuple[int, int, int]
    for hopping, (pair, cell) in enumerate(
        zip(model.hopping_pairs, model.hopping_cells, strict=True)
    ):
        blocks = blocks.at[
            cell_lookup[cell],
            pair[0],
            pair[1],
        ].add(model.hopping_amplitudes[hopping])
    diagonal: Int32[Array, " n_orb"] = jnp.arange(n_orbitals, dtype=jnp.int32)
    blocks = blocks.at[cell_lookup[zero_cell], diagonal, diagonal].add(
        model.onsite_energies
    )
    result: Tuple[
        Tuple[Tuple[int, int, int], ...],
        Complex128[Array, "n_r n_o n_o"],
    ] = (cells, blocks)
    return result


@jaxtyped(typechecker=beartype)
def rotate_tb_model(  # noqa: DOC503
    model: TBModel,
    rotation: Float64[Array, "3 3"],
) -> TBModel:
    """Construct a rotated complete-shell tight-binding model.

    Parameters
    ----------
    model : TBModel
        Bulk model in a real-harmonic orbital basis.
    rotation : Float64[Array, "3 3"]
        Proper active Cartesian rotation from the old frame to the new frame.

    Returns
    -------
    rotated_model : TBModel
        Ordinary model in the rotated Cartesian and orbital frame.

    Raises
    ------
    ValueError
        If an incomplete ``m`` shell receives a nonidentity rotation.
    EquinoxRuntimeError
        If ``rotation`` is non-finite, improper, or nonorthogonal.

    Notes
    -----
    The block-diagonal real-harmonic representation conjugates every exact
    translation block, including the onsite block. Dense translation blocks
    preserve generic onsite and hopping matrices without assuming shell
    degeneracy. The added static connectivity changes representation, not
    physical coupling.

    :see: :class:`~.test_slab_rotation.TestRotateTbModel`
    """
    rotation_array: Float64[Array, "3 3"] = jnp.asarray(
        rotation,
        dtype=jnp.float64,
    )
    missing: Dict[
        Tuple[int, int, int, int],
        Tuple[int, ...],
    ] = _missing_magnetic_numbers(model)
    if missing:
        error: TypeError
        try:
            identity_rotation: bool = bool(
                np.allclose(
                    np.asarray(rotation_array),
                    np.eye(3),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        except TypeError as error:
            message: str = (
                "incomplete shells require an eager identity/no-op rotation"
            )
            raise ValueError(message) from error
        if identity_rotation:
            return model
        diagnostics: str = "; ".join(
            f"{key}: missing m={values}"
            for key, values in sorted(missing.items())
        )
        message = (
            "nonidentity rotation requires complete registered m shells; "
            f"{diagnostics}"
        )
        raise ValueError(message)

    rotation_array = eqx.error_if(
        rotation_array,
        ~jnp.all(jnp.isfinite(rotation_array)),
        "rotate_tb_model: rotation must be finite",
    )
    identity: Float64[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)
    rotation_array = eqx.error_if(
        rotation_array,
        jnp.max(jnp.abs(rotation_array.T @ rotation_array - identity)) > 1e-10,  # noqa: PLR2004
        "rotate_tb_model: rotation must be orthogonal",
    )
    rotation_array = eqx.error_if(
        rotation_array,
        jnp.abs(jnp.linalg.det(rotation_array) - 1.0) > 1e-10,  # noqa: PLR2004
        "rotate_tb_model: rotation must be proper",
    )
    representation: Complex128[Array, "n_orb n_orb"] = _orbital_rotation(
        model,
        rotation_array,
    )
    cells: Tuple[Tuple[int, int, int], ...]
    blocks: Complex128[Array, "n_r n_o n_o"]
    cells, blocks = _translation_blocks(model)
    rotated_blocks: Complex128[Array, "n_r n_o n_o"] = (
        representation[None, :, :]
        @ blocks
        @ representation.conj().T[None, :, :]
    )
    n_orbitals: int = len(model.basis.n)
    hopping_pairs: Tuple[Tuple[int, int], ...] = tuple(
        (row, column)
        for _cell in cells
        for row in range(n_orbitals)
        for column in range(n_orbitals)
    )
    hopping_cells: Tuple[Tuple[int, int, int], ...] = tuple(
        cell
        for cell in cells
        for _row in range(n_orbitals)
        for _column in range(n_orbitals)
    )
    hopping_amplitudes: Complex128[Array, " n_hop"] = rotated_blocks.reshape(
        -1
    )
    rotated_lattice: Float64[Array, "3 3"] = (
        model.geometry.lattice @ rotation_array.T
    )
    rotated_geometry: CrystalGeometry = make_crystal_geometry(
        lattice=rotated_lattice,
        positions=model.geometry.positions,
        species=model.geometry.species,
    )
    rotated_model: TBModel = make_tb_model(
        hopping_amplitudes=hopping_amplitudes,
        onsite_energies=jnp.zeros_like(model.onsite_energies),
        soc_lambdas=model.soc_lambdas,
        geometry=rotated_geometry,
        basis=model.basis,
        hopping_pairs=hopping_pairs,
        hopping_cells=hopping_cells,
        shell_index=model.shell_index,
        spinor=model.spinor,
        orbital_positions=model.orbital_positions,
        depths=model.depths,
    )
    return rotated_model


__all__: list[str] = [
    "rotate_tb_model",
]
