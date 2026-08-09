r"""Construct atomic spin--orbit coupling in the real-cubic basis.

Extended Summary
----------------
This module implements the atomic spin--orbit term
:math:`\lambda\,\mathbf L\cdot\mathbf S`. It constructs orbital
angular-momentum matrices in the complex-harmonic basis. Their axes run from
:math:`m=-l` through :math:`m=+l`. The implementation applies the package
real-harmonic unitary and combines each operator with down--up block-major
spin.

Routine Listings
----------------
:func:`l_matrices`
    Construct orbital angular-momentum matrices in the complex basis.
:func:`soc_matrix`
    Assemble shell-resolved atomic SOC in an arbitrary spinor basis.
:func:`soc_shell_block`
    Construct a unit-strength real-cubic :math:`\mathbf L\cdot\mathbf S`.
:func:`spin_double_basis`
    Create a spin-doubled basis in the declared down--up block order.
:func:`spin_double_model`
    Create a spin-doubled model with spin-diagonal down--up blocks.

Notes
-----
The real-cubic operator transform is
:math:`O_\mathrm{real}=U^*O_\mathrm{complex}U^T`, convention C13 of the
ARPES forward canon. Spin convention C8 stores the original spinless block
as spin down (tag ``-1``) and appends spin up (tag ``+1``). The corresponding
Pauli matrices are the explicitly permuted down--up representation.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.maths import real_harmonic_unitary
from diffpes.types import (
    L_MAX,
    OrbitalBasis,
    TBModel,
    make_orbital_basis,
    make_tb_model,
)


def _validate_angular_momentum(l: int) -> None:
    """PRIVATE: Validate one static angular-momentum quantum number.

    Parameters
    ----------
    l : int
        Candidate static orbital angular momentum.

    Raises
    ------
    ValueError
        If ``l`` is not an integer inside ``[0, L_MAX]``.

    Notes
    -----
    The check reads the static Python value before any table
    construction, so ``l`` participates in the JAX trace signature.
    """
    if type(l) is not int or l < 0 or l > L_MAX:
        message: str = f"l={l} must satisfy 0 <= l <= {L_MAX}"
        raise ValueError(message)


def _validate_soc_structure(
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
    soc_lambdas: Float64[Array, " n_shells"],
) -> None:
    """PRIVATE: Validate static shell membership for SOC construction.

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital basis whose metadata defines the shell groups.
    shell_index : tuple[int, ...]
        Candidate orbital-to-shell IDs with ``-1`` for excluded orbitals.
    soc_lambdas : Float64[Array, " n_shells"]
        Shell coupling energies in eV; the check reads only the static
        shape.

    Raises
    ------
    ValueError
        If ``shell_index`` is not a tuple with one integer ``>= -1`` per
        orbital, or ``soc_lambdas`` is not one-dimensional with length
        ``max(shell_index) + 1``. Also raised when nonnegative IDs are
        not contiguous from zero or an active shell lacks an explicit
        spinor basis. Also raised when a shell spans more than one
        ``(atom, n, l)`` group or its two spin sectors do not carry
        matching unique ``m`` values.

    Notes
    -----
    Every check reads static Python metadata, so an invalid shell
    declaration fails before tracing. Matching spin sectors guarantee
    that the canonical ``soc_shell_block`` rows and columns exist for
    every listed orbital.
    """
    n_orbitals: int = len(basis.n)
    if type(shell_index) is not tuple:
        message: str = "shell_index must be a tuple"
        raise ValueError(message)
    if len(shell_index) != n_orbitals:
        message = "shell_index must have one entry per basis orbital"
        raise ValueError(message)
    if soc_lambdas.ndim != 1:
        message = "soc_lambdas must be one-dimensional"
        raise ValueError(message)
    if any(type(index) is not int or index < -1 for index in shell_index):
        message = (
            "shell_index entries must be integers greater than or equal to -1"
        )
        raise ValueError(message)
    expected_shells: int = max(shell_index, default=-1) + 1
    if soc_lambdas.shape[0] != expected_shells:
        message = "soc_lambdas length must equal max(shell_index) + 1"
        raise ValueError(message)
    if {index for index in shell_index if index >= 0} != set(
        range(expected_shells)
    ):
        message = "nonnegative shell_index IDs must be contiguous from 0"
        raise ValueError(message)
    if expected_shells > 0 and len(basis.spin) != n_orbitals:
        message = "active SOC shells require an explicit spinor basis"
        raise ValueError(message)

    shell: int
    for shell in range(expected_shells):
        orbitals: tuple[int, ...] = tuple(
            orbital
            for orbital, candidate in enumerate(shell_index)
            if candidate == shell
        )
        reference: int = orbitals[0]
        group: tuple[int, int, int] = (
            basis.atom_indices[reference],
            basis.n[reference],
            basis.l[reference],
        )
        if any(
            (
                basis.atom_indices[orbital],
                basis.n[orbital],
                basis.l[orbital],
            )
            != group
            for orbital in orbitals
        ):
            message = "each SOC shell must map to one (atom, n, l) group"
            raise ValueError(message)

        down_m: tuple[int, ...] = tuple(
            basis.m[orbital]
            for orbital in orbitals
            if basis.spin[orbital] == -1
        )
        up_m: tuple[int, ...] = tuple(
            basis.m[orbital]
            for orbital in orbitals
            if basis.spin[orbital] == 1
        )
        if (
            len(set(down_m)) != len(down_m)
            or len(set(up_m)) != len(up_m)
            or set(down_m) != set(up_m)
            or not down_m
        ):
            message = (
                "each SOC shell requires matching, unique orbital m values "
                "in both spin sectors"
            )
            raise ValueError(message)


def _shell_maps(
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
    shell: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """PRIVATE: Resolve one SOC shell to global and canonical block indices.

    Parameters
    ----------
    basis : OrbitalBasis
        Validated spinor basis with real-harmonic metadata.
    shell_index : tuple[int, ...]
        Static orbital-to-shell IDs.
    shell : int
        Nonnegative shell ID to resolve.

    Returns
    -------
    result : tuple[tuple[int, ...], tuple[int, ...], int]
        Global basis positions of the shell orbitals, their matching
        rows in the canonical down--up full-shell block, and the shell
        angular momentum.

    Notes
    -----
    The canonical block from ``soc_shell_block`` orders each spin sector
    as ``m = -l..+l`` with down before up. The local index is therefore
    ``m + l`` plus an offset of ``2*l + 1`` for spin up. Projected
    subsets such as ``t2g`` select only their own rows, so both index
    tuples can be shorter than the full shell.
    """
    global_indices: tuple[int, ...] = tuple(
        orbital
        for orbital, candidate in enumerate(shell_index)
        if candidate == shell
    )
    angular_momentum: int = basis.l[global_indices[0]]
    shell_size: int = 2 * angular_momentum + 1
    local_indices: tuple[int, ...] = tuple(
        (0 if basis.spin[orbital] == -1 else shell_size)
        + basis.m[orbital]
        + angular_momentum
        for orbital in global_indices
    )
    result: tuple[tuple[int, ...], tuple[int, ...], int] = (
        global_indices,
        local_indices,
        angular_momentum,
    )
    return result


@jaxtyped(typechecker=beartype)
def l_matrices(  # noqa: DOC502 -- validation is shared.
    l: int,  # noqa: E741
) -> tuple[
    Complex128[Array, "m1 m2"],
    Complex128[Array, "m1 m2"],
    Complex128[Array, "m1 m2"],
]:
    r"""Construct orbital angular-momentum matrices in the complex basis.

    The row and column axes run from :math:`m=-l` through :math:`m=+l`.
    Units use :math:`\hbar=1`. The returned order is ``(Lz, L+, L-)``.

    :see: :class:`~.test_soc.TestLMatrices`

    Parameters
    ----------
    l : int
        (**static** -- changing it triggers retracing) Angular momentum,
        restricted to ``0 <= l <= L_MAX``.

    Returns
    -------
    lz : Complex128[Array, "m1 m2"]
        Diagonal :math:`L_z` matrix.
    raising : Complex128[Array, "m1 m2"]
        :math:`L_+` ladder matrix.
    lowering : Complex128[Array, "m1 m2"]
        :math:`L_-` ladder matrix.

    Raises
    ------
    ValueError
        If ``l`` is not an integer in the supported interval.

    Notes
    -----
    The ladder entry from ``m`` to ``m+1`` is
    :math:`\sqrt{l(l+1)-m(m+1)}`. The lowering matrix is the Hermitian
    adjoint of the raising matrix.
    """
    _validate_angular_momentum(l)
    magnetic_numbers: Float64[Array, " m"] = jnp.arange(
        -l,
        l + 1,
        dtype=jnp.float64,
    )
    lz: Complex128[Array, "m1 m2"] = jnp.diag(magnetic_numbers).astype(
        jnp.complex128
    )
    size: int = 2 * l + 1
    raising: Complex128[Array, "m1 m2"] = jnp.zeros(
        (size, size),
        dtype=jnp.complex128,
    )
    magnetic: int
    for magnetic in range(-l, l):
        column: int = magnetic + l
        row: int = column + 1
        coefficient: float = math.sqrt(l * (l + 1) - magnetic * (magnetic + 1))
        raising = raising.at[row, column].set(coefficient)
    lowering: Complex128[Array, "m1 m2"] = raising.conj().T
    result: tuple[
        Complex128[Array, "m1 m2"],
        Complex128[Array, "m1 m2"],
        Complex128[Array, "m1 m2"],
    ] = (lz, raising, lowering)
    return result


@jaxtyped(typechecker=beartype)
def soc_shell_block(  # noqa: DOC502 -- validation is shared.
    l: int,  # noqa: E741
) -> Complex128[Array, "s1 s2"]:
    r"""Construct a unit-strength real-cubic :math:`\mathbf L\cdot\mathbf S`.

    The orbital order within each spin block is real-harmonic ``m=-l..+l``.
    The complete order is all spin down followed by all spin up. Multiplying
    the result by a shell coupling ``lambda`` gives the atomic SOC energy.

    :see: :class:`~.test_soc.TestSocShellBlock`

    Parameters
    ----------
    l : int
        (**static** -- changing it triggers retracing) Angular momentum,
        restricted to ``0 <= l <= L_MAX``.

    Returns
    -------
    block : Complex128[Array, "s1 s2"]
        Hermitian matrix of shape ``(2*(2*l+1), 2*(2*l+1))``.

    Raises
    ------
    ValueError
        If ``l`` is not an integer in the supported interval.

    Notes
    -----
    Convention C13 transforms each orbital operator as
    :math:`U^* O U^T`. Convention C8 uses
    :math:`\sigma_y=\left[\begin{smallmatrix}0&i\\-i&0\end{smallmatrix}
    \right]` and :math:`\sigma_z=\operatorname{diag}(-1,+1)` in down--up
    storage order. Spin operators are :math:`S_a=\sigma_a/2`.
    """
    _validate_angular_momentum(l)
    lz_complex: Complex128[Array, "m1 m2"]
    raising: Complex128[Array, "m1 m2"]
    lowering: Complex128[Array, "m1 m2"]
    lz_complex, raising, lowering = l_matrices(l)
    lx_complex: Complex128[Array, "m1 m2"] = 0.5 * (raising + lowering)
    ly_complex: Complex128[Array, "m1 m2"] = (raising - lowering) / (2j)
    unitary: Complex128[Array, "m1 m2"] = real_harmonic_unitary(l)

    def to_real(
        operator: Complex128[Array, "m1 m2"],
    ) -> Complex128[Array, "m1 m2"]:
        transformed: Complex128[Array, "m1 m2"] = (
            unitary.conj() @ operator @ unitary.T
        )
        return transformed

    lx_real: Complex128[Array, "m1 m2"] = to_real(lx_complex)
    ly_real: Complex128[Array, "m1 m2"] = to_real(ly_complex)
    lz_real: Complex128[Array, "m1 m2"] = to_real(lz_complex)
    sigma_x: Complex128[Array, "2 2"] = jnp.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=jnp.complex128,
    )
    sigma_y: Complex128[Array, "2 2"] = jnp.asarray(
        [[0.0, 1.0j], [-1.0j, 0.0]],
        dtype=jnp.complex128,
    )
    sigma_z: Complex128[Array, "2 2"] = jnp.asarray(
        [[-1.0, 0.0], [0.0, 1.0]],
        dtype=jnp.complex128,
    )
    block: Complex128[Array, "s1 s2"] = 0.5 * (
        jnp.kron(sigma_x, lx_real)
        + jnp.kron(sigma_y, ly_real)
        + jnp.kron(sigma_z, lz_real)
    )
    return block


@jaxtyped(typechecker=beartype)
def spin_double_basis(basis: OrbitalBasis) -> OrbitalBasis:
    """Create a spin-doubled basis in the declared down--up block order.

    The original orbital sequence becomes the spin-down block, tagged ``-1``.
    The function appends an identical spin-up sequence with tag ``+1``.
    Partners retain matching labels and quantum numbers. The explicit
    ``spin`` field carries their physical distinction.

    :see: :class:`~.test_soc.TestSpinDoubleBasis`

    Parameters
    ----------
    basis : OrbitalBasis
        Spinless orbital basis to duplicate.

    Returns
    -------
    doubled : OrbitalBasis
        Basis with twice as many entries and C8 spin tags.

    Raises
    ------
    ValueError
        If the input basis already has explicit spin tags.

    Notes
    -----
    Static metadata follows the original order in each spin block. The
    returned carrier passes through the standard orbital-basis factory.
    """
    if basis.spin:
        message: str = "spin_double_basis requires a spinless basis"
        raise ValueError(message)
    n_orbitals: int = len(basis.n)
    doubled: OrbitalBasis = make_orbital_basis(
        atom_indices=basis.atom_indices + basis.atom_indices,
        n=basis.n + basis.n,
        l=basis.l + basis.l,
        m=basis.m + basis.m,
        spin=(-1,) * n_orbitals + (1,) * n_orbitals,
        labels=basis.labels + basis.labels,
    )
    return doubled


@jaxtyped(typechecker=beartype)
def spin_double_model(model: TBModel) -> TBModel:
    """Create a spin-doubled model with spin-diagonal down--up blocks.

    The function copies the complete hopping list into the appended spin-up
    block and shifts its orbital indices. It copies onsite energies and shell
    membership in the same order. :func:`soc_matrix` supplies spin-flip terms
    through an on-site matrix.

    :see: :class:`~.test_soc.TestSpinDoubleModel`

    Parameters
    ----------
    model : TBModel
        Validated spinless model.

    Returns
    -------
    doubled : TBModel
        Revalidated spinor model with twice the Hamiltonian dimension.

    Raises
    ------
    ValueError
        If ``model`` already carries explicit spin.

    Notes
    -----
    The builder repeats each spinless hopping only within matching spin
    sectors. It preserves the exact integer-cell metadata in both copies.
    """
    if model.spinor or model.basis.spin:
        message: str = "spin_double_model requires a spinless model"
        raise ValueError(message)
    n_orbitals: int = len(model.basis.n)
    shifted_pairs: tuple[tuple[int, int], ...] = tuple(
        (orbital_i + n_orbitals, orbital_j + n_orbitals)
        for orbital_i, orbital_j in model.hopping_pairs
    )
    doubled_orbital_positions: Optional[Float64[Array, "n_so 3"]] = None
    if model.orbital_positions is not None:
        doubled_orbital_positions = jnp.concatenate(
            (model.orbital_positions, model.orbital_positions),
            axis=0,
        )
    doubled_depths: Optional[Float64[Array, " n_so"]] = None
    if model.depths is not None:
        doubled_depths = jnp.concatenate(
            (model.depths, model.depths),
            axis=0,
        )
    doubled: TBModel = make_tb_model(
        hopping_amplitudes=jnp.concatenate(
            (model.hopping_amplitudes, model.hopping_amplitudes)
        ),
        onsite_energies=jnp.concatenate(
            (model.onsite_energies, model.onsite_energies)
        ),
        soc_lambdas=model.soc_lambdas,
        geometry=model.geometry,
        basis=spin_double_basis(model.basis),
        hopping_pairs=model.hopping_pairs + shifted_pairs,
        hopping_cells=model.hopping_cells + model.hopping_cells,
        shell_index=model.shell_index + model.shell_index,
        spinor=True,
        orbital_positions=doubled_orbital_positions,
        depths=doubled_depths,
    )
    return doubled


@jaxtyped(typechecker=beartype)
def soc_matrix(  # noqa: DOC502 -- validation is shared.
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
    soc_lambdas: Float64[Array, " n_shells"],
) -> Complex128[Array, "n_so n_so"]:
    """Assemble shell-resolved atomic SOC in an arbitrary spinor basis.

    Each nonnegative ``shell_index`` identifies one ``(atom, n, l)`` group.
    A group must contain the same unique real-harmonic ``m`` channels in both
    spin sectors. Full atomic shells and projected subsets such as ``t2g`` are
    both supported. Static maps select the corresponding rows and columns
    from :func:`soc_shell_block` before one scatter-add per shell.

    :see: :class:`~.test_soc.TestSocMatrix`

    Parameters
    ----------
    basis : OrbitalBasis
        Orbital basis with explicit ``-1`` and ``+1`` spin metadata whenever
        an active shell is present.
    shell_index : tuple[int, ...]
        Orbital-to-shell IDs. ``-1`` excludes an orbital from atomic SOC.
    soc_lambdas : Float64[Array, "n_shells"]
        Differentiable shell coupling energies in eV.

    Returns
    -------
    matrix : Complex128[Array, "n_so n_so"]
        Hermitian on-site SOC matrix in the supplied global basis order.

    Raises
    ------
    ValueError
        If shell IDs, spin partners, or orbital shell metadata are
        inconsistent.
    EquinoxRuntimeError
        If any SOC coupling is non-finite.

    Notes
    -----
    The angular blocks are static constants. Gradients flow through their
    multiplication by ``soc_lambdas`` and through no table-construction
    parameter.
    """
    lambda_array: Float64[Array, " n_shells"] = jnp.asarray(
        soc_lambdas,
        dtype=jnp.float64,
    )
    _validate_soc_structure(basis, shell_index, lambda_array)
    lambda_array = eqx.error_if(
        lambda_array,
        ~jnp.all(jnp.isfinite(lambda_array)),
        "soc_matrix: soc lambdas finite",
    )
    n_orbitals: int = len(basis.n)
    matrix: Complex128[Array, "n_so n_so"] = jnp.zeros(
        (n_orbitals, n_orbitals),
        dtype=jnp.complex128,
    )
    shell: int
    for shell in range(lambda_array.shape[0]):
        global_tuple: tuple[int, ...]
        local_tuple: tuple[int, ...]
        angular_momentum: int
        global_tuple, local_tuple, angular_momentum = _shell_maps(
            basis,
            shell_index,
            shell,
        )
        global_indices: Int32[Array, " n_shell_orb"] = jnp.asarray(
            global_tuple,
            dtype=jnp.int32,
        )
        local_indices: Int32[Array, " n_shell_orb"] = jnp.asarray(
            local_tuple,
            dtype=jnp.int32,
        )
        full_block: Complex128[Array, "s1 s2"] = soc_shell_block(
            angular_momentum
        )
        projected_block: Complex128[Array, "p1 p2"] = full_block[
            local_indices[:, None],
            local_indices[None, :],
        ]
        matrix = matrix.at[
            global_indices[:, None],
            global_indices[None, :],
        ].add(lambda_array[shell] * projected_block)
    return matrix


__all__: list[str] = [
    "l_matrices",
    "soc_matrix",
    "soc_shell_block",
    "spin_double_basis",
    "spin_double_model",
]
