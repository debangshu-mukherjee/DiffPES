r"""Diagonalize native bands and adapt atom-resolved VASP projections.

Extended Summary
----------------
The module provides an eigenvalues-only fast path and an epsilon-regularized
Hermitian eigendecomposition whose eigenvector JVP remains finite at exact
degeneracy. Native diagonalization returns the geometry-bearing
``DiagonalizedBands`` interface. The VASP adapter retains its explicit
phase-loss policy and now preserves atom-resolved orbital data.

Routine Listings
----------------
:func:`eigh_safe`
    Diagonalize a Hermitian matrix with a regularized eigenvector JVP.
:func:`eigvalsh_bands`
    Compute only native tight-binding eigenvalues over k-points.
:func:`eigvalsh_bands_chunked`
    Compute eigenvalues with bounded live Hamiltonian storage.
:func:`diagonalize_tb`
    Diagonalize a native tight-binding model over k-points.
:func:`vasp_to_diagonalized`
    Convert atom-resolved VASP projections to approximate band vectors.

Notes
-----
Individual eigenvectors remain undefined inside a degenerate subspace. The
regularization prevents divergent autodiff. Scientific losses at a degeneracy
must consume fixed-group projectors, traces, or smooth spectral invariants.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Literal
from jaxtyping import Array, Complex128, Float64, Int32, jaxtyped

from diffpes.maths import safe_divide, safe_norm, safe_sqrt
from diffpes.types import (
    EPS,
    EPS_DEG,
    PHASE_LOSS_MESSAGE,
    BandStructure,
    CrystalGeometry,
    DiagonalizedBands,
    OrbitalBasis,
    OrbitalProjection,
    TBModel,
    make_diagonalized_bands,
)

from .hamiltonian import bloch_hamiltonian, bloch_hamiltonian_batch


def _checked_hermitian(
    hamiltonian: Complex128[Array, "n n"],
    *,
    context: str,
) -> Complex128[Array, "n n"]:
    """Reject non-finite or detectably non-Hermitian matrices under JAX."""
    checked: Complex128[Array, "n n"] = eqx.error_if(
        hamiltonian,
        ~jnp.all(jnp.isfinite(hamiltonian)),
        f"{context}: Hamiltonian entries must be finite",
    )
    checked = eqx.error_if(
        checked,
        ~jnp.allclose(
            checked,
            checked.conj().T,
            rtol=EPS,
            atol=EPS,
        ),
        f"{context}: Hamiltonian must be Hermitian",
    )
    return checked  # noqa: RET504 -- both checks must thread the returned value.


@jax.custom_jvp
@jaxtyped(typechecker=beartype)
def eigh_safe(  # noqa: DOC502 -- validation is delegated to a traced helper.
    hamiltonian: Complex128[Array, "n n"],
) -> tuple[Float64[Array, " n"], Complex128[Array, "n n"]]:
    r"""Diagonalize a Hermitian matrix with a regularized eigenvector JVP.

    The function preserves standard Hermitian eigenpairs while regularizing
    only inverse-gap factors in eigenvector tangents.

    :see: :class:`~.test_diagonalize.TestEighSafe`

    Parameters
    ----------
    hamiltonian : Complex128[Array, "n n"]
        Complex Hermitian matrix.

    Returns
    -------
    eigensystem : tuple[Float64[Array, " n"], Complex128[Array, "n n"]]
        Ascending eigenvalues and corresponding eigenvector columns.

    Raises
    ------
    EquinoxRuntimeError
        If the input contains a non-finite entry or is not Hermitian within
        the types-owned numerical tolerance.

    Notes
    -----
    The primal result equals :func:`jax.numpy.linalg.eigh`. Its custom JVP
    replaces the nondegenerate factor :math:`1/\Delta` by
    :math:`\Delta/(\Delta^2+\epsilon^2)`, with ``epsilon = EPS_DEG`` eV.
    Eigenvector tangents are therefore finite but biased by
    :math:`O(\epsilon^2/g^2)` for a gap ``g`` near the regularization scale.
    No correctness claim applies to individual vectors inside a degenerate
    group.
    """
    checked_hamiltonian: Complex128[Array, "n n"] = _checked_hermitian(
        hamiltonian,
        context="eigh_safe",
    )
    eigenvalues: Float64[Array, " n"]
    eigenvectors: Complex128[Array, "n n"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(checked_hamiltonian)
    eigensystem: tuple[Float64[Array, " n"], Complex128[Array, "n n"]] = (
        eigenvalues,
        eigenvectors,
    )
    return eigensystem


@eigh_safe.defjvp
def _eigh_safe_jvp(
    primals: tuple[Complex128[Array, "n n"]],
    tangents: tuple[Complex128[Array, "n n"]],
) -> tuple[
    tuple[Float64[Array, " n"], Complex128[Array, "n n"]],
    tuple[Float64[Array, " n"], Complex128[Array, "n n"]],
]:
    """Apply the Lorentzian-regularized Hermitian eigensystem JVP."""
    hamiltonian: Complex128[Array, "n n"]
    hamiltonian_tangent: Complex128[Array, "n n"]
    (hamiltonian,) = primals
    (hamiltonian_tangent,) = tangents
    checked_hamiltonian: Complex128[Array, "n n"] = _checked_hermitian(
        hamiltonian,
        context="eigh_safe",
    )
    eigenvalues: Float64[Array, " n"]
    eigenvectors: Complex128[Array, "n n"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(checked_hamiltonian)
    projected: Complex128[Array, "n n"] = (
        eigenvectors.conj().T @ hamiltonian_tangent @ eigenvectors
    )
    eigenvalue_tangent: Float64[Array, " n"] = jnp.real(
        jnp.diagonal(projected)
    )
    gaps: Float64[Array, "n n"] = eigenvalues[None, :] - eigenvalues[:, None]
    inverse_gaps: Float64[Array, "n n"] = gaps / (
        gaps * gaps + EPS_DEG * EPS_DEG
    )
    eigenvector_tangent: Complex128[Array, "n n"] = eigenvectors @ (
        inverse_gaps * projected
    )
    primal_output: tuple[Float64[Array, " n"], Complex128[Array, "n n"]] = (
        eigenvalues,
        eigenvectors,
    )
    tangent_output: tuple[Float64[Array, " n"], Complex128[Array, "n n"]] = (
        eigenvalue_tangent,
        eigenvector_tangent,
    )
    result: tuple[
        tuple[Float64[Array, " n"], Complex128[Array, "n n"]],
        tuple[Float64[Array, " n"], Complex128[Array, "n n"]],
    ] = (primal_output, tangent_output)
    return result  # noqa: RET504 -- assign-before-return is required.


@jaxtyped(typechecker=beartype)
def eigvalsh_bands(  # noqa: DOC502 -- validation is delegated to a traced helper.
    model: TBModel,
    kpoints: Float64[Array, "n_k 3"],
) -> Float64[Array, "n_k n_bands"]:
    """Compute only native tight-binding eigenvalues over k-points.

    This fast path avoids eigenvector construction and returns one ascending
    spectrum for every supplied k-point.

    :see: :class:`~.test_diagonalize.TestEigvalshBands`

    Parameters
    ----------
    model : TBModel
        Validated native tight-binding model.
    kpoints : Float64[Array, "n_k 3"]
        Fractional reciprocal-space k-points.

    Returns
    -------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Ascending band energies in eV.

    Raises
    ------
    EquinoxRuntimeError
        If an assembled Hamiltonian contains a non-finite entry or is not
        Hermitian within the types-owned numerical tolerance.

    Notes
    -----
    This path avoids eigenvectors and their inverse-gap derivative factors.
    Losses at exact crossings must still be symmetric within each degenerate
    eigenvalue group.
    """
    hamiltonians: Complex128[Array, "n_k n_orb n_orb"] = (
        bloch_hamiltonian_batch(model, kpoints)
    )

    def diagonalize_matrix(
        hamiltonian: Complex128[Array, "n_orb n_orb"],
    ) -> Float64[Array, " n_bands"]:
        checked_hamiltonian: Complex128[Array, "n_orb n_orb"] = (
            _checked_hermitian(
                hamiltonian,
                context="eigvalsh_bands",
            )
        )
        spectrum: Float64[Array, " n_bands"] = jnp.linalg.eigvalsh(
            checked_hamiltonian
        )
        return spectrum

    eigenvalues: Float64[Array, "n_k n_bands"] = jax.vmap(diagonalize_matrix)(
        hamiltonians
    )
    return eigenvalues


@jaxtyped(typechecker=beartype)
def eigvalsh_bands_chunked(  # noqa: DOC502, DOC503
    model: TBModel,
    kpoints: Float64[Array, "n_k 3"],
    chunk_size: int = 32,
) -> Float64[Array, "n_k n_bands"]:
    """Compute eigenvalues with bounded live Hamiltonian storage.

    A fixed-size scan partitions the k-point axis. Each scan body assembles
    and diagonalizes only ``chunk_size`` Hamiltonians. A
    :func:`jax.checkpoint` wrapper makes reverse mode recompute the body
    instead of retaining every chunk's dense intermediates.

    :see: :class:`~.test_diagonalize.TestEigvalshBandsChunked`

    Parameters
    ----------
    model : TBModel
        Validated native tight-binding model.
    kpoints : Float64[Array, "n_k 3"]
        Fractional reciprocal-space k-points. Callers sweeping path lengths
        should pad to one fixed ``n_k`` and mask the returned band loss.
    chunk_size : int, optional
        Positive static number of k-points per chunk. ``n_k`` must be exactly
        divisible by this value. Default is 32.

    Returns
    -------
    eigenvalues : Float64[Array, "n_k n_bands"]
        Ascending band energies in eV.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not a positive integer or does not divide the
        padded k-point count.
    EquinoxRuntimeError
        If an assembled Hamiltonian is non-finite or non-Hermitian.

    Notes
    -----
    Padding keeps the array shape fixed and therefore permits one compilation
    across a k-path-length sweep. Thickness, termination, orbital count, and
    connectivity remain static design choices and legitimately retrace. The
    implementation never materializes a global ``(n_k, n_orb, n_orb)``
    Hamiltonian array.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        message: str = "chunk_size must be a positive integer"
        raise ValueError(message)
    n_kpoints: int = kpoints.shape[0]
    if n_kpoints % chunk_size:
        message = "padded k-point count must be divisible by chunk_size"
        raise ValueError(message)
    chunks: Float64[Array, "n_chunk chunk_size 3"] = jnp.reshape(
        kpoints,
        (-1, chunk_size, 3),
    )

    def diagonalize_chunk(
        points: Float64[Array, "chunk_size 3"],
    ) -> Float64[Array, "chunk_size n_bands"]:
        def diagonalize_point(
            point: Float64[Array, " 3"],
        ) -> Float64[Array, " n_bands"]:
            hamiltonian: Complex128[Array, "n_orb n_orb"] = bloch_hamiltonian(
                model,
                point,
            )
            checked: Complex128[Array, "n_orb n_orb"] = _checked_hermitian(
                hamiltonian,
                context="eigvalsh_bands_chunked",
            )
            spectrum: Float64[Array, " n_bands"] = jnp.linalg.eigvalsh(checked)
            return spectrum

        values: Float64[Array, "chunk_size n_bands"] = jax.vmap(
            diagonalize_point
        )(points)
        return values

    checkpointed_chunk: Callable[[Array], Array] = jax.checkpoint(
        diagonalize_chunk
    )

    def scan_body(
        carry: None,
        points: Float64[Array, "chunk_size 3"],
    ) -> tuple[None, Float64[Array, "chunk_size n_bands"]]:
        output: tuple[None, Float64[Array, "chunk_size n_bands"]] = (
            carry,
            checkpointed_chunk(points),
        )
        return output

    _: None
    chunk_eigenvalues: Float64[Array, "n_chunk chunk_size n_bands"]
    _, chunk_eigenvalues = jax.lax.scan(scan_body, None, chunks)
    eigenvalues: Float64[Array, "n_k n_bands"] = jnp.reshape(
        chunk_eigenvalues,
        (n_kpoints, len(model.basis.n)),
    )
    return eigenvalues


@jaxtyped(typechecker=beartype)
def diagonalize_tb(
    model: TBModel,
    kpoints: Float64[Array, "n_k 3"],
) -> DiagonalizedBands:
    """Diagonalize a native tight-binding model over k-points.

    The function assembles and diagonalizes each Bloch Hamiltonian, converts
    eigenvectors to band-major order, and attaches model context.

    :see: :class:`~.test_diagonalize.TestDiagonalizeTB`

    Parameters
    ----------
    model : TBModel
        Validated native tight-binding model.
    kpoints : Float64[Array, "n_k 3"]
        Fractional reciprocal-space k-points.

    Returns
    -------
    bands : DiagonalizedBands
        Geometry-bearing eigensystem with band-major eigenvectors.

    Notes
    -----
    :func:`jax.vmap` applies :func:`eigh_safe` to each fractional k-point.
    The result retains the model geometry, static orbital basis, and any
    explicit orbital positions that define its basis-position gauge. Optional
    The result propagates orbital depths as an identity carrier.
    """

    def diagonalize_point(
        point: Float64[Array, " 3"],
    ) -> tuple[Float64[Array, " n_bands"], Complex128[Array, "n_orb n_bands"]]:
        hamiltonian: Complex128[Array, "n_orb n_orb"] = bloch_hamiltonian(
            model,
            point,
        )
        eigensystem: tuple[
            Float64[Array, " n_bands"], Complex128[Array, "n_orb n_bands"]
        ] = eigh_safe(hamiltonian)
        return eigensystem  # noqa: RET504 -- assign-before-return is required.

    eigenvalues: Float64[Array, "n_k n_bands"]
    eigenvector_columns: Complex128[Array, "n_k n_orb n_bands"]
    eigenvalues, eigenvector_columns = jax.vmap(diagonalize_point)(kpoints)
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"] = jnp.swapaxes(
        eigenvector_columns,
        -1,
        -2,
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=kpoints,
        geometry=model.geometry,
        basis=model.basis,
        fermi_energy=0.0,
        orbital_positions=model.orbital_positions,
        depths=model.depths,
    )
    return bands


@jaxtyped(typechecker=beartype)
def vasp_to_diagonalized(  # noqa: DOC503 -- traced checks raise indirectly.
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    geometry: CrystalGeometry,
    orbital_basis: OrbitalBasis,
    phase_loss: Literal["warn", "ignore", "error"] = "warn",
) -> DiagonalizedBands:
    """Convert atom-resolved VASP projections to approximate band vectors.

    VASP PROCAR provides orbital weights rather than complex coefficients.
    This adapter selects each basis orbital at its registered atom, takes the
    positive square root, and normalizes the resulting vectors. Relative
    phases are irrecoverably absent, so phase-sensitive observables remain an
    explicitly approximate path.

    :see: :class:`~.test_diagonalize.TestVaspToDiagonalized`

    Parameters
    ----------
    bands : BandStructure
        VASP eigenvalues, k-points, and Fermi energy.
    orb_proj : OrbitalProjection
        Atom- and orbital-resolved PROCAR weights with final axis in VASP's
        nine-orbital order.
    geometry : CrystalGeometry
        Geometry associated with the VASP calculation.
    orbital_basis : OrbitalBasis
        Atom mapping and real-harmonic channels to retain (**static** --
        changing them triggers retracing).
    phase_loss : Literal["warn", "ignore", "error"], optional
        Policy for missing complex phases. Default is ``"warn"``.

    Returns
    -------
    diagonalized : DiagonalizedBands
        Approximate normalized coefficients with geometry and basis context.

    Raises
    ------
    ValueError
        If the phase policy has an invalid value or requests an error.
        Also raised when the geometry and projection atom counts disagree,
        for a spin-resolved basis, an invalid atom index, or an unsupported
        basis channel.
    EquinoxRuntimeError
        If a selected projection weight is non-finite or negative, or if any
        selected band vector has zero norm.

    Notes
    -----
    The adapter gathers one atom and VASP channel for each basis orbital.
    Every selected band vector must contain nonzero projection weight because
    a zero vector cannot define a normalized approximate eigenstate. Spinful
    bases remain unsupported until the adapter can resolve spin channels.
    The adapter checks projection weights for finiteness and nonnegativity
    before calling :func:`diffpes.maths.safe_sqrt`. It checks the resulting
    norm before calling :func:`diffpes.maths.safe_divide`.
    Those helpers provide registered boundary conventions; they do not
    perform the adapter's domain validation.
    """
    if phase_loss not in ("warn", "ignore", "error"):
        message: str = (
            "phase_loss must be one of {'warn', 'ignore', 'error'}, "
            f"got '{phase_loss}'"
        )
        raise ValueError(message)
    if phase_loss == "error":
        raise ValueError(PHASE_LOSS_MESSAGE)
    if phase_loss == "warn":
        warnings.warn(PHASE_LOSS_MESSAGE, RuntimeWarning, stacklevel=2)

    n_projection_atoms: int = orb_proj.projections.shape[2]
    n_geometry_atoms: int = geometry.positions.shape[0]
    if n_geometry_atoms != n_projection_atoms:
        message = (
            "vasp_to_diagonalized: geometry and projection atom counts "
            "must agree"
        )
        raise ValueError(message)
    if orbital_basis.spin:
        message = (
            "vasp_to_diagonalized: spin-resolved orbital bases are not "
            "supported"
        )
        raise ValueError(message)
    if any(
        index >= n_projection_atoms for index in orbital_basis.atom_indices
    ):
        message = "orbital basis atom_indices exceed the projection atom axis"
        raise ValueError(message)

    vasp_lm_to_index: dict[tuple[int, int], int] = {
        (0, 0): 0,
        (1, -1): 1,
        (1, 0): 2,
        (1, 1): 3,
        (2, -2): 4,
        (2, -1): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8,
    }
    orbital_channels: list[int] = []
    angular: int
    magnetic: int
    for angular, magnetic in zip(
        orbital_basis.l,
        orbital_basis.m,
        strict=True,
    ):
        channel: int | None = vasp_lm_to_index.get((angular, magnetic))
        if channel is None:
            message = (
                f"Orbital (l={angular}, m={magnetic}) is not in VASP "
                "9-orbital set"
            )
            raise ValueError(message)
        orbital_channels.append(channel)

    atom_index_array: Int32[Array, " n_orb"] = jnp.asarray(
        orbital_basis.atom_indices,
        dtype=jnp.int32,
    )
    channel_array: Int32[Array, " n_orb"] = jnp.asarray(
        orbital_channels,
        dtype=jnp.int32,
    )
    raw_selected_weights: Float64[Array, "n_k n_bands n_orb"] = (
        orb_proj.projections[:, :, atom_index_array, channel_array]
    )
    selected_weights: Float64[Array, "n_k n_bands n_orb"] = eqx.error_if(
        raw_selected_weights,
        ~jnp.all(jnp.isfinite(raw_selected_weights)),
        "vasp_to_diagonalized: selected projection weights must be finite",
    )
    selected_weights = eqx.error_if(
        selected_weights,
        ~jnp.all(selected_weights >= 0.0),
        "vasp_to_diagonalized: selected weights must be nonnegative",
    )
    coefficients: Float64[Array, "n_k n_bands n_orb"] = safe_sqrt(
        selected_weights
    )
    normalization: Float64[Array, "n_k n_bands 1"] = safe_norm(
        coefficients,
        axis=-1,
        keepdims=True,
    )
    normalization = eqx.error_if(
        normalization,
        jnp.any(normalization == 0.0),
        "vasp_to_diagonalized: selected projection norm must be nonzero",
    )
    normalized: Float64[Array, "n_k n_bands n_orb"] = safe_divide(
        coefficients,
        normalization,
    )
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"] = normalized.astype(
        jnp.complex128
    )
    diagonalized: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=bands.eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=bands.kpoints,
        geometry=geometry,
        basis=orbital_basis,
        fermi_energy=bands.fermi_energy,
    )
    return diagonalized


__all__: list[str] = [
    "diagonalize_tb",
    "eigh_safe",
    "eigvalsh_bands",
    "eigvalsh_bands_chunked",
    "vasp_to_diagonalized",
]
