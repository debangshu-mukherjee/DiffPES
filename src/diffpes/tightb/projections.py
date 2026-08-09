"""Reduce tight-binding eigenvectors to gauge-invariant observables.

Extended Summary
----------------
This module exposes squared orbital weights, band and fixed-group projectors,
operator traces, degeneracy-averaged expectation values, and fat-band
weights. Scientific calculations at an exact degeneracy use a registered
fixed group rather than selecting a group from a differentiable threshold.

Routine Listings
----------------
:func:`orbital_weights`
    Compute the squared orbital amplitudes of normalized eigenvectors.
:func:`band_projectors`
    Materialize each U(1)-gauge-invariant rank-one band projector.
:func:`group_projector`
    Construct the projector onto one registered, fixed band group.
:func:`group_trace`
    Trace a Hermitian operator over one fixed band group.
:func:`expectation_path`
    Compute operator expectations with diagnostic degeneracy averaging.
:func:`fat_bands`
    Compute degeneracy-averaged weights of selected model orbitals.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex128, Float64, jaxtyped

from diffpes.types import EPS, MATRIX_NDIM, DiagonalizedBands

from .operators import orbital_projector


def _validate_selection(
    selection: tuple[int, ...],
    upper_bound: int,
    *,
    name: str,
) -> None:
    """PRIVATE: Validate a static tuple of unique array indices.

    Parameters
    ----------
    selection : tuple[int, ...]
        Candidate static index tuple.
    upper_bound : int
        Exclusive upper bound for every index.
    name : str
        Argument name used in error messages.

    Raises
    ------
    ValueError
        If the selection is not a tuple, is empty, holds a non-integer,
        indexes outside ``[0, upper_bound)``, or repeats an index.

    Notes
    -----
    All checks run on static Python values before tracing, so the
    validated selection participates in the JAX trace signature.
    """
    if type(selection) is not tuple:
        message: str = f"{name} must be a tuple"
        raise ValueError(message)
    if not selection:
        message = f"{name} must contain at least one index"
        raise ValueError(message)
    if any(type(index) is not int for index in selection):
        message = f"{name} must contain integers"
        raise ValueError(message)
    if any(index < 0 or index >= upper_bound for index in selection):
        message = f"{name} indices must lie in [0, {upper_bound})"
        raise ValueError(message)
    if len(set(selection)) != len(selection):
        message = f"{name} must not contain duplicate indices"
        raise ValueError(message)


def _checked_operator(
    operator: Complex128[Array, "n_orb n_orb"],
    n_orbitals: int,
    *,
    context: str,
) -> Complex128[Array, "n_orb n_orb"]:
    """PRIVATE: Validate the static shape and traced Hermitian contract.

    Parameters
    ----------
    operator : Complex128[Array, "n_orb n_orb"]
        Candidate operator in the model orbital basis.
    n_orbitals : int
        Required static operator dimension.
    context : str
        Caller name used in error messages.

    Returns
    -------
    checked : Complex128[Array, "n_orb n_orb"]
        The unchanged operator with both runtime checks threaded through.

    Raises
    ------
    ValueError
        If the operator is not a square matrix of the required size.

    Notes
    -----
    The shape check is static. Chained :func:`equinox.error_if` guards
    then raise ``EquinoxRuntimeError`` for a non-finite entry or a
    deviation from the conjugate transpose beyond the absolute ``EPS``
    tolerance.
    """
    if operator.ndim != MATRIX_NDIM or operator.shape != (
        n_orbitals,
        n_orbitals,
    ):
        message: str = (
            f"{context}: operator must have shape ({n_orbitals}, {n_orbitals})"
        )
        raise ValueError(message)
    checked: Complex128[Array, "n_orb n_orb"] = eqx.error_if(
        operator,
        ~jnp.all(jnp.isfinite(operator)),
        f"{context}: operator entries must be finite",
    )
    checked = eqx.error_if(
        checked,
        ~jnp.allclose(checked, checked.conj().T, rtol=0.0, atol=EPS),
        f"{context}: operator must be Hermitian",
    )
    return checked  # noqa: RET504 -- both runtime checks must be threaded.


@jaxtyped(typechecker=beartype)
def orbital_weights(
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"],
) -> Float64[Array, "n_k n_bands n_orb"]:
    r"""Compute the squared orbital amplitudes of normalized eigenvectors.

    Multiplication by an arbitrary per-band U(1) phase leaves every returned
    value unchanged.

    :see: :class:`~.test_projections.TestOrbitalWeights`

    Parameters
    ----------
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Band-major complex orbital coefficients.

    Returns
    -------
    weights : Float64[Array, "n_k n_bands n_orb"]
        Values :math:`|c_{kno}|^2`.

    Notes
    -----
    Individual band weights are not invariant under rotations inside an
    exactly degenerate subspace. Use :func:`group_trace` for a scientific
    observable at such a degeneracy.
    """
    weights: Float64[Array, "n_k n_bands n_orb"] = jnp.abs(eigenvectors) ** 2
    return weights


@jaxtyped(typechecker=beartype)
def band_projectors(
    eigenvectors: Complex128[Array, "n_k n_bands n_orb"],
) -> Complex128[Array, "n_k n_bands n_orb n_orb"]:
    r"""Materialize each U(1)-gauge-invariant rank-one band projector.

    Form an outer product from every band-major eigenvector without choosing
    an orbital phase convention.

    :see: :class:`~.test_projections.TestBandProjectors`

    Parameters
    ----------
    eigenvectors : Complex128[Array, "n_k n_bands n_orb"]
        Band-major complex orbital coefficients.

    Returns
    -------
    projectors : Complex128[Array, "n_k n_bands n_orb n_orb"]
        Outer products :math:`|\psi_{kn}\rangle\langle\psi_{kn}|`.

    Notes
    -----
    This result occupies ``n_k * n_bands * n_orb**2`` complex elements.
    Hot paths that need only an observable should use :func:`group_trace` or
    :func:`expectation_path` and avoid this materialization.
    """
    projectors: Complex128[Array, "n_k n_bands n_orb n_orb"] = jnp.einsum(
        "kbi,kbj->kbij",
        eigenvectors,
        eigenvectors.conj(),
    )
    return projectors


@jaxtyped(typechecker=beartype)
def group_projector(  # noqa: DOC502 -- validation is delegated.
    bands: DiagonalizedBands,
    group: tuple[int, ...],
) -> Complex128[Array, "n_k n_orb n_orb"]:
    r"""Construct the projector onto one registered, fixed band group.

    The sum is invariant under arbitrary unitary rotations among the selected
    eigenvectors, including rotations inside a Kramers pair.

    :see: :class:`~.test_projections.TestGroupProjector`

    Parameters
    ----------
    bands : DiagonalizedBands
        Geometry-bearing, band-major eigensystem.
    group : tuple[int, ...]
        Fixed unique band indices (**static** -- changing them retraces).

    Returns
    -------
    projector : Complex128[Array, "n_k n_orb n_orb"]
        Fixed-group projector at every k-point.

    Raises
    ------
    ValueError
        If ``group`` is empty, duplicated, or outside the band axis.

    Notes
    -----
    Static validation fixes group membership before the traced outer-product
    reduction. The sum removes every internal basis choice.
    """
    _validate_selection(
        group,
        bands.eigenvalues.shape[1],
        name="group",
    )
    selected: Complex128[Array, "n_k n_group n_orb"] = bands.eigenvectors[
        :, group, :
    ]
    projector: Complex128[Array, "n_k n_orb n_orb"] = jnp.einsum(
        "kgi,kgj->kij",
        selected,
        selected.conj(),
    )
    return projector


@jaxtyped(typechecker=beartype)
def group_trace(  # noqa: DOC502 -- validation is delegated.
    bands: DiagonalizedBands,
    operator: Complex128[Array, "n_orb n_orb"],
    group: tuple[int, ...],
) -> Float64[Array, " n_k"]:
    r"""Trace a Hermitian operator over one fixed band group.

    This is the normative exact-degeneracy observable. It equals
    :math:`\operatorname{Tr}(P_G O)` and is invariant under every unitary
    basis change within the registered group.

    :see: :class:`~.test_projections.TestGroupTrace`

    Parameters
    ----------
    bands : DiagonalizedBands
        Geometry-bearing, band-major eigensystem.
    operator : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian operator in the model basis.
    group : tuple[int, ...]
        Fixed unique band indices (**static** -- changing them retraces).

    Returns
    -------
    traces : Float64[Array, "n_k"]
        Real fixed-group operator trace at each k-point.

    Raises
    ------
    ValueError
        If static shapes or group indices are invalid.
    EquinoxRuntimeError
        If ``operator`` is non-finite or non-Hermitian.

    Notes
    -----
    The implementation contracts the fixed-group projector with the
    Hermitian operator and returns the real component.
    """
    n_orbitals: int = bands.eigenvectors.shape[2]
    checked_operator: Complex128[Array, "n_orb n_orb"] = _checked_operator(
        operator,
        n_orbitals,
        context="group_trace",
    )
    projector: Complex128[Array, "n_k n_orb n_orb"] = group_projector(
        bands,
        group,
    )
    traces_complex: Complex128[Array, " n_k"] = jnp.einsum(
        "kij,ji->k",
        projector,
        checked_operator,
    )
    traces: Float64[Array, " n_k"] = jnp.real(traces_complex)
    return traces


@jaxtyped(typechecker=beartype)
def expectation_path(  # noqa: DOC502 -- validation is delegated.
    bands: DiagonalizedBands,
    operator: Complex128[Array, "n_orb n_orb"],
    degen_tol: float = 1e-10,
) -> Float64[Array, "n_k n_bands"]:
    r"""Compute operator expectations with diagnostic degeneracy averaging.

    Average raw quadratic forms over connected components of bands whose
    pairwise energy separation falls below ``degen_tol``. The fixed-shape
    component mask makes the result invariant under a complete unitary
    rotation of an exactly degenerate block.

    :see: :class:`~.test_projections.TestExpectationPath`

    Parameters
    ----------
    bands : DiagonalizedBands
        Geometry-bearing, band-major eigensystem.
    operator : Complex128[Array, "n_orb n_orb"]
        Finite Hermitian operator in the model basis.
    degen_tol : float, optional
        Positive diagnostic energy threshold in eV. Default is ``1e-10``.

    Returns
    -------
    expectations : Float64[Array, "n_k n_bands"]
        Degeneracy-averaged values :math:`\langle\psi|O|\psi\rangle`.

    Raises
    ------
    ValueError
        If the operator shape is invalid.
    EquinoxRuntimeError
        If the operator is invalid or ``degen_tol`` is not finite and
        positive.

    Notes
    -----
    The threshold graph and its connected components are nondifferentiable
    and exist for diagnostics only. Transitive closure prevents overlapping
    pairwise neighborhoods from assigning inconsistent averages to a chain
    of nearby bands. Exact-crossing gradient claims must use a fixed
    :func:`group_trace`.
    """
    n_orbitals: int = bands.eigenvectors.shape[2]
    checked_operator: Complex128[Array, "n_orb n_orb"] = _checked_operator(
        operator,
        n_orbitals,
        context="expectation_path",
    )
    tolerance: Float64[Array, ""] = jnp.asarray(degen_tol, dtype=jnp.float64)
    tolerance = eqx.error_if(
        tolerance,
        ~jnp.isfinite(tolerance) | (tolerance <= 0.0),
        "expectation_path: degen_tol must be finite and positive",
    )
    raw_complex: Complex128[Array, "n_k n_bands"] = jnp.einsum(
        "kbi,ij,kbj->kb",
        bands.eigenvectors.conj(),
        checked_operator,
        bands.eigenvectors,
    )
    raw: Float64[Array, "n_k n_bands"] = jnp.real(raw_complex)
    gaps: Float64[Array, "n_k n_bands n_bands"] = jnp.abs(
        bands.eigenvalues[:, :, None] - bands.eigenvalues[:, None, :]
    )
    connected: Array = gaps < tolerance
    pivot: int
    for pivot in range(bands.eigenvalues.shape[1]):
        connected = connected | (
            connected[:, :, pivot, None] & connected[:, None, pivot, :]
        )
    mask: Float64[Array, "n_k n_bands n_bands"] = connected.astype(jnp.float64)
    numerator: Float64[Array, "n_k n_bands"] = jnp.einsum(
        "kij,kj->ki",
        mask,
        raw,
    )
    denominator: Float64[Array, "n_k n_bands"] = jnp.sum(mask, axis=-1)
    expectations: Float64[Array, "n_k n_bands"] = numerator / denominator
    return expectations


@jaxtyped(typechecker=beartype)
def fat_bands(  # noqa: DOC502 -- validation is delegated.
    bands: DiagonalizedBands,
    orbital_select: tuple[int, ...],
) -> Float64[Array, "n_k n_bands"]:
    """Compute degeneracy-averaged weights of selected model orbitals.

    Build one orbital-selection projector and evaluate it along the complete
    diagonalized path.

    :see: :class:`~.test_projections.TestFatBands`

    Parameters
    ----------
    bands : DiagonalizedBands
        Geometry-bearing, band-major eigensystem.
    orbital_select : tuple[int, ...]
        Fixed unique orbital indices (**static** -- changing them retraces).

    Returns
    -------
    weights : Float64[Array, "n_k n_bands"]
        Selected-orbital weights, averaged within diagnostic degeneracies.

    Raises
    ------
    ValueError
        If the selection is empty, duplicated, or outside the orbital axis.

    Notes
    -----
    This convenience function uses the default threshold of
    :func:`expectation_path`. Use :func:`group_trace` for scientific
    derivatives at an exact degeneracy.
    """
    operator: Complex128[Array, "n_orb n_orb"] = orbital_projector(
        bands.basis,
        orbital_select,
    )
    weights: Float64[Array, "n_k n_bands"] = expectation_path(bands, operator)
    return weights


__all__: list[str] = [
    "band_projectors",
    "expectation_path",
    "fat_bands",
    "group_projector",
    "group_trace",
    "orbital_weights",
]
