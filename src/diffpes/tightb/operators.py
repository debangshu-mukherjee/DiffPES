r"""Construct Hermitian observables in a tight-binding orbital basis.

Extended Summary
----------------
The builders in this module construct spin, atomic :math:`L\cdot S`, and
orbital-selection operators using the package-wide real-harmonic and spinor
conventions.

Routine Listings
----------------
:func:`spin_operator`
    Construct :math:`S_{\widehat n}=\widehat n\cdot\sigma/2`.
:func:`ls_operator`
    Construct unit-strength atomic :math:`L\cdot S` by shell.
:func:`orbital_projector`
    Construct a diagonal projector onto selected basis orbitals.
:func:`surface_projector`
    Construct surface-sensitive orbital probability weights.
:func:`layer_resolved_weights`
    Compute per-band surface weights as an off-degeneracy diagnostic.
:func:`layer_resolved_group_traces`
    Compute surface traces over complete, isolated fixed band groups.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Complex128, Float, Float64, jaxtyped

from diffpes.types import (
    DEGENERACY_GROUP_TOL_EV,
    EPS,
    GROUP_COMPLEMENT_GAP_MIN_EV,
    DiagonalizedBands,
    OrbitalBasis,
    ScalarFloat,
)

from .soc import soc_matrix


def _validate_spin_pairs(basis: OrbitalBasis) -> None:
    """Require one down and one up state for every spatial orbital."""
    n_orbitals: int = len(basis.n)
    if len(basis.spin) != n_orbitals:
        message: str = "spin_operator requires an explicit spinor basis"
        raise ValueError(message)
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    orbital: int
    for orbital in range(n_orbitals):
        key: tuple[int, int, int, int] = (
            basis.atom_indices[orbital],
            basis.n[orbital],
            basis.l[orbital],
            basis.m[orbital],
        )
        groups.setdefault(key, []).append(basis.spin[orbital])
    if any(sorted(spins) != [-1, 1] for spins in groups.values()):
        message = (
            "spin_operator requires exactly one -1 and one +1 state for "
            "every spatial orbital"
        )
        raise ValueError(message)


def _validate_orbital_selection(
    basis: OrbitalBasis,
    orbital_select: tuple[int, ...],
) -> None:
    """Validate one static orbital selection."""
    if type(orbital_select) is not tuple:
        message: str = "orbital_select must be a tuple"
        raise ValueError(message)
    if not orbital_select:
        message = "orbital_select must contain at least one index"
        raise ValueError(message)
    if any(type(index) is not int for index in orbital_select):
        message = "orbital_select must contain integers"
        raise ValueError(message)
    n_orbitals: int = len(basis.n)
    if any(index < 0 or index >= n_orbitals for index in orbital_select):
        message = f"orbital_select indices must lie in [0, {n_orbitals})"
        raise ValueError(message)
    if len(set(orbital_select)) != len(orbital_select):
        message = "orbital_select must not contain duplicate indices"
        raise ValueError(message)


def _validate_fixed_groups(
    fixed_groups: tuple[tuple[int, ...], ...],
    n_bands: int,
) -> None:
    """Validate static, disjoint registered band groups."""
    if type(fixed_groups) is not tuple:
        message: str = "fixed_groups must be a tuple"
        raise ValueError(message)
    if not fixed_groups:
        message = "fixed_groups must contain at least one group"
        raise ValueError(message)
    selected: set[int] = set()
    group: tuple[int, ...]
    for group in fixed_groups:
        if type(group) is not tuple:
            message = "fixed_groups must contain tuples"
            raise ValueError(message)
        if not group:
            message = "fixed_groups must not contain empty groups"
            raise ValueError(message)
        if any(type(index) is not int for index in group):
            message = "fixed_groups must contain integer band indices"
            raise ValueError(message)
        if any(index < 0 or index >= n_bands for index in group):
            message = f"fixed_groups indices must lie in [0, {n_bands})"
            raise ValueError(message)
        if len(set(group)) != len(group):
            message = "fixed_groups must not contain duplicate indices"
            raise ValueError(message)
        overlap: set[int] = selected.intersection(group)
        if overlap:
            message = "fixed_groups must be disjoint"
            raise ValueError(message)
        selected.update(group)


def _surface_weighted_expectations(
    eigenvectors: Complex[Array, "n_k n_bands n_orb"],
    projector_diagonal: Float[Array, " n_orb"],
) -> Float[Array, "n_k n_bands"]:
    """Compute band probabilities for a diagonal surface operator."""
    orbital_probabilities: Float[Array, "n_k n_bands n_orb"] = (
        jnp.abs(eigenvectors) ** 2
    )
    expectations: Float[Array, "n_k n_bands"] = jnp.einsum(
        "kbo,o->kb",
        orbital_probabilities,
        projector_diagonal,
    )
    return expectations


@jaxtyped(typechecker=beartype)
def spin_operator(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    axis: Float[Array, " 3"],
) -> Complex[Array, "n_so n_so"]:
    r"""Construct :math:`S_{\widehat n}=\widehat n\cdot\sigma/2`.

    The operator follows the declared ``(down, up)`` convention, including
    :math:`\sigma_y=[[0,i],[-i,0]]` and
    :math:`\sigma_z=\operatorname{diag}(-1,+1)`. Spatial partners are matched
    from basis metadata, so spin blocks need not be contiguous.

    :see: :class:`~.test_operators.TestSpinOperator`

    Parameters
    ----------
    basis : OrbitalBasis
        Spinor basis containing one ``-1`` and one ``+1`` state for every
        spatial orbital.
    axis : Float[Array, "3"]
        Finite Cartesian unit vector in the sample frame.

    Returns
    -------
    operator : Complex[Array, "n_so n_so"]
        Hermitian spin-one-half operator in the supplied basis order.

    Raises
    ------
    ValueError
        If the basis does not contain complete, unique spin pairs.
    EquinoxRuntimeError
        If ``axis`` is non-finite or not a unit vector.

    Notes
    -----
    Static orbital metadata identifies the partner pairs. Array operations
    then place the three Pauli components in the supplied basis order.
    """
    _validate_spin_pairs(basis)
    checked_axis: Float[Array, " 3"] = eqx.error_if(
        axis,
        ~jnp.all(jnp.isfinite(axis)),
        "spin_operator: axis must be finite",
    )
    norm_squared: Float[Array, ""] = jnp.sum(checked_axis * checked_axis)
    checked_axis = eqx.error_if(
        checked_axis,
        ~jnp.isclose(norm_squared, 1.0, rtol=EPS, atol=EPS),
        "spin_operator: axis must be a unit vector",
    )

    keys: tuple[tuple[int, int, int, int], ...] = tuple(
        (
            basis.atom_indices[index],
            basis.n[index],
            basis.l[index],
            basis.m[index],
        )
        for index in range(len(basis.n))
    )
    same_spatial: Float64[Array, "n_so n_so"] = jnp.asarray(
        [
            [float(row_key == column_key) for column_key in keys]
            for row_key in keys
        ],
        dtype=jnp.float64,
    )
    spin: Float64[Array, " n_so"] = jnp.asarray(
        basis.spin,
        dtype=jnp.float64,
    )
    row_spin: Float64[Array, "n_so 1"] = spin[:, None]
    column_spin: Float64[Array, "1 n_so"] = spin[None, :]
    sigma_x: Complex128[Array, "n_so n_so"] = (
        same_spatial * (row_spin != column_spin)
    ).astype(jnp.complex128)
    sigma_y: Complex128[Array, "n_so n_so"] = same_spatial * jnp.where(
        (row_spin == -1.0) & (column_spin == 1.0),
        1.0j,
        jnp.where(
            (row_spin == 1.0) & (column_spin == -1.0),
            -1.0j,
            0.0j,
        ),
    )
    sigma_z: Complex128[Array, "n_so n_so"] = (
        same_spatial * (row_spin == column_spin) * row_spin
    ).astype(jnp.complex128)
    operator: Complex128[Array, "n_so n_so"] = 0.5 * (
        checked_axis[0] * sigma_x
        + checked_axis[1] * sigma_y
        + checked_axis[2] * sigma_z
    )
    return operator


@jaxtyped(typechecker=beartype)
def ls_operator(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    shell_index: tuple[int, ...],
) -> Complex[Array, "n_so n_so"]:
    r"""Construct unit-strength atomic :math:`L\cdot S` by shell.

    Use the SOC shell map while replacing every physical coupling with one.
    The resulting operator exposes atomic angular correlations directly.

    :see: :class:`~.test_operators.TestLsOperator`

    Parameters
    ----------
    basis : OrbitalBasis
        Real-harmonic spinor basis.
    shell_index : tuple[int, ...]
        Static orbital-to-shell IDs. ``-1`` excludes an orbital; nonnegative
        IDs must follow the :func:`diffpes.tightb.soc.soc_matrix` contract.

    Returns
    -------
    operator : Complex[Array, "n_so n_so"]
        Hermitian sum of unit-strength :math:`L\cdot S` shell blocks.

    Raises
    ------
    ValueError
        If shell metadata or spin partners violate the SOC contract.

    Notes
    -----
    The result is dimensionless in units with :math:`\hbar=1`. Multiplication
    by shell-specific couplings is part of Hamiltonian construction, not this
    diagnostic operator.
    """
    n_shells: int = max(shell_index, default=-1) + 1
    strengths: Float64[Array, " n_shells"] = jnp.ones(
        (n_shells,),
        dtype=jnp.float64,
    )
    operator: Complex[Array, "n_so n_so"] = soc_matrix(
        basis,
        shell_index,
        strengths,
    )
    return operator


@jaxtyped(typechecker=beartype)
def orbital_projector(  # noqa: DOC502 -- validation is delegated.
    basis: OrbitalBasis,
    orbital_select: tuple[int, ...],
) -> Complex[Array, "n n"]:
    """Construct a diagonal projector onto selected basis orbitals.

    Mark each requested static index with one and leave all other diagonal
    entries at zero.

    :see: :class:`~.test_operators.TestOrbitalProjector`

    Parameters
    ----------
    basis : OrbitalBasis
        Model basis defining the operator dimension.
    orbital_select : tuple[int, ...]
        Fixed unique orbital indices (**static** -- changing them retraces).

    Returns
    -------
    projector : Complex[Array, "n n"]
        Hermitian idempotent diagonal selection matrix.

    Raises
    ------
    ValueError
        If the selection is empty, duplicated, or outside the basis.

    Notes
    -----
    The static tuple controls selection and therefore participates in JAX
    tracing. The returned complex matrix remains a numerical array leaf.
    """
    _validate_orbital_selection(basis, orbital_select)
    diagonal: Float64[Array, " n"] = jnp.zeros(
        (len(basis.n),),
        dtype=jnp.float64,
    )
    diagonal = diagonal.at[jnp.asarray(orbital_select)].set(1.0)
    projector: Complex128[Array, "n n"] = jnp.diag(diagonal).astype(
        jnp.complex128
    )
    return projector


@jaxtyped(typechecker=beartype)
def surface_projector(  # noqa: DOC503 -- runtime checks use eqx.error_if.
    depths: Float[Array, " n_orb"],
    intensity_escape_length_ang: ScalarFloat,
) -> Float[Array, " n_orb"]:
    r"""Construct surface-sensitive orbital probability weights.

    Each orbital at depth :math:`d_o` in Angstrom receives the probability
    weight :math:`\exp(-d_o/\lambda_I)`, where :math:`\lambda_I` is explicitly
    an intensity escape length. Coherent amplitude attenuation, which uses
    the factor :math:`1/(2\lambda_I)`, is a separate matrix-element operation.

    :see: :class:`~.test_operators.TestSurfaceProjector`

    Parameters
    ----------
    depths : Float[Array, "n_orb"]
        Nonnegative orbital depths below the top surface in Angstrom.
    intensity_escape_length_ang : ScalarFloat
        Finite, positive intensity escape length in Angstrom.

    Returns
    -------
    weights : Float[Array, "n_orb"]
        Diagonal entries of the surface probability operator.

    Raises
    ------
    ValueError
        If ``depths`` is not one-dimensional.
    EquinoxRuntimeError
        If a depth is non-finite or negative, or the escape length is not
        finite and positive.

    Notes
    -----
    For escape lengths at or below ``EPS``, the implementation evaluates the
    exact positive-side limit: one at zero depth and zero below the surface.
    A sanitized division branch yields finite values and zero limiting
    subgradients instead of ``0 * inf``.
    """
    if depths.ndim != 1:
        message: str = "surface_projector: depths must be one-dimensional"
        raise ValueError(message)
    checked_depths: Float[Array, " n_orb"] = eqx.error_if(
        depths,
        ~jnp.all(jnp.isfinite(depths)) | jnp.any(depths < -1e-12),  # noqa: PLR2004
        "surface_projector: depths must be finite and nonnegative",
    )
    checked_depths = jnp.maximum(checked_depths, 0.0)
    escape_length: Float64[Array, ""] = jnp.asarray(
        intensity_escape_length_ang,
        dtype=jnp.float64,
    )
    escape_length = eqx.error_if(
        escape_length,
        ~jnp.isfinite(escape_length) | (escape_length <= 0.0),
        "surface_projector: intensity escape length must be finite and "
        "positive",
    )
    ordinary_length: Array = escape_length > EPS
    sanitized_length: Float[Array, ""] = jnp.where(
        ordinary_length,
        escape_length,
        1.0,
    )
    ordinary_weights: Float[Array, " n_orb"] = jnp.exp(
        -checked_depths / sanitized_length
    )
    limiting_weights: Float[Array, " n_orb"] = jnp.where(
        checked_depths == 0.0,
        1.0,
        0.0,
    )
    weights: Float[Array, " n_orb"] = jnp.where(
        ordinary_length,
        ordinary_weights,
        limiting_weights,
    )
    return weights


@jaxtyped(typechecker=beartype)
def layer_resolved_weights(
    bands: DiagonalizedBands,
    intensity_escape_length_ang: ScalarFloat,
) -> Float[Array, "n_k n_bands"]:
    """Compute per-band surface weights as an off-degeneracy diagnostic.

    Contract each band's orbital probabilities with the diagonal surface
    operator. These individual values depend on the eigenvector basis inside
    a degenerate subspace and therefore carry no degeneracy-invariance claim.

    :see: :class:`~.test_operators.TestLayerResolvedWeights`

    Parameters
    ----------
    bands : DiagonalizedBands
        Band eigensystem carrying non-``None`` orbital ``depths``.
    intensity_escape_length_ang : ScalarFloat
        Finite, positive intensity escape length in Angstrom.

    Returns
    -------
    weights : Float[Array, "n_k n_bands"]
        Raw surface-operator expectation for every eigenvector.

    Raises
    ------
    ValueError
        If ``bands.depths`` is ``None``.

    Notes
    -----
    This diagonal fast path avoids materializing an ``(n_orb, n_orb)``
    operator. Use :func:`layer_resolved_group_traces` at a degeneracy.
    """
    if bands.depths is None:
        message: str = "layer_resolved_weights requires bands.depths"
        raise ValueError(message)
    projector_diagonal: Float[Array, " n_orb"] = surface_projector(
        bands.depths,
        intensity_escape_length_ang,
    )
    weights: Float[Array, "n_k n_bands"] = _surface_weighted_expectations(
        bands.eigenvectors,
        projector_diagonal,
    )
    return weights


@jaxtyped(typechecker=beartype)
def layer_resolved_group_traces(  # noqa: DOC502, DOC503
    bands: DiagonalizedBands,
    fixed_groups: tuple[tuple[int, ...], ...],
    intensity_escape_length_ang: ScalarFloat,
) -> Float[Array, "n_k n_group"]:
    r"""Compute surface traces over complete, isolated fixed band groups.

    The result is :math:`\operatorname{Tr}(P_g P_{\rm surface})` for every
    statically registered group and is invariant under arbitrary unitary
    mixing among that group's eigenvectors.

    :see: :class:`~.test_operators.TestLayerResolvedGroupTraces`

    Parameters
    ----------
    bands : DiagonalizedBands
        Band eigensystem carrying non-``None`` orbital ``depths``.
    fixed_groups : tuple[tuple[int, ...], ...]
        Static, nonempty, disjoint groups of unique band indices. Changing
        membership retraces and defines a new evidence identity.
    intensity_escape_length_ang : ScalarFloat
        Finite, positive intensity escape length in Angstrom.

    Returns
    -------
    traces : Float[Array, "n_k n_group"]
        Gauge-invariant surface trace for every k-point and group.

    Raises
    ------
    ValueError
        If the input lacks depths or contains malformed static group metadata.
    EquinoxRuntimeError
        If an excluded band lies within ``DEGENERACY_GROUP_TOL_EV`` of a
        group, or its complement gap is below
        ``GROUP_COMPLEMENT_GAP_MIN_EV`` at any k-point.

    Notes
    -----
    Runtime energy comparisons certify a fixed declaration; they never
    discover or alter group membership. A closed complement gap requires a
    newly registered static merged group.
    """
    if bands.depths is None:
        message: str = "layer_resolved_group_traces requires bands.depths"
        raise ValueError(message)
    n_bands: int = bands.eigenvalues.shape[1]
    _validate_fixed_groups(fixed_groups, n_bands)
    projector_diagonal: Float[Array, " n_orb"] = surface_projector(
        bands.depths,
        intensity_escape_length_ang,
    )
    checked_eigenvectors: Complex[Array, "n_k n_bands n_orb"] = (
        bands.eigenvectors
    )
    all_bands: set[int] = set(range(n_bands))
    group_index: int
    group: tuple[int, ...]
    for group_index, group in enumerate(fixed_groups):
        complement: tuple[int, ...] = tuple(
            sorted(all_bands.difference(group))
        )
        if not complement:
            continue
        cross_gaps: Float[Array, "n_k n_group n_complement"] = jnp.abs(
            bands.eigenvalues[:, group, None]
            - bands.eigenvalues[:, None, complement]
        )
        cuts_degeneracy: Array = jnp.any(cross_gaps <= DEGENERACY_GROUP_TOL_EV)
        checked_eigenvectors = eqx.error_if(
            checked_eigenvectors,
            cuts_degeneracy,
            "layer_resolved_group_traces: fixed group "
            f"{group_index} cuts a degenerate multiplet",
        )
        closes_gap: Array = jnp.any(cross_gaps < GROUP_COMPLEMENT_GAP_MIN_EV)
        checked_eigenvectors = eqx.error_if(
            checked_eigenvectors,
            closes_gap,
            "layer_resolved_group_traces: fixed group "
            f"{group_index} is not complement-isolated",
        )
    per_band: Float[Array, "n_k n_bands"] = _surface_weighted_expectations(
        checked_eigenvectors,
        projector_diagonal,
    )
    traces: Float[Array, "n_k n_group"] = jnp.stack(
        [jnp.sum(per_band[:, group], axis=-1) for group in fixed_groups],
        axis=-1,
    )
    return traces


__all__: list[str] = [
    "layer_resolved_group_traces",
    "layer_resolved_weights",
    "ls_operator",
    "orbital_projector",
    "spin_operator",
    "surface_projector",
]
