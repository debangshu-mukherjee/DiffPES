"""Compute static atomic configurations and Slater screening estimates.

Extended Summary
----------------
The module fills neutral ground-state subshells through element 103 and
applies Slater's original grouped screening rules. These host-side helpers
initialize differentiable radial carriers; they do not enter traced kernels.

Routine Listings
----------------
:func:`electron_configuration`
    Return the neutral ground-state subshell configuration.
:func:`slater_zeff`
    Compute a subshell effective charge from Slater screening.
:func:`slater_zeta`
    Compute a Slater exponent from the effective principal number.
"""

from beartype import beartype
from jaxtyping import jaxtyped


def _aufbau_configuration(atomic_number: int) -> dict[tuple[int, int], int]:
    """PRIVATE: Return the Madelung sequence with ground-state exceptions.

    Parameters
    ----------
    atomic_number : int
        Validated atomic number from 1 through 103.

    Returns
    -------
    configuration : dict[tuple[int, int], int]
        Occupancy for each occupied ``(n, l)`` subshell.

    Implementation Logic
    --------------------
    Fills a static nineteen-subshell Madelung table from 1s through 7p
    until no electrons remain.  A static exception map then overrides
    the anomalous neutral ground states: chromium, copper, the 4d and
    5d anomalies, and the lanthanide and actinide rows.  An override
    occupancy of zero removes the subshell.
    """
    orbitals: tuple[tuple[int, int, int], ...] = (
        (1, 0, 2),
        (2, 0, 2),
        (2, 1, 6),
        (3, 0, 2),
        (3, 1, 6),
        (4, 0, 2),
        (3, 2, 10),
        (4, 1, 6),
        (5, 0, 2),
        (4, 2, 10),
        (5, 1, 6),
        (6, 0, 2),
        (4, 3, 14),
        (5, 2, 10),
        (6, 1, 6),
        (7, 0, 2),
        (5, 3, 14),
        (6, 2, 10),
        (7, 1, 6),
    )
    remaining: int = atomic_number
    configuration: dict[tuple[int, int], int] = {}
    principal: int
    angular: int
    capacity: int
    for principal, angular, capacity in orbitals:
        occupancy: int = min(remaining, capacity)
        if occupancy:
            configuration[(principal, angular)] = occupancy
        remaining -= occupancy
        if remaining == 0:
            break

    exceptions: dict[int, dict[tuple[int, int], int]] = {
        24: {(4, 0): 1, (3, 2): 5},
        29: {(4, 0): 1, (3, 2): 10},
        41: {(5, 0): 1, (4, 2): 4},
        42: {(5, 0): 1, (4, 2): 5},
        44: {(5, 0): 1, (4, 2): 7},
        45: {(5, 0): 1, (4, 2): 8},
        46: {(5, 0): 0, (4, 2): 10},
        47: {(5, 0): 1, (4, 2): 10},
        57: {(4, 3): 0, (5, 2): 1, (6, 0): 2},
        58: {(4, 3): 1, (5, 2): 1, (6, 0): 2},
        64: {(4, 3): 7, (5, 2): 1, (6, 0): 2},
        78: {(6, 0): 1, (5, 2): 9},
        79: {(6, 0): 1, (5, 2): 10},
        89: {(5, 3): 0, (6, 2): 1, (7, 0): 2},
        90: {(5, 3): 0, (6, 2): 2, (7, 0): 2},
        91: {(5, 3): 2, (6, 2): 1, (7, 0): 2},
        92: {(5, 3): 3, (6, 2): 1, (7, 0): 2},
        93: {(5, 3): 4, (6, 2): 1, (7, 0): 2},
        96: {(5, 3): 7, (6, 2): 1, (7, 0): 2},
        103: {(5, 3): 14, (6, 2): 0, (7, 0): 2, (7, 1): 1},
    }
    override: dict[tuple[int, int], int]
    if atomic_number in exceptions:
        override = exceptions[atomic_number]
        key: tuple[int, int]
        occupancy: int
        for key, occupancy in override.items():
            if occupancy == 0:
                configuration.pop(key, None)
            else:
                configuration[key] = occupancy
    return configuration


@jaxtyped(typechecker=beartype)
def electron_configuration(
    atomic_number: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return the neutral ground-state subshell configuration.

    The function fills the Madelung sequence and applies the measured
    chromium, copper, transition-metal, lanthanide, and actinide exceptions
    through lawrencium.

    :see: :class:`~.test_screening.TestElectronConfiguration`

    Parameters
    ----------
    atomic_number : int
        Atomic number from 1 through 103.

    Returns
    -------
    configuration : tuple[tuple[int, int, int], ...]
        Occupied ``(n, l, occupancy)`` rows in increasing ``(n+l, n)`` order.

    Raises
    ------
    ValueError
        If the atomic number lies outside 1 through 103.

    Notes
    -----
    The exception set uses conventional neutral isolated-atom
    configurations. Quantum numbers remain host-side integers.
    """
    if (
        type(atomic_number) is not int or not 1 <= atomic_number <= 103  # noqa: PLR2004
    ):
        message: str = "atomic_number must be an integer from 1 through 103"
        raise ValueError(message)
    occupied: dict[tuple[int, int], int] = _aufbau_configuration(atomic_number)
    configuration: tuple[tuple[int, int, int], ...] = tuple(
        (principal, angular, occupied[(principal, angular)])
        for principal, angular in sorted(
            occupied,
            key=lambda quantum: (
                quantum[0] + quantum[1],
                quantum[0],
            ),
        )
    )
    return configuration


@jaxtyped(typechecker=beartype)
def slater_zeff(
    atomic_number: int,
    n: int,
    l: int,  # noqa: E741
) -> float:
    """Compute a subshell effective charge from Slater screening.

    For an ``ns`` or ``np`` electron, same-group electrons contribute 0.35
    each except 1s partners contribute 0.30. The preceding shell contributes
    0.85 per electron and deeper shells contribute one. For ``nd`` and
    ``nf`` electrons, same-subshell partners contribute 0.35 and every inner
    electron contributes one.

    :see: :class:`~.test_screening.TestSlaterZeff`

    Parameters
    ----------
    atomic_number : int
        Atomic number from 1 through 103.
    n : int
        Principal quantum number of an occupied subshell.
    l : int
        Angular momentum of an occupied subshell.

    Returns
    -------
    effective_charge : float
        ``atomic_number - screening``.

    Raises
    ------
    ValueError
        If the requested subshell is invalid or unoccupied.

    Notes
    -----
    Decimal screening factors represent exact rules rather than fitted
    floating-point data.
    """
    configuration: tuple[tuple[int, int, int], ...] = electron_configuration(
        atomic_number
    )
    occupancies: dict[tuple[int, int], int] = {
        (principal, angular): occupancy
        for principal, angular, occupancy in configuration
    }
    if (
        type(n) is not int
        or type(l) is not int
        or n < 1
        or l < 0
        or l >= n
        or occupancies.get((n, l), 0) == 0
    ):
        message: str = "requested n, l subshell must be occupied"
        raise ValueError(message)

    screening: float = 0.0
    principal: int
    angular: int
    occupancy: int
    if l <= 1:
        for (principal, angular), occupancy in occupancies.items():
            if principal == n and angular <= 1:
                partners: int = occupancy - int(angular == l)
                factor: float = 0.30 if n == 1 else 0.35
                screening += factor * partners
            elif principal == n - 1:
                screening += 0.85 * occupancy
            elif principal <= n - 2:
                screening += float(occupancy)
    else:
        for (principal, angular), occupancy in occupancies.items():
            if principal == n and angular == l:
                screening += 0.35 * (occupancy - 1)
            elif principal < n or (principal == n and angular < l):
                screening += float(occupancy)
    effective_charge: float = float(atomic_number) - screening
    return effective_charge


@jaxtyped(typechecker=beartype)
def slater_zeta(  # noqa: DOC502
    atomic_number: int,
    n: int,
    l: int,  # noqa: E741
) -> float:
    """Compute a Slater exponent from the effective principal number.

    The function divides ``slater_zeff`` by Slater's sequence
    ``(1, 2, 3, 3.7, 4.0, 4.2)``. The last registered value applies to
    principal shells six and seven.

    :see: :class:`~.test_screening.TestSlaterZeta`

    Notes
    -----
    Divide the effective charge by Slater's static effective principal value.

    Parameters
    ----------
    atomic_number : int
        Atomic number from 1 through 103.
    n : int
        Principal quantum number of an occupied subshell.
    l : int
        Angular momentum of an occupied subshell.

    Returns
    -------
    zeta : float
        Effective Slater exponent in inverse Bohr.

    Raises
    ------
    ValueError
        If the requested subshell is invalid or unoccupied.
    """
    effective_principal_values: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        3.7,
        4.0,
        4.2,
    )
    effective_charge: float = slater_zeff(atomic_number, n, l)
    effective_principal: float = effective_principal_values[
        min(n, len(effective_principal_values)) - 1
    ]
    zeta: float = effective_charge / effective_principal
    return zeta


__all__: list[str] = [
    "electron_configuration",
    "slater_zeff",
    "slater_zeta",
]
