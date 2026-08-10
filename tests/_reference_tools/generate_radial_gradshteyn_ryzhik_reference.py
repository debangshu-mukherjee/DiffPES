"""Generate independent G&R 6.621.3 radial-transform references.

The generator evaluates Gradshteyn--Ryzhik 6.621.3 with 80 working decimal
digits and writes 50-digit normalized STO and hydrogenic dipole integrals.
It is an offline authority: package tests consume only the frozen CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
from beartype.typing import Dict, Tuple


@dataclass(frozen=True)
class RadialCase:
    """Describe one normalized radial-transform reference case."""

    reference_id: str
    mode: str
    n: int
    angular_momentum: int
    radial_parameter: str
    l_prime: int
    k_bohr_inv: str


CASES: Tuple[RadialCase, ...] = (
    RadialCase("sto_1s", "slater", 1, 0, "0.75", 1, "0.25"),
    RadialCase("sto_2p_to_s", "slater", 2, 1, "1.30", 0, "0.80"),
    RadialCase("sto_3p_to_d", "slater", 3, 1, "2.10", 2, "1.70"),
    RadialCase("sto_4d_to_f", "slater", 4, 2, "0.55", 3, "0.90"),
    RadialCase("hydrogenic_1s", "hydrogenic", 1, 0, "1.20", 1, "0.40"),
    RadialCase("hydrogenic_2p_to_s", "hydrogenic", 2, 1, "1.00", 0, "0.70"),
    RadialCase("hydrogenic_3p_to_d", "hydrogenic", 3, 1, "2.40", 2, "1.30"),
    RadialCase("hydrogenic_4d_to_p", "hydrogenic", 4, 2, "3.20", 1, "2.20"),
    RadialCase("hydrogenic_7g_to_h", "hydrogenic", 7, 4, "3.50", 5, "3.20"),
)


def _gr_66213_power(
    power: int,
    l_prime: int,
    decay: mp.mpf,
    momentum: mp.mpf,
) -> mp.mpf:
    r"""PRIVATE: Evaluate ``integral r**power exp(-a r) j_l(k r) dr``.

    Parameters
    ----------
    power : int
        Radial power of the integrand.
    l_prime : int
        Final-state spherical-Bessel order.
    decay : mp.mpf
        Exponential decay constant in 1/Bohr.
    momentum : mp.mpf
        Photoelectron momentum in 1/Bohr.

    Returns
    -------
    value : mp.mpf
        The integral in Bohr units at the working precision.

    Notes
    -----
    This is G&R 6.621.3 after substituting
    ``j_l(x) = sqrt(pi / (2 x)) J_(l+1/2)(x)``: a Gamma-function
    prefactor times the Gauss hypergeometric ``2F1`` at argument
    ``-(k/a)**2``.
    """
    mu: mp.mpf = mp.mpf(power) + mp.mpf("0.5")
    nu: mp.mpf = mp.mpf(l_prime) + mp.mpf("0.5")
    first: mp.mpf = (mu + nu) / 2
    second: mp.mpf = (mu + nu + 1) / 2
    prefactor: mp.mpf = (
        mp.sqrt(mp.pi / (2 * momentum))
        * momentum**nu
        * mp.gamma(mu + nu)
        / (2**nu * decay ** (mu + nu) * mp.gamma(nu + 1))
    )
    value: mp.mpf = prefactor * mp.hyp2f1(
        first,
        second,
        nu + 1,
        -((momentum / decay) ** 2),
    )
    return value


def _slater_reference(case: RadialCase) -> mp.mpf:
    """PRIVATE: Evaluate one normalized Slater radial dipole integral.

    Parameters
    ----------
    case : RadialCase
        Frozen case with quantum numbers, the Slater exponent in
        1/Bohr, and the momentum in 1/Bohr.

    Returns
    -------
    value : mp.mpf
        Normalized ``<R_nl | r | j_l'>`` integral in Bohr units.

    Notes
    -----
    The normalized STO radial part is ``N r^{n-1} e^{-zeta r}`` with
    ``N = (2 zeta)^{n+1/2} / sqrt((2n)!)``; the dipole integral then
    reduces to one :func:`_gr_66213_power` call with power ``n + 2``.
    """
    zeta: mp.mpf = mp.mpf(case.radial_parameter)
    normalization: mp.mpf = (2 * zeta) ** (case.n + mp.mpf("0.5"))
    normalization /= mp.sqrt(mp.factorial(2 * case.n))
    value: mp.mpf = normalization * _gr_66213_power(
        case.n + 2,
        case.l_prime,
        zeta,
        mp.mpf(case.k_bohr_inv),
    )
    return value


def _hydrogenic_reference(case: RadialCase) -> mp.mpf:
    """PRIVATE: Evaluate one normalized hydrogenic radial dipole integral.

    Parameters
    ----------
    case : RadialCase
        Frozen case with quantum numbers, the effective charge, and
        the momentum in 1/Bohr.

    Returns
    -------
    value : mp.mpf
        Normalized ``<R_nl | r | j_l'>`` integral in Bohr units.

    Implementation Logic
    --------------------
    The associated Laguerre polynomial expands into its explicit
    binomial sum; each monomial term maps to one
    :func:`_gr_66213_power` call with power ``3 + l + term`` and the
    hydrogenic decay ``Z/n``, and the normalized prefactor scales the
    sum.
    """
    charge: mp.mpf = mp.mpf(case.radial_parameter)
    decay: mp.mpf = charge / case.n
    laguerre_order: int = case.n - case.angular_momentum - 1
    alpha: int = 2 * case.angular_momentum + 1
    normalization: mp.mpf = (2 * decay) ** mp.mpf("1.5")
    normalization *= mp.sqrt(
        mp.factorial(laguerre_order)
        / (2 * case.n * mp.factorial(case.n + case.angular_momentum))
    )
    total: mp.mpf = mp.mpf("0")
    term: int
    for term in range(laguerre_order + 1):
        laguerre_coefficient: mp.mpf = (
            (-1) ** term
            * mp.binomial(laguerre_order + alpha, laguerre_order - term)
            / mp.factorial(term)
        )
        radial_coefficient: mp.mpf = laguerre_coefficient * (2 * decay) ** (
            case.angular_momentum + term
        )
        total += radial_coefficient * _gr_66213_power(
            3 + case.angular_momentum + term,
            case.l_prime,
            decay,
            mp.mpf(case.k_bohr_inv),
        )
    value: mp.mpf = normalization * total
    return value


def _direct_quadrature(case: RadialCase) -> mp.mpf:
    """PRIVATE: Evaluate the same reference by direct quadrature.

    Parameters
    ----------
    case : RadialCase
        Frozen case with quantum numbers, the radial parameter, and
        the momentum in 1/Bohr.

    Returns
    -------
    value : mp.mpf
        The dipole integral in Bohr units from ``mp.quad`` on
        ``[0, inf)``.

    Implementation Logic
    --------------------
    The integrand assembles the normalized Slater or hydrogenic radial
    function explicitly, multiplies by ``r^3`` and the spherical
    Bessel function ``j_l'(k r)`` (with its exact ``k r = 0`` limit),
    and integrates with adaptive arbitrary-precision quadrature as an
    independent check on the closed form.
    """
    momentum: mp.mpf = mp.mpf(case.k_bohr_inv)
    parameter: mp.mpf = mp.mpf(case.radial_parameter)

    def spherical_bessel(radius: mp.mpf) -> mp.mpf:
        argument: mp.mpf = momentum * radius
        if argument == 0:
            return mp.mpf(int(case.l_prime == 0))
        return mp.sqrt(mp.pi / (2 * argument)) * mp.besselj(
            case.l_prime + mp.mpf("0.5"),
            argument,
        )

    if case.mode == "slater":
        normalization: mp.mpf = (2 * parameter) ** (
            case.n + mp.mpf("0.5")
        ) / mp.sqrt(mp.factorial(2 * case.n))

        def integrand(radius: mp.mpf) -> mp.mpf:
            radial: mp.mpf = (
                normalization
                * radius ** (case.n - 1)
                * mp.exp(-parameter * radius)
            )
            return radial * radius**3 * spherical_bessel(radius)

    else:
        decay: mp.mpf = parameter / case.n
        laguerre_order: int = case.n - case.angular_momentum - 1
        alpha: int = 2 * case.angular_momentum + 1
        normalization = (2 * decay) ** mp.mpf("1.5") * mp.sqrt(
            mp.factorial(laguerre_order)
            / (2 * case.n * mp.factorial(case.n + case.angular_momentum))
        )

        def integrand(radius: mp.mpf) -> mp.mpf:
            scaled: mp.mpf = 2 * decay * radius
            radial: mp.mpf = (
                normalization
                * mp.exp(-decay * radius)
                * scaled**case.angular_momentum
                * mp.laguerre(laguerre_order, alpha, scaled)
            )
            return radial * radius**3 * spherical_bessel(radius)

    value: mp.mpf = mp.quad(integrand, [0, mp.inf])
    return value


def generate(output: Path) -> None:
    """Write the frozen reference CSV."""
    mp.mp.dps = 80
    rows: list[Dict[str, str]] = []
    case: RadialCase
    for case in CASES:
        if case.mode == "slater":
            closed_value: mp.mpf = _slater_reference(case)
        else:
            closed_value = _hydrogenic_reference(case)
        quadrature_value: mp.mpf = _direct_quadrature(case)
        absolute_difference: mp.mpf = abs(closed_value - quadrature_value)
        scale: mp.mpf = max(mp.mpf(1), abs(closed_value))
        if absolute_difference > mp.mpf("1e-20") * scale:
            message: str = (
                f"{case.reference_id}: G&R/direct mismatch "
                f"{mp.nstr(absolute_difference, 8)}"
            )
            raise RuntimeError(message)
        rows.append(
            {
                "reference_id": case.reference_id,
                "mode": case.mode,
                "n": str(case.n),
                "angular_momentum": str(case.angular_momentum),
                "radial_parameter": case.radial_parameter,
                "l_prime": str(case.l_prime),
                "k_bohr_inv": case.k_bohr_inv,
                "expected_unphased": mp.nstr(closed_value, 50),
                "gr_direct_abs_difference": mp.nstr(absolute_difference, 10),
                "working_digits": str(mp.mp.dps),
                "frozen_digits": "50",
                "authority": "Gradshteyn-Ryzhik 6.621.3",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the output path and generate the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tests/test_diffpes/_reference_data/"
            "radial_gradshteyn_ryzhik_66213_reference.csv"
        ),
    )
    arguments: argparse.Namespace = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
