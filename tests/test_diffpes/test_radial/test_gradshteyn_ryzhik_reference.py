"""Certify radial transforms against frozen G&R 6.621.3 references.

Extended Summary
----------------
The tests compare normalized radial transforms and retain arbitrary-precision
authority metadata for every frozen case.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict, List, TextIO
from jaxtyping import Array, Float64

from diffpes.radial import (
    gauss_legendre_nodes,
    hydrogenic_radial,
    radial_integral,
    slater_radial,
)

REFERENCE_PATH: Path = (
    Path(__file__).parents[1]
    / "_reference_data"
    / "radial_gradshteyn_ryzhik_66213_reference.csv"
)


class TestGradshteynRyzhikReference:
    """Compare production STO and hydrogenic dipole transforms with G&R.

    The battery covers nine normalized Slater and hydrogenic radial cases.
    It applies 1,024-node Gauss--Legendre quadrature through the public API and
    compares the results with frozen 50-digit formula values.
    """

    def test_normalized_radial_battery_matches_frozen_50_digit_values(
        self,
    ) -> None:
        """Match all frozen values through the public r-cubed integral API.

        The offline generator evaluates G&R 6.621.3 at 80 working digits,
        cross-checks it by direct arbitrary-precision quadrature, and freezes
        50 digits. This test deliberately has no mpmath runtime dependency.

        Notes
        -----
        It reads frozen rows and evaluates the public radial-integral API.
        """
        stream: TextIO
        with REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
            rows: List[Dict[str, str]] = list(csv.DictReader(stream))

        assert {row["mode"] for row in rows} == {"slater", "hydrogenic"}
        assert len(rows) == 9
        nodes: Float64[Array, " 1024"]
        weights: Float64[Array, " 1024"]
        nodes, weights = gauss_legendre_nodes(1024, 120.0)
        row: Dict[str, str]
        for row in rows:
            n_value: int = int(row["n"])
            angular_momentum: int = int(row["angular_momentum"])
            parameter: Float64[Array, ""] = jnp.asarray(
                float(row["radial_parameter"]),
                dtype=jnp.float64,
            )
            radial_values: Float64[Array, " 1024"]
            if row["mode"] == "slater":
                radial_values = slater_radial(
                    nodes,
                    n=n_value,
                    zeta=parameter,
                )
            else:
                radial_values = hydrogenic_radial(
                    nodes,
                    n=n_value,
                    angular_momentum=angular_momentum,
                    z_eff=parameter,
                )
            l_prime: int = int(row["l_prime"])
            actual: complex = complex(
                radial_integral(
                    jnp.asarray(
                        float(row["k_bohr_inv"]),
                        dtype=jnp.float64,
                    ),
                    nodes,
                    weights,
                    radial_values,
                    l_prime,
                )
            )
            expected: complex = (1j) ** l_prime * float(
                row["expected_unphased"]
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1.0e-9,
                atol=1.0e-12,
                err_msg=row["reference_id"],
            )

    def test_frozen_reference_records_authority_and_precision(self) -> None:
        """Require each row to retain its independent reference provenance.

        The test checks authority labels, precision, and cross-check errors.

        Notes
        -----
        It reads the metadata columns and enforces their registered bounds.
        """
        stream: TextIO
        with REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
            rows: List[Dict[str, str]] = list(csv.DictReader(stream))

        row: Dict[str, str]
        for row in rows:
            assert row["authority"] == "Gradshteyn-Ryzhik 6.621.3"
            assert int(row["working_digits"]) >= 50
            assert int(row["frozen_digits"]) == 50
            difference: float = float(row["gr_direct_abs_difference"])
            expected_scale: float = max(
                1.0,
                abs(float(row["expected_unphased"])),
            )
            assert difference <= 1.0e-20 * expected_scale
