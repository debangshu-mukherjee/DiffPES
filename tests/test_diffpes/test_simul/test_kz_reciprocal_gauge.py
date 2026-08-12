"""Distinguish bulk reciprocal covariance from physical repeated-zone contrast.

The bulk-kz driver wraps the normal integration coordinate over one primitive
reciprocal period.  That operation relies on basis-gauge covariance at fixed
parallel momentum and fixed outgoing state.  It does not make a measured
ARPES matrix element periodic when the detected parallel momentum moves into
a neighbouring surface zone.
"""

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Any, Dict, Tuple, cast
from jaxtyping import Array, Complex128, Float64
from numpy.typing import NDArray
from scipy import special

from diffpes.matrixel import (
    contract_polarization,
    matrix_element_intensity,
    orbital_transition_channels,
    project_band_channels,
    transition_source,
)
from diffpes.simul import assemble_spectral_intensity_chunk, effects
from diffpes.types import (
    ExperimentGeometry,
    MatrixElementParams,
    OrbitalBasis,
    SelfEnergyModel,
    SurfaceCell,
    make_matrix_element_params,
    make_orbital_basis,
    make_self_energy_model,
)
from tests.test_diffpes.test_simul import test_kz_driver_scans as driver_tests


def _gauge_shifted_hamiltonian(
    hamiltonian: Float64[Array, "..."],
    fractional_positions: Float64[Array, "..."],
    reciprocal_shift: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Apply the basis-position gauge for one integer shift.

    Parameters
    ----------
    hamiltonian : Float64[Array, "..."]
        Hermitian orbital Hamiltonian at the unshifted momentum.
    fractional_positions : Float64[Array, "..."]
        Fractional orbital centres.
    reciprocal_shift : Float64[Array, "..."]
        Integer reciprocal-lattice shift in fractional coordinates.

    Returns
    -------
    shifted : Float64[Array, "..."]
        Gauge-covariant Hamiltonian at the shifted momentum.
    """
    phases: Complex128[Array, "..."] = jnp.exp(
        2.0j * jnp.pi * (fractional_positions @ reciprocal_shift)
    )
    gauge: Float64[Array, "..."] = jnp.diag(phases)
    shifted: Float64[Array, "..."] = gauge.conj().T @ hamiltonian @ gauge
    return shifted


class TestKzReciprocalGauge:
    """Verify the scientifically restricted reciprocal-periodicity claim.

    The case applies a reciprocal normal shift and requires covariance without
    erasing the physical contrast between repeated zones.
    """

    def test_normal_covariance_does_not_erase_repeated_zone_contrast(
        self,
    ) -> None:
        """Preserve only the fixed-outgoing-state normal-period intensity.

        The normal reciprocal shift applies cancelling diagonal gauges to the
        basis-position Hamiltonian and orbital source. The projected band
        weight and resolvent intensity therefore stay invariant.
        By contrast, a detected in-plane repeated-zone point has a different
        outgoing direction. Its physical matrix element can—and here must—
        differ from a folded periodic scalar.

        Notes
        -----
        A ``2*pi`` cubic direct lattice makes each Cartesian reciprocal period
        equal to one inverse Angstrom.  The second orbital has generic x/z
        fractional coordinates, exposing both gauge transformations.
        """
        fractional_positions: Float64[Array, "2 3"] = jnp.asarray(
            ((0.0, 0.0, 0.0), (0.27, 0.0, 0.31)),
            dtype=jnp.float64,
        )
        positions_cart: Float64[Array, "..."] = (
            2.0 * jnp.pi * fractional_positions
        )
        normal_shift: Float64[Array, "3"] = jnp.asarray((0.0, 0.0, 1.0))
        in_plane_shift: Float64[Array, "3"] = jnp.asarray((1.0, 0.0, 0.0))
        initial_base: Float64[Array, "3"] = jnp.asarray((0.12, 0.0, 0.17))
        initial_momenta: Float64[Array, "..."] = jnp.stack(
            (
                initial_base,
                initial_base + normal_shift,
                initial_base + in_plane_shift,
            )
        )

        final_norm: float = 3.0
        base_final_z: float = math.sqrt(final_norm**2 - initial_base[0] ** 2)
        zone_final_z: float = math.sqrt(
            final_norm**2 - (initial_base[0] + in_plane_shift[0]) ** 2
        )
        final_base: Float64[Array, "..."] = jnp.asarray(
            (initial_base[0], initial_base[1], base_final_z)
        )
        final_momenta: Float64[Array, "..."] = jnp.stack(
            (
                final_base,
                final_base,
                jnp.asarray(
                    (
                        initial_base[0] + in_plane_shift[0],
                        initial_base[1],
                        zone_final_z,
                    )
                ),
            )
        )

        basis: OrbitalBasis = make_orbital_basis(
            atom_indices=(0, 1),
            n=(1, 1),
            l=(0, 0),
            m=(0, 0),
            labels=("A:1s", "B:1s"),
        )
        matrix_params: MatrixElementParams = make_matrix_element_params(
            basis,
            (0, 1),
            sigma_shell=jnp.asarray((1.0, 0.83)),
            phase_shift_angles_shell=jnp.asarray((0.19, -0.23)),
        )
        radial_values: Complex128[Array, "..."] = jnp.broadcast_to(
            jnp.asarray(
                ((0.0j, 1.0j), (0.0j, 0.74j)),
                dtype=jnp.complex128,
            ),
            (3, 2, 2),
        )
        orbital_channels: Complex128[Array, "..."] = (
            orbital_transition_channels(
                initial_momenta,
                final_momenta,
                positions_cart,
                jnp.zeros((2,), dtype=jnp.float64),
                radial_values,
                matrix_params,
                jnp.asarray(8.0),
                basis,
            )
        )
        polarization: Complex128[Array, "3"] = jnp.asarray(
            (0.62 + 0.11j, -0.27 + 0.43j, 0.51 - 0.08j),
            dtype=jnp.complex128,
        )
        orbital_rows: Complex128[Array, "..."] = contract_polarization(
            orbital_channels,
            polarization,
        )

        base_hamiltonian: Complex128[Array, "2 2"] = jnp.asarray(
            ((-0.31, 0.28 + 0.17j), (0.28 - 0.17j, 0.43)),
            dtype=jnp.complex128,
        )
        hamiltonians: Complex128[Array, "..."] = jnp.stack(
            (
                base_hamiltonian,
                _gauge_shifted_hamiltonian(
                    base_hamiltonian,
                    fractional_positions,
                    normal_shift,
                ),
                _gauge_shifted_hamiltonian(
                    base_hamiltonian,
                    fractional_positions,
                    in_plane_shift,
                ),
            )
        )
        sources: Complex128[Array, "..."] = transition_source(orbital_rows)[
            :, None, :, :
        ]
        self_energy: SelfEnergyModel = make_self_energy_model(gamma=0.06)
        spectral_intensity: Float64[Array, "..."] = (
            assemble_spectral_intensity_chunk(
                hamiltonians,
                sources,
                jnp.asarray((-0.12,)),
                self_energy,
                jnp.asarray(0.0),
                jnp.asarray(35.0),
            )[:, 0]
        )

        eigenvalues: Float64[Array, "..."]
        eigenvectors: Complex128[Array, "..."]
        eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hamiltonians)
        del eigenvalues
        band_channels: Complex128[Array, "..."] = project_band_channels(
            orbital_channels,
            jnp.swapaxes(eigenvectors, -1, -2),
        )
        band_amplitudes: Complex128[Array, "..."] = contract_polarization(
            band_channels,
            polarization,
        )
        band_intensity: Float64[Array, "..."] = matrix_element_intensity(
            band_amplitudes
        )

        chex.assert_trees_all_close(
            spectral_intensity[1],
            spectral_intensity[0],
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        chex.assert_trees_all_close(
            band_intensity[1],
            band_intensity[0],
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert not jnp.allclose(
            spectral_intensity[2],
            spectral_intensity[0],
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        assert not jnp.allclose(
            band_intensity[2],
            band_intensity[0],
            rtol=1.0e-8,
            atol=1.0e-10,
        )


class TestKzRegisteredNodeBoundary:
    """Verify that traced bulk scans consume the checked node array.

    The case supplies a shifted quadrature grid and requires rejection in both
    eager and compiled execution.
    """

    def test_shifted_grid_rejects_eager_and_jit(self) -> None:
        """Reject a shape-correct but nonregistered midpoint schedule.

        The candidate differs from every registered centre by ``0.01``. The
        production scan must consume the value returned by
        :func:`equinox.error_if`. Discarding that value lets compiled
        dead-code elimination erase the rejection.

        Notes
        -----
        Exercise the public photon-energy scan in both execution modes. The
        shared tiny fixture supplies valid carriers so the shifted nodes are
        the only violated contract.
        """
        fixture: Dict[str, object] = driver_tests._driver_fixture()  # noqa: SLF001
        shifted_nodes: Float64[Array, "..."] = (
            effects.kz_fractional_nodes(4) + 0.01
        )
        photon_energies: Float64[Array, "1"] = jnp.asarray((28.0,))

        def evaluate(
            candidate_nodes: Float64[Array, "..."],
        ) -> Float64[Array, "..."]:
            """Run the public finite-width scan with candidate nodes."""
            result: Float64[Array, "..."] = driver_tests._simulate_scan(  # noqa: SLF001
                fixture,
                photon_energies,
                mode="bulk_kz",
                kz_nodes_frac=candidate_nodes,
            )
            return result

        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="registered uniform fractional centers",
        ):
            evaluate(shifted_nodes)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="registered uniform fractional centers",
        ):
            jax.jit(evaluate)(shifted_nodes)


class TestCoherentSlabSurfaceFrame:
    """Bind each coherent slab to its consumed surface frame.

    The case supplies inconsistent surface cells and requires the coherent
    slab path to reject them in eager and compiled execution.
    """

    def test_mismatched_surface_cell_rejects_eager_and_jit(self) -> None:
        """Reject a valid surface cell from a different slab geometry.

        The planted cell changes one in-plane lattice length while retaining
        valid shapes and exact integer metadata. Production must compare it
        with the depth-bearing bands before using the reciprocal source map.

        Notes
        -----
        Exercise the public coherent scan eagerly and through an outer JIT.
        The checked reciprocal leaf feeds the source-coordinate conversion,
        so compiled dead-code elimination cannot discard the guard.
        """
        fixture: Dict[str, object] = driver_tests._driver_fixture()  # noqa: SLF001
        matched_cell: SurfaceCell = cast(
            SurfaceCell,
            fixture["surface_cell"],
        )
        mismatched_vectors: Float64[Array, "..."] = (
            matched_cell.in_plane_vectors.at[0, 0].set(
                matched_cell.in_plane_vectors[0, 0] + 0.1
            )
        )
        mismatched_cell: SurfaceCell = eqx.tree_at(
            lambda item: item.in_plane_vectors,
            matched_cell,
            mismatched_vectors,
        )
        photon_energies: Float64[Array, "1"] = jnp.asarray((28.0,))

        def evaluate(candidate_cell: SurfaceCell) -> Float64[Array, "..."]:
            """Run the coherent public scan with one candidate cell."""
            candidate_fixture: Dict[str, object] = dict(fixture)
            candidate_fixture["surface_cell"] = candidate_cell
            result: Float64[Array, "..."] = driver_tests._simulate_scan(  # noqa: SLF001
                candidate_fixture,
                photon_energies,
                mode="coherent_slab",
            )
            return result

        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="must match the DiagonalizedBands surface frame",
        ):
            evaluate(mismatched_cell)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="must match the DiagonalizedBands surface frame",
        ):
            jax.jit(evaluate)(mismatched_cell)


class TestKzProductionQuadratureReplay:
    """Bind the selected node count to the public production scan.

    The case replays the selected quadrature values, sum, and mean-free-path
    directional derivative against the authenticated reference.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(3200)
    def test_selected_value_sum_and_mean_free_path_jvp_match_reference(
        self,
    ) -> None:
        """Compare the 2048-node production replay with 4096 nodes.

        The public bulk-kz route evaluates one orbital, two momenta, two
        energies, and one photon energy. It checks pointwise values, the
        summed intrinsic observable, and a mean-free-path forward JVP.

        Notes
        -----
        The scan is pre-detector. One identical fixed detector-volume and
        exposure factor multiplies both summed values. That count scale
        therefore cancels from the registered relative-error comparison.
        """
        fixture: Dict[str, object] = driver_tests._driver_fixture()  # noqa: SLF001
        base_geometry: ExperimentGeometry = cast(
            ExperimentGeometry,
            fixture["geometry"],
        )
        energy_axis: Float64[Array, "..."] = cast(
            Float64[Array, "..."], fixture["energy_axis"]
        )[:2]
        photon_energies: Float64[Array, "1"] = jnp.asarray((28.0,))
        mean_free_path: Float64[Array, "..."] = (
            base_geometry.mean_free_path_ang
        )

        def replay(
            node_count: int,
        ) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
            """Return one public value and lambda-direction JVP replay."""
            nodes: Float64[Array, "..."] = effects.kz_fractional_nodes(
                node_count
            )

            def evaluate(
                candidate_length: Float64[Array, "..."],
            ) -> Float64[Array, "..."]:
                """Evaluate the public scan at one escape length."""
                candidate_geometry: ExperimentGeometry = eqx.tree_at(
                    lambda item: item.mean_free_path_ang,
                    base_geometry,
                    candidate_length,
                )
                value: Float64[Array, "..."] = driver_tests._simulate_scan(  # noqa: SLF001
                    fixture,
                    photon_energies,
                    mode="bulk_kz",
                    geometry=candidate_geometry,
                    energy_axis=energy_axis,
                    kz_nodes_frac=nodes,
                    checkpoint=True,
                )[0]
                return value

            def value_and_jvp(
                candidate_length: Float64[Array, "..."],
            ) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
                """Differentiate one replay in the unit lambda direction."""
                result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
                    jax.jvp(
                        evaluate,
                        (candidate_length,),
                        (jnp.ones_like(candidate_length),),
                    )
                )
                return result

            compiled: Any = jax.jit(value_and_jvp)
            result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
                compiled(mean_free_path)
            )
            return result

        selected: Float64[Array, "..."]
        selected_jvp: Float64[Array, "..."]
        reference: Float64[Array, "..."]
        reference_jvp: Float64[Array, "..."]
        selected, selected_jvp = replay(2048)
        reference, reference_jvp = replay(4096)

        relative_floor: float = 1.0e-14
        pointwise_scale: Float64[Array, "..."] = jnp.maximum(
            jnp.abs(reference),
            relative_floor,
        )
        pointwise_error: Float64[Array, "..."] = jnp.max(
            jnp.abs(selected - reference) / pointwise_scale
        )
        selected_sum: Float64[Array, "..."] = jnp.sum(selected)
        reference_sum: Float64[Array, "..."] = jnp.sum(reference)
        summed_error: Float64[Array, "..."] = jnp.abs(
            selected_sum - reference_sum
        ) / (jnp.maximum(jnp.abs(reference_sum), relative_floor))
        reference_jvp_norm: Float64[Array, "..."] = jnp.max(
            jnp.abs(reference_jvp)
        )
        jvp_error: Float64[Array, "..."] = jnp.max(
            jnp.abs(selected_jvp - reference_jvp)
        ) / jnp.maximum(reference_jvp_norm, relative_floor)

        assert reference_jvp_norm > 1.0e-10
        assert pointwise_error <= 1.0e-5
        assert summed_error <= 1.0e-6
        assert jvp_error <= 1.0e-4


class TestKzLargePeriodVoigtLimit:
    """Verify the secondary periodized-Voigt full-line parity limit.

    The cases compare the wrapped Fourier form with SciPy periodization and
    require convergence to the full-line profile for a large period.
    """

    def test_scipy_periodization_matches_wrapped_fourier(self) -> None:
        """Bind SciPy image periodization to the wrapped-Voigt authority.

        The explicit Fourier coefficients equal the product of Gaussian and
        Cauchy coefficients used by the production convolution test. A
        separate SciPy image sum evaluates the same periodic profile.

        Notes
        -----
        Bound both omitted tails below ``1e-12``. The image bound uses the
        distant Cauchy envelope, while the Fourier bound uses the maximum
        geometric ratio after the first omitted harmonic.
        """
        period_inv_ang: float = 1_000.0
        sigma_inv_ang: float = 0.35
        gamma_inv_ang: float = 0.08
        offsets_inv_ang: Float64[NDArray, " n_offset"] = np.asarray(
            (-3.0, -0.35, 0.0, 0.8, 2.6),
            dtype=np.float64,
        )

        image_radius: int = 250_000
        image_indices: Float64[NDArray, " n_image"] = np.arange(
            -image_radius,
            image_radius + 1,
            dtype=np.float64,
        )
        image_arguments: Float64[NDArray, "n_offset n_image"] = (
            offsets_inv_ang[:, None] + period_inv_ang * image_indices[None, :]
        )
        periodized_scipy: Float64[NDArray, " n_offset"] = np.sum(
            special.voigt_profile(
                image_arguments,
                sigma_inv_ang,
                gamma_inv_ang,
            ),
            axis=-1,
        )
        half_shifted_radius: float = image_radius + 0.5
        image_remainder_bound: float = (
            8.0
            * gamma_inv_ang
            / (np.pi * period_inv_ang**2)
            * (1.0 / half_shifted_radius**2 + 1.0 / half_shifted_radius)
        )

        fourier_terms: int = 64_000
        harmonics: Float64[NDArray, " n_mode"] = np.arange(
            1,
            fourier_terms + 1,
            dtype=np.float64,
        )
        cauchy_rate: float = 2.0 * np.pi * gamma_inv_ang / period_inv_ang
        gaussian_rate: float = (
            0.5 * (2.0 * np.pi * sigma_inv_ang / period_inv_ang) ** 2
        )
        coefficients: Float64[NDArray, " n_mode"] = np.exp(
            -cauchy_rate * harmonics - gaussian_rate * harmonics**2
        )
        wrapped_fourier: Float64[NDArray, " n_offset"] = (
            1.0
            + 2.0
            * np.sum(
                coefficients[:, None]
                * np.cos(
                    2.0
                    * np.pi
                    * harmonics[:, None]
                    * offsets_inv_ang[None, :]
                    / period_inv_ang
                ),
                axis=0,
            )
        ) / period_inv_ang
        first_omitted: float = float(fourier_terms + 1)
        first_coefficient: float = float(
            np.exp(
                -cauchy_rate * first_omitted - gaussian_rate * first_omitted**2
            )
        )
        subsequent_ratio_bound: float = float(
            np.exp(-cauchy_rate - gaussian_rate * (2.0 * first_omitted + 1.0))
        )
        fourier_remainder_bound: float = (
            2.0
            / period_inv_ang
            * first_coefficient
            / (1.0 - subsequent_ratio_bound)
        )

        assert image_remainder_bound <= 1.0e-12
        assert fourier_remainder_bound <= 1.0e-12
        np.testing.assert_allclose(
            periodized_scipy,
            wrapped_fourier,
            rtol=1.0e-8,
            atol=image_remainder_bound + fourier_remainder_bound,
        )

    def test_periodized_voigt_reduces_to_scipy_full_line(self) -> None:
        """Match a large-period image sum to ``scipy``'s full-line Voigt.

        The primary truth is the independent wrapped-Fourier fixture. This
        secondary test instead periodizes SciPy's full-line profile explicitly
        and verifies that its central image is the large-period limit. A
        conservative tail bound covers every omitted image.

        Notes
        -----
        Split the Gaussian expectation at ``|Y|=|x|/2`` for
        ``|x| <= G/2``. This bounds each distant image by
        ``4*gamma/(pi*x**2)`` plus a negligible Gaussian-tail term. Sum the
        envelope beyond the retained image radius to obtain the uniform bound.
        """
        period_inv_ang: float = 200_000.0
        sigma_inv_ang: float = 0.35
        gamma_inv_ang: float = 0.08
        offsets_inv_ang: Float64[NDArray, " n_offset"] = np.asarray(
            (-3.0, -1.4, -0.35, 0.0, 0.8, 2.6),
            dtype=np.float64,
        )
        image_radius: int = 8
        image_indices: Float64[NDArray, " n_image"] = np.arange(
            -image_radius,
            image_radius + 1,
            dtype=np.float64,
        )
        image_arguments: Float64[NDArray, "n_offset n_image"] = (
            offsets_inv_ang[:, None] + period_inv_ang * image_indices[None, :]
        )
        periodized: Float64[NDArray, " n_offset"] = np.sum(
            special.voigt_profile(
                image_arguments,
                sigma_inv_ang,
                gamma_inv_ang,
            ),
            axis=-1,
        )
        full_line: Float64[NDArray, " n_offset"] = special.voigt_profile(
            offsets_inv_ang,
            sigma_inv_ang,
            gamma_inv_ang,
        )

        half_shifted_radius: float = image_radius + 0.5
        cauchy_remainder_bound: float = (
            8.0
            * gamma_inv_ang
            / (np.pi * period_inv_ang**2)
            * (1.0 / half_shifted_radius**2 + 1.0 / half_shifted_radius)
        )
        nearest_omitted_distance: float = (image_radius + 0.5) * period_inv_ang
        gaussian_remainder_bound: float = special.erfc(
            nearest_omitted_distance / (2.0 * np.sqrt(2.0) * sigma_inv_ang)
        ) / (np.pi * gamma_inv_ang)
        reference_remainder_bound: float = (
            cauchy_remainder_bound + gaussian_remainder_bound
        )

        assert reference_remainder_bound <= 1.0e-12
        np.testing.assert_allclose(
            periodized,
            full_line,
            rtol=1.0e-8,
            atol=0.0,
        )
