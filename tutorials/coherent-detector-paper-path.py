# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A Coherent Tight-Binding Model to Detector Counts
#
# This paper-path example connects one native tight-binding model to a
# coherent single-$k_z$ ARPES source, a Fermi-surface map, an independently
# fitted analyser transmission, and native detector counts. It uses the
# canonical detector driver rather than manually reordering its effects.
#
# The example is intentionally small enough for CPU documentation builds. The
# one-orbital model makes the coherent source especially transparent; richer
# orbital bases use the same carriers and driver.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from diffpes.simul import (
    assemble_spectral_intensity_chunk,
    sample_poisson_counts,
    simulate_arpes,
    transmission_shape,
)
from diffpes.tightb import (
    bloch_hamiltonian_batch,
    diagonalize_tb,
)
from diffpes.types import (
    fermi_surface_map,
    make_arpes_cube,
    make_crystal_geometry,
    make_detector_calibration,
    make_detector_effects,
    make_experiment_geometry,
    make_final_state_spec,
    make_kgrid,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_tb_model,
)

# %% [markdown]
# ## 1. Build a Native Tight-Binding Raster
#
# A square lattice with one $s$ orbital has four explicitly Hermitian-closed
# hopping records. Choosing lattice rows $2\pi I$ makes the fractional and
# sample-Cartesian in-plane coordinates numerically identical in this
# pedagogical model.

# %%
crystal = make_crystal_geometry(
    lattice=2.0 * jnp.pi * jnp.eye(3),
    positions=jnp.zeros((1, 3)),
    species=("X",),
)
basis = make_orbital_basis(
    atom_indices=(0,),
    n=(1,),
    l=(0,),
    m=(0,),
    labels=("1s",),
)
model = make_tb_model(
    hopping_amplitudes=0.18 * jnp.ones(4, dtype=jnp.complex128),
    onsite_energies=jnp.asarray([-0.36]),
    soc_lambdas=jnp.zeros(0),
    geometry=crystal,
    basis=basis,
    hopping_pairs=((0, 0), (0, 0), (0, 0), (0, 0)),
    hopping_cells=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
    shell_index=(-1,),
    depths=jnp.asarray([0.25]),
)

k_axis = jnp.linspace(-0.22, 0.22, 7)
mesh_x, mesh_y = jnp.meshgrid(k_axis, k_axis, indexing="xy")
kpoints = jnp.stack((mesh_x, mesh_y, jnp.zeros_like(mesh_x)), axis=-1).reshape(
    (-1, 3)
)
kgrid = make_kgrid(kpoints, mesh_shape=(7, 7), kz=0.0)
hamiltonians = bloch_hamiltonian_batch(model, kpoints)
bands = diagonalize_tb(model, kpoints)

print("Hamiltonian raster:", hamiltonians.shape)
print(
    "band-energy range (eV):",
    tuple(
        float(value)
        for value in (bands.eigenvalues.min(), bands.eigenvalues.max())
    ),
)

# %% [markdown]
# ## 2. Assemble the Coherent Source Cube
#
# The source ket below is one explicit coherent outgoing channel. The public
# resolvent assembly evaluates the Hamiltonian directly, applies a causal
# self-energy, and samples the Fermi occupation on the requested energy axis.
# It does not convolve or normalize the result.

# %%
energy_axis = jnp.linspace(-0.24, 0.08, 33)
self_energy = make_self_energy_model(gamma=0.035)
transition_sources = jnp.ones(
    (kpoints.shape[0], energy_axis.shape[0], 1, 1),
    dtype=jnp.complex128,
)
source_flat = assemble_spectral_intensity_chunk(
    hamiltonians,
    transition_sources,
    energy_axis,
    self_energy,
    bands.fermi_energy,
    25.0,
)
source_intensity = source_flat.reshape((7, 7, 33)).transpose((1, 0, 2))
source_cube = make_arpes_cube(
    source_intensity,
    k_axis,
    k_axis,
    energy_axis,
    provenance="one-orbital coherent resolvent tutorial",
)
fermi_map = fermi_surface_map(source_cube, tol_ev=0.02)

fig, ax = plt.subplots(figsize=(4.8, 4.0))
image = ax.imshow(
    fermi_map.T,
    origin="lower",
    extent=(
        float(k_axis[0]),
        float(k_axis[-1]),
        float(k_axis[0]),
        float(k_axis[-1]),
    ),
    cmap="inferno",
)
ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
ax.set_title(r"coherent $E_F \pm 20$ meV map")
fig.colorbar(image, ax=ax, label="mean spectral intensity")
plt.show()

# %% [markdown]
# `fermi_surface_map` is a display reduction over an explicit top-hat window.
# The complete sampled-energy cube remains the physical source carrier.

# %% [markdown]
# ## 3. Fit an Independent Analyser Calibration
#
# Fit two raw transmission slopes to a small synthetic calibration scan. The
# calibration domain fixes the basis and mean-one normalization, so cropping
# the ARPES query cannot redefine the response. This fit determines shape,
# not absolute throughput; exposure remains an explicit separate parameter.

# %%
calibration = make_detector_calibration(
    u_bin_edges=jnp.linspace(-0.050, 0.050, 9),
    v_bin_edges=jnp.linspace(-0.050, 0.050, 9),
    energy_bin_edges_ev=jnp.linspace(-0.22, 0.06, 15),
    psf_fwhm_u=0.008,
    psf_fwhm_v=0.010,
    psf_fwhm_energy_ev=0.025,
    transmission_reference_domain_ev=jnp.asarray([44.5, 46.0]),
)
calibration_energy = jnp.linspace(44.6, 45.9, 24)
planted_slopes = jnp.asarray([-0.65, 0.30])
measured_transmission = transmission_shape(
    calibration_energy, planted_slopes, calibration
)


def transmission_loss(raw_slopes):
    prediction = transmission_shape(
        calibration_energy, raw_slopes, calibration
    )
    return jnp.mean((prediction - measured_transmission) ** 2)


optimizer = optax.adam(learning_rate=0.08)
fitted_slopes = jnp.zeros(2)
optimizer_state = optimizer.init(fitted_slopes)


@jax.jit
def fit_step(raw_slopes, state):
    loss, gradient = jax.value_and_grad(transmission_loss)(raw_slopes)
    updates, state = optimizer.update(gradient, state, raw_slopes)
    raw_slopes = optax.apply_updates(raw_slopes, updates)
    return raw_slopes, state, loss


for _ in range(120):
    fitted_slopes, optimizer_state, fit_loss = fit_step(
        fitted_slopes, optimizer_state
    )

fit_error = jnp.max(
    jnp.abs(
        transmission_shape(calibration_energy, fitted_slopes, calibration)
        - measured_transmission
    )
)
print("fitted raw slopes:", fitted_slopes)
print(f"maximum calibration residual: {float(fit_error):.3e}")

# %% [markdown]
# ## 4. Run the Canonical Detector/Count Driver
#
# The driver now rebuilds the coherent source with the production
# matrix-element channels and applies the registered order exactly once:
# source mapping and domain mixing, true-kinetic-energy transmission, native
# detector resolution, background, sensitivity, exposure, and bin volume.
# The explicit exposure is chosen from the deterministic rate scale to give
# order-$10^2$ expected events; it is fixed before the PRNG key is used.

# %%
experiment = make_experiment_geometry(
    photon_energy_ev=50.0,
    polarization=jnp.asarray([1.0 + 0.0j, 0.25j, 0.0j]),
    work_function_ev=4.5,
    temperature_k=25.0,
    mean_free_path_ang=8.0,
)
radial_spec = make_radial_spec(
    basis,
    (0,),
    mode="fixed",
    fixed_integrals_shell=jnp.asarray([[0.0, 1.0]]),
)
matrix_element_params = make_matrix_element_params(
    basis,
    (0,),
    sigma_shell=jnp.asarray([1.0]),
    phase_shift_angles_shell=jnp.asarray([0.15]),
)
detector_effects = make_detector_effects(
    domain_logits=jnp.asarray([0.0]),
    domain_euler_angles_rad=jnp.zeros((1, 3)),
    transmission_raw_slopes=fitted_slopes,
    background_coefficients=jnp.asarray([-8.0]),
    sensitivity_coefficients=jnp.asarray([]),
    exposure=2.0e8,
    background_mode="flat",
    sensitivity_mode="constant",
    domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
)
detector = simulate_arpes(
    (hamiltonians,),
    (bands,),
    radial_spec,
    matrix_element_params,
    make_radial_quadrature_spec(),
    make_final_state_spec(),
    experiment,
    self_energy,
    kgrid,
    energy_axis,
    calibration,
    detector_effects,
    k_chunk=8,
    energy_chunk=8,
    checkpoint=True,
)
observed_counts = sample_poisson_counts(
    jax.random.key(20260810), detector.expected_counts
)

expected_map = detector.expected_counts[0].sum(axis=-1)
observed_map = observed_counts[0].sum(axis=-1)
print("native detector raster:", detector.expected_counts.shape)
print(
    f"expected/observed totals: {float(expected_map.sum()):.1f} / {int(observed_map.sum())}"
)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
axes[0].imshow(expected_map.T, origin="lower", cmap="magma")
axes[0].set_title("expected counts")
axes[1].imshow(observed_map.T, origin="lower", cmap="magma")
axes[1].set_title("one explicit-key Poisson draw")
for axis in axes:
    axis.set_xlabel("native detector u bin")
    axis.set_ylabel("native detector v bin")
plt.show()

# %% [markdown]
# The integer draw is deliberately outside the differentiable graph. Fit or
# design against `detector.expected_counts`; call a sampler only when an
# acquisition realization is required.
#
# ## Where to Go Next
#
# - Read the [spectral broadening and self-energy guide](../guides/spectral-broadening-and-self-energy.md).
# - Inspect [native tight-binding models](tight-binding-models.md) for
#   multi-orbital and Slater--Koster construction.
# - Use the [quickstart](quickstart.md) to compare the eigen and resolvent
#   spectral paths.
