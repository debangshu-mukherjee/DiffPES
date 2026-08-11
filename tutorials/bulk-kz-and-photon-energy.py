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
# # Bulk $k_z$ Integration and Photon-Energy Scans
#
# This example builds a small $k_z$-dispersive tight-binding model, inspects
# the registered wrapped-Lorentzian bin weights, and evaluates a compact
# photon-energy scan through the public canonical driver. The example keeps
# the scientific production recommendation separate from its deliberately
# small documentation workload.

# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffpes.simul import (
    hv_map_at_energy,
    kz_fractional_nodes,
    kz_wrapped_lorentzian_bin_weights,
    simulate_hv_scan,
)
from diffpes.types import (
    make_crystal_geometry,
    make_experiment_geometry,
    make_final_state_spec,
    make_kpath,
    make_matrix_element_params,
    make_orbital_basis,
    make_radial_quadrature_spec,
    make_radial_spec,
    make_self_energy_model,
    make_surface_cell,
    make_tb_model,
)

# %% [markdown]
# ## 1. Choose One of Four Mutually Exclusive Modes
#
# The canonical drivers distinguish four physical routes. They are modes,
# not accuracy levels that can be combined:
#
# | `kz_mode` | Electronic input | Surface/node input | Meaning |
# |---|---|---|---|
# | `native_direct` | explicit Hamiltonian plus native bands | neither | retain the caller's single native $k_z$ |
# | `bulk_direct` | bulk `TBModel` | `SurfaceCell`, no nodes | exact zero-width limit with finite-energy final-state kinematics |
# | `bulk_kz` | bulk `TBModel` | `SurfaceCell` plus registered nodes | finite escape-depth integral over one wrapped reciprocal period |
# | `coherent_slab` | explicit Hamiltonian plus depth-bearing bands | `SurfaceCell`, no bulk model or nodes | coherent depth-resolved slab emission |
#
# Passing carriers from two rows together is rejected. This tutorial uses
# `bulk_kz`; the other rows remain distinct public contracts.

# %%
modes = (
    "native_direct",
    "bulk_direct",
    "bulk_kz",
    "coherent_slab",
)
print("registered mode choices:", ", ".join(modes))

# %% [markdown]
# ## 2. Inspect the Wrapped $k_z$ Quadrature
#
# The registered Plan-08b G6 profile recommends `n_kz=2048`. The public
# helper returns uniform bin centres over $[-1/2,1/2)$; the second helper
# integrates the wrapped Lorentzian analytically over their bin edges. A
# centre near $+1/2$ therefore has visible mass at both ends of the plotted
# primitive period, while its total mass remains one.
#
# The recommendation is calibrated for the frozen G6 profile. It is not a
# universal default: inputs outside that profile require their own convergence
# study. Computing these one-dimensional weights is cheap and creates no
# spectral all-node carrier.

# %%
recommended_n_kz = 2048
recommended_nodes = kz_fractional_nodes(recommended_n_kz)
recommended_edges = jnp.linspace(-0.5, 0.5, recommended_n_kz + 1)
recommended_weights = kz_wrapped_lorentzian_bin_weights(
    recommended_edges,
    jnp.asarray(0.46),
    mean_free_path_ang=7.5,
    period_inv_ang=2.0 * jnp.pi / 3.2,
)
print("recommended node count:", recommended_nodes.shape[0])
print("wrapped mass:", float(recommended_weights.sum()))

fig, ax = plt.subplots(figsize=(6.2, 3.5))
ax.plot(recommended_nodes, recommended_weights)
ax.set_xlabel("surface-fractional $k_z$")
ax.set_ylabel("analytic bin mass")
ax.set_title("wrapped Lorentzian centred at 0.46")
plt.show()

# %% [markdown]
# ## 3. Build a Compact Bulk Model
#
# The identity surface below makes the coordinate roles easy to see. Paired
# hoppings along $x$ and $z$ keep the model Hermitian and make both path
# momentum and photon-energy-dependent $k_z$ observable.

# %%
lattice_scale = 3.2
crystal = make_crystal_geometry(
    lattice_scale * jnp.eye(3),
    jnp.zeros((1, 3)),
    ("X",),
)
basis = make_orbital_basis(
    atom_indices=(0,),
    n=(1,),
    l=(0,),
    m=(0,),
    labels=("1s",),
)
bulk_model = make_tb_model(
    hopping_amplitudes=jnp.asarray(
        (-0.12, -0.12, -0.38, -0.38), dtype=jnp.complex128
    ),
    onsite_energies=jnp.asarray((-0.05,)),
    soc_lambdas=jnp.zeros((0,)),
    geometry=crystal,
    basis=basis,
    hopping_pairs=((0, 0), (0, 0), (0, 0), (0, 0)),
    hopping_cells=((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)),
    shell_index=(-1,),
)
surface_cell = make_surface_cell(
    in_plane_vectors=lattice_scale
    * jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
    stacking_vector=lattice_scale * jnp.asarray((0.0, 0.0, 1.0)),
    rotation=jnp.eye(3),
    interlayer_spacing_ang=lattice_scale,
    miller=(0, 0, 1),
    in_plane_coeffs=((1, 0, 0), (0, 1, 0)),
    stacking_coeffs=(0, 0, 1),
)

path_coordinate = jnp.linspace(-0.055, 0.055, 5)
path_points = jnp.stack(
    (
        path_coordinate,
        jnp.zeros_like(path_coordinate),
        jnp.zeros_like(path_coordinate),
    ),
    axis=-1,
)
kpath = make_kpath(path_points, n_per_segment=1, kz=0.0)

radial_spec = make_radial_spec(
    basis,
    (0,),
    mode="fixed",
    fixed_integrals_shell=jnp.asarray(((0.0, 1.0),)),
)
matrix_element_params = make_matrix_element_params(
    basis,
    (0,),
    sigma_shell=jnp.asarray((1.13,)),
    phase_shift_angles_shell=jnp.asarray((0.21,)),
)
geometry = make_experiment_geometry(
    photon_energy_ev=28.0,
    polarization=jnp.asarray((1.0, 0.35j, 0.0)),
    sample_azimuth=0.14,
    work_function_ev=4.3,
    inner_potential_ev=11.0,
    temperature_k=45.0,
    mean_free_path_ang=7.5,
)

# %% [markdown]
# ## 4. Run a Small $h\nu$ Scan
#
# To keep documentation execution light, the next cell uses only eight
# $k_z$ nodes. That count is a pedagogical runtime choice and is **not** G6
# production evidence. Use the calibrated 2048-node profile above for its
# registered domain, or perform a fresh convergence study for other inputs.
#
# `simulate_hv_scan` reevaluates exact finite-energy kinematics and matrix
# elements at every photon energy. `hv_map_at_energy` then interpolates one
# requested binding energy and orients the result as path momentum by photon
# energy.

# %%
demo_nodes = kz_fractional_nodes(8)
energy_axis = jnp.linspace(-0.42, -0.06, 7)
photon_energies = jnp.asarray((25.5, 28.0, 31.5))

scan = simulate_hv_scan(
    None,
    None,
    radial_spec,
    matrix_element_params,
    make_radial_quadrature_spec(),
    make_final_state_spec(),
    geometry,
    make_self_energy_model(gamma=0.055),
    kpath,
    energy_axis,
    photon_energies,
    k_chunk=5,
    energy_chunk=7,
    checkpoint=True,
    bulk_model=bulk_model,
    surface_cell=surface_cell,
    kz_nodes_frac=demo_nodes,
    kz_mode="bulk_kz",
)
constant_energy_map = hv_map_at_energy(scan, energy_axis, -0.21)
print("scan shape (hnu, path, energy):", scan.shape)
print("map shape (path, hnu):", constant_energy_map.shape)

fig, ax = plt.subplots(figsize=(5.8, 3.8))
image = ax.imshow(
    constant_energy_map,
    origin="lower",
    aspect="auto",
    extent=(
        float(photon_energies[0]),
        float(photon_energies[-1]),
        float(path_coordinate[0]),
        float(path_coordinate[-1]),
    ),
    cmap="magma",
)
ax.set_xlabel(r"$h\nu$ (eV)")
ax.set_ylabel("source path coordinate")
ax.set_title(r"bulk-$k_z$ intensity at $\omega=-0.21$ eV")
fig.colorbar(image, ax=ax, label="intrinsic intensity")
plt.show()

# %% [markdown]
# ## 5. Boundaries and Memory Semantics
#
# Chinook's retained compatibility fixture is a single-$k_z$, $\omega=0$
# boundary comparison. It is context for the direct Fermi-level seam, not an
# oracle for the finite-$\omega$ wrapped bulk integral. Here every sampled
# $\omega$ receives its own final-state centre before the bulk Hamiltonian is
# evaluated.
#
# Production `bulk_kz` does not materialize a
# `K x bands x energy x n_kz` spectrum or a `K x n_kz x 3` kinematic table.
# Its checkpointed scan consumes one scalar node, maps only the current
# `(K, 3)` coordinates, and accumulates the `(K, energy)` observable. The
# one-dimensional node and weight arrays plotted above are therefore not a
# hidden full-node spectral carrier.
#
# The returned scan is still proportional to `n_hv` because those requested
# maps are physical outputs. Plan-08b's separate S2 gate measures whether
# auxiliary checkpoint memory remains flat beyond that required output axis.
#
# Continue with the [coherent detector paper path](coherent-detector-paper-path.ipynb)
# to apply analyser response and generate expected detector counts.
