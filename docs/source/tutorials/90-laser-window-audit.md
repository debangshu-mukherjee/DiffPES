# Laser Window Audit

The photoemission horizon for laser photon energies, drawn over the
internal DFT band structures of Bi2Se3 and PdTe2. A calibrated
tight-binding Dirac cone then turns the accessible window into a
simulated spectrum. The notebook reads the local `data/DFT` tree.

## Load the Public API

The kinematics calls convert photon energies into momentum horizons.
The readers supply band structures, Fermi levels, and reciprocal
lattices. The tight-binding and spectral calls build the simulated cone
at the end.


```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from diffpes.inout import read_eigenval, read_outcar, read_poscar
from diffpes.simul import (
    assemble_spectral_intensity_chunk,
    final_state_k_inv_ang,
    kinetic_energy_ev,
)
from diffpes.tightb import (
    bloch_hamiltonian_batch,
    diagonalize_tb,
    kpath_arc_length,
    kpoints_cart_to_frac,
)
from diffpes.types import (
    make_crystal_geometry,
    make_kpath,
    make_orbital_basis,
    make_self_energy_model,
    make_tb_model,
)
```

## Draw the Photoemission Horizon

`kinetic_energy_ev` applies energy conservation for each photon energy.
`final_state_k_inv_ang` converts the kinetic energy into the maximal
parallel momentum. The audit covers laser lines at 6.05, 6.4, 7.0, and
10.8 eV. The work function enters as 4.5 eV.


```python
LASER_LINES_EV = np.asarray([6.05, 6.4, 7.0, 10.8])
WORK_FUNCTION_EV = 4.5
photon_grid_ev = jnp.linspace(5.0, 12.0, 281)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for trial_work_function in (4.0, 4.5, 5.0, 5.5):
    kinetic_grid, _ = jax.vmap(
        kinetic_energy_ev, in_axes=(0, None, None)
    )(photon_grid_ev, trial_work_function, jnp.asarray(0.0))
    horizon, _ = final_state_k_inv_ang(kinetic_grid)
    ax.plot(
        photon_grid_ev, horizon, label=f"W = {trial_work_function} eV"
    )
for laser_line in LASER_LINES_EV:
    ax.axvline(laser_line, color="0.75", linewidth=0.8)
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel(r"$k_{\parallel}$ horizon ($\AA^{-1}$)")
ax.set_title("horizon at zero binding energy")
ax.legend()
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_4_0.png)




```python
kinetic_grid, _ = jax.vmap(kinetic_energy_ev, in_axes=(0, None, None))(
    photon_grid_ev, WORK_FUNCTION_EV, jnp.asarray(0.0)
)
full_reach, _ = final_state_k_inv_ang(kinetic_grid)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for acceptance_deg in (15.0, 30.0, 90.0):
    reach = full_reach * np.sin(np.deg2rad(acceptance_deg))
    ax.plot(
        photon_grid_ev,
        reach,
        label=rf"$\pm{acceptance_deg:.0f}^\circ$ acceptance",
    )
for laser_line in LASER_LINES_EV:
    ax.axvline(laser_line, color="0.75", linewidth=0.8)
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel(r"reachable $k_{\parallel}$ ($\AA^{-1}$)")
ax.set_title("analyser acceptance at W = 4.5 eV")
ax.legend()
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_5_0.png)




```python
binding_window_ev = jnp.where(kinetic_grid > 0.0, kinetic_grid, 0.0)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(photon_grid_ev, binding_window_ev, color="tab:blue")
for laser_line in LASER_LINES_EV:
    ax.axvline(laser_line, color="0.75", linewidth=0.8)
    ax.plot(
        laser_line,
        laser_line - WORK_FUNCTION_EV,
        marker="o",
        color="tab:red",
    )
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel("accessible binding window (eV)")
ax.set_title("binding-energy depth per photon energy")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_6_0.png)



## Load the Anchor Band Paths

`read_outcar` supplies the Fermi level of each calculation. `make_kpath`
and `kpath_arc_length` convert each fractional path into a cumulative
momentum axis in inverse Angstrom, centered at Gamma.


```python
DATA_ROOT = Path("..") / "data" / "DFT"
BI2SE3_DIR = DATA_ROOT / "Bi2Se3" / "6QL" / "Output few bands"
PDTE2_DIR = DATA_ROOT / "PdTe2" / "4ML" / "Output"
bi2se3_fermi_ev = float(
    read_outcar(str(BI2SE3_DIR / "OUTCAR_SCF")).fermi_energy
)
pdte2_fermi_ev = float(
    read_outcar(
        str(PDTE2_DIR / "MGM" / "OAM DATA" / "OUTCAR")
    ).fermi_energy
)
bi2se3_geo = read_poscar(str(BI2SE3_DIR / "POSCAR"))
pdte2_geo = read_poscar(
    str(DATA_ROOT / "PdTe2" / "4ML" / "PdTe2_4ML_0x_0y_0z.vasp")
)
bi2se3_bands = read_eigenval(
    str(BI2SE3_DIR / "MGM" / "EIGENVAL"), fermi_energy=bi2se3_fermi_ev
)
pdte2_mgm = read_eigenval(
    str(PDTE2_DIR / "MGM" / "EIGENVAL"), fermi_energy=pdte2_fermi_ev
)
pdte2_kgk = read_eigenval(
    str(PDTE2_DIR / "KGK" / "EIGENVAL"), fermi_energy=pdte2_fermi_ev
)
bi2se3_dist = kpath_arc_length(
    make_kpath(bi2se3_bands.kpoints), bi2se3_geo
)
bi2se3_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(bi2se3_bands.kpoints), axis=1))
)
bi2se3_axis = np.asarray(bi2se3_dist - bi2se3_dist[bi2se3_gamma])
bi2se3_shift = np.asarray(bi2se3_bands.eigenvalues) - bi2se3_fermi_ev
pdte2_mgm_dist = kpath_arc_length(
    make_kpath(pdte2_mgm.kpoints), pdte2_geo
)
pdte2_mgm_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(pdte2_mgm.kpoints), axis=1))
)
pdte2_mgm_axis = np.asarray(
    pdte2_mgm_dist - pdte2_mgm_dist[pdte2_mgm_gamma]
)
pdte2_mgm_shift = np.asarray(pdte2_mgm.eigenvalues) - pdte2_fermi_ev
pdte2_kgk_dist = kpath_arc_length(
    make_kpath(pdte2_kgk.kpoints), pdte2_geo
)
pdte2_kgk_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(pdte2_kgk.kpoints), axis=1))
)
pdte2_kgk_axis = np.asarray(
    pdte2_kgk_dist - pdte2_kgk_dist[pdte2_kgk_gamma]
)
pdte2_kgk_shift = np.asarray(pdte2_kgk.eigenvalues) - pdte2_fermi_ev
print("Bi2Se3 Fermi energy (eV):", bi2se3_fermi_ev)
print("PdTe2 Fermi energy (eV):", pdte2_fermi_ev)
```

    Bi2Se3 Fermi energy (eV): 0.0178
    PdTe2 Fermi energy (eV): 2.1733


## Overlay the Horizon on the Bands

Each curve bounds the states one laser line reaches. The boundary at
each energy comes from the same kinematics calls. States outside the
curves stay dark at that photon energy.


```python
overlay_energy_ev = jnp.linspace(-1.5, 0.0, 301)
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(bi2se3_axis, bi2se3_shift, color="0.6", linewidth=0.5)
for laser_line in LASER_LINES_EV:
    kinetic_line, _ = kinetic_energy_ev(
        float(laser_line), WORK_FUNCTION_EV, overlay_energy_ev
    )
    boundary, _ = final_state_k_inv_ang(kinetic_line)
    ax.plot(boundary, overlay_energy_ev, label=f"{laser_line} eV")
    ax.plot(-boundary, overlay_energy_ev, color=ax.lines[-1].get_color())
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-1.5, 0.4)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Bi2Se3 surface state inside the laser horizons")
ax.legend(loc="lower right")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_10_0.png)




```python
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(pdte2_mgm_axis, pdte2_mgm_shift, color="0.6", linewidth=0.5)
for laser_line in LASER_LINES_EV:
    kinetic_line, _ = kinetic_energy_ev(
        float(laser_line), WORK_FUNCTION_EV, overlay_energy_ev
    )
    boundary, _ = final_state_k_inv_ang(kinetic_line)
    ax.plot(boundary, overlay_energy_ev, label=f"{laser_line} eV")
    ax.plot(-boundary, overlay_energy_ev, color=ax.lines[-1].get_color())
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-1.5, 0.4)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("PdTe2 near-Gamma states inside the laser horizons")
ax.legend(loc="lower right")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_11_0.png)




```python
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.plot(pdte2_kgk_axis, pdte2_kgk_shift, color="0.6", linewidth=0.4)
for laser_line in LASER_LINES_EV:
    kinetic_line, _ = kinetic_energy_ev(
        float(laser_line), WORK_FUNCTION_EV, overlay_energy_ev
    )
    boundary, _ = final_state_k_inv_ang(kinetic_line)
    ax.plot(boundary, overlay_energy_ev, label=f"{laser_line} eV")
    ax.plot(-boundary, overlay_energy_ev, color=ax.lines[-1].get_color())
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-2.2, 0.4)
ax.set_xlabel(r"$k - k_\Gamma$ along K--$\Gamma$--K ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("PdTe2 full retained path against the laser horizons")
ax.legend(loc="lower right")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_12_0.png)



## Summarize the Reachable Path Fraction

The bars count the path points inside the horizon at 0.3 eV binding
energy. The 10.8 eV line covers most of both retained paths. The
6.05 eV line keeps a narrow cone around Gamma.


```python
probe_binding_ev = jnp.asarray(-0.3)
bar_offsets = np.arange(LASER_LINES_EV.shape[0], dtype=np.float64)
bi2se3_fractions = []
pdte2_fractions = []
for laser_line in LASER_LINES_EV:
    kinetic_probe, _ = kinetic_energy_ev(
        float(laser_line), WORK_FUNCTION_EV, probe_binding_ev
    )
    probe_horizon, _ = final_state_k_inv_ang(kinetic_probe)
    bi2se3_fractions.append(
        float(np.mean(np.abs(bi2se3_axis) <= float(probe_horizon)))
    )
    pdte2_fractions.append(
        float(np.mean(np.abs(pdte2_kgk_axis) <= float(probe_horizon)))
    )
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.bar(
    bar_offsets - 0.18,
    bi2se3_fractions,
    width=0.36,
    label="Bi2Se3 M--Gamma--M",
)
ax.bar(
    bar_offsets + 0.18,
    pdte2_fractions,
    width=0.36,
    label="PdTe2 K--Gamma--K",
)
ax.set_xticks(bar_offsets)
ax.set_xticklabels([f"{value} eV" for value in LASER_LINES_EV])
ax.set_ylabel("reachable fraction of the path")
ax.set_title("path coverage at 0.3 eV binding energy")
ax.legend()
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_14_0.png)



## Calibrate a Dirac Cone to the Surface State

A linear fit through the surface-state branch gives the Dirac velocity.
A two-site honeycomb model reproduces that velocity through one hopping
amplitude. Its Dirac point sits at the corner of the model zone.


```python
gamma_column = bi2se3_shift[bi2se3_gamma]
surface_band = int(np.argmin(np.abs(gamma_column + 0.05)))
fit_mask = (np.abs(bi2se3_axis) > 0.02) & (np.abs(bi2se3_axis) < 0.12)
fit_slope, fit_intercept = np.polyfit(
    np.abs(bi2se3_axis[fit_mask]),
    bi2se3_shift[fit_mask, surface_band],
    1,
)
dirac_velocity_ev_ang = float(abs(fit_slope))
dirac_energy_ev = float(gamma_column[surface_band])
lattice_constant_ang = 2.0
hopping_ev = (
    2.0 * dirac_velocity_ev_ang / (np.sqrt(3.0) * lattice_constant_ang)
)
print("surface band index:", surface_band)
print("Dirac velocity (eV Ang):", round(dirac_velocity_ev_ang, 3))
print("Dirac point (eV):", round(dirac_energy_ev, 3))
print("honeycomb hopping (eV):", round(hopping_ev, 3))
```

    surface band index: 172
    Dirac velocity (eV Ang): 2.682
    Dirac point (eV): -0.023
    honeycomb hopping (eV): 1.549



```python
lattice = jnp.asarray(
    [
        [lattice_constant_ang, 0.0, 0.0],
        [
            lattice_constant_ang / 2.0,
            lattice_constant_ang * jnp.sqrt(3.0) / 2.0,
            0.0,
        ],
        [0.0, 0.0, 20.0],
    ]
)
crystal = make_crystal_geometry(
    lattice=lattice,
    positions=jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]
    ),
    species=("X", "X"),
)
basis = make_orbital_basis(
    atom_indices=(0, 1),
    n=(1, 1),
    l=(0, 0),
    m=(0, 0),
    labels=("1s", "2s"),
)
model = make_tb_model(
    hopping_amplitudes=hopping_ev * jnp.ones(6, dtype=jnp.complex128),
    onsite_energies=jnp.zeros(2),
    soc_lambdas=jnp.zeros(0),
    geometry=crystal,
    basis=basis,
    hopping_pairs=((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)),
    hopping_cells=(
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
    ),
    shell_index=(-1, -1),
    depths=jnp.zeros(2),
)
dirac_frac = jnp.asarray([1.0 / 3.0, 2.0 / 3.0, 0.0])
dirac_cart = np.asarray(dirac_frac @ crystal.reciprocal)
path_direction = dirac_cart / np.linalg.norm(dirac_cart)
cone_axis = np.linspace(-0.30, 0.30, 181)
cone_cart = dirac_cart[None, :] + cone_axis[:, None] * path_direction
cone_frac = kpoints_cart_to_frac(jnp.asarray(cone_cart), crystal)
cone_hamiltonians = bloch_hamiltonian_batch(model, cone_frac)
cone_bands = diagonalize_tb(model, cone_frac)
print("cone Hamiltonians:", cone_hamiltonians.shape)
```

    cone Hamiltonians: (181, 2, 2)



```python
cone_energies = np.asarray(cone_bands.eigenvalues) + dirac_energy_ev
fig, ax = plt.subplots(figsize=(6.2, 4.8))
ax.plot(bi2se3_axis, bi2se3_shift, color="0.75", linewidth=0.5)
ax.plot(cone_axis, cone_energies[:, 0], color="tab:orange", linewidth=1.6)
ax.plot(
    cone_axis,
    cone_energies[:, 1],
    color="tab:orange",
    linewidth=1.6,
    label="calibrated cone",
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-1.0, 0.6)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("calibrated cone over the DFT surface state")
ax.legend(loc="lower right")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_18_0.png)



## Simulate the Cone Spectrum

Unit transition sources enter the resolvent assembly. A 20 meV constant
self-energy sets the linewidth. The Fermi level sits above the Dirac
point by the fitted offset, and the temperature is 100 K.


```python
spectrum_energy_ev = jnp.linspace(-0.8, 0.25, 211)
cone_self_energy = make_self_energy_model(gamma=0.02)
cone_sources = jnp.ones(
    (cone_axis.shape[0], spectrum_energy_ev.shape[0], 1, 2),
    dtype=jnp.complex128,
)
cone_intensity = assemble_spectral_intensity_chunk(
    cone_hamiltonians,
    cone_sources,
    spectrum_energy_ev,
    cone_self_energy,
    jnp.asarray(-dirac_energy_ev),
    100.0,
)
cone_image = np.asarray(cone_intensity).T
fig, ax = plt.subplots(figsize=(6.2, 4.8))
image = ax.imshow(
    cone_image,
    origin="lower",
    aspect="auto",
    extent=(
        float(cone_axis[0]),
        float(cone_axis[-1]),
        float(spectrum_energy_ev[0]),
        float(spectrum_energy_ev[-1]),
    ),
    cmap="magma",
)
ax.set_xlabel(r"$k - k_D$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("simulated cone spectral intensity")
fig.colorbar(image, ax=ax, label="spectral intensity")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_20_0.png)



## Mask the Spectrum with the Laser Horizon

The 6.05 eV panel keeps the narrow cone that a fixed-wavelength laser
sees. The 10.8 eV panel restores most of the momentum range. The dashed
curves trace the horizon from the kinematics calls.


```python
energy_grid = np.asarray(spectrum_energy_ev)
momentum_grid = cone_axis
low_kinetic, _ = kinetic_energy_ev(
    6.05, WORK_FUNCTION_EV, spectrum_energy_ev
)
low_boundary, _ = final_state_k_inv_ang(low_kinetic)
low_boundary = np.asarray(low_boundary)
low_mask = (
    np.abs(momentum_grid[None, :]) <= low_boundary[:, None]
).astype(np.float64)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
image = ax.imshow(
    cone_image * low_mask,
    origin="lower",
    aspect="auto",
    extent=(
        float(momentum_grid[0]),
        float(momentum_grid[-1]),
        float(energy_grid[0]),
        float(energy_grid[-1]),
    ),
    cmap="magma",
)
ax.plot(low_boundary, energy_grid, color="w", linestyle="--", linewidth=1.0)
ax.plot(
    -low_boundary, energy_grid, color="w", linestyle="--", linewidth=1.0
)
ax.set_xlabel(r"$k - k_D$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("simulated spectrum inside the 6.05 eV horizon")
fig.colorbar(image, ax=ax, label="spectral intensity")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_22_0.png)




```python
high_kinetic, _ = kinetic_energy_ev(
    10.8, WORK_FUNCTION_EV, spectrum_energy_ev
)
high_boundary, _ = final_state_k_inv_ang(high_kinetic)
high_boundary = np.asarray(high_boundary)
high_mask = (
    np.abs(momentum_grid[None, :]) <= high_boundary[:, None]
).astype(np.float64)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
image = ax.imshow(
    cone_image * high_mask,
    origin="lower",
    aspect="auto",
    extent=(
        float(momentum_grid[0]),
        float(momentum_grid[-1]),
        float(energy_grid[0]),
        float(energy_grid[-1]),
    ),
    cmap="magma",
)
ax.plot(
    high_boundary, energy_grid, color="w", linestyle="--", linewidth=1.0
)
ax.plot(
    -high_boundary, energy_grid, color="w", linestyle="--", linewidth=1.0
)
ax.set_xlabel(r"$k - k_D$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("simulated spectrum inside the 10.8 eV horizon")
fig.colorbar(image, ax=ax, label="spectral intensity")
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_23_0.png)



## Cut the Simulated Spectrum

The momentum cut sits at 0.15 eV binding energy. The energy cut sits on
the cone branch at 0.10 inverse Angstrom. The masked curve drops to zero
where the 6.05 eV horizon ends.


```python
cut_row = int(np.argmin(np.abs(energy_grid + 0.15)))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(momentum_grid, cone_image[cut_row], label="full cone")
ax.plot(
    momentum_grid,
    (cone_image * low_mask)[cut_row],
    label="6.05 eV horizon",
)
ax.set_xlabel(r"$k - k_D$ ($\AA^{-1}$)")
ax.set_ylabel("spectral intensity")
ax.set_title("momentum cut at 0.15 eV binding energy")
ax.legend()
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_25_0.png)




```python
cut_column = int(np.argmin(np.abs(momentum_grid - 0.10)))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(energy_grid, cone_image[:, cut_column], label="full cone")
ax.plot(
    energy_grid,
    (cone_image * low_mask)[:, cut_column],
    label="6.05 eV horizon",
)
ax.axvline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$E - E_F$ (eV)")
ax.set_ylabel("spectral intensity")
ax.set_title("energy cut at 0.10 inverse Angstrom")
ax.legend()
plt.show()
```



![png](90-laser-window-audit_files/90-laser-window-audit_26_0.png)
