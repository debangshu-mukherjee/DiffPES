# kz Broadening and Photon-Energy Scans

Finite photoelectron escape depth makes bulk ARPES an incoherent average over
out-of-plane momentum. diffpes evaluates that average explicitly and keeps its
mean-free-path and inner-potential dependence inside the differentiable
forward model. This bulk construction is an alternative to a coherent slab
depth sum; the two escape-depth descriptions are never combined.

## Finite-energy center and wrapped line shape

For relative energy $\omega=E-E_F$, the free-electron final-state center is

$$
k_z^0(k_\parallel,h\nu,V_0,W,\omega) =
\sqrt{\frac{2m}{\hbar^2}(h\nu-W+\omega+V_0)-|k_\parallel|^2}.
$$

Production spectra evaluate this expression at every sampled $\omega$ through
{func}`~diffpes.simul.kz_from_inner_potential`. The separately named
{func}`~diffpes.simul.kz_from_inner_potential_at_fermi` approximation is not
broadcast over a finite energy window.

An intensity escape length $\lambda$ gives the Lorentzian half-width
$\gamma=1/(2\lambda)$. Consider a primitive reciprocal shift of the *normal
integration coordinate* at fixed detected $k_\parallel$ and outgoing $k_f$.
The bulk Hamiltonian, spectral operator, and transition source co-transform.
The complete quadratic integrand $s^\dagger A s$ is therefore invariant, so
diffpes analytically wraps the infinite Lorentzian over one surface-normal
reciprocal period $G_\perp$. In the surface-fractional coordinate $u$, its
dimensionless width is $\gamma_u=\gamma/G_\perp$ and its density is

$$
P_1(u;c,\gamma_u) =
\frac{\sinh(2\pi\gamma_u)}
{\cosh(2\pi\gamma_u)-\cos(2\pi(u-c))}.
$$

{func}`~diffpes.simul.kz_wrapped_lorentzian_bin_weights` integrates this
density analytically over fixed fractional bins. The positive masses sum to
one without truncating images, cropping a tail, or renormalizing a finite
window. {func}`~diffpes.simul.broaden_kz` then applies those masses as an
incoherent reduction.

This is the free-electron-final-state escape-depth construction described by
Strocov (J. Electron Spectrosc. Relat. Phenom. 130, 65, 2003) and by
Damascelli, Hussain, and Shen (Rev. Mod. Phys. 75, 473, 2003, section II.B).
Chinook provides useful compatibility evidence only for its single-kz
kinematics: it chooses one $k_z$ per $(k_\parallel,h\nu)$ and uses escape depth
for amplitude attenuation. It cannot validate this finite-width wrapped
integral or its additional $\partial I/\partial\lambda$ and
$\partial I/\partial V_0$ paths.

The surface mapping uses a complete {class}`~diffpes.types.SurfaceCell`.
Nodes are dimensionless centres in the third surface-fractional coordinate,
not scalar values appended to bulk fractional $(k_x,k_y)$. This distinction
preserves the lateral coupling of an oblique stacking vector and the primitive
reciprocal period. This normal-coordinate covariance does not assert that a
move to a neighboring *detected surface zone* is periodic. That move changes
physical $k_\parallel$ and $k_f$, so radial and spherical-harmonic
matrix-element factors retain genuine repeated-zone contrast.

The Plan 08b G6 calibration tested candidate counts
$\{32,64,128,256,512,1024,2048\}$ against node doubling and a 4096-node small
fixture. The smallest count meeting the registered pointwise/count
$10^{-5}$, integrated-count $10^{-6}$, and gradient $10^{-4}$ budgets was
`n_kz=2048`; the independent wrapped-reference remainder was below
$10^{-12}$. Use

```python
from diffpes.simul import kz_fractional_nodes

kz_nodes_frac = kz_fractional_nodes(2048)
```

for inputs inside that frozen calibration profile. There is no silent public
node-count default. A different physical profile requires an explicitly
recalibrated count.

## Four mutually exclusive driver modes

The `kz_mode` argument selects one physical route. Invalid or mixed carrier
combinations fail at the eager boundary.

| Mode | Native Hamiltonian/bands | Bulk model | Surface cell | kz nodes | Meaning |
|---|---|---|---|---|---|
| `native_direct` | Required | Absent | Absent | Absent | Retains Plan 08a's already-diagonalized single-kz route exactly. |
| `bulk_direct` | Empty/absent | Required | Required | Absent | Evaluates the bulk model once at the exact finite-$\omega$ center; this is the analytic $\gamma=0$ path. |
| `bulk_kz` | Empty/absent | Required | Required | At least two registered centres | Streams a finite-width wrapped bulk integral; matrix-element depth attenuation is disabled. |
| `coherent_slab` | Required, with depth-bearing bands | Absent | Required | Absent | Evaluates one coherent slab-depth amplitude sum at exact finite-$\omega$ final momentum; wrapped broadening is disabled. |

For {func}`~diffpes.simul.simulate_arpes` and
{func}`~diffpes.simul.simulate_arpes_cut`, bulk and surface inputs are
per-domain tuples. The new inputs are keyword-only; every Plan 08a positional
argument, the explicit {class}`~diffpes.types.DetectorCalibration`, and the
detector-space domain mixture remain unchanged. A bulk call therefore uses
empty native tuples:

```python
from diffpes.simul import simulate_arpes

raster = simulate_arpes(
    (),
    (),
    radial_spec,
    matrix_element_params,
    radial_quadrature,
    final_state,
    geometry,
    self_energy,
    kgrid,
    energy_axis,
    detector_calibration,
    detector_effects,
    bulk_models_by_domain=(bulk_model,),
    surface_cells_by_domain=(surface_cell,),
    kz_nodes_frac=kz_nodes_frac,
    kz_mode="bulk_kz",
)
```

The complete physical ordering is

$$
|\hat\epsilon\!\cdot\!M|^2
\rightarrow A(k,\omega)
\rightarrow f_{FD}(\omega;T)
\rightarrow \text{kz integral}
\rightarrow \widetilde T(E_{kin})
\rightarrow \text{native detector resolution}
\rightarrow \text{expected counts}.
$$

Domain rotation and conservative mapping occur before the detector-space
mixture. Transmission acts at true kinetic energy before resolution. Display
normalization and random count sampling remain outside the deterministic
driver.

## Photon-energy scans

{func}`~diffpes.simul.simulate_hv_scan` is the single-domain, pre-detector
primitive. It returns an array with shape `[n_hv, n_k, n_e]` and accepts the
same four mutually exclusive source modes using singular `bulk_model` and
`surface_cell` arguments. It deliberately accepts no transmission,
resolution, detector calibration, count sampling, or display normalization.

```python
from diffpes.simul import hv_map_at_energy, simulate_hv_scan

scan = simulate_hv_scan(
    None,
    None,
    radial_spec,
    matrix_element_params,
    radial_quadrature,
    final_state,
    geometry,
    self_energy,
    kpath,
    energy_axis,
    photon_energies_ev,
    bulk_model=bulk_model,
    surface_cell=surface_cell,
    kz_nodes_frac=kz_nodes_frac,
    kz_mode="bulk_kz",
)

map_k_hv = hv_map_at_energy(scan, energy_axis, energy_ev=-0.15)
```

{func}`~diffpes.simul.hv_map_at_energy` linearly interpolates only along the
sampled energy axis and returns `[n_k, n_hv]` for plotting. The query must lie
inside that axis.

The photon-energy axis is a `jax.lax.scan`, not a Python loop or a
five-point interpolation. Each row recomputes the exact finite-$\omega$
kinematics and photon-energy-dependent matrix elements. Photon-energy values,
$V_0$, $W$, $\omega$, and the additional bulk-kz $\lambda$ path remain
differentiable.

## Memory contract and physical interpretation

Bulk integration is node-local. The production graph may hold one node's
kinematics, Hamiltonian, sources, and a single `[K, E]` accumulator. It must
not materialize a complete `[K, B, E, n_kz]`, `[K, B, B, n_kz]`,
`[K, E, n_kz]`, or `[K, n_kz, 3]` carrier. Checkpointed `lax.scan` keeps
reverse-mode storage flat in `n_kz`. The hν scan adds no growing auxiliary
storage beyond its required returned scan.

Physically, the explicit Lorentzian integral is the free-electron-final-state
limit of a damped complex-$k_z$ treatment. Its linewidth is the observable
shadow of the finite escape depth. It is more than a single-kz compatibility
calculation, but it is not a full damped atomic radial matrix element. That
candidate remains outside the current mode set and public API.

See [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) for the
exact final-state branch and [Coherent Spectral
Assembly](simulation-levels.md) for the resolvent and detector boundaries.
