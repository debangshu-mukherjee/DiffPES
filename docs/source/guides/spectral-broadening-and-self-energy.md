# Spectral Broadening and Self-Energy

An ideal band structure contains infinitely sharp delta functions
$\delta(E - E_b(\mathbf{k}))$. Several effects reshape a measured ARPES
spectrum. These effects include instrumental resolution, finite quasiparticle
lifetime, thermal occupation, and finite angular acceptance. The
{mod}`diffpes.simul` package provides a differentiable primitive for each
effect. This guide derives the profiles and defines their width conventions.
It also shows how an energy-dependent self-energy changes the lineshape.

Energy widths use eV, momentum widths use $\text{Å}^{-1}$, and temperature
uses Kelvin. All calculations use float64.

## Gaussian Broadening: Instrumental Resolution

A Gaussian approximates the combined resolution of the beamline and analyzer.
`gaussian(energy_range, center, sigma)` evaluates the unit-area profile

$$
G(E; E_0, \sigma) = \frac{1}{\sqrt{2\pi}\,\sigma}
    \exp\!\left(-\frac{(E - E_0)^2}{2\sigma^2}\right)
$$

**Width convention:** `sigma` is the standard deviation, not the FWHM. The conversion is

$$
\mathrm{FWHM} = 2\sqrt{2\ln 2}\;\sigma \approx 2.355\,\sigma
$$

Thus, a quoted 10 meV FWHM corresponds to `sigma = 0.00425` eV.

```python
import jax.numpy as jnp
from diffpes.simul import gaussian

energy_axis = jnp.linspace(-1.0, 1.0, 2001)
g = gaussian(energy_axis, center=0.0, sigma=0.02)
print(float(g.max()))                       # 19.947 = 1 / (sqrt(2 pi) * 0.02)
print(float(jnp.trapezoid(g, energy_axis)))  # ~1.0 (unit area)
```

The `basic`, `basicplus`, and `advanced` levels convolve each band delta
function with $G$.

## Lorentzian Lifetime and the Pseudo-Voigt Profile

A quasiparticle with finite lifetime $\tau$ has a Lorentzian spectral function.
Its half-width is $\gamma = \hbar / (2\tau)$. The physical lineshape is a
**Voigt profile**. It convolves the intrinsic Lorentzian with the instrumental
Gaussian. A true Voigt has no closed form. Therefore,
`voigt(energy_range, center, sigma, gamma)` uses the Thompson-Cox-Hastings
(1987) pseudo-Voigt approximation. Its relative error is less than 1%:

$$
V(E) \approx \eta\, L(E; \gamma_V) + (1 - \eta)\, G(E; \sigma_V)
$$

The empirical quintic computes the effective Voigt FWHM $f_V$. Its inputs are
$f_G = 2\sqrt{2\ln 2}\,\sigma$ and $f_L = 2\gamma$:

$$
f_V = \left( f_G^5 + 2.69269 f_G^4 f_L + 2.42843 f_G^3 f_L^2
      + 4.47163 f_G^2 f_L^3 + 0.07842 f_G f_L^4 + f_L^5 \right)^{1/5}
$$

The mixing ratio is
$\eta = 1.36603\,\rho - 0.47719\,\rho^2 + 0.11116\,\rho^3$. Here,
$\rho = f_L / f_V$, and the implementation clips $\rho$ to $[0, 1]$.

**Width convention:** `gamma` is the Lorentzian **half**-width at half-maximum (HWHM).

```python
from diffpes.simul import voigt

v = voigt(energy_axis, center=0.0, sigma=0.02, gamma=0.05)
print(float(v.max()))  # 5.646 - broader and lower than the pure Gaussian
```

The `novice`, `expert`, and `soc` levels use this profile. The
`simulate_tb_radial` model also uses it with constant `params.sigma` and
`params.gamma`.

## Fermi-Dirac Occupation

Only occupied states photoemit. diffpes multiplies each band contribution by
the Fermi-Dirac factor at electronic temperature $T$:

$$
f(E) = \frac{1}{1 + e^{(E - E_F)/k_B T}}, \qquad k_B = 8.617 \times 10^{-5}\ \mathrm{eV/K}
$$

```python
from diffpes.simul import fermi_dirac

occ = fermi_dirac(energy=0.01, fermi_energy=0.0, temperature=20.0)
print(float(occ))  # 0.003 - 10 meV above E_F at 20 K is essentially empty
```

The thermal energy $k_B T$ is 1.3 meV at 15 K. It is 25 meV at room
temperature. At low positive temperature, `sigma` usually controls the
observed Fermi edge width. The finite-temperature function rejects
nonpositive or non-finite temperature. Diffpes defines no implicit numerical
zero-temperature step.

diffpes applies $f$ at each **band center** $E_b(\mathbf{k})$. It does not
multiply the final spectrum pointwise. This difference matters only for peaks
within a few widths of $E_F$.

## Momentum Broadening

Finite angular acceptance and the photon spot size broaden the spectrum along
$k$. `apply_momentum_broadening(intensity, k_distances, dk)` convolves each
energy column with a Gaussian. The input intensity has shape `(K, E)`, and
`dk` is the standard deviation in $\text{Å}^{-1}$. A row-normalized dense
kernel performs the convolution:

$$
I'(k_i, E) = \frac{\sum_j e^{-(k_i - k_j)^2 / 2\,\delta k^2}\; I(k_j, E)}
                  {\sum_j e^{-(k_i - k_j)^2 / 2\,\delta k^2}}
$$

Row normalization conserves spectral weight for nonuniform k-point spacing.
The `k_distances` values contain cumulative distances along the k-path:

```python
import jax.numpy as jnp
import diffpes
from diffpes.simul import apply_momentum_broadening

n_k = 61
k_frac = jnp.linspace(-0.5, 0.5, n_k)
eigenbands = jnp.stack(
    [-1.2 + 0.9 * jnp.cos(2 * jnp.pi * k_frac),
     -2.6 + 0.4 * jnp.cos(2 * jnp.pi * k_frac + jnp.pi)], axis=1)
surface_orb = jnp.ones((n_k, 2, 1, 9)) / 9.0

spectrum = diffpes.simul.simulate_expanded(
    level="novice", eigenbands=eigenbands, surface_orb=surface_orb,
    ef=0.0, sigma=0.03, gamma=0.05, fidelity=2000,
    temperature=20.0, photon_energy=21.2,
)

k_distances = jnp.linspace(0.0, 1.2, n_k)   # cumulative path length (1/Angstrom)
blurred = apply_momentum_broadening(spectrum.intensity, k_distances, dk=0.03)
print(blurred.shape)  # (61, 2000)
```

Typical experimental `dk` values are 0.01-0.05 $\text{Å}^{-1}$. JAX
traces the complete kernel. Therefore, `jax.grad` can differentiate with
respect to `dk`. The fitting model can use momentum resolution as a parameter.

## Energy-Dependent Self-Energy $\Gamma(E)$

A constant `gamma` is an approximate model. In real materials, the imaginary
electron self-energy depends on energy. Therefore, the Lorentzian width also
depends on energy:

$$
\Gamma(E) = -\,\mathrm{Im}\,\Sigma(E), \qquad
A(\mathbf{k}, E) \sim \frac{1}{\pi}\,
\frac{\Gamma(E)}{\left(E - E_b(\mathbf{k})\right)^2 + \Gamma(E)^2}
$$

`evaluate_self_energy(energy, config)` evaluates $\Gamma(E)$ from a
{class}`~diffpes.types.SelfEnergyConfig`. The configuration selects one of
three modes:

| `mode` | Model | Typical use |
|--------|-------|-------------|
| `"constant"` | $\Gamma(E) = c_0$ | Equivalent to a fixed `params.gamma` |
| `"polynomial"` | $\Gamma(E) = \sum_n c_n E^n$ (coefficients highest-degree first, as in `jnp.polyval`) | Fermi-liquid $\Gamma = \Gamma_0 + \beta (E - E_F)^2$ |
| `"tabulated"` | Piecewise-linear interpolation of $(\varepsilon_i, \Gamma_i)$ nodes | Self-energies from GW or DMFT |

```python
import jax.numpy as jnp
from diffpes.simul import evaluate_self_energy
from diffpes.types import make_self_energy_config

# Fermi-liquid-like: Gamma(E) = 0.5 E^2 + 0.02 (eV), E measured from E_F
se_cfg = make_self_energy_config(
    mode="polynomial",
    coefficients=jnp.array([0.5, 0.0, 0.02]),  # [E^2, E^1, E^0]
)
energy_axis = jnp.linspace(-1.0, 1.0, 2001)
gamma_e = evaluate_self_energy(energy_axis, se_cfg)
print(float(gamma_e[1000]))  # 0.02 at E = 0 (sharp quasiparticles at E_F)
print(float(gamma_e[0]))     # 0.52 at E = -1 eV (short-lived deep states)

# Tabulated mode: nodes from a many-body calculation
se_tab = make_self_energy_config(
    mode="tabulated",
    coefficients=jnp.array([0.30, 0.10, 0.02, 0.02]),   # Gamma values (eV)
    energy_nodes=jnp.array([-3.0, -1.0, -0.1, 0.0]),    # energies (eV)
)
```

### What a self-energy does to the lineshape

For an EDC at fixed $\mathbf{k}$, a quasiparticle peak has Lorentzian FWHM
$2\Gamma(E)$. The fixed Gaussian resolution convolves this peak. A
Fermi-liquid model uses $\Gamma \propto (E - E_F)^2$. Its peaks sharpen near
the Fermi level and broaden at higher binding energy. JAX traces the
`coefficients` array. Therefore, `jax.grad` gives each pixel's sensitivity to
the self-energy shape. This sensitivity supports fitting $\Gamma(E)$ to
measured EDC widths.

diffpes does not model the real part $\mathrm{Re}\,\Sigma$. That component
shifts and renormalizes band positions, including kinks and mass enhancement.
diffpes broadens the bare input bands.

## Putting It Together

`simulate_tb_radial` accepts an optional self-energy and optional momentum
broadening. When `self_energy` is present, $\Gamma(E)$ replaces the constant
`params.gamma` at each energy-axis point:

```python
from diffpes.types import (
    make_diagonalized_bands, make_orbital_basis,
    make_polarization_config, make_simulation_params, make_slater_params,
)

basis = make_orbital_basis(n_values=(2, 2, 2), l_values=(1, 1, 1),
                           m_values=(-1, 0, 1))
slater = make_slater_params(zeta=jnp.array([1.6, 1.6, 1.6]), orbital_basis=basis)
eigvecs = jnp.tile(jnp.eye(3, dtype=jnp.complex128)[jnp.newaxis, :2, :], (n_k, 1, 1))
kpts = jnp.stack([k_frac, jnp.zeros(n_k), jnp.zeros(n_k)], axis=1)
diag = make_diagonalized_bands(eigenbands, eigvecs, kpts, fermi_energy=0.0)

spec = diffpes.simul.simulate_tb_radial(
    diag_bands=diag,
    slater_params=slater,
    params=make_simulation_params(energy_min=-3.5, energy_max=0.5,
                                  fidelity=800, sigma=0.03, gamma=0.05),
    pol_config=make_polarization_config(theta=jnp.deg2rad(45.0)),
    work_function=4.3,
    self_energy=se_cfg,   # Gamma(E) replaces the constant gamma
    dk=0.02,              # momentum broadening (1/Angstrom)
)
print(spec.intensity.shape)  # (61, 800)
```

## Summary of Width Conventions

| Parameter | Meaning | Convention |
|-----------|---------|------------|
| `sigma` | Gaussian (instrumental) width | Standard deviation, eV; FWHM $= 2.355\,\sigma$ |
| `gamma` | Lorentzian (lifetime) width | HWHM, eV; Lorentzian FWHM $= 2\gamma$ |
| `dk` | Momentum resolution | Standard deviation, $\text{Å}^{-1}$ |
| `temperature` | Fermi-Dirac electronic temperature | Kelvin; $k_B T$ in eV internally |
| `SelfEnergyConfig.coefficients` | $\Gamma(E)$ model parameters | eV; replaces `gamma` per energy point |

## Further Reading

- [Simulation Levels](simulation-levels.md) identifies the levels that use
  Gaussian and Voigt broadening.
- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) defines the
  $(E, k)$ axes for broadening.
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md)
  describes the weights that precede broadening.
- API reference: {doc}`../api/simul` (`gaussian`, `voigt`, `fermi_dirac`, `apply_momentum_broadening`, `evaluate_self_energy`), {doc}`../api/types` (`make_self_energy_config`, `make_simulation_params`)
