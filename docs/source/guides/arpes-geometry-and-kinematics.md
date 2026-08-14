# ARPES Geometry and Kinematics

ARPES measures photoelectron energy and detector angles. Diffpes converts
these measurements into crystal momentum with differentiable JAX functions.

## Experiment Geometry

{class}`~diffpes.types.ExperimentGeometry` stores one beamline and sample
configuration. Every numerical field is a traced JAX leaf. The `slit` value
is static because it selects a detector convention.

```python
import jax.numpy as jnp

from diffpes.simul import polarization_from_angles
from diffpes.types import make_experiment_geometry

polarization = polarization_from_angles(0.75, 0.1, "p")
experiment = make_experiment_geometry(
    photon_energy_ev=50.0,
    polarization=polarization,
    incidence_theta=0.75,
    incidence_phi=0.1,
    sample_azimuth=0.05,
    work_function_ev=4.5,
    inner_potential_ev=12.0,
    slit="H",
)
```

The factory normalizes the polarization vector and verifies that it is
transverse to the declared photon direction. This operation removes its
intensity-scale gauge. The factory rejects invalid inputs during eager and
compiled execution.

## Crystal Coordinates

{class}`~diffpes.types.CrystalGeometry` stores real-space lattice rows in
Angstrom. Its `reciprocal` rows use inverse Angstrom and include $2\pi$.

Diffpes uses one row-vector conversion contract:

$$
\mathbf{k}_{\mathrm{cart}}=\mathbf{k}_{\mathrm{frac}}B,
\qquad
\mathbf{k}_{\mathrm{frac}}
=\frac{\mathbf{k}_{\mathrm{cart}}A^T}{2\pi}.
$$

Use {func}`~diffpes.tightb.kpoints_frac_to_cart` and
{func}`~diffpes.tightb.kpoints_cart_to_frac` for these conversions. Do not
normalize fractional coordinates before conversion. A fractional direction
does not represent a Cartesian direction for a non-orthogonal lattice.

## Paths and Rasters

{func}`~diffpes.tightb.build_kpath` interpolates labeled anchors. It returns a
{class}`~diffpes.types.KPath` with traced fractional coordinates.
{func}`~diffpes.tightb.kpath_arc_length` supplies the Cartesian plotting axis.

{func}`~diffpes.tightb.build_bz_mesh` returns a fixed-size reciprocal mesh and
a first-zone mask. The mask avoids a data-dependent gather under `jit`.
{func}`~diffpes.tightb.first_bz_mask` uses squared Wigner-Seitz distances and a
static reciprocal shell. Its singular-value bound proves that the shell is
complete for the supplied points. Increase `shell_radius` when a skew or
anisotropic basis fails this conservative check. The function raises an error
instead of returning an uncertified mask. The mesh builder also rejects a basis when
reciprocal-vector inequalities cannot prove that its fixed fractional cube
contains the complete first zone. Use a reduced basis in that case.

The hard Boolean membership mask has no boundary derivative. Holding its
current values fixed does not define a lattice-shape derivative for a
mask-weighted observable. Lattice-shape inference therefore requires either a
preregistered fixed mask or a separate smooth shape-derivative scheme.

Two builders create ARPES rasters:

- {func}`~diffpes.tightb.build_arpes_kmesh` creates a fixed-$k_z$ map.
- {func}`~diffpes.tightb.build_kmesh_hv_at_fermi` creates an explicitly
  at-Fermi $(k_\parallel,h\nu)$ parity map.

Both builders rotate laboratory momentum into the sample frame. Their
{class}`~diffpes.types.KGrid` outputs retain static raster shapes.

```{figure} figures/geometry-energy-slices.png
:alt: Constant-energy momentum maps of graphene at three binding energies

Constant-energy $(k_x, k_y)$ maps of a graphene spectral cube at three
binding energies. Rasters like these are what `build_arpes_kmesh`
produces for the spectral assemblers; the pockets grow and warp as the
slice moves below the Fermi level.
```

## Free-Electron Kinematics

The three-step model gives the vacuum kinetic energy:

$$
E_{\mathrm{kin}}=h\nu-W+\omega,\qquad \omega=E-E_F.
$$

{func}`~diffpes.simul.kinetic_energy_ev` returns this signed raw energy and
the explicit mask `Ekin > 0`. It never replaces forbidden emission with a
positive energy.

The final-state magnitude is

$$
k_f=\sqrt{\frac{E_{\mathrm{kin}}}
{\hbar^2/(2m_e)}}.
$$

{func}`~diffpes.simul.final_state_k_inv_ang` returns this expression and its
validity mask. Forbidden inputs receive a zero sentinel and false mask.
DiffPES uses
$\hbar^2/(2m_e)=3.8099821\,\mathrm{eV\,\mathring{A}^2}$.

The free-electron inner-potential model gives

$$
k_z(\omega)=\sqrt{\frac{(h\nu-W+\omega)-
(\hbar^2/2m_e)k_\parallel^2+V_0}
{\hbar^2/(2m_e)}}.
$$

{func}`~diffpes.simul.kz_from_inner_potential` returns complex $k_z$ and a
propagating-channel mask for an explicit $\omega$. A negative radicand
produces an evanescent channel. Forbidden surface emission returns a zero
sentinel and false mask, including when $k_\parallel^2$ exceeds the vacuum
aperture $(2m_e/\hbar^2)E_{\rm kin}$. The separately named
{func}`~diffpes.simul.kz_from_inner_potential_at_fermi` is only the
$\omega=0$ parity approximation.

```{figure} figures/geometry-kz-hv.png
:alt: Probed out-of-plane momentum versus photon energy for several parallel momenta
:width: 74%

The out-of-plane momentum probed at each photon energy from
`kz_from_inner_potential`. Changing $h\nu$ scans $k_z$; larger
$k_\parallel$ lowers the curve.
```

For a propagating channel,

$$
\frac{\partial k_z}{\partial V_0}
=\frac{1}{2(\hbar^2/2m_e)k_z}.
$$

This nonzero derivative supplies the $V_0$ row in an experiment-design
Jacobian.

## Detector Coordinates

The analyzer uses two angles, `tx` and `ty`. Diffpes applies active rotations
to column vectors. The slit convention is

$$
R_H=R_x(t_y)R_y(t_x),
\qquad
R_V=R_x(t_x)R_y(t_y).
$$

{func}`~diffpes.simul.detector_rotation` constructs this frame.
{func}`~diffpes.simul.detector_angles_to_kpar` maps detector angles to
parallel momentum. {func}`~diffpes.simul.kpar_to_detector_angles` provides
the inverse inside the physical disk $|k_\parallel|<k_f$.

{func}`~diffpes.simul.emission_angles` converts Cartesian momentum to polar
and azimuthal angles. Azimuth is undefined at normal emission. The function
returns a guarded value there, but derivative tests exclude that point.

## Polarization Frame

{func}`~diffpes.simul.polarization_from_angles` constructs s, p, circular, or
linear polarization. It returns the complex Cartesian amplitude without an
intensity reduction.

{func}`~diffpes.simul.sample_azimuth_rotation` constructs the active
sample-to-laboratory orientation.
{func}`~diffpes.simul.lab_polarization_to_sample` applies its inverse to the
fixed laboratory photon field. No detector angle enters this transformation,
so the sample-frame polarization is identical at every detector pixel.

{func}`~diffpes.simul.rotate_frame_vectors` has a different role: it maps
detector-fixed real axes, such as analyzer spin axes, through the composed
detector and inverse-sample orientations.

The spherical components use order $(q=-1,0,+1)$:

$$
\epsilon_{-1}=\frac{\epsilon_x-i\epsilon_y}{\sqrt{2}},
\quad
\epsilon_0=\epsilon_z,
\quad
\epsilon_{+1}=-\frac{\epsilon_x+i\epsilon_y}{\sqrt{2}}.
$$

{func}`~diffpes.simul.polarization_to_spherical` performs this complex-linear
transform. It preserves $\sum_q|\epsilon_q|^2$.

## Information Flow

Geometry fields stay inside the JAX graph. A photon-energy scan can therefore
differentiate $k_z$ with respect to $V_0$, $W$, and every photon-energy point.

```python
import jax
import jax.numpy as jnp

from diffpes.simul import kz_from_inner_potential

k_parallel = jnp.linspace(0.0, 1.0, 8)
photon_energy = jnp.linspace(25.0, 100.0, 6)

def scan(parameters):
    v0 = parameters[0]
    work_function = parameters[1]
    energies = parameters[2:]
    kz, _ = jax.vmap(
        lambda energy: kz_from_inner_potential(
            energy,
            work_function,
            v0,
            jnp.asarray(0.0),
            k_parallel,
        )
    )(energies)
    return jnp.real(kz)

parameters = jnp.concatenate((jnp.array([12.0, 4.5]), photon_energy))
jacobian = jax.jacfwd(scan)(parameters)
```

The work function and Fermi-relative energy enter only as
$h\nu-W+\omega$, so $J_W=-J_\omega$ exactly. A photon-energy scan does not
lift this gauge without an external energy or work-function reference. It can
still constrain $V_0$ through the $k_z$ dispersion.

See [Geometry and kinematics](../tutorials/geometry-and-kinematics.md) for a
complete executable example.
