---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Complete-Group Sensitivity at a Dark Corridor

This executable example isolates the semantics of
`band_group_weight_sensitivity`. A one-parameter amplitude has a destructive
interference zero. We differentiate complete isolated band-group weights and
then apply the logarithmic validity mask.

The synthetic callback stands in for the longer
`unpack_matrixel_params` → channel assembly → band projection → polarization
contraction pipeline. Its required output shape is `[K, B, S]`.

```{code-cell} ipython3
import diffpes
import jax.numpy as jnp

geometry = diffpes.types.make_crystal_geometry(
    lattice=jnp.eye(3) * 5.0,
    positions=jnp.array([[0.0, 0.0, 0.0]]),
    species=("X",),
)
basis = diffpes.types.make_orbital_basis(
    atom_indices=(0, 0),
    n=(2, 2),
    l=(0, 1),
    m=(0, 0),
)
kpoints = jnp.array(
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
)
eigenvalues = jnp.array(
    [[-1.0, 1.0], [-0.9, 1.1], [-0.8, 1.2]]
)
eigenvectors = jnp.broadcast_to(
    jnp.eye(2, dtype=jnp.complex128),
    (kpoints.shape[0], 2, 2),
)
bands = diffpes.types.make_diagonalized_bands(
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    kpoints=kpoints,
    geometry=geometry,
    basis=basis,
)
experiment = diffpes.types.make_experiment_geometry(
    photon_energy_ev=21.2,
    polarization=jnp.array([0.0, 1.0, 0.0], dtype=jnp.complex128),
)
```

The first band amplitude is
$M_0(\theta)=e^{i\theta}-e^{-i\theta}=2i\sin\theta$. It is dark at
$\theta=0$. The second band is a bright reference. Each singleton group is
complete and remains separated from its complement by more than the required
energy gap.

```{code-cell} ipython3
def rebuild(candidate, bands, experiment):
    del experiment
    angle = candidate[0]
    dark_amplitude = jnp.exp(1j * angle) - jnp.exp(-1j * angle)
    bright_amplitude = jnp.asarray(1.0 + 0.0j)
    one_row = jnp.stack([dark_amplitude, bright_amplitude])
    return jnp.broadcast_to(
        one_row[None, :, None],
        (bands.eigenvalues.shape[0], 2, 1),
    )


theta = jnp.array([0.0])
weights, dweights = diffpes.simul.band_group_weight_sensitivity(
    theta,
    rebuild,
    bands,
    experiment,
    band_groups=((0,), (1,)),
)
print("weights at the corridor:")
print(weights)
print("dw/dtheta shape:", dweights.shape)
```

`weights` has shape `[K, group]`; `dweights` has shape
`[theta, K, group]`. These are matrix-element weights and derivatives. They
do not include occupation, spectral broadening, exposure, background, or
detector response.

## Logarithmic Derivatives Need a Mask

$d\log w/d\theta=(dw/d\theta)/w$ has no value at the dark corridor. The
helper returns a zero sentinel there and a false validity mask. Downstream
code must use the mask.

```{code-cell} ipython3
log_derivative, valid = (
    diffpes.simul.log_band_group_weight_sensitivity(
        weights,
        dweights,
        min_band_group_weight=1.0e-12,
    )
)
print("valid at theta=0:")
print(valid)
print("masked log derivative:")
print(log_derivative)
assert not bool(jnp.any(valid[:, 0]))
assert bool(jnp.all(valid[:, 1]))
```

Move away from the corridor and the first group becomes valid. Its analytic
weight is $4\sin^2\theta$, so
$d\log w/d\theta=2\cot\theta$.

```{code-cell} ipython3
theta_lit = jnp.array([0.2])
weights_lit, dweights_lit = (
    diffpes.simul.band_group_weight_sensitivity(
        theta_lit,
        rebuild,
        bands,
        experiment,
        band_groups=((0,), (1,)),
    )
)
log_lit, valid_lit = diffpes.simul.log_band_group_weight_sensitivity(
    weights_lit,
    dweights_lit,
    min_band_group_weight=1.0e-12,
)
expected = 2.0 / jnp.tan(theta_lit[0])
print(float(log_lit[0, 0, 0]), float(expected))
assert bool(jnp.all(valid_lit))
assert jnp.allclose(log_lit[0, :, 0], expected)
```

For a degenerate multiplet, pass the entire multiplet as one tuple. A partial
group is basis-gauge dependent and is rejected before differentiation.

`expected_counts` can convert an already mapped detector density into native
bin counts. These `weights`, `dweights`, and `log_derivative` names still
retain their literal matrix-element meaning.
