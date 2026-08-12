# JAX Transformability and Gradients

diffpes numerical kernels use JAX arrays and fixed-shape PyTrees. Continuous
physical inputs support `jit`, `vmap`, forward- and reverse-mode automatic
differentiation. Static selectors such as radial mode, basis quantum numbers,
and band-group membership select compiled program structure.

## Coherent Spectral Gradients

The spectral primitives differentiate through the causal self-energy and
gauge-invariant band-weight fast path:

```python
def total_spectral_weight(gamma):
    model = diffpes.types.make_self_energy_model(gamma=gamma)
    sigma = diffpes.simul.evaluate_self_energy(omega, model)
    intensity = jax.vmap(
        lambda energy, sigma_i: diffpes.simul.spectral_intensity_eigen(
            eigenvalues, band_weights, energy, sigma_i, 1.0e-4
        )
    )(omega, sigma)
    return jnp.sum(intensity)

d_total_d_gamma = jax.grad(total_spectral_weight)(0.08)
```

At degeneracy, use `spectral_intensity_resolvent`; it avoids raw eigenvector
derivatives and retains the full transition source.

## Coherent Matrix-Element Coordinates

Use `pack_matrixel_params` to obtain a real optimizer vector:

```python
theta, tree_definition, packing_metadata = (
    diffpes.matrixel.pack_matrixel_params(
        radial,
        me_params,
        experiment.mean_free_path_ang,
    )
)

def rebuild(candidate):
    radial_i, params_i, mean_free_path_i = (
        diffpes.matrixel.unpack_matrixel_params(
            candidate,
            tree_definition,
            packing_metadata,
            radial,
            me_params,
        )
    )
    # Assemble channels with radial_i, params_i, and mean_free_path_i.
    return spin_amplitudes
```

Slater exponents and contraction coefficients, hydrogenic effective charges,
shell scales, physical channel phases, and mean free path are active according
to the radial mode. Grid samples and fixed calibrated integrals remain
outside this inversion view.

## Gauge Directions

Intensity does not identify a common phase on every final-state channel.
Normalized Slater contractions also do not identify a common coefficient
scale for each shell. Obtain the corresponding packed unit tangents with:

```python
phase_null = diffpes.matrixel.matrix_element_phase_gauge_direction(
    radial, me_params, experiment.mean_free_path_ang
)
scale_nulls = diffpes.matrixel.radial_coefficient_scale_gauge_directions(
    radial, me_params, experiment.mean_free_path_ang
)
```

These are expected null directions, not failed derivatives. Optimizers and
information analyses should fix or project them.

## Complete Band-Group Sensitivities

Individual eigenvectors inside a degenerate subspace can rotate without
changing the physics. `band_group_weight_sensitivity` therefore accepts
static, non-overlapping, complete isolated band groups. Its callback returns
complex amplitudes with shape `[K, B, S]`. The helper sums
$\sum_s |M_s|^2$, sums each complete group, and computes
`dw/dtheta` with `jax.jacfwd`.

```python
weights, dweights = diffpes.matrixel.band_group_weight_sensitivity(
    theta,
    rebuild,
    bands,
    experiment,
    band_groups=((0, 1), (2,)),
)
```

`log_band_group_weight_sensitivity` converts these derivatives to
$d\log w/d\theta$ only where `w` exceeds a caller-supplied positive floor.
It returns zero sentinels and `False` in the validity mask for dark or
sub-floor groups. Always carry that mask downstream.

These quantities are band-group matrix-element weights. They are not
expected detector counts. Exposure, background, detector response, and
counting statistics are outside the current detector-independent model.

## Differentiability Boundaries

- File parsing, data-table loading, and Python carrier construction occur
  outside JIT.
- Static mode strings, orbital tuples, shell maps, and group tuples trigger
  retracing when changed.
- An explicit `emission_valid` mask defines the vacuum kinematics domain.
- Yeh--Lindau interpolation does not extrapolate across endpoints or gaps.
- Log sensitivity is undefined at a dark corridor and is represented by a
  false mask, not a finite physical derivative.

The executable
[matrix-element sensitivity tutorial](../tutorials/matrix-element-sensitivity.md)
demonstrates the complete-group and dark-mask behavior.
