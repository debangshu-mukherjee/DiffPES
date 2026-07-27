# JAX Transformability and Gradients

diffpes numerical kernels use JAX arrays and fixed-shape PyTrees. Continuous
physical inputs support `jit`, `vmap`, forward- and reverse-mode automatic
differentiation. Static selectors such as radial mode, basis quantum numbers,
and band-group membership select compiled program structure.

## Incoherent Tier Gradients

The two expanded tiers differentiate through occupation, broadening, and
their orbital-weight reduction:

```python
def total_novice_weight(sigma):
    spectrum = diffpes.simul.simulate_expanded(
        level="novice",
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        sigma=sigma,
        gamma=0.08,
        fidelity=1000,
    )
    return jnp.sum(spectrum.intensity)

d_total_d_sigma = jax.grad(total_novice_weight)(0.04)
```

For `level="basic"`, derivatives also pass through the Yeh--Lindau
interpolator with respect to photon energy inside a valid table segment.
Table boundaries, gaps, element identity, and subshell identity are domain
or static choices, not smooth variables.

Both tiers consume projection probabilities. Their gradients do not recover
the discarded relative phase of orbital coefficients.

## Coherent Matrix-Element Coordinates

Use `pack_matrixel_params` to obtain a real optimizer vector:

```python
theta, tree_definition, packing_metadata = (
    diffpes.simul.pack_matrixel_params(
        radial,
        me_params,
        experiment.mean_free_path_ang,
    )
)

def rebuild(candidate):
    radial_i, params_i, mean_free_path_i = (
        diffpes.simul.unpack_matrixel_params(
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
phase_null = diffpes.simul.matrix_element_phase_gauge_direction(
    radial, me_params, experiment.mean_free_path_ang
)
scale_nulls = diffpes.simul.radial_coefficient_scale_gauge_directions(
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
weights, dweights = diffpes.simul.band_group_weight_sensitivity(
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
counting statistics enter in Plan 08.

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
