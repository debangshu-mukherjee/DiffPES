# Slabs and Surface-Resolved Bands

Construct a finite open slab from a periodic tight-binding model. The example
uses a one-orbital chain because its finite spectrum is known exactly. The
same API accepts a three-dimensional bulk model and a primitive Miller tuple.
This static page records each call and its expected analytic check.

```python
import jax.numpy as jnp

import diffpes as dp
```

The periodic bulk model has one hopping in each normal direction. Exact
integer cells remain separate from the differentiable hopping values.

```python
geometry = dp.types.make_crystal_geometry(
    lattice=jnp.eye(3),
    positions=jnp.zeros((1, 3)),
    species=("X",),
)
basis = dp.types.make_orbital_basis(
    atom_indices=(0,),
    n=(1,),
    l=(0,),
    m=(0,),
    labels=("s",),
)
hopping = -0.8
bulk = dp.types.make_tb_model(
    hopping_amplitudes=jnp.asarray([hopping, hopping], dtype=jnp.complex128),
    onsite_energies=jnp.zeros(1),
    soc_lambdas=jnp.zeros(0),
    geometry=geometry,
    basis=basis,
    hopping_pairs=((0, 0), (0, 0)),
    hopping_cells=((0, 0, 1), (0, 0, -1)),
    shell_index=(-1,),
)
```

Build a six-Angstrom material span normal to `(001)`. For this unit-spacing
chain that span contains seven atomic planes (six interplane intervals). The
returned `SlabSpec` records the exact surface coefficients and atom
provenance. Every retained hopping has zero normal-image cell component;
vacuum size is not used as a substitute for this graph invariant.

`gen_slab` is the host-side convenience entry point. Call
`freeze_slab_topology` once outside JAX transforms. Then call `rebuild_slab`
under `jit`, `grad`, or `vmap` while the discrete topology remains valid.

```python
slab, slab_spec = dp.tightb.gen_slab(
    bulk,
    miller=(0, 0, 1),
    thickness_ang=6.0,
    vacuum_ang=8.0,
)
print("layers:", slab_spec.n_layers)
print("depths (Angstrom):", slab.depths)
print("normal-image cells:", {cell[2] for cell in slab.hopping_cells})
```

The finite-chain eigenvalues are
$E_m=2t\cos[m\pi/(N+1)]$. This provides an analytic correctness check rather
than a comparison with another slab implementation.

```python
k_parallel = jnp.zeros((1, 3))
bands = dp.tightb.diagonalize_tb(slab, k_parallel)
modes = jnp.arange(1, slab_spec.n_layers + 1)
expected = jnp.sort(
    2.0 * hopping * jnp.cos(modes * jnp.pi / (slab_spec.n_layers + 1))
)
print("maximum analytic error:", jnp.max(jnp.abs(bands.eigenvalues[0] - expected)))
```

Depths are probability coordinates for surface diagnostics. Here
`intensity_escape_length_ang` is an intensity escape length, so the orbital
weight is `exp(-depth/lambda_I)`. Coherent photoemission amplitudes use the
separate `exp(-depth/(2*lambda_I))` law owned by the matrix-element stage.

```python
surface_weights = dp.tightb.layer_resolved_weights(
    bands,
    intensity_escape_length_ang=2.0,
)
print(surface_weights)
```

Individual band weights are plotting diagnostics away from degeneracy. At a
degeneracy, use `layer_resolved_group_traces` with a preregistered complete,
complement-isolated band group.
