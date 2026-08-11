# Native Tight-Binding Models

Build nearest-neighbor graphene twice: first from an explicit, Hermitian-closed
hopping list and then from one Slater--Koster integral. Add atomic spin--orbit
coupling and calculate fat bands with a Rashba spin texture. Finish with a
broadened density of states. Every numerical object remains a JAX PyTree.
The same models can later become optimization variables.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffpes.tightb import (
    build_kpath,
    build_sk_model,
    diagonalize_tb,
    dos_gaussian,
    eigvalsh_bands,
    expectation_path,
    fat_bands,
    fermi_level_from_filling,
    kpath_arc_length,
    spin_double_model,
    spin_operator,
)
from diffpes.types import (
    make_crystal_geometry,
    make_orbital_basis,
    make_slater_koster_params,
    make_tb_model,
)
```

## Graphene from an Explicit Hopping List

`TBModel` stores exact integer cell translations separately from the
differentiable complex hopping amplitudes. A physical nearest-neighbor bond is
therefore represented by a directed entry `(i, j, R)` and its conjugate
partner `(j, i, -R)`.

```python
a = 2.46
graphene_geometry = make_crystal_geometry(
    lattice=jnp.asarray(
        [
            [a, 0.0, 0.0],
            [0.5 * a, 0.5 * jnp.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, 10.0],
        ]
    ),
    positions=jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]
    ),
    species=("C", "C"),
)
graphene_basis = make_orbital_basis(
    atom_indices=(0, 1),
    n=(2, 2),
    l=(1, 1),
    m=(0, 0),
    labels=("A_pz", "B_pz"),
)

hopping = -2.7
graphene_hand = make_tb_model(
    hopping_amplitudes=hopping * jnp.ones(6, dtype=jnp.complex128),
    onsite_energies=jnp.zeros(2),
    soc_lambdas=jnp.zeros(0),
    geometry=graphene_geometry,
    basis=graphene_basis,
    hopping_pairs=(
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 0),
        (1, 0),
    ),
    hopping_cells=(
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
    ),
    shell_index=(-1, -1),
)
print(f"{len(graphene_hand.hopping_pairs)} directed hopping records")
```

```{admonition} Bloch-gauge convention
:class: important

diffpes uses the basis-position gauge. A hopping has phase
$\exp[2\pi i\,k\cdot(R+\tau_j-\tau_i)]$, where `R` is the exact integer cell
and $\tau_i$ is the fractional basis position. A cell-origin convention gives
the same eigenvalues but eigenvectors that differ by a diagonal,
momentum-dependent phase. Keep this distinction when comparing orbital
coefficients between codes. Native atom-centred models derive $\tau_i$ from
the assigned atom. Wannier models instead carry one explicit
`orbital_positions` row per orbital, so distinct centres on the same atom are
not collapsed.
```

Build the fractional $\Gamma$--K--M--$\Gamma$ path and diagonalize the model.
Repeated segment endpoints are intentional and make each symmetry point easy
to label.

```python
anchors = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
graphene_path = build_kpath(
    anchors,
    graphene_geometry,
    n_per_segment=51,
    labels=("Gamma", "K", "M", "Gamma"),
)
path_distance = kpath_arc_length(graphene_path, graphene_geometry)
hand_bands = diagonalize_tb(graphene_hand, graphene_path.kpoints)
print(hand_bands.eigenvalues.shape)
print("Dirac-point energies (eV):", hand_bands.eigenvalues[51])
```

## The Same Material from Slater--Koster Parameters

For an in-plane carbon--carbon bond, two $p_z$ orbitals couple through
$V_{pp\pi}$. The builder discovers the static neighbor topology once,
then retains exact cells. Away from cutoff crossings, differentiation covers
the bond geometry and Slater--Koster value.

```python
graphene_sk_params = make_slater_koster_params(
    values=jnp.asarray([hopping]),
    keys=("C-C:pp_pi",),
)
graphene_sk = build_sk_model(
    geometry=graphene_geometry,
    basis=graphene_basis,
    sk_params=graphene_sk_params,
    onsite_energies=jnp.zeros(2),
    soc_lambdas=jnp.zeros(0),
    shell_index=(-1, -1),
    cutoff=1.5,
)
sk_bands = diagonalize_tb(graphene_sk, graphene_path.kpoints)
maximum_difference = jnp.max(
    jnp.abs(sk_bands.eigenvalues - hand_bands.eigenvalues)
)
print(f"hand-built versus SK maximum error: {float(maximum_difference):.3e} eV")
```

The discrete neighbor search is not a differentiable operation. Freeze it
outside an optimizer and keep parameter updates within a region where bonds
do not enter or leave the cutoff. The returned amplitudes, onsite energies,
SOC strengths, positions, and lattice remain ordinary differentiable leaves.

## Atomic Spin--Orbit Coupling

Atomic SOC is $\lambda\,\mathbf L\cdot\mathbf S$. Start from a spinless,
complete $p$ shell, attach one shell identifier, and then duplicate the model.
The declared spin order is all spin-down orbitals followed by all spin-up
orbitals; `spinor=True` never doubles this basis again.

```python
p_basis = make_orbital_basis(
    atom_indices=(0, 0, 0),
    n=(2, 2, 2),
    l=(1, 1, 1),
    m=(-1, 0, 1),
    labels=("p_y", "p_z", "p_x"),
)
p_atom = make_tb_model(
    hopping_amplitudes=jnp.zeros(0, dtype=jnp.complex128),
    onsite_energies=jnp.zeros(3),
    soc_lambdas=jnp.asarray([0.30]),
    geometry=make_crystal_geometry(
        lattice=4.0 * jnp.eye(3),
        positions=jnp.zeros((1, 3)),
        species=("X",),
    ),
    basis=p_basis,
    hopping_pairs=(),
    hopping_cells=(),
    shell_index=(0, 0, 0),
)
p_atom_soc = spin_double_model(p_atom)
atomic_bands = eigvalsh_bands(
    p_atom_soc,
    jnp.zeros((1, 3)),
)
print("atomic SOC multiplets (eV):", atomic_bands[0])
```

The two $j=1/2$ states lie at $-\lambda$ and the four $j=3/2$ states at
$+\lambda/2$. The splitting is therefore $3\lambda/2$.

## Fat Bands

Raw eigenvector components depend on a phase convention and, at degeneracy,
on an arbitrary basis inside the degenerate subspace. `fat_bands` instead
forms an orbital projector and averages its expectation inside diagnostic
degenerate groups. At graphene's Dirac point both sublattices carry one-half
of each averaged state.

```python
a_sublattice = fat_bands(sk_bands, (0,))
print("A-sublattice weights at K:", a_sublattice[51])

fig, ax = plt.subplots(figsize=(7, 4))
for band in range(sk_bands.eigenvalues.shape[1]):
    points = ax.scatter(
        path_distance,
        sk_bands.eigenvalues[:, band],
        c=a_sublattice[:, band],
        s=16,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
ticks = jnp.asarray(graphene_path.label_indices)
ax.set_xticks(path_distance[ticks], graphene_path.labels)
ax.set_ylabel("energy (eV)")
ax.set_title(r"Graphene fat bands: A-sublattice weight")
fig.colorbar(points, ax=ax, label="projector expectation")
plt.show()
```

For derivatives exactly at a crossing, register the band group explicitly
and use `group_projector` or `group_trace`. The energy-threshold averaging in
`fat_bands` and `expectation_path` is a plotting diagnostic, not a
differentiable band selector.

## A Rashba Spin Texture

Spin texture is another projector-based diagnostic. This small native model
places one spatial orbital in each spin sector and uses conjugate-closed
spin-flip hoppings. The resulting square-lattice Rashba field winds around
$\Gamma$.

```python
rashba_geometry = make_crystal_geometry(
    lattice=jnp.diag(jnp.asarray([3.2, 3.2, 12.0])),
    positions=jnp.zeros((1, 3)),
    species=("X",),
)
rashba_basis = make_orbital_basis(
    atom_indices=(0, 0),
    n=(1, 1),
    l=(0, 0),
    m=(0, 0),
    spin=(-1, 1),
    labels=("s_down", "s_up"),
)
alpha = 0.27
rashba_model = make_tb_model(
    hopping_amplitudes=alpha
    * jnp.asarray(
        [-0.5, -0.5, 0.5, 0.5, -0.5j, 0.5j, 0.5j, -0.5j],
        dtype=jnp.complex128,
    ),
    onsite_energies=jnp.zeros(2),
    soc_lambdas=jnp.zeros(0),
    geometry=rashba_geometry,
    basis=rashba_basis,
    hopping_pairs=(
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
    ),
    hopping_cells=(
        (1, 0, 0),
        (-1, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, -1, 0),
        (0, 1, 0),
    ),
    shell_index=(-1, -1),
    spinor=True,
)

angle = jnp.linspace(0.0, 2.0 * jnp.pi, 48, endpoint=False)
rashba_kpoints = jnp.stack(
    (
        0.16 * jnp.cos(angle),
        0.16 * jnp.sin(angle),
        jnp.zeros_like(angle),
    ),
    axis=-1,
)
rashba_bands = diagonalize_tb(rashba_model, rashba_kpoints)
sx = expectation_path(
    rashba_bands,
    spin_operator(rashba_basis, jnp.asarray([1.0, 0.0, 0.0])),
)
sy = expectation_path(
    rashba_bands,
    spin_operator(rashba_basis, jnp.asarray([0.0, 1.0, 0.0])),
)
spin_magnitude = jnp.sqrt(sx[:, 1] ** 2 + sy[:, 1] ** 2)
print(
    "upper-band |<S>| range:",
    float(jnp.min(spin_magnitude)),
    float(jnp.max(spin_magnitude)),
)

fig, ax = plt.subplots(figsize=(5, 5))
ax.quiver(
    rashba_kpoints[:, 0],
    rashba_kpoints[:, 1],
    sx[:, 1],
    sy[:, 1],
    spin_magnitude,
    cmap="plasma",
    pivot="mid",
)
ax.set_xlabel(r"$k_1$")
ax.set_ylabel(r"$k_2$")
ax.set_aspect("equal")
ax.set_title("Upper-band Rashba spin texture")
plt.show()
```

Each arrow has spin-one-half magnitude. Lattice sine factors make the arrows
exactly perpendicular to the lattice Rashba field; near $\Gamma$ this is the
familiar tangential helical texture.

## Density of States and Filling

Sample a small uniform reciprocal mesh, broaden each eigenvalue with a
normalized Gaussian, and solve the finite-temperature filling equation.
`dos_gaussian` returns the same `DensityOfStates` carrier used by parsed
electronic-structure data.

```python
mesh_axis = jnp.arange(24, dtype=jnp.float64) / 24.0
k1, k2 = jnp.meshgrid(mesh_axis, mesh_axis, indexing="ij")
mesh_kpoints = jnp.stack(
    (jnp.ravel(k1), jnp.ravel(k2), jnp.zeros(k1.size)),
    axis=-1,
)
mesh_eigenvalues = eigvalsh_bands(graphene_sk, mesh_kpoints)
k_weights = jnp.full(
    mesh_kpoints.shape[0],
    1.0 / mesh_kpoints.shape[0],
)
energy_axis = jnp.linspace(-9.0, 9.0, 801)
graphene_dos = dos_gaussian(
    mesh_eigenvalues,
    k_weights,
    energy_axis,
    sigma=0.12,
)
chemical_potential = fermi_level_from_filling(
    mesh_eigenvalues,
    k_weights,
    n_electrons=1.0,
    temperature_k=300.0,
)
integrated_states = jnp.trapezoid(graphene_dos.total_dos, energy_axis)
print(f"half-filled chemical potential: {float(chemical_potential):.3e} eV")
print(f"integrated states per cell: {float(integrated_states):.8f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(graphene_dos.energy, graphene_dos.total_dos)
ax.axvline(float(chemical_potential), color="black", linestyle="--")
ax.set_xlabel("energy (eV)")
ax.set_ylabel("DOS (states/eV/cell)")
ax.set_title("Gaussian-broadened graphene DOS")
plt.show()
```

The positive temperature keeps the filling equation smooth. Its Optimistix
root is implicitly differentiable with respect to the band energies and,
through them, the hopping parameters.

## Importing a Wannier90 Model

For a material model generated elsewhere, use a format-specific reader rather
than rewriting its rows by hand. `read_wannier90_hr` requires explicit
Cartesian Wannier centres because `hr.dat` does not contain them;
`read_wannier90_tb` reads both the centres and full position matrices from
`seedname_tb.dat`. If multiple orbitals assigned to one atom have distinct
centres, pass the atomic `geometry` explicitly: Wannier centres do not by
themselves determine nuclear positions.

```python
from diffpes.inout import read_wannier90_hr, read_wannier90_tb

model, operators = read_wannier90_hr(
    "seedname_hr.dat",
    geometry,
    basis,
    centres_cart,
)

model, operators = read_wannier90_tb(
    "seedname_tb.dat",
    basis,
    spin_layout="block_down_up",
    geometry=geometry,
)
```

Parsing is deliberately a host-side operation. The returned model's numerical
leaves can be fine-tuned with JAX, while exact cells, orbital metadata, and
the one-time serialized spin permutation remain static. Native
diagonalization propagates explicit orbital positions into
`DiagonalizedBands` for later phase-sensitive matrix elements.
