# Expanded Wrappers and Conventions

The expanded wrappers accept plain arrays and scalars, construct validated
PyTrees, and delegate to the two retained incoherent spectrum functions.

| Typed function | Plain-input wrapper | Orbital weighting |
|---|---|---|
| `simulate_novice` | `simulate_novice_expanded` | uniform non-s |
| `simulate_basic` | `simulate_basic_expanded` | Yeh--Lindau subshell |

`simulate_expanded(level=...)` dispatches `"novice"` or `"basic"`
case-insensitively. Any other value raises `ValueError`.

## Array Layout

`eigenbands` has shape `[K, B]`. `surface_orb` has shape `[K, B, A, 9]`
and uses atom-major VASP ordering:

```text
index:   0    1    2    3    4     5     6     7     8
orbital: s    py   pz   px   dxy   dyz   dz2   dxz   dx2-y2
```

Python slices are zero-based and end-exclusive:

- non-s: `slice(1, 9)`
- p: `slice(1, 4)`
- d: `slice(4, 9)`

The values are projection probabilities, not complex coefficients. Expanded
wrappers therefore cannot reproduce orbital or inter-centre interference.

## Novice Dispatch

```python
from diffpes.simul import simulate_expanded

spectrum = simulate_expanded(
    level="novice",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.04,
    gamma=0.08,
    fidelity=2000,
    temperature=15.0,
    photon_energy=21.2,
)
```

The energy axis extends one eV beyond the minimum and maximum band energies.
`sigma` and `gamma` are the Gaussian and Lorentzian Voigt components.
`photon_energy` is retained by the shared carrier but does not change the
uniform novice weights.

## Basic Dispatch

The basic tier needs the atomic identity of every flattened projection
channel. Supply an atom-major `OrbitalBasis` with nine rows per atom and one
atomic number per atom:

```python
from diffpes.types import make_orbital_basis

basis = make_orbital_basis(
    atom_indices=(0,) * 9,
    n=(3, 3, 3, 3, 3, 3, 3, 3, 3),
    l=(0, 1, 1, 1, 2, 2, 2, 2, 2),
    m=(0, -1, 0, 1, -2, -1, 0, 1, 2),
)
spectrum = simulate_expanded(
    level="basic",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    photon_energy=40.8,
    basis=basis,
    atomic_numbers=(14,),
)
```

Within each atom, all p rows share one principal quantum number. The same
rule applies to all d rows. The table lookup rejects unsupported
element/subshell combinations or photon energies.

## Typed and Coherent Boundaries

Use typed `simulate_novice` or `simulate_basic` when carriers already exist.
Use `simulate_context` or `run_vasp_workflow` to connect parsed VASP data to
the same two dispatch routes.

Do not pass coherent complex eigenvectors through `surface_orb`: that would
discard their phase. Instead use the carrier factories and the
`assemble_orbital_transition_channels`, `project_band_channels`, and
polarization-contraction primitives described in
[Matrix Elements and Polarization](matrix-elements-and-polarization.md).

The level selector is static under JIT. Continuous scalar and array values
remain differentiable, but changing `"novice"` to `"basic"` selects a
different compiled program.
