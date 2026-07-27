# VASP Data Ingestion

diffpes simulates ARPES from DFT output. The `diffpes.inout` package parses
six VASP file types into validated PyTrees. These files are POSCAR, EIGENVAL,
PROCAR, KPOINTS, DOSCAR, and CHGCAR. The `diffpes.simul` package assembles the
PyTrees into simulation inputs. This guide describes the reader signatures,
Fermi-energy handling, orbital projections, workflows, and HDF5 persistence.

The readers use plain Python I/O because file parsing is sequential. The
readers are not compatible with `jit`. Each reader returns a JAX PyTree from
a `diffpes.types` factory. JAX can transform and differentiate the returned
arrays.

## Reader Summary

| Function | Input file | Returns |
|---|---|---|
| `read_poscar(filename="POSCAR")` | POSCAR / CONTCAR | `CrystalGeometry` |
| `read_eigenval(filename="EIGENVAL", fermi_energy=0.0, return_mode="legacy")` | EIGENVAL | `BandStructure` or `SpinBandStructure` |
| `read_procar(filename="PROCAR", return_mode="legacy")` | PROCAR (`LORBIT=11/12`) | `OrbitalProjection` or `SpinOrbitalProjection` |
| `read_kpoints(filename="KPOINTS")` | KPOINTS (line / explicit / auto) | `KPathInfo` |
| `read_doscar(filename="DOSCAR", return_mode="legacy")` | DOSCAR | `DensityOfStates` or `FullDensityOfStates` |
| `read_chgcar(filename)` | CHGCAR (charge / spin / SOC) | `VolumetricData` or `SOCVolumetricData` |

The spin-aware readers share the `return_mode` option. The default
`"legacy"` mode returns the simple type and only the first spin channel.
`"full"` returns the spin-resolved type when the file contains `ISPIN=2` or
SOC data. Otherwise, `"full"` returns the simple type.

## A Worked Example on the In-Repo Fixtures

The test suite includes minimal VASP files in
`tests/test_diffpes/test_inout/fixtures/`. The parsers use these files in
tests. Run this example from the repository root:

```python
from diffpes.inout import (
    read_chgcar, read_doscar, read_eigenval,
    read_kpoints, read_poscar, read_procar,
)

fixtures = "tests/test_diffpes/test_inout/fixtures/"

geometry = read_poscar(fixtures + "POSCAR")
bands = read_eigenval(fixtures + "EIGENVAL", fermi_energy=0.0)
projection = read_procar(fixtures + "PROCAR")
kpath = read_kpoints(fixtures + "KPOINTS_line")
dos = read_doscar(fixtures + "DOSCAR")
volume = read_chgcar(fixtures + "CHGCAR_charge")

print(geometry.species, geometry.positions.shape)    # per-atom species, (6, 3)
print(bands.eigenvalues.shape, bands.kpoints.shape)  # (1, 1) (1, 3)
print(projection.projections.shape)                  # (2, 2, 1, 9)
print(kpath.mode, tuple(kpath.labels))               # Line-mode ('G', 'X', 'M')
print(dos.energy.shape, float(dos.fermi_energy))     # (5,) 0.5
print(volume.charge.shape, volume.grid_shape)        # (2, 2, 2) (2, 2, 2)
```

Spin-resolved fixtures include `EIGENVAL_spin`, `PROCAR_spin`, `PROCAR_soc`,
`DOSCAR_spin`, `CHGCAR_spin`, and `CHGCAR_soc`. Pass `return_mode="full"` to
return the `Spin*` or SOC types. The fixtures are intentionally small and do
not have consistent dimensions across files. EIGENVAL has one k-point, while
PROCAR has two. Therefore, the workflow example uses a real VASP run
directory.

## Reader Outputs

- **`read_poscar`** returns a `CrystalGeometry`. It applies the universal
  scaling factor to `lattice [3, 3]`. It derives
  `reciprocal [3, 3]` and converts `positions [N, 3]` to fractional
  coordinates. The result also contains per-atom `species`. The
  reader supports VASP 4, VASP 5+, and `Selective dynamics` blocks.
- **`read_eigenval`** parses eigenvalues at each k-point. It sorts the bands
  by energy at each point. In `"full"` mode, `ISPIN=2` files return both
  channels in a `SpinBandStructure`.
- **`read_procar`** detects the block layout: 1 block (`ISPIN=1`), 2 blocks
  (`ISPIN=2`, up/down), or 4 blocks (SOC: total, Sx, Sy, Sz). In full mode,
  the reader encodes spin data as six non-negative channels:
  `[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]` on a `SpinOrbitalProjection`.
- **`read_kpoints`** handles line-mode paths (with symmetry-point labels),
  explicit lists, and automatic grids. `KPathInfo.mode` records the input
  mode. The result also contains labels, segment counts, and grid metadata.
- **`read_doscar`** returns the total DOS and Fermi energy. Full mode adds the
  integrated DOS, spin channels, and available per-atom PDOS blocks.
- **`read_chgcar`** parses one or more volumetric grids. It supports charge,
  `ISPIN=2` magnetization, and three-component SOC magnetization. SOC data
  returns as `SOCVolumetricData`.

## Fermi-Energy Handling

EIGENVAL does not contain the Fermi energy, but DOSCAR does.
`read_eigenval` accepts an explicit `fermi_energy` argument with default
`0.0`. It stores this value on the returned `BandStructure`. The reader does
not shift the eigenvalues. Downstream Fermi-Dirac weighting uses
`bands.fermi_energy`.

The workflow helper uses a fixed priority. An explicit `fermi_energy` value
has the highest priority. Otherwise, the helper reads the value from DOSCAR.
Without either source, it uses `0.0`. The helper raises `FileNotFoundError`
when a required DOSCAR file is missing.

## The Orbital Projection Layout

`projections` has shape `[nkpt, nband, natom, 9]`, with the last axis in
the VASP orbital ordering:

```text
index:   0    1    2    3    4     5     6     7     8
orbital: s    py   pz   px   dxy   dyz   dz2   dxz   dx2-y2
```

The parser does not store the VASP `tot` column. Standard zero-based,
end-exclusive slices select orbital families. Use `slice(1, 9)` for non-s,
`slice(1, 4)` for p, `slice(4, 9)` for d (see
[Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
for the full convention). Three `diffpes.inout` helpers provide common
reductions. All helpers preserve differentiability:

```python
from diffpes.inout import aggregate_atoms, reduce_orbitals, select_atoms

surface = select_atoms(projection, [0])        # keep atom 0 (zero-based)
summed = aggregate_atoms(projection, [0])      # [K, B, 9], atom axis summed
spd = reduce_orbitals(projection.projections)  # [K, B, A, 3] s/p/d totals
```

## From Files to a Spectrum

`diffpes.simul` provides three assembly levels for a VASP run directory. Use
this one-call form:

```python
from diffpes.simul import run_vasp_workflow

spectrum = run_vasp_workflow(
    level="novice",
    directory="path/to/vasp_run",   # EIGENVAL, PROCAR, DOSCAR, KPOINTS
    photon_energy=21.2,
    sigma=0.04,
    gamma=0.08,
    fidelity=2500,
)
```

Alternatively, separate loading from simulation. This form supports
inspection, selection, and reuse of the parsed context:

```python
from diffpes.simul import load_vasp_context, prepare_projection, simulate_context

context = load_vasp_context(
    directory="path/to/vasp_run",
    procar_mode="full",       # preserve spin data when present
    check_dimensions=True,    # cross-file consistency checks
)
# context is a WorkflowContext: .bands, .orb_proj, .kpath, .dos

prepared = prepare_projection(context.orb_proj, atom_indices=[0, 1],
                              attach_oam=True)
spectrum = simulate_context(context, level="novice", photon_energy=21.2,
                            dk=0.02, normalize=True)
```

`load_vasp_context` reads EIGENVAL and PROCAR. It also reads available DOSCAR
and KPOINTS files. The function resolves the Fermi energy as described above.
With `check_dimensions=True`, it checks the k-point and band counts before
simulation. `simulate_context` passes the eigenvalues and prepared projections
to `simulate_expanded`. It can also apply momentum broadening and z-score
normalization. The retained dispatch values are `"novice"` and `"basic"`.
The basic route also requires an atom-major `OrbitalBasis` and one atomic
number per selected atom. Both routes consume projection probabilities and
are incoherent. Use `DiagonalizedBands` and the matrix-element primitives
when complex band coefficients and interference must be preserved.

## HDF5 Round-Trip

Parse large PROCAR files once, and persist the PyTrees:

```python
from diffpes.inout import load_from_h5, save_to_h5

save_to_h5(
    "vasp_run.h5",
    compression="gzip",
    bands=bands, projection=projection, geometry=geometry,
)

data = load_from_h5("vasp_run.h5")      # {"bands": ..., "projection": ..., ...}
bands_back = load_from_h5("vasp_run.h5", name="bands")
```

Each keyword becomes an HDF5 group, and each array field becomes a dataset.
JSON attributes store static metadata such as labels, modes, and `grid_shape`.
The group also stores the type name. `load_from_h5` uses this name to
reconstruct the exact PyTree class. Optional `None` fields remain `None`.
Unknown group types raise an error during loading.

## Related Reading

- [PyTree Architecture](pytree-architecture.md) describes the reader output
  types and their validation contract.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  describes the `simulate_*_expanded` functions used by `simulate_context`.
- [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
  describes differentiation after PyTree construction.
- API reference: {doc}`../api/inout`, {doc}`../api/simul`,
  {doc}`../api/types`.
