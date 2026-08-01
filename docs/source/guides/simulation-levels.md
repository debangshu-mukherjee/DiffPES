# Simulation Tiers and the Coherent Pipeline

diffpes separates inexpensive projection spectra from coherent
photoemission amplitudes. This distinction is physical, not merely an API
choice. A VASP `PROCAR` projection contains probabilities
$|c_{n o}(\mathbf{k})|^2$; it does not contain the relative phases needed for
interference between orbitals or atomic centres.

## The Two Incoherent Tiers

| Tier | Broadening | Orbital weight | Intended use |
|---|---|---|---|
| `novice` | Voigt | uniform over non-s channels | quick band-map checks |
| `basic` | Gaussian | element- and subshell-resolved Yeh--Lindau data | photon-energy trends without interference |

Both tiers calculate one probability-level reduction,

$$
W_n(\mathbf{k})=\sum_o |c_{no}(\mathbf{k})|^2 w_o,
$$

and then apply occupation and energy broadening. They cannot describe dark
corridors, sublattice interference, or polarization selection rules that
depend on amplitude phase.

Use the typed functions `simulate_novice` and `simulate_basic`, or dispatch
plain arrays through `simulate_expanded`:

```python
spectrum = diffpes.simul.simulate_expanded(
    level="novice",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    sigma=0.04,
    gamma=0.08,
    fidelity=2000,
)
```

The `basic` route additionally requires an atom-major `OrbitalBasis` and
`atomic_numbers`. It uses `yeh_lindau_orbital_weights`, which gathers
subshell cross sections from the packaged Yeh--Lindau table. The underlying
`yeh_lindau_cross_section` interpolation is log-log monotone within valid
table segments. It rejects extrapolation and missing-data gaps.

## The Coherent Primitive Pipeline

Quantitative matrix-element work uses functions from
`diffpes.simul.matrixel` directly:

1. Build an `OrbitalBasis`, `RadialSpec`, `MatrixElementParams`,
   `RadialQuadratureSpec`, and `FinalStateSpec`.
2. Supply explicit vacuum final momenta `k_f_cart` and their emission-validity
   mask from the kinematics layer.
3. Use `assemble_orbital_transition_channels` to retain one complex Cartesian
   transition vector per orbital and outgoing spin.
4. Apply `contract_polarization` or `contract_experiment_polarization` only
   after the transition vector is assembled.
5. Use `project_band_channels` with complex band coefficients.
6. Apply `matrix_element_intensity` once. It sums outgoing-spin modulus
   squares incoherently while preserving interference inside each spin
   channel.

This pipeline does not read the experiment's inner potential to invent a
vacuum momentum. Inner-potential estimates belong to bulk $k_z$ inference;
the matrix element consumes the explicit physical vacuum vector.

## Choosing an Interface

- Use `novice` to inspect dispersions and broadening cheaply.
- Use `basic` for isolated-atom photon-energy trends when probability-only
  projections are sufficient.
- Use the coherent primitives when relative orbital phase, atomic positions,
  polarization, surface attenuation, spinors, or parameter sensitivities
  matter.

Expected detector counts are outside these tiers and primitives. Exposure,
background, detector response, and counting statistics are not yet supported.

See [Matrix Elements and Polarization](matrix-elements-and-polarization.md)
for the amplitude convention and
[Matrix-element sensitivity](../tutorials/matrix-element-sensitivity.md) for
complete-group derivatives.
