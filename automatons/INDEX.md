# Automaton index

Use this index to select an agent-runnable diffpes experiment.
Each entry supports discovery, parameter inspection, validation, and a
CPU-safe smoke run.

## Use the interface

1. Read a file summary below.
2. Run `--describe` to inspect accepted parameters and returned artifacts.
3. Run `--validate` to check merged parameters without calculation.
4. Run `--smoke --json` before a full calculation.
5. Parse the final stdout line as a JSON object.

## Forward simulation

| File | Summary |
| --- | --- |
| `forward_bands.py` | Bands on a selected momentum path. |
| `forward_spectral_cut.py` | Spectral intensity along a momentum path. |
| `forward_arpes_cube.py` | Intensity over momentum and energy. |
| `forward_detector_acquisition.py` | Detector counts from a simulated source. |
| `photon_energy_scan.py` | Intensity over photon energy. |
| `polarization_dichroism.py` | Polarization-dependent intensity contrast. |
| `vasp_bands_to_arpes.py` | VASP bands translated into ARPES observables. |

## Measurement ingest

| File | Summary |
| --- | --- |
| `arpes_ingest.py` | Load and normalize a spectral cut. |
| `match_measured_to_simulated.py` | Rank simulated cuts against a measurement. |
| `resolution_sweep.py` | Compare energy and momentum resolution settings. |
| `counting_statistics.py` | Compare count statistics across exposures. |

## Inversion

| File | Summary |
| --- | --- |
| `fit_hopping_parameters.py` | Fit tight-binding hopping parameters. |
| `fit_self_energy.py` | Fit a self-energy model. |
| `fit_experiment_geometry.py` | Fit polarization and geometry parameters. |

## Identifiability

| File | Summary |
| --- | --- |
| `information_spectrum.py` | Report information rank and null directions. |
| `experiment_design_compare.py` | Rank candidate acquisition settings. |
| `audit_derivatives.py` | Compare automatic and finite-difference derivatives. |

## Diagnostics and operations

| File | Summary |
| --- | --- |
| `convergence_study.py` | Check numerical convergence across grids. |
| `parameter_grid.py` | Evaluate a rectangular parameter grid. |
| `certify_forward.py` | Persist and verify a forward certificate. |
| `export_model.py` | Export a portable spectral calculation. |
| `bump_pin.py` | Rewrite experiment version pins. |

The [README](README.md) gives a concise catalog.
The [agent guide](../docs/source/guides/running-experiments-as-an-agent.md)
defines the execution protocol.
