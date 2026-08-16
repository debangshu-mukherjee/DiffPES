# Agent-runnable experiments

The `automatons/` directory contains standalone diffpes experiments for
automation and continuous integration.

Each file uses the public API and writes records below its output directory.
Read [INDEX.md](INDEX.md) to choose an experiment.
Read the [agent guide](../docs/source/guides/running-experiments-as-an-agent.md)
before an automated run.

Run a small forward simulation with:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/dp-mpl .venv/bin/python \
  automatons/forward_bands.py --smoke --outdir /tmp/dp-bands --json
```

Use `--describe` before a run.
Use `--smoke` for CPU-safe trial runs.
Read the final stdout line as one JSON result.

## Catalog

| File | Purpose |
| --- | --- |
| `forward_bands.py` | Simulate tight-binding bands along a momentum path. |
| `forward_spectral_cut.py` | Simulate an energy-momentum spectral cut. |
| `forward_arpes_cube.py` | Simulate a momentum-energy intensity cube. |
| `forward_detector_acquisition.py` | Simulate detector counts for one acquisition. |
| `photon_energy_scan.py` | Simulate a photon-energy scan. |
| `polarization_dichroism.py` | Compare intensity for two polarizations. |
| `vasp_bands_to_arpes.py` | Convert a VASP band input into an ARPES result. |
| `arpes_ingest.py` | Ingest and normalize a measured spectral cut. |
| `match_measured_to_simulated.py` | Score measured and simulated spectral cuts. |
| `resolution_sweep.py` | Measure response changes across resolution settings. |
| `counting_statistics.py` | Measure count statistics across exposures. |
| `fit_hopping_parameters.py` | Recover tight-binding hopping values. |
| `fit_self_energy.py` | Recover self-energy parameters. |
| `fit_experiment_geometry.py` | Recover experiment geometry parameters. |
| `information_spectrum.py` | Measure parameter information and null directions. |
| `experiment_design_compare.py` | Rank candidate experiment settings. |
| `audit_derivatives.py` | Check derivative evidence for a forward map. |
| `convergence_study.py` | Measure observables across numerical resolutions. |
| `parameter_grid.py` | Evaluate a grid of physical parameters. |
| `certify_forward.py` | Create and verify a forward certificate. |
| `export_model.py` | Export and compare a spectral model. |
| `bump_pin.py` | Update the declared diffpes version pins. |

## Schemas

The [parameter description schema](schema/automaton_params.schema.json)
defines the `--describe` payload.
The [result schema](schema/automaton_result.schema.json) defines the final
stdout JSON record.
