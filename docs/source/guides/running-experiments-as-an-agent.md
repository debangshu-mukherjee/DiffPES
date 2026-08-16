# Running experiments as an agent

Automatons are standalone diffpes experiment files in the repository root.
They use the public API and return one machine-readable JSON record.

Start with the [catalog](https://github.com/debangshu-mukherjee/diffpes/blob/main/automatons/INDEX.md)
and its [short descriptions](https://github.com/debangshu-mukherjee/diffpes/blob/main/automatons/README.md).
The schemas define the [description record](https://github.com/debangshu-mukherjee/diffpes/blob/main/automatons/schema/automaton_params.schema.json)
and the [result record](https://github.com/debangshu-mukherjee/diffpes/blob/main/automatons/schema/automaton_result.schema.json).

## Discover and inspect

Read the catalog before selecting a file.
Run `--describe` to inspect parameters, returned fields, and artifact roles.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/dp-mpl .venv/bin/python \
  automatons/forward_bands.py --describe --json
```

The final stdout line contains the description JSON object.
The `params_schema` member lists accepted parameter names and types.

## Validate and run

Pass parameter JSON with `--params <file>`, `--params -`, or inline JSON.
Explicit command flags override values from the parameter document.
Run `--validate` before expensive work when inputs come from another agent.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/dp-mpl .venv/bin/python \
  automatons/forward_bands.py --validate --params '{"n_k": 24}' --json
```

Use a dedicated output directory for each run.
Use one fixed seed when reproducibility matters.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/dp-mpl .venv/bin/python \
  automatons/forward_bands.py --smoke --seed 123 \
  --outdir /tmp/dp-forward-bands --json
```

Smoke mode uses small in-code fixtures and CPU-safe sizes.
Each smoke run must finish within 60 seconds on a CPU host.

## Estimate resources

Run `--estimate` to inspect declared resource needs before a calculation.
An experiment with an estimate emits `spec.estimate(args)` as JSON.
An experiment without an estimate emits this JSON object:

```json
{
  "est_wall_s": null,
  "needs_gpu": false,
  "est_mem_gb": null,
  "cache_warm": null
}
```

Use the estimate to select a suitable host and output location.

## Parse and trust the result

Read the final stdout line as JSON.
Ignore any earlier human-readable output.
The result includes `status`, `params`, `metrics`, `artifacts`, and
`result_key`.

The `result_key` identifies the experiment, merged parameters, seed, and
diffpes version.
Use it to compare equivalent requests across independent runs.
Artifact paths are relative to `--outdir` and stay below that directory.
Each artifact record gives a role, MIME type, path, and optional preview.

An exit status of zero reports an `ok` result.
Input range and unsupported-parameter failures exit with status 2.
Deadline expiration exits with status 124 and reports `timeout`.
Other failures exit with status 1 and provide `error_kind`.

## Runtime controls

Use `--cache` to enable the persistent JAX compilation cache.
Set `DIFFPES_JAX_CACHE_DIR` to choose the cache directory.
Use `--unchecked` only when runtime type checks block a diagnosis.
Use `--deadline <seconds>` to limit wall time on POSIX hosts.
Use `--json` when another process consumes stdout.

## Experiment classes

Forward simulation files create bands, cuts, cubes, detector images, and
photon-energy scans.
Measurement ingest files normalize data and compare candidate simulations.
Inversion files recover physical or experimental parameters from planted data.
Identifiability files report information rank, design rankings, and derivative
evidence.
Diagnostics files check convergence, grids, certification, and exported models.

## Agent pattern

1. Read the catalog and choose one file.
2. Call `--describe` and build a parameter document.
3. Call `--validate` and repair invalid input.
4. Run `--smoke --json` with a fixed seed.
5. Parse the final record and inspect declared artifacts.
6. Repeat a successful request before using it in a larger workflow.

The catalog and schemas are part of the repository contract.
Keep them aligned when a new experiment file lands.
