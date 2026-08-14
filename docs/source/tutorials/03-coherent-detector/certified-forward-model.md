# Inspect and persist a certified forward run

This tutorial starts after you call {func}`diffpes.certify.certify_forward`.
The function returns a {class}`~diffpes.types.CertifiedResult`. Its `value`
contains the ordinary JAX result. Its `certificate` contains the scientific
assurance record.

```python
from diffpes.certify import (
    certify_forward,
    explain_claim,
    prepare_certification,
    register_builtin_models,
    summarize_certificate,
)
from diffpes.types import (
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    make_execution_manifest,
)

register_builtin_models()

manifest = make_execution_manifest(
    execution_id="tb-radial-example",
    model_ref=f"{TB_RADIAL_MODEL_ID}@{TB_RADIAL_MODEL_VERSION}",
    schema_version="1",
    package_version="development",
    source_checksum="working-tree",
    environment_checksum="tutorial-environment",
    backend="cpu",
    precision_policy="float64",
    deterministic=True,
    started_at_utc="2026-07-21T00:00:00Z",
)

context = prepare_certification(
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    manifest,
    policy_id="org.diffpes.policy.research.v1",
)

# A single PyTree: bands, radial parameters, simulation parameters,
# polarization, work function, optional self-energy, radial grid, and dk.
model_inputs = (
    bands,
    radial,
    simulation,
    polarization,
    4.5,
    None,
    radial_grid,
    None,
)
run = certify_forward(context, model_inputs)
```

JAX runs the forward model and its scientific checks together. The envelope
does not change the spectrum, JVP, or VJP from the ordinary model path.

## Read the record

Start with the stable text summary:

```python
print(summarize_certificate(run.certificate))
```

The summary identifies the model, implementation, policy outcome, and input
artifacts. It also lists conventions, transformations, information losses,
claim states, derivatives, and information diagnostics.

Request a claim by its stable ID to examine its result:

```python
print(
    explain_claim(
        run.certificate,
        "claim.output.finite",
    )
)
```

The explanation includes the subject, continuous margin, evidence method,
reference values, residuals, and tolerances. The report treats zero local
sensitivity as a local observation. It does not claim global independence.

## Save beside the result

Use JSON when the certificate travels independently:

```python
from diffpes.inout import save_certificate_json

save_certificate_json(run.certificate, "tb-radial.certificate.json")
```

Use an HDF5 attachment when an HDF5 file already stores the numerical result:

```python
from diffpes.inout import attach_certificate_h5, save_to_h5

save_to_h5("tb-radial.h5", spectrum=run.value)
attach_certificate_h5("tb-radial.h5", "spectrum", run.certificate)
```

The HDF5 attachment contains the exact authoritative JSON bytes. Both forms
load as the same {class}`~diffpes.types.ForwardCertificate`:

```python
from diffpes.inout import load_certificate_h5, load_certificate_json

from_json = load_certificate_json("tb-radial.certificate.json")
from_h5 = load_certificate_h5("tb-radial.h5", "spectrum")
```

The stored consistency marker detects accidental storage mismatches only. It
does not provide security or physical assurance. The certificate records the
contracts, differentiable diagnostics, evidence, and policy evaluation that
provide scientific assurance.

## Compare reruns

```python
from diffpes.certify import diff_certificates

change = diff_certificates(from_json, later_run.certificate)
print(change.summary)
```

The comparison separates model, input, and semantic changes from numerical,
environmental, and audit-only differences. Review the scientific differences
before you compare spectra. Identical array shapes do not imply identical
physical meaning.
