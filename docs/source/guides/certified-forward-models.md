# Certified forward models

diffpes certification is a scientific-assurance record for one execution of a
forward model. It records the model, semantic conventions, normalized inputs,
and ordered transformations. It also records validity checks, derivative
evidence, information losses, and the policy outcome for an observable.

Certification does **not** assert universal model correctness or input
artifact truthfulness. The recorded model domain, checks, evidence, and
numerical tolerances bound a statement such as "validated under
`org.diffpes.policy.research.v1`."

## The JAX-native assurance path

The scientific part of certification executes through the same JAX program as
the ordinary forward model. Contract predicates are array functions. Domain
margins and residuals remain differentiable leaves. JVPs and VJPs compute the
information-flow evidence. The discrete pass/fail fields show views of those
continuous quantities.

The certified executor retains one `jax.linearize` result. All requested JVP
probes and the transpose operation use this retained linear map. The structural
dependency cache uses the exact model ID, PyTree structure, shape, and dtype.
`checkify` returns structured errors for failed hard domain checks.

This separation matters during optimization. A policy can report a point
outside a verified photon-energy domain. The algebraic domain margin can still
supply a useful gradient toward the domain. Likewise, a small VJP reports
local first-order insensitivity at the evaluation point. It does not prove
global independence.

diffpes performs filesystem discovery and persistence outside the JAX
program. These operations cannot satisfy physics, numerical, or
differentiability claims.

## What a certificate records

A {class}`~diffpes.types.ForwardCertificate` contains:

- a versioned forward-model specification and execution manifest;
- source and normalized-artifact references;
- semantic conventions and validity-domain predicates;
- an ordered provenance graph with preserved, introduced, and destroyed
  information;
- claims with continuous margins and references to supporting evidence;
- derivative, structural-dependency, local-sensitivity, and information-spectrum
  diagnostics;
- the evaluated policy report and any unmet requirements.

The graphs answer different questions and remain distinct. Provenance records
the process that produced a result. Structural dependency records which input
leaves occur in the traced program. JVP/VJP evidence records local first-order
information flow. Transformation records identify preserved or lost
interpretations. Examples include absolute intensity calibration after
normalization and high-frequency structure after convolution.

## Certification levels

A named policy recomputes levels from claims. A stored `certified=True` flag
does not define a level.

| Level | Bounded meaning |
| --- | --- |
| `identified` | The certificate identifies the model, inputs, conventions, environment, and output. |
| `validated` | Required contracts and in-domain runtime checks passed. |
| `differentiable` | The evaluator computed the declared derivatives, which met their evidence tolerances. |
| `verified` | Required analytic or external-reference comparisons passed. |
| `benchmarked` | A named independent model or dataset comparison passed in its declared domain. |
| `reproducible` | Resolver-backed re-execution met the recorded reproduction tolerance. |

`failed`, `not_checked`, `not_applicable`, and `out_of_domain` remain visible on
individual claims. A waiver explains an unmet policy requirement; it never
turns the corresponding claim into a pass.

Each waiver records an author, reason, claim scope, issue time, and expiry time.
The preparation boundary requires absolute UTC times. It rejects a malformed,
premature, or expired waiver before compiled execution.

## Artifact resolution and reproduction

An artifact resolver returns normalized scientific content. It can also return
the exact source bytes. diffpes checks separate domain-separated SHA-256
identities for normalized content and exact bytes.

The reproduction runner requires one normalized-input artifact and one result
artifact. It resolves both artifacts and runs the exact registered model. The
report stores maximum absolute and relative errors against an explicit
tolerance.

```python
from diffpes.certify import mapping_artifact_resolver, reproduce_forward

resolver = mapping_artifact_resolver(
    {"input": normalized_inputs, "result": stored_result}
)
report = reproduce_forward(certificate, resolver=resolver)
```

Only a successful resolver-backed report supports the `reproducible` policy
level. A checksum match alone does not support that level.

## Registry handshakes

The packaged registry manifest lists each built-in model and transformation.
diffpes generates model cards directly from the registered model specification.
The registry validator detects a missing entry or changed model card.

Owner handshakes name required model, transformation, convention, and evidence
IDs. The kinematics and geometry layer registers four semantic transformation
contracts through this interface. Its handshake becomes complete only when the
caller supplies all six declared evidence IDs.

## Inspection

The inspection API answers scientific questions without loading the result
arrays:

```python
from diffpes.certify import (
    diff_certificates,
    explain_claim,
    summarize_certificate,
)

print(summarize_certificate(certificate))
print(explain_claim(certificate, "claim.output.finite"))
comparison = diff_certificates(reference, candidate)
```

`diff_certificates` separates scientific model, input, transformation,
evidence, numerical, environment, and audit-time differences. The comparison
therefore distinguishes a timestamp-only rerun from a model or convention
change.

## Portable records

Canonical JSON is the authoritative portable representation. The serializer
stores numeric arrays losslessly with dtype and shape metadata. It does not
round arrays into decimal lists. HDF5 stores the exact JSON record and several
convenience attributes next to a numerical result:

```python
from diffpes.inout import (
    attach_certificate_h5,
    certificate_identity,
    finalize_certificate,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)

save_certificate_json(certificate, "run.certificate.json")
restored = load_certificate_json("run.certificate.json")

finalized = finalize_certificate(certificate)
identity = certificate_identity(finalized)

attach_certificate_h5("spectrum.h5", "spectrum", certificate)
restored_from_h5 = load_certificate_h5("spectrum.h5", "spectrum")
```

The record includes a consistency marker that detects accidental storage
mismatches. This bookkeeping value provides no security, authenticity, or
physical-validity claim. The loader rejects unknown schema major versions.
It retains unknown minor extension data in the certificate extension object.
A load and save cycle therefore preserves that data.

The I/O boundary replaces the traced kernel placeholder with the canonical
certificate identity. This identity covers all scientific and numerical
fields. It excludes the audit execution ID and wall-clock timestamp. SHA-256
provides collision-resistant content addressing, not authentication. The outer
storage CRC32 remains a separate transport-corruption check over the complete
document and never participates in scientific identity.

## Reading claims responsibly

A certificate can establish that the declared implementation used the
recorded inputs. It can also establish that the implementation met named tests
in a declared domain. It cannot assess an approximation for an unrecorded
experiment. It cannot prove that an upstream DFT file or instrument file is
truthful. Local Jacobian rank cannot establish global identifiability. Those
conclusions require upstream evidence and scientific judgment.
