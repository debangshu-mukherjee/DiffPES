# Matrix-element scalability reproducibility evidence

Regenerate the literal CPU evidence from the repository root:

```bash
MPLCONFIGDIR=/tmp/diffpes-mpl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python tests/_reference_tools/measure_matrix_element_scaling.py
```

The harness passes the 4096-k-point arrays, 18-orbital eigenvectors, radial
channels, matrix-element leaves, energy scales, and polarizations as dynamic
arguments. The XLA argument allocation is therefore about 23.8 MB rather than
the invalid eight-byte closed-over-constant result.

`channel_scan_hlo.txt.gz` and `channel_scan_jaxpr.txt.gz` contain both the
scalar-energy value/gradient program and the separate eight-energy reduced
scan. Their gzip metadata and JAX callback addresses are canonicalized, so
unchanged source and toolchain inputs regenerate byte-identical IR files.
The scan returns only `(8, 4096, 6)` complete-group weights and contains no
`(8, 4096, 18)` or transposed K-E-B allocation.

`memory_analysis` is the XLA compiler allocation authority. The separately
recorded whole-process peak RSS includes Python, compiler state, allocator
caches, and all scalability executables and is diagnostic only.

Throughput timing is host-specific. The JSON retains two warmups, all seven
synchronized repetitions for each of four routes, compilation times, medians,
and both ratios. The executable test recomputes every statistic from the raw
measurements. Another host need not reproduce identical wall-clock durations.
