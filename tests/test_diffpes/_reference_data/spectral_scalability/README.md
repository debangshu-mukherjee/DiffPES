# Spectral scalability reproducibility evidence

Regenerate the isolated CPU artifact from the repository root after the
spectral implementation is stable:

```bash
MPLCONFIGDIR=/tmp/diffpes-mpl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python tests/_reference_tools/measure_spectral_scaling.py
```

`cpu_benchmark.json` records an ahead-of-time compilation of the literal
registered `_stream_spectral_intensity` value-and-Hamiltonian-gradient target.
The target has `256 k`, `512 omega`, one spinless outgoing source, `32 orbitals`,
and static `(32, 32)` chunks. It uses a numerical Kramers--Kronig self-energy
with `n_kk=4096` and `n_tail=256`. The target was compiled and measured but not
executed; that choice is recorded explicitly. Small executable companions
compare checkpointed values and gradients with the unchunked production
assembler. They also count traces across three active `(k, omega)` sizes inside
one fixed padded shape.

XLA `memory_analysis` is the allocation authority. It reported argument,
output, temporary, and alias allocations of `4,211,032`, `4,194,328`,
`50,187,248`, and `0` bytes, respectively. Their derived peak-live total is
`58,592,608` bytes. This total is compared directly with the frozen spinless
solve-tape model: `16*n_k*omega_chunk*n_orb**2 = 134,217,728` bytes. Its `1.5x`
ceiling is `201,326,592` bytes. No projection from a smaller executable
authorizes this requirement. Whole-process peak RSS remains diagnostic because it
includes Python, compilation, allocator caches, and the companion executables.
The recorded high-water values were `464,285,696` bytes before and `686,735,360`
bytes after the literal-target compile. The executable companions had not run.
Compilation took `5.206802` seconds on the recorded TFRT CPU host. The compact
`k_i[K,3] + final_norm[E] + valid[E]` schedule accounts for `10,752` diagnostic
bytes and reconstructs final momentum only inside each live block.

The small unchunked comparison reported maximum absolute errors of
`5.9164567891575885e-31` for values and exactly `0` for Hamiltonian gradients;
its maximum reference gradient was `0.10284887664454907`. The fixed padded
schedule traced once and retained compile-cache sizes `[0, 1, 1, 1]` across
three active shapes.

The dtype-boundary record lowers the production Lineax solve, requires complex128
operator/RHS/solution IR, and records that the typed public boundary rejects a
complex64 call. Source hashes bind the generator, numerical implementation,
carrier, dependency metadata, and lock file to the measurement. The pytest
artifact handshake recomputes every digest and every allocation identity.
The committed `cpu_benchmark.json` SHA-256 is
`73e14ff43beabbbbad71d7dbe1ee1ba8defa1d5f9e16fb24febf115b40daa50f`.
