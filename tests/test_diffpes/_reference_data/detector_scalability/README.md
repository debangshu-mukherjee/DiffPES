# Detector-driver scalability evidence

Regenerate the isolated CPU artifact from the repository root only after the
detector mapper, effects chain, and canonical drivers are stable:

```bash
MPLCONFIGDIR=/tmp/diffpes-mpl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cpu \
.venv/bin/python tests/_reference_tools/measure_detector_scaling.py
```

The harness ahead-of-time compiles the literal `256 x 256 x 400` expected-count
cube with 20 bands and 20 explicit Hamiltonian orbitals. The source grid uses
the released signed-diagonal boundary-aware cubature path. The executable is
compile-only on CPU because running its 26,214,400 detector bins is not part of
the memory requirement.

XLA `memory_analysis` supplies argument, output, temporary, alias, and derived
peak-live allocation for the forward and complete-Hamiltonian-gradient
programs. The respective frozen ceilings are 2 and 12 decimal gigabytes.
Whole-process RSS is diagnostic. A recursive literal-shape JAXPR audit rejects
flattened `(K,B,E)` tensors, raster `(kx,ky,E,B)` tensors, and every
permutation of the complete `(K,E,3)` or `(kx,ky,E,3)` final-momentum carrier.
The audit includes nested JAXPR constants and rejects flattened or
block-factored shapes by total element count as well. The last invariant
authenticates the compact kinematics schedule responsible for avoiding the
otherwise unavoidable 629 MB full-cube allocation.

Small executable companions compare checkpointed and non-rematerialized
full-driver values and Hamiltonian gradients at `rtol=1e-12`. They also count
one compilation across three native-FWHM and fixed-length photon-energy sweeps,
then compare a two-geometry `vmap` with direct rows. Their generic rotated
targets remain strictly enclosed inside every source exterior face.

The frozen CPU run passed with 1,679,527,776 bytes peak-live forward allocation.
Value plus the complete Hamiltonian gradient used 2,518,490,824 bytes. Its
recursive audit found no full KBE or full-kinematics carrier. Rematerialized
values and gradients were bitwise equal. The compile-reuse check recorded
cache sizes `[0, 1, 1, 1]`, and the geometry-batching check had zero vmap
error.

The JSON records hashes for this harness, the complete production driver and
detector stack, relevant carriers, dependency metadata, and the lock file. The
consumer authenticates artifact SHA-256
`85cfbf86d58d957605a2977ea1921a7568a32b06de9d6ab241a3fe9849fbe190`
before parsing the record.
