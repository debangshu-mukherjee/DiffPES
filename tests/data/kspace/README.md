# Chinook k-space verification

These offline references verify `kz-kinematics-reference`,
`tilted-polarization-reference`, and `kmesh-reduction-reference`. DiffPES tests consume only the immutable JSON files and do
not install, import, or execute Chinook.

The authoritative generator is maintained outside the DiffPES source and test
trees in the external verification repository under `verification/kspace/`. It runs manually in a
dedicated pinned environment. Only generated data, hashes, and provenance cross
into this directory; no Chinook-importing Python belongs under `tests/`.

- Chinook commit: `24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`
- DiffPES baseline commit: `afe36cfbb703510f01de6da376b35627eaac8d4d`
- Environment SHA-256: `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- Platform: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- Machine: `x86_64`
- Python: `3.11.13`

The offline generator applies a Python 3.11 compatibility shim for Chinook's
legacy `collections.Iterable` import. It does not change numerical code. The
kinematics artifact records the active/passive and slit-axis mappings beside every
raw-source-derived value.

## Checksums

- `chinook_env_freeze.txt`: `6d00cb4df251508b6392273b1df166f6a17abe8f6691cffead45c636e8ef2531`
- `kz_kpt_reference.json`: `729c68d4a03bddaa89bc20ce072192ff50b792259cad68aa928d4cf7218da8e0`
- `mesh_reduce_reference.json`: `ea9e5414acfe948c8813067c38026538ea24517b94fd631c2734a018f017b386`
- `tilt_polarization_reference.json`: `2f0aa6c7af16a5bc74ac620adc037eb912af40e982e673565cf78236b4545973`
