# Analytic Kramers--Kronig reference

This directory contains the independent analytic reference for the Kramers--Kronig operator. The
archive uses the frozen 1001-point relative-energy grid from -1 eV to 1 eV.
All real parts are subtracted at 0 eV.

`generate_kk_analytic_reference.py` evaluates the retarded-pole and Wigner
semicircle closed forms with 80-decimal-digit `mpmath`. It verifies the KK sign
and normalization by direct principal-value quadrature at seven fixed points.
It never imports `diffpes.simul.spectral` or other DiffPES production code.

Run the generator with:

```text
.venv/bin/python tests/test_diffpes/_reference_data/kk_analytic_reference/generate_kk_analytic_reference.py
```

`manifest.json` records all fixture values, the numerical arbiter, array
shapes, and the SHA-256 digest of `kk_reference.npz`.
