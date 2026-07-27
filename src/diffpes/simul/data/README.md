# Yeh–Lindau atomic subshell cross sections

`yeh_lindau_1985.npz` is a deterministic, compact derivative of the
Regoutz-group digitisation of J. J. Yeh and I. Lindau, *Atomic Data and
Nuclear Data Tables* **32**, 1–155 (1985),
doi:10.1016/0092-640X(85)90016-6.

The source workbook is Figshare dataset
doi:10.6084/m9.figshare.12389750.v3, file ID `22867790`, distributed under
CC BY 4.0. `yeh_lindau_1985.json` records its SHA-256, the derived archive
SHA-256, units, interpolation convention, and every supported positive
energy interval. Regenerate the archive with:

```console
python scripts/generate_yeh_lindau_data.py Excel_Yeh_Lindau_1985_PICS.xlsx
```

Cross sections are stored in megabarn. Missing cells remain `NaN`, published
zeros remain zero, and the runtime interpolator neither fills gaps nor
extrapolates. The stored slopes are shape-preserving cubic Hermite
derivatives in `log(sigma)` versus `log(photon_energy)`.

When using these data, cite both the original Yeh–Lindau paper and the
Regoutz-group digitisation named above.
