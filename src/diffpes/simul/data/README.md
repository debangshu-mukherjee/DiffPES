# Yeh–Lindau atomic subshell cross sections

`yeh_lindau_1985.npz` is a deterministic, compact derivative of the
Regoutz-group digitisation of J. J. Yeh and I. Lindau, *Atomic Data and
Nuclear Data Tables* **32**, 1–155 (1985),
doi:10.1016/0092-640X(85)90016-6.

The source workbook is Figshare dataset
doi:10.6084/m9.figshare.12389750.v3, file ID `22867790`, distributed under
CC BY 4.0. `yeh_lindau_1985.json` records its SHA-256, the derived archive
SHA-256, generator SHA-256, extraction method, units, interpolation
convention, replay spot checks, and every supported positive energy interval.
Regenerate the archive with:

```console
python tests/_reference_tools/generate_yeh_lindau_data.py \
  Excel_Yeh_Lindau_1985_PICS.xlsx
```

Cross sections are stored in megabarn. Missing cells remain `NaN`, published
zeros remain zero, and the runtime interpolator neither fills gaps nor
extrapolates. The stored slopes are shape-preserving cubic Hermite
derivatives in `log(sigma)` versus `log(photon_energy)`.

The versioned Figshare workbook is the executable numerical authority. Its
dataset record says that the values were manually mined from the original
tabulated data; the Regoutz-group project page records internal peer review
and Prof. Lindau's agreement to make the dataset available. The primary paper
DOI and volume-wide pages 1–155 provide the bibliographic locator. The
workbook metadata does not supply a cell-by-cell paper page/table map, so this
package makes no independent-PDF-transcription claim. This scope is explicit
in the manifest and in the evidence amendment for source provenance. Frozen
Figshare API metadata and project-page snapshots under
`tests/test_diffpes/_reference_data/yeh_lindau_authority/` bind these
authority claims by SHA-256.

When using these data, cite both the original Yeh–Lindau paper and the
Regoutz-group digitisation named above.
