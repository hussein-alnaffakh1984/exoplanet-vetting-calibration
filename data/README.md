# Data files (download separately)

These inputs are public but too large to store in the repository. Place the
files here (or set the `EXO_DATA_DIR` environment variable to their folder).

| File | Source | Notes |
| --- | --- | --- |
| `all_global.csv` | Macedo & Zalewski (2024), Mendeley Data DOI 10.17632/wctcv34962.3 | 2001 global-view flux columns + `label` |
| `koi_dr25_full.csv` | NASA Exoplanet Archive, Kepler DR25 KOI table | `koi_disposition`, seven features in `src/config.py`, plus `_err1`/`_err2` |
| `TOI.csv` | NASA Exoplanet Archive, TESS TOI catalogue | `tfopwg_disp` + `pl_*` columns; record the download date |

`flux_global.npz` is generated automatically on first run; do not commit it.
