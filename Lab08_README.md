# MDI3003 — Experiment 08: Agricultural Predictive Analytics

Course: MDI3003 — Advanced Predictive Analytics · Instructor: Dr. Durgesh Kumar, Assistant Professor (Senior), SCOPE
Fall Semester 2026–2027

Rice yield regression (core) and an independent crop-label classification extension, built and executed per the Experiment 08 student manual's Appendix A reference implementation.

## What this is

- **Core task:** predict Rice yield (t/ha) for district–season–harvest-year observations in West Bengal using only `state, district, season, year` (no post-harvest fields), evaluated with a chronological train/validation/locked-test split.
- **Extension:** an independent soil/environment crop-label classifier (`N, P, K, temperature, humidity, ph, rainfall → label`), never mixed with the yield data.

All numbers in the report and CSVs come from one real, executed run of `lab08.py` on the canonical files in `data/` — nothing is synthetic or hand-typed.

## Repository layout

```
Lab08.ipynb              Executed notebook — runs both tasks end to end, displays every table/figure
Lab08_Report.docx/.pdf   Full write-up: problem contract, data governance, methodology,
                                    evidence tables, figures, error analysis, limitations, discussion
Lab08_Validation_Results.csv
Lab08_Test_Results.csv
Lab08_Error_Analysis.csv
Lab08_README.md          This file

lab08.py                           Reference pipeline (regression + classification, shared CLI)
config.json                        Regression core config
config_classification.json         Classification extension config
requirements.txt                   Minimum compatible package versions
requirements-tested.txt            Exact versions used for this run

data/
  rice_canonical.csv               Rice-only canonical file, West Bengal, 1997-2019 (1,279 rows)
  crop_labels_canonical.csv        Crop-recommendation canonical file (2,200 rows, 22 balanced classes)

models/
  core/*.joblib                    Trained pipelines for the regression core (median, ridge_trend, tree, forest,
                                    selected_bundle)
  classification/*.joblib          Trained pipelines for the classification extension

figures/
  core/*.png                       Training EDA, validation comparison, actual-vs-predicted, residuals
  classification/*.png             Training label counts, pH distribution, validation comparison, confusion matrix

artifacts/
  core/                            config.json, versions.json, split_manifest.csv, rolling_origins.csv,
                                    year_robustness.csv, test_predictions.csv, acceptance.json, selection.json
  classification/                  Same categories plus per_class.json (precision/recall/F1 per crop label)

outputs/                           Raw run directories exactly as lab08.py wrote them (core/ and classification/),
                                    kept alongside the consolidated models/figures/artifacts/ above for traceability
```

## How to reproduce

```bash
pip install -r requirements.txt

# Regression core
python lab08.py --config config.json --stage validate
python lab08.py --config config.json --stage test

# Classification extension
python lab08.py --config config_classification.json --stage validate
python lab08.py --config config_classification.json --stage test
```

Or open `Lab08.ipynb` and run all cells — it calls the same two commands and then loads/display every saved table and figure. `validate` must run before `test` for each config (the code refuses to re-run validation into an existing `outputs/<name>/selection.json`, and `test` checks the saved data/config hashes and split manifest before unlocking).

## Data provenance

| | Core (regression) | Extension (classification) |
|---|---|---|
| Content | District-wise, season-wise Rice production statistics, West Bengal, 1997–2019 | Soil/environment sensor readings mapped to a recommended crop label, 22 balanced classes |
| Rows | 1,279 (from 326,846 raw multi-crop, multi-state rows) | 2,200 |
| Canonical file SHA-256 | `4d7e02e28fff608bc13f405e3ab3db141b009d9a9c949c0b06fb74324a10b706` | `3de03aeebd2998d862b9e8eb35c87ef5239b8a92128550c0bde583d2aee9d76b` |
| Target | `yield_t_ha` (t/ha, provided directly, verified finite and non-negative) | `label` (categorical crop name) |
| Predictor whitelist | `state, district, season, year` — production/area excluded (target leakage) | `N, P, K, temperature, humidity, ph, rainfall` |

Full source description, licence caveats, and the exact leakage/unit/duplicate-key audit are in the report's Sections 1–3.

## Headline results

**Regression core** (chronological split: train 1997–2015, validation 2016–2017, locked test 2018–2019):

| Model | Validation MAE (t/ha) | Test MAE (t/ha) | Test R² |
|---|---|---|---|
| forest (selected) | 0.258 | 0.364 | 0.402 |
| ridge_trend | 0.281 | — | — |
| tree | 0.308 | — | — |
| median (baseline) | 0.520 | 0.617 | -0.637 |

**Classification extension** (stratified 60/20/20 split):

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|
| forest (selected) | 0.993 | 0.991 | 0.991 |
| logistic | 0.973 | — | — |
| majority (baseline) | 0.004 | 0.045 | 0.004 |

Full validation tables, the locked-test comparison, five largest error cases with cause hypotheses, rolling-origin stability checks, and the confusion matrix are in the report.

## Limitations (see report Section 12 for the complete list)

- Fixed teaching hyperparameters; not tuned for optimality.
- Core regression uses only categorical location/time features — no weather, soil, or irrigation data — so single-year shocks cannot be anticipated.
- Single-state (West Bengal) scope; not validated for other states or unseen districts.
- Rolling-origin and two-year robustness checks are descriptive stability evidence, not confidence intervals.
- The classification extension's near-perfect scores reflect a compact, augmented benchmark with undocumented label-generation provenance and should not be read as real-world crop-recommendation accuracy.
- Retrospective demonstration only — no operational deployment or agronomic decision-making is supported by these results.

## Environment

```
python==3.12.3
```
See `requirements-tested.txt` for exact package versions used to produce these results.
