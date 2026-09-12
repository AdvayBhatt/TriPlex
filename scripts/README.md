# Script roles and data formats

## Submission forecasts

Run from the repository root. `run_selected_pipeline.py` is the selected real-data submission forecast; `run_pipeline.py` is the reference predictor and synthetic judge demonstration. Both read `data/raw/C*_Phenotype_Data_V2.csv` and extracted genotype CSVs below `data/raw/genotypes/C1` and C2. They preserve candidates independently of target outcomes and enforce plot budgets. See the [root README](../README.md) for commands and outputs.

The selected runner imports shared mathematical helpers from `experiments/`. It does not run the six imported scripts below or consume their dataset caches.

## Retrospective data-audit imports

These files were selectively imported from data-audit. They preserve earlier analyses and are **not January 2008 forecast entry points**. Their numerical methods are retained so historical results are not silently changed.

| Script | Historical role |
|---|---|
| `audit_data.py` | Describes observed phenotype, genotype and environment files, including 2008 outcomes |
| `build_genotypes.py` | Reads original genotype ZIPs and writes legacy `geno_C*.npz` caches |
| `build_dataset.py` | Cleans observed outcomes and builds legacy `dataset_C*.npz` targets/PCs |
| `decompose.py` | Between/within-family and environmental diagnostics |
| `selection_index.py` | Retrospective yield/moisture selection-index analysis |
| `recommend.py` | Separate flat-ridge recommendations, historical location effects and location-yield intervals |

### Required input format

The imported genotype reader expects these original ZIP paths and internal member names, rather than the forecast runner's extracted layout:

```text
data/raw/ImputedC1Populations.zip -> ImputedPopulationsC1/C1.<population>_Imputed.csv
data/raw/ImputedC2Populations.zip -> ImputedPopulationsC2/C2.<population>_Imputed.csv
data/raw/C1_Phenotype_Data_V2.csv
data/raw/C2_Phenotype_Data_V2.csv
data/raw/environmental_features.csv   (for analyses that read it)
```

Run `--help` for each script's data, cache and output options. Legacy caches normally live in `data/processed/`, separately from `data/processed/forecast_cache`. Load only caches you generated or reviewed: old NPZ readers enable pickle for object arrays.

### Known boundaries

- `build_dataset.py` cleans all observed years before identifying target candidates. Missing 2008 yields, outliers and observation counts therefore affect candidate eligibility; it cannot construct an operational January roster.
- `prepare_markers` uses all supplied rows for missingness and allele-frequency filtering. Only imputation and scaling use the training mask. Forecast and grouped-CV statistics from this path do not have the selected pipeline's preprocessing guarantees.
- `recommend.py` selects a fraction of lines, not a site-plot budget; it forces at least one line and can include extra ties in its exported advancement flag. Its intervals and location forecasts are from its own flat model, not the selected C1/C2 policy.
- Several data-audit experiments were not imported, including `rank_lines.py`, `demo.py` and environmental/neural experiments. Do not follow the old branch's complete command chain here. Main's judge command is `run_pipeline.py --sample`.

Use these scripts to investigate historical findings. Before repurposing them for forecasts, separate outcomes from eligibility, fit every transform inside training folds, and replace the line quota with tested plot allocation.
