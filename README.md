# TriPlex -- Precision Agriculture Hackathon 2026, Track 1

**Team:** Advay Bhattacharya, Dhanush (Texas A&M)
**Challenge:** Broad-Acre Performance Prediction (C1 heterotic pool, corn breeding)

---

## Problem Statement

A commercial corn breeding program is entering 2008 with reduced field-plot capacity. From a pipeline of inbred lines tested across environments from 2001-2007, only the top performers can advance. Each wrong advancement decision costs a full field season and delays genetic gain by 2-3 years. The business question is straightforward: **which lines do you advance, and how confident are you?**

We build an end-to-end pipeline that ingests raw phenotype, environmental, and genomic data and produces a ranked advance list with uncertainty bounds, a composite selection index that balances yield against harvest moisture, and a stability score that identifies lines with broad vs. environment-specific adaptation.

---

## Run Instructions

### Judge mode (fast, no large files required)

```bash
# From repo root
pip install -r requirements.txt
python scripts/run_pipeline.py --sample
```

Output appears in `outputs/line_rankings_sample.csv` in under 1 second. The sample slice is 100 phenotype rows and 100 genomic rows committed to the repo under `data/raw/sample_data/`.

### Full data mode (HPRC recommended)

```bash
python scripts/run_pipeline.py
# or specify clusters explicitly:
python scripts/run_pipeline.py --clusters 1 2
```

Reads `data/raw/C{n}_Phenotype_Data_V2.csv` and the matching imputed genomic files per population. Output goes to `outputs/line_rankings_full.csv`. Full data: 77,352 lines ranked, ~15 minutes on a single CPU core.

---

## Solution Overview

```
data/raw/
  C1_Phenotype_Data_V2.csv   (141 MB, gitignored)
  environmental_features.csv
  genotypes/C1/ImputedPopulationsC1/C1.{pop}_Imputed.csv  (one per population)
  sample_data/               (committed -- 100-row slices for judge mode)
        |
scripts/run_pipeline.py   -- self-contained end-to-end pipeline
        |
outputs/line_rankings_*.csv  -- ranked advance list with composite index
```

We chose **Challenge 1 (Broad-Acre Performance Prediction)** as the primary framing. The key design choice is population-stratified Ridge regression with line-level cross-validation, which maps directly to G-BLUP (the industry-standard genomic selection model) while keeping the implementation auditable and runtime manageable.

---

## Technical Approach

### Data structure

Each line is identified by `LINE_UNIQUE_ID` in the format `C{cluster}.{population}.{line}` (e.g., `C1.1.191`). Phenotype rows represent a single line-environment observation (line x location x year). Genomic files are per-population matrices with rows indexed by zero-padded 11-digit IDs (e.g., `00000000191`); parent rows (`PID...`) are excluded.

Join pipeline (implemented in `run_pipeline.py`):

1. Parse `LINE_UNIQUE_ID` via regex `C(\d+)\.(\d+)\.(\d+)` to extract cluster, population, line integer.
2. Group phenotype rows by (cluster, population).
3. For each population, load the matching imputed genomic file. Convert progeny row IDs to integers by stripping leading zeros. Filter to progeny rows only.
4. Optionally join environmental features on (YEAR, LOC).
5. Fit model, compute rankings, collect results.

### Model

**Algorithm:** Ridge regression (L2 penalty), equivalent to G-BLUP at genomic scale. At the typical per-population size of ~150 lines and ~300 SNP columns, Ridge regression with an appropriate alpha grid is the most stable linear estimator available.

**Alpha grid:** `[100, 1000, 10000, 100000]` -- tuned for the genomic scale where n ~ 150 and p ~ 300. Values below 100 are effectively unregularized at this scale. Alpha is selected by inner cross-validation via `RidgeCV`.

**Feature pipeline per row:**
- SNP marker dosage values (population-specific subset, clipped to valid dosage range [-1, 2])
- Environmental features (climate + soil, joined by year and location)
- Mean imputation for missing values (biologically neutral at mean dosage)
- Standard scaling before Ridge

**Cross-validation:** 5-fold GroupKFold with `groups=LINE_UNIQUE_ID`. All observations for a line are held out together. This is the correct structure for the breeding prediction problem: the model must predict a line's performance from its genomic profile, not from co-observed environment data. Out-of-fold predictions are clipped to ±3 SD of the population's observed yield to prevent extrapolation.

### Outputs per line

| Column | Description |
|--------|-------------|
| `GCA_pred` | Mean cross-validated predicted yield (bu/acre) across environments |
| `GCA_obs` | Mean observed yield across environments (training reference) |
| `stability` | `1 / (1 + CV)` where CV = std(YLD) / |mean(YLD)| across environments; higher = more broadly adapted |
| `n_env` | Number of environment-year observations for this line |
| `pred_lo` / `pred_hi` | GCA_pred ± 1 population RMSE (prediction interval) |
| `MST_obs` | Mean harvest moisture across environments (lower = commercially preferred) |
| `composite_score` | GCA_pred minus moisture penalty (see below) |
| `composite_rank` | Rank by composite score (used for advance list) |
| `yield_rank` | Rank by GCA_pred alone (reference) |
| `advance` | True for top 10% of lines by composite rank |

### Composite selection index

```
composite_score = GCA_pred - 5.0 * (MST_obs - fleet_mean_MST)
```

The moisture penalty of 5.0 bu/acre per unit reflects the drying cost economics: a 1-point drop in harvest moisture saves ~$0.04/bu in drying costs; at ~125 bu/acre average yield, that is approximately 5 bu/acre-equivalent. Lines with missing MST receive zero moisture adjustment (they are not penalized for missing data). The advance list is the top 10% by composite rank.

### Stability score

`1 / (1 + CV)` where CV is the coefficient of variation of observed yield across all environments a line was tested in. Score of 1.0 means perfectly consistent yield everywhere; lower scores indicate G×E interaction. Lines with only one environment observation receive 0.5 (unknown stability, treated as neutral). Stability is reported as a secondary filter but does not enter the composite rank -- the advance list is driven by the composite score.

---

## Results

### Full data run (77,352 lines, C1 + C2 pools, 499 populations)

| Model | RMSE (bu/acre) |
|-------|----------------|
| Baseline (LOC+YEAR group mean, in-sample) | 20.17 |
| Ridge CV (5-fold line-level, out-of-fold) | 18.19 |

The Ridge model reduces prediction error by **10% over the baseline** (group mean), measured out-of-fold (honest). Advance list: 7,735 lines (10% budget, by composite rank). Full rankings: `outputs/line_rankings_full.csv`.

### Sample mode run (judge-reproducible)

37 lines ranked on the 100-row sample slice in under 1 second. Top 4 by composite rank advance. Results: `outputs/line_rankings_sample.csv`.

---

## Commercial Recommendations

Use `composite_rank` to drive advancement decisions, not raw yield rank. The composite index penalizes wet lines (high MST) that cost more to dry -- a 3-point moisture advantage is economically equivalent to ~15 bu/acre of additional yield, which moves a line's effective rank substantially.

Practical advance criteria we recommend:

1. **Composite rank:** Primary sort. Advance the top 10% by composite score.
2. **Stability filter:** For seed supply decisions, prefer lines with `stability > 0.85` (broadly adapted). Lines with `stability < 0.60` carry high G×E risk and should advance only to targeted environments, not broad deployment.
3. **Prediction interval:** Lines where `pred_lo` falls below 120 bu/acre (a rough commercial floor) carry elevated downside risk and warrant a second field year before broad release even if their composite rank is high.
4. **MST priority:** If drying capacity is constrained at the conditioning facility, sort the advance list by `MST_obs` ascending within composite-rank tiers to manage logistics.

The rankings in `outputs/line_rankings_full.csv` are ready to drop into the field advancement spreadsheet as-is, with `advance = True` as the go/no-go column.

---

## Constraints and Limitations

**Leakage caveat.** GroupKFold by LINE_UNIQUE_ID holds out a full line at a time, which is the correct CV structure for genomic prediction. However, sibling lines in the same population (related breeding material) may share marker haplotypes, so cross-validated R² is still upwardly biased relative to true out-of-population prediction. The RMSE reported above is honest within the CV structure; R² should not be used to benchmark against public genomic selection literature without accounting for within-population relatedness.

**Incomplete genomic coverage.** Not all phenotyped lines have a matching row in the genomic file (particularly early-generation material from 2001-2003). These lines are excluded from rankings. Future work: impute missing lines using family-average SNP profiles.

**Environmental gap in sample mode.** The 20-row environmental sample does not overlap fully with the 100-row phenotype sample by year and location. In full mode the overlap is near-complete, and environmental features are the second most important predictor group after SNPs.

**No G×E interaction modeling.** The current model treats environments as additive effects via the feature matrix. A reaction norm or factor analytic G×E model would improve accuracy for location-specific recommendations, at significantly higher implementation cost.

**Temporal generalization.** The model is trained on 2001-2007 and predicts 2008. Year-to-year weather shifts and phenological changes are not modeled. Prediction intervals should be interpreted as within-training-distribution bounds; they do not capture novel-year risk.

**Alpha grid boundary.** The RidgeCV alpha grid caps at 100,000. For very small populations (< 50 lines) the optimal alpha may exceed this bound, resulting in a slightly underregularized model. Populations with fewer than 5 unique lines are skipped entirely.

---

## Repository Structure

```
TriPlex/
  data/
    raw/
      sample_data/          # committed -- 100-row sample slices for judge mode
        sample_C1_phenotype_100rows.csv
        sample_C1.1_100rows.csv
        sample_environmental_20rows.csv
      C1_Phenotype_Data_V2.csv   # gitignored -- 141 MB
      C2_Phenotype_Data_V2.csv   # gitignored -- 141 MB
      environmental_features.csv
      genotypes/C1/ImputedPopulationsC1/   # gitignored -- per-pop genomic files
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_modeling.ipynb
  scripts/
    run_pipeline.py    # end-to-end pipeline (primary judge entry point)
    run_modeling.py    # streaming Ridge model on preprocessed merged_full.csv
  outputs/
    line_rankings_full.csv     # 77,352 lines, full run
    line_rankings_sample.csv   # sample run, judge-reproducible
  ref/                         # hackathon brief and data documentation
  requirements.txt
  README.md
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
```

Install: `pip install -r requirements.txt`
