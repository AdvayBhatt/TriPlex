# TriPlex

**Precision Digital Agriculture Hackathon 2026 -- Corn Breeding Track**

Texas A&M University team project. We integrate phenotypic, environmental, and genomic data from a commercial corn breeding program (2001-2007) to predict maize line General Combining Ability (GCA) and produce a ranked list of lines to advance into the 2008 field season.

---

## Problem Statement

A commercial corn breeding team faces budget cuts heading into January 2008. Field plot capacity is reduced, so they need data-driven recommendations for which inbred lines from the 115 Relative Maturity pipeline to advance. Every wrong advancement decision delays genetic gain by a full season and costs millions in lost yield improvement. We build an end-to-end prediction system that translates three years of multi-omics breeding data into a ranked, uncertainty-qualified line advancement list.

---

## Solution Overview

```
Raw Data (pheno + env + geno)
        |
  02_preprocessing.ipynb   -- merge pipeline, outputs data/processed/merged_*.csv
        |
  03_modeling.ipynb        -- baseline + Ridge regression + CV + uncertainty + rankings
        |
  outputs/line_rankings_*.csv   -- final advancement recommendations
```

We chose Challenge 1 (Broad-Acre Performance Prediction) as our primary framing, with stability metrics (prediction interval width) incorporated to flag high-risk lines.

---

## Data

The dataset covers the C1 heterotic pool across 499 populations and 2001-2007 growing seasons.

- `data/raw/C1_Phenotype_Data_V2.csv` (141 MB, gitignored) -- full C1 phenotype
- `data/raw/C2_Phenotype_Data_V2.csv` (141 MB, gitignored) -- full C2 phenotype
- `data/raw/environmental_features.csv` -- 86 climate and soil features by location-year
- `data/raw/genotypes/C1/` -- 499 imputed genomic files, one per population
- `data/raw/sample_data/` -- 100-row slices committed to git for judge mode

Join keys:

- Phenotype to environment via YEAR + LOC
- Phenotype to genomic via LINE_UNIQUE_ID format `C{cluster}.{population}.{line}`, matched to geno row index after stripping leading zeros

---

## Technical Approach

### Preprocessing (`notebooks/02_preprocessing.ipynb`)

1. Drop junk duplicate columns from the pre-merged phenotype file (Unnamed, projects_x/y, etc.)
2. Normalize YEAR column (YEAR_x renamed to YEAR)
3. Parse LINE_UNIQUE_ID into CLUSTER_ID, POP_NUM, LINE_NUM using regex `C(\d+)\.(\d+)\.(\d+)`
4. Left join phenotype to environment on (YEAR, LOC)
5. Loop over each (cluster, population) group, load the matching imputed geno file, filter to progeny rows only (11-digit zero-padded index, excluding parent PIDs), and left join on LINE_NUM
6. Output flat feature matrix to `data/processed/`

### Modeling (`notebooks/03_modeling.ipynb`)

**Baseline.** Location-year mean yield, computed per (LOC, YEAR) group. Captures pure environmental effects with no genomic information. Evaluated in-sample as a reference floor.

**Primary model.** Ridge regression (L2-penalized least squares) on the full feature matrix. Ridge regression is computationally equivalent to G-BLUP, the industry-standard genomic selection model. It handles the wide-matrix problem (more SNP columns than rows in any single population) by shrinking all marker effects toward zero proportionally, avoiding overfitting without discarding any marker.

Feature pipeline per observation:
- 7 phenotypic trait features (MST, PHT, EHT, TWT, RTLP, STLP, ERM)
- Up to 84 environmental features (climate and soil by location-year)
- 2912 SNP marker features (one per marker column, imputed to column mean for missing values)

Mean imputation for SNPs is biologically neutral since the mean SNP value approximates the heterozygous allele (near zero on the -1, 0, 1 scale).

Alpha (regularization strength) is tuned via inner 5-fold cross-validation using RidgeCV.

**Cross-validation structure.** We cross-validate at the line level rather than the row level. Each unique line may appear across multiple location-years; naive row-level CV would leak training information across folds. Holding out all observations for a line at once reflects the actual breeding prediction problem (predict a new line's performance from its genomic profile).

**Uncertainty quantification.** We bootstrap the cross-validation residuals (500 draws) to produce 90% prediction intervals per line. Lines whose lower prediction bound falls below the population median are flagged as high-risk regardless of their point estimate.

**GCA estimation.** We aggregate predictions to line level by averaging predicted yield across all environments the line was tested in. This is our GCA estimate. Lines are ranked by GCA and the top 20 low-risk lines are recommended for advancement.

---

## Model Validation (Sample Mode Results)

These results are from sample data (97 rows, 1 population, degraded env and SNP coverage). Full-data results replace these once the full run completes.

| Model | RMSE (bu/acre) | R² |
|-------|---------------|-----|
| Baseline (LOC+YEAR mean, in-sample) | 14.43 | 0.496 |
| Ridge regression (5-fold line-level CV) | 18.98 | 0.128 |

The baseline R² of 0.496 is inflated because it is evaluated in-sample. A cross-validated baseline would be substantially lower. The Ridge R² of 0.128 reflects prediction from phenotypic traits only since environmental and most SNP features are absent in the sample slice.

90% prediction interval mean width in sample mode: 57.56 bu/acre (expected to narrow significantly on full data).

---

## Commercial Recommendations

See `outputs/line_rankings_sample.csv` (sample) or `outputs/line_rankings_full.csv` (full run).

The output table includes for each line:

- `rank` -- overall rank by predicted GCA
- `GCA_pred` -- mean predicted yield across tested environments (bu/acre)
- `GCA_obs` -- mean observed yield (training data reference)
- `pred_lo_mean` / `pred_hi_mean` -- 90% prediction interval bounds
- `high_risk` -- True if the lower prediction bound falls below the population median
- `advance` -- True for the top 20 lines that are not high-risk

Lines with `advance = True` are our recommendations for the 2008 field season under the reduced plot budget constraint.

---

## Run Instructions

### Judge Mode (sample data, no large files needed)

```bash
pip install -r requirements.txt
# Open notebooks in order and run all cells with SAMPLE_MODE = True (default)
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
# Output appears in outputs/line_rankings_sample.csv
```

### Full Data Mode

Requires the large raw data files (gitignored). Recommended to run on HPC.

1. Set `SAMPLE_MODE = False` at the top of both notebooks
2. Run `02_preprocessing.ipynb` -- this loops over 499 geno files and may take several minutes
3. Run `03_modeling.ipynb` -- output goes to `outputs/line_rankings_full.csv`

---

## Constraints and Limitations

**Incomplete genomic coverage.** Not all phenotyped lines have a matching genomic file, particularly in early years. These lines receive NaN SNP features and their GCA estimates rely on phenotypic and environmental features only. Prediction uncertainty is higher for these lines.

**Environmental gap in sample mode.** The sample environmental slice does not overlap with the sample phenotype slice. In full mode, this gap closes. Environmental features are the second most important predictor group after SNPs.

**Cross-population SNP consistency.** Different population geno files may not share identical marker sets. If markers differ across files, the merged feature matrix will be sparse across populations. Future work should harmonize the marker panel or use within-population models.

**Temporal generalization.** The model is trained on 2001-2007 and predicts 2008. Weather patterns and genotype-by-environment interactions may shift year to year. Prediction intervals widen for environments or line types not well represented in the training window.

**No GxE interaction modeling.** The current model treats environment as additive features. A full genotype-by-environment interaction model (e.g., reaction norm or factor analytic model) would improve accuracy in location-specific recommendations.

---

## Repository Structure

```
TriPlex/
  data/
    raw/
      sample_data/      # committed -- 100-row slices for judge mode
      genotypes/C1/     # gitignored -- 499 imputed geno files
    processed/          # gitignored -- merged output from preprocessing
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_modeling.ipynb
  outputs/              # rankings CSVs
  ref/                  # hackathon brief, data guide, background docs
  src/                  # (reserved for utility modules)
  requirements.txt
  README.md
```
