# TriPlex — TAMIDS Corn Breeding Track

TriPlex helps breeding managers choose **previously untested 2008 lines for field testing**, using information available in January 2008. It addresses the brief's broad-acre challenge in the 115 RM pipeline. C1 and C2 remain separate breeding pools; measured opposite-pool testcross yield is a proxy for breeding merit, not an identified pure GCA estimate.

The current pipeline forecasts field-adjusted yield advantage, reports uncertainty from historical forecasts, and assigns a limited number of plots to a ranked shortlist. Historical notebooks and the old `line_rankings_full.csv` are exploratory work, **not the current 2008 recommendations**.

## Start here

- **Current selected predictor:** `python scripts/run_selected_pipeline.py --output-dir outputs/my_forecast` (full data; prediction only). Add `--evaluate` for explicitly retrospective scoring.
- **Reference predictor and synthetic demo:** `python scripts/run_pipeline.py --sample --output-dir outputs/my_demo`.
- **Current findings:** [performance](report/PERFORMANCE_EXPERIMENTS.md), [robustness](report/ROBUSTNESS_REVIEW.md). Selected 2008 correlations are 0.180 / 0.182; improvements are exploratory and advancement-gain uncertainty remains substantial.
- **Mixed-model research:** [results and methods](report/MIXED_MODEL_REVIEW.md). Historical joint adjustment was fitted and tested, with and without tester effects. Neither variant passed the adoption rule; the selected policy remains unchanged. Reproduce with `experiments/mixed_model.py`.

| Folder | Purpose |
|---|---|
| `scripts/` | The two pipeline commands above |
| `src/` | Reusable forecast, selected-model and synthetic-data implementation |
| `experiments/` | Model comparisons, mixed-model research and robustness audits; see its README |
| `tests/` | Regression tests and independent artifact verifiers |
| `model_configs/` | Frozen selected-model policy |
| `ref/`, `data/` | Original reference documents and datasets |
| `report/` | Detailed findings and historical review evidence |
| `archive/source_snapshots/` | Compressed source used to verify historical results after reorganizing files |

The detailed instructions below describe the **reference** predictor. Its scores are not the selected model's scores. Earlier reports retain their original paths as dated records; implementation files formerly under `scripts/` now live in `src/` or `experiments/`.

## Run instructions

Tested with Python 3.13 and the versions in `requirements.txt`. From the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --sample
python -m unittest discover -s tests -v
```

Judge mode generates a deterministic **synthetic** dataset for two pools, eight years and new families each year. It uses the same loader, preprocessing, rolling forecasts, calibration, baselines, ranking and allocation as the full run. It includes 2008 lines with no outcomes, which remain in the shortlist. It needs no large raw files and completes in seconds. Synthetic accuracy is not evidence of real breeding performance. The provided 100-row, one-family sample is retained for inspection but cannot demonstrate new-family validation; the brief explicitly permits synthetic judge data.

```sh
# Full data, both pools; default training period is exactly 2001–2007
python scripts/run_pipeline.py

# Explicit budget: 4,000 plots PER POOL, two plots per candidate-location
python scripts/run_pipeline.py --plot-budget 4000 --replicates 2 --output-dir outputs/budget_4000

# Prediction without evaluating any 2008 outcomes
python scripts/run_pipeline.py --predict-only --output-dir outputs/prediction_only

# Optional external planting roster, with no outcome columns
python scripts/run_pipeline.py --candidates candidates.csv --predict-only --output-dir outputs/roster_forecast

# Optional sensitivity analysis; 2000 exists in the raw data but is excluded by default
python scripts/run_pipeline.py --start-year 2000 --output-dir outputs/sensitivity_2000
```

An external roster needs `LINE_UNIQUE_ID,LOC,CROSS`, one row per intended line-location. Include both clusters or specify `--clusters 1` / `--clusters 2`. Default candidates come from 2008 phenotype **metadata only**, treating its locations as the known planned planting roster. That is an explicit retrospective reconstruction assumption; an operational planting roster is preferable.

Output directories must be empty unless `--overwrite` is explicitly supplied. Existing historical rankings are never overwritten by the default commands. To repeat the demonstration, use `--sample --overwrite` or a new output directory. Run from the repo root because default paths are relative to it.

Full data must be expanded as:

```text
data/raw/C1_Phenotype_Data_V2.csv
data/raw/C2_Phenotype_Data_V2.csv
data/raw/genotypes/C1/C1.1_Imputed.csv
data/raw/genotypes/C2/C2.1_Imputed.csv
```

Nested directories below each genotype pool are also supported. ZIP archives must be extracted first. The environmental feature table is retained for exploration; this model handles environments through historical site-year adjustment and does not use its growing-season weather predictors. Dependencies do not include notebook-only packages or R.

## Method and decision boundary

1. Read the references first; see `report/DAY1_REVIEW.md` for the complete inventory and data audit. Keep historical records from 2001–2007 and construct all 2008 candidates independently of yield availability.
2. Normalize numeric IDs, including leading zeros and known `.0`, `.1` or `#1` suffix forms. Export raw-to-normalized aliases. This treats suffixes as identifier-format aliases; it is a documented data assumption. Never guess corrupted alphanumeric IDs. Drop unresolvable historical IDs with an audit trail; fail on unresolvable candidate IDs. Conflicting DNA records for the same canonical identity are excluded rather than arbitrarily choosing the first.
3. Validate marker panels by name and reorder consistently. Accept only the provided `-1, 0, 1` coding and missing values; exclude parent rows. Content-hashed, non-pickle genotype caches avoid reparsing all 999 CSVs on every run.
4. Collapse duplicate line-site-year outcome records by their mean. Keep observed finite yields, fields with at least 20 distinct lines, and training/scoring lines with at least two locations. Subtract the mean of each site-year field, then average adjusted yield across sites for each line. Training and validation/target outcomes are adjusted **separately**. No yield-based outlier thresholds are selected from future outcomes.
5. Fit one additive marker ridge model per cluster. Marker missingness filtering (at least 80% training coverage), mean imputation and standardization use training lines only. Constant markers are removed. Missing or wholly uninformative candidate DNA receives an explicit phenotypic-only fallback, retaining the candidate. The default fixed ridge penalty is 30,000, inherited as an exploratory starting point from the audited approach, not claimed to be optimal or independently selected by this validation.
6. Forecast 2005 from 2001–2004, 2006 from 2001–2005, and 2007 from 2001–2006. Assert that forecast populations do not occur in training. Every population in these data belongs to one season. This tests the task's combination of new populations and future seasons; a random row split does not.
7. Form central 90% prediction-error intervals using earlier rolling-forecast residuals only, separately for marker predictions and phenotypic fallbacks. The first validation year has no interval. The final 2008 forecast uses residuals from all three historical origins. Fewer than 20 source-specific errors means no interval and a review flag. These are empirical **prediction intervals for realized adjusted line means**, not confidence intervals for pure breeding value. Family dependence, shift and the final refit prevent a claim of exact 90% coverage.
8. Refit on 2001–2007, save predictions and a planting plan, then optionally score against 2008 outcomes in a separate output. Target yields, moisture, height, test weight, lodging and actual 2008 weather cannot affect the rankings or allocation. Prior exposure to 2008 results means all reported 2008 performance remains retrospective/exploratory.

The score is in bushels/acre **relative to the site's contemporaneous cohort**, averaged over observed sites for evaluation. It is not an absolute 2008 yield forecast, a forecast of stability, a moisture-adjusted economic return, or a causal response to advancing the line. Unbalanced family placement can affect field centering. Tester/family confounding and unmodeled genotype-by-environment interactions remain limitations.

## Baselines and evaluation

The brief requests phenotypic BLUP and environmental means. Both are reported on the same adjusted-yield scale:

- A phenotypic-only random line-intercept model is fit by REML after field centering. A genuinely unseen line with no modeled relationship has a zero random effect, so its prediction is the fitted intercept. Variance estimates condition on estimated field means and are not a joint field/line variance decomposition.
- The environmental-mean baseline predicts zero yield advantage after field adjustment.

Neither constant baseline can rank unseen lines, so its correlation is undefined rather than falsely reported as zero. Their RMSE still provides a valid comparison. Genomic ridge can have positive ranking correlation while losing on RMSE; inspect both. No pedigree BLUP or parental-genotype blend is silently included.

Reports contain Pearson and Spearman correlations, RMSE, within-population centered correlation, top-decile gain, the actual allocation's observed gain, and empirical interval coverage. Pearson uncertainty resamples whole populations (200 replicates). Top-decile gain is computed within the scorable subset, whereas advancement is decided among **all** candidates before scoring. Missing outcomes can bias the evaluable subset; gains are descriptive, not guaranteed future genetic gain.

**Selected model** — two-stage family/within-family ridge for C1; flat marker ridge for C2; planned-site adjustment. Primary deliverable: `outputs/layout_selected_verified/`. Independent verification passed for both pools.

| Result | C1 | C2 |
|---|---:|---:|
| Candidates ranked | 7,432 | 8,536 |
| Pearson r (retrospective 2008) | 0.180 | 0.182 |
| Model RMSE, adjusted bu/acre | 9.651 | 10.259 |
| Environmental-baseline RMSE | ~9.811 | ~10.381 |
| Actual advancement gain, adjusted bu/acre | +3.330 | +2.676 |
| Lines advanced / plots allocated | 757 / 3,994 | 861 / 4,698 |

The selected model beats the environmental-mean baseline on RMSE; the reference predictor does not. Advancement gain is roughly 2× higher than the reference predictor. See `report/PERFORMANCE_EXPERIMENTS.md` for the comparison.

---

**Reference predictor** — flat ridge, no family stratification, no planned-site adjustment. Verified default full-data run (2001–2007 training; retrospective 2008 scoring):

| Result | C1 | C2 |
|---|---:|---:|
| Candidates ranked / scored | 7,432 / 7,397 | 8,536 / 8,519 |
| Pearson r | 0.1646 | 0.1048 |
| Model / environmental-baseline RMSE | 10.0533 / 9.8109 | 11.0102 / 10.3806 |
| Actual advancement gain, adjusted bu/acre | +1.5167 | +1.3262 |
| Nominal 90% interval coverage | 93.16% | 88.84% |
| Lines advanced / plots allocated | 762 / 3,994 | 890 / 4,698 |

**Reference predictor RMSE loses to the simple baseline.** Historical allocation gains for 2005/2006/2007 were -0.30/+3.47/-0.61 bu/acre for C1 and +0.94/+2.02/+1.66 for C2. Even the highest-ranked lines (`C1.427.36`, `C2.368.110`) have 90% prediction intervals spanning zero and include new sites. Treat the output as a screening shortlist for breeder review. The final cached full run took about 95 seconds locally with four numerical threads; this is not a hardware-independent runtime guarantee.

See `report/CODE_ALIGNMENT.md` for verified real-data results, checks, judging coverage and remaining limitations. Do not reuse the old README's accuracy, stability, runtime or moisture-composite claims. Do not present the audit branch's “3× published benchmark” as an apples-to-apples comparison.

## Outputs and commercial use

Default folders are `outputs/judge_forecast/` and `outputs/forecast_2008/`. Each cluster exports:

| File | Purpose |
|---|---|
| `C1_rankings.csv` / `C2_rankings.csv` | Every candidate, predicted advantage, source, rank, intervals, review flags, advancement and plot cost |
| `C*_plot_plan.csv` | Selected line-location combinations and replication; totals match the budget report |
| `C*_validation_YYYY.csv` | Historical forecast predictions, evaluation truth and errors |
| `C*_retrospective_evaluation.csv` | Optional scored 2008 subset; never used to revise saved predictions |
| `C*_calibration.csv` | Historical residuals used for final uncertainty intervals |
| `C*_marker_model.npz` | Marker names, retained-marker mask, imputation/scaling and fitted coefficients; no pickle |
| `C*_report.json` | Data issues, baseline parameters, fit dates, validation, allocation and optional retrospective metrics |
| `C*_id_aliases.csv` | Raw-to-canonical ID provenance |
| `C*_historical_sites.csv` | Descriptive historical site yields; not a future weather forecast |
| `run_manifest.json` | Parameters, dependency versions, source hashes, input hashes and elapsed time |

The provisional policy uses **10% of the listed candidate plots in each pool**, rounded down, with one plot per line-location. This is a team assumption, not a supplied budget. In predicted-rank order, advance a line if its complete listed site bundle fits the remaining budget; otherwise skip it and consider the next line. Ties break by canonical ID. Export unused capacity. This is a transparent feasible heuristic, not a claim to solve a global resource-allocation optimum. It can prefer a lower-ranked cheaper bundle after a higher-ranked bundle no longer fits.

Breeding managers should inspect the intervals and `needs_review` flags (new sites, DNA fallback or unavailable intervals) before adopting the shortlist. The prototype has no minimum family representation, seed availability, tester availability, fixed site overhead or diversity constraint. It does not allocate a reserve for exploration, prove superiority to the midparent, or model stability. Those need explicit breeding policy and/or additional data. `--plot-budget` is a per-pool cap, not a shared total across pools.

## Repository and branch boundaries

The reference/demo path is `scripts/run_pipeline.py`, backed by `src/triplex.py` and the synthetic generator. The selected full-data predictor is `scripts/run_selected_pipeline.py`. Notebooks are retained as Day 1 exploration; see `notebooks/README.md`. Dhanush's public `data-audit` branch was reviewed separately at `51d39d95d3cab7d311bcb0f157f07544d4203eec`; it has not been merged or edited by this work. This implementation adopts the cross-population marker-learning idea and corrects the decision boundary explicitly.

Generated data, full rankings, models and review caches stay local. Do not upload the large raw datasets. Review and stage individual intended files; avoid blanket staging of `report/`, which also contains review evidence and extracted reference material. No commit, push or branch switch is required to run or inspect this workflow.
