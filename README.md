# TriPlex -- Genomic Prediction for Maize Line Advancement

## 1. Problem Statement

It is January 2008 at a commercial maize breeding company. Roughly 16,000 new inbred lines are about to be planted across the Corn Belt for testcross evaluation. None has ever been field-tested. The default planning assumption is a budget of 10% of the original candidate-site plots in each pool. The task is to decide **which lines get the ground**, using only what is knowable before planting: their DNA, their parentage, and the locations where they are scheduled to go.

Of the 7,432 lines in the 2008 Cluster 1 cohort, exactly zero appeared in the eight preceding years. Same in Cluster 2: zero of 8,536. Every population is new. There is no track record to look up, so prediction must travel through genetics.

**What is at stake.** Every line that should have been tested but was skipped is a potential commercial hybrid that the programme will never see. Every plot spent on a mediocre line is a plot not spent on a better one. At scale, the difference between random allocation and informed allocation is worth millions of dollars per breeding cycle.

**What makes this hard.** The 2008 candidates are progeny of crosses absent from training. Random line splits can put siblings on both sides, overstating transfer to a new family. Validation must therefore test new populations.

## 2. Solution Overview

TriPlex targets broad-acre testcross performance within each pool, as allowed by the [project brief](ref/PRECISION_AGRICULTURE_HACKATHON_2026_TRACK1.pdf). It is a proxy for breeding merit, not a direct estimate of pure GCA or specific hybrid performance. C1 uses the following two-stage approach; C2 uses the flat ridge described in Section 3.2:

1. **Family-level prediction.** The densely genotyped parents of each 2008 population have training-era relatives. A ridge regression on parental marker profiles predicts each new family's mean breeding value (GCA).
2. **Within-family prediction.** A separate ridge regression on progeny SNP markers predicts each line's deviation from its family mean, capturing the Mendelian sampling that distinguishes siblings.
3. **Site adjustment.** Predictions are centered against the complete planned planting roster at each location before allocation, removing systematic location biases from the ranking.

Historical data-audit diagnostics found that flat-model predictions varied mainly between families, while observed outcomes also varied substantially within families. This motivated separate family-mean and within-family components for C1; those diagnostics are not a variance decomposition of the final selected model.

Marker ridge corresponds to genomic BLUP under matching relationship-matrix scaling; the combined C1 predictor is a two-component model. The heavy regularization (alpha = 10,000 for parents, 30,000 for progeny markers, 3,000,000 for Cluster 2's flat model) is deliberately strong because genetic effects are additive, individually tiny, and spread across thousands of loci.

### Workflow

```
Historical trials + candidate DNA/parents + complete planned roster
                              |
           scripts/run_selected_pipeline.py
            + model_configs/selected.json
                              |
    Rankings + 90% prediction intervals + site plot plans
                              |
        Optional separate retrospective evaluation
```

## 3. Technical Approach

### 3.1 Data Cleaning and Leakage Prevention

The selected and reference forecast runners separate historical outcomes from candidate metadata. Candidate eligibility does not depend on 2008 yield availability. Marker filters, imputation and scaling use training rows only; planned-site centering intentionally uses the complete candidate roster and its predictions, never its outcomes. These guarantees do not apply to the imported retrospective cache builders; see [script roles and limitations](scripts/README.md).

- **ID normalization.** Line IDs are canonicalized to three-part form (e.g., `C1.435.109`). Cluster 2 IDs carry a fourth component in the raw files; a naive join matches nothing. Six populations have malformed progeny identifiers. Conflicting DNA records for the same canonical ID are excluded rather than arbitrarily choosing one.
- **Yield adjustment.** Raw yield is centered by subtracting each field's (site x year) mean, estimating performance relative to other lines in that field. This is a trial-performance target, not a direct separation of genetic and environmental variance.
- **Training target filters.** Fields with fewer than 20 lines and lines observed at fewer than 2 locations are excluded from target estimation. Repeated line/site/year observations are averaged. These rules do not filter the forecast roster; the imported analysis path has separate outlier rules.
- **Leakage constraint.** Seven harvest-measured traits are explicitly excluded from all inputs: `ERM, MST, PHT, RTLP, STLP, TWT, EHT`. These are measured at harvest in October off the same plot that produces the yield. A model using them scores well in testing and is impossible to run in January when the inputs do not exist. The exclusion is named in code, not implied by omission.
- **Progeny marker QC.** Markers with greater than 50% training missingness or minor allele frequency below 1% are dropped. Retained counts depend on pool and training partition. Remaining gaps are mean-imputed and z-score standardized using training statistics; the parent component has its own preprocessing.

### 3.2 Model Architecture

**Cluster 1: Two-component family model** (`parents_10000`)
- Stage 1: Ridge(alpha=30,000) on progeny markers, predicting both raw adjusted yield and within-family deviation as a two-column response.
- Stage 2: Ridge(alpha=10,000) on parent markers, predicting family mean GCA.
- Combined prediction: family mean from parents + within-family deviation from progeny markers.
- Lines without valid parent data fall back to the flat progeny-only column.

**Cluster 2: Flat high-regularization ridge** (`flat_3000000`)
- Ridge(alpha=3,000,000) on progeny markers, predicting raw adjusted yield.
- The extreme regularization was selected because Cluster 2 responded better to heavy shrinkage than to the two-component decomposition during model selection.

**Site adjustment.** Both clusters apply planned-site centering: per-site mean predictions are subtracted from each line's prediction at each scheduled location, then averaged per line. This removes systematic location biases from the ranking before allocation.

### 3.3 Model Selection Protocol

Model selection scores used development years (2004, 2005, 2006), but analysts had already inspected later outcomes before freezing the final policy. All 2008 results are retrospective/exploratory, not an untouched prospective test. Restricting selection scores to development years does not erase that prior exposure. The protocol:

- For each development year, train on all preceding years and predict that year's new populations.
- Compare models by allocation gain over budget-matched random selection (500 random draws, seed 8371).
- Selection rule: largest mean annual allocation gain, positive in at least 2 of 3 development years. Ties broken by lower mean RMSE.
- The final runner uses 2007 forecast residuals for empirical prediction intervals and fits through 2007 to predict 2008. Historical training starts in 2001.
- The winning model and adjustment are frozen in `model_configs/selected.json`. All subsequent runs use this config without re-searching.

### 3.4 Validation Design

Random line cross-validation can overstate accuracy here because siblings from a cross can appear in both training and validation. Population-separated historical forecasts better match the new-family decision.

This pipeline enforces **leave-population-out validation**: all siblings stay on the same side of the split. Training populations are required to be disjoint from forecast populations as a hard constraint. This mirrors the actual task, where every 2008 family is new.

## 4. Results

### 4.1 Prediction Accuracy

Retrospective evaluation against held-out 2008 outcomes (field-adjusted yield, bu/acre):

| Metric | C1 | C2 |
|---|---:|---:|
| Candidates ranked | 7,432 | 8,536 |
| Pearson r (retrospective 2008) | 0.180 | 0.182 |
| Model RMSE (bu/acre) | 9.651 | 10.259 |
| Environmental-baseline RMSE | 9.811 | 10.381 |
| Realized allocation gain (bu/acre) | +3.330 | +2.676 |
| Lines advanced / plots allocated | 757 / 3,994 | 861 / 4,698 |
| Scorable candidates | 7,397 | 8,519 |
| Coverage of nominal 90% prediction intervals | 90.81% | 87.63% |

The selected model beats the environmental-mean baseline on RMSE in both clusters. Realized allocation gain is the observed mean adjusted yield of scorable advanced lines minus the scorable cohort mean. It is not a predicted score, guaranteed future genetic gain, or a causal economic return. C2 prediction intervals under-cover. See the [performance investigation](report/PERFORMANCE_EXPERIMENTS.md) and [robustness review](report/ROBUSTNESS_REVIEW.md).

### 4.2 Historical analyses and interpretation

The data-audit branch investigated family structure, environmental variation, moisture, and alternative models. Its flat-model correlation of 0.186 and experimental blend result of 0.191 use other analysis settings; they are not selected-model metrics. See the [historical handbook](docs/Maize_Prediction_Handbook.html) for that exploratory context and [script guide](scripts/README.md) for the partial integration boundary.

Sparse, largely unreplicated observations make yield targets noisy. Reliability estimates and comparisons to published studies depend on target construction, populations and validation design; they do not establish a universal accuracy ceiling or a controlled threefold improvement over the literature. The selected pipeline's evidence is the within-dataset comparison in Section 4.1, with its retrospective limitations.

## 5. Run Instructions

Tested with Python 3.13 and the versions in `requirements.txt`. From the repository root:

```sh
python -m pip install -r requirements.txt
```

### Judge mode (synthetic data, no large files needed)

```sh
python scripts/run_pipeline.py --sample --output-dir outputs/judge_demo
```

Generates a deterministic synthetic dataset for two pools, eight years, and new families each year. Runs the **reference predictor**, including loading, preprocessing, rolling forecasts, calibration, baselines, ranking and allocation. Completes in seconds; synthetic results do not reproduce the real-data metrics above. The selected parent/progeny model is separately exercised by the test suite.

### Full data run (selected model)

```sh
python scripts/run_selected_pipeline.py --output-dir outputs/my_forecast
```

Reads frozen model config from `model_configs/selected.json`. Add `--evaluate` for separate retrospective 2008 scoring. Add `--candidates path/to/roster.csv` to supply a complete decision-time roster with columns `LINE_UNIQUE_ID,LOC,CROSS`. `--plot-budget 1000` caps plots at 1,000 **per cluster**; `--replicates` sets plots per line-location. Choose a new, empty output directory for each run.

### Full data run (reference predictor)

```sh
python scripts/run_pipeline.py --output-dir outputs/my_reference --predict-only
```

The reference runner evaluates target outcomes by default; `--predict-only` skips evaluation. The selected runner evaluates only with `--evaluate`.

### Data requirements

Full data must be extracted to:
```
data/raw/C1_Phenotype_Data_V2.csv
data/raw/C2_Phenotype_Data_V2.csv
data/raw/genotypes/C1/C1.1_Imputed.csv  (nested directories supported)
data/raw/genotypes/C2/C2.1_Imputed.csv
```

Use the original phenotype files, whose year column is `YEAR_x`. The forecast runners cache genotype arrays in `data/processed/forecast_cache` and use four numerical threads by default. Full runs need the complete data and substantially more memory/time than judge mode. Weather/soil covariates are not inputs to the selected runner; site information enters through the planned roster. The imported analysis scripts require a different ZIP layout documented in [scripts/README.md](scripts/README.md).

### Tests

```sh
python -m unittest discover -s tests -v
```

The optional R integration test requires a locally configured R installation; the Python forecast runners do not require R. Historical `tests/verify_*.py` scripts require local full-data outputs and sometimes archived sources or experiment artifacts; they are not clean-checkout smoke tests.

## 6. Commercial Recommendations

### 6.1 Line Advancement

At the default **10% of candidate-site plots per cluster**, the selected model advances 757 C1 lines using 3,994 plots and 861 C2 lines using 4,698 plots. Their retrospective realized gains are +3.33 and +2.68 bu/acre; these are not their forecast scores. The allocator visits lines in rank order and selects each line's complete scheduled-site bundle if it fits the remaining budget. This is budget-feasible, not a knapsack optimum or an exact 10% quota of lines.

Inspect the [selected recommendation example](docs/selected_recommendations.md): ten advanced lines per pool with forecast scores, 90% prediction intervals and plot costs. The [CSV](docs/selected_shortlist.csv) and [provenance record](docs/selected_shortlist_provenance.json) provide a compact derived example without raw data. It is an excerpt, not the complete field plan.

### 6.2 Outputs and location plans

The selected runner writes `C*_rankings.csv` (all candidates, relative scores, intervals and decisions), `C*_plot_plan.csv` (selected line-location pairs and plot counts), `C*_candidate_roster.csv`, historical `C*_calibration.csv`, numeric `C*_model.npz`, pool reports and `manifest.json`. With `--evaluate`, it also writes `C*_retrospective.csv` after saving the forecast.

These are **location plot plans, not absolute location-yield forecasts**. The separate retrospective `recommend.py` estimates location yields using another model; its output must not be presented as predictions from the selected policy. Novel-site exposure is flagged, but the selected runner's intervals are not widened separately for each site.

### 6.3 Operational Guidance

- **Inspect the `needs_review` flags.** The selected runner flags novel-site exposure or missing line DNA. Missing-data uncertainty is not separately calibrated. Its intervals describe errors in relative trial performance, not confidence intervals for pure GCA or per-line stability.
- **The ranking is a screening tool, not a final decision.** Breeding managers should consider minimum family representation, seed availability, tester availability, fixed site overhead, and diversity constraints before committing. The prototype does not model these.
- **Broad-acre screening is the chosen scope.** The project brief permits it. The selected forecast does not establish that every line is stably adapted everywhere and does not provide per-line stability scores.
- **Midparent improvement remains a breeder criterion.** The scenario motivates it, but the selected output does not estimate a midparent contrast. Historical midparent flags from the imported model are not selected-model outputs.
- **Moisture is easier to predict than yield** (r = 0.27 vs 0.19) and could inform a drying-cost economic index, though at standard grain prices the shortlist overlap with yield-only ranking is 89%.

## 7. Constraints and Limitations

### Failure Modes

- **Novel germplasm.** Accuracy depends on relatedness to training lines. A 2008 family unrelated to anything in 2000-2007 will predict poorly, and the relatedness analysis does not reliably flag which ones in advance.
- **New seasons.** The selected model predicts relative performance and does not forecast actual growing-season weather or absolute site yield. Transfer to a new season remains uncertain.
- **New locations.** Novel-site candidates are flagged; the selected calibration does not establish site-specific coverage.
- **Roster assumption.** Retrospective trial metadata must represent the complete planting plan available in January. Operational use should supply an independently recorded decision-time roster.
- **Thin lines.** A line grown in 2 fields is ranked alongside one grown in 7. Reliability weighting was tested and hurt, so lines are weighted equally.

### Data Limits No Model Can Overcome

- **SCA is inestimable by design.** Every line is crossed to exactly one tester (0 of 73,532 have two), and every population uses exactly one tester (468 of 468). Line-by-tester interaction has no degrees of freedom.
- **G x E cannot be separated from plot error** at the individual-field level: 99.8% of cells hold one unreplicated plot.
- **Imputed genotypes are ~97% inferred.** Progeny were genotyped at ~3% of markers; the rest was filled in computationally from parental chromosomes. Imputation error is a real and unquantified uncertainty source.
- **Prior exposure to 2008 results.** All reported 2008 performance is retrospective/exploratory. The model selection protocol used only 2004-2006 development years, but the analyst had seen 2008 outcomes before freezing the final model config.

### Historical model experiments

Alternative models and preprocessing were explored in data-audit and in the main-branch investigations. Their reported results are historical comparisons, not evidence that flexible models can never work for this biology. The selected model is supported by the documented chronological comparisons and remains subject to uncertainty and prior outcome exposure.

### Next Steps Not Attempted

1. **Spatial adjustment of unreplicated trials.** Attacks plot error directly. Needs plot row/range coordinates.
2. **Within-family prediction improvement.** Two-thirds of genetic variation sits there, and the model barely touches it (within-family r = 0.11). The largest unexploited opportunity.
3. **Factor-analytic multi-environment model.** Would replace several hand-rolled estimators with one properly specified fit.

---

## Repository Structure

| Folder | Purpose |
|---|---|
| `scripts/` | Selected/reference forecast entry points and explicitly labeled retrospective imports; see [script guide](scripts/README.md) |
| `src/` | Core forecast engine (`triplex.py`) and frozen selected model (`selected_model.py`) |
| `experiments/` | Model comparisons, diagnostics, roster adjustment experiments |
| `model_configs/` | Frozen model policy (`selected.json`) |
| `tests/` | Regression tests and independent artifact verifiers |
| `data/` | Supplied samples; full raw datasets and caches are ignored |
| `ref/` | Original reference documents |
| `report/` | Performance experiments, robustness review, mixed-model research |
| `docs/` | Selected shortlist example/provenance and historical Maize Prediction Handbook |

## Supplementary Material

The [Maize Prediction Handbook](docs/Maize_Prediction_Handbook.html) preserves historical data-audit analyses, model experiments and background explanations. It is not the current submission specification. Use this README and the selected recommendation example for current model choices, outputs and metrics.
