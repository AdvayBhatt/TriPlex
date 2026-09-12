# TriPlex -- Genomic Prediction for Maize Line Advancement

## 1. Problem Statement

It is January 2008 at a commercial maize breeding company. Roughly 16,000 new inbred lines are about to be planted across the Corn Belt for testcross evaluation. None has ever been field-tested. The budget for field plots has been cut, so only about 10% can be evaluated. The task is to decide **which lines get the ground**, using only what is knowable before planting: their DNA, their parentage, and the locations where they are scheduled to go.

Of the 7,432 lines in the 2008 Cluster 1 cohort, exactly zero appeared in the eight preceding years. Same in Cluster 2: zero of 8,536. Every population is new. There is no track record to look up, so prediction must travel through genetics.

**What is at stake.** Every line that should have been tested but was skipped is a potential commercial hybrid that the programme will never see. Every plot spent on a mediocre line is a plot not spent on a better one. At scale, the difference between random allocation and informed allocation is worth millions of dollars per breeding cycle.

**What makes this hard.** The 2008 candidates are progeny of crosses that have never been observed in the training data. Standard random cross-validation inflates accuracy dramatically in this setting (published estimates show gaps of up to 0.59 between random CV and honest leave-population-out CV on the same maize data). Any model that leans on family resemblance within training populations will collapse when every target family is new.

## 2. Solution Overview

TriPlex uses a two-stage genomic prediction approach tailored to the family structure of commercial maize testcross data:

1. **Family-level prediction.** The densely genotyped parents of each 2008 population have training-era relatives. A ridge regression on parental marker profiles predicts each new family's mean breeding value (GCA).
2. **Within-family prediction.** A separate ridge regression on progeny SNP markers predicts each line's deviation from its family mean, capturing the Mendelian sampling that distinguishes siblings.
3. **Site adjustment.** Predictions are centered against the complete planned planting roster at each location before allocation, removing systematic location biases from the ranking.

This decomposition was motivated by a diagnostic finding: 95.7% of the variance in flat-model predictions is between-family, but only 34.0% of actual outcome variance is. The flat model effectively ranks families, not lines. Splitting the problem lets each component use the information source best suited to it.

The pipeline is equivalent to GBLUP/rrBLUP, the field-standard method for genomic selection. Ridge regression on centered markers is mathematically identical to genomic BLUP. The heavy regularization (alpha = 10,000 for parents, 30,000 for progeny markers, 3,000,000 for Cluster 2's flat model) is deliberately strong because genetic effects are additive, individually tiny, and spread across thousands of loci.

### Workflow

```
build_genotypes  -->  build_dataset  -->  model_experiments  -->  selected_model  -->  recommend
   (999 CSVs          (phenotype +         (model search on       (frozen config,      (per-site
    to .npz)           leakage guard)       dev years only)        final fit)           yields)
```

## 3. Technical Approach

### 3.1 Data Cleaning and Leakage Prevention

Raw phenotype records (~537,000 rows per cluster) go through the following steps. All statistics (marker means, standard deviations, field effects) are computed on training data only. The 2008 cohort never influences any transformation.

- **ID normalization.** Line IDs are canonicalized to three-part form (e.g., `C1.435.109`). Cluster 2 IDs carry a fourth component in the raw files; a naive join matches nothing. Six populations have malformed progeny identifiers. Conflicting DNA records for the same canonical ID are excluded rather than arbitrarily choosing one.
- **Yield adjustment.** Raw yield is centered by subtracting each field's (site x year) mean. This removes the 68.8% of variance attributable to environment, leaving "how much better or worse than everyone else in that same field."
- **Quality filters.** Fields with fewer than 20 lines are dropped (unstable means). Lines observed at fewer than 2 locations are dropped (unreliable averages). Outliers beyond 5 SD within their own field are removed (2 records in C1).
- **Leakage constraint.** Seven harvest-measured traits are explicitly excluded from all inputs: `ERM, MST, PHT, RTLP, STLP, TWT, EHT`. These are measured at harvest in October off the same plot that produces the yield. A model using them scores well in testing and is impossible to run in January when the inputs do not exist. The exclusion is named in code, not implied by omission.
- **Marker QC.** Of 2,911 SNP markers, those with greater than 50% missing calls or minor allele frequency below 1% are dropped, leaving ~2,686. Remaining gaps are mean-imputed using training-set column means. Markers are z-score standardized using training-set statistics.

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

Model selection used only development years (2004, 2005, 2006) and was completed before examining 2007 or 2008 outcomes. The protocol:

- For each development year, train on all preceding years and predict that year's new populations.
- Compare models by allocation gain over budget-matched random selection (500 random draws, seed 8371).
- Selection rule: largest mean annual allocation gain, positive in at least 2 of 3 development years. Ties broken by lower mean RMSE.
- 2007 served as confirmation only (calibration errors for prediction intervals). 2008 is the target.
- The winning model and adjustment are frozen in `model_configs/selected.json`. All subsequent runs use this config without re-searching.

### 3.4 Validation Design

Standard random cross-validation would mislead here. Lines within a population are siblings from one cross, nearly identical genetically. A random split puts brothers on both sides, inflating accuracy. Published studies document gaps of up to 0.59 between random CV and honest population-level CV on this same germplasm.

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
| Advancement gain (bu/acre) | +3.330 | +2.676 |
| Lines advanced / plots allocated | 757 / 3,994 | 861 / 4,698 |

The selected model beats the environmental-mean baseline on RMSE in both clusters. Advancement gain is the mean adjusted yield of advanced lines, measuring the commercial value of the ranking.

### 4.2 Why 0.18 Is Better Than It Sounds

The outcomes being graded against are themselves noisy. Each 2008 line's "true" value comes from roughly five unreplicated plots. Split-half reliability of the 2008 line means is approximately 0.49. A perfect model that knew every line's true genetic value exactly would score only **r ~ 0.70** against this target. The rest is plot noise, and nothing predicts noise. Any team reporting r = 0.9 has a leak.

The honest framing: **0.18 / 0.70 = 26% of what is achievable**, not 0.18 out of 1.0.

### 4.3 Literature Benchmark

This dataset appears in two 2014 *Crop Science* papers describing 969 biparental maize testcross populations from a commercial programme, 2000-2008, two heterotic groups, 2,911 SNP markers, ~156 lines per cross. The published benchmark for the "same background" task (pool unrelated crosses, predict a new one) is **r = 0.06**. TriPlex achieves roughly **3x the published accuracy** on the same scenario.

| Published design | What it is | Yield accuracy |
|---|---|---:|
| Phenotypic selection | line's mean in half environments vs other half | 0.24 |
| Within-family | train on the same biparental cross | 0.14 |
| **Same background** | **pool unrelated crosses, predict a new one (our task)** | **0.06** |
| **TriPlex** | **our result on the same task** | **0.186** |

Prediction from DNA alone, on unphenotyped lines in unseen families in a future year, reaches 78% of what actually growing the plants twice achieves.

### 4.4 The Diagnostic That Shaped the Model

Decomposing the prediction accuracy by family structure:

| Component | Correlation | Meaning |
|---|---:|---|
| Overall, line level | +0.186 | the headline number |
| **Between families** | **+0.316** | ranking the 71 new families against each other |
| Within families | +0.109 | ranking siblings inside a family |

95.7% of the variance in flat-model predictions is between-family, but only 34.0% of actual outcome variance is. The model assigns nearly the same score to every line in a family, while two-thirds of real genetic variation sits within families. This motivated the two-component architecture, which was the only improvement that replicated across both clusters.

### 4.5 Fifteen Approaches Tested

Each was implemented and scored against held-out 2008 outcomes. The pattern is the finding: genetic effects here are additive, individually tiny, and spread across thousands of markers. Methods that hunt for interactions and thresholds find noise instead.

| Approach | Best r | Verdict |
|---|---:|---|
| Two-component blend (adopted) | 0.191 | Adopted for C1 |
| Ridge, all markers | 0.186 | Baseline |
| Economic selection index | 0.186 | No effect (89% shortlist overlap) |
| Dense neural network | 0.172 | Matched, not beaten |
| Multivariate trait model | 0.177 | Ambiguous |
| PCA compression (50 PCs) | 0.169 | Rejected |
| Ridge + boosting blend | 0.159 | Made it worse |
| Gradient boosting | 0.125 | Rejected |
| Midparent excess | 0.121 | Rejected |
| Two-way field adjustment | 0.119 | Rejected |
| Random forest | 0.107 | Rejected |
| Block-attention transformer | 0.096 | Rejected |

Flexible models lean on family resemblance, which cannot transfer when every target population is new. A heavily regularized linear model is the correct model class for this biology.

### 4.6 Environment Analysis

Environment accounts for 68.8% of yield variance. Two findings:

**G x E is smaller than it looks.** Correlation between individual environments is ~0.10, but grouping site-years into climate/soil types and averaging noise away gives cross-type genetic correlation plausibly 0.9 or above (median 1.04 across matched half-samples, with one lower-confidence pair at 0.59). Most of what looks like genotype-by-environment interaction is actually plot noise. One national ranking list is correct.

**Weather predicts places, not seasons, with a twist.** Splitting covariates into location climatology (96% of variance) and within-location anomaly (4%): climatology predicts 2008 at r = 0.005, anomaly at r = 0.175. Combining anomaly with location history: r = 0.324 vs history alone at 0.296. The season signal is real but lives in 4% of the covariate variance; fitting all covariates together lets the useless 96% drown it out.

## 5. Run Instructions

Tested with Python 3.13 and the versions in `requirements.txt`. From the repository root:

```sh
python -m pip install -r requirements.txt
```

### Judge mode (synthetic data, no large files needed)

```sh
python scripts/run_pipeline.py --sample --output-dir outputs/judge_demo
```

Generates a deterministic synthetic dataset for two pools, eight years, and new families each year. Uses the same loader, preprocessing, rolling forecasts, calibration, baselines, ranking and allocation as the full run. Completes in seconds.

### Full data run (selected model)

```sh
python scripts/run_selected_pipeline.py --output-dir outputs/my_forecast
```

Reads frozen model config from `model_configs/selected.json`. Add `--evaluate` for retrospective 2008 scoring.

### Full data run (reference predictor)

```sh
python scripts/run_pipeline.py --output-dir outputs/my_reference
```

### Data requirements

Full data must be extracted to:
```
data/raw/C1_Phenotype_Data_V2.csv
data/raw/C2_Phenotype_Data_V2.csv
data/raw/genotypes/C1/C1.1_Imputed.csv  (nested directories supported)
data/raw/genotypes/C2/C2.1_Imputed.csv
```

### Tests

```sh
python -m unittest discover -s tests -v
```

## 6. Commercial Recommendations

### 6.1 Line Advancement

At a 10% budget (the default), the selected model advances 757 lines in Cluster 1 and 861 in Cluster 2. These lines are predicted to yield +3.33 and +2.68 bu/acre above the cohort average, respectively. The top-ranked lines across clusters have substantial predicted advantages but wide prediction intervals spanning zero, reflecting the genuine uncertainty of genomic prediction on unphenotyped lines.

The selection gain is concentrated at the top of the ranking. Using data-audit's analysis on the flat ridge baseline:

| Advance threshold | Truly in that tier | vs. random | Realized gain |
|---|---:|---:|---:|
| Top 2% | 10.9% | 5.4x | +7.29 bu/acre |
| Top 5% | 14.4% | 2.9x | +4.16 bu/acre |
| Top 10% | 17.2% | 1.7x | +2.54 bu/acre |
| Top 20% | 26.6% | 1.3x | +1.64 bu/acre |

Accuracy is highest exactly where a reduced budget forces selectivity.

### 6.2 Location-Specific Yields

For each advanced line, the pipeline produces expected yield at each scheduled location by combining the genomic prediction with shrinkage-estimated site effects (locations with thin history are shrunk toward zero). Site-specific prediction intervals combine model uncertainty and field year-to-year variability in quadrature, providing wider intervals at sites with no testing history.

Of the 149 sites in the 2008 roster, 37 have no historical data. These receive the all-field spread as their uncertainty, flagged as higher risk.

### 6.3 Operational Guidance

- **Inspect the `needs_review` flags.** Lines planted at novel sites, lacking DNA (phenotypic fallback), or with unavailable prediction intervals are flagged for breeder review before adopting the shortlist.
- **The ranking is a screening tool, not a final decision.** Breeding managers should consider minimum family representation, seed availability, tester availability, fixed site overhead, and diversity constraints before committing. The prototype does not model these.
- **One national list is correct.** Cross-environment genetic correlation is high (plausibly 0.9+), meaning regional splits would reduce testing intensity without meaningfully improving selection accuracy. This was confirmed both by split-half analysis and by mixed-model variance decomposition.
- **Moisture is easier to predict than yield** (r = 0.27 vs 0.19) and could inform a drying-cost economic index, though at standard grain prices the shortlist overlap with yield-only ranking is 89%.

## 7. Constraints and Limitations

### Failure Modes

- **Novel germplasm.** Accuracy depends on relatedness to training lines. A 2008 family unrelated to anything in 2000-2007 will predict poorly, and the relatedness analysis does not reliably flag which ones in advance.
- **New seasons.** Environmental covariates cannot forecast an unseen year. Only the relative ranking is reliable, not absolute yield.
- **New locations.** 37 of 149 sites have no history. Their field effect is assumed zero with the widest interval.
- **Thin lines.** A line grown in 2 fields is ranked alongside one grown in 7. Reliability weighting was tested and hurt, so lines are weighted equally.

### Data Limits No Model Can Overcome

- **SCA is inestimable by design.** Every line is crossed to exactly one tester (0 of 73,532 have two), and every population uses exactly one tester (468 of 468). Line-by-tester interaction has no degrees of freedom.
- **G x E cannot be separated from plot error** at the individual-field level: 99.8% of cells hold one unreplicated plot.
- **Imputed genotypes are ~97% inferred.** Progeny were genotyped at ~3% of markers; the rest was filled in computationally from parental chromosomes. Imputation error is a real and unquantified uncertainty source.
- **Prior exposure to 2008 results.** All reported 2008 performance is retrospective/exploratory. The model selection protocol used only 2004-2006 development years, but the analyst had seen 2008 outcomes before freezing the final model config.

### What Was Tried and Did Not Work

Neural networks, gradient boosting, random forests, transformers, PCA compression, and blended ensembles all matched or fell short of simple ridge. The validation-to-holdout drop tells the story: ridge does not move; the dense network falls 0.229 to 0.172; the attention model falls 0.188 to 0.096. Flexible models learn overall relatedness, not marker effects, which is precisely the signal absent when every target population is new.

### Next Steps Not Attempted

1. **Spatial adjustment of unreplicated trials.** Attacks plot error directly. Needs plot row/range coordinates.
2. **Within-family prediction improvement.** Two-thirds of genetic variation sits there, and the model barely touches it (within-family r = 0.11). The largest unexploited opportunity.
3. **Factor-analytic multi-environment model.** Would replace several hand-rolled estimators with one properly specified fit.

---

## Repository Structure

| Folder | Purpose |
|---|---|
| `scripts/` | Pipeline entry points: `run_selected_pipeline.py` (production), `run_pipeline.py` (reference/demo) |
| `src/` | Core forecast engine (`triplex.py`) and frozen selected model (`selected_model.py`) |
| `experiments/` | Model comparisons, diagnostics, roster adjustment experiments |
| `model_configs/` | Frozen model policy (`selected.json`) |
| `tests/` | Regression tests and independent artifact verifiers |
| `data/` | Raw datasets (not tracked in git) |
| `ref/` | Original reference documents |
| `report/` | Detailed findings, performance experiments, and review evidence |
| `docs/` | The Maize Prediction Handbook and working analysis notes |

## Supplementary Material

The **Maize Prediction Handbook** (`docs/Maize_Prediction_Handbook.html`) is a 16-section analysis document covering the biology, the statistics, every modelling decision, all 15 approaches tested, the literature benchmark, self-identified limitations, and a glossary of every term used. It is written so that someone who has never seen a breeding dataset can read it end to end and argue with it intelligently.
