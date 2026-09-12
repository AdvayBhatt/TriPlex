# Performance investigation — 12 September 2026

## Result

The new fixed models improve the original corrected pipeline on the same 2008 candidate roster, target, scoring subset and plot budgets. Both now beat the zero/mean baseline on RMSE. The improvement comes from better regularization, parental information for C1, and making predictions relative to the other planned lines at each site. No future yield, harvest trait or actual growing-season weather enters prediction.

**These are exploratory results, not an untouched test.** The first selection experiment already examined 2008 before the roster-adjustment follow-up was proposed. The follow-up choices still used only 2004–2006 forecast results, then were checked on 2007 and 2008. That protects the computation boundary, but does not erase adaptive researcher exposure. No presentation or submission work was done in this investigation.

| 2008 retrospective metric | C1 previous | C1 selected | C2 previous | C2 selected |
|---|---:|---:|---:|---:|
| Pearson correlation | 0.1646 | **0.1799** | 0.1048 | **0.1818** |
| RMSE, adjusted bu/acre | 10.0533 | **9.6511** | 11.0102 | **10.2588** |
| Mean-baseline RMSE | 9.8109 | 9.8109 | 10.3806 | 10.3806 |
| Actual allocation gain, adjusted bu/acre | +1.5167 | **+3.3296** | +1.3262 | **+2.6758** |
| Candidates retained | 7,432 | 7,432 | 8,536 | 8,536 |
| Candidates scorable | 7,397 | 7,397 | 8,519 | 8,519 |
| Advanced lines | 762 | 757 | 890 | 861 |
| Allocated plots | 3,994 | 3,994 | 4,698 | 4,698 |

Allocation gain is the observed mean field-adjusted yield of scorable selected lines minus the scorable cohort mean. It is not guaranteed future genetic gain or a causal economic return. All candidates are selected before outcomes are scored. Different numbers of advanced lines reflect their different site-bundle costs under the same plot budget.

Nominal 90% prediction intervals cover 90.81% of scorable C1 outcomes and 87.63% of C2 outcomes. C2 undercoverage remains a limitation. The intervals use 2007 errors from the fixed selected method, not 2008 errors, and are not confidence intervals for pure breeding value.

## 1. Explaining the gap to `data-audit`

The full available `ref/` context remains the starting point: January 2008, historical 2001–2007 data, known candidate genotypes and planned locations, separate pools, and a broad-acre screening decision. The target and eligibility rules from `CODE_ALIGNMENT.md` were kept fixed throughout model comparisons. The raw data and existing outputs were preserved.

Eight diagnostic variants per pool isolated numerical precision, marker coverage, rare-marker filtering, inclusion of 2000, legacy ID treatment and outcome cleaning. These diagnostics used already-inspected 2008 outcomes to explain a known discrepancy, **not to select the historical model winner**.

| Single diagnostic change from the current method, unless noted | C1 r | C2 r |
|---|---:|---:|
| Reconstructed current control | 0.1646 | 0.1048 |
| Float64 instead of float32 | 0.1646 | 0.1048 |
| Allow 50% missing markers rather than 20% | 0.1692 | 0.1399 |
| 50% missingness + training MAF >= 0.01 | 0.1708 | 0.1372 |
| Add 2000, retaining the current marker rules | 0.1578 | 0.1088 |
| Revert ID inclusion behavior only | 0.1701 | 0.1048 |
| Legacy outcome cleaning only | 0.1641 | 0.1052 |
| Combined audit-like years/cleaning/marker rules, corrected IDs and train-only filters | 0.1721 | 0.1468 |

The 80% coverage requirement introduced in my rewrite was unvalidated and discarded useful markers, especially for C2. Higher precision was not the explanation. Restoring an ID parsing error was not adopted just because dropping records changed accuracy. The corrected parser recovers legitimate historical records, which changes the C1 training cohort substantially.

An additional forensic reproduction used the previously generated, reviewed legacy caches. It reproduced the prior plain ridge at **0.18618/0.15030** on its own target. On the current common target these scores are 0.18539/0.15096. Restricting its marker filters to training data changes them to 0.18107/0.14683 on the common target. Thus joint training/target marker filtering contributed about 0.0043/0.0041 to that diagnostic score difference. The remaining C1 difference also involves its 60,207-line legacy training cohort versus the larger recovered cohort and associated target aggregation. Do not attribute all of the original gap to the omitted parental blend.

Evidence: `outputs/model_comparison/baseline_diagnostics.csv` and `audit_bridge.csv`. The forensic script is intentionally separate and requires locally reviewed legacy caches; it is not used by the selected predictor.

## 2. Historical model comparison and the first failure

Eleven configurations were compared on forecasts for 2004, 2005 and 2006, each fitted using strictly earlier seasons starting in 2001:

- The prior corrected model as a control.
- Four marker ridge penalties: 3,000; 30,000; 300,000; 3,000,000, with the 50%/MAF rule fitted only on training data.
- Three two-component models: family mean predicted from midparent genotypes with penalties 100, 1,000 or 10,000, plus within-family deviations predicted from progeny markers with penalty 30,000.
- Three equal-weight raw-yield-unit blends of those two-component models with the 30,000 marker ridge.

Selection prioritized average annual allocation gain over budget-matched random allocation, requiring positive gain in at least two of three origins; lower average RMSE broke ties. Each random comparison used 500 random candidate orders with the exact same full-site-bundle allocation rule and budget. These are conditional comparisons on observed trial outcomes, not p-values for future generalization.

The first frozen choices were the parent-10,000 model for C1 and marker ridge-3,000,000 for C2. Calibration slopes/intercepts were fitted from development forecasts with equal weight per origin. That first attempt was **not** an unqualified success:

- C1's parent model improved 2007 correlation and selection gain, but its calibrated RMSE remained worse than the mean. The predeclared adoption gate rejected it. Calibration of the control alone improved 2008 RMSE to about 9.700 without changing rankings.
- C2's heavily regularized marker model passed the 2007 gate but lost 2008 ranking correlation, falling to about 0.087. Its calibration improved RMSE relative to the old model, but still did not beat the mean.

Those results are preserved in `comparison_results.json`, `frozen_choices.json`, the confirmation files and `C*_candidate_rankings.csv`. They were not silently replaced by the later result.

## 3. Correcting a mismatch between predictions and the fixed target

The target is already field-relative. A model prediction can nevertheless carry a family/cohort offset that does not correspond to the field-relative quantity being scored. The known planting roster permits this correction **without any field outcomes**:

```text
raw prediction for line i = p_i
planned-site reference for site e = mean(p_j for all planned lines j at site e)
field-relative forecast for line i = mean over its planned sites of (p_i - site reference)
```

The complete pre-allocation roster defines the reference. Do not recenter on the selected shortlist afterward. This is not an absolute yield estimate or pure GCA estimate; it remains a forecast of the existing relative-yield proxy. It assumes the supplied roster represents the planned comparison set. Changing the roster can change predictions, including those for unchanged lines, because their comparison group changes. This dependency uses information available at decision time.

The follow-up compared three transformations of the already-defined historical model predictions: unchanged, a single plot-weighted cohort offset, and planned-site adjustment. The model choices and a nonnegative calibration slope capped at one were determined from 2004–2006 only. Both selected slopes were one. The winning choices were:

- **C1:** parental family-mean model, penalty 10,000, plus marker-predicted within-family deviations, penalty 30,000; then planned-site adjustment.
- **C2:** marker ridge, penalty 3,000,000; then planned-site adjustment. A parental model was tested but was not the selected C2 method.

The correction changes no outcome, scoring subset, candidate eligibility or budget. It uses no observed 2008 trait. It supplies the model with a consequence of the known planting plan that the previous predictor ignored.

The selected models also improved the later historical 2007 check:

| 2007 check | C1 control | C1 selected | C2 control | C2 selected |
|---|---:|---:|---:|---:|
| Correlation | 0.0804 | 0.1764 | 0.2027 | 0.3166 |
| RMSE | 11.3979 | 10.0911 | 9.9580 | 9.5024 |
| Mean-baseline RMSE | 10.2438 | 10.2438 | 9.7386 | 9.7386 |
| Allocation gain | -0.6126 | +2.2441 | +1.6646 | +2.6706 |

The full development and later-check results are in `roster_development_metrics.json` and `roster_results.json`. Development scores are used for selection and therefore optimistic as estimates of future performance. The follow-up was motivated after earlier results were seen; neither the later 2007 revisit nor 2008 is advertised as an untouched evaluation.

## 4. Runnable selected models

The original `scripts/run_pipeline.py` and `outputs/forecast_2008/` remain as the reference. The new runner fits only the selected models, rather than rerunning the search:

```sh
# Prediction only; no target evaluation by default. Use a new output folder.
python scripts/run_selected_pipeline.py --output-dir outputs/my_selected_run

# Explicit retrospective evaluation after saving predictions
python scripts/run_selected_pipeline.py --evaluate --output-dir outputs/my_evaluated_run

# Optional complete reference roster and explicit per-pool plot cap
python scripts/run_selected_pipeline.py --candidates candidates.csv --plot-budget 4000 --replicates 2 --output-dir outputs/my_roster_run
```

Policy: `model_configs/selected.json`. The full-data runner uses the existing runtime dependencies and expanded data layout. It has no dependency on the legacy audit cache or development-result files. It requires the full data; the unchanged original runner retains the synthetic judge mode. No presentation materials were changed.

The final verified outputs are in **`outputs/selected_forecast_verified/`**. Each pool includes complete rankings, the complete reference roster, the selected plot plan, numeric model parameters saved without pickle, historical calibration residuals, a data/model report and optional retrospective outcomes. The manifest records policy, parameters and source hashes. Output folders must be new and empty to preserve prior results.

`selected_model.py` fits the selected estimators independently using scikit-learn's ridge implementation, whereas the search reused matrix sufficient statistics across penalties. The independent estimator reproduces the selected experiment's predictions and allocation. Parent-missing cases fall back to the line-marker model; missing line markers can still use available parental information. Prediction uncertainty under those missing-data patterns is not separately calibrated, so those cases require review; all supplied 2008 candidates have line DNA.

## 5. Verification and limits

**21 regression tests passed.** The tests retain the original decision-boundary checks and add independent ridge-path equivalence, train-only filter behavior, historical selection/calibration rules, parent/line-model behavior, numeric-model save/reload, planned-roster adjustment and zero-budget random-comparison handling.

A complete selected-pipeline test deletes future yields, alters harvest values, adds future weather values and uses an external outcome-free roster. Rankings, intervals and allocations remain exactly unchanged; the prediction-only run emits no target evaluation. Site adjustment is also tested for duplicate-roster invariance, absence of outcome dependence and a zero plot-weighted mean over the full reference roster.

The full-data verifier imports no model implementation. It rebuilds the marker and parent predictions directly from saved coefficients and cached genotype calls, independently reconstructs the planned-site adjustment, recomputes interval endpoints, checks all candidate identities and plot budgets, and compares the result with the earlier experiment. Both pools pass. Verification commands:

```sh
python -m unittest discover -s tests -v
python tests/verify_selected_run.py outputs/selected_forecast_verified
```

The gains are observed on this dataset, not a guarantee of future breeding value. Family dependence, incomplete outcomes, unbalanced trials, tester confounding and the reconstructed planned roster remain important. The metric still measures relative trial performance, not absolute yield or stability. C2 interval coverage is below nominal. A future claim of generalization needs genuinely new outcomes or a prospectively frozen evaluation; repeated reanalysis of 2008 cannot supply that.

The current work is local performance research plus a separately runnable selected predictor; the earlier method remains available for comparison.
