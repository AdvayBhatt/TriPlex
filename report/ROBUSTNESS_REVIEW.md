# Model robustness review — 12 September 2026

The selected models remain reasonable working models, but the performance work is not exhausted. Their error improvement is better supported than their improvement in scarce-plot allocation gain. C2's within-family prediction and both pools' uncertainty for sparsely tested lines deserve attention next. No trained model, policy, original result, raw dataset, branch or submission material was changed in this audit.

The reference constraints remain January 2008 prediction from historical 2001–2007 records and decision-time information, separate breeding pools, and useful advancement under a plot cap. This review tests the existing field-relative yield target. It does not establish pure GCA, absolute yield or causal genetic gain. The reconstructed roster must represent information available at the decision, rather than an outcome-dependent reconstruction of who survived to harvest.

## Chronological selection

For outer year 2006, select among the existing 33 model/adjustment configurations using only forecasts from 2004–2005. For outer year 2007, use only 2004–2006. Keep the existing mean allocation-gain-over-random selection criterion, require at least two positive origins, and estimate the calibration slope only from earlier forecasts. The base forecasts themselves were trained only on seasons before their forecast year. Both outer checks select the current C1 parental model and C2 heavily regularized marker model, with site adjustment and slope one.

| Pool / outer year | Reference gain | Selected gain | Reference RMSE | Selected RMSE | Zero-baseline RMSE |
|---|---:|---:|---:|---:|---:|
| C1 / 2006 | 3.470 | 3.057 | 11.072 | 10.456 | 10.655 |
| C1 / 2007 | -0.613 | 2.244 | 11.398 | 10.091 | 10.244 |
| C2 / 2006 | 2.023 | 4.493 | 10.770 | 10.517 | 10.770 |
| C2 / 2007 | 1.665 | 2.671 | 9.958 | 9.502 | 9.739 |

All yields/errors are in adjusted bu/acre. C1 does not improve allocation gain in every year. The two later checks support the chosen configurations, but these are only two historical outer origins. The configuration library and target adjustment were developed after later outcomes had already been inspected. This chronological replay cannot erase that adaptive exposure or become a pristine nested prospective experiment. No new model family or hyperparameter was chosen in this review.

## Paired family uncertainty in 2008

Resample entire populations, with replacement, 2,000 times. Apply the same family multiplicities to both methods, preserving missing outcomes and using the complete candidate-family universe. Recalculate weighted evaluation metrics for the fixed predictions and fixed allocation decisions. This preserves within-family dependence. The resulting percentile ranges are conditional on this season's sites, fitted models and allocations; they do not incorporate refitting, adaptive model selection or uncertainty over new seasons. Families can also share sites, so these are not joint family-and-site confidence guarantees.

| Quantity | C1 estimate [95% bootstrap range] | C2 estimate [95% bootstrap range] |
|---|---:|---:|
| Selected gain over cohort | 3.330 [1.059, 5.559] | 2.676 [-2.120, 6.027] |
| Gain improvement over reference | 1.813 [-0.674, 4.370] | 1.350 [-2.767, 4.481] |
| RMSE difference from reference | -0.402 [-0.755, -0.057] | -0.751 [-1.366, -0.171] |
| RMSE difference from zero baseline | -0.160 [-0.378, 0.035] | -0.122 [-0.240, -0.022] |

Negative RMSE differences favor the selected model. Both gain-improvement ranges include zero. In particular, beating all 500 random allocations in the earlier experiment did not establish that C2's gain transfers reliably across families. C1's small RMSE advantage over the zero baseline is also uncertain under this resampling.

As a separate shared-site sensitivity check, independently reconstruct each target from distinct finite line-site means, fields with at least 20 lines, and lines with at least two sites. Delete each observed site in turn and recompute line outcomes, retaining the fixed forecasts and selection decisions. Newly unscorable lines leave evaluation only. Both pools retain positive selected gain and lower RMSE than the reference after every single-site deletion:

| Leave-one-site-out range | C1 | C2 |
|---|---:|---:|
| Selected gain | 2.889 to 3.656 | 2.370 to 3.041 |
| Gain improvement over reference | 1.601 to 2.429 | 1.041 to 1.729 |

Thus no single observed site explains the advantage. This is a sensitivity range, not a bootstrap confidence interval, and does not test simultaneous correlated site failures or a new season. A full joint family/site and model-refitting uncertainty analysis remains outstanding.

## Where the gain comes from

Subtract each family's observed mean from its lines' outcomes and predictions to measure within-family correlation. This is retrospective diagnosis only: observed family outcomes are never predictors. Decompose selected gain exactly into the change in observed family means and the change in deviations from those means.

| Diagnostic, 2008 | C1 | C2 |
|---|---:|---:|
| Within-family correlation, reference | 0.0847 | 0.0921 |
| Within-family correlation, selected | 0.1586 | 0.0844 |
| Selected gain: family-mean component | 0.712 | 2.147 |
| Selected gain: within-family component | 2.618 | 0.528 |
| Families represented in advancement / all candidates | 54 / 71 | 38 / 86 |

C2's overall improvement mostly comes from differences between families, while its within-family correlation slightly decreases. C1 improves its within-family ranking. These are components of the observed trial-relative target, not estimates of genetic variance components or pure breeding merit. The C2 pattern explains why family composition matters so much and motivates a targeted within-family experiment rather than an unrestricted search for more model types.

## Planting-roster sensitivity

Freeze raw model scores, every candidate's own sites, costs and the plot budget. Randomly omit comparator lines from the site-reference calculation, then score all original candidates against the resulting references. Repeat 100 times at omission probabilities 5%, 10% and 20%. Repeat separately with entire families omitted. If a site has no retained comparators, retain its complete-roster reference. Outcomes do not enter these perturbations.

This simulates incomplete comparator information, not moving actual planting sites or changing the feasible plan. Fractions are omission probabilities; actual omitted counts vary. Shortlist retention is the fraction of original advanced lines still advanced.

| 20% omission scenario | C1 median / worst retention | C2 median / worst retention |
|---|---:|---:|
| Independent lines | 97.9% / 96.7% | 98.6% / 97.6% |
| Whole families | 87.3% / 76.6% | 84.5% / 70.2% |

At 10% whole-family omission, median retention is 92.8% / 92.0%. The correction is stable to small scattered omissions but materially sensitive to missing families. A high overall rank correlation does not guarantee the same shortlist. Preserve the complete pre-allocation reference roster; do not recenter on the selected shortlist. Tests of actual site reassignment and out-of-distribution planting plans remain separate work.

## Prediction intervals

Existing intervals use 2007 residual quantiles, unchanged here. Nominal 90% coverage varies substantially across decision-relevant groups:

| 2008 group | C1 coverage (scorable n) | C2 coverage (scorable n) |
|---|---:|---:|
| All | 90.8% (7,397) | 87.6% (8,519) |
| Advanced | 87.9% (750) | 81.7% (858) |
| Planned at 1–3 sites | 79.1% (383) | 65.6% (305) |
| Planned at 4–6 sites | 91.0% (6,348) | 88.3% (7,519) |
| Planned at 7+ sites | 95.9% (666) | 90.1% (695) |
| At least one previously unseen site | 93.6% (5,119) | 88.6% (6,775) |
| All sites previously seen | 84.5% (2,278) | 83.8% (1,744) |

These overlapping descriptive groups are not independent comparisons or evidence that novel sites cause better coverage. One constant interval width per pool does not represent the differing uncertainty of outcomes averaged over different numbers of sites. Missing outcomes still limit evaluation: 35 C1 and 17 C2 candidates are not scored, including 7 and 3 advanced lines. Nothing here validates their interval coverage.

For the chronological checks, pooling only earlier forecast residuals yields 2006 coverage of 91.6% / 90.1% and 2007 coverage of 92.4% / 91.9%. This supports investigating historical-only interval calibration further, but does not justify choosing new interval parameters to hit 90% on already-inspected 2008.

## Next modeling work

1. Test a separate, more weakly regularized within-family component for C2 while preserving the strongly regularized family-level signal. Freeze a small candidate set and select using earlier forecast origins. Report family and within-family performance separately, alongside actual plot-budget gain. Improvement is a hypothesis, not promised.
2. Test uncertainty calibration by planned site count using earlier residuals, with minimum group sizes and fallback rules. Assess coverage on advanced lines as well as all candidates. Do not retune to 2008 coverage.
3. If results remain uncertain, extend the temporal audit with earlier origins and, where practical, refit-based family/site uncertainty. New prospective outcomes would still be needed for a clean generalization claim.

No performance ceiling has been established. The current results are useful working evidence, with specific weaknesses now identified.

## Reproduction and verification

New audit: `scripts/stress_test.py`. Final evidence: `outputs/robustness_final/`. The folder records the protocol, audit-source hash, hashes of recorded input evidence, chronological choices, bootstrap summaries, individual site deletions, roster perturbations and coverage groups. Intermediate `robustness*` folders are ignored local attempts; `robustness_final` is the completed run.

```text
python scripts/stress_test.py --output-dir outputs/new_robustness_run
python -m unittest discover -s tests -q
python tests/verify_robustness.py outputs/robustness_final
python tests/verify_selected_run.py outputs/selected_forecast_verified
```

All 26 regression tests passed. New tests exercise exclusion of outer-year outcomes from selection, paired metrics with missing outcomes, duplicate-invariant site-target reconstruction, outcome-free roster perturbation, bootstrap serialization and the exact gain decomposition. The independent verifier imports no audit/model functions: it reproduces the family bootstrap from aggregated sufficient statistics, checks key coverage groups and chronological choices, and checks recorded hashes. The separate full-data model verifier still reproduces both selected models, site adjustment, intervals and budgets. Nothing was committed, pushed, merged or switched to another branch.
