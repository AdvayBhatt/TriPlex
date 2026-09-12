# Historical mixed-model experiment

> Historical investigation. Paths to full outputs, archived sources and earlier review notes refer to local research artifacts, not files shipped in the public checkout. For current commands and a public derived shortlist, see the [README](../README.md) and [selected example](../docs/selected_recommendations.md).

Completed 12 September 2026. **Retain the existing selected models for both pools.** The mixed-model experiment is implemented and independently checked, but neither variant earns replacement under the predeclared allocation-gain rule. No new 2008 evaluation was performed.

## C1 result

| Method | Mean 2004–2006 gain | Mean 2004–2006 RMSE | 2007 gain | 2007 RMSE | 2007 correlation |
|---|---:|---:|---:|---:|---:|
| Current selected | 4.198 | 10.871 | 2.244 | 10.091 | 0.1764 |
| Joint adjustment | 4.403 | 10.851 | 2.129 | 10.088 | 0.1776 |
| Joint + tester | 4.024 | 10.950 | 2.906 | 10.005 | 0.2169 |

The historical development rule chooses joint adjustment without tester. It improves average development gain and RMSE slightly, but loses allocation gain in 2007. Retain the current C1 model. The tester variant has a promising 2007 result, but choosing it because of that result would reverse the earlier model-selection decision after inspecting the check year. It remains a hypothesis for a subsequent explicitly exploratory study, not an adopted winner. Both confirmation variants are disclosed; the adoption decision uses only the preselected one.

## C2 result

| Method | Mean 2004–2006 gain | Mean 2004–2006 RMSE |
|---|---:|---:|
| Current selected | 5.027 | 10.466 |
| Joint adjustment | 4.788 | 10.447 |
| Joint + tester | 4.878 | 10.522 |

Joint adjustment slightly reduces average error but loses advancement gain. The tester variant loses both average gain and error. Neither passes development selection, so no additional C2 2007 mixed-model check or 2008 refit was needed. This follows the breeding decision objective rather than selecting whichever metric happens to improve.

## Complete development comparisons

Gain is the scorable selected-line mean minus the scorable cohort mean, under the same full-candidate budget policy. It is not causal genetic gain. Units are adjusted bu/acre.

| Pool / year | Current gain | Joint gain | Joint + tester gain |
|---|---:|---:|---:|
| C1 / 2004 | 3.193 | 3.421 | 3.533 |
| C1 / 2005 | 6.344 | 7.022 | 5.181 |
| C1 / 2006 | 3.057 | 2.766 | 3.358 |
| C2 / 2004 | 4.740 | 4.039 | 4.118 |
| C2 / 2005 | 5.847 | 5.751 | 6.103 |
| C2 / 2006 | 4.493 | 4.572 | 4.414 |

The tester results are genuinely variable across years; the earlier blanket argument that tester effects could not be modeled was too strong. Conversely, the successful fit and improved C1 tester result in 2007 do not establish a consistent predictive improvement. These findings do not exhaust joint genomic models, alternative first-stage targets/weights, or C2 within-family prediction.

## Why test this

The reference documents ask for preplant advancement of unseen lines, using historical trials and genotypes. Simple field-centering was a baseline choice, not a demonstrated optimal trial adjustment. In environments containing only a few families, field means can include substantial family composition effects. A joint historical model can estimate environmental and genetic contributions together, subject to its assumptions.

This experiment follows the mentor's direction without asserting that the informal formula determines a unique valid model. `GENERATION_NAME` contains F2, BC, DH and DH2, so “Gen” may mean generation. Here generation is explicitly a fixed categorical effect; genomic prediction occurs in the second stage. We do not simultaneously fit unrestricted fixed effects for every line plus genomic random effects.

## What the data supports and what it does not

In the raw 2001–2007 metadata there are 402 C1 families and 399 C2 families, with 930 and 870 location-year environments. A typical line has seven environments. The median environment contains two families. There are 48 and 41 recorded testers, some shared across many families, despite each family having at most one recorded tester.

The stricter existing yield/ID eligibility rules leave 401 C1 and 397 C2 families for the full historical trial-adjustment design. Of those, 27 and five families lack tester labels. These counts differ from the raw metadata inventory because the fit uses the same eligible historical line/site cohort as the current predictor.

Within each year, the family/environment comparison graph is connected from 2003 onward in both pools. C1 has two disconnected components in 2001 and three in 2002; C2 has two in each early year. Contrasts between disconnected groups rely on random-effects assumptions rather than direct shared-site comparisons. No family spans years. Population and year effects therefore also require the hierarchical interpretation, not arbitrary independent fixed effects for every population and year.

The data has no verified plot row/range/replicate layout. Duplicate records at the same line-site-year are averaged, not treated as independent replications. A separate line-by-environment random effect cannot be cleanly separated from residual variation with one retained entry mean per cell. We fit population-by-environment, which has multiple lines per cell. This is not a spatial field model.

## Model and prediction boundary

For each pool and forecast origin independently, train only on 2001 through the previous year. Fit raw historical yield entry means with:

```text
yield = year + generation
      + random environment
      + random population
      + random line
      + random population-by-environment
      + residual
```

A second variant adds a random tester effect. Each unknown tester receives its own family-specific unknown label, avoiding an invented common tester for all missing records. That is a nuisance-model convention, not an imputation of actual tester identity. For single-family testers, separation of tester and family contributions relies on variance assumptions learned from the shared-tester portion of the data. Even with a fitted hierarchy, these data do not establish pure tester-independent GCA.

REML fitting uses installed R 4.4.2 and `lme4` with the `bobyqa` optimizer. The library implements sparse mixed-model estimation; see [Bates et al., Fitting Linear Mixed-Effects Models Using lme4](https://www.jstatsoft.org/article/view/v067i01). Development fits disable additional numerical derivative checks to limit runtime. Saved optimizer return codes and singularity flags are checked; optimizer success is not a proof of a global optimum or correct model assumptions. Package-version startup warnings are recorded by the runtime; the synthetic integration test executes successfully in this installation.

For the training target, subtract fitted year/generation, environment and population-by-environment contributions from each historical yield. In the tester variant also subtract the fitted tester contribution. Average by line. Keep the family signal, line signal and line-average residual; do not feed only shrunken line BLUPs into another shrinkage model.

Then fit exactly the existing genomic configuration: C1 parental-family ridge plus within-family marker ridge, C2 heavily regularized marker ridge. Marker filtering, imputation, scaling, penalties and final planned-site adjustment remain unchanged. The isolated change is historical target construction.

This is **two-stage mixed-model adjustment followed by genomic regression**, not a single joint genomic REML model. It does not propagate first-stage adjustment uncertainty into the second stage, use a genomic relationship matrix within the mixed fit, or model soil/weather response. Historical environment labels are used; actual future weather and harvest traits are not inputs. Candidate generation/tester effects are not added back to the forecast.

## Comparison protocol

- Development origins: 2004, 2005 and 2006. Fit every stage using strictly earlier data.
- Keep the existing full candidate roster, field-relative evaluation target, genotype rules and 10% site-bundle plot budget.
- Select by mean annual allocation gain, requiring positive gain in at least two origins and mean RMSE no worse than the current selected predictor. Retain the current predictor unless the candidate also improves its mean allocation gain.
- Freeze the candidate before the 2007 check. Adoption requires greater 2007 allocation gain and no worse RMSE than the current selected model.
- Do not use 2008 to select or evaluate these variants. Historical years have nevertheless been inspected in earlier research, so this is still retrospective exploratory work.

The unchanged evaluation target measures realized trial-relative testcross performance. A model that removes tester contribution might better approximate a different breeding-merit target yet score worse here. These data do not supply clean independent GCA truth with which to resolve that distinction. Do not interpret a failed predictive test as proof that tester effects do not exist.

## Files and verification

The experiment is in `experiments/mixed_model.py` and `experiments/trial_adjustment.R`. Development artifacts are under `outputs/mixed_model/`; C1 confirmation is under `outputs/mixed_confirmation_C1/`. Each origin saves historical input rows, adjusted training targets, variance components, fixed effects, optimizer diagnostics, candidate predictions and evaluation metrics. `structure.json` records the separate connectivity check. `selection_summary.json` records both final choices; C1's pre-2007 frozen choice and confirmation decision are also retained. Both output folders contain execution-source snapshots checked against their protocol hashes; subsequent import-path and summary-command changes are not misrepresented as the executed source.

The new tests verify the historical metadata boundary, unknown-tester grouping, agreement of the second-stage genomic code with the existing selected predictor on identical targets, and an actual R fit on synthetic balanced trials. The R integration test checks that nuisance adjustment preserves within-family contrasts when every line has the same sites. `tests/verify_mixed_model.py` independently checks temporal separation, candidate/target identities, optimizer codes, forecast metrics and budget allocations against the saved historical evaluation cohort.

**All 30 regression tests pass.** All 14 real historical forecasts (12 development, two C1 confirmation) pass the independent checks, have optimizer return code zero and have no singular-fit flag. Full derivative/Hessian checks were not performed for the real fits. The synthetic judge pipeline passes after the layout changes. A new full-data prediction-only run at `outputs/layout_selected_verified/` independently reproduces both existing selected models, rankings, intervals and plot allocations with current-source hashes. Original forecast, selected and robustness artifacts also continue to pass verification using their matching historical source snapshots where appropriate.

```text
python experiments/mixed_model.py --output-dir outputs/new_mixed_development
python experiments/mixed_model.py --output-dir outputs/new_mixed_development --summarize
python experiments/mixed_model.py --years 2007 --clusters 1 --output-dir outputs/new_mixed_confirmation
python experiments/mixed_model.py --output-dir outputs/new_mixed_development --summarize --confirmation-folder outputs/new_mixed_confirmation
python tests/verify_mixed_model.py outputs/mixed_model
python tests/verify_mixed_model.py outputs/mixed_confirmation_C1
python -m unittest discover -s tests -q
python tests/verify_selected_run.py outputs/layout_selected_verified
```

The cleanup leaves only `run_pipeline.py` and `run_selected_pipeline.py` in `scripts/`. Reusable implementation is in `src/`; research is in `experiments/`. The compressed pre-layout source snapshot permits honest verification of old result manifests without pretending the reorganized source has its old hash. Prior outputs and raw data were preserved. No branch switch, commit, push or merge was performed.
