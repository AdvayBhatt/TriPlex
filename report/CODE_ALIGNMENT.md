# Code alignment and verification — 12 September 2026

## Outcome

The runnable workflow now implements the brief's January 2008, new-line broad-acre decision. It uses historical phenotypes and candidate DNA, validates on successive historical seasons, keeps all 2008 candidates, reports empirical prediction intervals, and exports a budget-feasible planting plan for each pool. The original notebooks remain clearly labeled Day 1 exploration.

**The implementation is more defensible; its predictive performance is still modest.** This is not a demonstrated improvement over Dhanush's experimental blend. The corrected default run gives retrospective Pearson correlations of **0.165 for C1 and 0.105 for C2**. Both models have worse RMSE than the constant environmental-mean baseline, and C1's historical advancement gains are inconsistent. Report these limitations alongside the positive ranking signal.

The final source was run on both full datasets and independently checked against its exported artifacts. All 15,968 target candidates are retained. No commit, push, merge or branch switch was performed. Local `main` remains at `34c51298bdbbfaddd9a8e361aa0390277499d39e`; the checked `origin/data-audit` reference remains `51d39d95d3cab7d311bcb0f157f07544d4203eec`.

## Reference and judging alignment

The five available reference documents were reviewed before choosing the method; their contents and contradictions are catalogued in [DAY1_REVIEW.md](DAY1_REVIEW.md). The track brief and scenario were checked again during implementation. There is no numerical judging-weight schedule in the available brief, so none is invented here.

| Requirement from `ref/` | Current implementation | Practical limit |
|---|---|---|
| January 2008 decision using 2001–2007 history | Exact default training years; 2008 candidate roster built separately from outcomes | Locations reconstructed from recorded 2008 metadata are assumed to be known planting plans |
| Broad-acre prediction is allowed | One transferable marker model per pool, predicting average field-adjusted yield advantage | No absolute future yield, site-specific response or stability forecast |
| Separate heterotic pools and GCA focus | Separate C1/C2 fits and budgets; opposite-pool testcross yield is a breeding-merit proxy | Tester/family confounding prevents claiming a pure GCA estimate |
| Phenotypic, genomic and environmental integration | Historical yields adjusted for site-year environment, joined to candidate markers; historical site summaries exported | Weather/soil covariates are not used in the predictive model; no claim of omics or G×E modeling beyond what is implemented |
| Phenotypic BLUP and environmental-mean baselines | REML random line intercept after field centering, plus zero adjusted-yield baseline | New unrelated lines cannot be ranked by either constant baseline; REML is conditional on estimated field means |
| Appropriate validation | Forecast 2005, 2006 and 2007 from strictly earlier years; assert no population overlap | Few temporal origins, dependence within populations, and prior exposure to 2008 limit inference |
| Uncertainty and failures | Historical forecast-error intervals, source-specific calibration, coverage checks, family bootstrap for correlation, missing-data and novel-site flags | Empirical intervals are not guaranteed coverage or confidence bounds for pure genetic merit |
| Line advancement and scarce plots | All-candidate rankings and explicit line-location replication plans constrained by a per-pool budget | 10% budget and one replicate are assumptions; no seed, diversity, site-overhead or tester constraints |
| Small judge demonstration | Same pipeline on deterministic synthetic two-pool, multi-year data | Synthetic scores are demonstration results only |
| Communication and reproducibility | README, pinned dependencies, source/input hashes, model files, manifests, tests and this report | Slides and the team's final commercial policy still need to reflect these results |

## Changes and why they were necessary

### Prediction boundary and validation

The earlier local pipeline fitted within each population across available years and produced historical rankings. Its line-grouped folds did not test transfer into wholly new populations, which is the 2008 task. Actual season weather and observed moisture/stability information could enter its decision score.

The replacement learns shared marker effects across historical populations, separately within each cluster. No target yield is used to decide candidate membership, preprocess markers, fit models, calibrate intervals, rank lines or allocate plots. Harvest traits and the weather feature table do not enter prediction. The evaluator runs after predictions and the planting plan are saved.

For historical validation, outcomes are adjusted independently inside the held-out season. Field means from that season define the evaluation target; they are not fed into the historical fit. The first forecast year has no prior calibration errors and therefore no interval. Later validation intervals use only errors from earlier origins. Final 2008 intervals use 2005–2007 errors.

2008 was already examined by the team, including multiple models on `data-audit`. It cannot now be advertised as an untouched final test. The fixed ridge penalty, 30,000, is an exploratory starting value inherited from that investigation, not an independently optimized choice. No 2008-driven parameter search was added here.

### Data integrity

The loader supports the actual expanded genotype layout, including nested directories beneath each pool, and validates the shared 2,911-marker panel by name. All calls must be -1, 0, 1 or missing. Parents are excluded. Missingness filtering, imputation and scaling use historical training lines only.

Numeric normalization resolves leading zeros and documented suffix forms. This recovers historical records missed by the earlier parser: the earlier review's roughly 85% C1 historical genotype-match figure was specific to that parser, not a permanent biological coverage limit. In the corrected 2001–2007 fit, 65,052 of 65,127 eligible C1 training lines have unambiguous DNA. All 65,125 eligible C2 training lines have DNA.

The 181 corrupted historical C1 identifiers account for 715 rows in the selected training years; they are excluded and listed in the audit. Seventy-five C1 canonical identities have conflicting genotype rows and are excluded from marker fitting rather than resolved by keeping the first. They remain eligible for the phenotypic baseline. C2 has no such conflicts in this run. Raw-to-canonical aliases are exported for inspection. No 2008 candidate requires a DNA fallback in the supplied data.

Training/scoring outcomes need finite yield, a field with at least 20 distinct lines, and at least two locations for the line. Repeated line-site-year records are averaged before adjustment. These rules restrict fitting and **scoring only**: 35 C1 and 17 C2 candidates without sufficient outcomes still receive predictions and remain eligible for advancement.

### Uncertainty, baselines and allocation

The earlier residual/stability columns were not calibrated prediction intervals for new populations. They are replaced by the 5th and 95th percentiles of earlier forecast errors, added to the current prediction. Missing DNA uses a clearly labeled phenotypic fallback; insufficient source-specific calibration produces an empty interval and a review flag, not fabricated precision. Wholly missing or uninformative candidate markers also trigger fallback.

The phenotypic-only baseline estimates residual and line variance by REML after field centering. For unseen lines, the predicted random effect is zero. The environmental-mean baseline is zero on the adjusted scale. Correlation is undefined for these constant predictions; their RMSE is reported. These variance estimates are not comparable to Dhanush's joint environment/population/population×environment decomposition.

The allocation policy traverses predicted rank, retaining a full planned site bundle if it fits the remaining budget. Ties use canonical ID. The default budget is 10% of unique candidate-site plots in each pool, rounded down, with one replicate. This explains why advancement counts are not exactly 10% of line counts. C2 contains repeated recorded line-site combinations; these do not create extra hypothetical plots. The exported plan is checked against the ranking's costs and budget.

The policy is feasible and transparent, but it is a heuristic. It does not solve a global optimization, impose minimum family coverage, model seed availability, select testers, or reserve plots for exploration. New sites and missing information are flagged for review without pretending that those uncertainties have been solved.

### Repository and execution

The entry point is now thin orchestration in `scripts/run_pipeline.py`; `scripts/triplex.py` contains the auditable statistical workflow and `scripts/demo_data.py` supplies synthetic judge data. Requirements were previously empty and now pin the five runtime dependencies tested locally. The pipeline has no runtime dependency on the review copy of `data-audit`.

Output paths are separate from the old historical rankings. Nonempty destinations require an explicit overwrite flag. Overwrite clears only known generated files for the selected pools and run manifests, preventing stale evaluation files from appearing current after a prediction-only run. Raw data, unrelated files and other pools' outputs are not cleared. Consult the current manifest to identify which pools belong to a run.

Generated output directories ignore their own contents. Original data and the previous `outputs/line_rankings_full.csv` remain in place. Each notebook has a prominent scope note and retains its original exploration/results. The pre-existing `.gitignore` changes, deletion of root `run_modeling.py`, and old sample ranking were not reversed or treated as changes made by this task.

## Verified real-data results

Final execution: Python 3.13.1, four numerical threads, versions recorded in the manifest. The final complete run took **94.8 seconds with genotype caches available** on this machine. Judge mode completed in about one second of pipeline time in a clean temporary directory containing only the scripts and requirements, using already-installed dependencies. Installation in a fresh virtual environment was not tested. The first full implementation pass, which built caches, took about 198 seconds; these are local observations rather than hardware-independent guarantees.

| Quantity | C1 | C2 |
|---|---:|---:|
| Eligible historical training lines | 65,127 | 65,125 |
| Training lines with unambiguous informative DNA | 65,052 | 65,125 |
| Retained training markers | 2,412 | 2,373 |
| All 2008 candidates ranked | 7,432 | 8,536 |
| Candidates with evaluable outcomes | 7,397 | 8,519 |
| Candidates advanced under default policy | 762 | 890 |
| Allocated plots / budget | 3,994 / 3,994 | 4,698 / 4,698 |
| Advanced candidates with evaluable outcomes | 744 | 885 |
| Retrospective Pearson r | 0.1646 | 0.1048 |
| Population-bootstrap 95% interval for r | [0.0445, 0.2725] | [-0.0046, 0.2202] |
| Retrospective Spearman rho | 0.1200 | 0.1058 |
| Within-population Pearson r | 0.0847 | 0.0921 |
| Model RMSE, adjusted bu/acre | 10.0533 | 11.0102 |
| Environmental-mean baseline RMSE | 9.8109 | 10.3806 |
| Phenotypic-only baseline RMSE | 9.8116 | 10.3815 |
| Scorable top-decile mean advantage over scorable cohort | +1.4939 | +1.3960 |
| Actual advanced/scorable mean advantage over scorable cohort | +1.5167 | +1.3262 |
| Nominal 90% interval coverage on scorable 2008 lines | 93.16% | 88.84% |

Top-ranked examples are `C1.427.36` (predicted +14.90, interval -5.96 to +31.34 adjusted bu/acre) and `C2.368.110` (+14.04, interval -5.21 to +29.36). Both involve at least one historically unseen site and are flagged for review. Even the highest-ranked line in either pool has a lower interval endpoint below zero. The shortlist therefore should not be described as confidently superior lines. Intervals characterize realized adjusted performance and are broad.

### Historical rolling validation

The advancement-gain column below evaluates the actual budget-feasible allocation among candidates whose outcomes can be scored. It is not the same as selecting exactly the top 10% of scorable lines.

| Pool | Forecast year | Pearson r | Allocation gain, adjusted bu/acre | Empirical 90% interval coverage |
|---|---:|---:|---:|---:|
| C1 | 2005 | 0.0537 | -0.3012 | Not available: no prior forecast errors |
| C1 | 2006 | 0.1788 | +3.4703 | 93.35% |
| C1 | 2007 | 0.0804 | -0.6126 | 91.39% |
| C2 | 2005 | 0.1719 | +0.9394 | Not available: no prior forecast errors |
| C2 | 2006 | 0.2211 | +2.0228 | 92.03% |
| C2 | 2007 | 0.2027 | +1.6646 | 92.48% |

This is meaningful evidence of instability, especially for C1. The fixed model loses to the environmental-mean baseline on RMSE in all listed folds, although C2 2006 is nearly tied. Positive pooled correlation alone does not establish robust commercial value. C2's retrospective correlation bootstrap interval also includes zero.

## Comparison with Dhanush's separate branch

The cross-population marker model is the useful foundation taken from the audited approach. The separate branch's reproduced blend scores (approximately 0.191/0.154) are **not** scores for this implementation. Its fit/evaluation population, inclusion of 2000, outlier handling, marker preprocessing and parental blend differ. This task did not import its experimental blend or claim equal/higher accuracy.

The corrected workflow adds a stricter candidate/outcome separation, chronology-aware validation, train-only marker filtering, explicit handling of conflicting identities, complete candidate outputs, historical empirical error intervals, per-location plot plans, run provenance and a synthetic demonstration of new-family forecasting. The branch's broader method search and R analysis remain useful research material, not silently merged code.

For a fair model comparison next, freeze one candidate roster and one historical evaluation contract and test the branch's preferred blend through that same boundary. Do not keep choosing methods based on 2008 and continue calling 2008 a clean test. Do not repeat the “3× benchmark” claim: the earlier review established that the reported published within-family comparison and the pooled cohort score are different evaluation quantities.

## Verification performed

Commands used for the final implementation:

```sh
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --sample --overwrite
python scripts/run_pipeline.py --overwrite
python tests/verify_full_run.py outputs/judge_forecast
python tests/verify_full_run.py outputs/forecast_2008
git diff --check
```

**13 regression tests passed.** They exercise bounded ID normalization and bad candidate rejection; time/population overlap rejection; retention of candidates without outcomes; missing-DNA fallback; training-only marker filtering; saved-model roundtrip; candidate-DNA changes not affecting fitted parameters or other candidates beyond numerical rounding; zero/finite budget feasibility; source-specific intervals; marker-panel reordering; genotype conflict handling and cache consistency; invalid genotype calls; undefined constant-baseline correlation; malformed external rosters; and stale-output cleanup.

The end-to-end leakage test removes every target yield, replaces future harvest values, adds future weather values, and supplies an outcome-free external candidate roster. It compares the entire ranking output, including intervals and allocation, **exactly**. Historical decimal strings are preserved during mutation so the test changes only future inputs; reserializing historical floats would otherwise introduce unrelated roundoff. The prediction-only run does not emit a target evaluation.

The independent artifact verifier imports no forecast functions. For both real pools it reconstructs candidate IDs from the raw metadata, checks uniqueness/completeness, replays allocation, checks plot totals, recomputes interval endpoints from strictly prior errors, verifies source hashes, rebuilds predictions directly from saved numeric model parameters and cached marker calls, and recomputes retrospective correlation. **All checks passed** for judge mode and full data. Its results are in each output folder's `independent_verification.json`.

A separate clean-directory judge run passed with only the three runtime scripts and requirements copied in; no full raw dataset or review material was present. All three notebook files remained valid JSON after scope notes were added. The final tracked diff passed whitespace/error checks. Git references were read to confirm branch boundaries; no write to a Git reference occurred.

## Remaining decisions for the team

The code now matches the declared task, but the evidence does not justify calling it deployment-ready or the best available model. For the hackathon, present a reproducible screening prototype and its observed limitations. Prioritize agreement on the actual plot budget, replication, diversity and exploration policy; review novel-site flags; and make the slides use the same target, candidate counts, baselines and uncertainty language as the README.

Further modeling should focus on a fair historical comparison of the parental blend, stronger shrinkage selected without target-season outcomes, within-family prediction, and environment adjustment that respects trial design. Spatial adjustment needs actual plot layout/replicate metadata, which was not found in the supplied schemas. Historical climate or soil predictors need an explicit availability argument and a demonstrated historical gain. None of these is claimed as completed here.

No GitHub action is needed to inspect or run the changes. If you later publish them, review and stage individual intended source, test and documentation files; inspect your pre-existing edits separately. Avoid blanket staging of `report/` or generated outputs. The user, not this task, should choose when and where to commit or push.
