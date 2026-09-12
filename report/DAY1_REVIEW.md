# TriPlex: Day 1 review and Day 2 starting point

Reviewed 12 September 2026. This is an assessment, not a model rewrite. Original data and outputs were preserved. The workspace changed during review: root modeling scripts were removed and `scripts/run_pipeline.py` was introduced in local commit `34c51298bdbbfaddd9a8e361aa0390277499d39e`. Findings below distinguish that current entry point from earlier notebooks/scripts. Public `data-audit` was checked against GitHub and reviewed at `51d39d95d3cab7d311bcb0f157f07544d4203eec`.

**Decision:** use the audit branch's cross-family genomic prediction framework as the foundation, after correcting its leakage and deployment issues. Keep the local work's streaming approach, convenient command-line entry point, and commercial-index intent. The existing local rankings are retrospective results and should not be submitted as a January 2008 advancement list.

## 1. What the references actually ask for

All five files in `ref/` were inspected, including slide text/notes and embedded images.

| Reference | What it establishes |
|---|---|
| `PRECISION_AGRICULTURE_HACKATHON_2026_TRACK1.pdf`, 8 pages | January 2008 decision, historical 2001–2007 information, new-season advancement under reduced plot capacity. Choose broad-acre, environment-specific, or multi-trait optimization. Required: baselines, appropriate validation, uncertainty, actual line recommendations, resource allocation, limitations, and a small-data judge mode. Only supplied breeding datasets; no large raw data uploads. |
| `Hackathon Scenario And Help.docx` | Initial screening in the 115 RM pipeline. Improve inbred germplasm within each cluster. Observed yield is from an opposite-pool testcross. GCA is the intended focus; SCA is explicitly outside scope. Broad-acre prediction is permitted. Most lines do not outperform the midparent sufficiently to advance. |
| `Data Overview.pptx`, 5 slides | Explicitly says 999 populations, 499 C1 and 500 C2, 2000–2008, 16–356 lines per population; parents typed at 2,911 markers and progeny at 49–123. The genotype screenshot includes chromosome and cM metadata above parental/progeny rows. Speaker notes describe an older 969-population version. |
| `Hybrid_Breeding_Maize_Course.pptx`, 57 slides | Hybrid vigor, additive GCA versus cross-specific SCA, separate heterotic pools, selfing/DH, testcross screening, progressively narrower trials, seed-production constraints, and commercial tradeoffs. Its example costs and advancement percentages illustrate the breeding funnel; they are not the hackathon's fixed budget or measured project outcomes. |
| `CORN_BREEDING_DATA_GUIDE.md` | Joining conventions and starter examples. Useful orientation, but several examples assume cleaner names and IDs than the actual files. Treat it as guidance to verify, not executable ground truth. |

The task is **rank previously untested 2008 candidate lines using preplant information**, separately respecting C1/C2 breeding pools. It is not simply predict a randomly hidden historical row. Known candidate DNA, pedigree and planned sites are legitimate inputs. Actual April–October 2008 weather and same-season measured traits do not exist in January. Soil and historical climate can be inputs, subject to documented availability. ERM's exact measurement provenance is not established here; exclude it until its availability is supported rather than claiming every trait was literally measured in October.

Measured testcross performance is a proxy for breeding merit. Neither raw average yield nor field-centered yield is automatically a pure, fully identified GCA estimate across testers. A single tester per population limits that interpretation.

The brief mentions 2001–2007; the overview and raw data also contain 2000. A final submission should explicitly justify including 2000 and report a 2001–2007 sensitivity check. The budget fraction is not specified: 10%, top 20, and the course's example percentages are team policy choices.

## 2. Verified local data inventory

Independent scan: both phenotype CSVs, the full environment table, and **all 999 genotype CSVs**. Detailed counts, schemas, call frequencies, missingness and unusual IDs are in `report/review/data_profile.json`; its `shape` includes two temporary audit columns, so the source phenotype width is 33, not 35.

| Quantity | C1 | C2 |
|---|---:|---:|
| Phenotype rows | 536,936 | 535,340 |
| Raw unique IDs | 78,065 | 76,635 |
| IDs after retaining first three dot-separated components | 78,015 | 76,635 |
| Historical normalized lines, through 2007 | 70,583 | 68,099 |
| 2008 candidate lines | 7,432 | 8,536 |
| 2008 populations | 71 | 86 |
| Candidate line or population overlap with earlier years | 0 | 0 |
| 2008 records with yield / all records | 37,522 / 39,947 | 45,268 / 47,339 |
| Locations over all years | 405 | 390 |
| 2008 locations / previously unseen | 153 / 38 | 146 / 36 |
| Genotype files | 499 | 500 |
| Historical normalized lines matched to genotype | 60,311 | 68,099 |
| 2008 lines matched to genotype | 7,432 | 8,536 |
| Missing genotype calls, including parents | 13.63% | 11.82% |

Every population appears in only one year. Every population with an observed tester has at most one tester. Missing tester records exist, so this is not proof that every row contains a known tester.

**Genotypes:** the same ordered 2,911-marker panel occurs in every file, including across clusters. All observed calls are exactly -1, 0, or 1. No evidence supports treating these files as 0–2 dosage data. Parent rows begin `PID`; progeny rows follow. C1 historical line coverage is about 85.45%, while C2 coverage is 100% under the audit parser. Do not generalize C1's missingness to C2.

**IDs:** all C2 phenotype IDs have a fourth component. The old local end-anchored regex rejects all of them; it also rejects 4,488 C1 records. First-three-component normalization merges 50 raw C1 IDs, so preserve raw IDs and document collisions. Some genotype IDs carry `.1` or `#1` suffixes; C1.126 has 181 unrecoverable corrupted IDs. The audit keeps the first of 75 duplicate parsed C1 rows. Additional valid unpadded integers occur in C1.107 (19), C1.237 (19), and C2.309 (102); the local strict 11-digit filter discards those too. These are not all corrupt records. The normalization rule should be documented as an assumption about suffixes, not erase provenance silently.

**Environment:** 1,185 rows, 86 columns, no missing cells and no duplicate `(YEAR, LOC)` keys. Of 84 covariates, 42 start with `X` (April–October PRCP/TAVG/DP01/DP10/HTDD/CLDD); 42 are soil measurements at six depths (cfvo/clay/silt/nitrogen/sand/phh2o/soc). Local code selects only `X*`, thus excludes all soil. C1 has complete environment-key coverage. **C2 has 209 unmatched site-years covering 52,412 rows, including 5,778 rows in 2008.** A complete environment table does not imply complete coverage of both phenotype tables. This contradicts the audit documentation's broad coverage claim.

**Trait missingness:** yield 3.89%/4.36%, moisture 2.63%/2.84%, plant height 70.92%/71.18%, ear height 77.91%/77.17%, test weight 11.87%/12.48% (C1/C2). Historical traits can be separate training targets; missing trait targets should not be treated as observed average performance.

**Samples:** 100 phenotype rows, 100 genotype rows including two parents, 20 environment rows. There are 80 raw phenotype lines and 38 with sample DNA; there are no shared phenotype/environment site-years. The sample is an execution demonstration, not useful validation of new-family forecasting.

**Absent locally:** full `merged_full.csv`, imputed ZIP archives, unimputed genome archives, and the referenced `Data Dictionary.docx`. The genotype CSVs are already expanded directly under `data/raw/genotypes/C1/` and `C2/`. Genetic map extraction and the precise 96.9% pre-imputation missingness claim cannot be independently verified from absent archives. The overview supports sparse progeny typing, but not that exact raw missingness percentage.

**Spatial information:** current phenotype schemas have site latitude/longitude and SET, but no explicit plot row/range/replicate coordinates. SET is not a substitute for a spatial layout. The rows may also represent line-site summaries rather than individual original plots (the related publication describes entry means). Do not equate lack of duplicate CSV rows with proof that original field trials were unreplicated. Spatial correction needs original trial-design metadata.

## 3. Local method versus audit method

| Aspect | Current local `scripts/run_pipeline.py` | `data-audit` |
|---|---|---|
| Unit fitted | Separate model per population, observation rows | One line-level model per cluster, shared across historical populations |
| Target | Raw yield, averaged after prediction | Mean site-year-centered yield |
| Inputs | Progeny SNPs + actual season `X*` weather | Main ranking uses progeny SNPs; experimental blend adds parental genotype model |
| Validation | Hold lines out within their own population | Hold whole historical populations out; evaluate 2008 cohort separately |
| Can learn from prior years for a new population? | No cross-population transfer exists | Yes, marker effects transfer across populations |
| Moisture index | Observed same-season MST | Separate experimental script predicts MST from DNA |
| Uncertainty | Same population row-level RMSE added/subtracted to line means | Historical family-held-out residual SD; normal 80% bounds, not coverage-calibrated |
| Main output cohort | Historical and 2008 lines together | Cleaned, phenotyped subset of 2008 lines |
| Judge mode | Real sample, runs successfully | Synthetic data through production scripts, runs successfully |
| Data layout | Expects extra `ImputedPopulationsC*` folder missing here | Expects ZIPs missing here |

Both choose a sensible additive starting model and both attempt actionable outputs. The audit's validation target is substantially closer to the brief. These methods' RMSE/correlation numbers measure different tasks and cannot be ranked as if they were interchangeable scores.

## 4. Local findings, prioritized

**P1: There is no January 2008 prediction path.** `run_pipeline.py` processes all years within each population and drops missing yield before predicting (line 108); the GroupKFold loop at 168 trains on siblings from that population's same season. Since every 2008 population is new, those training labels would not be available. Withholding yield in a direct function check returns zero predictions. Introduce a historical training stage and a separate candidate-only prediction stage.

**P1: Advancement uses future measured moisture.** Lines 186–187 aggregate observed MST, and 347–348 subtract it from predicted yield. The index cannot run for unharvested candidates. Use predicted moisture from a historical model, or omit the penalty. Likewise the observed-yield stability score cannot be claimed to be known for 2008 candidates.

**P1: Actual future weather is used.** The `YEAR_x + LOC` join loads April–October weather for the line's own season. This is retrospective information, even though harvest traits are excluded from the yield feature matrix. Historical climate or explicit preplant forecasts are the appropriate alternative.

**P1: Full mode finds no genotype files in this checkout.** Line 294 adds `ImputedPopulationsC{cluster}` below `genotypes/C{cluster}`. Neither directory exists locally. Missing paths are silently skipped, and empty results exit successfully. Resolve actual supported layouts and fail clearly when zero populations can run. The new script fixes dependence on `merged_full.csv`, which the removed root scripts required.

**P1: Existing advancement table is the wrong cohort.** `outputs/line_rankings_full.csv` has 77,352 rows, all C1, despite the newer README claiming C1+C2. It contains 7,735 advancement flags, of which only **626** are 2008; **7,109** are historical. The full table contains 7,429 2008 lines. Per-year advancement counts: 2000 14; 2001 168; 2002 99; 2003 32; 2004 2,461; 2005 2,369; 2006 587; 2007 1,379; 2008 626. Do not treat this artifact as a planting recommendation.

**P2: Validation still has avoidable contamination.** OOF clipping uses the entire population's labels (175–176), including held-out labels. Inner `RidgeCV` does not hold entire lines out; imputation/scaling also occur before that inner CV. Outer grouping protects raw IDs, but suffix variants mapped to identical DNA may straddle folds. Use a canonical line key and training-only transformations/clipping, with grouped inner tuning if tuning is retained.

**P2: Raw yield and observed CV do not identify genetic merit or adaptation.** Across populations with different sites/years, raw means reward favorable environments. A low yield CV can reflect a narrow site range or noise, not broad adaptation. `n_env` counts rows, not distinct environments. Neither ±one row-level RMSE around a line mean nor the proposed 0.85/0.60 stability cutoffs has demonstrated calibration.

**P2: Moisture economics confuse dollars with bushels.** At the README's assumed $0.04 per bushel per moisture point and 125 bu/ac, the drying saving is $5/ac per point. To express that as bu/ac, divide by a stated grain price. It is not automatically 5 bu/ac. This is dimensional arithmetic using the README's assumptions, not a verified current price recommendation.

**P2: Reproducibility and reported improvement are incomplete.** Local `requirements.txt` is empty. The README's 18.19 versus 20.17 RMSE compares population-averaged CV error with a global in-sample environmental baseline, not a matched held-out comparison. The current entry point does not calculate or export that comparison. Claims that feature groups are important, stability thresholds are validated, or full results cover both pools need evidence.

Earlier implementation findings retained for context: `03_modeling.ipynb` includes harvest traits, duplicates MST by also selecting all `M*` columns, excludes soil, and ranks in-sample fitted predictions. Its residual bootstrap stores 500 full prediction vectors (about 2 GB for 511,642 float64 rows, before copies) and averages row interval endpoints rather than estimating uncertainty of line means. Its risk rule compares lower bounds to the median lower bound across all lines, not to a population median performance threshold. `02_preprocessing.ipynb` defaults to full mode, has the same incorrect nested genomic path, rejects suffix IDs, and can append incompatible CSV widths if genomic files are missing. Earlier root/scripts streaming variants clipped -1 calls to 0; the new entry point changed this to [-1,2], so that specific destructive clipping finding is historical, not a current-code defect.

## 5. Audit-branch findings, prioritized

**P1: Prediction eligibility depends on future outcomes.** `build_dataset.clean` drops missing yield, removes outcome-defined outliers, and requires at least two yield observations before assembling the candidate table. Main outputs therefore cover 7,397 C1 and 8,519 C2 candidates instead of all 7,432/8,536. Missing/low-count 2008 labels should restrict evaluation, not prediction. `recommend.load_plots` also drops missing yield before deriving the supposed planting schedule. Separate candidate/schedule data from truth.

**P1: Absolute-yield recommendations include 2008 truth.** `recommend.py:68` calculates `grand` over all years, and line 122 calculates fallback field SD over all years. Both influence forecasts/intervals. Estimate both from years before 2008. Historical field shrinkage does not remove contamination in the all-year intercept. Main genomic ranking is a separate path and is not contaminated by this particular intercept bug.

**P2: Marker filtering contradicts training-only claims.** `build_dataset.py:199,204` calculates marker missingness and frequency over all lines, including 2008. Means/scales are historical-only, but filters are not. Known target DNA could be intentionally used in a transductive method; it must not be described as a fully training-only transform. For a clean protocol, fit filters on historical training rows and refit within CV folds.

**P2: Reported CV is also the tuning score.** `rank_lines.py:133` selects alpha using grouped CV, then reports scores on those same folds. PCA, marker standardization, and target field centering are prepared outside folds. Training field means can include validation-family outcomes. This does not mean 2008 labels directly entered the main ridge fit, because years/cohorts are separate, but the CV estimate is not nested and independent. Use rolling historical years for model selection; keep the final forecast year separate. A parent-derived feature should be cross-fitted rather than calculated from the training row's own target.

**P1 for the optional experiment: Multivariate target construction includes 2008 outcomes.** `multivariate.py:119` estimates G/R covariances from all plots, then uses them to build historical training targets. Fit the covariance model exclusively on historical data before judging that experiment. Missing trait targets elsewhere are filled with zero; this creates artificial observations and can distort comparisons.

**P2: The email's best model is not the delivered model.** `relatedness.py` prints a 50/50 standardized blend, but does not save it to the ranking or recommendation outputs. `rank_lines.py` saves plain ridge. `recommend.py` refits plain ridge with its own default alpha, without consuming the tuning result. The blend is dimensionless after standardization; attaching bushel intervals from plain ridge would be wrong. Persist the chosen model, calibration, prediction units and matching uncertainty together.

**P2: Fifteen 2008 comparisons make this an explored test set.** Numerous variants were accepted/rejected after examining 2008. The claim of one final untouched pass is inconsistent with the experimental history. The small blend gains can be reported as exploratory reproduced results, but require historical-year replication and family-aware uncertainty to support a robust improvement claim.

**P2: Clean installation is not specified.** Audit `requirements.txt` contains only pandas/numpy; production scripts import scipy and scikit-learn. Optional neural work also needs torch; R needs readr/lme4. The demo passes in the installed environment, not in a proven clean requirements-only installation. The archive-only loader also needs adaptation for this checkout.

**P2: Intervals and allocation need narrower claims.** Normal residual-SD intervals have no reported held-out empirical coverage. They describe errors against noisy adjusted phenotypes, not pure genetic uncertainty, and do not separate plot error from G×E. `location_allocation` is a site summary, not an allocation satisfying an integer plot budget. Selecting a fraction of lines does not fix plot count when lines have different scheduled sites. Thresholding tied scores can also exceed the intended selection count. High-yield sites are not necessarily the most informative trial sites; unknown sites may provide valuable information.

**Scientific interpretation:** environment-centered means still depend on which families share each field. The R population model's residual includes within-population line differences and omitted interactions, not just plot error. An unweighted variance of family means divided by line variance is not an exact variance decomposition with unequal family sizes; the reported two-thirds remainder is observed variation, not established within-family genetic variation. `genetic_correlation.py` improves on the old calculation but its correlations do not use exactly identical line subsets and counts; values above 1 do not establish absence of G×E. Broad-acre is a reasonable hackathon scope, not a proved universally optimal national list. Location fixed effects can have an unseen-site fallback; claiming categorical locations are categorically unusable is too strong.

Other audit script limitations: `audit_data.py` samples four marker panels per cluster, not every file (the independent review here checks all). The neural seed is set after model construction, limiting exact reproducibility; neural models retain an early-stopping training split while the ridge baseline uses all historical rows. Attention blocks follow existing column order without loading/validating the map and can cross chromosome boundaries. `decompose.py` calculates climate normals including 2008 weather. The handbook's history-plus-weather-anomaly improvement would not be usable in January with actual future anomalies. Several reported sweeps/weighting experiments have no standalone reproduction entry point in the branch. `R/variance_components.R` is the scalable population-level model; `teammate_original_FIXED.R` still builds large fixed-effect matrices and its VarCorr table excludes the fixed terms. R model outputs were reviewed as reported, not rerun here.

## 6. The published comparison

The relevant source is Jacobson et al. (2014), *General Combining Ability Model for Genomewide Selection in a Biparental Cross*, DOI 10.2135/cropsci2013.11.0774 ([author-hosted paper](https://bernardo-group.org/wp-content/uploads/2015/11/Jacobson-et-al-2014.pdf)). Table 2 reports 0.06 for the same-background model, averaged across 30 selected test populations. The paper controls tester and assesses prediction within crosses; pooled cross-family 2008 correlation is a different statistic. Its SB+GCA value is 0.12. Similar data provenance is plausible but exact dataset identity/protocol equivalence is not established by matching marker counts.

Consequently, avoid “3× the published benchmark” and “78% of phenotypic selection.” Use the publication as context for the difficulty of across-family prediction. Also correct the handbook's claim that prediction R² is literally Pearson r squared: out-of-sample R² depends on calibration and can be negative despite positive correlation. An illustrative 0.24 upper result is not below 0.186; it cannot on its own prove a two-stage model is inferior. Negative experiments establish performance of the tested configurations, not that more data, markers, G×E or neural models can never help.

## 7. Verification record and boundaries

Both judge entry points were executed with outputs redirected under `report/review/`. Local sample run produces 37 lines and 4 advancements. Audit demo reproduces correlation 0.643 against synthetic genetic truth and 0.499 against synthetic observed outcomes. These measure different synthetic/real-sample demonstrations, not comparative forecasting performance.

The audit reproduction uses unchanged branch modeling functions; only the genotype reader is adapted to the existing expanded local CSVs. It recreates compressed genotype matrices and cleaned line targets, then runs `relatedness.py` at alpha 30,000. It does not rerun the full alpha grid, PCA sweep, neural training, every optional experiment, or R. The run completed successfully. Detailed output is in `report/review/reproduction_log.txt`.

| Reproduced method | C1 Pearson | C1 Spearman | C1 top-10% gain (bu/ac) | C2 Pearson | C2 Spearman | C2 top-10% gain (bu/ac) |
|---|---:|---:|---:|---:|---:|---:|
| Flat pooled ridge | 0.186 | 0.142 | 2.54 | 0.150 | 0.151 | 2.34 |
| Parent-family + within-family components | 0.169 | 0.150 | 2.81 | 0.144 | 0.152 | 3.43 |
| 50/50 standardized blend | 0.191 | 0.152 | 2.85 | 0.154 | 0.156 | 3.20 |

Training/test counts reproduce 60,207/7,397 in C1 and 67,716/8,519 in C2. Gains refer to the observed, field-adjusted evaluation target and are not guaranteed economic or genetic gains. The blend improves on flat ridge in both clusters, but the two-component model alone has greater top-10% gain in C2 despite lower correlation; choice of decision metric matters.

Two focused checks confirmed the audit findings independently of documentation: changing only 2008 yields in a small input changed `recommend.field_effects`' forecast baseline from 116 to 176; changing only candidate missing genotype calls changed `prepare_markers`' retained marker count from one to two. These demonstrate the dependency paths, not their numerical effect on the full real-data scores.

The raw data scan confirms zero 2008 historical overlap and matching genotype panels. Tests also confirm that the local prediction function produces no predictions when yields are withheld, that both nested genotype directories are absent, and that full ranking cluster/year composition contradicts the README. Original notebooks and outputs were not overwritten. Review data caches are local artifacts, not submission data.

## 8. Practical Day 2 order

1. Establish the forecast contract: all 15,968 candidate lines, preplant feature allowlist, separate evaluation truth, separate clusters, explicit assumed plot budget. Freeze the currently explored 2008 results as exploratory.
2. Repair the audit foundation: local data adapters, training-only preprocessing/statistics, full candidate coverage, reproducible dependencies, and one saved model-to-recommendation path. Keep a plain ridge reference.
3. Validate on rolling pre-2008 years with family separation. Evaluate all candidates with usable truth under a fixed scoring protocol. Report pooled and within-family Pearson/Spearman, top-budget gain/precision, and family-resampled uncertainty; compare blend and ridge using identical splits.
4. Make the decision operational: predict moisture if included, document dimensional economic weights, enforce an actual plot budget, retain family/pool diversity where justified, and empirically check interval coverage. Use expected field yield cautiously; it is distinct from breeding-value ranking.
5. Finish the submission narrative and judge mode. Explain why broad-acre was chosen and what evidence supports each claim. Demonstrate that hiding all 2008 yield/trait columns leaves candidate predictions unchanged.

For Dhanush's offered tasks, the highest-value immediate ownership is **repairing and packaging his prediction/recommendation path and documenting the validation limitations**. An on-site teammate can check whether original trial coordinates exist. Pursue spatial work only if that metadata is available. Within-family prediction is worthwhile research, but should follow a reproducible, valid baseline and submission package.
