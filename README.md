# TriPlex -- `deepmoy-pipeline` branch

Genomic prediction pipeline for the corn breeding hackathon, built from scratch on
top of the data findings already established on `main` and `data-audit`, with an
independent re-verification of the raw files, two real parsing bugs found and fixed
along the way, and a two-fold walk-forward validation instead of a single 2008 holdout.

## 1. Problem, in the terms this pipeline treats it as

It is January 2008. A reduced field-plot budget means the breeding programme can
only advance a fraction of its Cluster 1 / Cluster 2 candidate lines. Every 2008
line comes from a population that has **never been tested before** -- verified
independently here: zero population overlap between 2008 and any prior year, in
both clusters, and that holds for *every* year back to 2000, not just 2008. There
is no per-line phenotypic history to look up. Because every population is
testcrossed to exactly one opposite-cluster tester, SCA has no degrees of freedom
to estimate -- this is structurally a **GCA-only, additive genomic prediction**
problem: predict a line's breeding value from DNA alone, rank the candidates, and
advance the plot budget where it buys the most genetic gain.

## 2. What's different about this branch

`main` and `data-audit` already did a lot of the hard diagnostic work (2008 labels
present in the data, harvest-trait leakage risk, G x E premise not holding up,
ridge beating every fancier model tried). This branch re-derives the pipeline
independently rather than importing either one, in order to cross-check their
numbers -- and in doing so found two things worth flagging to the team:

**Two real parsing bugs, not just the known suffix quirk.** `data-audit`'s own
`normalize_line_id` (and an earlier version of this pipeline) assumes a
`LINE_UNIQUE_ID` suffix like `C2.1.1.0` is a harmless float-formatting artifact and
safely truncates it. That's true *almost* everywhere, but not always:

- Populations `C1.125`, `C1.39`, `C2.52`, `C2.195` use a genuine decimal suffix
  (`.1`, `.2`) that marks a **second, different individual** sharing the same
  integer line number -- truncating merges two different lines' yield records
  into one. Fixed by dropping these rows explicitly (5,760 across both clusters)
  rather than silently merging them.
- Population `C1.34` doesn't even use a dot -- its suffix is `#1`/`#2`
  (`"000000325#1"`), embedded directly in the phenotype file's own `LINE_UNIQUE_ID`,
  not just in the genotype files. The naive parse crashes on this; a looser one
  silently mismatches it. Fixed with the same drop-don't-merge rule.
- Population `C1.126`'s corrupted ids (`"00000DS%130"`-style, no recoverable line
  number) turned out to exist in the **phenotype file too**, not only the genotype
  file as previously documented. Fixed by treating these as unparseable.

**A shrinkage experiment that failed, tested and rejected with evidence** (see
`scripts/compare_shrinkage.py`): an empirical-Bayes (BLUP-style) shrinkage of the
field-adjusted line mean -- pulling thin-data lines toward zero, the fix for
`data-audit`'s diagnosed reason their own two-way field/line fit didn't work --
sounded like a clear improvement. Tested head-to-head against the plain
(unshrunk) mean residual on real 2008 outcomes, it did not help, and clearly hurt
the metric that matters most for this task:

| Cluster | Target | Held-out pearson | Held-out gain (bu/ac, top 10%) |
|---|---|---|---|
| C1 | shrunk | 0.132 | 0.26 |
| C1 | **unshrunk** | **0.159** | **1.39** |
| C2 | shrunk | 0.128 | 0.83 |
| C2 | **unshrunk** | 0.121 | **1.74** |

Shrinkage pulls thin-data lines hardest toward zero -- but the top/bottom of a
ranking is disproportionately made of exactly those less-replicated lines, so it
specifically blunts the tail we're selecting on. **This pipeline uses the plain
field-adjusted mean**, which also brings it in line with `data-audit`'s own target.

**A genuine instability finding from running two walk-forward folds instead of
one** (see section 5) -- the single biggest reason to look at more than one
held-out year before trusting a correlation number.

## 3. Pipeline

```
data/raw/  (gitignored, same layout as main)
  C1_Phenotype_Data_V2.csv, C2_Phenotype_Data_V2.csv, environmental_features.csv
  genotypes/C{1,2}/ImputedPopulationsC{1,2}/C{n}.{pop}_Imputed.csv
        |
scripts/clean_phenotype.py    raw CSV -> data/processed/clean_C{n}.parquet
scripts/clean_genotypes.py    ~1,000 population CSVs -> data/processed/geno_C{n}.npz
scripts/build_target.py       field adjustment + shrinkage experiment -> target_C{n}_cutoff{Y}.parquet
scripts/build_features.py     per-fold marker QC/impute/scale -> features_C{n}_cutoff{Y}.npz
scripts/rank_lines.py          baselines + ridge, CV and held-out scoring -> outputs/
scripts/compare_shrinkage.py   diagnostic: shrunk vs. unshrunk target, same X matrices
scripts/viz_stats.py           aggregate stats for a data-cleaning dashboard (no raw files needed downstream)
```

Every cleaning step counts and prints what it drops -- nothing disappears silently.

### Cleaning (plot records)

| Step | C1 dropped | C2 dropped |
|---|---|---|
| Unparseable `LINE_UNIQUE_ID` (corrupted, incl. `C1.126` phenotype-side) | 715 | 0 |
| Ambiguous decimal/`#`-suffix line id (see above) | 3,773 | 1,987 |
| No yield recorded | 24,579 | 25,158 |
| Field with <20 lines | 3 | 15 |
| Outlier >5 SD within its own field | 2 | 5 |
| Line grown in <2 fields | 242 | 50 |
| **Final** | **511,395** | **510,112** |

### Genotypes

Both clusters share one identical 2,911-marker panel. Structural cleaning
(parents excluded, progeny ids parsed, the same five quirky populations handled
consistently with the phenotype side) yields 77,142 / 77,142 usable progeny rows
for C1 (one population, `C1.126`, reduced to zero usable rows and dropped
entirely) and 76,635 / 76,635 for C2 (no populations lost). After joining to the
cleaned phenotype line set: **100% genotype coverage** on both clusters, for
every fold -- every phenotyped line that survives cleaning has a matching
genotype row. (`data-audit` reports 85.4% training-year coverage; we traced the
gap to their `normalize_line_id` merge-rather-than-drop choice on the ambiguous
rows plus differing `clean()` thresholds, not a bug on either side -- see
`scripts/clean_genotypes.py` and `scripts/clean_phenotype.py` docstrings for the
full trace.)

### Target: field-adjusted mean yield (`YLD_ADJ`)

`YLD_BE` minus its own field's (`YEAR`+`LOC`) mean, averaged per line across every
field it was grown in. Field variance component is ~86% of the total (13.6% line
/ 86.4% field-plus-residual for C1, 10.9%/89.1% for C2), consistent with both
other branches' numbers. Shrinkage was tested and rejected (section 2).

### Marker QC and scaling -- fit per fold, on training lines only

Drop markers >50% missing or MAF <1% (thresholds and the resulting mean/sd used
for imputation and standardization are all computed from `YEAR<=cutoff` lines
only, then applied unchanged to the held-out year). float64 throughout -- ridge's
normal equations on ~2,700 correlated (LD-linked) marker columns are numerically
delicate; float32 measurably pushed the selected alpha to the edge of the search
grid.

### Model

Ridge regression on all QC'd markers (~2,700), the GBLUP-equivalent linear model,
picked over gradient boosting / PCA-compressed features / neural nets on the
strength of `data-audit`'s own published sweep (ridge beat all of them, and every
alternative degraded more from CV to real holdout, meaning they were leaning on
family resemblance that cannot transfer to brand-new 2008 populations). Not
re-litigated here. Compared against two baselines:

- `mean` -- training cohort average (do-nothing floor)
- `parent` -- average of training lines sharing a parent (`CROSS`), pure pedigree

Alpha tuned by **leave-population-out** grouped CV (never random k-fold --
siblings within a population would leak across the split).

## 4. Validation design: two walk-forward folds, not one

Because every year's populations are already disjoint from every other year (see
section 1), a year boundary *is* a population-disjoint split for free. Two folds:

- **train <=2006 -> score on real 2007** (a validation checkpoint)
- **train <=2007 -> score on real 2008** (the actual January-2008 decision)

Both years' true yields are already in the data; both are held out of training
and used only for final scoring.

## 5. Results

| Fold | Cluster | CV pearson | Held-out pearson | Held-out gain (bu/ac, top 10%) |
|---|---|---|---|---|
| <=2006 -> 2007 | C1 | 0.241 | **0.048** | **-1.16** |
| <=2006 -> 2007 | C2 | 0.257 | 0.195 | +2.14 |
| <=2007 -> 2008 | C1 | 0.189 | 0.159 | +1.39 |
| <=2007 -> 2008 | C2 | 0.231 | 0.121 | +1.74 |

The 2007 fold for C1 is the headline finding of this branch: CV suggested the
*best*-looking fold of the four (0.241), but real 2007 performance collapsed to
near zero, and the top-10% selection by this model would have **underperformed**
a do-nothing baseline (gain -1.16 vs. the mean baseline's +1.64). C2 did not show
the same collapse in the same year. A single held-out year -- which is all a
2008-only evaluation would show -- would have hidden this entirely.

**Read this as evidence for volatility, not a broken model.** The underlying
signal is modest everywhere (r ~ 0.12-0.26), and individual line means are only
~49% reliable to begin with (`data-audit`'s reliability analysis, not
re-derived here) -- at that noise level, a single held-out year can plausibly
swing from the best fold to a wash by chance. The actionable conclusion is that
**any recommendation from this model needs wide, explicit uncertainty bounds**,
not a confident point estimate -- and that a one-year holdout, on its own,
overstates how much you can trust the number it produces.

Per-line rankings: `outputs/ranked_2007_C{1,2}.csv`, `outputs/ranked_2008_C{1,2}.csv`.
Full model comparison: `outputs/model_comparison_cutoff{2006,2007}.csv`.

### Top-ranked 2008 lines (the production fold, train<=2007)

`pred` is the ridge model's GCA estimate (bu/ac, field-adjusted scale); `y` is
the line's real 2008 field-adjusted yield, shown for audit only -- it plays no
part in ranking or training. Given the instability finding above, treat the
ranking (which lines are near the top) as the useful signal, not the exact
`pred` value for any one line.

**Cluster 1**

| line | pred | y (real 2008) | n_fields | population |
|---|---|---|---|---|
| C1.427.16 | 11.35 | 13.29 | 5 | 427 |
| C1.427.36 | 11.31 | -13.49 | 5 | 427 |
| C1.401.42 | 11.28 | 1.19 | 6 | 401 |
| C1.379.88 | 11.14 | 10.67 | 5 | 379 |
| C1.401.18 | 10.71 | -4.60 | 6 | 401 |
| C1.427.85 | 10.57 | -8.03 | 5 | 427 |
| C1.401.51 | 10.46 | -10.74 | 5 | 401 |
| C1.401.81 | 10.40 | 6.24 | 6 | 401 |
| C1.401.87 | 10.35 | -0.33 | 6 | 401 |
| C1.401.7 | 10.26 | -4.51 | 6 | 401 |

**Cluster 2**

| line | pred | y (real 2008) | n_fields | population |
|---|---|---|---|---|
| C2.388.67.0 | 9.94 | 0.43 | 5 | 388 |
| C2.424.142.0 | 9.37 | 15.56 | 5 | 424 |
| C2.440.115.0 | 8.99 | 9.46 | 5 | 440 |
| C2.440.102.0 | 8.98 | 8.22 | 5 | 440 |
| C2.417.66.0 | 8.88 | 27.37 | 5 | 417 |
| C2.417.78.0 | 8.83 | 15.09 | 5 | 417 |
| C2.417.88.0 | 8.82 | 28.07 | 3 | 417 |
| C2.388.65.0 | 8.79 | 17.80 | 4 | 388 |
| C2.440.123.0 | 8.60 | 1.75 | 5 | 440 |
| C2.417.62.0 | 8.60 | 12.16 | 4 | 417 |

Note the scatter between `pred` and real `y` even within this top-10 -- C1.427.36
was ranked #2 and actually underperformed badly (-13.49); C2.417.88.0 wasn't
ranked #1 despite the single best real outcome (28.07). That's the r~0.13-0.16
signal made concrete: the ranking carries real information in aggregate (see the
gain metric in section 5), but is not reliable line-by-line, which is exactly
why section 5 argues for uncertainty bounds over point estimates.

## 6. Run instructions

```bash
pip install -r requirements.txt
# expects data/raw/ laid out exactly as in main's README

py scripts/clean_phenotype.py
py scripts/clean_genotypes.py
py scripts/build_target.py   --cutoff-year 2007   # also run with --cutoff-year 2006
py scripts/build_features.py --cutoff-year 2007   # also run with --cutoff-year 2006
py scripts/rank_lines.py     --cutoff-year 2007   # also run with --cutoff-year 2006
```

`compare_shrinkage.py` and `viz_stats.py` are diagnostics, not part of the main
path -- the former reuses already-built feature matrices, the latter only needs
the cleaned parquet/npz files, no raw archives.

## 7. Known limitations, not yet addressed here

- **No uncertainty quantification yet** -- section 5's instability finding argues
  this is not optional; combining leave-population-out CV residual variance with
  historical per-location variance (as `data-audit` does) is the natural next step.
- **No multi-trait / selection index** -- DNA predicting moisture/test-weight
  better than yield (per `data-audit`) is an unexploited lever here.
- **Pedigree feature is a simple parent-mean baseline only**, not combined with
  markers as an extra ridge feature.
- **No environmental covariates** -- consistent with both other branches' finding
  that weather/soil predicts across locations but not across years, so this was
  deprioritized rather than overlooked.
- The residual gap between this branch's held-out C1 number (0.159) and
  `data-audit`'s (0.186), even after matching target construction, is not fully
  reconciled -- likely small cumulative differences in cleaning thresholds, not a
  bug, but not chased down further.

## Requirements

```
pandas
numpy
scikit-learn
scipy
pyarrow
```
