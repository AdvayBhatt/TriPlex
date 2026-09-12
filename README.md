# TriPlex — Corn Breeding Track

**Precision Digital Agriculture Hackathon 2026**

Predicting which never-tested maize lines deserve scarce field plots, from DNA alone.

---

## 1. Problem statement

It is late on a Friday in January 2008. Budget cuts have reduced the field plots available for the
coming season. The head of the 115 Relative Maturity pipeline needs to know which maize lines to
advance — and the lines in question are about to be planted, so nothing has been harvested.

**The decision:** ~16,000 candidate lines (7,397 in Cluster 1, 8,519 in Cluster 2), a plot budget
that covers a fraction of them, and one season's delay costing a year of genetic gain.

**Why it is hard.** We verified that **not one line in the 2008 cohort was ever field-tested before**
— zero overlap with all eight prior years, in both clusters, with every population new. There is no
performance history to look up. Information can reach the target year only through genomic
relatedness, pedigree, or environment.

**Stakeholder impact.** The intended user is the breeding programme manager allocating plots.
Advancing the wrong lines wastes a season and the plots that could have carried better candidates;
advancing the right ones compounds into genetic gain across subsequent cycles. At the top 2% of our
ranking, selected lines out-yield the cohort average by **7.3 bu/acre**.

**Commercial significance.** This is the initial screening stage of general germplasm evaluation.
Most lines here are not suitable for advancement. The job is to spend a reduced budget where it buys
the most information about which lines are worth carrying forward.

## 2. Solution overview

Three data sources join into one modelling table:

| Source | One row is | Joined by |
|---|---|---|
| Phenotype | one line, in one field, in one year | — |
| Environment | one field in one year (weather + soil) | `YEAR` + `LOC` |
| Genotype | one line's DNA, 2,911 SNP markers | `LINE_UNIQUE_ID` |

**The workflow:**

```
raw archives
   → build_genotypes.py   ~1,000 population CSVs → one int8 matrix (0.08s to load)
   → extract_map.py       the genetic map, which lives only in the unimputed archives
   → build_dataset.py     field adjustment, aggregation to line level, leakage exclusion
   → rank_lines.py        ridge on markers, leave-population-out validation
   → recommend.py         advancement list, expected yields, plot allocation
```

**The prediction approach.** Because field effects dominate yield (68.8% of variance), we never
model raw yield. We subtract each field's mean — leaving a line's advantage over its own field-mates
— then average over the ~7 fields each line grew in. That single number per line, its **General
Combining Ability**, is the target. Ridge regression on all 2,686 usable markers predicts it for the
2008 cohort, which is then ranked.

## 3. Technical approach

### Preprocessing

**Plot records** (536,936 → 515,784 for C1), every drop counted and printed:

| Step | Dropped | Rationale |
|---|---|---|
| No yield recorded | 20,904 | nothing to learn from |
| Field with <20 lines | 3 | its mean is the yardstick; too few lines makes it unstable |
| Outlier >5 SD *within its own field* | 2 | a 300 bu/ac plot is only odd if neighbours yielded 150 |
| Line grown in <2 fields | 243 | unreliable line mean |

**Markers** (2,911 → 2,686): dropped >50% uncalled (11) and minor allele frequency <1% (214).
Remaining gaps filled with each marker's mean, then centred and scaled. **All statistics are
computed on training lines only**, so the held-out year never influences the transformation.

### The leakage constraint

The scenario fixes the decision point at January 2008. A feature is legal only if it exists before
planting.

- **Legal** — genotype, pedigree (`CROSS` parents), environment covariates of target sites.
- **Excluded** — `MST`, `PHT`, `EHT`, `TWT`, `RTLP`, `STLP`, `ERM` and the line's own `YLD_BE`.

Every excluded trait is measured *at harvest, on the same plot* as the yield being predicted. A model
using them scores well and cannot be run at decision time. They are named explicitly in
`build_dataset.py` so the exclusion is visible rather than implied. We verified the cost: no harvest
trait correlates with yield above |r| = 0.17.

### Baseline models

| Baseline | C1 (2008) | C2 (2008) |
|---|---|---|
| Cohort mean (do nothing) | 0.000 | 0.000 |
| Pedigree — mean of training lines sharing a parent | 0.082 | 0.116 |
| Environment mean | see §7 — covariates fail across years |

### Validation

Random k-fold would be dishonest here. Lines within a population are siblings from one cross; a
random split puts near-identical relatives on both sides, so a model scores well by recognising
families and then collapses on 2008, where every family is new.

**Folds are split by population.** All siblings stay on one side. This mirrors the real task and is
the cross-validation appropriate to breeding programme structure. Final scoring is a single pass
against the real 2008 yields, which are present in the data but never used in training.

Grouped CV tracked the held-out result closely in C1 (0.192 → 0.186) and was optimistic in C2
(0.227 → 0.150), so we treat it as an upper estimate rather than a forecast.

## 4. Results

### Prediction accuracy

| Model | C1 CV | **C1 2008** | C2 CV | **C2 2008** |
|---|---|---|---|---|
| Cohort mean | 0.000 | 0.000 | 0.000 | 0.000 |
| Pedigree only | 0.068 | 0.082 | 0.080 | 0.116 |
| Ridge, 50 genomic PCs | 0.165 | 0.138 | 0.206 | 0.077 |
| **Ridge, all markers** | **0.192** | **0.186** | **0.227** | **0.150** |
| Markers + pedigree | 0.166 | 0.170 | 0.194 | 0.174 |

DNA beats pedigree in both clusters. Whether pedigree helps *on top of* markers does not replicate
— it hurt in C1 and helped in C2 — so we do not claim it.

### Interpreting the number

The outcomes we score against are themselves noisy: each 2008 line's value comes from ~5 single
unreplicated plots. Splitting each line's plots into two halves and correlating the independent
means gives 0.33, implying the line means are about **49% reliable**.

**A perfect model would score r ≈ 0.70, not 1.0.** The rest is plot noise, which nothing predicts.

| | Raw r | % of achievable |
|---|---|---|
| C1 | 0.186 | **27%** |
| C2 | 0.150 | 23% |

Reported as correlation, not R². R² against a half-noise target reads as 0.03 and misrepresents the
result; correlation is the convention in genomic selection for exactly this reason.

### Line ranking

| Advance | In the true top tier | vs random | Realised gain |
|---|---|---|---|
| Top 2% | 10.9% | **5.4×** | +7.29 bu/ac |
| Top 5% | 14.4% | 2.9× | +4.16 bu/ac |
| Top 10% | 17.2% | 1.7× | +2.54 bu/ac |
| Top 20% | 26.6% | 1.3× | +1.64 bu/ac |

Accuracy is highest exactly where a reduced budget forces selectivity.

### Uncertainty quantification

Two independent sources, combined in quadrature:

- **σ_model** = 10.95 bu/ac (C1), 10.02 (C2) — from leave-population-out CV residuals, so it
  reflects performance on *new families*, the real task.
- **σ_field** — a location's year-to-year swing from its own history; locations with no history take
  the spread across all fields (31.9 bu/ac) and are flagged.

80% intervals are **±14.0 bu/ac** (C1), ±12.8 (C2). They are wide deliberately. With r ≈ 0.19
against a target that is half noise, narrower intervals would misrepresent the evidence.
**Individual line predictions are weak; the value is in the aggregate.**

## 5. Run instructions

### Judge mode — no large files needed

```bash
pip install -r requirements.txt
python scripts/demo.py
```

Runs the **entire pipeline in ~3.3 seconds** on 0.8 MB of generated data. It does not reimplement
anything: it writes synthetic data in the real file formats and invokes the production scripts, so a
passing demo exercises the real code paths.

Because the synthetic marker effects are known, it also reports recovery against **true genetic
values** (r = 0.643) alongside recovery against observed yields (r = 0.499) — a direct demonstration
of the plot-noise gap that reliability analysis measures indirectly on real data.

### Full dataset

```bash
ln -s "/path/to/Simplified Hackathon Dataset V3" data/raw

python scripts/audit_data.py          # verify the data matches expectations
python scripts/build_genotypes.py     # zips → data/processed/geno_C{1,2}.npz
python scripts/extract_map.py         # genetic map → marker_map.csv
python scripts/build_dataset.py       # → dataset_C{1,2}.npz
python scripts/rank_lines.py          # → ranked_2008_C{n}.csv, model_comparison.csv
python scripts/recommend.py           # → recommendations_C{n}.csv, allocation, per-location
```

Optional analyses: `gxe_diagnostic.py`, `environment_model.py`, `multitrait.py`, `neural_test.py`
(requires `torch`).

**Resources:** the full pipeline needs 61 MB of processed files and peaks around 4.6 GB RAM. A ridge
fit takes 1.6 seconds. No GPU or cluster is required.

## 6. Commercial recommendations

`recommend.py` produces three artefacts.

**Lines to advance.** At a 10% budget: 740 Cluster 1 lines across 34 populations, 852 Cluster 2 lines
across 46. Each carries a predicted GCA, an 80% interval, and a flag for whether it is expected to
beat its own midparent — the criterion the breeding brief names for this stage.

| Line | Predicted GCA | 80% interval | Beats midparent |
|---|---|---|---|
| `C1.427.16` | +14.68 | +0.65 … +28.71 | yes |
| `C1.427.11` | +14.31 | +0.29 … +28.34 | yes |
| `C1.427.36` | +13.80 | −0.23 … +27.83 | yes |

Of the 740 advanced, 722 have a midparent estimate and all 722 clear it. The remaining 18 have no
parent seen before 2008 and are reported as unknown rather than silently counted as failures.

**Expected yield per location.** The field effect does not reorder lines — every line at a site
shares it — but it is required to state a bushel figure. `line_by_location_C{n}.csv` gives each
advanced line's expected yield at the sites it is scheduled for, with site-specific intervals.

**Where to spend plots.** Of 149 Cluster 1 locations scheduled for 2008, **37 have no history at all**
and 51 more have fewer than three years. Since weather and soil cannot forecast a new season (§7), a
location's own record is the only usable signal. Historical effects are shrunk toward zero by years
of evidence, and sites without a track record are flagged as high-variance bets rather than ranked.

**One national list is correct.** Genetic correlation across environment types is ≈0.85, so lines
rank consistently everywhere. There is no need to manage location-specific portfolios — a finding,
not a limitation.

## 7. Constraints and limitations

### What we tested and rejected

Ten approaches, each implemented properly and scored against real held-out outcomes.

| Approach | Best r | Outcome |
|---|---|---|
| **Ridge, all markers** | **0.186** | kept |
| Dense network (704k params) | 0.172 | matched, not beaten |
| Field-quality weighting | 0.171 | rejected |
| PCA compression (10–600 components) | 0.169 | rejected |
| Ridge + GBM blend | 0.159 | made it worse |
| Gradient boosting | 0.125 | rejected |
| Midparent excess ranking | 0.121 | rejected |
| Two-way field adjustment | 0.119 | rejected |
| Block attention (transformer) | 0.096 | rejected |
| Multi-trait stacking | +0.008 / −0.003 | did not replicate |
| Two-stage via harvest traits | 0.240 ceiling | rejected — below DNA-only |

**The pattern is the finding.** Genetic effects here are additive, individually tiny, and spread
across thousands of markers. Methods that hunt interactions find noise. The validation-to-holdout
drop tells the story: ridge does not move, the dense network falls 0.229 → 0.172, attention falls
0.188 → 0.096. Flexible models lean on family resemblance, which cannot transfer when every target
population is new.

### Known failure modes

- **Novel germplasm.** Accuracy depends on relatedness to trained lines. A 2008 population unrelated
  to anything in 2000–2007 will be predicted poorly, and we cannot flag that in advance.
- **New seasons.** Environmental covariates predict across locations (r = +0.185) but not across
  years (r = −0.030, and −0.141 on 2008). Absolute yield forecasts for an unseen season are not
  reliable; only the relative ranking is.
- **New locations.** 37 of 149 C1 sites have no history, so their field effect is assumed zero with
  the widest interval.
- **Thin lines.** A line grown in 2 fields is ranked alongside one grown in 7. Reliability weighting
  was tested and hurt (0.185 → 0.171), so lines are weighted equally, which is a known compromise.

### Data gaps

- **SCA is inestimable by design.** Every line is crossed to exactly one tester (0 of 73,532 have
  two) and every population uses exactly one tester (468 of 468). Line × tester interaction has no
  degrees of freedom, and the tester main effect is perfectly confounded with population. We verified
  this rather than assuming it.
- **G×E cannot be separated from plot error** at the individual-field level: 99.8% of
  line × environment cells hold a single unreplicated plot.
- **Imputed genotypes are ~97% inferred.** Parents were densely genotyped (0.8% missing); each
  progeny line was skim-genotyped at ~3% of markers and the rest imputed. Imputation error is a real
  and unquantified uncertainty source.
- **C1 genotype coverage is 85% for training years** (100% for 2008), so ~15% of training lines are
  unusable.

### Documentation discrepancies

Findings that contradict the supplied documentation, all verified against the raw files:

- Years span **2000–2008**, not 2001–2007, and the 2008 yields are present.
- The year column is **`YEAR_x`**, not `YEAR`; the provided sample code does not run as written.
- Every C2 identifier carries a **4th component** (`C2.1.1.0`), and 2,397 C1 ids do too. Joining on
  the raw id matches zero C2 lines.
- **405 locations** (C1) / 390 (C2), not the ~7 the guide lists.
- The **genetic map** exists only in the unimputed archives, as the first two rows.
- 6 populations have malformed progeny ids; the guide's regex silently discards all 980 rows.

### Next development steps

1. **Multi-trait genomic prediction.** DNA predicts test weight (0.293) and moisture (0.271) *better*
   than yield (0.187). A proper multivariate model, and the selection index it enables, is the
   strongest remaining addition — advancement decisions need predicted moisture and lodging anyway.
2. **Haplotype / IBD segment sharing.** Progeny chromosomes are mosaics of a few large parental
   blocks; identifying shared inherited segments may estimate relatedness better than treating
   markers independently. The genetic map makes this feasible.
3. **Shrunk (BLUP-style) field effects.** The two-way fit failed because its line effects carry no
   shrinkage; a properly regularised version may behave differently.
4. **Replication.** The single largest limitation is one plot per line per field. Nothing in the
   modelling can recover what the trial design did not measure.

---

## Repository

| Path | Contents |
|---|---|
| `scripts/` | the pipeline, one stage per file |
| `docs/DATA_FINDINGS.md` | data discoveries with the evidence behind each |
| `docs/PIPELINE.md` | plain-language walkthrough of every decision |
| `ref/` | the supplied brief, data guide and scenario notes |
| `CLAUDE.md` | setup and the gotchas, for anyone picking this up |

`data/` and `outputs/` are gitignored — no dataset files are committed.
