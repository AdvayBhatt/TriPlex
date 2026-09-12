# Data findings — read before modeling

Reconnaissance on the full dataset. Regenerate any number here with
`python scripts/audit_data.py` (writes `outputs/data_audit.md`).

Several of these contradict `ref/CORN_BREEDING_DATA_GUIDE.md` and the track PDF. Where they differ,
the files win — the documentation appears to describe an earlier version of the dataset.

---

## 1. 2008 is a completely new cohort — this defines the problem

Line overlap between 2008 and every prior year is **exactly zero**, in both clusters:

| | Lines ≤2007 | Lines 2008 | Overlap | New populations |
|---|---|---|---|---|
| C1 | 70,633 | 7,432 | **0** | 71 of 71 |
| C2 | 68,099 | 8,536 | **0** | 86 of 86 |

**Implication:** there is no per-line phenotypic history to look up for the target year. Any model
that predicts 2008 from a line's own past records has nothing to work with. Information can only
reach 2008 through three channels:

1. **Genomic relatedness** — the SNP data, and the main intended route.
2. **Pedigree** — testers recur strongly (8/10 in C1, 10/13 in C2 appear pre-2008) and some parent
   germplasm recurs (15/42 C1, 15/54 C2). Cheap to encode, likely a strong baseline.
3. **Environment** — year/location covariates, which transfer regardless of germplasm.

This is what makes it a genuine genomic prediction problem rather than a lookup exercise.

## 2. The 2008 labels are in the data

The docs describe the data as 2001–2007. It is actually **2000–2008**, and the 2008 rows have yield
filled in — 37,522/39,947 in C1, 45,268/47,339 in C2.

**Implication:** we can train on ≤2007, predict 2008, and score ourselves honestly against held-out
truth. This is the single biggest advantage available. Teams working from the documentation alone
will have no real validation signal and will be reporting cross-validation numbers that don't
measure the actual task.

We should still *build* the pipeline as if 2008 were unlabeled, and use the labels only for
evaluation — otherwise we leak and the judged result won't reflect real performance.

## 3. Location cannot be a categorical feature

405 distinct locations in C1, 390 in C2 — not the ~7 the guide lists. In 2008, 38 of 153 C1
locations (36 of 146 in C2) had never appeared before.

**Implication:** one-hot encoded `LOC` dead-ends on unseen sites. Environments must be represented
by their weather and soil covariates. This is exactly what `environmental_features.csv` is for, and
it is complete — 86 columns, **0% missing**, covering every YEAR+LOC including 2008.

## 4. Genomics is cleaner than expected

- All **999** population files (499 C1 + 500 C2) share one **identical 2,911-marker panel** — the
  same panel across both clusters. Genotypes stack into a single matrix with no intersection step.
- **100% coverage**: every phenotyped population has a genotype file, in both clusters, all years.
- Files are small (~1–1.4 MB). Full matrix is ~160k lines × 2,911 markers — fits in memory.

Caveats: the "Imputed" files are still **12–14% NA**, and the first two rows of each file are the
population's parents (`PID<number>`), not progeny — filter them out or they corrupt the merge.

### Coverage is 100% by population, but 85% by line

Worth separating, because it sets the training-set size:

| | Lines with genotypes |
|---|---|
| Training (≤2007), C1 | 60,311 / 70,633 — **85.4%** |
| **Target (2008), C1** | 7,432 / 7,432 — **100%** |

Every population has a genotype file, but not every phenotyped *line* within a population has a
genotype row. We lose ~15% of training lines; we lose nothing in the prediction set, which is the
half that matters.

### The unimputed archives are not redundant — they hold the genetic map

Easy to dismiss as "raw versions of the imputed files." They aren't. Each unimputed CSV has 2,915
columns (4 metadata + the same 2,911 markers) and its **first two rows are not genotypes**:

- **Row 1** — each marker's chromosome (1–10, maize's full complement).
- **Row 2** — each marker's position in **centiMorgans**, monotonically increasing within chromosome.

That's a genetic linkage map: 2,911 markers over **1,825 cM**, mean spacing **0.63 cM**. Identical
between C1 and C2. `scripts/extract_map.py` pulls it out without extracting the 700 MB file.

It also explains how the data was generated. Splitting missingness by row type:

| | Missing calls |
|---|---|
| Parent rows | **0.8%** |
| Progeny rows | **96.9%** |

Parents were densely genotyped; each progeny line was skim-genotyped at only ~3% of markers (~90 of
2,911), and the rest was **imputed** from the map plus parental haplotypes — which works precisely
because biparental populations have the long LD blocks described above.

Two things follow:

1. **Parent genotypes are a much cleaner signal than progeny genotypes.** Progeny calls in the
   imputed files are ~97% statistical inference.
2. **Imputation error is a genuine uncertainty source** and belongs in the limitations section the
   rubric asks for. Nobody else will mention it.

### Progeny ID quirks (6 of 999 populations)

Progeny IDs are normally 11-digit zero-padded line numbers, but:

- **C1.34, C1.39, C1.125, C2.52, C2.195** carry a replicate suffix — `000000001.1`, `000000001#1`.
  The leading digits are the real line number; `build_genotypes.py` parses them and keeps the first
  row when a suffix maps two rows to the same line (75 rows dropped in C1).
- **C1.126** has corrupted IDs (`00000TF%405`, `000000BE5%3`) with no recoverable line number.
  All 181 rows are dropped.

All six are 2001–2003 populations — **none are in the 2008 prediction set**, so this costs a little
training data and nothing else. The regex in the provided data guide (`^\d{11}$`) silently discards
all of them.

## 5. Phenotypes are testcrosses → predict GCA

Every yield number is the performance of a *hybrid*: a C1 line crossed to a C2 tester (or vice
versa). We are estimating **General Combining Ability** — the line's average contribution across
hybrid combinations, which is predominantly **additive**.

**Implication:** additive genomic models (GBLUP, ridge regression on markers, rrBLUP) are
well-matched to the biology here. The scenario doc explicitly puts SCA out of scope, so we should
not spend time on dominance/epistasis interaction terms.

## 6. Data hygiene gotchas

- The year column is **`YEAR_x`**, not `YEAR`. The provided sample code does not run as written.
- Junk columns throughout: `Unnamed: 0{,_x,_y}`, `projects_x/y`, `shorthand_x/y`, duplicate `YEAR_y`.
- Trait completeness varies a lot. `YLD_BE` ~96%, `MST` ~97%, but `PHT` ~29% and `EHT` ~22% —
  multi-trait work is constrained by height sparsity, not by yield.
- Mean yield trends upward across years (166 → 206 bu/acre). Year effects are real and must be
  modeled or removed, or the model will read genetic gain as noise.

---

## What this suggests for approach

The zero-overlap finding argues for building the **genomic prediction path first** — it is the only
channel with enough resolution to rank individual lines, and everything else (environmental
covariates, GxE, multi-trait indices) layers on top of it.

A defensible progression, each step scoreable against real 2008 truth:

1. **Baselines** (required by the rubric): environmental mean, and population/pedigree mean — i.e.
   "how well can you do knowing only the cross and the location?"
2. **Additive genomic model** on the stacked marker matrix → GCA estimates for 2008 lines.
3. **Environmental covariates** added for broad-acre stability, or a GxE term for the
   environment-specific challenge.
4. **Selection index / Pareto analysis** if we take the multi-trait challenge.

Open decision: which of the three challenge options to make primary. Note that (1) and (2) are
shared infrastructure regardless of that choice, so they are safe to build now.
