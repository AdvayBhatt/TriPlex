# What the pipeline does, in plain terms

Feeds directly into the README sections the rubric requires (solution overview,
technical approach, validation). Numbers below are Cluster 1.

## The one-sentence version

We take eight years of field trial records, strip out the effect of *where* each
plant was grown so that only genetics remain, and train a model to predict a maize
line's genetic merit from its DNA — then use it to rank 16,000 lines nobody has
ever field-tested.

## The three data sources and how they join

| Source | One row is | Joins by |
|---|---|---|
| Phenotype | one plant line, in one field, in one year | — |
| Environment | one field in one year (weather + soil) | `YEAR` + `LOC` |
| Genotype | one plant line's DNA (2,911 markers) | `LINE_UNIQUE_ID` |

`LINE_UNIQUE_ID` looks like `C1.435.109` = cluster 1, population 435, line 109.
That maps to genotype file `C1.435_Imputed.csv`, row `109`.

**Gotcha that cost us a debugging cycle:** C1 ids have 3 parts (`C1.1.191`) but
every C2 id has a 4th (`C2.1.1.0`), and 2,397 C1 ids do too. Joining on the raw id
matches *zero* C2 lines. We trim to the first 3 parts, which matches 100%.

## What we predict (the output)

Not raw yield. Two problems make raw yield unusable:

**1. Fields differ enormously.** Environment is ~68% of all yield variation. The
same line yielded 163 bu/acre at one site and 263 at another in the same year. A
line grown in good fields looks good even when its genetics are ordinary.

*Fix:* subtract each field's average yield. What's left is "how much better or
worse than everyone else in that same field" — a fair comparison.

**2. A single plot is noisy.** One plot of one line barely predicts the same line
elsewhere (r ≈ 0.1). And 99.8% of line-field combinations have only one plot.

*Fix:* average each line's adjusted yield over all ~7 fields it grew in. Averaging
cancels noise.

The result is **one row per line** with target **`YLD_ADJ`** — a line's average
advantage over its field-mates, in bu/acre. Spread is about ±10.7 bu/acre.

## What we feed in (the inputs)

The scenario is **January 2008**: lines are about to be planted, nothing is
harvested. A feature is legal only if it exists before planting.

**Legal**
- **Genotype** — 2,686 SNP markers. DNA, readable from a seed sample.
- **Pedigree** — the parent lines of the cross, from `CROSS`.
- *(Phase 2)* **Environment** — weather and soil of the target locations.

**Banned — every harvest-measured trait**
`MST`, `PHT`, `EHT`, `TWT`, `RTLP`, `STLP`, `ERM`, and the line's own `YLD_BE`.

These are measured *on the same plot, at the same time* as the yield we predict.
Using them is target leakage: the model would score beautifully and be impossible
to run at decision time. They are named explicitly in `build_dataset.py` so the
exclusion is visible, not implied.

We checked the cost of excluding them: no harvest trait correlates with yield above
|r| = 0.17. We give up almost nothing.

They remain useful for multi-trait prediction and the selection index — just never
as inputs for predicting yield.

## The cleaning steps

Every drop is counted and printed, so nothing disappears silently.

**Plot records** (536,936 → 515,784):

| Step | Dropped | Why |
|---|---|---|
| No yield recorded | 20,904 | nothing to learn from |
| Field with <20 lines | 3 | its average is the yardstick we subtract; too few lines makes it unstable |
| Outlier >5 SD *within its own field* | 2 | a 300 bu/acre plot is only suspicious if neighbours yielded 150 |
| Line grown in <2 fields | 243 | its mean would be unreliable |

**Lines** (77,752 → 67,604): dropped lines with no DNA — nothing to predict from.
C1 genotype coverage is 85% for training years but **100% for 2008**, so we lose
training data and nothing in the prediction set.

**Markers** (2,911 → 2,686):

| Step | Dropped | Why |
|---|---|---|
| >50% uncalled | 11 | imputing them invents data |
| Rare variant (MAF <1%) | 214 | identical in ~99% of lines, so it can't tell them apart |

Remaining gaps are filled with each marker's average, then every column is centred
and scaled. **All of these statistics come from training lines only**, so nothing
about 2008 influences the transformation.

## How we check it honestly

**Random k-fold would lie to us.** Lines inside a population are siblings from one
cross. A random split puts near-identical relatives on both sides, so the model
scores well by memorising families — then collapses on 2008, where every population
is new.

So folds split **by population**: all siblings stay on one side. That mirrors the
real task, and it's the "cross-validation appropriate for breeding program
structure" the rubric asks for.

In C1 it tracked closely (CV 0.192 → 2008 0.186). In C2 it was optimistic
(CV 0.227 → 2008 0.150). So grouped CV is a usable guide for iterating without
touching 2008, but it is **not** a precise forecast — treat it as an upper estimate.

## What we got

Cluster 1 (2,686 markers, 60,207 train / 7,397 test lines):

| Model | CV r | 2008 r | Gain (bu/ac) |
|---|---|---|---|
| Do nothing (cohort mean) | 0.000 | 0.000 | −1.25 |
| Pedigree only | 0.068 | 0.082 | −0.62 |
| Ridge on 50 genomic PCs | 0.165 | 0.138 | +0.76 |
| **Ridge on all markers** | **0.192** | **0.186** | **+2.54** |
| Markers + pedigree | 0.166 | 0.170 | +1.85 |

Cluster 2 (2,444 markers, 67,716 train / 8,519 test lines):

| Model | CV r | 2008 r | Gain (bu/ac) |
|---|---|---|---|
| Do nothing (cohort mean) | 0.000 | 0.000 | +2.08 |
| Pedigree only | 0.080 | 0.116 | +0.23 |
| Ridge on 50 genomic PCs | 0.206 | 0.077 | +1.38 |
| Ridge on all markers | 0.227 | 0.150 | +2.34 |
| **Markers + pedigree** | 0.194 | **0.174** | +2.33 |

"Gain" = advance the top 10% by model score; how much those lines actually
out-yield the cohort average.

What holds across both clusters:

- **DNA beats pedigree.** 0.186 vs 0.082 in C1, 0.150 vs 0.116 in C2. Genomic
  prediction earns its place in both.
- **Full markers beat PCs.** PCA keeps family structure but discards the
  within-family variation that separates siblings — exactly what has to be ranked
  once every population is new.

What does **not** hold:

- **Whether pedigree helps on top of markers is cluster-dependent.** It hurt in C1
  (0.186 → 0.170) and helped in C2 (0.150 → 0.174). One cluster is not evidence of
  a general rule; do not claim it as a finding.

Also treat the "gain" column with care for the do-nothing model: when every
prediction is identical, the top 10% is an arbitrary tie-break, which is why C1
shows −1.25 and C2 shows +2.08. Both are noise around zero, not signal.

r ≈ 0.15–0.19 is modest. It is in the normal published range for predicting
*entirely new families*, the hard case here. It is an honest baseline, not yet a
winning number.

## File map

| Script | Produces |
|---|---|
| `audit_data.py` | verifies the raw data matches expectations |
| `build_genotypes.py` | zips → `geno_C{n}.npz` (int8, loads in 0.08s) |
| `extract_map.py` | `marker_map.csv` — chromosome + cM per marker |
| `gxe_diagnostic.py` | how much genotype-by-environment interaction exists |
| `build_dataset.py` | `dataset_C{n}.npz` — the clean modelling table |
| `rank_lines.py` | `ranked_2008_C{n}.csv`, `model_comparison.csv` |

## What is not done yet

- Environment features are extracted but **not yet used as model inputs** — that is
  Phase 2, and it is the team's chosen angle.
- Lines are weighted equally regardless of how many fields they grew in.
- Field adjustment is simple subtraction, not a joint line+field fit.
- No uncertainty intervals on predictions yet (the rubric asks for them).
