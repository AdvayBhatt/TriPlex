# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TriPlex — team entry for the **Precision Digital Agriculture Hackathon 2026, Corn Breeding Track**.

**The task:** using a commercial maize breeding program's records (2000–2008), predict 2008 testcross
performance for new maize lines and turn those predictions into line-advancement recommendations
under a reduced field-plot budget. The deliverable is a prediction pipeline plus a README covering
problem statement, method, validation and uncertainty, commercial recommendations, and limitations.

Reference material is in `ref/` — read `ref/Hackathon Scenario And Help.docx` for breeding context
before modeling, and `ref/PRECISION_AGRICULTURE_HACKATHON_2026_TRACK1.pdf` for the three challenge
options and judging criteria.

## Setup

Data is **not** in the repo (`data/` and `outputs/` are gitignored). Point `data/raw` at the dataset:

```bash
ln -s "/path/to/Simplified Hackathon Dataset V3" data/raw
python scripts/audit_data.py            # verify the data matches expectations
python scripts/build_genotypes.py       # one-time: zips -> data/processed/geno_C{1,2}.npz
```

`scripts/audit_data.py` recomputes every structural claim in `docs/DATA_FINDINGS.md` from the raw
files. Run it after any data refresh; `--clusters 1` halves the runtime while iterating.

## Data layout

Under `data/raw/`:

| File | Contents |
|---|---|
| `C{1,2}_Phenotype_Data_V2.csv` | ~537k / ~535k rows, 33 cols, 141 MB each. Field trial records. |
| `environmental_features.csv` | 1,185 rows × 86 cols. One record per YEAR+LOC. |
| `ImputedC{1,2}Populations.zip` | 499 / 500 per-population SNP files (`ImputedPopulationsC1/C1.<pop>_Imputed.csv`). Modeling data. |
| `Unimputed_C{1,2}_Genome_Data.zip` | Single ~700 MB CSV each: the **genetic map** (rows 1–2), dense parent genotypes, sparse raw progeny calls. |

## Non-obvious facts that will bite you

Verified against the actual files; several contradict `ref/CORN_BREEDING_DATA_GUIDE.md` and the
track PDF. Full evidence in `docs/DATA_FINDINGS.md`.

**Phenotype CSVs are an un-cleaned merge.** The year column is **`YEAR_x`**, not `YEAR` (a redundant
`YEAR_y` also exists). `Unnamed: 0{,_x,_y}`, `projects_x/y`, `shorthand_x/y` carry no signal. The
sample code in the provided data guide does not run as written.

**Years are 2000–2008, not 2001–2007**, and 2008 rows have `YLD_BE` populated (~94% of C1, ~96% of
C2). Train on ≤2007 and score against 2008 — do not assume the target year is unlabeled.

**No line tested in 2008 was tested before.** Line overlap with prior years is exactly zero in both
clusters (0 of 7,432 C1; 0 of 8,536 C2); all 71 C1 and 86 C2 populations in 2008 are new. Per-line
phenotypic history cannot reach the target year. The only bridges are genomic relatedness, pedigree
(`GERMPLASM_ID`, `CROSS`, `GERMPLASM_ID_TESTER` — testers and some parents do recur), and environment.

**Location won't generalize as a categorical.** 405 locations in C1 / 390 in C2; 36–38 of the 2008
sites are new. Represent environments by their weather and soil covariates instead.

**All 999 population files share one identical 2,911-marker panel** — including across C1 and C2 —
so genotypes stack into a single matrix with no intersection step. Coverage is 100% by *population*
but ~85% by *line* for training years; the 2008 prediction set is 100% covered.

**"Imputed" genotypes are still 12–14% NA.** Handle missingness explicitly.

**Never unzip the genotype archives.** Reading all 999 populations in place takes ~35s and the cost
is CSV parsing, not decompression, so extracting 2.6 GB buys nothing. Run `build_genotypes.py` once
instead: it writes `data/processed/geno_C{1,2}.npz` (int8, ~18 MB each) which reloads in **0.08s**.
Rows are keyed by `LINE_UNIQUE_ID` so the matrix joins straight to phenotype data. Missing calls are
the sentinel **`-2`**, not NaN (int8 can't hold NaN).

**The genetic map lives only in the unimputed archives.** Rows 1–2 of the unimputed CSV are marker
chromosome (1–10) and position in cM — not genotypes. `extract_map.py` pulls them without extracting
the 700 MB file, writing `data/processed/marker_map.csv` in the same column order as the genotype
matrices. 2,911 markers over 1,825 cM, mean spacing **0.63 cM**.

**Imputed progeny genotypes are mostly inferred, not observed.** In the raw data the parents are
densely genotyped (0.8% missing) but each progeny line was skim-genotyped at only ~3% of markers
(96.9% missing); imputation filled the rest using the map and parental haplotypes. Two consequences:
parent genotypes are a much cleaner signal than progeny genotypes, and imputation error is a real
uncertainty source that belongs in the limitations section.

**Genotype file structure:** first two rows are the population's parents (`PID<number>`); progeny
follow with 11-digit zero-padded IDs where `int(id)` is the line number matching the third component
of `LINE_UNIQUE_ID`. Values are `-1 / 0 / 1`.

**Phenotypes are testcrosses, not the lines themselves.** C1 lines are crossed to a C2 tester and
vice versa, so the measured trait reflects **General Combining Ability** — predominantly additive.
Model additive genomic effects; SCA is explicitly out of scope for this track.

**Trait completeness varies widely.** `YLD_BE` (the primary target, bu/acre) and `MST` are ~96–97%
complete; `PHT` ~29%, `EHT` ~22%. Multi-trait work is constrained by height sparsity. Mean yield
trends upward across years (166 → 206), so year effects must be modeled or removed.

## Linking keys

- **Phenotype ↔ Environment:** `YEAR_x` + `LOC` (the environmental file's key column is plain `YEAR`).
- **Phenotype ↔ Genotype:** `LINE_UNIQUE_ID` = `C{cluster}.{population}.{line}` → file
  `ImputedPopulationsC{cluster}/C{cluster}.{population}_Imputed.csv`, row where `int(id) == line`.
- **Geographic:** `LONGITUDE` + `LATITUDE`.

Environmental columns are monthly April–October (`X04`–`X10`) `PRCP`, `TAVG`, `DP01`, `DP10`,
`HTDD`, `CLDD`, plus soil `clay/silt/sand/cfvo/nitrogen/phh2o/soc` at six depths. Soil values need
unit conversion (pH is ×10, texture is ‰) — see `ref/` data dictionary.

## Conventions

Phenotype CSVs need `low_memory=False` or an explicit dtype map; prefer `usecols=` since loading all
33 columns of both clusters costs ~1 GB. Read genotype files directly from the zips (see
`read_genotype()` in `scripts/audit_data.py`) rather than extracting ~600 MB per cluster.

Judging requires a **judge mode** that runs end-to-end on the small sample files without the 141 MB
CSVs, and **baseline models** (phenotypic BLUP, environmental means) to compare against. Keep data
paths and sample sizes configurable rather than hard-coded.
