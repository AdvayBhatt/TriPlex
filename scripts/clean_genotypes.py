"""
clean_genotypes.py -- Stage 1: genotype data cleaning (structural).

Reads every per-population imputed genomic CSV for a cluster and stacks them
into one matrix, with the same row-level scrutiny the phenotype cleaning
script applies to plot records.

Per population file:
    - The first two rows are the population's parents ("PID..."), not
      progeny -- they are pulled out into a separate parent table, never
      mixed into the line-level matrix.
    - Progeny row ids are normally an 11-digit zero-padded line number
      ("00000000001"). A handful of populations carry a replicate suffix
      (".1" / "#1") -- the leading 11 digits are the real line number, and
      when a suffix maps two rows to the same line, the first is kept.
      A few rows are corrupted beyond recovery (no 11-digit run at all) and
      are dropped, population and all, since there's no line number to
      recover.
    - Every row is renamed to LINE_UNIQUE_ID = "C{cluster}.{population}.{line}"
      so it lines up directly with the cleaned phenotype table's own id.

This stage does NOT do statistical marker QC (missingness / MAF filtering,
imputation, scaling) -- those thresholds must be fit on the training
(pre-cutoff-year) lines only, which depends on which walk-forward fold is
being run. That happens at the feature-extraction stage, per fold. Here we
only build the clean, complete (including NAs) structural matrix once.

Usage:
    py scripts/clean_genotypes.py
    py scripts/clean_genotypes.py --clusters 1 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NA_SENTINEL = -9  # valid dosages are {-1, 0, 1}; -9 marks a missing call
# Progeny ids are normally 11-digit zero-padded, but a few populations use a
# shorter 9-digit base with a replicate suffix ("000000337#1", "000000189.1")
# -- accept any leading digit run, with an optional "." or "#" suffix.
_PROGENY_RE = re.compile(r"^(\d+)(?:[.#]\d+)?$")

GENO_DIRS = {
    1: "genotypes/C1/ImputedPopulationsC1",
    2: "genotypes/C2/ImputedPopulationsC2",
}


def load_population(path: Path, cluster: int, population: int):
    """Return (parent_df, progeny_df) for one population file, or (None, None)
    if the file is unreadable. progeny_df is indexed by LINE_UNIQUE_ID."""
    df = pd.read_csv(path, index_col=0)

    is_parent = df.index.astype(str).str.startswith("PID")
    parents = df[is_parent].copy()

    progeny = df[~is_parent].copy()
    ids = pd.Series(progeny.index.astype(str), index=progeny.index)
    line_int = ids.str.extract(_PROGENY_RE)[0]
    n_corrupt = int(line_int.isna().sum())
    progeny = progeny[line_int.notna()].copy()
    line_int = line_int.dropna().astype(int)

    # Duplicate suffix rows (e.g. "00000000001" and "00000000001.1") -> keep first
    dup = line_int.duplicated(keep="first")
    n_dup = int(dup.sum())
    progeny = progeny[~dup.to_numpy()]
    line_int = line_int[~dup]

    progeny.index = [f"C{cluster}.{population}.{i}" for i in line_int]
    progeny.index.name = "LINE_UNIQUE_ID"

    return parents, progeny, n_corrupt, n_dup


def clean_cluster(cluster: int, data_dir: Path):
    geno_dir = data_dir / GENO_DIRS[cluster]
    files = sorted(geno_dir.glob(f"C{cluster}.*_Imputed.csv"),
                    key=lambda p: int(re.search(r"\.(\d+)_", p.name).group(1)))
    print(f"\n{'=' * 68}\nCLUSTER {cluster} genotypes: {len(files)} population files\n"
          f"{'=' * 68}")

    parent_frames, progeny_frames = [], []
    master_cols = None
    n_corrupt_total = n_dup_total = n_dropped_pops = 0

    for path in files:
        population = int(re.search(r"\.(\d+)_", path.name).group(1))
        parents, progeny, n_corrupt, n_dup = load_population(path, cluster, population)
        n_corrupt_total += n_corrupt
        n_dup_total += n_dup

        if master_cols is None:
            master_cols = list(parents.columns)
        # Align columns defensively -- verified identical across files, but
        # reindex rather than assume, in case a population differs.
        parents = parents.reindex(columns=master_cols)
        progeny = progeny.reindex(columns=master_cols)

        if len(progeny) == 0:
            n_dropped_pops += 1
            continue

        parent_frames.append(parents)
        progeny_frames.append(progeny)

    parents_all = pd.concat(parent_frames)
    progeny_all = pd.concat(progeny_frames)

    print(f"corrupted progeny ids (no recoverable line number): {n_corrupt_total:,} dropped")
    print(f"duplicate-suffix rows (kept first occurrence):       {n_dup_total:,} dropped")
    print(f"populations with zero usable progeny after cleaning: {n_dropped_pops}")
    print(f"markers: {len(master_cols):,}")
    print(f"final: {len(progeny_all):,} progeny lines, {len(parents_all):,} parent rows")

    geno_matrix = progeny_all.fillna(NA_SENTINEL).to_numpy(dtype=np.int8)
    parent_matrix = parents_all.fillna(NA_SENTINEL).to_numpy(dtype=np.int8)
    pct_missing = float((geno_matrix == NA_SENTINEL).mean()) * 100
    print(f"missing calls in progeny matrix: {pct_missing:.1f}%")

    return {
        "line_ids": progeny_all.index.to_numpy(),
        "matrix": geno_matrix,
        "marker_names": np.array(master_cols),
        "parent_ids": parents_all.index.to_numpy(),
        "parent_matrix": parent_matrix,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for c in args.clusters:
        result = clean_cluster(c, args.data_dir)
        out_path = args.out_dir / f"geno_C{c}.npz"
        np.savez_compressed(out_path, na_sentinel=NA_SENTINEL, **result)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
