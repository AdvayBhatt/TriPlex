"""
build_features.py -- Stage 2b: marker feature matrix, per walk-forward fold.

Joins the field-adjusted target table (build_target.py) to the cleaned
genotype matrix (clean_genotypes.py), then does the statistical marker QC
that MUST be fit on training-fold lines only:

    1. Drop markers >50% missing (imputing them would mostly invent data).
    2. Drop markers with MAF <1% (near-fixed across lines -- no
       discriminating power between siblings).
    3. Mean-impute remaining gaps using the TRAINING mean per marker.
    4. Standardize (center/scale) using TRAINING mean/std per marker.

All four statistics come from is_train==True rows only, then are applied
unchanged to the held-out year -- exactly the boundary that keeps the
held-out year's evaluation honest.

Note on ids: the target table's own LINE_UNIQUE_ID can carry the raw
LINE_UNIQUE_ID quirk (a 4th dot-component on ~2,400 C1 rows and effectively
all of C2, from a float-formatted LINE column). The genotype matrix's row
ids are always the clean "C{cluster}.{population}.{line}" form, so the join
key here is rebuilt from the already-parsed population/line_int columns,
not the raw string.

Usage:
    py scripts/build_features.py --cutoff-year 2007
    py scripts/build_features.py --cutoff-year 2006
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_genotypes(path: Path):
    d = np.load(path, allow_pickle=True)
    line_ids = d["line_ids"]
    matrix = d["matrix"]
    marker_names = d["marker_names"]
    na_sentinel = int(d["na_sentinel"])
    idx_map = {lid: i for i, lid in enumerate(line_ids)}
    return idx_map, matrix, marker_names, na_sentinel


def marker_qc_and_scale(X_train: np.ndarray, X_test: np.ndarray, na_sentinel: int,
                         marker_names: np.ndarray):
    is_na_train = X_train == na_sentinel
    n_train = X_train.shape[0]

    miss_frac = is_na_train.mean(axis=0)
    n_valid = (~is_na_train).sum(axis=0).astype(np.float64)
    n_valid_safe = np.where(n_valid == 0, np.nan, n_valid)
    n_one = (X_train == 1).sum(axis=0)
    n_zero = (X_train == 0).sum(axis=0)
    p = (n_one + 0.5 * n_zero) / n_valid_safe
    maf = np.minimum(p, 1 - p)
    maf = np.nan_to_num(maf, nan=0.0)

    keep = (miss_frac <= 0.5) & (maf >= 0.01)
    n_dropped_missing = int((miss_frac > 0.5).sum())
    n_dropped_maf = int((~keep).sum() - n_dropped_missing)
    print(f"  markers: {X_train.shape[1]:,} -> {int(keep.sum()):,}  "
          f"(dropped {n_dropped_missing:,} >50% missing, {n_dropped_maf:,} MAF<1%)")

    Xtr = X_train[:, keep].astype(np.float64)
    Xte = X_test[:, keep].astype(np.float64)
    kept_names = marker_names[keep]

    na_tr = Xtr == na_sentinel
    na_te = Xte == na_sentinel
    Xtr[na_tr] = np.nan
    Xte[na_te] = np.nan

    train_mean = np.nanmean(Xtr, axis=0)
    train_std = np.nanstd(Xtr, axis=0)
    train_std[train_std == 0] = 1.0

    Xtr = np.where(np.isnan(Xtr), train_mean, Xtr)
    Xte = np.where(np.isnan(Xte), train_mean, Xte)

    Xtr = (Xtr - train_mean) / train_std
    Xte = (Xte - train_mean) / train_std

    # float64, not float32: ridge's normal equations on ~2,700 correlated
    # (LD-linked) marker columns are numerically delicate, and float32
    # precision loss shows up as ill-conditioning and pushes the selected
    # alpha to the edge of the search grid to compensate.
    return Xtr.astype(np.float64), Xte.astype(np.float64), kept_names


def build(cluster: int, cutoff_year: int, data_dir: Path):
    target = pd.read_parquet(data_dir / f"target_C{cluster}_cutoff{cutoff_year}.parquet")
    target["LINE_KEY"] = (
        f"C{cluster}." + target["population"].astype("Int64").astype(str)
        + "." + target["line_int"].astype("Int64").astype(str)
    )

    idx_map, matrix, marker_names, na_sentinel = load_genotypes(
        data_dir / f"geno_C{cluster}.npz"
    )

    has_geno = target["LINE_KEY"].isin(idx_map)
    n_before = len(target)
    target = target[has_geno].copy()
    print(f"  lines with genotype: {len(target):,} / {n_before:,} "
          f"({100 * len(target) / n_before:.1f}%)")
    for split, mask in [("train", target["is_train"]), ("test", target["is_test"])]:
        n = int(mask.sum())
        print(f"    {split}: {n:,}")

    rows = target["LINE_KEY"].map(idx_map).to_numpy()
    X_all = matrix[rows]

    train_mask = target["is_train"].to_numpy()
    test_mask = target["is_test"].to_numpy()

    X_train_raw = X_all[train_mask]
    X_test_raw = X_all[test_mask]
    X_train, X_test, kept_markers = marker_qc_and_scale(
        X_train_raw, X_test_raw, na_sentinel, marker_names
    )

    y_train = target.loc[train_mask, "YLD_ADJ"].to_numpy(dtype=np.float32)
    y_test = target.loc[test_mask, "YLD_ADJ"].to_numpy(dtype=np.float32)

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "marker_names": kept_markers,
        "line_ids_train": target.loc[train_mask, "LINE_UNIQUE_ID"].to_numpy(),
        "line_ids_test": target.loc[test_mask, "LINE_UNIQUE_ID"].to_numpy(),
        "population_train": target.loc[train_mask, "population"].to_numpy(),
        "population_test": target.loc[test_mask, "population"].to_numpy(),
        "parent1_train": target.loc[train_mask, "parent1"].to_numpy(),
        "parent2_train": target.loc[train_mask, "parent2"].to_numpy(),
        "parent1_test": target.loc[test_mask, "parent1"].to_numpy(),
        "parent2_test": target.loc[test_mask, "parent2"].to_numpy(),
        "n_fields_train": target.loc[train_mask, "n_fields"].to_numpy(),
        "n_fields_test": target.loc[test_mask, "n_fields"].to_numpy(),
    }
    print(f"  final: X_train {X_train.shape}, X_test {X_test.shape}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cutoff-year", type=int, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for c in args.clusters:
        print(f"\ncluster {c}, cutoff {args.cutoff_year}:")
        result = build(c, args.cutoff_year, args.data_dir)
        out_path = args.out_dir / f"features_C{c}_cutoff{args.cutoff_year}.npz"
        np.savez_compressed(out_path, **result)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
