"""Phase 1: turn the raw files into one clean, leakage-free modelling table.

WHAT WE ARE PREDICTING, IN PLAIN TERMS
--------------------------------------
Each maize line was grown in several fields. We want one number per line: "how
good is this line's genetics?", so we can rank lines and advance the best ones.

Two problems stand between the raw data and that number.

1. FIELDS DIFFER ENORMOUSLY. Environment is ~68% of all yield variation. A line
   grown in good fields looks good even if its genetics are ordinary. So we do not
   use raw yield. We subtract each field's average yield, which leaves "how much
   better or worse than everyone else in that same field" -- a fair comparison.

2. A SINGLE PLOT IS NOISY. One plot of one line in one field barely predicts the
   same line in another field (r ~ 0.1, see gxe_diagnostic.py). So we average each
   line's adjusted yield over all the fields it was grown in. Averaging cancels
   noise; the line mean is far more reliable than any single plot.

The result is one row per line, with a target called YLD_ADJ.

WHAT WE ARE ALLOWED TO USE AS FEATURES
--------------------------------------
The scenario is January 2008: the lines are about to be planted, nothing has been
harvested. So a feature is legal only if it exists before planting.

  LEGAL    genotype (SNP markers)      -- DNA, known from a seed sample
           pedigree (parent germplasm) -- known from the cross records
           environment covariates      -- weather/soil of the target locations

  ILLEGAL  MST, PHT, EHT, TWT, RTLP, STLP, ERM, and the line's own YLD_BE
           -- all measured AT HARVEST, on the same plot as the yield we predict.
           Using them leaks the answer and cannot be run at decision time.

Those traits are still useful later for multi-trait prediction and the selection
index; they are simply never inputs for predicting yield.

OUTPUT  data/processed/dataset_C{n}.npz
    lines, y (YLD_ADJ), n_obs, is_2008, parent1, parent2, PCs, and the fitted
    marker filter so 2008 lines are transformed exactly like training lines.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --clusters 1 --n-components 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from build_genotypes import MISSING

YEAR, TRAIT, ID = "YEAR_x", "YLD_BE", "LINE_UNIQUE_ID"
HOLDOUT = 2008

# Measured at harvest -- never features for predicting yield. Named explicitly so
# the exclusion is visible rather than implied by omission.
HARVEST_TRAITS = ["ERM", "MST", "PHT", "RTLP", "STLP", "TWT", "EHT"]


def normalize_line_id(s: pd.Series) -> pd.Series:
    """Trim LINE_UNIQUE_ID to the 3-part form the genotype files are keyed by.

    The documented format is C{cluster}.{population}.{line}, but the clusters are
    inconsistent: every C2 id carries a 4th component (C2.1.1.0, where the suffix
    is 0 or 1) and 2,397 C1 ids do too. Joining on the raw id silently matches
    nothing for C2. Trimming to three parts matches 100% of the genotype matrix;
    where a suffix distinguished two records of one line they are averaged, which
    is what line-level aggregation wants anyway.
    """
    return s.astype(str).str.split(".").str[:3].str.join(".")


def load_phenotype(data_root: Path, cluster: int) -> pd.DataFrame:
    df = pd.read_csv(
        data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID, TRAIT, "CROSS"], low_memory=False,
    )
    df[ID] = normalize_line_id(df[ID])
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    return df


def clean(df: pd.DataFrame, min_env_lines: int, min_line_obs: int,
          outlier_z: float) -> tuple[pd.DataFrame, dict]:
    """Standard hygiene, each step reported so nothing is dropped silently."""
    log = {"start": len(df)}

    df = df.dropna(subset=[TRAIT])
    log["no_yield"] = log["start"] - len(df)

    # A field's average is the yardstick we subtract, so it must be well estimated.
    # Fields testing only a handful of lines give unstable averages.
    counts = df.groupby("ENV")[ID].transform("nunique")
    df = df[counts >= min_env_lines]
    log["small_env"] = log["start"] - log["no_yield"] - len(df)

    # Outliers relative to the field they grew in: a 300 bu/acre plot is only
    # suspicious if its neighbours yielded 150.
    g = df.groupby("ENV")[TRAIT]
    z = (df[TRAIT] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    keep = z.abs() <= outlier_z
    dropped_out = int((~keep & z.notna()).sum())
    df = df[keep | z.isna()]
    log["outliers"] = dropped_out

    # Lines grown in too few fields get unreliable means.
    obs = df.groupby(ID)[TRAIT].transform("size")
    before = len(df)
    df = df[obs >= min_line_obs]
    log["thin_lines"] = before - len(df)

    log["final"] = len(df)
    return df, log


def to_line_level(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the field effect, then average per line."""
    df = df.assign(YLD_ADJ=df[TRAIT] - df.groupby("ENV")[TRAIT].transform("mean"))
    out = df.groupby(ID).agg(
        y=("YLD_ADJ", "mean"),
        n_obs=("YLD_ADJ", "size"),
        n_env=("ENV", "nunique"),
        first_year=(YEAR, "min"),
        cross=("CROSS", "first"),
    ).reset_index()
    out["is_2008"] = out.first_year == HOLDOUT
    parts = out.cross.fillna("/").astype(str).str.replace(r"\*\d+", "", regex=True)
    out["parent1"] = parts.str.split("/").str[0]
    out["parent2"] = parts.str.split("/").str[-1]
    return out


def prepare_markers(geno_path: Path, lines, train_mask: np.ndarray,
                    max_missing: float = 0.5, min_maf: float = 0.01,
                    log: dict | None = None) -> np.ndarray:
    """Filter, impute and standardise the marker matrix.

    Everything is estimated on TRAINING lines only -- marker means for imputation,
    and the centre/scale -- so no information from the held-out 2008 cohort leaks
    into the transformation.

    float64 throughout: ridge on ~2,700 correlated columns is numerically delicate
    and float32 produces overflow warnings in the normal equations.
    """
    log = {} if log is None else log
    d = np.load(geno_path, allow_pickle=True)
    pos = pd.Index(d["lines"]).get_indexer(pd.Index(lines))
    if (pos < 0).any():
        raise ValueError(f"{(pos < 0).sum()} lines missing from genotype matrix")

    X = d["X"][pos].astype("float64")
    X[X == MISSING] = np.nan
    log["markers_start"] = X.shape[1]

    # Drop markers that are mostly uncalled -- imputing them invents data.
    keep = np.isnan(X).mean(axis=0) <= max_missing
    log["drop_missing"] = int((~keep).sum())

    # Drop markers with almost no variation: one that is identical in 99% of lines
    # cannot tell them apart. (Coding is -1/0/1, so rescale to 0..1 for frequency.)
    freq = np.nanmean((X + 1) / 2, axis=0)
    keep &= np.minimum(freq, 1 - freq) >= min_maf
    # A marker never observed in training has no mean to impute with.
    keep &= (~np.isnan(X[train_mask])).sum(axis=0) > 0
    log["drop_maf"] = int(X.shape[1] - log["drop_missing"] - keep.sum())

    X = X[:, keep]
    log["markers_kept"] = int(keep.sum())

    col_mean = np.nanmean(X[train_mask], axis=0)
    gaps = np.isnan(X)
    X[gaps] = np.take(col_mean, np.where(gaps)[1])

    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    if not np.isfinite(X).all():
        raise ValueError("non-finite values after standardisation")
    return X


def genomic_pcs(geno_path: Path, lines: pd.Series, train_mask: np.ndarray,
                n_components: int, max_missing: float, min_maf: float,
                log: dict) -> np.ndarray:
    """Compress markers to a few dozen axes, mostly describing family structure.

    Kept as a compact feature set for interaction models later (Phase 5), where a
    full marker matrix crossed with environment terms would be unwieldy. For plain
    yield ranking the full marker matrix is the stronger input -- PCA discards the
    within-family variation that distinguishes siblings.
    """
    X = prepare_markers(geno_path, lines, train_mask, max_missing, min_maf, log)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(X[train_mask])
    log["pc_variance"] = float(pca.explained_variance_ratio_.sum())
    return pca.transform(X).astype("float32")


def build(data_root: Path, cluster: int, out_dir: Path, args) -> None:
    print(f"\n{'=' * 62}\nCLUSTER {cluster}\n{'=' * 62}")
    raw = load_phenotype(data_root, cluster)
    df, log = clean(raw, args.min_env_lines, args.min_line_obs, args.outlier_z)

    print("Cleaning (plot records):")
    print(f"  start                     {log['start']:>9,}")
    print(f"  dropped, no yield         {log['no_yield']:>9,}")
    print(f"  dropped, tiny field       {log['small_env']:>9,}")
    print(f"  dropped, outlier |z|>{args.outlier_z}  {log['outliers']:>9,}")
    print(f"  dropped, line too thin    {log['thin_lines']:>9,}")
    print(f"  kept                      {log['final']:>9,}")

    lvl = to_line_level(df)
    # Keep only lines we have DNA for -- without it there is nothing to predict from.
    geno_path = out_dir / f"geno_C{cluster}.npz"
    have = pd.Index(np.load(geno_path, allow_pickle=True)["lines"])
    before = len(lvl)
    lvl = lvl[lvl[ID].isin(have)].reset_index(drop=True)

    train_mask = (~lvl.is_2008).to_numpy()
    print(f"\nLines: {before:,} -> {len(lvl):,} with genotypes "
          f"({before - len(lvl):,} dropped)")
    print(f"  training (<=2007): {train_mask.sum():,}")
    print(f"  held-out 2008:     {(~train_mask).sum():,}")
    print(f"  fields per line:   median {lvl.n_env.median():.0f}")
    print(f"  target YLD_ADJ:    mean {lvl.y.mean():.2f}, sd {lvl.y.std():.2f} bu/acre")

    glog: dict = {}
    pcs = genomic_pcs(geno_path, lvl[ID], train_mask, args.n_components,
                      args.max_marker_missing, args.min_maf, glog)
    print(f"\nMarkers: {glog['markers_start']:,} -> {glog['markers_kept']:,} "
          f"({glog['drop_missing']:,} too many gaps, {glog['drop_maf']:,} too rare)")
    print(f"  {args.n_components} PCs capture "
          f"{100 * glog['pc_variance']:.1f}% of marker variance")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dataset_C{cluster}.npz"
    np.savez_compressed(
        path,
        lines=lvl[ID].to_numpy(), y=lvl.y.to_numpy("float32"),
        n_obs=lvl.n_obs.to_numpy(), n_env=lvl.n_env.to_numpy(),
        is_2008=lvl.is_2008.to_numpy(),
        parent1=lvl.parent1.to_numpy(), parent2=lvl.parent2.to_numpy(),
        pcs=pcs,
    )
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--min-env-lines", type=int, default=20)
    ap.add_argument("--min-line-obs", type=int, default=2)
    ap.add_argument("--outlier-z", type=float, default=5.0)
    ap.add_argument("--max-marker-missing", type=float, default=0.5)
    ap.add_argument("--min-maf", type=float, default=0.01)
    args = ap.parse_args()

    for c in args.clusters:
        build(args.data_root, c, args.out_dir, args)


if __name__ == "__main__":
    main()
