"""Multivariate line-value estimation: use correlated traits to de-noise yield.

WHY THIS IS DIFFERENT FROM THE MULTI-TRAIT STACKING WE ALREADY REJECTED
-----------------------------------------------------------------------
`multitrait.py` predicted each trait from markers separately and stacked the
predictions. It did not replicate, and it could not, because stacking never
estimates the thing that actually carries information here: the RESIDUAL
covariance between traits measured on the same plot.

In a single-plot trial the error in yield is not independent of the error in
moisture or test weight from that same plot. A poor stand, a wet corner, a
compaction strip perturbs all of them together. So a plot whose moisture and test
weight both came in oddly is a plot whose yield reading should be trusted less --
and partially corrected. Recovering that requires estimating two covariance
matrices and combining them, which is what this does.

THE MODEL
    observed trait means for a line:   ybar = g + e,     e ~ N(0, R / n)
    genetic values across traits:      g    ~ N(0, G)
    best linear prediction:            ghat = G (G + R/n)^-1 ybar

The yield element of ghat borrows from the other traits in proportion to how much
genetic signal they share (G) relative to how much noise they share (R). Because
each line has its own plot count n, lines tested less get shrunk harder -- which is
also the reliability weighting we tried and failed to get right before, now falling
out of the model instead of being bolted on.

G and R are both estimated from the data:
    R  within-line covariance of plot deviations from that line's own mean
    T  between-line covariance of the line means
    G  = T - R * mean(1/n)          (T contains genetic signal plus averaged noise)

VALIDATION
The improved values are used as a better TRAINING TARGET. Scoring stays against the
original, unmodified 2008 line means, so the yardstick does not move and the
comparison is honest.

Usage:
    python scripts/multivariate.py --clusters 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from build_dataset import prepare_markers

YEAR, ID = "YEAR_x", "LINE_UNIQUE_ID"
HOLDOUT = 2008
# Traits with enough coverage to estimate covariances reliably (C1: 96/97/88/68%).
TRAITS = ["YLD_BE", "MST", "TWT", "STLP"]


def nearest_pd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues -- moment estimators of G are not guaranteed PD."""
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps * max(w.max(), 1.0))
    return V @ np.diag(w) @ V.T


def load_plots(data_root: Path, cluster: int) -> pd.DataFrame:
    df = pd.read_csv(data_root / f"C{cluster}_Phenotype_Data_V2.csv",
                     usecols=[YEAR, "LOC", ID] + TRAITS, low_memory=False)
    df[ID] = df[ID].astype(str).str.split(".").str[:3].str.join(".")
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    # Same field adjustment as the univariate pipeline, applied to every trait.
    for c in TRAITS:
        df[c] = df[c] - df.groupby("ENV")[c].transform("mean")
    return df


def estimate_covariances(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split observed variation into within-line (plot noise) and genetic parts."""
    g = df.groupby(ID)
    means = g[TRAITS].mean()
    counts = g[TRAITS].count()

    # R: covariance of plot deviations from each line's own mean -> pure plot noise.
    dev = df[TRAITS] - df[ID].map(means[TRAITS[0]]).to_frame().assign(
        **{c: df[ID].map(means[c]) for c in TRAITS})[TRAITS].to_numpy()
    dev = pd.DataFrame(dev, columns=TRAITS)
    R = dev.cov().to_numpy()
    # Deviations within a line of n plots have n-1 df, so the naive covariance is
    # biased downward by (n-1)/n; rescale by the mean plot count.
    nbar = counts[TRAITS[0]].mean()
    R *= nbar / max(nbar - 1.0, 1.0)

    T = means.cov().to_numpy()
    inv_n = (1.0 / counts.replace(0, np.nan)).mean().to_numpy()
    G = nearest_pd(T - R * np.diag(np.ones(len(TRAITS))) * 0 - R * inv_n.mean())
    return G, R, means.assign(n=counts[TRAITS[0]])


def multivariate_blup(means: pd.DataFrame, G: np.ndarray, R: np.ndarray) -> pd.Series:
    """ghat = G (G + R/n)^-1 ybar, computed per distinct plot count."""
    Y = means[TRAITS].to_numpy("float64")
    # Lines missing a trait get 0 (the field-centred mean), which the shrinkage
    # then pulls back toward zero anyway.
    Y = np.nan_to_num(Y)
    out = np.zeros(len(Y))
    n = means["n"].to_numpy("float64")
    for k in np.unique(n):
        idx = n == k
        W = G @ np.linalg.inv(G + R / max(k, 1.0))
        out[idx] = (Y[idx] @ W.T)[:, 0]     # yield is TRAITS[0]
    return pd.Series(out, index=means.index, name="y_mv")


def run(cluster: int, args) -> None:
    print(f"\n{'=' * 74}\nCLUSTER {cluster}\n{'=' * 74}")
    plots = load_plots(args.data_root, cluster)
    G, R, means = estimate_covariances(plots)

    def corrmat(M):
        d = np.sqrt(np.diag(M))
        return M / np.outer(d, d)

    print("genetic correlations (G) between traits:")
    print(pd.DataFrame(corrmat(G), index=TRAITS, columns=TRAITS).round(3).to_string())
    print("\nplot-level residual correlations (R):")
    print(pd.DataFrame(corrmat(R), index=TRAITS, columns=TRAITS).round(3).to_string())
    gy, ry = corrmat(G)[0, 1:], corrmat(R)[0, 1:]
    print(f"\nyield vs others -- genetic {np.round(gy,3)}  residual {np.round(ry,3)}")
    print("A gap between the two is what the multivariate model exploits;")
    print("if they were identical there would be nothing to gain.")

    mv = multivariate_blup(means, G, R)

    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines, y_orig, is08 = d["lines"], d["y"].astype("float64"), d["is_2008"]
    tr = ~is08
    y_mv = mv.reindex(pd.Index(lines)).to_numpy()
    ok = np.isfinite(y_mv)
    y_mv = np.where(ok, y_mv, 0.0)
    print(f"\nmultivariate estimate available for {ok.sum():,} of {len(lines):,} lines")
    print(f"correlation with the original univariate target: "
          f"{np.corrcoef(y_mv[ok], y_orig[ok])[0,1]:.3f}")

    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", lines, tr)
    yte = y_orig[~tr]

    def score(train_y, label):
        p = Ridge(alpha=args.alpha).fit(X[tr], train_y[tr]).predict(X[~tr])
        k = max(int(0.10 * len(p)), 1)
        sel = np.argsort(-p)[:k]
        print(f"  {label:<40}r = {np.corrcoef(p, yte)[0,1]:+.3f}   "
              f"rho = {spearmanr(p, yte).statistic:+.3f}   "
              f"gain = {yte[sel].mean()-yte.mean():+.2f}")

    print(f"\n-- scored against the ORIGINAL 2008 line means (fixed yardstick) --")
    score(y_orig, "train on univariate target (baseline)")
    score(y_mv, "train on MULTIVARIATE target")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--alpha", type=float, default=30000)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
