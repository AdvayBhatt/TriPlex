"""Two-component prediction: family mean from parents, then rank within family.

WHY SPLIT THE PROBLEM
The decomposition in decompose.py showed that 95.7% of the variance in a single
pooled ridge's predictions is between-family, while only 34% of the variance in real
outcomes is. One flat model over 60,000 lines is therefore solving the family-ranking
problem and barely touching the within-family problem, even though two thirds of the
genetic variation lives there.

These are genuinely different problems with different data:

  family mean       predictable from the two PARENTS, which were densely genotyped
                    (0.8% missing) -- roughly 430 training observations, one per family
  within-family     requires marker-QTL associations that survive into a family whose
                    particular segregating segments were never trained on -- 60,000
                    observations, but the hard part

Fitting them separately lets each use the right data and the right amount of
shrinkage, instead of averaging the two problems together.

RELATEDNESS STRATIFICATION
Published work on this germplasm finds accuracy collapses as the training set becomes
less related to the target: full-sib ~0.59, half-sib ~0.25, unrelated ~0.05. So a
single pooled accuracy figure hides enormous variation. This script classifies each
2008 family by how related its parents are to parents already measured, and reports
accuracy per class -- which says in advance which families are predictable at all.

Usage:
    python scripts/relatedness.py --clusters 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV

from build_dataset import prepare_markers
from build_genotypes import MISSING


def corr(a, b) -> float:
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def midparent_matrix(geno_path: Path, train_pops: np.ndarray) -> tuple:
    """One averaged genotype vector per population, from its two parents.

    Parents carry the cleanest genotypes in the dataset (0.8% missing versus ~13%
    for progeny, because progeny were skim-genotyped and imputed). The mid-parent
    average is the expected genotype of the family, and therefore the natural
    predictor of the family's mean performance.
    """
    d = np.load(geno_path, allow_pickle=True)
    P = d["P"].astype("float64")
    P[P == MISSING] = np.nan
    ppop = d["parent_pop"]

    pops, rows = [], []
    for pop in np.unique(ppop):
        block = P[ppop == pop]
        if len(block) == 0:
            continue
        pops.append(int(pop))
        rows.append(np.nanmean(block, axis=0))
    M = np.vstack(rows)

    # Impute and standardise on TRAINING populations only.
    tr = np.isin(pops, train_pops)
    col = np.nanmean(M[tr], axis=0)
    gaps = np.isnan(M)
    M[gaps] = np.take(np.nan_to_num(col), np.where(gaps)[1])
    keep = np.isfinite(M).all(axis=0) & (M[tr].std(axis=0) > 1e-8)
    M = M[:, keep]
    mu, sd = M[tr].mean(0), M[tr].std(0)
    M = (M - mu) / sd
    return np.array(pops), M, tr


def run(cluster: int, args) -> None:
    print(f"\n{'=' * 76}\nCLUSTER {cluster}\n{'=' * 76}")
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines, y, is08 = d["lines"], d["y"].astype("float64"), d["is_2008"]
    tr = ~is08
    pop = np.array([int(ln.split(".")[1]) for ln in lines])

    geno = args.data_dir / f"geno_C{cluster}.npz"
    X = prepare_markers(geno, lines, tr)

    fam = pd.DataFrame({"pop": pop, "y": y, "is08": is08}).groupby("pop").agg(
        mean=("y", "mean"), n=("y", "size"), is08=("is08", "first"))
    pops, M, mtr = midparent_matrix(geno, fam.index[~fam.is08].to_numpy())
    idx = pd.Index(pops)
    have = fam.index.isin(pops)
    fam = fam[have]
    order = idx.get_indexer(fam.index)
    M = M[order]
    ftr = (~fam.is08).to_numpy()

    print(f"{len(fam)} families with parental genotypes "
          f"({ftr.sum()} training, {(~ftr).sum()} held-out), {M.shape[1]:,} markers")

    # ---- component A: family mean from mid-parent genotypes -------------------
    a = RidgeCV(alphas=np.logspace(0, 6, 30)).fit(M[ftr], fam["mean"].to_numpy()[ftr])
    fam_pred = a.predict(M)
    r_fam = corr(fam_pred[~ftr], fam["mean"].to_numpy()[~ftr])
    print(f"\n-- component A: family mean from PARENTS --")
    print(f"   held-out family-mean accuracy r = {r_fam:+.3f} on {(~ftr).sum()} families")

    # ---- component B: within-family deviation from the line's own markers ------
    fmean = pd.Series(fam["mean"].to_numpy(), index=fam.index)
    dev = y - pd.Series(pop).map(fmean).to_numpy()
    ok = np.isfinite(dev)
    b = Ridge(alpha=args.alpha).fit(X[tr & ok], dev[tr & ok])
    dev_pred = b.predict(X)
    r_dev = corr(dev_pred[~tr], dev[~tr])
    print(f"-- component B: within-family deviation from markers --")
    print(f"   held-out within-family accuracy r = {r_dev:+.3f}")

    # ---- combined, versus the flat pooled model -------------------------------
    combined = pd.Series(pop).map(
        pd.Series(fam_pred, index=fam.index)).to_numpy() + dev_pred
    flat = Ridge(alpha=args.alpha).fit(X[tr], y[tr]).predict(X)
    yte = y[~tr]

    def report(p, label):
        k = max(int(0.10 * len(p)), 1)
        sel = np.argsort(-p)[:k]
        print(f"   {label:<38}r = {corr(p, yte):+.3f}   rho = "
              f"{spearmanr(p, yte).statistic:+.3f}   gain = {yte[sel].mean()-yte.mean():+.2f}")

    print(f"\n-- HELD-OUT 2008, line level --")
    report(flat[~tr], "flat pooled ridge (baseline)")
    report(combined[~tr], "two-component (parents + within)")
    # A blend hedges the two: the flat model is better calibrated overall, the
    # two-component model is sharper between families.
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    report(0.5 * z(flat[~tr]) + 0.5 * z(combined[~tr]), "50/50 blend of the two")

    # ---- relatedness stratification -------------------------------------------
    # Correlation between mid-parent genotype vectors is a cheap proxy for how much
    # ancestry a held-out family shares with anything already measured.
    K = M[~ftr] @ M[ftr].T / M.shape[1]
    best = K.max(axis=1)
    q = np.quantile(best, [1 / 3, 2 / 3])
    band = np.digitize(best, q)
    names = ["least related", "mid", "most related"]

    te_pop = pop[~tr]
    fam_te = fam.index[~ftr].to_numpy()
    band_of = dict(zip(fam_te, band))
    line_band = np.array([band_of.get(p, -1) for p in te_pop])

    print(f"\n-- accuracy by relatedness of the family's parents to training --")
    print(f"   {'band':<16}{'families':>10}{'lines':>8}{'flat r':>9}{'2-comp r':>10}")
    for bi, nm in enumerate(names):
        m = line_band == bi
        if m.sum() < 50:
            continue
        print(f"   {nm:<16}{(band==bi).sum():>10}{m.sum():>8}"
              f"{corr(flat[~tr][m], yte[m]):>9.3f}{corr(combined[~tr][m], yte[m]):>10.3f}")
    print(f"\n   max mid-parent relatedness to training: "
          f"median {np.median(best):.3f}, range {best.min():.3f} to {best.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--alpha", type=float, default=30000)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
