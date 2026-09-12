"""Two diagnostics that decide where the remaining effort should go.

DIAGNOSTIC 1 -- where does our accuracy actually come from?
A line's breeding value splits into two parts:

    GCA(line) = family mean  +  deviation of that line within its family

These have completely different predictability. The family mean is predictable from
how related the family's PARENTS are to parents we have already measured. The
within-family deviation requires marker-QTL associations that survive into a family
whose particular segregating segments were never trained on -- the hard part.

A single pooled ridge conflates them. If nearly all of our r=0.186 is the
between-family component, the model is effectively doing parental GCA prediction and
effort should go there, not into fancier whole-genome regressions.

DIAGNOSTIC 2 -- why do weather covariates fail across years?
Environmental covariates predict a new LOCATION (r=+0.185) but not a new YEAR
(r=-0.030). The suspicion is that the 84 covariates act as a location fingerprint
rather than a season descriptor: half are time-invariant soil, and a location's
monthly weather is dominated by its own climatological normal rather than that
year's anomaly.

Splitting every covariate into

    EC = location climatology  +  within-location anomaly

and fitting the two blocks separately tests this directly. If the anomaly block
explains ~nothing, the null result is demonstrated rather than merely observed.

Usage:
    python scripts/decompose.py --clusters 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

from build_dataset import prepare_markers

YEAR, TRAIT, ID = "YEAR_x", "YLD_BE", "LINE_UNIQUE_ID"
HOLDOUT = 2008


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def between_within(cluster: int, args) -> None:
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines, y, is08 = d["lines"], d["y"].astype("float64"), d["is_2008"]
    tr = ~is08
    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", lines, tr)
    pop = np.array([ln.split(".")[1] for ln in lines])

    alpha = args.alpha
    pred = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[~tr])

    t = pd.DataFrame({"pop": pop[~tr], "y": y[~tr], "pred": pred})
    fam = t.groupby("pop").agg(y=("y", "mean"), pred=("pred", "mean"), n=("y", "size"))
    # Deviations of each line from its own family's mean, on both sides.
    t["y_dev"] = t.y - t["pop"].map(fam.y)
    t["pred_dev"] = t.pred - t["pop"].map(fam.pred)

    overall = corr(t.pred.to_numpy(), t.y.to_numpy())
    between = corr(fam.pred.to_numpy(), fam.y.to_numpy())
    within = corr(t.pred_dev.to_numpy(), t.y_dev.to_numpy())

    # How much of the variance on each side is between-family at all?
    vb_y = fam.y.var() / t.y.var()
    vb_p = fam.pred.var() / t.pred.var()

    print(f"\n{'=' * 72}\nDIAGNOSTIC 1 -- CLUSTER {cluster}: where does the accuracy live?\n{'=' * 72}")
    print(f"{len(t):,} held-out lines in {len(fam)} new families\n")
    print(f"  overall (line level)            r = {overall:+.3f}")
    print(f"  BETWEEN families (family means) r = {between:+.3f}   ({len(fam)} families)")
    print(f"  WITHIN families (deviations)    r = {within:+.3f}")
    print()
    print(f"  share of observed variance that is between-family : {100*vb_y:.1f}%")
    print(f"  share of PREDICTED variance that is between-family: {100*vb_p:.1f}%")

    if np.isfinite(within) and np.isfinite(between):
        if abs(within) < 0.4 * abs(between):
            print("\n  -> Accuracy is overwhelmingly BETWEEN-family. The model is in effect")
            print("     ranking families, not lines within them. Effort belongs in parental")
            print("     GCA prediction and relatedness-weighted training sets.")
        elif abs(within) > 0.8 * abs(between):
            print("\n  -> Both components contribute. Within-family signal is real, so")
            print("     whole-genome regression is doing genuine work.")
        else:
            print("\n  -> Between-family dominates but within-family is not zero.")

    # Predicting the family mean from the PARENTS alone is the natural competitor to
    # a 2,686-marker model, and needs only two genotypes per family.
    par = pd.DataFrame({"pop": pop, "p1": d["parent1"], "p2": d["parent2"]}).drop_duplicates("pop")
    tr_fam = (pd.DataFrame({"pop": pop[tr], "y": y[tr]}).groupby("pop").y.mean()
              .rename("y").reset_index().merge(par, on="pop"))
    stacked = pd.concat([tr_fam[["p1", "y"]].rename(columns={"p1": "p"}),
                         tr_fam[["p2", "y"]].rename(columns={"p2": "p"})])
    pmean = stacked.groupby("p").y.mean()
    te_fam = fam.reset_index().merge(par, on="pop")
    mid = pd.concat([te_fam.p1.map(pmean), te_fam.p2.map(pmean)], axis=1).mean(axis=1)
    ok = mid.notna().to_numpy()
    if ok.sum() >= 5:
        print(f"\n  family means from PARENTS alone (no progeny markers):")
        print(f"     r = {corr(mid[ok].to_numpy(), te_fam.y.to_numpy()[ok]):+.3f} "
              f"on {ok.sum()} of {len(te_fam)} families")
        print(f"     vs marker model on the same families: "
              f"{corr(te_fam.pred.to_numpy()[ok], te_fam.y.to_numpy()[ok]):+.3f}")


def ec_decomposition(cluster: int, args) -> None:
    """Split covariates into location climatology and within-location anomaly."""
    ph = pd.read_csv(args.data_root / f"C{cluster}_Phenotype_Data_V2.csv",
                     usecols=[YEAR, "LOC", TRAIT], low_memory=False).dropna(subset=[TRAIT])
    ph["ENV"] = ph[YEAR].astype(str) + "_" + ph["LOC"]
    eff = (ph.groupby("ENV")[TRAIT].mean() - ph[TRAIT].mean()).rename("effect")
    meta = ph.drop_duplicates("ENV").set_index("ENV")[[YEAR, "LOC"]]
    n = ph.groupby("ENV").size().rename("n")

    env = pd.read_csv(args.data_root / "environmental_features.csv")
    env["ENV"] = env.YEAR.astype(str) + "_" + env.LOC
    cols = [c for c in env.columns if c not in ("YEAR", "LOC", "ENV")]
    tab = meta.join(eff).join(n).join(env.set_index("ENV")[cols], how="inner")
    tab = tab[tab.n >= args.min_plots]

    # climatology = each location's mean covariate value; anomaly = deviation from it
    clim = tab.groupby("LOC")[cols].transform("mean")
    anom = tab[cols] - clim

    print(f"\n{'=' * 72}\nDIAGNOSTIC 2 -- CLUSTER {cluster}: are the covariates a place or a season?\n{'=' * 72}")
    var_clim = clim.var().sum()
    var_anom = anom.var().sum()
    print(f"{len(tab):,} site-years, {len(cols)} covariates")
    print(f"  variance that is location climatology : {100*var_clim/(var_clim+var_anom):.1f}%")
    print(f"  variance that is within-location anomaly: {100*var_anom/(var_clim+var_anom):.1f}%")

    y = tab.effect.to_numpy()
    tr = (tab[YEAR] < HOLDOUT).to_numpy()

    def fit(block: pd.DataFrame, label: str) -> None:
        Z = block.to_numpy("float64")
        mu, sd = Z[tr].mean(0), Z[tr].std(0)
        sd[sd < 1e-8] = 1.0
        Z = (Z - mu) / sd
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Z[tr], y[tr])
        print(f"    {label:<34} 2008 r = {corr(m.predict(Z[~tr]), y[~tr]):+.3f}")

    print("\n  predicting the 2008 field effect from:")
    fit(tab[cols], "full covariates (as before)")
    fit(clim, "location climatology block only")
    fit(anom, "within-location ANOMALY block only")

    hist = tab[tr].groupby("LOC").effect.mean()
    base = tab[~tr].LOC.map(hist)
    ok = base.notna().to_numpy()
    if ok.sum() >= 5:
        print(f"    {'location historical mean':<34} 2008 r = "
              f"{corr(base[ok].to_numpy(), y[~tr][ok]):+.3f}  ({ok.sum()} sites)")
    print("\n  If the anomaly block is ~0, the covariates describe WHERE a trial was,")
    print("  not WHAT the season did -- so they cannot forecast an unseen year.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--alpha", type=float, default=30000)
    ap.add_argument("--min-plots", type=int, default=20)
    args = ap.parse_args()
    for c in args.clusters:
        between_within(c, args)
        ec_decomposition(c, args)


if __name__ == "__main__":
    main()
