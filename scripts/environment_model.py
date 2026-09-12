"""Phase 2a: characterise environments, and predict them from weather and soil.

Environment is ~68% of all yield variation -- the single biggest lever in the data.
This script asks whether that lever is *predictable* from the covariates we hold
before planting, which decides two things:

  1. Whether we can say anything about 2008's 38 brand-new locations, which have
     no yield history at all.
  2. Where to spend a reduced plot budget: testing at a site whose productivity we
     cannot anticipate wastes plots.

TWO-WAY EFFECTS, NOT RAW MEANS
------------------------------
A field's raw average yield is confounded with which germplasm happened to be
tested there -- environments carry a median of only 2 populations, so a field that
drew good families looks like a good field. We separate the two with an alternating
fit: estimate field effects holding line effects fixed, then line effects holding
field effects fixed, and repeat until stable. This also gives cleaner line effects
than the simple subtraction used in Phase 1.

HONEST SPLIT
------------
Environment models are trained on 2000-2007 site-years and tested on 2008. Results
are broken out for locations SEEN before versus locations that are genuinely NEW,
because only the second number tells us whether the weather/soil covariates carry
real information rather than just memorising site names.

Usage:
    python scripts/environment_model.py
    python scripts/environment_model.py --clusters 1 --n-types 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

YEAR, TRAIT, ID = "YEAR_x", "YLD_BE", "LINE_UNIQUE_ID"
HOLDOUT = 2008


def two_way_effects(df: pd.DataFrame, iters: int = 30, tol: float = 1e-4) -> tuple:
    """Split yield into a field effect and a line effect by alternating fits.

    Neither can be estimated cleanly on its own because the design is unbalanced:
    good lines are not spread evenly across fields. Alternating converges to the
    additive decomposition y ~ mu + env + line.
    """
    y = df[TRAIT].to_numpy("float64")
    env_key = df["ENV"].to_numpy()
    line_key = df[ID].to_numpy()
    mu = y.mean()

    env_eff = pd.Series(0.0, index=pd.unique(env_key))
    line_eff = pd.Series(0.0, index=pd.unique(line_key))
    prev = None
    for _ in range(iters):
        resid = y - mu - line_eff.reindex(line_key).to_numpy()
        env_eff = pd.Series(resid, index=env_key).groupby(level=0).mean()
        resid = y - mu - env_eff.reindex(env_key).to_numpy()
        line_eff = pd.Series(resid, index=line_key).groupby(level=0).mean()
        cur = env_eff.to_numpy()
        if prev is not None and np.abs(cur - prev).max() < tol:
            break
        prev = cur.copy()
    return env_eff, line_eff, mu


def load_environment(data_root: Path) -> pd.DataFrame:
    env = pd.read_csv(data_root / "environmental_features.csv")
    env["ENV"] = env.YEAR.astype(str) + "_" + env.LOC
    return env


def run(cluster: int, args) -> None:
    print(f"\n{'=' * 70}\nCLUSTER {cluster}\n{'=' * 70}")
    ph = pd.read_csv(
        args.data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID, TRAIT], low_memory=False,
    ).dropna(subset=[TRAIT])
    ph[ID] = ph[ID].astype(str).str.split(".").str[:3].str.join(".")
    ph["ENV"] = ph[YEAR].astype(str) + "_" + ph["LOC"]

    env_eff, line_eff, mu = two_way_effects(ph)
    print(f"two-way fit: {len(env_eff):,} field effects, {len(line_eff):,} line effects")
    print(f"  field effect sd {env_eff.std():.2f} bu/ac  vs  "
          f"line effect sd {line_eff.std():.2f} bu/ac")

    meta = ph.drop_duplicates("ENV").set_index("ENV")[[YEAR, "LOC"]]
    tab = meta.join(env_eff.rename("effect"))
    tab["n_plots"] = ph.groupby("ENV").size()
    # Field effects from a handful of plots are too noisy to model or to score against.
    tab = tab[tab.n_plots >= args.min_plots]

    feats = load_environment(args.data_root).set_index("ENV")
    cols = [c for c in feats.columns if c not in ("YEAR", "LOC", "ENV")]
    tab = tab.join(feats[cols], how="inner")
    print(f"site-years with both an effect and covariates: {len(tab):,}")

    train = tab[tab[YEAR] < HOLDOUT]
    test = tab[tab[YEAR] == HOLDOUT]
    seen_locs = set(train.LOC)
    print(f"  train {len(train):,} site-years  ->  test {len(test):,} "
          f"({(~test.LOC.isin(seen_locs)).sum()} at locations never seen before)")

    X = tab[cols].to_numpy("float64")
    tr = (tab[YEAR] < HOLDOUT).to_numpy()

    # 84 covariates with a condition number in the millions: soil is measured at six
    # depths and monthly temperatures are near-duplicates. PCA de-correlates them.
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    sd_x[sd_x < 1e-8] = 1.0
    Z = (X - mu_x) / sd_x
    pca = PCA(n_components=args.n_components, svd_solver="full", random_state=0).fit(Z[tr])
    P = pca.transform(Z)
    print(f"  {len(cols)} covariates -> {args.n_components} PCs "
          f"({100 * pca.explained_variance_ratio_.sum():.1f}% of variance)")

    ytr = train.effect.to_numpy()
    yte = test.effect.to_numpy()
    model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(P[tr], ytr)
    pred = model.predict(P[~tr])

    def score(mask, label):
        if mask.sum() < 5:
            print(f"   {label:<28} too few site-years ({mask.sum()})")
            return
        r = np.corrcoef(pred[mask], yte[mask])[0, 1]
        rho = spearmanr(pred[mask], yte[mask]).statistic
        rmse = float(np.sqrt(((pred[mask] - yte[mask]) ** 2).mean()))
        print(f"   {label:<28} n={mask.sum():>4}  r={r:>6.3f}  rho={rho:>6.3f}  "
              f"rmse={rmse:>6.2f} bu/ac  (sd={yte[mask].std():.2f})")

    print(f"\n-- predicting 2008 field effects from weather + soil --")
    is_new = (~test.LOC.isin(seen_locs)).to_numpy()
    score(np.ones(len(yte), bool), "all 2008 site-years")
    score(~is_new, "returning locations")
    score(is_new, "NEW locations (no history)")

    # A location's own past average is the obvious competitor: does weather/soil add
    # anything beyond "this site was good last year"?
    hist = train.groupby("LOC").effect.mean()
    base = test.LOC.map(hist)
    have = base.notna().to_numpy()
    if have.sum() >= 5:
        b = base[have].to_numpy()
        print(f"\n   baseline, location's historical mean: n={have.sum():>4}  "
              f"r={np.corrcoef(b, yte[have])[0, 1]:>6.3f}")
        print(f"   covariate model on those same sites:            "
              f"r={np.corrcoef(pred[have], yte[have])[0, 1]:>6.3f}")

    # Environment types: group similar site-years so line-by-environment work later
    # has enough data per group. Individual locations are far too thin.
    km = KMeans(n_clusters=args.n_types, n_init=10, random_state=0).fit(P[tr])
    tab["etype"] = km.predict(P)
    print(f"\n-- {args.n_types} environment types (k-means on climate/soil PCs) --")
    summary = tab.groupby("etype").agg(
        site_years=("effect", "size"), mean_effect=("effect", "mean"),
        sd_effect=("effect", "std"), n_2008=(YEAR, lambda s: (s == HOLDOUT).sum()))
    print(summary.to_string(float_format=lambda v: f"{v:7.2f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = tab[[YEAR, "LOC", "effect", "n_plots", "etype"]].copy()
    out["pred_effect"] = np.nan
    out.loc[~tr, "pred_effect"] = pred
    out.to_csv(args.out_dir / f"environments_C{cluster}.csv")
    line_eff.rename("effect").to_csv(args.out_dir / f"line_effects_C{cluster}.csv")
    print(f"\nwrote {args.out_dir}/environments_C{cluster}.csv and line_effects_C{cluster}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--n-types", type=int, default=6)
    ap.add_argument("--min-plots", type=int, default=20)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
