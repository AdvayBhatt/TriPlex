"""Turn ranked lines into the actual commercial deliverable.

The model outputs a relative score ("line C1.427.16 scores +14.7"), which is not
something a breeding manager can act on. This script produces the three things the
brief actually asks for:

  1. WHICH LINES TO ADVANCE, under a stated plot budget, each with a prediction
     interval and a flag for whether it is expected to beat its own parents.

  2. EXPECTED YIELD IN BUSHELS, at the specific locations where each line is
     already scheduled to be planted. This is where the field effect belongs: it
     does not change the ranking (every line at a site shares it) but it is
     required to state an absolute number.

  3. WHERE TO SPEND THE PLOTS. A location's own history predicts its productivity
     at r = 0.259, while weather and soil manage -0.14 (see environment_model.py).
     So locations with a track record are the defensible places to test, and
     2008's new sites are flagged as high-variance bets.

UNCERTAINTY
-----------
Two independent sources, combined in quadrature:

  sigma_model  how far a line's true value typically falls from our prediction,
               measured empirically from leave-population-out CV residuals -- so it
               reflects performance on NEW families, which is the real task.
  sigma_field  how much a location's effect swings year to year, from its own
               history. Locations with no history get the spread across all fields,
               which is much wider, and are marked accordingly.

Intervals are deliberately wide. With r ~ 0.19 against a target that is itself
about half noise, narrow intervals would be dishonest.

Usage:
    python scripts/recommend.py --clusters 1 --budget 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from build_dataset import prepare_markers

YEAR, TRAIT, ID = "YEAR_x", "YLD_BE", "LINE_UNIQUE_ID"
SHRINK = 3  # years of history at which a location keeps half its apparent effect
HOLDOUT = 2008


def load_plots(data_root: Path, cluster: int) -> pd.DataFrame:
    df = pd.read_csv(
        data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID, TRAIT], low_memory=False,
    ).dropna(subset=[TRAIT])
    df[ID] = df[ID].astype(str).str.split(".").str[:3].str.join(".")
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    return df


def field_effects(plots: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Each location's historical productivity, and how much it swings by year."""
    grand = plots[TRAIT].mean()
    env = plots.groupby(["ENV", "LOC", YEAR])[TRAIT].mean().reset_index()
    env["effect"] = env[TRAIT] - grand
    hist = env[env[YEAR] < HOLDOUT].groupby("LOC").effect.agg(
        raw_effect="mean", year_to_year_sd="std", years="size")
    # One good season is mostly luck. Shrink toward zero by years of evidence:
    # with SHRINK=3, a single year keeps 1/4 of its apparent effect, six keep 2/3.
    hist["hist_effect"] = hist.raw_effect * hist.years / (hist.years + SHRINK)
    return hist, grand


def run(cluster: int, args) -> None:
    print(f"\n{'=' * 72}\nCLUSTER {cluster}\n{'=' * 72}")
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines, y, is08 = d["lines"], d["y"].astype("float64"), d["is_2008"]
    tr = ~is08
    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", lines, tr)
    pops = np.array([ln.split(".")[1] for ln in lines])

    # Out-of-fold residuals on NEW families give an honest sigma for the intervals.
    oof = np.full(tr.sum(), np.nan)
    Xtr, ytr = X[tr], y[tr]
    for a, b in GroupKFold(n_splits=args.folds).split(Xtr, ytr, pops[tr]):
        oof[b] = Ridge(alpha=args.alpha).fit(Xtr[a], ytr[a]).predict(Xtr[b])
    sigma_model = float(np.nanstd(ytr - oof))
    print(f"leave-population-out residual sd (sigma_model): {sigma_model:.2f} bu/ac")

    model = Ridge(alpha=args.alpha).fit(Xtr, ytr)
    pred = model.predict(X[~tr])

    rec = pd.DataFrame({"line": lines[~tr], "pred_gca": pred})
    rec["population"] = [ln.split(".")[1] for ln in rec.line]

    # Midparent: constant within a population, so a threshold flag, not a ranking key.
    tdf = pd.DataFrame({"y": y[tr], "p1": d["parent1"][tr], "p2": d["parent2"][tr]})
    gca = pd.concat([tdf[["p1", "y"]].rename(columns={"p1": "p"}),
                     tdf[["p2", "y"]].rename(columns={"p2": "p"})]).groupby("p").y.mean()
    t8 = pd.DataFrame({"p1": d["parent1"][~tr], "p2": d["parent2"][~tr]})
    mp = np.column_stack([t8.p1.map(gca), t8.p2.map(gca)]).astype("float64")
    rec["n_parents_known"] = (~np.isnan(mp)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        rec["midparent"] = np.where(rec.n_parents_known > 0, np.nanmean(mp, axis=1), np.nan)
    # pd.NA rather than False where the midparent is unknown: ">" against NaN returns
    # False, which would silently count "unknown" as "does not beat".
    rec["beats_midparent"] = pd.Series(
        np.where(rec.midparent.notna(), rec.pred_gca > rec.midparent, None),
        index=rec.index, dtype="object")

    z = norm.ppf(0.5 + args.interval / 2)
    rec["gca_lo"] = rec.pred_gca - z * sigma_model
    rec["gca_hi"] = rec.pred_gca + z * sigma_model

    plots = load_plots(args.data_root, cluster)
    hist, grand = field_effects(plots)
    all_sd = float(plots.groupby("ENV")[TRAIT].mean().std())

    # Where each 2008 line is actually scheduled -- the brief says locations are known.
    sched = plots[plots[YEAR] == HOLDOUT][[ID, "LOC"]].drop_duplicates()
    sched = sched.merge(rec, left_on=ID, right_on="line")
    sched = sched.join(hist, on="LOC")
    sched["known_site"] = sched.hist_effect.notna()
    sched["field_effect"] = sched.hist_effect.fillna(0.0)
    sched["sigma_field"] = sched.year_to_year_sd.fillna(all_sd)
    sched["expected_yield"] = grand + sched.field_effect + sched.pred_gca
    sigma_tot = np.sqrt(sigma_model**2 + sched.sigma_field**2)
    sched["yield_lo"] = sched.expected_yield - z * sigma_tot
    sched["yield_hi"] = sched.expected_yield + z * sigma_tot

    n_adv = max(int(round(args.budget * len(rec))), 1)
    top = rec.nlargest(n_adv, "pred_gca")
    print(f"\n-- ADVANCE {n_adv} of {len(rec):,} lines "
          f"({100*args.budget:.0f}% budget), {top.population.nunique()} populations --")
    show = top.head(args.show)[
        ["line", "pred_gca", "gca_lo", "gca_hi", "beats_midparent"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    has_mp = int(rec.midparent.notna().sum())
    top_known = top.beats_midparent.notna()
    print(f"\n   midparent known for {has_mp:,} of {len(rec):,} lines "
          f"({rec.n_parents_known.gt(0).sum():,} with >=1 parent seen pre-{HOLDOUT})")
    print(f"   of the {n_adv} advanced, {int(top_known.sum())} have a midparent estimate and "
          f"{int((top.beats_midparent == True).sum())} are predicted to beat it")
    print(f"   {int(args.interval*100)}% interval width: +/-{z*sigma_model:.1f} bu/ac -- "
          f"wide, because accuracy is r~0.19 against a half-noise target")

    print(f"\n-- EXPECTED YIELD at scheduled locations (first {args.show}) --")
    ex = sched.nlargest(args.show, "pred_gca")[
        ["line", "LOC", "known_site", "expected_yield", "yield_lo", "yield_hi"]]
    print(ex.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))

    print(f"\n-- WHERE TO SPEND PLOTS --")
    locs = sched.groupby("LOC").agg(lines_2008=("line", "nunique"),
                                    known=("known_site", "first"),
                                    hist_effect=("field_effect", "first"),
                                    yr_sd=("sigma_field", "first"),
                                    raw_effect=("raw_effect", "first"),
                                    years=("years", "first"))
    locs["years"] = locs.years.fillna(0)
    locs["confidence"] = np.where(locs.known, "has history", "NO history - high risk")
    established = locs[locs.known & (locs.years >= args.min_years)]
    best = established.nlargest(args.show, "hist_effect")
    print(f"   {len(locs)} locations; {int((~locs.known).sum())} have no history at all")
    print(f"   {int((locs.known & (locs.years < args.min_years)).sum())} more have fewer than "
          f"{args.min_years} years, too thin to rank on")
    print(f"\n   best sites with >={args.min_years} years of history "
          f"(effect shrunk toward 0 by years of evidence):")
    print(best[["lines_2008", "raw_effect", "hist_effect", "yr_sd", "years"]].to_string(
        float_format=lambda v: f"{v:7.1f}"))
    if (~locs.known).any():
        print(f"\n   no-history sites carry sigma_field={all_sd:.1f} bu/ac vs "
              f"{locs[locs.known].yr_sd.median():.1f} for established ones --")
        print(f"   testing there buys less information per plot.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec.sort_values("pred_gca", ascending=False).assign(
        advance=lambda f: f.pred_gca >= top.pred_gca.min()
    ).to_csv(args.out_dir / f"recommendations_C{cluster}.csv", index=False)
    locs.to_csv(args.out_dir / f"location_allocation_C{cluster}.csv")
    sched.to_csv(args.out_dir / f"line_by_location_C{cluster}.csv", index=False)
    print(f"\nwrote recommendations_C{cluster}.csv, location_allocation_C{cluster}.csv, "
          f"line_by_location_C{cluster}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--budget", type=float, default=0.10,
                    help="fraction of the cohort the plot budget allows")
    ap.add_argument("--alpha", type=float, default=30000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.80)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--min-years", type=int, default=3,
                    help="years of history a location needs before we rank it")
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
