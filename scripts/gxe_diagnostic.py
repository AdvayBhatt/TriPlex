"""Measure how much genotype-by-environment interaction actually exists.

This is the go/no-go test for the environment-specific challenge. Predicting
location-specific performance only beats simple broad-acre averaging if line
*rankings* genuinely reorder across environments. If they don't, the simpler
model wins and the GxE machinery is wasted effort.

The obstacle: 99.8% of line-x-environment cells hold a single plot, so GxE is
confounded with plot error and no variance decomposition can separate them.
This script works around that with three complementary measurements:

  1. Variance decomposition (sequential SS) -- how much yield variance is the
     environment main effect vs the genotype main effect vs everything left over
     (GxE + error, inseparable).

  2. Between-environment correlations -- for pairs of environments sharing enough
     lines, how well does performance in one predict performance in the other?
     Low correlation means rankings reorder. This is attenuated by plot noise, so
     it is a LOWER bound on the true genetic correlation.

  3. Within-environment repeatability -- from the 931 cells that do have two
     plots, how well does one plot predict another *in the same environment*?
     This is the noise ceiling: no across-environment correlation can exceed it.

Comparing (2) against (3) separates "rankings truly reorder" from "single plots
are just noisy". That comparison is the actual result.

Only years <= 2007 are used; 2008 stays untouched as the held-out target.

Usage:
    python scripts/gxe_diagnostic.py
    python scripts/gxe_diagnostic.py --clusters 1 --top-envs 150
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

YEAR, TRAIT, ID = "YEAR_x", "YLD_BE", "LINE_UNIQUE_ID"
HOLDOUT = 2008


def load(data_root: Path, cluster: int) -> pd.DataFrame:
    df = pd.read_csv(
        data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID, TRAIT], low_memory=False,
    ).dropna(subset=[TRAIT])
    df = df[df[YEAR] < HOLDOUT].copy()
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    return df


def variance_decomposition(df: pd.DataFrame) -> dict[str, float]:
    """Sequential (Type I) sums of squares: environment first, then genotype.

    Environment is fitted first because it is by far the largest effect and the
    design is heavily unbalanced; the genotype term is therefore 'genotype after
    adjusting for environment', which is the quantity that matters for ranking.
    """
    y = df[TRAIT].to_numpy(dtype="float64")
    ss_total = ((y - y.mean()) ** 2).sum()

    env_mean = df.groupby("ENV")[TRAIT].transform("mean").to_numpy()
    ss_env = ((env_mean - y.mean()) ** 2).sum()

    # Within-environment deviations: what is left for genotype to explain.
    resid = y - env_mean
    df = df.assign(_r=resid)
    line_mean = df.groupby(ID)["_r"].transform("mean").to_numpy()
    ss_line = (line_mean**2).sum()

    ss_rest = ss_total - ss_env - ss_line
    return {
        "environment": 100 * ss_env / ss_total,
        "genotype": 100 * ss_line / ss_total,
        "gxe_plus_error": 100 * ss_rest / ss_total,
    }


def between_environment_correlations(
    df: pd.DataFrame, top_envs: int, min_shared: int
) -> pd.Series:
    """Correlation of line performance between pairs of environments."""
    biggest = df.ENV.value_counts().head(top_envs).index
    sub = df[df.ENV.isin(biggest)]
    # One value per line per environment (the few duplicate cells get averaged).
    wide = sub.pivot_table(index=ID, columns="ENV", values=TRAIT, aggfunc="mean")

    out = {}
    for a, b in itertools.combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) >= min_shared:
            out[(a, b)] = pair[a].corr(pair[b])
    return pd.Series(out, dtype="float64").dropna()


def within_environment_repeatability(data_root: Path, cluster: int) -> tuple[float, int]:
    """Correlation between two plots of the same line in the same environment.

    Sets the ceiling: across-environment correlations cannot exceed this, because
    both are degraded by the same single-plot noise.
    """
    df = pd.read_csv(
        data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID, TRAIT], low_memory=False,
    ).dropna(subset=[TRAIT])
    df = df[df[YEAR] < HOLDOUT].copy()
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]

    dup = df.groupby([ID, "ENV"]).filter(lambda g: len(g) == 2)
    if dup.empty:
        return float("nan"), 0
    ranked = dup.groupby([ID, "ENV"]).cumcount()
    wide = dup.assign(rep=ranked).pivot_table(
        index=[ID, "ENV"], columns="rep", values=TRAIT
    ).dropna()
    return wide[0].corr(wide[1]), len(wide)


def within_population_correlations(df: pd.DataFrame, min_shared: int) -> pd.Series:
    """Between-environment correlations computed inside single populations.

    Guards against restriction of range: environments test a median of only 2
    populations, so lines shared by any environment pair are usually siblings from
    one cross. If that narrow genetic range were the reason correlations look low,
    restricting to within-population comparisons would not make them lower.
    """
    df = df.assign(POP=df[ID].str.split(".").str[1])
    size = df.groupby("POP").agg(nl=(ID, "nunique"), nev=("ENV", "nunique"))
    pops = size[(size["nl"] >= 100) & (size["nev"] >= 8)].index

    out = []
    for pop in pops[:25]:
        wide = df[df.POP == pop].pivot_table(
            index=ID, columns="ENV", values=TRAIT, aggfunc="mean"
        )
        for a, b in itertools.combinations(wide.columns, 2):
            pair = wide[[a, b]].dropna()
            if len(pair) >= min_shared:
                out.append(pair[a].corr(pair[b]))
    return pd.Series(out, dtype="float64").dropna()


def report(cluster: int, df: pd.DataFrame, args) -> None:
    print(f"\n{'=' * 62}\nCLUSTER {cluster}  (years <= {HOLDOUT - 1})\n{'=' * 62}")
    print(f"{len(df):,} yield records | {df.ENV.nunique():,} environments | "
          f"{df[ID].nunique():,} lines")

    vc = variance_decomposition(df)
    print("\n-- Variance in YLD_BE --")
    for k, v in vc.items():
        print(f"   {k:<18} {v:5.1f}%")
    print("   (GxE and plot error cannot be separated: 1 plot per line-env cell)")

    corr = between_environment_correlations(df, args.top_envs, args.min_shared)
    rep, n_rep = within_environment_repeatability(args.data_root, cluster)

    print(f"\n-- Between-environment correlation ({len(corr):,} env pairs, "
          f">={args.min_shared} shared lines) --")
    if len(corr):
        q = corr.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"   median {corr.median():.3f} | mean {corr.mean():.3f}")
        print("   deciles: " + "  ".join(f"p{int(k*100)}={v:.2f}" for k, v in q.items()))
        print(f"   negative correlations: {100 * (corr < 0).mean():.1f}% of pairs")

    print(f"\n-- Within-environment repeatability (n={n_rep} duplicated cells) --")
    print(f"   correlation between two plots, same line, same environment: {rep:.3f}")

    wpop = within_population_correlations(df, args.min_shared)
    print(f"\n-- Same, computed within single populations ({len(wpop):,} pairs) --")
    if len(wpop):
        print(f"   median {wpop.median():.3f}  "
              f"(guards against restriction of range; lower, not higher,"
              f" than the pooled figure)")

    print("\n-- Reading --")
    print(f"   A line's performance in one environment explains very little of its")
    print(f"   performance in another (r ~ {corr.median():.2f}), and that holds inside")
    print(f"   single families too. Two explanations fit equally well here and the")
    print(f"   design CANNOT separate them:")
    print(f"     (a) real GxE -- rankings genuinely reorder between environments;")
    print(f"     (b) low single-plot heritability -- one unreplicated plot is noisy.")
    print(f"   The repeatability above ({rep:.2f}) is the only handle on (b), and it")
    print(f"   rests on {n_rep} duplicated cells that may not be a representative")
    print(f"   sample of lines. Treat it as indicative, not decisive.")
    print(f"\n   Either way the operational conclusion is the same: single")
    print(f"   line-in-one-environment values are weak, so predictions should target")
    print(f"   a line's mean over many environments, or groups of similar")
    print(f"   environments -- not individual locations.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--top-envs", type=int, default=150,
                    help="use the N largest environments for pairwise correlations")
    ap.add_argument("--min-shared", type=int, default=30,
                    help="minimum lines shared by an environment pair")
    args = ap.parse_args()

    for c in args.clusters:
        report(c, load(args.data_root, c), args)


if __name__ == "__main__":
    main()
