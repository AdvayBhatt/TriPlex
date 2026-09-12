"""
build_target.py -- Stage 2a: field-adjusted, shrunk line target (YLD_ADJ).

Two problems with raw YLD_BE as a training target:

  1. Fields differ enormously (a line yields 163 bu/ac at one site, 263 at
     another, same year) -- environment dominates raw yield variance. Fix:
     subtract each field's own mean. What's left is "how much better or worse
     than everyone else in that same field", a fair comparison.

  2. A line's mean residual, especially from only 2-7 plots, is noisy. Naive
     two-way (line + field) fitting was tried elsewhere and made things worse,
     because unshrunk line effects are noisy enough that alternating passes
     the noise back and forth between the two effect sets.

This script fixes (2) without reintroducing that problem: field adjustment
stays simple subtraction (step 1, unchanged), but the line-level mean
residual is then shrunk toward zero by an empirical-Bayes (BLUP-style)
factor B_i = sigma2_line / (sigma2_line + sigma2_resid / n_i), estimated
from a one-way random-effects ANOVA of (field-adjusted residual ~ line)
computed on TRAINING-FOLD data only (YEAR <= cutoff). A line seen in only 2
fields gets pulled hard toward zero; one seen in 10 fields barely moves.

The resulting YLD_ADJ is computed for every line, train and held-out alike,
using this one fixed pair of variance components -- so the shrinkage rule
itself never sees the held-out year, even though the same rule is used to
build that year's scoring target.

Usage:
    py scripts/build_target.py --cutoff-year 2007   # train <=2007, score on 2008
    py scripts/build_target.py --cutoff-year 2006   # train <=2006, score on 2007
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def one_way_variance_components(resid: pd.Series, group: pd.Series) -> tuple[float, float]:
    """Method-of-moments (ANOVA) estimate of sigma2_line, sigma2_resid for an
    unbalanced one-way random-effects model: resid_ij = line_i + e_ij."""
    df = pd.DataFrame({"resid": resid.to_numpy(), "group": group.to_numpy()})
    g = df.groupby("group")["resid"]
    n_i = g.size()
    grand_mean = df["resid"].mean()
    group_mean = g.mean()

    k = len(n_i)          # number of lines
    N = int(n_i.sum())    # total observations

    ss_within = ((df["resid"] - group_mean.reindex(df["group"]).to_numpy()) ** 2).sum()
    ms_within = ss_within / (N - k)

    ss_between = (n_i * (group_mean - grand_mean) ** 2).sum()
    ms_between = ss_between / (k - 1)

    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)

    sigma2_resid = float(ms_within)
    sigma2_line = max(float((ms_between - ms_within) / n0), 0.0)
    return sigma2_line, sigma2_resid


def build_target(cluster: int, cutoff_year: int, data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        data_dir / f"clean_C{cluster}.parquet",
        columns=["YEAR", "FIELD", "LINE_UNIQUE_ID", "population", "line_int",
                 "YLD_BE", "parent1", "parent2"],
    )

    # Step 1: field adjustment (simple subtraction, computed within each
    # field -- a field only ever belongs to one year, so this never mixes
    # information across years or across the train/test boundary).
    field_mean = df.groupby("FIELD")["YLD_BE"].transform("mean")
    df["resid"] = df["YLD_BE"] - field_mean

    # Step 2: estimate shrinkage variance components on TRAINING FOLD ONLY.
    train_mask = df["YEAR"] <= cutoff_year
    sigma2_line, sigma2_resid = one_way_variance_components(
        df.loc[train_mask, "resid"], df.loc[train_mask, "LINE_UNIQUE_ID"]
    )
    print(f"  cutoff={cutoff_year}: sigma2_line={sigma2_line:.2f}  "
          f"sigma2_resid={sigma2_resid:.2f}  "
          f"(line variance is {100 * sigma2_line / (sigma2_line + sigma2_resid):.1f}% of the total)")

    # Step 3: apply that fixed shrinkage rule to every line (train + held-out).
    line = df.groupby("LINE_UNIQUE_ID").agg(
        YEAR=("YEAR", "first"),
        population=("population", "first"),
        line_int=("line_int", "first"),
        parent1=("parent1", "first"),
        parent2=("parent2", "first"),
        n_fields=("resid", "size"),
        mean_resid=("resid", "mean"),
    ).reset_index()

    # Compared head-to-head against the plain (unshrunk) mean residual via
    # compare_shrinkage.py: shrinkage did not improve held-out 2008 accuracy
    # and clearly hurt the top-decile "gain" metric (it pulls thin-data lines
    # -- disproportionately the tail we're selecting on -- hardest toward
    # zero). YLD_ADJ is therefore the plain mean residual; the shrunk version
    # is kept alongside for reference, not used downstream.
    B = sigma2_line / (sigma2_line + sigma2_resid / line["n_fields"])
    line["shrinkage_B"] = B
    line["YLD_ADJ"] = line["mean_resid"]
    line["YLD_ADJ_SHRUNK"] = B * line["mean_resid"]
    line["is_train"] = line["YEAR"] <= cutoff_year
    line["is_test"] = line["YEAR"] == cutoff_year + 1

    print(f"  {len(line):,} lines  |  train (<={cutoff_year}): {int(line['is_train'].sum()):,}  "
          f"|  test ({cutoff_year + 1}): {int(line['is_test'].sum()):,}  "
          f"|  mean shrinkage B: {B.mean():.3f} (median n_fields={int(line['n_fields'].median())})")

    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cutoff-year", type=int, required=True,
                     help="train on YEAR<=cutoff, test/score on YEAR==cutoff+1")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for c in args.clusters:
        print(f"\ncluster {c}:")
        line = build_target(c, args.cutoff_year, args.data_dir)
        out_path = args.out_dir / f"target_C{c}_cutoff{args.cutoff_year}.parquet"
        line.to_parquet(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
