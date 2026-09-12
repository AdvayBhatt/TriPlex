"""Genetic correlation between environment types, with MATCHED reliabilities.

The earlier estimate compared a cross-type correlation (+0.357) against a
within-type split-half floor (+0.265) and inferred rg ~ 0.85. That comparison is not
sound: the two sides average over different numbers of site-years, so they carry
different amounts of noise, and taken literally the arithmetic implies rg > 1.

This recomputes it so both sides are measured the same way. Each environment type's
site-years are randomly split in half. Then for a pair of types A and B:

    within  = corr( line means in A1 , line means in A2 )   <- same type, half data
    cross   = corr( line means in A1 , line means in B1 )   <- different type, half data

Both use half-sized samples on the same lines, so plot noise attenuates both equally
and the ratio estimates the genetic correlation directly:

    rg ~ cross / sqrt( within_A * within_B )

Averaged over many random splits to remove split-to-split luck.
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd

def run(cluster: int, args) -> None:
    ph = pd.read_csv(args.data_root / f"C{cluster}_Phenotype_Data_V2.csv", low_memory=False,
                     usecols=["YEAR_x","LOC","LINE_UNIQUE_ID","YLD_BE"]).dropna(subset=["YLD_BE"])
    ph["LINE_UNIQUE_ID"] = ph.LINE_UNIQUE_ID.astype(str).str.split(".").str[:3].str.join(".")
    ph["ENV"] = ph.YEAR_x.astype(str) + "_" + ph.LOC
    ph["adj"] = ph.YLD_BE - ph.groupby("ENV").YLD_BE.transform("mean")
    env = pd.read_csv(args.data_dir / f"environments_C{cluster}.csv", index_col=0)
    ph = ph.join(env["etype"], on="ENV").dropna(subset=["etype"])
    ph["etype"] = ph.etype.astype(int)

    types = sorted(ph.etype.unique())
    rng = np.random.default_rng(0)
    cross_acc, within_acc, rg_acc = {}, {t: [] for t in types}, []

    for rep in range(args.reps):
        halves = {}
        for t in types:
            evs = ph.loc[ph.etype == t, "ENV"].unique()
            rng.shuffle(evs)
            halves[t] = (set(evs[: len(evs) // 2]), set(evs[len(evs) // 2:]))

        def means(t, half):
            s = ph[(ph.etype == t) & ph.ENV.isin(halves[t][half])]
            g = s.groupby("LINE_UNIQUE_ID").adj
            m, n = g.mean(), g.size()
            return m[n >= args.min_plots]

        m = {(t, h): means(t, h) for t in types for h in (0, 1)}
        for t in types:
            j = pd.concat([m[(t, 0)], m[(t, 1)]], axis=1, join="inner").dropna()
            if len(j) >= args.min_lines:
                within_acc[t].append(j.iloc[:, 0].corr(j.iloc[:, 1]))
        for a, b in itertools.combinations(types, 2):
            j = pd.concat([m[(a, 0)], m[(b, 0)]], axis=1, join="inner").dropna()
            if len(j) >= args.min_lines:
                cross_acc.setdefault((a, b), []).append(
                    (j.iloc[:, 0].corr(j.iloc[:, 1]), len(j)))

    print(f"\n{'='*72}\nCLUSTER {cluster} -- matched-reliability genetic correlation\n{'='*72}")
    wi = {t: np.nanmean(v) for t, v in within_acc.items() if v}
    print("within-type repeatability (half vs half, same type):")
    for t, v in sorted(wi.items()):
        print(f"   type {t}: {v:+.3f}   ({len(within_acc[t])} reps)")
    print("\ncross-type correlation and implied genetic correlation:")
    print(f"   {'pair':<10}{'n lines':>9}{'cross r':>10}{'rg':>9}")
    for (a, b), vals in sorted(cross_acc.items()):
        if a not in wi or b not in wi or wi[a] <= 0 or wi[b] <= 0:
            continue
        cr = np.nanmean([v[0] for v in vals]); nl = int(np.mean([v[1] for v in vals]))
        rg = cr / np.sqrt(wi[a] * wi[b])
        rg_acc.append(min(rg, 1.5))
        print(f"   {a} vs {b:<5}{nl:>9}{cr:>10.3f}{rg:>9.3f}")
    if rg_acc:
        print(f"\n   median implied rg across type pairs: {np.median(rg_acc):.3f}")
        print(f"   (values above 1.0 are sampling noise; rg cannot exceed 1)")
        print(f"\n   Break-even for splitting the programme is rg ~ 0.78.")
        v = np.median(rg_acc)
        print(f"   -> {'ONE national list remains correct' if v > 0.78 else 'splitting may be justified'}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--min-plots", type=int, default=2)
    ap.add_argument("--min-lines", type=int, default=100)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)

if __name__ == "__main__":
    main()
