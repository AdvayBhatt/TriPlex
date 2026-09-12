"""
viz_stats.py -- compute aggregate stats from the cleaned data for visualization.

Reads the cleaned phenotype/genotype files (and, only for trait-completeness,
a few columns from the raw CSVs) and writes one small JSON with everything
a dashboard needs, so the browser never has to touch the multi-hundred-MB
source files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")

CLEANING_FUNNEL = {
    1: {"labels": ["Loaded", "Yield recorded", "Field ≥20 lines",
                   "Within 5 SD", "Line ≥2 fields"],
        "values": [536936, 516032, 516029, 516027, 515784]},
    2: {"labels": ["Loaded", "Yield recorded", "Field ≥20 lines",
                   "Within 5 SD", "Line ≥2 fields"],
        "values": [535340, 511983, 511968, 511963, 511913]},
}

TRAIT_COLS = ["YLD_BE", "MST", "PHT", "EHT", "RTLP", "STLP", "TWT", "ERM"]


def trait_completeness(cluster: int) -> dict:
    path = DATA_RAW / f"C{cluster}_Phenotype_Data_V2.csv"
    df = pd.read_csv(path, usecols=TRAIT_COLS, low_memory=False)
    n = len(df)
    return {t: round(100 * df[t].notna().mean(), 1) for t in TRAIT_COLS}, n


def year_summary(cluster: int) -> dict:
    df = pd.read_parquet(DATA_PROC / f"clean_C{cluster}.parquet",
                          columns=["YEAR", "YLD_BE", "LINE_UNIQUE_ID", "population"])
    g = df.groupby("YEAR")
    out = {
        "years": [int(y) for y in g.size().index],
        "mean_yield": [round(float(v), 2) for v in g["YLD_BE"].mean()],
        "sd_yield": [round(float(v), 2) for v in g["YLD_BE"].std()],
        "n_lines": [int(v) for v in g["LINE_UNIQUE_ID"].nunique()],
        "n_populations": [int(v) for v in g["population"].nunique()],
    }
    return out


def population_overlap(cluster: int) -> dict:
    df = pd.read_parquet(DATA_PROC / f"clean_C{cluster}.parquet",
                          columns=["YEAR", "population"])
    years = sorted(df["YEAR"].unique())
    pops_by_year = {int(y): set(df.loc[df.YEAR == y, "population"]) for y in years}
    pct_new = []
    seen = set()
    for y in years:
        pops = pops_by_year[int(y)]
        new = pops - seen
        pct_new.append(round(100 * len(new) / len(pops), 1) if pops else 0.0)
        seen |= pops
    return {"years": [int(y) for y in years], "pct_new_populations": pct_new}


def marker_qc(cluster: int) -> dict:
    d = np.load(DATA_PROC / f"geno_C{cluster}.npz", allow_pickle=True)
    m = d["matrix"]
    sentinel = int(d["na_sentinel"])
    is_na = m == sentinel

    miss_frac = is_na.mean(axis=0)

    # Simple allele-frequency proxy for MAF from {-1,0,1} dosage coding:
    # p = P(1) + 0.5*P(0) among non-missing calls; MAF = min(p, 1-p)
    valid = ~is_na
    n_valid = valid.sum(axis=0).astype(np.float64)
    n_valid[n_valid == 0] = np.nan
    n_one = (m == 1).sum(axis=0)
    n_zero = (m == 0).sum(axis=0)
    p = (n_one + 0.5 * n_zero) / n_valid
    maf = np.minimum(p, 1 - p)

    def hist(values, edges):
        values = values[~np.isnan(values)]
        counts, _ = np.histogram(values, bins=edges)
        return counts.tolist()

    miss_edges = np.linspace(0, 1, 21)
    maf_edges = np.linspace(0, 0.5, 21)
    return {
        "missing_hist": hist(miss_frac, miss_edges),
        "missing_edges": miss_edges.round(3).tolist(),
        "maf_hist": hist(maf, maf_edges),
        "maf_edges": maf_edges.round(3).tolist(),
        "pct_markers_over_50pct_missing": round(100 * float((miss_frac > 0.5).mean()), 2),
        "pct_markers_under_1pct_maf": round(100 * float((maf < 0.01).mean()), 2),
        "n_markers": int(m.shape[1]),
        "n_lines": int(m.shape[0]),
    }


def main():
    out = {"clusters": {}}
    for c in (1, 2):
        print(f"cluster {c}...")
        completeness, n_rows = trait_completeness(c)
        out["clusters"][str(c)] = {
            "cleaning_funnel": CLEANING_FUNNEL[c],
            "year_summary": year_summary(c),
            "population_overlap": population_overlap(c),
            "marker_qc": marker_qc(c),
            "trait_completeness": completeness,
            "trait_completeness_n": n_rows,
        }
    out_path = DATA_PROC / "viz_stats.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
