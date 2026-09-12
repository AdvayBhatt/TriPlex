"""
clean_phenotype.py -- Stage 1: phenotype data cleaning.

Reads the raw C1/C2 phenotype CSVs and produces a cleaned, plot-level dataset
per cluster, with every drop counted and printed so nothing disappears silently.

Cleaning steps (in order):
    1. Load only the columns needed -- never load harvest-measured traits
       (MST, PHT, EHT, TWT, RTLP, STLP, ERM) as features. They are target
       leakage for yield prediction: measured at harvest, on the same plot,
       at the same time as YLD_BE.
    2. Parse LINE_UNIQUE_ID -> cluster, population, line integer, trimming to
       the first 3 dot-separated parts (some ids, esp. in C2, carry a 4th
       component from a float-formatted LINE column, e.g. "C2.1.1.0").
    3. Drop rows with no yield recorded.
    4. Drop fields (YEAR x LOC) with fewer than MIN_FIELD_LINES lines -- a
       field's mean is the yardstick every line in it gets compared against;
       too few lines makes that yardstick unstable.
    5. Drop within-field outliers beyond OUTLIER_SD standard deviations of
       that field's own mean (a relative test: a 300 bu/ac plot is only
       suspicious next to 150 bu/ac neighbors, not against a global cutoff).
    6. Drop lines grown in fewer than MIN_LINE_FIELDS fields -- a one-plot
       line mean is not a usable estimate.
    7. Parse CROSS into parent1 / parent2 for the pedigree feature.

2008 rows are NOT excluded here -- they are cleaned by the same row-level
rules as every other year (that is legitimate; it's about plot/field
data quality, not about the label). The train/test year boundary is a
modeling-stage concern, applied when statistics (marker scaling, field-effect
shrinkage, etc.) are fit -- never on rows after the chosen cutoff year.

Usage:
    py scripts/clean_phenotype.py
    py scripts/clean_phenotype.py --clusters 1 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIN_FIELD_LINES = 20
OUTLIER_SD = 5.0
MIN_LINE_FIELDS = 2

# Captures cluster, population, and everything after as the raw line string
# -- LINE is usually a plain integer ("C1.1.191"), but on 4 populations
# (C1.125, C1.39, C2.195, C2.52) it's a genuine decimal ("C1.125.159.2"),
# where the fractional part distinguishes a real second individual sharing
# the same integer line number, not just a float-formatting artifact
# (which shows up as a harmless trailing ".0" almost everywhere in C2).
_ID_RE = re.compile(r"^C(\d+)\.(\d+)\.(.+)$")

RAW_COLS = {
    1: ["YEAR_x", "LOC", "LONGITUDE", "LATITUDE", "YLD_BE", "CLUSTER",
        "LINE_UNIQUE_ID", "CROSS", "GERMPLASM_ID_TESTER"],
    2: ["YEAR_x", "LOC", "LONGITUDE", "LATITUDE", "YLD_BE", "CLUSTER",
        "LINE_UNIQUE_ID", "CROSS", "GERMPLASM_ID_TESTER"],
}


def parse_ids(line_unique_id: pd.Series) -> pd.DataFrame:
    """Extract (cluster, population, line_int) from LINE_UNIQUE_ID.

    The line component is usually a plain integer. Where it carries a
    suffix, separated by either "." (most of C2, plus populations C1.125 /
    C1.39 / C2.52 / C2.195) or "#" (population C1.34 only, e.g.
    "000000325#1") -- a ".0" is a harmless float-formatting artifact and is
    dropped safely, but a nonzero suffix ("1", "2") marks a genuinely
    different individual sharing the same integer line number. Those rows
    are flagged as ambiguous rather than silently collapsed onto the
    integer-only line, since the genotype file's own cleaning already keeps
    only one such individual per integer and there is no reliable way to
    tell, from the phenotype side alone, which of the two a given plot
    record belongs to.
    """
    extracted = line_unique_id.astype(str).str.extract(_ID_RE)
    extracted.columns = ["cluster_id", "population", "line_str"]
    line_parts = extracted["line_str"].str.split(r"[.#]", n=1, expand=True, regex=True)
    line_int = line_parts[0]
    frac = line_parts[1] if line_parts.shape[1] > 1 else pd.Series(
        [None] * len(extracted), index=extracted.index)
    ambiguous = frac.notna() & (frac != "0")

    # A handful of ids (population C1.126) are corrupted beyond recovery,
    # e.g. "00000DS%130" -- no digit-only line number to extract. Blank
    # those out so they fall into the unparseable-id drop, not a crash.
    line_int = line_int.where(line_int.str.match(r"^\d+$"))

    out = pd.DataFrame({
        "cluster_id": extracted["cluster_id"],
        "population": extracted["population"],
        "line_int": line_int,
    }).astype("Int64")
    out["ambiguous_replicate"] = ambiguous.fillna(False)
    return out


def load_raw(path: Path, cluster: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_COLS[cluster], low_memory=False)
    df = df.rename(columns={"YEAR_x": "YEAR"})
    return df


def clean_cluster(cluster: int, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"C{cluster}_Phenotype_Data_V2.csv"
    print(f"\n{'=' * 68}\nCLUSTER {cluster}\n{'=' * 68}")

    df = load_raw(path, cluster)
    n0 = len(df)
    print(f"loaded                                   {n0:>10,}")

    ids = parse_ids(df["LINE_UNIQUE_ID"])
    df = pd.concat([df, ids], axis=1)
    df = df.dropna(subset=["cluster_id", "population", "line_int"])
    n_id = len(df)
    print(f"unparseable LINE_UNIQUE_ID                {n0 - n_id:>10,} dropped "
          f"({n_id:,} remain)")

    n_before_ambig = len(df)
    df = df[~df["ambiguous_replicate"]]
    n_ambig = n_before_ambig - len(df)
    print(f"ambiguous decimal-suffix line id           {n_ambig:>10,} dropped "
          f"({len(df):,} remain)")

    # Step 3: no yield recorded
    df = df.dropna(subset=["YLD_BE"])
    n_yld = len(df)
    print(f"no yield recorded                         {n_id - n_yld:>10,} dropped "
          f"({n_yld:,} remain)")

    # Step 4: fields with too few lines
    df["FIELD"] = df["YEAR"].astype(str) + "_" + df["LOC"].astype(str)
    field_n = df.groupby("FIELD")["LINE_UNIQUE_ID"].transform("nunique")
    keep = field_n >= MIN_FIELD_LINES
    n_field = keep.sum()
    print(f"field with <{MIN_FIELD_LINES} lines                     "
          f"{len(df) - n_field:>10,} dropped ({n_field:,} remain)")
    df = df[keep]

    # Step 5: within-field outliers
    field_mean = df.groupby("FIELD")["YLD_BE"].transform("mean")
    field_std = df.groupby("FIELD")["YLD_BE"].transform("std")
    z = (df["YLD_BE"] - field_mean) / field_std.replace(0, np.nan)
    keep = z.abs() <= OUTLIER_SD
    keep = keep.fillna(True)  # single-std-dev fields (std==0/NaN): nothing to flag
    n_outlier = keep.sum()
    print(f"outlier >{OUTLIER_SD:.0f} SD within its own field        "
          f"{len(df) - n_outlier:>10,} dropped ({n_outlier:,} remain)")
    df = df[keep]

    # Step 6: lines grown in too few fields
    line_n = df.groupby("LINE_UNIQUE_ID")["FIELD"].transform("nunique")
    keep = line_n >= MIN_LINE_FIELDS
    n_line = keep.sum()
    print(f"line grown in <{MIN_LINE_FIELDS} fields                    "
          f"{len(df) - n_line:>10,} dropped ({n_line:,} remain)")
    df = df[keep]

    # Step 7: pedigree
    parts = df["CROSS"].astype(str).str.split("/")
    df["parent1"] = parts.str[0]
    df["parent2"] = parts.str[-1]

    n_2008 = int((df["YEAR"] == 2008).sum())
    n_lines_final = df["LINE_UNIQUE_ID"].nunique()
    print(f"\nfinal: {len(df):,} plot records, {n_lines_final:,} unique lines, "
          f"{n_2008:,} rows in 2008")

    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for c in args.clusters:
        cleaned = clean_cluster(c, args.data_dir)
        out_path = args.out_dir / f"clean_C{c}.parquet"
        cleaned.to_parquet(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
