"""Reproducible audit of the corn breeding dataset.

Regenerates every structural claim the team is relying on, so nobody has to take
the provided documentation (or a teammate) at their word. Several findings here
contradict CORN_BREEDING_DATA_GUIDE.md and the track PDF -- the checks below are
the evidence.

Usage:
    python scripts/audit_data.py                      # writes outputs/data_audit.md
    python scripts/audit_data.py --data-root path/to/dataset
    python scripts/audit_data.py --clusters 1         # skip C2 for a faster run
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# The phenotype CSVs are an un-cleaned merge: the year column is YEAR_x, not YEAR.
YEAR = "YEAR_x"
TRAITS = ["ERM", "MST", "PHT", "RTLP", "STLP", "TWT", "YLD_BE", "EHT"]
TARGET_YEAR = 2008
ID = "LINE_UNIQUE_ID"

# Columns we actually need; loading all 33 for both clusters costs ~1 GB.
PHENO_COLS = [
    YEAR, "LOC", "LONGITUDE", "LATITUDE", ID, "CLUSTER",
    "GERMPLASM_ID", "GERMPLASM_ID_TESTER", "CROSS", "GENERATION_NAME",
] + TRAITS


def population_of(ids: pd.Series) -> pd.Series:
    """C1.42.7 -> 42. Population is the second component of LINE_UNIQUE_ID."""
    return ids.str.split(".").str[1].astype(int)


def load_phenotype(data_root: Path, cluster: int) -> pd.DataFrame:
    path = data_root / f"C{cluster}_Phenotype_Data_V2.csv"
    df = pd.read_csv(path, usecols=PHENO_COLS, low_memory=False)
    df["POP"] = population_of(df[ID])
    return df


def genotype_populations(data_root: Path, cluster: int) -> set[int]:
    """Population numbers that have a genotype file, parsed from names in the zip."""
    zpath = data_root / f"ImputedC{cluster}Populations.zip"
    pattern = re.compile(rf"C{cluster}\.(\d+)_Imputed\.csv$")
    with zipfile.ZipFile(zpath) as z:
        return {int(m.group(1)) for n in z.namelist() if (m := pattern.search(n))}


def read_genotype(data_root: Path, cluster: int, pop: int, **kwargs) -> pd.DataFrame:
    """Read one population's markers straight out of the zip (no 600 MB extraction)."""
    zpath = data_root / f"ImputedC{cluster}Populations.zip"
    inner = f"ImputedPopulationsC{cluster}/C{cluster}.{pop}_Imputed.csv"
    with zipfile.ZipFile(zpath) as z:
        return pd.read_csv(io.BytesIO(z.read(inner)), index_col=0, **kwargs)


def audit_cluster(data_root: Path, cluster: int, out: list[str]) -> None:
    df = load_phenotype(data_root, cluster)
    prior, target = df[df[YEAR] < TARGET_YEAR], df[df[YEAR] == TARGET_YEAR]

    out.append(f"\n## Cluster {cluster} phenotypes\n")
    out.append(f"- Rows: {len(df):,} | unique lines: {df[ID].nunique():,}")
    out.append(f"- Year range: **{df[YEAR].min()}-{df[YEAR].max()}** "
               f"(docs claim 2001-2007)")

    per_year = df.groupby(YEAR).agg(rows=(ID, "size"), mean_yield=("YLD_BE", "mean"))
    out.append("\n| Year | Rows | Mean YLD_BE |\n|---|---|---|")
    for year, r in per_year.iterrows():
        out.append(f"| {year} | {r.rows:,.0f} | {r.mean_yield:.1f} |")

    # The load-bearing finding: is there any line-level history for the target year?
    lines_prior, lines_target = set(prior[ID]), set(target[ID])
    pops_prior, pops_target = set(prior.POP), set(target.POP)
    overlap = len(lines_prior & lines_target)
    out.append(f"\n### {TARGET_YEAR} is an entirely new cohort\n")
    out.append(f"- Lines tested before {TARGET_YEAR}: {len(lines_prior):,}")
    out.append(f"- Lines tested in {TARGET_YEAR}: {len(lines_target):,}")
    out.append(f"- **Overlap: {overlap}** "
               f"({100 * overlap / max(len(lines_target), 1):.1f}%)")
    out.append(f"- Populations in {TARGET_YEAR}: {len(pops_target)} "
               f"({len(pops_target - pops_prior)} never seen before)")
    out.append(f"- {TARGET_YEAR} rows with a YLD_BE label: "
               f"{target.YLD_BE.notna().sum():,} / {len(target):,} "
               f"-- usable as held-out ground truth")

    # If no line history carries over, these are the only bridges to the target year.
    for label, col in [("Testers", "GERMPLASM_ID_TESTER"), ("Parent germplasm", "GERMPLASM_ID")]:
        t, p = set(target[col].dropna()), set(prior[col].dropna())
        out.append(f"- {label} in {TARGET_YEAR}: {len(t)}, of which {len(t & p)} appear pre-{TARGET_YEAR}")

    # Location generalization: LOC as a plain categorical cannot cover unseen sites.
    locs_prior, locs_target = set(prior.LOC), set(target.LOC)
    out.append(f"\n### Locations\n")
    out.append(f"- Distinct locations overall: **{df.LOC.nunique()}** (docs list ~7)")
    out.append(f"- {TARGET_YEAR} locations: {len(locs_target)}, "
               f"of which {len(locs_target - locs_prior)} are new")

    out.append(f"\n### Trait completeness (% non-null)\n")
    out.append("| Trait | Overall | " + str(TARGET_YEAR) + " |\n|---|---|---|")
    for t in TRAITS:
        out.append(f"| {t} | {df[t].notna().mean() * 100:.1f}% | "
                   f"{target[t].notna().mean() * 100:.1f}% |")

    # Genotype coverage decides whether genomic prediction is even possible here.
    gpops = genotype_populations(data_root, cluster)
    covered = len(pops_target & gpops)
    out.append(f"\n### Genotype coverage\n")
    out.append(f"- Population files available: {len(gpops)}")
    out.append(f"- {TARGET_YEAR} populations with genotypes: {covered}/{len(pops_target)} "
               f"({100 * covered / max(len(pops_target), 1):.0f}%)")
    all_pops = set(df.POP)
    out.append(f"- All phenotyped populations with genotypes: "
               f"{len(all_pops & gpops)}/{len(all_pops)}")

    # Marker panels must be identical for populations to stack into one matrix.
    sample_pops = sorted(gpops)[:: max(len(gpops) // 4, 1)][:4]
    panels = {p: set(read_genotype(data_root, cluster, p, nrows=2).columns) for p in sample_pops}
    sizes = {len(v) for v in panels.values()}
    shared = len(set.intersection(*panels.values()))
    out.append(f"- Marker panel across populations {sample_pops}: "
               f"sizes={sizes}, shared={shared} "
               f"-- {'identical, so genotypes stack directly' if len(sizes) == 1 and shared == sizes.pop() else 'DIFFERENT, intersection required'}")

    geno = read_genotype(data_root, cluster, sample_pops[0])
    parents = [i for i in geno.index if str(i).startswith("PID")]
    out.append(f"- Example file C{cluster}.{sample_pops[0]}: {geno.shape[0]} rows x "
               f"{geno.shape[1]} markers; first rows are parents {parents[:2]}")
    out.append(f"- Missingness in *imputed* genotypes: "
               f"**{geno.isna().to_numpy().mean() * 100:.1f}%**")
    out.append(f"- Genotype call values: "
               f"{sorted(pd.unique(geno.to_numpy().ravel())[:5].tolist(), key=str)}")


def audit_environment(data_root: Path, out: list[str]) -> None:
    env = pd.read_csv(data_root / "environmental_features.csv")
    out.append("\n## Environmental features\n")
    out.append(f"- Shape: {env.shape[0]:,} rows x {env.shape[1]} columns "
               f"(key: YEAR + LOC)")
    out.append(f"- Year range: {env.YEAR.min()}-{env.YEAR.max()} | "
               f"distinct locations: {env.LOC.nunique()}")
    out.append(f"- Missing values: **{env.isna().to_numpy().mean() * 100:.2f}%**")
    out.append(f"- Rows for {TARGET_YEAR}: {(env.YEAR == TARGET_YEAR).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", type=Path, default=Path("outputs/data_audit.md"))
    args = ap.parse_args()

    if not args.data_root.exists():
        raise SystemExit(
            f"Data not found at {args.data_root}.\n"
            "Symlink or copy the dataset there, e.g.\n"
            '  ln -s "/path/to/Simplified Hackathon Dataset V3" data/raw'
        )

    out = ["# Data audit", "",
           "Generated by `scripts/audit_data.py`. Every number below is recomputed "
           "from the raw files; several contradict the provided documentation."]

    audit_environment(args.data_root, out)
    for c in args.clusters:
        audit_cluster(args.data_root, c, out)

    report = "\n".join(out) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
