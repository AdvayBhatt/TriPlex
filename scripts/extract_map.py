"""Extract the genetic linkage map from the unimputed genome archives.

The map is the one thing the *imputed* per-population files do not contain. In the
unimputed CSV the first two data rows are not genotypes: row 1 is each marker's
chromosome (1-10) and row 2 is its position in centiMorgans. Identical in C1 and C2.

Only the first few MB of the ~700 MB CSV are read, so nothing is extracted to disk.

Output: data/processed/marker_map.csv  (marker, chromosome, cM), in file order --
which is the same column order as geno_C{1,2}.npz.

Usage:
    python scripts/extract_map.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

META_COLS = 4  # projects, projectID, shorthand, LINE


def extract(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if not n.startswith("__MACOSX"))
        with z.open(name) as f:
            # Three lines is all we need; read a chunk rather than the whole member.
            head = f.read(4_000_000).decode("utf8", errors="replace").split("\n")

    fields = [[c.strip() for c in head[i].split(",")][META_COLS:] for i in range(3)]
    markers, chrom, cm = fields

    df = pd.DataFrame({
        "marker": markers,
        "chromosome": [int(c) for c in chrom],
        "cM": [float(p) for p in cm],
    })
    if df.marker.duplicated().any():
        raise ValueError("duplicate marker names in map")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/marker_map.csv"))
    args = ap.parse_args()

    c1 = extract(args.data_root / "Unimputed_C1_Genome_Data.zip")
    c2 = extract(args.data_root / "Unimputed_C2_Genome_Data.zip")
    if not c1.equals(c2):
        raise ValueError("C1 and C2 maps differ -- they are expected to be identical")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    c1.to_csv(args.out, index=False)

    span = c1.groupby("chromosome").cM.max().sum()
    print(f"{len(c1)} markers across {c1.chromosome.nunique()} chromosomes")
    print(f"total map length {span:.0f} cM | mean spacing {span / len(c1):.2f} cM")
    print(c1.groupby("chromosome").agg(markers=("marker", "size"),
                                       max_cM=("cM", "max")).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
