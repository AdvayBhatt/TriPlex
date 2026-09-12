"""RETROSPECTIVE DATA-AUDIT IMPORT. See scripts/README.md for limitations.
Use scripts/run_selected_pipeline.py for the January forecast.

Convert the per-population genotype CSVs into one fast-loading matrix per cluster.

The zips hold ~1,000 small CSVs. Reading them is not slow because of compression --
it is slow because of CSV parsing (~35s for all of them, every single time). This
script pays that cost once and writes an int8 matrix that reloads in about a second,
so model iteration never re-parses CSVs.

Do NOT unzip the archives; this reads them in place.

Output (per cluster, in data/processed/):
    geno_C{n}.npz  containing
        markers   (M,)    marker names, shared 2,911-panel
        lines     (L,)    LINE_UNIQUE_ID, e.g. "C1.435.109" -- joins to phenotype data
        X         (L, M)  int8 genotypes for progeny
        parents   (P,)    parent germplasm IDs, e.g. "PID1409169"
        P         (P, M)  int8 genotypes for parents
        parent_pop(P,)    population each parent belongs to

Genotype coding is int8: -1 / 0 / 1 as in the source, with **-2 meaning missing**
(int8 cannot hold NaN). Restore NaN downstream with:
    Xf = X.astype("float32"); Xf[X == MISSING] = np.nan

Usage:
    python scripts/build_genotypes.py
    python scripts/build_genotypes.py --clusters 1
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import genotype_populations, read_genotype

MISSING = -2

# Progeny ids are normally 11-digit zero-padded line numbers, but 6 of the 999
# populations carry a replicate suffix ("000000001.1", "000000001#1") and one
# (C1.126) has corrupted ids ("00000TF%405") with no recoverable line number.
# All six are 2001-2003 populations -- none are in the 2008 prediction set.
PROGENY_ID = re.compile(r"^(\d+)(?:[.#]\d+)?$")


def build_cluster(data_root: Path, cluster: int, out_dir: Path) -> None:
    pops = sorted(genotype_populations(data_root, cluster))
    print(f"[C{cluster}] {len(pops)} populations")

    markers: np.ndarray | None = None
    stats = {"unparsed": 0, "duplicate": 0}
    prog_blocks, prog_ids = [], []
    par_blocks, par_ids, par_pops = [], [], []
    t0 = time.time()

    for i, pop in enumerate(pops, 1):
        g = read_genotype(data_root, cluster, pop)

        # The panel is identical across every population (verified in audit_data.py).
        # Reorder defensively rather than trusting column order, and fail loudly if
        # a population ever breaks the assumption.
        if markers is None:
            markers = g.columns.to_numpy()
        elif not np.array_equal(g.columns.to_numpy(), markers):
            if set(g.columns) != set(markers):
                raise ValueError(f"C{cluster}.{pop} has a different marker panel")
            g = g[markers]

        # int8 with a sentinel: the full matrix is ~460 MB this way vs ~1.8 GB as float32.
        block = g.to_numpy(dtype="float32")
        block = np.where(np.isnan(block), MISSING, block).astype("int8")

        # Rows starting with PID are the population's two parents; the rest are progeny
        # with 11-digit zero-padded ids whose integer value is the line number.
        idx = np.asarray([str(v) for v in g.index], dtype="U16")
        is_parent = np.asarray([s.startswith("PID") for s in idx])

        par_blocks.append(block[is_parent])
        par_ids.append(idx[is_parent])
        par_pops.append(np.full(is_parent.sum(), pop))

        # Keep only progeny whose line number is recoverable, and the first row
        # when a replicate suffix maps two rows onto the same line.
        keep, names, seen = [], [], set()
        for row, raw in enumerate(idx):
            if is_parent[row]:
                continue
            m = PROGENY_ID.match(raw)
            if m is None:
                stats["unparsed"] += 1
                continue
            line_no = int(m.group(1))
            if line_no in seen:
                stats["duplicate"] += 1
                continue
            seen.add(line_no)
            keep.append(row)
            names.append(f"C{cluster}.{pop}.{line_no}")

        prog_blocks.append(block[keep])
        prog_ids.append(np.asarray(names, dtype="U24"))

        if i % 100 == 0 or i == len(pops):
            print(f"  {i}/{len(pops)} populations ({time.time() - t0:.0f}s)")

    X = np.vstack(prog_blocks)
    P = np.vstack(par_blocks)
    lines = np.concatenate(prog_ids)
    parents = np.concatenate(par_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"geno_C{cluster}.npz"
    np.savez_compressed(
        path, markers=markers, lines=lines, X=X,
        parents=parents, P=P, parent_pop=np.concatenate(par_pops),
    )

    print(f"[C{cluster}] progeny {X.shape} | parents {P.shape} "
          f"({len(np.unique(parents))} distinct germplasm)")
    if stats["unparsed"] or stats["duplicate"]:
        print(f"[C{cluster}] dropped {stats['unparsed']} rows with unrecoverable ids, "
              f"{stats['duplicate']} replicate rows")
    print(f"[C{cluster}] missing calls: {(X == MISSING).mean() * 100:.1f}%")
    print(f"[C{cluster}] wrote {path} ({path.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s total)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Data not found at {args.data_root}. See CLAUDE.md for setup.")

    for c in args.clusters:
        build_cluster(args.data_root, c, args.out_dir)


if __name__ == "__main__":
    import sys
    print("Retrospective data-audit analysis; see scripts/README.md. "
          "Use run_selected_pipeline.py for forecasts and plot budgets.", file=sys.stderr)
    main()
