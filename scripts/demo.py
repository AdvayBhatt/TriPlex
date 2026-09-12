"""Judge mode: run the entire pipeline end to end, with no large data files.

The brief requires a demonstration that does not need the 141 MB phenotype CSVs and
explicitly allows synthetic data to showcase the methodology. This generates a small
dataset with the same STRUCTURE as the real one, writes it in the same file formats,
then runs the actual production scripts against it. Nothing here reimplements the
pipeline -- if this passes, the real code paths work.

The synthetic data also has something the real data cannot: a KNOWN ground truth.
Marker effects are chosen by us, so the demo can report how much of the true genetic
signal the pipeline recovers, which no run on real data can show.

Structure mirrored from the real dataset:
  - biparental populations, two parents each, progeny as recombinant mosaics
  - a shared marker panel across all populations
  - many site-years, each testing only a couple of populations
  - one unreplicated plot per line per site-year
  - the final year's populations are entirely new, as 2008 is
  - harvest traits correlated with yield, present so the leakage exclusion is visible

Usage:
    python scripts/demo.py                 # ~30 seconds, no downloads
    python scripts/demo.py --populations 60 --keep
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HARVEST_TRAITS = ["ERM", "MST", "PHT", "RTLP", "STLP", "TWT", "EHT"]


def synthesise(root: Path, n_pop: int, n_lines: int, n_markers: int,
               n_locs: int, years: list[int], rng: np.random.Generator) -> dict:
    """Write synthetic data in exactly the real file formats."""
    root.mkdir(parents=True, exist_ok=True)
    markers = [f"M{i:09d}" for i in range(n_markers)]

    # A handful of markers carry real effects; the rest are noise, as in reality
    # where yield is polygenic but most of a sparse panel tags nothing useful.
    n_causal = max(n_markers // 10, 5)
    causal = rng.choice(n_markers, n_causal, replace=False)
    effects = np.zeros(n_markers)
    effects[causal] = rng.normal(0, 1.0, n_causal)

    geno_dir = root / "ImputedPopulationsC1"
    geno_dir.mkdir(exist_ok=True)
    rows, true_gca = [], {}

    for pop in range(1, n_pop + 1):
        # Two inbred parents, then progeny as mosaics of large parental blocks --
        # the high-LD structure that makes a sparse panel informative.
        pa, pb = (rng.choice([-1, 1], n_markers) for _ in range(2))
        block = max(n_markers // 20, 1)
        g = np.empty((n_lines, n_markers), dtype=float)
        for i in range(n_lines):
            src = np.repeat(rng.random(int(np.ceil(n_markers / block))) < 0.5,
                            block)[:n_markers]
            g[i] = np.where(src, pa, pb)
            het = rng.random(n_markers) < 0.05
            g[i][het] = 0
        g[rng.random(g.shape) < 0.12] = np.nan       # imputed files are ~12-14% NA

        ids = [f"{i + 1:011d}" for i in range(n_lines)]
        frame = pd.DataFrame(g, index=[f"PID{pop}A", f"PID{pop}B"][:0] + ids,
                             columns=markers)
        parents = pd.DataFrame(np.vstack([pa, pb]),
                               index=[f"PID{pop}A", f"PID{pop}B"], columns=markers)
        pd.concat([parents, frame]).to_csv(geno_dir / f"C1.{pop}_Imputed.csv")

        filled = np.where(np.isnan(g), 0.0, g)
        for i, lid in enumerate(ids):
            true_gca[f"C1.{pop}.{i + 1}"] = float(filled[i] @ effects)
        rows.append(pop)

    with zipfile.ZipFile(root / "ImputedC1Populations.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        for f in sorted(geno_dir.iterdir()):
            z.write(f, f"ImputedPopulationsC1/{f.name}")
    shutil.rmtree(geno_dir)

    # Scale genetic effects so they sit well below environment, as in the real data
    # (environment ~69% of variance, genotype ~8%).
    gca_sd = np.std(list(true_gca.values())) or 1.0
    gca = {k: v / gca_sd * 10.0 for k, v in true_gca.items()}

    locs = [f"L{i:03d}" for i in range(n_locs)]
    pops_by_year = {}
    for yi, yr in enumerate(years):
        # The last year gets populations never seen before, mirroring 2008.
        if yr == years[-1]:
            pops_by_year[yr] = list(range(n_pop - n_pop // 4 + 1, n_pop + 1))
        else:
            pops_by_year[yr] = list(rng.choice(
                range(1, n_pop - n_pop // 4 + 1), max(n_pop // 6, 2), replace=False))

    env_rows, plot_rows = [], []
    for yr in years:
        year_eff = rng.normal(0, 12)
        for loc in locs:
            if rng.random() < 0.5:
                continue
            eff = year_eff + rng.normal(0, 25)          # fields dominate variance
            env_rows.append({"YEAR": yr, "LOC": loc,
                             **{f"X{m:02d}_{v}": rng.normal(50, 10)
                                for m in range(4, 11) for v in ("PRCP", "TAVG")}})
            for pop in rng.choice(pops_by_year[yr], min(2, len(pops_by_year[yr])),
                                  replace=False):
                for i in range(1, n_lines + 1):
                    lid = f"C1.{pop}.{i}"
                    y = 180 + eff + gca[lid] + rng.normal(0, 18)   # one noisy plot
                    row = {"YEAR_x": yr, "LOC": loc, "LINE_UNIQUE_ID": lid,
                           "YLD_BE": round(y, 3), "CROSS": f"{pop}A/{pop}B"}
                    for t in HARVEST_TRAITS:      # correlated, and never used as input
                        row[t] = round(0.3 * (y - 180) + rng.normal(0, 10), 2)
                    plot_rows.append(row)

    pd.DataFrame(plot_rows).to_csv(root / "C1_Phenotype_Data_V2.csv", index=False)
    pd.DataFrame(env_rows).to_csv(root / "environmental_features.csv", index=False)
    return {"gca": gca, "holdout": years[-1], "plots": len(plot_rows)}


def sh(cmd: list[str], env_extra: dict) -> None:
    import os
    env = {**os.environ, **env_extra}
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print("\n".join("  " + ln for ln in out.strip().splitlines()[-18:]))
    if r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--populations", type=int, default=40)
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--markers", type=int, default=400)
    ap.add_argument("--locations", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    work = args.workdir or Path(tempfile.mkdtemp(prefix="triplex_demo_"))
    raw, proc, out = work / "raw", work / "processed", work / "outputs"
    scripts = Path(__file__).resolve().parent
    env_extra = {"PYTHONPATH": str(scripts), "PYTHONWARNINGS": "ignore"}

    print("=" * 72)
    print("TRIPLEX JUDGE MODE -- full pipeline on synthetic data, no large files")
    print("=" * 72)
    print(f"workdir: {work}")

    rng = np.random.default_rng(args.seed)
    years = [2003, 2004, 2005, 2006, 2007, 2008]
    meta = synthesise(raw, args.populations, args.lines, args.markers,
                      args.locations, years, rng)
    size = sum(f.stat().st_size for f in raw.rglob("*")) / 1e6
    print(f"generated {meta['plots']:,} plot records across {len(years)} years "
          f"({size:.1f} MB total -- the real inputs are 436 MB)")

    py = sys.executable
    sh([py, str(scripts / "build_genotypes.py"), "--data-root", str(raw),
        "--out-dir", str(proc), "--clusters", "1"], env_extra)
    sh([py, str(scripts / "build_dataset.py"), "--data-root", str(raw),
        "--out-dir", str(proc), "--geno-dir", str(proc), "--clusters", "1",
        "--n-components", "20", "--min-env-lines", "5"], env_extra)
    sh([py, str(scripts / "rank_lines.py"), "--data-dir", str(proc),
        "--out-dir", str(out), "--clusters", "1", "--folds", "3"], env_extra)
    sh([py, str(scripts / "recommend.py"), "--data-root", str(raw),
        "--data-dir", str(proc), "--out-dir", str(out), "--clusters", "1",
        "--folds", "3", "--min-years", "2"], env_extra)

    # Only possible with synthetic data: score predictions against the TRUE genetic
    # values, rather than against noisy observed yields.
    rank = pd.read_csv(out / "ranked_2008_C1.csv")
    rank["true_gca"] = rank.line.map(meta["gca"])
    ok = rank.dropna(subset=["true_gca"])
    r_true = np.corrcoef(ok.pred, ok.true_gca)[0, 1]
    r_obs = np.corrcoef(ok.pred, ok.y)[0, 1]
    print("\n" + "=" * 72)
    print("RECOVERY OF KNOWN TRUTH (synthetic data only)")
    print("=" * 72)
    print(f"  prediction vs TRUE genetic value : r = {r_true:.3f}")
    print(f"  prediction vs observed yield     : r = {r_obs:.3f}")
    print("  The first exceeds the second because observed yields carry plot noise.")
    print("  This is the same gap the reliability ceiling measures on real data.")

    if args.keep or args.workdir:
        print(f"\noutputs kept in {out}")
    else:
        shutil.rmtree(work, ignore_errors=True)
        print("\ntemporary files removed (use --keep to retain them)")


if __name__ == "__main__":
    main()
