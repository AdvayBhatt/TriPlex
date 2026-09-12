"""
run_pipeline.py  --  end-to-end genomic selection pipeline
===========================================================
Reads raw phenotype CSVs + per-population genomic files directly.
No pre-built merged file required.

JUDGE MODE (default -- runs on sample data in <30 seconds):
    python scripts/run_pipeline.py --sample

FULL MODE (runs on HPRC with full data, ~15 min):
    python scripts/run_pipeline.py

Outputs:
    outputs/line_rankings_sample.csv   (sample / judge mode)
    outputs/line_rankings_full.csv     (full mode)
"""

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
ALPHAS          = [100, 1_000, 10_000, 100_000]
MST_ECON_WEIGHT = 5.0    # bu/acre per moisture-unit penalty
YEAR_COL        = "YEAR_x"
TARGET          = "YLD_BE"
ID_COL          = "LINE_UNIQUE_ID"


# ── ID helpers ─────────────────────────────────────────────────────────────────
_ID_RE = re.compile(r"^C(\d+)\.(\d+)\.(\d+)")

def parse_line_id(lid: str):
    """Return (cluster, pop, line_int) from 'C1.1.191'."""
    m = _ID_RE.match(str(lid))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def geno_row_id(line_int: int) -> str:
    """Convert integer line number to zero-padded 11-digit genomic row ID."""
    return f"{line_int:011d}"


# ── Genomic file loading ────────────────────────────────────────────────────────
def load_geno(geno_path: Path, line_ints: list[int]) -> pd.DataFrame:
    """
    Load a population genomic CSV and return only the progeny rows
    whose integer IDs are in line_ints.

    The first column (unnamed) holds row identifiers:
      - Parent rows:  "PID200761", etc.  -> skip
      - Progeny rows: "00000000191", etc. -> keep, convert to int

    Returns DataFrame indexed by line_int (integer), columns = SNP names.
    """
    raw = pd.read_csv(geno_path, index_col=0, low_memory=False)
    # Keep only progeny rows (all-digit index values)
    progeny_mask = raw.index.astype(str).str.match(r"^\d{11}$")
    raw = raw[progeny_mask].copy()
    raw.index = raw.index.astype(str).str.lstrip("0").astype(int)
    raw.index.name = "line_int"
    # Deduplicate: if a line_int appears more than once in the file, keep the first
    if raw.index.duplicated().any():
        raw = raw[~raw.index.duplicated(keep="first")]

    # Keep only unique requested lines (pheno can repeat a line_int across environments;
    # geno_df must stay unique so that later .loc[line_ints] fans them out correctly)
    raw_set = set(raw.index)
    wanted = list(dict.fromkeys(li for li in line_ints if li in raw_set))
    if not wanted:
        return pd.DataFrame()
    sub = raw.loc[wanted]
    # Replace NA strings with np.nan, convert to float
    sub = sub.replace("NA", np.nan).astype(float)
    return sub


# ── Single-population model ─────────────────────────────────────────────────────
def process_population(
    pheno_pop: pd.DataFrame,
    geno_df: pd.DataFrame,
    env_df: pd.DataFrame | None,
) -> list[dict]:
    """
    Fit ridge CV for one population and return a list of per-line result dicts.

    pheno_pop : phenotype rows for this population (may span multiple years/locs)
    geno_df   : SNP matrix indexed by line_int
    env_df    : optional environmental features (joined on YEAR_x + LOC)
    """
    # Aggregate phenotype to line-level mean and std for stability
    pheno_pop = pheno_pop.dropna(subset=[TARGET])
    if pheno_pop[ID_COL].nunique() < 5 or len(pheno_pop) < 15:
        return []

    parsed = pheno_pop[ID_COL].map(parse_line_id)
    pheno_pop = pheno_pop.copy()
    pheno_pop["line_int"] = parsed.map(lambda x: x[2] if x else np.nan)
    pheno_pop = pheno_pop.dropna(subset=["line_int"])
    pheno_pop["line_int"] = pheno_pop["line_int"].astype(int)

    # Join genomic data
    have_geno = set(geno_df.index) if geno_df is not None and len(geno_df) > 0 else set()
    pheno_pop = pheno_pop[pheno_pop["line_int"].isin(have_geno)]
    if len(pheno_pop) < 15:
        return []

    # Optional: join environmental features on YEAR_x + LOC
    if env_df is not None and YEAR_COL in pheno_pop.columns and "LOC" in pheno_pop.columns:
        env_cols_feat = [c for c in env_df.columns if c.startswith("X")]
        if env_cols_feat:
            pheno_pop = pheno_pop.merge(
                env_df[["YEAR", "LOC"] + env_cols_feat].rename(columns={"YEAR": YEAR_COL}),
                on=[YEAR_COL, "LOC"],
                how="left",
            )

    # Build feature matrix: SNP columns for each line in this population
    snp_cols = list(geno_df.columns)
    env_feat_cols = [c for c in pheno_pop.columns if c.startswith("X")]

    line_ints = pheno_pop["line_int"].values
    y = pheno_pop[TARGET].values.astype(np.float32)
    groups = pheno_pop[ID_COL].values  # group = unique line (for GroupKFold)

    # Build X: one row per phenotype observation, SNPs from genomic file.
    # geno_df has a unique index; .loc[line_ints] fans repeated line_ints out
    # correctly (one SNP row per pheno row), keeping X and y aligned.
    snp_data = geno_df.loc[line_ints, snp_cols].values.astype(np.float32)
    snp_data = np.clip(snp_data, -1.0, 2.0)  # clip to valid dosage range

    feat_parts = [snp_data]
    if env_feat_cols:
        env_data = pheno_pop[env_feat_cols].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(np.float32)
        feat_parts.append(env_data)

    X = np.hstack(feat_parts)

    n_lines = len(np.unique(groups))
    n_splits = min(5, n_lines)
    if n_splits < 2:
        return []

    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="mean")),
        ("sc",    StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])

    # Clip OOF to ±3 SD of observed yield in this population
    y_mean, y_std = float(y.mean()), float(y.std())
    oof = np.clip(oof, y_mean - 3 * y_std, y_mean + 3 * y_std)

    valid = ~np.isnan(oof)
    if valid.sum() < 2:
        return []

    pop_rmse = float(np.sqrt(mean_squared_error(y[valid], oof[valid])))

    # MST per line if available
    has_mst = "MST" in pheno_pop.columns
    line_mst = (
        pheno_pop.groupby(ID_COL)["MST"].mean().to_dict() if has_mst else {}
    )

    results = []
    for lid in np.unique(groups):
        mask = groups == lid
        v = ~np.isnan(oof[mask])
        if v.sum() == 0:
            continue
        line_yields = y[mask]
        gca_pred = float(oof[mask][v].mean())
        gca_obs  = float(line_yields.mean())

        # Stability: inverse CV across environments
        if len(line_yields) > 1 and abs(gca_obs) > 1e-6:
            cv_val   = float(line_yields.std()) / abs(gca_obs)
            stability = round(1.0 / (1.0 + cv_val), 4)
        else:
            stability = 0.5

        mst_val = line_mst.get(lid, np.nan)
        parsed_id = parse_line_id(lid)

        results.append({
            "LINE_UNIQUE_ID":  lid,
            "CLUSTER_ID":      parsed_id[0] if parsed_id else np.nan,
            "POP_NUM":         parsed_id[1] if parsed_id else np.nan,
            "GCA_pred":        round(gca_pred, 4),
            "GCA_obs":         round(gca_obs,  4),
            "stability":       stability,
            "n_env":           int(mask.sum()),
            "pred_lo":         round(gca_pred - pop_rmse, 2),
            "pred_hi":         round(gca_pred + pop_rmse, 2),
            "MST_obs":         round(float(mst_val), 3) if not np.isnan(mst_val) else np.nan,
        })
    return results


# ── Sample (judge) mode ─────────────────────────────────────────────────────────
def run_sample(data_root: Path, out_dir: Path) -> None:
    sample_dir = data_root / "raw" / "sample_data"
    pheno_path = sample_dir / "sample_C1_phenotype_100rows.csv"
    geno_path  = sample_dir / "sample_C1.1_100rows.csv"
    env_path   = sample_dir / "sample_environmental_20rows.csv"

    for p in [pheno_path, geno_path]:
        if not p.exists():
            sys.exit(f"Sample file not found: {p}\nExpected under {sample_dir}")

    print(f"[sample] Loading phenotype: {pheno_path.name}")
    pheno = pd.read_csv(pheno_path, low_memory=False)
    pheno = pheno.rename(columns={"YEAR_x": YEAR_COL}) if YEAR_COL not in pheno.columns else pheno

    env_df = None
    if env_path.exists():
        try:
            env_df = pd.read_csv(env_path, low_memory=False)
        except Exception:
            pass

    # Determine which populations are in the sample phenotype file
    pheno["_parsed"] = pheno[ID_COL].map(parse_line_id)
    pheno = pheno.dropna(subset=["_parsed"])
    pheno["_pop"] = pheno["_parsed"].map(lambda x: x[1])

    all_results = []
    for pop_num, pop_pheno in pheno.groupby("_pop"):
        # Infer which genotype file covers this population
        # Sample file is C1.1 -- only pop 1 will have genomic data
        line_ints = pop_pheno["_parsed"].map(lambda x: x[2]).dropna().astype(int).tolist()
        print(f"[sample] Population C1.{pop_num}: {len(pop_pheno)} rows, "
              f"{len(line_ints)} unique lines -- loading {geno_path.name}")
        try:
            geno_df = load_geno(geno_path, line_ints)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
        if geno_df.empty:
            print(f"  No genomic data matched -- skipping")
            continue
        res = process_population(pop_pheno, geno_df, env_df)
        all_results.extend(res)

    _finalize(all_results, out_dir / "line_rankings_sample.csv", mode="sample")


# ── Full mode ───────────────────────────────────────────────────────────────────
def run_full(data_root: Path, out_dir: Path, clusters: list[int]) -> None:
    raw_dir = data_root / "raw"
    env_path = raw_dir / "environmental_features.csv"
    env_df = None
    if env_path.exists():
        try:
            env_df = pd.read_csv(env_path, low_memory=False)
            print(f"Environmental features loaded: {env_df.shape}")
        except Exception:
            pass

    all_results = []
    t0 = time.time()

    for cluster in clusters:
        pheno_path = raw_dir / f"C{cluster}_Phenotype_Data_V2.csv"
        if not pheno_path.exists():
            print(f"[C{cluster}] Phenotype file not found at {pheno_path}, skipping")
            continue

        geno_root = raw_dir / "genotypes" / f"C{cluster}" / f"ImputedPopulationsC{cluster}"

        print(f"\n[C{cluster}] Loading phenotype ({pheno_path.stat().st_size // 1_000_000} MB)...")
        pheno = pd.read_csv(pheno_path, low_memory=False)
        pheno = pheno.rename(columns={"YEAR_x": YEAR_COL}) if YEAR_COL not in pheno.columns else pheno
        pheno["_parsed"] = pheno[ID_COL].map(parse_line_id)
        pheno = pheno.dropna(subset=["_parsed"])
        pheno["_pop"] = pheno["_parsed"].map(lambda x: x[1])

        pops = sorted(pheno["_pop"].unique())
        print(f"[C{cluster}] {len(pops)} populations, {len(pheno)} rows total")

        pop_count = 0
        for pop_num in pops:
            pop_pheno = pheno[pheno["_pop"] == pop_num]
            line_ints = pop_pheno["_parsed"].map(lambda x: x[2]).dropna().astype(int).tolist()

            geno_path = geno_root / f"C{cluster}.{pop_num}_Imputed.csv"
            if not geno_path.exists():
                continue

            try:
                geno_df = load_geno(geno_path, line_ints)
            except Exception as e:
                print(f"  [C{cluster}.{pop_num}] geno error: {e}")
                continue

            if geno_df.empty:
                continue

            res = process_population(pop_pheno, geno_df, env_df)
            all_results.extend(res)
            pop_count += 1

            if pop_count % 25 == 0:
                elapsed = time.time() - t0
                print(f"  Pop {pop_count:>3} | lines so far: {len(all_results):>6} | {elapsed:.0f}s")

        print(f"[C{cluster}] Done: {pop_count} populations processed")

    _finalize(all_results, out_dir / "line_rankings_full.csv", mode="full")


# ── Shared finalization: composite index, ranking, output ──────────────────────
def _finalize(results: list[dict], out_path: Path, mode: str) -> None:
    if not results:
        print(f"No results to write -- check input data and genomic file paths.")
        return

    df = pd.DataFrame(results)

    # Composite selection index
    mst_global_mean = df["MST_obs"].mean()
    mst_dev = df["MST_obs"].fillna(mst_global_mean) - mst_global_mean
    df["composite_score"] = df["GCA_pred"] - MST_ECON_WEIGHT * mst_dev
    df["yield_rank"]      = df["GCA_pred"].rank(ascending=False).astype(int)
    df["composite_rank"]  = df["composite_score"].rank(ascending=False).astype(int)

    budget_n = max(1, int(round(len(df) * 0.10)))
    df["advance"] = df["composite_rank"] <= budget_n
    df = df.sort_values("composite_rank").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"=== {mode.upper()} MODE RESULTS ===")
    print(f"  Lines ranked    : {len(df)}")
    print(f"  Advance list    : {df['advance'].sum()} lines (10% budget, composite rank)")
    print(f"  MST econ weight : {MST_ECON_WEIGHT} bu/acre per moisture unit")
    if len(df) > 0:
        print(f"\nTop 10 lines (composite rank):")
        show_cols = ["LINE_UNIQUE_ID", "composite_rank", "GCA_pred",
                     "MST_obs", "composite_score", "stability",
                     "pred_lo", "pred_hi", "advance"]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df.head(10)[show_cols].to_string(index=False))
    print(f"\n  Output -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genomic selection pipeline: phenotype + genomics -> ranked advance list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run in judge/demo mode on sample data (fast, no full dataset required)",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root of the data directory (contains raw/ subdirectory)",
    )
    parser.add_argument(
        "--out-dir", default="outputs",
        help="Directory to write ranked CSV output",
    )
    parser.add_argument(
        "--clusters", nargs="+", type=int, default=[1, 2],
        help="Clusters to process in full mode (e.g. 1 2)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)

    t0 = time.time()
    if args.sample:
        print("Running in SAMPLE / JUDGE MODE")
        print(f"Data root: {data_root.resolve()}\n")
        run_sample(data_root, out_dir)
    else:
        print("Running in FULL MODE")
        print(f"Data root: {data_root.resolve()}\n")
        run_full(data_root, out_dir, args.clusters)

    print(f"\nTotal time: {(time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
