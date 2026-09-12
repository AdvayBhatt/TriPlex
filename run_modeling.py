"""
Population-by-population Ridge CV modeling for genomic selection.
Streams merged_full.csv one population at a time -- never loads the full
dataset in memory. Each population is ~1000 rows, trivially fits in 32GB.

Outputs per line:
  GCA_pred       -- cross-validated yield prediction (bu/acre)
  GCA_obs        -- observed mean yield across environments
  stability      -- 0-1 score; higher = more consistent across environments
  pred_lo/hi     -- ±1 population CV-RMSE prediction interval
  MST_obs        -- historical mean moisture at harvest
  composite_score -- GCA_pred adjusted for moisture (commercial index)
  composite_rank -- rank by composite_score (used for advance list)

Run from the repo root:
    python3 run_modeling.py
"""

import pandas as pd
import numpy as np
import os, gc, time, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH   = 'data/processed/merged_full.csv'
OUTPUT_PATH = 'outputs/line_rankings_full.csv'
CHUNKSIZE   = 3000

# Ridge alpha grid appropriate for genomic selection (n ~ 150 lines, p ~ 300 SNPs).
# Values <100 are effectively unregularized at this scale and cause extrapolation.
ALPHAS = [100, 1_000, 10_000, 100_000]

# Economic weight: bu/acre value per unit moisture reduction.
# Each 1-point drop in harvest moisture saves ~$0.04/bu in drying costs;
# at ~125 bu/acre average, that is ~5 bu/acre-equivalent per moisture unit.
MST_ECON_WEIGHT = 5.0

t0 = time.time()

# ── 1. Column catalogue ───────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    all_cols = f.readline().strip().split(',')

snp_cols = [c for c in all_cols if c.startswith('M') and c != 'MST']
env_cols = [c for c in all_cols if c.startswith('X')]
has_mst  = 'MST' in all_cols
n_snp    = len(snp_cols)

print(f"SNP: {n_snp} | Env: {len(env_cols)} | MST column: {has_mst}")
print(f"Features per population: SNP (population-specific subset) + {len(env_cols)} env\n")

# ── 2. Baseline (LOC+YEAR group mean, read non-SNP cols only) ────────────────
non_snp = [c for c in all_cols if c not in snp_cols]
print("Computing LOC+YEAR baseline (non-SNP columns only)...")
df_meta = pd.read_csv(DATA_PATH, usecols=non_snp, low_memory=False)
df_meta = df_meta.dropna(subset=['YLD_BE'])
group_means   = df_meta.groupby(['LOC', 'YEAR'])['YLD_BE'].transform('mean')
baseline_rmse = np.sqrt(mean_squared_error(df_meta['YLD_BE'], group_means))
baseline_r2   = r2_score(df_meta['YLD_BE'], group_means)
print(f"  Baseline RMSE: {baseline_rmse:.2f} bu/acre  R²: {baseline_r2:.3f}  (in-sample)")
del df_meta, group_means
gc.collect()

# ── 3. Ridge CV per population ────────────────────────────────────────────────
def process_pop(buf, cluster_id, pop_num):
    """
    Run line-level GroupKFold RidgeCV on one population's rows.
    Returns (list_of_line_dicts, pop_rmse, pop_r2).

    Key safeguards:
      - SNP values clipped to [0, 2] (imputed dosage range) before modeling.
      - OOF predictions clipped to ±3 SD of observed yield within this population.
      - Alpha range [100, 1000, 10000, 100000] appropriate for genomic scale.

    New outputs per line:
      - stability: 1 / (1 + CV_yield) where CV = std/mean across environments.
        Higher = more consistent across locations/years (broadly adapted).
      - pred_lo/hi: GCA_pred ± pop_rmse, a 1-sigma prediction interval.
      - MST_obs: historical mean moisture at harvest (lower is commercially preferred).
    """
    df = pd.concat(buf, ignore_index=True).dropna(subset=['YLD_BE'])
    n_lines = df['LINE_UNIQUE_ID'].nunique()
    if n_lines < 5 or len(df) < 15:
        return [], None, None

    # Population-specific SNP columns (non-null only for this population)
    pop_snps  = [c for c in snp_cols if c in df.columns and df[c].notna().any()]
    feat_cols = pop_snps + env_cols
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    y = df['YLD_BE'].values.astype(np.float32)
    lines = df['LINE_UNIQUE_ID'].values

    # ── Safeguard 1: clip SNP dosage to valid imputation range [0, 2] ────────
    n_pop_snps = len(pop_snps)
    X[:, :n_pop_snps] = np.clip(X[:, :n_pop_snps], 0.0, 2.0)

    n_splits = min(5, n_lines)
    pipe = Pipeline([
        ('imp',   SimpleImputer(strategy='mean')),
        ('sc',    StandardScaler()),
        ('ridge', RidgeCV(alphas=ALPHAS)),
    ])

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, lines):
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])

    # ── Safeguard 2: clip OOF to ±3 SD of observed yield (this population) ──
    y_mean, y_std = y.mean(), y.std()
    oof = np.clip(oof, y_mean - 3 * y_std, y_mean + 3 * y_std)

    valid = ~np.isnan(oof)
    if valid.sum() < 2:
        return [], None, None

    pop_rmse = float(np.sqrt(mean_squared_error(y[valid], oof[valid])))
    pop_r2   = float(r2_score(y[valid], oof[valid]))

    # Pre-compute per-line MST means (fast groupby instead of row-by-row lookup)
    if has_mst and 'MST' in df.columns:
        line_mst = df.groupby('LINE_UNIQUE_ID')['MST'].mean().to_dict()
    else:
        line_mst = {}

    out = []
    for line in np.unique(lines):
        m = lines == line
        v = ~np.isnan(oof[m])
        if v.sum() == 0:
            continue

        line_yields = y[m]
        gca_pred    = float(oof[m][v].mean())

        # Stability: inverse of coefficient of variation across environments.
        # Single-environment lines get 0.5 (unknown stability, treated as neutral).
        if len(line_yields) > 1 and float(line_yields.mean()) != 0:
            cv        = float(line_yields.std()) / abs(float(line_yields.mean()))
            stability = round(1.0 / (1.0 + cv), 4)
        else:
            stability = 0.5

        mst_val = line_mst.get(line, np.nan)

        out.append({
            'LINE_UNIQUE_ID': line,
            'CLUSTER_ID':     int(cluster_id),
            'POP_NUM':        int(pop_num),
            'GCA_pred':       round(gca_pred, 4),
            'GCA_obs':        round(float(line_yields.mean()), 4),
            'stability':      stability,
            'n_env':          int(m.sum()),
            'pred_lo':        round(gca_pred - pop_rmse, 2),
            'pred_hi':        round(gca_pred + pop_rmse, 2),
            'MST_obs':        round(float(mst_val), 3) if not np.isnan(mst_val) else np.nan,
        })
    return out, pop_rmse, pop_r2

# ── 4. Stream CSV, buffer by population ──────────────────────────────────────
snp_dtype = {c: 'float32' for c in snp_cols}

all_results = []
rmse_list   = []
r2_list     = []
current_key = None
buf         = []
pop_count   = 0

print(f"Streaming {DATA_PATH} (chunksize={CHUNKSIZE})...")
print("Progress every 25 populations.\n")

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNKSIZE, dtype=snp_dtype,
                         low_memory=False):
    for (cid, pid), grp in chunk.groupby(['CLUSTER_ID', 'POP_NUM'], sort=False):
        key = (cid, pid)
        if current_key is None:
            current_key = key

        if key == current_key:
            buf.append(grp)
        else:
            res, rmse, r2 = process_pop(buf, current_key[0], current_key[1])
            all_results.extend(res)
            if rmse is not None:
                rmse_list.append(rmse)
                r2_list.append(r2)
            pop_count += 1
            if pop_count % 25 == 0 or pop_count == 1:
                elapsed  = time.time() - t0
                avg_rmse = np.mean(rmse_list) if rmse_list else float('nan')
                print(f"  Pop {pop_count:>3} | lines so far: {len(all_results):>5} | "
                      f"mean CV RMSE: {avg_rmse:.2f} | {elapsed:.0f}s elapsed")
            buf = [grp]
            current_key = key

# Last population
if buf:
    res, rmse, r2 = process_pop(buf, current_key[0], current_key[1])
    all_results.extend(res)
    if rmse is not None:
        rmse_list.append(rmse)
        r2_list.append(r2)
    pop_count += 1

# ── 5. Aggregate, build composite index, and rank ─────────────────────────────
print(f"\nAggregating {len(all_results)} line-predictions across {pop_count} populations...")

results_df = pd.DataFrame(all_results)

# Yield rank (reference)
results_df['yield_rank'] = results_df['GCA_pred'].rank(ascending=False).astype(int)

# Composite selection index: reward high yield, penalize high moisture.
# Moisture penalty uses deviation from fleet mean so lines with missing MST
# are not unfairly penalized -- they receive a zero moisture adjustment.
mst_global_mean = results_df['MST_obs'].mean()
mst_deviation   = results_df['MST_obs'].fillna(mst_global_mean) - mst_global_mean
results_df['composite_score'] = results_df['GCA_pred'] - MST_ECON_WEIGHT * mst_deviation
results_df['composite_rank']  = results_df['composite_score'].rank(ascending=False).astype(int)

# Advance list and final sort driven by composite rank
results_df['advance'] = results_df['composite_rank'] <= 20
results_df = results_df.sort_values('composite_rank').reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

elapsed   = time.time() - t0
mean_rmse = np.mean(rmse_list) if rmse_list else float('nan')
mean_r2   = np.mean(r2_list)   if r2_list   else float('nan')

print(f"\n{'='*62}")
print(f"=== Full-data model summary ===")
print(f"  Baseline RMSE   : {baseline_rmse:.2f} bu/acre  (R² {baseline_r2:.3f})")
print(f"  Ridge CV RMSE   : {mean_rmse:.2f} bu/acre  (R² {mean_r2:.3f})")
print(f"  Features        : population SNPs + {len(env_cols)} env covariates")
print(f"  Populations     : {pop_count}")
print(f"  Lines ranked    : {len(results_df)}")
print(f"  Advance list    : {int(results_df['advance'].sum())} lines (by composite score)")
print(f"  MST econ weight : {MST_ECON_WEIGHT} bu/acre per moisture unit")
print(f"  Global MST mean : {mst_global_mean:.2f}")
print(f"  Elapsed         : {elapsed/60:.1f} min")
print(f"\nTop 10 lines (by composite rank):")
cols_show = ['LINE_UNIQUE_ID', 'composite_rank', 'GCA_pred', 'MST_obs',
             'composite_score', 'stability', 'pred_lo', 'pred_hi', 'advance']
print(results_df.head(10)[cols_show].to_string(index=False))
print(f"\nRankings saved -> {OUTPUT_PATH}")
