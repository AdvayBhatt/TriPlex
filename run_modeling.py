"""
Population-by-population Ridge CV modeling for genomic selection.
Streams merged_full.csv one population at a time -- never loads the full
dataset in memory. Each population is ~1000 rows, trivially fits in 32GB.

Run from the repo root:
    python3 run_modeling.py
"""

import pandas as pd
import numpy as np
import os, gc, time, warnings
warnings.filterwarnings('ignore')   # suppress sklearn and pandas dtype chatter

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

t0 = time.time()

# ── 1. Column catalogue ───────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    all_cols = f.readline().strip().split(',')

snp_cols = [c for c in all_cols if c.startswith('M') and c != 'MST']
env_cols = [c for c in all_cols if c.startswith('X')]
n_snp    = len(snp_cols)

# Features: SNP markers + environmental covariates only.
# Agronomic traits (PHT, EHT, etc.) are in-season measurements unavailable
# before planting and are excluded to prevent leakage.
print(f"SNP: {n_snp} | Env: {len(env_cols)}")
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
        Out-of-range values from imputation QC failures would otherwise cause
        the linear model to extrapolate wildly for individual lines.
      - OOF predictions clipped to ±3 SD of observed yield within this
        population, preventing single corrupted-marker lines from dominating
        the cross-population ranking.
      - Alpha range [100, 1000, 10000, 100000] appropriate for genomic
        selection where p_SNP ~ n_lines.
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

    out = []
    for line in np.unique(lines):
        m = lines == line
        v = ~np.isnan(oof[m])
        if v.sum() == 0:
            continue
        out.append({
            'LINE_UNIQUE_ID': line,
            'CLUSTER_ID':     int(cluster_id),
            'POP_NUM':        int(pop_num),
            'GCA_pred':       float(oof[m][v].mean()),
            'GCA_obs':        float(y[m].mean()),
            'n_env':          int(m.sum()),
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

# ── 5. Aggregate and rank ─────────────────────────────────────────────────────
print(f"\nAggregating {len(all_results)} line-predictions across {pop_count} populations...")

results_df = pd.DataFrame(all_results)
results_df['rank']    = results_df['GCA_pred'].rank(ascending=False).astype(int)
results_df            = results_df.sort_values('rank').reset_index(drop=True)
results_df['advance'] = results_df['rank'] <= 20

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

elapsed   = time.time() - t0
mean_rmse = np.mean(rmse_list) if rmse_list else float('nan')
mean_r2   = np.mean(r2_list)   if r2_list   else float('nan')

print(f"\n{'='*58}")
print(f"=== Full-data model summary ===")
print(f"  Baseline RMSE : {baseline_rmse:.2f} bu/acre  (R² {baseline_r2:.3f})")
print(f"  Ridge CV RMSE : {mean_rmse:.2f} bu/acre  (R² {mean_r2:.3f})")
print(f"  Features      : population SNPs + {len(env_cols)} env covariates")
print(f"  Populations   : {pop_count}")
print(f"  Lines ranked  : {len(results_df)}")
print(f"  Advance list  : {int(results_df['advance'].sum())} lines")
print(f"  Elapsed       : {elapsed/60:.1f} min")
print(f"\nTop 10 lines:")
cols_show = ['LINE_UNIQUE_ID', 'rank', 'GCA_pred', 'GCA_obs', 'advance']
print(results_df.head(10)[cols_show].to_string(index=False))
print(f"\nRankings saved -> {OUTPUT_PATH}")
