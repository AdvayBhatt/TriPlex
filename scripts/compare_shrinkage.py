"""
compare_shrinkage.py -- diagnostic: does BLUP-style shrinkage on YLD_ADJ
help, hurt, or just change the yardstick, versus data-audit's unshrunk
simple-mean target?

Reuses the already-built marker matrices (no need to rebuild features --
X doesn't depend on which target we score against), and re-pulls both
YLD_ADJ (shrunk) and YLD_ADJ_RAW (unshrunk) from the target parquet, aligned
to the same line order already baked into features_C{cluster}_cutoff{year}.npz.

Usage:
    py scripts/compare_shrinkage.py --cutoff-year 2007
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ALPHAS = [1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def tune_alpha(X, y, groups, folds):
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(folds, n_groups))
    best_alpha, best_score = ALPHAS[0], -np.inf
    for a in ALPHAS:
        scores = []
        for tr, te in gkf.split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            pred = m.predict(X[te])
            if np.std(pred) > 0:
                scores.append(float(np.corrcoef(pred, y[te])[0, 1]))
        if scores and np.mean(scores) > best_score:
            best_alpha, best_score = a, float(np.mean(scores))
    return best_alpha, best_score


def run(cluster: int, cutoff_year: int, data_dir: Path, folds: int):
    feat = np.load(data_dir / f"features_C{cluster}_cutoff{cutoff_year}.npz", allow_pickle=True)
    target = pd.read_parquet(data_dir / f"target_C{cluster}_cutoff{cutoff_year}.parquet").set_index("LINE_UNIQUE_ID")

    X_train, X_test = feat["X_train"], feat["X_test"]
    groups = feat["population_train"]

    print(f"\n{'=' * 60}\nCLUSTER {cluster}, cutoff {cutoff_year}\n{'=' * 60}")
    for target_col, label in [("YLD_ADJ", "shrunk (BLUP)"), ("YLD_ADJ_RAW", "unshrunk (simple mean)")]:
        y_train = target.loc[feat["line_ids_train"], target_col].to_numpy(dtype=np.float64)
        y_test = target.loc[feat["line_ids_test"], target_col].to_numpy(dtype=np.float64)

        alpha, cv_score = tune_alpha(X_train, y_train, groups, folds)
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        pearson = float(np.corrcoef(pred, y_test)[0, 1])

        n_sel = max(int(round(0.10 * len(y_test))), 1)
        chosen = np.argsort(-pred)[:n_sel]
        gain = float(y_test[chosen].mean() - y_test.mean())

        print(f"  target={label:<24} alpha={alpha:<8} CV_pearson={cv_score:.3f}  "
              f"HELDOUT_pearson={pearson:.3f}  gain_bu={gain:.2f}  "
              f"target_sd(train)={y_train.std():.2f}  target_sd(test)={y_test.std():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cutoff-year", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args.cutoff_year, args.data_dir, args.folds)


if __name__ == "__main__":
    main()
