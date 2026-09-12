"""
rank_lines.py -- Stage 3: fit the genomic prediction model and rank lines.

Trains on the pre-cutoff fold (all lines with YEAR <= cutoff) and scores
against the real held-out year (cutoff + 1), which is already sitting in the
data -- but was never touched during feature/target construction or here.

Models compared, cheapest to strongest:
    mean      predict the training average for every test line (do-nothing floor)
    parent    average of training lines sharing a parent -- pedigree, no DNA
    ridge     ridge regression on all QC'd markers -- the GBLUP-equivalent
              genomic prediction model, and the one we're actually shipping

Cross-validation for alpha tuning is grouped BY POPULATION, not by line and
never random k-fold: lines within a population are siblings from one cross,
so a random split would let the model score well by recognising families,
which cannot transfer to the held-out year (every population there is new).

Scoring reports both correlation (pearson/spearman -- is the ranking right)
and "gain": the realised mean of the top N% the model would advance, in
bu/acre above the cohort average -- the number a breeding manager acts on.

Usage:
    py scripts/rank_lines.py --cutoff-year 2007
    py scripts/rank_lines.py --cutoff-year 2006
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ALPHAS = [1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def parent_prediction(train_p1, train_p2, train_y, target_p1, target_p2) -> np.ndarray:
    """Pure pedigree baseline: average adjusted yield of training lines sharing
    either parent. Falls back to the training mean for an unseen parent."""
    stacked = pd.DataFrame({
        "p": np.concatenate([train_p1, train_p2]),
        "y": np.concatenate([train_y, train_y]),
    })
    means = stacked.groupby("p")["y"].mean()
    p1 = pd.Series(target_p1).map(means)
    p2 = pd.Series(target_p2).map(means)
    return pd.concat([p1, p2], axis=1).mean(axis=1).fillna(train_y.mean()).to_numpy()


def tune_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> tuple[float, float]:
    """Pick ridge strength by leave-population-out CV, so tuning itself never
    leaks across families."""
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


def evaluate(name: str, pred: np.ndarray, truth: np.ndarray, frac: float) -> dict:
    n_sel = max(int(round(frac * len(truth))), 1)
    chosen = np.argsort(-pred)[:n_sel]
    ideal = np.sort(truth)[::-1][:n_sel].mean()
    gain = float(truth[chosen].mean() - truth.mean())
    pearson = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 0 else 0.0
    spearman = float(spearmanr(pred, truth).statistic) if np.std(pred) > 0 else 0.0
    pct_ideal = 100 * gain / (ideal - truth.mean()) if ideal > truth.mean() else 0.0
    return {"model": name, "pearson": pearson, "spearman": spearman,
            "gain_bu": gain, "pct_of_ideal": pct_ideal}


def run(cluster: int, cutoff_year: int, args) -> pd.DataFrame:
    d = np.load(args.data_dir / f"features_C{cluster}_cutoff{cutoff_year}.npz",
                allow_pickle=True)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    groups = d["population_train"]

    print(f"\n{'=' * 68}\nCLUSTER {cluster}, cutoff {cutoff_year} -> test {cutoff_year + 1}\n{'=' * 68}")
    print(f"train {len(y_train):,} lines / {len(np.unique(groups))} populations  ->  "
          f"test {len(y_test):,} lines (all new populations)")
    print(f"features: {X_train.shape[1]:,} markers")

    alpha, cv_score = tune_alpha(X_train, y_train, groups, args.folds)
    print(f"ridge alpha selected by leave-population-out CV: {alpha}  (CV pearson {cv_score:.3f})")

    # Grouped-CV predictions for every training line (for the CV-vs-holdout comparison)
    gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    rows = []
    for tr, te in gkf.split(X_train, y_train, groups):
        fold = {
            "mean": np.full(len(te), y_train[tr].mean()),
            "parent": parent_prediction(d["parent1_train"][tr], d["parent2_train"][tr],
                                         y_train[tr], d["parent1_train"][te], d["parent2_train"][te]),
            "ridge": Ridge(alpha=alpha).fit(X_train[tr], y_train[tr]).predict(X_train[te]),
        }
        for name, p in fold.items():
            rows.append({**evaluate(name, p, y_train[te], args.select), "split": "CV"})
    cv = pd.DataFrame(rows).groupby("model", as_index=False).mean(numeric_only=True)
    cv["split"] = f"CV (leave-population-out, train<={cutoff_year})"

    # Final fit on ALL training lines, scored against the real held-out year.
    final_ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    preds = {
        "mean": np.full(len(y_test), y_train.mean()),
        "parent": parent_prediction(d["parent1_train"], d["parent2_train"], y_train,
                                     d["parent1_test"], d["parent2_test"]),
        "ridge": final_ridge.predict(X_test),
    }
    ho = pd.DataFrame([evaluate(n, p, y_test, args.select) for n, p in preds.items()])
    ho["split"] = f"HELD-OUT {cutoff_year + 1}"

    out = pd.concat([cv, ho], ignore_index=True)
    order = ["mean", "parent", "ridge"]
    out["model"] = pd.Categorical(out["model"], order, ordered=True)
    out = out.sort_values(["split", "model"])

    for split in out["split"].unique():
        s = out[out["split"] == split]
        print(f"\n-- {split} --")
        print(f"{'model':<10}{'pearson':>9}{'spearman':>10}{'gain bu/ac':>12}{'% of ideal':>12}")
        for _, r in s.iterrows():
            print(f"{r['model']:<10}{r['pearson']:>9.3f}{r['spearman']:>10.3f}"
                  f"{r['gain_bu']:>12.2f}{r['pct_of_ideal']:>11.1f}%")

    ranked = pd.DataFrame({
        "line": d["line_ids_test"], "pred": preds["ridge"], "y": y_test,
        "n_fields": d["n_fields_test"], "population": d["population_test"],
    }).sort_values("pred", ascending=False)
    out_path = args.out_dir / f"ranked_{cutoff_year + 1}_C{cluster}.csv"
    ranked.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"top {args.top_n} lines by predicted GCA:")
    print(ranked.head(args.top_n).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    out.insert(0, "cluster", cluster)
    out.insert(1, "cutoff_year", cutoff_year)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cutoff-year", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select", type=float, default=0.10,
                     help="fraction of the cohort advanced, for the gain metric")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.concat([run(c, args.cutoff_year, args) for c in args.clusters], ignore_index=True)
    out_path = args.out_dir / f"model_comparison_cutoff{args.cutoff_year}.csv"
    res.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
