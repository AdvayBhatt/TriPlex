"""Phase 1 model: rank 2008 lines by predicted genetic merit, using legal features only.

Trains on lines from 2000-2007 and predicts the 2008 cohort, whose true yields are
held out until scoring. Nothing measured at harvest is used as an input.

WHY THE CROSS-VALIDATION LOOKS UNUSUAL
--------------------------------------
Random k-fold would be dishonest here. Lines inside one population are siblings
from a single cross, so a random split puts near-identical relatives in both train
and test. The model then looks excellent by memorising families, and collapses on
2008 -- where every population is brand new.

So folds are split BY POPULATION: all siblings stay together on one side. That
mirrors the real task and is the "cross-validation appropriate for breeding program
structure" the rubric asks for. Both are reported, because the gap between them is
itself a result worth showing.

MODELS
    mean          predict the training average for everyone (the do-nothing floor)
    parent        average of training lines sharing a parent -- pedigree, no DNA
    ridge_pcs     ridge on 50 genomic PCs
    ridge_markers ridge on all ~2,700 markers -- this is rrBLUP/GBLUP in ridge form,
                  and the main genomic prediction model
    ridge+parent  markers plus the pedigree prediction as an extra feature

PCs lose to full markers because PCA keeps family structure and discards the
within-family variation that separates siblings -- which is exactly what has to be
ranked once every 2008 population is new.

SCORING
    Correlation tells you if the ranking is right. But the decision is "advance the
    top N%", so we also report what that selection actually yields: the realised
    mean of the lines the model picks, in bu/acre above the cohort average. That is
    the number a breeding manager cares about.

Usage:
    python scripts/rank_lines.py
    python scripts/rank_lines.py --clusters 1 --select 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from build_dataset import prepare_markers

ALPHAS = [1, 10, 100, 300, 1000, 3000, 10000, 30000]


def population_of(lines: np.ndarray) -> np.ndarray:
    return np.array([ln.split(".")[1] for ln in lines])


def parent_prediction(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Average adjusted yield of training lines sharing either parent.

    Pure pedigree: no markers. This is the baseline genomic prediction has to beat,
    and it is a genuinely strong one -- 28 of the 75 parents behind 2008 populations
    were already used before 2008.
    """
    stacked = pd.concat([
        train[["parent1", "y"]].rename(columns={"parent1": "p"}),
        train[["parent2", "y"]].rename(columns={"parent2": "p"}),
    ])
    means = stacked.groupby("p").y.mean()
    p1 = target.parent1.map(means)
    p2 = target.parent2.map(means)
    # Fall back to the overall training mean when a parent was never seen before.
    return pd.concat([p1, p2], axis=1).mean(axis=1).fillna(train.y.mean()).to_numpy()


def tune_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> float:
    """Pick ridge strength by grouped CV, so tuning does not leak across families."""
    gkf = GroupKFold(n_splits=folds)
    best, best_score = ALPHAS[0], -np.inf
    for a in ALPHAS:
        scores = []
        for tr, te in gkf.split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            pred = m.predict(X[te])
            if np.std(pred) > 0:
                scores.append(np.corrcoef(pred, y[te])[0, 1])
        if scores and np.mean(scores) > best_score:
            best, best_score = a, float(np.mean(scores))
    return best


def evaluate(name: str, pred: np.ndarray, truth: np.ndarray, frac: float) -> dict:
    n_sel = max(int(round(frac * len(truth))), 1)
    chosen = np.argsort(-pred)[:n_sel]
    ideal = np.sort(truth)[::-1][:n_sel].mean()
    # How much of the achievable gain the model actually captured.
    gain = truth[chosen].mean() - truth.mean()
    return {
        "model": name,
        "pearson": float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 0 else 0.0,
        "spearman": float(spearmanr(pred, truth).statistic) if np.std(pred) > 0 else 0.0,
        "gain_bu": gain,
        "pct_of_ideal": 100 * gain / (ideal - truth.mean()) if ideal > truth.mean() else 0.0,
    }


def run(cluster: int, args) -> pd.DataFrame:
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    df = pd.DataFrame({
        "line": d["lines"], "y": d["y"].astype("float64"), "n_obs": d["n_obs"],
        "is_2008": d["is_2008"], "parent1": d["parent1"], "parent2": d["parent2"],
    })
    df["pop"] = population_of(d["lines"])
    tr_mask = ~df.is_2008.to_numpy()

    pcs = d["pcs"].astype("float64")
    markers = prepare_markers(args.data_dir / f"geno_C{cluster}.npz",
                              d["lines"], tr_mask)

    train, test = df[tr_mask].reset_index(drop=True), df[~tr_mask].reset_index(drop=True)
    ytr, yte = train.y.to_numpy(), test.y.to_numpy()
    groups = train["pop"].to_numpy()

    print(f"\n{'=' * 68}\nCLUSTER {cluster}\n{'=' * 68}")
    print(f"train {len(train):,} lines / {train['pop'].nunique()} populations  ->  "
          f"test {len(test):,} lines / {test['pop'].nunique()} populations (all new)")
    print(f"features: {markers.shape[1]:,} markers, {pcs.shape[1]} PCs")

    blocks = {"ridge_pcs": pcs, "ridge_markers": markers}
    alphas = {k: tune_alpha(v[tr_mask], ytr, groups, args.folds)
              for k, v in blocks.items()}
    print("alpha by leave-population-out CV: " +
          ", ".join(f"{k}={v}" for k, v in alphas.items()))

    gkf = GroupKFold(n_splits=args.folds)
    cv_rows = []
    for tr, te in gkf.split(ytr, ytr, groups):
        sub_tr, sub_te = train.iloc[tr], train.iloc[te]
        par = parent_prediction(sub_tr, sub_te)
        par_self = parent_prediction(sub_tr, sub_tr)
        fold = {"mean": np.full(len(te), ytr[tr].mean()), "parent": par}
        for name, X in blocks.items():
            Xb = X[tr_mask]
            fold[name] = Ridge(alpha=alphas[name]).fit(Xb[tr], ytr[tr]).predict(Xb[te])
        Mb = markers[tr_mask]
        fold["ridge+parent"] = Ridge(alpha=alphas["ridge_markers"]).fit(
            np.column_stack([Mb[tr], par_self]), ytr[tr]
        ).predict(np.column_stack([Mb[te], par]))
        for nm, p in fold.items():
            cv_rows.append({**evaluate(nm, p, ytr[te], args.select), "split": "CV"})

    cv = pd.DataFrame(cv_rows).groupby("model", as_index=False).mean(numeric_only=True)
    cv["split"] = "CV (leave-population-out)"

    # Final fit on all training lines, scored against the held-out 2008 truth.
    par_test = parent_prediction(train, test)
    par_self = parent_prediction(train, train)
    preds = {"mean": np.full(len(test), ytr.mean()), "parent": par_test}
    for name, X in blocks.items():
        preds[name] = Ridge(alpha=alphas[name]).fit(X[tr_mask], ytr).predict(X[~tr_mask])
    preds["ridge+parent"] = Ridge(alpha=alphas["ridge_markers"]).fit(
        np.column_stack([markers[tr_mask], par_self]), ytr
    ).predict(np.column_stack([markers[~tr_mask], par_test]))

    ho = pd.DataFrame([evaluate(n, p, yte, args.select) for n, p in preds.items()])
    ho["split"] = "HELD-OUT 2008"

    out = pd.concat([cv, ho], ignore_index=True)
    order = ["mean", "parent", "ridge_pcs", "ridge_markers", "ridge+parent"]
    out["model"] = pd.Categorical(out.model, order, ordered=True)
    out = out.sort_values(["split", "model"])

    for split in out.split.unique():
        s = out[out.split == split]
        print(f"\n-- {split} --")
        print(f"{'model':<14}{'pearson':>9}{'spearman':>10}"
              f"{'gain bu/ac':>12}{'% of ideal':>12}")
        for _, r in s.iterrows():
            print(f"{r.model:<14}{r.pearson:>9.3f}{r.spearman:>10.3f}"
                  f"{r.gain_bu:>12.2f}{r.pct_of_ideal:>11.1f}%")

    best = preds["ridge_markers"]
    ranked = test.assign(pred=best).sort_values("pred", ascending=False)
    top = ranked.head(args.top_n)[["line", "pred", "y", "n_obs"]]
    print(f"\n-- top {args.top_n} 2008 lines by ridge_markers (actual y shown for audit) --")
    print(top.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    out.insert(0, "cluster", cluster)
    ranked.assign(cluster=cluster)[["cluster", "line", "pred", "y", "n_obs"]].to_csv(
        args.out_dir / f"ranked_2008_C{cluster}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select", type=float, default=0.10,
                    help="fraction of the cohort advanced, for the gain metric")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.concat([run(c, args) for c in args.clusters], ignore_index=True)
    res.to_csv(args.out_dir / "model_comparison.csv", index=False)
    print(f"\nwrote {args.out_dir}/model_comparison.csv and ranked_2008_C*.csv")


if __name__ == "__main__":
    main()
