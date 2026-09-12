"""Does predicting the other traits from DNA help predict yield?

The two-stage idea (DNA -> harvest traits -> yield) is capped: even with the TRUE
measured traits in hand, yield prediction tops out at r = 0.24, below what DNA
alone already achieves. But the sideways version might still pay off. Traits share
genetic architecture, so the genomic signal for moisture or plant height may carry
yield information that the yield model alone misses.

This tests it honestly:

  1. Fit ridge from markers to EACH trait separately (yield included).
  2. Get out-of-fold predictions on training lines, folds split by population, so
     the stacker never sees a prediction made by a model that saw that line.
  3. Fit a stacker: actual yield ~ the predicted traits.
  4. Score on held-out 2008 and compare against single-trait ridge.

If the stacker leans on traits other than yield and beats the baseline, multi-trait
helps. If it just reproduces the yield model, it does not.

No leakage: every input to the stacker is itself predicted from DNA. The measured
harvest traits are used only as TRAINING TARGETS for historical lines, never as
features for a 2008 line.

Usage:
    python scripts/multitrait.py --clusters 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

from build_dataset import prepare_markers

TRAITS = ["YLD_BE", "ERM", "MST", "PHT", "RTLP", "STLP", "TWT", "EHT"]
YEAR, ID = "YEAR_x", "LINE_UNIQUE_ID"


def line_level_traits(data_root: Path, cluster: int) -> pd.DataFrame:
    """Field-adjust every trait the same way the target is built, then average."""
    df = pd.read_csv(
        data_root / f"C{cluster}_Phenotype_Data_V2.csv",
        usecols=[YEAR, "LOC", ID] + TRAITS, low_memory=False,
    )
    df[ID] = df[ID].astype(str).str.split(".").str[:3].str.join(".")
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    for c in TRAITS:
        df[c] = df[c] - df.groupby("ENV")[c].transform("mean")
    return df.groupby(ID)[TRAITS].mean()


def evaluate(pred: np.ndarray, truth: np.ndarray, frac: float = 0.10) -> tuple:
    k = max(int(frac * len(truth)), 1)
    sel = np.argsort(-pred)[:k]
    return (float(np.corrcoef(pred, truth)[0, 1]),
            float(spearmanr(pred, truth).statistic),
            float(truth[sel].mean() - truth.mean()))


def run(cluster: int, args) -> None:
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines = d["lines"]
    tr_mask = ~d["is_2008"]
    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", lines, tr_mask)

    traits = line_level_traits(args.data_root, cluster).reindex(lines)
    # A line missing a trait gets 0, i.e. "average", since traits are field-centred.
    coverage = traits.notna().mean()
    Y = traits.fillna(0.0).to_numpy("float64")

    groups = np.array([ln.split(".")[1] for ln in lines])[tr_mask]
    ytr_all, yte_all = Y[tr_mask], Y[~tr_mask]
    y_idx = TRAITS.index("YLD_BE")
    yte = yte_all[:, y_idx]

    print(f"\n{'=' * 66}\nCLUSTER {cluster}\n{'=' * 66}")
    print(f"{tr_mask.sum():,} train / {(~tr_mask).sum():,} test lines, "
          f"{X.shape[1]:,} markers")
    print("trait coverage: " + ", ".join(f"{t}={100*coverage[t]:.0f}%" for t in TRAITS))

    # Out-of-fold predictions for every trait, folds split by population.
    oof = np.zeros_like(ytr_all)
    gkf = GroupKFold(n_splits=args.folds)
    Xtr = X[tr_mask]
    for tr, te in gkf.split(Xtr, ytr_all[:, 0], groups):
        m = Ridge(alpha=args.alpha).fit(Xtr[tr], ytr_all[tr])
        oof[te] = m.predict(Xtr[te])

    full = Ridge(alpha=args.alpha).fit(Xtr, ytr_all)
    test_pred = full.predict(X[~tr_mask])

    print(f"\n-- how well DNA predicts each trait (out-of-fold, new families) --")
    for i, t in enumerate(TRAITS):
        r = np.corrcoef(oof[:, i], ytr_all[:, i])[0, 1]
        print(f"   {t:<8} r = {r:6.3f}")

    # Baseline: the yield model on its own.
    base = evaluate(test_pred[:, y_idx], yte)

    # Stacker trained on out-of-fold predictions, so it cannot exploit leakage.
    stack = LinearRegression().fit(oof, ytr_all[:, y_idx])
    multi = evaluate(stack.predict(test_pred), yte)

    print(f"\n-- stacker weights (what it leans on) --")
    for t, w in sorted(zip(TRAITS, stack.coef_), key=lambda kv: -abs(kv[1])):
        print(f"   {t:<8} {w:+7.3f}")

    print(f"\n-- HELD-OUT 2008 --")
    print(f"{'model':<28}{'pearson':>9}{'spearman':>10}{'gain bu/ac':>12}")
    print(f"{'ridge, yield only':<28}{base[0]:>9.3f}{base[1]:>10.3f}{base[2]:>12.2f}")
    print(f"{'multi-trait stacked':<28}{multi[0]:>9.3f}{multi[1]:>10.3f}{multi[2]:>12.2f}")
    delta = multi[0] - base[0]
    print(f"\n   difference in pearson: {delta:+.3f} "
          f"({'multi-trait helps' if delta > 0.005 else 'no meaningful gain'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=30000,
                    help="reuse the alpha chosen by rank_lines.py")
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
