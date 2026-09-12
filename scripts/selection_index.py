"""Selection index: rank on economic value, not yield alone.

Advancement is not a yield contest. Grain delivered wet must be mechanically dried,
and that cost is real. The public formulation used in maize variety filings is:

    income per acre = price * yield  -  0.02 * (harvest moisture - 15.5) * yield

so one point of harvest moisture costs 0.02 * yield / price bushels per acre --
about 0.75 bu/ac at 150 bu and $4.00. That gives a defensible, citable exchange rate
between the two traits without having to invent economic weights.

Both components are predicted from DNA, so the index is legal at decision time. And
moisture is the easier of the two to predict -- markers reach r = 0.271 for moisture
against 0.187 for yield -- so the moisture half of the index is more trustworthy than
the yield half.

Usage:
    python scripts/selection_index.py --clusters 1 --price 4.00
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from build_dataset import prepare_markers

YEAR, ID = "YEAR_x", "LINE_UNIQUE_ID"

def line_means(data_root: Path, cluster: int, traits) -> pd.DataFrame:
    df = pd.read_csv(data_root / f"C{cluster}_Phenotype_Data_V2.csv",
                     usecols=[YEAR, "LOC", ID] + traits, low_memory=False)
    df[ID] = df[ID].astype(str).str.split(".").str[:3].str.join(".")
    df["ENV"] = df[YEAR].astype(str) + "_" + df["LOC"]
    for c in traits:
        df[c] = df[c] - df.groupby("ENV")[c].transform("mean")
    return df.groupby(ID)[traits].mean()

def run(cluster: int, args) -> None:
    print(f"\n{'='*74}\nCLUSTER {cluster}\n{'='*74}")
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    lines, y, is08 = d["lines"], d["y"].astype("float64"), d["is_2008"]
    tr = ~is08
    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", lines, tr)

    lm = line_means(args.data_root, cluster, ["YLD_BE", "MST"]).reindex(pd.Index(lines))
    mst = np.nan_to_num(lm["MST"].to_numpy())

    p_yield = Ridge(alpha=args.alpha).fit(X[tr], y[tr]).predict(X)
    p_mst = Ridge(alpha=args.alpha).fit(X[tr], mst[tr]).predict(X)
    print(f"marker prediction of moisture, held out: "
          f"r = {np.corrcoef(p_mst[~tr], mst[~tr])[0,1]:+.3f}")
    print(f"marker prediction of yield,    held out: "
          f"r = {np.corrcoef(p_yield[~tr], y[~tr])[0,1]:+.3f}")

    # bushels forfeited per point of moisture above the 15.5% standard
    cost = 0.02 * args.yield_level / args.price
    index = p_yield - cost * p_mst
    print(f"\nmoisture penalty: {cost:.2f} bu/ac per point "
          f"(at {args.yield_level:.0f} bu/ac and ${args.price:.2f}/bu)")

    yte, mte = y[~tr], mst[~tr]
    k = max(int(args.budget * (~tr).sum()), 1)
    print(f"\n-- advancing the top {int(args.budget*100)}% ({k} lines) --")
    print(f"   {'ranked by':<26}{'realised yield':>16}{'realised moisture':>19}")
    for lab, p in [("yield alone", p_yield[~tr]), ("economic index", index[~tr])]:
        sel = np.argsort(-p)[:k]
        print(f"   {lab:<26}{yte[sel].mean()-yte.mean():>+15.2f}"
              f"{mte[sel].mean()-mte.mean():>+19.2f}")
    sy = np.argsort(-p_yield[~tr])[:k]; si = np.argsort(-index[~tr])[:k]
    net_y = (yte[sy].mean()-yte.mean()) - cost*(mte[sy].mean()-mte.mean())
    net_i = (yte[si].mean()-yte.mean()) - cost*(mte[si].mean()-mte.mean())
    print(f"\n   net economic value, yield alone   : {net_y:+.2f} bu/ac equivalent")
    print(f"   net economic value, economic index: {net_i:+.2f} bu/ac equivalent")
    print(f"   overlap between the two shortlists: "
          f"{100*len(set(sy.tolist())&set(si.tolist()))/k:.0f}%")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1])
    ap.add_argument("--alpha", type=float, default=30000)
    ap.add_argument("--price", type=float, default=4.00)
    ap.add_argument("--yield-level", type=float, default=150.0)
    ap.add_argument("--budget", type=float, default=0.10)
    args = ap.parse_args()
    for c in args.clusters:
        run(c, args)

if __name__ == "__main__":
    main()
