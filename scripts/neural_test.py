"""Do neural models beat ridge here? Tested rather than argued.

Earlier turns asserted that attention-based models would overfit this data. That was
reasoning from the data shape, not evidence. This runs the experiment.

Two architectures, both trained and selected exactly like the linear models:

  MLP        markers -> 256 -> 64 -> 1, dropout, the standard dense baseline.
  BlockAttn  markers are split into contiguous blocks along the genetic map, each
             block linearly embedded into a token, then a transformer encoder layer
             lets blocks attend to one another before pooling to a prediction. This
             is the "cross-feature attention over SNPs" idea in its cheapest honest
             form: attending over ~64 genomic regions rather than 2,686 individual
             markers, which would be both intractable and hopeless at this sample size.

Blocks follow the genetic map, so a token is a chromosome segment -- the unit that is
actually inherited together under high LD, and therefore the only grouping with a
biological justification.

Validation matches the rest of the project: a population-grouped holdout for early
stopping, so siblings never straddle the split, then a single score against the real
2008 cohort. Anything else would flatter the model.

Usage:
    python scripts/neural_test.py --clusters 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from build_dataset import prepare_markers


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, drop: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BlockAttn(nn.Module):
    """Markers -> genomic-region tokens -> self-attention -> prediction."""

    def __init__(self, n_in: int, n_blocks: int = 64, d: int = 32,
                 heads: int = 4, drop: float = 0.3):
        super().__init__()
        self.n_blocks = n_blocks
        self.pad = (-n_in) % n_blocks
        self.block = (n_in + self.pad) // n_blocks
        self.embed = nn.Linear(self.block, d)
        self.pos = nn.Parameter(torch.randn(1, n_blocks, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x):
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad))
        t = self.embed(x.view(x.size(0), self.n_blocks, self.block)) + self.pos
        return self.head(self.enc(t).mean(1)).squeeze(-1)


def train(model, Xtr, ytr, Xva, yva, epochs: int, lr: float, wd: float,
          batch: int, patience: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    n = len(Xtr)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).cpu().numpy()
        # Early stopping on correlation, which is the metric we actually report.
        r = np.corrcoef(pv, yva)[0, 1] if np.std(pv) > 1e-8 else -1.0
        if r > best:
            best, bad = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def run(cluster: int, args) -> None:
    d = np.load(args.data_dir / f"dataset_C{cluster}.npz", allow_pickle=True)
    y = d["y"].astype("float64")
    tr = ~d["is_2008"]
    X = prepare_markers(args.data_dir / f"geno_C{cluster}.npz", d["lines"], tr)
    pops = np.array([ln.split(".")[1] for ln in d["lines"]])

    # Population-grouped validation split for early stopping.
    rng = np.random.default_rng(args.seed)
    train_pops = np.unique(pops[tr])
    va_pops = set(rng.choice(train_pops, max(len(train_pops) // 5, 1), replace=False))
    is_va = np.array([p in va_pops for p in pops]) & tr
    is_tr = tr & ~is_va

    ys = y[is_tr].std()
    ymu = y[is_tr].mean()
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Xva, Xte = t(X[is_tr]), t(X[is_va]), t(X[~tr])
    ytr = t((y[is_tr] - ymu) / ys)
    yva_raw, yte = y[is_va], y[~tr]

    print(f"\n{'=' * 74}\nCLUSTER {cluster}\n{'=' * 74}")
    print(f"train {is_tr.sum():,} / val {is_va.sum():,} lines "
          f"({len(va_pops)} held-out populations) / test {(~tr).sum():,}")
    print(f"{X.shape[1]:,} markers, torch {torch.__version__}, {args.threads} threads\n")

    rows = []
    base = Ridge(alpha=30000).fit(X[tr], y[tr]).predict(X[~tr])
    rows.append(("ridge (baseline)", base, 0.0))

    for name, ctor in [("MLP", lambda: MLP(X.shape[1], args.hidden, args.dropout)),
                       ("BlockAttn", lambda: BlockAttn(X.shape[1], args.blocks))]:
        t0 = time.time()
        model = ctor()
        n_par = sum(p.numel() for p in model.parameters())
        vr = train(model, Xtr, ytr, Xva, yva_raw, args.epochs, args.lr,
                   args.weight_decay, args.batch, args.patience, args.seed)
        model.eval()
        with torch.no_grad():
            pred = model(Xte).cpu().numpy() * ys + ymu
        rows.append((f"{name} ({n_par/1e3:.0f}k params)", pred, time.time() - t0))
        print(f"  {name}: best val r = {vr:+.3f}  [{time.time()-t0:.0f}s]")

    print(f"\n-- HELD-OUT 2008 --")
    print(f"{'model':<28}{'pearson':>9}{'spearman':>10}{'gain':>9}{'time':>8}")
    for name, p, secs in rows:
        k = max(int(0.10 * len(p)), 1)
        sel = np.argsort(-p)[:k]
        r = np.corrcoef(p, yte)[0, 1] if np.std(p) > 1e-8 else 0.0
        rho = spearmanr(p, yte).statistic if np.std(p) > 1e-8 else 0.0
        print(f"{name:<28}{r:>9.3f}{rho:>10.3f}"
              f"{yte[sel].mean()-yte.mean():>9.2f}{secs:>7.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for c in args.clusters:
        run(c, args)


if __name__ == "__main__":
    main()
