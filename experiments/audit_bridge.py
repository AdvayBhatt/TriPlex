"""Forensic reproduction from this workspace's reviewed legacy caches; never a deployable model.

These NPZ files were generated locally during the prior branch audit. Their object
arrays require pickle support; do not point this script at untrusted NPZ files.
"""
import gc
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from model_experiments import design
from triplex import ID, META, YEAR, YIELD, correlation, line_targets, normalize


def run():
    cache = Path('report/review/reproduction')
    out = Path('outputs/model_comparison')
    results = []
    for cluster in [1, 2]:
        with np.load(cache / f'dataset_C{cluster}.npz', allow_pickle=True) as d:
            lines, y, target = d['lines'].astype(str), d['y'].astype(float), d['is_2008']
        with np.load(cache / f'geno_C{cluster}.npz', allow_pickle=True) as g:
            pos = pd.Index(g['lines']).get_indexer(lines)
            assert (pos >= 0).all()
            x = g['X'][pos].astype('float64')
        x[x == -2] = np.nan
        raw = pd.read_csv(f'data/raw/C{cluster}_Phenotype_Data_V2.csv', usecols=META+[YIELD], low_memory=False)
        normalized, _ = normalize(raw.loc[raw[YEAR].eq(2008)])
        truth = line_targets(normalized)[0].set_index(ID).y.reindex(lines[target]).to_numpy()
        for name in ['legacy_joint_filter', 'legacy_train_only_filter']:
            if name == 'legacy_joint_filter':
                count = np.isfinite(x).sum(0)
                mean = np.nansum(x, axis=0) / np.maximum(count, 1)
                freq = (mean+1)/2
                keep = (count >= .5*len(x)) & (np.minimum(freq,1-freq) >= .01)
                # Mimic the legacy filter, but keep training imputation/scaling identical.
                a, b, info = design(x[~target][:, keep], x[target][:, keep], 1., 0.)
            else:
                a, b, info = design(x[~target], x[target], .5, .01)
            fit = Ridge(alpha=30000., solver='cholesky').fit(a, y[~target])
            predictions = fit.predict(b)
            result = dict(cluster=cluster, variant=name, training_lines=int((~target).sum()),
                          markers=info['markers'], legacy_truth_r=correlation(predictions,y[target]),
                          common_truth_r=correlation(predictions,truth))
            results.append(result)
            print(result, flush=True)
            pd.DataFrame(results).to_csv(out / 'audit_bridge.csv',index=False)
            del a,b,fit
            gc.collect()
        del x
        gc.collect()


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
