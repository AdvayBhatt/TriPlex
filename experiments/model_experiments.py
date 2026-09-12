"""Controlled model experiments; does not modify the established forecast outputs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from triplex import (ID, META, YEAR, YIELD, candidate_table, correlation, family,
                     line_targets, load_genotypes, normalize, read_inputs, write_json)


def design(train, test, max_missing=.5, min_maf=.01, dtype='float64'):
    """All decisions about markers use training data only, including frequency."""
    train = train.astype(dtype)
    count = np.isfinite(train).sum(0)
    mean = np.nansum(train, axis=0) / np.maximum(count, 1)
    freq = (mean + 1) / 2
    keep = (count >= (1 - max_missing) * len(train)) & (count > 0)
    keep &= np.minimum(freq, 1 - freq) >= min_maf
    train = train[:, keep]
    mean = mean[keep].astype(dtype)
    train = np.where(np.isnan(train), mean, train)
    scale = train.std(0)
    varying = scale > 1e-6
    kept = np.flatnonzero(keep)[varying]
    mean, scale = mean[varying], scale[varying]
    train = (train[:, varying] - mean) / scale
    test = test[:, kept].astype(dtype)
    test = (np.where(np.isnan(test), mean, test) - mean) / scale
    return train, test, dict(markers=len(kept), keep=kept, mean=mean, scale=scale)


def ridge_path(x, y, test, alphas):
    """Reuse the sufficient statistics for penalties and multiple response columns."""
    y = np.asarray(y, dtype=x.dtype)
    if y.ndim == 1:
        y = y[:, None]
    xm, ym = x.mean(0), y.mean(0)
    gram = x.T @ x - len(x) * np.outer(xm, xm)
    rhs = x.T @ y - len(x) * np.outer(xm, ym)
    predictions, models = {}, {}
    for alpha in alphas:
        system = gram.copy()
        system.flat[::len(system) + 1] += alpha
        coef = cho_solve(cho_factor(system, lower=True, check_finite=False), rhs, check_finite=False)
        intercept = ym - xm @ coef
        predictions[alpha] = test @ coef + intercept
        models[alpha] = (coef, intercept)
    return predictions, models


def outcome_variant(history, mode='current'):
    if mode == 'current':
        return line_targets(history)[0]
    d = history.loc[history[YIELD].notna()].copy()
    if mode == 'legacy_ids':
        # Isolate records the old first-three-components parser could join.
        raw_keys = d._raw_id.str.split('.').str[:3].str.join('.')
        return line_targets(d.loc[raw_keys.eq(d[ID])])[0]
    if mode != 'legacy_clean':
        raise ValueError(mode)
    size = d.groupby([YEAR, 'LOC'])[ID].transform('nunique')
    d = d.loc[size >= 20].copy()
    group = d.groupby([YEAR, 'LOC'])[YIELD]
    z = (d[YIELD] - group.transform('mean')) / group.transform('std').replace(0, np.nan)
    d = d.loc[z.abs().le(5) | z.isna()].copy()
    d = d.loc[d.groupby(ID)[YIELD].transform('size') >= 2].copy()
    d['y'] = d[YIELD] - d.groupby([YEAR, 'LOC'])[YIELD].transform('mean')
    return d.groupby(ID).agg(y=('y', 'mean'), n_sites=('LOC', 'nunique'), year=(YEAR, 'first')).reset_index()


def training_arrays(history, metadata, geno, mode='current'):
    if history[YEAR].max() >= metadata[YEAR].min():
        raise ValueError('Forecast must occur after training')
    if set(history[ID].map(family)) & set(metadata[ID].map(family)):
        raise ValueError('Forecast population overlaps training')
    level = outcome_variant(history, mode)
    x, have = geno.rows(level[ID])
    level = level.loc[have].reset_index(drop=True)
    candidates = candidate_table(metadata)
    test, have_test = geno.rows(candidates[ID])
    # Never silently drop candidates in an experiment.
    full_test = np.full((len(candidates), len(geno.markers)), np.nan, dtype='float32')
    full_test[have_test] = test
    return level, x, candidates, full_test


def allocation_mask(costs, budget, order):
    selected = np.zeros(len(costs), dtype=bool)
    remaining = budget
    for index in order:
        if costs[index] <= remaining:
            selected[index] = True
            remaining -= costs[index]
    return selected


def random_reference(candidates, truth, repetitions=500, budget=None, replicates=1):
    joined = candidates.merge(truth[[ID, 'y']], on=ID, how='left', validate='one_to_one')
    costs, y = joined.n_sites.to_numpy(int) * replicates, joined.y.to_numpy()
    budget = int(np.floor(.1 * costs.sum())) if budget is None else budget
    if budget < costs.min():
        return np.full(repetitions, np.nan)
    rng = np.random.default_rng(8371)
    gain = []
    for _ in range(repetitions):
        selected = allocation_mask(costs, budget, rng.permutation(len(y)))
        gain.append(float(np.nanmean(y[selected]) - np.nanmean(y)))
    return np.array(gain)


def metrics(candidates, prediction, truth, random_gains=None, budget=None, replicates=1):
    d = candidates.assign(prediction=prediction).merge(truth[[ID, 'y']], on=ID, how='left', validate='one_to_one')
    costs = d.n_sites.to_numpy(int) * replicates
    budget = int(np.floor(.1 * costs.sum())) if budget is None else budget
    order = np.lexsort((d[ID].to_numpy(), -d.prediction.to_numpy()))
    chosen = allocation_mask(costs, budget, order)
    finite = d.y.notna()
    y, p = d.loc[finite, 'y'].to_numpy(), d.loc[finite, 'prediction'].to_numpy()
    gain = float(d.loc[chosen, 'y'].mean() - d.y.mean())
    result = dict(pearson=correlation(p, y), rmse=float(np.sqrt(np.mean((p-y)**2))),
                  baseline_rmse=float(np.sqrt(np.mean(y*y))), allocation_gain=gain,
                  advanced=int(chosen.sum()), plots=int(costs[chosen].sum()), budget=budget,
                  candidates=len(d), scored=int(finite.sum()))
    if random_gains is not None:
        finite_random = random_gains[np.isfinite(random_gains)]
        if len(finite_random) and np.isfinite(gain):
            result.update(random_gain_mean=float(finite_random.mean()),
                          random_gain_95=np.quantile(finite_random, [.025, .975]).tolist(),
                          gain_over_random_mean=gain-float(finite_random.mean()),
                          random_allocation_percentile=float(np.mean(finite_random < gain)))
        else:
            result.update(random_gain_mean=None, random_gain_95=None,
                          gain_over_random_mean=None, random_allocation_percentile=None)
    return result


def diagnose(root, out, clusters):
    """Predeclared one-change diagnostics on already-inspected 2008; never select by these scores."""
    variants = [
        ('current_float32', 2001, .2, 0., 'current', 'float32'),
        ('precision_only', 2001, .2, 0., 'current', 'float64'),
        ('coverage50_only', 2001, .5, 0., 'current', 'float64'),
        ('coverage50_maf01', 2001, .5, .01, 'current', 'float64'),
        ('year2000_only', 2000, .2, 0., 'current', 'float64'),
        ('legacy_ids_only', 2001, .2, 0., 'legacy_ids', 'float64'),
        ('legacy_clean_only', 2001, .2, 0., 'legacy_clean', 'float64'),
        ('combined_audit_like_train_only', 2000, .5, .01, 'legacy_clean', 'float64'),
    ]
    rows = []
    for c in clusters:
        history, metadata, _ = read_inputs(root, c, 2000, 2008)
        geno = load_genotypes(root, c, root.parent / 'processed/forecast_cache')
        raw = pd.read_csv(root / f'C{c}_Phenotype_Data_V2.csv', usecols=META+[YIELD], low_memory=False)
        target, _ = normalize(raw.loc[raw[YEAR].eq(2008)])
        truth = line_targets(target)[0]
        for name, start, missing, maf, mode, precision in variants:
            print(f'C{c} diagnostic: {name}', flush=True)
            level, x, candidates, test = training_arrays(history.loc[history[YEAR] >= start], metadata, geno, mode)
            x, test, prep = design(x, test, missing, maf, precision)
            fit = Ridge(alpha=30000., solver='cholesky').fit(x, level.y.to_numpy(dtype=precision))
            pred = fit.predict(test)
            row = dict(cluster=c, variant=name, train_lines=len(level), markers=prep['markers'],
                       **metrics(candidates, pred, truth))
            rows.append(row)
            print(json.dumps(row), flush=True)
            pd.DataFrame(rows).to_csv(out / 'baseline_diagnostics.csv', index=False)
            del x, test, prep, fit
            gc.collect()
    return rows


def parent_matrix(root, cluster, geno):
    path = root.parent / f'processed/forecast_cache/parents_C{cluster}_{geno.audit["source_sha256"]}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return pd.DataFrame(d['values'], index=d['families'], columns=d['markers'])
    rows, families = [], []
    for file in sorted((root / f'genotypes/C{cluster}').rglob('*_Imputed.csv')):
        d = pd.read_csv(file, index_col=0, nrows=2)
        if not d.index.astype(str).str.startswith('PID').all() or len(d) != 2:
            continue
        if set(d.columns) != set(geno.markers):
            raise ValueError(f'Parent panel mismatch: {file}')
        values = d.loc[:, geno.markers].to_numpy(float)
        if not (np.isnan(values) | np.isin(values, [-1, 0, 1])).all():
            raise ValueError(f'Invalid parent calls: {file}')
        counts = np.isfinite(values).sum(0)
        mean = np.divide(np.nansum(values, axis=0), counts, out=np.full(len(counts), np.nan), where=counts > 0)
        if not np.isfinite(mean).any():
            continue
        rows.append(mean)
        families.append(file.stem.split('_')[0])
    frame = pd.DataFrame(rows, index=families, columns=geno.markers)
    np.savez_compressed(path, values=frame.to_numpy(), families=frame.index.to_numpy(str), markers=geno.markers)
    return frame


def predictions_for_origin(history, metadata, geno, parents):
    level, raw_x, candidates, raw_test = training_arrays(history, metadata, geno)
    family_labels = level[ID].map(family)
    family_mean = level.groupby(family_labels).y.mean()
    deviation = level.y - family_labels.map(family_mean)
    ys = np.column_stack([level.y, deviation])
    predictions = {}
    # Keep the current method as a control, independent of diagnostic outcomes.
    x, test, _ = design(raw_x, raw_test, .2, 0., 'float32')
    control = Ridge(alpha=30000., solver='cholesky').fit(x, level.y.to_numpy('float32'))
    predictions['control'] = control.predict(test)
    del x, test, control
    x, test, _ = design(raw_x, raw_test, .5, .01)
    path, _ = ridge_path(x, ys, test, [3000., 30000., 300000., 3000000.])
    for alpha, values in path.items():
        predictions[f'flat_{int(alpha)}'] = values[:, 0]
    del x, test, raw_x, raw_test
    gc.collect()
    train_families = family_mean.index.intersection(parents.index)
    tx = parents.reindex(train_families).to_numpy()
    px = parents.reindex(candidates.population).to_numpy()
    tx, px, _ = design(tx, px, 1., 0.)
    # Three predeclared family penalties; outer historical forecasts select the model.
    for alpha in [100., 1000., 10000.]:
        fit = Ridge(alpha=alpha, solver='cholesky').fit(tx, family_mean.reindex(train_families).to_numpy())
        fp = fit.predict(px)
        combined = fp + path[30000.][:, 1]
        missing = ~candidates.population.isin(parents.index)
        combined[missing] = path[30000.][missing, 0]
        predictions[f'parents_{int(alpha)}'] = combined
        # Raw bu/acre blend: no target-cohort standardization or target outcome information.
        predictions[f'blend_{int(alpha)}'] = .5 * combined + .5 * path[30000.][:, 0]
    return candidates, predictions


def history_experiment(root, out, clusters):
    rows = []
    for c in clusters:
        history, _, _ = read_inputs(root, c, 2001, 2008)
        geno = load_genotypes(root, c, root.parent / 'processed/forecast_cache')
        parents = parent_matrix(root, c, geno)
        for year in [2004, 2005, 2006]:
            print(f'C{c}: development forecast {year}', flush=True)
            held = history.loc[history[YEAR].eq(year)]
            candidates, predictions = predictions_for_origin(history.loc[history[YEAR] < year], held[META], geno, parents)
            truth = line_targets(held)[0]
            random_gains = random_reference(candidates, truth)
            saved = candidates.merge(truth[[ID, 'y']], on=ID, how='left')
            for name, values in predictions.items():
                saved[name] = values
                rows.append(dict(cluster=c, year=year, model=name, **metrics(candidates, values, truth, random_gains)))
            saved.to_csv(out / f'C{c}_development_{year}.csv', index=False)
            write_json(out / 'development_metrics.json', rows)
            pd.DataFrame(rows).drop(columns=['random_gain_95']).to_csv(out / 'development_metrics.csv', index=False)
        print(f'C{c}: development complete; 2007/2008 not evaluated by this stage', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['diagnose', 'develop'], required=True)
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/model_comparison'))
    p.add_argument('--clusters', type=int, nargs='+', default=[1, 2], choices=[1, 2])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / '.gitignore').write_text('*\n')
    write_json(args.output_dir / f'{args.stage}_protocol.json', dict(
        parameters=vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        development_years=[2004, 2005, 2006], confirmation_year=2007,
        selection='Largest mean annual gain over budget-matched random allocation, positive in >=2 of 3 origins; tie: lower mean RMSE',
        calibration='Nonnegative slope capped at one and intercept fit on development forecast predictions only',
        target='Current field-centered line means, >=20 lines/field and >=2 sites/line; all candidates retained',
        diagnostic_caveat='2008 diagnostics explain known gaps only; not used to choose historical model candidates'))
    with threadpool_limits(limits=4):
        if args.stage == 'diagnose':
            diagnose(args.data_dir, args.output_dir, args.clusters)
        else:
            history_experiment(args.data_dir, args.output_dir, args.clusters)


if __name__ == '__main__':
    main()
