"""Auditable broad-acre prediction with an explicit historical/forecast boundary.

Yield targets are line means after subtracting year-location means. Marker effects
transfer across populations; unavailable genotypes get a phenotypic-only baseline.
Target-season yields are accessed only by the retrospective evaluator.
"""
import hashlib
import importlib.metadata
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ID = 'LINE_UNIQUE_ID'
YEAR = 'YEAR_x'
YIELD = 'YLD_BE'
META = [ID, YEAR, 'LOC', 'CROSS']
SCHEMA = 'forecast-v1'


class InsufficientOutcomesError(ValueError):
    """An outcome partition cannot support the declared evaluation target."""


def canonical(value):
    """Known numeric aliases only; never guess a corrupted identifier."""
    m = re.fullmatch(r'C([12])\.(\d+)\.(\d+)(?:[.#]\d+)?', str(value))
    return f'C{int(m[1])}.{int(m[2])}.{int(m[3])}' if m else None


def family(line):
    return str(line).rsplit('.', 1)[0]


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def normalize(frame, strict=False):
    out = frame.copy()
    out['_raw_id'] = out[ID].astype(str)
    ids = out[ID].map(canonical)
    bad = out.loc[ids.isna(), ID].astype(str).unique().tolist()
    if strict and bad:
        raise ValueError(f'Unresolvable candidate IDs: {bad[:10]}')
    out[ID] = ids
    out = out.loc[ids.notna()].copy()
    if out['LOC'].isna().any():
        raise ValueError('Missing location metadata')
    out['LOC'] = out['LOC'].astype(str)
    out['CROSS'] = out['CROSS'].fillna('unknown').astype(str)
    if out.groupby(ID)['CROSS'].nunique().max() > 1:
        raise ValueError('Conflicting pedigree metadata for the same normalized line')
    return out, bad


def read_inputs(root, cluster, start, target, candidate_path=None):
    path = root / f'C{cluster}_Phenotype_Data_V2.csv'
    # Read metadata separately so candidate eligibility cannot depend on outcome availability.
    meta = pd.read_csv(path, usecols=META, low_memory=False)
    if candidate_path:
        supplied = pd.read_csv(candidate_path, usecols=[ID, 'LOC', 'CROSS'])
        if supplied[ID].map(canonical).isna().any():
            raise ValueError('Unresolvable candidate IDs in external roster')
        supplied = supplied[supplied[ID].astype(str).str.startswith(f'C{cluster}.')].copy()
        supplied[YEAR] = target
    else:
        supplied = meta.loc[meta[YEAR].eq(target)].copy()
    candidates, _ = normalize(supplied, strict=True)
    if candidates.empty:
        raise ValueError(f'No C{cluster} candidates for {target}')
    # CSV parsing touches the column, but only historical rows enter any fit or transformation.
    history = pd.read_csv(path, usecols=META + [YIELD], low_memory=False)
    history = history.loc[history[YEAR].between(start, target - 1)].copy()
    original_history_rows = len(history)
    history, bad = normalize(history)
    expected = f'C{cluster}.'
    if not history[ID].str.startswith(expected).all() or not candidates[ID].str.startswith(expected).all():
        raise ValueError('Cluster labels disagree with phenotype file')
    return history, candidates, {'excluded_historical_ids': bad, 'excluded_historical_rows': original_history_rows - len(history),
                                'historical_rows': len(history), 'candidate_rows': len(candidates),
                                'candidate_lines': candidates[ID].nunique(),
                                'historical_input_sha256': hashlib.sha256(pd.util.hash_pandas_object(
                                    history[META + [YIELD]], index=False).values.tobytes()).hexdigest(),
                                'candidate_input_sha256': hashlib.sha256(pd.util.hash_pandas_object(
                                    candidates[META], index=False).values.tobytes()).hexdigest()}


def candidate_table(metadata):
    out = metadata.groupby(ID, sort=True).agg(CROSS=('CROSS', 'first'), n_sites=('LOC', 'nunique')).reset_index()
    out['population'] = out[ID].map(family)
    return out


@dataclass
class Genotypes:
    lines: np.ndarray
    markers: np.ndarray
    values: np.ndarray
    audit: dict

    def __post_init__(self):
        self.index = pd.Index(self.lines)

    def rows(self, ids):
        positions = self.index.get_indexer(ids)
        present = positions >= 0
        present[present] &= (self.values[positions[present]] != 2).any(axis=1)
        x = self.values[positions[present]].astype('float32')
        x[x == 2] = np.nan
        return x, present


def load_genotypes(root, cluster, cache):
    paths = sorted((root / f'genotypes/C{cluster}').rglob('*_Imputed.csv'))
    if not paths:
        raise FileNotFoundError(f'Expand genotype CSVs under {root}/genotypes/C{cluster}/ first')
    digest = hashlib.sha256(SCHEMA.encode())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    signature = digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f'C{cluster}_{signature}.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as g:
            return Genotypes(g['lines'], g['markers'], g['values'], json.loads(str(g['audit'])))
    panel, rows, ids, seen, conflicts, malformed = None, [], [], {}, set(), []
    duplicate_rows = 0
    for path in paths:
        pop = path.stem.split('_')[0]
        if not re.fullmatch(fr'C{cluster}\.\d+', pop):
            raise ValueError(f'Unexpected population filename: {path}')
        g = pd.read_csv(path, index_col=0, low_memory=False)
        if panel is None:
            panel = list(g.columns)
        if len(g.columns) != len(panel) or set(g.columns) != set(panel):
            raise ValueError(f'Marker panels differ: {path}')
        values = g.loc[:, panel].to_numpy(dtype='float32')
        if not (np.isnan(values) | np.isin(values, [-1, 0, 1])).all():
            raise ValueError(f'Invalid genotype calls in {path}; expected -1, 0, 1 or NA')
        values = np.where(np.isnan(values), 2, values).astype('int8')
        for raw, row in zip(g.index.astype(str), values):
            if raw.startswith('PID'):
                continue
            lid = canonical(pop + '.' + raw)
            if lid is None:
                malformed.append(pop + '.' + raw)
                continue
            if lid in seen:
                duplicate_rows += 1
                if not np.array_equal(rows[seen[lid]], row):
                    conflicts.add(lid)
                continue
            seen[lid] = len(rows)
            rows.append(row)
            ids.append(lid)
    keep = np.array([x not in conflicts for x in ids])
    audit = dict(files=len(paths), markers=len(panel), source_sha256=signature,
                 malformed_ids=malformed, conflicting_ids=sorted(conflicts), duplicate_rows=duplicate_rows)
    result = Genotypes(np.array(ids)[keep], np.array(panel), np.array(rows, dtype='int8')[keep], audit)
    np.savez_compressed(destination, lines=result.lines, markers=result.markers,
                        values=result.values, audit=json.dumps(audit))
    return result


def line_targets(history, min_field_lines=20, min_sites=2):
    """Build truth independently in each time partition; no target-season filtering of candidates."""
    d = history.loc[np.isfinite(pd.to_numeric(history[YIELD], errors='coerce'))].copy()
    d[YIELD] = pd.to_numeric(d[YIELD])
    # Repeated records at the same line/site/year do not count as independent sites.
    d = d.groupby([ID, YEAR, 'LOC'], as_index=False)[YIELD].mean()
    sizes = d.groupby([YEAR, 'LOC'])[ID].transform('nunique')
    d = d.loc[sizes >= min_field_lines].copy()
    d['adjusted'] = d[YIELD] - d.groupby([YEAR, 'LOC'])[YIELD].transform('mean')
    levels = d.groupby(ID).agg(y=('adjusted', 'mean'), n_sites=('LOC', 'nunique'),
                              year=(YEAR, 'first')).reset_index()
    levels = levels.loc[levels.n_sites >= min_sites].reset_index(drop=True)
    d = d.loc[d[ID].isin(levels[ID])]
    if levels.empty:
        raise InsufficientOutcomesError('No usable outcomes: need 20 lines per field and two sites per line')
    levels['population'] = levels[ID].map(family)
    return levels, d


def phenotypic_blup(plots):
    """REML random line intercept after field centering; new unrelated lines get effect zero.

    These variance components condition on estimated field means, so they are an
    approximation, not a joint multi-environment genetic variance decomposition.
    """
    g = plots.groupby(ID)['adjusted'].agg(['size', 'mean', 'var'])
    n = g['size'].to_numpy(float)
    means = g['mean'].to_numpy(float)
    ss = np.nansum((n - 1) * g['var'].fillna(0).to_numpy())
    count = n.sum()

    def objective(theta, details=False):
        residual, line = np.exp(theta)
        denom = residual + n * line
        weights = n / denom
        intercept = np.sum(weights * means) / weights.sum()
        q = ss / residual + np.sum(weights * (means - intercept) ** 2)
        logdet = np.sum((n - 1) * np.log(residual) + np.log(denom))
        loss = .5 * (logdet + np.log(weights.sum()) + q)
        return (intercept, residual, line) if details else loss

    variance = max(float(plots.adjusted.var()), 1.)
    fit = minimize(objective, np.log([variance * .75, variance * .25]),
                   method='L-BFGS-B', bounds=[(-12, 20), (-12, 20)])
    if not fit.success:
        raise RuntimeError(f'Phenotypic baseline REML failed: {fit.message}')
    intercept, residual, line = objective(fit.x, True)
    return dict(intercept=intercept, residual_variance=residual, line_variance=line,
                observations=int(count), fitted_lines=len(g), converged=bool(fit.success))


@dataclass
class MarkerModel:
    keep: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, x):
        x = x[:, self.keep]
        x = np.where(np.isnan(x), self.mean, x)
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def save(self, path, markers):
        np.savez_compressed(path, keep=self.keep, mean=self.mean, scale=self.scale,
                            coef=self.coef, intercept=self.intercept, markers=markers)


def fit_markers(x, y, alpha):
    # Selection, imputation and scaling see only historical training lines.
    present = np.isfinite(x)
    counts = present.sum(axis=0)
    means = np.nansum(x, axis=0) / np.maximum(counts, 1)
    filled = np.where(present, x, means).astype('float32')
    scale = filled.std(axis=0)
    keep = (counts >= .8 * len(x)) & (scale > 1e-6)
    if not keep.any():
        raise ValueError('No usable training markers')
    mean, sd = means[keep].astype('float32'), scale[keep]
    z = (filled[:, keep] - mean) / sd
    fit = Ridge(alpha=alpha, solver='cholesky').fit(z, y)
    return MarkerModel(keep, mean, sd, fit.coef_, float(fit.intercept_))


def forecast(history, metadata, geno, alpha):
    if history.empty or metadata.empty or history[YEAR].max() >= metadata[YEAR].min():
        raise ValueError('Training years must be strictly earlier than forecast years')
    if set(history[ID].map(family)) & set(metadata[ID].map(family)):
        raise ValueError('This workflow requires new forecast populations; overlap found')
    levels, plots = line_targets(history)
    baseline = phenotypic_blup(plots)
    x, have = geno.rows(levels[ID])
    if have.sum() < 20:
        raise ValueError('Fewer than 20 training lines with DNA')
    model = fit_markers(x, levels.loc[have, 'y'].to_numpy('float32'), alpha)
    training_dna = int(have.sum())
    out = candidate_table(metadata)
    x, have = geno.rows(out[ID])
    informative = np.isfinite(x[:, model.keep]).any(axis=1)
    x = x[informative]
    have[have] &= informative
    out['predicted_yield_advantage'] = baseline['intercept']
    out.loc[have, 'predicted_yield_advantage'] = model.predict(x)
    out['prediction_source'] = np.where(have, 'marker_ridge', 'phenotypic_fallback')
    out['phenotypic_blup_baseline'] = baseline['intercept']
    out['environment_mean_baseline'] = 0.
    novel = metadata.loc[~metadata.LOC.isin(history.LOC), [ID, 'LOC']].drop_duplicates().groupby(ID).size()
    out['n_novel_sites'] = out[ID].map(novel).fillna(0).astype(int)
    out['needs_review'] = (~have) | out.n_novel_sites.gt(0)
    out = out.sort_values(['predicted_yield_advantage', ID], ascending=[False, True]).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    info = dict(train_years=sorted(history[YEAR].unique()), forecast_years=sorted(metadata[YEAR].unique()),
                train_lines=len(levels), train_lines_with_dna=training_dna, retained_markers=int(model.keep.sum()),
                candidates=len(out), candidates_without_dna=int((~have).sum()), phenotypic_blup=baseline)
    return out, model, info


def allocate(predictions, budget=None, replicates=1):
    """Rank-first full site bundles; deterministic, budget-feasible, not a knapsack optimum."""
    if replicates < 1 or (budget is not None and budget < 0):
        raise ValueError('Invalid plot budget or replication')
    out = predictions.copy()
    out['plots_if_advanced'] = out.n_sites.astype(int) * replicates
    budget = int(np.floor(out.plots_if_advanced.sum() * .1)) if budget is None else int(budget)
    remaining, selected = budget, []
    for cost in out.plots_if_advanced:
        advance = cost <= remaining
        selected.append(advance)
        if advance:
            remaining -= cost
    out['advance'] = selected
    out['allocated_plots'] = np.where(out.advance, out.plots_if_advanced, 0)
    return out, dict(plot_budget=budget, plots_allocated=budget - remaining, unused_plots=remaining,
                     lines_advanced=int(out.advance.sum()), replicates_per_location=replicates)


def add_intervals(predictions, residuals):
    """Central 90% historical forecast-error intervals; no claim of exact future coverage."""
    out = predictions.copy()
    out['lower_90'] = np.nan
    out['upper_90'] = np.nan
    out['calibration_n'] = 0
    for source in out.prediction_source.unique():
        errors = residuals.loc[residuals.prediction_source.eq(source), 'error'].dropna()
        mask = out.prediction_source.eq(source)
        out.loc[mask, 'calibration_n'] = len(errors)
        if len(errors) >= 20:
            lo, hi = np.quantile(errors, [.05, .95])
            out.loc[mask, 'lower_90'] = out.loc[mask, 'predicted_yield_advantage'] + lo
            out.loc[mask, 'upper_90'] = out.loc[mask, 'predicted_yield_advantage'] + hi
    if 'needs_review' in out:
        out['needs_review'] |= out.lower_90.isna()
    return out


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-10 and np.std(b) > 1e-10 else None


def evaluate(predictions, truth):
    scored = predictions.merge(truth[[ID, 'y']], on=ID, validate='one_to_one').reset_index(drop=True)
    a, b = scored.predicted_yield_advantage.to_numpy(), scored.y.to_numpy()
    if len(scored) == 0:
        return {'scored_lines': 0, 'candidate_lines': len(predictions)}, scored
    centered = scored[['population', 'y', 'predicted_yield_advantage']].copy()
    for col in ['y', 'predicted_yield_advantage']:
        centered[col] -= centered.groupby('population')[col].transform('mean')
    selected = scored.loc[scored.advance, 'y']
    top = scored.sort_values(['predicted_yield_advantage', ID], ascending=[False, True]).head(max(1, int(np.ceil(.1 * len(scored)))))
    result = dict(candidate_lines=len(predictions), scored_lines=len(scored), pearson=correlation(a, b),
                  spearman=float(spearmanr(a, b).statistic) if np.std(a) > 1e-10 else None,
                  rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                  within_population_pearson=correlation(centered.predicted_yield_advantage, centered.y),
                  top_decile_gain=float(top.y.mean() - scored.y.mean()),
                  advanced_lines_scored=len(selected), advanced_gain=float(selected.mean() - scored.y.mean()) if len(selected) else None)
    for col in ['phenotypic_blup_baseline', 'environment_mean_baseline']:
        result[col] = dict(rmse=float(np.sqrt(np.mean((scored[col] - b) ** 2))),
                           pearson=correlation(scored[col], b))
    finite = scored.lower_90.notna() & scored.upper_90.notna()
    result['interval_lines_scored'] = int(finite.sum())
    result['interval_90_coverage'] = float(scored.loc[finite, 'y'].between(
        scored.loc[finite, 'lower_90'], scored.loc[finite, 'upper_90']).mean()) if finite.any() else None
    # Resample whole populations, acknowledging family dependence in pooled accuracy.
    groups = list(scored.groupby('population').indices.values())
    rng, bootstrap = np.random.default_rng(42), []
    if len(groups) >= 3:
        for _ in range(200):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            r = correlation(a[idx], b[idx])
            if r is not None:
                bootstrap.append(r)
    result['pearson_population_bootstrap_95'] = np.quantile(bootstrap, [.025, .975]).tolist() if bootstrap else None
    scored['error'] = scored.y - scored.predicted_yield_advantage
    return result, scored


def run(args):
    started = time.perf_counter()
    manifest = dict(schema=SCHEMA, synthetic=args.sample, parameters=vars(args).copy(), clusters={},
                    dependencies={name: importlib.metadata.version(name) for name in
                                  ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']},
                    source_sha256={name: hashlib.sha256(next(Path(__file__).resolve().parents[1]/folder/name
                                   for folder in ('scripts','src')
                                   if (Path(__file__).resolve().parents[1]/folder/name).exists()).read_bytes()).hexdigest()
                                   for name in ['triplex.py', 'run_pipeline.py', 'demo_data.py']})
    for cluster in args.clusters:
        print(f'C{cluster}: reading historical records and candidate metadata', flush=True)
        history, candidates, input_audit = read_inputs(args.data_dir, cluster, args.start_year, args.target_year, args.candidates)
        aliases = pd.concat([history[['_raw_id', ID]], candidates[['_raw_id', ID]]]).drop_duplicates()
        aliases.loc[aliases._raw_id.ne(aliases[ID])].to_csv(args.output_dir / f'C{cluster}_id_aliases.csv', index=False)
        geno = load_genotypes(args.data_dir, cluster, args.data_dir.parent / 'processed/forecast_cache')
        residuals = pd.DataFrame(columns=[ID, 'prediction_source', 'error', 'forecast_year'])
        validation = []
        for year in args.validation_years:
            print(f'C{cluster}: historical forecast for {year}', flush=True)
            before = history.loc[history[YEAR] < year]
            heldout = history.loc[history[YEAR].eq(year)]
            pred, _, info = forecast(before, heldout[META], geno, args.alpha)
            pred = add_intervals(pred, residuals)
            pred, plan = allocate(pred, args.plot_budget, args.replicates)
            truth, _ = line_targets(heldout)
            metrics, scored = evaluate(pred, truth)
            validation.append(dict(year=year, fit=info, allocation=plan, metrics=metrics))
            scored['forecast_year'] = year
            errors = scored[[ID, 'prediction_source', 'error', 'forecast_year']]
            residuals = errors.copy() if residuals.empty else pd.concat([residuals, errors], ignore_index=True)
            scored.to_csv(args.output_dir / f'C{cluster}_validation_{year}.csv', index=False)
        print(f'C{cluster}: fitting final forecast for {args.target_year}', flush=True)
        predictions, model, info = forecast(history, candidates, geno, args.alpha)
        predictions = add_intervals(predictions, residuals)
        predictions, plan = allocate(predictions, args.plot_budget, args.replicates)
        predictions.to_csv(args.output_dir / f'C{cluster}_rankings.csv', index=False)
        schedule = candidates[[ID, 'LOC']].drop_duplicates().merge(
            predictions.loc[predictions.advance, [ID, 'rank']], on=ID, validate='many_to_one')
        schedule['plots'] = args.replicates
        schedule.sort_values(['rank', 'LOC']).to_csv(args.output_dir / f'C{cluster}_plot_plan.csv', index=False)
        if int(schedule.plots.sum()) != plan['plots_allocated']:
            raise AssertionError('Plot plan does not match allocation')
        model.save(args.output_dir / f'C{cluster}_marker_model.npz', geno.markers)
        residuals.to_csv(args.output_dir / f'C{cluster}_calibration.csv', index=False)
        # Descriptive historical environments, never a claim to know 2008 growing-season weather.
        history.groupby('LOC')[YIELD].agg(['count', 'mean', 'std']).to_csv(args.output_dir / f'C{cluster}_historical_sites.csv')
        result = dict(input_audit=input_audit, genotype_audit=geno.audit, validation=validation,
                      final_fit=info, allocation=plan, target_evaluation=None,
                      novel_candidate_sites=sorted(set(candidates.LOC) - set(history.LOC)))
        # Separate evaluator. Nothing below changes the saved rankings, intervals, or allocation.
        if not args.predict_only:
            print(f'C{cluster}: retrospective target-season scoring (exploratory, not untouched)', flush=True)
            raw = pd.read_csv(args.data_dir / f'C{cluster}_Phenotype_Data_V2.csv', usecols=META + [YIELD], low_memory=False)
            target, _ = normalize(raw.loc[raw[YEAR].eq(args.target_year)])
            if target[YIELD].notna().any():
                try:
                    truth, _ = line_targets(target)
                except InsufficientOutcomesError as exc:
                    result['target_evaluation'] = {'status': str(exc)}
                else:
                    metrics, scored = evaluate(predictions, truth)
                    result['target_evaluation'] = metrics
                    scored.to_csv(args.output_dir / f'C{cluster}_retrospective_evaluation.csv', index=False)
            else:
                result['target_evaluation'] = {'status': 'No target outcomes available'}
        manifest['clusters'][f'C{cluster}'] = result
        write_json(args.output_dir / f'C{cluster}_report.json', result)
        print(f'C{cluster}: {len(predictions)} candidates, {plan["lines_advanced"]} advanced, {plan["plots_allocated"]} plots', flush=True)
    manifest['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / 'run_manifest.json', manifest)
    print(f'Finished in {manifest["elapsed_seconds"]:.1f}s. Results: {args.output_dir}', flush=True)
