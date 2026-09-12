"""Conditional robustness audit. Never changes a trained predictor or its policy.

Historical outer checks are 2006 and 2007; selection uses 2004 onward strictly
before the outer year. The model library itself was designed retrospectively.
Family bootstrap holds sites, fitted predictions and allocation fixed. Site
deletion is a separate sensitivity analysis, not a joint confidence interval.
"""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from model_experiments import allocation_mask, metrics
from roster_experiment import adjust, slope_calibration
from select_model import choose
from triplex import ID, META, YEAR, YIELD, normalize, write_json


def select_before(rows, frames, year):
    past = rows.loc[rows.year < year]
    years = sorted(past.year.unique())
    if len(years) < 2 or set(years) - set(frames):
        raise ValueError('At least two complete earlier origins required')
    winner, _ = choose(past)
    if winner == 'control':
        winner = 'control__raw'
    return winner, slope_calibration([frames[y] for y in years], winner), years


def selected_mask(frame, values):
    costs = frame.n_sites.to_numpy(int)
    return allocation_mask(costs, int(np.floor(.1 * costs.sum())),
                           np.lexsort((frame[ID].to_numpy(), -np.asarray(values))))


def paired_stats(y, p, q, selected, control, weights=None):
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(q) & (w > 0)
    def mean(v, mask):
        m = valid & mask
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan
    all_rows = np.ones(len(y), bool)
    gain = mean(y, selected) - mean(y, all_rows)
    return np.array([gain, mean(y, selected) - mean(y, control),
                     np.sqrt(mean((p-y)**2, all_rows)) - np.sqrt(mean((q-y)**2, all_rows)),
                     np.sqrt(mean((p-y)**2, all_rows)) - np.sqrt(mean(y*y, all_rows))])


STAT_NAMES = ['selected_gain', 'gain_difference_vs_control',
              'rmse_difference_vs_control', 'rmse_difference_vs_zero']


def family_bootstrap(frame, repeats=2000):
    codes, families = pd.factorize(frame.population, sort=True)
    y, p, q = (frame[k].to_numpy(float) for k in ['y', 'p', 'q'])
    a, b = frame.a.to_numpy(bool), frame.b.to_numpy(bool)
    rng = np.random.default_rng(9122026)
    draws = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(len(families), size=len(families)), minlength=len(families))
        draws.append(paired_stats(y, p, q, a, b, counts[codes]))
    draws = np.array(draws)
    estimate = paired_stats(y, p, q, a, b)
    return {name: dict(estimate=estimate[i], percentile_95=np.nanquantile(draws[:, i], [.025, .975]).tolist(),
                       valid_draws=int(np.isfinite(draws[:, i]).sum())) for i, name in enumerate(STAT_NAMES)}


def site_deletion(frame, observed):
    # Rebuild the target independently: equal weight per distinct line-site,
    # fields with >=20 finite lines and lines observed at >=2 sites.
    d = observed.loc[np.isfinite(observed[YIELD])].groupby([ID, 'LOC'], as_index=False)[YIELD].mean()
    d = d.loc[d.groupby('LOC')[ID].transform('size') >= 20].copy()
    d['z'] = d[YIELD] - d.groupby('LOC')[YIELD].transform('mean')
    index = pd.Series(np.arange(len(frame)), index=frame[ID])
    d['i'] = d[ID].map(index)
    if d.i.isna().any():
        raise ValueError('Outcome contains an unknown candidate')
    counts = np.bincount(d.i, minlength=len(frame))
    sums = np.bincount(d.i, weights=d.z, minlength=len(frame))
    base = np.divide(sums, counts, out=np.full(len(frame), np.nan), where=counts >= 2)
    np.testing.assert_allclose(base, frame.y, equal_nan=True, atol=1e-9)
    rows = []
    for site, block in d.groupby('LOC'):
        n, s = counts.copy(), sums.copy()
        n[block.i] -= 1
        s[block.i] -= block.z.to_numpy()
        y = np.divide(s, n, out=np.full(len(frame), np.nan), where=n >= 2)
        stats = paired_stats(y, frame.p.to_numpy(), frame.q.to_numpy(), frame.a.to_numpy(), frame.b.to_numpy())
        rows.append(dict(site=site, scored=int(np.isfinite(y).sum()), **dict(zip(STAT_NAMES, stats))))
    return pd.DataFrame(rows)


def roster_sensitivity(frame, roster, repeats=100, unit='line'):
    # Simulate incomplete competitor information while retaining each line's own
    # planned sites. Costs, candidate eligibility and plot budget stay fixed.
    # Site references use only retained competitors, with full-roster fallback
    # if no competitors remain. No outcomes enter this calculation.
    schedule = roster[[ID, 'LOC']].drop_duplicates().merge(frame[[ID, 'raw_model_score']], on=ID, validate='many_to_one')
    base = adjust(roster, frame, frame.raw_model_score.to_numpy(), 'site')
    np.testing.assert_allclose(base, frame.p, atol=1e-9)
    original = selected_mask(frame, base)
    full_mean = schedule.groupby('LOC').raw_model_score.mean()
    rng = np.random.default_rng(12309)
    rows = []
    for fraction in [.05, .1, .2]:
        for repetition in range(repeats):
            if unit == 'line':
                keep = frame.loc[rng.random(len(frame)) >= fraction, ID]
            elif unit == 'family':
                families = frame.population.unique()
                retained = families[rng.random(len(families)) >= fraction]
                keep = frame.loc[frame.population.isin(retained), ID]
            else:
                raise ValueError(unit)
            reference = schedule.loc[schedule[ID].isin(keep)].groupby('LOC').raw_model_score.mean().reindex(full_mean.index).fillna(full_mean)
            residual = schedule.raw_model_score - schedule.LOC.map(reference)
            p = residual.groupby(schedule[ID]).mean().reindex(frame[ID]).to_numpy()
            chosen = selected_mask(frame, p)
            rows.append(dict(omitted_fraction=fraction, repetition=repetition,
                             rank_spearman=float(spearmanr(base, p).statistic),
                             selected_retention=float((chosen & original).sum()/original.sum()),
                             score_rmse=float(np.sqrt(np.mean((p-base)**2)))))
    return pd.DataFrame(rows)


def interval_groups(frame):
    covered = frame.y.between(frame.lower_90, frame.upper_90)
    finite = frame.y.notna()
    masks = {'all': finite, 'advanced': finite & frame.a,
             'any_novel_site': finite & frame.n_novel_sites.gt(0),
             'all_sites_seen': finite & frame.n_novel_sites.eq(0),
             'planned_sites_1_3': finite & frame.n_sites.le(3),
             'planned_sites_4_6': finite & frame.n_sites.between(4, 6),
             'planned_sites_7_plus': finite & frame.n_sites.ge(7)}
    return [dict(group=k, n=int(m.sum()), coverage=float(covered[m].mean()),
                 width=float((frame.upper_90-frame.lower_90)[m].mean()),
                 mean_error=float((frame.y-frame.p)[m].mean())) for k, m in masks.items() if m.any()]


def family_diagnostics(frame):
    d = frame.loc[frame.y.notna()].copy()
    family_y = d.groupby('population').y.transform('mean')
    within_y = d.y-family_y
    within_p = d.p-d.groupby('population').p.transform('mean')
    within_q = d.q-d.groupby('population').q.transform('mean')
    return dict(families=int(frame.population.nunique()),
                advanced_families=int(frame.loc[frame.a, 'population'].nunique()),
                within_family_correlation_selected=float(within_y.corr(within_p)),
                within_family_correlation_control=float(within_y.corr(within_q)),
                selected_gain_family_component=float(family_y[d.a].mean()-family_y.mean()),
                selected_gain_within_family_component=float(within_y[d.a].mean()-within_y.mean()))


def run(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new, empty output folder')
    out.mkdir(parents=True, exist_ok=True)
    (out/'.gitignore').write_text('*\n')
    protocol = dict(status='Exploratory robustness audit; previously inspected years remain inspected',
                    outer_years=[2006, 2007], inner_years='2004 through outer year minus one',
                    library='Existing 11 models x 3 adjustments; no new model search',
                    selection='Existing mean gain over random rule, >=2 positive years; earlier-only slope',
                    uncertainty='2000 paired whole-family bootstrap draws conditional on sites, fitted models and allocations; separate leave-one-observed-site-out target sensitivity',
                    roster='100 draws each at 5%, 10%, 20% missing competitor lines or whole families; fixed own sites, costs and budget',
                    intervals='Audit existing 2008 intervals by pre-outcome groups; 2007 earlier-only residual calibration',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'protocol.json', protocol)
    comparison = Path('outputs/model_comparison')
    verified = Path('outputs/selected_forecast_verified')
    inputs = {}
    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    historical_metrics = pd.read_json(comparison/'roster_development_metrics.json')
    inputs[str(comparison/'roster_development_metrics.json')] = hashlib.sha256((comparison/'roster_development_metrics.json').read_bytes()).hexdigest()
    nested, summary, groups = [], {}, []
    for c in [1, 2]:
        key = f'C{c}'
        print(f'{key}: chronological selection and conditional uncertainty', flush=True)
        frames = {y: read(comparison/f'{key}_roster_development_{y}.csv') for y in [2004, 2005, 2006]}
        for year in [2006, 2007]:
            winner, slope, years = select_before(historical_metrics.loc[historical_metrics.cluster.eq(c)], frames, year)
            write_json(out/f'{key}_choice_before_{year}.json', dict(model=winner, slope=slope, years=years))
            if year == 2006:
                held = frames[year]
                p, q = slope * held[winner].to_numpy(), held['control__raw'].to_numpy()
            else:
                # Check cached 2007 predictions correspond to the newly chosen
                # historical policy; fail instead of silently using another model.
                import json
                old = json.loads((comparison/'roster_frozen_choices.json').read_text())[key]
                assert winner == old['model'] and abs(slope-old['slope']) < 1e-10
                held = read(comparison/f'{key}_roster_rankings_2007.csv')
                truth = read(comparison/f'{key}_roster_scored_2007.csv')[[ID, 'y']]
                control = read(comparison/f'{key}_confirmation_2007.csv')[[ID, 'control_raw']]
                held = held.merge(truth, on=ID, how='left', validate='one_to_one').merge(control, on=ID, validate='one_to_one')
                p, q = held.predicted_yield_advantage.to_numpy(), held.control_raw.to_numpy()
            truth = held[[ID, 'y']].dropna()
            errors = np.concatenate([(f.y-slope*f[winner]).dropna().to_numpy() for y, f in frames.items() if y < year])
            lo, hi = np.quantile(errors, [.05, .95])
            mask = held.y.notna()
            nested.append(dict(cluster=c, year=year, model=winner, slope=slope, selection_years=years,
                               selected=metrics(held[[ID, 'n_sites']], p, truth), control=metrics(held[[ID, 'n_sites']], q, truth),
                               earlier_residual_interval_coverage=float(held.y[mask].between(p[mask]+lo, p[mask]+hi).mean())))
        frame = read(verified/f'{key}_rankings.csv').rename(columns={'predicted_yield_advantage': 'p', 'advance': 'a'})
        truth = read(verified/f'{key}_retrospective.csv')[[ID, 'y']]
        control = read(Path('outputs/forecast_2008')/f'{key}_rankings.csv')[[ID, 'predicted_yield_advantage', 'advance']].rename(columns={'predicted_yield_advantage': 'q', 'advance': 'b'})
        frame = frame.merge(truth, on=ID, how='left', validate='one_to_one').merge(control, on=ID, validate='one_to_one')
        summary[key] = dict(family_bootstrap=family_bootstrap(frame), family_diagnostics=family_diagnostics(frame))
        groups.extend([dict(cluster=c, **r) for r in interval_groups(frame)])
        roster = read(verified/f'{key}_candidate_roster.csv')
        sensitivity = roster_sensitivity(frame, roster)
        sensitivity.to_csv(out/f'{key}_roster_sensitivity.csv', index=False)
        summary[key]['roster_sensitivity'] = {str(f): dict(retention_median=d.selected_retention.median(), retention_min=d.selected_retention.min(), rank_correlation_min=d.rank_spearman.min(), score_rmse_median=d.score_rmse.median()) for f,d in sensitivity.groupby('omitted_fraction')}
        family_sensitivity = roster_sensitivity(frame, roster, unit='family')
        family_sensitivity.to_csv(out/f'{key}_family_roster_sensitivity.csv', index=False)
        summary[key]['family_roster_sensitivity'] = {str(f): dict(retention_median=d.selected_retention.median(), retention_min=d.selected_retention.min(), rank_correlation_min=d.rank_spearman.min(), score_rmse_median=d.score_rmse.median()) for f,d in family_sensitivity.groupby('omitted_fraction')}
        print(f'{key}: independently rebuilding truth and deleting each observed site', flush=True)
        path = Path('data/raw')/f'{key}_Phenotype_Data_V2.csv'
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = pd.read_csv(path, usecols=META+[YIELD], low_memory=False)
        observed, _ = normalize(raw.loc[raw[YEAR].eq(2008)])
        deletion = site_deletion(frame, observed)
        deletion.to_csv(out/f'{key}_site_deletion.csv', index=False)
        summary[key]['site_deletion_ranges'] = {name: [deletion[name].min(), deletion[name].max()] for name in STAT_NAMES}
    write_json(out/'nested_checks.json', nested)
    write_json(out/'summary.json', summary)
    pd.DataFrame(groups).to_csv(out/'interval_coverage.csv', index=False)
    write_json(out/'input_hashes.json', inputs)
    print('Robustness audit complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/robustness'))
    run(parser.parse_args().output_dir)
