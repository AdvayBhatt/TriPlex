"""Exploratory second-stage check: align predictions with the fixed field-relative target.

The first model-selection experiment already examined 2008. This follow-up is
explicitly exploratory; choices still use only historical development forecasts.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_experiments import metrics, random_reference, parent_matrix, predictions_for_origin, allocation_mask
from select_model import choose
from triplex import ID, META, YEAR, YIELD, line_targets, load_genotypes, normalize, read_inputs, write_json


def adjust(metadata, candidates, prediction, mode):
    if mode == 'raw':
        return np.asarray(prediction)
    if mode == 'cohort':
        return prediction - np.average(prediction, weights=candidates.n_sites)
    if mode != 'site':
        raise ValueError(mode)
    schedule = metadata[[ID, 'LOC']].drop_duplicates().merge(
        candidates[[ID]].assign(p=prediction), on=ID, validate='many_to_one')
    schedule['adjusted'] = schedule.p - schedule.groupby('LOC').p.transform('mean')
    return schedule.groupby(ID).adjusted.mean().reindex(candidates[ID]).to_numpy()


def slope_calibration(frames, name):
    numerator, denominator = 0., 0.
    for frame in frames:
        d = frame.loc[frame.y.notna()]
        numerator += np.mean(d[name] * d.y)
        denominator += np.mean(d[name] ** 2)
    return float(np.clip(numerator / denominator, 0, 1)) if denominator > 1e-12 else 0.


def develop(root, folder):
    rows, choices = [], {}
    for c in [1, 2]:
        history, _, _ = read_inputs(root, c, 2001, 2008)
        frames = []
        for year in [2004, 2005, 2006]:
            saved = pd.read_csv(folder / f'C{c}_development_{year}.csv')
            candidates = saved[[ID, 'CROSS', 'n_sites', 'population']]
            metadata = history.loc[history[YEAR].eq(year), META]
            truth = saved[[ID, 'y']].dropna()
            random = random_reference(candidates, truth)
            transformed = candidates.merge(truth, on=ID, how='left')
            models = [name for name in saved if name not in [ID, 'CROSS', 'n_sites', 'population', 'y']]
            for model in models:
                for mode in ['raw', 'cohort', 'site']:
                    name = model + '__' + mode
                    values = adjust(metadata, candidates, saved[model].to_numpy(), mode)
                    transformed[name] = values
                    rows.append(dict(cluster=c, year=year, model=name, **metrics(candidates, values, truth, random)))
            transformed.to_csv(folder / f'C{c}_roster_development_{year}.csv', index=False)
            frames.append(transformed)
        records = pd.DataFrame([r for r in rows if r['cluster'] == c])
        # Same primary gain criterion. RMSE breaks raw/cohort ties without consulting later years.
        winner, summary = choose(records)
        summary.to_csv(folder / f'C{c}_roster_summary.csv', index=False)
        choices[f'C{c}'] = dict(model=winner, slope=slope_calibration(frames, winner))
    write_json(folder / 'roster_development_metrics.json', rows)
    write_json(folder / 'roster_frozen_choices.json', choices)
    print(json.dumps(choices, indent=2), flush=True)
    return choices


def evaluate_later(root, folder, choices):
    results = {}
    for c in [1, 2]:
        choice = choices[f'C{c}']
        model, mode = choice['model'].split('__')
        history, target_metadata, _ = read_inputs(root, c, 2001, 2008)
        geno = load_genotypes(root, c, root.parent / 'processed/forecast_cache')
        parents = parent_matrix(root, c, geno)
        results[f'C{c}'] = {}
        confirmation_errors = None
        for year in [2007, 2008]:
            print(f'C{c}: roster-adjusted {year}, frozen model {choice["model"]}', flush=True)
            metadata = history.loc[history[YEAR].eq(year), META] if year == 2007 else target_metadata[META]
            candidates, predictions = predictions_for_origin(history.loc[history[YEAR] < year], metadata, geno, parents)
            values = choice['slope'] * adjust(metadata, candidates, predictions[model], mode)
            rankings = candidates.assign(predicted_yield_advantage=values)
            costs = rankings.n_sites.to_numpy(int)
            order = np.lexsort((rankings[ID].to_numpy(), -values))
            rankings['advance'] = allocation_mask(costs, int(np.floor(.1*costs.sum())), order)
            rankings['allocated_plots'] = np.where(rankings.advance, costs, 0)
            if confirmation_errors is not None:
                lo, hi = np.quantile(confirmation_errors, [.05, .95])
                rankings['lower_90'], rankings['upper_90'] = values + lo, values + hi
            rankings = rankings.iloc[order].reset_index(drop=True)
            rankings['rank'] = np.arange(1, len(rankings)+1)
            rankings.to_csv(folder / f'C{c}_roster_rankings_{year}.csv', index=False)
            # Save predictions before accessing evaluation outcomes.
            raw = pd.read_csv(root / f'C{c}_Phenotype_Data_V2.csv', usecols=META+[YIELD], low_memory=False)
            target, _ = normalize(raw.loc[raw[YEAR].eq(year)])
            truth = line_targets(target)[0]
            random = random_reference(candidates, truth)
            measured = dict(selected=metrics(candidates, values, truth, random),
                            control=metrics(candidates, predictions['control'], truth, random))
            # Fixed secondary comparator: roster centering of the control, no new tuning.
            centered = adjust(metadata, candidates, predictions['control'], 'cohort')
            measured['control_centered'] = metrics(candidates, centered, truth, random)
            scored = rankings.merge(truth[[ID, 'y']], on=ID, validate='one_to_one')
            if year == 2007:
                confirmation_errors = (scored.y - scored.predicted_yield_advantage).to_numpy()
            else:
                measured['selected']['interval_90_coverage'] = float(scored.y.between(scored.lower_90, scored.upper_90).mean())
            scored.to_csv(folder / f'C{c}_roster_scored_{year}.csv', index=False)
            results[f'C{c}'][str(year)] = measured
            write_json(folder / 'roster_results.json', results)
            print(json.dumps(measured), flush=True)


if __name__ == '__main__':
    root, folder = Path('data/raw'), Path('outputs/model_comparison')
    write_json(folder / 'roster_protocol.json', dict(
        status='Exploratory follow-up after the first 2008 evaluation; not an untouched test',
        target='Unchanged field-centered line means', variants=['raw', 'planned-plot-weighted cohort centering', 'planned-site centering'],
        selection_years=[2004, 2005, 2006], later_checks=[2007, 2008],
        rule='Same historical mean allocation gain criterion, >=2 positive years; RMSE tie-break; slope fit on development only',
        caveat='Predictions are relative to the complete supplied planting roster; changing that roster changes the reference'))
    with threadpool_limits(limits=4):
        choices = develop(root, folder)
        evaluate_later(root, folder, choices)
