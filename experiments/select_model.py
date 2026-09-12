"""Freeze development choices, check 2007, then evaluate the adopted choice in 2008."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_experiments import metrics, parent_matrix, predictions_for_origin, random_reference, allocation_mask
from triplex import ID, META, YEAR, YIELD, line_targets, load_genotypes, normalize, read_inputs, write_json


def fit_calibration(frames, model):
    """Equal weight to each historical origin; no confirmation/target outcomes."""
    x, y, weights = [], [], []
    for frame in frames:
        d = frame.loc[frame.y.notna()]
        x.extend(d[model])
        y.extend(d.y)
        weights.extend(np.full(len(d), 1 / len(d)))
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(weights)
    w /= w.sum()
    xm, ym = w @ x, w @ y
    variance = w @ ((x-xm)**2)
    slope = float(np.clip((w @ ((x-xm)*(y-ym))) / variance, 0, 1)) if variance > 1e-12 else 0.
    return dict(slope=slope, intercept=float(ym-slope*xm), years=[2004, 2005, 2006])


def choose(rows):
    summary = rows.groupby('model').agg(mean_gain=('gain_over_random_mean', 'mean'),
                                       positive_years=('gain_over_random_mean', lambda x: int((x > 0).sum())),
                                       mean_rmse=('rmse', 'mean')).reset_index()
    eligible = summary.loc[summary.positive_years >= 2]
    if eligible.empty:
        return 'control', summary
    winner = eligible.sort_values(['mean_gain', 'mean_rmse', 'model'], ascending=[False, True, True]).iloc[0].model
    return winner, summary


def freeze(folder, clusters):
    metrics_frame = pd.read_csv(folder / 'development_metrics.csv')
    choice = dict(development_years=[2004, 2005, 2006], confirmation_year=2007,
                  adoption_rule='Adopt winner only if 2007 allocation gain exceeds zero and the control, and calibrated RMSE <= zero baseline. Otherwise retain control; retain calibration only if 2007 RMSE improves.',
                  clusters={})
    for c in clusters:
        records = metrics_frame.loc[metrics_frame.cluster.eq(c)]
        if set(records.year) != {2004, 2005, 2006}:
            raise ValueError('Incomplete development origins')
        model, summary = choose(records)
        summary.to_csv(folder / f'C{c}_development_summary.csv', index=False)
        frames = [pd.read_csv(folder / f'C{c}_development_{year}.csv') for year in [2004, 2005, 2006]]
        choice['clusters'][f'C{c}'] = dict(model=model, calibration=fit_calibration(frames, model),
                                         control_calibration=fit_calibration(frames, 'control'),
                                         development_files={str(year): hashlib.sha256(
                                             (folder / f'C{c}_development_{year}.csv').read_bytes()).hexdigest()
                                                            for year in [2004, 2005, 2006]})
    write_json(folder / 'frozen_choices.json', choice)
    print(json.dumps(choice, indent=2), flush=True)
    return choice


def calibrate(values, calibration):
    return calibration['intercept'] + calibration['slope'] * values


def run(root, folder, clusters):
    # Persist the choice before loading outcomes from either later season.
    frozen = freeze(folder, clusters)
    results = {}
    for c in clusters:
        choice = frozen['clusters'][f'C{c}']
        history, target_metadata, _ = read_inputs(root, c, 2001, 2008)
        geno = load_genotypes(root, c, root.parent / 'processed/forecast_cache')
        parents = parent_matrix(root, c, geno)
        held = history.loc[history[YEAR].eq(2007)]
        print(f'C{c}: confirmation forecast 2007 for frozen {choice["model"]}', flush=True)
        candidates, predictions = predictions_for_origin(history.loc[history[YEAR] < 2007], held[META], geno, parents)
        truth = line_targets(held)[0]
        random = random_reference(candidates, truth)
        comparison = {}
        outputs = candidates.merge(truth[[ID, 'y']], on=ID, how='left')
        for name, value in [('control_raw', predictions['control']),
                            ('control_calibrated', calibrate(predictions['control'], choice['control_calibration'])),
                            ('selected_raw', predictions[choice['model']]),
                            ('selected_calibrated', calibrate(predictions[choice['model']], choice['calibration']))]:
            comparison[name] = metrics(candidates, value, truth, random)
            outputs[name] = value
        outputs.to_csv(folder / f'C{c}_confirmation_2007.csv', index=False)
        selected, control = comparison['selected_calibrated'], comparison['control_calibrated']
        adopt = (selected['gain_over_random_mean'] > max(0, control['gain_over_random_mean']) and
                 selected['rmse'] <= selected['baseline_rmse'])
        model = choice['model'] if adopt else 'control'
        calibration = choice['calibration'] if adopt else choice['control_calibration']
        if not adopt and control['rmse'] > comparison['control_raw']['rmse']:
            calibration = dict(slope=1., intercept=0.)
        decision = dict(selected_model_confirmed=bool(adopt), adopted_model=model, calibration=calibration,
                        confirmation_metrics=comparison)
        write_json(folder / f'C{c}_adoption_before_2008.json', decision)
        print(json.dumps(decision), flush=True)
        print(f'C{c}: final retrospective 2008 check for {model}', flush=True)
        candidates, predictions = predictions_for_origin(history, target_metadata[META], geno, parents)
        pred = calibrate(predictions[model], calibration)
        rankings = candidates.assign(predicted_yield_advantage=pred, model=model)
        # Use only pre-2008 errors. Refit calibration is not performed on confirmation data.
        # Development errors fitted the calibration. Use 2007 errors for intervals;
        # 2007 also gates adoption, so this is not an untouched coverage guarantee.
        confirmation_prediction = calibrate(predictions_for_saved_confirmation(outputs, choice, model), calibration)
        errors = (outputs.y - confirmation_prediction).dropna().to_numpy()
        lo, hi = np.quantile(errors, [.05, .95])
        rankings['lower_90'] = pred + lo
        rankings['upper_90'] = pred + hi
        rankings['interval_calibration_year'] = 2007
        order = np.lexsort((rankings[ID].to_numpy(), -pred))
        costs = rankings.n_sites.to_numpy(int)
        rankings['advance'] = allocation_mask(costs, int(np.floor(.1*costs.sum())), order)
        rankings['allocated_plots'] = np.where(rankings.advance, costs, 0)
        rankings = rankings.iloc[order].reset_index(drop=True)
        rankings['rank'] = np.arange(1, len(rankings)+1)
        rankings.to_csv(folder / f'C{c}_candidate_rankings.csv', index=False)
        raw = pd.read_csv(root / f'C{c}_Phenotype_Data_V2.csv', usecols=META+[YIELD], low_memory=False)
        target, _ = normalize(raw.loc[raw[YEAR].eq(2008)])
        truth = line_targets(target)[0]
        random = random_reference(candidates, truth)
        target_metrics = dict(adopted=metrics(candidates, pred, truth, random),
                              control_raw=metrics(candidates, predictions['control'], truth, random))
        scored = rankings.merge(truth[[ID, 'y']], on=ID, validate='one_to_one')
        target_metrics['adopted']['interval_90_coverage'] = float(scored.y.between(scored.lower_90, scored.upper_90).mean())
        scored.to_csv(folder / f'C{c}_candidate_retrospective.csv', index=False)
        results[f'C{c}'] = dict(**decision, retrospective_2008=target_metrics)
        write_json(folder / 'comparison_results.json', results)
        print(json.dumps(target_metrics), flush=True)


def predictions_for_saved_confirmation(outputs, choice, model):
    return outputs.selected_raw if model == choice['model'] else outputs.control_raw


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/model_comparison'))
    p.add_argument('--clusters', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    args = p.parse_args()
    with threadpool_limits(limits=4):
        run(args.data_dir, args.output_dir, args.clusters)
