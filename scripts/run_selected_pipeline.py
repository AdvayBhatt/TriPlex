"""Run the fixed improved models without repeating the model search. Default: predict only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
for folder in ('src', 'experiments'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_experiments import parent_matrix, metrics, random_reference
from roster_experiment import adjust
from selected_model import fit_selected
from triplex import (ID, META, YEAR, YIELD, allocate, line_targets, load_genotypes,
                     normalize, read_inputs, write_json)


def run(args):
    policy = json.loads(args.policy.read_text())
    start, year, calibration_year = policy['start_year'], policy['target_year'], policy['calibration_year']
    if not start < calibration_year < year:
        raise ValueError('Invalid policy chronology')
    manifest = dict(policy=policy, parameters=vars(args), clusters={},
                    sources={name: hashlib.sha256(next(Path(__file__).resolve().parents[1]/folder/name
                                 for folder in ('scripts','src','experiments')
                                 if (Path(__file__).resolve().parents[1]/folder/name).exists()).read_bytes()).hexdigest()
                             for name in ['run_selected_pipeline.py', 'selected_model.py', 'model_experiments.py',
                                          'roster_experiment.py', 'select_model.py', 'triplex.py']})
    for cluster in args.clusters:
        key = f'C{cluster}'
        choice = policy['clusters'][key]
        history, metadata, audit = read_inputs(args.data_dir, cluster, start, year, args.candidates)
        geno = load_genotypes(args.data_dir, cluster, args.data_dir.parent / 'processed/forecast_cache')
        parents = parent_matrix(args.data_dir, cluster, geno)
        print(f'{key}: calibrating from the historical {calibration_year} forecast', flush=True)
        held = history.loc[history[YEAR].eq(calibration_year)]
        candidates, raw, _ = fit_selected(history.loc[history[YEAR] < calibration_year], held[META], geno, parents, choice['model'])
        pred = choice['slope'] * adjust(held[META], candidates, raw, choice['adjustment'])
        calibration = candidates.assign(prediction=pred).merge(line_targets(held)[0][[ID, 'y']], on=ID)
        calibration['error'] = calibration.y-calibration.prediction
        lo, hi = np.quantile(calibration.error, [.05, .95])
        calibration.to_csv(args.output_dir / f'{key}_calibration.csv', index=False)
        print(f'{key}: fitting {choice["model"]} through {year-1}', flush=True)
        candidates, raw, state = fit_selected(history, metadata, geno, parents, choice['model'])
        pred = choice['slope'] * adjust(metadata, candidates, raw, choice['adjustment'])
        rankings = candidates.assign(raw_model_score=raw, predicted_yield_advantage=pred,
                                     lower_90=pred+lo, upper_90=pred+hi, calibration_n=len(calibration))
        novel = metadata.loc[~metadata.LOC.isin(history.LOC)].groupby(ID).LOC.nunique()
        rankings['n_novel_sites'] = rankings[ID].map(novel).fillna(0).astype(int)
        _, has_dna = geno.rows(rankings[ID])
        rankings['needs_review'] = rankings.n_novel_sites.gt(0) | ~has_dna
        rankings = rankings.sort_values(['predicted_yield_advantage', ID], ascending=[False, True]).reset_index(drop=True)
        rankings['rank'] = np.arange(1, len(rankings)+1)
        rankings, allocation = allocate(rankings, args.plot_budget, args.replicates)
        rankings.to_csv(args.output_dir / f'{key}_rankings.csv', index=False)
        roster = metadata[META].drop_duplicates()
        roster.to_csv(args.output_dir / f'{key}_candidate_roster.csv', index=False)
        plan = roster[[ID, 'LOC']].drop_duplicates().merge(rankings.loc[rankings.advance, [ID, 'rank']], on=ID)
        plan['plots'] = args.replicates
        plan.sort_values(['rank', 'LOC']).to_csv(args.output_dir / f'{key}_plot_plan.csv', index=False)
        assert int(plan.plots.sum()) == allocation['plots_allocated']
        np.savez_compressed(args.output_dir / f'{key}_model.npz', **state)
        result = dict(model=choice, input_audit=audit, genotype_audit=geno.audit, allocation=allocation,
                      uncertainty='2007 residuals; empirical prediction intervals, no future coverage guarantee',
                      reference='Full input roster before budget selection', retrospective_evaluation=None)
        if args.evaluate:
            raw_truth = pd.read_csv(args.data_dir / f'{key}_Phenotype_Data_V2.csv', usecols=META+[YIELD], low_memory=False)
            target, _ = normalize(raw_truth.loc[raw_truth[YEAR].eq(year)])
            truth = line_targets(target)[0]
            random = random_reference(candidates, truth, budget=args.plot_budget, replicates=args.replicates)
            measured = metrics(candidates, pred, truth, random, budget=args.plot_budget, replicates=args.replicates)
            scored = rankings.merge(truth[[ID, 'y']], on=ID, validate='one_to_one')
            measured['interval_90_coverage'] = float(scored.y.between(scored.lower_90, scored.upper_90).mean())
            result['retrospective_evaluation'] = measured
            scored.to_csv(args.output_dir / f'{key}_retrospective.csv', index=False)
        manifest['clusters'][key] = result
        write_json(args.output_dir / f'{key}_report.json', result)
        print(f'{key}: {len(rankings)} candidates; {allocation["lines_advanced"]} advanced; {allocation["plots_allocated"]} plots', flush=True)
    write_json(args.output_dir / 'manifest.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/selected_forecast'))
    p.add_argument('--policy', type=Path, default=Path('model_configs/selected.json'))
    p.add_argument('--clusters', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    p.add_argument('--candidates', type=Path, help='Complete planned reference roster: LINE_UNIQUE_ID, LOC, CROSS')
    p.add_argument('--plot-budget', type=int)
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--evaluate', action='store_true', help='Score 2008 retrospectively after saving predictions')
    args = p.parse_args()
    if args.replicates < 1 or (args.plot_budget is not None and args.plot_budget < 0):
        p.error('Replication must be positive and budget nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Choose a new empty output directory to preserve earlier results')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / '.gitignore').write_text('*\n')
    with threadpool_limits(limits=4):
        run(args)
