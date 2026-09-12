"""Independent numeric reconstruction of the selected models and planned-site adjustment."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from artifact_sources import verify_source


def verify(folder=Path('outputs/selected_forecast')):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for file, digest in manifest['sources'].items():
        verify_source(file, digest)
    results = {}
    for key, report in manifest['clusters'].items():
        rankings = pd.read_csv(folder / f'{key}_rankings.csv')
        reference = pd.read_csv(f'outputs/forecast_2008/{key}_rankings.csv')
        assert set(rankings.LINE_UNIQUE_ID) == set(reference.LINE_UNIQUE_ID)
        assert rankings.LINE_UNIQUE_ID.is_unique
        assert rankings.predicted_yield_advantage.is_monotonic_decreasing
        signature = report['genotype_audit']['source_sha256']
        cache = Path(manifest['parameters']['data_dir']).parent / 'processed/forecast_cache'
        with np.load(folder / f'{key}_model.npz', allow_pickle=False) as model, np.load(cache/f'{key}_{signature}.npz', allow_pickle=False) as dna:
            pos = pd.Index(dna['lines']).get_indexer(rankings.LINE_UNIQUE_ID)
            assert (pos >= 0).all()
            np.testing.assert_array_equal(dna['markers'], model['markers'])
            raw = dna['values'][pos].astype(float)
            raw[raw == 2] = np.nan
            x = raw[:, model['snp_keep']]
            x = (np.where(np.isnan(x), model['snp_mean'], x)-model['snp_mean'])/model['snp_scale']
            marker = x @ model['snp_coef'].T + model['snp_intercept']
            predicted = marker[:, 0]
            if str(model['kind']) == 'parents':
                with np.load(cache / f'parents_{key}_{signature}.npz', allow_pickle=False) as parents:
                    parent_pos = pd.Index(parents['families']).get_indexer(rankings.population)
                    assert (parent_pos >= 0).all()
                    px = parents['values'][parent_pos][:, model['parent_keep']]
                    px = (np.where(np.isnan(px), model['parent_mean'], px)-model['parent_mean'])/model['parent_scale']
                    predicted = px @ model['parent_coef'] + model['parent_intercept'] + marker[:, 1]
        np.testing.assert_allclose(predicted, rankings.raw_model_score, atol=1e-9)
        roster = pd.read_csv(folder / f'{key}_candidate_roster.csv')
        schedule = roster[['LINE_UNIQUE_ID','LOC']].drop_duplicates().merge(
            rankings[['LINE_UNIQUE_ID']].assign(raw=predicted), on='LINE_UNIQUE_ID', validate='many_to_one')
        schedule['relative'] = schedule.raw-schedule.groupby('LOC').raw.transform('mean')
        expected = schedule.groupby('LINE_UNIQUE_ID').relative.mean().reindex(rankings.LINE_UNIQUE_ID).to_numpy()
        expected *= report['model']['slope']
        np.testing.assert_allclose(expected, rankings.predicted_yield_advantage, atol=1e-9)
        assert abs(np.average(expected, weights=rankings.n_sites)) < 1e-8
        errors = pd.read_csv(folder / f'{key}_calibration.csv').error
        np.testing.assert_allclose(rankings.lower_90, expected+np.quantile(errors,.05), atol=1e-9)
        np.testing.assert_allclose(rankings.upper_90, expected+np.quantile(errors,.95), atol=1e-9)
        plan = pd.read_csv(folder / f'{key}_plot_plan.csv')
        assert set(plan.LINE_UNIQUE_ID) == set(rankings.loc[rankings.advance,'LINE_UNIQUE_ID'])
        assert plan.plots.sum() == rankings.allocated_plots.sum() == report['allocation']['plots_allocated']
        assert plan.plots.sum() <= report['allocation']['plot_budget']
        assert not plan.duplicated(['LINE_UNIQUE_ID','LOC']).any()
        # Separate implementation of the estimator must reproduce the earlier experiment.
        experiment = pd.read_csv(f'outputs/model_comparison/{key}_roster_rankings_2008.csv').set_index('LINE_UNIQUE_ID')
        ranked = rankings.set_index('LINE_UNIQUE_ID')
        np.testing.assert_allclose(ranked.predicted_yield_advantage.sort_index(),
                                   experiment.predicted_yield_advantage.sort_index(), atol=1e-7)
        assert (ranked.advance.sort_index() == experiment.advance.sort_index()).all()
        results[key] = dict(candidates=len(rankings), saved_model_rebuilt=True, roster_adjustment_rebuilt=True,
                            intervals_verified=True, budgets_verified=True, experiment_reproduced=True)
    (folder/'independent_verification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/selected_forecast_verified'))
