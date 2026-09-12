"""Check exported full-data artifacts independently of the forecast implementation.

Run from the repository root: python tests/verify_full_run.py outputs/forecast_2008
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_sources import verify_source


def verify(folder):
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    args = manifest['parameters']
    root = Path(args['data_dir'])
    for name, signature in manifest['source_sha256'].items():
        verify_source(name, signature)
    checks = {}
    for c in args['clusters']:
        prefix = f'C{c}'
        report = manifest['clusters'][prefix]
        ranks = pd.read_csv(folder / f'{prefix}_rankings.csv')
        raw = pd.read_csv(root / f'{prefix}_Phenotype_Data_V2.csv', usecols=['LINE_UNIQUE_ID', 'YEAR_x', 'LOC'])
        raw = raw.loc[raw.YEAR_x.eq(args['target_year'])].copy()
        def key(text):
            parts = re.fullmatch(r'C([12])\.(\d+)\.(\d+)(?:[.#]\d+)?', text)
            assert parts is not None, text
            return f'C{int(parts[1])}.{int(parts[2])}.{int(parts[3])}'
        raw['LINE_UNIQUE_ID'] = raw.LINE_UNIQUE_ID.map(key)
        ids = set(raw.LINE_UNIQUE_ID)
        assert ranks.LINE_UNIQUE_ID.is_unique
        assert ids == set(ranks.LINE_UNIQUE_ID)
        assert ranks.predicted_yield_advantage.is_monotonic_decreasing
        assert ranks['rank'].tolist() == list(range(1, len(ranks) + 1))
        assert np.isfinite(ranks.predicted_yield_advantage).all()
        schedule = pd.read_csv(folder / f'{prefix}_plot_plan.csv')
        assert not schedule.duplicated(['LINE_UNIQUE_ID', 'LOC']).any()
        assert set(schedule.LINE_UNIQUE_ID) == set(ranks.loc[ranks.advance, 'LINE_UNIQUE_ID'])
        plan = report['allocation']
        assert schedule.plots.sum() == ranks.allocated_plots.sum() == plan['plots_allocated']
        assert plan['plots_allocated'] <= plan['plot_budget']
        assert (schedule.plots == args['replicates']).all()
        available = raw.groupby('LINE_UNIQUE_ID').LOC.nunique()
        assert (ranks.set_index('LINE_UNIQUE_ID').n_sites.sort_index() == available.sort_index()).all()
        remaining = plan['plot_budget']
        for row in ranks.itertuples():
            assert row.advance == (row.plots_if_advanced <= remaining)
            if row.advance:
                remaining -= row.plots_if_advanced
        assert remaining == plan['unused_plots']
        calibration = pd.read_csv(folder / f'{prefix}_calibration.csv')
        assert set(calibration.forecast_year) == set(args['validation_years'])
        assert calibration.forecast_year.max() < args['target_year']
        assert not set(calibration.LINE_UNIQUE_ID) & ids
        for source, selected in ranks.groupby('prediction_source'):
            errors = calibration.loc[calibration.prediction_source.eq(source), 'error']
            assert (selected.calibration_n == len(errors)).all()
            if len(errors) >= 20:
                lo, hi = np.quantile(errors, [.05, .95])
                np.testing.assert_allclose(selected.lower_90, selected.predicted_yield_advantage + lo)
                np.testing.assert_allclose(selected.upper_90, selected.predicted_yield_advantage + hi)
        for fold in report['validation']:
            assert max(fold['fit']['train_years']) < fold['year']
            saved = pd.read_csv(folder / f'{prefix}_validation_{fold["year"]}.csv')
            prior = calibration.loc[calibration.forecast_year < fold['year']]
            for source, selected in saved.groupby('prediction_source'):
                errors = prior.loc[prior.prediction_source.eq(source), 'error']
                if len(errors) >= 20:
                    np.testing.assert_allclose(selected.lower_90,
                                               selected.predicted_yield_advantage + np.quantile(errors, .05))
                else:
                    assert selected.lower_90.isna().all()
        # Rebuild predictions directly from the saved numeric model and cached calls.
        signature = report['genotype_audit']['source_sha256']
        cache = root.parent / f'processed/forecast_cache/{prefix}_{signature}.npz'
        with np.load(cache, allow_pickle=False) as dna, np.load(folder / f'{prefix}_marker_model.npz', allow_pickle=False) as model:
            np.testing.assert_array_equal(dna['markers'], model['markers'])
            position = pd.Index(dna['lines']).get_indexer(ranks.LINE_UNIQUE_ID)
            assert (position >= 0).all(), 'Full supplied 2008 cohort is expected to have DNA'
            x = dna['values'][position][:, model['keep']].astype('float32')
            x[x == 2] = np.nan
            x = np.where(np.isnan(x), model['mean'], x)
            rebuilt = ((x - model['mean']) / model['scale']) @ model['coef'] + model['intercept']
            np.testing.assert_allclose(rebuilt, ranks.predicted_yield_advantage, atol=1e-5, rtol=1e-5)
        truth = pd.read_csv(folder / f'{prefix}_retrospective_evaluation.csv')
        assert set(truth.LINE_UNIQUE_ID) <= ids
        r = np.corrcoef(truth.predicted_yield_advantage, truth.y)[0, 1]
        assert abs(r - report['target_evaluation']['pearson']) < 1e-10
        checks[prefix] = dict(candidates=len(ranks), scored=len(truth), advanced=int(ranks.advance.sum()),
                              plots=int(schedule.plots.sum()), source_hashes_match=True,
                              candidates_complete=True, historical_intervals_verified=True,
                              model_predictions_rebuilt=True, budget_verified=True)
    (folder / 'independent_verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/forecast_2008'))
