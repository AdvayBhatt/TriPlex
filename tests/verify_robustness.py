"""Independent artifact checks; imports no project model or audit functions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from artifact_sources import verify_source


def verify(folder):
    protocol = json.loads((folder/'protocol.json').read_text())
    verify_source('stress_test.py', protocol['source_sha256'])
    for path, expected in json.loads((folder/'input_hashes.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    summary = json.loads((folder/'summary.json').read_text())
    intervals = pd.read_csv(folder/'interval_coverage.csv')
    for c in [1,2]:
        key = f'C{c}'
        d = pd.read_csv(Path('outputs/selected_forecast_verified')/f'{key}_retrospective.csv')
        reference = pd.read_csv(Path('outputs/forecast_2008')/f'{key}_rankings.csv')
        d = d.merge(reference[['LINE_UNIQUE_ID','predicted_yield_advantage','advance']], on='LINE_UNIQUE_ID', suffixes=('', '_old'), validate='one_to_one')
        d['e2'] = (d.y-d.predicted_yield_advantage)**2
        d['old_e2'] = (d.y-d.predicted_yield_advantage_old)**2
        d['zero_e2'] = d.y**2
        d['selected_y'] = d.y*d.advance
        d['control_y'] = d.y*d.advance_old
        g = d.groupby('population', sort=True).agg(n=('y','size'), y=('y','sum'), a=('advance','sum'),
                 b=('advance_old','sum'), ay=('selected_y','sum'), by=('control_y','sum'),
                 e2=('e2','sum'), old_e2=('old_e2','sum'), zero_e2=('zero_e2','sum'))
        # Sampling all candidate families, including those with no scorable
        # outcomes, matches the audit's population universe.
        ranks = pd.read_csv(Path('outputs/selected_forecast_verified')/f'{key}_rankings.csv')
        g = g.reindex(sorted(ranks.population.unique()), fill_value=0)
        rng = np.random.default_rng(9122026)
        counts = np.stack([np.bincount(rng.integers(len(g), size=len(g)), minlength=len(g)) for _ in range(2000)])
        sums = counts @ g.to_numpy(float)
        n,y,a,b,ay,by,e2,old_e2,zero_e2 = sums.T
        samples = {'selected_gain':ay/a-y/n, 'gain_difference_vs_control':ay/a-by/b,
                   'rmse_difference_vs_control':np.sqrt(e2/n)-np.sqrt(old_e2/n),
                   'rmse_difference_vs_zero':np.sqrt(e2/n)-np.sqrt(zero_e2/n)}
        for name, values in samples.items():
            np.testing.assert_allclose(np.quantile(values,[.025,.975]), summary[key]['family_bootstrap'][name]['percentile_95'], atol=1e-10)
        checks = intervals.loc[intervals.cluster.eq(c)].set_index('group')
        for name, mask in {'all':np.ones(len(d),bool), 'advanced':d.advance,
                            'planned_sites_1_3':d.n_sites.le(3),
                            'all_sites_seen':d.n_novel_sites.eq(0)}.items():
            assert mask.sum() == checks.loc[name,'n']
            np.testing.assert_allclose(d.loc[mask,'y'].between(d.loc[mask,'lower_90'],d.loc[mask,'upper_90']).mean(),checks.loc[name,'coverage'])
        print(f'{key}: independent family-bootstrap and coverage checks passed')
    rows = pd.read_json('outputs/model_comparison/roster_development_metrics.json')
    for result in json.loads((folder/'nested_checks.json').read_text()):
        past = rows.loc[rows.cluster.eq(result['cluster']) & rows.year.lt(result['year'])]
        grouped = past.groupby('model').agg(gain=('gain_over_random_mean','mean'), positive=('gain_over_random_mean',lambda x:(x>0).sum()), rmse=('rmse','mean')).reset_index()
        chosen = grouped.loc[grouped.positive.ge(2)].sort_values(['gain','rmse','model'],ascending=[False,True,True]).iloc[0].model
        assert chosen == result['model']
        assert result['selection_years'] == sorted(past.year.unique())
    print('Input hashes and chronological choices passed')


if __name__ == '__main__':
    verify(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/robustness_final'))
