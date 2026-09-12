import sys
import tempfile
import contextlib
import io
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
for folder in ('src', 'experiments'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
from model_experiments import allocation_mask, design, metrics, random_reference, ridge_path, predictions_for_origin
from demo_data import generate
from triplex import META, YEAR, family, load_genotypes, read_inputs
from select_model import choose, fit_calibration
from roster_experiment import adjust
from selected_model import fit_selected, predict_state
from run_selected_pipeline import run as run_selected


class ExperimentTests(unittest.TestCase):
    def test_selected_pipeline_is_invariant_to_future_yields_and_traits(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / 'raw'
            generate(root)
            for path in (root/'genotypes/C1').glob('*.csv'):
                progeny = pd.read_csv(path, index_col=0)
                parents = progeny.iloc[:2].copy()
                parents.index = ['PIDtestA', 'PIDtestB']
                pd.concat([parents, progeny]).to_csv(path)
            args = SimpleNamespace(data_dir=root, output_dir=base/'first',
                                   policy=Path(__file__).resolve().parents[1]/'model_configs/selected.json',
                                   clusters=[1], candidates=None, plot_budget=None, replicates=1, evaluate=True)
            args.output_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                run_selected(args)
            expected = pd.read_csv(args.output_dir/'C1_rankings.csv')
            text = pd.read_csv(root/'C1_Phenotype_Data_V2.csv', dtype=str, keep_default_na=False)
            target = text.YEAR_x.eq('2008')
            args.candidates = base/'roster.csv'
            text.loc[target,['LINE_UNIQUE_ID','LOC','CROSS']].to_csv(args.candidates,index=False)
            text.loc[target,'YLD_BE'] = ''
            text.loc[target,'MST'] = '999999'
            text['future_weather'] = '999999'
            text.to_csv(root/'C1_Phenotype_Data_V2.csv',index=False)
            args.output_dir, args.evaluate = base/'second', False
            args.output_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                run_selected(args)
            pd.testing.assert_frame_equal(expected, pd.read_csv(args.output_dir/'C1_rankings.csv'), check_exact=True)
            self.assertFalse((args.output_dir/'C1_retrospective.csv').exists())

    def test_roster_adjustment_uses_plans_only_and_handles_duplicate_metadata(self):
        candidates = pd.DataFrame({'LINE_UNIQUE_ID': ['C1.1.1', 'C1.1.2', 'C1.2.1'], 'n_sites': [2, 1, 1]})
        metadata = pd.DataFrame({'LINE_UNIQUE_ID': ['C1.1.1', 'C1.1.1', 'C1.1.2', 'C1.2.1'],
                                 'LOC': ['A', 'B', 'A', 'B'], 'YLD_BE': [1, 2, 3, 4]})
        predictions = np.array([4., 2., 0.])
        result = adjust(metadata, candidates, predictions, 'site')
        np.testing.assert_allclose(result, [1.5, -1, -2])
        changed = pd.concat([metadata.assign(YLD_BE=999999)]*2)
        np.testing.assert_array_equal(result, adjust(changed, candidates, predictions, 'site'))
        self.assertAlmostEqual(np.average(result, weights=candidates.n_sites), 0.)
        centered = adjust(metadata, candidates, predictions, 'cohort')
        self.assertAlmostEqual(np.average(centered, weights=candidates.n_sites), 0.)

    def test_parent_and_blend_predictions_ignore_future_outcomes_and_other_family_dna(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            generate(root)
            geno = load_genotypes(root, 1, Path(folder) / 'cache')
            history, candidates, _ = read_inputs(root, 1, 2001, 2008)
            values = geno.values.astype(float)
            values[values == 2] = np.nan
            parents = pd.DataFrame(values, index=[family(x) for x in geno.lines], columns=geno.markers).groupby(level=0).mean()
            original, predictions = predictions_for_origin(history, candidates[META], geno, parents)
            for model in ['parents_10000', 'flat_3000000']:
                selected_candidates, actual, state = fit_selected(history, candidates[META], geno, parents, model)
                np.testing.assert_allclose(actual, predictions[model], atol=1e-8)
                path = Path(folder) / 'saved_model.npz'
                np.savez_compressed(path, **state)
                with np.load(path, allow_pickle=False) as saved:
                    loaded = {key: saved[key] for key in saved.files}
                x, _ = geno.rows(selected_candidates.LINE_UNIQUE_ID)
                px = parents.reindex(selected_candidates.population).to_numpy()
                np.testing.assert_allclose(predict_state(loaded, x, px), actual, atol=1e-10)
            mutated = candidates.assign(YLD_BE=999999, MST=-999999)
            changed = parents.copy()
            target_family = original.population.iloc[0]
            changed.loc[target_family] = 1.
            after, new = predictions_for_origin(history, mutated, geno, changed)
            self.assertEqual(len(predictions), 11)
            self.assertEqual(original.LINE_UNIQUE_ID.tolist(), after.LINE_UNIQUE_ID.tolist())
            unaffected = original.population.ne(target_family)
            for name in predictions:
                np.testing.assert_allclose(predictions[name][unaffected], new[name][unaffected], atol=1e-8)

    def test_selection_requires_multiple_positive_origins(self):
        records = pd.DataFrame({'model': ['spike']*3 + ['steady']*3,
                                'gain_over_random_mean': [100, -1, -1, 1, 2, 3],
                                'rmse': [10]*6})
        winner, _ = choose(records)
        self.assertEqual(winner, 'steady')

    def test_calibration_uses_equal_origin_weights_and_cannot_reverse_ranks(self):
        a = pd.DataFrame({'model': [1., 2., 3.], 'y': [2., 4., 6.]})
        b = pd.DataFrame({'model': [1., 2., 3.], 'y': [1., 2., 3.]})
        expected = fit_calibration([a, b], 'model')
        duplicated = fit_calibration([pd.concat([a]*10), b], 'model')
        self.assertAlmostEqual(expected['slope'], duplicated['slope'])
        self.assertAlmostEqual(expected['intercept'], duplicated['intercept'])
        self.assertEqual(expected['slope'], 1.)
        reverse = a.assign(y=-a.y)
        self.assertEqual(fit_calibration([reverse], 'model')['slope'], 0.)

    def test_ridge_path_matches_independent_estimator(self):
        rng = np.random.default_rng(27)
        x = rng.normal(2, 1, (60, 12))
        y = rng.normal(3, 2, (60, 2))
        test = rng.normal(1, 3, (15, 12))
        predictions, _ = ridge_path(x, y, test, [3., 300.])
        for alpha, values in predictions.items():
            expected = Ridge(alpha=alpha, solver='cholesky').fit(x, y).predict(test)
            np.testing.assert_allclose(values, expected, atol=1e-10)

    def test_filter_imputation_and_scale_ignore_forecast_rows(self):
        train = np.array([[1, np.nan, 1], [-1, np.nan, 1], [0, 1, 1], [1, np.nan, 1.]])
        a, b, info = design(train, np.array([[1, 1, -1.]]), .5, .01)
        a2, b2, info2 = design(train, np.array([[1, -1, 0.]]), .5, .01)
        np.testing.assert_array_equal(info['keep'], [0])
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)
        np.testing.assert_array_equal(info['mean'], info2['mean'])

    def test_random_policy_preserves_budget_and_candidate_missingness(self):
        costs = np.array([5, 3, 2, 8])
        for order in [np.arange(4), np.arange(4)[::-1]]:
            selected = allocation_mask(costs, 7, order)
            self.assertLessEqual(costs[selected].sum(), 7)
        candidates = pd.DataFrame({'LINE_UNIQUE_ID': [f'C1.1.{i}' for i in range(20)], 'n_sites': [3]*20})
        truth = candidates.iloc[:19][['LINE_UNIQUE_ID']].assign(y=np.arange(19))
        a = random_reference(candidates, truth, repetitions=20)
        b = random_reference(candidates, truth, repetitions=20)
        np.testing.assert_array_equal(a, b)
        score = metrics(candidates, np.arange(20), truth, a)
        self.assertEqual(score['candidates'], 20)
        self.assertEqual(score['scored'], 19)
        self.assertEqual(score['plots'], 6)
        empty = random_reference(candidates, truth, repetitions=20, budget=0)
        self.assertTrue(np.isnan(empty).all())
        zero = metrics(candidates, np.arange(20), truth, empty, budget=0)
        self.assertEqual(zero['advanced'], 0)
        self.assertIsNone(zero['random_allocation_percentile'])


if __name__ == '__main__':
    unittest.main()
