"""Regression tests for decision-time availability and commercial output invariants."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
for folder in ('src', 'experiments'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
from demo_data import generate
from run_pipeline import clear_generated_outputs
from triplex import (ID, META, YEAR, Genotypes, MarkerModel, add_intervals, allocate,
                     canonical, evaluate, fit_markers, forecast, line_targets,
                     load_genotypes, normalize, read_inputs, run)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / 'raw'
        generate(cls.root)
        cls.geno = load_genotypes(cls.root, 1, Path(cls.temp.name) / 'cache')
        cls.history, cls.candidates, _ = read_inputs(cls.root, 1, 2001, 2008)
        cls.limits = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()
        cls.temp.cleanup()

    def test_id_aliases_are_bounded_and_bad_candidate_fails(self):
        self.assertEqual(canonical('C1.034.00000325#1'), 'C1.34.325')
        self.assertEqual(canonical('C2.309.102.0'), 'C2.309.102')
        self.assertIsNone(canonical('C1.126.00000DS%130'))
        self.assertIsNone(canonical('C1.1.2unexpected'))
        bad = self.candidates.copy()
        bad.loc[bad.index[0], ID] = 'corrupt'
        with self.assertRaisesRegex(ValueError, 'Unresolvable'):
            normalize(bad, strict=True)

    def test_overwrite_removes_stale_results_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name in ['C1_retrospective_evaluation.csv', 'C1_validation_2004.csv', 'run_manifest.json',
                         'personal_notes.txt', 'line_rankings_full.csv', 'C2_rankings.csv']:
                (path / name).write_text('preserve unless explicitly generated for selected pool')
            clear_generated_outputs(path, [1])
            self.assertEqual({p.name for p in path.iterdir()},
                             {'personal_notes.txt', 'line_rankings_full.csv', 'C2_rankings.csv'})

    def test_invalid_external_roster_id_is_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidates.csv'
            pd.DataFrame({ID: ['broken'], 'LOC': ['S1'], 'CROSS': ['1/2']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unresolvable'):
                read_inputs(self.root, 1, 2001, 2008, path)

    def test_forecast_requires_earlier_years_and_new_families(self):
        with self.assertRaisesRegex(ValueError, 'strictly earlier'):
            forecast(self.history, self.history[META], self.geno, 30000)
        bad = self.candidates.copy()
        bad[ID] = self.history[ID].iloc[0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            forecast(self.history, bad, self.geno, 30000)

    def test_missing_outcomes_do_not_remove_candidates(self):
        pred, _, info = forecast(self.history, self.candidates, self.geno, 30000)
        self.assertEqual(len(pred), 96)
        raw = pd.read_csv(self.root / 'C1_Phenotype_Data_V2.csv')
        truth, _ = line_targets(raw.loc[raw[YEAR].eq(2008)])
        self.assertEqual(len(truth), 92)
        self.assertEqual(info['train_years'], list(range(2001, 2008)))
        self.assertTrue(set(self.candidates[ID]) == set(pred[ID]))

    def test_missing_dna_uses_explicit_fallback(self):
        target_id = self.candidates[ID].iloc[0]
        keep = self.geno.lines != target_id
        reduced = Genotypes(self.geno.lines[keep], self.geno.markers, self.geno.values[keep], {})
        pred, _, info = forecast(self.history, self.candidates, reduced, 30000)
        self.assertEqual(len(pred), 96)
        self.assertEqual(info['candidates_without_dna'], 1)
        row = pred.set_index(ID).loc[target_id]
        self.assertEqual(row.prediction_source, 'phenotypic_fallback')
        self.assertEqual(row.predicted_yield_advantage, row.phenotypic_blup_baseline)

    def test_marker_preprocessing_is_train_only_and_model_roundtrips(self):
        x = np.array([[0, np.nan, 1], [1, np.nan, 1], [-1, np.nan, 1]] * 10, dtype='float32')
        model = fit_markers(x, np.arange(30, dtype='float32'), 10)
        np.testing.assert_array_equal(model.keep, [True, False, False])
        a = model.predict(np.array([[0, -1, 0]], dtype='float32'))
        b = model.predict(np.array([[0, 1, 1]], dtype='float32'))
        np.testing.assert_array_equal(a, b)
        path = Path(self.temp.name) / 'model.npz'
        model.save(path, np.array(['a', 'b', 'c']))
        with np.load(path, allow_pickle=False) as saved:
            loaded = MarkerModel(*(saved[k] for k in ['keep', 'mean', 'scale', 'coef']), float(saved['intercept']))
        np.testing.assert_array_equal(model.predict(x), loaded.predict(x))

    def test_changing_candidate_dna_cannot_change_other_candidates_or_model(self):
        pred, model, _ = forecast(self.history, self.candidates, self.geno, 30000)
        values = self.geno.values.copy()
        target_id = self.candidates[ID].iloc[0]
        values[self.geno.index.get_loc(target_id)] = 2
        mutated = Genotypes(self.geno.lines, self.geno.markers, values, {})
        after, model_after, _ = forecast(self.history, self.candidates, mutated, 30000)
        np.testing.assert_array_equal(model.coef, model_after.coef)
        np.testing.assert_array_equal(model.keep, model_after.keep)
        a = pred.set_index(ID).drop(target_id).predicted_yield_advantage.sort_index()
        b = after.set_index(ID).drop(target_id).predicted_yield_advantage.sort_index()
        # Removing an uninformative row changes the BLAS batch shape; allow only
        # float32 rounding, while fitted parameters above must remain identical.
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        self.assertEqual(after.set_index(ID).loc[target_id, 'prediction_source'], 'phenotypic_fallback')

    def test_allocation_is_budget_feasible_and_zero_is_supported(self):
        pred = pd.DataFrame({'n_sites': [4, 2, 1], 'rank': [1, 2, 3]})
        selected, plan = allocate(pred, 6, 2)
        self.assertEqual(selected.advance.tolist(), [False, True, True])
        self.assertEqual(plan['plots_allocated'], 6)
        zero, plan = allocate(pred, 0)
        self.assertFalse(zero.advance.any())
        self.assertEqual(plan['plots_allocated'], 0)

    def test_intervals_use_prior_errors_and_do_not_invent_fallback_coverage(self):
        pred = pd.DataFrame({'prediction_source': ['marker_ridge', 'phenotypic_fallback'],
                             'predicted_yield_advantage': [4., 1.]})
        errors = pd.DataFrame({'prediction_source': ['marker_ridge'] * 100, 'error': np.arange(100)})
        out = add_intervals(pred, errors)
        self.assertAlmostEqual(out.lower_90.iloc[0], 8.95)
        self.assertAlmostEqual(out.upper_90.iloc[0], 98.05)
        self.assertTrue(np.isnan(out.lower_90.iloc[1]))

    def test_genotype_conflicts_panel_order_and_invalid_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dna = root / 'genotypes/C1'
            dna.mkdir(parents=True)
            a = pd.DataFrame([[0, 1], [1, 1], [-1, 0]], index=['00001', '00001#1', '00002'], columns=['a', 'b'])
            a.to_csv(dna / 'C1.1_Imputed.csv')
            b = pd.DataFrame([[1, -1]], index=['00003'], columns=['b', 'a'])
            b.to_csv(dna / 'C1.2_Imputed.csv')
            result = load_genotypes(root, 1, root / 'cache')
            self.assertEqual(result.audit['conflicting_ids'], ['C1.1.1'])
            self.assertNotIn('C1.1.1', result.lines)
            x, _ = result.rows(['C1.2.3'])
            np.testing.assert_array_equal(x, [[-1, 1]])
            cached = load_genotypes(root, 1, root / 'cache')
            np.testing.assert_array_equal(result.values, cached.values)
            b.iloc[0, 0] = 9
            b.to_csv(dna / 'C1.2_Imputed.csv')
            with self.assertRaisesRegex(ValueError, 'Invalid genotype'):
                load_genotypes(root, 1, root / 'cache')

    def test_constant_baseline_correlation_is_undefined(self):
        pred, _, _ = forecast(self.history, self.candidates, self.geno, 30000)
        pred, _ = allocate(pred)
        pred = add_intervals(pred, pd.DataFrame(columns=['prediction_source', 'error']))
        raw = pd.read_csv(self.root / 'C1_Phenotype_Data_V2.csv')
        truth, _ = line_targets(raw.loc[raw[YEAR].eq(2008)])
        metrics, _ = evaluate(pred, truth)
        self.assertIsNone(metrics['phenotypic_blup_baseline']['pearson'])
        self.assertIsNone(metrics['environment_mean_baseline']['pearson'])

    def test_full_workflow_invariant_to_future_labels_traits_and_external_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'raw'
            generate(data)
            args = SimpleNamespace(data_dir=data, output_dir=root / 'first', clusters=[1],
                                   start_year=2001, target_year=2008, validation_years=[2005, 2006, 2007],
                                   alpha=30000., plot_budget=None, replicates=1, candidates=None,
                                   predict_only=False, sample=True)
            args.output_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            original = pd.read_csv(args.output_dir / 'C1_rankings.csv')
            report = json.loads((args.output_dir / 'C1_report.json').read_text())
            self.assertEqual(report['validation'][0]['metrics']['interval_lines_scored'], 0)
            self.assertGreater(report['validation'][1]['metrics']['interval_lines_scored'], 0)
            raw = pd.read_csv(data / 'C1_Phenotype_Data_V2.csv')
            mask = raw[YEAR].eq(2008)
            external = root / 'candidates.csv'
            raw.loc[mask, [ID, 'LOC', 'CROSS']].to_csv(external, index=False)
            # Preserve historical decimal strings exactly: a pandas float roundtrip
            # would change historical inputs as well as the intended future values.
            text = pd.read_csv(data / 'C1_Phenotype_Data_V2.csv', dtype=str, keep_default_na=False)
            mask = text[YEAR].eq('2008')
            text.loc[mask, 'YLD_BE'] = ''
            text.loc[mask, 'MST'] = '999999'
            text['X2008_weather'] = '999999'
            text.to_csv(data / 'C1_Phenotype_Data_V2.csv', index=False)
            args.output_dir = root / 'after'
            args.output_dir.mkdir()
            args.candidates = external
            args.predict_only = True
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            after = pd.read_csv(args.output_dir / 'C1_rankings.csv')
            assert_frame_equal(original, after, check_exact=True)
            self.assertFalse((args.output_dir / 'C1_retrospective_evaluation.csv').exists())


if __name__ == '__main__':
    unittest.main()
