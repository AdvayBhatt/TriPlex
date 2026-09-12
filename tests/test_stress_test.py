import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
for folder in ('src', 'experiments'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
import numpy as np
import pandas as pd
from stress_test import select_before, paired_stats, site_deletion, roster_sensitivity, family_bootstrap, family_diagnostics
from triplex import ID, YIELD
from roster_experiment import adjust


class RobustnessTests(unittest.TestCase):
    def test_bootstrap_serializes_and_gain_components_add(self):
        import json
        frame = pd.DataFrame({'population':['a','a','b','b'], 'y':[0.,2.,4.,6.],
                              'p':[1.,2.,3.,4.], 'q':[0.,1.,0.,1.],
                              'a':[False,True,False,True], 'b':[True,False,True,False]})
        result = family_bootstrap(frame, repeats=20)
        json.dumps(result, allow_nan=False)
        parts = family_diagnostics(frame)
        self.assertAlmostEqual(parts['selected_gain_family_component'] + parts['selected_gain_within_family_component'],
                               result['selected_gain']['estimate'])

    def test_selection_cannot_see_outer_outcomes(self):
        rows = pd.DataFrame([dict(year=y, model=m, gain_over_random_mean=g,
                                  rmse=1.) for y in [2004, 2005, 2006]
                             for m,g in [('a__site', 2.), ('b__site', 1.)]])
        frames = {y: pd.DataFrame({'y':[1.,2.], 'a__site':[1.,2.], 'b__site':[2.,3.]}) for y in [2004,2005,2006]}
        before = select_before(rows, frames, 2006)
        rows.loc[rows.year.eq(2006), 'gain_over_random_mean'] = -1e9
        frames[2006]['y'] = 1e9
        self.assertEqual(before, select_before(rows, frames, 2006))
        self.assertEqual(before[2], [2004,2005])
        with self.assertRaises(ValueError):
            select_before(rows, frames, 2005)

    def test_paired_identical_policies_and_missing_outcomes(self):
        y = np.array([1.,2.,np.nan,4.])
        p = np.array([0.,3.,100.,4.])
        a = np.array([False,True,True,False])
        stats = paired_stats(y,p,p,a,a, np.array([1.,2.,10.,1.]))
        self.assertAlmostEqual(stats[1], 0.)
        self.assertAlmostEqual(stats[2], 0.)
        self.assertAlmostEqual(stats[0], -.25)

    def test_site_deletion_rebuilds_equal_line_site_target(self):
        ids = [f'C1.1.{i}' for i in range(20)]
        observed = pd.DataFrame([{ID:k, 'LOC':str(s), YIELD:float(i+s*10)}
                                 for i,k in enumerate(ids) for s in range(3)])
        # An identical duplicate must not change the line-site target.
        observed = pd.concat([observed, observed.iloc[[0]]], ignore_index=True)
        frame = pd.DataFrame({ID:ids,'y':np.arange(20)-9.5,'p':np.arange(20)-9.5,
                              'q':np.zeros(20),'a':np.arange(20)>=18,'b':np.arange(20)<2})
        result = site_deletion(frame, observed)
        self.assertTrue((result.scored == 20).all())
        np.testing.assert_allclose(result.gain_difference_vs_control, 18.)

    def test_roster_audit_ignores_outcomes(self):
        ids = [f'C1.1.{i}' for i in range(30)]
        frame = pd.DataFrame({ID:ids,'n_sites':3,'raw_model_score':np.arange(30,dtype=float)})
        roster = pd.DataFrame([{ID:k, 'LOC':str(s)} for k in ids for s in range(3)])
        frame['p'] = adjust(roster, frame, frame.raw_model_score.to_numpy(), 'site')
        first = roster_sensitivity(frame, roster, repeats=2)
        frame['y'] = 1e10
        roster[YIELD] = -1e10
        pd.testing.assert_frame_equal(first, roster_sensitivity(frame, roster, repeats=2))


if __name__ == '__main__':
    unittest.main()
