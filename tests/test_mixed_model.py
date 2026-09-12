import sys
import unittest
from pathlib import Path
import tempfile
import subprocess
import json
import numpy as np
import pandas as pd
for folder in ('src','experiments','scripts'):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/folder))
from mixed_model import historical_design, genomic_prediction, choose_adjustment
from triplex import ID, YEAR, YIELD, candidate_table, load_genotypes, read_inputs, META
from selected_model import fit_selected
from model_experiments import parent_matrix
from demo_data import generate


class MixedModelTests(unittest.TestCase):
    def test_selection_requires_gain_error_and_optimizer_checks(self):
        baseline=pd.DataFrame({'year':[2004,2005,2006],'allocation_gain':[2.,2.,2.],'rmse':[10.,10.,10.]})
        rows=pd.DataFrame([dict(year=y,model=m,allocation_gain=g,rmse=e,optimizer_code=o)
              for y in [2004,2005,2006] for m,g,e,o in [('good',3.,9.,0),('bad_error',8.,11.,0),('failed',9.,8.,1)]])
        self.assertEqual(choose_adjustment(rows,baseline)[0],'good')
        rows.loc[rows.model.eq('good'),'allocation_gain']=1.
        self.assertEqual(choose_adjustment(rows,baseline)[0],'current_selected')
        with self.assertRaises(ValueError):
            choose_adjustment(pd.concat([rows,rows.iloc[[0]].assign(year=2007)]),baseline)
        with self.assertRaises(ValueError):
            choose_adjustment(rows.iloc[1:],baseline)

    @unittest.skipUnless(Path('C:/3ps/R-4.4.2/bin/Rscript.exe').exists(), 'Optional R integration test')
    def test_r_adjustment_retains_within_family_signal_on_balanced_sites(self):
        rng=np.random.default_rng(91)
        records=[]
        for fam in range(12):
            year=2001+fam//6
            for line in range(5):
                genetic=float(rng.normal(0,4))
                for site in range(3):
                    records.append(dict(line=f'C1.{fam}.{line}', population=str(fam), year=year,
                        generation='F2' if fam%2 else 'BC', environment=f'{year}/{site}',
                        tester=str(fam%3), **{'yield':140+5*fam+site*15+genetic+rng.normal(0,1)}))
        d=pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            source=folder/'history.csv'
            prefix=folder/'fit'
            d.to_csv(source,index=False)
            result=subprocess.run(['C:/3ps/R-4.4.2/bin/Rscript.exe','experiments/trial_adjustment.R',
                       str(source),str(prefix),'2003'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            raw=d.groupby('line')['yield'].mean()
            for variant in ['joint','joint_tester']:
                adjusted=pd.read_csv(str(prefix)+f'_{variant}_targets.csv').set_index(ID).y.reindex(raw.index)
                families=raw.index.str.rsplit('.',n=1).str[0]
                np.testing.assert_allclose(adjusted-adjusted.groupby(families).transform('mean'),
                                          raw-raw.groupby(families).transform('mean'),atol=1e-8)
                fit=json.loads(Path(str(prefix)+f'_{variant}_fit.json').read_text())
                self.assertEqual(fit['max_training_year'],2002)
                self.assertEqual(fit['optimizer_code'],0)

    def test_historical_metadata_boundary_and_unknown_testers(self):
        rows=[{ID:f'C1.{year}.{i}',YEAR:year,'LOC':str(site),'CROSS':'a/b',YIELD:float(i+site)}
              for year in [2001,2002] for i in range(25) for site in range(2)]
        history=pd.DataFrame(rows)
        extra=history[[ID,YEAR]].drop_duplicates().assign(GENERATION_NAME='F2',GERMPLASM_ID_TESTER=np.nan)
        levels, d=historical_design(history,extra,2003)
        self.assertEqual(len(levels),50)
        self.assertEqual(d.tester.nunique(),2)
        future=extra.iloc[[0]].assign(**{YEAR:2003,'GENERATION_NAME':'future','GERMPLASM_ID_TESTER':999})
        _, again=historical_design(history,pd.concat([extra,future]),2003)
        pd.testing.assert_frame_equal(d,again)
        with self.assertRaises(ValueError):
            historical_design(history,extra,2002)

    def test_genomic_stage_matches_existing_selected_model_on_same_targets(self):
        from triplex import line_targets
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw'
            generate(root)
            # Synthetic files normally have no parent rows: create two parents
            # from available marker rows, keeping progeny data intact.
            for path in root.glob('genotypes/*/*.csv'):
                d=pd.read_csv(path,index_col=0)
                parents=d.iloc[:2].copy()
                parents.index=['PID1','PID2']
                pd.concat([parents,d]).to_csv(path)
            for c, model in [(1,'parents_10000'),(2,'flat_3000000')]:
                history,meta,_=read_inputs(root,c,2001,2008)
                geno=load_genotypes(root,c,root.parent/'processed/forecast_cache')
                parents=parent_matrix(root,c,geno)
                candidates=candidate_table(meta)
                expected=fit_selected(history,meta,geno,parents,model)[1]
                actual=genomic_prediction(line_targets(history)[0],candidates,geno,parents,c)
                np.testing.assert_allclose(actual,expected,atol=1e-9)


if __name__=='__main__':
    unittest.main()
