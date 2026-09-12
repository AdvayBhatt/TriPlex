"""Independently check saved forecast metrics and strictly historical R inputs."""
import json
from pathlib import Path
import sys
import hashlib
import zipfile
import numpy as np
import pandas as pd


def verify(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    snapshot=folder/'execution_sources.zip'
    for name,digest in protocol['sources'].items():
        current=Path('experiments')/name
        if hashlib.sha256(current.read_bytes()).hexdigest()!=digest:
            with zipfile.ZipFile(snapshot) as z:
                assert hashlib.sha256(z.read(name)).hexdigest()==digest
            print(f'{name}: matching execution snapshot verified')
    rows=json.loads((folder/'metrics.json').read_text())
    for row in rows:
        c,year,variant=row['cluster'],row['year'],row['model']
        prefix=folder/f'C{c}_{year}'
        historical=pd.read_csv(str(prefix)+'_historical.csv',low_memory=False)
        assert historical.year.max()<year
        assert historical.year.min()>=2001
        assert not historical.duplicated(['line','environment']).any()
        targets=pd.read_csv(str(prefix)+f'_{variant}_targets.csv')
        assert targets.LINE_UNIQUE_ID.is_unique
        assert set(targets.LINE_UNIQUE_ID)==set(historical.line)
        assert np.isfinite(targets.y).all()
        pred=pd.read_csv(str(prefix)+f'_{variant}_predictions.csv')
        assert not set(pred.LINE_UNIQUE_ID)&set(historical.line)
        assert not set(pred.population)&set(historical.population)
        if year<2007:
            held=pd.read_csv(f'outputs/model_comparison/C{c}_roster_development_{year}.csv')
        else:
            held=pd.read_csv(f'outputs/model_comparison/C{c}_roster_rankings_{year}.csv').merge(
                pd.read_csv(f'outputs/model_comparison/C{c}_roster_scored_{year}.csv')[['LINE_UNIQUE_ID','y']],on='LINE_UNIQUE_ID',how='left')
        assert set(pred.LINE_UNIQUE_ID)==set(held.LINE_UNIQUE_ID)
        d=pred.merge(held[['LINE_UNIQUE_ID','y','n_sites']],on='LINE_UNIQUE_ID',suffixes=('','_truth'),validate='one_to_one')
        np.testing.assert_array_equal(d.n_sites,d.n_sites_truth)
        d=d.sort_values(['prediction','LINE_UNIQUE_ID'],ascending=[False,True])
        remaining=int(np.floor(.1*d.n_sites.sum()))
        selected=[]
        for cost in d.n_sites:
            take=cost<=remaining
            selected.append(take)
            if take: remaining-=cost
        finite=d.y.notna()
        measured=dict(pearson=d.loc[finite,'prediction'].corr(d.loc[finite,'y']),
                      rmse=np.sqrt(np.mean((d.loc[finite,'prediction']-d.loc[finite,'y'])**2)),
                      allocation_gain=d.loc[selected,'y'].mean()-d.y.mean(),
                      plots=int(d.loc[selected,'n_sites'].sum()),advanced=int(sum(selected)))
        for name,value in measured.items():
            np.testing.assert_allclose(value,row[name],atol=1e-9,err_msg=f'{c}/{year}/{variant}/{name}')
        fit=json.loads(Path(str(prefix)+f'_{variant}_fit.json').read_text())
        assert fit['max_training_year']==int(historical.year.max())
        assert fit['n']==len(historical)
        assert fit['optimizer_code']==0, fit
    print(f'{len(rows)} forecasts: chronology, candidate identity, target identity, optimizer status, metrics and allocation independently verified')


if __name__=='__main__':
    verify(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/mixed_model'))
