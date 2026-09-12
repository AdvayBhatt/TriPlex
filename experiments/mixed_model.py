"""Compare historical mixed-model targets without changing the production policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from model_experiments import design, parent_matrix, metrics
from roster_experiment import adjust
from triplex import ID, META, YEAR, YIELD, family, line_targets, load_genotypes, read_inputs, write_json, candidate_table


def historical_design(history, extra, year):
    if history[YEAR].max() >= year:
        raise ValueError('Historical adjustment cannot include the forecast year')
    levels, plots = line_targets(history)
    attrs = extra.loc[extra[YEAR].lt(year)].copy()
    for column in ['GENERATION_NAME','GERMPLASM_ID_TESTER']:
        if attrs.groupby(ID)[column].nunique().max() > 1:
            raise ValueError('Conflicting historical metadata: '+column)
    attrs = attrs.groupby(ID)[['GENERATION_NAME','GERMPLASM_ID_TESTER']].first()
    d = plots.merge(attrs, on=ID, validate='many_to_one')
    d['population'] = d[ID].map(family)
    d['tester'] = d.GERMPLASM_ID_TESTER.astype('string').fillna('unknown_'+d.population)
    d['generation'] = d.GENERATION_NAME.fillna('unknown')
    d['environment'] = d[YEAR].astype(str)+'/'+d.LOC.astype(str)
    d = d.rename(columns={ID:'line',YIELD:'yield',YEAR:'year'})
    return levels, d[['line','yield','year','generation','environment','population','tester']]


def genomic_prediction(levels, candidates, geno, parents, c):
    raw, available = geno.rows(levels[ID])
    levels = levels.loc[available].reset_index(drop=True)
    test, has_test = geno.rows(candidates[ID])
    if not has_test.all():
        raise ValueError('This experiment requires DNA for all candidates')
    x, tx, _ = design(raw, test, .5, .01)
    if c == 2:
        return Ridge(alpha=3000000., solver='cholesky').fit(x,levels.y).predict(tx)
    labels = levels[ID].map(family)
    means = levels.groupby(labels).y.mean()
    fit = Ridge(alpha=30000.,solver='cholesky').fit(x,np.column_stack([levels.y,levels.y-labels.map(means)]))
    flat, deviation = fit.predict(tx).T
    families = means.index.intersection(parents.index)
    px, pt, _ = design(parents.reindex(families).to_numpy(),parents.reindex(candidates.population).to_numpy(),1.,0.)
    pf = Ridge(alpha=10000.,solver='cholesky').fit(px,means.reindex(families)).predict(pt)
    missing = ~np.isfinite(parents.reindex(candidates.population).to_numpy()).any(axis=1)
    return np.where(missing,flat,pf+deviation)


def choose_adjustment(rows, baseline):
    if set(rows.year) != {2004,2005,2006} or set(baseline.year) != {2004,2005,2006}:
        raise ValueError('Selection requires exactly the three development origins')
    if rows.duplicated(['model','year']).any() or not rows.groupby('model').year.nunique().eq(3).all():
        raise ValueError('Every candidate must have exactly one result per development year')
    summary=rows.groupby('model').agg(gain=('allocation_gain','mean'),rmse=('rmse','mean'),
         positive=('allocation_gain',lambda x:int((x>0).sum())),
         optimizer_failures=('optimizer_code',lambda x:int((x!=0).sum()))).reset_index()
    eligible=summary.loc[(summary.positive>=2)&summary.optimizer_failures.eq(0)&
                         (summary.rmse<=baseline.rmse.mean())&
                         (summary.gain>baseline.allocation_gain.mean())]
    winner=eligible.sort_values(['gain','rmse','model'],ascending=[False,True,True]).iloc[0].model if len(eligible) else 'current_selected'
    return winner,summary


def summarize(out, confirmation=None):
    rows=pd.read_json(out/'metrics.json')
    base=pd.read_json('outputs/model_comparison/roster_development_metrics.json')
    choices={}
    for c in sorted(rows.cluster.unique()):
        name='parents_10000__site' if c==1 else 'flat_3000000__site'
        baseline=base.loc[base.cluster.eq(c)&base.model.eq(name)]
        winner,summary=choose_adjustment(rows.loc[rows.cluster.eq(c)],baseline)
        choice=dict(cluster=int(c),development_years=[2004,2005,2006],winner=winner,
             candidates=summary.to_dict('records'),baseline_gain=baseline.allocation_gain.mean(),
             baseline_rmse=baseline.rmse.mean(),confirmation_year=2007,
             adoption_rule='Require 2007 allocation gain greater than current selected, and RMSE no worse; otherwise retain current policy')
        frozen=out/f'C{c}_frozen_before_2007.json'
        if frozen.exists():
            assert json.loads(frozen.read_text())['winner']==winner, 'Existing frozen choice disagrees'
        else:
            # Persist before any requested confirmation files are accessed.
            write_json(frozen,choice)
        choices[f'C{c}']=choice
    if confirmation:
        later=pd.read_json(confirmation/'metrics.json')
        reference=json.loads(Path('outputs/model_comparison/roster_results.json').read_text())
        for key,choice in choices.items():
            subset=later.loc[later.cluster.eq(choice['cluster'])&later.year.eq(2007)&later.model.eq(choice['winner'])]
            if len(subset):
                result=subset.iloc[0].to_dict()
                control=reference[key]['2007']['selected']
                choice['confirmation']=dict(candidate=result,baseline=control,
                    passes=bool(result['optimizer_code']==0 and result['allocation_gain']>control['allocation_gain'] and result['rmse']<=control['rmse']))
    write_json(out/'selection_summary.json',choices)
    print(json.dumps(choices,indent=2),flush=True)


def run(args):
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'.gitignore').write_text('*\n')
    write_json(out/'protocol.json',dict(years=args.years, clusters=args.clusters,
      variants=['joint','joint_tester'], development=[2004,2005,2006], confirmation=2007,
      training='2001 through forecast year minus one; same usable lines/sites as baseline',
      model='year + generation fixed; environment, population, line, population:environment random; tester optional',
      targets='Raw yields minus fitted nuisance effects, averaged by line; retain residual, family and line signal',
      missing_tester='Separate unknown level per family; avoids treating all unknown testers as one known tester',
      genomic='Existing C1 parental+within-family, C2 ridge; existing marker filters/penalties; full roster adjustment',
      evaluation='Unchanged field-relative targets, full candidate roster and 10% plot budget; no 2008 examination',
      rule='Choose by mean historical allocation gain; require positive gain in >=2 origins and mean RMSE no worse than current selected; test frozen winner on 2007',
      caveats='Two-stage nuisance adjustment, not joint genomic REML or identified pure GCA. All historical years previously inspected.',
      sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('trial_adjustment.R')]}))
    results=[]
    for c in args.clusters:
        history, _, _ = read_inputs(Path('data/raw'),c,2001,2008)
        raw = pd.read_csv(f'data/raw/C{c}_Phenotype_Data_V2.csv',usecols=[ID,YEAR,'GENERATION_NAME','GERMPLASM_ID_TESTER'],low_memory=False)
        from triplex import canonical
        raw[ID] = raw[ID].map(canonical)
        raw = raw.loc[raw[ID].notna() & raw[YEAR].between(2001,2007)]
        geno = load_genotypes(Path('data/raw'),c,Path('data/processed/forecast_cache'))
        parents = parent_matrix(Path('data/raw'),c,geno)
        for year in args.years:
            print(f'C{c} forecast {year}: preparing strictly historical adjustment',flush=True)
            training = history.loc[history[YEAR].lt(year)]
            held = history.loc[history[YEAR].eq(year)]
            levels,d = historical_design(training,raw,year)
            prefix = out/f'C{c}_{year}'
            source = out/f'C{c}_{year}_historical.csv'
            d.to_csv(source,index=False)
            subprocess.run([str(args.rscript),str(Path(__file__).with_name('trial_adjustment.R')),str(source),str(prefix),str(year)],check=True)
            candidates = candidate_table(held[META])
            truth = line_targets(held)[0][[ID,'y']]
            for variant in ['joint','joint_tester']:
                adjusted = pd.read_csv(str(prefix)+f'_{variant}_targets.csv')
                assert set(adjusted[ID]) == set(levels[ID])
                p = genomic_prediction(adjusted,candidates,geno,parents,c)
                p = adjust(held[META],candidates,p,'site')
                candidates.assign(prediction=p).to_csv(str(prefix)+f'_{variant}_predictions.csv',index=False)
                report = json.loads(Path(str(prefix)+f'_{variant}_fit.json').read_text())
                results.append(dict(cluster=c,year=year,model=variant,singular=report['singular'],optimizer_code=report['optimizer_code'],**metrics(candidates,p,truth)))
                write_json(out/'metrics.json',results)
                print(json.dumps(results[-1]),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/mixed_model'))
    p.add_argument('--rscript',type=Path,default=Path('C:/3ps/R-4.4.2/bin/Rscript.exe'))
    p.add_argument('--years',type=int,nargs='+',default=[2004,2005,2006],choices=[2004,2005,2006,2007])
    p.add_argument('--clusters',type=int,nargs='+',default=[1,2],choices=[1,2])
    p.add_argument('--summarize',action='store_true',help='Freeze development choices from completed saved results; no refitting')
    p.add_argument('--confirmation-folder',type=Path,help='Compare previously frozen choices with saved 2007 results')
    with threadpool_limits(limits=4):
        args=p.parse_args()
        if args.summarize:
            summarize(args.output_dir,args.confirmation_folder)
        else:
            run(args)
