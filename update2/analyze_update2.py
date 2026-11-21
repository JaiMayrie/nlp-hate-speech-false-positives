"""Reproducible Update 2 analysis of the saved Update 1 predictions.

No models are trained and no new neural inference is performed by this script.
Inputs are the original results/ CSV files; exports contain no raw text.
Run from the repository root:
    python update2/analyze_update2.py --input-dir results --output-dir update2
Requires Python 3.10+, numpy, and pandas. AI-assisted implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = ['women','trans people','gay people','black people','disabled people','Muslims','immigrants']
FUNCTIONS = ['ident_neutral_nh', 'ident_pos_nh']


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    f1_h=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    f1_n=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return dict(n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tp+tn)/len(y),
                precision_hate=p,recall_hate=r,macro_f1=(f1_h+f1_n)/2,
                fpr=fp/(fp+tn),fnr=fn/(fn+tp))


def run(input_dir: Path, output_dir: Path, seed: int=42, resamples: int=1000) -> dict:
    require(resamples>0, 'Bootstrap resamples must be positive.')
    rd=output_dir/'results'; ed=output_dir/'evidence'
    rd.mkdir(parents=True,exist_ok=True); ed.mkdir(parents=True,exist_ok=True)
    checks=[]; hashes={}; dfs={}; computed=[]
    cols=['id','label','functionality','target_ident','templ_id','hate_score','prediction']
    for condition in ['released','finetuned']:
        path=input_dir/f'dehatebert_{condition}_predictions.csv'
        payload=path.read_bytes(); hashes[path.name]=dict(sha256=hashlib.sha256(payload).hexdigest(),
            git_blob_sha1=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest())
        df=pd.read_csv(path)
        require(set(cols)<=set(df.columns),f'Missing columns: {path}')
        require(df.id.is_unique,f'Duplicate case IDs: {path}')
        require(len(df)==3728,f'Unexpected case count in {path}')
        require(df.label.isin([0,1]).all(),f'Invalid labels: {path}')
        require(df.hate_score.between(0,1).all(),f'Invalid probabilities: {path}')
        require(np.array_equal((df.hate_score>=0.5).astype(int),df.prediction),f'Prediction threshold mismatch: {path}')
        dfs[condition]=df.sort_values('id').reset_index(drop=True)
        m=metrics(df.label.to_numpy(),df.prediction.to_numpy())
        saved=json.loads((input_dir/f'dehatebert_{condition}_metrics.json').read_text())
        for key,value in m.items():
            if key in saved:
                require(np.isclose(value,saved[key],rtol=1e-10,atol=1e-10),f'Metric mismatch: {condition}/{key}')
        computed.append(dict(model='dehatebert',condition=condition,threshold=0.5,**m))
        checks.append(f'{condition}: unique IDs, 3728 cases, legal labels/scores, threshold agreement, and saved overall metrics verified')
    identity=['id','label','functionality','target_ident','templ_id']
    pd.testing.assert_frame_equal(dfs['released'][identity],dfs['finetuned'][identity])
    p=dfs['released'].merge(dfs['finetuned'],on=identity,suffixes=('_released','_finetuned'),validate='one_to_one')
    checks.append('One-to-one paired join preserves all case labels and metadata across the two conditions')
    pd.DataFrame(computed).to_csv(rd/'overall_recheck.csv',index=False)
    trans=p.groupby(['label','prediction_released','prediction_finetuned']).size().reset_index(name='count')
    trans.to_csv(rd/'prediction_transitions.csv',index=False)
    details=[]
    for name,g in p.groupby('functionality',sort=True):
        y=g.label.to_numpy(); a=g.prediction_released.to_numpy(); b=g.prediction_finetuned.to_numpy()
        require(len(set(y))==1,f'Mixed labels for {name}')
        details.append(dict(functionality=name,label=int(y[0]),n=len(g),
          correct_both=int(((a==y)&(b==y)).sum()),wrong_both=int(((a!=y)&(b!=y)).sum()),
          corrected=int(((a!=y)&(b==y)).sum()),new_errors=int(((a==y)&(b!=y)).sum())))
    pd.DataFrame(details).to_csv(rd/'category_transitions.csv',index=False)
    p['score_change']=p.hate_score_finetuned-p.hate_score_released
    p.to_csv(rd/'paired_cases.csv',index=False)
    shifts=[]
    for label,g in p.groupby('label'):
        shifts.append(dict(label=int(label),n=len(g),scores_decreased=int((g.score_change<0).sum()),
           scores_increased=int((g.score_change>0).sum()),scores_equal=int((g.score_change==0).sum()),
           fraction_decreased=float((g.score_change<0).mean()),
           released_mean=float(g.hate_score_released.mean()),finetuned_mean=float(g.hate_score_finetuned.mean()),
           released_median=float(g.hate_score_released.median()),finetuned_median=float(g.hate_score_finetuned.median()),
           median_paired_change=float(g.score_change.median())))
    pd.DataFrame(shifts).to_csv(rd/'score_shift_by_label.csv',index=False)
    subset=p[p.functionality.isin(FUNCTIONS)&(p.label==0)&p.target_ident.isin(GROUPS)].copy()
    keys=['functionality','templ_id']
    coverage=subset.groupby(keys).target_ident.agg(lambda x:set(x))
    keep=coverage[coverage.apply(lambda x:x==set(GROUPS))].index
    subset=subset.set_index(keys).loc[keep].reset_index()
    require(len(subset)>0,'No complete seven-group templates; do not report unmatched group FPRs as matched.')
    cell=subset.groupby(keys+['target_ident'])[['prediction_released','prediction_finetuned']].mean().reset_index()
    mats={c:cell.pivot(index=keys,columns='target_ident',values=f'prediction_{c}').reindex(columns=GROUPS).sort_index()
          for c in ['released','finetuned']}
    require(mats['released'].index.equals(mats['finetuned'].index),'Template indices differ.')
    require(not any(m.isna().any().any() for m in mats.values()),'Missing template-group cells.')
    n=len(mats['released']); rng=np.random.default_rng(seed); draws=rng.integers(0,n,size=(resamples,n))
    boot={c:m.to_numpy()[draws].mean(axis=1) for c,m in mats.items()}
    summary=[]
    for condition,m in mats.items():
        point=m.mean(axis=0).to_numpy(); low,high=np.quantile(boot[condition],[.025,.975],axis=0)
        for i,group in enumerate(GROUPS):
            summary.append(dict(condition=condition,target_ident=group,templates=n,
              cases=int((subset.target_ident==group).sum()),false_positives=int(subset.loc[subset.target_ident==group,f'prediction_{condition}'].sum()),
              fpr=float(point[i]),bootstrap_95_low=float(low[i]),bootstrap_95_high=float(high[i])))
    pd.DataFrame(summary).to_csv(rd/'matched_identity_fpr.csv',index=False)
    gaps=[]
    for c,m in mats.items():
        point=m.mean(axis=0).to_numpy(); samples=boot[c].max(axis=1)-boot[c].min(axis=1)
        low,high=np.quantile(samples,[.025,.975])
        gaps.append(dict(condition=c,templates=n,fpr_max_minus_min=float(point.max()-point.min()),
             bootstrap_95_low=float(low),bootstrap_95_high=float(high)))
    pd.DataFrame(gaps).to_csv(rd/'matched_group_gaps.csv',index=False)
    deltas=[]
    bd=boot['finetuned']-boot['released']; low,high=np.quantile(bd,[.025,.975],axis=0)
    point=(mats['finetuned']-mats['released']).mean(axis=0).to_numpy()
    for i,group in enumerate(GROUPS):
        deltas.append(dict(target_ident=group,templates=n,delta_fpr=float(point[i]),bootstrap_95_low=float(low[i]),bootstrap_95_high=float(high[i])))
    pd.DataFrame(deltas).to_csv(rd/'matched_fpr_changes.csv',index=False)
    subset.to_csv(rd/'matched_template_cases.csv',index=False)
    for c,m in mats.items():
        m.to_csv(rd/f'matched_template_matrix_{c}.csv')
    checks.append(f'Matched subset has {len(subset)} cases across {n} complete seven-group templates; equal template weighting used')
    checks.append('Paired bootstrap uses the same sampled template indices for all identities and both conditions')
    for condition in ['released','finetuned']:
        m=next(x for x in computed if x['condition']==condition)
        require(sum(x['count'] for x in trans.to_dict('records'))==3728,'Transition total incorrect')
        require(m['tn']+m['fp']==1165 and m['fn']+m['tp']==2563,'Class denominators changed')
    checks.append('Class denominators and transition totals reconcile with overall metrics')
    manifest=dict(analysis='Update 2 post-hoc analysis of Update 1 saved predictions; no new training or inference',
        created_utc=datetime.now(timezone.utc).isoformat(),bootstrap_seed=seed,bootstrap_resamples=resamples,
        bootstrap_unit='whole (functionality, templ_id) with all seven identities kept together',
        bootstrap_method='non-stratified percentile intervals; empirical template resampling, not population coverage',
        matched_functions=FUNCTIONS,matched_templates=n,matched_cases=len(subset),
        caveat='All-zero empirical errors give degenerate [0,0] bootstrap intervals; they do not imply zero population risk.',
        environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        input_hashes=hashes,verification_checks=checks,
        unchanged='original predictions, labels, training settings, main threshold and Update 1 model experiment')
    (ed/'analysis_audit.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('\nTransitions:\n'+trans.to_string(index=False))
    print('\n'+json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results'))
    parser.add_argument('--output-dir',type=Path,default=Path('update2'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resamples',type=int,default=1000)
    args=parser.parse_args()
    run(args.input_dir,args.output_dir,args.seed,args.resamples)
