from pathlib import Path
import ast,json,hashlib,math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
base=Path(__file__).resolve().parents[1]
r=base/'results'
checks=[]
def ok(name,cond):
    if not cond: raise AssertionError(name)
    checks.append(name)
nbp=base/'Jaida_NLP_Update1_FineTuning.ipynb'
nb=json.loads(nbp.read_text())
node=ast.parse(''.join(nb['cells'][3]['source']))
source=ast.literal_eval(node.body[0].value)
ok('Notebook embedded project.py matches exported project.py byte-for-byte',source== (base/'project.py').read_text())
ok('Notebook contains no error outputs',not any(o['output_type']=='error' for c in nb['cells'] for o in c.get('outputs',[])))
ok('First-model and export cells executed; second-model cell unexecuted',nb['cells'][5]['execution_count'] is not None and nb['cells'][6]['execution_count'] is not None and nb['cells'][7]['execution_count'] is None)
frames={}
for condition in ['released','finetuned']:
    p=pd.read_csv(r/f'dehatebert_{condition}_predictions.csv')
    frames[condition]=p
    ok(f'{condition}: 3728 unique case IDs',len(p)==3728 and p.id.is_unique)
    ok(f'{condition}: score range and labels',p.hate_score.between(0,1).all() and set(p.label)=={0,1})
    ok(f'{condition}: stored labels match 0.5 threshold',np.array_equal(p.prediction,(p.hate_score>=.5).astype(int)))
    def calc(q,t):
        y=q.label.to_numpy(); pred=(q.hate_score.to_numpy()>=t).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        return dict(n=len(y),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),precision_hate=precision_score(y,pred,zero_division=0),recall_hate=recall_score(y,pred,zero_division=0),fpr=float(fp/(tn+fp)) if tn+fp else None,fnr=float(fn/(tp+fn)) if tp+fn else None)
    def equal(actual,expected):
        return (expected is None or pd.isna(expected)) if actual is None else math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10)
    expected=json.loads((r/f'dehatebert_{condition}_metrics.json').read_text())
    calc_m=calc(p,.5)
    ok(f'{condition}: every overall metric agrees with prediction rows',all(equal(v,expected[k]) for k,v in calc_m.items()))
    cats=pd.read_csv(r/f'dehatebert_{condition}_categories.csv')
    ok(f'{condition}: all 29 functionality tables agree',len(cats)==29 and all(all(equal(v,getattr(row,k)) for k,v in calc(p[p.functionality==row.functionality],.5).items() if k in cats.columns) for row in cats.itertuples()))
    thresholds=pd.read_csv(r/f'dehatebert_{condition}_thresholds.csv')
    ok(f'{condition}: all nine threshold tables agree',len(thresholds)==9 and all(all(equal(v,getattr(row,k)) for k,v in calc(p,row.threshold).items()) for row in thresholds.itertuples()))
    fpids=pd.read_csv(r/f'dehatebert_{condition}_false_positive_ids.csv')
    ok(f'{condition}: false-positive IDs agree',set(fpids.id)==set(p.loc[(p.label==0)&(p.prediction==1),'id']))
    print(condition,json.dumps(calc_m))
    print('thresholds\n',thresholds[['threshold','accuracy','macro_f1','fpr','fnr']].to_string(index=False))
    print('nonhateful categories\n',cats[cats.fpr.notna()][['functionality','n','fp','fpr']].to_string(index=False))
ok('Before/after case IDs, gold labels, functionalities, identities and templates match',frames['released'][['id','label','functionality','target_ident','templ_id']].equals(frames['finetuned'][['id','label','functionality','target_ident','templ_id']]))
ids={}
counts=pd.read_csv(r/'dataset_counts.csv')
for split in ['train','val','test']:
    d=pd.read_csv(r/f'hatexplain_{split}_ids.csv'); ids[split]=set(d.id)
    ok(f'{split}: unique IDs and saved binary mapping',d.id.is_unique and np.array_equal(d.label,d.label_original.map({'normal':0,'offensive':0,'hatespeech':1})))
    c=counts[(counts.dataset=='HateXplain')&(counts.split==split)].iloc[0]
    ok(f'{split}: saved class counts agree',len(d)==c['n'] and (d.label==1).sum()==c.hate and (d.label==0).sum()==c.non_hate)
ok('Train/validation/test saved IDs are mutually disjoint',not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test'])
hist=pd.read_csv(r/'dehatebert/training_history.csv')
cfg=json.loads((r/'dehatebert/run_config.json').read_text())
status=json.loads((r/'dehatebert/status.json').read_text())
ok('Two epochs completed and highest validation macro-F1 selected',status['status']=='completed' and len(hist)==2 and int(hist.loc[hist.validation_macro_f1.idxmax(),'epoch'])==cfg['selected_epoch']==2)
ok('Explicit label mapping agrees with positive-class index',cfg['id2label']=={'0':'NON_HATE','1':'HATE'} and cfg['hate_index']==1)
summary=pd.read_csv(r/'model_comparison.csv')
ok('Summary includes only two conditions of DeHateBERT',len(summary)==2 and set(summary.model)=={'dehatebert'} and set(summary.condition)=={'released','finetuned'})
audit={'status':'passed','scope':'Independent recomputation from exported prediction rows, notebook/code consistency, split-ID/label/count checks, and training-log checks. Neural inference/training was not repeated, model weights are not in the archive, and raw-text duplicate filtering was not independently re-run.','passed_check_count':len(checks),'checks':checks,'source_archive':'Jaida_Update1_Results_20261031_021541.zip','notebook_sha256':hashlib.sha256(nbp.read_bytes()).hexdigest(),'project_py_sha256':hashlib.sha256((base/'project.py').read_bytes()).hexdigest()}
(base/'evidence/verification_recheck.json').write_text(json.dumps(audit,indent=2))
print('\nCHECKS PASSED',len(checks))
print('Notebook metadata keys',nb['metadata'].keys())
print('Notebook stream outputs:')
for c in nb['cells']:
 for o in c.get('outputs',[]):
  if o['output_type']=='stream': print(''.join(o['text'])[:6000])
