"""Refit the existing comparison plans for an explicitly separate Q1 target.

Private inputs and predictions remain outside Git. Never recode the original Q1.
"""
import argparse,hashlib,json,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.entrypoint import (
    _resolve_model_plan,_require_comparison_models,_validate_analysis_set_feature_scope,
    _runtime_time_legality_audit,_require_frozen_runtime_contract)
from attention_pipeline.supervised_learning.runner import run_nested_loso
from attention_pipeline.supervised_learning.task import BinaryTaskSpec

KEYS=['participant_group_id','session_id','block_id','probe_event_id']
def safe_name(sid):return str(sid).replace(':','_')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def read(p):return pd.read_csv(p,low_memory=False)
def fit_chunk(frame,chosen,spec,sid,groups):
    return run_nested_loso(frame,model_feature_schemes=chosen,task_spec=spec,run_id=sid+'__q1_12vs34',analysis_set_id=sid,outer_group_subset=groups)
def ci(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(20260830);b=values[rng.integers(0,len(values),(1000,len(values)))].mean(axis=1)
    return float(values.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))

def summarize(pred,pcol,target):
    scores=[];participants={}
    for model,g in pred.groupby('model_id',sort=False):
        records=[];total=float(g.q1_binary.sum())
        for pid,t in g.groupby('participant_group_id',sort=True):
            y=t.q1_binary.to_numpy(int);p=np.clip(t[pcol].to_numpy(float),1e-15,1-1e-15);prior=np.clip((total-y.sum())/(len(g)-len(t)),1e-15,1-1e-15)
            loss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)));base=float(np.mean(-y*np.log(prior)-(1-y)*np.log1p(-prior)));classes=len(np.unique(y))==2
            rpos=float(np.mean(p[y==1]>=.5)) if (y==1).any() else np.nan;rneg=float(np.mean(p[y==0]<.5)) if (y==0).any() else np.nan
            records.append({'participant_group_id':pid,'loss':loss,'baseline_loss':base,'improvement':base-loss,'brier':float(np.mean((p-y)**2)),'auroc':float(roc_auc_score(y,p)) if classes else np.nan,'balanced_accuracy':float((rpos+rneg)/2) if classes else np.nan,'positive_recall':rpos,'negative_recall':rneg})
        table=pd.DataFrame(records);participants[model]=table
        rec={'analysis_set_id':str(g.analysis_set_id.iloc[0]),'model_id':model,'target':target,'n_probes':len(g),'n_participants':len(table),'n_sessions':g.session_id.nunique(),'positive_probes':int(g.q1_binary.sum()),'positive_fraction':float(g.q1_binary.mean()),'n_two_class_participants':int(table.auroc.notna().sum())}
        for col in ['loss','baseline_loss','improvement','brier','auroc','balanced_accuracy','positive_recall','negative_recall']:
            point,lower,upper=ci(table[col]);rec[col]=point;rec[col+'_ci_low']=lower;rec[col+'_ci_high']=upper
        rec['relative_loss_reduction']=rec['improvement']/rec['baseline_loss'];scores.append(rec)
    return scores,participants

def audit(result,frame,oldpred,oldfolds,models):
    assert result.failures.empty,result.failures
    p=result.predictions;assert not p.model_failed.any() and p.p_q1_in_1_2.notna().all()
    assert 'p_q1_equals_1' not in p and (p.q1_binary==p.q1_nominal_4class.isin([1,2]).astype(int)).all()
    assert not p.duplicated(['model_id']+KEYS).any()
    assert len(p)==len(frame)*len(models)
    for model,g in p.groupby('model_id'):
        old=oldpred[oldpred.model_id==model]
        cols=KEYS+['q1_nominal_4class']
        left=g[cols].astype(str).sort_values(KEYS).reset_index(drop=True);right=old[cols].astype(str).sort_values(KEYS).reset_index(drop=True)
        # CSV round-trips may express original integer labels as 1 or 1.0.
        left['q1_nominal_4class']=pd.to_numeric(left.q1_nominal_4class);right['q1_nominal_4class']=pd.to_numeric(right.q1_nominal_4class)
        pd.testing.assert_frame_equal(left,right,check_dtype=False)
    groups=set(frame.participant_group_id.astype(str));inner_count=0
    oldmap={(f['model_id'],str(f['outer_fold_group'])):f for f in oldfolds}
    for f in result.fold_audits:
        tr=set(f['outer_train_group_ids']);te=set(f['outer_test_group_ids']);old=oldmap[(f['model_id'],str(f['outer_fold_group']))]
        assert not f['failed'] and tr.isdisjoint(te) and tr|te==groups and te=={f['outer_fold_group']}
        assert tr==set(old['outer_train_group_ids']) and te==set(old['outer_test_group_ids'])
        assert set(f['final_refit']['preprocessing']['fit_group_ids'])==tr
        scores=f['selection']['candidate_participant_macro_log_loss'];best=min(scores.values());selected=f['selection']['selected_c']
        assert any(float(k.split('C=')[-1])==selected and np.isclose(v,best,rtol=0,atol=1e-14) for k,v in scores.items())
        assert len(f['selection']['inner_fold_audits'])==len(old['selection']['inner_fold_audits'])==5
        for fold,oldinner in zip(f['selection']['inner_fold_audits'],old['selection']['inner_fold_audits']):
            it=set(fold['train_group_ids']);iv=set(fold['validation_group_ids'])
            assert it.isdisjoint(iv) and it|iv==tr and not fold['failure_by_c']
            assert it==set(oldinner['train_group_ids']) and iv==set(oldinner['validation_group_ids'])
            assert set(fold['preprocessing']['fit_group_ids'])==it;inner_count+=1
    return {'outer_folds':len(result.fold_audits),'inner_folds':inner_count,'failures':0,'raw_labels_keys_and_splits_match_primary':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repair-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--config',type=Path,default=Path('configs/supervised_learning_q1_12vs34_sensitivity.yaml'));ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    plan=yaml.safe_load(a.config.read_text(encoding='utf-8'));base=load_config(plan['base_config']);_require_frozen_runtime_contract(base.data);legality=_runtime_time_legality_audit(base.data);families,_=_resolve_model_plan(base.data);spec=BinaryTaskSpec(**{**plan['task'],'positive_values':tuple(plan['task']['positive_values']),'negative_values':tuple(plan['task']['negative_values'])})
    out=a.output_root;out.mkdir(exist_ok=a.resume,parents=True)
    lineage=read(a.repair_root/'report_results/result_lineage.csv');pairplan=read(a.repair_root/'report_results/paired_results.csv')
    priorities=['AS.behavior_reference','AS.sensor_only_joint','AS.behavior_plus_ocular','AS.behavior_plus_movement','AS.behavior_plus_cardiopulmonary','AS.full']
    lineage=lineage.iloc[sorted(range(len(lineage)),key=lambda i:(priorities.index(lineage.iloc[i].analysis_set_id) if lineage.iloc[i].analysis_set_id in priorities else 99,str(lineage.iloc[i].analysis_set_id)))]
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
    execution={'code_sha':code,'working_tree_dirty':bool(dirty.strip()),'runner_sha256':sha('src/attention_pipeline/supervised_learning/runner.py'),'script_sha256':sha(__file__),'workers':a.workers}
    if a.resume:
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));assert manifest['config_sha256']==sha(a.config) and manifest['task']==plan
        allnew=read(out/'model_results.csv').to_dict('records');allold=read(out/'primary_target_results.csv').to_dict('records');allpairs=read(out/'paired_results.csv').to_dict('records') if (out/'paired_results.csv').stat().st_size>5 else []
        for completed in manifest['completed_sets']:
            assert sha(out/safe_name(completed['analysis_set_id'])/'probe_predictions.csv')==completed['prediction_sha256']
        manifest.setdefault('resumed_executions',[]).append(execution)
    else:
        manifest={'status':'RUNNING','task':plan,**execution,'config_sha256':sha(a.config),'python':sys.version,'time_legality':legality,'completed_sets':[]};allnew=[];allold=[];allpairs=[]
    manifest['status']='RUNNING';dump(out/'manifest.json',manifest);t0=time.time();pool=ProcessPoolExecutor(max_workers=a.workers)
    finished={v['analysis_set_id'] for v in manifest['completed_sets']}
    for _,line in lineage.iterrows():
        sid=line.analysis_set_id;inp=a.repair_root/'inputs'/f'{safe_name(sid)}.csv';frame=read(inp);models=_require_comparison_models(frame);chosen={k:families[k] for k in models};_,predictors=_validate_analysis_set_feature_scope(frame,chosen,analysis_set_id=sid)
        if sid in finished:
            assert sha(inp)==next(v['input_sha256'] for v in manifest['completed_sets'] if v['analysis_set_id']==sid)
            print('REUSE completed audited set '+sid,flush=True);continue
        oldrun=Path(line.source_run);assert sha(oldrun/'probe_predictions.csv')==line.prediction_sha256
        oldmanifest=json.loads((oldrun/'run_manifest.json').read_text(encoding='utf-8-sig'));oldinput=Path(oldmanifest['provenance']['input_table'])
        assert sha(oldinput)==oldmanifest['provenance']['input_sha256']
        columns=list(dict.fromkeys(KEYS+['q1_nominal_4class']+list(predictors)))
        pd.testing.assert_frame_equal(frame[columns].sort_values(KEYS).reset_index(drop=True),read(oldinput)[columns].sort_values(KEYS).reset_index(drop=True),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
        oldpred=read(oldrun/'probe_predictions.csv');oldfolds=json.loads((oldrun/'fold_audits.json').read_text(encoding='utf-8-sig'))
        print(f'START {sid}: {len(frame)} probes, {len(models)} models',flush=True)
        groups=sorted(frame.participant_group_id.astype(str).unique());chunks=[list(v) for v in np.array_split(groups,min(a.workers,len(groups)))]
        futures=[pool.submit(fit_chunk,frame,chosen,spec,sid,g) for g in chunks];parts=[f.result() for f in futures]
        run=parts[0];run.predictions=pd.concat([p.predictions for p in parts],ignore_index=True);run.fold_audits=[f for p in parts for f in p.fold_audits];run.failures=pd.concat([p.failures for p in parts],ignore_index=True);run.metadata.update({'partial_outer_run':False,'evaluated_participant_groups':groups,'execution':execution})
        check=audit(run,frame,oldpred,oldfolds,models);dest=out/safe_name(sid);dest.mkdir();run.predictions.to_csv(dest/'probe_predictions.csv',index=False,encoding='utf-8-sig');dump(dest/'fold_audits.json',run.fold_audits);dump(dest/'run_metadata.json',run.metadata)
        new,pt=summarize(run.predictions,'p_q1_in_1_2','12_vs_34');old,_=summarize(oldpred,'p_q1_equals_1','1_vs_234');allnew+=new;allold+=old
        for _,pair in pairplan[pairplan.analysis_set_id==sid].iterrows():
            b=pt[pair.baseline_model_id].set_index('participant_group_id').loss;c=pt[pair.added_model_id].set_index('participant_group_id').loss
            assert b.index.equals(c.index);point,lower,upper=ci(b-c)
            allpairs.append({'analysis_set_id':sid,'comparison_type':pair.comparison_type,'baseline_model_id':pair.baseline_model_id,'added_model_id':pair.added_model_id,'improvement':point,'ci_low':lower,'ci_high':upper,'n_participants':len(b)})
        for name,records in [('model_results',allnew),('primary_target_results',allold),('paired_results',allpairs)]:pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
        check.update({'analysis_set_id':sid,'models':len(models),'input_sha256':sha(inp),'prediction_sha256':sha(dest/'probe_predictions.csv'),'old_prediction_sha256':line.prediction_sha256,'elapsed_seconds_total':time.time()-t0});manifest['completed_sets'].append(check);dump(out/'manifest.json',manifest)
        print(f'DONE {sid}: {check["outer_folds"]} outer folds; elapsed {time.time()-t0:.1f}s',flush=True)
    pool.shutdown();assert len(allnew)==plan['expected_model_rows'] and len(lineage)==plan['expected_analysis_sets'] and len(allpairs)==plan['expected_pairs']
    manifest.update({'status':'PASS','model_rows':len(allnew),'paired_rows':len(allpairs),'outer_folds':sum(v['outer_folds'] for v in manifest['completed_sets']),'inner_folds':sum(v['inner_folds'] for v in manifest['completed_sets']),'elapsed_seconds':time.time()-t0});dump(out/'manifest.json',manifest);print('ALL PASS',flush=True)

if __name__=='__main__':main()
