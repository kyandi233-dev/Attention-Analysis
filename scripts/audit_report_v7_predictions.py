"""Independently recompute report prediction metrics from every archived prediction."""
import argparse,json,hashlib,sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
def read(p):return pd.read_csv(p,encoding='utf-8-sig',low_memory=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
sys.stdout.reconfigure(encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('--repair-root',type=Path,required=True);a=ap.parse_args();root=a.repair_root;out=root/'report_results';models=read(out/'model_results.csv');diagnostics=read(out/'probability_diagnostics.csv');lineage=read(out/'result_lineage.csv');checked=[];folds=0;inner=0
for _,row in lineage.iterrows():
 run=Path(row.source_run);m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8-sig'));source=read(m['provenance']['input_table']);current=read(root/'inputs'/(row.analysis_set_id.replace('::','__')+'.csv'));key=['session_id','block_id','probe_index_in_block'];cols=key+['participant_group_id','q1_nominal_4class']+m['declared_model_predictor_union'];equal(source[cols].sort_values(key).reset_index(drop=True),current[cols].sort_values(key).reset_index(drop=True));assert sha(run/'probe_predictions.csv')==row.prediction_sha256
 pred=read(run/'probe_predictions.csv');assert not pred.duplicated(['model_id']+key).any();assert pred.p_q1_equals_1.between(0,1).all();assert pred.q1_binary.eq(pred.q1_nominal_4class.eq(1).astype(int)).all()
 for mid,g in pred.groupby('model_id'):
  expected=models[(models.analysis_set_id==row.analysis_set_id)&(models.model_id==mid)].iloc[0];dg=diagnostics[(diagnostics.analysis_set_id==row.analysis_set_id)&(diagnostics.model_id==mid)].iloc[0];metrics=[]
  for participant,p in g.groupby('participant_group_id'):
   y=p.q1_binary.to_numpy();prob=p.p_q1_equals_1.to_numpy();metrics.append([log_loss(y,prob,labels=[0,1]),np.mean((prob-y)**2),roc_auc_score(y,prob) if len(np.unique(y))==2 else np.nan])
  values=np.nanmean(metrics,axis=0);assert np.isclose(values[0],expected.participant_equal_log_loss,rtol=0,atol=1e-12);assert np.isclose(values[1],dg.participant_macro_brier,rtol=0,atol=1e-12);assert np.isclose(values[2],dg.participant_macro_auroc,rtol=0,atol=1e-12)
  assert len(g)==expected.n_probes and g.participant_group_id.nunique()==expected.n_participants
  checked.append({'analysis_set':row.analysis_set_id,'model':mid,'n':len(g),'participant_equal_log_loss':values[0],'participant_macro_brier':values[1],'participant_macro_auroc':values[2]})
 audits=json.loads((run/'fold_audits.json').read_text(encoding='utf-8-sig'))
 for audit in audits:
  train=set(audit['outer_train_group_ids']);test=audit['outer_fold_group'];assert test not in train and not audit['failed'];assert set(audit['outer_test_group_ids'])=={test};folds+=1
  refit=audit['final_refit'];assert set(refit['train_group_ids'])==train and set(refit['test_group_ids'])=={test};assert set(refit['preprocessing']['fit_group_ids'])==train
  selection=audit['selection'];assert selection['selected_c'] in [.01,.1,1,10];assert len(selection['inner_fold_audits'])==5
  for f in selection['inner_fold_audits']:
   tr=set(f['train_group_ids']);va=set(f['validation_group_ids']);assert not tr&va and tr|va==train;assert set(f['preprocessing']['fit_group_ids'])==tr;assert not f['failure_by_c'];inner+=1
result={'status':'PASS','analysis_sets':len(lineage),'model_rows':len(checked),'outer_fold_records':folds,'inner_fold_records':inner,'metrics':checked,'inputs_predictors_labels_groups_equal':True,'prediction_files_match_recorded_hashes':True,'models_refitted':0,'tolerance':1e-12};(out/'v7_audit/prediction_recheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='metrics'})
