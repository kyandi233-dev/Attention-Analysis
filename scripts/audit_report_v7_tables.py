"""Row-bound numeric transcription audit; no model training or source modification."""
import argparse,json,re,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--inventory',default='table_inventory.json');a=ap.parse_args();root=a.root;r=a.results;out=r/'v7_audit'
tables=json.loads((out/a.inventory).read_text(encoding='utf-8'));checks=[];errors=[];sources={}
def read(path):
 sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return pd.read_csv(path)
def values(s):return re.findall(r'[+−-]?(?:\d+(?:\.\d*)?|\.\d+)',s.replace(',','') if re.search(r'\d,\d{3}',s) and '[' not in s else s)
def cell(t,i,j,expected):
 actual=tables[t]['rows'][i][j];got=values(actual);ex=np.atleast_1d(expected).tolist();ok=len(got)==len(ex)
 if ok:
  for token,v in zip(got,ex):
   decimals=len(token.split('.')[-1]) if '.' in token else 0;tol=.50001*10**(-decimals)
   if '<' in actual:ok=ok and float(v)<float(token)
   elif not np.isfinite(float(v)) or abs(float(token.replace('−','-'))-float(v))>tol:ok=False
 item={'table_index':t,'row':i,'column':j,'shown':actual,'source_values':ex,'pass':ok};checks.append(item)
 if not ok:errors.append(item)
def rows(t,df,mapping):
 assert len(df)==len(tables[t]['rows'])-1,(t,len(df),len(tables[t]['rows']))
 for i,(_,x) in enumerate(df.iterrows(),1):
  for j,fields in mapping.items():cell(t,i,j,[x[f] for f in fields] if isinstance(fields,list) else fields(x) if callable(fields) else x[fields])
ocular=root/'FormalScience/Ocular/tables';movement=root/'FormalScience/Movement/tables';behavior=root/'Behavior/formal_v3'
progress=read(ocular/'ocular_task_progression.csv');oq1=read(ocular/'ocular_q1_models.csv');oq2=read(ocular/'ocular_q2_models.csv');links=read(ocular/'ocular_behavior_links.csv')
rows(25,progress,{2:'estimate',3:['ci_low','ci_high'],4:'q_value_bh_within_family',5:['participant_group_n','session_n','n_rows']})
for t,df,term,j in [(26,oq1,'ocular_within_z',2),(27,oq1,'ocular_between_z',2),(28,oq2,'ocular_within_z',1),(29,oq2,'ocular_between_z',1)]:
 rows(t,df[df.term==term],{j:'estimate',j+1:['ci_low','ci_high'],j+2:'q_value_bh_within_family',j+3:['participant_group_n','session_n','n_rows']})
for t,term in [(30,'ocular_within_z'),(31,'ocular_between_z')]:
 df=links[links.term==term].copy()
 rows(t,df,{2:'estimate',3:['ci_low','ci_high'],4:'q_value_bh_within_family',5:['participant_group_n','session_n','n_rows']})
for t,name,j,coef in [(37,'movement_task_progression',2,'estimate'),(38,'movement_q1_models',2,'estimate_per_predictor_sd'),(39,'movement_q2_models',1,'estimate_per_predictor_sd'),(40,'movement_behavior_links',2,'estimate_per_predictor_sd')]:rows(t,read(movement/(name+'.csv')),{j:coef,j+1:['ci_low','ci_high'],j+2:'n_rows'})
rows(41,read(movement/'movement_feature_coverage.csv'),{1:'finite_probe_n',2:lambda x:x.finite_fraction*100,3:'participant_group_n',4:'session_n'})
bc=read(behavior/'b1_b2_participant_cluster_bootstrap.csv');rows(34,bc[bc.metric!='omission_rate'],{1:'estimate_b2_minus_b1',2:'ci_low',3:'ci_high',4:'participant_group_n'})
error=read(behavior/'error_trajectory_summary.csv')
for i,(_,x) in enumerate(error.iterrows(),1):
 if pd.notna(x.participant_centered_rt_mean_ms):
  for j,v in [(2,x.participant_centered_rt_mean_ms),(3,x.participant_centered_rt_sem_ms),(4,x.participant_centered_rt_mean_ms-1.96*x.participant_centered_rt_sem_ms),(5,x.participant_centered_rt_mean_ms+1.96*x.participant_centered_rt_sem_ms)]:cell(35,i,j,v)
 for j,field in [(1,'relative_trial'),(6,'participant_group_n'),(7,'session_n')]:
  if field in x:cell(35,i,j,x[field])
coverage=read(root/'FormalScience/Behavior/tables/behavior_feature_coverage.csv')
for i,(_,x) in enumerate(coverage.iterrows(),1):
 v=json.loads(x.coverage_summary)
 for j,key in [(1,'finite_probe_n'),(2,'finite_fraction'),(3,'participant_group_n'),(4,'session_n')]:cell(36,i,j,v[key])
data=json.loads((r/'report_payload.json').read_text(encoding='utf-8'));models=pd.DataFrame(data['models']);pairs=pd.DataFrame(data['pairs']);session=read(r/'session_results.csv')
rows(46,models,{2:'n_participants',3:'n_probes',4:'participant_equal_log_loss',5:'ci_lower',6:'ci_upper'})
rows(33,models,{1:['n_participants','n_probes'],2:'participant_equal_log_loss',3:['ci_lower','ci_upper'],4:'participant_macro_auroc',5:['participant_macro_auroc_ci_lower','participant_macro_auroc_ci_upper'],6:'n_participants_auroc_estimable'})
rows(47,models,{2:'participant_macro_auroc',3:'participant_macro_auroc_ci_lower',4:'participant_macro_auroc_ci_upper',5:'participant_macro_brier',6:'n_participants_auroc_estimable',7:'n_participants_auroc_single_class'})
for i,(_,x) in enumerate(models.iterrows(),1):
 if x.calibration_slope_reporting=='reportable':cell(47,i,8,x.calibration_slope)
rows(32,pairs,{2:['point_estimate','ci_lower','ci_upper']})
def model(s,m):return models[(models.analysis_set_id==s)&(models.model_id==m)].iloc[0]
for i,(s,m,label) in enumerate(data['independent'],1):
 x=model(s,m);ss=session[(session.analysis_set_id==s)&(session.model_id==m)].iloc[0]
 for j,f in [(1,['participant_equal_log_loss','ci_lower','ci_upper']),(2,['participant_macro_auroc','participant_macro_auroc_ci_lower','participant_macro_auroc_ci_upper']),(3,['participant_macro_brier']),(4,['n_participants','n_sessions','n_probes']),(5,['n_participants_auroc_estimable'])]:cell(14,i,j,[x[k] for k in f])
 for j,f in [(1,'n_estimable'),(2,'n_single_class'),(3,'mean_auroc'),(4,'median_auroc')]:cell(15,i,j,ss[f])
for i,k in enumerate(data['devices'],1):
 x=model('AS.device::'+k,k)
 for j,f in [(2,'n_participants'),(3,'n_probes'),(4,'participant_equal_log_loss'),(5,'participant_macro_auroc')]:cell(20,i,j,x[f])
for t,mods,full in [(18,['ocular','movement','cardiopulmonary'],False),(19,['behavior','ocular','movement','cardiopulmonary'],True)]:
 for i,k in enumerate(mods,1):
  x=pairs[(pairs.feature_id=='modality::'+k)&(pairs.analysis_set_id.eq('AS.full') if full else pairs.analysis_set_id.ne('AS.full'))].iloc[0]
  for j,v in [(1,x.point_estimate),(2,[x.ci_lower,x.ci_upper]),(3,x.n_participants),(4,model(x.analysis_set_id,x.added_model_id).n_probes)]:cell(t,i,j,v)
rows(16,read(root/'SupervisedRunsWindowSensitivityV1/window_sensitivity_summary.csv'),{0:'window_seconds',2:'participant_equal_log_loss',3:['ci_lower_95','ci_upper_95']})
h=json.loads((root/'SupervisedRunsOcularHeadmotionSensitivityV1/headmotion_sensitivity_summary.json').read_text(encoding='utf-8'))
for i,k in enumerate(['ocular_reference','ocular_plus_headmotion'],1):
 x=h['models'][k];cell(17,i,2,x['participant_equal_log_loss']);cell(17,i,3,[x['ci_lower_95'],x['ci_upper_95']])
questionnaire=read(root.parent/'derived/questionnaire_session_analysis_v1/ordinal_models_v1/ordinal_model_coefficients.csv')
rows(21,questionnaire,{2:'odds_ratio_per_1sd',3:['ci95_or_low','ci95_or_high'],4:'p'})
rows(24,questionnaire,{3:'estimate_log_odds',4:'se',5:'z',6:'p',7:'odds_ratio_per_1sd',8:['ci95_or_low','ci95_or_high']})
for i,(_,x) in enumerate(questionnaire.iterrows(),1):
 if pd.notna(x.p_fdr_exploratory):cell(24,i,9,x.p_fdr_exploratory)
probe=read(behavior/'probe_primary_30s.csv')
for j,c in enumerate(['q1_nominal_4class','q2_ordinal_4level']):
 for k in range(1,5):n=int(probe[c].eq(k).sum());cell(9,j*4+k,2,n);cell(9,j*4+k,3,100*n/len(probe))
for i,c in enumerate(['go_correct_rt_cv','go_correct_rt_theilsen_slope_ms_per_s','raw_go_omission_rate','commission_rate'],1):
 v=probe[c].dropna()
 for j,x in [(1,len(v)),(2,v.mean()),(3,v.std()),(4,v.median())]:
  if re.search(r'\d',tables[10]['rows'][i][j]):cell(10,i,j,x)
heart=read(r.parent/'repaired_source/m1_cardiopulmonary_taskb_source.csv')
# The v5 independently computed descriptions are reused only after exact source hash matching.
desc=json.loads((r/'v5_descriptive_results.json').read_text(encoding='utf-8'))
assert hashlib.sha256((r.parent/'repaired_source/m1_cardiopulmonary_taskb_source.csv').read_bytes()).hexdigest()==desc['input_hashes']['SupervisedRunsV4_Cardiopulmonary\\repair_20260928\\repaired_source\\m1_cardiopulmonary_taskb_source.csv']
for i,x in enumerate(desc['heart_stats'],1):
 for j,v in [(1,x['n']),(2,x['mean']),(3,x['sd']),(4,x['median']),(5,[x['q25'],x['q75']])]:cell(13,i,j,v)
 cell(42,i,1,x['n']);cell(42,i,2,100*x['n']/2320)
# Bind selected main text tables to their complete coefficient rows.
for i,row in enumerate(tables[11]['rows'][1:],1):
 fid={'瞳孔水平':'level_mean','瞳孔波动':'variability_mad','眨眼频率':'blink_event_rate_per_min'}[row[0]]
 point=float(row[2].replace('−','-'));candidate=progress[(progress.feature_id==fid)&(np.abs(progress.estimate-point)<.0005)];assert len(candidate)==1,(i,fid)
 x=candidate.iloc[0];cell(11,i,2,x.estimate);cell(11,i,3,[x.ci_low,x.ci_high])
mq1=read(movement/'movement_q1_models.csv');mq2=read(movement/'movement_q2_models.csv');subset=pd.concat([mq1[mq1.predictor=='body_motion_energy_median'],mq2[mq2.predictor=='body_motion_energy_median']]);rows(12,subset,{1:'estimate_per_predictor_sd',2:['ci_low','ci_high']})
# Actual standalone coverage, with the same feature order as Table 8.
feature_order=[k for k in data['labels'] if k.startswith('ocular.')]+[k for k in data['labels'] if k.startswith('movement.')]+[k for k in data['labels'] if k.startswith('cardiopulmonary.')]+[k for k in data['labels'] if k.startswith('behavior.')]
assert len(feature_order)==13
for i,fid in enumerate(feature_order,1):
 x=model('AS.standalone::'+fid,'standalone::'+fid);cell(8,i,4,x.n_sessions);cell(8,i,5,x.n_probes)
qa=json.loads((root.parent/'derived/questionnaire_session_analysis_v1/ordinal_models_v1/ordinal_model_audit.json').read_text(encoding='utf-8'))
cell(22,2,1,qa['n_sessions']);cell(22,3,1,[qa['repeat_participant_ids'],qa['repeat_ids_with_multiple_sessions']])
for i in range(1,5):cell(23,i,2,qa['outcome_counts'].get(str(i),0))
reference=read(root.parent/'derived/mmwave_estimator_improvement_v1_20260912_r1/final/CONTROL_VS_CANDIDATES_100_PROBES_LOCAL_ONLY.csv');err=reference.control_fused_hr_bpm-reference.ecg_hr_bpm;bias=err.mean()
for i,key in enumerate(['CORRECT_OR_NEAR_CORRECT','SELECTED_TARGET_WRONG_PEAK','HARMONIC_OR_HALF_DOUBLE_LOCK','TARGET_BIN_CHANNEL_MISS',None],1):
 v=err if key is None else err[reference.p2_failure_class==key];fraction=len(v)/len(reference);contribution=v.mean()*fraction
 for j,x in [(1,len(v)),(2,100*fraction),(3,v.mean()),(4,v.abs().mean())]:cell(43,i,j,x)
 for j,x in [(1,100*fraction),(2,v.mean()),(3,contribution),(4,100*contribution/bias)]:cell(44,i,j,x)
for i,key in enumerate(['control_time_hr_bpm','control_spectral_hr_bpm','control_fused_hr_bpm'],1):
 v=reference[key]-reference.ecg_hr_bpm;cell(45,i,1,v.abs().mean());cell(45,i,2,v.mean())
result={'status':'PASS' if not errors else 'FAIL','checked_numeric_cells':len(checks),'checked_tables':sorted(set(c['table_index'] for c in checks)),'source_sha256':sources,'errors':errors,'checks':checks}
(out/'table_numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','checks']},ensure_ascii=False));sys.exit(bool(errors))
