"""Check the numerical result prose against named current result rows."""
import argparse,json,re,sys,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--edited',action='store_true');a=ap.parse_args();root=a.root;r=a.results;o=r/'v7_audit';text=(o/'paragraph_inventory.txt').read_text(encoding='utf-8');ps={int(s.split('\t',1)[0]):s.split('\t',2)[-1] for s in text.splitlines() if re.match(r'^\d+\t',s)};checks=[];errors=[];sources={}
if a.edited:
 for change in json.loads((o/'edit_manifest.json').read_text(encoding='utf-8'))['changes']:
  if 'paragraph' in change:ps[change['paragraph']]=change['after']
def read(p):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def check(p,value,dp,absolute=False):
 expected=f'{abs(value) if absolute else value:.{dp}f}';tokens=re.findall(r'[+−-]?(?:\d+(?:\.\d*)?|\.\d+)',ps[p].replace(',',''));found=any(float(t.replace('−','-'))==float(expected) for t in tokens);item={'paragraph':p,'source_value':float(value),'decimal_places':dp,'expected':expected,'pass':found};checks.append(item)
 if not found:errors.append(item)
def series(p,x,fields,dp):
 for field in fields:check(p,x[field],dp)
b=root/'Behavior/formal_v3';oc=root/'FormalScience/Ocular/tables';mv=root/'FormalScience/Movement/tables';probes=read(b/'probe_primary_30s.csv');sessions=read(b/'session_metrics.csv');bc=read(b/'b1_b2_participant_cluster_bootstrap.csv').set_index('metric');bg=read(b/'block_cycle_gee.csv');q2=read(b/'q2_ordinal_gee_models.csv').set_index('predictor')
for value in [probes.participant_group_id.nunique(),probes.session_id.nunique(),2*probes.session_id.nunique(),len(probes),864*probes.session_id.nunique()]:check(256,value,0)
counts=probes[['participant_group_id','session_id']].drop_duplicates().groupby('participant_group_id').size()
for value in [(counts>1).sum(),(counts==2).sum(),(counts>=3).sum()]:check(256,value,0)
for value in [100*probes.q1_nominal_4class.eq(1).mean(),100*probes.q1_nominal_4class.ne(1).mean(),100*probes.q1_nominal_4class.eq(3).mean(),100*probes.q2_ordinal_4level.ge(3).mean()]:check(262,value,1)
check(264,probes.correct_go_rt_opportunities.mean(),2);check(264,100*probes.raw_go_omission_rate.eq(0).mean(),1);check(264,probes.nogo_opportunities.mean(),2)
for field in ['go_correct_rt_mean_ms','go_correct_rt_median_ms']:check(266,sessions[field].mean(),2)
for metric,dp in [('go_correct_rt_cv',3),('go_correct_rt_sd_ms',1)]:series(268,bc.loc[metric],['estimate_b2_minus_b1','ci_low','ci_high'],dp)
for term in ['cycle_bin','block2:cycle_bin']:
 x=bg[(bg.metric=='go_correct_rt_cv')&(bg.term==term)].iloc[0];series(269,x,['estimate','ci_low','ci_high'],4)
for field in ['participant_group_n','session_n','n_rows']:check(269,x[field],0)
for metric in ['commission_rate','dprime_loglinear']:
 x=q2.loc[metric];series(280,x,['ci_low','ci_high'],3);check(280,abs(x.estimate_per_predictor_sd),3)
series(281,q2.loc['clean_go_omission_rate'],['estimate_per_predictor_sd','ci_low','ci_high'],3)
eq1=read(oc/'ocular_q1_models.csv');eq2=read(oc/'ocular_q2_models.csv');links=read(oc/'ocular_behavior_links.csv')
for cat in [3,4]:
 x=eq1[(eq1.feature_id=='blink_event_rate_per_min')&(eq1.term=='ocular_within_z')&(eq1.contrast_category==cat)].iloc[0];series(306,x,['odds_ratio','or_ci_low','or_ci_high','q_value_bh_within_family'],3)
x=eq1[(eq1.feature_id!='blink_event_rate_per_min')&(eq1.term=='ocular_within_z')];check(309,x.q_value_bh_within_family.min(),3);check(309,x.q_value_bh_within_family.max(),3)
x=eq2[(eq2.feature_id=='blink_event_rate_per_min')&(eq2.term=='ocular_within_z')].iloc[0];series(311,x,['estimate','ci_low','ci_high','q_value_bh_within_family'],3)
x=links[(links.feature_id=='blink_event_rate_per_min')&(links.outcome=='raw_go_omission_rate')&(links.term=='ocular_within_z')].iloc[0];series(319,x,['estimate','ci_low','ci_high','odds_ratio','or_ci_low','or_ci_high'],3)
for fid,dp in [('quadratic_curvature_per_sec2',4),('blink_event_rate_per_min',3)]:
 outcome='go_correct_rt_theilsen_slope_ms_per_s' if dp==4 else 'commission_rate';x=links[(links.feature_id==fid)&(links.outcome==outcome)&(links.term=='ocular_between_z')].iloc[0];series(322,x,['estimate'],dp);series(322,x,['ci_low','ci_high'] if dp==4 else ['odds_ratio','or_ci_low','or_ci_high'],dp);check(322,x.q_value_bh_within_family,3)
move=read(mv/'movement_task_progression.csv')
for term in ['cycle_bin','block2:cycle_bin']:
 x=move[(move.metric=='body_motion_energy_median')&(move.term==term)].iloc[0];series(327,x,['ci_low','ci_high'],6);check(327,x.estimate,6,absolute=True)
desc=json.loads((r/'v5_descriptive_results.json').read_text(encoding='utf-8'))
for x in desc['heart_stats']:check(341,x['n'],0);check(341,x['mean'],2);check(341,x['sd'],2)
check(341,2200/2320*100,2);check(345,desc['reference']['mae'],2);check(345,desc['reference']['bias'],2)
data=json.loads((r/'report_payload.json').read_text(encoding='utf-8'));models=pd.DataFrame(data['models']);pairs=pd.DataFrame(data['pairs']);sess=read(r/'session_results.csv')
def model(s,m):return models[(models.analysis_set_id==s)&(models.model_id==m)].iloc[0]
sensor=model('AS.sensor_only_joint','sensor_only_joint');full=model('AS.full','full');beh=model('AS.behavior_reference','behavior_reference');eye=model('AS.behavior_plus_ocular','modality::ocular')
series(354,sensor,['participant_macro_auroc','participant_macro_auroc_ci_lower','participant_macro_auroc_ci_upper'],4)
for x in [sensor,beh]:series(355,x,['participant_equal_log_loss','training_fold_prevalence_log_loss'],4)
for sid,mid in [('AS.behavior_reference','behavior_reference'),('AS.behavior_plus_ocular','modality::ocular'),('AS.sensor_only_joint','sensor_only_joint')]:
 x=sess[(sess.analysis_set_id==sid)&(sess.model_id==mid)].iloc[0];series(360,x,['mean_auroc','median_auroc'],4);check(360,x.n_estimable,0)
for x in [beh,eye,sensor]:series(365,x,['calibration_slope','calibration_slope_ci_lower','calibration_slope_ci_upper'],3)
series(367,full,['calibration_slope','calibration_slope_ci_lower','calibration_slope_ci_upper','calibration_intercept','calibration_intercept_ci_lower','calibration_intercept_ci_upper'],3);check(367,full.participant_macro_calibration_in_the_large,5)
for mod in ['ocular','movement','cardiopulmonary']:
 x=pairs[(pairs.feature_id=='modality::'+mod)&(pairs.analysis_set_id!='AS.full')].iloc[0];series(383,x,['point_estimate','ci_lower','ci_upper'],5)
for mod in ['behavior','movement']:
 x=pairs[(pairs.feature_id=='modality::'+mod)&(pairs.analysis_set_id=='AS.full')].iloc[0];series(392,x,['point_estimate','ci_lower','ci_upper'],5)
check(392,full.participant_equal_log_loss,6)
for fid in ['behavior.nogo_commission.raw.v1','behavior.go_omission.raw.v1','behavior.rt_trend.theilsen.v1','behavior.rt_variability.cv.v1']:
 x=pairs[(pairs.feature_id==fid)&(pairs.analysis_set_id=='AS.full')].iloc[0];series(401,x,['point_estimate','ci_lower','ci_upper'],5)
for fid in ['cardiopulmonary.hr.fused.v1','cardiopulmonary.br.v1']:
 x=model('AS.standalone::'+fid,'standalone::'+fid);series(402,x,['participant_equal_log_loss','ci_lower','ci_upper'],4)
for mid in data['devices']:check(411,model('AS.device::'+mid,mid).n_probes/2320*100,2)
g1=read(oc/'g1_cross_signal_representation_summary.csv');g1=g1[(g1.cleaning_track=='rgb_plus_nir_qc')&(g1.buffer_id=='pre200_post200')&(g1.bin_width_sec==2)]
for metric in ['level_mean','variability_mad']:check(291,g1[g1.metric==metric].iloc[0].spearman_rho,3)
for metric in ['linear_slope_per_sec','quadratic_curvature_per_sec2']:check(291,100*g1[g1.metric==metric].iloc[0].dynamic_sign_agreement_fraction,1)
span=read(root/'NIR_G1/NIR_G1_20260912_fixed/freeze_support/g1_temporal_support_freeze_grid.csv');x=span[(span.signal=='seg_pupil_fraction_within_pupil_iris_hard')&(span.cleaning_track=='rgb_plus_nir_qc')&(span.buffer_id=='pre200_post200')&(span.bin_width_sec==2)].iloc[0]
check(292,x.candidate_row_n,0)
for field in ['linear_span_ge_10s_fraction','linear_span_ge_20s_fraction','quadratic_span_ge_10s_fraction','quadratic_span_ge_20s_fraction']:check(292,100*x[field],2)
progress=read(oc/'ocular_task_progression.csv');check(301,progress.q_value_bh_within_family.min(),3)
qe=read(b/'error_trajectory_summary.csv')
for position in [-1,1]:check(283,qe[(qe.error_type=='go_omission')&(qe.relative_trial==position)].iloc[0].participant_centered_rt_mean_ms,1)
for field in ['bias','lower_line','upper_line']:check(348,desc['reference'][field],2)
mw=read(mv/'movement_q1_models.csv')
for _,x in mw[(mw.predictor.str.contains('horizontal')&mw.contrast_category.eq(2))|(mw.predictor.str.contains('vertical')&mw.contrast_category.eq(4))].iterrows():series(338,x,['estimate_per_predictor_sd','ci_low','ci_high'],3)
paired=json.loads((root/'SupervisedRunsWindowPaired2030V1/paired_summary.json').read_text(encoding='utf-8'));check(375,paired['delta_ci_lower_95'],6);check(375,paired['delta_ci_upper_95'],6)
head=json.loads((root/'SupervisedRunsOcularHeadmotionSensitivityV1/headmotion_sensitivity_summary.json').read_text(encoding='utf-8'));series(379,head['paired_increment'],['point_estimate','ci_lower_95','ci_upper_95'],6)
reduced=json.loads((root/'SupervisedRunsPredefinedReducedV1/AS.full__predefined_reduced_nested__included_missing_aware/predefined_reduced_summary.json').read_text(encoding='utf-8'));series(403,reduced,['frozen_full_log_loss_reproduced','nested_selected_log_loss','delta_log_loss_full_minus_nested','delta_log_loss_ci_lower','delta_log_loss_ci_upper'],6)
qa=json.loads((root.parent/'derived/questionnaire_session_analysis_v1/ordinal_models_v1/ordinal_model_audit.json').read_text(encoding='utf-8'));series(417,qa,['n_sessions','repeat_participant_ids','repeat_ids_with_multiple_sessions'],0)
for v in qa['outcome_counts'].values():check(418,v,0)
if a.edited:
 import zipfile,xml.etree.ElementTree as ET
 manifest=json.loads((o/'edit_manifest.json').read_text(encoding='utf-8'))
 with zipfile.ZipFile(manifest['output']) as z:
  doc=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
  actual=[''.join(p.itertext()) for p in doc.findall('./w:body/w:p',ns)]
 def normalize(s):return re.sub(r'\s+','',re.sub(r'图\s*\d+','图',s))
 actual={normalize(p) for p in actual}
 for p in set(c['paragraph'] for c in checks):
  assert normalize(ps[p]) in actual,('Final artifact prose mismatch',p)
result={'status':'PASS' if not errors else 'FAIL','numerical_claim_checks':len(checks),'paragraphs_checked':len(set(c['paragraph'] for c in checks)),'checked_against_final_docx':bool(a.edited),'source_sha256':sources,'errors':errors,'checks':checks};(o/'prose_numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k not in ['checks','source_sha256']})
