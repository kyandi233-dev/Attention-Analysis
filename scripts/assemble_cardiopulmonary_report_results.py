"""Reuse validated archives and assemble report tables without training models."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from audit_cardiopulmonary_archive_reuse import audit_run,read,sha
from attention_pipeline.supervised_learning.evaluation import evaluate_prediction_archive,paired_log_loss_increment,binary_probe_log_loss
from attention_pipeline.supervised_learning.probability_diagnostics import build_probability_diagnostics
from attention_pipeline.supervised_learning.trajectory import summarize_session_discrimination

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--repair-root',type=Path,required=True);args=ap.parse_args()
    root=args.data_root; repair=args.repair_root; out=repair/'report_results';out.mkdir(exist_ok=False)
    sets=read(repair/'repaired/analysis_sets.csv')
    candidates={}
    for rr in [root/'SupervisedRunsV1',root/'SupervisedRunsV4_Cardiopulmonary',repair/'reruns']:
        for run in sorted(rr.glob('*__included_missing_aware')):
            m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8-sig'))
            candidates[m['analysis_set_id']]=run
    expected=set(sets.analysis_set_id.unique());assert set(candidates)==expected,(set(candidates)-expected,expected-set(candidates))
    reports=[];losses=[];pairs=[];diagnostics=[];bins=[];sessions=[];baselines=[];lineage=[]
    for sid,run in sorted(candidates.items()):
        print('assemble',sid,flush=True)
        report=audit_run(run,sets);reports.append(report)
        pred=read(run/'probe_predictions.csv');ev=evaluate_prediction_archive(pred)
        old_bs=json.loads((run/'participant_bootstrap.json').read_text())
        for b in ev.bootstrap_records:
            old=next(v for v in old_bs if v['model_id']==b['model_id'])
            for k in ['point_estimate','ci_lower','ci_upper']: assert np.isclose(b[k],old[k],atol=1e-12,rtol=0)
        score=ev.model_scores.merge(pd.DataFrame(ev.bootstrap_records)[['model_id','ci_lower','ci_upper']],on='model_id')
        score['n_sessions']=score.model_id.map(pred.groupby('model_id').session_id.nunique())
        losses.append(score)
        old_pair=json.loads((run/'paired_increment_bootstrap.json').read_text())
        for b in old_pair:
            if b.get('n_feature_absent_outer_folds',0):raise ValueError('estimability-specific pair needs explicit subset')
            result=paired_log_loss_increment(pred[pred.model_id==b['baseline_model_id']],pred[pred.model_id==b['added_model_id']])
            for k in ['point_estimate','ci_lower','ci_upper']: assert np.isclose(result.bootstrap[k],b[k],atol=1e-12,rtol=0),(sid,k)
            pairs.append({**b,'source_run':str(run),'interval_recomputed':True})
        session=summarize_session_discrimination(pred)
        for model,g in session.groupby('model_id'):
            valid=g[g.status=='estimable'];sessions.append({'analysis_set_id':sid,'model_id':model,'n_sessions':len(g),'n_estimable':len(valid),'n_single_class':len(g)-len(valid),'mean_auroc':valid.auroc.mean(),'median_auroc':valid.auroc.median()})
        # Reconstruct each outer-training constant baseline on exactly this model's sample.
        for model,g in pred.groupby('model_id'):
            pieces=[]
            for pid,test in g.groupby('participant_group_id'):
                train=g[g.participant_group_id!=pid]; p=float(train.q1_binary.mean())
                pieces.append(float(binary_probe_log_loss(test.q1_binary.to_numpy(),np.full(len(test),p)).mean()))
            baselines.append({'analysis_set_id':sid,'model_id':model,'training_fold_prevalence_log_loss':float(np.mean(pieces))})
        # Recompute diagnostics for report-facing models; all others keep audited archived values.
        diagpath=(root/'SupervisedRunsV1/probability_diagnostics/probability_diagnostics.csv' if 'SupervisedRunsV1' in str(run) else root/'SupervisedRunsV4_Cardiopulmonary/probability_diagnostics/probability_diagnostics.csv')
        if sid in ['AS.full','AS.sensor_only_joint','AS.behavior_plus_cardiopulmonary'] or str(run).startswith(str(repair)):
            subset=pred if sid!='AS.full' else pred[pred.model_id=='full']
            dg,cb=build_probability_diagnostics(subset)
            if sid=='AS.full':
                archived=read(diagpath);dg=pd.concat([dg,archived[(archived.analysis_set_id==sid)&(archived.model_id!='full')]],ignore_index=True)
            bins.append(cb)
        else:
            archived=read(diagpath);dg=archived[archived.analysis_set_id==sid].copy()
        assert set(dg.model_id)==set(pred.model_id)
        diagnostics.append(dg)
        lineage.append({'analysis_set_id':sid,'source_run':str(run),'prediction_sha256':sha(run/'probe_predictions.csv'),'disposition':'rerun_changed_input' if str(run).startswith(str(repair)) else 'reuse_verified_archive'})
        (out/'archive_validation.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    products={'model_results':pd.concat(losses,ignore_index=True),'paired_results':pd.DataFrame(pairs),'probability_diagnostics':pd.concat(diagnostics,ignore_index=True),'calibration_bins':pd.concat(bins,ignore_index=True),'session_results':pd.DataFrame(sessions),'constant_baselines':pd.DataFrame(baselines),'result_lineage':pd.DataFrame(lineage)}
    for name,df in products.items():df.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    assert len(products['model_results'])==56
    (out/'manifest.json').write_text(json.dumps({'status':'PASS','model_rows':56,'analysis_sets':25,'model_identities':int(products['model_results'].model_id.nunique()),'paired_rows':len(pairs),'outer_folds':sum(x['outer_folds'] for x in reports),'files':{n:{'sha256':sha(out/(n+'.csv')),'rows':len(d)} for n,d in products.items()}},indent=2),encoding='utf-8')

if __name__=='__main__': main()
