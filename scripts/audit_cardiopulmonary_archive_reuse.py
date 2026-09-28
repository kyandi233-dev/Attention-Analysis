"""Audit archived folds and reconstruct comparison inputs using existing builders.

No model fitting. Private row-level outputs stay under the supplied external root.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.feature_registry import build_feature_comparison_plan, load_registered_features
from attention_pipeline.multimodal_formal.supervised_comparison_sets import build_supervised_comparison_sets, write_supervised_comparison_sets
from attention_pipeline.multimodal_formal.supervised_input import materialize_supervised_input
from attention_pipeline.multimodal_formal.prediction_archive import normalize_task_a_predictions, validate_prediction_archive
from attention_pipeline.supervised_learning.evaluation import evaluate_prediction_archive

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return pd.read_csv(p, encoding='utf-8-sig', low_memory=False)

def equal(a,b):
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)

def audit_run(run, sets):
    m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8-sig'))
    inp=Path(m['provenance']['input_table'])
    assert sha(inp)==m['provenance']['input_sha256'], 'input hash mismatch'
    pred=read(run/'probe_predictions.csv')
    audit=validate_prediction_archive(normalize_task_a_predictions(pred),sets,membership_column=m['membership_type'],requested_analysis_set_ids=[m['analysis_set_id']])
    folds=json.loads((run/'fold_audits.json').read_text(encoding='utf-8-sig'))
    groups=set(read(inp).participant_group_id)
    inner_n=0
    for f in folds:
        tr=set(f['outer_train_group_ids']); te=set(f['outer_test_group_ids'])
        assert not f['failed'] and te=={f['outer_fold_group']}
        assert tr.isdisjoint(te) and tr|te==groups
        ref=f['final_refit']; assert set(ref['preprocessing']['fit_group_ids'])==tr
        assert set(ref['train_group_ids'])==tr and set(ref['test_group_ids'])==te
        scores=f['selection']['candidate_participant_macro_log_loss']
        selected=f['selection']['selected_c']
        assert any(float(k.split('C=')[-1])==selected and np.isclose(v,min(scores.values()),rtol=0,atol=1e-14) for k,v in scores.items())
        for inner in f['selection']['inner_fold_audits']:
            it=set(inner['train_group_ids']); iv=set(inner['validation_group_ids'])
            assert it.isdisjoint(iv) and it|iv==tr and (it|iv).isdisjoint(te)
            assert set(inner['preprocessing']['fit_group_ids'])==it
            assert not inner['failure_by_c']
            inner_n+=1
    assert len(folds)==len(groups)*m['n_models']
    ev=evaluate_prediction_archive(pred)
    old=read(run/'model_evaluation.csv')
    cols=['model_id','n_participants','n_probes','participant_equal_log_loss']
    equal(ev.model_scores[cols].sort_values('model_id').reset_index(drop=True),old[cols].sort_values('model_id').reset_index(drop=True))
    return {'analysis_set_id':m['analysis_set_id'],'input_sha256_verified':True,'outer_folds':len(folds),'inner_folds':inner_n,'models':m['n_models'],'archive_validation':audit,'loss_recomputed':True,'code_sha':m['provenance']['code_sha']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root=a.data_root; repo=Path(__file__).resolve().parents[1]
    oldsets=root/'SupervisedComparisonV3_Cardiopulmonary'; runs=root/'SupervisedRunsV4_Cardiopulmonary'
    config=repo/'configs/supervised_learning_v3_cardiopulmonary.yaml'
    sources={'behavior':root/'Behavior/formal_v3/probe_primary_30s.csv','ocular':root/'FormalScience/Ocular/tables/ocular_probe_features_wide.csv','movement':root/'FormalScience/Movement/tables/movement_probe_descriptive_source.csv','cardiopulmonary':root/'SupervisedComparisonV1/supplementary_cardiopulmonary_m1/m1_cardiopulmonary_taskb_source.csv'}
    cmd=[sys.executable,str(repo/'scripts/build_m1_cardiopulmonary_taskb_source.py'),'--m1-dir',str(root/'mmWave/mmwave_producer_contract_m1_01da845e_20260913'),'--identity-bridge',str(root/'mmWave/mmwave_cardiopulmonary_endpoint_guard_e352dce_20260913/mmwave_cardiopulmonary_taskb_source.csv'),'--probe-universe',str(oldsets/'formal_probe_identity.csv'),'--output',str(a.output/'repaired_source')]
    subprocess.run(cmd,check=True,stdout=(a.output/'source_build.log').open('w',encoding='utf-8'))
    cfg=load_config(config); registry=cfg.data['feature_registry']; plan=build_feature_comparison_plan(load_registered_features(registry))
    frames={k:read(v) for k,v in sources.items()}; meta=read(root/'SupervisedRunsV1/inputs/behavior_probe_metadata.csv')
    original_sets=read(oldsets/'analysis_sets.csv')
    report={'status':'RUNNING','sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in sources.items()},'archive_audits':[],'input_equivalence':[]}
    def save(): (a.output/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    save()
    for run in sorted(runs.glob('AS.*')):
        report['archive_audits'].append(audit_run(run,original_sets)); save(); print('audited',run.name,flush=True)
    built={}
    for kind in ('historical_rebuilt','repaired'):
        cp=frames['cardiopulmonary'] if kind=='historical_rebuilt' else read(a.output/'repaired_source/m1_cardiopulmonary_taskb_source.csv')
        result=build_supervised_comparison_sets(plan=plan,registry_features=registry['features'],behavior_probes=frames['behavior'],ocular_probes=frames['ocular'],movement_probes=frames['movement'],cardiopulmonary_probes=cp)
        write_supervised_comparison_sets(a.output/kind,result)
        built[kind]=(read(a.output/kind/'analysis_sets.csv'),read(a.output/kind/'probe_feature_status.csv'))
    inputs=a.output/'inputs'; inputs.mkdir()
    for sid in sorted(original_sets.analysis_set_id.unique()):
        tables={}
        for kind,(aset,status) in built.items():
            frame=materialize_supervised_input(aset,status,analysis_set_id=sid,membership_type='included_missing_aware',probe_metadata=meta)
            frame['required_outcomes']=json.dumps(['q1_nominal_4class'])
            tables[kind]=frame
        old=materialize_supervised_input(original_sets,read(oldsets/'probe_feature_status.csv'),analysis_set_id=sid,membership_type='included_missing_aware',probe_metadata=meta)
        old['required_outcomes']=json.dumps(['q1_nominal_4class'])
        key=['session_id','block_id','probe_index_in_block']
        def ordered(t): return t.sort_values(key).reset_index(drop=True).sort_index(axis=1)
        equal(ordered(old),ordered(tables['historical_rebuilt']))
        stored=runs/'inputs'/(sid.replace('::','__')+'.csv')
        if stored.exists(): equal(ordered(old),ordered(read(stored)))
        new=tables['repaired']; changed=False
        try: equal(ordered(old),ordered(new))
        except AssertionError: changed=True
        oldkeys=set(map(tuple,old[key].to_numpy())); newkeys=set(map(tuple,new[key].to_numpy()))
        common=new.set_index(key).loc[list(old.set_index(key).index)].reset_index()
        equal(ordered(old),ordered(common))
        new.to_csv(inputs/(sid.replace('::','__')+'.csv'),index=False,encoding='utf-8-sig')
        report['input_equivalence'].append({'analysis_set_id':sid,'historical_rebuild_equivalent':True,'stored_input_verified':stored.exists(),'old_rows':len(old),'new_rows':len(new),'changed':changed,'added_keys':len(newkeys-oldkeys),'removed_keys':len(oldkeys-newkeys),'existing_rows_all_columns_equal':True})
        save(); print('input',sid,len(old),len(new),flush=True)
    expected={'AS.sensor_only_joint','AS.standalone::cardiopulmonary.hr.fused.v1','AS.standalone::cardiopulmonary.br.v1'}
    assert {x['analysis_set_id'] for x in report['input_equivalence'] if x['changed']}==expected
    report['status']='PASS';save()

if __name__=='__main__':main()
