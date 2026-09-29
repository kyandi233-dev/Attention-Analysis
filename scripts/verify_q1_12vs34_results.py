"""Independently recompute aggregate metrics and make a target-comparison figure."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,balanced_accuracy_score

MAIN=[('行为','AS.behavior_reference','behavior_reference'),('眼部','AS.behavior_plus_ocular','modality::ocular'),('动作','AS.behavior_plus_movement','modality::movement'),('心肺','AS.behavior_plus_cardiopulmonary','modality::cardiopulmonary'),('八项传感联合','AS.sensor_only_joint','sensor_only_joint'),('十三项完整组合','AS.full','full')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repair-root',type=Path,required=True);a=ap.parse_args();r=a.root
    manifest=json.loads((r/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='PASS'
    new=pd.read_csv(r/'model_results.csv');old=pd.read_csv(r/'primary_target_results.csv');pairs=pd.read_csv(r/'paired_results.csv');frozen=pd.read_csv(a.repair_root/'report_results/model_results.csv');diag=pd.read_csv(a.repair_root/'report_results/probability_diagnostics.csv')
    assert len(new)==len(old)==56 and len(pairs)==28 and len(manifest['completed_sets'])==25
    tested=[];participant_losses={};label_counts=[];confusions={}
    for entry in manifest['completed_sets']:
        sid=entry['analysis_set_id'];safe=str(sid).replace(':','_');path=r/safe/'probe_predictions.csv';assert sha(path)==entry['prediction_sha256']
        pred=pd.read_csv(path);inp=pd.read_csv(a.repair_root/'inputs'/f'{safe}.csv')
        assert sha(a.repair_root/'inputs'/f'{safe}.csv')==entry['input_sha256']
        assert 'p_q1_equals_1' not in pred and pred.p_q1_in_1_2.between(0,1).all() and not pred.model_failed.any()
        label_counts.append({'analysis_set_id':sid,'n_probes':len(inp),'n_participants':inp.participant_group_id.nunique(),**{f'q1_{i}':int(inp.q1_nominal_4class.eq(i).sum()) for i in [1,2,3,4]}})
        for mid,g in pred.groupby('model_id'):
            assert len(g)==len(inp) and not g.duplicated(['session_id','block_id','probe_event_id']).any()
            assert (g.q1_binary==g.q1_nominal_4class.isin([1,2]).astype(int)).all()
            score=new[(new.analysis_set_id==sid)&(new.model_id==mid)].iloc[0]
            weights=1/g.groupby('participant_group_id').participant_group_id.transform('size').to_numpy(float)
            y=g.q1_binary.to_numpy(int);p=g.p_q1_in_1_2.to_numpy(float)
            hat=p>=.5
            confusions[(sid,mid)]={'true_positive':int(((y==1)&hat).sum()),'false_negative':int(((y==1)&~hat).sum()),'true_negative':int(((y==0)&~hat).sum()),'false_positive':int(((y==0)&hat).sum())}
            assert np.isclose(log_loss(y,p,sample_weight=weights,labels=[0,1]),score.loss,atol=1e-12,rtol=0)
            assert np.isclose(brier_score_loss(y,p,sample_weight=weights),score.brier,atol=1e-12,rtol=0)
            auc=[];ba=[];losses={};priors=[]
            for pid,t in g.groupby('participant_group_id'):
                losses[pid]=log_loss(t.q1_binary,t.p_q1_in_1_2,labels=[0,1])
                training=inp[inp.participant_group_id!=pid];prior=training.q1_nominal_4class.isin([1,2]).mean();priors.append(log_loss(t.q1_binary,np.full(len(t),prior),labels=[0,1]))
                if t.q1_binary.nunique()==2:auc.append(roc_auc_score(t.q1_binary,t.p_q1_in_1_2));ba.append(balanced_accuracy_score(t.q1_binary,t.p_q1_in_1_2.ge(.5)))
            assert np.isclose(np.mean(priors),score.baseline_loss,atol=1e-12,rtol=0)
            assert len(auc)==score.n_two_class_participants
            assert np.isclose(np.mean(auc),score.auroc,atol=1e-12,rtol=0)
            assert np.isclose(np.mean(ba),score.balanced_accuracy,atol=1e-12,rtol=0)
            participant_losses[(sid,mid)]=pd.Series(losses).sort_index();tested.append((sid,mid))
    for _,row in pairs.iterrows():
        d=participant_losses[(row.analysis_set_id,row.baseline_model_id)]-participant_losses[(row.analysis_set_id,row.added_model_id)]
        assert np.isclose(d.mean(),row.improvement,atol=1e-12,rtol=0)
        rng=np.random.default_rng(20260830);b=d.to_numpy()[rng.integers(0,len(d),(1000,len(d)))].mean(axis=1)
        assert np.allclose(np.quantile(b,[.025,.975]),[row.ci_low,row.ci_high],atol=1e-12,rtol=0)
    for _,row in old.iterrows():
        f=frozen[(frozen.analysis_set_id==row.analysis_set_id)&(frozen.model_id==row.model_id)].iloc[0];dg=diag[(diag.analysis_set_id==row.analysis_set_id)&(diag.model_id==row.model_id)].iloc[0]
        assert np.isclose(row.loss,f.participant_equal_log_loss,atol=1e-12,rtol=0)
        assert np.isclose(row.auroc,dg.participant_macro_auroc,atol=1e-12,rtol=0)
    pd.DataFrame(label_counts).to_csv(r/'label_distribution.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for label,sid,mid in MAIN:
        for frame in [old,new]:
            row=frame[(frame.analysis_set_id==sid)&(frame.model_id==mid)].iloc[0].to_dict();row['model_label']=label
            if row['target']=='12_vs_34':row.update(confusions[(sid,mid)])
            rows.append(row)
    pd.DataFrame(rows).to_csv(r/'main_model_comparison.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/simsun.ttc');plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={'wspace':.42})
    for frame,color,offset,label in [(old,'#0072B2',-.12,'原分组：1 对 2、3、4'),(new,'#E69F00',.12,'新分组：1、2 对 3、4')]:
        sel=pd.DataFrame([frame[(frame.analysis_set_id==sid)&(frame.model_id==mid)].iloc[0] for _,sid,mid in MAIN]);y=np.arange(6)+offset
        for ax,col in zip(axes,['auroc','improvement']):
            ax.errorbar(sel[col],y,xerr=np.vstack([sel[col]-sel[col+'_ci_low'],sel[col+'_ci_high']-sel[col]]),fmt='o',ms=4,capsize=2,color=color,label=label)
    for ax in axes:ax.set_yticks(range(6),[x[0] for x in MAIN]);ax.invert_yaxis();ax.grid(axis='x',color='#eeeeee')
    axes[0].axvline(.5,color='#777777',ls='--',lw=.8);axes[0].set_xlabel('受试者工作特征曲线下面积');axes[0].set_title('A  参与者内的类别排序')
    axes[1].axvline(0,color='#777777',ls='--',lw=.8);axes[1].set_xlabel('类别比例基线损失 − 模型损失');axes[1].set_title('B  相对各自类别比例基线的改善')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False);fig.subplots_adjust(left=.15,right=.98,top=.85,bottom=.15);fig.savefig(r/'target_comparison.png',facecolor='white');plt.close(fig)
    result={'status':'PASS','models_verified':len(tested),'pairs_verified':28,'label_mapping_verified':True,'primary_points_match_frozen_archive':True,'outer_folds':manifest['outer_folds'],'inner_folds':manifest['inner_folds'],'source_code_commits':[manifest['code_sha']]+[e['code_sha'] for e in manifest.get('resumed_executions',[])],'output_files':{name:sha(r/name) for name in ['model_results.csv','primary_target_results.csv','paired_results.csv','main_model_comparison.csv','label_distribution.csv','target_comparison.png']}}
    (r/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
