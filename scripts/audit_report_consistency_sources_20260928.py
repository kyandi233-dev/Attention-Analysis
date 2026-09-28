"""Read-only statistical/source audit and replacement descriptive Figure 17; no fits."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageChops

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--analysis-repo',type=Path,required=True);a=ap.parse_args()
    root=a.data_root; b=root/'Behavior/formal_v3'; trials=pd.read_csv(b/'trial_metrics.csv',low_memory=False)
    windows=pd.read_csv(b/'probe_window_sensitivity.csv'); coverage=[]
    for win,x in windows.groupby('window_seconds_nominal'):
        n=x.correct_go_rt_opportunities; valid=(n>=2)&(x.go_correct_rt_mean_ms>0)
        expected=x.go_correct_rt_sd_ms/x.go_correct_rt_mean_ms
        assert x.go_correct_rt_cv.notna().equals(valid)
        assert np.allclose(x.loc[valid,'go_correct_rt_cv'],expected[valid],equal_nan=True)
        coverage.append(dict(window_seconds=int(win),total=len(x),cv_valid=int(valid.sum()),mean_correct_go_rt_n=float(n.mean())))
    master=pd.read_csv(root.parent/'derived/subject_modalities_v1/subject_session_questionnaire_master.csv',dtype=str)
    key=lambda s:s.astype(str).str.replace('sub-','',regex=False).str.rstrip('_').str.lstrip('0')
    master['_key']=key(master.single_experiment_id); sessions=key(trials.session_id).unique(); matches=master[master._key.isin(sessions)]
    assert len(matches)==116 and matches.site.eq('北京').all()
    assert trials.groupby('session_id').block_num.nunique().eq(2).all() and trials.condition.eq('B').all()
    # The manuscript's historical figures are exactly the archived pilot PNGs.
    pilot=a.analysis_repo/'docs/030-behavior/history/BBB-v3.0'
    image_checks={}
    for n,fn in [(9,'051-04-Block主效应轨迹.png'),(13,'051-02-RT分布与ECDF.png')]:
        image_checks[str(n)]={'report_sha256':sha(a.results/f'audit15_original_fig{n}.png'),'archive_sha256':sha(pilot/fn)}
        image_checks[str(n)]['byte_equal']=image_checks[str(n)]['report_sha256']==image_checks[str(n)]['archive_sha256']
        left=Image.open(a.results/f'audit15_original_fig{n}.png').convert('RGB');right=Image.open(pilot/fn).convert('RGB')
        image_checks[str(n)]['pixels_equal']=left.size==right.size and ImageChops.difference(left,right).getbbox() is None
        assert image_checks[str(n)]['pixels_equal']
    block=pd.read_csv(b/'block_metrics.csv');cycle=pd.read_csv(b/'cycle_metrics.csv');metric='go_correct_rt_cv'
    wide=block.pivot_table(index='session_id',columns='block_id',values=metric,aggfunc='first').dropna()
    participant=cycle.groupby(['participant_group_id','block_id','cycle_bin'])[metric].mean().reset_index()
    summary=participant.groupby(['block_id','cycle_bin'])[metric].agg(['mean','sem','count']).reset_index()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    fig,axs=plt.subplots(1,2,figsize=(12,5.25),layout='constrained')
    for row in wide.itertuples():axs[0].plot([1,2],[row.B1,row.B2],color='#bbbbbb',alpha=.3,lw=.65)
    axs[0].plot([1,2],wide[['B1','B2']].mean(),color='#d85eae',lw=2.2,marker='o',label='Session-pair mean')
    axs[0].set(xticks=[1,2],xticklabels=['B1','B2'],xlabel='Block',ylabel='Correct Go RT coefficient of variation');axs[0].legend(frameon=False)
    for (name,cur),fmt,color in zip(summary.groupby('block_id'),['o-','s--'],['#297db8','#ff7f0e']):
        axs[1].errorbar(cur.cycle_bin,cur['mean'],yerr=cur['sem'],fmt=fmt,color=color,capsize=3,lw=2,label=name)
    axs[1].set(xticks=range(1,7),xlabel='Stage within block',ylabel='Correct Go RT coefficient of variation');axs[1].legend(title='Block',frameon=False)
    for ax,label in zip(axs,'AB'):ax.text(-.1,1.02,label,transform=ax.transAxes,fontweight='bold',fontsize=17)
    fig.savefig(a.results/'audit15_fig17.png');plt.close(fig)
    # Aggregate-only numeric evidence is safe for the formal repository.
    summary.to_csv(a.results/'audit15_fig17_aggregate.csv',index=False,encoding='utf-8-sig')
    audit={'window_cv_checks':coverage,'formal_cohort':{'sessions':len(matches),'site_counts':matches.site.value_counts().to_dict(),'blocks_per_session':2,'probes':int(trials.is_probe.eq(1).sum()),'trials':len(trials)},'historical_figure_identity':image_checks,'figure17':{'session_pairs':len(wide),'error_bars':'SEM across participants after within-participant aggregation','model_refits':0},'source_sha256':{str(p.relative_to(root)):sha(p) for p in [b/'trial_metrics.csv',b/'probe_window_sensitivity.csv',b/'block_metrics.csv',b/'cycle_metrics.csv']}}
    unit_rows=[]
    for path in (root/'RGB/10_analysis_ready').glob('*/*_motion_qc.parquet'):
        x=pd.read_parquet(path,columns=['body_motion_energy','global_motion_energy_per_sec','global_motion_energy']); valid=x.body_motion_energy.notna()
        unit_rows.append({'finite':int(valid.sum()),'rate_equal':bool(np.allclose(x.loc[valid,'body_motion_energy'],x.loc[valid,'global_motion_energy_per_sec'])),'dimensionless_equal':bool(np.allclose(x.loc[valid,'body_motion_energy'],x.loc[valid,'global_motion_energy']))})
    audit['movement_unit']={'files':len(unit_rows),'finite_rows':sum(x['finite'] for x in unit_rows),'all_equal_per_second':all(x['rate_equal'] for x in unit_rows),'all_equal_dimensionless':all(x['dimensionless_equal'] for x in unit_rows)}
    assert audit['movement_unit']['files']==115 and audit['movement_unit']['all_equal_per_second']
    (a.results/'audit15_source_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
