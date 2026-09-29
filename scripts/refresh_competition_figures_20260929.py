"""Recompute descriptive report figures from frozen inputs; never refit predictors."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args()
    r=a.results; b=a.data_root/'Behavior/formal_v3'; sources=[b/x for x in ['trial_metrics.csv','block_metrics.csv','cycle_metrics.csv']]
    font_manager.fontManager.addfont('C:/Windows/Fonts/simsun.ttc')
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
    palette={'behavior':'#0072B2','ocular':'#009E73','movement':'#E69F00','cardiopulmonary':'#CC79A7'}
    def save(fig,name): fig.savefig(r/f'v5_fig{name}.png',facecolor='white');plt.close(fig)
    # Formal trial distribution: exactly the same valid RT condition as the metrics.
    trials=pd.read_csv(sources[0],low_memory=False)
    # The legacy trial-level go_rt_valid column contains all correct Go RTs;
    # the science metrics apply the lower bound explicitly during aggregation.
    valid=trials.is_no_go.eq(0)&trials.correct.eq(1)&trials.rt.ge(100)
    rt=pd.to_numeric(trials.loc[valid,'rt'],errors='raise').to_numpy()
    assert np.isfinite(rt).all() and (rt>=100).all() and trials.session_id.nunique()==116
    fig,axs=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    axs[0].hist(rt,bins=np.arange(100,max(1201,np.ceil(rt.max()/25)*25+26),25),color=palette['behavior'],edgecolor='white',lw=.4)
    axs[0].set(xlabel='正确 Go 反应时（ms）',ylabel='试次数',xlim=(75,max(1200,rt.max()+20)))
    s=np.sort(rt);axs[1].plot(s,np.arange(1,len(s)+1)/len(s),color=palette['movement'],lw=2)
    axs[1].set(xlabel='正确 Go 反应时（ms）',ylabel='累积比例',ylim=(0,1),xlim=axs[0].get_xlim())
    for ax,letter in zip(axs,'AB'):ax.text(-.13,1.02,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    save(fig,13)
    # Coverage uses the audited independent feature sets; retain original modality colors.
    payload=json.loads((r/'report_payload.json').read_text(encoding='utf-8'))
    labels=payload['labels'];models=payload['models'];fids=[k for k in labels if not k.startswith('modality::')]
    counts=[next(x['n_probes'] for x in models if x['analysis_set_id']=='AS.standalone::'+k) for k in fids]
    fig,ax=plt.subplots(figsize=(11,5.34)); y=np.arange(len(fids))
    ax.barh(y,counts,color=[palette[k.split('.')[0]] for k in fids],height=.64)
    ax.set_yticks(y,[labels[k] for k in fids]);ax.invert_yaxis();ax.set(xlim=(0,2520),xlabel='有效探针数')
    ax.axvline(2320,color='#777777',ls='--',lw=1)
    for yy,n in zip(y,counts):ax.text(n+12,yy,str(n),va='center',fontsize=9)
    ax.legend(handles=[Patch(color=palette[k],label=n) for k,n in zip(palette,['行为','眼部','动作','心肺'])],ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.58,1.015))
    ax.set_axisbelow(True);ax.grid(axis='x',alpha=.22);fig.subplots_adjust(left=.26,right=.98,bottom=.12,top=.89);save(fig,16)
    # Repeated sessions first aggregate within participant. Resample complete participant vectors.
    block=pd.read_csv(sources[1]);cycle=pd.read_csv(sources[2]);metric='go_correct_rt_cv'
    bp=block.groupby(['participant_group_id','block_id'])[metric].mean().unstack()
    cp=cycle.groupby(['participant_group_id','block_id','cycle_bin'])[metric].mean().unstack(['block_id','cycle_bin'])
    assert len(bp)==len(cp)==61 and not bp.isna().any().any() and not cp.isna().any().any()
    rng=np.random.default_rng(20260929);indices=rng.integers(0,61,(20000,61))
    interval=lambda values: np.quantile(values[indices].mean(axis=1),[.025,.975],axis=0)
    bci=interval(bp.to_numpy());cci=interval(cp.to_numpy())
    fig,axs=plt.subplots(1,2,figsize=(11,4.65),layout='constrained')
    for row in bp.to_numpy():axs[0].plot([1,2],row,color='#bbbbbb',alpha=.4,lw=.6)
    bm=bp.mean().to_numpy();axs[0].errorbar([1,2],bm,yerr=[bm-bci[0],bci[1]-bm],fmt='o-',color='#CC79A7',capsize=4,lw=2,label='参与者均值及 95% 置信区间')
    axs[0].set(xticks=[1,2],xticklabels=['B1','B2'],xlabel='任务区块',ylabel='反应时变异系数');axs[0].legend(frameon=False,fontsize=8)
    aggregates=[]
    for name,fmt,color in [('B1','o-',palette['behavior']),('B2','s--',palette['movement'])]:
        ids=[i for i,k in enumerate(cp.columns) if k[0]==name];means=cp.iloc[:,ids].mean().to_numpy()
        axs[1].errorbar(range(1,7),means,yerr=[means-cci[0,ids],cci[1,ids]-means],fmt=fmt,color=color,capsize=3,lw=1.6,label=name)
        for j,i in enumerate(ids):aggregates.append({'block':name,'stage':j+1,'n_participants':61,'mean':means[j],'ci_lower':cci[0,i],'ci_upper':cci[1,i]})
    axs[1].set(xticks=range(1,7),xlabel='区块内阶段',ylabel='反应时变异系数');axs[1].legend(frameon=False)
    for ax,letter in zip(axs,'AB'):ax.text(-.12,1.02,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    save(fig,17);pd.DataFrame(aggregates).to_csv(r/'v5_fig17_aggregate.csv',index=False,encoding='utf-8-sig')
    # Current cardiopulmonary input, separate from behavioral complete-case selection.
    heart_path=r.parent/'repaired_source/m1_cardiopulmonary_taskb_source.csv';heart=pd.read_csv(heart_path)
    heart_stats=[]
    for col,label in [('mmwave_hr_fused_bpm_median','心率（次/min）'),('mmwave_breath_rate_breaths_per_min_median','呼吸率（次/min）')]:
        values=heart[col].dropna();assert len(values)==2200
        heart_stats.append({'label':label,'n':len(values),'mean':values.mean(),'sd':values.std(),'median':values.median(),'q25':values.quantile(.25),'q75':values.quantile(.75),'min':values.min(),'max':values.max()})
    # Use all 100 frozen reference pairs, replacing the obsolete 60-window chart.
    ref=pd.read_csv(a.reference);error=ref.control_fused_hr_bpm-ref.ecg_hr_bpm
    assert len(ref)==100 and np.isclose(error.abs().mean(),10.457079173363644)
    mean=(ref.control_fused_hr_bpm+ref.ecg_hr_bpm)/2;bias=error.mean();sd=error.std()
    fig,ax=plt.subplots(figsize=(8,5.1));ax.scatter(mean,error,s=20,color='#31868A',alpha=.8)
    for value,label,style in [(bias,'差值均值','-'),(bias-1.96*sd,'均值 − 1.96 × 标准差','--'),(bias+1.96*sd,'均值 + 1.96 × 标准差','--')]:
        ax.axhline(value,color='#777777',ls=style,lw=1);ax.text(.98,value+.45,f'{label} = {value:.2f}',ha='right',transform=ax.get_yaxis_transform(),fontsize=9)
    ax.set(xlabel='毫米波与心电参考心率的均值（次/min）',ylabel='毫米波心率 − 心电参考心率（次/min）');fig.tight_layout();save(fig,27)
    audit={'rt_valid_n':len(rt),'rt_total_trials':len(trials),'rt_sessions':116,'rt_min':float(rt.min()),'rt_max':float(rt.max()),'coverage':dict(zip(fids,counts)),
       'heart_stats':heart_stats,'heart_sessions':int(heart.loc[heart.mmwave_hr_fused_bpm_median.notna(),'session_id'].nunique()),
       'figure17':{'participants':61,'bootstrap_samples':20000,'seed':20260929,'aggregation':'within-participant before across-participant mean','paired_difference_mean':float((bp.B2-bp.B1).mean())},
       'reference':{'n':100,'mae':float(error.abs().mean()),'bias':float(bias),'sd':float(sd),'lower_line':float(bias-1.96*sd),'upper_line':float(bias+1.96*sd)},
       'input_hashes':{str(p.relative_to(a.data_root)) if p.is_relative_to(a.data_root) else p.name:sha(p) for p in sources+[heart_path,a.reference]},'predictive_model_refits':0}
    (r/'v5_descriptive_results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
