"""Deterministic competition figures from frozen scientific tables, no model fits."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from plot_cardiopulmonary_report_update import LABELS,DEVICE

COLORS={'behavior':'#0072B2','ocular':'#009E73','movement':'#E69F00','cardiopulmonary':'#CC79A7','joint':'#526779'}
FEATURES=['level_mean','variability_mad','linear_slope_per_sec','quadratic_curvature_per_sec2','blink_event_rate_per_min']
NAMES=['瞳孔水平','瞳孔波动','瞳孔线性变化','瞳孔二次曲率','眨眼频率']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args();root=a.root;r=a.results;out=r/'v7_audit';out.mkdir(exist_ok=True)
 for fn in ['simsun.ttc','times.ttf']:font_manager.fontManager.addfont('C:/Windows/Fonts/'+fn)
 plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':12.5,'axes.unicode_minus':True,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.dpi':300,'svg.fonttype':'none'})
 sources={};ledger=[]
 def read(path):
  p=Path(path);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
 def save(fig,n,data,question):
  fig.savefig(out/f'fig{n:02}_v7.png',facecolor='white',dpi=300);fig.savefig(out/f'fig{n:02}_v7.svg',facecolor='white');plt.close(fig)
  data.to_csv(out/f'fig{n:02}_data.csv',index=False,encoding='utf-8-sig');ledger.append({'old_figure':n,'question':question,'plotted_rows':len(data),'data_file':f'fig{n:02}_data.csv','png':f'fig{n:02}_v7.png'})
 def forest(ax,df,labels,point='estimate',lo='ci_low',hi='ci_high',color='ocular',zero=0):
  x=df[point].to_numpy();ax.errorbar(x,np.arange(len(df)),xerr=[x-df[lo].to_numpy(),df[hi].to_numpy()-x],fmt='o',ms=4,capsize=2,color=COLORS[color],lw=1.2)
  ax.set_yticks(range(len(df)),labels);ax.invert_yaxis();ax.axvline(zero,c='#9a9a9a',ls='--',lw=.8)
 models=read(r/'model_results.csv');diag=read(r/'probability_diagnostics.csv');base=read(r/'constant_baselines.csv');pairs=read(r/'paired_results.csv');bins=read(r/'calibration_bins.csv')
 stand=models[models.analysis_set_id.str.startswith('AS.standalone::')].copy();stand['feature_id']=stand.analysis_set_id.str.replace('AS.standalone::','',regex=False);stand=stand.set_index('feature_id').loc[[f for f in LABELS if not f.startswith('modality::')]].reset_index()
 fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ys=np.arange(len(stand));ax.barh(ys,stand.n_probes,color=[COLORS[f.split('.')[0]] for f in stand.feature_id]);ax.set_yticks(ys,[LABELS[f] for f in stand.feature_id]);ax.invert_yaxis();ax.set(xlabel='有效探针数',xlim=(0,2500));ax.axvline(2320,c='#888',ls='--',lw=.8)
 for y,n in zip(ys,stand.n_probes):ax.text(n+12,y,str(n),va='center',fontsize=11)
 ax.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in zip(list(COLORS)[:4],['行为','眼部','动作','心肺'])],ncol=4,loc='lower center',bbox_to_anchor=(.5,1.01));save(fig,15,stand,'各类输入的有效覆盖')
 b=read(root/'Behavior/formal_v3/block_metrics.csv');c=read(root/'Behavior/formal_v3/cycle_metrics.csv');bp=b.groupby(['participant_group_id','block_id']).go_correct_rt_cv.mean().unstack();cp=c.groupby(['participant_group_id','block_id','cycle_bin']).go_correct_rt_cv.mean().unstack(['block_id','cycle_bin']);assert len(bp)==len(cp)==61
 ix=np.random.default_rng(20260929).integers(0,61,(20000,61));ci=lambda v:np.quantile(v[ix].mean(axis=1),[.025,.975],axis=0)
 fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');v=bp.to_numpy();cl=ci(v);means=v.mean(axis=0)
 for row in v:axs[0].plot([1,2],row,c='#c8c8c8',alpha=.35,lw=.5)
 axs[0].errorbar([1,2],means,yerr=[means-cl[0],cl[1]-means],fmt='o-',color=COLORS['behavior'],capsize=3);axs[0].set(xticks=[1,2],xticklabels=['B1','B2'],xlabel='任务区块',ylabel='反应时变异系数')
 v=cp.to_numpy();cl=ci(v);means=v.mean(axis=0);rows=[]
 for block,fmt,color in [('B1','o-',COLORS['behavior']),('B2','s--','#72B7DC')]:
  idx=[i for i,k in enumerate(cp.columns) if k[0]==block];axs[1].errorbar(range(1,7),means[idx],yerr=[means[idx]-cl[0,idx],cl[1,idx]-means[idx]],fmt=fmt,color=color,capsize=3,label=block)
  for j,i in enumerate(idx):rows.append({'block':block,'stage':j+1,'mean':means[i],'ci_lower':cl[0,i],'ci_upper':cl[1,i]})
 axs[1].set(xlabel='区块内阶段',xticks=range(1,7),ylabel='反应时变异系数');axs[1].legend();axs[0].text(-.12,1.03,'A',transform=axs[0].transAxes,fontweight='bold');axs[1].text(-.12,1.03,'B',transform=axs[1].transAxes,fontweight='bold');agg=pd.DataFrame(rows);old=read(r/'v5_fig17_aggregate.csv');assert np.allclose(agg[['mean','ci_lower','ci_upper']],old[['mean','ci_lower','ci_upper']]);save(fig,16,agg,'持续任务的反应时波动变化')
 q=read(root/'Behavior/formal_v3/q1_nominal_models.csv');keys=['go_correct_rt_median_ms','go_correct_rt_cv','go_correct_rt_theilsen_slope_ms_per_s','raw_go_omission_rate','commission_rate','dprime_loglinear'];labs=['反应时中位数','反应时变异系数','反应时趋势','Go 遗漏率','No-Go 误按率','辨别力 d′'];assert set(keys)<=set(q.predictor)
 fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout='constrained');used=[]
 for ax,k in zip(axs,[2,3,4]):
  s=q[q.contrast_category==k].set_index('predictor').loc[keys].reset_index();used.append(s);forest(ax,s,labs,'estimate_per_predictor_sd',color='behavior');ax.set_xlabel('系数及 95% 置信区间');ax.text(.5,1.04,{2:'关注实验但未聚焦任务',3:'任务无关思维',4:'思维空白'}[k],ha='center',transform=ax.transAxes)
 save(fig,17,pd.concat(used),'六个行为维度与三类非任务聚焦报告的关系')
 e=read(root/'Behavior/formal_v3/error_trajectory_summary.csv');fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
 for kind,fmt,col,lab in [('go_omission','o-',COLORS['behavior'],'Go 遗漏'),('nogo_commission','s--','#72B7DC','No-Go 误按')]:
  s=e[e.error_type==kind].sort_values('relative_trial');assert len(s)==7;ax.errorbar(s.relative_trial,s.participant_centered_rt_mean_ms,yerr=1.96*s.participant_centered_rt_sem_ms,fmt=fmt,c=col,capsize=3,label=lab)
 ax.axhline(0,c='#999',lw=.8);ax.axvspan(-.08,.08,color='#eee');ax.set(xlabel='相对于错误事件的试次位置',ylabel='正确 Go 反应时偏离（ms）',xticks=range(-3,4));ax.legend();save(fig,18,e,'错误事件前后的参与者内反应时变化')
 # Frozen within-person estimates, never select points by significance.
 q1=read(root/'FormalScience/Ocular/tables/ocular_q1_models.csv');q2=read(root/'FormalScience/Ocular/tables/ocular_q2_models.csv');links=read(root/'FormalScience/Ocular/tables/ocular_behavior_links.csv')
 FEATURES[:]=list(q2[q2.term=='ocular_within_z'].feature_id)
 assert len(FEATURES)==5
 fig,axs=plt.subplots(1,3,figsize=(10,3.2),layout='constrained');used=[]
 for ax,k in zip(axs,[2,3,4]):
  s=q1[(q1.term=='ocular_within_z')&(q1.contrast_category==k)].set_index('feature_id').loc[FEATURES].reset_index();assert np.allclose(np.exp(s.estimate),s.odds_ratio);forest(ax,s,NAMES,'odds_ratio','or_ci_low','or_ci_high',zero=1);ax.set_xlabel('优势比及 95% 置信区间');ax.text(.5,1.04,{2:'关注实验但未聚焦任务',3:'任务无关思维',4:'思维空白'}[k],ha='center',transform=ax.transAxes);used.append(s)
 save(fig,21,pd.concat(used),'眼部变化与注意内容的人内关系')
 s=q2[q2.term=='ocular_within_z'].set_index('feature_id').loc[FEATURES].reset_index();fig,ax=plt.subplots(figsize=(7,3),layout='constrained');forest(ax,s,NAMES,'odds_ratio','or_ci_low','or_ci_high',zero=1);ax.set_xlabel('累计优势比及 95% 置信区间');save(fig,22,s,'眼部变化与清醒等级的人内关系')
 fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained');used=[]
 for ax,outcome,title in zip(axs.flat,['go_correct_rt_cv','go_correct_rt_theilsen_slope_ms_per_s','raw_go_omission_rate','commission_rate'],['反应时变异系数','反应时趋势（ms/s）','Go 遗漏率','No-Go 误按率']):
  s=links[(links.term=='ocular_within_z')&(links.outcome==outcome)].set_index('feature_id').loc[FEATURES].reset_index();used.append(s);binary=outcome in ['raw_go_omission_rate','commission_rate'];forest(ax,s,NAMES,*(['odds_ratio','or_ci_low','or_ci_high'] if binary else ['estimate','ci_low','ci_high']),zero=1 if binary else 0);ax.set_xlabel('优势比及 95% 置信区间' if binary else '原尺度系数及 95% 置信区间');ax.text(.5,1.04,title,ha='center',transform=ax.transAxes)
 save(fig,24,pd.concat(used),'眼部与近期行为的人内同步')
 m=read(root/'FormalScience/Movement/tables/movement_task_progression.csv');s=m[(m.metric=='body_motion_energy_median')&m.term.isin(['block2','cycle_bin','block2:cycle_bin'])].set_index('term').loc[['block2','cycle_bin','block2:cycle_bin']].reset_index();fig,ax=plt.subplots(figsize=(8,2.5),layout='constrained');forest(ax,s,['区块项（B2）','B1 区块内阶段变化','B2 与 B1 的阶段变化差异'],color='movement');ax.set_xlabel('动作强度系数（s⁻¹）及 95% 置信区间');save(fig,25,s,'身体动作强度随区块和阶段的模型估计')
 ref=root.parent/'derived/mmwave_estimator_improvement_v1_20260912_r1/final/CONTROL_VS_CANDIDATES_100_PROBES_LOCAL_ONLY.csv';df=read(ref)
 error=df.control_fused_hr_bpm-df.ecg_hr_bpm
 assert len(df)==100 and np.isclose(error.abs().mean(),10.457079173363644)
 xy=pd.DataFrame({'pair_mean_bpm':(df.control_fused_hr_bpm+df.ecg_hr_bpm)/2,'difference_bpm':error})
 bias=error.mean();sd=error.std(ddof=1)
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.scatter(xy.pair_mean_bpm,xy.difference_bpm,c=COLORS['cardiopulmonary'],s=16,alpha=.7)
 for value,ls in [(bias,'-'),(bias-1.96*sd,'--'),(bias+1.96*sd,'--')]:ax.axhline(value,c=COLORS['cardiopulmonary'],ls=ls,lw=1);ax.text(.99,value,f'{value:.2f}',transform=ax.get_yaxis_transform(),ha='right',va='bottom')
 ax.set(xlabel='心率估计与心电参考的均值（bpm）',ylabel='心率估计与心电参考之差（bpm）');save(fig,26,xy,'心率估计与心电参考的一致性')
 merged=models.merge(diag,on=['analysis_set_id','model_id','membership_type','n_probes','n_participants'],suffixes=('','_diagnostic')).merge(base,on=['analysis_set_id','model_id'])
 def row(sid,mid):return merged[(merged.analysis_set_id==sid)&(merged.model_id==mid)].iloc[0]
 def performance(keys,n):
  rows=pd.DataFrame([row(s,m) for s,m,_,_ in keys]);fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
  for j,(_,_,label,color) in enumerate(keys):
   x=rows.iloc[j];axs[0].plot(x.training_fold_prevalence_log_loss,j,'|',color='#888',ms=10)
   for ax,p,lo,hi in [(axs[0],'participant_equal_log_loss','ci_lower','ci_upper'),(axs[1],'participant_macro_auroc','participant_macro_auroc_ci_lower','participant_macro_auroc_ci_upper')]:ax.errorbar(x[p],j,xerr=[[x[p]-x[lo]],[x[hi]-x[p]]],fmt='o',c=COLORS[color],capsize=2,ms=4)
  for ax in axs:ax.set_yticks(range(len(keys)),[k[2] for k in keys]);ax.invert_yaxis()
  axs[0].set_xlabel('参与者等权对数损失');axs[1].set_xlabel('参与者平均曲线下面积');axs[1].axvline(.5,c='#999',ls='--',lw=.8);save(fig,n,rows,'各信息类别预测表现' if n==27 else '设备配置的预测表现')
 performance([('AS.behavior_reference','behavior_reference','行为（5 项）','behavior'),('AS.behavior_plus_ocular','modality::ocular','眼部（5 项）','ocular'),('AS.behavior_plus_movement','modality::movement','动作（1 项）','movement'),('AS.behavior_plus_cardiopulmonary','modality::cardiopulmonary','心肺（2 项）','cardiopulmonary'),('AS.sensor_only_joint','sensor_only_joint','传感联合（8 项）','joint')],27)
 performance([('AS.device::'+k,k,v,'joint') for k,v in DEVICE.items()],31)
 s=bins[(bins.analysis_set_id=='AS.full')&(bins.model_id=='full')];assert s.n_probes.sum()==1703;fig,ax=plt.subplots(figsize=(5,4),layout='constrained');ax.plot([0,1],[0,1],'--',c='#aaa');ax.scatter(s.mean_predicted_probability,s.observed_positive_rate,s=120*s.n_probes/s.n_probes.max(),c=COLORS['joint']);ax.set(xlabel='分箱平均预测概率',ylabel='观察到的任务聚焦比例',xlim=(0,1),ylim=(0,1));save(fig,28,s,'完整组合的概率校准')
 for n,s in [(29,pairs[pairs.added_model_id.str.startswith('behavior_plus_modality::')]),(30,pairs[(pairs.analysis_set_id=='AS.full')&(pairs.comparison_type=='full_leave_one_out')])]:
  s=s.copy();assert len(s)==(3 if n==29 else 13);fig,ax=plt.subplots(figsize=(8,2.5 if n==29 else 5),layout='constrained');labels=[]
  for j,(_,x) in enumerate(s.iterrows()):
   fid=x.feature_id;mod=x.added_model_id.split('::')[-1] if n==29 else fid.split('.')[0];label={'ocular':'眼部','movement':'动作','cardiopulmonary':'心肺'}[mod] if n==29 else LABELS[fid];labels.append(label);ax.errorbar(x.point_estimate,j,xerr=[[x.point_estimate-x.ci_lower],[x.ci_upper-x.point_estimate]],fmt='o',c=COLORS[mod],capsize=2,ms=4)
  ax.set_yticks(range(len(s)),labels);ax.invert_yaxis();ax.axvline(0,c='#999',ls='--',lw=.8);ax.set_xlabel('对数损失改善及 95% 置信区间' if n==29 else '移除后的模型损失减去完整组合损失');save(fig,n,s,'行为基础上的传感增量' if n==29 else '完整组合的预定义逐项贡献')
 (out/'plot_evidence.json').write_text(json.dumps({'source_sha256':sources,'figures':ledger,'palette':COLORS,'models_refitted':0},ensure_ascii=False,indent=2),encoding='utf-8')
 print('PLOTS',len(ledger))
if __name__=='__main__':main()
