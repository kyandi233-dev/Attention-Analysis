"""Render replacement report figures from audited aggregate tables only."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABELS={
'behavior.rt_level.median.v1':'反应时中位数','behavior.rt_variability.cv.v1':'反应时变异系数','behavior.rt_trend.theilsen.v1':'反应时趋势','behavior.go_omission.raw.v1':'Go 遗漏率','behavior.nogo_commission.raw.v1':'No-Go 误按率',
'ocular.pupil_level.rseg_hard.rgb_nir_qc.v1':'瞳孔水平','ocular.pupil_variability.rseg_hard.rgb_nir_qc.v1':'瞳孔波动','ocular.pupil_linear_trend.rseg_hard.rgb_nir_qc.v1':'瞳孔线性趋势','ocular.pupil_quadratic_curvature.rseg_hard.rgb_nir_qc.v1':'瞳孔二次曲率','ocular.blink_rate.rgb_event_rate.pre30s.v1':'眨眼频率','movement.body_motion_energy.median.pre30s.v1':'整体动作强度','cardiopulmonary.hr.fused.v1':'毫米波估计心率','cardiopulmonary.br.v1':'毫米波估计呼吸率',
'modality::behavior':'全部行为','modality::ocular':'全部眼部','modality::movement':'全部动作','modality::cardiopulmonary':'全部心肺'}
MOD={'behavior':'行为','ocular':'眼部','movement':'动作','cardiopulmonary':'心肺'}
DEVICE={'M0':'无传感设备','M2':'毫米波','M3':'可见光','M5':'近红外＋可见光','M6':'毫米波＋可见光','M7':'三类设备'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args();r=args.results
    out=r/'figures';out.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/simsun.ttc');plt.rcParams.update({'font.family':'SimSun','font.size':8,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
    models=pd.read_csv(r/'model_results.csv');diag=pd.read_csv(r/'probability_diagnostics.csv');pairs=pd.read_csv(r/'paired_results.csv');base=pd.read_csv(r/'constant_baselines.csv');bins=pd.read_csv(r/'calibration_bins.csv')
    merged=models.merge(diag,on=['analysis_set_id','model_id','membership_type','n_probes','n_participants'],suffixes=('','_diagnostic')).merge(base,on=['analysis_set_id','model_id'])
    def row(sid,mid):return merged[(merged.analysis_set_id==sid)&(merged.model_id==mid)].iloc[0]
    def save(fig,name):
        fig.patch.set_edgecolor('#ffff00');fig.patch.set_linewidth(5);fig.savefig(out/(name+'.png'),facecolor='white');plt.close(fig)
    def performances(rows,labels,name,ratio):
        fig,axs=plt.subplots(1,2,figsize=(6.27,6.27*ratio),gridspec_kw={'wspace':.38});ys=np.arange(len(rows))
        for ax in axs: ax.set_yticks(ys);ax.invert_yaxis();ax.grid(axis='x',color='#eeeeee',lw=.5)
        axs[0].set_yticklabels(labels);axs[1].set_yticklabels([])
        axs[0].plot([x.training_fold_prevalence_log_loss for x in rows],ys,'|',color='#999999',ms=10,label='训练折常数参照')
        axs[0].errorbar([x.participant_equal_log_loss for x in rows],ys,xerr=[[x.participant_equal_log_loss-x.ci_lower for x in rows],[x.ci_upper-x.participant_equal_log_loss for x in rows]],fmt='o',ms=3,color='#356887',lw=.8,label='预测模型')
        axs[1].errorbar([x.participant_macro_auroc for x in rows],ys,xerr=[[x.participant_macro_auroc-x.participant_macro_auroc_ci_lower for x in rows],[x.participant_macro_auroc_ci_upper-x.participant_macro_auroc for x in rows]],fmt='o',ms=3,color='#356887',lw=.8)
        axs[1].axvline(.5,color='#999999',ls='--',lw=.7);axs[0].set_xlabel('参与者等权对数损失');axs[1].set_xlabel('参与者宏平均曲线下面积');axs[0].legend(fontsize=6,frameon=False,loc='upper right');fig.subplots_adjust(left=.22,bottom=.24,top=.96,right=.98);save(fig,name)
    independent=[('AS.behavior_reference','behavior_reference','行为（5项）'),('AS.behavior_plus_ocular','modality::ocular','眼部（5项）'),('AS.behavior_plus_movement','modality::movement','动作（1项）'),('AS.behavior_plus_cardiopulmonary','modality::cardiopulmonary','心肺（2项）'),('AS.sensor_only_joint','sensor_only_joint','传感联合（8项）')]
    performances([row(s,m) for s,m,_ in independent],[n for _,_,n in independent],'fig28',2152650/5731510)
    performances([row('AS.device::'+m,m) for m in DEVICE],list(DEVICE.values()),'fig34',1886585/5731510)
    def forest(frame,labels,name,ratio):
        fig,ax=plt.subplots(figsize=(6.27,6.27*ratio));y=np.arange(len(frame));ax.errorbar(frame.point_estimate,y,xerr=[frame.point_estimate-frame.ci_lower,frame.ci_upper-frame.point_estimate],fmt='o',ms=3,color='#356887',lw=.8);ax.axvline(0,color='#999999',ls='--',lw=.7);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('对数损失差及 95% 置信区间');fig.subplots_adjust(left=.25,right=.97,bottom=.17,top=.96);save(fig,name)
    inc=pairs[pairs.comparison_type.isin(['behavior_modality_increment','modality_increment'])]
    if len(inc)!=3:inc=pairs[pairs.added_model_id.str.startswith('behavior_plus_modality::')]
    forest(inc,[MOD[x.split('::')[-1]] for x in inc.added_model_id],'fig32',2683510/5731510)
    full=pairs[(pairs.analysis_set_id=='AS.full')&(pairs.comparison_type=='full_leave_one_out')];assert len(full)==13
    forest(full,[LABELS[x] for x in full.feature_id],'fig33',3165475/5731510)
    cb=bins[(bins.analysis_set_id=='AS.full')&(bins.model_id=='full')]
    print('calibration columns',list(cb.columns),flush=True)
    xc=next(x for x in ['mean_predicted_probability','mean_probability','mean_prediction'] if x in cb)
    yc=next(x for x in ['observed_positive_rate','observed_rate','observed_fraction'] if x in cb)
    fig,ax=plt.subplots(figsize=(5.25,4.27));ax.plot([0,1],[0,1],ls='--',color='#999999');ax.scatter(cb[xc],cb[yc],s=cb.n_probes/3,color='#356887',alpha=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel='分箱平均预测概率',ylabel='观测任务聚焦比例');fig.tight_layout();save(fig,'fig29')
    coverage=pd.read_csv(r.parent/'repaired/feature_coverage.csv');print('coverage columns',list(coverage.columns),flush=True)
    # The registered standalone sets are the frozen finite/QC-admitted feature coverage.
    standalone=models[models.analysis_set_id.str.startswith('AS.standalone::')].copy();standalone['fid']=standalone.analysis_set_id.str.removeprefix('AS.standalone::');standalone=standalone.set_index('fid').loc[list(LABELS)[:13]].reset_index()
    fig,ax=plt.subplots(figsize=(6.27,3.04));y=np.arange(13);ax.barh(y,standalone.n_probes/2320,color='#7d9fb2');ax.set_yticks(y,[LABELS[x] for x in standalone.fid]);ax.invert_yaxis();ax.set_xlim(0,1.13);ax.set_xlabel('有效探针比例（总体 2,320 个探针）')
    for yy,n in zip(y,standalone.n_probes):ax.text(n/2320+.015,yy,f'{n:,}',va='center',fontsize=7)
    fig.subplots_adjust(left=.26,right=.99,bottom=.17,top=.98);save(fig,'fig16')
    payload={'models':merged.replace({np.nan:None}).to_dict('records'),'pairs':pairs.replace({np.nan:None}).to_dict('records'),'sessions':pd.read_csv(r/'session_results.csv').replace({np.nan:None}).to_dict('records'),'labels':LABELS,'devices':DEVICE,'independent':independent}
    (r/'report_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
