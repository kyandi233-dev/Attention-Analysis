"""Surgical OOXML report copy: preserve package, highlight every changed text run.

Run with the bundled document runtime. Inputs are aggregate results and original DOCX.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,zipfile,difflib,re
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from lxml import etree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(v,n=4):return f'{v:.{n}f}'.replace('-', '−')
def ci(d,prefix='',n=4):return '['+f(d[prefix+'ci_lower'],n)+', '+f(d[prefix+'ci_upper'],n)+']'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=json.loads((a.results/'report_payload.json').read_text(encoding='utf-8'));d=Document(a.input);paras=list(d.paragraphs);changes=[];media={}
    def row(s,m):return next(x for x in data['models'] if x['analysis_set_id']==s and x['model_id']==m)
    def pair(fid):return next(x for x in data['pairs'] if x['analysis_set_id']=='AS.full' and x['feature_id']==fid)
    def sess(s,m):return next(x for x in data['sessions'] if x['analysis_set_id']==s and x['model_id']==m)
    def settext(p,text,label,all_yellow=False):
        old=p.text
        if old==text and not all_yellow:return
        runs=list(p.runs); spans=[];start=0
        for run in runs:spans.append((start,start+len(run.text),deepcopy(run._r.rPr)));start+=len(run.text)
        default=next((x[2] for x in spans if x[2] is not None),None)
        def styleat(i):return next((st for lo,hi,st in spans if lo<=i<hi),default)
        for ch in list(p._p):
            if ch.tag!=qn('w:pPr'):p._p.remove(ch)
        def put(t,st,yellow):
            if not t:return
            rr=p.add_run(t)
            if st is not None:rr._r.insert(0,deepcopy(st))
            if yellow:
                pr=rr._r.get_or_add_rPr();h=pr.find(qn('w:highlight'))
                if h is None:h=OxmlElement('w:highlight');pr.append(h)
                h.set(qn('w:val'),'yellow')
        if all_yellow:put(text,default,True)
        else:
            for op,i,j,k,l in difflib.SequenceMatcher(None,old,text,autojunk=False).get_opcodes():
                if op=='equal':
                    for lo,hi,st in spans:
                        left=max(i,lo);right=min(j,hi)
                        if right>left:put(old[left:right],st,False)
                elif op in ['insert','replace']:put(text[k:l],styleat(i),True)
        changes.append({'location':label,'before':old,'after':text,'highlight':'yellow'})
    def p(i,text):
        # The original report uses ungrouped four-digit counts in running text.
        text=re.sub(r'(?<=\d),(?=\d{3}(?!\d))','',text)
        settext(paras[i],text,'paragraph '+str(i))
    def cell(c,text,label,all_yellow=False):
        settext(c.paragraphs[0],str(text),label,all_yellow)
        for extra in c.paragraphs[1:]:
            if extra.text:extra._p.getparent().remove(extra._p)
    def table(idx,rows):
        t=d.tables[idx];original_n=len(t.rows);template=deepcopy(t.rows[-1]._tr)
        while len(t.rows)<len(rows):t._tbl.append(deepcopy(template))
        while len(t.rows)>len(rows):t._tbl.remove(t.rows[-1]._tr)
        for ri,vals in enumerate(rows):
            for j,v in enumerate(vals):cell(t.rows[ri].cells[j],v,f'table {idx} row {ri} col {j}',ri>=original_n)
        if idx in [33,46,47]:
            for rr in t.rows[1:]:
                for cc in rr.cells:
                    for pp in cc.paragraphs:pp.paragraph_format.keep_with_next=False
            prev=t._tbl.getprevious()
            while prev is not None and prev.tag==qn('w:p'):
                pp=Paragraph(prev,d);pp.paragraph_format.keep_with_next=True
                if pp.text:
                    settext(pp,pp.text,f'table {idx} heading pagination',True);break
                prev=prev.getprevious()
    full=row('AS.full','full');sensor=row('AS.sensor_only_joint','sensor_only_joint');hr=row('AS.standalone::cardiopulmonary.hr.fused.v1','standalone::cardiopulmonary.hr.fused.v1');br=row('AS.standalone::cardiopulmonary.br.v1','standalone::cardiopulmonary.br.v1')
    p(240,'仅传感信息联合模型使用四项瞳孔指标、眨眼频率、动作强度以及毫米波估计心率和呼吸率，共 8 项指标，用于评价减少任务按键依赖时的预测表现。完整组合包含全部 13 项指标，逐类或逐项移除时，均在该次比较的共同样本上训练和评价。仅传感联合模型与完整组合移除行为后的模型使用同一组传感指标，各自的有效样本按对应纳入条件确定，并分别报告数据覆盖。独立传感联合不附加行为指标完整的条件。')
    p(238,'信息类别层评价行为、眼部、动作和心肺估计对注意内容报告的预测能力，并比较在行为基础上加入各类传感信息的变化。心肺类别与其行为增量共同采用行为及心肺有效集合；心率、呼吸率单指标另按各自有效值形成独立集合。')
    for ri in [7,8]:
        cell(d.tables[8].rows[ri].cells[4],'110',f'table 8 row {ri} sessions');cell(d.tables[8].rows[ri].cells[5],'2,200',f'table 8 row {ri} probes')
    cell(d.tables[8].rows[13].cells[4],'116','table 8 No-Go sessions');cell(d.tables[8].rows[13].cells[5],'2,320','table 8 No-Go probes')
    independent=data['independent']
    rows=[[c.text for c in d.tables[14].rows[0].cells]]
    for sid,mid,label in independent:
        x=row(sid,mid);rows.append([label,f(x['participant_equal_log_loss'])+' '+ci(x),f(x['participant_macro_auroc'])+' '+ci(x,'participant_macro_auroc_'),f(x['participant_macro_brier']),f"{x['n_participants']}/{x['n_sessions']}/{x['n_probes']}",str(x['n_participants_auroc_estimable'])])
    table(14,rows)
    p(363,'注：损失、布里尔分数及可估计参与者的曲线下面积按参与者等权汇总。单一类别参与者纳入损失评价，曲线下面积仅由同时具有两类标签者计算。心肺类别采用行为及心肺均有效的 58 人、110 场、2,198 个探针；独立传感联合为 57 人、98 场、1,705 个探针。区间反映固定折外预测下的参与者抽样不确定性。不同集合的分数不能作为直接性能排名。')
    p(364,f"行为和眼部模型的曲线下面积区间高于 0.5。8 项独立传感联合的曲线下面积为 {f(sensor['participant_macro_auroc'])}，95% 置信区间为 {ci(sensor,'participant_macro_auroc_')}，包含 0.5；动作与心肺类别的区间也包含 0.5。当前结果未显示传感联合具有明确高于机会水平的跨参与者排序能力。")
    p(365,'无输入参照使用各外层训练折的正例率作为常数概率，并按参与者等权评价。行为模型的损失为 0.6237，低于参照的 0.6438。8 项独立传感联合的损失为 '+f(sensor['participant_equal_log_loss'])+'，其对应参照为 '+f(sensor['training_fold_prevalence_log_loss'])+'。本项比较呈现各模型相对训练折常数概率的数值变化，成对改善区间尚未估计。')
    p(368,'注：左图为各模型及同一样本训练折常数参照的损失，右图为参与者宏平均曲线下面积及 95% 置信区间。心肺类别为 58 人、2,198 个探针；8 项独立传感联合为 57 人、1,705 个探针。各行按自身集合解释。')
    rows=[[c.text for c in d.tables[15].rows[0].cells]]
    for sid,mid,label in independent:
        x=sess(sid,mid);rows.append([label,str(x['n_estimable']),str(x['n_single_class']),f(x['mean_auroc']),f(x['median_auroc'])])
    table(15,rows);s=sess('AS.sensor_only_joint','sensor_only_joint')
    p(370,f"行为模型在 83 个同时包含两类报告的场次中，曲线下面积均值为 0.6919，中位数为 0.7262；眼部模型在 72 场中的均值为 0.5697，中位数为 0.5656。8 项独立传感联合在 {s['n_estimable']} 个可估计场次中的均值为 {f(s['mean_auroc'])}，中位数为 {f(s['median_auroc'])}。")
    p(375,'行为模型的校准斜率为 0.645，95% 置信区间为 [0.157, 1.234]。眼部模型为 −0.034，[−2.180, 0.439]。8 项独立传感联合的斜率原始估计为 '+f(sensor['calibration_slope'],3)+'，区间为 '+ci(sensor,'calibration_slope_',3)+'；其判别力区间包含 0.5，按预先规定的诊断规则不作斜率方向的实质解释。')
    p(376,'校准回归描述合并探针上的概率对应关系，参与者宏平均曲线下面积描述参与者内部的排序；两者的汇总尺度不同。校准诊断不参与特征或参数选择。区间由参与者簇自助法形成，每次仅重新拟合校准诊断回归，不重新训练预测模型。')
    p(377,'13 项完整组合的校准斜率为 '+f(full['calibration_slope'],3)+'，95% 置信区间为 '+ci(full,'calibration_slope_',3)+'；截距为 '+f(full['calibration_intercept'],3)+'，区间为 '+ci(full,'calibration_intercept_',3)+'。斜率区间上限略低于理想值 1。参与者等权的平均预测概率与实际正例率之差为 '+f(full['participant_macro_calibration_in_the_large'],5)+'，区间为 '+ci(full,'participant_macro_calibration_in_the_large_',3)+'。平均偏差与分箱校准分别反映总体及不同概率范围内的对应程度。')
    p(379,'图 29 十三项完整组合的折外概率校准')
    p(382,'在行为及心肺均有效的 58 人、2,198 个探针上，两项心肺估计类别模型的对数损失为 0.6347，95% 置信区间为 [0.5843, 0.6853]，曲线下面积为 0.4691，[0.4159, 0.5240]。在不附加行为完整性条件的 58 人、110 场、2,200 个探针上，心率单指标损失为 '+f(hr['participant_equal_log_loss'])+' '+ci(hr)+'，呼吸率单指标为 '+f(br['participant_equal_log_loss'])+' '+ci(br)+'。类别模型与两个单指标集合的样本条件不同。')
    p(383,'在行为基础上加入心肺估计的改善量为 0.00050，95% 置信区间为 [−0.00576, 0.00692]；单独加入呼吸率为 −0.00166，[−0.00284, −0.00062]，单独加入心率为 0.00218，[−0.00361, 0.00836]。这些成对结果采用 2,198 个共同探针。含毫米波的设备结果统一列于 5.8。心肺估计已满足本次预测的来源与时间条件，其生理准确性仍需独立同步参考验证；5.5 的测量评估与开发对照保留各自批次和分母。')
    p(388,paras[388].text+' 另在相同的 61 人、116 场、2,306 个探针上完整配对重训 20 s 与 30 s 行为模型，损失差（30 s − 20 s）为 −0.007751，95% 置信区间为 [−0.015151, 0.002301]。未发现明确差异，也不能据此认定两窗口等效或 30 s 最优。')
    p(390,paras[390].text.replace('在完整组合的','在不含心肺估计的原十一项组合的'))
    p(391,paras[391].text.replace('沿用二分类分析的比较样本，采用完整特征集合','沿用原十一项二分类分析的比较样本，采用十一项特征集合').replace('在完整组合的','在该十一项组合的'))
    p(737,paras[737].text.replace('完整组合的 60 名','原十一项组合的 60 名'))
    cat=pair('modality::behavior');mov=pair('modality::movement')
    p(415,'完整组合包含五项行为、五项眼部、一项动作和两项心肺估计。在相同的 57 人、98 场、1,703 个探针上，完整模型损失为 '+f(full['participant_equal_log_loss'],6)+'。移除行为后的损失增加 '+f(cat['point_estimate'],5)+'，95% 置信区间为 '+ci(cat,n=5)+'；移除动作后的差值为 '+f(mov['point_estimate'],5)+' '+ci(mov,n=5)+'。移除眼部或心肺类别的差值区间包含零。')
    rows=[[c.text for c in d.tables[19].rows[0].cells]]
    for k in ['behavior','ocular','movement','cardiopulmonary']:
        x=pair('modality::'+k);rows.append([data['labels']['modality::'+k],f(x['point_estimate'],5),ci(x,n=5),'57','1703'])
    table(19,rows)
    p(417,'注：四行均以同一十三项完整模型及 57 人、1,703 个探针为基准。差值为移除后的损失减去完整模型损失，正值表示保留该类信息时损失较低。移除行为后的八项模型仍受完整组合共同样本条件约束，与 1,705 个探针的独立传感联合分开报告。')
    p(418,'行为在其他传感指标已纳入时仍表现出正向条件预测价值，区间下限接近零。动作类别的移除差值为负且区间不包含零，与行为加入动作比较的方向一致。眼部和心肺类别未显示明确的条件增益。各项区间未作整组多重比较校正。')
    ids=['behavior.nogo_commission.raw.v1','behavior.go_omission.raw.v1','behavior.rt_trend.theilsen.v1','behavior.rt_variability.cv.v1']
    p(424,'完整组合移除误按率后，损失增加 '+f(pair(ids[0])['point_estimate'],5)+'，95% 置信区间为 '+ci(pair(ids[0]),n=5)+'。移除原始遗漏率、反应时趋势和变异系数后，损失变化分别为 '+ '、'.join(f(pair(fid)['point_estimate'],5)+' '+ci(pair(fid),n=5) for fid in ids[1:])+'。其中，反应时变异系数的区间包含零。这些结果表明，在当前完整组合和正则化模型中，不同行为指标的条件贡献有所差异。行为类别的总体价值与各单项指标的作用分别呈现。')
    p(425,paras[425].text.replace('全部 22 条成对结果见结果附表C','全部 28 条成对结果见附录 B.2'))
    p(426,paras[426].text+' 另在原十一项组合的 60 人、1,775 个探针上完成预定义精简流程的嵌套验证：完整模型损失为 0.618033，精简流程为 0.618623，差值（完整 − 精简）为 −0.000589，95% 置信区间为 [−0.009133, 0.008061]。该补充分析未显示精简改善，其结果不代表十三项组合已完成同样的精简检验。')
    p(428,'设备评价同时考虑可形成的指标、有效覆盖及预测表现。六种配置均保留五项任务行为输入，传感部分分别为无传感设备、毫米波、可见光、近红外与可见光、毫米波与可见光、三类设备联合。相应总输入数为 5、7、7、11、9 和 13 项。')
    devinfo={'M0':'五项行为（5项）','M2':'行为＋心肺（7项）','M3':'行为＋眨眼＋动作（7项）','M5':'行为＋眼部＋动作（11项）','M6':'行为＋眨眼＋动作＋心肺（9项）','M7':'行为＋眼部＋动作＋心肺（13项）'}
    rows=[[c.text for c in d.tables[20].rows[0].cells]]
    for mid,label in data['devices'].items():
        x=row('AS.device::'+mid,mid);rows.append([label,devinfo[mid],str(x['n_participants']),str(x['n_probes']),f(x['participant_equal_log_loss']),f(x['participant_macro_auroc'])])
    table(20,rows)
    p(434,'六种配置分别保留全部 2,320 个探针的 '+ '、'.join(f(row('AS.device::'+m,m)['n_probes']/2320*100,2)+'%' for m in data['devices'])+'。完整三设备组合为 57 人、98 场、1,703 个探针。各配置的分数同时受到输入与样本构成影响，不能据此作设备优劣排名。')
    p(435,'主要瞳孔指标结合近红外图像与可见光眨眼时点完成质量处理，因此采用双摄像头配置评价瞳孔信息。独立近红外配置需要替代指标及相应测量验证。六种配置均按实际输入数和各自样本报告；传感信息在行为之上的增量由共同样本成对比较评价。')
    p(522,'多模态比较为选择信息组合提供了具体依据。加入眼部或心肺后的损失改善区间包含零；加入动作后损失小幅增加。十三项完整组合移除动作的差值为负且区间不包含零，移除眼部或心肺的区间包含零。八项独立传感联合的排序区间包含机会水平，当前未发现其稳定区分能力。系统改进需关注指标质量和信息互补性，并在独立样本中继续检验。')
    p(526,paras[526].text.replace('完整组合采用1775个','十三项完整组合采用1703个、八项独立传感联合采用1705个'))
    p(657,'B.2 全部 28 项正式成对预测比较')
    p(662,'行为模型在不同共同样本上分别训练，因此表中保留对应的多行结果。全部 56 行包含 53 个模型及其对应的 25 个分析集合。对数损失评价包含单一标签参与者，曲线下面积由同时具有两类标签者汇总。完整组合包含 13 项指标，独立传感联合包含 8 项；原 11 项组合的四分类与精简验证另作补充报告。附录 F 与本表采用同一结果来源。')
    def feature(fid):return data['labels'].get(fid,fid)
    def name(mid):
        if mid in data['devices']:return data['devices'][mid]+'配置'
        if mid=='full':return '完整组合（13项）'
        if mid=='sensor_only_joint':return '独立传感联合（8项）'
        if mid=='behavior_reference':return '行为参照'
        if mid.startswith('full_minus_modality::'):return '完整组合−'+feature('modality::'+mid.split('::')[-1])
        if mid.startswith('full_minus::'):return '完整组合−'+feature(mid.split('::',1)[1])
        if mid.startswith('standalone::'):return feature(mid.split('::',1)[1])
        if mid.startswith('behavior_plus_modality::'):return '行为＋'+feature('modality::'+mid.split('::')[-1])
        if mid.startswith('behavior_plus::'):return '行为＋'+feature(mid.split('::',1)[1])
        if mid.startswith('modality::'):return feature(mid)
        return mid
    pairrows=[['比较层级','加入或保留的信息','损失改善量及95%置信区间','人/探针']]
    for x in data['pairs']:
        isfull=x['analysis_set_id']=='AS.full';model=row(x['analysis_set_id'],x['added_model_id'])
        pairrows.append(['完整组合删减' if isfull else '行为增加信息',feature(x['feature_id']),f(x['point_estimate'],6)+' '+ci(x,n=6),f"{x['n_participants']}/{model['n_probes']}"])
    table(32,pairrows)
    rows33=[[c.text for c in d.tables[33].rows[0].cells]];rows46=[[c.text for c in d.tables[46].rows[0].cells]];rows47=[[c.text for c in d.tables[47].rows[0].cells]]
    for x in data['models']:
        nm=name(x['model_id']);sid=x['analysis_set_id'];aset={'AS.behavior_reference':'行为参照','AS.behavior_plus_cardiopulmonary':'行为与心肺','AS.behavior_plus_ocular':'行为与眼部','AS.behavior_plus_movement':'行为与动作','AS.full':'十三项完整','AS.sensor_only_joint':'八项传感'}.get(sid,'单指标' if sid.startswith('AS.standalone::') else data['devices'].get(sid.split('::')[-1],sid));n=f"{x['n_participants']}/{x['n_probes']}"
        rows33.append([nm,n,f(x['participant_equal_log_loss']),ci(x),f(x['participant_macro_auroc']),ci(x,'participant_macro_auroc_'),str(x['n_participants_auroc_estimable'])])
        rows46.append([aset,nm,str(x['n_participants']),str(x['n_probes']),f(x['participant_equal_log_loss']),f(x['ci_lower']),f(x['ci_upper']),'可估计'])
        slope=f(x['calibration_slope'],3) if x['calibration_slope_reporting']=='reportable' else '不作实质解释'
        rows47.append([aset,nm,f(x['participant_macro_auroc']),f(x['participant_macro_auroc_ci_lower']),f(x['participant_macro_auroc_ci_upper']),f(x['participant_macro_brier']),str(x['n_participants_auroc_estimable']),str(x['n_participants_auroc_single_class']),slope])
    table(33,rows33);table(46,rows46);table(47,rows47)
    for i,fig in [(261,'fig16'),(366,'fig28'),(378,'fig29'),(410,'fig32'),(419,'fig33'),(431,'fig34')]:
        blip=paras[i]._p.xpath('.//a:blip')[0];rid=blip.get(qn('r:embed'));part=d.part.related_parts[rid];media[str(part.partname).lstrip('/')]=(a.results/'figures'/f'{fig}.png').read_bytes()
        cap=paras[i+1];settext(cap,cap.text,'figure caption '+str(i+1),True)
        changes.append({'location':'figure '+fig,'highlight':'yellow border and yellow caption','preserved_extent':True})
    # Retain exact original bytes for all unchanged package members.
    xml=etree.tostring(d._element,xml_declaration=True,encoding='UTF-8',standalone=True)
    with zipfile.ZipFile(a.input) as original,zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as result:
        for item in original.infolist():result.writestr(item,xml if item.filename=='word/document.xml' else media.get(item.filename,original.read(item.filename)))
    verify=Document(a.output);assert len(verify.tables)==51
    assert len(verify.tables[33].rows)==57 and len(verify.tables[46].rows)==57 and len(verify.tables[47].rows)==57
    manifest={'input':str(a.input),'input_sha256':sha(a.input),'output':str(a.output),'output_sha256':sha(a.output),'changed_regions':len(changes),'replaced_figures':list(media),'package_preservation':'all members except document.xml and six selected media retained byte-for-byte','changes':changes}
    (a.results/'report_edit_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='changes'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
