"""Restore manuscript voice and caption formatting; keep audit prose outside report."""
from pathlib import Path
from copy import deepcopy
import argparse,difflib,hashlib,json,re,zipfile
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
    assert sha(a.input)=='2118f960015325d6e7710cd0f6a17c5fc4fa0c0ad4509b067b5008e8540f5f34'
    if a.output.exists():assert sha(a.output)==json.loads((a.results/'v5_edit_manifest.json').read_text(encoding='utf-8'))['output_sha256']
    d=Document(a.input);original=Document(a.original);ps=list(d.paragraphs);assert len(ps)==763
    stats=json.loads((a.results/'v5_descriptive_results.json').read_text(encoding='utf-8'));changes=[]
    def settext(p,text,label,force=False):
        old=p.text
        if old==text and not force:return
        spans=[];start=0
        for run in p.runs:spans.append((start,start+len(run.text),deepcopy(run._r.rPr)));start+=len(run.text)
        fallback=next((st for lo,hi,st in spans if st is not None),None)
        for el in list(p._p):
            if el.tag!=qn('w:pPr'):p._p.remove(el)
        def put(t,style,yellow):
            if not t:return
            run=p.add_run(t)
            if style is not None:run._r.insert(0,deepcopy(style))
            if yellow:
                pr=run._r.get_or_add_rPr();h=pr.find(qn('w:highlight'))
                if h is None:h=OxmlElement('w:highlight');pr.append(h)
                h.set(qn('w:val'),'yellow')
        for op,i,j,k,l in difflib.SequenceMatcher(None,old,text,autojunk=False).get_opcodes():
            if op=='equal':
                for lo,hi,st in spans:
                    if min(j,hi)>max(i,lo):put(old[max(i,lo):min(j,hi)],st,False)
            elif op in ('insert','replace'):put(text[k:l],next((st for lo,hi,st in spans if lo<=i<hi),fallback),True)
        changes.append({'location':label,'before':old,'after':text})
    def p(i,text):settext(ps[i],text,f'paragraph {i}')
    replacements={
      161:'第二阶段的 19 份有效记录显示，在相同 B 条件下，Go 漏按率均值由第一个 Block 的 0.029 上升至第三个 Block 的 0.048，反应时变异系数均值由 0.369 上升至 0.485。相同条件下的连续记录呈现了任务推进过程中的行为变化，为正式实验采用统一 B 条件提供了依据。',
      185:'正确 Go 试次的反应时（reaction time, RT）用于描述反应速度，低于 100 ms 的按键作为提前反应单独记录。有效正确 Go RT 同时计算均值和中位数。反应稳定性采用 RT 标准差、四分位距、中位数绝对偏差和反应时变异系数（RT coefficient of variation [RT-CV]）描述。每个窗口包含至少 2 次有效正确 Go 反应时，以样本标准差除以均值计算 RT-CV。窗口内采用 Theil–Sen 稳健斜率描述反应速度随时间的变化方向（Sen, 1968; Theil, 1950）。',
      188:'图 13 正式实验的正确 Go 反应时分布',
      189:'注：A 为反应时分布，B 为累积分布。图示包含 116 个实验场次中 83661 次不低于 100 ms 的正确 Go 反应。',
      192:'Go 遗漏率为未作出反应的 Go 试次占全部 Go 试次的比例。为考察按键时序的影响，另将伴有刺激前按键或跨试次按键延续的漏按单独统计；其余漏按占全部 Go 试次的比例记为筛选后遗漏率，用于补充分析。',
      207:'注：从距离域复数信号中提取胸壁微动，经呼吸与心搏频段分离、时域和频域估计后，得到窗口级心率与呼吸率。',
      208:'毫米波记录胸壁微动中的呼吸和心搏信息，形成与行为、眼部和身体动作相配合的自主生理观测。分析提取窗口级心率与呼吸率，并结合运动幅度、相位连续性和谱峰集中程度控制信号质量。',
      209:'毫米波帧时间与任务事件时间对应，以每次思维探针前 30 s 为主要分析窗口。窗口内按时间顺序选取信号，并以区块边界限定分析范围，使心肺指标与同期行为及主观报告对应。',
      210:'呼吸和心搏相关变化分别采用 0.10–0.50 Hz 和 0.8–2.0 Hz 的四阶巴特沃斯带通滤波，并进行双向滤波。心搏信号结合变分模态分解（variational mode decomposition, VMD）提取窄带成分（Dragomiretskiy & Zosso, 2014）；模态数为 3，惩罚参数为 1000，更新步长为 0，采用均匀初始化和 10⁻⁶ 的收敛阈值。心率与呼吸率估计综合时域峰间隔、频域峰值及两者的一致性。',
      211:'毫米波采集数据为完成距离向快速傅里叶变换后的复数信号，由 2 发射×4 接收形成 8 个通道，每通道含 256 个距离单元。分析综合回波能量、相位变化和时间稳定性选择胸壁微动所在的距离单元及通道，再提取呼吸和心搏成分。',
      214:'注：A 为距离谱与选中的距离单元；B 为 20 s 相位片段及呼吸、心搏带通成分，后两者作纵向偏移以便观察；C 为心搏频谱，标出的谱峰对应 55 次/min。',
      239:'信息类别层分别评价行为、眼部、动作和心肺对注意内容报告的预测能力。在各类传感信息与行为的共同有效样本上，比较单类信息、行为基线及加入传感信息后的模型；独立单指标和八项传感联合模型分别使用其输入完整的样本。',
      250:'无输入参照以各外层训练折的正例率作为常数概率，在对应测试探针上按参与者等权计算对数损失。模型与参照的分数共同呈现；排序、损失和校准分别用于评价类别区分、概率预测误差和预测概率与实际比例的对应。',
      273:ps[273].text.replace('区块顺序包含持续作业、练习和休息等因素，结果描述了这一整体任务过程中的表现变化。','两种时间尺度的结果共同呈现了持续任务中的反应稳定性变化。'),
      276:'注：A 为参与者在 B1、B2 中的配对均值，灰线表示个体，深色线表示组均值；B 为每个区块六个连续阶段的组均值。先在参与者内汇总重复场次，再按参与者进行 20000 次自助抽样，误差线表示 95% 置信区间。每个阶段包含 4 个刺激循环。',
      285:'筛选后遗漏率与主观警觉呈负向关联，系数为 −0.404，95% 置信区间为 [−0.564, −0.244]，基于 2320 个探针。这一补充结果与误按率、辨别力的结果相互呼应，呈现了较低警觉状态下任务反应完整性与抑制控制的变化。',
      291:'遗漏前反应逐渐变慢，错误后随即加快，呈现了局部任务执行状态的调整。与整段任务中的波动相比，事件对齐的轨迹进一步展示了注意变化在短时间尺度上的行为表现。',
      331:ps[331].text.split(' 动作强度为')[0]+' 动作强度以相邻画面的归一化灰度差除以帧间隔计算，单位为 s⁻¹；阶段系数表示该指标随任务推进的变化。',
      343:'5.5 毫米波心肺记录与参考对照',
      344:'5.5.1 记录覆盖与心肺分布',
      345:'毫米波在 110 个实验场次中获得 2200 个有效心肺窗口，覆盖全部 2320 个思维探针的 94.83%。心率均值为 75.93 次/min，标准差为 10.68 次/min；呼吸率均值为 18.53 次/min，标准差为 4.60 次/min。与思维探针对齐的连续记录，将心肺活动纳入了多模态注意状态分析。',
      346:'表 13 心率与呼吸率的描述统计',
      347:'注：每项指标均包含 2200 个探针前 30 s 的有效窗口；方括号为第 25 和第 75 百分位数。',
      353:'5.5.2 同步心电参考对照',
      354:'算法开发阶段采用同一参与者的 5 场同步心电记录，形成 100 个配对窗口。毫米波融合心率的平均绝对误差为 10.46 次/min，平均差值为 −9.03 次/min。进一步将误差分解至谱峰识别、谐波区分和通道选择，明确了影响估计的主要环节；各类误差及估计方法的比较见附录 E。',
      356:'图 27 毫米波心率与心电参考的差值分布',
      357:'注：每个点表示一个配对窗口。横轴为两种心率的均值，纵轴为毫米波心率减心电参考心率。实线表示差值均值 −9.03 次/min，虚线表示均值上下各 1.96 个标准差，范围为 −30.76 至 12.69 次/min。',
      363:'注：指标按参与者等权汇总，曲线下面积由同时具有两类报告的参与者计算。眼部、动作和心肺采用与行为的共同有效样本，分别为 1775、2296 和 2198 个探针；八项传感联合为 1705 个探针。区间采用参与者自助抽样估计。',
      365:'无输入参照以外层训练折的任务聚焦比例给出常数概率。行为模型的对数损失为 0.6237，参照为 0.6438；八项传感联合模型为 0.6441，对应参照为 0.6348。结合排序指标，行为信息在新参与者的注意内容预测中表现出较稳定的信息价值。',
      385:'10 s 窗口中，误按率有 1740 个有效值，五项行为指标完整的探针为 1547 个；较长窗口提供了更多反应与抑制反应机会。在共同有效的 61 人、116 场、2306 个探针上，分别训练 20 s 与 30 s 行为模型，损失差（30 s − 20 s）为 −0.007751，95% 置信区间为 [−0.015151, 0.002301]。两窗口表现接近，结合窗口内有效反应数量，本研究保留 30 s 为主要分析窗口。',
      405:'注：各行在相同参与者、相同探针和相同训练划分上比较行为模型与增量模型，心肺比较包含 58 人、110 场、2198 个探针。',
      423:'各项成对比较用于考察预定信息组合中的条件预测价值，区间为未经整组多重校正的单项区间。补充分析还在原十一项组合的 60 人、1775 个探针上，采用嵌套验证比较完整模型与预定义精简流程：对数损失分别为 0.618033 和 0.618623，差值为 −0.000589，95% 置信区间为 [−0.009133, 0.008061]。精简流程与完整模型表现接近，为后续权衡指标数量与预测表现提供了参照。',
      431:'六种配置的有效覆盖率分别为 99.83%、94.74%、94.66%、76.51%、90.43% 和 73.41%；完整三设备组合包含 57 人、98 场、1703 个探针。覆盖率与各配置的预测结果共同呈现了不同硬件条件下可获得的信息及其表现。',
      439:ps[439].text.replace('毫米波分析采用窗口级心率与呼吸率，心率变异性的应用留待进一步的心电参考验证。','毫米波分析采用窗口级心率与呼吸率。'),
      443:'注：点为优势比，横线为 95% 置信区间，虚线为 1。橙色为确认性效标一致性模型，灰色为探索性模型。探索性行为分析中，误按率的多重比较校正结果为 q = .143。',
      449:'测评结果需要以使用者容易理解的方式呈现。FocusWave已完成面向测评准备、实时反馈和历史回顾的交互原型，将状态信息、估计可靠性与信号质量分层展示，并以示例数据呈现完整使用流程。画面采用平静、自然的视觉意象，探索低干扰的注意反馈方式。',
      454:'注：上方为任务诱发、多源采集和跨个体预测的研究流程，下方为状态反馈、单次总结和长期记录的交互原型。实线框与虚线框分别表示这两类环节。',
      456:'界面原型将每次测量组织为时间、状态、可靠性与数据来源四个方面。时间标识对应记录的时刻与范围，状态分量呈现注意状态的不同侧面，0–100 状态指数用于探索直观的汇总表达。估计可信程度与信号质量分别显示，信号不足时标记为暂时无法估计。当前模型提供任务聚焦报告的概率，界面在此基础上探索过程反馈与跨次数回顾的组织方式。',
      496:'注：设置界面集中呈现设备检查、模型范围、人工智能使用说明与数据管理信息。',
      519:'多模态比较使系统能够从“采集更多信号”进一步走向“识别有用信息”。行为信息构成新参与者预测的主要基础，眼部指标兼具独立区分信息和状态关联；在既有行为信息上，眼部与心肺的增量区间包含零，加入动作后损失小幅增加。十三项组合移除动作后的损失下降，进一步提示指标数量与有效信息并不完全对应。这些结果为特征优化和设备配置提供了具体依据。',
      522:'本研究将同一参与者的全部场次保留在同一验证侧，并采用参与者等权训练和评价，贴近系统首次面对新使用者的情境。排序、损失和概率校准共同构成了模型评价体系：排序反映类别区分，损失反映预测误差，校准反映预测概率与实际报告比例的对应。现有结果为进一步优化个体预测和概率表达提供了基础。',
      525:'毫米波通道在 110 场中形成了 2200 个与思维探针对齐的心肺窗口，将自主生理活动引入非接触注意测评。同步心电对照进一步量化了估计误差，并将主要误差对应到具体信号处理环节。这一组合贯通了信号提取、任务对齐、参考比较与预测评价，为心肺通道的持续改进提供了明确路径。',
      526:'心肺参考对照来自一名参与者的重复场次，后续将扩大同步心电与呼吸参考样本，覆盖更多个体和动作条件。当前研究聚焦窗口级心率与呼吸率；提高心搏定位精度后，可进一步拓展至逐搏变异指标。',
      528:'后续研究将围绕独立参与者、目标场景和个体校准展开。在现有 61 名参与者、116 场实验基础上增加不同人群及少数注意类别的观测，可提高类别估计的精度；在自然学习与工作任务中验证，将进一步连接受控测评与实际使用。对低反应次数窗口的稳定性、传感缺失与动作影响开展针对性评估，也有助于优化测量条件。',
      529:'方法优化将重点考察跨通道交互、非线性变化和个人校准，并将参数选择与重新训练纳入不确定性评价。界面中的连续状态指数、专注时长和画像将结合使用情境建立相应的评价标准；当前交互原型已为后续的用户研究和反馈设计提供了完整载体。',
      530:'FocusWave已完成从任务与状态参照、多源无接触采集到动态分析、跨参与者预测和交互呈现的系统构建。行为、眼部、动作及心肺信息的联合评价，使不同测量的作用更加清楚，也为系统精简和场景扩展提供了依据。通过继续完善测量精度与个体适配，系统可进一步服务于学习状态评估、持续作业监测与注意训练。',
      635:'· 心肺指标包括心率和呼吸率。',
      636:'· 场次级问卷分析以不低于 150 ms 的正确 Go 反应计算反应时中位数；探针前窗口分析采用 100 ms。',
      710:ps[710].text.replace('，并非每名参与者累计 2 次',''),
      713:ps[713].text.replace('覆盖数不等同于某次行为共同模型的样本数。',''),
      715:'对应正文 5.5，汇总心肺有效覆盖、同步心电对照及误差分解结果。',
      717:'心率与呼吸率分别在 110 个场次、2200 个探针前窗口内获得有效估计，覆盖率均为 94.83%。',
      721:'同步心电对照包含一名参与者的 5 场记录、100 个配对窗口。按谱峰、谐波和通道选择的判断结果分组，比较各类窗口的估计误差。',
      733:'误差分解将整体偏差对应到波峰识别、谐波区分和目标通道选择三个具体环节。结合时域、频域及融合路径的比较，可分别调整峰值选择与融合权重，使后续优化具有明确的检验目标。',
    }
    for i,text in replacements.items():p(i,text)
    # Remove nonessential old pilot inference, internal quality diagnostics, and audit addenda.
    deleted=[138,163,164,165,348,349,350,351,352,358,719,734,735,736,737,738]
    for i in deleted:
        changes.append({'location':f'delete paragraph {i}','before':ps[i].text,'after':''});ps[i]._p.getparent().remove(ps[i]._p)
    # Clear engineering names without changing the underlying variables or values.
    allp=list(d.paragraphs)+[p for t in d.tables for row in t.rows for c in row.cells for p in c.paragraphs]
    for pp in allp:
        text=pp.text.replace('无已检测时序歧义遗漏率','筛选后遗漏率').replace('无已检测时序歧义遗漏','筛选后遗漏')
        text=text.replace('时序歧义遗漏率','伴随提前或延续按键的遗漏率').replace('时序歧义遗漏','伴随提前或延续按键的遗漏')
        text=text.replace('心肺估计（补充）','心肺').replace('心肺（补充）','心肺')
        settext(pp,text,'term '+pp.text[:24])
    def rebuild_table(idx,values,widths):
        t=d.tables[idx];headers=deepcopy(t.rows[0]._tr);body=deepcopy(t.rows[1]._tr)
        for el in list(t._tbl):
            if el.tag==qn('w:tr'):t._tbl.remove(el)
        grid=t._tbl.tblGrid
        for el in list(grid):grid.remove(el)
        for width in widths:
            el=OxmlElement('w:gridCol');el.set(qn('w:w'),str(width));grid.append(el)
        for i,values_row in enumerate(values):
            row=deepcopy(headers if i==0 else body);cells=row.findall(qn('w:tc'));template=deepcopy(cells[0])
            for c in cells:row.remove(c)
            for text,width in zip(values_row,widths):
                c=deepcopy(template);row.append(c);pr=c.find(qn('w:tcPr'));w=pr.find(qn('w:tcW'));w.set(qn('w:w'),str(width))
                for bad in list(pr):
                    if bad.tag in (qn('w:gridSpan'),qn('w:vMerge')):pr.remove(bad)
                pp=c.find(qn('w:p'))
                for extra in list(c):
                    if extra.tag==qn('w:p') and extra is not pp:c.remove(extra)
                settext(Paragraph(pp,d),str(text),f'table {idx} row {i}',True)
                for run in Paragraph(pp,d).runs:
                    run.font.highlight_color=7
            t._tbl.append(row)
    vals=[['指标','窗口数','均值','标准差','中位数','四分位范围']]
    for x in stats['heart_stats']:vals.append([x['label'],str(x['n']),f"{x['mean']:.2f}",f"{x['sd']:.2f}",f"{x['median']:.2f}",f"[{x['q25']:.2f}, {x['q75']:.2f}]"])
    rebuild_table(13,vals,[2000,950,1000,1000,1000,2000])
    rebuild_table(42,[['指标','有效探针数','覆盖率'],['心率',2200,'94.83%'],['呼吸率',2200,'94.83%']],[3000,2400,2500])
    # Restore caption paragraph/run formatting from the user's original, not a generic theme.
    originals={}
    for pp in original.paragraphs:
        match=re.match(r'^([图表])\s+(\d+)\s',pp.text)
        if match:originals[(match[1],int(match[2]))]=pp
    restored=[]
    for pp in list(d.paragraphs):
        match=re.match(r'^([图表])\s+(\d+)\s',pp.text)
        if not match:continue
        kind,number=match[1],int(match[2]);template=originals.get((kind,number))
        if template is None:continue
        text=pp.text
        replacement=deepcopy(template._p);pp._p.addprevious(replacement);pp._p.getparent().remove(pp._p)
        restored_p=Paragraph(replacement,d)
        settext(restored_p,text,f'restore caption {kind}{number}',True)
        restored.append([kind,number])
    # Removing original Figure 9 shifts later figure numbers; update all textual references.
    def renumber(m):
        number=int(m.group(2));return m.group(1)+str(number-1 if number>9 else number)
    for pp in list(d.paragraphs)+[p for t in d.tables for row in t.rows for c in row.cells for p in c.paragraphs]:
        settext(pp,re.sub(r'(图\s*)(\d+)(?![\d.])',renumber,pp.text),'figure reference '+pp.text[:28])
    # Reintroduce native sequence fields with explicit restart numbers. This preserves
    # the original visible styles while avoiding inherited mixed literal/field numbering.
    for pp in d.paragraphs:
        match=re.match(r'^([图表])\s+(\d+)\s',pp.text)
        if not match:continue
        lo,hi=match.span(2);position=0
        for run in list(pp.runs):
            end=position+len(run.text)
            if position<hi and end>lo:
                text=run.text;pr=deepcopy(run._r.rPr);anchor=run._r
                if match[1]=='图' and int(match[2])>=9:
                    if pr is None:pr=OxmlElement('w:rPr')
                    h=pr.find(qn('w:highlight'))
                    if h is None:h=OxmlElement('w:highlight');pr.append(h)
                    h.set(qn('w:val'),'yellow')
                def newrun(text=None,field=None):
                    el=OxmlElement('w:r')
                    if pr is not None:el.append(deepcopy(pr))
                    if field is not None:el.append(field)
                    else:
                        t=OxmlElement('w:t');t.set(qn('xml:space'),'preserve');t.text=text;el.append(t)
                    anchor.addprevious(el)
                if lo>position:newrun(text[:lo-position])
                if position<=lo:
                    fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'begin');newrun(field=fld)
                    instr=OxmlElement('w:instrText');instr.set(qn('xml:space'),'preserve');instr.text=f' SEQ {match[1]} \\r {match[2]} \\* ARABIC ';newrun(field=instr)
                    fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'separate');newrun(field=fld)
                    newrun(match[2]);fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'end');newrun(field=fld)
                if hi<end:newrun(text[hi-position:])
                anchor.getparent().remove(anchor)
            position=end
    media={}
    for number,path in [(13,a.results/'v5_fig13.png'),(16,a.results/'v5_fig16.png'),(17,a.results/'v5_fig17.png'),(27,a.results/'v5_fig27.png')]+[(n,a.results/'v5_figures'/f'fig{n}.png') for n in [28,29,32,33,34]]:
        cap=next(p for p in ps if re.match(rf'^图\s+{number}\s',p.text)) if number not in [13,27] else ps[188 if number==13 else 356]
        # ps contains original wrappers even after caption restoration; original previous
        # drawing paragraphs are stable and were not replaced.
        idx=ps.index(cap);pic=ps[idx-1]._p.xpath('.//a:blip')[0]
        part=d.part.related_parts[pic.get(qn('r:embed'))];media[str(part.partname).lstrip('/')]=path.read_bytes()
        # Remove the previous display crop on Figure 13 now that it is a new formal plot.
        for sr in ps[idx-1]._p.xpath('.//a:srcRect'):sr.getparent().remove(sr)
    xml=etree.tostring(d._element,xml_declaration=True,encoding='UTF-8',standalone=True)
    with zipfile.ZipFile(a.input) as zin,zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():zout.writestr(item,xml if item.filename=='word/document.xml' else media.get(item.filename,zin.read(item.filename)))
    result={'input':str(a.input),'input_sha256':sha(a.input),'original_sha256':sha(a.original),'output':str(a.output),'output_sha256':sha(a.output),'changes':changes,'replaced_media':list(media),'restored_captions':restored,'removed_original_figure':9,'predictive_model_refits':0}
    (a.results/'v5_edit_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='changes'},ensure_ascii=False))

if __name__=='__main__':main()
