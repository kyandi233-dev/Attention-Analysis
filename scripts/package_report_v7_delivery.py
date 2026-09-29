"""Register the verified report in existing repositories and build its private handoff."""
import argparse,json,hashlib,shutil,zipfile,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--formal',type=Path,required=True);ap.add_argument('--code',type=Path,required=True);a=ap.parse_args();P=a.project;F=a.formal;A=a.code;R=P/'11_数据/_FormalAnalysis';O=R/'SupervisedRunsV4_Cardiopulmonary/repair_20260928/report_results/v7_audit'
m=json.loads((O/'edit_manifest.json').read_text(encoding='utf-8'));v=json.loads((O/'artifact_verification.json').read_text(encoding='utf-8'));plots=json.loads((O/'plot_evidence.json').read_text(encoding='utf-8'));inv=json.loads((O/'resolved_image_inventory.json').read_text(encoding='utf-8'))
assert v['status']=='PASS' and v['docx_sha256']==m['output_sha256']
for name in ['table_numeric_audit.json','prose_numeric_audit.json','prediction_recheck.json']:assert json.loads((O/name).read_text(encoding='utf-8'))['status']=='PASS'
record_name='09-29-4-国赛报告图表与全文数值方法核验.md';stem='09-29-4-v7';evidence=F/'运行记录与证据';assets=F/'国赛报告/assets';assets.mkdir(exist_ok=True)
def write(p,t):p.write_text(t,encoding='utf-8')
def prepend(p,t):
 s=p.read_text(encoding='utf-8');marker='## 2026-09-29 国赛报告 v7 图表与全文核验'
 if marker in s:
  assert t in s,('Existing registration differs',p)
  return
 i=s.find('\n');write(p,s[:i+1]+'\n'+marker+'\n\n'+t+'\n\n'+s[i+1:].lstrip())
def replace(p,old,new):
 s=p.read_text(encoding='utf-8')
 if old not in s:
  assert new in s,(p,old)
  return
 write(p,s.replace(old,new))
audit_names=['artifact_verification.json','edit_manifest.json','plot_evidence.json','prediction_recheck.json','table_numeric_audit.json','prose_numeric_audit.json']
for name in audit_names:shutil.copy2(O/name,evidence/(stem+'-'+name))
for plot in plots['figures']:
 for field in ['png','svg']:
  name=plot['png'] if field=='png' else plot['png'].replace('.png','.svg');shutil.copy2(O/name,assets/('20260929_v7_'+name))
 data=f"fig{plot['old_figure']:02}_data.csv"
 # Paired heart-reference rows stay in the private handoff, not Git.
 if plot['old_figure']!=26:shutil.copy2(O/data,evidence/(stem+'-'+data))
figure_rows=[]
for item in inv:
 old=item['figure'];new=m['figure_number_map'].get(str(old))
 if new is None:decision='移出副本';reason=m['removed_main_figures'][str(old)];source='原图与原稿保留'
 elif old in [x['old_figure'] for x in plots['figures']]:
  decision='保留并按结果重绘';reason=next(x for x in plots['figures'] if x['old_figure']==old)['question'];source=f'{stem}-fig{old:02}_data.csv' if old!=26 else '本机及私有交接包：fig26_data.csv'
 elif old in [33,34]:decision='恢复原始无标黄图片';reason='说明测量与系统呈现的关系';source='原始系统界面素材；逐字节相等'
 else:decision='保留原图';reason='理论、实验、方法或系统展示；不作为统计结果重配色';source='与当前输入文档嵌入图逐字节相等'
 figure_rows.append(f"| {old} | {new or '—'} | {item['caption']} | {decision} | {reason} | {source} |")
figure_rows.append('| 附录 B 原六图 | — | 眼部补充图 | 移出副本 | 与完整系数和覆盖表重复 | 原稿与原图保留 |')
record=f'''# 国赛报告图表与全文数值、方法核验

状态：完成。任务标识：FW-REPORT-V7-20260929。当前交付为 v7 标黄副本，替代 v6；原始报告及既有分析归档保留。本轮模型重训为 0。

## 成稿身份与实际验收

- 输入：`{m['input']}`；SHA-256：`{m['input_sha256']}`。这是本轮实际读取的 v6 文件，不能与先前记录的 379e648b…视为同一字节版本。
- 输出：`{m['output']}`；SHA-256：`{m['output_sha256']}`。
- Microsoft Word 原生打开及导出 110 页，逐页缩略审阅、结果图与题注重点放大检查；无空白页、无失效交叉引用字段。86 个题注均采用原稿题注样式、居中、9 磅、宋体与 Times New Roman、加粗及原生编号域：39 个图、47 个表。封面登记表不计为研究表。
- 附录 26 张表补齐题注；图表在当前文档的实际关系中定位，逐一校验全部 39 个图的嵌入内容。预览发现输入保存后内部图片部件重新编号，已修复按旧部件号替换造成的错位，再次原生渲染验收通过。
- 原图 33、34 现为图 27、28，与原始干净素材逐字节一致；其余系统界面和流程图内容保留。所有图内均未加入修改标黄；修改文字及题注保留标黄。

## 数值和方法检查

40 张统计结果表逐行绑定对应源表，共核对 2082 个数值单元格；正文 37 段包含的 191 处主要结果数值逐项核查，并验证确实存在于最终文档。方法、设备与概念表另按代码、配置及现有记录检查，未把它们计作 40 张统计表。

发现并修正：任务进程反应时变异系数模型应为 1392 条阶段记录，阶段系数为 0.0077，95% 置信区间 [0.0032, 0.0122]，交互系数 0.0002 [−0.0056, 0.0061]；表 8 动作指标单位恢复 s⁻¹；附录 C 原始遗漏率区间改为 [−0.004155, 0.01326]，与原始遗漏字段对应；表 21 的概率值按三位小数由 .011 改为 .010（源值 .010479862）。不是重新改变分析阈值。

原主目标 25 个分析集合的 56 行模型、3297 条外层折与 16485 条内层折记录重新读取核查。确认当前输入与冻结输入对应、参与者训练测试互斥、内层预处理只拟合训练侧、候选参数一致、无失败候选与预测行、概率有效、探针键不重复；参与者等权对数损失、布里尔分数和参与者平均受试者工作特征曲线下面积均从逐探针折外预测独立复算一致。三个受心肺源筛选影响的集合沿用已完成重训结果，其余集合复用等价输入，不再训练。

| 审查对象 | 当前实际方法与处理 |
|---|---|
| 数据结构 | 61 名参与者、116 场、2320 探针为正式队列；模型按自身或共同比较输入确定分母，重复场次始终跟随参与者 |
| 行为与任务进程 | 每窗口至少 2 个有效正确反应计算变异系数；区块变化按参与者聚类自助抽样，任务进程采用广义估计方程（generalized estimating equations [GEE]） |
| 注意内容解释模型 | 多项逻辑回归、参与者聚类稳健推断，与跨参与者二分类预测分开 |
| 困倦—清醒解释模型 | 行为和动作采用有序 GEE；眼部采用参与者聚类稳健有序逻辑回归，不能统称同一种实现 |
| 眼部分析 | 主要表示为瞳孔面积比例；区分个人均值与个人内偏离，控制区块和探针进程，按既定分析族校正多重比较 |
| 动作分析 | 身体动作强度为归一化画面变化率，单位 s⁻¹；同期动作—行为模型以动作指标为因变量、标准化行为指标为预测变量 |
| 心肺输入 | 生产输入有限值各 2200，行为与心肺共同样本 2198；独立传感联合 1705，十三项完整组合 1703。测量参考的 100 个配对窗口单独解释 |
| 二分类训练 | L2 正则化逻辑回归；留一参与者外层验证、参与者分组五折内部选择；候选参数为 0.01、0.1、1、10 |
| 增量、删减及设备 | 比较双方在同一集合上成对评价；十三项与历史十一项严格区分；六设备组合保留各自样本 |
| 校准和不确定性 | 概率损失按参与者等权；校准回归和分箱使用合并探针，置信区间按参与者重采样。校准图点面积对应分箱数量 |
| 问卷 | 68 场、46 个参与者标识，含 18 个重复参与者；累积有序模型纳入参与者随机截距，收敛记录通过 |

这次完整性指当前报告统计结果、现有分析输出和当前方法实现的对应检查，不等于重新处理全部原始信号、重训全部模型或独立重做文献与历史实验。旧四分类预测继续移出报告；1、2 对 3、4 的独立补充结果及完整私有包沿用 09-29-3，未替换主目标。

## 图的必要性与配色

按正式报告科研绘图规范逐图判断研究用途，共检查原正文 45 图及附录 6 图。14 张保留的统计结果图从真实结果重新绘制，提供 300 点/英寸位图及可编辑矢量图；蓝色表示行为、绿色表示眼部、橙色表示动作、粉色表示心肺、灰蓝表示联合配置。误差线与数值源绑定，不按显著性选择展示点。移出正文六图及附录六幅重复图；保留完整数值表，不因区间包含零而删图。

| 原图号 | 现图号 | 内容 | 决定 | 用途或理由 | 数据与图件核验 |
|---|---|---|---|---|---|
{chr(10).join(figure_rows)}

## 复现与资产边界

代码入口位于 Attention-Analysis 的 `scripts/report_figures_v7.py`、`repair_report_v7.py`、三个 `audit_report_v7_*`、`verify_report_v7_artifact.py`、`inspect_report_v7.py`、`render_report_v7_review.py` 和 `package_report_v7_delivery.py`。运行顺序及参数见代码仓库 `docs/060-formal-analysis/020-国赛报告图表与全文核验_20260929.md`。

正式仓库保存本记录、六份聚合核验、13 份聚合绘图数据、14 张位图和矢量图；报告、原始身份逐行文件、心电参考配对明细和页面渲染不进 Git。完整本轮交接包保存于本机 `_handoff/2026-09-29_report-v7/`，含报告、审阅 PDF、脚本、参数、绘图数据及核验。云端上传状态以追加回执为准，不把本地包当成已上传。
'''
write(evidence/record_name,record)
for p,prefix in [(F/'README.md','运行记录与证据/'),(evidence/'README.md',''),(F/'国赛报告/README.md','../运行记录与证据/'),(F/'正式报告/科研绘图/README.md','../../运行记录与证据/')]:
 prepend(p,f'[v7 成稿与逐项核验]({prefix}{record_name})：110 页、86 个原生题注；14 张统计图统一配色，移出 12 张重复或辅助图。40 表 2082 格、正文 191 处数值及 25 集合逐折预测核查通过。原图 33、34 恢复原始无标黄素材；模型重训为 0。输入及输出均按本轮实际校验值登记，历史版本保留。')
# Update the current chapter, leaving historical drafts untouched.
chapter=F/'国赛报告/章节草稿/5.2-行为表现与即时主观状态_20260913.md'
for x in m['changes']:
 if x.get('paragraph')==269:replace(chapter,x['before'],x['after'])
replace(F/'国赛报告/章节草稿/4.6-跨参与者监督学习与多模态比较.md','探针前窗口内整体身体画面变化的中位数（无量纲）','探针前窗口内归一化画面变化率的中位数（s⁻¹）')
replace(F/'国赛报告/附录/附录C-行为完整结果附表.md','| 0.004799 | -0.004106 | 0.01332 |','| 0.004799 | -0.004155 | 0.01326 |')
doc=f'''# 国赛报告 v7 图表与全文核验

任务 FW-REPORT-V7-20260929；无模型重训。正式判断与图表取舍见正式仓库运行记录 09-29-4；当前输入和最终输出校验值见该记录与 edit_manifest.json。

## 脚本及顺序

令 R 为 `D:/Project/厚粲杯/11_数据/_FormalAnalysis`，RR 为 R 下 `SupervisedRunsV4_Cardiopulmonary/repair_20260928/report_results`，O 为 RR 下 `v7_audit`。科学绘图与数值核查使用本机 Python 的 pandas、numpy、matplotlib、scikit-learn；文档处理使用内置运行时的 python-docx、Pillow、pypdfium2。

1. `inspect_report_v7.py SOURCE O`：保存原始段落、表格清单。
2. `report_figures_v7.py --root R --results RR`：14 张统计图、颜色映射、CSV（逗号分隔数据表）、PNG（位图）、SVG（可缩放矢量图）与输入散列。绘图脚本参数见 `--help`；参与者区块自助抽样 20000 次，种子 20260929。
3. `repair_report_v7.py --source SOURCE --results RR --assets ASSETS`：ASSETS 为国赛素材/系统界面图_20260913；按当前文档关系重新定位图片，生成新副本。
4. `inspect_report_v7.py OUTPUT O --prefix final_`；`audit_report_v7_tables.py --root R --results RR --inventory final_table_inventory.json`；`audit_report_v7_prose.py --root R --results RR --edited`；`audit_report_v7_predictions.py --repair-root R/SupervisedRunsV4_Cardiopulmonary/repair_20260928`。
5. Microsoft Word 只读打开输出，ExportAsFixedFormat 导出 O/v7_render.pdf。`render_report_v7_review.py O` 生成逐页预览；`verify_report_v7_artifact.py O` 检查图、表、字体、编号、媒体内容、原稿与最终副本校验值。
6. `package_report_v7_delivery.py --project PROJECT --formal FORMAL --code CODE` 登记既有正式索引并打包；相同登记不会重复附加，既有登记与预期不一致时停止。

脚本保留三类真实差错证据：正文旧阶段样本和系数、表格单位/区间/概率值、Office 保存后媒体部件重编号。统计表格核查按对应结果行及原精度比较，含小于号判断；正文核查还确认改后文字存在于实际成稿。图中所有模型估计直接读取正式冻结结果，没有重新选择特征或显著性门槛。

Git 仅保存源码、文档、聚合表图和验证；原文档、渲染页、参考心电的逐窗口明细留在本机和用户指定私有交接包。预测与逐折身份沿用原归档，不复制到公开仓库。公开说明不包含凭据。
'''
write(A/'docs/060-formal-analysis/020-国赛报告图表与全文核验_20260929.md',doc)
prepend(A/'docs/060-formal-analysis/README.md','[v7 脚本、参数和资产边界](020-国赛报告图表与全文核验_20260929.md)：14 张结果图、86 原生题注、2082 表格数值和 191 正文数值核查；全部 25 集合逐折核查，重训为 0。')
H=R/'_handoff/2026-09-29_report-v7';H.mkdir(parents=True,exist_ok=True);content=H/'content';content.mkdir(exist_ok=True)
for sub in ['report','scripts','evidence','figures']: (content/sub).mkdir(exist_ok=True)
shutil.copy2(m['output'],content/'report'/Path(m['output']).name);shutil.copy2(O/'v7_render.pdf',content/'report/v7_审阅版.pdf')
for p in A.glob('scripts/*report_v7*.py'):shutil.copy2(p,content/'scripts'/p.name)
shutil.copy2(A/'scripts/report_figures_v7.py',content/'scripts/report_figures_v7.py')
shutil.copy2(A/'scripts/plot_cardiopulmonary_report_update.py',content/'scripts/plot_cardiopulmonary_report_update.py')
for name in audit_names+['resolved_image_inventory.json','final_table_inventory.json','final_paragraph_inventory.txt']:shutil.copy2(O/name,content/'evidence'/name)
for pattern in ['fig*_v7.png','fig*_v7.svg','fig*_data.csv']:
 for p in O.glob(pattern):shutil.copy2(p,content/'figures'/p.name)
source_paths=set()
for name in ['plot_evidence.json','table_numeric_audit.json','prose_numeric_audit.json']:
 source_paths.update(json.loads((O/name).read_text(encoding='utf-8')).get('source_sha256',{}))
for name in ['report_payload.json','v5_descriptive_results.json','result_lineage.csv']:
 source_paths.add(str(O.parent/name))
for name in ['inventory.json','paragraph_inventory.txt','table_inventory.json']:
 shutil.copy2(O/name,content/'evidence'/name)
source_map=[]
for original in sorted(source_paths):
 source=Path(original)
 relative=source.relative_to(P)
 target=content/'inputs'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 source_map.append({'original_path':original,'package_path':str(target.relative_to(content)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
write(content/'source_map.json',json.dumps(source_map,ensure_ascii=False,indent=2))
write(content/'既有预测与训练归档.md','本轮未重新训练模型。25 个集合的完整实际输入、新旧逐探针预测、逐折审计、配置及训练源码沿用已上传并回读核对的私有包：https://drive.google.com/file/d/1mpNqnfPonNr57BCXtAd2nvJX93EllT5b/view 。其 SHA-256 为 2ac0546bbbb2b1960fdbfa729cc6fbfdb5b99700ee33223503dca8898f4287fa。本包包含本轮全部绘图、数值核验输入和源路径映射；不重复复制整套训练归档。\n')
shutil.copy2(evidence/record_name,content/'README.md');shutil.copy2(A/'docs/060-formal-analysis/020-国赛报告图表与全文核验_20260929.md',content/'运行说明.md')
files=[{'path':str(p.relative_to(content)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(content.rglob('*')) if p.is_file()];write(content/'manifest.json',json.dumps({'task':'FW-REPORT-V7-20260929','files':files},ensure_ascii=False,indent=2))
package=H/'FocusWave_国赛报告v7_图表与全文核验交接包.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(content.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(content)))
receipt={'package':str(package),'bytes':package.stat().st_size,'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'files':len(files)+1,'report_sha256':m['output_sha256'],'cloud':'PENDING'};write(H/'local_package_receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))
