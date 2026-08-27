# 七算法正式 Benchmark 管线闭环与验证工作记录

> 08-27（Asia/Shanghai）｜状态：工程闭环与单被试极小预览已完成；未运行 sub-031 单人 formal 或 sub-031～sub-040 全量。

## 背景

`nir-seven-algorithm-benchmark` 分支 `bc30ef7` 已实现七算法 adapter、统一结果语义和基于既有 crop manifest 的执行器，但尚未闭合 Issue #19 规定的 production `eyes.csv` → 原始 NIR 视频 → 1:1 source-pixel tight crop → 确定性抽样 → 七算法 → RITnet agreement → QC/完整性输出。现有 continuous 实现还以首帧 crop 尺度构造共享 detector，却按当前帧尺度写 provenance，存在实际参数与记录参数不一致；移动 tight bbox 也不是状态算法可直接共享 prior 的稳定坐标系。

## 已核验事实与任务边界

1. 当前仓库为 `D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis`，分支 `nir-seven-algorithm-benchmark`，HEAD `bc30ef7`，工作树在任务开始时干净，远端同分支同 HEAD。
2. production 根固定为 `D:\_AttentionData\Beijing-NIR\amd-directml`；sub-031～sub-040 均存在完整 `eyes.csv`。sub-035 同时存在一个 `running` 残留目录和一个 `complete` b8/b16 目录，发现逻辑必须只接受完整 run。
3. 原始数据 `E:\正式实验` 只读；视频路径以 production completion/eyes 证据为准，不靠目录名猜测。
4. `D:\CondaEnvs\pypupilext310\python.exe` 已核验为 Python 3.10.20，包含 PyPupilEXT 0.0.1、pupil-detectors 2.0.2、NumPy 1.26.4、OpenCV 4.10.0.84，七算法类均可导入。
5. 当前无人工 source-resolution pupil ellipse 金标准；RITnet 只能作 agreement comparator，任何输出不得称 accuracy 或正式科学结论。
6. 本轮不重跑 YOLO/RITnet，不覆盖 production，不启动 10 人全量，不上传被试图片、原始视频或逐帧表。
7. 项目级入口要求读取的 `000-项目定位和进度/000-厚璨杯项目记忆.md` 在当前工作区实际不存在；本轮以仓库 `AGENTS.md`、README、NIR 方法文档、配置、Issue #19 和 production 证据继续。

## 目标

1. 建立唯一正式 benchmark 入口，完成 production evidence 到可验证输出的全链路。
2. 冻结确定性、算法结果无关的采样规则和输入坐标契约。
3. 修复连续模式尺度 provenance，并消除移动 tight-crop 坐标系对状态 prior 的未声明破坏。
4. 严格实现 RITnet 320×160 椭圆到 source pixel 的各向异性仿射映射。
5. 建立输入不可用、算法未返回、官方 valid、几何 sanity、人工 credibility 六者分轨的输出和 QC。
6. 用单元测试、合成真实库 smoke 和 sub-031 极小真实预览证明代码可运行；预览后停止，等待全量批准。

## 计划步骤

1. 扩展 `configs/nir_pypupilext_native_benchmark.yaml`，冻结 production 根、输出根、采样、crop、RITnet 分析画布和完整性策略。
2. 新增 formal 模块：发现 complete production run、验证 evidence、确定性抽样、生成两类 1:1 source-pixel 输入（逐帧 tight crop；连续窗口固定 source canvas）、写 sample manifest。
3. 扩展 schema/runner：记录 source/analysis 坐标、实际参数、输入类型和 sequence；continuous 每帧更新尺度参数，且只允许固定输入画布。
4. 新增 agreement/summary/validation：RITnet 仿射映射、算法间描述性 agreement、subject/eye/phase 分层、temporal jitter、人工 QC 模板和 completion marker。
5. 新增仓库级 CLI 与测试，更新当前 NIR README/方法文档/工作记录索引。
6. 依次运行静态 import、目标单测、全仓相关回归、真实库合成 smoke、sub-031 极小真实预览；核对输出 schema、失败行、参数 provenance、坐标和 montage。

## 风险与停止点

- production 字段或完成标记不满足契约时停止，不猜列、不补造 bbox。
- 任一采样层不足目标数量时正式计划失败并报告；smoke 只能通过显式 override 缩小数量。
- tight bbox 浮点到像素采用 `floor(x1/y1)`、`ceil(x2/y2)` 并裁到视频边界；原浮点值与实际整数值同时保留。
- 状态算法不得在变化的 crop 坐标系中静默复用 prior；正式 continuous 输入必须是固定 source-pixel canvas，且与逐帧 tight 输入分开报告。
- RITnet 椭圆经各向异性缩放后必须用矩阵变换重新求轴与角度，不能分别把 axis_a/axis_b 乘单一比例。
- 任何已有输出路径存在时默认停止，不静默覆盖；不提供隐式清理。
- sub-031 极小真实预览通过后停止，不把 smoke 结果外推到 cohort，也不直接运行 10 人。

## 校验标准

1. 发现器对 sub-035 只选择 `status=complete` 的 b8/b16 run；重复完整候选必须报错。
2. sample manifest 主键唯一，角色计数达到配置；所有 ready crop 的尺寸等于实际整数 bbox 尺寸，像素未 resize。
3. 每个 ready frame×eye×algorithm 恰有一行；input unavailable 也有显式失败行，不静默丢失。
4. 每行 applied params 与 detector 实际参数一致；continuous 固定画布尺寸在 sequence 内恒定。
5. RITnet 中心和椭圆仿射映射通过解析测试；source center 可由 crop center+bbox 原点回算。
6. completion 只有在预期行数、主键、文件 hash、summary/QC 模板均通过时才写 `status=complete`。
7. 七算法真实库 smoke 无未解释异常；真实预览 montage 需人工目视确认坐标落位。

## 执行与决策过程

### 1. 环境与库可用性

用户原先在 `D:\PyPupilEXT` 源码根运行 import，导致本地未编译的 `pypupilext/` 遮蔽 conda 环境中已安装 wheel，因而报 `No module named 'pypupilext._pypupil'`。在源码根之外直接核验 `D:\CondaEnvs\pypupilext310\python.exe`：`_pypupil.cp310-win_amd64.pyd` 存在，PyPupilEXT 六算法与 `pupil_detectors.Detector2D` 全部可 import。因此不需要为 Pupil Labs 2D 另建环境，也不需要在两套 detector 环境之间串行交换图片。

### 2. production 入口与抽样闭环

新增 `formal.py`/`formal_cli.py` 与仓库级脚本，只接受 `status=complete` 且契约文件完整的 production run。sub-035 的 `running` 残留目录不会被选中；如有多个同等完整候选则拒绝静默选择。`eyes.csv` 先校验 identity、source video、bbox/RITnet 字段和重复主键，再在任何传统算法运行前完成确定性抽样。

对 sub-031 执行 formal 只读 `plan` 得到 `tight_frames=800`、`temporal_frames=300`，与 `300 Block1 + 300 Block2 + 100 high-quality + 100 difficult + 300 continuous` 契约一致。没有在 formal plan 阶段抽帧或写输出。

### 3. 坐标、连续状态与参数 provenance 修复

- tight bbox 保留 production 浮点值，实际像素按 `floor(min)/ceil(max)` 后裁到视频边界；两组坐标同时落盘。
- RITnet 320×160 ROI 椭圆通过完整仿射矩阵映射到 source pixel，各向异性缩放后用特征分解重求主/次轴与角度。
- continuous 不再在逐帧变化的 tight crop 中共享 detector state；每个窗口生成固定 source-pixel union canvas，以 `(subject, eye, sequence_id)` 分组。维度或 bbox 变化时直接报错。
- Swirski2D/Pupil Labs 2D 尺度参数在每帧前重新施加；PuRe/PuReST 实际像素直径边界与 detector 字段写入 `actual_applied`。修复后不再存在“首帧实际参数，当前帧理论 provenance”不一致。
- PyPupilEXT `size=(w,h)` 转换为 major/minor 后同步修正 OpenCV 角度；当原 `size[1]` 是主轴时增加 90°。

### 4. 输出、人工 QC 与完整性

输出现包含 sample manifest、CSV/Parquet 逐帧结果、subject×algorithm 摘要、算法两两描述性 agreement、temporal summary、人工 montage/空白 label 表、validation summary、completion 及所有核心产物 SHA256。人工 montage 每帧同时显示左右眼，列为 original、RITnet 和七算法，用于独立判断它们是否真正拟合瞳孔。

### 5. 自动测试与真实运行证据

1. 目标测试：在 `pypupilext310` 环境中执行两个 benchmark 测试文件，`27 passed`。覆盖七 API、三层语义、轴角、尺度规则、production run 选择、各向异性仿射、确定性抽样、连续帧和 moving-bbox 拒绝。
2. 真实编译库合成 smoke：`20260827-env-and-seven-algorithms`，28 行（4 图×7 算法），0 异常，七算法 overlay 全部落盘。
3. sub-031 production 极小闭环：`sub031-formal-smoke-20260827-v05`，30/30 输入 ready，210/210 结果，0 duplicate，0 input unavailable，completion 为 complete；后续用 `--stage validate` 独立重读仍通过。
4. 全仓通用测试不能在 detector 环境中统一收集，因该环境没有 SciPy/statsmodels；主分析环境的历史测试又依赖已不存在的旧数据/旧脚本与 pyarrow。这些与本次 benchmark 目标测试分开报告，不冒充“全仓全部通过”。
5. 修正了合成 smoke 测试把 PNG/manifest 写到仓库根的污染，改为 pytest 临时目录；并只清理了本轮测试创建的 4 张 `smoke_*.png` 和 1 个 `manifest.csv`。

## 最终决策结果

七算法执行器与 Issue #19 production 上层数据链现已在工程上闭环，能由本任务直接在指定 conda 环境中运行，不需要用户手工切换终端或为 Pupil Labs 2D 单建环境。但是“程序 completion=complete”只表示输入/行数/主键/产物契约完整，不表示任一算法准确。

sub-031 两张 smoke montage 已证明 overlay 坐标落位可视检，也显示不同算法会把眼睑、反光或其他黑区当成瞳孔。因此当前不根据 2 帧宣布“哪个最好”；正式排名必须先完成预定的人工 credibility 标注，再与 returned/valid/geometry/agreement/runtime/temporal 分轨统计。

## 已完成、未完成与待确认

已完成：仓库/分支/HEAD/远端/worktree 核验；Issue #19 与 production 契约核验；sub-031～040 complete run 发现；环境与七算法导入；production 到输出的代码闭环；目标测试、合成 smoke、sub-031 真实极小 smoke、formal 只读计划、文档与人工 QC 产物。

未完成：sub-031 formal 完整 800+300 帧 benchmark；每被试 12 帧人工 credibility 标注与复核；sub-031～040 批量 benchmark；基于人工 credibility 的正式算法比较结论。

待确认：用户检查当前两张 montage 的坐标与版式后，是否批准 sub-031 单被试完整 benchmark；单被试人工 QC 和完整性通过后，是否批准 sub-031～040 多被试写入。
