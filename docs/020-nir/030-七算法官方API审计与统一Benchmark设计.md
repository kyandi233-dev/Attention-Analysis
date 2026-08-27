# 030｜NIR 七传统瞳孔算法官方 API 审计与统一 Benchmark 设计

> 2026-08-27（Asia/Shanghai）｜依据 PyPupilEXT 官方 C++ 源码（openPupil/PyPupilEXT, commit a2999de 2024-10-19）与 pupil-detectors 官方源码（pupil-labs/pupil-detectors, tag 2.0.2 对应 cd297a0）逐文件核验，不是以仓库现有实现为准。

## 1. 目标

在 `nir-seven-algorithm-benchmark` 分支建立七种传统瞳孔检测算法（PuRe / PuReST / Pupil Labs 2D / ElSe / ExCuSe / Swirski2D / Starburst）的原生分辨率统一 benchmark，并把 **algorithm_returned / official_valid / geometry_sane** 三层语义彻底分开；与已冻结的 RITnet 正式结果只作 agreement 比较（RITnet 不是真值）。

## 2. 运行环境

- `pypupilext310` conda 环境（CPython 3.10.20）：pypupilext 0.0.1、pupil_detectors 2.0.2、numpy 1.26.4、opencv 4.10.0.84。
- PyPupilEXT wheel：仓库历史资产 `runtime/legacy/PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl`。
- 七个算法中六个（PuRe/PuReST/ElSe/ExCuSe/Swirski2D/Starburst）由 PyPupilEXT 的 `_pypupil.pyd`（pybind11 绑定，源码 `src/main.cpp`）暴露；**Pupil Labs 2D** 由 `pupil_detectors.Detector2D`（Cython 绑定 C++，`detector_2d.pyx`）暴露。

## 3. Pupil 对象与官方语义（`src/pupil-detection-methods/Pupil.h`）

Python 侧绑定（`src/main.cpp`）：

- `pupil.center` → `(x, y)` 元组；`pupil.size` → `(w, h)` 元组（**全轴长**，即椭圆包围盒两边全长，非半轴）；`pupil.angle` → float 度。
- `pupil.confidence` / `pupil.outline_confidence` / `pupil.eyelid` / `pupil.physicalDiameter` / `pupil.undistortedDiameter` / `pupil.algorithmName`。
- `pupil.valid(threshold)`：**pybind11 绑定未暴露 C++ 默认参数，必须显式传阈值**。官方默认阈值为 `NO_CONFIDENCE = -1.0`，因此调用 `pupil.valid(-1.0)`。
- `pupil.hasOutline()` = `size.width > 0 && size.height > 0`。
- `pupil.majorAxis()` = `max(size)`；`pupil.minorAxis()` = `min(size)`；`pupil.diameter()` = `majorAxis()`（官方"直径"=长轴全长，与几何平均直径 `sqrt(major*minor)` 不同）。
- 空/失败 sentinel（`clear()`）：`center=(-1,-1), size=(-1,-1), angle=-1, confidence=-1, outline_confidence=-1`。

**官方 `valid()` 实现**：

```cpp
return center.x > 0 && center.y > 0 && size.width > 0 && size.height > 0 &&
       (confidence > confidenceThreshold || outline_confidence > confidenceThreshold);
```

含义：**几何正 + 至少一个置信度超过阈值**。因为 `NO_CONFIDENCE=-1.0`，`valid(-1.0)` 等价于"有真实置信度且几何为正"。

**关键结论**：ElSe / ExCuSe / Swirski2D / Starburst 官方 `hasConfidence() == false`，`run()` 返回的 Pupil 中 `confidence = outline_confidence = -1`（ElSe/ExCuSe 用单参构造 `Pupil(scaledEllipse)`；Swirski2D/Starburst 不设置信度）。因此对这些算法，`Pupil.valid(-1.0)` **恒为 False**。这是官方语义，不是 bug：无置信度算法的"官方有效"信号不存在，必须用 `algorithm_returned`（= `hasOutline()`）+ `geometry_sane` 表达质量，`official_valid` 如实记为 False 并注释原因。PuRe / PuReST `hasConfidence() == true`（`pupil.confidence = selected.outlineContrast`），`valid()` 有真实含义。Pupil Labs 2D 返回 dict，`confidence ∈ [0,1]` 官方自带。

## 4. 逐算法审计

### 4.1 公共调用接口（`src/main.cpp` 绑定）

- `run(frame)` → 新 Pupil 对象（frame 为 uint8 灰度 2D numpy 数组，经 NDArrayConverter 转 cv::Mat）。
- `run(frame, pupil)` → 就地填充。
- `run(frame, roi, pupil, minPupilDiameterPx, maxPupilDiameterPx)` → ROI 版本，roi 为 `(x, y, w, h)` 元组。
- `runWithConfidence(...)`：所有重载 = `run(...)` **之后**再算 `outline_confidence = outlineContrastConfidence(frame, pupil)`，**有额外耗时**。主 runtime 必须测 `run()`；`runWithConfidence` 单独计时，不污染主算法 runtime。

### 4.2 PuRe

- 构造：`PuRe()`。字段：`meanCanthiDistanceMM=27.6`、`maxPupilDiameterMM=8.0`、`minPupilDiameterMM=2.0`、`baseSize=(320,240)`。
- `hasConfidence()==true`（原生置信度）。
- **内部 resize**（`init()`）：`scalingRatio = min(320/cols, 240/rows, 1.0)`；`run` 先下采样再检测，最后 `pupil.resize(1/scalingRatio)` 映射回原图坐标。→ 424×187 紧 crop 会被缩到 320×141，结果仍回到 crop 坐标。
- ROI 版本：`frame(roi)` 裁剪后按同样 scalingRatio 下采样；结果 `resize(1/scalingRatio)` 后 `center += roi.tl()` 回到全帧坐标。`userMin/MaxPupilDiameterPx > 0` 时以 `scalingRatio*user` 覆盖按"图像含双眼眦"假设（`estimateParameters`：`maxPupilDiameterPx = diag*8/27.6`）估计的直径范围。
- 失败：返回清空 Pupil（`hasOutline()==false`、`valid()==false`）。

### 4.3 PuReST

- 构造：`PuReST()`。字段同 PuRe。`reset()` 清 `previousPupil`。
- **有状态**：`run` 时若 `previousPupil.confidence == NO_CONFIDENCE` 走完整 `PuRe::run`，否则 `runTracking`；`previousPupil = pupil`。→ 逐帧独立比较必须 `reset()`；连续窗口必须按帧序连续调用。
- 缩放行为同 PuRe。

### 4.4 ElSe

- 构造：`ElSe()`。字段：`minAreaRatio=0.005`、`maxAreaRatio=0.2`。`hasConfidence()==false`。
- **内部 resize 仅当 max(rows, cols) > 640**（`#define IMG_SIZE 640`）；424×187 → **不缩放**。结果 `ellipse.center/size / scalingRatio` 映射回原图坐标。
- 失败 sentinel：中心出界时走 `blob_finder` 后强制 `ellipse.size = Size(0,0)` → `hasOutline()==false`。

### 4.5 ExCuSe

- 构造：`ExCuSe()`。字段：`max_ellipse_radi=50`、`good_ellipse_threshold=15`。`hasConfidence()==false`。
- **内部 resize 仅当 max(rows, cols) > 680**；424×187 → 不缩放。
- 失败 sentinel：`runexcuse` 返回 `center=(0,0), angle=0, size=(0,0)` → `hasOutline()==false`。

### 4.6 Swirski2D

- 构造：`Swirski2D()`。字段 `params`（TrackerParams）：`Radius_Min=40`、`Radius_Max=80`（**官方注释：对分辨率极敏感，需按图中瞳孔约略尺寸设定**）、`CannyBlur=1.6`、`CannyThreshold1=20`、`CannyThreshold2=40`、`StarburstPoints=0`、`PercentageInliers=20`、`InlierIterations=2`、`ImageAwareSupport=true`、`EarlyTerminationPercentage=95`、`EarlyRejection=true`、**`Seed=-1`**。
- `hasConfidence()==false`。**无内部 resize**。
- `Radius_Min/Max` 直接是 Haar 搜索的瞳孔**半径**范围（step 2，像素）。默认 40–80 半径 → 直径 80–160px，对 424×187 紧 crop 内约 6–16px 的瞳孔**完全失配**，必须冻结。
- **RANSAC 可复现**：`randomSubset(edgePoints, n, i + params.Seed)` 仅当 `params.Seed >= 0`；默认 -1 走非确定路径。→ 必须显式设 `Seed >= 0`。
- 失败 sentinel：`inliers.empty()` → `center=(0,0), size=(0,0)`。
- **ROI 版本官方 bug**：`pupil = run(frame(roi))` 后**不 shift 回全帧坐标**。→ benchmark 一律把 tight crop 作为 frame 传入、不传 ROI（输入是 crop，输出即在 crop 坐标）。

### 4.7 Starburst

- 构造：`Starburst()`。字段：`edge_threshold=16`、`rays=18`、`min_feature_candidates=10`、`corneal_reflection_ratio_to_image_size=2`、`crWindowSize=301`。`hasConfidence()==false`。
- **有状态**：`startPoint`、`lostFrameNum` 跨帧保持（失锁 5 帧回图像中心）。
- RANSAC 用 C 库全局 `rand()`（`get_random_num`），**无暴露 seed** → 进程内/跨进程不可严格复现，文档记录。
- 失败：仍构造 RotatedRect，`ellipse_axis = 2*[pupil_param[0], pupil_param[1]]` 可能为 0 或垃圾值 → 用 `hasOutline()` + `geometry_sane` 判断。
- 无内部 resize（粗定位数组仅用于降噪）。

### 4.8 Pupil Labs 2D（pupil_detectors.Detector2D）

- 构造：`Detector2D(properties=None)`；`detect(gray_img, color_img=None, roi=None, **kwargs)` → dict：
  - `ellipse` = `{"center": (x, y), "axes": (minor*2, major*2), "angle": rad*180/π - 90}`（**axes 第一项是短轴全轴长**）。
  - `diameter` = `max(axes)`；`confidence` ∈ [0,1]；`location` = center。
- 默认 properties：`coarse_detection=True, coarse_filter_min=128, coarse_filter_max=280, intensity_range=23, blur_size=5, canny_treshold=160, canny_ration=2, canny_aperture=5, pupil_size_min=10, pupil_size_max=100, ...`。
- **内部粗定位下采样**：仅当 `coarse_detection` 且 ROI 面积 > 320×240，粗定位阶段把图降采样 2（`integral[::2,::2]`），**精拟合仍在 ROI 全分辨率**。粗定位尺度 `coarse_filter_min/max`（除以 2 后使用）默认按全眼图像瞳孔 100–200px 标定；对紧 crop 需按尺度冻结。
- **有状态**：`mUse_strong_prior`/`mPrior_ellipse`/`mPupil_Size` 跨帧保持。
- 失败 sentinel：`result.confidence = 0.0` 且未填 ellipse（center=(0,0)、axes=(0,0)）。真实检出 confidence 通常 ≥0.3。
- 结果坐标为图像坐标（已加回 `roi.x/y`）；传 `roi=None` 即全图 ROI。

## 5. 统一 Schema（每 frame×eye×algorithm 一行）

| 字段 | 语义 |
|---|---|
| subject / phase / frame_idx / eye | 帧身份 |
| algorithm | 七算法规范名 |
| algorithm_returned | run() 无异常且 `hasOutline()`（size>0）；Pupil Labs 2D 为 axes>0 |
| official_valid | 严格 `Pupil.valid(-1.0)`；Pupil Labs 2D 为 confidence>0；无置信度算法如实 False |
| geometry_sane | returned 且中心在 crop 内、axis 不超 crop 比例、aspect 合理（见 core.geometry_sane） |
| center_x / center_y | crop 像素坐标 |
| major_axis / minor_axis | 全轴长（max/min of size） |
| angle_deg | 官方 angle 度 |
| diameter_geom | `sqrt(major*minor)` 几何平均直径 |
| area | `π*major*minor/4` |
| runtime_ms | 主 `run()` 计时（不含 confidence） |
| native_confidence | PuRe/PuReST/Pupil Labs 2D 原生；ElSe/ExCuSe/Swirski2D/Starburst 恒为官方 `NO_CONFIDENCE=-1.0`（=无原生置信度，不是 0） |
| outline_confidence | 仅 `--run-with-confidence` 时，单独计时 |
| confidence_runtime_ms | runWithConfidence 额外耗时，单独记录 |
| failure / exception | 异常信息 |
| bbox_x1/y1/x2/y2 | 源视频 tight bbox（供映射回源坐标） |
| input_width / input_height | crop 尺寸 |
| params_provenance | 算法参数与来源（JSON） |

## 6. 尺度规则与参数冻结（Swirski2D / PuRe 直径 / Pupil Labs 2D properties）

依据 Issue #19 实测（sub-031 frame 46988，紧 crop ≈424×187，RITnet 320×160 椭圆 ~5.4×12.2 → 映射紧 crop 后瞳孔直径约 6–16px，半径约 3–8px）：

- Swirski2D：`Radius_Min=max(2, round(0.02*min(W,H)))`、`Radius_Max=max(Radius_Min+4, round(0.10*min(W,H)))`；`Seed=0`。424×187 → Radius_Min≈4、Radius_Max≈19。provenance 记录该规则与来源。
- PuRe/PuReST：**不用 ROI 重载**。官方 `PuReST::run(frame, roi, pupil, userMin, userMax)` 有 bug（`previousPupil==NO_CONFIDENCE` 时调 `PuRe::run(frame, roi, pupil, -1, -1)` 不转发 user 直径；tracking 路径把 `userMax` 传两次）。改为用 `minPupilDiameterMM/maxPupilDiameterMM` 字段反推：`maxPupilDiameterPx = diag*(maxMM/27.6)`、`minPupilDiameterPx = (2*diag/3)*(minMM/27.6)`（diag 为下采样工作尺寸对角线），按目标 crop 直径反解 MM 并逐帧设置，覆盖 canthi 假设（紧 crop 不含眼眦，官方 estimateParameters 会错估）。
- Pupil Labs 2D：`pupil_size_min/max` 按同规则（min=2*Radius_Min, max=2*Radius_Max）；`coarse_filter_min/max` 按同规则缩放（紧 crop 时粗定位尺度也匹配）。具体数值在 synthetic + 真实 smoke 上验证后冻结。
- ElSe/ExCuSe/Starburst 对 424×187 不触发内部 resize，保持默认参数。

## 7. 状态算法处理

- **逐帧独立模式**：PuReST 每帧 `reset()`；Starburst 每帧新实例；Pupil Labs 2D 每帧新实例（消除 strong prior）。公平比较。
- **连续窗口模式**：对 temporal 窗口按帧序连续调用，但输入必须是该窗口所有 tight bbox 在 source pixel 中的固定 union canvas。状态以 `(subject, eye, sequence_id, algorithm)` 隔离；移动 tight crop 或变化尺寸直接拒绝。尺度敏感参数每帧重新施加，provenance 记录 detector 实际值。
- 两种模式都显式记录在输出与 CLI。

## 8. 边界

- benchmark 向算法提供 native source-resolution tight crop，**benchmark 本身不 resize**；但逐算法官方内部 resize/downscale 行为见 §4，文档如实记录，不宣称全链路原生分辨率。
- 不与 RITnet 椭圆 IoU 判"正确"；只做 agreement（center/diameter 分布、temporal）。
- 不覆盖 frozen production 数据；输出进独立下游目录；不上传原始视频/被试图片/逐帧数据到 GitHub。

## 9. 正式入口、抽样和审批门

唯一 production 入口为 `scripts/nir_pupil_benchmark.py`。使用经核验的 CPython 3.10 detector 环境，且从仓库根运行：

```powershell
$env:PYTHONPATH = "src"
$python = "D:\CondaEnvs\pypupilext310\python.exe"

# 只读计划；不抽帧、不运行算法
& $python scripts/nir_pupil_benchmark.py --stage plan --profile formal --subjects sub-031

# 极小单被试闭环；run-dir 必须为空目录
& $python scripts/nir_pupil_benchmark.py --stage all --profile smoke `
  --subjects sub-031 --run-dir "<new-empty-output-directory>"

# 对已有输出独立重读校验
& $python scripts/nir_pupil_benchmark.py --stage validate --profile smoke `
  --subjects sub-031 --run-dir "<completed-output-directory>"
```

formal 配置每被试先于算法抽取 `300 Block1 + 300 Block2 + 100 RITnet high-quality + 100 RITnet difficult`的独立 tight frames，另取 300 帧连续窗口。两类输入分轨报告。写入多被试必须显式加 `--approve-multi-subject`；现阶段仍需先完成单被试人工 credibility 复核，不得仅凭 completion 或自动几何门扩大。
