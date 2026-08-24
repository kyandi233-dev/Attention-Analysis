# RUNBOOK｜NIR Formal 执行接续与决策判断

本文件用于 `runtime/nir-formal/` 的正式分析接续、故障恢复和运行决策。目标是让后续执行者无需依赖聊天上下文，仅根据进程、`.run.lock`、`completion.json` 和输出产物即可判断下一步。

当前正式分支：`nvidia-cuda`。当前 package version：`1.0.1`。

## 1. 启动前固定检查

在仓库根目录：

```powershell
git switch nvidia-cuda
git pull
cd runtime\nir-formal
python -m pytest tests -q
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

只有以下条件同时满足才进入正式运行：

- `check-env` 能识别 NVIDIA GPU/CUDA；
- 正式模型资产存在；
- `discover --formal-only` 的被试与数据盘实际内容一致；
- `dry-run` 没有 duplicate、错误盘符或意外被试；
- 当前没有另一条正式 batch 或目标被试的 formal 进程仍在运行。

默认正式入口：

```powershell
python run_formal_batch.py
```

单被试诊断入口：

```powershell
python run_pipeline.py formal --video "<实际视频路径>" --device 0
```

## 2. 运行状态的唯一判断口径

正式运行目录内的 `completion.json` 是状态文件，但必须结合 `.run.lock` 和真实进程判断。

### `status = initializing`

含义：当前实例已经取得该输出目录的运行锁，正在 CUDA、YOLO、RITnet、时间戳或 phase window 等正式帧循环之前的初始化阶段。

判断：

- `.run.lock` 存在；
- lock 中 PID 在本机仍存活；
- 当前实例尚未进入正式帧处理。

动作：

- 不要因为 `processed_frames = 0` 结束进程；
- 继续观察 PID、CPU/GPU/显存和标准输出；
- 如果进程正常退出但没有发布终态，退出守卫应将状态收口为 `failed`；
- 如果进程被强制杀死、机器断电等导致守卫来不及执行，下一次启动会根据 PID 判断 stale lock。

### `status = running`

含义：初始化已经越过，正式运行逻辑已经接管该输出目录。

动作：

- 不重复启动同一 subject / 同一输出目录；
- `processed_frames` 当前不是实时 heartbeat，不用它判断“是否卡死”；
- 进程仍存活且 CPU/GPU 有活动时保持运行；
- 只有明确异常、进程退出或长时间无任何资源活动并伴随错误输出时才进入故障诊断。

### `status = complete`

含义：只有通过严格 completion validator 的正式全量结果才算完成。

动作：

- 正常 batch 应 skip；
- 不因人工怀疑而直接覆盖；
- 若需要重跑，先说明原因，再使用 `--force`。

### `status = smoke_complete`

含义：短测或 partial run 成功，不等于正式全量完成。

动作：

- 可以用于环境、速度、parity 或诊断；
- 不允许作为正式完成依据；
- 正式 batch 仍应继续跑完整版本。

### `status = failed`

含义：启动、读帧、产物完整性、验证或未捕获异常已明确失败。

动作：

1. 读取 `exception_type`、`error`、`failure_stage`、`finished_at_utc`；
2. 先解决根因；
3. 确认旧 PID 已退出且 `.run.lock` 不再被有效进程持有；
4. 对目标被试重新运行；
5. 不手工把 `failed` 改成 `complete`。

## 3. `.run.lock` 决策规则

同一正式输出目录只允许一个拥有者。

lock 中至少记录：PID、host、token、创建时间和命令。

看到 `.run.lock` 时按以下顺序判断：

1. **同 host 且 PID 存活**：视为有效锁，禁止第二实例启动。
2. **同 host 且 PID 已不存在**：视为可证明 stale lock，新实例允许恢复并重新取得锁。
3. **其他 host 持有**：默认视为有效锁，不自动删除，避免共享存储上的跨机器并发写入。
4. **lock 文件损坏/不可读**：不要手工猜测；先确认所有相关进程，再人工处理。

禁止使用“先检查文件不存在，再普通创建”的方式实现互斥。当前实现使用 exclusive create，避免两个进程同时通过检查。

## 4. 重复实例的处理规则

如果发现同一个 subject 同时有两条独立正式分析：

1. 先确认它们是否写向同一正式输出目录；
2. 只保留一条明确的当前诊断/正式实例；
3. 停止旧实例；
4. 确认旧 PID 已退出；
5. 检查 `.run.lock` 当前 owner；
6. 不在两个实例都存活时继续观察“哪个先完成”，因为它们可能同时覆盖 `frames.csv`、`eyes.csv`、`summary.json`、`run_manifest.json`、`phase_windows.json` 和 overlays。

当前守卫修复后，第二实例应在模型初始化之前因有效 `.run.lock` 被拒绝。

## 5. 进程仍活着但 0 帧时怎么判断

不要只看 `completion.json` 的 `processed_frames`。

### 情况 A：`initializing` + PID 存活

继续保留实例。检查：

```powershell
nvidia-smi
```

以及 Windows 任务管理器中的 PID、CPU 时间、内存和 GPU 占用。

只要进程仍有正常初始化活动，就不能仅凭 0 帧判定失败。

### 情况 B：`initializing` + PID 已退出

正常异常退出后应已经变为 `failed`。

若机器断电/强杀导致仍残留 `initializing`：

- 确认 PID 不存在；
- 下一次启动应将旧 lock 识别为 stale；
- 新运行重新取得锁并覆盖为新的 `initializing`；
- 不把旧 marker 当作当前进程状态。

### 情况 C：`running` + PID 存活

当前 `completion.json` 不是帧级 heartbeat。优先根据真实进程、GPU 活动、控制台输出和最终产物判断。

### 情况 D：没有相关 PID，但状态仍不是终态

这是异常退出或强杀后的遗留状态。不要宣称任务还在后台运行。确认 stale lock 后重新启动，并保留失败证据用于排查。

## 6. `sub-078` 当前接续规则

针对之前出现过重复实例和旧 `0 frames` marker 的 `sub-078`：

1. 先 `git pull` 到包含 formal run guard 的当前 `nvidia-cuda` HEAD；
2. 确认旧 `sub-078` Python PID 均已退出；
3. 确认没有有效 `.run.lock` 被旧 PID 持有；
4. 先单被试运行，不直接重新拉起全 batch：

```powershell
python run_formal_batch.py --subjects sub-078
```

5. 启动后应该先看到该输出目录的 `completion.json: initializing`；
6. 如果再开第二条相同 `sub-078`，第二条应被 `.run.lock` 拒绝；
7. 初始化完成后状态进入 `running`；
8. 最终只有严格验证通过后才允许 `complete`；
9. `sub-078` 验证正常后，再恢复剩余 batch。

## 7. 批处理恢复决策

批处理意外中断后，不从“上次记得跑到哪”恢复，而按输出目录逐个严格判断：

- valid `complete`：skip；
- `smoke_complete`：正式模式重跑；
- `failed`：根因处理后重跑；
- marker 缺失、损坏、身份不匹配、缺帧或产物不完整：重跑；
- 有有效 `.run.lock`：不启动第二实例；
- stale lock：确认 PID 不存在后恢复。

然后执行：

```powershell
python run_formal_batch.py --dry-run
python run_formal_batch.py
```

不要通过修改 completion marker 或删除错误产物来伪造已完成状态。

## 8. ORT CUDA 决策边界

默认正式复现仍使用 `pytorch-cuda`。

只有目标 NVIDIA 机器已经完成以下验证后，才允许把 `ort-cuda` 用作正式 worker profile：

- CUDA Execution Provider 正常；
- 短测无 CPU fallback；
- 与冻结 PyTorch CUDA 路径做 parity；
- 精度门通过；
- 稳定性通过；
- 速度确有收益。

运行：

```powershell
python run_formal_batch.py --backend ort-cuda
```

在这些门未通过前，只能视为可选短测 profile，不能因为“看起来更快”直接替换正式口径。

## 9. 修改管线后的最低验证

任何涉及 formal completion、run lock、batch skip、输出目录命名、backend identity 的修改，至少执行：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
python run_formal_batch.py --dry-run
```

如果修改了 GPU backend 或模型初始化逻辑，还必须在目标 NVIDIA 真机做短测。

修改完成后提交信息应说明：

- 改了什么生命周期行为；
- 是否改变科研参数；
- 是否改变输出 identity / package version；
- 测试结果；
- 尚未完成的真机验证。

## 10. 决策原则

正式分析的优先级固定为：

**科研结果完整性 > 防止并发覆盖 > 可恢复性 > 可诊断性 > 运行速度。**

遇到不确定状态时，不通过猜测把任务判成完成；优先保留证据、确认真实进程和锁 owner，再决定继续、失败或重跑。
