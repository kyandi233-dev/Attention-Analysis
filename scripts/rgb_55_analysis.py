# -*- coding: utf-8 -*-
"""
rgb_55_analysis.py
版本: v1.0 (2026-08-31)
功能: 正式报告 5.5 节「RGB 可见行为结果」的下游分析编排入口。

只读消费既有帧级产物 (RGB 10_analysis_ready 的 4 类 parquet) 与外部权威表
(行为 probe 表、毫米波 merge-ready 表), 不重跑任何视频/人脸/姿态模型。

产出 (输出到 21_analysis_tables_5.5, 不覆盖任何既有产物):
  tables/    rgb_probe_pre30s_strict_features.csv  严格 pre-probe 探针窗口特征 (2320 行)
             rgb_block_cycle_table.csv             block × cycle_bin 聚合表
             rgb_session_coverage.csv              场次级覆盖表
  models/    rgb_block_cycle_gee.csv               block×cycle Gaussian GEE (参与者聚类)
             rgb_q1_mnlogit.csv                    Q1 MNLogit 聚类稳健
             rgb_q2_ordinal_gee.csv                Q2 OrdinalGEE
             rgb_behavior_window_gee.csv           RGB 特征 ~ 近期行为指标
             rgb_mmwave_window_gee.csv             RGB 特征 ~ 毫米波运动代理 / HR / BR
             rgb_probe_within_between.csv          within/between 分解
  failures/  rgb_55_failures.csv                   全部 not_estimable 记录 (禁止空表冒充成功)
  figures/   4 张数据图 (PNG+SVG) + figure_manifest + coverage_audit
  run_manifest.json                                配置摘要 / 代码 commit / 输入 SHA-256

契约检查 (缺列/空主键/重复主键/schema 不一致/身份冲突 → 停止并报告):
  - cohort manifest include=true 场次 session 唯一且 repeat_participant_id 非空;
  - 行为 probe 表主键 (session, block, probe_order) 唯一, 覆盖 cohort 全部场次;
  - 帧表身份列与行为权威表一致 (participant_group_id 唯一且相同);
  - 毫米波两表合并后主键唯一, 与行为 probe 行一一对齐。

用法:
  python scripts/rgb_55_analysis.py
  python scripts/rgb_55_analysis.py --config configs/rgb_formal.yaml
  python scripts/rgb_55_analysis.py --paths-config configs/paths.local.yaml
  python scripts/rgb_55_analysis.py --subjects sub-031,sub-032   (代表场次 smoke)

依赖: pandas, numpy, statsmodels, pyarrow, matplotlib
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# 脚本从仓库根运行; 保证 src 在导入路径中 (pytest 场景由 pyproject pythonpath 提供)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import (
    canonical_session_id,
    included_cohort,
    load_cohort_manifest,
)
from attention_pipeline.rgb_formal.section_55 import (
    BEHAVIOR_PROBE_METRICS,
    MMWAVE_PROBE_METRICS,
    RGB_PROBE_METRICS,
    aggregate_probe_window,
    build_block_cycle_table,
    build_probe_within_between,
    build_session_coverage,
    fit_block_cycle_gee,
    fit_q1_mnlogit,
    fit_q2_ordinal_gee,
    fit_single_predictor_gee,
    merge_key_mmwave,
    normalize_mmwave_block,
)
from attention_pipeline.rgb_formal.figures_55 import build_figure_pack

RGB_55_RUNNER_VERSION = "rgb-5.5-analysis-v1.0"

# ---------------------------------------------------------------------------
# 硬编码参数 (集中声明)
# ---------------------------------------------------------------------------
# 毫米波与行为 probe 时间对齐的最大容忍差 (毫秒); 超过视为对齐失败并停止。
# 两边都以 formal master timeline 的 probe onset 为来源, 预期完全一致或仅差毫秒级。
PROBE_ONSET_MAX_DELTA_MS = 5000.0

# 帧级四轨文件的必需列 (契约检查: 缺列即停止)
MOTION_REQUIRED_COLUMNS = [
    "unix_ms", "block", "trial_num", "motion_valid", "body_motion_observable",
    "exposure_change_observable", "body_motion_energy", "global_motion_energy",
    "gray_mean", "exposure_change_abs", "exposure_change_signed", "cycle_num",
]
# 注意: pose_confirmation 真实 schema 没有 cycle_num 列 (三轨中仅 motion 与
# blink 帧带 cycle_num), block×cycle 聚合对 pose 帧按 unix_ms 对齐 motion cell,
# 不要求 pose 提供 cycle_num。
POSE_REQUIRED_COLUMNS = [
    "unix_ms", "block", "trial_num", "pose_shoulders_observable",
    "pose_lateral_right_per_sec", "pose_vertical_up_per_sec",
    "pose_radial_proximity_direction_score", "radial_world_z_proximity_rate",
]
BLINK_FRAMES_REQUIRED_COLUMNS = [
    "unix_ms", "block", "trial_num", "cycle_num", "blink_event_id",
    "left_eye_observable", "right_eye_observable", "blink_bilateral_observable",
    "bilateral_eye_consistent", "blink_closed_bilateral_candidate",
]
BLINK_EVENTS_REQUIRED_COLUMNS = [
    "blink_event_id", "start_unix_ms", "end_unix_ms", "duration_ms",
]


def _sha256(path: Path) -> str | None:
    """计算文件 SHA-256; 文件不存在时返回 None。"""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """写出 UTF-8-sig CSV (与仓库其余 runner 一致)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


class AnalysisContractError(RuntimeError):
    """数据契约被破坏时抛出; 调用方必须停止而不是修补。"""


def _check_columns(frame: pd.DataFrame, required: list[str], context: str) -> None:
    """缺列即停止 (不静默补列)。"""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AnalysisContractError(f"{context} 缺少列: {missing}")


def _load_session_frames(session: str, ready_root: Path) -> tuple[dict[str, pd.DataFrame | None], list[dict[str, str]]]:
    """读取单场四轨帧级 parquet (列投影); 文件缺失或损坏时该轨降级为 None。

    逐轨降级是显式报告而非静默修补: 缺失/损坏明细由调用方写入 run manifest
    (与 116 场口径 maintenance 表 partial_no_blink=100 行的已知缺失一致)。

    参数:
        session: 场次编号 (如 sub-031)
        ready_root: 10_analysis_ready 根目录
    返回:
        (frames dict, diagnostics list); frames 键为 motion/pose/blink_frames/
        blink_events, 值为 DataFrame 或 None; diagnostics 记录缺失/损坏明细
    """
    base = ready_root / session
    frames: dict[str, pd.DataFrame | None] = {}
    diagnostics: list[dict[str, str]] = []
    for track, filename, columns in (
        ("motion", f"{session}_motion_qc.parquet", MOTION_REQUIRED_COLUMNS),
        ("pose", f"{session}_pose_confirmation.parquet", POSE_REQUIRED_COLUMNS),
        ("blink_frames", f"{session}_blink_candidate_frames.parquet", BLINK_FRAMES_REQUIRED_COLUMNS),
        ("blink_events", f"{session}_blink_candidate_events.parquet", BLINK_EVENTS_REQUIRED_COLUMNS),
    ):
        path = base / filename
        if not path.is_file():
            frames[track] = None
            diagnostics.append({"session_id": session, "track": track, "issue": "file_missing"})
            continue
        try:
            frames[track] = pd.read_parquet(path, columns=columns + ["session_id", "participant_group_id"])
        except Exception as exc:
            # 单模态产物损坏 (如 sub-041 blink 帧文件 footer 截断) 只降级该轨,
            # 不修补、不重跑 producer; 明细写入 run manifest。
            frames[track] = None
            diagnostics.append({
                "session_id": session, "track": track,
                "issue": f"unreadable:{type(exc).__name__}",
            })
            continue
        if track == "blink_events":
            _check_columns(frames[track], BLINK_EVENTS_REQUIRED_COLUMNS, f"{session} blink events")
    return frames, diagnostics


def _verify_frame_identity(frames: dict[str, pd.DataFrame | None], session: str, expected_group: str) -> None:
    """帧表身份契约: 同场各轨 participant_group_id 唯一且与行为权威表一致。

    参数:
        frames: _load_session_frames 输出
        session: 场次编号
        expected_group: 行为权威表的 participant_group_id
    """
    for track, frame in frames.items():
        if frame is None or frame.empty:
            continue
        if "session_id" in frame.columns:
            observed_sessions = set(frame["session_id"].astype(str).dropna().unique())
            if observed_sessions != {session}:
                raise AnalysisContractError(
                    f"{session} {track} 帧表 session_id 冲突: {sorted(observed_sessions)}")
        if "participant_group_id" in frame.columns:
            groups = [g for g in frame["participant_group_id"].dropna().astype(str).unique() if g not in {"", "nan", "<NA>"}]
            if len(groups) != 1:
                raise AnalysisContractError(f"{session} {track} 帧表 participant_group_id 不唯一: {groups}")
            if groups and groups[0] != expected_group:
                raise AnalysisContractError(
                    f"{session} {track} 帧表身份 {groups[0]} 与行为权威身份 {expected_group} 不一致")


def _load_mmwave_merged(main_path: Path, e_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并毫米波主表 (1440 行) 与 E 表 (880 行), 返回 (宽表, 审计摘要)。

    合并键 = (session_id, 规范化 block, 块内探针序号); 契约: 键唯一、与行为
    probe 行一一对齐 (由调用方验证)。身份以行为表 participant_group_id 为准,
    毫米波 repeat_participant_id 不作身份来源。

    参数:
        main_path: mmwave_probe_merge_ready.csv
        e_path: mmwave_probe_merge_ready_E.csv
    返回:
        (合并表, 审计 dict)
    """
    for p in (main_path, e_path):
        if not p.is_file():
            raise AnalysisContractError(f"毫米波 merge-ready 表缺失: {p}")
    main = pd.read_csv(main_path, encoding="utf-8-sig", low_memory=False)
    extra = pd.read_csv(e_path, encoding="utf-8-sig", low_memory=False)
    merged = pd.concat([main, extra], ignore_index=True, sort=False)
    required = {"session_id", "block_id", "probe_index_in_block", "probe_onset_unix_ms"}
    _check_columns(merged, required, "mmwave merge-ready")
    merged["session_id"] = merged["session_id"].map(canonical_session_id)
    merged["block_norm"] = merged["block_id"].map(normalize_mmwave_block)
    merged["merge_key"] = (
        merged["session_id"].astype(str) + "|" + merged["block_norm"].astype(str)
        + "|" + pd.to_numeric(merged["probe_index_in_block"], errors="coerce").astype("Int64").astype(str)
    )
    duplicates = merged["merge_key"].duplicated()
    if duplicates.any():
        dup_keys = sorted(merged.loc[duplicates, "merge_key"].unique())[:5]
        raise AnalysisContractError(f"毫米波合并后主键重复: {dup_keys}")
    audit = {
        "main_rows": int(len(main)), "e_rows": int(len(extra)),
        "merged_rows": int(len(merged)),
        "merged_sessions": int(merged["session_id"].nunique()),
        "mmwave_state_counts": merged["mmwave_state"].value_counts(dropna=False).to_dict() if "mmwave_state" in merged else {},
        "identity_scheme_note": "identity taken from behavior probe table; mmwave repeat_participant_id unused",
    }
    return merged, audit


def run_rgb_55_analysis(
    config_path: str | Path = "configs/rgb_formal.yaml",
    *,
    paths_config: str | Path | None = None,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """执行 5.5 下游分析全流程, 返回 run manifest dict。"""
    config = load_config(config_path, paths_config=paths_config)
    ready_root = config.path_value("analysis_ready_root")
    output_root = config.path_value("analysis_tables_55_root")
    behavior_probe_path = config.path_value("behavior_probe_primary_30s")
    mmwave_main = config.path_value("mmwave_probe_merge_ready")
    mmwave_e = config.path_value("mmwave_probe_merge_ready_e")
    maintenance_features_path = config.path_value("analysis_ready_116cohort_root") / "rgb_probe_pre30s_features.csv"
    section = config.section("section_55")
    window_ms = float(section.get("probe_window_ms", 30000.0))

    # ---- 1. cohort manifest 契约 ----
    cohort = load_cohort_manifest(config, path_key="cohort_manifest")
    included = included_cohort(cohort, require_groups=True)
    cohort_sessions = sorted(included["session_id"].astype(str).tolist())
    if subjects is not None:
        requested = [canonical_session_id(v) for v in subjects]
        outside = sorted(set(requested) - set(cohort_sessions))
        if outside:
            raise AnalysisContractError(f"请求场次不在 include=true cohort 内: {outside}")
        cohort_sessions = requested

    # ---- 2. 行为权威 probe 表契约 ----
    behavior = pd.read_csv(behavior_probe_path, encoding="utf-8-sig", low_memory=False)
    _check_columns(behavior, [
        "session_id", "block_id", "probe_event_id", "probe_order_in_block",
        "anchor_trial_num", "probe_time_ms", "q1_nominal_4class", "q2_ordinal_4level",
        "participant_group_id",
    ], "行为 probe 表")
    behavior["session_id"] = behavior["session_id"].map(canonical_session_id)
    behavior = behavior[behavior["session_id"].isin(cohort_sessions)].copy()
    key = ["session_id", "block_id", "probe_order_in_block"]
    if behavior.duplicated(subset=key).any():
        dup = behavior.loc[behavior.duplicated(subset=key), key].head(5).to_dict("records")
        raise AnalysisContractError(f"行为 probe 主键重复: {dup}")
    if behavior["participant_group_id"].isna().any():
        raise AnalysisContractError("行为 probe 表存在空 participant_group_id")
    behavior_sessions = set(behavior["session_id"].astype(str))
    missing_sessions = sorted(set(cohort_sessions) - behavior_sessions)
    if missing_sessions:
        raise AnalysisContractError(f"行为 probe 表缺 cohort 场次: {missing_sessions}")

    # ---- 3. 逐场聚合 (严格 pre-probe 窗口 + block×cycle + 场次覆盖) ----
    probe_rows: list[dict[str, Any]] = []
    cycle_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    source_diagnostics: list[dict[str, str]] = []
    for session in cohort_sessions:
        behavior_sub = behavior[behavior["session_id"].astype(str).eq(session)]
        expected_group = str(behavior_sub["participant_group_id"].iloc[0])
        frames, diagnostics = _load_session_frames(session, ready_root)
        source_diagnostics.extend(diagnostics)
        _verify_frame_identity(frames, session, expected_group)
        for _, probe_row in behavior_sub.iterrows():
            probe_rows.append(aggregate_probe_window(
                motion=frames["motion"], pose=frames["pose"],
                blink_frames=frames["blink_frames"], blink_events=frames["blink_events"],
                probe_row=probe_row, window_ms=window_ms,
            ))
        if frames["motion"] is not None or frames["pose"] is not None or frames["blink_frames"] is not None:
            cycle_rows.append(build_block_cycle_table(
                motion=frames["motion"], pose=frames["pose"],
                blink_frames=frames["blink_frames"], blink_events=frames["blink_events"],
                session_id=session, participant_group_id=expected_group,
            ))
        coverage_rows.append(build_session_coverage(
            motion=frames["motion"], pose=frames["pose"],
            blink_frames=frames["blink_frames"], blink_events=frames["blink_events"],
            session_id=session, participant_group_id=expected_group,
        ))
    probe_features = pd.DataFrame(probe_rows)
    cycle_table = pd.concat(cycle_rows, ignore_index=True, sort=False) if cycle_rows else pd.DataFrame()
    session_coverage = pd.DataFrame(coverage_rows)

    # 契约: probe 特征表主键唯一 (与行为表同粒度)
    if probe_features.duplicated(subset=key).any():
        raise AnalysisContractError("RGB 探针窗口特征表主键重复")
    if len(probe_features) != len(behavior):
        raise AnalysisContractError(
            f"RGB 探针特征行数 {len(probe_features)} 与行为表 {len(behavior)} 不一致")

    # ---- 4. 毫米波合并与时间对齐契约 ----
    mmwave, mmwave_audit = _load_mmwave_merged(mmwave_main, mmwave_e)
    mmwave_sessions_in_cohort = mmwave[mmwave["session_id"].isin(cohort_sessions)]
    probe_features["merge_key"] = (
        probe_features["session_id"].astype(str) + "|" + probe_features["block_id"].astype(str)
        + "|" + pd.to_numeric(probe_features["probe_order_in_block"], errors="coerce").astype("Int64").astype(str)
    )
    mmwave_sub = mmwave_sessions_in_cohort[["merge_key", "probe_onset_unix_ms", *MMWAVE_PROBE_METRICS]].copy()
    if mmwave_sub["merge_key"].duplicated().any():
        raise AnalysisContractError("毫米波 cohort 子集主键重复")
    merged_probe = probe_features.merge(mmwave_sub, on="merge_key", how="left", validate="one_to_one", suffixes=("", "_mmwave"))
    aligned = merged_probe.dropna(subset=["probe_time_ms", "probe_onset_unix_ms"])
    if not aligned.empty:
        delta_ms = (pd.to_numeric(aligned["probe_onset_unix_ms"], errors="coerce")
                    - pd.to_numeric(aligned["probe_time_ms"], errors="coerce")).abs()
        max_delta = float(delta_ms.max())
        if max_delta > PROBE_ONSET_MAX_DELTA_MS:
            raise AnalysisContractError(
                f"毫米波与行为 probe onset 时间差超过 {PROBE_ONSET_MAX_DELTA_MS:.0f} ms: max={max_delta:.1f}")
        onset_alignment = {"matched_rows": int(len(aligned)), "max_abs_delta_ms": max_delta}
    else:
        onset_alignment = {"matched_rows": 0, "max_abs_delta_ms": None}
    unmatched_mmwave = len(mmwave_sub) - int(merged_probe["probe_onset_unix_ms"].notna().sum())
    if unmatched_mmwave > 0:
        raise AnalysisContractError(f"毫米波存在 {unmatched_mmwave} 行未匹配行为 probe 键")

    # ---- 5. 统计模型 ----
    # 契约: 预定义模型网格的列必须齐备, 缺失即停止 (禁止静默跳过预定义模型)
    missing_rgb = sorted(set(RGB_PROBE_METRICS) - set(probe_features.columns))
    if missing_rgb:
        raise AnalysisContractError(f"探针特征表缺少预定义 RGB 特征列: {missing_rgb}")
    missing_behavior = sorted(set(BEHAVIOR_PROBE_METRICS) - set(behavior.columns))
    if missing_behavior:
        raise AnalysisContractError(f"行为表缺少预定义行为指标列: {missing_behavior}")
    missing_mmwave = sorted(set(MMWAVE_PROBE_METRICS) - set(mmwave_sub.columns))
    if missing_mmwave:
        raise AnalysisContractError(f"毫米波表缺少预定义对照指标列: {missing_mmwave}")

    # 行为指标按 probe 主键合并进探针特征表 (one_to_one, 主键唯一已在上方验证)
    probe_with_behavior = probe_features.merge(
        behavior[key + list(BEHAVIOR_PROBE_METRICS)], on=key, how="left",
        validate="one_to_one", suffixes=("", "_behavior"))
    model_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    model_tables["block_cycle_gee"] = fit_block_cycle_gee(cycle_table, metrics=RGB_PROBE_METRICS)
    model_tables["q1_mnlogit"] = fit_q1_mnlogit(probe_features, predictors=RGB_PROBE_METRICS)
    model_tables["q2_ordinal_gee"] = fit_q2_ordinal_gee(probe_features, predictors=RGB_PROBE_METRICS)
    model_tables["behavior_window_gee"] = fit_single_predictor_gee(
        probe_with_behavior, outcomes=RGB_PROBE_METRICS, predictors=BEHAVIOR_PROBE_METRICS,
        analysis="rgb_behavior_window_gee", model_family="GaussianGEE_cluster")
    model_tables["mmwave_window_gee"] = fit_single_predictor_gee(
        merged_probe, outcomes=RGB_PROBE_METRICS, predictors=MMWAVE_PROBE_METRICS,
        analysis="rgb_mmwave_window_gee", model_family="GaussianGEE_cluster")
    within_between = build_probe_within_between(probe_features, metrics=RGB_PROBE_METRICS)

    # ---- 6. 图包 ----
    figure_manifest, figure_audit = build_figure_pack(
        probe_features=probe_features, cycle_table=cycle_table,
        session_coverage=session_coverage,
        output_root=output_root / "figures",
    )

    # ---- 7. 落盘 ----
    dirs = {
        "tables": output_root / "tables",
        "models": output_root / "models",
        "failures": output_root / "failures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    _write_csv(probe_features.drop(columns=["merge_key"], errors="ignore"), dirs["tables"] / "rgb_probe_pre30s_strict_features.csv")
    _write_csv(cycle_table, dirs["tables"] / "rgb_block_cycle_table.csv")
    _write_csv(session_coverage, dirs["tables"] / "rgb_session_coverage.csv")
    model_filenames = {
        "block_cycle_gee": "rgb_block_cycle_gee.csv",
        "q1_mnlogit": "rgb_q1_mnlogit.csv",
        "q2_ordinal_gee": "rgb_q2_ordinal_gee.csv",
        "behavior_window_gee": "rgb_behavior_window_gee.csv",
        "mmwave_window_gee": "rgb_mmwave_window_gee.csv",
    }
    for name, filename in model_filenames.items():
        results, _ = model_tables[name]
        _write_csv(results, dirs["models"] / filename)
    _write_csv(within_between, dirs["models"] / "rgb_probe_within_between.csv")
    all_failures = pd.concat(
        [failures for _, failures in model_tables.values()], ignore_index=True, sort=False)
    _write_csv(all_failures, dirs["failures"] / "rgb_55_failures.csv")
    _write_csv(figure_manifest, output_root / "figures" / "rgb_55_figure_manifest.csv")
    _write_csv(figure_audit, output_root / "figures" / "rgb_55_figure_coverage_audit.csv")

    # ---- 8. run manifest ----
    commit = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
    except Exception:
        pass
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "runner_version": RGB_55_RUNNER_VERSION,
        "git_commit": commit,
        "config_path": str(config_path),
        "config_digest": config.digest,
        "cohort": {
            "include_true_sessions": int(len(cohort_sessions)),
            "cohort_manifest_sha256": _sha256(config.registry_path("cohort_manifest")),
        },
        "inputs": {
            "behavior_probe_primary_30s": str(behavior_probe_path),
            "behavior_probe_sha256": _sha256(behavior_probe_path),
            "behavior_probe_rows": int(len(behavior)),
            "mmwave_main_sha256": _sha256(mmwave_main),
            "mmwave_e_sha256": _sha256(mmwave_e),
            "mmwave_audit": mmwave_audit,
            "mmwave_onset_alignment": onset_alignment,
            "maintenance_features_reference_sha256": _sha256(maintenance_features_path),
        },
        "probe_window": {
            "window_ms": window_ms,
            "strict_pre_probe": True,
            "same_block_required": True,
            "anchor_trial_excluded": True,
            "post_probe_excluded": True,
        },
        "outputs": {
            "probe_feature_rows": int(len(probe_features)),
            "probe_feature_sessions": int(probe_features["session_id"].nunique()),
            "missing_source_rows": int(probe_features["rgb_source_status"].eq("missing_source").sum()) if "rgb_source_status" in probe_features else 0,
            "partial_no_blink_rows": int(probe_features["rgb_source_status"].eq("partial_no_blink").sum()) if "rgb_source_status" in probe_features else 0,
            "cycle_table_rows": int(len(cycle_table)),
            "coverage_rows": int(len(session_coverage)),
            "failure_rows": int(len(all_failures)),
            "figure_generated": int(figure_audit["status"].eq("generated").sum()) if not figure_audit.empty else 0,
            "figure_not_estimable": int(figure_audit["status"].eq("not_estimable").sum()) if not figure_audit.empty else 0,
        },
        "source_diagnostics": source_diagnostics,
        "model_summary": {name: {
            "estimable_rows": int(len(results)),
            "failure_rows": int(len(failures)),
        } for name, (results, failures) in model_tables.items()},
        "pvalues_exported": False,
        "notes": [
            "Probe windows are strictly pre-probe: same block, trial_num < anchor trial, unix_ms < probe onset.",
            "Blink events are assigned via window-frame blink_event_id mapping; event table never joins by time range alone.",
            "Pose radial direction is reported as a dimensionless in-image direction-agreement candidate, not physical displacement.",
            "All models export B / SE / 95% CI only; no p-values are fabricated. Failures are written to failures/.",
            "Analysis results live outside the repository; only code/config/tests are committed.",
        ],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rgb_formal.yaml")
    parser.add_argument("--paths-config", default=None,
                        help="机器路径注册表; 缺省用 ATTENTION_ANALYSIS_PATHS_CONFIG")
    parser.add_argument("--subjects", default=None,
                        help="可选逗号分隔场次子集 (代表场次 smoke)")
    args = parser.parse_args()
    subjects = [x.strip() for x in args.subjects.split(",") if x.strip()] if args.subjects else None
    manifest = run_rgb_55_analysis(args.config, paths_config=args.paths_config, subjects=subjects)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
