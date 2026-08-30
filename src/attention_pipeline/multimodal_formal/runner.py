"""Multimodal fusion formal runner: alignment audit, LOSO ladder, outputs, manifest.

File: runner.py
Version: multimodal-fusion-v1.0.0
Purpose:
    编排多模态融合正式分析：
    1. 只读对齐审计（alignment.py）并落盘共同子集清单与覆盖率；
    2. 八组合 × 两结局 × 两模型的 LOSO 全量（外层折 = participant_group_id，
       joblib loky 并行，Windows 安全）；
    3. 汇总逐折性能（coverage 与性能同报）、M1-M7 相对 M0 的逐折配对
       差异与 bootstrap CI（按参与者重抽样）；
    4. Shapley 式平均边际贡献（仅描述，解释与预测分轨）；
    5. 执行 manifest（config digest、paths digest、代码 commit、输入 SHA、
       失败记录、参数快照）。

Contract:
    - 输入表只读；输出只写 --output-root 下新 run 目录，拒绝覆盖已有 run id；
    - 不修改任何单模态产物、NIR 冻结端点与其他 runner；
    - 不执行 formal_multimodal_v2.yaml fusion 状态解锁（项目负责人专属动作）；
    - 标准化/插补/筛选/选参只在训练折内（preprocessing.py）。

Usage:
    >>> python scripts/multimodal_fusion_analysis.py --config configs/multimodal_fusion.yaml \
    ...     --paths-config configs/paths.local.yaml --jobs 8

Dependencies:
    numpy, pandas, scikit-learn, joblib, PyYAML
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from attention_pipeline.config import Config, load_config

from . import MULTIMODAL_FUSION_VERSION
from .alignment import (
    KEY_COLUMNS,
    build_common_subset,
    load_aligned_tables,
    modality_feature_coverage,
    write_alignment_audit,
)
from .marginal import marginal_contributions
from .metrics import (
    METRIC_NAMES,
    bootstrap_paired_difference,
    fold_metrics_frame,
    summarize_fold_metric,
)
from .models import fit_predict_fold
from .preprocessing import PreprocessingState, apply_preprocessing, fit_preprocessing

# ---- 冻结参数（集中声明） ----
GROUP_COLUMN = "participant_group_id"
# 随机种子派生基数（保证逐折逐任务可复现）
SEED_BASE = 42
# joblib 并行后端（Windows 下必须 loky；不要用 multiprocessing 默认 fork 语义）
DEFAULT_BACKEND = "loky"
DEFAULT_N_JOBS = 8
# bootstrap 默认参数（科学配置 comparison.bootstrap 可覆盖）
DEFAULT_BOOTSTRAP_N = 2000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CI_PERCENTILES = (2.5, 97.5)


def _sha256(path: Path) -> str | None:
    """计算文件 SHA-256；文件不存在返回 None（不伪造 provenance）。"""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    """读取仓库当前 commit；读取失败返回 unknown（记录而非伪造）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _feature_columns(feature_blocks: dict[str, Any], blocks: tuple[str, ...]) -> tuple[list[str], tuple[str, ...]]:
    """由组合的模态块解析特征列与 NIR 指标名。

    Parameters
    ----------
    feature_blocks : config 的 feature_blocks 段。
    blocks : 组合的模态块名（如 ("behavior", "nir")）。

    Returns
    -------
    (columns, nir_metrics)：columns 为模型输入原始特征列（NIR 用原始指标名），
    nir_metrics 为需 within/between 分解的指标。
    """
    columns: list[str] = []
    nir_metrics: list[str] = []
    for block in blocks:
        if block == "nir_primary":
            metrics = [str(m) for m in feature_blocks.get("nir_primary", {}).get("metrics", [])]
            columns.extend(metrics)
            nir_metrics.extend(metrics)
            continue
        if block == "nir":
            # 兼容别名：nir -> nir_primary
            metrics = [str(m) for m in feature_blocks.get("nir_primary", {}).get("metrics", [])]
            columns.extend(metrics)
            nir_metrics.extend(metrics)
            continue
        raw = feature_blocks.get(block, [])
        columns.extend(str(c) for c in raw if isinstance(c, str))
    return columns, tuple(nir_metrics)


def _build_feature_frame(
    tables: dict[str, pd.DataFrame],
    common: pd.DataFrame,
    feature_blocks: dict[str, Any],
    outcome_column: str,
) -> pd.DataFrame:
    """构造共同子集全特征矩阵（父进程一次完成，worker 只做折切分）。

    Parameters
    ----------
    tables : 规范化模态表。
    common : 共同子集键表（含 participant_group_id 与标签）。
    feature_blocks : config feature_blocks。
    outcome_column : 当前结局列名。

    Returns
    -------
    共同子集行 × (键 + group + outcome + 全部特征列) 的矩阵。
    """
    frame = common.copy()
    for modality in ("behavior", "mmwave", "rgb"):
        cols = [c for c in feature_blocks.get(modality, []) if isinstance(c, str)]
        if not cols:
            continue
        frame = frame.merge(
            tables[modality][list(KEY_COLUMNS) + cols], on=list(KEY_COLUMNS), how="left"
        )
    nir_metrics = [str(m) for m in feature_blocks.get("nir_primary", {}).get("metrics", [])]
    if nir_metrics:
        frame = frame.merge(
            tables["nir"][list(KEY_COLUMNS) + nir_metrics], on=list(KEY_COLUMNS), how="left"
        )
    return frame


def loso_fold_split(
    frame: pd.DataFrame,
    fold_group: str,
    *,
    group_column: str = GROUP_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按留出参与者切分训练/测试折（participant-disjoint 契约）。

    Parameters
    ----------
    frame : 共同子集特征矩阵（含 group_column）。
    fold_group : 被留出的参与者组。
    group_column : 分组列名。

    Returns
    -------
    (train, test)：同一参与者的全部观测同侧，训练与测试参与者集合互斥。
    """
    train = frame[frame[group_column].astype(str).ne(fold_group)].copy()
    test = frame[frame[group_column].astype(str).eq(fold_group)].copy()
    return train, test


def _run_fold_task(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """单个 LOSO 折任务：留出一名参与者的全部观测，训练其余参与者。

    Parameters
    ----------
    payload : dict，含 feature_frame（共同子集全特征矩阵）、fold_group、
        fold_index、combinations（组合名 -> 模态块，有序）、feature_blocks、
        outcomes（outcome_key -> 列名）、models（模型名列表）、model_cfg、
        n_outcomes。

    Returns
    -------
    该折全部 (组合, 结局, 模型) 子记录列表；任何异常转为失败记录不抛出。
    """
    frame: pd.DataFrame = payload["feature_frame"]
    fold_group: str = payload["fold_group"]
    fold_index: int = payload["fold_index"]
    combinations: dict[str, list[str]] = payload["combinations"]
    feature_blocks: dict[str, Any] = payload["feature_blocks"]
    outcomes: dict[str, str] = payload["outcomes"]
    model_kinds: list[str] = payload["models"]
    model_cfg: dict[str, Any] = payload["model_cfg"]
    n_outcomes: int = payload["n_outcomes"]

    train, test = loso_fold_split(frame, fold_group)
    records: list[dict[str, Any]] = []
    if train.empty or test.empty:
        return [{
            "fold_group": fold_group, "combination": "ALL", "outcome": "ALL",
            "model": "ALL", "y_true": np.array([], dtype=int),
            "y_proba": np.full((0, n_outcomes), np.nan),
            "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
            "failed": True, "reason": "empty_train_or_test_fold",
        }]

    group_codes = pd.factorize(train[GROUP_COLUMN].astype(str))[0]
    for combination_index, (combination_name, blocks) in enumerate(combinations.items()):
        columns, nir_metrics = _feature_columns(feature_blocks, tuple(blocks))
        if not columns:
            continue
        for outcome_index, (outcome_key, outcome_column) in enumerate(outcomes.items()):
            y_train_full = pd.to_numeric(train[outcome_column], errors="coerce")
            y_test_full = pd.to_numeric(test[outcome_column], errors="coerce")
            for model_index, model_kind in enumerate(model_kinds):
                # 确定性种子：折序/组合序/结局序/模型序派生，与随机 hash 无关
                seed = SEED_BASE + fold_index * 1000 + combination_index * 100 + outcome_index * 10 + model_index
                try:
                    # 预处理只在训练折内拟合（组均值/中位数/mean/std/待删列）
                    state: PreprocessingState = fit_preprocessing(
                        train[[GROUP_COLUMN] + columns].copy(),
                        columns=columns,
                        group_col=GROUP_COLUMN,
                        nir_metrics=nir_metrics,
                    )
                    x_train = apply_preprocessing(
                        train, state, group_col=GROUP_COLUMN, nir_metrics=nir_metrics, is_train=True
                    )
                    x_test = apply_preprocessing(
                        test, state, group_col=GROUP_COLUMN, nir_metrics=nir_metrics, is_train=False
                    )
                    # 标签行过滤：结局缺失行不参与（审计已确认共同子集内 0 缺失）
                    train_mask = y_train_full.notna().to_numpy()
                    test_mask = y_test_full.notna().to_numpy()
                    y_train = y_train_full[train_mask].astype(int).to_numpy()
                    y_test = y_test_full[test_mask].astype(int).to_numpy()
                    record = fit_predict_fold(
                        x_train[train_mask], y_train,
                        x_test[test_mask], y_test,
                        group_codes[train_mask],
                        model_kind=model_kind,
                        model_cfg=model_cfg[model_kind],
                        n_outcomes=n_outcomes,
                        seed=seed,
                    )
                except Exception as exc:
                    record = {
                        "model_kind": model_kind,
                        "selected_C": None,
                        "c_selection_detail": {},
                        "n_train_rows": int(len(train)),
                        "n_test_rows": int(len(test)),
                        "y_true": y_test_full.dropna().astype(int).to_numpy(),
                        "y_proba": np.full((int(y_test_full.notna().sum()), n_outcomes), np.nan),
                        "n_classes_seen": 0,
                        "failed": True,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                record["fold_group"] = fold_group
                record["combination"] = combination_name
                record["outcome"] = outcome_key
                record["model"] = model_kind
                records.append(record)
    return records


def run_multimodal_fusion(
    config_path: str | Path = "configs/multimodal_fusion.yaml",
    *,
    paths_config: str | Path | None = None,
    run_id: str | None = None,
    jobs: int = DEFAULT_N_JOBS,
    fold_limit: int | None = None,
    combinations: list[str] | None = None,
    outcomes: list[str] | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """多模态融合正式分析全量入口。

    Parameters
    ----------
    config_path : 科学配置（configs/multimodal_fusion.yaml）。
    paths_config : 机器路径注册表。
    run_id : 输出 run id（默认 UTC 时间戳）。
    jobs : 折级并行 worker 数。
    fold_limit : 只跑前 N 个折（smoke 用；None 为全量）。
    combinations/outcomes/models : 子集覆盖（smoke/敏感性用；None 为全量）。

    Returns
    -------
    执行 manifest dict（含各输出路径与状态）。
    """
    config = load_config(config_path, paths_config=paths_config)
    data_root = config.path_value("data_root")
    output_root = config.path_value("output_root")

    # ---- 对齐审计（只读） ----
    outcome_cfg = config.section("outcomes")
    q1_col = str(outcome_cfg["q1"]["column"])
    q2_col = str(outcome_cfg["q2"]["column"])
    tables, common, audit = load_aligned_tables(data_root, outcome_columns=(q1_col, q2_col))
    if audit["status"] != "PASS_ALIGNMENT_AUDIT":
        raise ValueError(f"对齐审计未通过: {json.dumps(audit, ensure_ascii=False)}")
    feature_blocks = config.section("feature_blocks")
    coverage = modality_feature_coverage(tables, common, feature_blocks)

    # ---- run 目录 ----
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / run_id
    if run_root.exists():
        raise FileExistsError(f"run_id 已存在，拒绝覆盖: {run_root}")
    common_dir = run_root / "common_cohort"
    performance_dir = run_root / "performance"
    comparison_dir = run_root / "comparison"
    marginal_dir = run_root / "marginal"
    for directory in (common_dir, performance_dir, comparison_dir, marginal_dir):
        directory.mkdir(parents=True, exist_ok=False)
    write_alignment_audit(
        common_dir, tables=tables, common=common, audit=audit, feature_coverage=coverage
    )

    # ---- 共同子集全特征矩阵与折列表 ----
    feature_frame = _build_feature_frame(tables, common, feature_blocks, q1_col)
    groups = sorted(feature_frame[GROUP_COLUMN].astype(str).unique())
    if fold_limit is not None:
        groups = groups[:fold_limit]
    combos = config.section("combinations")
    if combinations is not None:
        unknown = set(combinations) - set(combos)
        if unknown:
            raise ValueError(f"未知组合: {sorted(unknown)}")
        combos = {k: v for k, v in combos.items() if k in combinations}
    selected_outcomes = {"q1": q1_col, "q2": q2_col}
    if outcomes is not None:
        selected_outcomes = {k: v for k, v in selected_outcomes.items() if k in outcomes}
    model_cfg = config.section("models")
    selected_models = ["logistic", "random_forest"]
    if models is not None:
        selected_models = [m for m in selected_models if m in models]

    # ---- 折级并行执行（每折一个任务，任务内循环组合×结局×模型） ----
    payloads: list[dict[str, Any]] = []
    for fold_index, group in enumerate(groups):
        payloads.append({
            "feature_frame": feature_frame,
            "fold_group": group,
            "fold_index": fold_index,
            "combinations": combos,
            "feature_blocks": feature_blocks,
            "outcomes": selected_outcomes,
            "models": selected_models,
            "model_cfg": {name: dict(cfg) for name, cfg in model_cfg.items()},
            "n_outcomes": 4,
        })
    backend = str(config.section("validation").get("fold_parallel_backend", DEFAULT_BACKEND))
    if jobs <= 1:
        task_results = [_run_fold_task(payload) for payload in payloads]
    else:
        task_results = Parallel(n_jobs=jobs, backend=backend)(
            delayed(_run_fold_task)(payload) for payload in payloads
        )
    records = [record for task in task_results for record in task]

    # ---- 逐折性能表（coverage 与性能同报） ----
    by_fold_frames: list[pd.DataFrame] = []
    for combination_name in combos:
        for outcome_key in selected_outcomes:
            for model_kind in selected_models:
                subset = [
                    r for r in records
                    if r["combination"] == combination_name
                    and r["outcome"] == outcome_key
                    and r["model"] == model_kind
                ]
                if not subset:
                    continue
                by_fold_frames.append(fold_metrics_frame(
                    subset, combination=combination_name, outcome=outcome_key, model_kind=model_kind
                ))
    by_fold = pd.concat(by_fold_frames, ignore_index=True, sort=False) if by_fold_frames else pd.DataFrame()
    by_fold_path = performance_dir / "performance_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False, encoding="utf-8-sig")

    # ---- 汇总表：组合 x 结局 x 模型 x 指标（逐折等权） ----
    summary_rows: list[dict[str, Any]] = []
    for (combination_name, outcome_key, model_kind), group in by_fold.groupby(
        ["combination", "outcome", "model"], sort=True
    ):
        row: dict[str, Any] = {
            "combination": combination_name, "outcome": outcome_key, "model": model_kind,
            "n_folds_total": int(len(group)),
            "n_folds_model_failed": int(group["model_failed"].sum()),
            "mean_n_test_rows": float(pd.to_numeric(group["n_test_rows"], errors="coerce").mean()),
        }
        for metric in METRIC_NAMES:
            summary = summarize_fold_metric(group, metric)
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_sd"] = summary["sd"]
            row[f"{metric}_n_folds_defined"] = summary["n_folds_defined"]
        summary_rows.append(row)
    performance_summary = pd.DataFrame(summary_rows)
    summary_path = performance_dir / "performance_summary.csv"
    performance_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # ---- M1-M7 相对 M0 的逐折配对差异 + bootstrap CI ----
    bootstrap_cfg = config.section("comparison").get("bootstrap", {})
    comparison_rows: list[dict[str, Any]] = []
    for (outcome_key, model_kind), group in by_fold.groupby(["outcome", "model"], sort=True):
        baseline = group[group["combination"].eq("M0")].set_index("fold_group")
        for combination_name in combos:
            if combination_name == "M0":
                continue
            current = group[group["combination"].eq(combination_name)].set_index("fold_group")
            paired = current.index.intersection(baseline.index)
            for metric in METRIC_NAMES:
                diff = pd.to_numeric(current.loc[paired, metric], errors="coerce") - \
                       pd.to_numeric(baseline.loc[paired, metric], errors="coerce")
                ci = bootstrap_paired_difference(
                    diff.to_numpy(dtype=float),
                    n=int(bootstrap_cfg.get("n", DEFAULT_BOOTSTRAP_N)),
                    seed=int(bootstrap_cfg.get("seed", DEFAULT_BOOTSTRAP_SEED)),
                    ci_percentiles=tuple(bootstrap_cfg.get("ci_percentiles", list(DEFAULT_CI_PERCENTILES))),
                )
                comparison_rows.append({
                    "outcome": outcome_key,
                    "model": model_kind,
                    "comparison": f"{combination_name}_vs_M0",
                    "metric": metric,
                    "paired_by": "fold",
                    "bootstrap_resample_unit": "participant",
                    **ci,
                })
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = comparison_dir / "incremental_vs_M0.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    # ---- 边际贡献（解释产物与预测产物分轨） ----
    marginal_frames: list[pd.DataFrame] = []
    for metric in METRIC_NAMES:
        source = performance_summary.rename(columns={f"{metric}_mean": "metric_mean"})
        marginal_frames.append(marginal_contributions(source, metric=metric))
    marginal = pd.concat(marginal_frames, ignore_index=True, sort=False)
    marginal_path = marginal_dir / "marginal_contribution.csv"
    marginal.to_csv(marginal_path, index=False, encoding="utf-8-sig")

    # ---- 执行 manifest ----
    input_sha = {}
    for modality, rel in {
        "behavior": "Behavior/formal_v3/probe_primary_30s.csv",
        "nir": "NIR/11_analysis_tables/probe_pupil_models/probe_pupil_model_table.csv",
        "mmwave_j": "mmWave/mmwave_probe_merge_ready.csv",
        "mmwave_e": "mmWave/mmwave_probe_merge_ready_E.csv",
        "rgb": "RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv",
        "identity_bridge": "mapping/mmwave_identity_bridge.csv",
    }.items():
        input_sha[modality] = _sha256(data_root / rel)
    import attention_pipeline as _attention_pipeline  # 包位置用于定位仓库根
    # __file__ = <repo>/src/attention_pipeline/__init__.py -> parents[2] = <repo>
    repo_root = Path(_attention_pipeline.__file__).resolve().parents[2]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "complete",
        "pipeline_version": MULTIMODAL_FUSION_VERSION,
        "science_config": str(Path(config_path).resolve()),
        "science_config_digest": config.digest,
        "paths_config": str(config.path_registry.path) if config.path_registry else None,
        "paths_config_digest": config.path_registry.digest if config.path_registry else None,
        "code_repo": str(repo_root),
        "code_commit": _git_commit(repo_root),
        "input_sha256": input_sha,
        "common_subset": audit["common_subset"],
        "alignment_audit_status": audit["status"],
        "n_fold_groups": int(len(groups)),
        "fold_groups": groups,
        "combinations": list(combos.keys()),
        "outcomes": list(selected_outcomes.keys()),
        "models": selected_models,
        "subset_overrides": {
            "fold_limit": fold_limit,
            "combinations_requested": combinations,
            "outcomes_requested": outcomes,
            "models_requested": models,
        },
        "n_task_payloads": int(len(payloads)),
        "n_fold_model_failures": int(by_fold["model_failed"].sum()) if not by_fold.empty else None,
        "feature_blocks": feature_blocks,
        "within_between_contract": config.section("preprocessing").get("nir_within_between"),
        "fusion_status_unlock_by_runner": False,
        "outputs": {
            "common_cohort": str(common_dir),
            "performance_by_fold": str(by_fold_path),
            "performance_summary": str(summary_path),
            "incremental_vs_M0": str(comparison_path),
            "marginal_contribution": str(marginal_path),
            "manifest": str(run_root / "manifest.json"),
        },
        "scientific_inference_authorized_by_code_alone": False,
        "explanation_and_prediction_separate": True,
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest
