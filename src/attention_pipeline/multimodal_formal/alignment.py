"""Probe-level input alignment: key normalization, identity bridge, common subset.

File: alignment.py
Version: multimodal-fusion-v1.0.0
Purpose:
    只读加载四模态 probe 特征表，把各表键名规范化为标准融合键
    ``(participant_group_id, session_id, block_id, probe_index_in_block, window_name)``，
    毫米波经身份桥表映射到 P 编码（R029=R096 跨批分裂按桥表修正条目处理），
    核验 Q1/Q2 标签与行为权威来源一致，构造四模态共同可用子集并输出对齐审计。

Contract:
    - 输入表只读，不写任何单模态目录；
    - 缺失模态只改变 availability，不改变参与者身份，不补零；
    - 标签以行为表为权威，其他模态仅做一致性校验，融合侧不重新推导；
    - 同一 session 不得出现在多个 P 组，同一 P 组不得拆入多折（下游分折保证）。

Usage:
    >>> from attention_pipeline.multimodal_formal.alignment import load_aligned_tables
    >>> aligned, audit = load_aligned_tables(data_root=Path(...))

Dependencies:
    numpy, pandas
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---- 冻结参数（集中声明；与 1.10 规格第 2/3 节一致） ----
# 标准 probe 融合键列名
KEY_COLUMNS = ("session_id", "block_id", "probe_index_in_block")
PRIMARY_WINDOW = "pre_30s"
# block 命名规范化：各表原始值 -> b1/b2
_BLOCK_CLEANUP = str.maketrans({"-": ""})
# 毫米波身份桥中标记跨批分裂的列（R029=R096 -> P-247758AEAC）
_BRIDGE_SPLIT_FLAG_COLUMN = "r_code_split_flag"
# 输入表相对路径（相对 data_root）
_INPUT_PATHS = {
    "behavior": "Behavior/formal_v3/probe_primary_30s.csv",
    "nir": "NIR/11_analysis_tables/probe_pupil_models/probe_pupil_model_table.csv",
    "mmwave": "mmWave/mmwave_probe_merge_ready.csv",
    "mmwave_e": "mmWave/mmwave_probe_merge_ready_E.csv",
    "rgb": "RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv",
    "bridge": "mapping/mmwave_identity_bridge.csv",
}
# 各模态 probe 索引列名（别名 -> 标准键）
_PROBE_INDEX_ALIASES = {
    "behavior": "probe_order_in_block",
    "nir": "probe_index_in_block",
    "mmwave": "probe_index_in_block",
    "rgb": "probe_order_in_block",
}
# 各模态 Q1/Q2 标签列名（行为为权威来源）
_LABEL_ALIASES = {
    "q1": {"behavior": "q1_nominal_4class", "nir": "probe_response",
           "mmwave": "label_probe_response", "rgb": "q1_nominal_4class"},
    "q2": {"behavior": "q2_ordinal_4level", "nir": "probe_vigilance",
           "mmwave": "label_probe_vigilance", "rgb": "q2_ordinal_4level"},
}
# NIR 表内 group 键（P 编码兼容别名，经 partition audit 背书）
_NIR_GROUP_ALIAS = "analysis_group_token"


def normalize_block(values: pd.Series) -> pd.Series:
    """规范化 block 编码：B1/B2、1/2、block-1/block-2 统一为 b1/b2。

    Parameters
    ----------
    values : 原始 block 列（字符串或数值）。

    Returns
    -------
    形如 ``b1``/``b2`` 的规范化字符串列。
    """
    text = values.astype(str).str.lower().str.replace("-", "", regex=False)
    text = text.str.replace("block", "", regex=False)
    return text.where(text.str.startswith("b"), "b" + text)


def _canonical_group(frame: pd.DataFrame, modality: str) -> pd.Series:
    """取某模态表的 canonical participant_group_id 列。

    Parameters
    ----------
    frame : 模态表。
    modality : behavior/nir/mmwave/rgb 之一。

    Returns
    -------
    P 编码序列；毫米波由桥表提前合并后取 participant_group_id。
    """
    if modality == "nir":
        return frame[_NIR_GROUP_ALIAS].astype(str)
    return frame["participant_group_id"].astype(str)


def load_modality_table(data_root: Path, modality: str, bridge: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """加载单个模态 probe 表并规范化为标准融合键。

    Parameters
    ----------
    data_root : 正式分析数据根（_FormalAnalysis）。
    modality : behavior/nir/mmwave/rgb；mmwave 自动合并 J+E 两批。
    bridge : 毫米波身份桥表（仅 mmwave 需要，含 participant_group_id）。

    Returns
    -------
    (table, problems)：table 含标准键与标签列；problems 为审计问题清单。
    """
    problems: list[str] = []
    if modality == "mmwave":
        parts = []
        for key in ("mmwave", "mmwave_e"):
            path = data_root / _INPUT_PATHS[key]
            if not path.is_file():
                problems.append(f"missing_input:{path}")
                continue
            parts.append(pd.read_csv(path, encoding="utf-8-sig", low_memory=False))
        if not parts:
            return pd.DataFrame(), problems
        frame = pd.concat(parts, ignore_index=True, sort=False)
        # 身份桥：R 编码仅作 provenance，分组一律用桥表 P 编码
        if bridge is None or "participant_group_id" not in bridge.columns:
            problems.append("mmwave_identity_bridge_missing_participant_group_id")
        else:
            frame = frame.merge(
                bridge[["session_id", "participant_group_id", _BRIDGE_SPLIT_FLAG_COLUMN]],
                on="session_id", how="left",
            )
            missing = int(frame["participant_group_id"].isna().sum())
            if missing:
                problems.append(f"mmwave_rows_without_bridge_identity:{missing}")
    else:
        path = data_root / _INPUT_PATHS[modality]
        if not path.is_file():
            problems.append(f"missing_input:{path}")
            return pd.DataFrame(), problems
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if modality == "nir":
            # NIR 表无 participant_group_id 列，使用 analysis_group_token 作为 P 编码兼容别名
            frame = frame.copy()
            frame["participant_group_id"] = frame[_NIR_GROUP_ALIAS].astype(str)

    index_col = _PROBE_INDEX_ALIASES[modality]
    if index_col not in frame.columns:
        problems.append(f"{modality}_missing_probe_index_column:{index_col}")
        return pd.DataFrame(), problems
    if "session_id" not in frame.columns:
        problems.append(f"{modality}_missing_session_id")
        return pd.DataFrame(), problems
    if "block_id" not in frame.columns and not (modality == "nir" and "block_num" in frame.columns):
        problems.append(f"{modality}_missing_block_column")
        return pd.DataFrame(), problems

    out = frame.copy()
    out["block_id"] = normalize_block(out["block_id"] if "block_id" in out.columns else out["block_num"])
    out["probe_index_in_block"] = pd.to_numeric(out[index_col], errors="coerce").astype("Int64")
    out["window_name"] = PRIMARY_WINDOW
    return out, problems


def build_label_consistency(
    tables: dict[str, pd.DataFrame],
    outcome: str,
) -> tuple[pd.DataFrame, int]:
    """核验非行为模态标签与行为权威标签的一致性。

    Parameters
    ----------
    tables : 已规范化的模态表字典（含 behavior）。
    outcome : q1 或 q2。

    Returns
    -------
    (audit_frame, total_mismatch)：每模态的匹配行数/不匹配数/标签缺失数，
    以及总不匹配数。
    """
    aliases = _LABEL_ALIASES[outcome]
    base = tables["behavior"][list(KEY_COLUMNS) + [aliases["behavior"]]].copy()
    base["pib"] = base["probe_index_in_block"]
    rows: list[dict[str, Any]] = []
    total_mismatch = 0
    for modality in ("nir", "mmwave", "rgb"):
        frame = tables[modality]
        label_col = aliases[modality]
        if label_col not in frame.columns:
            rows.append({
                "modality": modality, "outcome": outcome, "matched_rows": 0,
                "mismatch": None, "label_nan": None, "note": "label_column_missing",
            })
            continue
        # 对方表标签列重命名为固定名，避免与行为列同名时 merge 加 _x/_y 后缀
        sub = frame[list(KEY_COLUMNS) + [label_col]].rename(columns={label_col: "_other_label"})
        merged = base.merge(sub, on=list(KEY_COLUMNS), how="inner")
        base_labels = pd.to_numeric(merged[aliases["behavior"]], errors="coerce")
        other_labels = pd.to_numeric(merged["_other_label"], errors="coerce")
        mismatch = int((base_labels != other_labels).sum())
        nan_n = int(other_labels.isna().sum())
        total_mismatch += mismatch
        rows.append({
            "modality": modality, "outcome": outcome,
            "matched_rows": int(len(merged)), "mismatch": mismatch,
            "label_nan": nan_n, "note": "behavior_authoritative",
        })
    return pd.DataFrame(rows), total_mismatch


def build_common_subset(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造四模态共同可用探针观测子集（以 NIR 可用场次为约束）。

    Parameters
    ----------
    tables : 已规范化的模态表字典（含 participant_group_id 与标准键）。

    Returns
    -------
    (common, stats)：common 为共同子集（一行一探针观测，含各模态特征列并集
    与 availability 标记）；stats 为子集规模与各模态覆盖统计。
    """
    behavior = tables["behavior"]
    nir = tables["nir"]
    rgb = tables["rgb"]
    mmwave = tables["mmwave"]
    # 共同子集 = NIR 可用的 (session, block, probe) 与行为/RGB/毫米波键的交集
    common = behavior[list(KEY_COLUMNS) + ["participant_group_id"]].merge(
        nir[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner"
    )
    n_before = len(common)
    # 行为行数在 116 场口径下应为 2320；共同子集由 NIR 109 场决定
    common = common.merge(rgb[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner")
    common = common.merge(mmwave[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner")
    if len(common) != n_before:
        raise ValueError(
            f"common subset shrank after RGB/mmWave join: {n_before} -> {len(common)}; "
            "behavior/NIR keys must be a subset of RGB/mmWave keys"
        )
    stats = {
        "participant_group_n": int(common["participant_group_id"].nunique()),
        "session_n": int(common["session_id"].nunique()),
        "probe_n": int(len(common)),
        "behavior_coverage": int(len(common)),
        "nir_coverage": int(len(common)),
        "mmwave_coverage": int(len(common)),
        "rgb_coverage": int(len(common)),
    }
    return common, stats


def load_aligned_tables(
    data_root: Path,
    *,
    outcome_columns: tuple[str, str] = ("q1_nominal_4class", "q2_ordinal_4level"),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """只读加载并对齐四模态 probe 表，输出共同子集与对齐审计。

    Parameters
    ----------
    data_root : 正式分析数据根（_FormalAnalysis）。
    outcome_columns : 行为表内 Q1/Q2 权威标签列名。

    Returns
    -------
    (tables, common, audit)：tables 为规范化后四模态表；common 为共同子集
    键表（participant_group_id/session_id/block_id/probe_index_in_block）；
    audit 为对齐审计 dict（表清单、标签一致性、身份桥、共同子集规模、问题）。
    """
    problems: list[str] = []
    bridge_path = data_root / _INPUT_PATHS["bridge"]
    bridge = None
    if bridge_path.is_file():
        bridge = pd.read_csv(bridge_path, encoding="utf-8-sig", low_memory=False)
    else:
        problems.append(f"missing_input:{bridge_path}")

    tables: dict[str, pd.DataFrame] = {}
    table_audit: dict[str, Any] = {}
    for modality in ("behavior", "nir", "mmwave", "rgb"):
        frame, frame_problems = load_modality_table(data_root, modality, bridge=bridge)
        problems.extend(frame_problems)
        tables[modality] = frame
        table_audit[modality] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "sessions": int(frame["session_id"].nunique()) if not frame.empty else 0,
            "groups": int(frame["participant_group_id"].nunique()) if not frame.empty and "participant_group_id" in frame.columns else None,
            "duplicate_keys": int(frame.duplicated(subset=list(KEY_COLUMNS)).sum()) if not frame.empty else 0,
            "problems": frame_problems,
        }

    # 标签一致性核验（行为为权威）
    label_q1, mismatch_q1 = build_label_consistency(tables, "q1")
    label_q2, mismatch_q2 = build_label_consistency(tables, "q2")

    # 身份桥审计（毫米波 R 编码 -> P 编码）
    identity_audit: dict[str, Any] = {}
    mm = tables["mmwave"]
    if not mm.empty and bridge is not None:
        r2p = mm[["repeat_participant_id", "participant_group_id"]].drop_duplicates()
        split_mask = mm[_BRIDGE_SPLIT_FLAG_COLUMN].fillna(False).astype(bool) \
            if _BRIDGE_SPLIT_FLAG_COLUMN in mm.columns else pd.Series(False, index=mm.index)
        split_sessions = mm.loc[split_mask, ["session_id", "repeat_participant_id", "participant_group_id"]].drop_duplicates()
        identity_audit = {
            "bridge_rows": int(len(bridge)),
            "mmwave_rows_without_bridge_identity": int(mm["participant_group_id"].isna().sum()),
            "r_codes_mapping_to_multiple_groups": int(
                r2p.groupby("repeat_participant_id")["participant_group_id"].nunique().gt(1).sum()
            ),
            # 分裂统计口径：去重后的场次/参与者（桥表 2 行 = 2 场 = 1 个 P 组）
            "split_session_n": int(split_sessions["session_id"].nunique()) if not split_sessions.empty else 0,
            "split_participant_group_n": int(split_sessions["participant_group_id"].nunique()) if not split_sessions.empty else 0,
            "split_rows": split_sessions.to_dict("records"),
        }

    # 共同子集（由 NIR 可用场次约束的四模态共同可用观测）
    common, common_stats = build_common_subset(tables)
    # 行为标签与身份附加到共同子集
    behavior = tables["behavior"]
    common = common.merge(
        behavior[list(KEY_COLUMNS) + list(outcome_columns)],
        on=list(KEY_COLUMNS), how="left",
    )
    unresolved_label = int(common[list(outcome_columns)].isna().any(axis=1).sum())
    if unresolved_label:
        problems.append(f"common_subset_rows_without_behavior_label:{unresolved_label}")

    audit = {
        "status": "PASS_ALIGNMENT_AUDIT" if not problems and mismatch_q1 == 0 and mismatch_q2 == 0 else "FAILED_ALIGNMENT_AUDIT",
        "input_tables": table_audit,
        "key_contract": {
            "standard_keys": list(KEY_COLUMNS),
            "identity_key": "participant_group_id",
            "primary_window": PRIMARY_WINDOW,
            "block_normalization": "B1/B2, 1/2, block-1/block-2 -> b1/b2",
        },
        "label_consistency": {
            "q1": label_q1.to_dict("records"),
            "q2": label_q2.to_dict("records"),
            "total_mismatch": mismatch_q1 + mismatch_q2,
            "authority": "behavior table labels are authoritative; other modalities only checked",
        },
        "identity_bridge": identity_audit,
        "common_subset": common_stats,
        "problems": problems,
    }
    return tables, common, audit


def modality_feature_coverage(
    tables: dict[str, pd.DataFrame],
    common: pd.DataFrame,
    feature_blocks: dict[str, Any],
) -> pd.DataFrame:
    """计算共同子集内各模态主分析特征的非缺失覆盖（与性能同报）。

    Parameters
    ----------
    tables : 规范化模态表。
    common : 共同子集键表（含 participant_group_id 与标签）。
    feature_blocks : config 的 feature_blocks 段（nir 为 metrics 列表）。

    Returns
    -------
    一行一特征列：non_missing / total / coverage，附所属模态。
    """
    merged = common.copy()
    for modality in ("behavior", "mmwave", "rgb"):
        block = feature_blocks.get(modality, [])
        cols = [c for c in block if isinstance(c, str)]
        if not cols:
            continue
        frame = tables[modality]
        if frame.empty:
            continue
        merged = merged.merge(frame[list(KEY_COLUMNS) + cols], on=list(KEY_COLUMNS), how="left")
    # NIR 主指标（within/between 分解前的原始列）
    nir_cfg = feature_blocks.get("nir_primary", {})
    nir_metrics = [str(m) for m in nir_cfg.get("metrics", [])]
    if nir_metrics and not tables["nir"].empty:
        merged = merged.merge(
            tables["nir"][list(KEY_COLUMNS) + nir_metrics], on=list(KEY_COLUMNS), how="left"
        )

    rows: list[dict[str, Any]] = []
    modality_of = {}
    for modality, block in feature_blocks.items():
        if modality == "nir_primary":
            for metric in block.get("metrics", []):
                modality_of[metric] = "nir"
        else:
            for col in block:
                if isinstance(col, str):
                    modality_of[col] = modality
    total = len(merged)
    for col, modality in modality_of.items():
        if col not in merged.columns:
            rows.append({"modality": modality, "feature": col, "non_missing": 0,
                         "total": total, "coverage": 0.0, "note": "column_missing"})
            continue
        non_missing = int(merged[col].notna().sum())
        rows.append({
            "modality": modality, "feature": col, "non_missing": non_missing,
            "total": total, "coverage": round(non_missing / total, 6),
            "note": "",
        })
    return pd.DataFrame(rows)


def write_alignment_audit(
    output_dir: Path,
    *,
    tables: dict[str, pd.DataFrame],
    common: pd.DataFrame,
    audit: dict[str, Any],
    feature_coverage: pd.DataFrame,
) -> dict[str, str]:
    """把对齐审计产物写入独立输出目录（不改任何输入）。

    Parameters
    ----------
    output_dir : run 目录下的 common_cohort 子目录。
    tables/common/audit/feature_coverage : load_aligned_tables 及覆盖率产物。

    Returns
    -------
    写出的文件路径字典。
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    common_manifest = common[list(KEY_COLUMNS) + ["participant_group_id"]].drop_duplicates()
    target = output_dir / "common_cohort_manifest.csv"
    common_manifest.to_csv(target, index=False, encoding="utf-8-sig")
    paths["common_cohort_manifest"] = str(target)

    target = output_dir / "alignment_audit.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    paths["alignment_audit"] = str(target)

    target = output_dir / "feature_coverage.csv"
    feature_coverage.to_csv(target, index=False, encoding="utf-8-sig")
    paths["feature_coverage"] = str(target)

    label_rows: list[dict[str, Any]] = []
    for outcome in ("q1", "q2"):
        label_rows.extend(audit["label_consistency"][outcome])
    target = output_dir / "label_consistency.csv"
    pd.DataFrame(label_rows).to_csv(target, index=False, encoding="utf-8-sig")
    paths["label_consistency"] = str(target)
    return paths
