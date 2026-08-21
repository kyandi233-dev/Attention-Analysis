from __future__ import annotations

import math
import re
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.behavior.evidence import rolling_evidence, summarize_window
from attention_pipeline.behavior.extract import block_metrics, extract_trials
from attention_pipeline.behavior.reporting import (
    _design_audit,
    _field_dictionary,
    _program_recommendations,
    load_cohort,
    phase1_tables,
)
from attention_pipeline.behavior.evidence import cohort_probe_evidence
from attention_pipeline.config import load_config
from attention_pipeline.contracts import EYE_LEFT, EYE_RIGHT, PROBE_LABELS
from attention_pipeline.io import block_windows, load_timestamps, nearest_written_frame
from attention_pipeline.nir.metrics import rolling_perclos
from attention_pipeline.nir.review import (
    gate1_frame_plan,
    layer_status,
    preview_frame_plan,
    render_review_html,
    representative_frame_plan,
    resolve_gate1_dir,
    resolve_truth_dir,
)
from attention_pipeline.nir.roi import (
    EYE_CORNERS,
    ellipse_from_three_points,
    inverse_affine,
    map_ellipse_to_source,
    normalized_eye_roi,
    roi_border_status,
    transform_points,
)
from attention_pipeline.protocol import validate_protocol


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS  {name}")


def main() -> int:
    config = load_config(ROOT / "configs" / "preexperiment.yaml")
    protocol = validate_protocol(config)
    check("正式 BBB、18×24、30个双问题探针与432行协议", protocol["ok"])
    check("正式探针为注意4分类＋警觉度4点双问题调度", config.section("formal_protocol")["schedule_version"] == "sched-v1.0-4cat-vig-20260814")
    check("历史四类探针名义标签保留", set(PROBE_LABELS) == {1, 2, 3, 4})
    check(
        "历史 protocol 保持 ABCCBA 与固定探针位置",
        config.section("protocol")["block_order"] == ["A", "B", "C", "C", "B", "A"]
        and config.section("protocol")["probe_after_trials"] == [30, 82, 137, 191],
    )
    check("专注评分与正式 NIR 在审批门1关闭", not config.section("stages")["attention_score"] and not config.section("stages")["nir_extract"])

    trials = extract_trials(config, "sub-000")
    check("正式 RT 不被阈值删除", len(trials) == 1296 and trials["rt"].notna().sum() > 0)
    check("RT QC 标记存在", {"rt_qc_lt_100", "rt_qc_lt_150", "rt_qc_gt_1000", "rt_qc_gt_1150"}.issubset(trials))
    check("condition × position_in_cycle 保留", trials["condition_x_position"].notna().all())

    dframe = pd.DataFrame({
        "subject_id": ["x"] * 10, "block_num": [1] * 10, "condition": ["A"] * 10,
        "is_no_go": [0] * 8 + [1] * 2, "correct": [1] * 6 + [0] * 2 + [1] * 2,
        "commission": [0] * 8 + [1, 0], "rt": [200] * 6 + [np.nan] * 4,
        "omission": [0] * 6 + [1] * 2 + [0] * 2,
    })
    metric = block_metrics(dframe).iloc[0]
    expected = NormalDist().inv_cdf(6.5 / 9) - NormalDist().inv_cdf(1.5 / 3)
    check("d′ Hit=正确Go、FA=No-Go commission", math.isclose(metric["dprime_loglinear"], expected))

    synthetic = []
    for block_num, base in ((1, 0), (2, 1_000_000)):
        for index in range(20):
            synthetic.append({
                "subject_id": "x", "block_num": block_num, "condition": "A",
                "absolute_onset_time": base + index * 1000,
                "is_no_go": int(index % 4 == 0), "commission": int(index == 8),
                "rt": np.nan if index % 4 == 0 else 250 + index,
            })
    synthetic = pd.DataFrame(synthetic)
    windows = rolling_evidence(config, synthetic)
    check("行为证据窗不跨 block", windows.query("block_num == 2")["window_end_ms"].min() >= 1_000_000)
    summary = summarize_window(synthetic.query("block_num == 1"), 19_000, 30, 6)
    check("No-Go不足与Jeffreys区间显式输出", summary["window_status"] == "insufficient_nogo" and summary["commission_jeffreys_ci95_low"] < summary["commission_jeffreys_ci95_high"])

    with tempfile.TemporaryDirectory() as directory:
        timestamp_path = Path(directory) / "timestamps.csv"
        timestamp_path.write_text("10,1000,\n11,1033,dropped\n12,1066,\n", encoding="utf-8")
        timestamps = load_timestamps(timestamp_path)
        mapped = nearest_written_frame(timestamps, 1050)
        check("dropped 不占 AVI 位置", mapped["capture_frame_idx"] == 12 and mapped["avi_frame_idx"] == 1)

    timeline = config.path_value("raw_root") / "sub-000_" / "beh" / "master_timeline.csv"
    check("正式时间轴解析为六个 block", len(block_windows(timeline)) == 6)
    check("双眼使用被试解剖命名", EYE_CORNERS[EYE_RIGHT] == (33, 133) and EYE_CORNERS[EYE_LEFT] == (362, 263))

    image = np.zeros((300, 500, 3), dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    points[33], points[133] = (100, 120), (200, 120)
    roi, affine, _ = normalized_eye_roi(image, points, (33, 133))
    mapped_points = transform_points(points[[33, 133]], affine)
    check("固定 ROI 与仿射坐标", roi.shape == (160, 320, 3) and np.allclose(mapped_points, [[80, 80], [240, 80]]))
    ellipse = ellipse_from_three_points((100, 80), (130, 80), (100, 90))
    check("三点椭圆可恢复轴长", ellipse["major_diameter"] == 60 and ellipse["minor_diameter"] == 20)
    perclos = rolling_perclos([0.1, np.nan, 0.4], [0, 100, 200], threshold=0.2)
    check("缺失 EAR 不等于闭眼且不进分母", math.isclose(perclos[-1], 0.5))

    # --- 阶段0+1 新增：审批门1 与 ROI 补强契约（纯函数/只读，不读 AVI、不跑 MediaPipe） ---
    gate1 = gate1_frame_plan(config)
    check("审批门1帧计划=12帧且 frame_purpose 非空", len(gate1) == 12 and gate1["frame_purpose"].notna().all() and gate1["frame_purpose"].ne("").all())
    for _, row in gate1.iterrows():
        timestamp_rows = load_timestamps(Path(row["nir_timestamps"]))
        match = timestamp_rows.loc[timestamp_rows["capture_frame_idx"].eq(int(row["capture_frame_idx"]))]
        assert len(match) == 1 and not bool(match.iloc[0]["is_dropped"]), f"{row['subject']}/{row['capture_frame_idx']}"
        window = next(w for w in block_windows(Path(row["master_timeline"])) if w["block_num"] == int(row["block_num"]))
        assert window["start_ms"] <= float(match.iloc[0]["unix_ms"]) <= window["end_ms"], f"{row['subject']}/{row['capture_frame_idx']} 不在声明 block"
    check("审批门1每帧与时间戳/时间线逐帧一致（存在、非dropped、在声明block）", True)

    image = np.zeros((300, 500, 3), dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    points[33], points[133] = (100, 120), (200, 120)
    roi, affine, _ = normalized_eye_roi(image, points, (33, 133))
    original = np.array([[100.0, 120.0], [200.0, 120.0], [150.0, 145.0]])
    round_trip = transform_points(transform_points(original, affine), inverse_affine(affine))
    check("逆仿射坐标往返误差<1e-3", np.allclose(round_trip, original, atol=1e-3))
    check("ROI 完全在源图内→ready", roi_border_status((300, 500), affine) == "ready")
    check("ROI 四角越界→border_heavy", roi_border_status((120, 500), affine) == "border_heavy")

    scaled = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    mapped = map_ellipse_to_source((100, 50), (140, 50), (100, 80), inverse_affine(scaled))
    check(
        "椭圆映射回源图保形（轴长按 1/scale 还原）",
        math.isclose(mapped["major_diameter"], 40, abs_tol=1e-3)
        and math.isclose(mapped["minor_diameter"], 30, abs_tol=1e-3)
        and math.isclose(mapped["equivalent_diameter"], math.sqrt(1200), abs_tol=1e-3),
    )
    check("layer_status: no_face", layer_status(False, None, "") == {"face_status": "no_face", "roi_status": "missing", "pipeline_status": "no_face"})
    check("layer_status: 退化", layer_status(True, None, "") == {"face_status": "landmarks_invalid", "roi_status": "degenerate", "pipeline_status": "roi_missing"})
    check("layer_status: border_heavy", layer_status(True, np.zeros((4, 4), np.uint8), "border_heavy")["roi_status"] == "border_heavy")
    check("layer_status: 正常", layer_status(True, np.zeros((4, 4), np.uint8), "ready") == {"face_status": "observed", "roi_status": "ready", "pipeline_status": "ready_for_annotation"})

    with tempfile.TemporaryDirectory() as directory:
        from unittest import mock
        fake_config = mock.Mock()
        fake_config.path_value.return_value = Path(directory) / "gate1"
        fake_config.section.return_value = {"timezone": "Asia/Shanghai"}
        target = resolve_gate1_dir(fake_config)
        target.mkdir(parents=True, exist_ok=True)
        try:
            resolve_gate1_dir(fake_config)
            raise AssertionError("应拒绝覆盖已存在的审批门1目录")
        except RuntimeError:
            pass
        check("审批门1目录已存在且未 --force → 拒绝覆盖", True)
        check("--force → 允许覆盖", str(resolve_gate1_dir(fake_config, force=True)) == str(target))
    with tempfile.TemporaryDirectory() as directory:
        from unittest import mock
        fake_config = mock.Mock()
        fake_config.path_value.return_value = Path(directory) / "truth-528"
        fake_config.section.return_value = {"timezone": "Asia/Shanghai"}
        check("代表性真值目录用 truth_artifact_root", str(resolve_truth_dir(fake_config)) == str(fake_config.path_value.return_value))

    representative = representative_frame_plan(config)
    repeated = representative_frame_plan(config)
    check("264帧代表性设计完整", len(representative) == 264 and representative.groupby(["subject", "block_num"]).size().eq(4).all())
    check("代表性抽样完全可复现", representative.equals(repeated))
    check("12眼预览对应6个原始帧", len(preview_frame_plan(config, 12)) == 6)
    html = render_review_html([{
        "sample_id": "s1", "subject": "sub-000", "block_num": 1, "condition": "A",
        "temporal_stratum": 1, "eye": "eye_right", "pipeline_status": "ready_for_annotation",
        "face_status": "observed", "roi_status": "ready", "frame_purpose": "无旧crop基线",
        "context_path": "context/a.jpg", "roi_path": "roi/a.png",
    }])
    check("盲标支持复核者、三点、保存恢复和CSV导出", all(term in html for term in ("复核者 ID", "主标", "复标", "仲裁", "中心", "长轴端点", "短轴端点", "localStorage", "导出 CSV")))
    check("盲标页无算法叠加且显示分层状态", "algorithm" not in html.lower() and "Face" in html and "ROI" in html)
    check("盲标页 JS 无裸换行字符串（join/CSV 导出不被劈断）", "join('\\n')" in html and bool(re.search(r"['\"]\n['\"]", html)) is False)
    check("盲标页 JS 不用保留字 export 作标识符", "export.onclick" not in html and "getElementById('export')" in html)
    check("盲标页 JS 的 localStorage 读写被 try 包裹", "try{saved=JSON.parse(localStorage.getItem" in html.replace(" ", "") and "try{localStorage.setItem" in html.replace(" ", ""))
    check("盲标页键盘交互与放大标记", "toggleQuality" in html and "addEventListener('keydown'" in html and "aspect-ratio" in html and "arc(q.x,q.y,1.5" in html and "arc(q.x,q.y,4," not in html)
    cohort = load_cohort(config)
    audit = phase1_tables(config, cohort)
    check("11人行为审计为14256试次与264探针", len(cohort) == 14256 and int(cohort["is_probe"].sum()) == 264)
    check("行为报告不删除RT", int(audit["summary"].set_index("metric").loc["formal_trials", "value"]) == 14256)
    recommendations = _program_recommendations()
    check("程序建议按四类审批风险分层", set(recommendations["decision_class"]) == {"可直接纳入下一版", "需新实验版本验证", "需操作性定义讨论", "当前不应实施"})
    design = _design_audit().set_index("audit_item")
    check("探针界面没有默认选择", "current_choice初始为None" in design.loc["探针界面", "observed_design"])
    probes = cohort_probe_evidence(config, cohort)
    dictionary = _field_dictionary(cohort, probes).set_index(["layer", "field"])
    check("字段字典区分QC、候选证据与待批准字段", dictionary.loc[("v2逐试次", "rt_qc_lt_150"), "missing_semantics"] == "仅标记，不删除" and dictionary.loc[("探针前证据窗", "window_status"), "freeze_status"] == "候选字段，未冻结评分" and dictionary.loc[("建议新增", "schedule_hash"), "freeze_status"] == "待批准")
    print("\nGate 1 + behavior gate 3 contract checks: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

