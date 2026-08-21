import re
from pathlib import Path
from unittest import mock

import numpy as np

from attention_pipeline.io import block_windows, load_timestamps
from attention_pipeline.nir.review import (
    gate1_frame_plan,
    layer_status,
    preview_frame_plan,
    render_review_html,
    representative_frame_plan,
    resolve_gate1_dir,
)


def test_representative_sampling_is_complete_and_reproducible(config):
    first = representative_frame_plan(config)
    second = representative_frame_plan(config)
    assert len(first) == 264
    assert first.equals(second)
    assert first.groupby(["subject", "block_num"]).size().eq(4).all()


def test_preview_is_twelve_eye_samples_from_six_raw_frames(config):
    preview = preview_frame_plan(config, 12)
    assert len(preview) == 6
    assert preview.iloc[0]["subject"] == "sub-000"
    assert preview.iloc[-1]["subject"] == "sub-010"


def test_gate1_frame_plan_is_twelve_verified_frames(config):
    plan = gate1_frame_plan(config)
    assert len(plan) == 12
    assert plan["frame_purpose"].notna().all() and plan["frame_purpose"].ne("").all()
    assert plan["subject"].nunique() == 10
    for _, row in plan.iterrows():
        timestamp_rows = load_timestamps(Path(row["nir_timestamps"]))
        match = timestamp_rows.loc[timestamp_rows["capture_frame_idx"].eq(int(row["capture_frame_idx"]))].iloc[0]
        assert not bool(match["is_dropped"])
        window = next(
            w for w in block_windows(Path(row["master_timeline"])) if w["block_num"] == int(row["block_num"])
        )
        assert window["start_ms"] <= float(match["unix_ms"]) <= window["end_ms"]


def test_layer_status_four_branches():
    assert layer_status(False, None, "") == {
        "face_status": "no_face", "roi_status": "missing", "pipeline_status": "no_face",
    }
    assert layer_status(True, None, "") == {
        "face_status": "landmarks_invalid", "roi_status": "degenerate", "pipeline_status": "roi_missing",
    }
    assert layer_status(True, np.zeros((4, 4), dtype=np.uint8), "border_heavy")["roi_status"] == "border_heavy"
    assert layer_status(True, np.zeros((4, 4), dtype=np.uint8), "ready") == {
        "face_status": "observed", "roi_status": "ready", "pipeline_status": "ready_for_annotation",
    }


def test_resolve_gate1_dir_refuses_overwrite_without_force(tmp_path):
    fake_config = mock.Mock()
    fake_config.path_value.return_value = tmp_path / "gate1"
    fake_config.section.return_value = {"timezone": "Asia/Shanghai"}
    target = resolve_gate1_dir(fake_config)
    target.mkdir(parents=True, exist_ok=True)
    try:
        resolve_gate1_dir(fake_config)
        raise AssertionError("应拒绝覆盖已存在的审批门1目录")
    except RuntimeError:
        pass
    assert str(resolve_gate1_dir(fake_config, force=True)) == str(target)


def test_annotation_html_is_blind_and_supports_save_restore_export():
    html = render_review_html([{
        "sample_id": "s1", "subject": "sub-000", "block_num": 1, "condition": "A",
        "temporal_stratum": 1, "eye": "eye_right", "pipeline_status": "ready_for_annotation",
        "face_status": "observed", "roi_status": "ready", "frame_purpose": "无旧crop基线",
        "context_path": "context/a.jpg", "roi_path": "roi/a.png",
    }])
    assert "中心" in html and "长轴端点" in html and "短轴端点" in html
    assert "localStorage" in html and "导出 CSV" in html
    assert all(role in html for role in ("主标", "复标", "仲裁"))
    assert "algorithm" not in html.lower()
    assert "Face" in html and "ROI" in html


def test_annotation_html_has_no_bare_newline_inside_js_strings():
    html = render_review_html([{
        "sample_id": "s1", "subject": "sub-000", "block_num": 1, "condition": "A",
        "temporal_stratum": 1, "eye": "eye_right", "pipeline_status": "ready_for_annotation",
        "face_status": "observed", "roi_status": "ready", "frame_purpose": "无旧crop基线",
        "context_path": "context/a.jpg", "roi_path": "roi/a.png",
    }])
    assert "join('\\n')" in html
    assert re.search(r"['\"]\n['\"]", html) is None


def test_annotation_html_avoids_reserved_word_and_guards_localstorage():
    html = render_review_html([{
        "sample_id": "s1", "subject": "sub-000", "block_num": 1, "condition": "A",
        "temporal_stratum": 1, "eye": "eye_right", "pipeline_status": "ready_for_annotation",
        "face_status": "observed", "roi_status": "ready", "frame_purpose": "无旧crop基线",
        "context_path": "context/a.jpg", "roi_path": "roi/a.png",
    }])
    # export 是 JS 保留字，不能作为裸标识符
    assert "export.onclick" not in html
    assert "getElementById('export')" in html
    # localStorage 读写必须 try 包裹，避免 file:// 下 SecurityError 致空白页
    assert "try{saved=JSON.parse(localStorage.getItem" in html.replace(" ", "")
    assert "try{localStorage.setItem" in html.replace(" ", "")


def test_annotation_html_has_keyboard_shortcuts_and_zoomed_markers():
    html = render_review_html([{
        "sample_id": "s1", "subject": "sub-000", "block_num": 1, "condition": "A",
        "temporal_stratum": 1, "eye": "eye_right", "pipeline_status": "ready_for_annotation",
        "face_status": "observed", "roi_status": "ready", "frame_purpose": "无旧crop基线",
        "context_path": "context/a.jpg", "roi_path": "roi/a.png",
    }])
    assert "toggleQuality" in html
    assert "addEventListener('keydown'" in html
    assert "aspect-ratio" in html          # 画布放大
    assert "arc(q.x,q.y,1.5" in html       # 小标记点
    assert "arc(q.x,q.y,4," not in html    # 旧大圆点已移除
