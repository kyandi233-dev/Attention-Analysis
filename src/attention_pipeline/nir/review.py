from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..contracts import EYES
from ..io import block_windows, load_timestamps, nearest_written_frame, subject_paths
from ..metadata import run_metadata, source_id
from .roi import EYE_CORNERS, inverse_affine, normalized_eye_roi, roi_border_status


STRATUM_FRACTIONS = (0.125, 0.375, 0.625, 0.875)


def representative_frame_plan(config: Config) -> pd.DataFrame:
    rows = []
    raw_root = config.path_value("raw_root")
    for subject in config.data["subjects"]["include"]:
        paths = subject_paths(raw_root, subject)
        timestamps = load_timestamps(paths["nir_timestamps"])
        for block in block_windows(paths["master_timeline"]):
            for stratum, fraction in enumerate(STRATUM_FRACTIONS, start=1):
                target_ms = block["start_ms"] + fraction * (block["end_ms"] - block["start_ms"])
                mapped = nearest_written_frame(timestamps, target_ms)
                rows.append({
                    "subject": subject,
                    "block_num": block["block_num"],
                    "condition": block["condition"],
                    "temporal_stratum": stratum,
                    "target_fraction": fraction,
                    "target_ms": target_ms,
                    **mapped,
                    "nir_video": str(paths["nir_video"].resolve()),
                    "nir_timestamps": str(paths["nir_timestamps"].resolve()),
                    "master_timeline": str(paths["master_timeline"].resolve()),
                })
    result = pd.DataFrame(rows)
    if len(result) != 264:
        raise AssertionError(f"代表性原始帧计划应为 264，实际 {len(result)}")
    return result


def preview_frame_plan(config: Config, eye_samples: int = 12) -> pd.DataFrame:
    if eye_samples <= 0 or eye_samples % 2:
        raise ValueError("preview-eyes 必须为正偶数；每个原始视频帧包含双眼")
    full = representative_frame_plan(config)
    frame_count = eye_samples // 2
    indices = np.linspace(0, len(full) - 1, frame_count, dtype=int)
    return full.iloc[indices].reset_index(drop=True)


# 审批门1：12 个难度/压力覆盖原始帧（011 审计计划「第一审批门」表）。
# 帧号与 block 已在生成前经时间戳文件与 master_timeline 逐帧核验（存在、非 dropped、在声明 block 窗口内）。
GATE1_FRAMES = [
    {"subject": "sub-000", "block_num": 1, "capture_frame_idx": 10110, "frame_purpose": "无旧crop基线"},
    {"subject": "sub-002", "block_num": 2, "capture_frame_idx": 19503, "frame_purpose": "双眼no_face"},
    {"subject": "sub-004", "block_num": 3, "capture_frame_idx": 33858, "frame_purpose": "旧/new ROI配对"},
    {"subject": "sub-006", "block_num": 4, "capture_frame_idx": 40780, "frame_purpose": "旧/new ROI配对"},
    {"subject": "sub-008", "block_num": 5, "capture_frame_idx": 60944, "frame_purpose": "反光/质量压力"},
    {"subject": "sub-010", "block_num": 6, "capture_frame_idx": 78220, "frame_purpose": "无旧crop"},
    {"subject": "sub-000", "block_num": 1, "capture_frame_idx": 11392, "frame_purpose": "历史代理不可见压力"},
    {"subject": "sub-001", "block_num": 3, "capture_frame_idx": 34801, "frame_purpose": "P80-closed旧检测未返回"},
    {"subject": "sub-002", "block_num": 4, "capture_frame_idx": 46572, "frame_purpose": "P80-closed但旧检测返回"},
    {"subject": "sub-003", "block_num": 3, "capture_frame_idx": 34950, "frame_purpose": "部分开合边界"},
    {"subject": "sub-005", "block_num": 2, "capture_frame_idx": 28523, "frame_purpose": "可见低开放度旧检测返回"},
    {"subject": "sub-009", "block_num": 5, "capture_frame_idx": 56818, "frame_purpose": "可见低开放度旧检测未返回"},
]


def gate1_frame_plan(config: Config) -> pd.DataFrame:
    """按 GATE1_FRAMES 显式清单构建审批门1帧计划。

    每帧经 load_timestamps 映射到 AVI 位置并校验：存在、非 dropped、
    unix_ms 落在声明 block 的窗口内。任一失败 raise，不静默替换补抽。
    temporal_stratum 由帧在 block 内的位置分数换算（1–4），保留既有 schema。
    """
    raw_root = config.path_value("raw_root")
    rows = []
    for item in GATE1_FRAMES:
        subject = item["subject"]
        paths = subject_paths(raw_root, subject)
        timestamps = load_timestamps(paths["nir_timestamps"])
        matches = timestamps.loc[timestamps["capture_frame_idx"].eq(item["capture_frame_idx"])]
        if matches.empty:
            raise AssertionError(f"{subject} 时间戳缺少 capture frame {item['capture_frame_idx']}")
        frame_row = matches.iloc[0]
        if bool(frame_row["is_dropped"]) or not bool(np.isfinite(frame_row["unix_ms"])):
            raise AssertionError(f"{subject} capture frame {item['capture_frame_idx']} 为 dropped/无时间戳，禁止补抽")
        windows = block_windows(paths["master_timeline"])
        window = next((w for w in windows if w["block_num"] == item["block_num"]), None)
        if window is None:
            raise AssertionError(f"{subject} 没有 Block{item['block_num']} 窗口")
        unix_ms = float(frame_row["unix_ms"])
        if not (window["start_ms"] <= unix_ms <= window["end_ms"]):
            raise AssertionError(f"{subject} capture frame {item['capture_frame_idx']} 不在 Block{item['block_num']} 窗口内")
        fraction = (unix_ms - window["start_ms"]) / (window["end_ms"] - window["start_ms"])
        rows.append({
            "subject": subject,
            "block_num": item["block_num"],
            "condition": window["condition"],
            "temporal_stratum": min(int(fraction * 4), 3) + 1,
            "target_fraction": fraction,
            "target_ms": unix_ms,
            "capture_frame_idx": int(frame_row["capture_frame_idx"]),
            "avi_frame_idx": int(frame_row["avi_frame_idx"]),
            "unix_ms": unix_ms,
            "target_error_ms": 0.0,
            "frame_purpose": item["frame_purpose"],
            "nir_video": str(paths["nir_video"].resolve()),
            "nir_timestamps": str(paths["nir_timestamps"].resolve()),
            "master_timeline": str(paths["master_timeline"].resolve()),
        })
    result = pd.DataFrame(rows)
    if len(result) != 12:
        raise AssertionError(f"审批门1帧计划应为 12，实际 {len(result)}")
    return result


def layer_status(points_present: bool, roi, border_status: str) -> dict:
    """把 Face/ROI 层合并成分层状态三元组 {face_status, roi_status, pipeline_status}。"""
    if not points_present:
        return {"face_status": "no_face", "roi_status": "missing", "pipeline_status": "no_face"}
    if roi is None:
        return {"face_status": "landmarks_invalid", "roi_status": "degenerate", "pipeline_status": "roi_missing"}
    if border_status == "border_heavy":
        return {"face_status": "observed", "roi_status": "border_heavy", "pipeline_status": "ready_for_annotation"}
    return {"face_status": "observed", "roi_status": "ready", "pipeline_status": "ready_for_annotation"}


def _face_landmarks(image_bgr: np.ndarray, model_path: Path):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("生成真实 ROI 需要 mediapipe；请使用主环境或安装项目 [nir] 依赖") from exc
    usable_model = model_path
    if not str(model_path).isascii():
        usable_model = Path(tempfile.gettempdir()) / "attention_pipeline_v2_face_landmarker.task"
        if not usable_model.exists() or usable_model.stat().st_size != model_path.stat().st_size:
            shutil.copy2(model_path, usable_model)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(usable_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = image_bgr.shape[:2]
    return np.array([[point.x * w, point.y * h] for point in result.face_landmarks[0]], dtype=np.float32)


def _read_frame(video_path: Path, avi_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, avi_frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"无法读取 {video_path} AVI frame {avi_frame_idx}")
        return frame
    finally:
        cap.release()


def _imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    path.write_bytes(encoded.tobytes())


def _old_crop(config: Config, subject: str, eye: str, capture_idx: int, avi_idx: int) -> str:
    root = config.path_value("legacy_crop_root") / f"{subject}_" / eye
    for idx in (capture_idx, avi_idx):
        candidate = root / f"frame_{idx:08d}.png"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def resolve_review_dir(config: Config, key: str, force: bool = False) -> Path:
    """固定输出目录（artifacts/<key>），已存在且未 --force → 拒绝。"""
    output = config.path_value(key)
    if output.exists() and not force:
        raise RuntimeError(f"目录已存在: {output}（如需覆盖请显式加 --force）")
    return output


def resolve_gate1_dir(config: Config, tag: str | None = None, force: bool = False) -> Path:
    return resolve_review_dir(config, "gate1_artifact_root", force)


def resolve_truth_dir(config: Config, tag: str | None = None, force: bool = False) -> Path:
    return resolve_review_dir(config, "truth_artifact_root", force)


def resolve_preview_dir(config: Config, force: bool = False) -> Path:
    return resolve_review_dir(config, "preview_artifact_root", force)


def _build_review(config: Config, plan: pd.DataFrame, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    """共享复核包构建核心：读取计划帧、写上下文/ROI、生成分层状态 manifest 与盲标页。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    metadata = run_metadata(config)
    source_cache: dict[str, str] = {}
    roi_size = tuple(config.section("nir")["review"]["roi_size"])
    span = float(config.section("nir")["review"]["corner_span_fraction"])
    for _, row in plan.iterrows():
        frame = _read_frame(Path(row["nir_video"]), int(row["avi_frame_idx"]))
        for source_column in ("nir_video", "nir_timestamps", "master_timeline"):
            source_path = str(row[source_column])
            if source_path not in source_cache:
                source_cache[source_path] = source_id(Path(source_path))
        base = f"{row['subject']}_b{int(row['block_num'])}_s{int(row['temporal_stratum'])}_f{int(row['capture_frame_idx']):08d}"
        context_rel = Path("context") / f"{base}.jpg"
        _imwrite(output_dir / context_rel, frame)
        points = _face_landmarks(frame, config.path_value("face_landmarker_model"))
        for eye in EYES:
            sample_id = f"{base}_{eye}"
            roi_rel = ""
            affine_json = ""
            inverse_json = ""
            corner_distance = np.nan
            border_status = ""
            roi = None
            if points is not None:
                roi, affine, corner_distance = normalized_eye_roi(frame, points, EYE_CORNERS[eye], roi_size, span)
                if roi is not None:
                    border_status = roi_border_status(frame.shape[:2], affine, roi_size)
                    roi_path = Path("roi") / f"{sample_id}.png"
                    _imwrite(output_dir / roi_path, roi)
                    roi_rel = roi_path.as_posix()
                    affine_json = json.dumps(affine.tolist(), separators=(",", ":"))
                    inverse_json = json.dumps(inverse_affine(affine).tolist(), separators=(",", ":"))
            layered = layer_status(points is not None, roi, border_status)
            review_rows.append({
                "sample_id": sample_id,
                "subject": row["subject"],
                "block_num": int(row["block_num"]),
                "condition": row["condition"],
                "temporal_stratum": int(row["temporal_stratum"]),
                "capture_frame_idx": int(row["capture_frame_idx"]),
                "avi_frame_idx": int(row["avi_frame_idx"]),
                "unix_ms": int(row["unix_ms"]),
                "eye": eye,
                "pipeline_status": layered["pipeline_status"],
                "face_status": layered["face_status"],
                "roi_status": layered["roi_status"],
                "frame_purpose": str(row.get("frame_purpose", "")),
                "context_path": context_rel.as_posix(),
                "roi_path": roi_rel,
                "old_crop_path": _old_crop(config, row["subject"], eye, int(row["capture_frame_idx"]), int(row["avi_frame_idx"])),
                "corner_distance_source_px": corner_distance,
                "source_to_roi_affine": affine_json,
                "roi_to_source_affine": inverse_json,
                "source_video": row["nir_video"],
                "source_video_id": source_cache[row["nir_video"]],
                "source_timestamps": row["nir_timestamps"],
                "source_timestamps_id": source_cache[row["nir_timestamps"]],
                "source_master_timeline": row["master_timeline"],
                "source_master_timeline_id": source_cache[row["master_timeline"]],
                "pipeline_version": metadata["pipeline_version"],
                "config_digest": metadata["config_digest"],
                "generated_at": metadata["generated_at"],
            })
    review = pd.DataFrame(review_rows)
    review.to_csv(output_dir / "review_manifest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "review_manifest.json").write_text(
        review.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )
    html_path = output_dir / "review.html"
    html_path.write_text(render_review_html(review.to_dict(orient="records")), encoding="utf-8")
    contact_sheet(output_dir, review)
    return html_path, review


def build_preview(config: Config, eye_samples: int = 12) -> tuple[Path, pd.DataFrame]:
    plan = preview_frame_plan(config, eye_samples)
    return _build_review(config, plan, resolve_preview_dir(config))


def build_gate1_review(config: Config, tag: str | None = None, force: bool = False) -> tuple[Path, pd.DataFrame, Path]:
    output = resolve_gate1_dir(config, tag, force)
    html_path, review = _build_review(config, gate1_frame_plan(config), output)
    return html_path, review, output


def build_representative_review(config: Config, tag: str | None = None, force: bool = False) -> tuple[Path, pd.DataFrame, Path]:
    """阶段3：264 原始帧 / 528 单眼代表性集（固定四分位中点抽样，确定性可复现）。"""
    output = resolve_truth_dir(config, tag, force)
    html_path, review = _build_review(config, representative_frame_plan(config), output)
    return html_path, review, output


def contact_sheet(output: Path, review: pd.DataFrame, per_page: int = 40) -> Path:
    """把可用 ROI 拼成总览图；每页最多 per_page 眼，超过自动分页（528 眼拼一张会超大）。"""
    cards = []
    for _, row in review.iterrows():
        if not row["roi_path"]:
            continue
        roi = cv2.imdecode(np.fromfile(output / row["roi_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        canvas = cv2.copyMakeBorder(roi, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(canvas, row["sample_id"], (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cards.append(canvas)
    if not cards:
        raise RuntimeError("预览没有可用 ROI，检查 MediaPipe 与原始帧")
    pages = []
    for start in range(0, len(cards), per_page):
        page = cards[start : start + per_page]
        while len(page) % 2:
            page.append(np.full_like(page[0], 255))
        pages.append(np.vstack([np.hstack(page[i : i + 2]) for i in range(0, len(page), 2)]))
    first = None
    for idx, sheet in enumerate(pages, start=1):
        name = "000-contact-sheet.jpg" if len(pages) == 1 else f"contact-sheet-{idx:02d}.jpg"
        path = output / name
        _imwrite(path, sheet)
        if first is None:
            first = path
    return first


def render_review_html(samples: list[dict]) -> str:
    payload = json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>NIR 三点椭圆盲标预览</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei";margin:0;background:#f4f6f8;color:#17202a}}header{{padding:16px 24px;background:#17202a;color:white}}
main{{display:grid;grid-template-columns:360px 1fr;gap:18px;padding:18px}}.panel{{background:white;border-radius:10px;padding:14px;box-shadow:0 2px 9px #0001}}
canvas{{width:min(100%,800px);aspect-ratio:2/1;height:auto;image-rendering:auto;border:1px solid #ccd1d1;cursor:crosshair}}button,select,input{{margin:4px;padding:7px}}.quality label{{display:block}}.meta{{font-size:13px;line-height:1.55}}.hint{{color:#566573;font-size:13px}}
</style></head><body><header><b>NIR 三点椭圆盲标｜审批门 1 预览</b>　<span id="progress"></span></header>
<main><section class="panel"><div><button id="prev">上一帧</button><button id="next">下一帧</button><button id="reset">重置三点</button></div>
<div class="meta" id="meta"></div><hr><label>复核者 ID <input id="reviewer" placeholder="必填，如 R01"></label><label>角色 <select id="role"><option value="primary">主标</option><option value="secondary">复标</option><option value="adjudicator">仲裁</option></select></label><br><label>瞳孔可见性 <select id="visibility"><option value="">未标</option><option>可见</option><option>不确定</option><option>不可见</option></select></label>
<div class="quality"><b>质量/遮挡</b><label><input type="checkbox" value="blur">模糊</label><label><input type="checkbox" value="overexposure">过曝</label><label><input type="checkbox" value="reflection">反光</label><label><input type="checkbox" value="glasses_frame">镜框</label><label><input type="checkbox" value="eyelid_occlusion">眼睑遮挡</label></div>
<label>备注<br><input id="note" style="width:92%"></label><hr><button id="export">导出 CSV</button><p class="hint">快捷键：1可见 2不确定 3不可见｜B模糊 O过曝 R反光 G镜框 L眼睑｜←→切帧 X重置三点。可见时点三点：中心→长轴端点→短轴端点。</p></section>
<section class="panel"><canvas id="canvas" width="320" height="160"></canvas><p id="clickHint" class="hint"></p><details><summary>原始全帧上下文</summary><img id="context" style="max-width:100%"></details></section></main>
<script>
const samples={payload}; let index=0; const key='attention-v2-gate1-annotations'; let saved={{}}; try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}');}}catch(e){{saved={{}};}}
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),img=new Image(); const stages=['中心','长轴端点','短轴端点'];
function current(){{return samples[index]}} function recordKey(sample=current()){{return (reviewer.value.trim()||'anonymous')+'::'+sample.sample_id}} function record(){{const k=recordKey();return saved[k]||(saved[k]={{points:[],quality:[]}})}}
function persist(){{const r=record();r.reviewer_id=reviewer.value.trim();r.reviewer_role=role.value;r.visibility=visibility.value;r.note=note.value;r.quality=[...document.querySelectorAll('.quality input:checked')].map(x=>x.value);r.annotated_at=new Date().toISOString();try{{localStorage.setItem(key,JSON.stringify(saved));}}catch(e){{}}}}
function toggleQuality(v){{const b=document.querySelector('.quality input[value="'+v+'"]');if(b){{b.checked=!b.checked;persist();}}}}
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);if(img.complete&&img.naturalWidth)ctx.drawImage(img,0,0,canvas.width,canvas.height);const p=record().points;p.forEach((q,i)=>{{const col=['#e74c3c','#2ecc71','#3498db'][i];ctx.strokeStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(q.x-4,q.y);ctx.lineTo(q.x+4,q.y);ctx.moveTo(q.x,q.y-4);ctx.lineTo(q.x,q.y+4);ctx.stroke();ctx.beginPath();ctx.arc(q.x,q.y,1.5,0,Math.PI*2);ctx.fillStyle=col;ctx.fill()}});if(p.length===3){{const rx=Math.hypot(p[1].x-p[0].x,p[1].y-p[0].y),ry=Math.hypot(p[2].x-p[0].x,p[2].y-p[0].y),angle=Math.atan2(p[1].y-p[0].y,p[1].x-p[0].x);ctx.strokeStyle='#f1c40f';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0].x,p[0].y);ctx.lineTo(p[1].x,p[1].y);ctx.moveTo(p[0].x,p[0].y);ctx.lineTo(p[2].x,p[2].y);ctx.stroke();ctx.beginPath();ctx.ellipse(p[0].x,p[0].y,rx,ry,angle,0,Math.PI*2);ctx.stroke()}}clickHint.textContent=p.length<3?'下一点：'+stages[p.length]:'三点已完成';}}
function load(){{const s=current(),r=record();progress.textContent=`${{index+1}} / ${{samples.length}}`;meta.innerHTML=`<b>${{s.sample_id}}</b><br>被试 ${{s.subject}}｜Block ${{s.block_num}} ${{s.condition}}｜时间分层 ${{s.temporal_stratum}}<br>眼别 ${{s.eye}}｜Face ${{s.face_status||'-'}}｜ROI ${{s.roi_status||'-'}}<br>${{s.frame_purpose||''}}`;role.value=r.reviewer_role||role.value;visibility.value=r.visibility||'';note.value=r.note||'';document.querySelectorAll('.quality input').forEach(x=>x.checked=(r.quality||[]).includes(x.value));context.src=s.context_path;img.onload=draw;img.src=s.roi_path||s.context_path;draw();}}
canvas.onclick=e=>{{if(!current().roi_path)return alert('该样本为端到端 ROI 失败，仅标可见性和质量；不在变形的全帧上标椭圆。');if(visibility.value!=='可见')return alert('仅"可见"状态需要三点椭圆');const r=record();if(r.points.length>=3)return;const b=canvas.getBoundingClientRect();r.points.push({{x:(e.clientX-b.left)*canvas.width/b.width,y:(e.clientY-b.top)*canvas.height/b.height}});persist();draw()}};
reviewer.onchange=load;role.onchange=persist;visibility.onchange=()=>{{if(visibility.value!=='可见')record().points=[];persist();draw()}};note.oninput=persist;document.querySelectorAll('.quality input').forEach(x=>x.onchange=persist);prev.onclick=()=>{{persist();index=Math.max(0,index-1);load()}};next.onclick=()=>{{persist();index=Math.min(samples.length-1,index+1);load()}};reset.onclick=()=>{{record().points=[];persist();draw()}};
document.getElementById('export').onclick=()=>{{if(!reviewer.value.trim())return alert('导出前请填写复核者 ID');persist();const head=['reviewer_id','reviewer_role','annotated_at','sample_id','visibility','quality','note','center_x','center_y','major_x','major_y','minor_x','minor_y'];const rows=[head,...samples.map(s=>{{const r=saved[recordKey(s)]||{{}},p=r.points||[];return [r.reviewer_id||reviewer.value.trim(),r.reviewer_role||role.value,r.annotated_at||'',s.sample_id,r.visibility||'',(r.quality||[]).join('|'),r.note||'',...([0,1,2].flatMap(i=>p[i]?[p[i].x,p[i].y]:['','']))]}})];const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv],{{type:'text/csv'}}));a.download='annotations_'+reviewer.value.trim()+'.csv';a.click()}};document.addEventListener('keydown',e=>{{const t=(e.target&&e.target.tagName)||'';if(t==='INPUT'||t==='TEXTAREA')return;const k=e.key;if(k==='1'){{visibility.value='可见';persist();draw();}}else if(k==='2'){{visibility.value='不确定';record().points=[];persist();draw();}}else if(k==='3'){{visibility.value='不可见';record().points=[];persist();draw();}}else if(k==='b'||k==='B'){{toggleQuality('blur');}}else if(k==='o'||k==='O'){{toggleQuality('overexposure');}}else if(k==='r'||k==='R'){{toggleQuality('reflection');}}else if(k==='g'||k==='G'){{toggleQuality('glasses_frame');}}else if(k==='l'||k==='L'){{toggleQuality('eyelid_occlusion');}}else if(k==='ArrowLeft'||k==='a'||k==='A'){{persist();index=Math.max(0,index-1);load();}}else if(k==='ArrowRight'||k==='d'||k==='D'){{persist();index=Math.min(samples.length-1,index+1);load();}}else if(k==='x'||k==='X'||k==='Backspace'){{record().points=[];persist();draw();}}else{{return;}}e.preventDefault();}});load();
</script></body></html>'''
