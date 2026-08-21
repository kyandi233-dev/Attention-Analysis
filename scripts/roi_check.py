"""正式实验 mediapipe ROI 完整检验（主环境）。

> 2026-08-16（Asia/Shanghai）｜回答"mediapipe 在正式实验上到底行不行、能否靠写法优化提升"。
> Codex 在 sub-011 判三个全脸 ROI 候选全淘汰（画面为双眼特写）；但不同被试画面可能不同，
> 本脚本系统检验 7 个正式被试的画面类型 + 检出率 + 质量，为 ROI 方案定稿提供证据。

4 步：
  Step1 画面类型诊断：每被试全片均匀抽 12 帧 → 4×3 全帧联系表 → 人工填 frame_type（完整脸/双眼特写/无脸/混合）
  Step2 检出率矩阵：每被试 stride 500 抽样 × 4 配置（raw/CLAHE × 阈值 0.5/0.3）→ summary.csv + 图
  Step3 ROI 质量抽检：frame_type=完整脸 且检出≥60% 的被试×配置，抽 ROI 双眼并排图
  Step4 结论报告：分档判定 → 00-结论.md

用法（主环境，PYTHONPATH=src）：
    & 'D:/Code/python/python.exe' scripts/roi_check.py --step 1
    # 人工填 artifacts/roi-checks/01-frame-type/frame_type_labels.csv 的 frame_type 后：
    & 'D:/Code/python/python.exe' scripts/roi_check.py --step 2
    & 'D:/Code/python/python.exe' scripts/roi_check.py --step 3
    & 'D:/Code/python/python.exe' scripts/roi_check.py --step 4

输出：artifacts/roi-checks/（不触碰 E:\\正式实验 原件，不改 finish/）
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import roi_common
from attention_pipeline.nir.review import _imwrite, _read_frame
from attention_pipeline.nir.sequence import FaceLandmarkerSession

OUT = Path("artifacts/roi-checks")
ROOT = Path("E:/正式实验")
MODEL = Path(roi_common.DEFAULT_FACE_MODEL)
# 帧级配置：(name, clahe_on_frame, mediapipe_confidence)
CONFIGS = [("raw", False, 0.5), ("raw030", False, 0.3), ("clahe", True, 0.5), ("clahe030", True, 0.3)]
GOOD_TYPES = {"完整脸", "混合"}  # 视作"有完整脸可用"的画面类型
# 从 E:\\正式实验 探测被试（正式被试目录形如 sub-XXX_）
SUBJECTS = sorted(
    p.name.rstrip("_") for p in ROOT.iterdir()
    if p.is_dir() and p.name.startswith("sub-") and (ROOT / p.name / "nir").exists()
)


def subject_avi(subject: str) -> Path:
    return ROOT / f"{subject}_" / "nir" / f"{subject}_nir.avi"


def _clahe_tool():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _detect_sample(subject: str, stride: int, clahe: bool, confidence: float):
    """对某被试全片按 stride 抽样，返回 (hits, eye_spans, ms_per_frame)。"""
    avi = subject_avi(subject)
    cap = cv2.VideoCapture(str(avi))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    session = FaceLandmarkerSession(MODEL, confidence=confidence)
    cobj = _clahe_tool() if clahe else None
    hits, spans, t0 = [], [], time.perf_counter()
    for i in range(0, total, stride):
        try:
            f = _read_frame(avi, i)
        except RuntimeError:
            continue
        if cobj is not None:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            f = cv2.cvtColor(cobj.apply(g), cv2.COLOR_GRAY2BGR)
        pts = session.detect(f)
        if pts is not None:
            hits.append(i)
            spans.append(max(
                float(np.linalg.norm(pts[33] - pts[133])),
                float(np.linalg.norm(pts[362] - pts[263])),
            ))
    session.close()
    n = len(list(range(0, total, stride)))
    ms = (time.perf_counter() - t0) * 1000 / max(1, n)
    return hits, spans, ms


def step1(args) -> int:
    labels_dir = OUT / "01-frame-type"
    sheet_dir = labels_dir / "contact-sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sub in SUBJECTS:
        avi = subject_avi(sub)
        cap = cv2.VideoCapture(str(avi))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stride = max(1, total // args.contact_frames)
        thumbs = []
        for k in range(args.contact_frames):
            try:
                f = _read_frame(avi, k * stride)
            except RuntimeError:
                continue
            h, w = f.shape[:2]
            thumbs.append(cv2.resize(f, (max(1, int(w * 240 / h)), 240)))
        if not thumbs:
            print(f"  {sub}: 无法读帧", file=sys.stderr)
            rows.append({"subject": sub, "total_frames": total, "frame_type": "", "note": "读取失败"})
            continue
        blank = np.full_like(thumbs[0], 255)
        grid = []
        for r in range(4):
            seg = thumbs[r * 3:(r + 1) * 3]
            seg += [blank] * (3 - len(seg))
            grid.append(np.hstack(seg))
        _imwrite(sheet_dir / f"{sub}.jpg", np.vstack(grid))
        rows.append({"subject": sub, "total_frames": total, "frame_type": "", "note": ""})
    with (labels_dir / "frame_type_labels.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "total_frames", "frame_type", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"[step1] 联系表 → {sheet_dir}/ ; 请人工填 {labels_dir}/frame_type_labels.csv 的 frame_type "
          f"(完整脸/双眼特写/无脸/混合)")
    return 0


def step2(args) -> int:
    mat_dir = OUT / "02-detection-matrix"
    mat_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sub in SUBJECTS:
        for name, clahe, conf in CONFIGS:
            hits, spans, ms = _detect_sample(sub, args.detection_stride, clahe, conf)
            n = len(hits)  # hits 列表本身是样本；需要 n 样本数，用 _detect_sample 返回？下面重算
            # _detect_sample 未返回 n；用 summary 时由调用处补：这里直接从 avi 推算
            avi = subject_avi(sub)
            cap = cv2.VideoCapture(str(avi))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            n = len(list(range(0, total, args.detection_stride)))
            rate = len(hits) / n if n else 0.0
            s = np.array(spans)
            row = {
                "subject": sub, "config": name, "clahe": int(clahe), "confidence": conf,
                "n_samples": n, "hits": len(hits), "rate": round(rate, 4),
                "span_p10": round(float(np.percentile(s, 10)), 1) if len(s) else "",
                "span_p50": round(float(np.median(s)), 1) if len(s) else "",
                "span_p90": round(float(np.percentile(s, 90)), 1) if len(s) else "",
                "ms_per_frame": round(ms, 2),
            }
            summary.append(row)
            print(f"  {sub:9s} {name:9s} 检出 {rate:.3f} ({len(hits)}/{n})  span中位{row['span_p50']}px  {ms:.1f}ms/帧")
    with (mat_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[step2] → {mat_dir}/summary.csv")
    return 0


def step3(args) -> int:
    import pandas as pd
    from roi_mediapipe import MediaPipeRoi

    labels = pd.read_csv(OUT / "01-frame-type" / "frame_type_labels.csv")
    good = labels[labels["frame_type"].isin(GOOD_TYPES)]["subject"].tolist()
    det = pd.read_csv(OUT / "02-detection-matrix" / "summary.csv")
    q_dir = OUT / "03-roi-quality"
    q_dir.mkdir(parents=True, exist_ok=True)
    for sub in good:
        for name, clahe, conf in CONFIGS:
            row = det[(det["subject"] == sub) & (det["config"] == name)]
            if row.empty or float(row.iloc[0]["rate"]) < args.min_rate:
                continue
            roi = MediaPipeRoi(str(MODEL), 320, 160, corner_span=args.corner_span,
                               confidence=conf, clahe=clahe)
            avi = subject_avi(sub)
            cap = cv2.VideoCapture(str(avi))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            saved = 0
            for i in range(0, total, args.detection_stride):
                try:
                    f = _read_frame(avi, i)
                except RuntimeError:
                    continue
                eyes = roi.eyes(f)
                if not eyes:
                    continue
                if "image" in eyes["eye_right"]:
                    pair = np.hstack([eyes["eye_right"]["image"], np.full((160, 4), 255, np.uint8),
                                      eyes["eye_left"]["image"]])
                else:
                    pair = np.hstack([eyes["eye_right"], np.full((160, 4), 255, np.uint8),
                                      eyes["eye_left"]])
                _imwrite(q_dir / f"{sub}_{name}_q{saved:02d}_f{i}.png", pair)
                saved += 1
                if saved >= args.roi_quality_frames:
                    break
            roi.close()
            print(f"  {sub} {name}: {saved} 张 ROI → {q_dir}/")
    return 0


def step4(args) -> int:
    import pandas as pd

    labels = pd.read_csv(OUT / "01-frame-type" / "frame_type_labels.csv")
    det = pd.read_csv(OUT / "02-detection-matrix" / "summary.csv")
    lines = ["# mediapipe ROI 完整检验结论", "",
             f"> 08-16（Asia/Shanghai）｜画面类型由人工在 01-frame-type 联系表标注；判定线=检出率≥{int(args.min_rate * 100)}% + ROI 人眼合格。",
             "", "| 被试 | 画面类型 | raw | raw030 | clahe | clahe030 | 判定 | 推荐 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in labels.iterrows():
        sub = r["subject"]
        sub_det = det[det["subject"] == sub]
        rates = {cfg: "" for cfg, *_ in CONFIGS}
        for _, d in sub_det.iterrows():
            rates[d["config"]] = f"{d['rate']:.0%}"
        ft = r["frame_type"]
        is_good = ft in GOOD_TYPES
        max_rate = max((v for v in rates.values() if v), default="")
        usable = is_good and any(d["rate"] >= args.min_rate for _, d in sub_det.iterrows())
        verdict = "✅ mediapipe 可用" if usable else ("⚠️ 特写，需眼部专用" if ft == "双眼特写" else "❌ 不适用")
        rec = "进入 corner_span 选型" if usable else ("参考 Causa：眼部直接检测/固定 ROI" if ft == "双眼特写" else "记录时段分布")
        lines.append(f"| {sub} | {ft or '待标'} | {rates['raw']} | {rates['raw030']} | {rates['clahe']} | {rates['clahe030']} | {verdict} | {rec} |")
    report = OUT / "00-结论.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[step4] → {report}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="正式实验 mediapipe ROI 完整检验")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--contact-frames", type=int, default=12)
    parser.add_argument("--detection-stride", type=int, default=500)
    parser.add_argument("--corner-span", type=float, default=0.5)
    parser.add_argument("--roi-quality-frames", type=int, default=6)
    parser.add_argument("--min-rate", type=float, default=0.60)
    args = parser.parse_args(argv)
    return {1: step1, 2: step2, 3: step3, 4: step4}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
