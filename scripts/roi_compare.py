"""ROI 方案测速对比（主环境，一次性调查工具）。

> 08-16（Asia/Shanghai）｜E 盘拔出前写好；静态验证通过，盘插回后实测三方案。

回答"除了 mediapipe，还有谁可以选眼 ROI、谁更快"：同一批正式实验帧上对比
  - mediapipe FaceLandmarker（复用detector；速度与正确定位分开报告）
  - YuNet FaceDetectorYN（OpenCV 内置，230KB onnx）
  - YOLO-face（onnxruntime 零装包；需提供 onnx 模型）

每帧记录各后端：耗时 / 是否检出人脸 / 人脸框 / 眼关键点（左右眼中心或眼角），
并输出一张对比图（同一帧三后端标注叠加），供人眼核对框与眼点是否落在正确位置。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/roi_compare.py --avi E:/正式实验/sub-013_/nir/sub-013_nir.avi
    & 'D:/Code/python/python.exe' scripts/roi_compare.py --avi ... --yolo-model yolov8n-face.onnx

输出：--out 目录下 timeline.csv（frame,backend,ms,faces,box,eye_points）+ compare_<frame>.jpg
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from roi_common import ascii_model_path
from attention_pipeline.config import load_config

DEFAULT_FACE_MODEL = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/face_landmarker.task"
DEFAULT_YUNET = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/yunet_2023mar.onnx"
DEFAULT_YOLO = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/yolov8n-face.onnx"

COLORS = {"mediapipe": (0, 255, 0), "yunet": (0, 128, 255), "yolo": (255, 0, 255)}


# ------------------------------------------------------------- 各后端检测
def make_mediapipe(model_path: str):
    from attention_pipeline.nir.sequence import FaceLandmarkerSession
    session = FaceLandmarkerSession(Path(model_path))
    corner = {"eye_right": (33, 133), "eye_left": (362, 263)}  # 与 roi.EYE_CORNERS 一致

    def detect(frame):
        pts = session.detect(frame)
        if pts is None:
            return {"faces": 0, "box": None, "eye_points": None}
        box = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        eyes = {}
        for name, (i0, i1) in corner.items():  # 眼角两个 landmark 的中点作为该眼代表点
            p0, p1 = pts[i0], pts[i1]
            eyes[name] = (float((p0[0] + p1[0]) / 2), float((p0[1] + p1[1]) / 2))
        return {"faces": 1, "box": box, "eye_points": eyes}

    return detect, session.close


def make_yunet(model_path: str):
    det = cv2.FaceDetectorYN.create(ascii_model_path(model_path), "", (0, 0), 0.5, 0.3, 500)

    def detect(frame):
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        ok, faces = det.detect(frame)
        if not ok or faces is None or len(faces) == 0:
            return {"faces": 0, "box": None, "eye_points": None}
        x, y, bw, bh = [float(v) for v in faces[0, :4]]
        kp = faces[0, 4:14].reshape(5, 2).astype(float)  # 已是输入图像像素坐标
        return {"faces": 1, "box": [x, y, x + bw, y + bh],
                "eye_points": {"eye_right": tuple(kp[1]), "eye_left": tuple(kp[0])}}

    return detect, (lambda: None)


def make_yolo(model_path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_h, in_w = int(sess.get_inputs()[0].shape[2]), int(sess.get_inputs()[0].shape[3])

    def detect(frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (in_w, in_h)).astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        arr = np.asarray(sess.run(None, {in_name: blob})[0])
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T  # -> [N, C]
        c = arr.shape[1] if arr.ndim == 2 else 0
        s = np.array([w / in_w, h / in_h])
        if c >= 15:  # 带 5 关键点的头：bbox4 + kp10 + cls
            idx = int(np.argmax(arr[:, -1]))
            cx, cy, bw, bh = arr[idx, :4]
            box = [(cx - bw / 2) * s[0], (cy - bh / 2) * s[1], (cx + bw / 2) * s[0], (cy + bh / 2) * s[1]]
            kp = arr[idx, 4:14].reshape(5, 2) * s
            ep = {"eye_right": tuple(kp[1]), "eye_left": tuple(kp[0])}
        elif c == 5:  # 纯检测头：bbox4 + conf（本仓库 lindevs 版）；眼点用框内比例估计
            idx = int(np.argmax(arr[:, 4]))
            cx, cy, bw, bh = arr[idx, :4]
            x0, y0 = (cx - bw / 2) * s[0], (cy - bh / 2) * s[1]
            x1, y1 = (cx + bw / 2) * s[0], (cy + bh / 2) * s[1]
            box = [x0, y0, x1, y1]
            ep = {"eye_right": (x0 + 0.38 * (x1 - x0), y0 + 0.38 * (y1 - y0)),
                  "eye_left": (x0 + 0.62 * (x1 - x0), y0 + 0.38 * (y1 - y0))}
        else:
            print(f"[yolo] 未识别输出 shape {arr.shape}，跳过（需按实际模型改解析）")
            return {"faces": 0, "box": None, "eye_points": None}
        return {"faces": 1, "box": box, "eye_points": ep}

    return detect, (lambda: None)


def draw_annotations(vis, res: dict, label: str, color):
    if res["box"] is None:
        cv2.putText(vis, f"{label}: no face", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return
    x0, y0, x1, y1 = [int(v) for v in res["box"]]
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    cv2.putText(vis, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for name, (px, py) in (res["eye_points"] or {}).items():
        cv2.circle(vis, (int(px), int(py)), 6, color, -1)
        cv2.putText(vis, {"eye_right": "RE", "eye_left": "LE"}[name], (int(px) + 8, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/formal.yaml")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--avi", default=None)
    parser.add_argument("--frames", default="", help="逗号分隔帧号；缺省按 n-frames 均匀采样")
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--face-model", default=None)
    parser.add_argument("--yunet-model", default=None)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    selection = config.section("nir")["roi_selection"]
    subject = args.subject or str(selection["subject"])
    root = config.path_value("formal_data_root")
    args.avi = args.avi or str(root / subject / "nir" / f"{subject.rstrip('_')}_nir.avi")
    args.out = args.out or str(config.path_value("roi_selection_artifact_root") / "speed-diagnostic")
    args.face_model = args.face_model or str(config.path_value("face_landmarker_model"))
    args.yunet_model = args.yunet_model or str(config.path_value("yunet_model"))
    args.yolo_model = args.yolo_model or str(config.path_value("yolo_model"))

    cap = cv2.VideoCapture(args.avi)
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {args.avi}（硬盘在吗？）")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frames:
        frames = [int(v) for v in args.frames.split(",") if v.strip()]
    else:
        span = max(1, total - 1)
        frames = [round(i * span / max(1, args.n_frames - 1)) for i in range(args.n_frames)]

    backends = {"mediapipe": make_mediapipe(args.face_model)}
    if Path(args.yunet_model).exists():
        backends["yunet"] = make_yunet(args.yunet_model)
    else:
        print(f"[skip] YuNet 模型缺失：{args.yunet_model}")
    if args.yolo_model and Path(args.yolo_model).exists():
        backends["yolo"] = make_yolo(args.yolo_model)
    elif args.yolo_model:
        print(f"[skip] YOLO 模型缺失：{args.yolo_model}")
    if "yunet" not in backends and "yolo" not in backends:
        print("[提示] 只测到 mediapipe；下载 YuNet onnx 或给 YOLO 模型后可全测。")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    # 每后端 warmup 一帧（mediapipe 冷启动不计时）
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    _, warm = cap.read()
    for name, (detect, _) in backends.items():
        detect(warm)

    for fi, fnum in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        vis = frame.copy()
        cv2.putText(vis, f"frame {fnum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for name, (detect, _) in backends.items():
            t = time.perf_counter()
            res = detect(frame)
            ms = (time.perf_counter() - t) * 1000
            draw_annotations(vis, res, name, COLORS[name])
            rows.append({"frame": fnum, "backend": name, "ms": round(ms, 1),
                         "faces": res["faces"], "box": res["box"],
                         "eye_points": res["eye_points"]})
            summary.setdefault(name, []).append(ms)
        cv2.imencode(".jpg", vis)[1].tofile(str(outdir / f"compare_{fnum}.jpg"))

    with (outdir / "timeline.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "backend", "ms", "faces", "box", "eye_points"])
        for r in rows:
            writer.writerow([r["frame"], r["backend"], r["ms"], r["faces"], r["box"], r["eye_points"]])

    print(f"\n=== ROI 测速（warmup 后，{len(frames)} 帧平均）===")
    for name, ms_list in summary.items():
        print(f"  {name:10s} {np.mean(ms_list):6.1f} ms/帧   (n={len(ms_list)})")
    print(f"\n对比图 + timeline.csv → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



