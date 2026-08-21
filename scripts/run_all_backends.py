"""全算法封装统一入口：ROI 后端 × 瞳孔算法，产出同构对比供新设备选型。

> 08-16（Asia/Shanghai）｜不在本机选最优：把所有 ROI 后端（mediapipe/yunet/yolo/faceparts）
> 与瞳孔算法（PuReST/RITnet/DeepVOG/Iris）全部跑通，各自产出 manifest + detections，
> 汇总成一张对比表。新设备上跑完再选型（production 不冻结）。

ROI 后端（主环境，统一 .eyes(frame) 接口，走 roi_common.build_roi）：
  mediapipe / yunet / yolo / faceparts
瞳孔算法：
  PuReST   venv-pupil（nir_sequence_detect.py，读 manifest）
  RITnet   主环境 torch（models/RITnet-master/infer_ritnet.py，吃平铺 ROI 目录）
  DeepVOG  主环境 keras/tf（scripts/deepvog_pupil.py，吃平铺 ROI 目录；部署时装 keras）
  Iris     主环境 mediapipe（scripts/iris_landmark.py，吃整帧；仅全脸适用，特写失败）

用法（数据接入后）：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/run_all_backends.py --subject sub-011 --roi-backends faceparts,mediapipe --pupil-algos PuReST,RITnet
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import roi_common
from roi_mediapipe import MediaPipeRoi
from roi_yolo import YoloRoi
from roi_yunet import YuNetRoi
from roi_faceparts import FacePartsRoi
from attention_pipeline.config import load_config

ROI_BACKENDS = ("mediapipe", "yunet", "yolo", "faceparts")
PUPIL_ALGOS = ("PuReST", "RITnet", "DeepVOG", "Iris")
MODEL_KEYS = {
    "mediapipe": "face_landmarker_model",
    "yunet": "yunet_model",
    "yolo": "yolo_model",
    "faceparts": "faceparts_model",
}


def _provider(kind: str, config, args):
    if kind == "mediapipe":
        return MediaPipeRoi(config.path_value("face_landmarker_model"),
                            args.roi_w, args.roi_h, args.corner_span)
    if kind == "yunet":
        return YuNetRoi(config.path_value("yunet_model"), args.roi_w, args.roi_h, args.corner_span)
    if kind == "yolo":
        return YoloRoi(config.path_value("yolo_model"), args.roi_w, args.roi_h, args.corner_span)
    if kind == "faceparts":
        return FacePartsRoi(config.path_value("faceparts_model"), args.roi_w, args.roi_h,
                            args.corner_span, conf=getattr(args, "faceparts_conf", 0.25),
                            imgsz=getattr(args, "faceparts_imgsz", 640))
    raise ValueError(kind)


def _flatten_rois(manifest: Path, roi_root: Path, out_dir: Path) -> Path:
    """把 build_roi 的嵌套 ROI PNG（segments/<seq>/<eye>/fXXXX.png）平铺到 out_dir，
    文件名 {seq}__{eye}__f{off:04d}.png，匹配瞳孔脚本的 *_eye_*.png glob。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(manifest)
    for _, r in rows.iterrows():
        rel = str(r.get("roi_path", "")).strip()
        if not rel:
            continue
        src = roi_root / rel
        if not src.exists():
            continue
        seq = str(r.get("sequence_id", "")).replace("/", "_")
        eye = str(r.get("eye", "eye"))
        off = int(r.get("frame_offset", 0))
        dst = out_dir / f"{seq}__{eye}__f{off:04d}.png"
        if not dst.exists():
            shutil.copy2(src, dst)
    return out_dir


def _run_pupil(algo: str, config, args, outdir: Path, manifest: Path) -> Path:
    """对给定 ROI 产出运行一个瞳孔算法，返回其 detections CSV。"""
    script_dir = Path(__file__).resolve().parent
    if algo == "PuReST":
        return roi_common.run_detect(
            args.venv_python, manifest, outdir, args.px_min, args.px_max, args.pupil_min_mm,
            args.openness_visible, args.max_session_gap_ms, args.reset_after_quality_rejects,
        )
    if algo in ("RITnet", "DeepVOG"):
        roi_flat = _flatten_rois(manifest, outdir, outdir / "roi_flat")
        if algo == "RITnet":
            cmd = [
                sys.executable,
                str(script_dir.parent / "models" / "RITnet-master" / "infer_ritnet.py"),
                "--roi_dir", str(roi_flat), "--out", str(outdir / "ritnet"),
            ]
        else:
            cmd = [
                sys.executable, str(script_dir / "deepvog_pupil.py"),
                "--roi_dir", str(roi_flat), "--out", str(outdir / "deepvog_results.csv"),
            ]
        subprocess.run(cmd, check=True)
        return outdir / ("ritnet" / "ritnet_results.csv" if algo == "RITnet"
                         else "deepvog_results.csv")
    if algo == "Iris":
        raise NotImplementedError("Iris 吃整帧（非 ROI），需在 run_all_backends 之外单独跑 iris_landmark.py")
    raise ValueError(algo)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="全算法封装统一入口")
    parser.add_argument("--config", default="configs/formal.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--roi-backends", default=",".join(ROI_BACKENDS))
    parser.add_argument("--pupil-algos", default="PuReST")
    parser.add_argument("--seg-starts", default="")
    parser.add_argument("--frames-per-seg", type=int, default=None)
    parser.add_argument("--corner-span", type=float, default=None)
    parser.add_argument("--faceparts-conf", type=float, default=0.25)
    parser.add_argument("--faceparts-imgsz", type=int, default=640)
    parser.add_argument("--skip-detect", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    backends = [b for b in args.roi_backends.split(",") if b]
    algos = [a for a in args.pupil_algos.split(",") if a]
    unsupported = set(backends) - set(ROI_BACKENDS)
    if unsupported:
        parser.error(f"unsupported ROI backends: {sorted(unsupported)}")
    unsupported = set(algos) - set(PUPIL_ALGOS)
    if unsupported:
        parser.error(f"unsupported pupil algos: {sorted(unsupported)}")

    for backend in backends:
        # 复用 roi_common 解析参数（填充 roi_w/h、corner_span、px 门等）
        roi_common.resolve_common_args(args, MODEL_KEYS[backend])
        outdir = Path(args.out) / args.subject / backend
        outdir.mkdir(parents=True, exist_ok=True)
        provider = _provider(backend, config, args)
        try:
            manifest, n_eye_frames = roi_common.build_roi(provider, args, outdir)
        finally:
            provider.close()
        if args.skip_detect:
            continue
        for algo in algos:
            if algo == "Iris":
                continue  # Iris 吃整帧，独立跑
            det = _run_pupil(algo, config, args, outdir, manifest)
            print(f"[{backend} × {algo}] {det}", flush=True)
    print(f"\n[done] 产出见 {args.out}/{args.subject}/ 下各 backend 目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
