from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PYFEAT_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
PYFEAT_EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
LIBREFACE_AU_INT = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
LIBREFACE_AU_DET = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
LIBREFACE_EXPRESSIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
    "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _session(model: Path):
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"DmlExecutionProvider unavailable: {ort.get_available_providers()}")
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model), sess_options=so,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        enable_fallback=0,
    )
    sess.disable_fallback()
    return sess


def _imagenet_nchw(rgb_float01: np.ndarray) -> np.ndarray:
    x = (rgb_float01 - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)


def _pil_resize_shorter_center_crop(path: str, shorter: int = 256, crop: int = 224) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = float(shorter) / float(min(w, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left = max(0, (nw - crop) // 2)
        top = max(0, (nh - crop) // 2)
        im = im.crop((left, top, left + crop, top + crop))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return _imagenet_nchw(arr)


def _pil_resize_224(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return _imagenet_nchw(arr)


def _run_batched(session: Any, input_name: str, values: np.ndarray, batch_size: int) -> list[np.ndarray]:
    output_chunks: list[list[np.ndarray]] = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        out = session.run(None, {input_name: batch})
        output_chunks.append([np.asarray(v) for v in out])
    if not output_chunks:
        return []
    n_outputs = len(output_chunks[0])
    return [np.concatenate([chunk[i] for chunk in output_chunks], axis=0) for i in range(n_outputs)]


def run_libreface(args: argparse.Namespace) -> None:
    prep_dir = Path(args.prep_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_manifest = json.loads((prep_dir / "libreface_dml_prep_manifest.json").read_text(encoding="utf-8"))
    alignment = pd.read_parquet(prep_dir / "libreface_dml_alignment.parquet")
    gaze_index = pd.read_parquet(prep_dir / "libreface_dml_gaze_feature_index.parquet")
    gaze_features = np.load(prep_dir / "libreface_dml_gaze_features.npy")
    ok = alignment[alignment["alignment_success"].fillna(False).astype(bool)].copy()
    aligned_paths = ok["aligned_image_path"].astype(str).tolist()
    indices = ok["benchmark_index"].astype(int).tolist()
    if len(gaze_features) != len(gaze_index) or gaze_index["benchmark_index"].astype(int).tolist() != indices:
        raise RuntimeError("LibreFace prep alignment/gaze index mismatch")

    au_model = model_dir / "libreface2_au_joint.onnx"
    expr_model = model_dir / "libreface2_expression.onnx"
    gaze_model = model_dir / "libreface2_gaze_mlp.onnx"
    au_sess, expr_sess, gaze_sess = _session(au_model), _session(expr_model), _session(gaze_model)

    t0 = time.perf_counter()
    au_inputs = np.stack([_pil_resize_shorter_center_crop(p) for p in aligned_paths], axis=0)
    au_pre_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    au_int_prob, au_det_prob = _run_batched(au_sess, au_sess.get_inputs()[0].name, au_inputs, args.batch_size)
    au_infer_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    expr_inputs = np.stack([_pil_resize_224(p) for p in aligned_paths], axis=0)
    expr_pre_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    expr_scores = _run_batched(expr_sess, expr_sess.get_inputs()[0].name, expr_inputs, args.batch_size)[0]
    expr_infer_sec = time.perf_counter() - t0

    valid_gaze = np.isfinite(gaze_features).all(axis=1)
    gaze_out = np.full((len(indices), 2), np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    if valid_gaze.any():
        gaze_out[valid_gaze] = _run_batched(
            gaze_sess, gaze_sess.get_inputs()[0].name, gaze_features[valid_gaze].astype(np.float32), args.batch_size
        )[0]
    gaze_infer_sec = time.perf_counter() - t0

    au_int = pd.DataFrame(au_int_prob * 5.0, columns=[f"AU{x:02d}" for x in LIBREFACE_AU_INT])
    au_int.insert(0, "benchmark_index", indices)
    au_det = pd.DataFrame((au_det_prob >= 0.5).astype(np.int8), columns=[f"AU{x:02d}" for x in LIBREFACE_AU_DET])
    au_det.insert(0, "benchmark_index", indices)
    expr = pd.DataFrame(expr_scores, columns=[f"score__{x}" for x in LIBREFACE_EXPRESSIONS])
    expr.insert(0, "expression_label", [LIBREFACE_EXPRESSIONS[i] for i in np.argmax(expr_scores, axis=1)])
    expr.insert(0, "benchmark_index", indices)
    gaze = pd.DataFrame(gaze_out, columns=["yaw", "pitch"])
    gaze.insert(0, "benchmark_index", indices)

    au_int_path = out_dir / "libreface_dml_au_intensity.parquet"
    au_det_path = out_dir / "libreface_dml_au_detection.parquet"
    expr_path = out_dir / "libreface_dml_expression.parquet"
    gaze_path = out_dir / "libreface_dml_gaze.parquet"
    au_int.to_parquet(au_int_path, index=False, engine="pyarrow", compression="zstd")
    au_det.to_parquet(au_det_path, index=False, engine="pyarrow", compression="zstd")
    expr.to_parquet(expr_path, index=False, engine="pyarrow", compression="zstd")
    gaze.to_parquet(gaze_path, index=False, engine="pyarrow", compression="zstd")

    cpu_pre = float(prep_manifest["timing_sec"]["cpu_preprocess_total"])
    dml_stage = au_pre_sec + au_infer_sec + expr_pre_sec + expr_infer_sec + gaze_infer_sec
    total = cpu_pre + dml_stage
    manifest = {
        "schema_version": "rgb-face-real300-libreface-dml-v0.1",
        "candidate": "libreface2",
        "expected_frames": int(prep_manifest["expected_frames"]),
        "aligned_frames": len(indices),
        "batch_size": int(args.batch_size),
        "timing_sec": {
            "cpu_preprocess_from_fresh_prep": cpu_pre,
            "au_input_preprocess": au_pre_sec,
            "au_dml_inference": au_infer_sec,
            "expression_input_preprocess": expr_pre_sec,
            "expression_dml_inference": expr_infer_sec,
            "gaze_dml_inference": gaze_infer_sec,
            "dml_stage_total": dml_stage,
            "component_summed_end_to_end": total,
        },
        "input_frames_per_sec_component_summed": prep_manifest["expected_frames"] / total if total > 0 else None,
        "models": {p.stem: {"path": str(p), "sha256": _sha256(p)} for p in (au_model, expr_model, gaze_model)},
        "outputs": {"au_intensity": str(au_int_path), "au_detection": str(au_det_path), "expression": str(expr_path), "gaze": str(gaze_path)},
        "notes": [
            "End-to-end is component-summed across the fresh CPU prep process and the DirectML process; no saved CPU-model inference time is reused.",
            "Head pose and alignment landmarks remain in the fresh prep alignment parquet and are not recomputed by the learned ONNX heads.",
        ],
    }
    (out_dir / "libreface_dml_real300_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _generate_priors(h: int, w: int) -> np.ndarray:
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    anchors: list[list[float]] = []
    for sizes, step in zip(min_sizes, steps):
        fh, fw = math.ceil(h / step), math.ceil(w / step)
        for i in range(fh):
            cy = (i + 0.5) * step / h
            for j in range(fw):
                cx = (j + 0.5) * step / w
                for size in sizes:
                    anchors.append([cx, cy, size / w, size / h])
    return np.asarray(anchors, dtype=np.float32)


def _decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    boxes = np.empty_like(loc, dtype=np.float32)
    boxes[..., :2] = priors[None, :, :2] + loc[..., :2] * 0.1 * priors[None, :, 2:]
    boxes[..., 2:] = priors[None, :, 2:] * np.exp(loc[..., 2:] * 0.2)
    boxes[..., :2] -= boxes[..., 2:] / 2.0
    boxes[..., 2:] += boxes[..., :2]
    return boxes


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        area_i = np.maximum(0.0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0.0, boxes[i, 3] - boxes[i, 1])
        area_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-12)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _square_reflect_crop(rgb: np.ndarray, box: np.ndarray, size: int = 256) -> tuple[np.ndarray, tuple[float, float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * 1.2
    side = max(side, 1.0)
    ox, oy = cx - side / 2.0, cy - side / 2.0
    coords = (np.arange(size, dtype=np.float32) + 0.5) / float(size) * side
    map_x = np.broadcast_to((ox + coords)[None, :], (size, size)).astype(np.float32)
    map_y = np.broadcast_to((oy + coords)[:, None], (size, size)).astype(np.float32)
    crop = cv2.remap(
        rgb.astype(np.float32) / 255.0, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    return crop, (ox, oy, side)


def _preprocess_pyfeat_chip(chip01: np.ndarray) -> np.ndarray:
    resized = cv2.resize(chip01, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _imagenet_nchw(resized.astype(np.float32, copy=False))


def run_pyfeat(args: argparse.Namespace) -> None:
    root = Path(args.benchmark_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir).resolve()
    manifests = sorted(root.glob("*_face-continuous_frames.csv")) + sorted(root.glob("*_face-benchmark_frames.csv"))
    manifests = list(dict.fromkeys(manifests))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one frame manifest, found {len(manifests)}")
    frame_manifest = manifests[0]
    frames = pd.read_csv(frame_manifest)

    rf_model = model_dir / "pyfeat211_retinaface_r34.onnx"
    mt_model = model_dir / "pyfeat211_multitask_scientific_core.onnx"
    rf_sess, mt_sess = _session(rf_model), _session(mt_model)
    priors: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    stage = {
        "image_read_preprocess": 0.0,
        "retinaface_dml": 0.0,
        "decode_nms_crop": 0.0,
        "multitask_preprocess": 0.0,
        "multitask_dml": 0.0,
        "postprocess": 0.0,
    }
    total_start = time.perf_counter()

    for start in range(0, len(frames), args.retinaface_batch):
        batch_df = frames.iloc[start:start + args.retinaface_batch]
        t0 = time.perf_counter()
        rgb_list: list[np.ndarray] = []
        for p in batch_df["image_path"].astype(str):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(p)
            rgb_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb_list[0].shape[:2]
        if any(im.shape[:2] != (h, w) for im in rgb_list):
            raise RuntimeError("Mixed frame shapes are not supported by fixed-shape RetinaFace ONNX")
        rf_in = np.stack([
            np.transpose(im.astype(np.float32) - np.array([123.0, 117.0, 104.0], np.float32), (2, 0, 1))
            for im in rgb_list
        ])
        stage["image_read_preprocess"] += time.perf_counter() - t0
        if priors is None:
            priors = _generate_priors(h, w)

        t0 = time.perf_counter()
        loc, conf, _landm = rf_sess.run(None, {rf_sess.get_inputs()[0].name: rf_in})
        stage["retinaface_dml"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        boxes_all = _decode_boxes(np.asarray(loc), priors)
        boxes_all *= np.array([w, h, w, h], dtype=np.float32)[None, None, :]
        face_meta: list[dict[str, Any]] = []
        chips: list[np.ndarray] = []
        for bi, row in enumerate(batch_df.itertuples(index=False)):
            scores = np.asarray(conf)[bi, :, 1]
            mask = scores > 0.02
            boxes, sc = boxes_all[bi, mask], scores[mask]
            if len(sc) > 5000:
                order = np.argsort(sc)[::-1][:5000]
                boxes, sc = boxes[order], sc[order]
            keep = _nms(boxes, sc, 0.4)
            boxes, sc = boxes[keep], sc[keep]
            keep2 = sc >= 0.5
            boxes, sc = boxes[keep2][:750], sc[keep2][:750]
            if len(sc) == 0:
                face_meta.append({
                    "benchmark_index": int(row.benchmark_index),
                    "face_rank": 0,
                    "score": np.nan,
                    "crop": None,
                    "detected": False,
                })
                continue
            for rank, (box, score) in enumerate(zip(boxes, sc)):
                chip, crop = _square_reflect_crop(rgb_list[bi], box)
                chips.append(chip)
                face_meta.append({
                    "benchmark_index": int(row.benchmark_index),
                    "face_rank": rank,
                    "score": float(score),
                    "crop": crop,
                    "detected": True,
                })
        stage["decode_nms_crop"] += time.perf_counter() - t0

        outputs = None
        if chips:
            t0 = time.perf_counter()
            mt_inputs = np.stack([_preprocess_pyfeat_chip(c) for c in chips], axis=0)
            stage["multitask_preprocess"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            outputs = _run_batched(mt_sess, mt_sess.get_inputs()[0].name, mt_inputs, args.multitask_batch)
            stage["multitask_dml"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        out_cursor = 0
        for meta in face_meta:
            base = {
                "benchmark_index": meta["benchmark_index"],
                "face_rank": meta["face_rank"],
                "detected": meta["detected"],
            }
            if not meta["detected"] or outputs is None:
                rows.append(base)
                continue
            au, emo, va, gaze_raw, pose_raw, mesh, bs = [o[out_cursor] for o in outputs]
            out_cursor += 1
            ox, oy, side = meta["crop"]
            base.update({
                "FaceRectX": ox,
                "FaceRectY": oy,
                "FaceRectWidth": side,
                "FaceRectHeight": side,
                "FaceScore": meta["score"],
                "Pitch": -float(pose_raw[0]),
                "Roll": -float(pose_raw[2]),
                "Yaw": float(pose_raw[1]),
                "X": float(pose_raw[3]),
                "Y": float(pose_raw[4]),
                "Z": float(pose_raw[5]),
                "gaze_pitch": -float(gaze_raw[1]),
                "gaze_yaw": float(gaze_raw[0]),
            })
            base["gaze_angle"] = float(np.arccos(np.clip(
                np.cos(base["gaze_pitch"]) * np.cos(base["gaze_yaw"]), -1.0, 1.0
            )))
            for name, val in zip(PYFEAT_AUS, au):
                base[name] = float(val)
            for name, val in zip(PYFEAT_EMOTIONS, emo):
                base[name] = float(val)
            base["valence"], base["arousal"] = float(va[0]), float(va[1])
            for j in range(478):
                base[f"mesh_x_{j}"] = float(mesh[j, 0] * side + ox)
                base[f"mesh_y_{j}"] = float(mesh[j, 1] * side + oy)
                base[f"mesh_z_{j}"] = float(mesh[j, 2])
            for name, val in zip(BLENDSHAPES, bs):
                base[name] = float(val)
            rows.append(base)
        stage["postprocess"] += time.perf_counter() - t0

    total_sec = time.perf_counter() - total_start
    raw = pd.DataFrame(rows)
    out_path = out_dir / "pyfeat_dml_raw.parquet"
    raw.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    manifest = {
        "schema_version": "rgb-face-real300-pyfeat-dml-v0.1",
        "candidate": "pyfeat_detectorv2_scientific_core",
        "benchmark_frame_manifest": str(frame_manifest),
        "expected_frames": int(len(frames)),
        "output_rows": int(len(raw)),
        "detected_rows": int(raw.get("detected", pd.Series(dtype=bool)).fillna(False).sum()),
        "retinaface_batch": int(args.retinaface_batch),
        "multitask_batch": int(args.multitask_batch),
        "timing_sec": {**stage, "actual_raw_frame_end_to_end": total_sec},
        "input_frames_per_sec_end_to_end": len(frames) / total_sec if total_sec > 0 else None,
        "models": {
            rf_model.stem: {"path": str(rf_model), "sha256": _sha256(rf_model)},
            mt_model.stem: {"path": str(mt_model), "sha256": _sha256(mt_model)},
        },
        "raw_output": str(out_path),
        "notes": [
            "Raw frames -> RetinaFace ONNX/DML -> numpy decode/NMS -> isotropic 1.2 square reflection crop -> 224 ImageNet normalize -> multitask ONNX/DML.",
            "The crop geometry follows Py-Feat 2.1.1 extract_face_square_pad_torch. OpenCV bilinear/remap is used for the CPU-side sampling implementation; CPU-reference parity is the acceptance test for any interpolation drift.",
            "Identity is intentionally excluded, matching the Gate-0 scientific-core decision.",
            "Multi-face detections are retained as separate face_rank rows; no primary-face filtering is applied here.",
        ],
    }
    (out_dir / "pyfeat_dml_real300_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Real shared-frame DirectML runner for RGB Face candidates")
    sub = parser.add_subparsers(dest="candidate", required=True)

    lp = sub.add_parser("libreface")
    lp.add_argument("--prep-dir", required=True)
    lp.add_argument("--model-dir", required=True)
    lp.add_argument("--output-dir", required=True)
    lp.add_argument("--batch-size", type=int, default=16)

    pp = sub.add_parser("pyfeat")
    pp.add_argument("--benchmark-dir", required=True)
    pp.add_argument("--model-dir", required=True)
    pp.add_argument("--output-dir", required=True)
    pp.add_argument("--retinaface-batch", type=int, default=8)
    pp.add_argument("--multitask-batch", type=int, default=16)

    args = parser.parse_args()
    if args.candidate == "libreface":
        run_libreface(args)
    else:
        run_pyfeat(args)


if __name__ == "__main__":
    main()
