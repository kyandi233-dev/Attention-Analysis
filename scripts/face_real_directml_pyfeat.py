from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PYFEAT_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
PYFEAT_EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
DLIB68_FROM_MP478 = [
    127, 234, 93, 132, 58, 172, 136, 150, 176, 148, 152, 377, 400, 379, 365, 397, 356,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 6, 195, 4,
    240, 75, 1, 305, 460, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
]
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
        str(model),
        sess_options=so,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        enable_fallback=0,
    )
    sess.disable_fallback()
    return sess


def _imagenet_nchw(rgb_float01: np.ndarray) -> np.ndarray:
    x = (rgb_float01 - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)


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


def _decode_landmarks(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    shaped = pre.reshape(pre.shape[0], pre.shape[1], 5, 2)
    out = priors[None, :, None, :2] + shaped * 0.1 * priors[None, :, None, 2:]
    return out.reshape(pre.shape[0], pre.shape[1], 10).astype(np.float32, copy=False)


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


def _square_reflect_crop(
    rgb: np.ndarray, box: np.ndarray, size: int = 256
) -> tuple[np.ndarray, tuple[float, float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * 1.2
    side = max(side, 1.0)
    ox, oy = cx - side / 2.0, cy - side / 2.0
    coords = (np.arange(size, dtype=np.float32) + 0.5) / float(size) * side
    map_x = np.broadcast_to((ox + coords)[None, :], (size, size)).astype(np.float32)
    map_y = np.broadcast_to((oy + coords)[:, None], (size, size)).astype(np.float32)
    crop = cv2.remap(
        rgb.astype(np.float32) / 255.0,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
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
    manifests = list(dict.fromkeys(
        sorted(root.glob("*_face-continuous_frames.csv"))
        + sorted(root.glob("*_face-benchmark_frames.csv"))
    ))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one frame manifest, found {len(manifests)}")
    frame_manifest = manifests[0]
    frames = pd.read_csv(frame_manifest)
    if frames.empty or not {"benchmark_index", "image_path"}.issubset(frames.columns):
        raise ValueError(f"Invalid shared frame manifest: {frame_manifest}")

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
            np.transpose(
                im.astype(np.float32) - np.array([123.0, 117.0, 104.0], np.float32),
                (2, 0, 1),
            )
            for im in rgb_list
        ])
        stage["image_read_preprocess"] += time.perf_counter() - t0
        if priors is None:
            priors = _generate_priors(h, w)

        t0 = time.perf_counter()
        loc, conf, landm = rf_sess.run(None, {rf_sess.get_inputs()[0].name: rf_in})
        stage["retinaface_dml"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        boxes_all = _decode_boxes(np.asarray(loc), priors)
        landmarks_all = _decode_landmarks(np.asarray(landm), priors)
        boxes_all *= np.array([w, h, w, h], dtype=np.float32)[None, None, :]
        landmarks_all *= np.tile(np.array([w, h], dtype=np.float32), 5)[None, None, :]
        face_meta: list[dict[str, Any]] = []
        chips: list[np.ndarray] = []

        for bi, row in enumerate(batch_df.itertuples(index=False)):
            scores = np.asarray(conf)[bi, :, 1]
            mask = scores > 0.02
            boxes, sc, lms5 = boxes_all[bi, mask], scores[mask], landmarks_all[bi, mask]
            if len(sc) > 5000:
                order = np.argsort(sc)[::-1][:5000]
                boxes, sc, lms5 = boxes[order], sc[order], lms5[order]
            keep = _nms(boxes, sc, 0.4)
            boxes, sc, lms5 = boxes[keep], sc[keep], lms5[keep]
            keep2 = sc >= 0.5
            boxes, sc, lms5 = boxes[keep2][:750], sc[keep2][:750], lms5[keep2][:750]

            if len(sc) == 0:
                face_meta.append({
                    "benchmark_index": int(row.benchmark_index),
                    "face_rank": 0,
                    "score": np.nan,
                    "crop": None,
                    "detected": False,
                    "image_path": str(row.image_path),
                    "frame_h": h,
                    "frame_w": w,
                    "rf_box": None,
                    "rf_landmarks5": None,
                })
                continue

            for rank, (box, score, lmk5) in enumerate(zip(boxes, sc, lms5)):
                chip, crop = _square_reflect_crop(rgb_list[bi], box)
                chips.append(chip)
                face_meta.append({
                    "benchmark_index": int(row.benchmark_index),
                    "face_rank": rank,
                    "score": float(score),
                    "crop": crop,
                    "detected": True,
                    "image_path": str(row.image_path),
                    "frame_h": h,
                    "frame_w": w,
                    "rf_box": box.astype(np.float32),
                    "rf_landmarks5": lmk5.astype(np.float32),
                })
        stage["decode_nms_crop"] += time.perf_counter() - t0

        outputs = None
        if chips:
            t0 = time.perf_counter()
            mt_inputs = np.stack([_preprocess_pyfeat_chip(c) for c in chips], axis=0)
            stage["multitask_preprocess"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            outputs = _run_batched(
                mt_sess, mt_sess.get_inputs()[0].name, mt_inputs, args.multitask_batch
            )
            stage["multitask_dml"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        out_cursor = 0
        for meta in face_meta:
            base: dict[str, Any] = {
                "benchmark_index": meta["benchmark_index"],
                "face_rank": meta["face_rank"],
                "detected": meta["detected"],
                "image_path": meta.get("image_path"),
                "FrameHeight": meta.get("frame_h"),
                "FrameWidth": meta.get("frame_w"),
            }
            if not meta["detected"] or outputs is None:
                rows.append(base)
                continue

            au, emo, va, gaze_raw, pose_raw, mesh, bs = [o[out_cursor] for o in outputs]
            out_cursor += 1
            ox, oy, side = meta["crop"]
            rf_box = meta["rf_box"]
            lmk5 = meta["rf_landmarks5"]
            base.update({
                "rf_bbox_x1": float(rf_box[0]),
                "rf_bbox_y1": float(rf_box[1]),
                "rf_bbox_x2": float(rf_box[2]),
                "rf_bbox_y2": float(rf_box[3]),
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
                "gaze_raw_yaw": float(gaze_raw[0]),
                "gaze_raw_pitch": float(gaze_raw[1]),
                "pose_raw_0": float(pose_raw[0]),
                "pose_raw_1": float(pose_raw[1]),
                "pose_raw_2": float(pose_raw[2]),
                "pose_raw_3": float(pose_raw[3]),
                "pose_raw_4": float(pose_raw[4]),
                "pose_raw_5": float(pose_raw[5]),
            })
            for j in range(5):
                base[f"rf_landmark5_x_{j}"] = float(lmk5[2 * j])
                base[f"rf_landmark5_y_{j}"] = float(lmk5[2 * j + 1])

            base["gaze_angle"] = float(np.arccos(np.clip(
                np.cos(base["gaze_pitch"]) * np.cos(base["gaze_yaw"]), -1.0, 1.0
            )))
            for name, val in zip(PYFEAT_AUS, au):
                base[name] = float(val)
            for name, val in zip(PYFEAT_EMOTIONS, emo):
                base[name] = float(val)
            base["valence"], base["arousal"] = float(va[0]), float(va[1])

            for j in range(478):
                base[f"mesh_norm_x_{j}"] = float(mesh[j, 0])
                base[f"mesh_norm_y_{j}"] = float(mesh[j, 1])
                base[f"mesh_norm_z_{j}"] = float(mesh[j, 2])
                base[f"mesh_x_{j}"] = float(mesh[j, 0] * side + ox)
                base[f"mesh_y_{j}"] = float(mesh[j, 1] * side + oy)
                base[f"mesh_z_{j}"] = float(mesh[j, 2])
            for dlib_idx, mp_idx in enumerate(DLIB68_FROM_MP478):
                base[f"x_{dlib_idx}"] = float(mesh[mp_idx, 0] * side + ox)
                base[f"y_{dlib_idx}"] = float(mesh[mp_idx, 1] * side + oy)
            for name, val in zip(BLENDSHAPES, bs):
                base[name] = float(val)
            rows.append(base)
        stage["postprocess"] += time.perf_counter() - t0

    total_sec = time.perf_counter() - total_start
    raw = pd.DataFrame(rows)
    out_path = out_dir / "pyfeat_dml_raw.parquet"
    raw.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")

    manifest = {
        "schema_version": "rgb-face-real300-pyfeat-dml-v0.2",
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
        "retained_output_families": [
            "RetinaFace decoded bbox + score",
            "RetinaFace decoded 5-point landmarks",
            "20 AU probabilities",
            "7 emotion probabilities",
            "valence/arousal",
            "raw + canonical gaze",
            "raw + canonical 6DoF pose",
            "478x3 normalized mesh",
            "478x3 original-frame convenience mesh",
            "dlib-68 compatibility landmark view",
            "52 blendshapes",
            "frame provenance",
        ],
        "notes": [
            "Raw frames -> RetinaFace ONNX/DML -> numpy decode/NMS -> isotropic 1.2 square reflection crop -> 224 ImageNet normalize -> multitask ONNX/DML.",
            "The crop geometry follows Py-Feat 2.1.1 extract_face_square_pad_torch. OpenCV remap/resize performs the CPU-side sampling; CPU-reference parity is the acceptance test for interpolation drift.",
            "Identity is intentionally excluded, matching the Gate-0 scientific-core decision for this single-participant-video measurement use case.",
            "Multi-face detections are retained as separate face_rank rows; no primary-face filtering is applied here.",
            "No inexpensive native scientific-core output is discarded in this validation layer.",
        ],
    }
    (out_dir / "pyfeat_dml_real300_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Real 300-frame Py-Feat DirectML runner")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retinaface-batch", type=int, default=8)
    parser.add_argument("--multitask-batch", type=int, default=16)
    run_pyfeat(parser.parse_args())


if __name__ == "__main__":
    main()
