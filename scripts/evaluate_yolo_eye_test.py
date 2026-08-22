"""Evaluate the trained single-class NIR eye detector on frozen splits.

The script keeps model selection on ``selection-split`` and evaluates the
frozen operating threshold on ``evaluation-split``.  It also writes an
auditable per-image/per-subject matching table without forcing two detections
per image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Box:
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_error_normalized(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ac = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    bc = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    diag = ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5
    return (((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5) / diag if diag > 0 else np.nan


def greedy_match(truth: list[Box], predictions: list[Box], match_iou: float) -> list[dict[str, Any]]:
    """Greedy one-to-one matching in descending prediction confidence order."""
    used_truth: set[int] = set()
    rows: list[dict[str, Any]] = []
    for pred_idx, pred in sorted(enumerate(predictions), key=lambda x: x[1].confidence, reverse=True):
        candidates = [(iou_xyxy(pred.xyxy, gt.xyxy), gt_idx) for gt_idx, gt in enumerate(truth) if gt_idx not in used_truth]
        best_iou, gt_idx = max(candidates, default=(0.0, None))
        if gt_idx is not None and best_iou >= match_iou:
            used_truth.add(gt_idx)
            rows.append({"kind": "match", "truth_index": gt_idx, "prediction_index": pred_idx,
                         "iou": best_iou, "center_error_norm": center_error_normalized(pred.xyxy, truth[gt_idx].xyxy),
                         "confidence": pred.confidence})
        else:
            rows.append({"kind": "fp", "truth_index": None, "prediction_index": pred_idx,
                         "iou": best_iou, "center_error_norm": np.nan, "confidence": pred.confidence})
    for gt_idx in range(len(truth)):
        if gt_idx not in used_truth:
            rows.append({"kind": "fn", "truth_index": gt_idx, "prediction_index": None,
                         "iou": 0.0, "center_error_norm": np.nan, "confidence": np.nan})
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _read_truth(label_path: Path, image_size: tuple[int, int]) -> list[Box]:
    width, height = image_size
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 or int(values[0]) != 0:
            raise ValueError(f"Invalid single-class YOLO label: {label_path}: {line}")
        _, cx, cy, w, h = map(float, values)
        boxes.append(Box(((cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height)))
    return boxes


def _load_manifest(path: Path, root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image", "subject", "batch", "split", "n_boxes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame["image_path"] = frame["image"].map(lambda p: root / Path(str(p).replace("/", "\\")))
    frame["label_path"] = frame["image_path"].map(lambda p: root / "labels" / p.parent.name / f"{p.stem}.txt")
    return frame


def _validate_splits(frame: pd.DataFrame) -> None:
    by_split = {split: set(group.subject) for split, group in frame.groupby("split")}
    for left in by_split:
        for right in by_split:
            if left < right and by_split[left] & by_split[right]:
                raise ValueError(f"Subject leakage between {left} and {right}: {sorted(by_split[left] & by_split[right])}")
    for _, row in frame.iterrows():
        if not row.image_path.exists() or not row.label_path.exists():
            raise FileNotFoundError(f"Missing image/label pair: {row.image}")
        actual = len(row.label_path.read_text(encoding="utf-8").splitlines())
        if actual != int(row.n_boxes):
            raise ValueError(f"Manifest label count mismatch: {row.image}: {actual} != {row.n_boxes}")


def _predict(model: Any, image_path: Path, args: argparse.Namespace, conf: float) -> list[Box]:
    result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=conf, iou=args.nms_iou,
                           max_det=300, device=args.device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [Box(tuple(map(float, box)), float(score)) for box, score, cls in zip(boxes, scores, classes) if cls == 0]


def _operating_threshold(model: Any, frame: pd.DataFrame, args: argparse.Namespace) -> tuple[float, pd.DataFrame]:
    rows = []
    for _, item in frame.iterrows():
        from PIL import Image
        with Image.open(item.image_path) as image:
            size = image.size
        truth = _read_truth(item.label_path, size)
        predictions = _predict(model, item.image_path, args, 0.001)
        for threshold in np.linspace(0.05, 0.95, 19):
            filtered = [p for p in predictions if p.confidence >= threshold]
            matches = greedy_match(truth, filtered, args.match_iou)
            tp = sum(r["kind"] == "match" for r in matches)
            fp = sum(r["kind"] == "fp" for r in matches)
            fn = sum(r["kind"] == "fn" for r in matches)
            rows.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn})
    sweep = pd.DataFrame(rows).groupby("threshold", as_index=False)[["tp", "fp", "fn"]].sum()
    metrics = sweep.apply(lambda r: pd.Series(_metric(int(r.tp), int(r.fp), int(r.fn))), axis=1)
    sweep = pd.concat([sweep, metrics], axis=1)
    best_f1 = sweep.f1.max()
    threshold = float(sweep[sweep.f1 == best_f1].threshold.max())
    return threshold, sweep


def _evaluate_images(model: Any, frame: pd.DataFrame, args: argparse.Namespace, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    from PIL import Image
    for _, item in frame.iterrows():
        with Image.open(item.image_path) as image:
            size = image.size
        truth = _read_truth(item.label_path, size)
        predictions = _predict(model, item.image_path, args, threshold)
        matches = greedy_match(truth, predictions, args.match_iou)
        for match in matches:
            detail_rows.append({"image": item.image, "subject": item.subject, "batch": item.batch,
                                "n_truth": len(truth), "n_predictions": len(predictions), **match})
        image_rows.append({"image": item.image, "subject": item.subject, "batch": item.batch,
                           "n_truth": len(truth), "n_predictions": len(predictions),
                           "tp": sum(r["kind"] == "match" for r in matches),
                           "fp": sum(r["kind"] == "fp" for r in matches),
                           "fn": sum(r["kind"] == "fn" for r in matches),
                           "all_annotated_eyes_found": len(truth) > 0 and all(r["kind"] == "match" for r in matches if r["truth_index"] is not None),
                           "two_eye_truth": len(truth) == 2,
                           "two_eye_success": len(truth) == 2 and sum(r["kind"] == "match" for r in matches) == 2,
                           "extra_box": len(predictions) > len(truth),
                           "zero_eye": len(predictions) == 0,
                           "single_eye": len(predictions) == 1})
    return pd.DataFrame(detail_rows), pd.DataFrame(image_rows)


def _summary(detail: pd.DataFrame, images: pd.DataFrame, split: str, subject: str | None = None) -> dict[str, Any]:
    scoped = images if subject is None else images[images.subject == subject]
    d = detail if subject is None else detail[detail.subject == subject]
    tp, fp, fn = int((d.kind == "match").sum()), int((d.kind == "fp").sum()), int((d.kind == "fn").sum())
    result: dict[str, Any] = {"split": split, "subject": subject or "__overall__", "images": int(len(scoped)),
                              "truth_boxes": int(scoped.n_truth.sum()), "predictions": int(scoped.n_predictions.sum()),
                              "tp": tp, "fp": fp, "fn": fn, **_metric(tp, fp, fn),
                              "all_annotated_eyes_found_rate": float(scoped.all_annotated_eyes_found.mean()) if len(scoped) else 0.0,
                              "two_eye_success_rate": float(scoped.loc[scoped.two_eye_truth, "two_eye_success"].mean()) if scoped.two_eye_truth.any() else np.nan,
                              "extra_box_rate": float(scoped.extra_box.mean()) if len(scoped) else 0.0,
                              "zero_eye_rate": float(scoped.zero_eye.mean()) if len(scoped) else 0.0,
                              "single_eye_rate": float(scoped.single_eye.mean()) if len(scoped) else 0.0}
    matched = d[d.kind == "match"]
    result["iou_mean"] = float(matched.iou.mean()) if len(matched) else np.nan
    result["iou_median"] = float(matched.iou.median()) if len(matched) else np.nan
    result["iou_p10"] = float(matched.iou.quantile(0.10)) if len(matched) else np.nan
    result["center_error_norm_median"] = float(matched.center_error_norm.median()) if len(matched) else np.nan
    result["min_confidence"] = float(d.loc[d.kind != "fn", "confidence"].min()) if (d.kind != "fn").any() else np.nan
    return result


def _native_val(model: Any, args: argparse.Namespace, data: Path | str, split: str, output: Path,
                name: str, plots: bool = True) -> dict[str, Any]:
    result = model.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                       workers=args.workers, conf=0.001, iou=args.nms_iou, plots=plots,
                       project=str(output.resolve()), name=name, exist_ok=True, verbose=False)
    box = result.box
    return {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50), "mAP50-95": float(box.map)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if model.names != {0: "eye"}:
        raise ValueError(f"Expected {{0: 'eye'}}, got {model.names}")
    root = args.data.resolve().parent
    with args.manifest.open(encoding="utf-8-sig") as f:
        manifest = pd.read_csv(f.name)
    manifest = _load_manifest(args.manifest, root)
    _validate_splits(manifest)
    selection = manifest[manifest.split == args.selection_split].copy()
    evaluation = manifest[manifest.split == args.evaluation_split].copy()
    if selection.empty or evaluation.empty:
        raise ValueError("Selection or evaluation split is empty")
    threshold, sweep = _operating_threshold(model, selection, args)
    sweep.to_csv(args.output / "val_threshold_sweep.csv", index=False)
    native = _native_val(model, args, args.data.resolve(), args.evaluation_split, args.output,
                         f"native_{args.evaluation_split}", plots=True)
    detail, images = _evaluate_images(model, evaluation, args, threshold)
    detail.to_csv(args.output / "per_image_predictions.csv", index=False)
    images.to_csv(args.output / "per_image_summary.csv", index=False)
    images[(images.fp > 0) | (images.fn > 0)].to_csv(args.output / "failure_index.csv", index=False)
    overall = _summary(detail, images, args.evaluation_split)
    subjects = [_summary(detail, images, args.evaluation_split, s) for s in sorted(evaluation.subject.unique())]
    per_subject = pd.DataFrame(subjects)
    subject_yaml_dir = args.output / "subject_val_yamls"
    subject_yaml_dir.mkdir(exist_ok=True)
    native_subject_rows = []
    for subject in sorted(evaluation.subject.unique()):
        subject_yaml = subject_yaml_dir / f"{subject}.yaml"
        subject_list = subject_yaml_dir / f"{subject}.txt"
        subject_images = evaluation[evaluation.subject == subject]["image"].tolist()
        subject_list.write_text("\n".join(str((root / Path(p.replace("/", "\\"))).resolve()) for p in subject_images) + "\n", encoding="utf-8")
        subject_yaml.write_text(yaml.safe_dump({"path": str(root), "train": str(subject_list.resolve()), "val": str(subject_list.resolve()),
                                                 "names": {0: "eye"}}, sort_keys=False), encoding="utf-8")
        subject_native = _native_val(model, args, subject_yaml, "val", args.output / "native_subject",
                                     f"{subject}", plots=False)
        native_subject_rows.append({"subject": subject, "AP50": subject_native["mAP50"],
                                    "mAP50-95": subject_native["mAP50-95"],
                                    "native_precision": subject_native["precision"],
                                    "native_recall": subject_native["recall"]})
    native_subject = pd.DataFrame(native_subject_rows)
    native_subject.to_csv(args.output / "native_subject_metrics.csv", index=False)
    per_subject = per_subject.merge(native_subject, on="subject", how="left")
    per_subject.to_csv(args.output / "per_subject_metrics.csv", index=False)
    combined = {**overall, "native_ap": native, "operating_threshold": threshold,
                "selection_split": args.selection_split, "evaluation_split": args.evaluation_split}
    (args.output / "overall_metrics.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([combined]).to_csv(args.output / "overall_metrics.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        plot_frame = per_subject.set_index("subject")[["f1", "recall", "AP50", "mAP50-95"]]
        ax = plot_frame.plot.bar(figsize=(11, 5), ylim=(0, 1), rot=30)
        ax.set_ylabel("score")
        ax.set_title("YOLO26n eye detector: frozen test metrics by subject")
        ax.grid(axis="y", alpha=0.25)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(args.output / "per_subject_metrics.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    (args.output / "run_manifest.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "model": str(args.model.resolve()), "model_sha256": _sha256(args.model), "model_bytes": args.model.stat().st_size,
        "data": str(args.data.resolve()), "manifest": str(args.manifest.resolve()),
        "arguments": vars(args), "python": sys.version, "platform": platform.platform(),
        "packages": {name: __import__(name).__version__ for name in ("ultralytics", "torch", "cv2", "numpy", "pandas")},
        "torch_cuda_available": bool(__import__("torch").cuda.is_available()),
        "split_counts": manifest.groupby("split").size().to_dict(),
        "split_subject_counts": manifest.groupby("split").subject.nunique().to_dict(),
        "test_boxes": int(evaluation.n_boxes.sum()), "operating_threshold": threshold,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "operating_threshold": threshold,
                      "overall": overall, "native_ap": native}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
