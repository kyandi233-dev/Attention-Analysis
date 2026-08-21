"""NIR 眼框数据集抽帧（两批分开训练）。

策略（见 docs/010-nir/08-17-01 步骤A/B 与 08-17-02 标注指南）：
  - 每 (subject, block) 按时间均匀抽 N 帧，只在 block_start/block_stop 任务窗口内；
  - 用 unix_ms → frame_idx 映射（不按帧号/30）；
  - 确定性 seed，同 (subject,block) 帧号去重；
  - batch1 / batch2 分开，各自按被试划分 train/val/test（不按帧划分）。

数据事实（2026-08-19 核实）：
  - batch1 = sub-011~030 + sub-9504，21 人 × 3 block（环境一：桌子矮一点、被试更容易离开画面）；
  - batch2 = sub-031~055 + sub-061，26 人 × 2 block（环境二：桌子正常；用户确认边界为 >=31）；
  - 两批实验环境（桌椅/光照/取景）不同，分开训练两个模型。

用法（主分析环境需有 opencv-python + pandas；本机用 C:/Python314）：
    $env:PYTHONPATH='src'
    & 'C:/Python314/python.exe' scripts/extract_eye_dataset.py --out datasets/nir-eye-dataset-v1
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.nir.formal import (
    block_window,
    formal_subject_paths,
    load_nir_timestamps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH1_DEFAULT = [f"sub-{i:03d}" for i in range(11, 31)] + ["sub-9504"]
BATCH2_DEFAULT = [f"sub-{i:03d}" for i in range(31, 56)] + ["sub-061"]
# 边界 >=31（用户确认）：sub-030 属环境一（3 block）；sub-031..055/061 属环境二（2 block）。
# sub-056..060 不存在。

FRAME_FIELDS = [
    "image_id", "subject", "batch", "block", "frame_idx", "unix_ms",
    "image_path", "source_video",
]
ANNOTATION_FIELDS = [
    "image_id", "subject", "video", "frame_idx", "frame_type",
    "n_eye_boxes", "quality", "annotator", "review_status", "notes",
]


def discover_blocks(master_timeline: Path) -> list[int]:
    """从 master_timeline 发现同时有 block_start/block_stop 的 block 编号。"""
    timeline = pd.read_csv(master_timeline)
    starts = timeline[timeline["event"] == "block_start"]
    stops = timeline[timeline["event"] == "block_stop"]
    stop_by = {str(d): int(t) for d, t in zip(stops["detail"], stops["unix_ms"])}
    blocks = []
    for detail, start_ms in zip(starts["detail"], starts["unix_ms"]):
        detail = str(detail)
        if detail in stop_by and int(start_ms) < stop_by[detail]:
            try:
                num = int(detail.split("_")[0].replace("Block", ""))
            except ValueError:
                continue
            blocks.append(num)
    return sorted(set(blocks))


def frame_indices_for_block(
    timestamps: pd.DataFrame,
    start_ms: int,
    stop_ms: int,
    n: int,
    margin_ms: int,
) -> list[tuple[int, int]]:
    """在 block 窗口边距内按时间均匀取 n 个帧，返回 [(frame_idx, unix_ms), ...]。

    用区间中点（非端点）避免贴边；同窗口帧号去重（碰撞 +1 帧）。
    """
    unix = timestamps["unix_ms"].to_numpy()
    frame = timestamps["frame_idx"].to_numpy()
    ms_by_frame = {int(f): int(u) for f, u in zip(frame, unix)}
    last_frame = int(frame[-1])

    a = start_ms + margin_ms
    b = stop_ms - margin_ms
    if b - a < 500:  # block 太短，退回无边距窗口
        a, b = start_ms, stop_ms
    centers = a + (np.arange(n) + 0.5) / n * (b - a)

    out = []
    used: set[int] = set()
    for center in centers:
        pos = int(np.searchsorted(unix, center, side="left"))
        if pos >= len(unix):
            continue
        fidx = int(frame[pos])
        while fidx in used and fidx < last_frame:
            fidx += 1
        if fidx in used or fidx not in ms_by_frame:
            continue
        used.add(fidx)
        ms = ms_by_frame[fidx]
        if not (start_ms <= ms <= stop_ms):
            continue
        out.append((fidx, ms))
    return out


def extract_gray(cap, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def save_jpg(path: Path, gray: np.ndarray) -> None:
    """cv2.imwrite 在中文路径下静默失败；用 imencode + write_bytes 规避（同 review._imwrite）。"""
    ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def make_split(subjects: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    """按被试确定性划分 train/val/test（70/15/15）。"""
    rng = random.Random(seed)
    order = sorted(subjects)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * 0.70))
    n_val = max(1, int(round(n * 0.15)))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_train = n - n_val
        n_test = 0
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="NIR 眼框数据集抽帧：按 block 时间均匀抽帧，两批分开。"
    )
    parser.add_argument("--root", default="E:/正式实验")
    parser.add_argument("--out", default="datasets/nir-eye-dataset-v1")
    parser.add_argument("--frames-per-block", type=int, default=5)
    parser.add_argument("--margin-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--blocks", default="", help="逗号分隔 block 编号，默认全部")
    parser.add_argument("--batch1", default=",".join(BATCH1_DEFAULT))
    parser.add_argument("--batch2", default=",".join(BATCH2_DEFAULT))
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    margin_ms = int(round(args.margin_sec * 1000))
    block_filter = [int(b) for b in args.blocks.split(",") if b.strip()]
    batch1 = [s.strip() for s in args.batch1.split(",") if s.strip()]
    batch2 = [s.strip() for s in args.batch2.split(",") if s.strip()]

    images_dir = out / "images"
    labels_dir = out / "labels_yolo"
    manifests_dir = out / "manifests"
    previews_dir = out / "previews"
    for d in (images_dir, labels_dir, manifests_dir, previews_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames: list[dict] = []
    skipped: list[tuple[str, str, str]] = []
    per_batch_counts = {"batch1": {}, "batch2": {}}

    for batch, subjects in (("batch1", batch1), ("batch2", batch2)):
        batch_images = images_dir / batch
        batch_images.mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            try:
                paths = formal_subject_paths(root, subject)
            except Exception as exc:  # 目录名异常
                skipped.append((subject, "paths", str(exc)))
                continue
            missing = [name for name in ("nir_video", "nir_timestamps", "master_timeline")
                       if not paths[name].exists()]
            if missing:
                skipped.append((subject, "missing", ",".join(missing)))
                continue
            try:
                timestamps = load_nir_timestamps(paths["nir_timestamps"])
            except Exception as exc:
                skipped.append((subject, "timestamps", str(exc)))
                continue
            blocks = discover_blocks(paths["master_timeline"])
            if block_filter:
                blocks = [b for b in blocks if b in block_filter]
            if not blocks:
                skipped.append((subject, "blocks", "none"))
                continue

            cap = cv2.VideoCapture(str(paths["nir_video"]))
            if not cap.isOpened():
                skipped.append((subject, "video", "cannot open"))
                continue

            subject_frames: list[dict] = []
            extract_fail = 0
            for block in blocks:
                start_ms, stop_ms = block_window(paths["master_timeline"], block)
                picks = frame_indices_for_block(
                    timestamps, start_ms, stop_ms, args.frames_per_block, margin_ms
                )
                for fidx, ms in picks:
                    image_id = f"subject_{subject}_frame_{fidx:06d}"
                    subject_frames.append({
                        "image_id": image_id,
                        "subject": subject,
                        "batch": batch,
                        "block": block,
                        "frame_idx": fidx,
                        "unix_ms": ms,
                        "image_path": f"images/{batch}/{image_id}.jpg",
                        "source_video": str(paths["nir_video"]),
                    })
            # 同一被试集中 seek 提取，减少重复打开视频
            for rec in sorted(subject_frames, key=lambda r: r["frame_idx"]):
                gray = extract_gray(cap, rec["frame_idx"])
                if gray is None:
                    extract_fail += 1
                    continue
                save_jpg(batch_images / f"{rec['image_id']}.jpg", gray)
            cap.release()

            per_batch_counts[batch][subject] = len(subject_frames)
            if extract_fail:
                skipped.append((subject, "extract_fail", str(extract_fail)))
            frames.extend(subject_frames)
            print(f"[extract] {batch} {subject} blocks={blocks} frames={len(subject_frames)}",
                  flush=True)

    # ---- manifests / frames.csv ----
    frames_csv = manifests_dir / "frames.csv"
    with frames_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(frames)

    # ---- split_subject.csv + 各 split 图片清单 txt ----
    split_rows = []
    for batch, subjects in (("batch1", batch1), ("batch2", batch2)):
        train, val, test = make_split(subjects, args.seed)
        for subject in subjects:
            split_rows.append({"subject": subject, "batch": batch,
                               "split": ("train" if subject in train else
                                         "val" if subject in val else "test")})
        # 真实路径清单按 images 目录实际文件写（图片名含 subject_<stem>_frame_）
        batch_files = sorted(p.resolve().as_posix() for p in (out / f"images/{batch}").glob("*.jpg"))
        for label, group in (("train", train), ("val", val), ("test", test)):
            members = set(group)
            selected = [p for p in batch_files
                        if any(f"/{s}_frame_" in p for s in members)]
            (manifests_dir / f"split_{batch}_{label}.txt").write_text(
                "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    with (manifests_dir / "split_subject.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "batch", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    # ---- annotations.csv 占位（按 08-17-02 字段）----
    with (manifests_dir / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for rec in frames:
            writer.writerow({
                "image_id": rec["image_id"], "subject": rec["subject"],
                "video": Path(rec["source_video"]).name,
                "frame_idx": rec["frame_idx"],
                "frame_type": "", "n_eye_boxes": "", "quality": "",
                "annotator": "", "review_status": "", "notes": "",
            })

    # ---- classes.txt / dataset yaml ----
    (out / "classes.txt").write_text("eye\n", encoding="utf-8")
    for batch in ("batch1", "batch2"):
        yaml_path = out / f"dataset_{batch}.yaml"
        lines = [
            f"# {batch}：NIR 眼框单类检测（train/val/test 按被试划分，见 manifests/split_subject.csv）",
            f"path: {str(out.resolve().as_posix())}",
            f"train: {str((manifests_dir / f'split_{batch}_train.txt').resolve().as_posix())}",
            f"val: {str((manifests_dir / f'split_{batch}_val.txt').resolve().as_posix())}",
            f"test: {str((manifests_dir / f'split_{batch}_test.txt').resolve().as_posix())}",
            "names:",
            "  0: eye",
            "",
        ]
        yaml_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- 汇总 ----
    total = len(frames)
    print("\n=== 抽帧汇总 ===")
    for batch in ("batch1", "batch2"):
        counts = per_batch_counts[batch]
        n_subj = sum(1 for v in counts.values() if v > 0)
        print(f"{batch}: {len(counts)} 被试, 有帧 {n_subj}, 帧数 {sum(counts.values())}")
    print(f"总帧数: {total}")
    if skipped:
        print("\n-- 跳过/失败 --")
        for row in skipped:
            print("  ", row)
    print(f"\n输出目录: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
