#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_ritnet.py — RITnet 最小推理封装（只读实验脚本，不改生产代码）

输入：单眼 ROI 图（本项目 320x160，WxH，灰度）
流程：预处理(gamma0.8 + CLAHE + 归一化) -> DenseNet2D 推理 -> 4类语义分割
      -> 瞳孔(类别3)掩膜 -> 在 320x160 坐标下拟合椭圆
输出：每眼一行 CSV；可对照 ground_truth_528.csv 计算中心误差/直径误差/IoU/伪阳性。

用法：
  python infer_ritnet.py --roi_dir <dir> --out <dir> [--gt_csv <csv>] [--limit N] [--overlays N]
"""
import argparse
import csv
import glob
import io
import json
import os
import sys

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from densenet import DenseNet2D

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pkl")

TARGET_W, TARGET_H = 640, 400   # RITnet 训练输入（OpenEDS 640x400 横构图）
ROI_W, ROI_H = 320, 160         # 本项目固定眼角 ROI


def load_model(device):
    model = DenseNet2D(dropout=True, prob=0.2)
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def preprocess(roi_gray, device):
    """roi_gray: (ROI_H, ROI_W) uint8 -> tensor (1,1,TARGET_H,TARGET_W)"""
    img = cv2.resize(roi_gray, (TARGET_W, TARGET_H))          # 拉伸到训练尺寸
    table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)  # gamma 0.8
    img = cv2.LUT(img, table)
    img = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(img)
    x = torch.from_numpy(img).float().to(device)
    x = (x / 255.0 - 0.5) / 0.5
    return x[None, None]


def infer(model, roi_gray, device):
    with torch.no_grad():
        logits = model(preprocess(roi_gray, device))          # (1,4,TH,TW)
        probs = torch.softmax(logits, dim=1)
    pred = logits[0].argmax(0).cpu().numpy()                  # (TH,TW)
    prob_pupil = probs[0, 3].cpu().numpy()                    # 瞳孔类概率图
    return pred, prob_pupil


def pupil_ellipse_roi(pred, prob_pupil):
    """在模型尺度提取瞳孔掩膜 -> 缩回 ROI 坐标 -> 拟合椭圆。
    返回 dict 或 None（无足够大瞳孔连通域）。"""
    pupil = (pred == 3).astype(np.uint8)
    mask_small = cv2.resize(pupil, (ROI_W, ROI_H), interpolation=cv2.INTER_NEAREST)
    mask_small = (mask_small > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_small * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 5:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
    # 瞳孔区 softmax 均值（弱置信度）
    conf = float(prob_pupil[pupil.astype(bool)].mean()) if pupil.any() else 0.0
    return dict(
        found=True,
        center_x=float(cx), center_y=float(cy),
        major_diameter=float(MA), minor_diameter=float(ma),
        angle_deg=float(ang),
        mask_area=float(cv2.contourArea(c)),
        equiv_diameter=float(2 * np.sqrt(cv2.contourArea(c) / np.pi)),
        pupil_confidence=conf,
    )


def ellipse_mask(center, major, minor, angle, h=ROI_H, w=ROI_W):
    m = np.zeros((h, w), np.uint8)
    if major <= 0 or minor <= 0:
        return m
    (cx, cy), (MA, ma), ang = center, (float(major), float(minor)), float(angle)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                (int(round(MA / 2)), int(round(ma / 2))), ang, 0, 360, 255, -1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt_csv", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=0, help="保存前 N 个 overlay")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL_PATH}", flush=True)

    # 真值
    gt = {}
    if args.gt_csv and os.path.exists(args.gt_csv):
        raw = open(args.gt_csv, "rb").read()
        for enc in ("utf-8-sig", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        for r in csv.DictReader(io.StringIO(text)):
            gt[r["sample_id"]] = r

    files = sorted(glob.glob(os.path.join(args.roi_dir, "*_eye_*.png")))
    print(f"roi files found: {len(files)}", flush=True)
    if args.limit:
        files = files[: args.limit]

    model = load_model(device)
    results = []
    for i, fp in enumerate(files):
        sid = os.path.basename(fp)[: -len(".png")]
        roi = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            print(f"  ! read fail {sid}", flush=True)
            continue
        pred, prob_pupil = infer(model, roi, device)
        e = pupil_ellipse_roi(pred, prob_pupil)
        row = {"sample_id": sid}
        row.update(e or {"found": False})
        if sid in gt:
            g = gt[sid]
            row["visibility"] = g["visibility"]
            row["evaluation_excluded"] = g["evaluation_excluded"]
            for k in ("center_x", "center_y", "major_diameter", "minor_diameter",
                      "equivalent_diameter", "angle_deg"):
                v = g.get(k, "")
                row["gt_" + k] = v if v != "" else None
        results.append(row)
        # overlay（原 ROI + 瞳孔掩膜 + 椭圆）
        if args.overlays and i < args.overlays:
            vis = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            if e:
                m = ellipse_mask((e["center_x"], e["center_y"]), e["major_diameter"],
                                 e["minor_diameter"], e["angle_deg"])
                overlay = vis.copy()
                overlay[m > 0] = (255, 80, 80)
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
                cv2.ellipse(vis, (int(round(e["center_x"])), int(round(e["center_y"]))),
                            (int(round(e["major_diameter"] / 2)), int(round(e["minor_diameter"] / 2))),
                            e["angle_deg"], 0, 360, (0, 200, 255), 1)
            if sid in gt and gt[sid]["visibility"] == "可见" and gt[sid]["center_x"]:
                g = gt[sid]
                cv2.ellipse(vis, (int(float(g["center_x"])), int(float(g["center_y"]))),
                            (int(float(g["major_diameter"]) / 2), int(float(g["minor_diameter"]) / 2)),
                            float(g["angle_deg"]), 0, 360, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(args.out, f"overlay_{sid}.png"), vis)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    out_csv = os.path.join(args.out, "ritnet_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        cols = ["sample_id", "found", "center_x", "center_y", "major_diameter", "minor_diameter",
                "angle_deg", "equiv_diameter", "mask_area", "pupil_confidence",
                "visibility", "evaluation_excluded",
                "gt_center_x", "gt_center_y", "gt_major_diameter", "gt_minor_diameter",
                "gt_equivalent_diameter", "gt_angle_deg"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {out_csv}  ({len(results)} rows)", flush=True)

    # 指标
    found = [r for r in results if r.get("found")]
    vis_found = [r for r in found if r.get("visibility") == "可见"
                 and not r.get("evaluation_excluded") and r.get("gt_center_x") is not None]
    vis_all = [r for r in results if r.get("visibility") == "可见"
               and not r.get("evaluation_excluded") and r.get("gt_center_x") is not None]
    invis_all = [r for r in results if r.get("visibility") == "不可见"]
    invis_found = [r for r in invis_all if r.get("found")]

    cxs, dys, ious, rels = [], [], [], []
    for r in vis_found:
        gt_cx, gt_cy = float(r["gt_center_x"]), float(r["gt_center_y"])
        ce = np.hypot(r["center_x"] - gt_cx, r["center_y"] - gt_cy)
        cxs.append(ce)
        gd = float(r["gt_equivalent_diameter"])
        if gd > 0:
            rels.append(abs(r["equiv_diameter"] - gd) / gd)
        m_gt = ellipse_mask((gt_cx, gt_cy), float(r["gt_major_diameter"]),
                            float(r["gt_minor_diameter"]), float(r["gt_angle_deg"]))
        m_pr = ellipse_mask((r["center_x"], r["center_y"]), r["major_diameter"],
                            r["minor_diameter"], r["angle_deg"])
        inter = np.logical_and(m_gt > 0, m_pr > 0).sum()
        union = np.logical_or(m_gt > 0, m_pr > 0).sum()
        ious.append(float(inter) / float(union) if union else 0.0)
        dys.append(float(r["equiv_diameter"]) - gd)

    def mean(x):
        return float(np.mean(x)) if x else None

    summary = {
        "n_total": len(results),
        "n_found": len(found),
        "n_visible_gt": len(vis_all),
        "n_visible_found": len(vis_found),
        "visible_detection_rate": (len(vis_found) / len(vis_all)) if vis_all else None,
        "center_error_px_mean": mean(cxs),
        "center_error_normalized_mean": (mean(cxs) / ROI_H) if cxs else None,  # corner span=160
        "equiv_diameter_rel_error_mean": mean(rels),
        "equiv_diameter_diff_px_mean": mean(dys),
        "ellipse_iou_mean": mean(ious),
        "n_invisible": len(invis_all),
        "n_invisible_false_positive": len(invis_found),
        "invisible_false_positive_rate": (len(invis_found) / len(invis_all)) if invis_all else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== 摘要 ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
