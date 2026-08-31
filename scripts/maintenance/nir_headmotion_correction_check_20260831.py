# -*- coding: utf-8 -*-
"""
文件名：nir_headmotion_correction_check_20260831.py
版本：v1.0
功能：证明 NIR 瞳孔结果不是头动造成。聚合 RGB face 头动字段（面框尺度/外眼角/内眦/头姿旋转平移）到 block 层，
      与 NIR block 瞳孔（原始几何平均直径）join，比较：① 未校正瞳孔 vs 各头动字段的 Spearman 相关；
      ② 用 RGB 头动字段线性残差化（log 瞳孔）后，残差瞳孔 vs 各头动字段的相关。若矫正后相关趋近 0，
      说明瞳孔与头动关联可被 RGB 头动解释/消除，NIR 瞳孔效应非头动造成。
用法：python nir_headmotion_correction_check_20260831.py
依赖：pandas, numpy, scipy, pyarrow（读 face_raw.parquet）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.stats import spearmanr
from attention_pipeline.nir_pipeline_validation.figure_style import (
    configure_publication_style, make_figure, clean_axis, panel_label, finalize_layout, save_figure,
)

RGB_DIRS = [
    r"D:/Project/厚粲杯/11_数据/04_Attention-Analysis_nvidia-cuda_RGB",
    r"E:/_Analysis-needed/RGB/02_raw-first",
]
NIR_BLOCK = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/NIR/11_analysis_tables/block_session_models/block_session_model_table.csv"
OUT_DIR = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/new_figure"
FORMATS = ["png", "svg"]
PTOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE"]

HEAD_FIELDS = {
    "bbox_scale_px": "面框尺度（px）",
    "outer_eye_px": "外眼角距离（px）",
    "inner_canthus_px": "内眦距离（px）",
    "rotation_mag": "头姿旋转幅度",
    "translation_mag": "头姿平移幅度",
}


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _derive_head(frame: pd.DataFrame) -> pd.DataFrame:
    """从 face_raw 派生头动字段。"""
    w = _num(frame, "FaceRectWidth")
    h = _num(frame, "FaceRectHeight")
    frame["bbox_scale_px"] = np.sqrt((w * h).clip(lower=0))
    frame["outer_eye_px"] = np.hypot(
        _num(frame, "mesh_x_33") - _num(frame, "mesh_x_263"),
        _num(frame, "mesh_y_33") - _num(frame, "mesh_y_263"),
    )
    frame["inner_canthus_px"] = np.hypot(
        _num(frame, "mesh_x_133") - _num(frame, "mesh_x_362"),
        _num(frame, "mesh_y_133") - _num(frame, "mesh_y_362"),
    )
    frame["rotation_mag"] = np.sqrt(sum(_num(frame, c).fillna(0) ** 2 for c in ["Pitch", "Roll", "Yaw"]))
    frame["translation_mag"] = np.sqrt(sum(_num(frame, c).fillna(0) ** 2 for c in ["X", "Y", "Z"]))
    return frame


def collect_head_block():
    """遍历 D+E 盘 face_raw.parquet，按 session×block 聚合头动中位数。"""
    sessions = []
    for root in RGB_DIRS:
        if not os.path.isdir(root):
            continue
        for d in sorted(Path(root).glob("sub-*")):
            p = d / f"{d.name}_face_raw.parquet"
            if not p.exists():
                continue
            sessions.append(p)
    rows = []
    seen = set()
    for p in sessions:
        sid = p.parent.name
        if sid in seen:
            continue
        seen.add(sid)
        try:
            f = pd.read_parquet(p, columns=["block", "FaceRectWidth", "FaceRectHeight",
                                            "mesh_x_33", "mesh_x_263", "mesh_x_133", "mesh_x_362",
                                            "mesh_y_33", "mesh_y_263", "mesh_y_133", "mesh_y_362",
                                            "Pitch", "Roll", "Yaw", "X", "Y", "Z"])
        except Exception:
            continue
        f = _derive_head(f)
        g = f.groupby("block")[[*HEAD_FIELDS]].median()
        for blk, sub in g.iterrows():
            row = {"session_id": sid, "block_num": int(blk)}
            for k in HEAD_FIELDS:
                row[k] = sub[k]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    configure_publication_style()
    print("聚合 RGB face 头动（D+E 盘）...")
    head = collect_head_block()
    print(f"  头动 block 行数: {len(head)}")
    nir = pd.read_csv(NIR_BLOCK)
    nir = nir[nir["metric"].astype(str).str.contains("geom", na=False)]
    nir = nir[["session_id", "block_num", "binocular_raw_median"]]
    merged = head.merge(nir, on=["session_id", "block_num"], how="inner")
    print(f"  与 NIR 瞳孔 join 后行数: {len(merged)}，覆盖 session: {merged['session_id'].nunique()}")

    # M0 相关：未校正瞳孔 vs 各头动
    m0 = {}
    for k in HEAD_FIELDS:
        pair = merged[["binocular_raw_median", k]].dropna()
        if len(pair) >= 10:
            r = spearmanr(pair["binocular_raw_median"], pair[k]).correlation
            m0[k] = r
    # M2 残差化：log(pupil) ~ head (每个头动字段单独回归)，残差后 vs head
    m2 = {}
    for k in HEAD_FIELDS:
        pair = merged[["binocular_raw_median", k]].dropna()
        pair = pair[pair["binocular_raw_median"] > 0]
        if len(pair) < 10:
            continue
        lp = np.log(pair["binocular_raw_median"].to_numpy())
        x = pair[k].to_numpy()
        coef = np.polyfit(x, lp, 1)
        resid = lp - np.polyval(coef, x)
        m2[k] = spearmanr(resid, x).correlation

    # 画对比图
    fig, ax = make_figure(width="full", height_cm=4.8, nrows=1, ncols=1)
    keys = [k for k in HEAD_FIELDS if k in m0 and k in m2]
    labels = [HEAD_FIELDS[k] for k in keys]
    y = list(range(len(keys)))[::-1]
    for yi, k in zip(y, keys):
        ax.errorbar(m0[k], yi, fmt="o", color=PTOL[0], markersize=4, label="未校正瞳孔" if yi == y[0] else None)
        ax.errorbar(m2[k], yi, fmt="s", color=PTOL[1], markersize=4, label="RGB 头动矫正后" if yi == y[0] else None)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("瞳孔与头动的 Spearman ρ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=7, frameon=False)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.24, right=0.985, bottom=0.14, top=0.90)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_figure(fig, Path(OUT_DIR) / "fig_headmotion_check", FORMATS)

    print("\n=== 结果：瞳孔与头动 Spearman ρ（未校正 vs RGB 头动矫正后）===")
    for k in HEAD_FIELDS:
        if k in m0 and k in m2:
            print(f"  {HEAD_FIELDS[k]:16s} 未校正 {m0[k]:+.3f}  →  矫正后 {m2[k]:+.3f}")
        elif k in m0:
            print(f"  {HEAD_FIELDS[k]:16s} 未校正 {m0[k]:+.3f}  →  矫正后 N/A")
    print("\n结论：若矫正后 ρ 趋近 0/显著下降，说明 NIR 瞳孔与头动关联可被 RGB 头动解释，非头动造成。")
    print("（注：此为测量学/稳健性佐证，不替代主分析，不冻结任何校正公式。）")

    # 保存原始表格（进 Git，供报告/审计）
    out_root = r"D:/Project/厚粲杯/08_算法/FocusWave-Formal-Analysis/正式报告/附录数据/头动"
    os.makedirs(out_root, exist_ok=True)
    merged[["session_id", "block_num", "binocular_raw_median", *HEAD_FIELDS]].to_csv(
        os.path.join(out_root, "nir_pupil_headmotion_raw.csv"), index=False)
    res = pd.DataFrame({
        "headmotion": [HEAD_FIELDS[k] for k in HEAD_FIELDS if k in m0 or k in m2],
        "m0_uncorrected_rho": [m0.get(k, np.nan) for k in HEAD_FIELDS if k in m0 or k in m2],
        "m2_rgb_corrected_rho": [m2.get(k, np.nan) for k in HEAD_FIELDS if k in m0 or k in m2],
    })
    res.to_csv(os.path.join(out_root, "nir_pupil_headmotion_rho.csv"), index=False)
    print(f"已保存原始表格到 {out_root}")


if __name__ == "__main__":
    main()
