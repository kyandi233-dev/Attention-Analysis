"""阶段4：六算法单帧基准评估（主环境 3.13）。

流程：读 ground_truth_528 + review_manifest → 生成可运行样本清单 → subprocess
调 venv-pupil 的 scripts/nir_detect_batch.py 产出 detections.csv → 评估三种率
（端到端 / 算法层分开）→ 图表 → benchmark_report.md。

指标口径：
- returned = 算法返回椭圆且过几何合理性门（中心在画内、3≤minor≤major≤0.65·min(h,w)、aspect≥0.25）
- usable   = returned 且对真值椭圆过 usable_fit（IoU、中心误差、等径相对误差）
- face_failure 20 眼（无 ROI）只进端到端分母，不进算法层
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import Config
from ..metadata import run_metadata, source_id
from .roi import ellipse_iou

GEOMETRY_IMAGE_SIZE = (320, 160)
GEOMETRY_MIN_AXIS = 3.0
GEOMETRY_MAX_AXIS_FRACTION = 0.65
GEOMETRY_MIN_ASPECT = 0.25


# ---------- 纯函数（可测） ----------

def geometry_plausible(
    center, major_diameter, minor_diameter,
    image_size=GEOMETRY_IMAGE_SIZE, min_axis=GEOMETRY_MIN_AXIS,
    max_axis_fraction=GEOMETRY_MAX_AXIS_FRACTION, min_aspect=GEOMETRY_MIN_ASPECT,
) -> bool:
    """算法返回的椭圆是否"几何合理"（返回质量门，v1 口径）。"""
    width, height = image_size
    cx, cy = center
    if not (0 <= cx < width and 0 <= cy < height):
        return False
    major = abs(float(major_diameter))
    minor = abs(float(minor_diameter))
    if major < minor:
        major, minor = minor, major
    if minor < min_axis or major > max_axis_fraction * min(width, height):
        return False
    if major <= 0 or minor / major < min_aspect:
        return False
    return True


def equivalent_diameter(major_diameter: float, minor_diameter: float) -> float:
    """等效直径 = sqrt(长轴直径 × 短轴直径)；与 ground_truth 三点椭圆口径一致。"""
    return math.sqrt(max(float(major_diameter), 0.0) * max(float(minor_diameter), 0.0))


def usable_fit(
    truth: dict, detected: dict,
    iou_min: float, center_error_max_px: float, diameter_relative_error_max: float,
) -> bool:
    """检测椭圆是否可视为真值椭圆的可用拟合。detected 需含 center/axes/angle。"""
    if ellipse_iou(truth, detected) < iou_min:
        return False
    center_error = math.hypot(
        detected["center_x"] - truth["center_x"],
        detected["center_y"] - truth["center_y"],
    )
    if center_error > center_error_max_px:
        return False
    truth_eq = equivalent_diameter(truth["major_diameter"], truth["minor_diameter"])
    if truth_eq <= 0:
        return False
    relative_error = abs(
        equivalent_diameter(detected["major_diameter"], detected["minor_diameter"]) - truth_eq
    ) / truth_eq
    return relative_error <= diameter_relative_error_max


def rate_metrics(
    det: pd.DataFrame,
    truth_lookup: dict,
    visible_truth_ids: set,
    visible_all_ids: set,
    invisible_roi_ids: set,
    invisible_all_ids: set,
    subject_of: dict,
    thresholds: dict,
    photometric_threshold: float | None = None,
) -> dict:
    """对一个 (算法×预处理) 计算全部率与门控。det 为该组合的检测长表。

    photometric_threshold 非 None 时，returned 判定额外要求 photometric_contrast>阈值
    （缺失/NaN → 不通过；阈值固定取自 v1，不在 sweep 内调）。
    """
    truth_rows = det[det["sample_id"].isin(visible_truth_ids)].copy()

    def _returned_ok(row) -> bool:
        if int(row["returned"]) != 1:
            return False
        if not geometry_plausible((row["center_x"], row["center_y"]), row["major_diameter"], row["minor_diameter"]):
            return False
        if photometric_threshold is not None:
            if "photometric_contrast" not in det.columns:
                return False
            try:
                if not (float(row["photometric_contrast"]) > photometric_threshold):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    returned_mask = truth_rows.apply(_returned_ok, axis=1)
    usable_mask = pd.Series(False, index=truth_rows.index)
    returned_rows = truth_rows[returned_mask]
    if len(returned_rows):
        usable_mask.loc[returned_rows.index] = returned_rows.apply(
            lambda r: usable_fit(
                truth_lookup[r["sample_id"]],
                {
                    "center_x": r["center_x"], "center_y": r["center_y"],
                    "major_diameter": r["major_diameter"], "minor_diameter": r["minor_diameter"],
                    "angle_deg": r["angle_deg"],
                },
                thresholds["iou_min"], thresholds["center_error_max_px"], thresholds["diameter_relative_error_max"],
            ),
            axis=1,
        )
    returned_n = int(returned_mask.sum())
    usable_n = int(usable_mask.sum())
    algorithm_layer_visible = len(visible_truth_ids)
    end_to_end_visible = len(visible_all_ids)
    wrong_among_returned = (returned_n - usable_n) / returned_n if returned_n else math.nan

    inv_rows = det[det["sample_id"].isin(invisible_roi_ids)]
    fp_mask = inv_rows.apply(_returned_ok, axis=1)
    fp_n = int(fp_mask.sum())
    fp_rate_algorithm = fp_n / len(invisible_roi_ids) if invisible_roi_ids else math.nan
    fp_rate_end_to_end = fp_n / len(invisible_all_ids) if invisible_all_ids else math.nan

    per_subject = {}
    for subject in sorted({subject_of[s] for s in visible_truth_ids}):
        sub_ids = {s for s in visible_truth_ids if subject_of[s] == subject}
        sub_usable = int(usable_mask[truth_rows["sample_id"].isin(sub_ids)].sum())
        per_subject[subject] = sub_usable / len(sub_ids)

    error_rows = []
    for _, r in truth_rows[returned_mask].iterrows():
        truth = truth_lookup[r["sample_id"]]
        det_ellipse = {
            "center_x": r["center_x"], "center_y": r["center_y"],
            "major_diameter": r["major_diameter"], "minor_diameter": r["minor_diameter"],
            "angle_deg": r["angle_deg"],
        }
        truth_eq = equivalent_diameter(truth["major_diameter"], truth["minor_diameter"])
        error_rows.append({
            "sample_id": r["sample_id"],
            "iou": ellipse_iou(truth, det_ellipse),
            "center_error_px": math.hypot(det_ellipse["center_x"] - truth["center_x"], det_ellipse["center_y"] - truth["center_y"]),
            "diameter_relative_error": abs(equivalent_diameter(det_ellipse["major_diameter"], det_ellipse["minor_diameter"]) - truth_eq) / truth_eq if truth_eq > 0 else math.nan,
        })
    error_df = pd.DataFrame(error_rows)

    return {
        "returned_n": returned_n,
        "usable_n": usable_n,
        "algorithm_layer_visible_rate": usable_n / algorithm_layer_visible if algorithm_layer_visible else math.nan,
        "end_to_end_visible_rate": usable_n / end_to_end_visible if end_to_end_visible else math.nan,
        "wrong_among_returned_rate": wrong_among_returned,
        "fp_rate_algorithm_layer": fp_rate_algorithm,
        "fp_rate_end_to_end": fp_rate_end_to_end,
        "median_iou": float(error_df["iou"].median()) if len(error_df) else math.nan,
        "median_center_error_px": float(error_df["center_error_px"].median()) if len(error_df) else math.nan,
        "median_diameter_relative_error": float(error_df["diameter_relative_error"].median()) if len(error_df) else math.nan,
        "subjects_passing_usable": int(sum(v >= thresholds["per_subject_usable_min"] for v in per_subject.values())),
        "per_subject_usable": per_subject,
    }


# ---------- 数据装配 ----------

def load_benchmark_inputs(config: Config, review_tag: str = "20260814") -> pd.DataFrame:
    """合并 ground_truth_528 + review_manifest 的 roi_path，重命名真值椭圆列为 truth_*。"""
    review_dir = config.path_value("truth_artifact_root")
    gt = pd.read_csv(review_dir / "ground_truth_528.csv")
    man = pd.read_csv(review_dir / "review_manifest.csv")
    merged = gt.merge(man[["sample_id", "roi_path", "subject"]], on="sample_id", how="left")
    # review_manifest 的 roi_path 相对 review 包目录，统一转绝对路径（适配器/图册/评估共用）
    merged["roi_path"] = merged["roi_path"].map(
        lambda p: str((review_dir / p).resolve()) if isinstance(p, str) and p else ""
    )
    return merged.rename(columns={
        "center_x": "truth_center_x", "center_y": "truth_center_y",
        "major_x": "truth_major_x", "major_y": "truth_major_y",
        "minor_x": "truth_minor_x", "minor_y": "truth_minor_y",
        "major_diameter": "truth_major_diameter", "minor_diameter": "truth_minor_diameter",
        "angle_deg": "truth_angle_deg", "equivalent_diameter": "truth_equivalent_diameter",
    })


def runnable_eyes(merged: pd.DataFrame) -> pd.DataFrame:
    """有 ROI 且 face 层成功的眼（算法可运行集）。"""
    return merged[merged["roi_path"].notna() & merged["roi_path"].ne("")].copy()


def resolve_benchmark_dir(config: Config, kind: str = "default", force: bool = False) -> Path:
    """基准目录：artifacts/benchmark-single/<kind>/（default=off-the-shelf，tuned=参数调优）。"""
    output = config.path_value("benchmark_artifact_root") / kind
    if output.exists() and not force:
        raise RuntimeError(f"基准目录已存在: {output}（如需覆盖请显式加 --force）")
    return output


def write_manifest(runnable: pd.DataFrame, path: Path) -> None:
    runnable[["sample_id", "roi_path"]].to_csv(path, index=False, encoding="utf-8")


def run_detection_subprocess(
    config: Config,
    manifest: Path,
    out: Path,
    limit: int = 0,
    algorithms: list[str] | None = None,
    preprocessing: list[str] | None = None,
    params: dict | None = None,
    config_name: str = "default",
) -> None:
    """调用 venv-pupil 适配器；输出 detections.csv。默认用配置的算法/预处理；参数调优时覆盖。"""
    adapter = Path(__file__).resolve().parents[3] / "scripts" / "nir_detect_batch.py"
    python = config.section("runtimes")["pypupilext_python"]
    algorithms = algorithms or list(config.section("nir")["pypupilext_algorithms"])
    preprocessing = preprocessing or list(config.section("nir")["preprocessing"])
    cmd = [
        str(python), str(adapter),
        "--manifest", str(manifest),
        "--out", str(out),
        "--algorithms", ",".join(algorithms),
        "--preprocessing", ",".join(preprocessing),
        "--config-name", config_name,
    ]
    if params:
        # 适配器 --params 期望 {algorithm: {属性: 值}}；sweep 每次只跑一个算法，按该算法键嵌套
        cmd += ["--params", json.dumps({algorithms[0]: params}, ensure_ascii=False)]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(f"检测适配器失败:\n{completed.stderr[-2000:]}")


def run_benchmark(config: Config, tag: str | None, smoke: bool, force: bool = False) -> dict:
    """阶段4 检测阶段：建清单 + subprocess 跑适配器。"""
    merged = load_benchmark_inputs(config)
    runnable = runnable_eyes(merged).copy()
    review_dir = config.path_value("truth_artifact_root")
    output = resolve_benchmark_dir(config, "default", force)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.csv"
    write_manifest(runnable, manifest)
    smoke_eyes = int(config.section("nir")["benchmark"].get("smoke_eyes", 12))
    limit = smoke_eyes if smoke else 0
    detections_path = output / "detections.csv"
    run_detection_subprocess(config, manifest, detections_path, limit)
    review_dir = config.path_value("truth_artifact_root")
    metadata = run_metadata(config, sources=[
        review_dir / "ground_truth_528.csv",
        review_dir / "review_manifest.csv",
        Path(__file__).resolve().parents[3] / "scripts" / "nir_detect_batch.py",
    ])
    (output / "benchmark_manifest.json").write_text(
        json.dumps({**metadata, "smoke": smoke, "limit": limit, "runnable_eyes": len(runnable)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output_dir": str(output), "detections": str(detections_path), "runnable_eyes": len(runnable)}


def load_detections(config: Config, kind: str = "default") -> pd.DataFrame:
    output = config.path_value("benchmark_artifact_root") / kind
    return pd.read_csv(output / "detections.csv")


# ---------- 评估聚合 ----------

def admission_thresholds(config: Config, corner_span_px: int = 160) -> dict:
    nir = config.section("nir")
    usable = nir["usable_fit"]
    admission = nir["admission"]
    return {
        "iou_min": float(usable["ellipse_iou_min"]),
        "center_error_max_px": float(usable["center_error_corner_span_max"]) * corner_span_px,
        "diameter_relative_error_max": float(usable["equivalent_diameter_relative_error_max"]),
        "visible_end_to_end_rate_min": float(admission["visible_end_to_end_usable_rate_min"]),
        "invisible_fp_rate_max": float(admission["invisible_false_positive_rate_max"]),
        "wrong_among_returned_max": float(admission["wrong_among_returned_rate_max"]),
        "subjects_passing_min": int(admission["subjects_passing_min"]),
        "per_subject_usable_min": float(admission["per_subject_usable_rate_min"]),
        "per_subject_invisible_fp_max": float(admission["per_subject_invisible_false_positive_rate_max"]),
    }


def evaluation_context(config: Config) -> dict:
    """构建评估所需的眼集合与真值查找表。"""
    merged = load_benchmark_inputs(config)
    thresholds = admission_thresholds(config)
    visible = merged[merged["visibility"] == "可见"]
    visible_all_ids = set(visible["sample_id"])
    visible_truth = visible[
        visible["roi_path"].notna() & visible["roi_path"].ne("")
        & visible["truth_center_x"].notna()
    ]
    visible_truth_ids = set(visible_truth["sample_id"])
    invisible = merged[merged["visibility"] == "不可见"]
    invisible_all_ids = set(invisible["sample_id"])
    invisible_roi = invisible[invisible["roi_path"].notna() & invisible["roi_path"].ne("")]
    invisible_roi_ids = set(invisible_roi["sample_id"])
    subject_of = dict(zip(merged["sample_id"], merged["subject"]))
    truth_lookup = {}
    for _, r in visible_truth.iterrows():
        truth_lookup[r["sample_id"]] = {
            "center_x": r["truth_center_x"], "center_y": r["truth_center_y"],
            "major_diameter": r["truth_major_diameter"], "minor_diameter": r["truth_minor_diameter"],
            "angle_deg": r["truth_angle_deg"],
        }
    return {
        "merged": merged,
        "thresholds": thresholds,
        "visible_truth_ids": visible_truth_ids,
        "visible_all_ids": visible_all_ids,
        "invisible_roi_ids": invisible_roi_ids,
        "invisible_all_ids": invisible_all_ids,
        "subject_of": subject_of,
        "truth_lookup": truth_lookup,
    }


def evaluate_all(config: Config, tag: str) -> tuple[pd.DataFrame, dict]:
    """对每个 (算法×预处理) 计算指标 + 门控判定，返回汇总表与上下文。"""
    ctx = evaluation_context(config)
    merged = ctx["merged"]
    thresholds = ctx["thresholds"]
    det = load_detections(config, "default")
    visible_truth_ids = ctx["visible_truth_ids"]
    visible_all_ids = ctx["visible_all_ids"]
    invisible_roi_ids = ctx["invisible_roi_ids"]
    invisible_all_ids = ctx["invisible_all_ids"]
    subject_of = ctx["subject_of"]
    truth_lookup = ctx["truth_lookup"]

    photo = float(config.section("nir")["benchmark"].get("photometric_threshold", 0.02))
    summary_rows = []
    for (algorithm, preprocessing), group in det.groupby(["algorithm", "preprocessing"]):
        ungated = rate_metrics(
            group, truth_lookup,
            visible_truth_ids, visible_all_ids,
            invisible_roi_ids, invisible_all_ids,
            subject_of, thresholds,
            photometric_threshold=None,
        )
        metrics = rate_metrics(
            group, truth_lookup,
            visible_truth_ids, visible_all_ids,
            invisible_roi_ids, invisible_all_ids,
            subject_of, thresholds,
            photometric_threshold=photo,
        )
        passed = (
            metrics["end_to_end_visible_rate"] >= thresholds["visible_end_to_end_rate_min"]
            and metrics["fp_rate_end_to_end"] <= thresholds["invisible_fp_rate_max"]
            and (pd.isna(metrics["wrong_among_returned_rate"]) or metrics["wrong_among_returned_rate"] <= thresholds["wrong_among_returned_max"])
            and metrics["subjects_passing_usable"] >= thresholds["subjects_passing_min"]
        )
        summary_rows.append({
            "algorithm": algorithm, "preprocessing": preprocessing,
            "returned_n": metrics["returned_n"], "usable_n": metrics["usable_n"],
            "visible_truth_n": len(visible_truth_ids), "visible_all_n": len(visible_all_ids),
            "algorithm_layer_visible_rate": metrics["algorithm_layer_visible_rate"],
            "end_to_end_visible_rate": metrics["end_to_end_visible_rate"],
            "wrong_among_returned_rate": metrics["wrong_among_returned_rate"],
            "fp_rate_algorithm_layer": metrics["fp_rate_algorithm_layer"],
            "fp_rate_end_to_end": metrics["fp_rate_end_to_end"],
            "median_iou": metrics["median_iou"],
            "median_center_error_px": metrics["median_center_error_px"],
            "median_diameter_relative_error": metrics["median_diameter_relative_error"],
            "subjects_passing_usable": metrics["subjects_passing_usable"],
            "ungated_end_to_end_visible_rate": ungated["end_to_end_visible_rate"],
            "ungated_algorithm_layer_visible_rate": ungated["algorithm_layer_visible_rate"],
            "ungated_fp_rate_end_to_end": ungated["fp_rate_end_to_end"],
            "photometric_threshold": photo,
            "passed_admission": passed,
        })
    summary = pd.DataFrame(summary_rows).sort_values(["algorithm", "preprocessing"]).reset_index(drop=True)
    context = {
        "thresholds": thresholds,
        "visible_truth_n": len(visible_truth_ids),
        "visible_all_n": len(visible_all_ids),
        "invisible_roi_n": len(invisible_roi_ids),
        "invisible_all_n": len(invisible_all_ids),
    }
    return summary, context


# ---------- 图表与报告 ----------

def _setup_cjk_font() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_rates(output: Path, summary: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_cjk_font()

    labels = [f"{r['algorithm']}\n{r['preprocessing']}" for _, r in summary.iterrows()]
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, summary["end_to_end_visible_rate"], width, label="可见端到端可用率")
    ax.bar(x, summary["fp_rate_end_to_end"], width, label="不可见误检率")
    ax.bar(x + width, summary["wrong_among_returned_rate"], width, label="返回中错误率")
    th = summary["end_to_end_visible_rate"].iloc[0]  # placeholder
    ax.axhline(0.85, color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.05, color="tab:orange", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.05, color="tab:green", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("比例")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("六算法 × {raw, CLAHE}：三种率与门槛（虚线=门槛）")
    fig.tight_layout()
    path = output / "plots" / "01_rates.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_subject_matrix(output: Path, det: pd.DataFrame, merged: pd.DataFrame, algorithms: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_cjk_font()

    # 每算法 raw 的逐被试可用率
    visible_truth = merged[
        (merged["visibility"] == "可见")
        & merged["roi_path"].notna() & merged["roi_path"].ne("")
        & merged["truth_center_x"].notna()
    ]
    truth_lookup = {
        r.sample_id: {
            "center_x": r.truth_center_x, "center_y": r.truth_center_y,
            "major_diameter": r.truth_major_diameter, "minor_diameter": r.truth_minor_diameter,
            "angle_deg": r.truth_angle_deg,
        }
        for r in visible_truth.itertuples()
    }
    thresholds = {"iou_min": 0.7, "center_error_max_px": 16, "diameter_relative_error_max": 0.2}
    matrix = {}
    for algorithm in algorithms:
        sub = det[(det["algorithm"] == algorithm) & (det["preprocessing"] == "raw")]
        rates = {}
        for subject, group_ids in visible_truth.groupby("subject"):
            ids = set(group_ids["sample_id"])
            rows = sub[sub["sample_id"].isin(ids)]
            returned = rows["returned"].astype(int).eq(1) & rows.apply(
                lambda r: geometry_plausible((r["center_x"], r["center_y"]), r["major_diameter"], r["minor_diameter"]), axis=1
            )
            usable = 0
            returned_rows = rows[returned]
            if len(returned_rows):
                usable = int(returned_rows.apply(
                    lambda r: usable_fit(truth_lookup[r.sample_id], {
                        "center_x": r.center_x, "center_y": r.center_y,
                        "major_diameter": r.major_diameter, "minor_diameter": r.minor_diameter,
                        "angle_deg": r.angle_deg,
                    }, thresholds["iou_min"], thresholds["center_error_max_px"], thresholds["diameter_relative_error_max"]),
                    axis=1,
                ).sum())
            rates[subject] = usable / len(ids)
        matrix[algorithm] = rates
    frame = pd.DataFrame(matrix).T  # 行=算法，列=被试
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(frame.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(frame.shape[1]), frame.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(frame.shape[0]), frame.index, fontsize=8)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            v = frame.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.55 else "black")
    ax.set_title("逐被试可见眼可用率（raw，门槛≥0.75）")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    path = output / "plots" / "02_per_subject_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_failure_gallery(output: Path, det: pd.DataFrame, merged: pd.DataFrame, algorithm: str, preprocessing: str) -> Path:
    """对选定算法：抽 可见且(可用/错/漏) 各若干眼，叠加真值(绿)+检测(红) 到 ROI。"""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_cjk_font()

    visible_truth = merged[
        (merged["visibility"] == "可见")
        & merged["roi_path"].notna() & merged["roi_path"].ne("")
        & merged["truth_center_x"].notna()
    ]
    truth_lookup = {
        r.sample_id: {
            "center_x": r.truth_center_x, "center_y": r.truth_center_y,
            "major_diameter": r.truth_major_diameter, "minor_diameter": r.truth_minor_diameter,
            "angle_deg": r.truth_angle_deg,
        }
        for r in visible_truth.itertuples()
    }
    thresholds = {"iou_min": 0.7, "center_error_max_px": 16, "diameter_relative_error_max": 0.2}
    det = det[(det["algorithm"] == algorithm) & (det["preprocessing"] == preprocessing)].set_index("sample_id")

    def draw_overlay(sample_id: str) -> np.ndarray:
        row = merged[merged["sample_id"] == sample_id].iloc[0]
        image = cv2.imdecode(np.fromfile(row["roi_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        truth = truth_lookup[sample_id]
        cv2.ellipse(image, (int(truth["center_x"]), int(truth["center_y"])),
                    (int(truth["major_diameter"] / 2), int(truth["minor_diameter"] / 2)),
                    int(truth["angle_deg"]), 0, 360, (0, 255, 0), 2)
        if sample_id in det.index:
            d = det.loc[sample_id]
            if d["returned"] == 1 and geometry_plausible((d["center_x"], d["center_y"]), d["major_diameter"], d["minor_diameter"]):
                cv2.ellipse(image, (int(d["center_x"]), int(d["center_y"])),
                            (int(d["major_diameter"] / 2), int(d["minor_diameter"] / 2)),
                            int(d["angle_deg"]), 0, 360, (0, 0, 255), 1)
        return image

    shown = []
    for sample_id in visible_truth["sample_id"]:
        d = det.loc[sample_id] if sample_id in det.index else None
        returned = d is not None and d["returned"] == 1 and geometry_plausible((d["center_x"], d["center_y"]), d["major_diameter"], d["minor_diameter"])
        usable = returned and usable_fit(truth_lookup[sample_id], {
            "center_x": d.center_x, "center_y": d.center_y,
            "major_diameter": d.major_diameter, "minor_diameter": d.minor_diameter, "angle_deg": d.angle_deg,
        }, thresholds["iou_min"], thresholds["center_error_max_px"], thresholds["diameter_relative_error_max"])
        kind = "usable" if usable else ("wrong" if returned else "missed")
        shown.append((sample_id, kind))
    fig, axes = plt.subplots(3, 6, figsize=(18, 8))
    for ax in axes.flat:
        ax.axis("off")
    for idx, kind in enumerate(["usable", "wrong", "missed"]):
        picks = [s for s, k in shown if k == kind][: 6]
        for j, sample_id in enumerate(picks):
            ax = axes[idx, j]
            img = draw_overlay(sample_id)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{kind}\n{sample_id}", fontsize=7)
            ax.axis("off")
    fig.suptitle(f"{algorithm} ({preprocessing})：可见眼可用/错/漏 抽样（绿=真值 红=检测）", fontsize=11)
    fig.tight_layout()
    path = output / "plots" / "03_failure_gallery.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_report(config: Config, tag: str, summary: pd.DataFrame, context: dict) -> Path:
    """生成 benchmark_report.md（图文，PNG 在 plots/ 同级）。"""
    output = config.path_value("benchmark_artifact_root") / "default"
    merged = load_benchmark_inputs(config)
    det = load_detections(config, "default")
    algorithms = list(config.section("nir")["pypupilext_algorithms"])
    threshold = context["thresholds"]

    summary.to_csv(output / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    rate_plot = _plot_rates(output, summary)
    best_raw = summary[summary["preprocessing"] == "raw"].sort_values("end_to_end_visible_rate", ascending=False).iloc[0]
    subject_plot = _plot_subject_matrix(output, det, merged, algorithms)
    gallery_plot = _plot_failure_gallery(output, det, merged, best_raw["algorithm"], "raw")

    lines = []
    lines.append(f"# 阶段4｜六算法单帧基准报告")
    lines.append("")
    lines.append(f"> {pd.Timestamp.now().isoformat(timespec='seconds')}（Asia/Shanghai）｜对 528 眼人工真值中的 480 眼（有 ROI）跑六算法 × {{raw, CLAHE}} 单帧检测。")
    lines.append("")
    lines.append("## 门槛")
    lines.append("")
    lines.append(f"| 项 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 可见端到端可用率 | ≥ {threshold['visible_end_to_end_rate_min']} |")
    lines.append(f"| 不可见误检率（端到端） | ≤ {threshold['invisible_fp_rate_max']} |")
    lines.append(f"| 返回中错误率 | ≤ {threshold['wrong_among_returned_max']} |")
    lines.append(f"| 每被试可用率 | ≥ {threshold['per_subject_usable_min']}（≥{threshold['subjects_passing_min']} 被试） |")
    lines.append(f"| usable_fit | IoU≥{threshold['iou_min']} 且 中心误差≤{threshold['center_error_max_px']:.0f}px 且 等径相对误差≤{threshold['diameter_relative_error_max']} |")
    lines.append("")
    lines.append(f"真值：{context['visible_truth_n']} 可见（含真值椭圆）+ {context['invisible_all_n']} 不可见；face_failure 不计算法层、计入端到端分母。")
    lines.append("")
    lines.append("## 三种率（端到端口径）")
    lines.append("")
    lines.append("![三种率](plots/01_rates.png)")
    lines.append("")
    lines.append("| 算法 | 预处理 | 可见端到端可用率 | 不可见误检率 | 返回中错误率 | 中位IoU | 中位中心误差px | 通过门槛 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        wrong = r["wrong_among_returned_rate"]
        wrong_str = f"{wrong:.3f}" if pd.notna(wrong) else "—"
        lines.append(
            f"| {r['algorithm']} | {r['preprocessing']} | {r['end_to_end_visible_rate']:.3f} | "
            f"{r['fp_rate_end_to_end']:.3f} | {wrong_str} | "
            f"{r['median_iou']:.3f} | {r['median_center_error_px']:.1f} | {'✅' if r['passed_admission'] else '❌'} |"
        )
    lines.append("")
    lines.append(f"当前最接近门槛的是 **{best_raw['algorithm']} / raw**（端到端可用率 {best_raw['end_to_end_visible_rate']:.3f}）。")
    lines.append("")
    lines.append("## 逐被试可用率（raw）")
    lines.append("")
    lines.append("![逐被试矩阵](plots/02_per_subject_matrix.png)")
    lines.append("")
    lines.append("## 失败抽样")
    lines.append("")
    lines.append("![失败图册](plots/03_failure_gallery.png)")
    lines.append("")
    lines.append("## 结论与边界")
    lines.append("")
    lines.append("- 本基准为**库默认参数 off-the-shelf** 对比；Swirski2D/PuRe 默认 mm→px 假设面向全帧含眼眦，对 320×160 紧 ROI 可能失配，表现不代表参数调优后上限。")
    lines.append("- 置信度字段已记录未做门控（未验证阈值不入正式）。")
    lines.append("- 单帧模式对 PuReST 不利（追踪器主场在连续序列，见阶段5）。")
    lines.append("")

    report_path = output / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------- 阶段4b：参数调优 sweep ----------

def run_tuned_benchmark(config: Config, tag: str | None, force: bool = False) -> dict:
    """按 tuning 网格逐配置跑检测（每配置只跑其算法，raw），合并成带 config 列的 detections.csv。"""
    if not tag:
        timezone = config.section("pipeline").get("timezone", "Asia/Shanghai")
        tag = datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d")
    merged = load_benchmark_inputs(config)
    runnable = runnable_eyes(merged).copy()
    output = config.path_value("benchmark_artifact_root") / "tuned"
    if output.exists() and not force:
        raise RuntimeError(f"tuned 基准目录已存在: {output}（如需覆盖请显式加 --force）")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.csv"
    write_manifest(runnable, manifest)
    tuning = config.section("nir")["benchmark"]["tuning"]
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for item in tuning:
        algorithm = item["algorithm"]
        cname = item["config"]
        params = item.get("params", {})
        per = config_dir / f"{algorithm}_{cname}.csv"
        run_detection_subprocess(
            config, manifest, per,
            algorithms=[algorithm], preprocessing=["raw"], params=params, config_name=cname,
        )
        frames.append(pd.read_csv(per))
    det = pd.concat(frames, ignore_index=True)
    det.to_csv(output / "detections.csv", index=False, encoding="utf-8-sig")
    review_dir = config.path_value("truth_artifact_root")
    metadata = run_metadata(config, sources=[
        review_dir / "ground_truth_528.csv",
        review_dir / "review_manifest.csv",
        Path(__file__).resolve().parents[3] / "scripts" / "nir_detect_batch.py",
    ])
    (output / "benchmark_manifest.json").write_text(
        json.dumps({**metadata, "tuned": True, "grid": tuning, "runnable_eyes": len(runnable)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output_dir": str(output), "detections": str(output / "detections.csv"), "configs": len(tuning)}


def evaluate_tuned(config: Config, tag: str) -> tuple[pd.DataFrame, dict]:
    """对每个 (算法×参数配置) 计算门控/未门控指标 + admission 判定。"""
    ctx = evaluation_context(config)
    thresholds = ctx["thresholds"]
    photo = float(config.section("nir")["benchmark"].get("photometric_threshold", 0.02))
    output = config.path_value("benchmark_artifact_root") / "tuned"
    det = pd.read_csv(output / "detections.csv")
    rows = []
    for (algorithm, cname), group in det.groupby(["algorithm", "config"]):
        m_ungated = rate_metrics(
            group, ctx["truth_lookup"], ctx["visible_truth_ids"], ctx["visible_all_ids"],
            ctx["invisible_roi_ids"], ctx["invisible_all_ids"], ctx["subject_of"], thresholds,
            photometric_threshold=None,
        )
        m_gated = rate_metrics(
            group, ctx["truth_lookup"], ctx["visible_truth_ids"], ctx["visible_all_ids"],
            ctx["invisible_roi_ids"], ctx["invisible_all_ids"], ctx["subject_of"], thresholds,
            photometric_threshold=photo,
        )
        passed = (
            m_gated["end_to_end_visible_rate"] >= thresholds["visible_end_to_end_rate_min"]
            and m_gated["fp_rate_end_to_end"] <= thresholds["invisible_fp_rate_max"]
            and (pd.isna(m_gated["wrong_among_returned_rate"]) or m_gated["wrong_among_returned_rate"] <= thresholds["wrong_among_returned_max"])
            and m_gated["subjects_passing_usable"] >= thresholds["subjects_passing_min"]
        )
        rows.append({
            "algorithm": algorithm, "config": cname,
            "end_to_end_visible_rate": m_gated["end_to_end_visible_rate"],
            "algorithm_layer_visible_rate": m_gated["algorithm_layer_visible_rate"],
            "fp_rate_end_to_end": m_gated["fp_rate_end_to_end"],
            "wrong_among_returned_rate": m_gated["wrong_among_returned_rate"],
            "returned_n": m_gated["returned_n"], "usable_n": m_gated["usable_n"],
            "median_iou": m_gated["median_iou"], "median_center_error_px": m_gated["median_center_error_px"],
            "subjects_passing_usable": m_gated["subjects_passing_usable"],
            "ungated_end_to_end_visible_rate": m_ungated["end_to_end_visible_rate"],
            "ungated_fp_rate_end_to_end": m_ungated["fp_rate_end_to_end"],
            "passed_admission": passed,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tuned_summary.csv", index=False, encoding="utf-8-sig")
    return summary, {"thresholds": thresholds, "photometric_threshold": photo}


def _plot_sweep(output: Path, summary: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_cjk_font()

    labels = [f"{r['algorithm']}\n{r['config']}" for _, r in summary.iterrows()]
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width, summary["end_to_end_visible_rate"], width, label="可见端到端可用率(门控)")
    ax.bar(x, summary["fp_rate_end_to_end"], width, label="不可见误检率(门控)")
    ax.bar(x + width, summary["wrong_among_returned_rate"], width, label="返回中错误率(门控)")
    ax.axhline(0.85, color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.05, color="tab:orange", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.05, color="tab:green", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("比例")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("参数调优 18 配置：光度门控后三种率（虚线=门槛）")
    fig.tight_layout()
    path = output / "plots" / "01_sweep_rates.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_tuned_report(config: Config, tag: str, summary: pd.DataFrame, context: dict) -> Path:
    output = config.path_value("benchmark_artifact_root") / "tuned"
    thresholds = context["thresholds"]
    photo = context["photometric_threshold"]
    sweep_plot = _plot_sweep(output, summary)
    best = summary.sort_values("end_to_end_visible_rate", ascending=False).iloc[0]
    ctx = evaluation_context(config)
    det = pd.read_csv(output / "detections.csv")
    gallery_plot = _plot_failure_gallery(output, det[det["config"] == best["config"]], ctx["merged"], best["algorithm"], "raw")

    lines = []
    lines.append(f"# 阶段4b｜六算法参数调优报告")
    lines.append("")
    lines.append(f"> {pd.Timestamp.now().isoformat(timespec='seconds')}（Asia/Shanghai）｜18 个预指定参数配置 × 480 眼 × raw；光度门控阈值 {photo} 固定（v1 口径）。")
    lines.append("")
    lines.append("## 门槛")
    lines.append("")
    lines.append(f"| 可见端到端可用率 ≥{thresholds['visible_end_to_end_rate_min']} | 不可见误检率 ≤{thresholds['invisible_fp_rate_max']} | 返回中错误率 ≤{thresholds['wrong_among_returned_max']} | 每被试可用率 ≥{thresholds['per_subject_usable_min']}（≥{thresholds['subjects_passing_min']}被试） |")
    lines.append("")
    lines.append("## 参数网格（预指定，几何失配推理）")
    lines.append("")
    grid = config.section("nir")["benchmark"]["tuning"]
    lines.append("| 算法 | 配置 | 参数 |")
    lines.append("|---|---|---|")
    for item in grid:
        lines.append(f"| {item['algorithm']} | {item['config']} | `{item.get('params', {})}` |")
    lines.append("")
    lines.append("## 18 配置：光度门控后三种率")
    lines.append("")
    lines.append("![sweep 三种率](plots/01_sweep_rates.png)")
    lines.append("")
    lines.append("| 算法 | 配置 | 端到端可用率(门控) | 未门控可用率 | 误检(门控) | 未门控误检 | 返回中错误 | 通过 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        wrong = r["wrong_among_returned_rate"]
        wrong_str = f"{wrong:.3f}" if pd.notna(wrong) else "—"
        lines.append(
            f"| {r['algorithm']} | {r['config']} | {r['end_to_end_visible_rate']:.3f} | {r['ungated_end_to_end_visible_rate']:.3f} | "
            f"{r['fp_rate_end_to_end']:.3f} | {r['ungated_fp_rate_end_to_end']:.3f} | {wrong_str} | "
            f"{'✅' if r['passed_admission'] else '❌'} |"
        )
    lines.append("")
    lines.append(f"当前最好：**{best['algorithm']}/{best['config']}**（门控后端到端可用率 {best['end_to_end_visible_rate']:.3f}）。")
    lines.append("")
    lines.append("## 最优配置失败抽样")
    lines.append("")
    lines.append("![失败图册](plots/02_failure_gallery.png)")
    lines.append("")
    lines.append("## 结论与边界")
    lines.append("")
    lines.append(f"- 光度门控固定阈值 {photo}，**不在 sweep 内调**（避免测试集过拟合）。")
    lines.append("- 参数候选预指定（默认值 vs 320×160 紧 ROI 几何失配），非从结果反推。")
    lines.append("- 若仍无达标：失配不止参数层（可能需换 ROI/预处理/门控结构），如实报告，不强选。")
    lines.append("")

    report_path = output / "benchmark_report_tuned.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


