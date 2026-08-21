from __future__ import annotations

import math

import cv2
import numpy as np

from ..contracts import EYE_LEFT, EYE_RIGHT


EYE_CORNERS = {
    EYE_RIGHT: (33, 133),
    EYE_LEFT: (362, 263),
}


def normalized_eye_roi(
    image: np.ndarray,
    points_xy: np.ndarray,
    corner_indices: tuple[int, int],
    output_size: tuple[int, int] = (320, 160),
    corner_span_fraction: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    out_w, out_h = map(int, output_size)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy 必须为 N×2 像素坐标")
    if out_w <= 0 or out_h <= 0 or not 0 < corner_span_fraction < 1:
        raise ValueError("ROI 参数无效")
    p0 = points_xy[corner_indices[0]].astype(np.float32)
    p1 = points_xy[corner_indices[1]].astype(np.float32)
    delta = p1 - p0
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance < 8:
        return None, None, math.nan
    perpendicular = np.array([-delta[1], delta[0]], dtype=np.float32) / distance
    p2 = 0.5 * (p0 + p1) + perpendicular * (0.5 * distance)
    target_distance = out_w * corner_span_fraction
    q0 = np.array([(out_w - target_distance) / 2, out_h / 2], dtype=np.float32)
    q1 = np.array([(out_w + target_distance) / 2, out_h / 2], dtype=np.float32)
    q2 = np.array([out_w / 2, out_h / 2 + target_distance / 2], dtype=np.float32)
    affine = cv2.getAffineTransform(np.vstack([p0, p1, p2]), np.vstack([q0, q1, q2]))
    roi = cv2.warpAffine(
        image, affine, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return np.ascontiguousarray(roi), affine, distance


def transform_points(points_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return cv2.transform(np.asarray(points_xy, dtype=np.float32)[None, :, :], affine)[0]


def inverse_affine(affine: np.ndarray) -> np.ndarray:
    """返回仿射矩阵的逆（2×3），与 transform_points 配套做源图↔ROI 坐标往返。"""
    return cv2.invertAffineTransform(np.asarray(affine, dtype=np.float64).reshape(2, 3))


def map_ellipse_to_source(
    center, major_endpoint, minor_endpoint, inverse_aff: np.ndarray
) -> dict:
    """把 ROI 空间的三点椭圆经逆仿射映射回源帧系。

    固定眼角 ROI 的仿射是相似变换（旋转＋均匀缩放＋平移），保椭圆，
    因此对三点逐一逆变换后重建椭圆即可，返回源图系椭圆参数字典。
    """
    src_center = transform_points(np.asarray([center], dtype=np.float32), inverse_aff)[0]
    src_major = transform_points(np.asarray([major_endpoint], dtype=np.float32), inverse_aff)[0]
    src_minor = transform_points(np.asarray([minor_endpoint], dtype=np.float32), inverse_aff)[0]
    return ellipse_from_three_points(src_center, src_major, src_minor)


def roi_border_status(
    image_shape: tuple[int, int],
    affine: np.ndarray,
    output_size: tuple[int, int] = (320, 160),
    border_margin_px: int = 0,
) -> str:
    """把 ROI 四角经逆仿射映射回源图，检查是否越界/贴边。

    任一角落在图外或距图边不足 border_margin_px → "border_heavy"，否则 "ready"。
    调用方在 affine 为 None 或 ROI 退化时自行按 missing/degenerate 处理。
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    out_w, out_h = int(output_size[0]), int(output_size[1])
    inverse = inverse_affine(affine)
    corners = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    source_corners = transform_points(corners, inverse)
    inside = (
        (source_corners[:, 0] >= border_margin_px)
        & (source_corners[:, 0] <= width - 1 - border_margin_px)
        & (source_corners[:, 1] >= border_margin_px)
        & (source_corners[:, 1] <= height - 1 - border_margin_px)
    )
    return "ready" if bool(inside.all()) else "border_heavy"


def ellipse_to_poly(center, axes_half, angle_deg: float, points: int = 64) -> np.ndarray:
    """椭圆 → 多边形顶点（N×2，int）。

    axes_half = (半长轴, 半短轴)；angle_deg 与 cv2.ellipse 约定一致。
    用于把真值/检测椭圆近似为凸多边形后求 IoU。
    """
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    axes = (int(round(abs(float(axes_half[0])))), int(round(abs(float(axes_half[1])))))
    if axes[0] <= 0 or axes[1] <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    return cv2.ellipse2Poly((cx, cy), axes, int(round(float(angle_deg))) % 360, 0, 360, points)


def ellipse_iou(truth: dict, detected: dict, points: int = 64) -> float:
    """两椭圆的 IoU。truth/detected 均为含 center_x/center_y/major_diameter/minor_diameter/angle_deg 的 dict。"""
    def polygon(ellipse: dict) -> np.ndarray:
        return ellipse_to_poly(
            (ellipse["center_x"], ellipse["center_y"]),
            (ellipse["major_diameter"] / 2, ellipse["minor_diameter"] / 2),
            ellipse["angle_deg"],
            points,
        )
    poly_truth = polygon(truth)
    poly_detected = polygon(detected)
    if len(poly_truth) < 3 or len(poly_detected) < 3:
        return 0.0
    to_f32 = lambda poly: poly.reshape(-1, 1, 2).astype(np.float32)
    intersection_area, intersection = cv2.intersectConvexConvex(to_f32(poly_truth), to_f32(poly_detected))
    area_intersection = float(intersection_area) if intersection is not None and len(intersection) else 0.0
    area_truth = float(cv2.contourArea(to_f32(poly_truth)))
    area_detected = float(cv2.contourArea(to_f32(poly_detected)))
    union = area_truth + area_detected - area_intersection
    if union <= 0:
        return 0.0
    return float(area_intersection / union)


def ellipse_from_three_points(center, major_endpoint, minor_endpoint) -> dict:
    center = np.asarray(center, dtype=float)
    major = np.asarray(major_endpoint, dtype=float) - center
    minor = np.asarray(minor_endpoint, dtype=float) - center
    major_radius = float(np.linalg.norm(major))
    minor_radius = float(np.linalg.norm(minor))
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("长短轴端点必须与中心不同")
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "major_diameter": 2 * major_radius,
        "minor_diameter": 2 * minor_radius,
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
        "equivalent_diameter": 2 * math.sqrt(major_radius * minor_radius),
    }

