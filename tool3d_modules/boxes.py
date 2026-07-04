# -*- coding: utf-8 -*-
"""三维框模块，负责框生成、几何计算、密度统计、质量过滤和重叠抑制。"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def filter_points_by_percentile(
    points: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """按坐标分位数过滤极端点，降低飞点对三维框的影响。"""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 10:
        return points

    low = np.percentile(points, lower_percentile, axis=0)
    high = np.percentile(points, upper_percentile, axis=0)

    keep = np.all((points >= low) & (points <= high), axis=1)

    if keep.sum() < 5:
        return points

    filtered = points[keep]

    if filtered.shape[0] < 20:
        return filtered

    center = np.median(filtered, axis=0)
    distances = np.linalg.norm(filtered - center, axis=1)
    distance_threshold = np.percentile(distances, upper_percentile)
    distance_keep = distances <= distance_threshold

    if int(distance_keep.sum()) < max(5, int(filtered.shape[0] * 0.70)):
        return filtered

    return filtered[distance_keep]


def compute_aabb_3d(points: np.ndarray) -> Dict:
    """根据点集计算轴对齐三维框。"""
    points = np.asarray(points, dtype=np.float64)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    center = (min_xyz + max_xyz) / 2.0
    size = max_xyz - min_xyz

    xmin, ymin, zmin = min_xyz
    xmax, ymax, zmax = max_xyz

    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)

    return {
        "box_type": "AABB",
        "center": center,
        "size": size,
        "heading_angle": 0.0,
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "corners": corners,
    }


def compute_pca_obb_3d(
    points: np.ndarray,
    up_axis: int = 2
) -> Dict:
    """根据点集在水平面上估计主方向三维有向框。"""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 3:
        return compute_aabb_3d(points)

    if up_axis not in [0, 1, 2]:
        raise ValueError("up_axis 只能是 0, 1, 2")

    horizontal_axes = [i for i in range(3) if i != up_axis]

    xy = points[:, horizontal_axes]
    z = points[:, up_axis]

    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean

    cov = np.cov(xy_centered.T)

    if not np.isfinite(cov).all():
        return compute_aabb_3d(points)

    eig_vals, eig_vecs = np.linalg.eigh(cov)

    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]

    # 保证局部坐标系方向稳定
    if np.linalg.det(eig_vecs) < 0:
        eig_vecs[:, 1] *= -1

    local_xy = xy_centered @ eig_vecs

    min_local = local_xy.min(axis=0)
    max_local = local_xy.max(axis=0)

    center_local = (min_local + max_local) / 2.0
    size_local = max_local - min_local

    center_xy = xy_mean + center_local @ eig_vecs.T

    zmin = z.min()
    zmax = z.max()
    center_z = (zmin + zmax) / 2.0
    height = zmax - zmin

    center = np.zeros(3, dtype=np.float64)
    center[horizontal_axes] = center_xy
    center[up_axis] = center_z

    # 第一主方向
    main_dir = eig_vecs[:, 0]

    # heading_angle 只在默认水平轴为 x-y 时有明确意义
    heading_angle = math.atan2(main_dir[1], main_dir[0])

    # size 的语义：
    # [length, width, height]
    size = np.array([
        size_local[0],
        size_local[1],
        height
    ], dtype=np.float64)

    # 计算 8 个角点
    lx_min, ly_min = min_local
    lx_max, ly_max = max_local

    local_corners_2d = np.array([
        [lx_min, ly_min],
        [lx_max, ly_min],
        [lx_max, ly_max],
        [lx_min, ly_max],
    ], dtype=np.float64)

    world_corners_2d = xy_mean + local_corners_2d @ eig_vecs.T

    corners = []

    for zz in [zmin, zmax]:
        for p2d in world_corners_2d:
            p3d = np.zeros(3, dtype=np.float64)
            p3d[horizontal_axes] = p2d
            p3d[up_axis] = zz
            corners.append(p3d)

    corners = np.asarray(corners, dtype=np.float64)

    return {
        "box_type": "PCA_OBB",
        "center": center,
        "size": size,
        "heading_angle": float(heading_angle),
        "horizontal_axes": horizontal_axes,
        "up_axis": int(up_axis),
        "main_direction_2d": main_dir,
        "corners": corners,
    }


def build_3d_boxes_from_segments(
    points3d: np.ndarray,
    segments: List[Dict],
    box_type: str = "pca",
    min_points: int = 30,
    up_axis: int = 2,
    use_percentile_filter: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> List[Dict]:
    """根据三维实例分割结果生成三维框列表。"""
    if box_type not in ["aabb", "pca"]:
        raise ValueError("box_type 只能是 'aabb' 或 'pca'")

    points3d = np.asarray(points3d, dtype=np.float64)

    boxes = []

    for seg in segments:
        point_indices = np.asarray(seg["point_indices"], dtype=np.int64)

        if point_indices.size < min_points:
            continue

        instance_points = points3d[point_indices]

        finite = np.isfinite(instance_points).all(axis=1)
        instance_points = instance_points[finite]

        if instance_points.shape[0] < min_points:
            continue

        raw_num_points = int(instance_points.shape[0])

        if use_percentile_filter:
            box_points = filter_points_by_percentile(
                instance_points,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile
            )
        else:
            box_points = instance_points

        if box_points.shape[0] < min_points:
            box_points = instance_points

        if box_type == "aabb":
            box = compute_aabb_3d(box_points)
        else:
            box = compute_pca_obb_3d(box_points, up_axis=up_axis)

        box.update({
            "instance_id": int(seg["instance_id"]),
            "class_name": str(seg["class_name"]),
            "class_id": int(seg["class_id"]),
            "num_points": int(box_points.shape[0]),
            "raw_num_points": raw_num_points,
            "mask_path": seg.get("mask_path", None),
        })

        boxes.append(box)

    return boxes


def get_box_enclosing_aabb(box: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """把任意三维框转换为外接轴对齐框。"""
    if "min_xyz" in box and "max_xyz" in box:
        min_xyz = np.asarray(box["min_xyz"], dtype=np.float64).reshape(3)
        max_xyz = np.asarray(box["max_xyz"], dtype=np.float64).reshape(3)
        return min_xyz, max_xyz

    if "corners" in box:
        corners = np.asarray(box["corners"], dtype=np.float64).reshape(-1, 3)
        return corners.min(axis=0), corners.max(axis=0)

    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size = np.asarray(box["size"], dtype=np.float64).reshape(3)
    half = np.abs(size) / 2.0

    return center - half, center + half


def compute_box_volume_from_size(box: Dict, eps: float = 1e-6) -> float:
    """根据三维框尺寸计算体积。"""
    size = np.asarray(box.get("size", [0, 0, 0]), dtype=np.float64).reshape(3)
    size = np.abs(size)

    if not np.isfinite(size).all():
        return 0.0

    volume = float(np.prod(np.maximum(size, eps)))
    return volume


def points_inside_box(points: np.ndarray, box: Dict, eps: float = 1e-6) -> np.ndarray:
    """判断点集中的每个点是否位于指定三维框内部。"""
    points = np.asarray(points, dtype=np.float64)

    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    finite = np.isfinite(points).all(axis=1)

    if (
        str(box.get("box_type", "")).upper() == "PCA_OBB"
        and "main_direction_2d" in box
        and "horizontal_axes" in box
        and "up_axis" in box
        and "center" in box
        and "size" in box
    ):
        center = np.asarray(box["center"], dtype=np.float64).reshape(3)
        size = np.abs(np.asarray(box["size"], dtype=np.float64).reshape(3))
        horizontal_axes = [int(x) for x in box["horizontal_axes"]]
        up_axis = int(box["up_axis"])
        main_dir = np.asarray(box["main_direction_2d"], dtype=np.float64).reshape(2)

        norm = float(np.linalg.norm(main_dir))
        if norm > eps and np.isfinite(center).all() and np.isfinite(size).all():
            main_dir = main_dir / norm
            side_dir = np.array([-main_dir[1], main_dir[0]], dtype=np.float64)
            basis = np.stack([main_dir, side_dir], axis=1)

            local_xy = (points[:, horizontal_axes] - center[horizontal_axes]) @ basis
            local_up = points[:, up_axis] - center[up_axis]
            half = size / 2.0 + eps

            return (
                finite
                & (np.abs(local_xy[:, 0]) <= half[0])
                & (np.abs(local_xy[:, 1]) <= half[1])
                & (np.abs(local_up) <= half[2])
            )

    min_xyz, max_xyz = get_box_enclosing_aabb(box)
    min_xyz = np.asarray(min_xyz, dtype=np.float64).reshape(3) - eps
    max_xyz = np.asarray(max_xyz, dtype=np.float64).reshape(3) + eps

    return finite & np.all((points >= min_xyz) & (points <= max_xyz), axis=1)


def count_semantic_points_inside_box(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    box: Dict,
    eps: float = 1e-6
) -> int:
    """统计指定实例语义点中落在三维框内部的点数。"""
    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32)

    instance_id = int(box.get("instance_id", 0))
    class_id = int(box.get("class_id", -1))

    semantic_mask = point_instance_ids == instance_id
    if class_id >= 0:
        semantic_mask &= point_class_ids == class_id

    candidate_indices = np.where(semantic_mask)[0]
    if candidate_indices.size == 0:
        return 0

    inside = points_inside_box(points3d[candidate_indices], box, eps=eps)
    return int(inside.sum())


def compute_aabb_iou_3d(box_a: Dict, box_b: Dict, eps: float = 1e-6) -> float:
    """使用外接轴对齐框近似计算两个三维框的交并比。"""
    a_min, a_max = get_box_enclosing_aabb(box_a)
    b_min, b_max = get_box_enclosing_aabb(box_b)

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)

    inter_vol = float(np.prod(inter_size))

    a_vol = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 0.0)))

    union = a_vol + b_vol - inter_vol

    if union <= eps:
        return 0.0

    return float(inter_vol / union)


def add_density_info_to_box(
    box: Dict,
    points3d: Optional[np.ndarray] = None,
    point_class_ids: Optional[np.ndarray] = None,
    point_instance_ids: Optional[np.ndarray] = None,
    eps: float = 1e-6
) -> Dict:
    """为三维框补充体积、框内语义点数、密度和质量评分。"""
    new_box = dict(box)

    num_points = int(new_box.get("num_points", 0))
    density_num_points = num_points
    density_source = "box_num_points"

    if points3d is not None and point_class_ids is not None and point_instance_ids is not None:
        density_num_points = count_semantic_points_inside_box(
            points3d=points3d,
            point_class_ids=point_class_ids,
            point_instance_ids=point_instance_ids,
            box=new_box,
            eps=eps
        )
        density_source = "semantic_points_inside_box"
    volume = compute_box_volume_from_size(new_box, eps=eps)

    if volume <= eps:
        density = float("inf") if density_num_points > 0 else 0.0
    else:
        density = float(density_num_points / volume)

    # 综合评分：密度优先，同时考虑点数。
    # 防止一个很小但点数极少的碎片因为密度高而压过真实物体。
    quality = float(density * math.log1p(max(density_num_points, 1)))

    new_box["box_volume"] = float(volume)
    new_box["density_num_points"] = int(density_num_points)
    new_box["num_semantic_points_in_box"] = int(density_num_points)
    new_box["density_source"] = density_source
    new_box["point_density"] = float(density)
    new_box["box_quality_score"] = float(quality)

    return new_box


def apply_box_density_filter_and_nms(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    segments: List[Dict],
    boxes: List[Dict],
    background_class_id: int = -1,
    enable_density_filter: bool = True,
    min_box_density: float = 30.0,
    min_box_inner_points: int = 0,
    max_box_volume: Optional[float] = None,
    enable_box_nms: bool = True,
    box_nms_iou_thresh: float = 0.10,
    box_nms_class_aware: bool = False,
    remove_suppressed_box_points: bool = True
) -> Dict:
    """根据框内点数、密度、体积和重叠关系过滤三维框。"""
    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32).copy()
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32).copy()

    boxes_with_quality = [
        add_density_info_to_box(
            box,
            points3d=points3d,
            point_class_ids=point_class_ids,
            point_instance_ids=point_instance_ids
        )
        for box in boxes
    ]

    removed_instance_ids = set()
    density_removed = []
    nms_removed = []

    # -------------------------
    # 1. 密度过滤 / 体积过滤
    # -------------------------
    candidate_boxes = []

    for box in boxes_with_quality:
        instance_id = int(box["instance_id"])
        density = float(box.get("point_density", 0.0))
        volume = float(box.get("box_volume", 0.0))
        num_points = int(box.get("num_points", 0))
        density_num_points = int(box.get("density_num_points", num_points))

        remove_reason = None

        if enable_density_filter and density < float(min_box_density):
            remove_reason = "low_box_density"

        if int(min_box_inner_points) > 0 and density_num_points < int(min_box_inner_points):
            remove_reason = "too_few_box_inner_points"

        if max_box_volume is not None and volume > float(max_box_volume):
            remove_reason = "too_large_box_volume"

        if remove_reason is not None:
            removed_instance_ids.add(instance_id)
            density_removed.append({
                "instance_id": instance_id,
                "class_name": str(box.get("class_name", "")),
                "class_id": int(box.get("class_id", -1)),
                "num_points": num_points,
                "density_num_points": density_num_points,
                "box_volume": volume,
                "point_density": density,
                "reason": remove_reason,
            })
        else:
            candidate_boxes.append(box)

    # -------------------------
    # 2. 3D box NMS，抑制重叠框
    # -------------------------
    kept_boxes = []

    if enable_box_nms and len(candidate_boxes) > 1:
        # 质量高的优先保留
        order = sorted(
            range(len(candidate_boxes)),
            key=lambda i: (
                float(candidate_boxes[i].get("box_quality_score", 0.0)),
                int(candidate_boxes[i].get("density_num_points", candidate_boxes[i].get("num_points", 0)))
            ),
            reverse=True
        )

        suppressed = np.zeros(len(candidate_boxes), dtype=bool)

        for order_i in order:
            if suppressed[order_i]:
                continue

            cur_box = candidate_boxes[order_i]
            kept_boxes.append(cur_box)

            cur_class_id = int(cur_box.get("class_id", -1))

            for order_j in order:
                if order_j == order_i or suppressed[order_j]:
                    continue

                other_box = candidate_boxes[order_j]
                other_class_id = int(other_box.get("class_id", -1))

                if box_nms_class_aware and cur_class_id != other_class_id:
                    continue

                iou = compute_aabb_iou_3d(cur_box, other_box)

                if iou >= float(box_nms_iou_thresh):
                    suppressed[order_j] = True

                    other_instance_id = int(other_box["instance_id"])
                    removed_instance_ids.add(other_instance_id)

                    nms_removed.append({
                        "instance_id": other_instance_id,
                        "class_name": str(other_box.get("class_name", "")),
                        "class_id": other_class_id,
                        "num_points": int(other_box.get("num_points", 0)),
                        "density_num_points": int(other_box.get("density_num_points", other_box.get("num_points", 0))),
                        "point_density": float(other_box.get("point_density", 0.0)),
                        "box_volume": float(other_box.get("box_volume", 0.0)),
                        "suppressed_by_instance_id": int(cur_box["instance_id"]),
                        "iou": float(iou),
                        "reason": "box_nms_overlap",
                    })
    else:
        kept_boxes = candidate_boxes

    # -------------------------
    # 3. 被删除实例的语义点改回背景
    # -------------------------
    removed_point_count = 0

    if remove_suppressed_box_points and len(removed_instance_ids) > 0:
        for instance_id in removed_instance_ids:
            remove_mask = point_instance_ids == int(instance_id)
            removed_point_count += int(remove_mask.sum())

            point_class_ids[remove_mask] = background_class_id
            point_instance_ids[remove_mask] = 0

    # -------------------------
    # 4. 重建 segments
    # -------------------------
    filtered_segments = []

    for seg in segments:
        instance_id = int(seg["instance_id"])

        if instance_id in removed_instance_ids:
            continue

        indices = np.where(point_instance_ids == instance_id)[0]

        if indices.size == 0:
            continue

        filtered_segments.append({
            **seg,
            "point_indices": indices,
            "num_points": int(indices.size),
        })

    stats = {
        "enabled_density_filter": bool(enable_density_filter),
        "min_box_density": float(min_box_density),
        "min_box_inner_points": int(min_box_inner_points),
        "max_box_volume": None if max_box_volume is None else float(max_box_volume),
        "enabled_box_nms": bool(enable_box_nms),
        "box_nms_iou_thresh": float(box_nms_iou_thresh),
        "box_nms_class_aware": bool(box_nms_class_aware),
        "remove_suppressed_box_points": bool(remove_suppressed_box_points),
        "before_boxes": int(len(boxes)),
        "after_boxes": int(len(kept_boxes)),
        "removed_boxes": int(len(boxes) - len(kept_boxes)),
        "removed_instance_ids": sorted([int(x) for x in removed_instance_ids]),
        "removed_point_count": int(removed_point_count),
        "density_removed": density_removed,
        "nms_removed": nms_removed,
    }

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "segments": filtered_segments,
        "boxes": kept_boxes,
        "box_quality_filter_stats": stats,
    }
