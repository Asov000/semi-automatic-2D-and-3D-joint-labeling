# -*- coding: utf-8 -*-
"""三维语义过滤模块，通过邻域和连通域规则清理语义飞点。"""

from typing import Dict, List

import numpy as np


def get_semantic_filter_params(strength: int) -> Dict:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    strength = int(strength)

    table = {
        0: {
            "enabled": False,
            "radius": 0.00,
            "min_neighbors": 0,
            "min_component_points": 0,
            "min_component_ratio": 0.00,
            "keep_largest_only": False,
            "boundary_only": True,
            "boundary_percentile": 0.0,
            "robust_box_lower_percentile": 0.0,
            "robust_box_upper_percentile": 100.0,
            "robust_box_core_percentile": 100.0,
            "outside_hard_remove_radius": 0.0,
            "outside_hard_remove_box_ratio": 0.0,
            "outside_neighbor_boost": 0.0,
        },
        1: {
            "enabled": True,
            "radius": 0.08,
            "min_neighbors": 4,
            "min_component_points": 8,
            "min_component_ratio": 0.005,
            "keep_largest_only": False,
            "boundary_only": True,
            "boundary_percentile": 8.0,
            "robust_box_lower_percentile": 2.0,
            "robust_box_upper_percentile": 98.0,
            "robust_box_core_percentile": 88.0,
            "outside_hard_remove_radius": 4.0,
            "outside_hard_remove_box_ratio": 0.35,
            "outside_neighbor_boost": 1.0,
        },
        2: {
            "enabled": True,
            "radius": 0.10,
            "min_neighbors": 6,
            "min_component_points": 15,
            "min_component_ratio": 0.010,
            "keep_largest_only": False,
            "boundary_only": True,
            "boundary_percentile": 12.0,
            "robust_box_lower_percentile": 3.0,
            "robust_box_upper_percentile": 97.0,
            "robust_box_core_percentile": 82.0,
            "outside_hard_remove_radius": 3.0,
            "outside_hard_remove_box_ratio": 0.25,
            "outside_neighbor_boost": 2.0,
        },
        3: {
            "enabled": True,
            "radius": 0.15,
            "min_neighbors": 8,
            "min_component_points": 25,
            "min_component_ratio": 0.020,
            "keep_largest_only": True,
            "boundary_only": True,
            "boundary_percentile": 18.0,
            "robust_box_lower_percentile": 5.0,
            "robust_box_upper_percentile": 95.0,
            "robust_box_core_percentile": 75.0,
            "outside_hard_remove_radius": 2.0,
            "outside_hard_remove_box_ratio": 0.18,
            "outside_neighbor_boost": 3.0,
        },
    }

    if strength not in table:
        raise ValueError("semantic_filter_strength 只能是 0, 1, 2, 3")

    params = dict(table[strength])
    params["strength"] = strength
    return params


def _connected_components_from_neighbor_lists(neighbor_lists: List[List[int]], valid_local_mask: np.ndarray) -> List[np.ndarray]:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    valid_local_mask = np.asarray(valid_local_mask, dtype=bool)
    n = int(valid_local_mask.shape[0])
    visited = np.zeros(n, dtype=bool)
    components = []

    valid_set = set(np.where(valid_local_mask)[0].tolist())

    for start in list(valid_set):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            for nb in neighbor_lists[cur]:
                if nb in valid_set and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        components.append(np.asarray(comp, dtype=np.int64))

    components.sort(key=lambda x: x.size, reverse=True)
    return components


def _aabb_boundary_candidate_mask(points: np.ndarray, percentile: float) -> np.ndarray:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    if percentile <= 0:
        return np.zeros((points.shape[0],), dtype=bool)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    span = np.maximum(max_xyz - min_xyz, 1e-6)

    normalized = (points - min_xyz) / span
    nearest_face_distance = np.minimum(normalized, 1.0 - normalized).min(axis=1)
    threshold = np.percentile(nearest_face_distance, float(percentile))

    return nearest_face_distance <= threshold


def _robust_box_filter_candidates(
    points: np.ndarray,
    boundary_percentile: float,
    lower_percentile: float,
    upper_percentile: float,
    core_percentile: float
) -> Dict:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])

    if n == 0:
        return {
            "candidate_mask": np.zeros((0,), dtype=bool),
            "outside_distance": np.zeros((0,), dtype=np.float64),
            "outside_mask": np.zeros((0,), dtype=bool),
            "boundary_mask": np.zeros((0,), dtype=bool),
            "box_min": np.zeros((3,), dtype=np.float64),
            "box_max": np.zeros((3,), dtype=np.float64),
        }

    core_points = points
    if 0.0 < core_percentile < 100.0 and n >= 20:
        center = np.median(points, axis=0)
        scale = np.median(np.abs(points - center), axis=0)
        scale = np.maximum(scale, 1e-6)
        normalized_distance = np.linalg.norm((points - center) / scale, axis=1)
        core_threshold = np.percentile(normalized_distance, float(core_percentile))
        core_mask = normalized_distance <= core_threshold

        if int(core_mask.sum()) >= max(10, int(n * 0.25)):
            core_points = points[core_mask]

    low = np.percentile(core_points, float(lower_percentile), axis=0)
    high = np.percentile(core_points, float(upper_percentile), axis=0)

    if not np.all(high > low):
        low = points.min(axis=0)
        high = points.max(axis=0)

    span = np.maximum(high - low, 1e-6)
    below = np.maximum(low - points, 0.0)
    above = np.maximum(points - high, 0.0)
    outside_vector = below + above
    outside_distance = np.linalg.norm(outside_vector, axis=1)
    outside_mask = outside_distance > 0.0

    normalized = (points - low) / span
    inside_mask = np.all((normalized >= 0.0) & (normalized <= 1.0), axis=1)
    nearest_face_distance = np.minimum(normalized, 1.0 - normalized).min(axis=1)

    boundary_mask = np.zeros((n,), dtype=bool)
    if boundary_percentile > 0 and int(inside_mask.sum()) > 0:
        inside_distances = nearest_face_distance[inside_mask]
        threshold = np.percentile(inside_distances, float(boundary_percentile))
        boundary_mask = inside_mask & (nearest_face_distance <= threshold)

    return {
        "candidate_mask": outside_mask | boundary_mask,
        "outside_distance": outside_distance,
        "outside_mask": outside_mask,
        "boundary_mask": boundary_mask,
        "box_min": low,
        "box_max": high,
    }


def filter_semantic_outliers_3d(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    segments: List[Dict],
    strength: int = 0,
    background_class_id: int = -1,
    min_points: int = 30
) -> Dict:
    """对三维语义点云执行离群点过滤。"""
    strength = int(strength)
    params = get_semantic_filter_params(strength)

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32).copy()
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32).copy()

    before_labeled = int((point_class_ids >= 0).sum())

    if not params["enabled"] or before_labeled == 0:
        return {
            "point_class_ids": point_class_ids,
            "point_instance_ids": point_instance_ids,
            "segments": segments,
            "filter_stats": {
                "semantic_filter_strength": strength,
                "enabled": False,
                "before_labeled_points": before_labeled,
                "after_labeled_points": before_labeled,
                "removed_labeled_points": 0,
                "removed_ratio": 0.0,
            }
        }

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise ImportError(
            "3D 语义图滤波需要 scipy.spatial.cKDTree。请确认 scipy 已安装。"
        ) from e

    radius = float(params["radius"])
    min_neighbors = int(params["min_neighbors"])
    min_component_points = int(params["min_component_points"])
    min_component_ratio = float(params["min_component_ratio"])
    keep_largest_only = bool(params["keep_largest_only"])
    boundary_only = bool(params.get("boundary_only", True))
    boundary_percentile = float(params.get("boundary_percentile", 10.0))
    robust_box_lower_percentile = float(params.get("robust_box_lower_percentile", 2.0))
    robust_box_upper_percentile = float(params.get("robust_box_upper_percentile", 98.0))
    robust_box_core_percentile = float(params.get("robust_box_core_percentile", 85.0))
    outside_hard_remove_radius = float(params.get("outside_hard_remove_radius", 3.0))
    outside_hard_remove_box_ratio = float(params.get("outside_hard_remove_box_ratio", 0.25))
    outside_neighbor_boost = float(params.get("outside_neighbor_boost", 1.0))

    removed_total = 0
    filtered_segments = []
    segment_stats = []

    for seg in segments:
        instance_id = int(seg["instance_id"])
        class_id = int(seg["class_id"])

        # 以实例为单位处理。这样不会把两个同类但空间分离的物体误合并。
        obj_indices = np.where(point_instance_ids == instance_id)[0]
        raw_n = int(obj_indices.size)

        if raw_n == 0:
            continue

        # 点太少时不做半径连通滤波，避免小目标被直接吃掉。
        if raw_n < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })
            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        obj_points = points3d[obj_indices]
        finite = np.isfinite(obj_points).all(axis=1)

        if finite.sum() < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })
            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        # 对非有限点直接视为离群点。
        finite_local_indices = np.where(finite)[0]
        finite_points = obj_points[finite]

        if boundary_only:
            candidate_info = _robust_box_filter_candidates(
                finite_points,
                boundary_percentile=boundary_percentile,
                lower_percentile=robust_box_lower_percentile,
                upper_percentile=robust_box_upper_percentile,
                core_percentile=robust_box_core_percentile
            )
            candidate_finite_indices = np.where(candidate_info["candidate_mask"])[0]
            outside_distance = candidate_info["outside_distance"]
            keep_local = np.ones(raw_n, dtype=bool)

            if candidate_finite_indices.size > 0:
                tree = cKDTree(finite_points)
                candidate_points = finite_points[candidate_finite_indices]
                candidate_neighbor_lists = tree.query_ball_point(candidate_points, r=radius)
                candidate_neighbor_counts = np.array(
                    [len(x) for x in candidate_neighbor_lists],
                    dtype=np.int32
                )
                candidate_outside_distance = outside_distance[candidate_finite_indices]
                robust_box_size = candidate_info["box_max"] - candidate_info["box_min"]
                robust_box_diag = float(np.linalg.norm(np.maximum(robust_box_size, 0.0)))
                hard_remove_distance = max(
                    radius * outside_hard_remove_radius,
                    robust_box_diag * outside_hard_remove_box_ratio,
                    1e-6
                )
                hard_remove_mask = candidate_outside_distance >= hard_remove_distance
                distance_boost = np.ceil(
                    (candidate_outside_distance / max(radius, 1e-6)) * outside_neighbor_boost
                ).astype(np.int32)
                required_neighbors = min_neighbors + distance_boost
                sparse_candidate_mask = candidate_neighbor_counts < required_neighbors
                remove_candidate_mask = hard_remove_mask | sparse_candidate_mask
                raw_candidate_indices = finite_local_indices[candidate_finite_indices]
                remove_local_indices = raw_candidate_indices[remove_candidate_mask]
                keep_local[remove_local_indices] = False

            keep_local[~finite] = False

            keep_global = obj_indices[keep_local]
            remove_global = obj_indices[~keep_local]

            if remove_global.size > 0:
                point_class_ids[remove_global] = background_class_id
                point_instance_ids[remove_global] = 0

            removed = int(remove_global.size)
            after_n = int(keep_global.size)
            removed_total += removed

            if after_n > 0:
                filtered_segments.append({
                    **seg,
                    "point_indices": keep_global,
                    "num_points": after_n,
                    "raw_num_points_before_semantic_filter": raw_n,
                    "semantic_filter_removed_points": removed,
                    "semantic_filter_strength": strength,
                    "semantic_filter_boundary_only": True,
                    "semantic_filter_boundary_candidate_points": int(candidate_finite_indices.size),
                    "semantic_filter_outside_candidate_points": int(candidate_info["outside_mask"].sum()),
                })

            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": after_n,
                "removed": removed,
                "skipped": False,
                "boundary_only": True,
                "boundary_candidate_points": int(candidate_finite_indices.size),
                "outside_candidate_points": int(candidate_info["outside_mask"].sum()),
                "max_outside_distance": float(outside_distance.max()) if outside_distance.size else 0.0,
            })
            continue

        tree = cKDTree(finite_points)
        finite_neighbor_lists = tree.query_ball_point(finite_points, r=radius)
        finite_neighbor_counts = np.array([len(x) for x in finite_neighbor_lists], dtype=np.int32)

        dense_finite_mask = finite_neighbor_counts >= min_neighbors

        # 映射回 obj_points 的局部索引。
        dense_local_mask = np.zeros(raw_n, dtype=bool)
        dense_local_mask[finite_local_indices[dense_finite_mask]] = True

        # 为了复用邻接表，需要构建 raw_n 长度的邻接表，邻接索引为 obj_points 局部索引。
        neighbor_lists = [[] for _ in range(raw_n)]
        finite_to_raw = finite_local_indices
        for finite_i, raw_i in enumerate(finite_to_raw):
            neighbor_lists[int(raw_i)] = [int(finite_to_raw[j]) for j in finite_neighbor_lists[finite_i]]

        components = _connected_components_from_neighbor_lists(
            neighbor_lists=neighbor_lists,
            valid_local_mask=dense_local_mask
        )

        if len(components) == 0:
            # 极端情况下不全删，保留原实例，避免误伤整物体。
            keep_local = np.ones(raw_n, dtype=bool)
        else:
            if keep_largest_only:
                keep_components = [components[0]]
            else:
                threshold = max(min_component_points, int(round(raw_n * min_component_ratio)))
                keep_components = [comp for comp in components if comp.size >= threshold]

                # 如果阈值过严导致全无，则至少保留最大的主体。
                if len(keep_components) == 0:
                    keep_components = [components[0]]

            keep_local = np.zeros(raw_n, dtype=bool)
            for comp in keep_components:
                keep_local[comp] = True

        keep_global = obj_indices[keep_local]
        remove_global = obj_indices[~keep_local]

        if remove_global.size > 0:
            point_class_ids[remove_global] = background_class_id
            point_instance_ids[remove_global] = 0

        removed = int(remove_global.size)
        after_n = int(keep_global.size)
        removed_total += removed

        if after_n > 0:
            filtered_segments.append({
                **seg,
                "point_indices": keep_global,
                "num_points": after_n,
                "raw_num_points_before_semantic_filter": raw_n,
                "semantic_filter_removed_points": removed,
                "semantic_filter_strength": strength,
            })

        segment_stats.append({
            "instance_id": instance_id,
            "class_id": class_id,
            "before": raw_n,
            "after": after_n,
            "removed": removed,
            "skipped": False,
        })

    after_labeled = int((point_class_ids >= 0).sum())

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "segments": filtered_segments,
        "filter_stats": {
            "semantic_filter_strength": strength,
            "enabled": True,
            "radius": radius,
            "min_neighbors": min_neighbors,
            "min_component_points": min_component_points,
            "min_component_ratio": min_component_ratio,
            "keep_largest_only": keep_largest_only,
            "boundary_only": boundary_only,
            "boundary_percentile": boundary_percentile,
            "robust_box_lower_percentile": robust_box_lower_percentile,
            "robust_box_upper_percentile": robust_box_upper_percentile,
            "robust_box_core_percentile": robust_box_core_percentile,
            "outside_hard_remove_radius": outside_hard_remove_radius,
            "outside_hard_remove_box_ratio": outside_hard_remove_box_ratio,
            "outside_neighbor_boost": outside_neighbor_boost,
            "before_labeled_points": before_labeled,
            "after_labeled_points": after_labeled,
            "removed_labeled_points": int(before_labeled - after_labeled),
            "removed_ratio": float((before_labeled - after_labeled) / max(before_labeled, 1)),
            "segment_stats": segment_stats,
        }
    }
