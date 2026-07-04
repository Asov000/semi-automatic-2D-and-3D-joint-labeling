# -*- coding: utf-8 -*-
"""三维点赋值模块，将点云投影到二维 mask 并生成类别和实例标签。"""

from typing import Dict, List, Tuple

import numpy as np

from .common import to_binary_mask
from .projection import filter_visible_points_by_zbuffer, project_points_to_image_with_indices


def assign_3d_points_to_2d_masks(
    points3d: np.ndarray,
    K: np.ndarray,
    Rtilt: np.ndarray,
    image_shape: Tuple[int, int, int],
    masks_2d: List[Dict],
    points_are_after_rtilt: bool = True,
    use_matlab_pixel: bool = True,
    use_zbuffer: bool = True,
    zbuffer_tolerance: float = 0.03,
    background_class_id: int = -1,
    overlap_policy: str = "later"
) -> Dict:
    """将三维点投影到二维 mask，并生成点级类别和实例标签。"""
    if overlap_policy not in ["later", "first"]:
        raise ValueError("overlap_policy 只能是 'later' 或 'first'")

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    num_points = points3d.shape[0]

    uv, depth, point_indices, uv_float = project_points_to_image_with_indices(
        points3d=points3d,
        K=K,
        image_shape=image_shape,
        Rtilt=Rtilt,
        points_are_after_rtilt=points_are_after_rtilt,
        use_matlab_pixel=use_matlab_pixel
    )

    valid_projected_mask = np.zeros(num_points, dtype=bool)
    valid_projected_mask[point_indices] = True

    if use_zbuffer:
        visible_local_mask = filter_visible_points_by_zbuffer(
            uv=uv,
            depth=depth,
            image_shape=image_shape,
            tolerance=zbuffer_tolerance
        )

        uv = uv[visible_local_mask]
        depth = depth[visible_local_mask]
        point_indices = point_indices[visible_local_mask]
        uv_float = uv_float[visible_local_mask]
    else:
        visible_local_mask = np.ones(len(point_indices), dtype=bool)

    visible_projected_mask = np.zeros(num_points, dtype=bool)
    visible_projected_mask[point_indices] = True

    point_class_ids = np.full(num_points, background_class_id, dtype=np.int32)
    point_instance_ids = np.zeros(num_points, dtype=np.int32)

    instance_meta = {}

    for item in masks_2d:
        mask = to_binary_mask(item["mask"])

        class_name = str(item["class_name"])
        class_id = int(item["class_id"])
        instance_id = int(item["instance_id"])

        u = uv[:, 0]
        v = uv[:, 1]

        hit = mask[v, u] > 0

        if overlap_policy == "first":
            hit = hit & (point_instance_ids[point_indices] == 0)

        selected_indices = point_indices[hit]

        if selected_indices.size == 0:
            continue

        point_class_ids[selected_indices] = class_id
        point_instance_ids[selected_indices] = instance_id

        instance_meta[instance_id] = {
            "instance_id": instance_id,
            "class_name": class_name,
            "class_id": class_id,
            "mask_path": item.get("mask_path", None),
        }

    segments = []

    for instance_id in sorted(instance_meta.keys()):
        indices = np.where(point_instance_ids == instance_id)[0]

        if indices.size == 0:
            continue

        meta = instance_meta[instance_id]

        segments.append({
            "instance_id": int(instance_id),
            "class_name": meta["class_name"],
            "class_id": int(meta["class_id"]),
            "point_indices": indices,
            "num_points": int(indices.size),
            "mask_path": meta.get("mask_path", None),
        })

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "valid_projected_mask": valid_projected_mask,
        "visible_projected_mask": visible_projected_mask,
        "segments": segments,
        "uv": uv,
        "uv_float": uv_float,
        "depth": depth,
        "point_indices": point_indices,
    }
