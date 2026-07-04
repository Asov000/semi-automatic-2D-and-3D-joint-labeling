# -*- coding: utf-8 -*-
"""三维导出模块，负责保存点云 mask、三维框和带颜色点云文件。"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import ensure_dir, jsonable


def save_point_masks(
    save_dir: str,
    sample_name: str,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    valid_projected_mask: Optional[np.ndarray] = None,
    visible_projected_mask: Optional[np.ndarray] = None
) -> str:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    ensure_dir(save_dir)

    save_path = os.path.join(save_dir, f"{sample_name}_point_masks.npz")

    np.savez_compressed(
        save_path,
        point_class_ids=point_class_ids.astype(np.int32),
        point_instance_ids=point_instance_ids.astype(np.int32),
        valid_projected_mask=valid_projected_mask.astype(bool) if valid_projected_mask is not None else None,
        visible_projected_mask=visible_projected_mask.astype(bool) if visible_projected_mask is not None else None,
    )

    return save_path


def save_3d_boxes_json(
    save_path: str,
    boxes: List[Dict],
    extra_info: Optional[Dict] = None
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    ensure_dir(os.path.dirname(save_path))

    data = {
        "boxes": boxes
    }

    if extra_info is not None:
        data["extra_info"] = extra_info

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=4)


def save_3d_boxes_txt(
    save_path: str,
    boxes: List[Dict]
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", encoding="utf-8") as f:
        for box in boxes:
            class_name = str(box["class_name"])
            class_id = int(box["class_id"])
            instance_id = int(box["instance_id"])

            center = np.asarray(box["center"], dtype=np.float64)
            size = np.asarray(box["size"], dtype=np.float64)
            heading = float(box.get("heading_angle", 0.0))
            num_points = int(box.get("num_points", 0))

            line = (
                f"{class_name} "
                f"{class_id:d} "
                f"{instance_id:d} "
                f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f} "
                f"{heading:.6f} "
                f"{num_points:d}"
            )

            f.write(line + "\n")


def save_labeled_pointcloud_ply(
    save_path: str,
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    class_id_to_color: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    ensure_dir(os.path.dirname(save_path))

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)

    if class_id_to_color is None:
        class_id_to_color = {}

    def _color_to_uint8(color):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        arr = np.asarray(color, dtype=np.float64).reshape(3)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)

    try:
        from run import DEFAULT_CLASS_COLOR_PALETTE

        default_colors = DEFAULT_CLASS_COLOR_PALETTE
    except Exception:
        default_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
            (128, 255, 0),
        ]

    colors = np.zeros((points3d.shape[0], 3), dtype=np.uint8)
    colors[:, :] = np.array([160, 160, 160], dtype=np.uint8)

    unique_class_ids = sorted([int(x) for x in np.unique(point_class_ids) if int(x) >= 0])

    for cid in unique_class_ids:
        # 兼容 {0: color} 和 {"0": color} 两种 key
        if cid in class_id_to_color:
            color = _color_to_uint8(class_id_to_color[cid])
        elif str(cid) in class_id_to_color:
            color = _color_to_uint8(class_id_to_color[str(cid)])
        else:
            # 关键修复：用 cid 取色，而不是 enumerate 后的 i
            color = np.array(default_colors[cid % len(default_colors)], dtype=np.uint8)

        colors[point_class_ids == cid] = color

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points3d.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points3d, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
