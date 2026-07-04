# -*- coding: utf-8 -*-
"""三维可视化模块，使用 Open3D 显示语义点云和三维框。"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import load_mat_points


def get_default_color_palette() -> List[Tuple[float, float, float]]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    try:
        from run import DEFAULT_CLASS_COLOR_PALETTE

        return [
            normalize_color_to_open3d(color)
            for color in DEFAULT_CLASS_COLOR_PALETTE
        ]
    except Exception:
        return [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.5, 0.0),
            (0.5, 0.0, 1.0),
            (0.0, 0.5, 1.0),
            (0.5, 1.0, 0.0),
        ]


def normalize_color_to_open3d(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    arr = np.asarray(color, dtype=np.float64).reshape(3)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return float(arr[0]), float(arr[1]), float(arr[2])


def get_class_color_by_id(
    class_id: int,
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> Tuple[float, float, float]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    cid = int(class_id)

    if class_id_to_color is not None:
        if cid in class_id_to_color:
            return normalize_color_to_open3d(class_id_to_color[cid])
        if str(cid) in class_id_to_color:
            return normalize_color_to_open3d(class_id_to_color[str(cid)])

    palette = get_default_color_palette()
    return palette[cid % len(palette)]


def get_instance_color_by_id(instance_id: int) -> Tuple[float, float, float]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    palette = get_default_color_palette()
    iid = int(instance_id)
    return palette[(iid - 1) % len(palette)]


def build_open3d_labeled_pointcloud(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: Optional[np.ndarray] = None,
    color_mode: str = "class",
    show_background: bool = True,
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    import open3d as o3d

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)

    if point_instance_ids is not None:
        point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32)

    if color_mode not in ["class", "instance"]:
        raise ValueError("color_mode 只能是 'class' 或 'instance'")

    if not show_background:
        keep = point_class_ids >= 0
        points_show = points3d[keep]
        class_ids_show = point_class_ids[keep]

        if point_instance_ids is not None:
            instance_ids_show = point_instance_ids[keep]
        else:
            instance_ids_show = None
    else:
        points_show = points3d
        class_ids_show = point_class_ids
        instance_ids_show = point_instance_ids

    colors = np.zeros((points_show.shape[0], 3), dtype=np.float64)
    colors[:, :] = np.array([0.55, 0.55, 0.55], dtype=np.float64)  # 未标注背景点灰色

    if color_mode == "class":
        unique_ids = sorted([int(x) for x in np.unique(class_ids_show) if int(x) >= 0])

        for cid in unique_ids:
            # 关键修复：直接用 class_id 取色，不能用 enumerate(unique_ids) 的序号取色
            color = get_class_color_by_id(cid, class_id_to_color=class_id_to_color)
            colors[class_ids_show == cid] = np.array(color, dtype=np.float64)

    else:
        if instance_ids_show is None:
            raise ValueError("color_mode='instance' 时必须传入 point_instance_ids")

        unique_ids = sorted([int(x) for x in np.unique(instance_ids_show) if int(x) > 0])

        for iid in unique_ids:
            # 实例颜色也按 instance_id 稳定取色，不再按当前样本出现顺序取色
            color = get_instance_color_by_id(iid)
            colors[instance_ids_show == iid] = np.array(color, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def build_open3d_box_lineset(
    box: Dict,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    import open3d as o3d

    corners = np.asarray(box["corners"], dtype=np.float64)

    if corners.shape != (8, 3):
        raise ValueError(f"box['corners'] 必须是 8x3，当前 shape={corners.shape}")

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def build_open3d_box_center_sphere(
    box: Dict,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius: float = 0.035
):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    import open3d as o3d

    center = np.asarray(box["center"], dtype=np.float64)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)

    return sphere


def visualize_labeled_pointcloud_and_boxes(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: Optional[np.ndarray],
    boxes: List[Dict],
    color_mode: str = "class",
    show_background: bool = True,
    show_box_center: bool = True,
    point_size: float = 2.0,
    window_name: str = "3D labeled point cloud + boxes",
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> None:
    """使用 Open3D 显示语义点云和三维框。"""
    import open3d as o3d

    geometries = []

    pcd = build_open3d_labeled_pointcloud(
        points3d=points3d,
        point_class_ids=point_class_ids,
        point_instance_ids=point_instance_ids,
        color_mode=color_mode,
        show_background=show_background,
        class_id_to_color=class_id_to_color
    )

    geometries.append(pcd)

    # 按你的要求：所有 3D 框固定为绿色，不再按类别变化。
    box_color = (0.0, 1.0, 0.0)

    for box in boxes:
        line_set = build_open3d_box_lineset(
            box=box,
            color=box_color
        )
        geometries.append(line_set)

        if show_box_center:
            center_sphere = build_open3d_box_center_sphere(
                box=box,
                color=box_color,
                radius=0.035
            )
            geometries.append(center_sphere)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6,
        origin=[0, 0, 0]
    )
    geometries.append(coord)

    print("\n========== 3D Box 信息 ==========")
    for box in boxes:
        center = np.asarray(box["center"], dtype=np.float64)
        size = np.asarray(box["size"], dtype=np.float64)

        print(
            f"class={box['class_name']} | "
            f"class_id={box['class_id']} | "
            f"instance_id={box['instance_id']} | "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
            f"size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) | "
            f"heading={float(box.get('heading_angle', 0.0)):.3f} | "
            f"points={box.get('num_points', 0)}"
        )

    print("\n========== 3D 点云颜色模式 ==========")
    print(f"color_mode={color_mode}")
    print("注意：若 color_mode='instance'，颜色表示实例，不表示类别。要和 2D 类别颜色一致，请使用 color_mode='class'。")
    print("3D 框颜色：固定绿色")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=900)

    for g in geometries:
        vis.add_geometry(g)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def visualize_result_from_memory(
    result: Dict,
    color_mode: str = "class",
    show_background: bool = True
) -> None:
    """直接可视化内存中的三维标注结果。"""
    visualize_labeled_pointcloud_and_boxes(
        points3d=result["points3d"],
        point_class_ids=result["point_class_ids"],
        point_instance_ids=result["point_instance_ids"],
        boxes=result["boxes"],
        color_mode=color_mode,
        show_background=show_background,
        show_box_center=True,
        point_size=2.0,
        window_name="3D annotation result",
        class_id_to_color=result.get("class_id_to_color", None)
    )


def visualize_saved_3d_annotation(
    root_dir: str,
    image_id: int,
    annotation_root: Optional[str] = None,
    color_mode: str = "class",
    show_background: bool = True
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    sample_name = f"{image_id:06d}"

    if annotation_root is None:
        annotation_root = os.path.join(root_dir, "annotation_3d")

    sample_annotation_dir = os.path.join(annotation_root, sample_name)

    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    point_mask_path = os.path.join(sample_annotation_dir, f"{sample_name}_point_masks.npz")
    boxes_json_path = os.path.join(sample_annotation_dir, f"{sample_name}_3d_boxes.json")

    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"点云文件不存在: {pc_path}")

    if not os.path.exists(point_mask_path):
        raise FileNotFoundError(f"点云 mask 文件不存在: {point_mask_path}")

    if not os.path.exists(boxes_json_path):
        raise FileNotFoundError(f"3D box json 文件不存在: {boxes_json_path}")

    points3d_rgb = load_mat_points(pc_path)
    points3d = points3d_rgb[:, 0:3]

    data = np.load(point_mask_path, allow_pickle=True)

    point_class_ids = data["point_class_ids"]
    point_instance_ids = data["point_instance_ids"]

    with open(boxes_json_path, "r", encoding="utf-8") as f:
        box_data = json.load(f)

    boxes = box_data["boxes"]
    class_id_to_color = box_data.get("extra_info", {}).get("class_id_to_color", None)

    print("[Visualize] 点云:", pc_path)
    print("[Visualize] 点云 mask:", point_mask_path)
    print("[Visualize] 3D boxes:", boxes_json_path)
    print("[Visualize] 点数量:", points3d.shape[0])
    print("[Visualize] 已标注点数量:", int((point_class_ids >= 0).sum()))
    print("[Visualize] 3D 框数量:", len(boxes))

    visualize_labeled_pointcloud_and_boxes(
        points3d=points3d,
        point_class_ids=point_class_ids,
        point_instance_ids=point_instance_ids,
        boxes=boxes,
        color_mode=color_mode,
        show_background=show_background,
        show_box_center=True,
        point_size=2.0,
        window_name=f"{sample_name} labeled point cloud + boxes",
        class_id_to_color=class_id_to_color
    )
