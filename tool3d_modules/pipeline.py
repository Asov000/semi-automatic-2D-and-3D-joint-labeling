# -*- coding: utf-8 -*-
"""三维标注主流程模块，串联读取、投影、赋值、建框、过滤和保存。"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .assignment import assign_3d_points_to_2d_masks
from .boxes import add_density_info_to_box, apply_box_density_filter_and_nms, build_3d_boxes_from_segments
from .common import ensure_dir
from .export import save_3d_boxes_json, save_3d_boxes_txt, save_labeled_pointcloud_ply, save_point_masks
from .io import load_binary_masks_from_segmentation_dir, load_calib, load_mat_points
from .mask_nms import semantic_mask_nms
from .semantic_filter import filter_semantic_outliers_3d


def generate_3d_annotations_for_one_sample(
    root_dir: str,
    image_id: int,
    segmentation_dir: str,
    save_root: Optional[str] = None,
    box_type: str = "pca",
    min_points: int = 30,
    up_axis: int = 2,
    use_zbuffer: bool = True,
    zbuffer_tolerance: float = 0.03,
    use_percentile_filter: bool = True,
    lower_percentile: float = 3.0,
    upper_percentile: float = 97.0,
    semantic_filter_strength: int = 0,
    box_density_filter_enabled: bool = True,
    min_box_density: float = 30.0,
    min_box_inner_points: int = 0,
    max_box_volume: Optional[float] = None,
    box_nms_enabled: bool = True,
    box_nms_iou_thresh: float = 0.10,
    box_nms_class_aware: bool = False,
    remove_suppressed_box_points: bool = True,
    save_ply: bool = True,
    save_point_mask: bool = True,
    save_boxes: bool = True,
    class_name_to_color: Optional[Dict[str, Tuple[int, int, int]]] = None,
    enable_mask_nms: bool = True,
    mask_nms_iou_threshold: float = 0.65,
    mask_nms_overlap_min_threshold: float = 0.80,
    mask_nms_same_class_iou_threshold: float = 0.90,
    mask_nms_same_class_overlap_min_threshold: float = 0.95
) -> Dict:
    """为单个样本生成三维语义点云、三维框和导出文件。"""
    sample_name = f"{image_id:06d}"

    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    image_path = os.path.join(root_dir, "image", sample_name + ".jpg")
    calib_path = os.path.join(root_dir, "calib", sample_name + ".txt")

    if save_root is None:
        save_root = os.path.join(root_dir, "annotation_3d")

    save_dir = os.path.join(save_root, sample_name)
    should_save_outputs = bool(save_point_mask or save_boxes or save_ply)

    if should_save_outputs:
        ensure_dir(save_dir)

    print("[3DTool] 读取点云:", pc_path)
    print("[3DTool] 读取图像:", image_path)
    print("[3DTool] 读取标定:", calib_path)
    print("[3DTool] 读取2D mask:", segmentation_dir)

    points3d_rgb = load_mat_points(pc_path)
    points3d = points3d_rgb[:, 0:3]

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"图像读取失败: {image_path}")

    Rtilt, K = load_calib(calib_path)

    masks_2d, class_to_id = load_binary_masks_from_segmentation_dir(
        segmentation_dir=segmentation_dir,
        image_name=sample_name
    )

    class_id_to_color = {}
    if class_name_to_color:
        for class_name, class_id in class_to_id.items():
            if class_name in class_name_to_color:
                class_id_to_color[int(class_id)] = class_name_to_color[class_name]

    if len(masks_2d) == 0:
        raise RuntimeError(f"没有读取到有效 2D mask: {segmentation_dir}")

    if enable_mask_nms:
        masks_2d, mask_nms_stats = semantic_mask_nms(
            masks_2d,
            iou_threshold=mask_nms_iou_threshold,
            overlap_min_threshold=mask_nms_overlap_min_threshold,
            suppress_same_class=True,
            same_class_iou_threshold=mask_nms_same_class_iou_threshold,
            same_class_overlap_min_threshold=mask_nms_same_class_overlap_min_threshold
        )
    else:
        mask_nms_stats = {
            "enabled": False,
            "before": int(len(masks_2d)),
            "after": int(len(masks_2d)),
            "removed": [],
            "reason": "disabled_by_run_config",
        }

    if len(masks_2d) == 0:
        raise RuntimeError(f"2D mask NMS removed all masks: {segmentation_dir}")

    assign_result = assign_3d_points_to_2d_masks(
        points3d=points3d,
        K=K,
        Rtilt=Rtilt,
        image_shape=image.shape,
        masks_2d=masks_2d,
        points_are_after_rtilt=True,
        use_matlab_pixel=True,
        use_zbuffer=use_zbuffer,
        zbuffer_tolerance=zbuffer_tolerance,
        background_class_id=-1,
        overlap_policy="later"
    )

    boxes = build_3d_boxes_from_segments(
        points3d=points3d,
        segments=assign_result["segments"],
        box_type=box_type,
        min_points=min_points,
        up_axis=up_axis,
        use_percentile_filter=use_percentile_filter,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile
    )

    # =========================
    # 3D 框质量修正：
    # 1. 低密度框删除；
    # 2. 重叠框 NMS；
    # 3. 被删除实例的点云语义改回背景。
    # =========================
    box_quality_filter_result = apply_box_density_filter_and_nms(
        points3d=points3d,
        point_class_ids=assign_result["point_class_ids"],
        point_instance_ids=assign_result["point_instance_ids"],
        segments=assign_result["segments"],
        boxes=boxes,
        background_class_id=-1,
        enable_density_filter=box_density_filter_enabled,
        min_box_density=min_box_density,
        min_box_inner_points=min_box_inner_points,
        max_box_volume=max_box_volume,
        enable_box_nms=box_nms_enabled,
        box_nms_iou_thresh=box_nms_iou_thresh,
        box_nms_class_aware=box_nms_class_aware,
        remove_suppressed_box_points=remove_suppressed_box_points
    )

    assign_result["point_class_ids"] = box_quality_filter_result["point_class_ids"]
    assign_result["point_instance_ids"] = box_quality_filter_result["point_instance_ids"]
    assign_result["segments"] = box_quality_filter_result["segments"]
    boxes = box_quality_filter_result["boxes"]
    box_quality_filter_stats = box_quality_filter_result["box_quality_filter_stats"]

    semantic_filter_result = filter_semantic_outliers_3d(
        points3d=points3d,
        point_class_ids=assign_result["point_class_ids"],
        point_instance_ids=assign_result["point_instance_ids"],
        segments=assign_result["segments"],
        strength=semantic_filter_strength,
        background_class_id=-1,
        min_points=min_points
    )

    assign_result["point_class_ids"] = semantic_filter_result["point_class_ids"]
    assign_result["point_instance_ids"] = semantic_filter_result["point_instance_ids"]
    assign_result["segments"] = semantic_filter_result["segments"]
    semantic_filter_stats = semantic_filter_result["filter_stats"]

    boxes = build_3d_boxes_from_segments(
        points3d=points3d,
        segments=assign_result["segments"],
        box_type=box_type,
        min_points=min_points,
        up_axis=up_axis,
        use_percentile_filter=use_percentile_filter,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile
    )
    boxes = [
        add_density_info_to_box(
            box,
            points3d=points3d,
            point_class_ids=assign_result["point_class_ids"],
            point_instance_ids=assign_result["point_instance_ids"]
        )
        for box in boxes
    ]

    point_mask_path = None
    boxes_json_path = None
    boxes_txt_path = None

    if save_point_mask:
        point_mask_path = save_point_masks(
            save_dir=save_dir,
            sample_name=sample_name,
            point_class_ids=assign_result["point_class_ids"],
            point_instance_ids=assign_result["point_instance_ids"],
            valid_projected_mask=assign_result["valid_projected_mask"],
            visible_projected_mask=assign_result["visible_projected_mask"]
        )

    if save_boxes:
        boxes_json_path = os.path.join(save_dir, f"{sample_name}_3d_boxes.json")
        boxes_txt_path = os.path.join(save_dir, f"{sample_name}_3d_boxes.txt")

        extra_info = {
            "sample_name": sample_name,
            "pc_path": pc_path,
            "image_path": image_path,
            "calib_path": calib_path,
            "segmentation_dir": segmentation_dir,
            "num_points": int(points3d.shape[0]),
            "num_projected_points": int(assign_result["valid_projected_mask"].sum()),
            "num_visible_projected_points": int(assign_result["visible_projected_mask"].sum()),
            "num_labeled_points": int((assign_result["point_class_ids"] >= 0).sum()),
            "class_to_id": class_to_id,
            "class_id_to_color": class_id_to_color,
            "mask_nms_stats": mask_nms_stats,
            "box_type": box_type,
            "min_points": min_points,
            "use_zbuffer": use_zbuffer,
            "zbuffer_tolerance": zbuffer_tolerance,
            "use_percentile_filter": use_percentile_filter,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "enable_mask_nms": bool(enable_mask_nms),
            "mask_nms_iou_threshold": float(mask_nms_iou_threshold),
            "mask_nms_overlap_min_threshold": float(mask_nms_overlap_min_threshold),
            "mask_nms_same_class_iou_threshold": float(mask_nms_same_class_iou_threshold),
            "mask_nms_same_class_overlap_min_threshold": float(mask_nms_same_class_overlap_min_threshold),
            "semantic_filter_strength": int(semantic_filter_strength),
            "semantic_filter_stats": semantic_filter_stats,
            "boxes_rebuilt_after_semantic_filter": True,
            "box_density_filter_enabled": bool(box_density_filter_enabled),
            "min_box_density": float(min_box_density),
            "min_box_inner_points": int(min_box_inner_points),
            "max_box_volume": None if max_box_volume is None else float(max_box_volume),
            "box_nms_enabled": bool(box_nms_enabled),
            "box_nms_iou_thresh": float(box_nms_iou_thresh),
            "box_nms_class_aware": bool(box_nms_class_aware),
            "remove_suppressed_box_points": bool(remove_suppressed_box_points),
            "box_quality_filter_stats": box_quality_filter_stats,
        }

        save_3d_boxes_json(
            save_path=boxes_json_path,
            boxes=boxes,
            extra_info=extra_info
        )

        save_3d_boxes_txt(
            save_path=boxes_txt_path,
            boxes=boxes
        )

    ply_path = None

    if save_ply:
        ply_path = os.path.join(save_dir, f"{sample_name}_labeled_points.ply")
        save_labeled_pointcloud_ply(
            save_path=ply_path,
            points3d=points3d,
            point_class_ids=assign_result["point_class_ids"],
            class_id_to_color=class_id_to_color
        )

    result = {
        "sample_name": sample_name,
        "points3d": points3d,
        "point_class_ids": assign_result["point_class_ids"],
        "point_instance_ids": assign_result["point_instance_ids"],
        "segments": assign_result["segments"],
        "boxes": boxes,
        "class_to_id": class_to_id,
        "class_id_to_color": class_id_to_color,
        "mask_nms_stats": mask_nms_stats,
        "semantic_filter_stats": semantic_filter_stats,
        "box_quality_filter_stats": box_quality_filter_stats,
        "save_paths": {
            "point_masks": point_mask_path,
            "boxes_json": boxes_json_path,
            "boxes_txt": boxes_txt_path,
            "labeled_ply": ply_path,
        }
    }

    print("[3DTool] 完成")
    print("[3DTool] 3D 实例数量:", len(assign_result["segments"]))
    print("[3DTool] 3D 框数量:", len(boxes))
    print("[3DTool] 3D语义图滤波强度:", int(semantic_filter_strength))
    print("[3DTool] 语义滤波移除点数量:", semantic_filter_stats.get("removed_labeled_points", 0))
    print("[3DTool] 低密度/重叠框移除数量:", box_quality_filter_stats.get("removed_boxes", 0))
    print("[3DTool] 低密度/重叠框移除点数量:", box_quality_filter_stats.get("removed_point_count", 0))
    print("[3DTool] 保存目录:", save_dir)

    return result


def generate_3d_annotations_batch(
    root_dir: str,
    segmentation_root: str,
    image_ids: List[int],
    save_root: Optional[str] = None,
    box_type: str = "pca",
    min_points: int = 30,
    semantic_filter_strength: int = 0,
    enable_mask_nms: bool = True,
    mask_nms_iou_threshold: float = 0.65,
    mask_nms_overlap_min_threshold: float = 0.80,
    mask_nms_same_class_iou_threshold: float = 0.90,
    mask_nms_same_class_overlap_min_threshold: float = 0.95
) -> List[Dict]:
    """批量处理多个样本并生成三维标注结果。"""
    results = []

    for image_id in image_ids:
        sample_name = f"{image_id:06d}"
        segmentation_dir = os.path.join(segmentation_root, f"{sample_name}_segmentation")

        if not os.path.isdir(segmentation_dir):
            print(f"[Batch] 跳过，未找到 segmentation_dir: {segmentation_dir}")
            continue

        try:
            result = generate_3d_annotations_for_one_sample(
                root_dir=root_dir,
                image_id=image_id,
                segmentation_dir=segmentation_dir,
                save_root=save_root,
                box_type=box_type,
                min_points=min_points,
                semantic_filter_strength=semantic_filter_strength,
                enable_mask_nms=enable_mask_nms,
                mask_nms_iou_threshold=mask_nms_iou_threshold,
                mask_nms_overlap_min_threshold=mask_nms_overlap_min_threshold,
                mask_nms_same_class_iou_threshold=mask_nms_same_class_iou_threshold,
                mask_nms_same_class_overlap_min_threshold=mask_nms_same_class_overlap_min_threshold
            )
            results.append(result)

        except Exception as e:
            print(f"[Batch] 处理失败: {sample_name}")
            print(f"[Batch] 错误: {e}")

    return results
