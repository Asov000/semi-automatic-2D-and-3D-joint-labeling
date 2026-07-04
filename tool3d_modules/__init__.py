# -*- coding: utf-8 -*-
"""三维标注工具包导出入口，兼容旧版工具调用方式。"""

from .assignment import assign_3d_points_to_2d_masks
from .boxes import (
    add_density_info_to_box,
    apply_box_density_filter_and_nms,
    build_3d_boxes_from_segments,
    compute_aabb_3d,
    compute_aabb_iou_3d,
    compute_box_volume_from_size,
    compute_pca_obb_3d,
    filter_points_by_percentile,
    get_box_enclosing_aabb,
)
from .common import ensure_dir, jsonable, to_binary_mask
from .export import save_3d_boxes_json, save_3d_boxes_txt, save_labeled_pointcloud_ply, save_point_masks
from .io import (
    find_classes_json,
    load_binary_masks_from_segmentation_dir,
    load_calib,
    load_detection_class_mapping,
    load_mat_points,
    recover_class_name_from_safe_name,
)
from .mask_nms import compute_mask_overlap, semantic_mask_nms
from .pipeline import generate_3d_annotations_batch, generate_3d_annotations_for_one_sample
from .projection import filter_visible_points_by_zbuffer, project_points_to_image_with_indices
from .semantic_filter import (
    _connected_components_from_neighbor_lists,
    filter_semantic_outliers_3d,
    get_semantic_filter_params,
)
from .visualization import (
    build_open3d_box_center_sphere,
    build_open3d_box_lineset,
    build_open3d_labeled_pointcloud,
    get_class_color_by_id,
    get_default_color_palette,
    get_instance_color_by_id,
    normalize_color_to_open3d,
    visualize_labeled_pointcloud_and_boxes,
    visualize_result_from_memory,
    visualize_saved_3d_annotation,
)

__all__ = [name for name in globals() if not name.startswith('__')]
