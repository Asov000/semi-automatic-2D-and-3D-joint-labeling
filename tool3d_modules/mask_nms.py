# -*- coding: utf-8 -*-
"""二维语义 mask 抑制模块，删除重叠、包含或重复的 mask。"""

from typing import Dict, List, Tuple

import numpy as np


def compute_mask_overlap(mask_a, mask_b) -> Dict:
    """计算两个二维 mask 的交集、并集、交并比和覆盖率。"""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)

    inter = int(np.logical_and(a, b).sum())
    area_a = int(a.sum())
    area_b = int(b.sum())
    union = area_a + area_b - inter
    min_area = max(min(area_a, area_b), 1)

    return {
        "intersection": inter,
        "iou": float(inter / max(union, 1)),
        "overlap_min": float(inter / min_area),
        "area_a": area_a,
        "area_b": area_b,
    }


def semantic_mask_nms(
    items: List[Dict],
    iou_threshold: float = 0.65,
    overlap_min_threshold: float = 0.80,
    suppress_same_class: bool = False,
    same_class_iou_threshold: float = 0.90,
    same_class_overlap_min_threshold: float = 0.95
) -> Tuple[List[Dict], Dict]:
    """对二维语义 mask 执行非极大值抑制。"""
    if len(items) <= 1:
        return items, {
            "enabled": True,
            "iou_threshold": float(iou_threshold),
            "overlap_min_threshold": float(overlap_min_threshold),
            "suppress_same_class": bool(suppress_same_class),
            "same_class_iou_threshold": float(same_class_iou_threshold),
            "same_class_overlap_min_threshold": float(same_class_overlap_min_threshold),
            "before": int(len(items)),
            "after": int(len(items)),
            "removed": [],
        }

    order = sorted(
        range(len(items)),
        key=lambda i: (
            float(items[i].get("score", 1.0)),
            int(np.asarray(items[i]["mask"]).astype(bool).sum())
        ),
        reverse=True
    )

    kept_indices = []
    removed = []

    for idx in order:
        item = items[idx]
        class_name = str(item.get("class_name", ""))
        suppress_info = None

        for kept_idx in kept_indices:
            kept_item = items[kept_idx]
            kept_class_name = str(kept_item.get("class_name", ""))
            same_class = kept_class_name == class_name

            if same_class and not suppress_same_class:
                continue

            overlap = compute_mask_overlap(item["mask"], kept_item["mask"])

            if same_class:
                should_suppress = (
                    overlap["iou"] >= float(same_class_iou_threshold)
                    or overlap["overlap_min"] >= float(same_class_overlap_min_threshold)
                )
            else:
                should_suppress = (
                    overlap["iou"] >= float(iou_threshold)
                    or overlap["overlap_min"] >= float(overlap_min_threshold)
                )

            if should_suppress:
                suppress_info = {
                    "removed_index": int(idx),
                    "removed_class": class_name,
                    "removed_score": float(item.get("score", 1.0)),
                    "kept_index": int(kept_idx),
                    "kept_class": kept_class_name,
                    "kept_score": float(kept_item.get("score", 1.0)),
                    "iou": overlap["iou"],
                    "overlap_min": overlap["overlap_min"],
                    "intersection": overlap["intersection"],
                    "reason": "semantic_mask_nms_overlap",
                }
                break

        if suppress_info is None:
            kept_indices.append(idx)
        else:
            removed.append(suppress_info)

    kept_indices = sorted(kept_indices)
    filtered_items = [items[idx] for idx in kept_indices]

    stats = {
        "enabled": True,
        "iou_threshold": float(iou_threshold),
        "overlap_min_threshold": float(overlap_min_threshold),
        "suppress_same_class": bool(suppress_same_class),
        "same_class_iou_threshold": float(same_class_iou_threshold),
        "same_class_overlap_min_threshold": float(same_class_overlap_min_threshold),
        "before": int(len(items)),
        "after": int(len(filtered_items)),
        "removed": removed,
    }

    return filtered_items, stats
