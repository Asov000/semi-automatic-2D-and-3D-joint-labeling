# -*- coding: utf-8 -*-
"""已完成 mask 转换模块，负责批量生成检测框和导出文本。"""

import os
from typing import Dict, List, Optional

from .bbox import mask_to_voc_bbox, mask_to_yolo_bbox
from .mask import refine_mask


def refine_saved_masks(
    saved_masks: List[Dict],
    **refine_kwargs,
) -> List[Dict]:
    new_items = []

    for item in saved_masks:
        new_item = dict(item)
        new_item["mask"] = refine_mask(item["mask"], **refine_kwargs)
        new_items.append(new_item)

    return new_items


def saved_masks_to_yolo_lines(
    saved_masks: List[Dict],
    class_to_id: Dict[str, int],
    image_width: int,
    image_height: int,
    decimals: int = 6,
    use_refined_mask: bool = False,
    refine_kwargs: Optional[Dict] = None,
) -> List[str]:
    if refine_kwargs is None:
        refine_kwargs = {}

    lines = []

    for item in saved_masks:
        class_name = item["class_name"]
        if class_name not in class_to_id:
            raise KeyError(f"类别 {class_name} 不在 class_to_id 中")

        mask = item["mask"]

        if use_refined_mask:
            mask = refine_mask(mask, **refine_kwargs)

        line = mask_to_yolo_bbox(
            mask,
            class_id=class_to_id[class_name],
            image_width=image_width,
            image_height=image_height,
            decimals=decimals,
        )

        if line is not None:
            lines.append(line)

    return lines


def saved_masks_to_voc_objects(
    saved_masks: List[Dict],
    use_refined_mask: bool = False,
    refine_kwargs: Optional[Dict] = None,
) -> List[Dict]:
    if refine_kwargs is None:
        refine_kwargs = {}

    objects = []

    for item in saved_masks:
        class_name = item["class_name"]
        mask = item["mask"]

        if use_refined_mask:
            mask = refine_mask(mask, **refine_kwargs)

        obj = mask_to_voc_bbox(mask, class_name=class_name)

        if obj is not None:
            objects.append(obj)

    return objects


def save_yolo_txt(save_path: str, yolo_lines: List[str]) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")
