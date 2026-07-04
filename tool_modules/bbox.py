# -*- coding: utf-8 -*-
"""二维框工具模块，负责 mask 到 YOLO 和 Pascal VOC 标注的转换。"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

import numpy as np

from .mask import to_binary_mask
from .types import ArrayLike


def mask_to_bbox_xyxy(
    mask: ArrayLike,
    inclusive: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    binary = to_binary_mask(mask)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    if inclusive:
        return xmin, ymin, xmax, ymax

    return xmin, ymin, xmax + 1, ymax + 1


def bbox_xyxy_to_yolo(
    bbox_xyxy: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    class_id: int,
    inclusive: bool = True,
    decimals: int = 6,
) -> str:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于 0")

    xmin, ymin, xmax, ymax = bbox_xyxy

    if inclusive:
        x1 = xmin
        y1 = ymin
        x2 = xmax + 1
        y2 = ymax + 1
    else:
        x1 = xmin
        y1 = ymin
        x2 = xmax
        y2 = ymax

    x1 = max(0, min(float(x1), float(image_width)))
    x2 = max(0, min(float(x2), float(image_width)))
    y1 = max(0, min(float(y1), float(image_height)))
    y2 = max(0, min(float(y2), float(image_height)))

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_w = box_w / image_width
    norm_h = box_h / image_height

    values = [x_center, y_center, norm_w, norm_h]
    values = [max(0.0, min(1.0, v)) for v in values]

    return (
        f"{int(class_id)} "
        f"{values[0]:.{decimals}f} "
        f"{values[1]:.{decimals}f} "
        f"{values[2]:.{decimals}f} "
        f"{values[3]:.{decimals}f}"
    )


def mask_to_yolo_bbox(
    mask: ArrayLike,
    class_id: int,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    decimals: int = 6,
) -> Optional[str]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    binary = to_binary_mask(mask)
    h, w = binary.shape[:2]

    if image_width is None:
        image_width = w
    if image_height is None:
        image_height = h

    bbox = mask_to_bbox_xyxy(binary, inclusive=True)

    if bbox is None:
        return None

    return bbox_xyxy_to_yolo(
        bbox,
        image_width=image_width,
        image_height=image_height,
        class_id=class_id,
        inclusive=True,
        decimals=decimals,
    )


def mask_to_voc_bbox(
    mask: ArrayLike,
    class_name: str,
    difficult: int = 0,
    truncated: int = 0,
    pose: str = "Unspecified",
) -> Optional[Dict]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    bbox = mask_to_bbox_xyxy(mask, inclusive=True)

    if bbox is None:
        return None

    xmin, ymin, xmax, ymax = bbox

    return {
        "name": str(class_name),
        "pose": pose,
        "truncated": int(truncated),
        "difficult": int(difficult),
        "bndbox": {
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax),
        },
    }


def build_voc_xml_string(
    filename: str,
    image_width: int,
    image_height: int,
    objects: List[Dict],
    folder: str = "",
    image_depth: int = 3,
    segmented: int = 1,
) -> str:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    annotation = ET.Element("annotation")

    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = folder

    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename

    size_elem = ET.SubElement(annotation, "size")

    width_elem = ET.SubElement(size_elem, "width")
    width_elem.text = str(int(image_width))

    height_elem = ET.SubElement(size_elem, "height")
    height_elem.text = str(int(image_height))

    depth_elem = ET.SubElement(size_elem, "depth")
    depth_elem.text = str(int(image_depth))

    segmented_elem = ET.SubElement(annotation, "segmented")
    segmented_elem.text = str(int(segmented))

    for obj in objects:
        if obj is None:
            continue

        obj_elem = ET.SubElement(annotation, "object")

        name_elem = ET.SubElement(obj_elem, "name")
        name_elem.text = str(obj["name"])

        pose_elem = ET.SubElement(obj_elem, "pose")
        pose_elem.text = str(obj.get("pose", "Unspecified"))

        truncated_elem = ET.SubElement(obj_elem, "truncated")
        truncated_elem.text = str(int(obj.get("truncated", 0)))

        difficult_elem = ET.SubElement(obj_elem, "difficult")
        difficult_elem.text = str(int(obj.get("difficult", 0)))

        bndbox_elem = ET.SubElement(obj_elem, "bndbox")
        box = obj["bndbox"]

        for key in ["xmin", "ymin", "xmax", "ymax"]:
            elem = ET.SubElement(bndbox_elem, key)
            elem.text = str(int(box[key]))

    rough_string = ET.tostring(annotation, encoding="utf-8")
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def save_voc_xml(
    save_path: str,
    filename: str,
    image_width: int,
    image_height: int,
    objects: List[Dict],
    folder: str = "",
    image_depth: int = 3,
    segmented: int = 1,
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    xml_str = build_voc_xml_string(
        filename=filename,
        image_width=image_width,
        image_height=image_height,
        objects=objects,
        folder=folder,
        image_depth=image_depth,
        segmented=segmented,
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
