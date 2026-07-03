# -*- coding: utf-8 -*-
"""
mask_tool_functions.py

纯工具函数：不依赖 PyQt 前端。

功能：
1. 对标注后的二值掩膜进行后处理：
   - 去除小噪声区域
   - 填充内部空洞
   - 小范围闭运算修补边界裂缝
   - 限制外部扩张，避免 mask 蔓延过多

2. 从 mask 计算 2D bbox，并转换为 YOLO 检测框格式。

3. 从 mask 计算 2D bbox，并转换为 Pascal VOC 检测框格式 / XML。
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


ArrayLike = Union[np.ndarray, List[List[int]]]


# ============================================================
# 基础 mask 工具
# ============================================================

def to_binary_mask(mask: ArrayLike, threshold: float = 0) -> np.ndarray:
    if mask is None:
        raise ValueError("mask 不能为 None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            # 如果误传了 RGB mask，这里取最大通道作为前景判断
            arr = np.max(arr, axis=2)

    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D 或单通道图像，当前 shape={arr.shape}")

    binary = (arr > threshold).astype(np.uint8)
    return binary


def _make_odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _ellipse_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = _make_odd_kernel_size(kernel_size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def remove_small_components(
    mask: ArrayLike,
    min_area: int = 64,
    keep_largest: bool = False
) -> np.ndarray:
    """
    去除前景中的小连通域噪声。

    参数：
        mask:
            输入二值 mask。
        min_area:
            小于该面积的连通域会被删除。
        keep_largest:
            True 时只保留最大连通域，适合 SAM 单目标 mask。
            False 时保留所有面积 >= min_area 的连通域。

    返回：
        cleaned_mask: 0/1 uint8
    """
    binary = to_binary_mask(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    output = np.zeros_like(binary, dtype=np.uint8)

    component_ids = list(range(1, num_labels))
    areas = [stats[i, cv2.CC_STAT_AREA] for i in component_ids]

    if keep_largest:
        largest_idx = int(np.argmax(areas))
        largest_label = component_ids[largest_idx]
        largest_area = areas[largest_idx]

        if min_area <= 0 or largest_area >= min_area:
            output[labels == largest_label] = 1
    else:
        for comp_id, area in zip(component_ids, areas):
            if area >= min_area:
                output[labels == comp_id] = 1

    return output


def fill_mask_holes(mask: ArrayLike) -> np.ndarray:
    """
    填充 mask 内部空洞。

    原理：
        将背景区域做连通域分析。
        与图像边界相连的背景认为是真背景；
        不与边界相连的背景认为是 mask 内部空洞，将其填成前景。

    返回：
        filled_mask: 0/1 uint8
    """
    binary = to_binary_mask(mask)

    background = (binary == 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    if num_labels <= 1:
        return binary

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    output = binary.copy()

    for label_id in range(1, num_labels):
        if label_id not in border_labels:
            output[labels == label_id] = 1

    return output.astype(np.uint8)


def refine_mask(
    mask: ArrayLike,
    min_area: int = 64,
    keep_largest: bool = True,
    close_kernel_size: int = 5,
    close_iterations: int = 1,
    open_kernel_size: int = 0,
    open_iterations: int = 1,
    fill_holes: bool = True,
    max_external_expand_px: int = 0
) -> np.ndarray:
    """
    对 SAM 标注后的二值 mask 做后处理。

    目标：
        1. 内部不应有空洞；
        2. 外部轮廓尽量闭合；
        3. 不让 mask 向外大范围蔓延到不该包含的区域。

    处理流程：
        1. 二值化；
        2. 删除小连通域 / 可选只保留最大连通域；
        3. 可选开运算，去除细小毛刺；
        4. 闭运算，修补小裂缝和小缺口；
        5. 填充内部空洞；
        6. 限制外部扩张范围；
        7. 再次去小区域并填洞。

    参数：
        mask:
            输入 mask。
        min_area:
            小连通域过滤阈值。
        keep_largest:
            True 时只保留最大连通域，适合单个目标。
        close_kernel_size:
            闭运算核大小。越大越容易闭合缺口，但也越可能外扩。
            建议 3 或 5。
        close_iterations:
            闭运算次数。
        open_kernel_size:
            开运算核大小。默认 0 表示不开启。
        open_iterations:
            开运算次数。
        fill_holes:
            是否填充内部空洞。
        max_external_expand_px:
            允许 mask 外部最多扩张多少像素。
            - 0：最严格，外部不允许扩张，只允许填内部洞；
            - 1~3：允许小范围修补边界；
            - None 或负数：不限制外部扩张，不建议默认使用。

    返回：
        refined_mask: 0/1 uint8
    """
    original = to_binary_mask(mask)

    # 先去掉明显小噪声
    clean = remove_small_components(
        original,
        min_area=min_area,
        keep_largest=keep_largest
    )

    # 可选开运算：去除细小毛刺
    if open_kernel_size and open_kernel_size > 1:
        open_kernel = _ellipse_kernel(open_kernel_size)
        clean = cv2.morphologyEx(
            clean,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=max(1, int(open_iterations))
        )
        clean = to_binary_mask(clean)

    # 闭运算：修补边界裂缝和小缺口
    if close_kernel_size and close_kernel_size > 1:
        close_kernel = _ellipse_kernel(close_kernel_size)
        closed = cv2.morphologyEx(
            clean,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=max(1, int(close_iterations))
        )
        closed = to_binary_mask(closed)
    else:
        closed = clean.copy()

    if fill_holes:
        filled = fill_mask_holes(closed)
    else:
        filled = closed.copy()

    # 限制外部扩张：
    # - 闭运算可能会让边界向外扩张；
    # - 但内部空洞填充是合理的，不应该被 max_external_expand_px 限制。
    if max_external_expand_px is not None and max_external_expand_px >= 0:
        if max_external_expand_px == 0:
            allowed_external = clean.astype(bool)
        else:
            px = int(max_external_expand_px)
            allow_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * px + 1, 2 * px + 1)
            )
            allowed_external = cv2.dilate(clean, allow_kernel, iterations=1).astype(bool)

        # holes_to_fill 是由填洞新增的区域，属于内部区域，可以保留
        holes_to_fill = filled.astype(bool) & (~closed.astype(bool))

        # closed / filled 中由闭运算造成的外部扩张，需要受 allowed_external 限制
        refined = ((filled.astype(bool) & allowed_external) | holes_to_fill).astype(np.uint8)
    else:
        refined = filled.astype(np.uint8)

    # 后处理后再过滤一次小连通域
    refined = remove_small_components(
        refined,
        min_area=min_area,
        keep_largest=keep_largest
    )

    if fill_holes:
        refined = fill_mask_holes(refined)

    return refined.astype(np.uint8)


# ============================================================
# bbox / YOLO / VOC
# ============================================================

def mask_to_bbox_xyxy(
    mask: ArrayLike,
    inclusive: bool = True
) -> Optional[Tuple[int, int, int, int]]:
    """
    从 mask 计算外接 2D 框。

    参数：
        mask:
            输入 mask。
        inclusive:
            True:
                返回 Pascal VOC 风格坐标：
                (xmin, ymin, xmax, ymax)，其中 xmax/ymax 是前景像素最大坐标。
            False:
                返回 half-open 坐标：
                (xmin, ymin, xmax_exclusive, ymax_exclusive)。

    返回：
        bbox:
            若 mask 为空，返回 None。
    """
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
    else:
        return xmin, ymin, xmax + 1, ymax + 1


def bbox_xyxy_to_yolo(
    bbox_xyxy: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    class_id: int,
    inclusive: bool = True,
    decimals: int = 6
) -> str:
    """
    将 xyxy bbox 转换为 YOLO 检测框格式。

    YOLO 格式：
        class_id x_center y_center width height

    坐标全部归一化到 0~1。

    参数：
        bbox_xyxy:
            (xmin, ymin, xmax, ymax)
        image_width:
            图像宽度 W
        image_height:
            图像高度 H
        class_id:
            类别 ID
        inclusive:
            bbox_xyxy 是否为 VOC 式闭区间坐标。
        decimals:
            小数保留位数。

    返回：
        yolo_line: str
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于 0")

    xmin, ymin, xmax, ymax = bbox_xyxy

    if inclusive:
        # VOC 式闭区间转 half-open 区间
        x1 = xmin
        y1 = ymin
        x2 = xmax + 1
        y2 = ymax + 1
    else:
        x1 = xmin
        y1 = ymin
        x2 = xmax
        y2 = ymax

    # 裁剪到图像范围
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
    decimals: int = 6
) -> Optional[str]:
    """
    直接从 mask 生成 YOLO 检测框格式。

    返回：
        "class_id x_center y_center width height"
        如果 mask 为空，返回 None。
    """
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
        decimals=decimals
    )


def mask_to_voc_bbox(
    mask: ArrayLike,
    class_name: str,
    difficult: int = 0,
    truncated: int = 0,
    pose: str = "Unspecified"
) -> Optional[Dict]:
    """
    直接从 mask 生成 Pascal VOC 风格的 object 标注。

    VOC bbox 格式：
        xmin, ymin, xmax, ymax

    注意：
        这里的 xmax/ymax 是前景像素的最大坐标，和 Pascal VOC 习惯一致。

    返回：
        dict:
        {
            "name": class_name,
            "pose": "Unspecified",
            "truncated": 0,
            "difficult": 0,
            "bndbox": {
                "xmin": ...,
                "ymin": ...,
                "xmax": ...,
                "ymax": ...
            }
        }

        如果 mask 为空，返回 None。
    """
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
            "ymax": int(ymax)
        }
    }


def build_voc_xml_string(
    filename: str,
    image_width: int,
    image_height: int,
    objects: List[Dict],
    folder: str = "",
    image_depth: int = 3,
    segmented: int = 1
) -> str:
    """
    根据 VOC object 列表生成 Pascal VOC XML 字符串。

    参数：
        filename:
            图片文件名，如 test.jpg。
        image_width/image_height:
            图像宽高。
        objects:
            mask_to_voc_bbox 返回的 dict 列表。
        folder:
            VOC XML 中的 folder 字段。
        image_depth:
            通道数，RGB 通常为 3。
        segmented:
            VOC segmented 字段，分割任务可设为 1。

    返回：
        xml_str: 格式化后的 XML 字符串。
    """
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
    segmented: int = 1
) -> None:
    """
    保存 Pascal VOC XML 文件。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    xml_str = build_voc_xml_string(
        filename=filename,
        image_width=image_width,
        image_height=image_height,
        objects=objects,
        folder=folder,
        image_depth=image_depth,
        segmented=segmented
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


# ============================================================
# 批量处理 saved_masks 的辅助函数
# ============================================================

def refine_saved_masks(
    saved_masks: List[Dict],
    **refine_kwargs
) -> List[Dict]:
    """
    批量处理前端 self.saved_masks 风格的数据。

    每个 item 至少需要包含：
        item["mask"]
        item["class_name"]

    返回：
        new_saved_masks:
            和原 saved_masks 结构一致，但 mask 被替换为后处理后的 mask。
    """
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
    refine_kwargs: Optional[Dict] = None
) -> List[str]:
    """
    将 saved_masks 批量转换为 YOLO 检测框 txt 行。

    参数：
        saved_masks:
            前端保存的 mask 列表。
        class_to_id:
            类别名到 ID 的映射，如 {"chair": 0, "table": 1}
        image_width/image_height:
            原图宽高。
        use_refined_mask:
            是否先对 mask 做 refine_mask 后再算框。
        refine_kwargs:
            refine_mask 参数。

    返回：
        yolo_lines:
            ["0 0.512 0.433 0.221 0.382", ...]
    """
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
            decimals=decimals
        )

        if line is not None:
            lines.append(line)

    return lines


def saved_masks_to_voc_objects(
    saved_masks: List[Dict],
    use_refined_mask: bool = False,
    refine_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """
    将 saved_masks 批量转换为 VOC object 列表。
    """
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
    """
    保存 YOLO 检测框 txt。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")



