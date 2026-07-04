# -*- coding: utf-8 -*-
"""二维框工具模块，负责 mask 到 YOLO 和 Pascal VOC 标注的转换。"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

import numpy as np

# 将输入 mask 统一转换为二值 mask
from .mask import to_binary_mask

# ArrayLike 一般表示 numpy 数组、list 等可转数组的数据类型
from .types import ArrayLike


def mask_to_bbox_xyxy(
    mask: ArrayLike,
    inclusive: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """
    根据二值 mask 计算目标外接矩形框。

    参数：
    mask:
        输入 mask，可以是 0/1、0/255 或其他可转为二值图的数组。

    inclusive:
        是否使用闭区间坐标。
        True:
            返回 [xmin, ymin, xmax, ymax]，
            其中 xmax 和 ymax 表示前景区域实际占据的最后一个像素。
        False:
            返回 [xmin, ymin, xmax + 1, ymax + 1]，
            更接近 Python 切片的左闭右开格式。

    返回：
    bbox:
        (xmin, ymin, xmax, ymax)

    如果 mask 为空，则返回 None。
    """

    # 先将输入 mask 转为标准二值 mask
    binary = to_binary_mask(mask)

    # 找到所有前景像素的位置
    # ys 是行坐标，也就是 y
    # xs 是列坐标，也就是 x
    ys, xs = np.where(binary > 0)

    # 如果没有任何前景点，说明 mask 为空
    if len(xs) == 0 or len(ys) == 0:
        return None

    # 计算前景区域的最小/最大 x、y 坐标
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    # inclusive=True 时，直接返回前景像素覆盖范围
    if inclusive:
        return xmin, ymin, xmax, ymax

    # inclusive=False 时，将右下角扩展 1 像素
    # 使 bbox 变成类似 Python 切片的 [xmin:xmax, ymin:ymax] 格式
    return xmin, ymin, xmax + 1, ymax + 1


def bbox_xyxy_to_yolo(
    bbox_xyxy: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    class_id: int,
    inclusive: bool = True,
    decimals: int = 6,
) -> str:
    """
    将 xyxy 格式 bbox 转换为 YOLO 标注格式。

    YOLO 格式为：
        class_id x_center y_center width height

    其中：
        x_center、y_center、width、height 都是相对于图像宽高的归一化值。

    参数：
    bbox_xyxy:
        bbox 坐标，格式为 (xmin, ymin, xmax, ymax)。

    image_width:
        图像宽度。

    image_height:
        图像高度。

    class_id:
        类别 ID。

    inclusive:
        bbox_xyxy 是否为闭区间坐标。
        如果为 True，则 xmax/ymax 需要 +1 后再计算宽高。

    decimals:
        YOLO 小数保留位数。

    返回：
    YOLO 格式字符串，例如：
        "0 0.512345 0.421875 0.125000 0.250000"
    """

    # 图像宽高必须合法
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于 0")

    # 解包 xyxy 坐标
    xmin, ymin, xmax, ymax = bbox_xyxy

    # 如果 bbox 是闭区间：
    # xmax/ymax 表示最后一个有效像素
    # 因此计算宽高时需要 +1
    if inclusive:
        x1 = xmin
        y1 = ymin
        x2 = xmax + 1
        y2 = ymax + 1

    # 如果 bbox 已经是左闭右开格式，则直接使用
    else:
        x1 = xmin
        y1 = ymin
        x2 = xmax
        y2 = ymax

    # 将 bbox 限制在图像范围内，防止越界
    x1 = max(0, min(float(x1), float(image_width)))
    x2 = max(0, min(float(x2), float(image_width)))
    y1 = max(0, min(float(y1), float(image_height)))
    y2 = max(0, min(float(y2), float(image_height)))

    # 计算 bbox 宽高
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)

    # 计算中心点并归一化
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height

    # 计算宽高并归一化
    norm_w = box_w / image_width
    norm_h = box_h / image_height

    # 防止由于异常坐标导致归一化值超出 0~1
    values = [x_center, y_center, norm_w, norm_h]
    values = [max(0.0, min(1.0, v)) for v in values]

    # 拼接成 YOLO 标签格式字符串
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
    """
    直接从 mask 生成 YOLO 格式 bbox 标签。

    参数：
    mask:
        输入 mask。

    class_id:
        类别 ID。

    image_width:
        图像宽度。
        如果不传，则默认使用 mask 的宽度。

    image_height:
        图像高度。
        如果不传，则默认使用 mask 的高度。

    decimals:
        YOLO 坐标小数保留位数。

    返回：
    YOLO 格式字符串。
    如果 mask 为空，则返回 None。
    """

    # 将 mask 转成标准二值图
    binary = to_binary_mask(mask)

    # 获取 mask 尺寸
    h, w = binary.shape[:2]

    # 如果未指定图像宽高，则默认使用 mask 自身尺寸
    if image_width is None:
        image_width = w
    if image_height is None:
        image_height = h

    # 根据 mask 计算 xyxy bbox
    bbox = mask_to_bbox_xyxy(binary, inclusive=True)

    # 空 mask 无法生成 bbox
    if bbox is None:
        return None

    # 将 xyxy bbox 转换为 YOLO 格式
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
    """
    直接从 mask 生成 Pascal VOC 格式的 object 字典。

    参数：
    mask:
        输入 mask。

    class_name:
        类别名称。

    difficult:
        Pascal VOC 中的 difficult 字段。
        一般 0 表示不困难，1 表示困难样本。

    truncated:
        Pascal VOC 中的 truncated 字段。
        一般 0 表示目标未被截断，1 表示目标被截断。

    pose:
        Pascal VOC 中的 pose 字段。
        默认使用 "Unspecified"。

    返回：
    VOC object 字典，例如：
    {
        "name": "chair",
        "pose": "Unspecified",
        "truncated": 0,
        "difficult": 0,
        "bndbox": {
            "xmin": 10,
            "ymin": 20,
            "xmax": 100,
            "ymax": 200
        }
    }

    如果 mask 为空，则返回 None。
    """

    # 根据 mask 计算 bbox
    bbox = mask_to_bbox_xyxy(mask, inclusive=True)

    # 空 mask 直接返回 None
    if bbox is None:
        return None

    # 解包 bbox
    xmin, ymin, xmax, ymax = bbox

    # 构造 Pascal VOC object 字典
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
    """
    构建 Pascal VOC XML 标注字符串。

    参数：
    filename:
        图像文件名，例如 "000001.jpg"。

    image_width:
        图像宽度。

    image_height:
        图像高度。

    objects:
        目标列表。
        每个目标一般由 mask_to_voc_bbox() 生成。

    folder:
        图像所在文件夹名称。

    image_depth:
        图像通道数。
        RGB 图像一般为 3。

    segmented:
        VOC 中的 segmented 字段。
        1 表示该图像有分割标注，0 表示没有。

    返回：
    xml_str:
        格式化后的 Pascal VOC XML 字符串。
    """

    # 创建根节点 <annotation>
    annotation = ET.Element("annotation")

    # 写入 <folder>
    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = folder

    # 写入 <filename>
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename

    # 写入 <size>
    size_elem = ET.SubElement(annotation, "size")

    # 图像宽度
    width_elem = ET.SubElement(size_elem, "width")
    width_elem.text = str(int(image_width))

    # 图像高度
    height_elem = ET.SubElement(size_elem, "height")
    height_elem.text = str(int(image_height))

    # 图像通道数
    depth_elem = ET.SubElement(size_elem, "depth")
    depth_elem.text = str(int(image_depth))

    # 写入 <segmented>
    segmented_elem = ET.SubElement(annotation, "segmented")
    segmented_elem.text = str(int(segmented))

    # 遍历所有目标，写入 <object>
    for obj in objects:
        # 跳过空目标
        if obj is None:
            continue

        # 创建 object 节点
        obj_elem = ET.SubElement(annotation, "object")

        # 类别名称
        name_elem = ET.SubElement(obj_elem, "name")
        name_elem.text = str(obj["name"])

        # 姿态字段
        pose_elem = ET.SubElement(obj_elem, "pose")
        pose_elem.text = str(obj.get("pose", "Unspecified"))

        # 是否截断
        truncated_elem = ET.SubElement(obj_elem, "truncated")
        truncated_elem.text = str(int(obj.get("truncated", 0)))

        # 是否困难样本
        difficult_elem = ET.SubElement(obj_elem, "difficult")
        difficult_elem.text = str(int(obj.get("difficult", 0)))

        # bbox 节点
        bndbox_elem = ET.SubElement(obj_elem, "bndbox")
        box = obj["bndbox"]

        # 依次写入 xmin / ymin / xmax / ymax
        for key in ["xmin", "ymin", "xmax", "ymax"]:
            elem = ET.SubElement(bndbox_elem, key)
            elem.text = str(int(box[key]))

    # 将 ElementTree 转成原始 XML 字符串
    rough_string = ET.tostring(annotation, encoding="utf-8")

    # 使用 minidom 格式化 XML，使其更易读
    parsed = minidom.parseString(rough_string)

    # 返回带缩进的 XML 字符串
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
    """
    保存 Pascal VOC XML 标注文件。

    参数：
    save_path:
        XML 文件保存路径。

    filename:
        图像文件名。

    image_width:
        图像宽度。

    image_height:
        图像高度。

    objects:
        VOC object 列表。

    folder:
        图像所在文件夹名称。

    image_depth:
        图像通道数。

    segmented:
        VOC segmented 字段。

    返回：
    无返回值，直接将 XML 写入 save_path。
    """

    # 创建 XML 保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 构建 VOC XML 字符串
    xml_str = build_voc_xml_string(
        filename=filename,
        image_width=image_width,
        image_height=image_height,
        objects=objects,
        folder=folder,
        image_depth=image_depth,
        segmented=segmented,
    )

    # 写入 XML 文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(xml_str)