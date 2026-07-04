# -*- coding: utf-8 -*-
"""二维分割结果检查脚本，验证 mask、检测框、导出文本和可视化结果。"""

import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 从 tool_modules 中导入已有的工具函数
# 这里使用别名，避免和当前脚本中封装的函数重名
from tool_modules import (
    bbox_xyxy_to_yolo as _bbox_xyxy_to_yolo_line,  # 将 xyxy 框转换为 YOLO 格式字符串
    mask_to_bbox_xyxy as _mask_to_bbox_xyxy,       # 根据 mask 计算 xyxy 外接框
    mask_to_voc_bbox as _mask_to_voc_bbox,         # 根据 mask 计算 VOC 格式 bbox
    refine_mask as _refine_mask_binary,            # 对二值 mask 做后处理
)

# SAM 输出目录
SEG_DIR = r"C:\Users\25918\Desktop\sam_output\test2_segmentation"

# 二值 mask 保存目录
BINARY_DIR = os.path.join(SEG_DIR, "binary_masks")

# 测试处理后的结果保存目录
OUTPUT_DIR = os.path.join(SEG_DIR, "processed_test")


def refine_mask(
    mask,
    min_area=64,
    keep_largest=False,
    close_kernel_size=5,
    fill_holes=True,
    max_external_expand_px=1,
):
    """
    对原始 mask 进行后处理。

    参数：
    mask:
        输入二值 mask，可以是 0/255 或 0/1 格式。

    min_area:
        最小连通区域面积，小于该面积的区域会被过滤掉。

    keep_largest:
        是否只保留最大连通区域。
        如果目标只有一个实例，可以设为 True；
        如果同类有多个实例，建议 False。

    close_kernel_size:
        闭运算核大小，用于连接小断裂、平滑边界。

    fill_holes:
        是否填充 mask 内部空洞。

    max_external_expand_px:
        最大外扩像素，控制 mask 外扩范围，避免后处理过度膨胀。

    返回：
    refined:
        后处理后的二值 mask，格式为 0/255。
    """

    # 调用 tool_modules 中已有的 refine_mask 函数
    refined = _refine_mask_binary(
        mask,
        min_area=min_area,
        keep_largest=keep_largest,
        close_kernel_size=close_kernel_size,
        fill_holes=fill_holes,
        max_external_expand_px=max_external_expand_px,
    )

    # 统一转换为 0/255 的 uint8 二值图
    return (refined > 0).astype(np.uint8) * 255


def mask_to_bbox_xyxy(mask):
    """
    根据二值 mask 计算 xyxy 格式外接框。

    参数：
    mask:
        输入二值 mask。

    返回：
    bbox:
        [xmin, ymin, xmax, ymax]
        如果 mask 为空，则返回 None。
    """

    # inclusive=True 表示 xmax / ymax 包含边界像素
    return _mask_to_bbox_xyxy(mask, inclusive=True)


def bbox_xyxy_to_yolo(bbox, image_width, image_height, class_id):
    """
    将 xyxy 格式 bbox 转换为 YOLO 格式。

    参数：
    bbox:
        [xmin, ymin, xmax, ymax]

    image_width:
        图像宽度。

    image_height:
        图像高度。

    class_id:
        YOLO 类别 ID。

    返回：
    tuple:
        (class_id, x_center, y_center, width, height)
        其中 x_center / y_center / width / height 都是归一化后的值。
    """

    # 调用已有工具函数，返回的是字符串，例如：
    # "0 0.512345 0.421231 0.123456 0.234567"
    line = _bbox_xyxy_to_yolo_line(
        bbox,
        image_width=image_width,
        image_height=image_height,
        class_id=class_id,
        inclusive=True,
        decimals=12,
    )

    # 将字符串拆分为各个字段
    parts = line.split()

    # 转成更方便程序使用的数值格式
    return (
        int(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4])
    )


def mask_to_yolo_bbox(mask, image_width, image_height, class_id):
    """
    直接从 mask 计算 YOLO 格式 bbox。

    参数：
    mask:
        输入二值 mask。

    image_width:
        图像宽度。

    image_height:
        图像高度。

    class_id:
        类别 ID。

    返回：
    YOLO bbox:
        (class_id, x_center, y_center, width, height)

    如果 mask 为空，则返回 None。
    """

    # 先由 mask 计算 xyxy 外接框
    bbox = mask_to_bbox_xyxy(mask)

    # 如果 mask 为空，bbox 为 None
    if bbox is None:
        return None

    # 再把 xyxy bbox 转成 YOLO 格式
    return bbox_xyxy_to_yolo(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        class_id=class_id,
    )


def mask_to_voc_bbox(mask, class_name):
    """
    直接从 mask 计算 VOC 格式 bbox。

    参数：
    mask:
        输入二值 mask。

    class_name:
        类别名称。

    返回：
    dict:
        {
            "name": class_name,
            "bndbox": {
                "xmin": ...,
                "ymin": ...,
                "xmax": ...,
                "ymax": ...
            }
        }

    如果 mask 为空，则返回 None。
    """

    # 调用已有工具函数生成 VOC 格式目标信息
    obj = _mask_to_voc_bbox(mask, class_name=class_name)

    if obj is None:
        return None

    # 只保留类别名和 bbox 信息
    return {
        "name": obj["name"],
        "bndbox": obj["bndbox"],
    }


def load_classes_json(seg_dir):
    """
    加载 SAM 分割结果目录中的类别信息 json 文件。

    参数：
    seg_dir:
        分割结果目录。

    返回：
    data:
        json 文件内容。
        如果没有找到 *_classes.json，则返回 None。
    """

    # 查找以 _classes.json 结尾的文件
    json_files = [
        f for f in os.listdir(seg_dir)
        if f.endswith("_classes.json")
    ]

    # 没有类别 json 文件时返回 None
    if len(json_files) == 0:
        return None

    # 默认读取第一个匹配到的 json 文件
    json_path = os.path.join(seg_dir, json_files[0])

    # 读取 json 内容
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_original_image(class_info):
    """
    根据类别 json 中记录的 image_path 读取原始图像。

    参数：
    class_info:
        load_classes_json() 读取出来的类别信息字典。

    返回：
    image_rgb:
        RGB 格式原图。
        如果读取失败，则返回 None。
    """

    # 没有类别信息时，无法读取原图路径
    if class_info is None:
        return None

    # 从 json 中读取原图路径
    image_path = class_info.get("image_path", None)

    if image_path is None:
        return None

    # 判断原图路径是否存在
    if not os.path.exists(image_path):
        print(f"原图路径不存在：{image_path}")
        return None

    # OpenCV 默认读取 BGR 图像
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return None

    # 转成 RGB，方便 matplotlib 显示
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def parse_binary_mask_filename(filename):
    """
    解析二值 mask 文件名，提取实例 ID 和类别名称。

    参数：
    filename:
        mask 文件名。

    文件名预期格式类似：
        xxx_001_chair.png
        xxx_002_table.png

    返回：
    instance_id:
        实例编号，例如 1、2、3。

    class_name:
        类别名称，例如 chair、table。

    如果无法解析，则返回：
        None, "unknown"
    """

    # 去掉文件扩展名
    name = os.path.splitext(filename)[0]

    # 匹配最后的 "_三位数字_类别名"
    # 例如：000616_001_chair
    match = re.search(r"_(\d{3})_(.+)$", name)

    if match is None:
        return None, "unknown"

    # 第一个括号是实例 ID
    instance_id = int(match.group(1))

    # 第二个括号是类别名称
    class_name = match.group(2)

    return instance_id, class_name


def draw_bbox_on_image(image_rgb, bbox, color=(255, 0, 0), thickness=2):
    """
    在 RGB 图像上绘制 bbox。

    参数：
    image_rgb:
        RGB 格式图像。

    bbox:
        [xmin, ymin, xmax, ymax]

    color:
        框颜色，RGB 格式。

    thickness:
        框线宽度。

    返回：
    vis:
        绘制 bbox 后的图像。
    """

    # 复制图像，避免修改原图
    vis = image_rgb.copy()

    # 如果 bbox 为空，直接返回原图
    if bbox is None:
        return vis

    xmin, ymin, xmax, ymax = bbox

    # 绘制矩形框
    # 注意：这里 image_rgb 是 RGB 图像，但 cv2.rectangle 只是写入数值，
    # 所以 color 按 RGB 理解即可。
    cv2.rectangle(
        vis,
        (xmin, ymin),
        (xmax, ymax),
        color,
        thickness
    )

    return vis


def overlay_mask_on_image(image_rgb, mask, color=(255, 0, 0), alpha=0.45):
    """
    将 mask 半透明叠加到原图上。

    参数：
    image_rgb:
        RGB 格式原图。

    mask:
        二值 mask，非零区域表示前景。

    color:
        mask 叠加颜色，RGB 格式。

    alpha:
        透明度。
        越大，mask 颜色越明显。

    返回：
    vis:
        叠加 mask 后的图像。
    """

    # 转成 float32，方便做透明度混合
    vis = image_rgb.copy().astype(np.float32)

    # mask 中大于 0 的区域视为前景区域
    mask_bool = mask > 0

    # 颜色转成数组
    color_arr = np.array(color, dtype=np.float32)

    # 对 mask 前景区域进行 alpha 混合
    # 新像素 = 原像素 * (1 - alpha) + mask颜色 * alpha
    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + color_arr * alpha

    # 限制范围到 0~255，并转回 uint8
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    return vis


def test_segmentation_masks(seg_dir):
    """
    测试指定分割目录下的所有二值 mask。

    主要功能：
    1. 读取 binary_masks 文件夹中的 mask
    2. 对每个 mask 做后处理
    3. 计算 xyxy bbox
    4. 转换为 YOLO bbox
    5. 转换为 VOC bbox
    6. 保存 refined mask
    7. 保存 bbox 可视化图
    8. 保存 YOLO / VOC 标签文本

    参数：
    seg_dir:
        SAM 分割输出目录。
    """

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载类别信息 json
    class_info = load_classes_json(seg_dir)

    # 根据 json 读取原图
    image_rgb = load_original_image(class_info)

    # binary_masks 目录
    binary_dir = os.path.join(seg_dir, "binary_masks")

    # 检查 binary_masks 是否存在
    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"找不到 binary_masks 文件夹：{binary_dir}")

    # 收集所有 mask 图片文件
    mask_files = [
        f for f in os.listdir(binary_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    # 没有 mask 时直接报错
    if len(mask_files) == 0:
        raise FileNotFoundError(f"binary_masks 文件夹里没有 mask 图片：{binary_dir}")

    # 按文件名排序，保证处理顺序稳定
    mask_files = sorted(mask_files)

    # 读取类别映射
    # 预期格式可能类似：
    # {
    #     "background": 0,
    #     "chair": 1,
    #     "table": 2
    # }
    if class_info is not None and "class_to_id" in class_info:
        class_to_id = class_info["class_to_id"]
    else:
        # 如果没有类别 json，则只设置 background
        class_to_id = {"background": 0}

    # YOLO 一般要求类别从 0 开始
    # 而当前 class_to_id 可能把 background 设置为 0，
    # 真实类别从 1 开始，所以这里要减 1。
    yolo_class_to_id = {}

    for class_name, class_id in class_to_id.items():
        # background 不参与 YOLO 训练标签
        if class_name == "background":
            continue

        # 类别 ID 减 1，使真实类别从 0 开始
        yolo_class_to_id[class_name] = int(class_id) - 1

    # 保存所有 YOLO 标签行
    yolo_lines = []

    # 保存所有 VOC 标签行
    voc_lines = []

    print("========== 开始测试 mask ==========")
    print(f"seg_dir: {seg_dir}")
    print(f"binary_masks 数量: {len(mask_files)}")
    print()

    # 遍历每个 mask 文件
    for idx, filename in enumerate(mask_files):
        # 当前 mask 路径
        mask_path = os.path.join(binary_dir, filename)

        # 从文件名解析实例 ID 和类别名称
        instance_id, class_name = parse_binary_mask_filename(filename)

        # 以灰度图方式读取 mask
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 如果读取失败，跳过
        if mask_gray is None:
            print(f"读取失败，跳过：{mask_path}")
            continue

        # mask 高度和宽度
        h, w = mask_gray.shape[:2]

        # 如果无法读取原图，就创建黑底图用于展示
        if image_rgb is None:
            image_rgb_show = np.zeros((h, w, 3), dtype=np.uint8)

        else:
            # 复制原图
            image_rgb_show = image_rgb.copy()

            # 如果原图尺寸和 mask 尺寸不一致，则 resize 到 mask 尺寸
            if image_rgb_show.shape[:2] != (h, w):
                image_rgb_show = cv2.resize(
                    image_rgb_show,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

        # 对原始 mask 做后处理：
        # 1. 去除小面积噪声
        # 2. 闭运算连接断裂区域
        # 3. 填充内部空洞
        # 4. 控制最大外扩范围
        refined_mask = refine_mask(
            mask_gray,
            min_area=64,
            keep_largest=False,
            close_kernel_size=5,
            fill_holes=True,
            max_external_expand_px=1
        )

        # 根据后处理后的 mask 计算外接矩形框
        bbox = mask_to_bbox_xyxy(refined_mask)

        # 如果 mask 为空，则跳过
        if bbox is None:
            print(f"{filename}: 空 mask，跳过")
            continue

        # 根据类别名称获取 YOLO 类别 ID
        # 如果类别没找到，默认设为 0
        class_id = yolo_class_to_id.get(class_name, 0)

        # 由 mask 计算 YOLO bbox
        yolo_bbox = mask_to_yolo_bbox(
            refined_mask,
            image_width=w,
            image_height=h,
            class_id=class_id
        )

        # 拼接成 YOLO 标签行：
        # class_id x_center y_center width height
        yolo_line = (
            f"{yolo_bbox[0]} "
            f"{yolo_bbox[1]:.6f} "
            f"{yolo_bbox[2]:.6f} "
            f"{yolo_bbox[3]:.6f} "
            f"{yolo_bbox[4]:.6f}"
        )

        # 保存 YOLO 标签行
        yolo_lines.append(yolo_line)

        # 根据 mask 计算 VOC 格式目标
        voc_obj = mask_to_voc_bbox(
            refined_mask,
            class_name=class_name
        )

        # 拼接成简化 VOC 标签行：
        # class_name xmin ymin xmax ymax
        voc_line = (
            f"{class_name} "
            f"{voc_obj['bndbox']['xmin']} "
            f"{voc_obj['bndbox']['ymin']} "
            f"{voc_obj['bndbox']['xmax']} "
            f"{voc_obj['bndbox']['ymax']}"
        )

        # 保存 VOC 标签行
        voc_lines.append(voc_line)

        # 打印当前 mask 的测试结果
        print(f"[{idx + 1}] {filename}")
        print(f"    class_name: {class_name}")
        print(f"    bbox xyxy: {bbox}")
        print(f"    YOLO: {yolo_line}")
        print(f"    VOC : {voc_line}")
        print()

        # 保存后处理后的 mask
        refined_save_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".png", "_refined.png")
        )
        cv2.imwrite(refined_save_path, refined_mask)

        # 将原始 mask 叠加到原图上，红色表示原始 mask
        overlay_raw = overlay_mask_on_image(
            image_rgb_show,
            mask_gray,
            color=(255, 0, 0),
            alpha=0.45
        )

        # 将后处理 mask 叠加到原图上，绿色表示 refined mask
        overlay_refined = overlay_mask_on_image(
            image_rgb_show,
            refined_mask,
            color=(0, 255, 0),
            alpha=0.45
        )

        # 在 refined mask 的叠加图上绘制 bbox
        bbox_vis = draw_bbox_on_image(
            overlay_refined,
            bbox,
            color=(255, 255, 0),
            thickness=2
        )

        # OpenCV 保存图像时使用 BGR，因此 RGB 转 BGR
        bbox_vis_bgr = cv2.cvtColor(bbox_vis, cv2.COLOR_RGB2BGR)

        # 保存 bbox 可视化图
        vis_save_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".png", "_bbox_vis.png")
        )
        cv2.imwrite(vis_save_path, bbox_vis_bgr)

        # 使用 matplotlib 展示四张图：
        # 1. 原图
        # 2. 原始 mask
        # 3. 后处理 mask
        # 4. bbox 可视化结果
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(image_rgb_show)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Raw Mask")
        plt.imshow(mask_gray, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Refined Mask")
        plt.imshow(refined_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title(f"{class_name} bbox")
        plt.imshow(bbox_vis)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # 保存 YOLO 标签 txt
    yolo_save_path = os.path.join(OUTPUT_DIR, "labels_yolo.txt")

    with open(yolo_save_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")

    # 保存 VOC 标签 txt
    voc_save_path = os.path.join(OUTPUT_DIR, "labels_voc.txt")

    with open(voc_save_path, "w", encoding="utf-8") as f:
        for line in voc_lines:
            f.write(line + "\n")

    print("========== 测试完成 ==========")
    print(f"处理结果保存到：{OUTPUT_DIR}")
    print(f"YOLO 标签保存到：{yolo_save_path}")
    print(f"VOC 标签保存到：{voc_save_path}")