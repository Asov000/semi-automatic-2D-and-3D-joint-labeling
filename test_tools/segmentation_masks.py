# -*- coding: utf-8 -*-
"""二维分割结果检查脚本，验证 mask、检测框、导出文本和可视化结果。"""

import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tool_modules import (
    bbox_xyxy_to_yolo as _bbox_xyxy_to_yolo_line,
    mask_to_bbox_xyxy as _mask_to_bbox_xyxy,
    mask_to_voc_bbox as _mask_to_voc_bbox,
    refine_mask as _refine_mask_binary,
)

SEG_DIR = r"C:\Users\25918\Desktop\sam_output\test2_segmentation"
BINARY_DIR = os.path.join(SEG_DIR, "binary_masks")
OUTPUT_DIR = os.path.join(SEG_DIR, "processed_test")


def refine_mask(
    mask,
    min_area=64,
    keep_largest=False,
    close_kernel_size=5,
    fill_holes=True,
    max_external_expand_px=1,
):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    refined = _refine_mask_binary(
        mask,
        min_area=min_area,
        keep_largest=keep_largest,
        close_kernel_size=close_kernel_size,
        fill_holes=fill_holes,
        max_external_expand_px=max_external_expand_px,
    )
    return (refined > 0).astype(np.uint8) * 255


def mask_to_bbox_xyxy(mask):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    return _mask_to_bbox_xyxy(mask, inclusive=True)


def bbox_xyxy_to_yolo(bbox, image_width, image_height, class_id):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    line = _bbox_xyxy_to_yolo_line(
        bbox,
        image_width=image_width,
        image_height=image_height,
        class_id=class_id,
        inclusive=True,
        decimals=12,
    )
    parts = line.split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def mask_to_yolo_bbox(mask, image_width, image_height, class_id):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    bbox = mask_to_bbox_xyxy(mask)

    if bbox is None:
        return None

    return bbox_xyxy_to_yolo(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        class_id=class_id,
    )


def mask_to_voc_bbox(mask, class_name):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    obj = _mask_to_voc_bbox(mask, class_name=class_name)
    if obj is None:
        return None
    return {
        "name": obj["name"],
        "bndbox": obj["bndbox"],
    }


def load_classes_json(seg_dir):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    json_files = [
        f for f in os.listdir(seg_dir)
        if f.endswith("_classes.json")
    ]

    if len(json_files) == 0:
        return None

    json_path = os.path.join(seg_dir, json_files[0])

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_original_image(class_info):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if class_info is None:
        return None

    image_path = class_info.get("image_path", None)

    if image_path is None:
        return None

    if not os.path.exists(image_path):
        print(f"原图路径不存在：{image_path}")
        return None

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def parse_binary_mask_filename(filename):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""

    name = os.path.splitext(filename)[0]

    match = re.search(r"_(\d{3})_(.+)$", name)

    if match is None:
        return None, "unknown"

    instance_id = int(match.group(1))
    class_name = match.group(2)

    return instance_id, class_name


def draw_bbox_on_image(image_rgb, bbox, color=(255, 0, 0), thickness=2):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    vis = image_rgb.copy()

    if bbox is None:
        return vis

    xmin, ymin, xmax, ymax = bbox

    cv2.rectangle(
        vis,
        (xmin, ymin),
        (xmax, ymax),
        color,
        thickness
    )

    return vis


def overlay_mask_on_image(image_rgb, mask, color=(255, 0, 0), alpha=0.45):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    vis = image_rgb.copy().astype(np.float32)

    mask_bool = mask > 0
    color_arr = np.array(color, dtype=np.float32)

    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + color_arr * alpha

    vis = np.clip(vis, 0, 255).astype(np.uint8)

    return vis


def test_segmentation_masks(seg_dir):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_info = load_classes_json(seg_dir)
    image_rgb = load_original_image(class_info)

    binary_dir = os.path.join(seg_dir, "binary_masks")

    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"找不到 binary_masks 文件夹：{binary_dir}")

    mask_files = [
        f for f in os.listdir(binary_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if len(mask_files) == 0:
        raise FileNotFoundError(f"binary_masks 文件夹里没有 mask 图片：{binary_dir}")

    mask_files = sorted(mask_files)

    # 类别映射
    if class_info is not None and "class_to_id" in class_info:
        class_to_id = class_info["class_to_id"]
    else:
        class_to_id = {"background": 0}

    # YOLO 一般要求类别从 0 开始，所以 background 不参与
    yolo_class_to_id = {}

    for class_name, class_id in class_to_id.items():
        if class_name == "background":
            continue
        yolo_class_to_id[class_name] = int(class_id) - 1

    yolo_lines = []
    voc_lines = []

    print("========== 开始测试 mask ==========")
    print(f"seg_dir: {seg_dir}")
    print(f"binary_masks 数量: {len(mask_files)}")
    print()

    for idx, filename in enumerate(mask_files):
        mask_path = os.path.join(binary_dir, filename)

        instance_id, class_name = parse_binary_mask_filename(filename)

        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_gray is None:
            print(f"读取失败，跳过：{mask_path}")
            continue

        h, w = mask_gray.shape[:2]

        if image_rgb is None:
            # 如果原图读不到，就用黑底图显示
            image_rgb_show = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image_rgb_show = image_rgb.copy()

            if image_rgb_show.shape[:2] != (h, w):
                image_rgb_show = cv2.resize(
                    image_rgb_show,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

        # 后处理
        refined_mask = refine_mask(
            mask_gray,
            min_area=64,
            keep_largest=False,
            close_kernel_size=5,
            fill_holes=True,
            max_external_expand_px=1
        )

        # 计算 bbox
        bbox = mask_to_bbox_xyxy(refined_mask)

        if bbox is None:
            print(f"{filename}: 空 mask，跳过")
            continue

        # 计算 YOLO
        class_id = yolo_class_to_id.get(class_name, 0)

        yolo_bbox = mask_to_yolo_bbox(
            refined_mask,
            image_width=w,
            image_height=h,
            class_id=class_id
        )

        yolo_line = (
            f"{yolo_bbox[0]} "
            f"{yolo_bbox[1]:.6f} "
            f"{yolo_bbox[2]:.6f} "
            f"{yolo_bbox[3]:.6f} "
            f"{yolo_bbox[4]:.6f}"
        )

        yolo_lines.append(yolo_line)

        # 计算 VOC
        voc_obj = mask_to_voc_bbox(
            refined_mask,
            class_name=class_name
        )

        voc_line = (
            f"{class_name} "
            f"{voc_obj['bndbox']['xmin']} "
            f"{voc_obj['bndbox']['ymin']} "
            f"{voc_obj['bndbox']['xmax']} "
            f"{voc_obj['bndbox']['ymax']}"
        )

        voc_lines.append(voc_line)

        print(f"[{idx + 1}] {filename}")
        print(f"    class_name: {class_name}")
        print(f"    bbox xyxy: {bbox}")
        print(f"    YOLO: {yolo_line}")
        print(f"    VOC : {voc_line}")
        print()

        # 保存处理后的 mask
        refined_save_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".png", "_refined.png")
        )
        cv2.imwrite(refined_save_path, refined_mask)

        # 可视化
        overlay_raw = overlay_mask_on_image(
            image_rgb_show,
            mask_gray,
            color=(255, 0, 0),
            alpha=0.45
        )

        overlay_refined = overlay_mask_on_image(
            image_rgb_show,
            refined_mask,
            color=(0, 255, 0),
            alpha=0.45
        )

        bbox_vis = draw_bbox_on_image(
            overlay_refined,
            bbox,
            color=(255, 255, 0),
            thickness=2
        )

        # 保存可视化图
        bbox_vis_bgr = cv2.cvtColor(bbox_vis, cv2.COLOR_RGB2BGR)

        vis_save_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".png", "_bbox_vis.png")
        )
        cv2.imwrite(vis_save_path, bbox_vis_bgr)

        # 展示
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

    # 保存 YOLO txt
    yolo_save_path = os.path.join(OUTPUT_DIR, "labels_yolo.txt")

    with open(yolo_save_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")

    # 保存 VOC txt
    voc_save_path = os.path.join(OUTPUT_DIR, "labels_voc.txt")

    with open(voc_save_path, "w", encoding="utf-8") as f:
        for line in voc_lines:
            f.write(line + "\n")

    print("========== 测试完成 ==========")
    print(f"处理结果保存到：{OUTPUT_DIR}")
    print(f"YOLO 标签保存到：{yolo_save_path}")
    print(f"VOC 标签保存到：{voc_save_path}")
