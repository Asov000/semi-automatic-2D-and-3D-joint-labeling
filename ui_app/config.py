# -*- coding: utf-8 -*-
"""界面配置模块，从启动配置中读取路径、模型参数和后处理默认值。"""

import os
import torch

try:
    from run import (
        BOX_3D_LOWER_PERCENTILE,
        BOX_3D_UPPER_PERCENTILE,
        CLASS_COLOR_TABLE,
        DEFAULT_CLASS_COLOR_PALETTE,
        ENABLE_2D_MASK_CLOSE,
        ENABLE_2D_MASK_FILL_HOLES,
        ENABLE_2D_MASK_NMS,
        ENABLE_2D_MASK_OPEN,
        ENABLE_3D_BOX_NMS,
        MASK_NMS_IOU_THRESHOLD,
        MASK_NMS_OVERLAP_MIN_THRESHOLD,
        MASK_NMS_SAME_CLASS_IOU_THRESHOLD,
        MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD,
        MIN_3D_BOX_INNER_POINTS,
        REFINE_2D_MASK_CLOSE_ITERATIONS,
        REFINE_2D_MASK_CLOSE_KERNEL_SIZE,
        REFINE_2D_MASK_KEEP_LARGEST,
        REFINE_2D_MASK_MAX_EXTERNAL_EXPAND_PX,
        REFINE_2D_MASK_MIN_AREA,
        REFINE_2D_MASK_OPEN_ITERATIONS,
        REFINE_2D_MASK_OPEN_KERNEL_SIZE,
        SAM3_DEFAULT_CONF,
        SAM3_DEFAULT_DEVICE,
        SAM3_DEFAULT_HALF,
        SAM3_DEFAULT_HALF_MODE,
        SAM3_DEFAULT_IMGSZ,
        SAM3_DEFAULT_IOU,
        SAM3_DEFAULT_MAX_DET,
        SAM3_DEFAULT_TEXT_PROMPTS,
        SAM3_DEFAULT_VERBOSE,
        SAM3_MODEL_PATH,
        USE_3D_BOX_PERCENTILE_FILTER,
        USE_3D_ZBUFFER,
    )
except Exception:
    CLASS_COLOR_TABLE = {
        "wall": (255, 0, 0),
        "floor": (0, 255, 0),
        "chair": (255, 0, 255),
        "table": (255, 255, 128),
        "sofa": (0, 0, 255),
        "bed": (255, 128, 0),
        "cabinet": (128, 0, 255),
        "window": (0, 255, 255),
        "door": (0, 128, 255),
        "other": (128, 255, 0),
    }
    DEFAULT_CLASS_COLOR_PALETTE = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
    ]
    MASK_NMS_IOU_THRESHOLD = 0.65
    MASK_NMS_OVERLAP_MIN_THRESHOLD = 0.80
    MASK_NMS_SAME_CLASS_IOU_THRESHOLD = 0.90
    MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD = 0.95
    ENABLE_2D_MASK_NMS = True
    REFINE_2D_MASK_MIN_AREA = 64
    REFINE_2D_MASK_KEEP_LARGEST = False
    ENABLE_2D_MASK_CLOSE = True
    REFINE_2D_MASK_CLOSE_KERNEL_SIZE = 5
    REFINE_2D_MASK_CLOSE_ITERATIONS = 1
    ENABLE_2D_MASK_OPEN = False
    REFINE_2D_MASK_OPEN_KERNEL_SIZE = 3
    REFINE_2D_MASK_OPEN_ITERATIONS = 1
    ENABLE_2D_MASK_FILL_HOLES = True
    REFINE_2D_MASK_MAX_EXTERNAL_EXPAND_PX = 2
    USE_3D_ZBUFFER = True
    USE_3D_BOX_PERCENTILE_FILTER = True
    BOX_3D_LOWER_PERCENTILE = 3.0
    BOX_3D_UPPER_PERCENTILE = 97.0
    ENABLE_3D_BOX_NMS = True
    MIN_3D_BOX_INNER_POINTS = 5000
    SAM3_MODEL_PATH = r"D:\sam3.pt"
    SAM3_DEFAULT_TEXT_PROMPTS = "chair, table, sofa"
    SAM3_DEFAULT_CONF = 0.25
    SAM3_DEFAULT_IOU = 0.70
    SAM3_DEFAULT_IMGSZ = 1024
    SAM3_DEFAULT_MAX_DET = 100
    SAM3_DEFAULT_HALF_MODE = "自动"
    SAM3_DEFAULT_HALF = True
    SAM3_DEFAULT_DEVICE = None
    SAM3_DEFAULT_VERBOSE = False

IMAGE_FOLDER = r"C:\Users\25918\Desktop\test"
SAM_CHECKPOINT = r"D:\SAM\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = r"C:\Users\25918\Desktop\sam_output"

SUNRGBD_ROOT_DIR = r"D:\frustum-convnet\sunrgbd\mysunrgbd\training"
ANNOTATION_3D_SAVE_ROOT = os.path.join(OUTPUT_DIR, "annotations_3d")
BOX_3D_TYPE = "pca"
MIN_3D_BOX_POINTS = 30
BOX_3D_UP_AXIS = 2
ZBUFFER_TOLERANCE = 0.03

ENABLE_3D_BOX_DENSITY_FILTER = True
MIN_3D_BOX_DENSITY = 10000
MAX_3D_BOX_VOLUME = None
BOX_3D_NMS_IOU_THRESH = 0.10
BOX_3D_NMS_CLASS_AWARE = False
REMOVE_SUPPRESSED_3D_BOX_POINTS = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

REFINE_KWARGS = {
    "min_area": REFINE_2D_MASK_MIN_AREA,
    "keep_largest": REFINE_2D_MASK_KEEP_LARGEST,
    "close_kernel_size": REFINE_2D_MASK_CLOSE_KERNEL_SIZE if ENABLE_2D_MASK_CLOSE else 0,
    "close_iterations": REFINE_2D_MASK_CLOSE_ITERATIONS,
    "open_kernel_size": REFINE_2D_MASK_OPEN_KERNEL_SIZE if ENABLE_2D_MASK_OPEN else 0,
    "open_iterations": REFINE_2D_MASK_OPEN_ITERATIONS,
    "fill_holes": ENABLE_2D_MASK_FILL_HOLES,
    "max_external_expand_px": REFINE_2D_MASK_MAX_EXTERNAL_EXPAND_PX,
}


def normalize_rgb_color(color):
    """把输入颜色裁剪并转换为合法的 RGB 整数三元组。"""
    r, g, b = color
    return (
        int(max(0, min(255, r))),
        int(max(0, min(255, g))),
        int(max(0, min(255, b))),
    )


def get_registered_class_color(class_name: str, offset: int = 0):
    """读取类别的全局注册颜色，不存在时自动分配新颜色。"""
    name = str(class_name).strip()
    if name in CLASS_COLOR_TABLE:
        return normalize_rgb_color(CLASS_COLOR_TABLE[name])

    color = DEFAULT_CLASS_COLOR_PALETTE[offset % len(DEFAULT_CLASS_COLOR_PALETTE)]
    CLASS_COLOR_TABLE[name] = normalize_rgb_color(color)
    return CLASS_COLOR_TABLE[name]


def set_registered_class_color(class_name: str, color):
    """更新指定类别的全局注册颜色。"""
    name = str(class_name).strip()
    CLASS_COLOR_TABLE[name] = normalize_rgb_color(color)
    return CLASS_COLOR_TABLE[name]
