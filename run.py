# -*- coding: utf-8 -*-
"""项目启动入口和全局配置文件，集中管理模型路径、后处理开关和三维参数。"""

import os
import sys


# 全局类别颜色表，二维掩膜、导出文件和三维语义显示都会复用这里的颜色。
# 如果要调整某个类别的系统默认颜色，只需要修改这张表。
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

# SAM3 自动框和掩膜预测的默认参数。
SAM3_MODEL_PATH = r"D:\sam3.pt"
SAM3_DEFAULT_TEXT_PROMPTS = "chair, table, sofa"
SAM3_DEFAULT_CONF = 0.6
SAM3_DEFAULT_IOU = 1
SAM3_DEFAULT_IMGSZ = 1024
SAM3_DEFAULT_MAX_DET = 100
SAM3_DEFAULT_HALF_MODE = "自动"
SAM3_DEFAULT_HALF = True
SAM3_DEFAULT_DEVICE = None
SAM3_DEFAULT_VERBOSE = False

# 二维和三维后处理的全局开关。
# 需要统一控制导出行为时，优先在这里调整默认值。
ENABLE_2D_MASK_NMS = True
MASK_NMS_IOU_THRESHOLD = 0.65
MASK_NMS_OVERLAP_MIN_THRESHOLD = 0.80
MASK_NMS_SAME_CLASS_IOU_THRESHOLD = 0.90
MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD = 0.95

REFINE_2D_MASK_MIN_AREA = 64
REFINE_2D_MASK_KEEP_LARGEST = False
ENABLE_2D_MASK_CLOSE = True
REFINE_2D_MASK_CLOSE_KERNEL_SIZE = 5
REFINE_2D_MASK_CLOSE_ITERATIONS = 1
ENABLE_2D_MASK_OPEN = False
REFINE_2D_MASK_OPEN_KERNEL_SIZE = 3
REFINE_2D_MASK_OPEN_ITERATIONS = 1
ENABLE_2D_MASK_FILL_HOLES = False
REFINE_2D_MASK_MAX_EXTERNAL_EXPAND_PX = 2

USE_3D_ZBUFFER = True
USE_3D_BOX_PERCENTILE_FILTER = True
BOX_3D_LOWER_PERCENTILE = 1
BOX_3D_UPPER_PERCENTILE = 99
ENABLE_3D_BOX_NMS = False
MIN_3D_BOX_INNER_POINTS = 500


def _ensure_main_on_path() -> None:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)


def main() -> int:
    """执行命令行入口逻辑，并启动主应用。"""
    _ensure_main_on_path()

    from ui_app.main import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
