# -*- coding: utf-8 -*-
"""界面共享上下文模块，集中导入 Qt、图像处理、模型和三维工具依赖。"""

import inspect
import json
import os
import sys
import traceback

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tool_modules import (
    refine_mask,
    save_voc_xml,
    save_yolo_txt,
    saved_masks_to_voc_objects,
    saved_masks_to_yolo_lines,
)

from .config import *

try:
    from tool3d_modules import (
        generate_3d_annotations_for_one_sample,
        save_3d_boxes_json,
        save_3d_boxes_txt,
        semantic_mask_nms,
        visualize_result_from_memory,
    )
    TOOL3D_AVAILABLE = True
    TOOL3D_IMPORT_ERROR = None
except Exception as _tool3d_error:
    generate_3d_annotations_for_one_sample = None
    save_3d_boxes_json = None
    save_3d_boxes_txt = None
    semantic_mask_nms = None
    visualize_result_from_memory = None
    TOOL3D_AVAILABLE = False
    TOOL3D_IMPORT_ERROR = _tool3d_error

try:
    from sam3_auto_modules import (
        SAM3AutoConfig,
        SAM3AutoMaskGenerator,
        build_sam3_overrides,
        normalize_text_prompts,
    )
    SAM3_AVAILABLE = True
    SAM3_IMPORT_ERROR = None
except Exception as _sam3_error:
    SAM3AutoConfig = None
    SAM3AutoMaskGenerator = None
    build_sam3_overrides = None
    normalize_text_prompts = None
    SAM3_AVAILABLE = False
    SAM3_IMPORT_ERROR = _sam3_error
