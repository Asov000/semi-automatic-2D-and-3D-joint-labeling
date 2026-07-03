# -*- coding: utf-8 -*-
"""

PyQt + SAM 批量半自动标注程序，整合 2D + 3D 标注导出。

功能：
1. 点击“完成该类标注”时，调用 tool.py 中的 refine_mask，
   保存到 saved_masks 的就是后处理后的干净 mask。
2. 点击“完成当前图片标注并保存 → 下一张”时，自动导出：
   - 2D 分割：semantic / instance / color / overlay / binary masks；
   - 2D 框：YOLO + VOC；
   - 3D 分割：point_class_ids / point_instance_ids / labeled ply；
   - 3D 框：3d_boxes.json / 3d_boxes.txt。

依赖：
- tool.py：
    refine_mask
    saved_masks_to_yolo_lines
    saved_masks_to_voc_objects
    save_yolo_txt
    save_voc_xml

- tool3d.py：
    generate_3d_annotations_for_one_sample
    visualize_result_from_memory

"""

import sys
import os
import cv2
import json
import traceback
import inspect
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QComboBox, QLineEdit, QMessageBox, QFileDialog,
    QListWidget, QListWidgetItem, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush

from segment_anything import sam_model_registry, SamPredictor


from tool import (
    refine_mask,
    saved_masks_to_yolo_lines,
    saved_masks_to_voc_objects,
    save_yolo_txt,
    save_voc_xml,
)

# =========================
# 3D 工具函数
# =========================
try:
    from tool3d import (
        generate_3d_annotations_for_one_sample,
        visualize_result_from_memory,
    )
    TOOL3D_AVAILABLE = True
    TOOL3D_IMPORT_ERROR = None
except Exception as _tool3d_error:
    generate_3d_annotations_for_one_sample = None
    visualize_result_from_memory = None
    TOOL3D_AVAILABLE = False
    TOOL3D_IMPORT_ERROR = _tool3d_error



IMAGE_FOLDER = r"C:\Users\25918\Desktop\test"

SAM_CHECKPOINT = r"C:\Users\25918\Downloads\SAM\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = r"C:\Users\25918\Desktop\sam_output"

# =========================
# 3D 标注相关路径
# =========================


# 该目录下必须包含：
#   pc/000001.mat
#   image/000001.jpg
#   calib/000001.txt
SUNRGBD_ROOT_DIR = r"D:\frustum-convnet\sunrgbd\mysunrgbd\training"

# 3D 标注保存根目录。
ANNOTATION_3D_SAVE_ROOT = os.path.join(OUTPUT_DIR, "annotations_3d")

# 3D 框类型：
#   "pca"  = PCA 近似有向框
#   "aabb" = 轴对齐框
BOX_3D_TYPE = "pca"

# 少于该点数的 3D 实例不生成 3D 框
MIN_3D_BOX_POINTS = 30

# SUNRGBD / upright_depth 常用 z 轴作为竖直方向
BOX_3D_UP_AXIS = 2

# 是否使用 z-buffer 过滤遮挡点
USE_3D_ZBUFFER = True
ZBUFFER_TOLERANCE = 0.03

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# =========================
# mask 后处理参数
# =========================

REFINE_KWARGS = {
    "min_area": 64,
    "keep_largest": True,
    "close_kernel_size": 5,
    "close_iterations": 1,
    "open_kernel_size": 0,
    "open_iterations": 1,
    "fill_holes": True,
    "max_external_expand_px": 2,
}


class ImageLabel(QLabel):
    mouse_clicked = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222222;")
        self.image_width = 0
        self.image_height = 0

    def set_image_size(self, w, h):
        self.image_width = w
        self.image_height = h

    def mousePressEvent(self, event):
        if self.image_width == 0 or self.image_height == 0:
            return

        label_w = self.width()
        label_h = self.height()

        scale = min(label_w / self.image_width, label_h / self.image_height)

        show_w = self.image_width * scale
        show_h = self.image_height * scale

        offset_x = (label_w - show_w) / 2
        offset_y = (label_h - show_h) / 2

        x = event.pos().x()
        y = event.pos().y()

        if x < offset_x or x > offset_x + show_w:
            return
        if y < offset_y or y > offset_y + show_h:
            return

        image_x = int((x - offset_x) / scale)
        image_y = int((y - offset_y) / scale)

        if event.button() == Qt.LeftButton:
            self.mouse_clicked.emit(image_x, image_y, 1)   # 前景点
        elif event.button() == Qt.RightButton:
            self.mouse_clicked.emit(image_x, image_y, 0)   # 背景点


class SAMAnnotator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SAM 批量半自动掩膜标注工具")
        self.resize(1500, 900)

        self.image_folder = IMAGE_FOLDER
        self.image_paths = []
        self.current_image_idx = 0

        self.image_path = None
        self.image_rgb = None

        # 最近一次保存的 2D segmentation 目录。
        # “查看当前标注点云”会复用这个目录里的 binary_masks。
        self.last_saved_segmentation_dir = None
        self.last_saved_image_name = None
        self.last_3d_result = None

        self.predictor = None

        # 当前正在编辑的 mask
        self.current_mask = None
        self.current_score = None

        # SAM 返回的候选 mask
        self.all_masks = None
        self.all_scores = None
        self.current_mask_index = 0

        # 当前 mask 使用的提示点
        self.points = []
        self.labels = []

        # 已经完成的 mask 列表
        self.saved_masks = []

        # 每张图片的临时标注缓存。
        # 作用：左右切换图片时，不丢失已经完成的 mask 和当前未完成的提示点。
        self.annotation_cache = {}

        # SAM 图像特征缓存。
        # 作用：返回已经打开过的图片时，不重复计算 predictor.set_image 的图像 embedding。
        # 注意：vit_h 的特征会占显存/内存，所以不要缓存太多张。
        self.image_feature_cache = {}
        self.image_feature_order = []
        self.max_image_feature_cache = 3

        # 类别颜色表，RGB 格式
        self.class_colors = {}

        # 默认颜色池，RGB
        self.default_colors = [
            (255, 0, 0),       # red
            (0, 255, 0),       # green
            (0, 0, 255),       # blue
            (255, 255, 0),     # yellow
            (255, 0, 255),     # magenta
            (0, 255, 255),     # cyan
            (255, 128, 0),     # orange
            (128, 0, 255),     # purple
            (0, 128, 255),     # sky blue
            (128, 255, 0),     # light green
        ]

        self.init_ui()
        self.init_class_colors()
        self.load_sam_model()
        self.load_image_folder(self.image_folder)

    # =========================
    # UI
    # =========================

    def init_ui(self):
        main_layout = QHBoxLayout()

        # 左侧：顶部翻页栏 + 图像显示区域
        left_layout = QVBoxLayout()

        nav_layout = QHBoxLayout()

        self.prev_image_btn = QPushButton("← 上一张")
        self.prev_image_btn.clicked.connect(self.go_prev_image)
        nav_layout.addWidget(self.prev_image_btn)

        self.image_counter_label = QLabel("当前图片：0/0")
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        self.image_counter_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        nav_layout.addWidget(self.image_counter_label, stretch=1)

        self.next_image_btn = QPushButton("下一张 →")
        self.next_image_btn.clicked.connect(self.go_next_image)
        nav_layout.addWidget(self.next_image_btn)

        left_layout.addLayout(nav_layout)

        self.image_label = ImageLabel()
        self.image_label.mouse_clicked.connect(self.on_mouse_click)
        left_layout.addWidget(self.image_label, stretch=1)

        main_layout.addLayout(left_layout, stretch=5)

        # 右侧工具栏
        side_layout = QVBoxLayout()

        self.info_label = QLabel("左键：前景点\n右键：背景点")
        side_layout.addWidget(self.info_label)

        # 文件夹选择
        self.open_folder_btn = QPushButton("打开图片文件夹")
        self.open_folder_btn.clicked.connect(self.open_image_folder)
        side_layout.addWidget(self.open_folder_btn)

        # 类别选择
        side_layout.addWidget(QLabel("当前类别："))

        self.class_combo = QComboBox()
        self.class_combo.addItems([
            "wall",
            "floor",
            "chair",
            "table",
            "sofa",
            "bed",
            "cabinet",
            "window",
            "door",
            "other"
        ])
        self.class_combo.currentTextChanged.connect(self.on_class_changed)
        side_layout.addWidget(self.class_combo)

        # 类别颜色
        side_layout.addWidget(QLabel("当前类别颜色："))

        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(28)
        self.color_preview.setStyleSheet(
            "background-color: rgb(255, 0, 0); border: 1px solid black;"
        )
        side_layout.addWidget(self.color_preview)

        self.choose_color_btn = QPushButton("选择类别颜色")
        self.choose_color_btn.clicked.connect(self.choose_class_color)
        side_layout.addWidget(self.choose_color_btn)

        # SAM 候选 mask 选择
        side_layout.addWidget(QLabel("SAM 候选 Mask："))

        self.mask_combo = QComboBox()
        self.mask_combo.addItems([
            "Mask 1",
            "Mask 2",
            "Mask 3"
        ])
        self.mask_combo.currentIndexChanged.connect(self.change_mask_index)
        side_layout.addWidget(self.mask_combo)

        # 添加新类别
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("输入新类别，例如 monitor")
        side_layout.addWidget(self.class_input)

        self.add_class_btn = QPushButton("添加类别")
        self.add_class_btn.clicked.connect(self.add_class)
        side_layout.addWidget(self.add_class_btn)

        # 当前点操作
        self.undo_btn = QPushButton("撤销上一个点")
        self.undo_btn.clicked.connect(self.undo_point)
        side_layout.addWidget(self.undo_btn)

        self.clear_btn = QPushButton("清空当前点")
        self.clear_btn.clicked.connect(self.clear_points)
        side_layout.addWidget(self.clear_btn)

        # 完成当前类别 mask
        self.commit_btn = QPushButton("✔ 完成该类标注")
        self.commit_btn.clicked.connect(self.commit_current_mask)
        side_layout.addWidget(self.commit_btn)

        # 已完成 mask 列表
        side_layout.addWidget(QLabel("该图片已完成掩膜："))

        self.saved_mask_list = QListWidget()
        self.saved_mask_list.setMinimumHeight(150)
        side_layout.addWidget(self.saved_mask_list)

        self.delete_saved_mask_btn = QPushButton("删除选中的已完成掩膜")
        self.delete_saved_mask_btn.clicked.connect(self.delete_selected_saved_mask)
        side_layout.addWidget(self.delete_saved_mask_btn)

        # =========================
        # 框标注保存格式：2D / 3D
        # =========================
        side_layout.addWidget(QLabel("框标注保存格式："))

        self.box_format_combo = QComboBox()
        self.box_format_combo.addItems([
            "不保存框",
            "仅2D框(YOLO+VOC)",
            "仅3D框",
            "2D+3D框",
        ])
        self.box_format_combo.setCurrentText("2D+3D框")
        side_layout.addWidget(self.box_format_combo)


        # =========================
        # 分割标注保存格式：2D / 3D
        # =========================
        side_layout.addWidget(QLabel("分割标注保存格式："))

        self.seg_format_combo = QComboBox()
        self.seg_format_combo.addItems([
            "仅2D分割",
            "2D+3D分割",
        ])
        self.seg_format_combo.setCurrentText("2D+3D分割")
        side_layout.addWidget(self.seg_format_combo)

        # =========================
        # 3D 语义图滤波强度
        # =========================
        side_layout.addWidget(QLabel("3D语义图滤波强度："))

        self.semantic_filter_combo = QComboBox()
        self.semantic_filter_combo.addItems([
            "0 不滤波",
            "1 弱滤波",
            "2 中滤波",
            "3 强滤波",
        ])
        self.semantic_filter_combo.setCurrentIndex(1)
        self.semantic_filter_combo.setToolTip(
            "用于去除投影到3D语义点云中的离群点。\n"
            "0：不处理；1：轻度；2：中等；3：强滤波，只保留主要连通块。"
        )
        side_layout.addWidget(self.semantic_filter_combo)

        # 查看当前标注点云
        self.view_3d_btn = QPushButton("查看当前标注点云")
        self.view_3d_btn.clicked.connect(self.view_current_annotated_pointcloud)
        side_layout.addWidget(self.view_3d_btn)

        # 完成当前图片，自动保存并跳转下一张
        self.finish_current_image_btn = QPushButton("完成当前图片标注并保存 → 下一张")
        self.finish_current_image_btn.clicked.connect(self.finish_current_image_and_next)
        side_layout.addWidget(self.finish_current_image_btn)

        side_layout.addStretch()

        self.status_label = QLabel("状态：等待操作")
        self.status_label.setWordWrap(True)
        side_layout.addWidget(self.status_label)

        main_layout.addLayout(side_layout, stretch=1)

        self.setLayout(main_layout)


    # =========================
    # 类别颜色相关
    # =========================

    def init_class_colors(self):
        for i in range(self.class_combo.count()):
            class_name = self.class_combo.itemText(i)
            self.class_colors[class_name] = self.default_colors[i % len(self.default_colors)]

        self.update_color_preview()

    def get_current_class_color(self):
        class_name = self.class_combo.currentText()

        if class_name not in self.class_colors:
            idx = len(self.class_colors)
            self.class_colors[class_name] = self.default_colors[idx % len(self.default_colors)]

        return self.class_colors[class_name]

    def update_color_preview(self):
        color = self.get_current_class_color()
        r, g, b = color

        self.color_preview.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid black;"
        )

        self.update_display()

    def on_class_changed(self, class_name):
        if class_name not in self.class_colors:
            idx = len(self.class_colors)
            self.class_colors[class_name] = self.default_colors[idx % len(self.default_colors)]

        self.update_color_preview()

    def choose_class_color(self):
        class_name = self.class_combo.currentText()
        old_color = self.get_current_class_color()
        old_qcolor = QColor(old_color[0], old_color[1], old_color[2])

        color = QColorDialog.getColor(old_qcolor, self, f"选择类别 {class_name} 的颜色")

        if color.isValid():
            new_color = (
                color.red(),
                color.green(),
                color.blue()
            )

            self.class_colors[class_name] = new_color

            # 已完成的同类别 mask 颜色也同步更新
            for item in self.saved_masks:
                if item["class_name"] == class_name:
                    item["color"] = new_color

            self.update_color_preview()
            self.refresh_saved_mask_list()
            self.update_display()

            self.status_label.setText(
                f"状态：已修改类别颜色\n"
                f"类别：{class_name}\n"
                f"颜色：RGB({color.red()}, {color.green()}, {color.blue()})"
            )

    # =========================
    # SAM 模型与图片文件夹加载
    # =========================

    def load_sam_model(self):
        self.status_label.setText("状态：正在加载 SAM 模型...")

        model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        model.to(device=SAM_DEVICE)

        self.predictor = SamPredictor(model)

        self.status_label.setText(f"状态：SAM 模型加载完成，当前设备：{SAM_DEVICE}")

    def load_image_folder(self, folder_path):
        """
        读取文件夹内所有图片。
        """
        if folder_path is None or not os.path.isdir(folder_path):
            QMessageBox.warning(
                self,
                "提示",
                f"图片文件夹不存在：\n{folder_path}\n\n请点击“打开图片文件夹”重新选择。"
            )
            self.image_paths = []
            self.current_image_idx = 0
            self.update_image_counter()
            return

        image_paths = []

        for name in os.listdir(folder_path):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(folder_path, name))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            QMessageBox.warning(
                self,
                "提示",
                f"该文件夹下没有找到图片：\n{folder_path}"
            )
            self.image_paths = []
            self.current_image_idx = 0
            self.update_image_counter()
            return

        self.image_folder = folder_path
        self.image_paths = image_paths
        self.current_image_idx = 0

        # 新打开一个文件夹时，清空旧文件夹的临时标注缓存和图像特征缓存。
        self.annotation_cache = {}
        self.image_feature_cache = {}
        self.image_feature_order = []

        self.load_current_image()

        self.status_label.setText(
            f"状态：已加载图片文件夹\n"
            f"文件夹：{folder_path}\n"
            f"图片数量：{len(self.image_paths)}"
        )

    def open_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择图片文件夹",
            self.image_folder if self.image_folder else ""
        )

        if folder_path:
            # 切换到新文件夹前，先把当前图片的临时标注状态写入内存缓存。
            self.cache_current_annotation_state()
            self.load_image_folder(folder_path)

    def load_current_image(self):
        if len(self.image_paths) == 0:
            self.image_path = None
            self.image_rgb = None
            self.image_label.clear()
            self.update_image_counter()
            return

        self.current_image_idx = max(0, min(self.current_image_idx, len(self.image_paths) - 1))
        image_path = self.image_paths[self.current_image_idx]

        self.image_path = image_path

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            QMessageBox.critical(self, "错误", f"无法读取图片：\n{image_path}")
            return

        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = self.image_rgb.shape[:2]
        self.image_label.set_image_size(w, h)

        # 这里是切换图片慢的主要位置。
        # 第一次打开某张图片必须计算 SAM 图像特征；返回已打开过的图片时优先复用缓存。
        cache_hit = self.set_predictor_image_with_cache(image_path, self.image_rgb)

        # 不再清空标注，而是恢复该图片之前的标注状态。
        self.restore_annotation_state_from_memory(image_path)

        self.update_display()
        self.update_image_counter()

        cache_text = "命中 SAM 特征缓存" if cache_hit else "首次计算 SAM 特征"

        self.status_label.setText(
            f"状态：图片加载完成（{cache_text}）\n"
            f"当前进度：{self.current_image_idx + 1}/{len(self.image_paths)}\n"
            f"图片：{os.path.basename(self.image_path)}\n"
            f"宽度：{w}\n"
            f"高度：{h}\n"
            f"已恢复 mask 数量：{len(self.saved_masks)}"
        )

    def clear_annotation_state(self):
        self.points = []
        self.labels = []

        self.current_mask = None
        self.current_score = None

        self.all_masks = None
        self.all_scores = None
        self.current_mask_index = 0
        self.mask_combo.setCurrentIndex(0)

        self.saved_masks = []
        self.saved_mask_list.clear()

    def clone_saved_masks(self, saved_masks):
        """
        深拷贝 saved_masks，避免切换图片后不同图片之间共享同一个 numpy mask。
        """
        cloned = []

        for item in saved_masks:
            new_item = dict(item)
            if "mask" in new_item and new_item["mask"] is not None:
                new_item["mask"] = new_item["mask"].copy()
            if "points" in new_item and new_item["points"] is not None:
                new_item["points"] = [p.copy() for p in new_item["points"]]
            if "labels" in new_item and new_item["labels"] is not None:
                new_item["labels"] = list(new_item["labels"])
            cloned.append(new_item)

        return cloned

    def cache_current_annotation_state(self):
        """
        将当前图片的标注状态保存到内存。
        这样左右切换图片时，已经完成的 mask 不会丢失。
        """
        if self.image_path is None:
            return

        self.annotation_cache[self.image_path] = {
            "saved_masks": self.clone_saved_masks(self.saved_masks),
            "points": [p.copy() for p in self.points],
            "labels": list(self.labels),
            "current_mask_index": int(self.current_mask_index),
        }

    def restore_annotation_state_from_memory(self, image_path):
        """
        从内存中恢复某张图片的标注状态。
        如果该图片之前没有标注过，则恢复为空状态。
        """
        self.points = []
        self.labels = []
        self.current_mask = None
        self.current_score = None
        self.all_masks = None
        self.all_scores = None
        self.current_mask_index = 0
        self.saved_masks = []
        self.saved_mask_list.clear()

        state = self.annotation_cache.get(image_path)
        if state is None:
            self.mask_combo.setCurrentIndex(0)
            return

        self.saved_masks = self.clone_saved_masks(state.get("saved_masks", []))
        self.points = [p.copy() for p in state.get("points", [])]
        self.labels = list(state.get("labels", []))
        self.current_mask_index = int(state.get("current_mask_index", 0))

        if 0 <= self.current_mask_index < self.mask_combo.count():
            self.mask_combo.setCurrentIndex(self.current_mask_index)
        else:
            self.current_mask_index = 0
            self.mask_combo.setCurrentIndex(0)

        self.refresh_saved_mask_list()

        # 如果之前有未完成的提示点，恢复图片后重新跑一次 SAM，继续编辑当前 mask。
        if len(self.points) > 0:
            self.run_sam_predict()

    def set_predictor_image_with_cache(self, image_path, image_rgb):
        """
        设置 SAM 当前图片。

        返回：
            True  表示命中缓存，没有重新计算图像 embedding；
            False 表示首次打开该图片，调用了 predictor.set_image。

        说明：
            predictor.set_image 是切换图片慢的主要原因，因为它会跑一次 SAM image encoder。
            对于已经打开过的图片，这里直接恢复 predictor.features，可显著加快“返回上一张”。
        """
        cache = self.image_feature_cache.get(image_path)

        if cache is not None:
            try:
                self.predictor.features = cache["features"]
                self.predictor.original_size = cache["original_size"]
                self.predictor.input_size = cache["input_size"]
                self.predictor.is_image_set = True

                # 维护 LRU 顺序
                if image_path in self.image_feature_order:
                    self.image_feature_order.remove(image_path)
                self.image_feature_order.append(image_path)

                return True
            except Exception:
                # 如果当前 segment_anything 版本属性不兼容，就回退到正常 set_image。
                self.image_feature_cache.pop(image_path, None)
                if image_path in self.image_feature_order:
                    self.image_feature_order.remove(image_path)

        self.predictor.set_image(image_rgb)

        # 保存特征缓存。不同 SAM 版本可能属性略有不同，因此做保护。
        try:
            self.image_feature_cache[image_path] = {
                "features": self.predictor.features,
                "original_size": self.predictor.original_size,
                "input_size": self.predictor.input_size,
            }

            if image_path in self.image_feature_order:
                self.image_feature_order.remove(image_path)
            self.image_feature_order.append(image_path)

            while len(self.image_feature_order) > self.max_image_feature_cache:
                old_path = self.image_feature_order.pop(0)
                self.image_feature_cache.pop(old_path, None)

        except Exception:
            pass

        return False

    def update_image_counter(self):
        total = len(self.image_paths)
        if total == 0:
            self.image_counter_label.setText("当前图片：0/0")
            self.prev_image_btn.setEnabled(False)
            self.next_image_btn.setEnabled(False)
            return

        current = self.current_image_idx + 1
        self.image_counter_label.setText(f"当前图片：{current}/{total}")

        self.prev_image_btn.setEnabled(self.current_image_idx > 0)
        self.next_image_btn.setEnabled(self.current_image_idx < total - 1)

    def has_unsaved_annotations(self):
        return (
            len(self.saved_masks) > 0
            or self.current_mask is not None
            or len(self.points) > 0
        )

    def confirm_discard_current_annotations(self):
        """
        左右切换图片前，如果当前图片还有未保存标注，则确认是否丢弃。
        """
        if not self.has_unsaved_annotations():
            return True

        reply = QMessageBox.question(
            self,
            "确认切换图片",
            "当前图片存在未保存标注。\n是否丢弃当前标注并切换图片？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        return reply == QMessageBox.Yes

    def go_prev_image(self):
        if len(self.image_paths) == 0:
            return

        if self.current_image_idx <= 0:
            return

        # 切换前先把当前图片的标注状态保存到内存。
        self.cache_current_annotation_state()

        self.current_image_idx -= 1
        self.load_current_image()

    def go_next_image(self):
        if len(self.image_paths) == 0:
            return

        if self.current_image_idx >= len(self.image_paths) - 1:
            return

        # 切换前先把当前图片的标注状态保存到内存。
        self.cache_current_annotation_state()

        self.current_image_idx += 1
        self.load_current_image()

    def go_next_image_after_save(self):
        """
        保存当前图片后自动跳转下一张。
        保存后也把当前图片标注状态留在内存里，之后返回该图片仍可继续修改。
        """
        if len(self.image_paths) == 0:
            return

        self.cache_current_annotation_state()

        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self.load_current_image()
        else:
            self.update_display()
            self.update_image_counter()

            QMessageBox.information(
                self,
                "全部完成",
                "当前图片已保存，并且已经是最后一张图片。"
            )

            self.status_label.setText(
                f"状态：已保存最后一张图片\n"
                f"输出根目录：{OUTPUT_DIR}\n"
                f"当前图片标注仍保留在界面中，可继续修改后再次保存。"
            )

    # =========================
    # 类别添加
    # =========================

    def add_class(self):
        new_class = self.class_input.text().strip()

        if not new_class:
            QMessageBox.warning(self, "提示", "类别不能为空")
            return

        existing_classes = [
            self.class_combo.itemText(i)
            for i in range(self.class_combo.count())
        ]

        if new_class not in existing_classes:
            self.class_combo.addItem(new_class)

            idx = len(self.class_colors)
            self.class_colors[new_class] = self.default_colors[idx % len(self.default_colors)]

            self.class_combo.setCurrentText(new_class)
            self.class_input.clear()

            self.status_label.setText(f"状态：已添加类别：{new_class}")
        else:
            self.class_combo.setCurrentText(new_class)
            self.status_label.setText(f"状态：类别已存在，已切换到：{new_class}")

        self.update_color_preview()

    # =========================
    # 鼠标点击与 SAM 预测
    # =========================

    def on_mouse_click(self, x, y, label):
        if self.image_rgb is None:
            return

        self.points.append([x, y])
        self.labels.append(label)

        if label == 1:
            point_type = "前景点"
        else:
            point_type = "背景点"

        self.status_label.setText(
            f"状态：添加{point_type}：({x}, {y})\n"
            f"当前点数量：{len(self.points)}"
        )

        self.run_sam_predict()

    def run_sam_predict(self):
        if self.image_rgb is None:
            return

        if len(self.points) == 0:
            self.current_mask = None
            self.current_score = None
            self.all_masks = None
            self.all_scores = None
            self.update_display()
            return

        point_coords = np.array(self.points)
        point_labels = np.array(self.labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        self.all_masks = masks
        self.all_scores = scores

        selected_index = self.mask_combo.currentIndex()

        if selected_index >= len(masks):
            selected_index = 0
            self.mask_combo.setCurrentIndex(0)

        self.current_mask_index = selected_index
        self.current_mask = masks[selected_index]
        self.current_score = scores[selected_index]

        self.update_display()

        self.status_label.setText(
            f"状态：SAM 分割完成\n"
            f"当前点数量：{len(self.points)}\n"
            f"当前选择：Mask {selected_index + 1}\n"
            f"当前 mask score：{self.current_score:.4f}\n"
            f"共返回 mask 数量：{len(masks)}"
        )

    def change_mask_index(self, index):
        self.current_mask_index = index

        if self.all_masks is None or self.all_scores is None:
            return

        if index >= len(self.all_masks):
            return

        self.current_mask = self.all_masks[index]
        self.current_score = self.all_scores[index]

        self.update_display()

        self.status_label.setText(
            f"状态：已切换到 Mask {index + 1}\n"
            f"当前 mask score：{self.current_score:.4f}"
        )

    # =========================
    # 图像显示
    # =========================

    def overlay_mask(self, show_img, mask, color, alpha):
        mask_bool = mask.astype(bool)
        color_arr = np.array(color, dtype=np.float32)

        show_img[mask_bool] = (
            show_img[mask_bool] * (1 - alpha) + color_arr * alpha
        )

        return show_img

    def update_display(self):
        if self.image_rgb is None:
            return

        show_img = self.image_rgb.copy().astype(np.float32)

        # 先显示已经完成的干净 mask
        for item in self.saved_masks:
            mask = item["mask"]
            color = item["color"]

            show_img = self.overlay_mask(
                show_img=show_img,
                mask=mask,
                color=color,
                alpha=0.38
            )

        # 再显示当前正在编辑的原始候选 mask
        if self.current_mask is not None:
            current_color = self.get_current_class_color()

            show_img = self.overlay_mask(
                show_img=show_img,
                mask=self.current_mask,
                color=current_color,
                alpha=0.55
            )

        show_img = np.clip(show_img, 0, 255).astype(np.uint8)
        show_img = np.ascontiguousarray(show_img)

        # 显示当前 mask 使用的前景点和背景点
        for point, label in zip(self.points, self.labels):
            x, y = point

            if label == 1:
                color = (0, 255, 0)      # 前景点：绿色
            else:
                color = (255, 0, 0)      # 背景点：红色

            cv2.drawMarker(
                show_img,
                (x, y),
                color,
                markerType=cv2.MARKER_STAR,
                markerSize=25,
                thickness=2
            )

        h, w, c = show_img.shape
        bytes_per_line = c * w

        q_img = QImage(
            show_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        self.update_display()

    # =========================
    # 当前点操作
    # =========================

    def undo_point(self):
        if len(self.points) == 0:
            return

        self.points.pop()
        self.labels.pop()

        self.run_sam_predict()

    def clear_points(self):
        self.points = []
        self.labels = []

        self.current_mask = None
        self.current_score = None

        self.all_masks = None
        self.all_scores = None

        self.update_display()

        self.status_label.setText("状态：已清空当前点，已完成 mask 不受影响")

    # =========================
    # 完成 / 删除 已完成 mask
    # =========================

    def commit_current_mask(self):
        """
        ✔ 按钮：
        完成当前类别的 mask。

        关键修改：
        - 这里直接调用 refine_mask；
        - saved_masks 中保存的是 clean_mask；
        - 后续所有导出都基于 clean_mask。
        """
        if self.current_mask is None:
            QMessageBox.warning(self, "提示", "当前还没有生成 mask，不能完成标注")
            return

        class_name = self.class_combo.currentText()
        color = self.get_current_class_color()

        raw_mask = self.current_mask.astype(np.uint8)

        try:
            clean_mask = refine_mask(raw_mask, **REFINE_KWARGS)
        except Exception as e:
            QMessageBox.critical(
                self,
                "mask 后处理失败",
                f"refine_mask 执行失败：\n{str(e)}"
            )
            return

        if int(clean_mask.sum()) == 0:
            QMessageBox.warning(
                self,
                "提示",
                "后处理后的 mask 为空，当前标注未保存。\n"
                "建议重新点选前景点，或者降低 min_area。"
            )
            return

        saved_item = {
            "class_name": class_name,
            "mask": clean_mask.astype(np.uint8),
            "score": float(self.current_score),
            "mask_index": int(self.current_mask_index + 1),
            "points": [p.copy() for p in self.points],
            "labels": self.labels.copy(),
            "color": color
        }

        self.saved_masks.append(saved_item)

        # 刷新右侧已完成 mask 列表
        self.refresh_saved_mask_list()

        # 清空当前提示点和当前 mask，但不清空已经完成的 mask
        self.points = []
        self.labels = []

        self.current_mask = None
        self.current_score = None

        self.all_masks = None
        self.all_scores = None
        self.current_mask_index = 0
        self.mask_combo.setCurrentIndex(0)

        self.update_display()

        self.status_label.setText(
            f"状态：已完成该类标注，已保存为干净 mask\n"
            f"类别：{class_name}\n"
            f"clean mask 像素数：{int(clean_mask.sum())}\n"
            f"已完成 mask 数量：{len(self.saved_masks)}\n"
            f"现在可以继续标注下一个类别"
        )

    def refresh_saved_mask_list(self):
        """
        刷新右侧已完成 mask 列表。
        删除、添加、改颜色后都调用这个函数。
        """
        self.saved_mask_list.clear()

        for idx, item in enumerate(self.saved_masks):
            class_name = item["class_name"]
            score = item["score"]
            mask_index = item["mask_index"]
            color = item["color"]

            list_text = (
                f"{idx + 1}. {class_name} | "
                f"Mask {mask_index} | "
                f"score {score:.4f}"
            )

            list_item = QListWidgetItem(list_text)

            q_color = QColor(color[0], color[1], color[2])
            list_item.setForeground(QBrush(q_color))

            self.saved_mask_list.addItem(list_item)

    def delete_selected_saved_mask(self):
        """
        删除右侧列表中选中的已完成 mask。
        删除后图像上对应的半透明 mask 也会同步消失。
        """
        current_row = self.saved_mask_list.currentRow()

        if current_row < 0:
            QMessageBox.warning(self, "提示", "请先在右侧列表中选中一个已完成掩膜")
            return

        if current_row >= len(self.saved_masks):
            QMessageBox.warning(self, "错误", "选中的掩膜索引异常")
            return

        deleted_item = self.saved_masks.pop(current_row)

        self.refresh_saved_mask_list()
        self.update_display()

        self.status_label.setText(
            f"状态：已删除已完成掩膜\n"
            f"类别：{deleted_item['class_name']}\n"
            f"剩余 mask 数量：{len(self.saved_masks)}"
        )

    # =========================
    # 自动保存当前图片
    # =========================

    def get_unique_class_names(self, saved_masks):
        class_names = []

        for item in saved_masks:
            class_name = item["class_name"]
            if class_name not in class_names:
                class_names.append(class_name)

        return class_names

    def build_semantic_class_to_id(self, class_names):
        """
        语义分割类别：
        - background 固定为 0；
        - 真实类别从 1 开始。
        """
        class_to_id = {"background": 0}
        for idx, class_name in enumerate(class_names, start=1):
            class_to_id[class_name] = idx
        return class_to_id

    def build_detection_class_to_id(self, class_names):
        """
        YOLO 检测类别：
        - 不需要 background；
        - 类别从 0 开始。
        """
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def get_refined_saved_masks_for_export(self):
        """
        导出前再次调用 refine_mask，保证最终落盘的一定是干净 mask。
        即使 commit_current_mask 已经处理过，这里再处理一次也更稳。
        """
        refined_items = []

        for item in self.saved_masks:
            new_item = dict(item)
            new_item["mask"] = refine_mask(item["mask"], **REFINE_KWARGS).astype(np.uint8)

            if int(new_item["mask"].sum()) > 0:
                refined_items.append(new_item)

        return refined_items

    # =========================
    # 2D + 3D 导出控制
    # =========================
    def get_3d_semantic_filter_strength(self):
        """
        获取 3D 语义图滤波强度。
            "0 不滤波"
            "1 弱滤波"
            "2 中滤波"
            "3 强滤波"
        """
        if not hasattr(self, "semantic_filter_combo"):
            return 0

        text = self.semantic_filter_combo.currentText().strip()
        try:
            return int(text.split()[0])
        except Exception:
            return 0

    def get_export_format_flags(self):
        """
        根据 UI 选择判断当前保存哪些标注格式。
            - 2D 分割始终保存，因为 3D 标注依赖 binary_masks；
            - 2D 框固定按常规 YOLO + VOC 同步保存；
            - 3D 分割保存 point_class_ids / point_instance_ids / labeled ply；
            - 3D 框保存 3d_boxes.json / 3d_boxes.txt。
        """
        box_format = self.box_format_combo.currentText()
        seg_format = self.seg_format_combo.currentText()
        semantic_filter_strength = self.get_3d_semantic_filter_strength()

        save_2d_boxes = box_format in [
            "仅2D框(YOLO+VOC)",
            "2D+3D框",
        ]

        save_3d_boxes = box_format in [
            "仅3D框",
            "2D+3D框",
        ]

        save_3d_segmentation = seg_format == "2D+3D分割"

        return {
            "box_format": box_format,
            "seg_format": seg_format,
            "save_2d_boxes": save_2d_boxes,
            "save_3d_boxes": save_3d_boxes,
            "save_3d_segmentation": save_3d_segmentation,
            "semantic_filter_strength": semantic_filter_strength,
        }

    def get_current_sample_name_and_id(self):
        """
        从当前图片文件名解析样本编号。

        当前版本默认要求：
            image/000001.jpg
            pc/000001.mat
            calib/000001.txt
        """
        if self.image_path is None:
            raise RuntimeError("当前没有加载图片")

        image_name = os.path.splitext(os.path.basename(self.image_path))[0]

        if not image_name.isdigit():
            raise ValueError(
                f"当前图片名不是纯数字编号：{image_name}\n\n"
                f"3D 标注默认要求图片名类似 000001.jpg，"
                f"并且对应：\n"
                f"  {SUNRGBD_ROOT_DIR}/pc/000001.mat\n"
                f"  {SUNRGBD_ROOT_DIR}/calib/000001.txt\n\n"
                f"如果你的图片名不是数字编号，需要先建立图片名到点云编号的映射。"
            )

        image_id = int(image_name)

        return image_name, image_id

    def check_tool3d_ready(self):
        """
        检查 3D 工具是否可用。
        """
        if not TOOL3D_AVAILABLE:
            raise ImportError(
                "tool3d.py 导入失败，无法使用 3D 标注功能。\n\n"
                "请确认：\n"
                "1. 前面写好的 3D 工具函数文件已经保存为 tool3d.py；\n"
                "2. tool3d.py 与当前 PyQt 文件在同一目录，或已经加入 PYTHONPATH；\n"
                "3. tool3d.py 中包含 generate_3d_annotations_for_one_sample "
                "和 visualize_result_from_memory。\n\n"
                f"原始错误：{repr(TOOL3D_IMPORT_ERROR)}"
            )

        if not os.path.isdir(SUNRGBD_ROOT_DIR):
            raise FileNotFoundError(
                f"SUNRGBD_ROOT_DIR 不存在：\n{SUNRGBD_ROOT_DIR}\n\n"
                f"请在代码顶部修改 SUNRGBD_ROOT_DIR。"
            )

    def run_3d_export_for_current_image(
        self,
        segmentation_dir,
        save_3d_boxes=True,
        save_3d_segmentation=True,
        force_save_for_view=False
    ):
        """
        根据当前图片的 2D binary_masks 生成 3D 标注。

        参数：
            segmentation_dir:
                当前图片的 2D 分割导出目录，例如：
                sam_output/000001_segmentation

            save_3d_boxes:
                是否保存 3D 框标注。

            save_3d_segmentation:
                是否保存 3D 点云分割标注。

            force_save_for_view:
                True 时，用于“查看当前标注点云”，会强制生成完整 3D 结果。
        """
        self.check_tool3d_ready()

        image_name, image_id = self.get_current_sample_name_and_id()

        if force_save_for_view:
            save_3d_boxes = True
            save_3d_segmentation = True

        semantic_filter_strength = self.get_3d_semantic_filter_strength()

        if not save_3d_boxes and not save_3d_segmentation:
            return None

        # 构造 3D 导出参数。
        # 新版 tool3d.py 支持 semantic_filter_strength；如果当前 tool3d.py 太旧，
        # 这里会先检查函数签名，避免因为多传参数导致 PyQt 直接崩溃。
        kwargs_3d = {
            "root_dir": SUNRGBD_ROOT_DIR,
            "image_id": image_id,
            "segmentation_dir": segmentation_dir,
            "save_root": ANNOTATION_3D_SAVE_ROOT,
            "box_type": BOX_3D_TYPE,
            "min_points": MIN_3D_BOX_POINTS,
            "up_axis": BOX_3D_UP_AXIS,
            "use_zbuffer": USE_3D_ZBUFFER,
            "zbuffer_tolerance": ZBUFFER_TOLERANCE,
            "use_percentile_filter": True,
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "save_ply": save_3d_segmentation,
            "save_point_mask": save_3d_segmentation,
            "save_boxes": save_3d_boxes,
            "semantic_filter_strength": semantic_filter_strength,
        }

        try:
            sig = inspect.signature(generate_3d_annotations_for_one_sample)
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            if not has_var_kwargs:
                supported_names = set(sig.parameters.keys())

                if (
                    semantic_filter_strength > 0
                    and "semantic_filter_strength" not in supported_names
                ):
                    raise TypeError(
                        "当前 tool3d.py 不支持 semantic_filter_strength，"
                        "请使用带 3D 语义离群点滤波的新版 tool3d.py。"
                    )

                kwargs_3d = {
                    k: v for k, v in kwargs_3d.items()
                    if k in supported_names
                }

        except TypeError:
            raise
        except Exception:
            # 极少数情况下 inspect 失败，仍然按新版参数调用，让真实错误向上抛出。
            pass

        result = generate_3d_annotations_for_one_sample(**kwargs_3d)

        self.last_3d_result = result
        return result

    def view_current_annotated_pointcloud(self):
        """
        查看当前图片对应的标注点云。

        流程：
            1. 先保存当前 2D mask 到 binary_masks；
            2. 基于 binary_masks 生成 3D 点云分割和 3D 框；
            3. 打开 Open3D 窗口显示点云和 3D 框；
               点云默认按类别上色，保证和 2D 类别色逻辑一致。
        """
        if self.image_rgb is None or self.image_path is None:
            QMessageBox.warning(self, "提示", "当前没有加载图片")
            return

        if len(self.saved_masks) == 0:
            QMessageBox.warning(self, "提示", "当前图片还没有完成任何 mask 标注")
            return

        try:
            ok = self.save_current_image_annotations(auto_jump_next=False)
            if not ok:
                return

            if self.last_saved_segmentation_dir is None:
                QMessageBox.warning(self, "提示", "没有找到当前图片的 2D mask 导出目录")
                return

            result = self.run_3d_export_for_current_image(
                segmentation_dir=self.last_saved_segmentation_dir,
                save_3d_boxes=True,
                save_3d_segmentation=True,
                force_save_for_view=True
            )

            if result is None:
                QMessageBox.warning(self, "提示", "3D 标注生成失败")
                return

            self.status_label.setText(
                f"状态：正在打开 3D 点云标注视图\n"
                f"图片：{os.path.basename(self.image_path)}\n"
                f"3D语义图滤波强度：{self.get_3d_semantic_filter_strength()}\n"
                f"3D 实例数量：{len(result.get('segments', []))}\n"
                f"3D 框数量：{len(result.get('boxes', []))}"
            )

            visualize_result_from_memory(
                result,
                color_mode="class",
                show_background=True
            )

        except Exception as e:
            err = traceback.format_exc()
            print(err)
            QMessageBox.critical(
                self,
                "查看标注点云失败",
                f"错误信息：\n{str(e)}"
            )

    def save_current_image_annotations(self, auto_jump_next=False):
        """
        保存当前图片的全部标注。

        2D 分割始终保存：
            - semantic_mask
            - instance_mask
            - color_mask
            - overlay_vis
            - binary_masks

        框标注根据 UI 保存：
            - 不保存框
            - 仅2D框：YOLO + VOC
            - 仅3D框：3D boxes
            - 2D+3D框：YOLO + VOC + 3D boxes

        分割标注根据 UI 保存：
            - 仅2D分割
            - 2D+3D分割
        """
        if self.image_rgb is None or self.image_path is None:
            QMessageBox.warning(self, "提示", "当前没有加载图片")
            return False

        if len(self.saved_masks) == 0:
            QMessageBox.warning(self, "提示", "当前图片还没有完成任何 mask 标注")
            return False

        format_flags = self.get_export_format_flags()

        try:
            refined_saved_masks = self.get_refined_saved_masks_for_export()
        except Exception as e:
            QMessageBox.critical(
                self,
                "保存失败",
                f"导出前 mask 后处理失败：\n{str(e)}"
            )
            return False

        if len(refined_saved_masks) == 0:
            QMessageBox.warning(
                self,
                "提示",
                "当前图片所有 mask 后处理后均为空，无法保存。"
            )
            return False

        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_dir = os.path.join(OUTPUT_DIR, f"{image_name}_segmentation")

        os.makedirs(output_dir, exist_ok=True)

        binary_dir = os.path.join(output_dir, "binary_masks")
        os.makedirs(binary_dir, exist_ok=True)

        h, w = self.image_rgb.shape[:2]

        class_names = self.get_unique_class_names(refined_saved_masks)

        semantic_class_to_id = self.build_semantic_class_to_id(class_names)
        detection_class_to_id = self.build_detection_class_to_id(class_names)

        if len(semantic_class_to_id) <= 256:
            semantic_dtype = np.uint8
        else:
            semantic_dtype = np.uint16

        semantic_mask = np.zeros((h, w), dtype=semantic_dtype)
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        overlay_vis = self.image_rgb.copy().astype(np.float32)

        overwrite_mode = "later"

        # =========================
        # 1. 保存 2D 分割 mask
        # =========================
        for instance_id, item in enumerate(refined_saved_masks, start=1):
            class_name = item["class_name"]
            mask = item["mask"].astype(bool)
            color = item["color"]

            class_id = semantic_class_to_id[class_name]

            if overwrite_mode == "later":
                valid_region = mask
            else:
                valid_region = mask & (semantic_mask == 0)

            semantic_mask[valid_region] = class_id
            instance_mask[valid_region] = instance_id
            color_mask[valid_region] = np.array(color, dtype=np.uint8)

            alpha = 0.45
            overlay_vis[valid_region] = (
                overlay_vis[valid_region] * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            )

            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[mask] = 255

            safe_class_name = (
                class_name
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            binary_save_name = f"{image_name}_{instance_id:03d}_{safe_class_name}.png"
            binary_save_path = os.path.join(binary_dir, binary_save_name)

            cv2.imwrite(binary_save_path, binary_mask)

        overlay_vis = np.clip(overlay_vis, 0, 255).astype(np.uint8)

        semantic_save_path = os.path.join(output_dir, f"{image_name}_semantic_mask.png")
        instance_save_path = os.path.join(output_dir, f"{image_name}_instance_mask.png")
        color_save_path = os.path.join(output_dir, f"{image_name}_color_mask.png")
        overlay_save_path = os.path.join(output_dir, f"{image_name}_overlay_vis.png")
        class_json_path = os.path.join(output_dir, f"{image_name}_classes.json")

        cv2.imwrite(semantic_save_path, semantic_mask)
        cv2.imwrite(instance_save_path, instance_mask)
        cv2.imwrite(color_save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR))

        # =========================
        # 2. 保存 2D 框标注：常规 YOLO + VOC
        # =========================
        yolo_save_path = None
        voc_save_path = None

        if format_flags["save_2d_boxes"]:
            yolo_dir = os.path.join(output_dir, "labels_2d_yolo")
            os.makedirs(yolo_dir, exist_ok=True)

            yolo_save_path = os.path.join(yolo_dir, f"{image_name}.txt")

            yolo_lines = saved_masks_to_yolo_lines(
                refined_saved_masks,
                class_to_id=detection_class_to_id,
                image_width=w,
                image_height=h,
                decimals=6,
                use_refined_mask=False,
                refine_kwargs=None
            )

            save_yolo_txt(yolo_save_path, yolo_lines)

            voc_dir = os.path.join(output_dir, "labels_2d_voc")
            os.makedirs(voc_dir, exist_ok=True)

            voc_save_path = os.path.join(voc_dir, f"{image_name}.xml")

            voc_objects = saved_masks_to_voc_objects(
                refined_saved_masks,
                use_refined_mask=False,
                refine_kwargs=None
            )

            save_voc_xml(
                save_path=voc_save_path,
                filename=os.path.basename(self.image_path),
                image_width=w,
                image_height=h,
                objects=voc_objects,
                folder=os.path.basename(self.image_folder),
                image_depth=3,
                segmented=1
            )

        # =========================
        # 3. 先保存 classes.json
        #    tool3d 会读取 detection_class_to_id。
        # =========================
        export_info = {
            "image_path": self.image_path,
            "height": h,
            "width": w,
            "mask_count": len(refined_saved_masks),
            "overwrite_mode": overwrite_mode,
            "semantic_class_to_id": semantic_class_to_id,
            "detection_class_to_id": detection_class_to_id,
            "box_format": format_flags["box_format"],
            "seg_format": format_flags["seg_format"],
            "semantic_filter_strength": format_flags["semantic_filter_strength"],
            "refine_kwargs": REFINE_KWARGS,
            "formats": {
                "2d_segmentation": {
                    "semantic_mask": os.path.basename(semantic_save_path),
                    "instance_mask": os.path.basename(instance_save_path),
                    "color_mask": os.path.basename(color_save_path),
                    "overlay_vis": os.path.basename(overlay_save_path),
                    "binary_masks_dir": "binary_masks",
                },
                "2d_detection": {
                    "yolo_txt": os.path.relpath(yolo_save_path, output_dir) if yolo_save_path else None,
                    "voc_xml": os.path.relpath(voc_save_path, output_dir) if voc_save_path else None,
                },
                "3d_segmentation": None,
                "3d_detection": None,
            }
        }

        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(export_info, f, ensure_ascii=False, indent=4)

        self.last_saved_segmentation_dir = output_dir
        self.last_saved_image_name = image_name

        # =========================
        # 4. 同步保存 3D 标注
        # =========================
        result_3d = None

        if format_flags["save_3d_boxes"] or format_flags["save_3d_segmentation"]:
            try:
                result_3d = self.run_3d_export_for_current_image(
                    segmentation_dir=output_dir,
                    save_3d_boxes=format_flags["save_3d_boxes"],
                    save_3d_segmentation=format_flags["save_3d_segmentation"],
                    force_save_for_view=False
                )

                if result_3d is not None:
                    save_paths = result_3d.get("save_paths", {})

                    if format_flags["save_3d_segmentation"]:
                        export_info["formats"]["3d_segmentation"] = {
                            "point_masks": save_paths.get("point_masks", None),
                            "labeled_ply": save_paths.get("labeled_ply", None),
                            "description": "point_class_ids / point_instance_ids / labeled ply"
                        }

                    if format_flags["save_3d_boxes"]:
                        export_info["formats"]["3d_detection"] = {
                            "boxes_json": save_paths.get("boxes_json", None),
                            "boxes_txt": save_paths.get("boxes_txt", None),
                            "box_type": BOX_3D_TYPE,
                            "description": "class_name class_id instance_id cx cy cz dx dy dz heading_angle num_points"
                        }

                    export_info["3d_summary"] = {
                        "num_3d_segments": len(result_3d.get("segments", [])),
                        "num_3d_boxes": len(result_3d.get("boxes", [])),
                        "semantic_filter_strength": format_flags["semantic_filter_strength"],
                    }

            except Exception as e:
                err = traceback.format_exc()
                print(err)

                QMessageBox.critical(
                    self,
                    "3D 标注保存失败",
                    f"2D 标注已经保存，但 3D 标注生成失败。\n\n"
                    f"错误信息：\n{str(e)}"
                )
                return False

        # 重新写入 classes.json，把 3D 路径也保存进去
        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(export_info, f, ensure_ascii=False, indent=4)

        status_text = (
            f"状态：当前图片已保存\n"
            f"图片：{os.path.basename(self.image_path)}\n"
            f"输出目录：{output_dir}\n"
            f"框标注：{format_flags['box_format']}\n"
            f"分割标注：{format_flags['seg_format']}\n"
            f"3D语义图滤波强度：{format_flags['semantic_filter_strength']}"
        )

        if result_3d is not None:
            status_text += (
                f"\n3D 实例数量：{len(result_3d.get('segments', []))}"
                f"\n3D 框数量：{len(result_3d.get('boxes', []))}"
            )

        self.status_label.setText(status_text)

        return True

    def finish_current_image_and_next(self):
        """
        点击后自动保存当前图片全部标注，并跳转到下一张图片。
        """
        if len(self.image_paths) == 0:
            QMessageBox.warning(self, "提示", "当前没有加载图片文件夹")
            return

        ok = self.save_current_image_annotations(auto_jump_next=True)

        if not ok:
            return

        self.go_next_image_after_save()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMAnnotator()
    window.show()
    sys.exit(app.exec_())
