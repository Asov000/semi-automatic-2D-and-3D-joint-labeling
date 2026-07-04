# -*- coding: utf-8 -*-
"""主窗口组装模块，负责组合界面、交互、导出、图片状态和自动标注能力。"""

from .context import *
from .exporting import ExportingMixin
from .image_state import ImageStateMixin
from .interaction import InteractionMixin
from .sam3_auto import SAM3AutoMixin
from .ui_setup import UISetupMixin


class SAMAnnotator(
    UISetupMixin,
    ImageStateMixin,
    SAM3AutoMixin,
    InteractionMixin,
    ExportingMixin,
    QWidget,
):
    """标注主窗口类，组合界面、交互、导出、图片状态和自动标注能力。"""
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
        self.edited_3d_boxes_by_image = {}

        self.predictor = None
        self.sam3_auto_generator = None
        self.sam3_last_config = None

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
        self.default_colors = list(DEFAULT_CLASS_COLOR_PALETTE)

        self.init_ui()
        self.init_class_colors()
        self.load_sam_model()
        self.load_image_folder(self.image_folder)
