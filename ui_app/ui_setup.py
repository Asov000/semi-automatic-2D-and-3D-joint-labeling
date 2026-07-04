# -*- coding: utf-8 -*-
"""界面布局模块，创建图片显示区、工具栏、参数输入框和操作按钮。"""

from .context import *
from .widgets import ImageLabel


class UISetupMixin:
    """界面布局混入类，负责创建主窗口所有控件。"""
    def init_ui(self):
        """创建主窗口界面控件，并连接对应的按钮事件。"""
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

        # SAM3 候选 mask 选择
        side_layout.addWidget(QLabel("SAM3 候选 Mask："))

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

        # =========================
        # SAM3 自动标注
        # =========================
        side_layout.addWidget(QLabel("SAM3 自动标注："))

        self.sam3_model_path_input = QLineEdit()
        self.sam3_model_path_input.setText(SAM3_MODEL_PATH)
        self.sam3_model_path_input.setPlaceholderText("SAM3 权重路径，例如 D:\\sam3.pt")
        side_layout.addWidget(self.sam3_model_path_input)

        self.sam3_select_model_btn = QPushButton("选择 SAM3 权重")
        self.sam3_select_model_btn.clicked.connect(self.choose_sam3_model_file)
        side_layout.addWidget(self.sam3_select_model_btn)

        self.sam3_text_input = QLineEdit()
        self.sam3_text_input.setText(SAM3_DEFAULT_TEXT_PROMPTS)
        self.sam3_text_input.setPlaceholderText("自动标注类别，例如 chair, table, sofa")
        side_layout.addWidget(self.sam3_text_input)

        sam3_param_layout_1 = QHBoxLayout()

        self.sam3_conf_input = QLineEdit()
        self.sam3_conf_input.setText(str(SAM3_DEFAULT_CONF))
        self.sam3_conf_input.setPlaceholderText("conf")
        sam3_param_layout_1.addWidget(QLabel("conf"))
        sam3_param_layout_1.addWidget(self.sam3_conf_input)

        self.sam3_iou_input = QLineEdit()
        self.sam3_iou_input.setText(str(SAM3_DEFAULT_IOU))
        self.sam3_iou_input.setPlaceholderText("iou")
        sam3_param_layout_1.addWidget(QLabel("iou"))
        sam3_param_layout_1.addWidget(self.sam3_iou_input)

        side_layout.addLayout(sam3_param_layout_1)

        sam3_param_layout_2 = QHBoxLayout()

        self.sam3_imgsz_input = QLineEdit()
        self.sam3_imgsz_input.setText(str(SAM3_DEFAULT_IMGSZ))
        self.sam3_imgsz_input.setPlaceholderText("imgsz")
        sam3_param_layout_2.addWidget(QLabel("imgsz"))
        sam3_param_layout_2.addWidget(self.sam3_imgsz_input)

        self.sam3_max_det_input = QLineEdit()
        self.sam3_max_det_input.setText(str(SAM3_DEFAULT_MAX_DET))
        self.sam3_max_det_input.setPlaceholderText("max_det")
        sam3_param_layout_2.addWidget(QLabel("max_det"))
        sam3_param_layout_2.addWidget(self.sam3_max_det_input)

        side_layout.addLayout(sam3_param_layout_2)

        self.sam3_half_combo = QComboBox()
        self.sam3_half_combo.addItems(["自动", "开启", "关闭"])
        self.sam3_half_combo.setCurrentText(SAM3_DEFAULT_HALF_MODE)
        side_layout.addWidget(QLabel("SAM3 half 精度："))
        side_layout.addWidget(self.sam3_half_combo)

        self.sam3_reset_btn = QPushButton("恢复 SAM3 默认参数")
        self.sam3_reset_btn.clicked.connect(self.reset_sam3_default_params)
        side_layout.addWidget(self.sam3_reset_btn)

        self.sam3_auto_btn = QPushButton("SAM3 自动标注当前图片")
        self.sam3_auto_btn.clicked.connect(self.run_sam3_auto_label_current_image)
        side_layout.addWidget(self.sam3_auto_btn)

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
        self.saved_mask_list.currentRowChanged.connect(lambda _row: self.update_display())
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

        # =========================
        # 3D 框点云密度过滤阈值
        # =========================
        density_layout = QHBoxLayout()

        density_layout.addWidget(QLabel("3D框密度阈值："))

        self.box_density_input = QLineEdit()
        self.box_density_input.setText(str(MIN_3D_BOX_DENSITY))
        self.box_density_input.setPlaceholderText("如 150 / 300 / 500")
        self.box_density_input.setToolTip(
            "用于过滤被飞点拉大的低质量3D框。\n"
            "计算方式：point_density = 框内点数 / 3D框体积。\n"
            "低于该阈值的实例会被删除，点云语义改回背景。\n"
            "值越大，过滤越严格。建议先试 150、300、500。"
        )

        density_layout.addWidget(self.box_density_input)

        side_layout.addLayout(density_layout)

        min_points_layout = QHBoxLayout()
        min_points_layout.addWidget(QLabel("3D框最少点数："))

        self.box_inner_points_input = QLineEdit()
        self.box_inner_points_input.setText(str(MIN_3D_BOX_INNER_POINTS))
        self.box_inner_points_input.setPlaceholderText("如 5000")
        self.box_inner_points_input.setToolTip(
            "用于删除语义点太少的 3D 框。\n"
            "计算方式：只统计当前实例语义点中落在该 3D 框内的点，背景点不参与。"
        )
        min_points_layout.addWidget(self.box_inner_points_input)

        side_layout.addLayout(min_points_layout)

        # 查看当前标注点云
        self.view_3d_btn = QPushButton("查看当前标注点云")
        self.view_3d_btn.clicked.connect(self.view_current_annotated_pointcloud)
        side_layout.addWidget(self.view_3d_btn)

        self.edit_3d_boxes_btn = QPushButton("交互式编辑 3D 框")
        self.edit_3d_boxes_btn.clicked.connect(self.edit_current_3d_boxes)
        side_layout.addWidget(self.edit_3d_boxes_btn)

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

    def init_class_colors(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        for i in range(self.class_combo.count()):
            class_name = self.class_combo.itemText(i)
            self.class_colors[class_name] = get_registered_class_color(class_name, i)

        self.update_color_preview()

    def get_current_class_color(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        class_name = self.class_combo.currentText()

        if class_name not in self.class_colors:
            idx = len(self.class_colors)
            self.class_colors[class_name] = get_registered_class_color(class_name, idx)

        return self.class_colors[class_name]

    def update_color_preview(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        color = self.get_current_class_color()
        r, g, b = color

        self.color_preview.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid black;"
        )

        self.update_display()

    def on_class_changed(self, class_name):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if class_name not in self.class_colors:
            idx = len(self.class_colors)
            self.class_colors[class_name] = get_registered_class_color(class_name, idx)

        self.update_color_preview()

    def choose_class_color(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
            set_registered_class_color(class_name, new_color)

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

    def add_class(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
            self.class_colors[new_class] = get_registered_class_color(new_class, idx)

            self.class_combo.setCurrentText(new_class)
            self.class_input.clear()

            self.status_label.setText(f"状态：已添加类别：{new_class}")
        else:
            self.class_combo.setCurrentText(new_class)
            self.status_label.setText(f"状态：类别已存在，已切换到：{new_class}")

        self.update_color_preview()
