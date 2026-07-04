# -*- coding: utf-8 -*-
"""自定义控件模块，提供支持图像坐标映射的 QLabel 控件。"""

from .context import QLabel, Qt, pyqtSignal


class ImageLabel(QLabel):
    """图片显示控件，支持把鼠标点击映射回原图坐标。"""
    mouse_clicked = pyqtSignal(int, int, int)

    def __init__(self):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222222;")
        self.image_width = 0
        self.image_height = 0

    def set_image_size(self, w, h):
        """记录原始图片尺寸，用于鼠标坐标映射。"""
        self.image_width = w
        self.image_height = h

    def mousePressEvent(self, event):
        """把控件点击位置换算为原图坐标，并发出前景或背景点信号。"""
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
