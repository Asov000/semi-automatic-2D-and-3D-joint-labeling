# -*- coding: utf-8 -*-
"""图片状态模块，负责图片加载、翻页、临时标注缓存和特征缓存。"""

from .context import *


class ImageStateMixin:
    """图片状态混入类，负责图片加载、翻页和缓存。"""
    def load_sam_model(self):
        """根据当前模式加载标注所需的分割模型。"""
        self.predictor = None
        self.status_label.setText("状态：已切换为 SAM3 标注模式，旧 SAM 模型不再加载")

    def load_legacy_sam_model(self):
        """加载经典 SAM 模型，并初始化点提示分割器。"""
        self.status_label.setText("状态：正在加载 SAM 模型...")

        model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        model.to(device=SAM_DEVICE)

        self.predictor = SamPredictor(model)

        self.status_label.setText(f"状态：SAM 模型加载完成，当前设备：{SAM_DEVICE}")

    def load_image_folder(self, folder_path):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
        """打开新的图片文件夹，并重置当前文件夹的临时状态。"""
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
        """读取当前索引对应的图片，并恢复该图片已有的临时标注。"""
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

        cache_text = "SAM3 模式，无需旧 SAM 图像特征缓存"

        self.status_label.setText(
            f"状态：图片加载完成（{cache_text}）\n"
            f"当前进度：{self.current_image_idx + 1}/{len(self.image_paths)}\n"
            f"图片：{os.path.basename(self.image_path)}\n"
            f"宽度：{w}\n"
            f"高度：{h}\n"
            f"已恢复 mask 数量：{len(self.saved_masks)}"
        )

    def clear_annotation_state(self):
        """清空当前图片的临时点、候选 mask 和已完成 mask。"""
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
        """深拷贝已完成的 mask，避免不同图片之间共享数组对象。"""
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
        """把当前图片的标注状态保存到内存缓存。"""
        if self.image_path is None:
            return

        self.annotation_cache[self.image_path] = {
            "saved_masks": self.clone_saved_masks(self.saved_masks),
            "points": [p.copy() for p in self.points],
            "labels": list(self.labels),
            "current_mask_index": int(self.current_mask_index),
        }

    def restore_annotation_state_from_memory(self, image_path):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
        """设置当前预测图片，并在可用时复用图像特征缓存。"""
        return False

    def legacy_set_predictor_image_with_cache(self, image_path, image_rgb):
        """为经典 SAM 设置当前图片，并维护图像特征缓存。"""
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
        """刷新界面上的当前图片序号显示。"""
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
        """判断当前图片是否存在尚未落盘的标注内容。"""
        return (
            len(self.saved_masks) > 0
            or self.current_mask is not None
            or len(self.points) > 0
        )

    def confirm_discard_current_annotations(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
        """保存当前临时状态后切换到上一张图片。"""
        if len(self.image_paths) == 0:
            return

        if self.current_image_idx <= 0:
            return

        # 切换前先把当前图片的标注状态保存到内存。
        self.cache_current_annotation_state()

        self.current_image_idx -= 1
        self.load_current_image()

    def go_next_image(self):
        """保存当前临时状态后切换到下一张图片。"""
        if len(self.image_paths) == 0:
            return

        if self.current_image_idx >= len(self.image_paths) - 1:
            return

        # 切换前先把当前图片的标注状态保存到内存。
        self.cache_current_annotation_state()

        self.current_image_idx += 1
        self.load_current_image()

    def go_next_image_after_save(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
