# -*- coding: utf-8 -*-
"""交互标注模块，负责鼠标点选、SAM 推理、mask 显示和标注提交。"""

from .context import *


class InteractionMixin:
    """交互混入类，负责点选分割、候选 mask 和显示刷新。"""
    def on_mouse_click(self, x, y, label):
        """处理图片上的鼠标点击，并将其转换为前景点或背景点。"""
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
        """根据当前提示点执行 SAM 分割，并刷新候选 mask。"""
        if self.image_rgb is None or self.image_path is None:
            return

        if len(self.points) == 0:
            self.current_mask = None
            self.current_score = None
            self.all_masks = None
            self.all_scores = None
            self.update_display()
            return

        class_name = self.class_combo.currentText().strip()
        if not class_name:
            QMessageBox.warning(self, "提示", "请先选择或添加当前类别")
            return

        try:
            generator = self.get_or_create_sam3_generator()
            auto_items = generator.predict(
                image_path=self.image_path,
                text_prompts=[class_name],
            )

            ranked_items = []
            for auto_item in auto_items:
                mask = auto_item["mask"].astype(np.uint8)
                point_score = 0.0
                positive_hits = 0
                background_violations = 0

                for point, label in zip(self.points, self.labels):
                    x, y = int(point[0]), int(point[1])
                    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
                        continue

                    hit = bool(mask[y, x] > 0)
                    if int(label) == 1:
                        if hit:
                            point_score += 20.0
                            positive_hits += 1
                        else:
                            point_score -= 20.0
                    else:
                        if hit:
                            point_score -= 25.0
                            background_violations += 1
                        else:
                            point_score += 5.0

                model_score = float(auto_item.get("score", 1.0))
                ranked_items.append((
                    model_score + point_score,
                    positive_hits,
                    -background_violations,
                    auto_item,
                    mask,
                ))

            if len(ranked_items) == 0:
                self.current_mask = None
                self.current_score = None
                self.all_masks = None
                self.all_scores = None
                self.update_display()
                self.status_label.setText(
                    f"状态：SAM3 未检测到当前类别候选 mask\n"
                    f"类别：{class_name}\n"
                    f"当前点数量：{len(self.points)}"
                )
                return

            ranked_items.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

            if ENABLE_2D_MASK_NMS and semantic_mask_nms is not None:
                nms_input_items = [
                    {
                        "class_name": class_name,
                        "mask": item[4],
                        "score": float(item[0]),
                        "ranked_item": item,
                    }
                    for item in ranked_items
                ]
                nms_output_items, self.last_candidate_mask_nms_stats = semantic_mask_nms(
                    nms_input_items,
                    iou_threshold=MASK_NMS_IOU_THRESHOLD,
                    overlap_min_threshold=MASK_NMS_OVERLAP_MIN_THRESHOLD,
                    suppress_same_class=True,
                    same_class_iou_threshold=MASK_NMS_SAME_CLASS_IOU_THRESHOLD,
                    same_class_overlap_min_threshold=MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD
                )
                ranked_items = [item["ranked_item"] for item in nms_output_items]

            masks = np.stack([item[4] for item in ranked_items], axis=0)
            scores = np.asarray([item[0] for item in ranked_items], dtype=np.float32)

            self.all_masks = masks
            self.all_scores = scores
            self.all_sam3_items = [item[3] for item in ranked_items]

            self.mask_combo.blockSignals(True)
            self.mask_combo.clear()
            for idx in range(len(masks)):
                self.mask_combo.addItem(f"Mask {idx + 1}")
            self.mask_combo.setCurrentIndex(0)
            self.mask_combo.blockSignals(False)

            self.current_mask_index = 0
            self.current_mask = masks[0]
            self.current_score = float(scores[0])

            self.ensure_class_exists(class_name)
            self.update_display()

            self.status_label.setText(
                f"状态：SAM3 当前类别候选分割完成\n"
                f"类别：{class_name}\n"
                f"当前点数量：{len(self.points)}\n"
                f"当前选择：Mask 1\n"
                f"当前 mask score：{self.current_score:.4f}\n"
                f"共返回 mask 数量：{len(masks)}"
            )

        except Exception as e:
            err = traceback.format_exc()
            print(err)
            QMessageBox.critical(
                self,
                "SAM3 点选标注失败",
                f"错误信息：\n{str(e)}"
            )

    def legacy_run_sam_predict(self):
        """使用经典 SAM 预测器执行点提示分割。"""
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
        """切换当前候选 mask，并刷新显示。"""
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

    def overlay_mask(self, show_img, mask, color, alpha):
        """把指定 mask 以半透明颜色叠加到图像上。"""
        mask_bool = mask.astype(bool)
        color_arr = np.array(color, dtype=np.float32)

        show_img[mask_bool] = (
            show_img[mask_bool] * (1 - alpha) + color_arr * alpha
        )

        return show_img

    def update_display(self):
        """刷新主图像区域，显示已完成 mask、当前候选 mask 和提示点。"""
        if self.image_rgb is None:
            return

        show_img = self.image_rgb.copy().astype(np.float32)

        selected_row = -1
        if hasattr(self, "saved_mask_list"):
            selected_row = self.saved_mask_list.currentRow()

        selected_mask = None
        selected_color = None

        # 先显示已经完成的干净 mask
        for idx, item in enumerate(self.saved_masks):
            mask = item["mask"]
            color = item["color"]
            is_selected = idx == selected_row

            show_img = self.overlay_mask(
                show_img=show_img,
                mask=mask,
                color=color,
                alpha=0.70 if is_selected else 0.38
            )

            if is_selected:
                selected_mask = mask
                selected_color = color

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

        if selected_mask is not None:
            mask_u8 = (selected_mask.astype(np.uint8) > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(show_img, contours, -1, (255, 255, 255), 3)
            cv2.drawContours(show_img, contours, -1, tuple(int(x) for x in selected_color), 1)

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
        """窗口尺寸变化时重新刷新图像显示。"""
        self.update_display()

    def undo_point(self):
        """撤销最近一次提示点，并重新执行分割。"""
        if len(self.points) == 0:
            return

        self.points.pop()
        self.labels.pop()

        self.run_sam_predict()

    def clear_points(self):
        """清空当前提示点和候选 mask。"""
        self.points = []
        self.labels = []

        self.current_mask = None
        self.current_score = None

        self.all_masks = None
        self.all_scores = None

        self.update_display()

        self.status_label.setText("状态：已清空当前点，已完成 mask 不受影响")

    def commit_current_mask(self):
        """将当前候选 mask 后处理后加入已完成标注列表。"""
        if self.current_mask is None and len(self.saved_masks) > 0:
            self.refresh_saved_mask_list()
            self.update_display()
            self.cache_current_annotation_state()
            self.status_label.setText(
                f"状态：当前没有人工待提交 mask，已保留右侧列表中的自动/已完成 mask\n"
                f"已完成 mask 数量：{len(self.saved_masks)}\n"
                f"如需落盘，请点击“完成当前图片标注并保存 → 下一张”。"
            )
            return

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
            "color": color,
            "source": "sam3_point"
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
        """刷新右侧已完成 mask 列表。"""
        if hasattr(self, "apply_saved_mask_semantic_nms"):
            self.saved_masks = self.apply_saved_mask_semantic_nms(self.saved_masks)

        self.saved_mask_list.clear()

        for idx, item in enumerate(self.saved_masks):
            class_name = item["class_name"]
            score = item["score"]
            mask_index = item["mask_index"]
            color = item["color"]

            source = item.get("source", "manual")

            if source == "sam3_auto":
                source_text = "SAM3自动"
            elif source == "sam3_point":
                source_text = "SAM3点选"
            else:
                source_text = "手动"

            list_text = (
                f"{idx + 1}. {class_name} | "
                f"{source_text} | "
                f"Mask {mask_index} | "
                f"score {score:.4f}"
            )

            list_item = QListWidgetItem(list_text)

            list_item.setForeground(QBrush(QColor(0, 0, 0)))

            self.saved_mask_list.addItem(list_item)

    def delete_selected_saved_mask(self):
        """删除右侧列表中选中的已完成 mask。"""
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
