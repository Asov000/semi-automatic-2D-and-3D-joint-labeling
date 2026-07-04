# -*- coding: utf-8 -*-
"""SAM3 自动标注模块，负责读取文本提示、执行自动分割并写入标注列表。"""

from .context import *


class SAM3AutoMixin:
    """SAM3 自动标注混入类，负责自动分割和结果写入。"""
    def choose_sam3_model_file(self):
        """通过文件选择器更新 SAM3 权重路径。"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SAM3 权重文件",
            os.path.dirname(self.sam3_model_path_input.text().strip())
            if self.sam3_model_path_input.text().strip()
            else "",
            "PyTorch Weights (*.pt *.pth);;All Files (*)"
        )

        if model_path:
            self.sam3_model_path_input.setText(model_path)
            # 权重切换后，下次自动标注重新加载
            self.sam3_auto_generator = None
            self.sam3_last_config = None

    def reset_sam3_default_params(self):
        """把 SAM3 参数恢复为配置中的默认值。"""
        self.sam3_model_path_input.setText(SAM3_MODEL_PATH)
        self.sam3_text_input.setText(SAM3_DEFAULT_TEXT_PROMPTS)
        self.sam3_conf_input.setText(str(SAM3_DEFAULT_CONF))
        self.sam3_iou_input.setText(str(SAM3_DEFAULT_IOU))
        self.sam3_imgsz_input.setText(str(SAM3_DEFAULT_IMGSZ))
        self.sam3_max_det_input.setText(str(SAM3_DEFAULT_MAX_DET))
        self.sam3_half_combo.setCurrentText(SAM3_DEFAULT_HALF_MODE)

        self.status_label.setText("状态：已恢复 SAM3 默认参数")

    def parse_sam3_text_prompts_from_ui(self):
        """从界面输入框中解析 SAM3 自动标注类别。"""
        raw_text = self.sam3_text_input.text().strip()

        if normalize_text_prompts is None:
            # 兜底解析
            raw_text = raw_text.replace("，", ",").replace("；", ",").replace(";", ",")
            prompts = [x.strip() for x in raw_text.split(",") if x.strip()]
        else:
            prompts = normalize_text_prompts(raw_text)

        return prompts

    def _parse_float_from_lineedit(self, lineedit, default_value, min_value=None, max_value=None):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        text = lineedit.text().strip()

        try:
            value = float(text)
        except Exception:
            value = float(default_value)

        if min_value is not None:
            value = max(float(min_value), value)

        if max_value is not None:
            value = min(float(max_value), value)

        return value

    def _parse_int_from_lineedit(self, lineedit, default_value, min_value=None, max_value=None):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        text = lineedit.text().strip()

        try:
            value = int(text)
        except Exception:
            value = int(default_value)

        if min_value is not None:
            value = max(int(min_value), value)

        if max_value is not None:
            value = min(int(max_value), value)

        return value

    def build_sam3_config_from_ui(self):
        """从界面读取 SAM3 参数，并构造自动推理配置。"""
        if not SAM3_AVAILABLE:
            raise ImportError(
                "SAM3 自动标注模块导入失败。\n\n"
                "请确认：\n"
                "1. sam3_auto.py 已经放在 main.py 同级目录；\n"
                "2. ultralytics 已安装；\n"
                "3. ultralytics 的 CLIP 依赖已正确安装；\n\n"
                f"原始错误：{repr(SAM3_IMPORT_ERROR)}"
            )

        model_path = self.sam3_model_path_input.text().strip()

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SAM3 权重不存在：\n{model_path}")

        conf = self._parse_float_from_lineedit(
            self.sam3_conf_input,
            SAM3_DEFAULT_CONF,
            min_value=0.0,
            max_value=1.0
        )

        iou = self._parse_float_from_lineedit(
            self.sam3_iou_input,
            SAM3_DEFAULT_IOU,
            min_value=0.0,
            max_value=1.0
        )

        imgsz = self._parse_int_from_lineedit(
            self.sam3_imgsz_input,
            SAM3_DEFAULT_IMGSZ,
            min_value=320
        )

        max_det = self._parse_int_from_lineedit(
            self.sam3_max_det_input,
            SAM3_DEFAULT_MAX_DET,
            min_value=1
        )

        half_mode = self.sam3_half_combo.currentText().strip()

        if half_mode == "开启":
            half = True
        elif half_mode == "关闭":
            half = False
        else:
            half = torch.cuda.is_available()

        config = SAM3AutoConfig(
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            half=half,
            device=SAM3_DEFAULT_DEVICE,
            verbose=SAM3_DEFAULT_VERBOSE
        )

        return config

    def get_or_create_sam3_generator(self):
        """获取或创建 SAM3 自动标注生成器。"""
        config = self.build_sam3_config_from_ui()

        if self.sam3_auto_generator is None:
            self.sam3_auto_generator = SAM3AutoMaskGenerator(config)
            self.sam3_last_config = config
            return self.sam3_auto_generator

        old_model_path = (
            self.sam3_last_config.model_path
            if self.sam3_last_config is not None
            else None
        )

        self.sam3_auto_generator.update_config(config)
        self.sam3_last_config = config

        # update_config 内部会在权重变化时重置 predictor
        if old_model_path != config.model_path:
            self.status_label.setText("状态：SAM3 权重路径变化，下次推理将重新加载模型")

        return self.sam3_auto_generator

    def ensure_class_exists(self, class_name):
        """确保指定类别存在于界面类别列表和颜色表中。"""
        class_name = str(class_name).strip()

        if not class_name:
            return

        existing_classes = [
            self.class_combo.itemText(i)
            for i in range(self.class_combo.count())
        ]

        if class_name not in existing_classes:
            self.class_combo.addItem(class_name)

        if class_name not in self.class_colors:
            idx = len(self.class_colors)
            self.class_colors[class_name] = get_registered_class_color(class_name, idx)

    def get_color_by_class_name(self, class_name):
        """根据类别名称获取稳定显示颜色。"""
        self.ensure_class_exists(class_name)
        return self.class_colors[class_name]

    def run_sam3_auto_label_current_image(self):
        """对当前图片执行 SAM3 自动标注，并把结果加入已完成 mask 列表。"""
        if self.image_rgb is None or self.image_path is None:
            QMessageBox.warning(self, "提示", "当前没有加载图片")
            return

        prompts = self.parse_sam3_text_prompts_from_ui()

        if len(prompts) == 0:
            QMessageBox.warning(self, "提示", "请先输入 SAM3 自动标注类别，例如 chair, table, sofa")
            return

        try:
            generator = self.get_or_create_sam3_generator()

            self.status_label.setText(
                f"状态：正在使用 SAM3 自动标注当前图片...\n"
                f"图片：{os.path.basename(self.image_path)}\n"
                f"类别：{', '.join(prompts)}"
            )
            QApplication.processEvents()

            auto_items = generator.predict(
                image_path=self.image_path,
                text_prompts=prompts
            )

            if len(auto_items) == 0:
                QMessageBox.information(self, "提示", "SAM3 未检测到有效实例")
                self.status_label.setText("状态：SAM3 未检测到有效实例")
                return

            added_count = 0
            skipped_count = 0

            for auto_idx, auto_item in enumerate(auto_items, start=1):
                class_name = str(auto_item.get("class_name", "unknown")).strip()
                score = float(auto_item.get("score", 1.0))
                raw_mask = auto_item["mask"].astype(np.uint8)

                try:
                    clean_mask = refine_mask(raw_mask, **REFINE_KWARGS)
                except Exception:
                    skipped_count += 1
                    continue

                if int(clean_mask.sum()) == 0:
                    skipped_count += 1
                    continue

                self.ensure_class_exists(class_name)
                color = self.get_color_by_class_name(class_name)

                saved_item = {
                    "class_name": class_name,
                    "mask": clean_mask.astype(np.uint8),
                    "score": score,
                    "mask_index": int(auto_idx),
                    "points": [],
                    "labels": [],
                    "color": color,
                    "source": "sam3_auto",
                    "bbox_xyxy": auto_item.get("bbox_xyxy", None),
                }

                self.saved_masks.append(saved_item)
                added_count += 1

            self.refresh_saved_mask_list()
            self.update_display()
            self.cache_current_annotation_state()

            self.status_label.setText(
                f"状态：SAM3 自动标注完成，但尚未保存到磁盘\n"
                f"图片：{os.path.basename(self.image_path)}\n"
                f"输入类别：{', '.join(prompts)}\n"
                f"新增 mask 数量：{added_count}\n"
                f"跳过 mask 数量：{skipped_count}\n"
                f"当前已完成 mask 总数：{len(self.saved_masks)}\n"
                f"如结果错误，可在右侧列表选中后删除"
            )

        except Exception as e:
            err = traceback.format_exc()
            print(err)

            QMessageBox.critical(
                self,
                "SAM3 自动标注失败",
                f"错误信息：\n{str(e)}"
            )
