# -*- coding: utf-8 -*-
"""标注导出模块，负责二维分割、二维框、三维语义点云和三维框保存。"""

from .context import *
from .box_editor import interactive_edit_boxes_open3d
import tempfile


class ExportingMixin:
    """导出混入类，负责二维和三维标注保存。"""
    def get_unique_class_names(self, saved_masks):
        """按出现顺序收集已完成 mask 中的类别名称。"""
        class_names = []

        for item in saved_masks:
            class_name = item["class_name"]
            if class_name not in class_names:
                class_names.append(class_name)

        return class_names

    def build_semantic_class_to_id(self, class_names):
        """构建语义分割类别到编号的映射，背景固定为零。"""
        class_to_id = {"background": 0}
        for idx, class_name in enumerate(class_names, start=1):
            class_to_id[class_name] = idx
        return class_to_id

    def build_detection_class_to_id(self, class_names):
        """构建检测框类别到编号的映射，类别从零开始。"""
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def get_refined_saved_masks_for_export(self):
        """导出前对已完成 mask 再次执行后处理和重叠抑制。"""
        refined_items = []

        for item in self.saved_masks:
            new_item = dict(item)
            new_item["mask"] = refine_mask(item["mask"], **REFINE_KWARGS).astype(np.uint8)

            if int(new_item["mask"].sum()) > 0:
                refined_items.append(new_item)

        return self.apply_saved_mask_semantic_nms(refined_items)

    def compute_mask_overlap_stats(self, mask_a, mask_b):
        """计算两个 mask 的重叠面积、并集、交并比和覆盖率。"""
        a = np.asarray(mask_a).astype(bool)
        b = np.asarray(mask_b).astype(bool)

        inter = int(np.logical_and(a, b).sum())
        area_a = int(a.sum())
        area_b = int(b.sum())
        union = area_a + area_b - inter
        min_area = max(min(area_a, area_b), 1)

        return {
            "intersection": inter,
            "iou": float(inter / max(union, 1)),
            "overlap_min": float(inter / min_area),
            "area_a": area_a,
            "area_b": area_b,
        }

    def apply_saved_mask_semantic_nms(
        self,
        items,
        iou_threshold=None,
        overlap_min_threshold=None
    ):
        """对已完成 mask 执行语义级重叠抑制。"""
        if iou_threshold is None:
            iou_threshold = MASK_NMS_IOU_THRESHOLD
        if overlap_min_threshold is None:
            overlap_min_threshold = MASK_NMS_OVERLAP_MIN_THRESHOLD

        if not ENABLE_2D_MASK_NMS:
            self.last_semantic_nms_stats = {
                "enabled": False,
                "before": int(len(items)),
                "after": int(len(items)),
                "removed": [],
                "reason": "disabled_by_run_config",
            }
            return items

        if semantic_mask_nms is None:
            filtered_items = items
            self.last_semantic_nms_stats = {
                "enabled": False,
                "before": int(len(items)),
                "after": int(len(items)),
                "removed": [],
                "reason": "semantic_mask_nms_import_failed",
            }
            return filtered_items

        filtered_items, stats = semantic_mask_nms(
            items,
            iou_threshold=iou_threshold,
            overlap_min_threshold=overlap_min_threshold,
            suppress_same_class=True,
            same_class_iou_threshold=MASK_NMS_SAME_CLASS_IOU_THRESHOLD,
            same_class_overlap_min_threshold=MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD
        )
        self.last_semantic_nms_stats = stats

        return filtered_items

    def normalize_saved_masks_with_semantic_nms(self):
        """用语义抑制结果更新已完成 mask 列表。"""
        before = len(self.saved_masks)
        self.saved_masks = self.apply_saved_mask_semantic_nms(self.saved_masks)
        after = len(self.saved_masks)
        return before - after

    def get_safe_class_name_for_mask_file(self, class_name):
        """把类别名称转换为适合文件名使用的安全字符串。"""
        return (
            str(class_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    def build_temporary_segmentation_dir_for_view(self, temp_root):
        """为临时三维预览构建二维 mask 导出目录。"""
        image_name, _image_id = self.get_current_sample_name_and_id()
        refined_saved_masks = self.get_refined_saved_masks_for_export()

        if len(refined_saved_masks) == 0:
            raise RuntimeError("当前图片没有可用于查看的有效 mask")

        segmentation_dir = os.path.join(temp_root, f"{image_name}_segmentation")
        binary_dir = os.path.join(segmentation_dir, "binary_masks")
        os.makedirs(binary_dir, exist_ok=True)

        h, w = self.image_rgb.shape[:2]
        class_names = self.get_unique_class_names(refined_saved_masks)
        detection_class_to_id = self.build_detection_class_to_id(class_names)

        for instance_id, item in enumerate(refined_saved_masks, start=1):
            class_name = item["class_name"]
            mask = item["mask"].astype(bool)

            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[mask] = 255

            safe_class_name = self.get_safe_class_name_for_mask_file(class_name)
            binary_save_name = f"{image_name}_{instance_id:03d}_{safe_class_name}.png"
            binary_save_path = os.path.join(binary_dir, binary_save_name)

            if not cv2.imwrite(binary_save_path, binary_mask):
                raise RuntimeError(f"临时 mask 写入失败: {binary_save_path}")

        class_json_path = os.path.join(segmentation_dir, f"{image_name}_classes.json")
        export_info = {
            "image_path": self.image_path,
            "height": h,
            "width": w,
            "mask_count": len(refined_saved_masks),
            "detection_class_to_id": detection_class_to_id,
            "class_color_table": {
                str(class_name): [int(v) for v in self.class_colors[class_name]]
                for class_name in class_names
                if class_name in self.class_colors
            },
            "semantic_nms_stats": getattr(self, "last_semantic_nms_stats", None),
            "temporary_view_only": True,
        }

        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(export_info, f, ensure_ascii=False, indent=4)

        return segmentation_dir

    def get_3d_semantic_filter_strength(self):
        """从界面读取三维语义滤波强度。"""
        if not hasattr(self, "semantic_filter_combo"):
            return 0

        text = self.semantic_filter_combo.currentText().strip()
        try:
            return int(text.split()[0])
        except Exception:
            return 0

    def get_export_format_flags(self):
        """根据界面选择计算当前需要导出的标注格式。"""
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
            "enable_2d_mask_nms": ENABLE_2D_MASK_NMS,
            "use_3d_zbuffer": USE_3D_ZBUFFER,
            "use_3d_box_percentile_filter": USE_3D_BOX_PERCENTILE_FILTER,
            "box_3d_lower_percentile": BOX_3D_LOWER_PERCENTILE,
            "box_3d_upper_percentile": BOX_3D_UPPER_PERCENTILE,
            "enable_3d_box_nms": ENABLE_3D_BOX_NMS,
            "min_3d_box_inner_points": self.get_3d_box_inner_points_threshold(),
        }

    def get_3d_box_density_threshold(self):
        """从界面读取三维框点云密度阈值。"""
        if not hasattr(self, "box_density_input"):
            return float(MIN_3D_BOX_DENSITY)

        text = self.box_density_input.text().strip()

        try:
            value = float(text)
        except Exception:
            value = float(MIN_3D_BOX_DENSITY)

        # 防止误输入负数
        value = max(0.0, value)

        return value

    def get_3d_box_inner_points_threshold(self):
        """从界面读取三维框内最少语义点数量。"""
        if not hasattr(self, "box_inner_points_input"):
            return int(MIN_3D_BOX_INNER_POINTS)

        text = self.box_inner_points_input.text().strip()

        try:
            value = int(text)
        except Exception:
            value = int(MIN_3D_BOX_INNER_POINTS)

        return max(0, value)

    def get_current_sample_name_and_id(self):
        """从当前图片文件名解析样本名称和数字编号。"""
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
        """检查三维标注工具和数据根目录是否可用。"""
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
        force_save_for_view=False,
        save_root=None
    ):
        """根据当前二维分割结果生成对应的三维标注。"""
        self.check_tool3d_ready()

        image_name, image_id = self.get_current_sample_name_and_id()

        semantic_filter_strength = self.get_3d_semantic_filter_strength()
        min_box_density = self.get_3d_box_density_threshold()
        min_box_inner_points = self.get_3d_box_inner_points_threshold()
        class_name_to_color = {
            str(class_name): tuple(int(v) for v in color)
            for class_name, color in self.class_colors.items()
        }

        should_compute = bool(force_save_for_view or save_3d_boxes or save_3d_segmentation)

        if not should_compute:
            return None

        # 构造 3D 导出参数。
        # 新版 tool3d.py 支持 semantic_filter_strength；如果当前 tool3d.py 太旧，
        # 这里会先检查函数签名，避免因为多传参数导致 PyQt 直接崩溃。
        kwargs_3d = {
            "root_dir": SUNRGBD_ROOT_DIR,
            "image_id": image_id,
            "segmentation_dir": segmentation_dir,
            "save_root": save_root if save_root is not None else ANNOTATION_3D_SAVE_ROOT,
            "box_type": BOX_3D_TYPE,
            "min_points": MIN_3D_BOX_POINTS,
            "up_axis": BOX_3D_UP_AXIS,
            "use_zbuffer": USE_3D_ZBUFFER,
            "zbuffer_tolerance": ZBUFFER_TOLERANCE,
            "use_percentile_filter": USE_3D_BOX_PERCENTILE_FILTER,
            "lower_percentile": BOX_3D_LOWER_PERCENTILE,
            "upper_percentile": BOX_3D_UPPER_PERCENTILE,
            "enable_mask_nms": ENABLE_2D_MASK_NMS,
            "mask_nms_iou_threshold": MASK_NMS_IOU_THRESHOLD,
            "mask_nms_overlap_min_threshold": MASK_NMS_OVERLAP_MIN_THRESHOLD,
            "mask_nms_same_class_iou_threshold": MASK_NMS_SAME_CLASS_IOU_THRESHOLD,
            "mask_nms_same_class_overlap_min_threshold": MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD,
            "save_ply": save_3d_segmentation,
            "save_point_mask": save_3d_segmentation,
            "save_boxes": save_3d_boxes,
            "semantic_filter_strength": semantic_filter_strength,

            # 新增：3D 框质量修正
            "box_density_filter_enabled": ENABLE_3D_BOX_DENSITY_FILTER,
            "min_box_density": min_box_density,
            "min_box_inner_points": min_box_inner_points,
            "max_box_volume": MAX_3D_BOX_VOLUME,
            "box_nms_enabled": ENABLE_3D_BOX_NMS,
            "box_nms_iou_thresh": BOX_3D_NMS_IOU_THRESH,
            "box_nms_class_aware": BOX_3D_NMS_CLASS_AWARE,
            "remove_suppressed_box_points": REMOVE_SUPPRESSED_3D_BOX_POINTS,
            "class_name_to_color": class_name_to_color,
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

    def apply_cached_3d_box_edits(self, result, image_name=None, write_files=True):
        """把当前图片缓存的三维框编辑结果应用到生成结果中。"""
        if result is None:
            return result

        if image_name is None:
            image_name = str(result.get("sample_name", ""))

        if not hasattr(self, "edited_3d_boxes_by_image"):
            self.edited_3d_boxes_by_image = {}

        if image_name not in self.edited_3d_boxes_by_image:
            return result

        result["boxes"] = [
            dict(box)
            for box in self.edited_3d_boxes_by_image[image_name]
        ]

        if write_files:
            self.write_3d_boxes_for_result(result)

        return result

    def write_3d_boxes_for_result(self, result):
        """把内存中的三维框结果写回 JSON 和文本文件。"""
        if result is None:
            return

        save_paths = result.get("save_paths", {})
        boxes = result.get("boxes", [])

        boxes_json_path = save_paths.get("boxes_json", None)
        boxes_txt_path = save_paths.get("boxes_txt", None)

        extra_info = None
        if boxes_json_path and os.path.exists(boxes_json_path):
            try:
                with open(boxes_json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                extra_info = old_data.get("extra_info", None)
            except Exception:
                extra_info = None

        if boxes_json_path and save_3d_boxes_json is not None:
            save_3d_boxes_json(
                save_path=boxes_json_path,
                boxes=boxes,
                extra_info=extra_info
            )

        if boxes_txt_path and save_3d_boxes_txt is not None:
            save_3d_boxes_txt(
                save_path=boxes_txt_path,
                boxes=boxes
            )

    def get_or_build_3d_result_for_box_edit(self):
        """获取当前图片可编辑的三维结果，必要时临时生成。"""
        image_name, _image_id = self.get_current_sample_name_and_id()

        result = getattr(self, "last_3d_result", None)
        if result is not None and str(result.get("sample_name", "")) == image_name:
            return self.apply_cached_3d_box_edits(result, image_name=image_name, write_files=False)

        with tempfile.TemporaryDirectory(prefix="sam_edit_3d_boxes_") as temp_root:
            segmentation_dir = self.build_temporary_segmentation_dir_for_view(temp_root)
            result = self.run_3d_export_for_current_image(
                segmentation_dir=segmentation_dir,
                save_3d_boxes=False,
                save_3d_segmentation=False,
                force_save_for_view=True,
                save_root=temp_root
            )

        if result is not None:
            self.last_3d_result = result
            result = self.apply_cached_3d_box_edits(result, image_name=image_name, write_files=False)

        return result

    def edit_current_3d_boxes(self):
        """打开交互式三维框编辑窗口，并缓存保存后的编辑结果。"""
        if self.image_rgb is None or self.image_path is None:
            QMessageBox.warning(self, "提示", "当前没有加载图片")
            return

        if len(self.saved_masks) == 0:
            QMessageBox.warning(self, "提示", "当前图片还没有完成任何 mask 标注")
            return

        try:
            image_name, _image_id = self.get_current_sample_name_and_id()
            result = self.get_or_build_3d_result_for_box_edit()
        except Exception as e:
            QMessageBox.critical(
                self,
                "3D 框编辑失败",
                f"生成当前 3D 框失败：\n{str(e)}"
            )
            return

        if result is None or len(result.get("boxes", [])) == 0:
            QMessageBox.warning(self, "提示", "当前没有可编辑的 3D 框")
            return

        try:
            edited_boxes, saved = interactive_edit_boxes_open3d(
                result=result,
                boxes=result.get("boxes", []),
                edit_step=0.05,
                rotate_step_degrees=5.0,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "交互式 3D 框编辑失败",
                f"打开或编辑 3D 框失败：\n{str(e)}"
            )
            return

        if not saved:
            self.status_label.setText(
                "状态：交互式 3D 框编辑窗口已关闭，未按 S 保存修改。"
            )
            return

        result["boxes"] = edited_boxes

        if not hasattr(self, "edited_3d_boxes_by_image"):
            self.edited_3d_boxes_by_image = {}
        self.edited_3d_boxes_by_image[image_name] = [dict(box) for box in edited_boxes]

        self.last_3d_result = result
        self.write_3d_boxes_for_result(result)

        self.status_label.setText(
            f"状态：已编辑当前图片的 3D 框\n"
            f"图片：{os.path.basename(self.image_path)}\n"
            f"当前 3D 框数量：{len(edited_boxes)}\n"
            f"提示：如果随后保存当前图片，已编辑的 3D 框会写入导出文件。"
        )

    def view_current_annotated_pointcloud(self):
        """生成当前图片的三维标注结果，并打开 Open3D 视图查看。"""
        if self.image_rgb is None or self.image_path is None:
            QMessageBox.warning(self, "提示", "当前没有加载图片")
            return

        if len(self.saved_masks) == 0:
            QMessageBox.warning(self, "提示", "当前图片还没有完成任何 mask 标注")
            return

        try:
            with tempfile.TemporaryDirectory(prefix="sam_view_3d_") as temp_root:
                view_segmentation_dir = self.build_temporary_segmentation_dir_for_view(temp_root)

                result = self.run_3d_export_for_current_image(
                    segmentation_dir=view_segmentation_dir,
                    save_3d_boxes=False,
                    save_3d_segmentation=False,
                    force_save_for_view=True,
                    save_root=temp_root
                )

                if result is None:
                    QMessageBox.warning(self, "提示", "3D 标注生成失败")
                    return

                result = self.apply_cached_3d_box_edits(result, write_files=False)

                self.status_label.setText(
                    f"状态：正在打开 3D 点云查看窗口，本次查看不会保存数据\n"
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

            return

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

            result = self.apply_cached_3d_box_edits(result, write_files=True)

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
        """保存当前图片的二维分割、二维框和可选三维标注。"""
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

            safe_class_name = self.get_safe_class_name_for_mask_file(class_name)

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
            "class_color_table": {
                str(class_name): [int(v) for v in self.class_colors[class_name]]
                for class_name in class_names
                if class_name in self.class_colors
            },
            "box_format": format_flags["box_format"],
            "seg_format": format_flags["seg_format"],
            "semantic_filter_strength": format_flags["semantic_filter_strength"],
            "postprocess_config": {
                "enable_2d_mask_nms": format_flags["enable_2d_mask_nms"],
                "mask_nms_iou_threshold": MASK_NMS_IOU_THRESHOLD,
                "mask_nms_overlap_min_threshold": MASK_NMS_OVERLAP_MIN_THRESHOLD,
                "mask_nms_same_class_iou_threshold": MASK_NMS_SAME_CLASS_IOU_THRESHOLD,
                "mask_nms_same_class_overlap_min_threshold": MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD,
                "refine_2d_mask": REFINE_KWARGS,
                "use_3d_zbuffer": format_flags["use_3d_zbuffer"],
                "use_3d_box_percentile_filter": format_flags["use_3d_box_percentile_filter"],
                "box_3d_lower_percentile": format_flags["box_3d_lower_percentile"],
                "box_3d_upper_percentile": format_flags["box_3d_upper_percentile"],
                "enable_3d_box_nms": format_flags["enable_3d_box_nms"],
                "min_3d_box_inner_points": format_flags["min_3d_box_inner_points"],
            },
            "semantic_nms_stats": getattr(self, "last_semantic_nms_stats", None),
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
                    result_3d = self.apply_cached_3d_box_edits(
                        result_3d,
                        image_name=image_name,
                        write_files=True
                    )
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
                        "mask_nms_stats": result_3d.get("mask_nms_stats", None),
                        "box_quality_filter_stats": result_3d.get("box_quality_filter_stats", None),
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
            f"3D语义图滤波强度：{format_flags['semantic_filter_strength']}\n"
            f"3D框密度阈值：{self.get_3d_box_density_threshold()}\n"
            f"3D框最少点数：{self.get_3d_box_inner_points_threshold()}"
        )

        if result_3d is not None:
            status_text += (
                f"\n3D 实例数量：{len(result_3d.get('segments', []))}"
                f"\n3D 框数量：{len(result_3d.get('boxes', []))}"
            )

        self.status_label.setText(status_text)

        return True

    def finish_current_image_and_next(self):
        """保存当前图片标注，并在成功后跳转到下一张图片。"""
        if len(self.image_paths) == 0:
            QMessageBox.warning(self, "提示", "当前没有加载图片文件夹")
            return

        ok = self.save_current_image_annotations(auto_jump_next=True)

        if not ok:
            return

        self.go_next_image_after_save()
