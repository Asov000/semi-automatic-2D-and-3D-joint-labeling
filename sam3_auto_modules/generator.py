# -*- coding: utf-8 -*-
"""SAM3 自动分割模块，负责加载模型、执行推理并解析 mask 结果。"""

import os
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .config import SAM3AutoConfig, build_sam3_overrides
from .utils import _get_class_name, _resize_mask_to_image, _to_numpy, normalize_text_prompts


def _safe_to_numpy(obj, attr_name: str):
    """执行模块内部辅助逻辑，供上层流程复用。"""
    if obj is None or not hasattr(obj, attr_name):
        return None

    value = getattr(obj, attr_name, None)
    if value is None:
        return None

    return _to_numpy(value)


def _safe_len(value) -> int:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    if value is None:
        return 0

    try:
        return len(value)
    except TypeError:
        return 0


class SAM3AutoMaskGenerator:
    """SAM3 自动分割生成器，封装模型加载、推理和结果解析。"""

    def __init__(self, config: Optional[Union[SAM3AutoConfig, Dict]] = None):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        self.config = None
        self.overrides = None
        self.predictor = None
        self.update_config(config or SAM3AutoConfig())

    def update_config(self, config: Union[SAM3AutoConfig, Dict]):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if isinstance(config, dict):
            base = SAM3AutoConfig()
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            config = base

        if not isinstance(config, SAM3AutoConfig):
            raise TypeError(f"config 必须是 SAM3AutoConfig 或 dict，当前是 {type(config)}")

        old_model_path = self.config.model_path if self.config is not None else None

        self.config = config
        self.overrides = build_sam3_overrides(config)

        # 权重路径变化时，重新加载 predictor
        if old_model_path != self.config.model_path:
            self.predictor = None

    def _ensure_predictor(self):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        if self.predictor is not None:
            return

        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f"SAM3 权重不存在：{self.config.model_path}")

        from ultralytics.models.sam import SAM3SemanticPredictor

        self.predictor = SAM3SemanticPredictor(overrides=self.overrides)

    def predict(self, image_path: str, text_prompts: Union[str, List[str]]) -> List[Dict]:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图片不存在：{image_path}")

        prompts = normalize_text_prompts(text_prompts)

        if len(prompts) == 0:
            raise ValueError("SAM3 自动标注类别不能为空")

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")

        image_h, image_w = image_bgr.shape[:2]

        self._ensure_predictor()

        self.predictor.set_image(image_path)

        results = self.predictor(
            text=prompts,
            save=False
        )

        return self._parse_results(
            results=results,
            prompts=prompts,
            image_h=image_h,
            image_w=image_w
        )

    @staticmethod
    def _parse_results(results, prompts: List[str], image_h: int, image_w: int) -> List[Dict]:
        """执行模块内部辅助逻辑，供上层流程复用。"""
        parsed = []

        if results is None:
            return parsed

        if not isinstance(results, (list, tuple)):
            results = [results]

        for result in results:
            masks_obj = getattr(result, "masks", None)
            boxes_obj = getattr(result, "boxes", None)
            names = getattr(result, "names", None)

            if masks_obj is None:
                continue

            masks_data = _to_numpy(getattr(masks_obj, "data", None))

            if masks_data is None:
                continue

            if masks_data.ndim == 2:
                masks_data = masks_data[None, :, :]

            confs = None
            class_ids = None
            boxes_xyxy = None

            if boxes_obj is not None:
                confs = _safe_to_numpy(boxes_obj, "conf")
                class_ids = _safe_to_numpy(boxes_obj, "cls")
                boxes_xyxy = _safe_to_numpy(boxes_obj, "xyxy")

            for i in range(masks_data.shape[0]):
                mask = _resize_mask_to_image(masks_data[i], image_h, image_w)

                if int(mask.sum()) == 0:
                    continue

                if confs is not None and i < _safe_len(confs):
                    score = float(confs[i])
                else:
                    score = 1.0

                if class_ids is not None and i < _safe_len(class_ids):
                    class_id = int(class_ids[i])
                else:
                    class_id = 0

                class_name = _get_class_name(names, class_id, prompts)

                if boxes_xyxy is not None and i < _safe_len(boxes_xyxy):
                    bbox = [int(x) for x in boxes_xyxy[i].tolist()]
                else:
                    ys, xs = np.where(mask > 0)
                    bbox = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ]

                parsed.append({
                    "mask": mask.astype(np.uint8),
                    "score": score,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox_xyxy": bbox,
                })

        return parsed
