# -*- coding: utf-8 -*-
"""SAM3 自动分割模块，负责加载模型、执行推理并解析 mask 结果。"""

import os
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

# 读取 SAM3 配置类和构建推理参数的函数
from .config import SAM3AutoConfig, build_sam3_overrides

# 工具函数：
# _get_class_name：根据类别 ID 获取类别名称
# _resize_mask_to_image：将模型输出 mask resize 回原图尺寸
# _to_numpy：将 tensor / list 等数据转为 numpy
# normalize_text_prompts：统一处理文本提示词格式
from .utils import (
    _get_class_name,
    _resize_mask_to_image,
    _to_numpy,
    normalize_text_prompts
)


def _safe_to_numpy(obj, attr_name: str):
    """
    从对象中读取某个属性，并转换为 numpy。

    参数：
    obj: 需要读取属性的对象，例如 boxes_obj
    attr_name: 属性名，例如 "conf"、"cls"、"xyxy"

    返回：
    如果属性存在，则返回 numpy 数组；
    如果对象为空或属性不存在，则返回 None。
    """

    # 如果对象为空，或者对象没有指定属性，直接返回 None
    if obj is None or not hasattr(obj, attr_name):
        return None

    # 获取指定属性
    value = getattr(obj, attr_name, None)

    # 如果属性值为空，返回 None
    if value is None:
        return None

    # 将属性值转换为 numpy 数组
    return _to_numpy(value)


def _safe_len(value) -> int:
    """
    获取对象长度。

    参数：
    value: 任意对象

    返回：
    如果对象支持 len()，返回其长度；
    如果对象为空或不支持 len()，返回 0。
    """

    # None 没有长度
    if value is None:
        return 0

    try:
        return len(value)
    except TypeError:
        return 0


class SAM3AutoMaskGenerator:
    """
    SAM3 自动分割生成器。

    主要功能：
    1. 管理 SAM3 推理配置
    2. 加载 SAM3 模型
    3. 根据文本提示词执行自动分割
    4. 解析模型输出的 mask、bbox、score 和类别信息
    """

    def __init__(self, config: Optional[Union[SAM3AutoConfig, Dict]] = None):
        """
        初始化 SAM3 自动分割器。

        参数：
        config:
            可以是 SAM3AutoConfig 对象，也可以是 dict。
            如果不传，则使用默认配置。
        """

        # 保存配置对象
        self.config = None

        # 保存最终传入模型 predictor 的推理参数
        self.overrides = None

        # SAM3 推理器，采用懒加载方式，真正预测时才加载
        self.predictor = None

        # 初始化配置
        self.update_config(config or SAM3AutoConfig())

    def update_config(self, config: Union[SAM3AutoConfig, Dict]):
        """
        更新 SAM3 配置。

        参数：
        config:
            可以是 SAM3AutoConfig 或 dict。
            如果是 dict，只会更新 SAM3AutoConfig 中存在的字段。

        作用：
        1. 更新 self.config
        2. 重新生成 self.overrides
        3. 如果模型路径发生变化，则清空 predictor，后续重新加载模型
        """

        # 如果传入的是 dict，则转换成 SAM3AutoConfig 对象
        if isinstance(config, dict):
            base = SAM3AutoConfig()

            # 只更新配置类中已有的字段，避免无效字段报错
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)

            config = base

        # 检查 config 类型是否合法
        if not isinstance(config, SAM3AutoConfig):
            raise TypeError(f"config 必须是 SAM3AutoConfig 或 dict，当前是 {type(config)}")

        # 记录旧的模型路径，用于判断是否需要重新加载模型
        old_model_path = self.config.model_path if self.config is not None else None

        # 更新配置对象
        self.config = config

        # 根据配置生成最终传给 predictor 的参数字典
        self.overrides = build_sam3_overrides(config)

        # 如果模型权重路径变化，说明原来的 predictor 已经不适用，需要重新加载
        if old_model_path != self.config.model_path:
            self.predictor = None

    def _ensure_predictor(self):
        """
        确保 SAM3 predictor 已经加载。

        说明：
        采用懒加载策略。
        只有第一次真正执行 predict 时才加载模型，
        避免初始化对象时就占用显存。
        """

        # 如果 predictor 已经存在，直接返回
        if self.predictor is not None:
            return

        # 检查模型权重文件是否存在
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f"SAM3 权重不存在：{self.config.model_path}")

        # 导入 SAM3 语义分割预测器
        # 这里依赖 ultralytics 中的 SAM3SemanticPredictor
        from ultralytics.models.sam import SAM3SemanticPredictor

        # 根据 overrides 创建 predictor
        self.predictor = SAM3SemanticPredictor(overrides=self.overrides)

    def predict(self, image_path: str, text_prompts: Union[str, List[str]]) -> List[Dict]:
        """
        对单张图片执行 SAM3 自动分割。

        参数：
        image_path:
            输入图片路径。

        text_prompts:
            文本提示词，可以是字符串，也可以是字符串列表。
            例如：
            "chair"
            或 ["chair", "table", "bed"]

        返回：
        List[Dict]，每个 dict 表示一个分割结果，包含：
        - mask: 二值掩膜
        - score: 置信度
        - class_id: 类别 ID
        - class_name: 类别名称
        - bbox_xyxy: 外接矩形框 [xmin, ymin, xmax, ymax]
        """

        # 检查图片路径是否存在
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图片不存在：{image_path}")

        # 统一处理文本提示词格式
        # 保证最终 prompts 是 List[str]
        prompts = normalize_text_prompts(text_prompts)

        # SAM3 自动标注必须有类别提示词
        if len(prompts) == 0:
            raise ValueError("SAM3 自动标注类别不能为空")

        # 用 OpenCV 读取图片，主要用于获取图像尺寸
        image_bgr = cv2.imread(image_path)

        # 图片读取失败时抛出异常
        if image_bgr is None:
            raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")

        # 获取原图高度和宽度
        image_h, image_w = image_bgr.shape[:2]

        # 确保 predictor 已经加载
        self._ensure_predictor()

        # 将当前图片设置给 SAM3 predictor
        self.predictor.set_image(image_path)

        # 执行文本提示分割
        results = self.predictor(
            text=prompts,
            save=False
        )

        # 解析 SAM3 输出结果
        return self._parse_results(
            results=results,
            prompts=prompts,
            image_h=image_h,
            image_w=image_w
        )

    @staticmethod
    def _parse_results(results, prompts: List[str], image_h: int, image_w: int) -> List[Dict]:
        """
        解析 SAM3 模型输出结果。

        参数：
        results:
            SAM3 predictor 的原始输出。

        prompts:
            文本提示词列表，用于辅助获取类别名称。

        image_h:
            原图高度。

        image_w:
            原图宽度。

        返回：
        parsed:
            解析后的分割结果列表。
        """

        # 保存最终解析结果
        parsed = []

        # 如果模型没有返回结果，直接返回空列表
        if results is None:
            return parsed

        # 统一 results 格式，保证后续可以遍历
        if not isinstance(results, (list, tuple)):
            results = [results]

        # 遍历每个 result
        for result in results:
            # 获取 mask 对象
            masks_obj = getattr(result, "masks", None)

            # 获取 bbox 对象，通常包含 conf、cls、xyxy 等信息
            boxes_obj = getattr(result, "boxes", None)

            # 获取类别名称映射
            names = getattr(result, "names", None)

            # 没有 mask，说明没有有效分割结果
            if masks_obj is None:
                continue

            # 读取 masks_obj.data，并转换为 numpy
            masks_data = _to_numpy(getattr(masks_obj, "data", None))

            # 如果 mask 数据为空，跳过
            if masks_data is None:
                continue

            # 如果只有单个 mask，形状可能是 H x W
            # 这里扩展成 1 x H x W，方便统一遍历
            if masks_data.ndim == 2:
                masks_data = masks_data[None, :, :]

            # 初始化置信度、类别 ID、检测框
            confs = None
            class_ids = None
            boxes_xyxy = None

            # 如果存在 boxes，则尝试提取对应属性
            if boxes_obj is not None:
                # 每个 mask / box 的置信度
                confs = _safe_to_numpy(boxes_obj, "conf")

                # 每个 mask / box 的类别 ID
                class_ids = _safe_to_numpy(boxes_obj, "cls")

                # 每个 mask / box 的检测框坐标，格式一般为 xyxy
                boxes_xyxy = _safe_to_numpy(boxes_obj, "xyxy")

            # 遍历每一个 mask
            for i in range(masks_data.shape[0]):
                # 将模型输出的 mask resize 回原图大小
                mask = _resize_mask_to_image(masks_data[i], image_h, image_w)

                # 如果 mask 为空，跳过
                if int(mask.sum()) == 0:
                    continue

                # 获取置信度
                # 如果模型没有返回 conf，则默认设置为 1.0
                if confs is not None and i < _safe_len(confs):
                    score = float(confs[i])
                else:
                    score = 1.0

                # 获取类别 ID
                # 如果模型没有返回 cls，则默认设置为 0
                if class_ids is not None and i < _safe_len(class_ids):
                    class_id = int(class_ids[i])
                else:
                    class_id = 0

                # 根据类别 ID 和 prompts 获取类别名称
                class_name = _get_class_name(names, class_id, prompts)

                # 如果模型直接返回了 bbox，则优先使用模型的 xyxy 框
                if boxes_xyxy is not None and i < _safe_len(boxes_xyxy):
                    bbox = [int(x) for x in boxes_xyxy[i].tolist()]

                # 如果没有 bbox，则根据 mask 自行计算外接矩形框
                else:
                    ys, xs = np.where(mask > 0)

                    bbox = [
                        int(xs.min()),  # xmin
                        int(ys.min()),  # ymin
                        int(xs.max()),  # xmax
                        int(ys.max()),  # ymax
                    ]

                # 保存当前目标的解析结果
                parsed.append({
                    "mask": mask.astype(np.uint8),  # 二值 mask
                    "score": score,                 # 置信度
                    "class_id": class_id,           # 类别 ID
                    "class_name": class_name,       # 类别名称
                    "bbox_xyxy": bbox,              # 外接框 xyxy
                })

        # 返回所有解析后的分割结果
        return parsed