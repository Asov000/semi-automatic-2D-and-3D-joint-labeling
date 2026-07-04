# -*- coding: utf-8 -*-
"""SAM3 辅助工具模块，负责文本提示解析、mask 尺寸调整和类别名称读取。"""

from typing import List, Union

import cv2
import numpy as np
import torch


def normalize_text_prompts(text: Union[str, List[str]]) -> List[str]:
    """
    规范化 SAM3 的文本提示词。

    参数：
    text:
        可以是字符串，也可以是字符串列表。

        示例：
        "chair, table, bed"
        ["chair", "table", "bed"]

    返回：
    prompts:
        处理后的类别提示词列表。
    """

    # 如果输入是字符串
    if isinstance(text, str):
        # 去除首尾空格
        raw = text.strip()

        # 将中文逗号、中文分号、英文分号统一替换成英文逗号
        # 方便后续统一用逗号切分
        raw = raw.replace("，", ",").replace("；", ",").replace(";", ",")

        # 将换行符也替换成逗号
        # 支持用户一行写一个类别
        raw = raw.replace("\n", ",")

        # 按逗号切分，并去掉空字符串
        prompts = [x.strip() for x in raw.split(",") if x.strip()]

    # 如果输入本身就是列表
    else:
        # 将每个元素转成字符串，并去除空白内容
        prompts = [str(x).strip() for x in text if str(x).strip()]

    # 返回标准化后的提示词列表
    return prompts


def _to_numpy(x):
    """
    将输入数据转换为 numpy 数组。

    参数：
    x:
        可以是 torch.Tensor、numpy 数组、list 或其他可转换对象。

    返回：
    numpy 数组；
    如果输入为 None，则返回 None。
    """

    # 空值直接返回 None
    if x is None:
        return None

    # 如果是 PyTorch Tensor，需要先 detach，再转到 CPU，最后转 numpy
    # detach()：脱离计算图
    # cpu()：确保数据在 CPU 上
    # numpy()：转为 numpy 数组
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    # 其他类型直接尝试转为 numpy 数组
    return np.asarray(x)


def _resize_mask_to_image(mask: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """
    将模型输出的 mask 调整到原图尺寸，并转成二值 mask。

    参数：
    mask:
        模型输出的 mask，可能是 H×W、1×H×W 或 H×W×1。

    image_h:
        原图高度。

    image_w:
        原图宽度。

    返回：
    二值 mask，形状为 image_h × image_w，数值为 0 或 1。
    """

    # 转成 numpy 数组
    mask = np.asarray(mask)

    # 如果 mask 是 3 维，需要压缩成 2 维
    if mask.ndim == 3:
        # 情况 1：形状为 1 × H × W
        if mask.shape[0] == 1:
            mask = mask[0]

        # 情况 2：形状为 H × W × 1
        elif mask.shape[-1] == 1:
            mask = mask[:, :, 0]

    # 如果经过处理后仍然不是二维 mask，说明格式异常
    if mask.ndim != 2:
        raise ValueError(f"mask 维度异常，当前 shape={mask.shape}")

    # 如果 mask 尺寸和原图尺寸不一致，则 resize 回原图大小
    if mask.shape[0] != image_h or mask.shape[1] != image_w:
        mask = cv2.resize(
            mask.astype(np.float32),       # resize 前转为 float
            (image_w, image_h),            # OpenCV resize 参数顺序是 宽、高
            interpolation=cv2.INTER_NEAREST # 最近邻插值，避免 mask 边界被平滑
        )

    # 将 mask 转成二值图
    # 大于 0.5 的位置视为前景，记为 1
    # 其他位置视为背景，记为 0
    return (mask > 0.5).astype(np.uint8)


def _get_class_name(names, class_id: int, prompts: List[str]) -> str:
    """
    根据 class_id 获取类别名称。

    参数：
    names:
        模型返回的类别名称映射，可能是 dict 或 list。

    class_id:
        类别 ID。

    prompts:
        用户输入的文本提示词列表。

    返回：
    class_name:
        类别名称字符串。
    """

    # 如果 names 是字典格式
    # 常见格式：{0: "chair", 1: "table"}
    if isinstance(names, dict):
        # 优先使用整数 key 查找
        if class_id in names:
            return str(names[class_id])

        # 有些结果中 key 可能是字符串形式，例如 {"0": "chair"}
        if str(class_id) in names:
            return str(names[str(class_id)])

    # 如果 names 是列表格式
    # 常见格式：["chair", "table", "bed"]
    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    # 如果模型没有提供 names，则尝试从用户输入的 prompts 中取类别名
    if 0 <= class_id < len(prompts):
        return str(prompts[class_id])

    # 如果都无法匹配，则返回默认类别名
    return f"class_{class_id}"