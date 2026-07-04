# -*- coding: utf-8 -*-
"""SAM3 辅助工具模块，负责文本提示解析、mask 尺寸调整和类别名称读取。"""

from typing import List, Union

import cv2
import numpy as np
import torch


def normalize_text_prompts(text: Union[str, List[str]]) -> List[str]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if isinstance(text, str):
        raw = text.strip()
        raw = raw.replace("，", ",").replace("；", ",").replace(";", ",")
        raw = raw.replace("\n", ",")
        prompts = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        prompts = [str(x).strip() for x in text if str(x).strip()]

    return prompts


def _to_numpy(x):
    """执行模块内部辅助逻辑，供上层流程复用。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_mask_to_image(mask: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    mask = np.asarray(mask)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(f"mask 维度异常，当前 shape={mask.shape}")

    if mask.shape[0] != image_h or mask.shape[1] != image_w:
        mask = cv2.resize(
            mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST
        )

    return (mask > 0.5).astype(np.uint8)


def _get_class_name(names, class_id: int, prompts: List[str]) -> str:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        if str(class_id) in names:
            return str(names[str(class_id)])

    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    if 0 <= class_id < len(prompts):
        return str(prompts[class_id])

    return f"class_{class_id}"
