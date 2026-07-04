# -*- coding: utf-8 -*-
"""三维工具通用模块，提供目录创建、JSON 转换和 mask 标准化。"""

import json
import os
from typing import List, Union

import numpy as np


def to_binary_mask(mask: Union[np.ndarray, List[List[int]]], threshold: float = 0) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if mask is None:
        raise ValueError("mask 不能为 None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = np.max(arr, axis=2)

    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D 或单通道图像，当前 shape={arr.shape}")

    return (arr > threshold).astype(np.uint8)


def ensure_dir(path: str) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    os.makedirs(path, exist_ok=True)


def jsonable(obj):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    return obj
