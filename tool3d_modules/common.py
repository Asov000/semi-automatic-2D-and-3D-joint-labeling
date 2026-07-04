# -*- coding: utf-8 -*-
"""
三维工具通用模块

该模块主要提供一些在 2D/3D 标注、点云处理、mask 保存过程中常用的基础工具函数：
1. mask 标准化为二值 mask
2. 自动创建目录
3. 将 numpy 类型对象转换为 JSON 可序列化对象
"""

import json
import os
from typing import List, Union

import numpy as np


def to_binary_mask(mask: Union[np.ndarray, List[List[int]]], threshold: float = 0) -> np.ndarray:
    """
    将输入的 mask 转换为标准的二维二值 mask。

    参数：
        mask:
            输入的掩膜，可以是 numpy 数组，也可以是二维 list。
            支持以下形式：
            1. 二维数组，shape = (H, W)
            2. 单通道三维数组，shape = (H, W, 1)
            3. 多通道三维数组，shape = (H, W, C)

        threshold:
            二值化阈值。
            大于该阈值的位置会被置为 1，否则置为 0。
            默认 threshold=0，适合处理常见的 0/255 mask。

    返回：
        np.ndarray:
            二维 uint8 类型二值 mask，shape = (H, W)。
            前景区域为 1，背景区域为 0。

    异常：
        ValueError:
            当 mask 为 None，或者 mask 不是二维 / 单通道图像时抛出。
    """

    # 如果传入的 mask 为空，无法继续处理，直接报错
    if mask is None:
        raise ValueError("mask 不能为 None")

    # 将输入转换为 numpy 数组，方便后续统一处理
    arr = np.asarray(mask)

    # 如果 mask 是三维数组，需要压缩成二维
    if arr.ndim == 3:
        # 情况一：单通道图像，例如 shape = (H, W, 1)
        # 直接取出第 0 个通道
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

        # 情况二：多通道图像，例如 RGB mask，shape = (H, W, 3)
        # 对多个通道取最大值，只要任意通道有前景，就认为该像素是前景
        else:
            arr = np.max(arr, axis=2)

    # 最终要求 mask 必须是二维数组
    # 如果仍然不是二维，说明输入格式不符合要求
    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D 或单通道图像，当前 shape={arr.shape}")

    # 根据阈值进行二值化：
    # arr > threshold 的位置为 True，转换成 uint8 后为 1
    # arr <= threshold 的位置为 False，转换成 uint8 后为 0
    return (arr > threshold).astype(np.uint8)


def ensure_dir(path: str) -> None:
    """
    确保指定目录存在。

    如果目录不存在，则自动创建；
    如果目录已经存在，则不会报错。

    参数：
        path:
            需要创建或检查的目录路径。

    返回：
        None
    """

    # exist_ok=True 表示：
    # 如果目录已存在，不抛出异常
    os.makedirs(path, exist_ok=True)


def jsonable(obj):
    """
    将对象转换为 JSON 可以保存的格式。

    在实际标注系统中，很多数据可能是 numpy 类型，例如：
    - np.ndarray
    - np.int64
    - np.float32

    这些类型不能直接被 json.dump() 保存，
    所以需要先转换为 Python 原生类型。

    参数：
        obj:
            任意待转换对象。

    返回：
        可被 JSON 序列化的对象。
    """

    # numpy 数组不能直接保存为 JSON
    # 需要转换为 Python list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy 整数类型，例如 np.int32、np.int64
    # 转换为 Python 原生 int
    if isinstance(obj, np.integer):
        return int(obj)

    # numpy 浮点类型，例如 np.float32、np.float64
    # 转换为 Python 原生 float
    if isinstance(obj, np.floating):
        return float(obj)

    # 如果是字典，需要递归处理每一个 value
    # 防止字典内部仍然包含 numpy 类型
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}

    # 如果是列表，需要递归处理列表中的每一个元素
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]

    # 如果是元组，JSON 中没有 tuple 类型
    # 因此转换为 list，并递归处理内部元素
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]

    # 其他 Python 原生类型，例如 str、int、float、bool、None
    # 本身就可以被 JSON 保存，直接返回
    return obj