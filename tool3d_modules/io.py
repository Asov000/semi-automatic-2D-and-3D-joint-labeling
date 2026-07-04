# -*- coding: utf-8 -*-
"""三维数据读取模块，负责读取点云、标定、类别映射和二维 mask。"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio

from .common import to_binary_mask


def load_mat_points(mat_path: str) -> np.ndarray:

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"点云文件不存在: {mat_path}")

    try:
        data = sio.loadmat(mat_path)

        if "points3d_rgb" in data:
            return np.asarray(data["points3d_rgb"], dtype=np.float64)

        for key, value in data.items():
            if key.startswith("__"):
                continue

            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")
                return arr[:, :6].astype(np.float64)

        raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")

    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            if "points3d_rgb" in f:
                arr = np.array(f["points3d_rgb"])

                if arr.shape[0] == 6:
                    arr = arr.T

                return arr.astype(np.float64)

            for key in f.keys():
                arr = np.array(f[key])

                if arr.ndim == 2:
                    if arr.shape[0] == 6:
                        arr = arr.T

                    if arr.shape[1] >= 6:
                        print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")
                        return arr[:, :6].astype(np.float64)

            raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")


def load_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    calib = np.loadtxt(calib_path)

    if calib.shape != (2, 9):
        raise ValueError(f"calib 文件格式不对，期望 shape=(2,9)，实际是 {calib.shape}")

    Rtilt = calib[0].reshape(3, 3, order="F")
    K = calib[1].reshape(3, 3, order="F")

    return Rtilt, K


def find_classes_json(segmentation_dir: str) -> Optional[str]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    for name in os.listdir(segmentation_dir):
        if name.endswith("_classes.json"):
            return os.path.join(segmentation_dir, name)

    return None


def load_detection_class_mapping(segmentation_dir: str) -> Dict[str, int]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    json_path = find_classes_json(segmentation_dir)

    if json_path is None:
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    mapping = info.get("detection_class_to_id", {})

    if mapping is None:
        mapping = {}

    return {str(k): int(v) for k, v in mapping.items()}


def recover_class_name_from_safe_name(
    safe_name: str,
    class_to_id: Dict[str, int]
) -> str:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if safe_name in class_to_id:
        return safe_name

    for class_name in class_to_id.keys():
        safe = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        if safe == safe_name:
            return class_name

    return safe_name


def load_binary_masks_from_segmentation_dir(
    segmentation_dir: str,
    image_name: Optional[str] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"segmentation_dir 不存在: {segmentation_dir}")

    binary_dir = os.path.join(segmentation_dir, "binary_masks")

    if not os.path.isdir(binary_dir):
        raise FileNotFoundError(f"binary_masks 目录不存在: {binary_dir}")

    class_to_id = load_detection_class_mapping(segmentation_dir)

    mask_files = [
        name for name in os.listdir(binary_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    mask_files = sorted(mask_files)

    masks = []
    next_class_id = 0 if len(class_to_id) == 0 else max(class_to_id.values()) + 1

    for default_instance_id, name in enumerate(mask_files, start=1):
        path = os.path.join(binary_dir, name)

        mask_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask_gray is None:
            print(f"[load_binary_masks] 跳过无法读取的 mask: {path}")
            continue

        mask = to_binary_mask(mask_gray)

        if int(mask.sum()) == 0:
            print(f"[load_binary_masks] 跳过空 mask: {path}")
            continue

        stem = os.path.splitext(name)[0]

        instance_id = default_instance_id
        safe_class_name = None

        if image_name is not None:
            prefix_pattern = f"{re.escape(image_name)}_"
            if re.match(prefix_pattern, stem):
                remain = stem[len(image_name) + 1:]

                # 形如 001_chair
                parts = remain.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    instance_id = int(parts[0])
                    safe_class_name = parts[1]

        if safe_class_name is None:
            # 通用匹配：xxx_001_chair
            m = re.match(r"^(.+)_([0-9]+)_(.+)$", stem)
            if m is not None:
                instance_id = int(m.group(2))
                safe_class_name = m.group(3)
            else:
                safe_class_name = f"object_{default_instance_id}"

        class_name = recover_class_name_from_safe_name(safe_class_name, class_to_id)

        if class_name not in class_to_id:
            class_to_id[class_name] = next_class_id
            next_class_id += 1

        class_id = class_to_id[class_name]

        masks.append({
            "instance_id": int(instance_id),
            "class_name": str(class_name),
            "class_id": int(class_id),
            "mask": mask.astype(np.uint8),
            "mask_path": path,
        })

    return masks, class_to_id
