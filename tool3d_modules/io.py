# -*- coding: utf-8 -*-

"""
三维数据读取模块

主要功能：
1. 读取 .mat 格式点云文件；
2. 读取相机标定文件 calib；
3. 读取类别映射 json；
4. 读取二维 binary mask；
5. 从 mask 文件名中解析类别名和实例 ID；
6. 最终输出可用于 2D mask → 3D 点云赋值的数据结构。
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio

# 从 common.py 中导入 mask 二值化函数
from .common import to_binary_mask


def load_mat_points(mat_path: str) -> np.ndarray:
    """
    读取 .mat 文件中的点云数据。

    支持两类 .mat 文件：
    1. 普通 MATLAB .mat 文件，可用 scipy.io.loadmat 读取；
    2. MATLAB v7.3 .mat 文件，本质是 HDF5，需要用 h5py 读取。

    期望读取到的数据格式：
        points3d_rgb: shape = (N, 6)

    其中：
        前 3 列：x, y, z
        后 3 列：r, g, b

    参数：
        mat_path:
            点云 .mat 文件路径。

    返回：
        np.ndarray:
            shape = (N, 6) 的点云数组。
    """

    # 如果点云文件不存在，直接报错
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"点云文件不存在: {mat_path}")

    try:
        # 尝试用 scipy 读取普通 .mat 文件
        data = sio.loadmat(mat_path)

        # 优先读取标准变量 points3d_rgb
        if "points3d_rgb" in data:
            return np.asarray(data["points3d_rgb"], dtype=np.float64)

        # 如果没有 points3d_rgb，则遍历 .mat 文件中的所有变量
        for key, value in data.items():
            # 跳过 MATLAB 自动生成的元信息字段
            if key.startswith("__"):
                continue

            arr = np.asarray(value)

            # 如果某个变量是二维数组，并且列数 >= 6，
            # 则认为它可能是点云数据
            if arr.ndim == 2 and arr.shape[1] >= 6:
                print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")

                # 只保留前 6 列，统一为 xyzrgb 格式
                return arr[:, :6].astype(np.float64)

        # 如果没有找到合适变量，则报错
        raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")

    except NotImplementedError:
        # 如果 scipy 无法读取，通常说明这是 MATLAB v7.3 格式
        # MATLAB v7.3 的 .mat 文件底层是 HDF5，因此用 h5py 读取
        with h5py.File(mat_path, "r") as f:

            # 优先读取 points3d_rgb
            if "points3d_rgb" in f:
                arr = np.array(f["points3d_rgb"])

                # 有些 HDF5 文件中保存为 6 x N，需要转置成 N x 6
                if arr.shape[0] == 6:
                    arr = arr.T

                return arr.astype(np.float64)

            # 如果没有 points3d_rgb，则遍历 HDF5 中所有变量
            for key in f.keys():
                arr = np.array(f[key])

                if arr.ndim == 2:
                    # 如果是 6 x N，则转置成 N x 6
                    if arr.shape[0] == 6:
                        arr = arr.T

                    # 如果列数 >= 6，则认为是点云数组
                    if arr.shape[1] >= 6:
                        print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")

                        # 统一只返回前 6 列
                        return arr[:, :6].astype(np.float64)

            # HDF5 中也找不到合适点云变量
            raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")


def load_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取相机标定文件。

    该函数假设 calib 文件格式为 shape = (2, 9)。

    第一行：
        Rtilt，3 x 3 倾斜校正矩阵展开后的结果。

    第二行：
        K，3 x 3 相机内参矩阵展开后的结果。

    注意：
        reshape 时使用 order="F"，
        表示按照 MATLAB 的列优先顺序恢复矩阵。
        这是为了兼容 SUNRGB-D 工具箱的保存格式。

    参数：
        calib_path:
            标定文件路径。

    返回：
        Rtilt:
            shape = (3, 3)

        K:
            shape = (3, 3)
    """

    # 检查标定文件是否存在
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    # 从 txt 中读取标定矩阵
    calib = np.loadtxt(calib_path)

    # 要求 calib 必须是 2 行 9 列
    if calib.shape != (2, 9):
        raise ValueError(f"calib 文件格式不对，期望 shape=(2,9)，实际是 {calib.shape}")

    # 第一行恢复为 Rtilt
    Rtilt = calib[0].reshape(3, 3, order="F")

    # 第二行恢复为相机内参 K
    K = calib[1].reshape(3, 3, order="F")

    return Rtilt, K


def find_classes_json(segmentation_dir: str) -> Optional[str]:
    """
    在分割结果目录中查找类别映射 json 文件。

    目标文件名格式：
        xxx_classes.json

    参数：
        segmentation_dir:
            分割结果目录。

    返回：
        如果找到，返回 json 文件路径；
        如果没找到，返回 None。
    """

    # 遍历 segmentation_dir 下所有文件
    for name in os.listdir(segmentation_dir):

        # 找到以 _classes.json 结尾的文件
        if name.endswith("_classes.json"):
            return os.path.join(segmentation_dir, name)

    # 没找到类别映射文件
    return None


def load_detection_class_mapping(segmentation_dir: str) -> Dict[str, int]:
    """
    读取类别名称到类别 ID 的映射关系。

    期望 json 文件中包含字段：
        detection_class_to_id

    例如：
        {
            "detection_class_to_id": {
                "chair": 0,
                "table": 1,
                "bed": 2
            }
        }

    参数：
        segmentation_dir:
            分割结果目录。

    返回：
        Dict[str, int]:
            类别名到类别 ID 的映射。
    """

    # 查找 xxx_classes.json 文件
    json_path = find_classes_json(segmentation_dir)

    # 如果没找到 json，则返回空映射
    if json_path is None:
        return {}

    # 读取 json 文件
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # 提取 detection_class_to_id 字段
    mapping = info.get("detection_class_to_id", {})

    # 防止字段存在但内容为 None
    if mapping is None:
        mapping = {}

    # 统一 key 为 str，value 为 int
    return {str(k): int(v) for k, v in mapping.items()}


def recover_class_name_from_safe_name(
    safe_name: str,
    class_to_id: Dict[str, int]
) -> str:
    """
    从安全文件名恢复类别名称。

    为什么需要恢复：
        有些类别名不能直接作为文件名，例如：
            night stand
            table/chair

        保存文件时可能被替换成：
            night_stand
            table_chair

    该函数会尝试根据已有 class_to_id 映射，
    把 safe_name 恢复成原始类别名。

    参数：
        safe_name:
            从文件名中解析出来的安全类别名。

        class_to_id:
            已有类别名到 ID 的映射。

    返回：
        str:
            恢复后的类别名。
            如果无法恢复，则返回 safe_name 本身。
    """

    # 如果 safe_name 本身就在类别映射中，直接返回
    if safe_name in class_to_id:
        return safe_name

    # 遍历已有类别名，尝试转换成安全文件名后进行匹配
    for class_name in class_to_id.keys():

        # 将类别名中的空格、斜杠替换为下划线
        safe = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        # 如果安全名匹配，则返回原始类别名
        if safe == safe_name:
            return class_name

    # 没有匹配成功，则直接返回 safe_name
    return safe_name


def load_binary_masks_from_segmentation_dir(
    segmentation_dir: str,
    image_name: Optional[str] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    从 segmentation_dir/binary_masks 中读取所有二值 mask。

    该函数会完成：
        1. 检查 segmentation_dir 是否存在；
        2. 检查 binary_masks 子目录是否存在；
        3. 读取类别映射 json；
        4. 遍历 binary_masks 下所有 mask 图像；
        5. 读取灰度 mask；
        6. 转换为二值 mask；
        7. 从文件名中解析 instance_id 和 class_name；
        8. 为每个 mask 分配 class_id；
        9. 返回 masks 列表和 class_to_id 映射。

    支持的文件名格式：

        格式 1：传入 image_name 时
            image_name_001_chair.png

            解析为：
                instance_id = 1
                class_name = chair

        格式 2：通用格式
            xxx_001_chair.png

            解析为：
                instance_id = 1
                class_name = chair

        格式 3：无法解析
            自动命名为 object_编号

    参数：
        segmentation_dir:
            分割结果目录。
            目录下必须存在 binary_masks 文件夹。

        image_name:
            当前图像名，可选。
            如果传入，会优先按照 image_name_实例ID_类别名 解析。

    返回：
        masks:
            mask 信息列表。
            每个元素结构为：
                {
                    "instance_id": 实例 ID,
                    "class_name": 类别名,
                    "class_id": 类别 ID,
                    "mask": 二值 mask,
                    "mask_path": mask 文件路径
                }

        class_to_id:
            类别名称到类别 ID 的映射。
    """

    # 检查 segmentation_dir 是否存在
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"segmentation_dir 不存在: {segmentation_dir}")

    # binary_masks 是存放每个实例二值 mask 的目录
    binary_dir = os.path.join(segmentation_dir, "binary_masks")

    # 检查 binary_masks 是否存在
    if not os.path.isdir(binary_dir):
        raise FileNotFoundError(f"binary_masks 目录不存在: {binary_dir}")

    # 读取类别映射
    # 如果没有 classes.json，则 class_to_id 为空
    class_to_id = load_detection_class_mapping(segmentation_dir)

    # 找到 binary_masks 下所有图像文件
    mask_files = [
        name for name in os.listdir(binary_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    # 排序，保证每次读取顺序一致
    mask_files = sorted(mask_files)

    # 保存所有 mask 信息
    masks = []

    # 如果已有类别映射，新类别从 max_id + 1 开始
    # 如果没有类别映射，则从 0 开始
    next_class_id = 0 if len(class_to_id) == 0 else max(class_to_id.values()) + 1

    # 遍历每个 mask 文件
    # default_instance_id 从 1 开始，作为无法解析 instance_id 时的默认值
    for default_instance_id, name in enumerate(mask_files, start=1):
        path = os.path.join(binary_dir, name)

        # 用灰度模式读取 mask
        mask_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 如果读取失败，则跳过该文件
        if mask_gray is None:
            print(f"[load_binary_masks] 跳过无法读取的 mask: {path}")
            continue

        # 将灰度 mask 转为标准二值 mask
        # 背景为 0，前景为 1
        mask = to_binary_mask(mask_gray)

        # 如果 mask 没有前景像素，则跳过
        if int(mask.sum()) == 0:
            print(f"[load_binary_masks] 跳过空 mask: {path}")
            continue

        # 去掉扩展名，得到文件主名
        stem = os.path.splitext(name)[0]

        # 默认实例 ID
        instance_id = default_instance_id

        # 从文件名中解析出的类别名
        safe_class_name = None

        # 如果传入 image_name，则优先按 image_name_001_chair 格式解析
        if image_name is not None:
            prefix_pattern = f"{re.escape(image_name)}_"

            # 判断文件名是否以 image_name_ 开头
            if re.match(prefix_pattern, stem):
                # 去掉 image_name_ 前缀
                remain = stem[len(image_name) + 1:]

                # 形如 001_chair
                parts = remain.split("_", 1)

                # 如果第一部分是数字，则认为是 instance_id
                if len(parts) == 2 and parts[0].isdigit():
                    instance_id = int(parts[0])
                    safe_class_name = parts[1]

        # 如果没有通过 image_name 解析成功，则使用通用正则匹配
        if safe_class_name is None:
            # 通用匹配：xxx_001_chair
            m = re.match(r"^(.+)_([0-9]+)_(.+)$", stem)

            if m is not None:
                # 第二个分组是实例 ID
                instance_id = int(m.group(2))

                # 第三个分组是类别名
                safe_class_name = m.group(3)
            else:
                # 如果文件名无法解析类别，则自动生成类别名
                safe_class_name = f"object_{default_instance_id}"

        # 尝试恢复原始类别名
        # 例如 night_stand 可能恢复成 night stand
        class_name = recover_class_name_from_safe_name(safe_class_name, class_to_id)

        # 如果该类别还没有 ID，则分配新的类别 ID
        if class_name not in class_to_id:
            class_to_id[class_name] = next_class_id
            next_class_id += 1

        # 获取当前类别 ID
        class_id = class_to_id[class_name]

        # 保存当前 mask 的完整信息
        masks.append({
            "instance_id": int(instance_id),
            "class_name": str(class_name),
            "class_id": int(class_id),
            "mask": mask.astype(np.uint8),
            "mask_path": path,
        })

    # 返回所有 mask，以及类别映射
    return masks, class_to_id
