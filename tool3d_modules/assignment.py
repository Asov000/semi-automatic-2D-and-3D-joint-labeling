# -*- coding: utf-8 -*-
"""
三维点赋值模块

功能：
    将 3D 点云投影到 2D 图像平面上，
    再根据 2D mask 判断每个 3D 点属于哪个语义类别和实例。

典型用途：
    1. 已有 2D 图像上的语义 mask
    2. 已有对应图像的 3D 点云
    3. 通过相机内参 K 和外参 / Rtilt 将点云投影到图像
    4. 判断投影点是否落入 mask 内
    5. 给 3D 点生成 point_class_ids 和 point_instance_ids
"""

from typing import Dict, List, Tuple

import numpy as np

# 将输入 mask 标准化为二维二值 mask
from .common import to_binary_mask

# 点云投影函数，以及基于 z-buffer 的可见点过滤函数
from .projection import filter_visible_points_by_zbuffer, project_points_to_image_with_indices


def assign_3d_points_to_2d_masks(
    points3d: np.ndarray,
    K: np.ndarray,
    Rtilt: np.ndarray,
    image_shape: Tuple[int, int, int],
    masks_2d: List[Dict],
    points_are_after_rtilt: bool = True,
    use_matlab_pixel: bool = True,
    use_zbuffer: bool = True,
    zbuffer_tolerance: float = 0.03,
    background_class_id: int = -1,
    overlap_policy: str = "later"
) -> Dict:
    """
    将三维点投影到二维 mask，并生成点级类别标签和实例标签。

    参数：
        points3d:
            输入点云，shape = (N, 3)。
            每一行表示一个三维点坐标，例如 [x, y, z]。

        K:
            相机内参矩阵，shape = (3, 3)。
            用于将相机坐标系下的 3D 点投影到 2D 图像平面。

        Rtilt:
            SUNRGB-D 等数据集中常见的倾斜矫正矩阵。
            用于坐标系变换。
            如果点云已经经过 Rtilt 处理，则 points_are_after_rtilt=True。

        image_shape:
            图像尺寸，一般为 (H, W, C)。
            用于判断投影点是否在图像范围内。

        masks_2d:
            二维 mask 列表。
            每个元素通常包含：
                {
                    "mask": 二维 mask,
                    "class_name": 类别名称,
                    "class_id": 类别 ID,
                    "instance_id": 实例 ID,
                    "mask_path": mask 文件路径，可选
                }

        points_are_after_rtilt:
            表示输入点云是否已经经过 Rtilt 变换。
            True 表示点云已经是矫正后的坐标；
            False 表示函数内部需要结合 Rtilt 进行处理。

        use_matlab_pixel:
            是否使用 MATLAB 风格像素坐标。
            SUNRGB-D 原始工具箱常用 MATLAB 坐标习惯，
            这里用于兼容不同投影方式。

        use_zbuffer:
            是否使用 z-buffer 过滤不可见点。
            True 时，同一个像素位置只保留更靠近相机的点，
            可以减少被遮挡点错误落入 2D mask 的情况。

        zbuffer_tolerance:
            z-buffer 深度容差。
            单位通常与点云深度单位一致。
            容差越大，允许更多近似同深度的点被保留。

        background_class_id:
            背景点类别 ID。
            默认 -1，表示没有被任何 mask 命中的点。

        overlap_policy:
            当多个 mask 同时覆盖同一个投影点时的处理策略。

            "later":
                后面的 mask 覆盖前面的结果。
                适合希望后处理结果覆盖前处理结果的情况。

            "first":
                第一次被命中的实例保留，后续 mask 不再覆盖。
                适合希望先标注结果优先的情况。

    返回：
        Dict:
            {
                "point_class_ids":
                    每个点的类别 ID，shape = (N,)

                "point_instance_ids":
                    每个点的实例 ID，shape = (N,)
                    0 表示没有被任何实例命中

                "valid_projected_mask":
                    每个原始 3D 点是否成功投影到图像范围内

                "visible_projected_mask":
                    每个原始 3D 点是否通过 z-buffer 可见性过滤

                "segments":
                    每个实例对应的点索引和元信息

                "uv":
                    可见投影点的整数像素坐标

                "uv_float":
                    可见投影点的浮点像素坐标

                "depth":
                    可见投影点的深度

                "point_indices":
                    可见投影点在原始点云中的索引
            }
    """

    # 检查重叠处理策略是否合法
    # 只允许两种方式：
    # later：后来的 mask 覆盖前面的 mask
    # first：先命中的 mask 保持不变，后面的 mask 不覆盖
    if overlap_policy not in ["later", "first"]:
        raise ValueError("overlap_policy 只能是 'later' 或 'first'")

    # 将输入点云转换为 float64 类型，并确保 shape 为 (N, 3)
    # reshape(-1, 3) 可以兼容部分被展平的点云输入
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    # 点云总点数
    num_points = points3d.shape[0]

    # 将 3D 点投影到 2D 图像平面
    #
    # uv:
    #   投影后的整数像素坐标，shape = (M, 2)
    #   每一行是 [u, v]
    #
    # depth:
    #   每个有效投影点对应的深度
    #
    # point_indices:
    #   有效投影点在原始 points3d 中的索引
    #
    # uv_float:
    #   未取整前的浮点像素坐标
    uv, depth, point_indices, uv_float = project_points_to_image_with_indices(
        points3d=points3d,
        K=K,
        image_shape=image_shape,
        Rtilt=Rtilt,
        points_are_after_rtilt=points_are_after_rtilt,
        use_matlab_pixel=use_matlab_pixel
    )

    # valid_projected_mask 用来记录原始点云中哪些点成功投影到了图像范围内
    # 初始全部为 False
    valid_projected_mask = np.zeros(num_points, dtype=bool)

    # point_indices 中的点表示成功投影到图像内的点
    # 将这些点标记为 True
    valid_projected_mask[point_indices] = True

    # 如果启用 z-buffer，则进一步过滤被遮挡的点
    if use_zbuffer:
        # 根据像素位置和深度，判断哪些投影点是可见的
        #
        # 基本思想：
        #   对同一个像素位置，如果多个 3D 点投影到这里，
        #   通常只保留深度最小，也就是最靠近相机的点。
        #
        # tolerance 允许一定深度误差，
        # 避免因为点云噪声导致过度过滤。
        visible_local_mask = filter_visible_points_by_zbuffer(
            uv=uv,
            depth=depth,
            image_shape=image_shape,
            tolerance=zbuffer_tolerance
        )

        # 只保留 z-buffer 判断为可见的点
        uv = uv[visible_local_mask]
        depth = depth[visible_local_mask]
        point_indices = point_indices[visible_local_mask]
        uv_float = uv_float[visible_local_mask]

    else:
        # 如果不启用 z-buffer，则所有已投影到图像内的点都认为是可见点
        visible_local_mask = np.ones(len(point_indices), dtype=bool)

    # visible_projected_mask 用来记录原始点云中哪些点最终被认为是可见点
    visible_projected_mask = np.zeros(num_points, dtype=bool)
    visible_projected_mask[point_indices] = True

    # 初始化每个点的类别标签
    # 默认全部为背景类别，例如 -1
    point_class_ids = np.full(num_points, background_class_id, dtype=np.int32)

    # 初始化每个点的实例标签
    # 0 表示未分配到任何实例
    point_instance_ids = np.zeros(num_points, dtype=np.int32)

    # 保存每个实例的元信息
    # key 为 instance_id
    # value 为类别名、类别 ID、mask 路径等信息
    instance_meta = {}

    # 遍历所有 2D mask，将可见 3D 投影点分配到对应 mask 中
    for item in masks_2d:
        # 将 mask 统一转换为二维二值 mask
        # 前景为 1，背景为 0
        mask = to_binary_mask(item["mask"])

        # 读取当前 mask 对应的语义类别名称
        class_name = str(item["class_name"])

        # 读取当前 mask 对应的类别 ID
        class_id = int(item["class_id"])

        # 读取当前 mask 对应的实例 ID
        instance_id = int(item["instance_id"])

        # 取出所有可见投影点的像素横坐标 u
        u = uv[:, 0]

        # 取出所有可见投影点的像素纵坐标 v
        v = uv[:, 1]

        # 判断每个投影点是否落在当前 mask 前景区域内
        #
        # mask[v, u]：
        #   因为图像数组索引是 [行, 列]，
        #   所以 v 对应行坐标，u 对应列坐标。
        #
        # hit 是一个布尔数组：
        #   True 表示该 3D 点投影到了当前 mask 内
        #   False 表示没有命中当前 mask
        hit = mask[v, u] > 0

        # 如果使用 first 策略：
        # 只允许还没有被任何实例分配过的点被当前 mask 选中
        #
        # point_instance_ids[point_indices] == 0
        # 表示这些投影点当前仍然属于未标注状态
        if overlap_policy == "first":
            hit = hit & (point_instance_ids[point_indices] == 0)

        # 根据 hit 结果，找到命中当前 mask 的原始点云索引
        selected_indices = point_indices[hit]

        # 如果没有任何点命中当前 mask，则跳过该 mask
        if selected_indices.size == 0:
            continue

        # 给命中的 3D 点赋予类别 ID
        point_class_ids[selected_indices] = class_id

        # 给命中的 3D 点赋予实例 ID
        point_instance_ids[selected_indices] = instance_id

        # 保存该实例的元信息
        # 后续生成 segments 时会用到
        instance_meta[instance_id] = {
            "instance_id": instance_id,
            "class_name": class_name,
            "class_id": class_id,
            "mask_path": item.get("mask_path", None),
        }

    # 用于保存每个实例对应的点云分割结果
    segments = []

    # 按照 instance_id 从小到大遍历所有实例
    for instance_id in sorted(instance_meta.keys()):
        # 找到所有属于当前 instance_id 的点索引
        indices = np.where(point_instance_ids == instance_id)[0]

        # 如果当前实例没有对应点，则跳过
        if indices.size == 0:
            continue

        # 获取当前实例的类别元信息
        meta = instance_meta[instance_id]

        # 将当前实例的点云分割结果加入 segments
        segments.append({
            "instance_id": int(instance_id),
            "class_name": meta["class_name"],
            "class_id": int(meta["class_id"]),

            # 当前实例包含的原始点云索引
            "point_indices": indices,

            # 当前实例包含的点数量
            "num_points": int(indices.size),

            # 当前实例对应的 2D mask 路径
            "mask_path": meta.get("mask_path", None),
        })

    # 返回完整的点级标注结果
    return {
        # 每个 3D 点的语义类别 ID
        "point_class_ids": point_class_ids,

        # 每个 3D 点的实例 ID
        "point_instance_ids": point_instance_ids,

        # 标记哪些原始 3D 点成功投影到了图像范围内
        "valid_projected_mask": valid_projected_mask,

        # 标记哪些原始 3D 点通过了 z-buffer 可见性过滤
        "visible_projected_mask": visible_projected_mask,

        # 每个实例对应的点云分割结果
        "segments": segments,

        # 可见投影点的整数像素坐标
        "uv": uv,

        # 可见投影点的浮点像素坐标
        "uv_float": uv_float,

        # 可见投影点的深度
        "depth": depth,

        # 可见投影点在原始 points3d 中的索引
        "point_indices": point_indices,
    }