# -*- coding: utf-8 -*-
"""
三维框模块

功能：
    1. 根据实例点云生成三维框；
    2. 支持普通 AABB 轴对齐框；
    3. 支持基于 PCA 主方向估计的水平有向框 PCA_OBB；
    4. 对生成的三维框计算体积、框内语义点数、点密度和质量分数；
    5. 根据密度、体积和重叠关系过滤低质量框；
    6. 可选择将被删除框对应的语义点重置为背景。

"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def filter_points_by_percentile(
    points: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    按坐标分位数过滤极端点，降低飞点对三维框的影响。

    作用：
        如果直接用所有点生成 3D 框，少量飞点会把框拉得很大。
        该函数先用坐标百分位数去掉极端点，再用中心距离进一步去掉远离主体的点。

    参数：
        points:
            输入点云，shape = (N, 3)。

        lower_percentile:
            下分位数，例如 1 表示每个坐标方向去掉最小的 1% 极端点。

        upper_percentile:
            上分位数，例如 99 表示每个坐标方向去掉最大的 1% 极端点。

    返回：
        np.ndarray:
            过滤后的点云。
    """

    # 统一转成 float64，保证后续几何计算稳定
    points = np.asarray(points, dtype=np.float64)

    # 点数太少时不做过滤，避免小目标被误删
    if points.shape[0] < 10:
        return points

    # 分别计算 x/y/z 三个方向的下界和上界百分位
    low = np.percentile(points, lower_percentile, axis=0)
    high = np.percentile(points, upper_percentile, axis=0)

    # 保留同时落在三个坐标分位范围内的点
    keep = np.all((points >= low) & (points <= high), axis=1)

    # 如果过滤后点太少，说明阈值可能过严，回退原始点
    if keep.sum() < 5:
        return points

    filtered = points[keep]

    # 如果过滤后点数仍然较少，直接返回，不再做二次距离过滤
    if filtered.shape[0] < 20:
        return filtered

    # 用中位数作为鲁棒中心，比 mean 更不容易被飞点影响
    center = np.median(filtered, axis=0)

    # 计算每个点到中心的欧氏距离
    distances = np.linalg.norm(filtered - center, axis=1)

    # 用距离百分位数作为二次过滤阈值
    distance_threshold = np.percentile(distances, upper_percentile)

    # 保留距离不超过阈值的点
    distance_keep = distances <= distance_threshold

    # 如果二次过滤后点数过少，回退到第一次过滤结果
    if int(distance_keep.sum()) < max(5, int(filtered.shape[0] * 0.70)):
        return filtered

    return filtered[distance_keep]


def compute_aabb_3d(points: np.ndarray) -> Dict:
    """
    根据点集计算轴对齐三维框 AABB。

    AABB:
        Axis-Aligned Bounding Box，轴对齐包围盒。
        框的边始终与全局 x/y/z 坐标轴平行。

    优点：
        计算简单，稳定。

    缺点：
        不能表达物体旋转方向。
        如果物体斜放，AABB 会比真实物体范围更大。

    参数：
        points:
            输入点云，shape = (N, 3)。

    返回：
        Dict:
            三维框信息，包括中心、尺寸、最小最大坐标和 8 个角点。
    """

    points = np.asarray(points, dtype=np.float64)

    # 分别计算 x/y/z 最小值和最大值
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    # 中心点 = min/max 中点
    center = (min_xyz + max_xyz) / 2.0

    # 尺寸 = max - min
    size = max_xyz - min_xyz

    xmin, ymin, zmin = min_xyz
    xmax, ymax, zmax = max_xyz

    # 构造 AABB 的 8 个角点
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)

    return {
        "box_type": "AABB",
        "center": center,
        "size": size,

        # AABB 没有旋转角，因此 heading_angle 固定为 0
        "heading_angle": 0.0,

        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "corners": corners,
    }


def compute_pca_obb_3d(
    points: np.ndarray,
    up_axis: int = 2
) -> Dict:
    """
    根据点集在水平面上估计主方向三维有向框 PCA_OBB。

    PCA_OBB:
        使用 PCA 主成分分析估计物体在水平面上的主方向，
        然后在该主方向坐标系下生成有向框。

    注意：
        这里不是完整 3D PCA OBB。
        它只在水平面上估计旋转方向，高度方向仍然沿 up_axis。
        对室内数据更稳定，例如 SUNRGB-D 常用 z 轴作为竖直方向。

    参数：
        points:
            输入点云，shape = (N, 3)。

        up_axis:
            竖直轴编号。
            0 表示 x 轴为竖直；
            1 表示 y 轴为竖直；
            2 表示 z 轴为竖直。

    返回：
        Dict:
            PCA 有向框信息，包括中心、尺寸、主方向、角点等。
    """

    points = np.asarray(points, dtype=np.float64)

    # 点数不足时无法稳定计算 PCA，回退为 AABB
    if points.shape[0] < 3:
        return compute_aabb_3d(points)

    if up_axis not in [0, 1, 2]:
        raise ValueError("up_axis 只能是 0, 1, 2")

    # 水平面由除 up_axis 之外的两个轴组成
    # 例如 up_axis=2 时，horizontal_axes=[0,1]，即 x-y 平面
    horizontal_axes = [i for i in range(3) if i != up_axis]

    # 取出水平面坐标和竖直方向坐标
    xy = points[:, horizontal_axes]
    z = points[:, up_axis]

    # 计算水平面均值
    xy_mean = xy.mean(axis=0)

    # 去中心化，用于 PCA
    xy_centered = xy - xy_mean

    # 计算二维协方差矩阵
    cov = np.cov(xy_centered.T)

    # 如果协方差异常，则回退 AABB
    if not np.isfinite(cov).all():
        return compute_aabb_3d(points)

    # 对协方差矩阵做特征值分解
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # 按特征值从大到小排序
    # 最大特征值对应的方向就是点云水平面主方向
    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]

    # 保证局部坐标系方向稳定，避免左右手系翻转
    if np.linalg.det(eig_vecs) < 0:
        eig_vecs[:, 1] *= -1

    # 将水平面点投影到 PCA 局部坐标系
    local_xy = xy_centered @ eig_vecs

    # 在 PCA 坐标系下计算局部 min/max
    min_local = local_xy.min(axis=0)
    max_local = local_xy.max(axis=0)

    # 局部中心和局部尺寸
    center_local = (min_local + max_local) / 2.0
    size_local = max_local - min_local

    # 将局部中心变换回世界坐标系
    center_xy = xy_mean + center_local @ eig_vecs.T

    # 竖直方向仍然使用 min/max
    zmin = z.min()
    zmax = z.max()
    center_z = (zmin + zmax) / 2.0
    height = zmax - zmin

    # 构造完整 3D 中心点
    center = np.zeros(3, dtype=np.float64)
    center[horizontal_axes] = center_xy
    center[up_axis] = center_z

    # 第一主方向，即长度方向
    main_dir = eig_vecs[:, 0]

    # heading_angle 只在默认水平轴为 x-y 时有明确意义
    # up_axis=2 时，atan2(y, x) 表示水平面旋转角
    heading_angle = math.atan2(main_dir[1], main_dir[0])

    # size 的语义：
    # size[0] = PCA 主方向长度
    # size[1] = PCA 侧方向宽度
    # size[2] = 竖直高度
    size = np.array([
        size_local[0],
        size_local[1],
        height
    ], dtype=np.float64)

    # 计算局部二维框的四个角点
    lx_min, ly_min = min_local
    lx_max, ly_max = max_local

    local_corners_2d = np.array([
        [lx_min, ly_min],
        [lx_max, ly_min],
        [lx_max, ly_max],
        [lx_min, ly_max],
    ], dtype=np.float64)

    # 将二维角点从 PCA 局部坐标系变回世界水平坐标
    world_corners_2d = xy_mean + local_corners_2d @ eig_vecs.T

    # 构造上下两个平面的 8 个 3D 角点
    corners = []

    for zz in [zmin, zmax]:
        for p2d in world_corners_2d:
            p3d = np.zeros(3, dtype=np.float64)
            p3d[horizontal_axes] = p2d
            p3d[up_axis] = zz
            corners.append(p3d)

    corners = np.asarray(corners, dtype=np.float64)

    return {
        "box_type": "PCA_OBB",
        "center": center,
        "size": size,
        "heading_angle": float(heading_angle),
        "horizontal_axes": horizontal_axes,
        "up_axis": int(up_axis),
        "main_direction_2d": main_dir,
        "corners": corners,
    }


def build_3d_boxes_from_segments(
    points3d: np.ndarray,
    segments: List[Dict],
    box_type: str = "pca",
    min_points: int = 30,
    up_axis: int = 2,
    use_percentile_filter: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> List[Dict]:
    """
    根据三维实例分割结果生成三维框列表。

    参数：
        points3d:
            原始点云，shape = (N, 3)。

        segments:
            实例分割结果列表。
            每个 segment 中包含当前实例的 point_indices。

        box_type:
            三维框类型。
            "aabb" 表示轴对齐框；
            "pca" 表示 PCA 水平有向框。

        min_points:
            生成三维框所需的最少点数。
            点太少的实例不生成框。

        up_axis:
            竖直轴编号，默认 2，即 z 轴为竖直方向。

        use_percentile_filter:
            是否在生成框前用百分位过滤飞点。

        lower_percentile / upper_percentile:
            百分位过滤参数。

    返回：
        List[Dict]:
            三维框列表。
    """

    if box_type not in ["aabb", "pca"]:
        raise ValueError("box_type 只能是 'aabb' 或 'pca'")

    points3d = np.asarray(points3d, dtype=np.float64)

    boxes = []

    # 遍历每个实例分割结果
    for seg in segments:
        point_indices = np.asarray(seg["point_indices"], dtype=np.int64)

        # 点数不足，不生成框
        if point_indices.size < min_points:
            continue

        # 取出当前实例对应点云
        instance_points = points3d[point_indices]

        # 去掉 NaN / Inf 点
        finite = np.isfinite(instance_points).all(axis=1)
        instance_points = instance_points[finite]

        if instance_points.shape[0] < min_points:
            continue

        raw_num_points = int(instance_points.shape[0])

        # 生成框前可选地过滤极端点，避免飞点撑大框
        if use_percentile_filter:
            box_points = filter_points_by_percentile(
                instance_points,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile
            )
        else:
            box_points = instance_points

        # 如果过滤后点太少，则回退使用原始实例点
        if box_points.shape[0] < min_points:
            box_points = instance_points

        # 根据指定类型生成三维框
        if box_type == "aabb":
            box = compute_aabb_3d(box_points)
        else:
            box = compute_pca_obb_3d(box_points, up_axis=up_axis)

        # 补充实例语义信息
        box.update({
            "instance_id": int(seg["instance_id"]),
            "class_name": str(seg["class_name"]),
            "class_id": int(seg["class_id"]),
            "num_points": int(box_points.shape[0]),
            "raw_num_points": raw_num_points,
            "mask_path": seg.get("mask_path", None),
        })

        boxes.append(box)

    return boxes


def get_box_enclosing_aabb(box: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    把任意三维框转换为外接轴对齐框。

    作用：
        有些操作，例如快速 IoU 计算，可以统一使用 AABB 近似。
        对 PCA_OBB 来说，这里会用它的 corners 计算外接 AABB。

    参数：
        box:
            三维框字典。

    返回：
        Tuple[np.ndarray, np.ndarray]:
            min_xyz, max_xyz。
    """

    # 如果本身就是 AABB，直接读取 min/max
    if "min_xyz" in box and "max_xyz" in box:
        min_xyz = np.asarray(box["min_xyz"], dtype=np.float64).reshape(3)
        max_xyz = np.asarray(box["max_xyz"], dtype=np.float64).reshape(3)
        return min_xyz, max_xyz

    # 如果有 8 个角点，则根据角点计算外接 AABB
    if "corners" in box:
        corners = np.asarray(box["corners"], dtype=np.float64).reshape(-1, 3)
        return corners.min(axis=0), corners.max(axis=0)

    # 如果只有 center 和 size，则退化为轴对齐框处理
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size = np.asarray(box["size"], dtype=np.float64).reshape(3)
    half = np.abs(size) / 2.0

    return center - half, center + half


def compute_box_volume_from_size(box: Dict, eps: float = 1e-6) -> float:
    """
    根据三维框尺寸计算体积。

    参数：
        box:
            三维框字典，需要包含 size。

        eps:
            最小尺寸保护值，避免体积为 0。

    返回：
        float:
            框体积。
    """

    size = np.asarray(box.get("size", [0, 0, 0]), dtype=np.float64).reshape(3)
    size = np.abs(size)

    # 如果 size 中存在 NaN / Inf，直接返回 0
    if not np.isfinite(size).all():
        return 0.0

    # 使用 eps 防止某个方向尺寸为 0 导致体积为 0
    volume = float(np.prod(np.maximum(size, eps)))
    return volume


def points_inside_box(points: np.ndarray, box: Dict, eps: float = 1e-6) -> np.ndarray:
    """
    判断点集中的每个点是否位于指定三维框内部。

    支持：
        1. PCA_OBB 有向框内部判断；
        2. 其他情况回退为外接 AABB 判断。

    参数：
        points:
            输入点云，shape = (N, 3)。

        box:
            三维框字典。

        eps:
            容差，避免浮点误差导致边界点被排除。

    返回：
        np.ndarray:
            bool 数组，shape = (N,)。
            True 表示点在框内。
    """

    points = np.asarray(points, dtype=np.float64)

    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    # 非有限点不能认为在框内
    finite = np.isfinite(points).all(axis=1)

    # 如果是 PCA_OBB，并且必要字段齐全，则用有向框方式判断
    if (
        str(box.get("box_type", "")).upper() == "PCA_OBB"
        and "main_direction_2d" in box
        and "horizontal_axes" in box
        and "up_axis" in box
        and "center" in box
        and "size" in box
    ):
        center = np.asarray(box["center"], dtype=np.float64).reshape(3)
        size = np.abs(np.asarray(box["size"], dtype=np.float64).reshape(3))
        horizontal_axes = [int(x) for x in box["horizontal_axes"]]
        up_axis = int(box["up_axis"])
        main_dir = np.asarray(box["main_direction_2d"], dtype=np.float64).reshape(2)

        # 主方向归一化
        norm = float(np.linalg.norm(main_dir))

        if norm > eps and np.isfinite(center).all() and np.isfinite(size).all():
            main_dir = main_dir / norm

            # side_dir 是主方向的垂直方向
            side_dir = np.array([-main_dir[1], main_dir[0]], dtype=np.float64)

            # 构造二维局部坐标基
            basis = np.stack([main_dir, side_dir], axis=1)

            # 将点转换到 PCA_OBB 的局部水平坐标系
            local_xy = (points[:, horizontal_axes] - center[horizontal_axes]) @ basis

            # 竖直方向局部坐标
            local_up = points[:, up_axis] - center[up_axis]

            # 半尺寸，加 eps 作为容差
            half = size / 2.0 + eps

            # 判断点是否在三个局部方向的范围内
            return (
                finite
                & (np.abs(local_xy[:, 0]) <= half[0])
                & (np.abs(local_xy[:, 1]) <= half[1])
                & (np.abs(local_up) <= half[2])
            )

    # 非 PCA_OBB 或字段不完整时，回退为外接 AABB 判断
    min_xyz, max_xyz = get_box_enclosing_aabb(box)
    min_xyz = np.asarray(min_xyz, dtype=np.float64).reshape(3) - eps
    max_xyz = np.asarray(max_xyz, dtype=np.float64).reshape(3) + eps

    return finite & np.all((points >= min_xyz) & (points <= max_xyz), axis=1)


def count_semantic_points_inside_box(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    box: Dict,
    eps: float = 1e-6
) -> int:
    """
    统计指定实例语义点中落在三维框内部的点数。

    注意：
        这里只统计属于当前 instance_id 的语义点。
        如果 box 中 class_id >= 0，还会进一步要求 class_id 一致。

    参数：
        points3d:
            原始点云，shape = (N, 3)。

        point_class_ids:
            每个点的类别 ID。

        point_instance_ids:
            每个点的实例 ID。

        box:
            当前三维框。

    返回：
        int:
            当前实例中落在该框内部的点数。
    """

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32)

    instance_id = int(box.get("instance_id", 0))
    class_id = int(box.get("class_id", -1))

    # 先按 instance_id 筛选语义点
    semantic_mask = point_instance_ids == instance_id

    # 如果 class_id 合法，则进一步要求类别一致
    if class_id >= 0:
        semantic_mask &= point_class_ids == class_id

    candidate_indices = np.where(semantic_mask)[0]

    if candidate_indices.size == 0:
        return 0

    # 判断候选语义点是否落在当前三维框内部
    inside = points_inside_box(points3d[candidate_indices], box, eps=eps)

    return int(inside.sum())


def compute_aabb_iou_3d(box_a: Dict, box_b: Dict, eps: float = 1e-6) -> float:
    """
    使用外接轴对齐框近似计算两个三维框的 IoU。

    注意：
        即使输入是 PCA_OBB，这里也会先转换成外接 AABB。
        因此这是近似 IoU，不是严格的 OBB IoU。

    参数：
        box_a:
            第一个三维框。

        box_b:
            第二个三维框。

    返回：
        float:
            两个框的 3D IoU。
    """

    # 将两个框统一转换为外接 AABB
    a_min, a_max = get_box_enclosing_aabb(box_a)
    b_min, b_max = get_box_enclosing_aabb(box_b)

    # 交集盒子的 min/max
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)

    # 交集尺寸，负数截断为 0
    inter_size = np.maximum(inter_max - inter_min, 0.0)

    # 交集体积
    inter_vol = float(np.prod(inter_size))

    # 两个框各自体积
    a_vol = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 0.0)))

    # 并集体积
    union = a_vol + b_vol - inter_vol

    if union <= eps:
        return 0.0

    return float(inter_vol / union)


def add_density_info_to_box(
    box: Dict,
    points3d: Optional[np.ndarray] = None,
    point_class_ids: Optional[np.ndarray] = None,
    point_instance_ids: Optional[np.ndarray] = None,
    eps: float = 1e-6
) -> Dict:
    """
    为三维框补充体积、框内语义点数、密度和质量评分。

    质量评分逻辑：
        point_density = 框内点数 / 框体积

        box_quality_score = density * log(1 + 点数)

    为什么要乘 log 点数：
        单纯密度高不一定可靠。
        一个很小的碎片可能密度很高，但点数很少。
        加入 log 点数后，可以让真实主体框比小碎片框更容易被保留。

    参数：
        box:
            原始三维框。

        points3d / point_class_ids / point_instance_ids:
            如果提供这些数据，会精确统计当前实例语义点落在框内的数量。

    返回：
        Dict:
            补充质量信息后的新框。
    """

    # 拷贝 box，避免直接修改原对象
    new_box = dict(box)

    # 默认使用 box 自带 num_points 作为密度点数
    num_points = int(new_box.get("num_points", 0))
    density_num_points = num_points
    density_source = "box_num_points"

    # 如果提供完整语义标签，则重新统计当前实例落在框内的语义点数量
    if points3d is not None and point_class_ids is not None and point_instance_ids is not None:
        density_num_points = count_semantic_points_inside_box(
            points3d=points3d,
            point_class_ids=point_class_ids,
            point_instance_ids=point_instance_ids,
            box=new_box,
            eps=eps
        )
        density_source = "semantic_points_inside_box"

    # 计算框体积
    volume = compute_box_volume_from_size(new_box, eps=eps)

    # 计算点密度
    if volume <= eps:
        density = float("inf") if density_num_points > 0 else 0.0
    else:
        density = float(density_num_points / volume)

    # 综合评分：
    # 密度越高越好，点数越多越可信
    quality = float(density * math.log1p(max(density_num_points, 1)))

    # 写入新字段
    new_box["box_volume"] = float(volume)
    new_box["density_num_points"] = int(density_num_points)
    new_box["num_semantic_points_in_box"] = int(density_num_points)
    new_box["density_source"] = density_source
    new_box["point_density"] = float(density)
    new_box["box_quality_score"] = float(quality)

    return new_box


def apply_box_density_filter_and_nms(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    segments: List[Dict],
    boxes: List[Dict],
    background_class_id: int = -1,
    enable_density_filter: bool = True,
    min_box_density: float = 30.0,
    min_box_inner_points: int = 0,
    max_box_volume: Optional[float] = None,
    enable_box_nms: bool = True,
    box_nms_iou_thresh: float = 0.10,
    box_nms_class_aware: bool = False,
    remove_suppressed_box_points: bool = True
) -> Dict:
    """
    根据框内点数、密度、体积和重叠关系过滤三维框。

    过滤逻辑：
        1. 先为每个框计算质量信息；
        2. 根据点密度过滤低质量框；
        3. 根据框内点数过滤过小实例；
        4. 根据最大体积过滤异常大框；
        5. 对剩余框做 3D NMS，删除重叠严重的框；
        6. 可选地把被删除框对应的语义点改回背景；
        7. 重建 segments。

    参数：
        points3d:
            原始点云，shape = (N, 3)。

        point_class_ids:
            每个点的语义类别 ID。

        point_instance_ids:
            每个点的实例 ID。

        segments:
            实例分割结果。

        boxes:
            待过滤的三维框列表。

        background_class_id:
            被删除实例对应点的类别 ID，默认 -1。

        enable_density_filter:
            是否启用密度过滤。

        min_box_density:
            最小框点密度。
            低于该值的框会被删除。

        min_box_inner_points:
            框内最少语义点数量。
            如果大于 0，则低于该点数的框会被删除。

        max_box_volume:
            最大允许体积。
            如果不为 None，体积超过该值的框会被删除。

        enable_box_nms:
            是否启用 3D NMS。

        box_nms_iou_thresh:
            3D NMS IoU 阈值。
            两个框 IoU 超过该阈值，低质量框会被抑制。

        box_nms_class_aware:
            是否按类别做 NMS。
            True 表示不同类别之间不互相抑制；
            False 表示所有类别之间都参与抑制。

        remove_suppressed_box_points:
            是否将被删除框对应的实例点重置为背景。

    返回：
        Dict:
            {
                "point_class_ids": 更新后的类别标签,
                "point_instance_ids": 更新后的实例标签,
                "segments": 更新后的实例分割,
                "boxes": 保留下来的三维框,
                "box_quality_filter_stats": 过滤统计信息
            }
    """

    # 统一输入格式，并拷贝标签，避免修改外部原始数组
    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32).copy()
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32).copy()

    # 为每个框补充体积、密度、质量分数
    boxes_with_quality = [
        add_density_info_to_box(
            box,
            points3d=points3d,
            point_class_ids=point_class_ids,
            point_instance_ids=point_instance_ids
        )
        for box in boxes
    ]

    # 保存被删除的 instance_id
    removed_instance_ids = set()

    # 记录被密度 / 体积过滤删除的框
    density_removed = []

    # 记录被 NMS 删除的框
    nms_removed = []

    # -------------------------
    # 1. 密度过滤 / 体积过滤
    # -------------------------

    candidate_boxes = []

    for box in boxes_with_quality:
        instance_id = int(box["instance_id"])
        density = float(box.get("point_density", 0.0))
        volume = float(box.get("box_volume", 0.0))
        num_points = int(box.get("num_points", 0))
        density_num_points = int(box.get("density_num_points", num_points))

        remove_reason = None

        # 点密度太低，说明框可能过大或者点太稀疏
        if enable_density_filter and density < float(min_box_density):
            remove_reason = "low_box_density"

        # 框内语义点太少，删除
        if int(min_box_inner_points) > 0 and density_num_points < int(min_box_inner_points):
            remove_reason = "too_few_box_inner_points"

        # 框体积过大，删除
        if max_box_volume is not None and volume > float(max_box_volume):
            remove_reason = "too_large_box_volume"

        # 如果命中任意删除条件，则加入删除列表
        if remove_reason is not None:
            removed_instance_ids.add(instance_id)

            density_removed.append({
                "instance_id": instance_id,
                "class_name": str(box.get("class_name", "")),
                "class_id": int(box.get("class_id", -1)),
                "num_points": num_points,
                "density_num_points": density_num_points,
                "box_volume": volume,
                "point_density": density,
                "reason": remove_reason,
            })
        else:
            candidate_boxes.append(box)

    # -------------------------
    # 2. 3D box NMS，抑制重叠框
    # -------------------------

    kept_boxes = []

    if enable_box_nms and len(candidate_boxes) > 1:
        # 质量高的框优先保留
        # 排序依据：
        #   1. box_quality_score
        #   2. density_num_points
        order = sorted(
            range(len(candidate_boxes)),
            key=lambda i: (
                float(candidate_boxes[i].get("box_quality_score", 0.0)),
                int(candidate_boxes[i].get(
                    "density_num_points",
                    candidate_boxes[i].get("num_points", 0)
                ))
            ),
            reverse=True
        )

        # 标记哪些框已经被抑制
        suppressed = np.zeros(len(candidate_boxes), dtype=bool)

        # 按质量从高到低遍历
        for order_i in order:
            if suppressed[order_i]:
                continue

            cur_box = candidate_boxes[order_i]
            kept_boxes.append(cur_box)

            cur_class_id = int(cur_box.get("class_id", -1))

            # 用当前保留框去抑制其他重叠框
            for order_j in order:
                if order_j == order_i or suppressed[order_j]:
                    continue

                other_box = candidate_boxes[order_j]
                other_class_id = int(other_box.get("class_id", -1))

                # 类别感知 NMS：
                # 如果开启，则不同类别之间不互相抑制
                if box_nms_class_aware and cur_class_id != other_class_id:
                    continue

                # 计算近似 3D IoU
                # 注意这里使用外接 AABB 近似，不是严格 OBB IoU
                iou = compute_aabb_iou_3d(cur_box, other_box)

                # IoU 超过阈值，则抑制质量较低的 other_box
                if iou >= float(box_nms_iou_thresh):
                    suppressed[order_j] = True

                    other_instance_id = int(other_box["instance_id"])
                    removed_instance_ids.add(other_instance_id)

                    nms_removed.append({
                        "instance_id": other_instance_id,
                        "class_name": str(other_box.get("class_name", "")),
                        "class_id": other_class_id,
                        "num_points": int(other_box.get("num_points", 0)),
                        "density_num_points": int(other_box.get(
                            "density_num_points",
                            other_box.get("num_points", 0)
                        )),
                        "point_density": float(other_box.get("point_density", 0.0)),
                        "box_volume": float(other_box.get("box_volume", 0.0)),
                        "suppressed_by_instance_id": int(cur_box["instance_id"]),
                        "iou": float(iou),
                        "reason": "box_nms_overlap",
                    })
    else:
        # 不启用 NMS 时，候选框全部保留
        kept_boxes = candidate_boxes

    # -------------------------
    # 3. 被删除实例的语义点改回背景
    # -------------------------

    removed_point_count = 0

    if remove_suppressed_box_points and len(removed_instance_ids) > 0:
        for instance_id in removed_instance_ids:
            # 找到该实例所有点
            remove_mask = point_instance_ids == int(instance_id)

            # 统计删除点数
            removed_point_count += int(remove_mask.sum())

            # 类别改成背景
            point_class_ids[remove_mask] = background_class_id

            # 实例 ID 改成 0，表示无实例
            point_instance_ids[remove_mask] = 0

    # -------------------------
    # 4. 重建 segments
    # -------------------------

    filtered_segments = []

    for seg in segments:
        instance_id = int(seg["instance_id"])

        # 已删除的实例不再保留
        if instance_id in removed_instance_ids:
            continue

        # 根据更新后的 point_instance_ids 重新找该实例的点
        indices = np.where(point_instance_ids == instance_id)[0]

        # 如果没有点了，则跳过
        if indices.size == 0:
            continue

        # 更新 segment 中的点索引和点数
        filtered_segments.append({
            **seg,
            "point_indices": indices,
            "num_points": int(indices.size),
        })

    # 汇总过滤统计信息
    stats = {
        "enabled_density_filter": bool(enable_density_filter),
        "min_box_density": float(min_box_density),
        "min_box_inner_points": int(min_box_inner_points),
        "max_box_volume": None if max_box_volume is None else float(max_box_volume),
        "enabled_box_nms": bool(enable_box_nms),
        "box_nms_iou_thresh": float(box_nms_iou_thresh),
        "box_nms_class_aware": bool(box_nms_class_aware),
        "remove_suppressed_box_points": bool(remove_suppressed_box_points),
        "before_boxes": int(len(boxes)),
        "after_boxes": int(len(kept_boxes)),
        "removed_boxes": int(len(boxes) - len(kept_boxes)),
        "removed_instance_ids": sorted([int(x) for x in removed_instance_ids]),
        "removed_point_count": int(removed_point_count),
        "density_removed": density_removed,
        "nms_removed": nms_removed,
    }

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "segments": filtered_segments,
        "boxes": kept_boxes,
        "box_quality_filter_stats": stats,
    }