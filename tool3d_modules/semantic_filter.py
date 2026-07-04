# -*- coding: utf-8 -*-
"""
三维语义过滤模块

功能：
    对已经完成 2D mask → 3D 点云赋值后的语义点云进行后处理，
    主要用于删除语义飞点、孤立点、边界误投影点和远离主体的异常点。

典型使用场景：
    1. 2D mask 投影到 3D 点云后，部分背景点被错误标注为物体；
    2. 一个物体实例周围存在零散飞点；
    3. 由于遮挡、投影误差、mask 边界不准，导致点云语义边界外扩；
    4. 希望通过滤波强度 0 / 1 / 2 / 3 控制清理力度。

核心思路：
    - strength=0：不滤波
    - strength=1：轻度去飞点
    - strength=2：中度去飞点
    - strength=3：强力过滤，只保留主体结构
"""

from typing import Dict, List

import numpy as np


def get_semantic_filter_params(strength: int) -> Dict:
    """
    根据语义滤波强度返回对应的滤波参数。

    参数：
        strength:
            语义滤波强度，只允许取 0、1、2、3。

            0：
                不启用滤波，保留所有语义点。

            1：
                轻度滤波，主要删除明显孤立点和少量边界飞点。

            2：
                中度滤波，删除更多边界异常点和稀疏点。

            3：
                强力滤波，过滤最严格，并且可以只保留最大连通主体。

    返回：
        Dict:
            当前强度对应的所有滤波参数。
    """

    # 确保 strength 是整数类型
    strength = int(strength)

    # 不同滤波强度对应的参数表
    table = {
        0: {
            # 是否启用滤波
            "enabled": False,

            # 邻域搜索半径，单位通常与点云坐标单位一致，例如米
            "radius": 0.00,

            # 半径邻域内至少需要多少个邻居，低于该值认为是稀疏点
            "min_neighbors": 0,

            # 连通域最少点数
            "min_component_points": 0,

            # 连通域最少占实例点数比例
            "min_component_ratio": 0.00,

            # 是否只保留最大连通域
            "keep_largest_only": False,

            # 是否只对边界和包围盒外部候选点进行过滤
            "boundary_only": True,

            # 边界候选点百分位
            "boundary_percentile": 0.0,

            # 鲁棒包围盒下界百分位
            "robust_box_lower_percentile": 0.0,

            # 鲁棒包围盒上界百分位
            "robust_box_upper_percentile": 100.0,

            # 用于估计主体核心点的百分位
            "robust_box_core_percentile": 100.0,

            # 包围盒外部点硬删除距离倍数
            "outside_hard_remove_radius": 0.0,

            # 包围盒外部点硬删除比例
            "outside_hard_remove_box_ratio": 0.0,

            # 离包围盒越远，对邻居数要求越高
            "outside_neighbor_boost": 0.0,
        },

        1: {
            # 轻度滤波
            "enabled": True,
            "radius": 0.08,
            "min_neighbors": 4,
            "min_component_points": 8,
            "min_component_ratio": 0.005,
            "keep_largest_only": False,
            "boundary_only": True,
            "boundary_percentile": 8.0,
            "robust_box_lower_percentile": 2.0,
            "robust_box_upper_percentile": 98.0,
            "robust_box_core_percentile": 88.0,
            "outside_hard_remove_radius": 4.0,
            "outside_hard_remove_box_ratio": 0.35,
            "outside_neighbor_boost": 1.0,
        },

        2: {
            # 中度滤波
            "enabled": True,
            "radius": 0.10,
            "min_neighbors": 6,
            "min_component_points": 15,
            "min_component_ratio": 0.010,
            "keep_largest_only": False,
            "boundary_only": True,
            "boundary_percentile": 12.0,
            "robust_box_lower_percentile": 3.0,
            "robust_box_upper_percentile": 97.0,
            "robust_box_core_percentile": 82.0,
            "outside_hard_remove_radius": 3.0,
            "outside_hard_remove_box_ratio": 0.25,
            "outside_neighbor_boost": 2.0,
        },

        3: {
            # 强力滤波
            "enabled": True,
            "radius": 0.15,
            "min_neighbors": 8,
            "min_component_points": 25,
            "min_component_ratio": 0.020,

            # 强滤波下只保留最大主体，适合去除大面积飞点
            "keep_largest_only": True,

            "boundary_only": True,
            "boundary_percentile": 18.0,
            "robust_box_lower_percentile": 5.0,
            "robust_box_upper_percentile": 95.0,
            "robust_box_core_percentile": 75.0,
            "outside_hard_remove_radius": 2.0,
            "outside_hard_remove_box_ratio": 0.18,
            "outside_neighbor_boost": 3.0,
        },
    }

    # 如果传入的强度不在允许范围内，直接报错
    if strength not in table:
        raise ValueError("semantic_filter_strength 只能是 0, 1, 2, 3")

    # 拷贝参数，避免外部修改原始 table
    params = dict(table[strength])

    # 额外保存当前 strength，方便后续统计和日志记录
    params["strength"] = strength

    return params


def _connected_components_from_neighbor_lists(
    neighbor_lists: List[List[int]],
    valid_local_mask: np.ndarray
) -> List[np.ndarray]:
    """
    根据邻接表和有效点 mask 计算连通域。

    这是一个内部辅助函数，主要用于：
        1. 已经通过半径邻域判断出哪些点是稠密点；
        2. 再根据邻接关系把稠密点划分为多个连通区域；
        3. 后续可以删除过小连通域，保留主体点云。

    参数：
        neighbor_lists:
            邻接表。
            neighbor_lists[i] 表示第 i 个点的邻居点索引列表。

        valid_local_mask:
            有效点布尔 mask。
            True 表示该点参与连通域计算；
            False 表示该点不参与。

    返回：
        List[np.ndarray]:
            连通域列表。
            每个元素是一个连通域内的点索引数组。
            返回结果会按连通域大小从大到小排序。
    """

    # 转换为 bool 类型，确保后续逻辑稳定
    valid_local_mask = np.asarray(valid_local_mask, dtype=bool)

    # 点数量
    n = int(valid_local_mask.shape[0])

    # visited 用来记录某个点是否已经被搜索过
    visited = np.zeros(n, dtype=bool)

    # 保存所有连通域
    components = []

    # 取出所有有效点索引，并转成 set，便于快速判断某个邻居是否有效
    valid_set = set(np.where(valid_local_mask)[0].tolist())

    # 遍历每一个有效点，使用 DFS 搜索连通域
    for start in list(valid_set):
        # 如果该点已经访问过，说明已经属于某个连通域，跳过
        if visited[start]:
            continue

        # 使用栈实现深度优先搜索
        stack = [start]
        visited[start] = True

        # 当前连通域的点索引
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            # 遍历当前点的所有邻居
            for nb in neighbor_lists[cur]:
                # 邻居必须满足：
                # 1. 是有效点
                # 2. 没有被访问过
                if nb in valid_set and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        # 保存当前连通域
        components.append(np.asarray(comp, dtype=np.int64))

    # 按连通域点数从大到小排序
    components.sort(key=lambda x: x.size, reverse=True)

    return components


def _aabb_boundary_candidate_mask(points: np.ndarray, percentile: float) -> np.ndarray:
    """
    根据 AABB 包围盒筛选边界候选点。

    AABB:
        Axis-Aligned Bounding Box，轴对齐包围盒。
        即分别用 x、y、z 三个方向的 min/max 构成包围盒。

    这个函数用于找出靠近包围盒边界的点，
    因为 mask 投影错误、边界扩散、语义飞点通常更容易出现在边界附近。

    参数：
        points:
            当前实例的点云，shape = (N, 3)。

        percentile:
            边界距离百分位。
            数值越大，越多点会被认为是边界候选点。

    返回：
        np.ndarray:
            bool 数组，shape = (N,)。
            True 表示该点是边界候选点。
    """

    points = np.asarray(points, dtype=np.float64)

    # 空点云直接返回空 bool 数组
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    # percentile <= 0 表示不筛选边界候选点
    if percentile <= 0:
        return np.zeros((points.shape[0],), dtype=bool)

    # 计算 AABB 的最小点和最大点
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    # 计算包围盒尺寸，避免除 0
    span = np.maximum(max_xyz - min_xyz, 1e-6)

    # 将点归一化到 [0, 1] 的包围盒空间中
    normalized = (points - min_xyz) / span

    # 计算每个点到最近包围盒面的距离
    # 越接近 0，说明越靠近边界
    nearest_face_distance = np.minimum(normalized, 1.0 - normalized).min(axis=1)

    # 根据百分位得到边界阈值
    threshold = np.percentile(nearest_face_distance, float(percentile))

    # 小于阈值的点认为是边界候选点
    return nearest_face_distance <= threshold


def _robust_box_filter_candidates(
    points: np.ndarray,
    boundary_percentile: float,
    lower_percentile: float,
    upper_percentile: float,
    core_percentile: float
) -> Dict:
    """
    使用鲁棒包围盒筛选需要进一步检查的候选点。

    该函数不是直接删除点，而是找出“可疑点”：
        1. 位于鲁棒包围盒外部的点；
        2. 位于鲁棒包围盒内部但靠近边界的点。

    为什么用鲁棒包围盒：
        普通 min/max 包围盒容易被飞点拉大。
        这里使用百分位数构造 box，例如 3%~97%，可以减少极端飞点影响。

    参数：
        points:
            当前实例的有限点云，shape = (N, 3)。

        boundary_percentile:
            内部边界点筛选百分位。

        lower_percentile:
            鲁棒包围盒下界百分位。

        upper_percentile:
            鲁棒包围盒上界百分位。

        core_percentile:
            用于筛选主体核心点的百分位。
            会先基于 median 和 MAD 估计核心点，再用核心点构造包围盒。

    返回：
        Dict:
            {
                "candidate_mask":
                    可疑候选点 mask，包括 box 外部点和边界点。

                "outside_distance":
                    每个点到鲁棒 box 外部的距离。

                "outside_mask":
                    是否位于鲁棒 box 外部。

                "boundary_mask":
                    是否是 box 内部边界候选点。

                "box_min":
                    鲁棒包围盒最小坐标。

                "box_max":
                    鲁棒包围盒最大坐标。
            }
    """

    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])

    # 空点云情况，返回空结构，避免后续报错
    if n == 0:
        return {
            "candidate_mask": np.zeros((0,), dtype=bool),
            "outside_distance": np.zeros((0,), dtype=np.float64),
            "outside_mask": np.zeros((0,), dtype=bool),
            "boundary_mask": np.zeros((0,), dtype=bool),
            "box_min": np.zeros((3,), dtype=np.float64),
            "box_max": np.zeros((3,), dtype=np.float64),
        }

    # 默认所有点都作为构造鲁棒 box 的点
    core_points = points

    # 当点数足够，并且 core_percentile 在合理范围内时，
    # 先筛选主体核心点，减少飞点对包围盒的影响
    if 0.0 < core_percentile < 100.0 and n >= 20:
        # 使用中位数作为鲁棒中心
        center = np.median(points, axis=0)

        # 使用 MAD 估计每个方向的鲁棒尺度
        scale = np.median(np.abs(points - center), axis=0)
        scale = np.maximum(scale, 1e-6)

        # 计算每个点相对于中心的归一化距离
        normalized_distance = np.linalg.norm((points - center) / scale, axis=1)

        # 根据 core_percentile 找出主体核心点
        core_threshold = np.percentile(normalized_distance, float(core_percentile))
        core_mask = normalized_distance <= core_threshold

        # 如果核心点数量足够，则用核心点构造包围盒
        if int(core_mask.sum()) >= max(10, int(n * 0.25)):
            core_points = points[core_mask]

    # 使用百分位数构造鲁棒 box
    low = np.percentile(core_points, float(lower_percentile), axis=0)
    high = np.percentile(core_points, float(upper_percentile), axis=0)

    # 如果 box 不合法，则回退到普通 min/max 包围盒
    if not np.all(high > low):
        low = points.min(axis=0)
        high = points.max(axis=0)

    # box 尺寸，避免除 0
    span = np.maximum(high - low, 1e-6)

    # 计算点在 box 下界之外的距离
    below = np.maximum(low - points, 0.0)

    # 计算点在 box 上界之外的距离
    above = np.maximum(points - high, 0.0)

    # box 外部方向向量
    outside_vector = below + above

    # 点到 box 外部的欧氏距离
    outside_distance = np.linalg.norm(outside_vector, axis=1)

    # outside_distance > 0 表示点在 box 外
    outside_mask = outside_distance > 0.0

    # 将点归一化到鲁棒 box 坐标系
    normalized = (points - low) / span

    # 判断点是否位于 box 内部
    inside_mask = np.all((normalized >= 0.0) & (normalized <= 1.0), axis=1)

    # 计算 box 内部点到最近 box 面的距离
    nearest_face_distance = np.minimum(normalized, 1.0 - normalized).min(axis=1)

    # 初始化边界点 mask
    boundary_mask = np.zeros((n,), dtype=bool)

    # 从 box 内部点中筛选靠近边界的候选点
    if boundary_percentile > 0 and int(inside_mask.sum()) > 0:
        inside_distances = nearest_face_distance[inside_mask]
        threshold = np.percentile(inside_distances, float(boundary_percentile))
        boundary_mask = inside_mask & (nearest_face_distance <= threshold)

    return {
        # 候选点 = box 外部点 或 box 内部边界点
        "candidate_mask": outside_mask | boundary_mask,

        # 每个点到 box 外部的距离
        "outside_distance": outside_distance,

        # 是否位于 box 外
        "outside_mask": outside_mask,

        # 是否属于 box 内部边界候选点
        "boundary_mask": boundary_mask,

        # 鲁棒 box 最小坐标
        "box_min": low,

        # 鲁棒 box 最大坐标
        "box_max": high,
    }


def filter_semantic_outliers_3d(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    segments: List[Dict],
    strength: int = 0,
    background_class_id: int = -1,
    min_points: int = 30
) -> Dict:
    """
    对三维语义点云执行离群点过滤。

    参数：
        points3d:
            原始点云，shape = (N, 3)。

        point_class_ids:
            每个点的语义类别 ID，shape = (N,)。
            background_class_id 通常为 -1。

        point_instance_ids:
            每个点的实例 ID，shape = (N,)。
            0 通常表示未分配实例。

        segments:
            实例分割结果列表。
            每个 segment 通常包含：
                {
                    "instance_id": 实例 ID,
                    "class_id": 类别 ID,
                    "class_name": 类别名称,
                    "point_indices": 当前实例点索引,
                    "num_points": 当前实例点数
                }

        strength:
            语义滤波强度，取值 0、1、2、3。

        background_class_id:
            被删除的语义点会被重置为该类别。
            默认 -1，表示背景。

        min_points:
            最小点数阈值。
            点数太少的实例不进行强滤波，避免小目标被误删。

    返回：
        Dict:
            {
                "point_class_ids":
                    滤波后的类别标签。

                "point_instance_ids":
                    滤波后的实例标签。

                "segments":
                    滤波后的实例列表。

                "filter_stats":
                    滤波统计信息。
            }
    """

    # 读取滤波强度对应参数
    strength = int(strength)
    params = get_semantic_filter_params(strength)

    # 统一输入格式，并拷贝标签数组，避免直接修改外部原始数组
    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32).copy()
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32).copy()

    # 滤波前有语义标签的点数量
    before_labeled = int((point_class_ids >= 0).sum())

    # 如果不启用滤波，或者没有任何已标注点，直接返回原始结果
    if not params["enabled"] or before_labeled == 0:
        return {
            "point_class_ids": point_class_ids,
            "point_instance_ids": point_instance_ids,
            "segments": segments,
            "filter_stats": {
                "semantic_filter_strength": strength,
                "enabled": False,
                "before_labeled_points": before_labeled,
                "after_labeled_points": before_labeled,
                "removed_labeled_points": 0,
                "removed_ratio": 0.0,
            }
        }

    # cKDTree 用于高效进行 3D 半径邻域搜索
    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise ImportError(
            "3D 语义图滤波需要 scipy.spatial.cKDTree。请确认 scipy 已安装。"
        ) from e

    # 从参数字典中读取具体滤波参数
    radius = float(params["radius"])
    min_neighbors = int(params["min_neighbors"])
    min_component_points = int(params["min_component_points"])
    min_component_ratio = float(params["min_component_ratio"])
    keep_largest_only = bool(params["keep_largest_only"])
    boundary_only = bool(params.get("boundary_only", True))
    boundary_percentile = float(params.get("boundary_percentile", 10.0))
    robust_box_lower_percentile = float(params.get("robust_box_lower_percentile", 2.0))
    robust_box_upper_percentile = float(params.get("robust_box_upper_percentile", 98.0))
    robust_box_core_percentile = float(params.get("robust_box_core_percentile", 85.0))
    outside_hard_remove_radius = float(params.get("outside_hard_remove_radius", 3.0))
    outside_hard_remove_box_ratio = float(params.get("outside_hard_remove_box_ratio", 0.25))
    outside_neighbor_boost = float(params.get("outside_neighbor_boost", 1.0))

    # 记录总共删除的点数
    removed_total = 0

    # 保存滤波后的 segment
    filtered_segments = []

    # 保存每个实例的滤波统计信息
    segment_stats = []

    # 逐个实例进行滤波
    # 注意：以实例为单位处理，避免把两个同类但空间分离的物体错误合并
    for seg in segments:
        instance_id = int(seg["instance_id"])
        class_id = int(seg["class_id"])

        # 根据 instance_id 从全局点云标签中取出当前实例点索引
        obj_indices = np.where(point_instance_ids == instance_id)[0]
        raw_n = int(obj_indices.size)

        # 当前实例没有点，直接跳过
        if raw_n == 0:
            continue

        # 如果点数太少，不进行半径邻域 / 连通域过滤
        # 这样可以避免小物体被错误删除
        if raw_n < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })

            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        # 取出当前实例的点坐标
        obj_points = points3d[obj_indices]

        # 判断每个点坐标是否是有限值
        # NaN / Inf 点无法参与正常几何计算
        finite = np.isfinite(obj_points).all(axis=1)

        # 如果有限点数量太少，也跳过滤波
        if finite.sum() < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })

            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        # 当前实例中有限点的局部索引
        finite_local_indices = np.where(finite)[0]

        # 当前实例中有限点的坐标
        finite_points = obj_points[finite]

        # boundary_only=True 时：
        # 只检查鲁棒 box 外部点和边界候选点，不对整个物体主体做连通域过滤
        if boundary_only:
            # 使用鲁棒包围盒筛选可疑候选点
            candidate_info = _robust_box_filter_candidates(
                finite_points,
                boundary_percentile=boundary_percentile,
                lower_percentile=robust_box_lower_percentile,
                upper_percentile=robust_box_upper_percentile,
                core_percentile=robust_box_core_percentile
            )

            # 在 finite_points 中的候选点索引
            candidate_finite_indices = np.where(candidate_info["candidate_mask"])[0]

            # 每个有限点到鲁棒 box 外部的距离
            outside_distance = candidate_info["outside_distance"]

            # 默认当前实例所有点都保留
            keep_local = np.ones(raw_n, dtype=bool)

            # 如果存在候选点，则进一步根据邻域密度和外部距离判断是否删除
            if candidate_finite_indices.size > 0:
                # 为当前实例有限点建立 KDTree
                tree = cKDTree(finite_points)

                # 取出候选点坐标
                candidate_points = finite_points[candidate_finite_indices]

                # 查询每个候选点在 radius 半径内的邻居
                candidate_neighbor_lists = tree.query_ball_point(candidate_points, r=radius)

                # 统计每个候选点的邻居数量
                candidate_neighbor_counts = np.array(
                    [len(x) for x in candidate_neighbor_lists],
                    dtype=np.int32
                )

                # 候选点到 box 外部的距离
                candidate_outside_distance = outside_distance[candidate_finite_indices]

                # 鲁棒 box 尺寸
                robust_box_size = candidate_info["box_max"] - candidate_info["box_min"]

                # 鲁棒 box 对角线长度
                robust_box_diag = float(np.linalg.norm(np.maximum(robust_box_size, 0.0)))

                # 硬删除距离阈值：
                # 如果点离鲁棒 box 太远，不管邻居数多少，都可以删除
                hard_remove_distance = max(
                    radius * outside_hard_remove_radius,
                    robust_box_diag * outside_hard_remove_box_ratio,
                    1e-6
                )

                # 距离超过 hard_remove_distance 的候选点直接删除
                hard_remove_mask = candidate_outside_distance >= hard_remove_distance

                # 距离 box 越远，要求的邻居数越高
                distance_boost = np.ceil(
                    (candidate_outside_distance / max(radius, 1e-6)) * outside_neighbor_boost
                ).astype(np.int32)

                # 每个候选点实际要求的最小邻居数
                required_neighbors = min_neighbors + distance_boost

                # 邻居数不足，认为是稀疏飞点
                sparse_candidate_mask = candidate_neighbor_counts < required_neighbors

                # 最终删除条件：
                # 1. 离鲁棒 box 过远
                # 2. 或者邻域内点太少
                remove_candidate_mask = hard_remove_mask | sparse_candidate_mask

                # 将 finite_points 局部索引映射回 obj_points 局部索引
                raw_candidate_indices = finite_local_indices[candidate_finite_indices]

                # 当前实例局部坐标中需要删除的点索引
                remove_local_indices = raw_candidate_indices[remove_candidate_mask]

                # 标记为不保留
                keep_local[remove_local_indices] = False

            # 非有限点直接删除
            keep_local[~finite] = False

            # 当前实例中保留下来的全局点索引
            keep_global = obj_indices[keep_local]

            # 当前实例中被删除的全局点索引
            remove_global = obj_indices[~keep_local]

            # 将被删除点重置为背景类别和无实例
            if remove_global.size > 0:
                point_class_ids[remove_global] = background_class_id
                point_instance_ids[remove_global] = 0

            # 当前实例删除点数
            removed = int(remove_global.size)

            # 当前实例滤波后点数
            after_n = int(keep_global.size)

            # 累加总删除点数
            removed_total += removed

            # 如果滤波后仍有点，则保留该 segment
            if after_n > 0:
                filtered_segments.append({
                    **seg,
                    "point_indices": keep_global,
                    "num_points": after_n,
                    "raw_num_points_before_semantic_filter": raw_n,
                    "semantic_filter_removed_points": removed,
                    "semantic_filter_strength": strength,
                    "semantic_filter_boundary_only": True,
                    "semantic_filter_boundary_candidate_points": int(candidate_finite_indices.size),
                    "semantic_filter_outside_candidate_points": int(candidate_info["outside_mask"].sum()),
                })

            # 保存当前实例的滤波统计
            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": after_n,
                "removed": removed,
                "skipped": False,
                "boundary_only": True,
                "boundary_candidate_points": int(candidate_finite_indices.size),
                "outside_candidate_points": int(candidate_info["outside_mask"].sum()),
                "max_outside_distance": float(outside_distance.max()) if outside_distance.size else 0.0,
            })

            # boundary_only 分支处理完成，进入下一个实例
            continue

        # 如果 boundary_only=False，则对整个实例做半径邻域和连通域过滤

        # 为当前实例有限点建立 KDTree
        tree = cKDTree(finite_points)

        # 查询每个点在 radius 半径内的邻居
        finite_neighbor_lists = tree.query_ball_point(finite_points, r=radius)

        # 统计邻居数量
        finite_neighbor_counts = np.array(
            [len(x) for x in finite_neighbor_lists],
            dtype=np.int32
        )

        # 邻居数达到阈值的点认为是稠密点
        dense_finite_mask = finite_neighbor_counts >= min_neighbors

        # 将 finite_points 的稠密点 mask 映射回 obj_points 的局部索引
        dense_local_mask = np.zeros(raw_n, dtype=bool)
        dense_local_mask[finite_local_indices[dense_finite_mask]] = True

        # 构建 raw_n 长度的邻接表
        # 邻接索引统一使用 obj_points 的局部索引
        neighbor_lists = [[] for _ in range(raw_n)]
        finite_to_raw = finite_local_indices

        for finite_i, raw_i in enumerate(finite_to_raw):
            neighbor_lists[int(raw_i)] = [
                int(finite_to_raw[j]) for j in finite_neighbor_lists[finite_i]
            ]

        # 根据稠密点 mask 和邻接表计算连通域
        components = _connected_components_from_neighbor_lists(
            neighbor_lists=neighbor_lists,
            valid_local_mask=dense_local_mask
        )

        # 如果没有任何连通域，说明过滤条件可能过严
        # 为避免误删整个物体，这里保留原实例
        if len(components) == 0:
            keep_local = np.ones(raw_n, dtype=bool)
        else:
            if keep_largest_only:
                # 只保留最大连通域
                keep_components = [components[0]]
            else:
                # 根据固定点数阈值和比例阈值共同决定小连通域是否保留
                threshold = max(
                    min_component_points,
                    int(round(raw_n * min_component_ratio))
                )

                # 保留点数足够的连通域
                keep_components = [
                    comp for comp in components if comp.size >= threshold
                ]

                # 如果阈值过严导致没有连通域被保留，
                # 至少保留最大主体，避免整物体被删光
                if len(keep_components) == 0:
                    keep_components = [components[0]]

            # 构造当前实例局部保留 mask
            keep_local = np.zeros(raw_n, dtype=bool)

            # 将保留连通域内的点标记为 True
            for comp in keep_components:
                keep_local[comp] = True

        # 映射回全局点云索引
        keep_global = obj_indices[keep_local]
        remove_global = obj_indices[~keep_local]

        # 删除点重置为背景和无实例
        if remove_global.size > 0:
            point_class_ids[remove_global] = background_class_id
            point_instance_ids[remove_global] = 0

        # 当前实例删除数量
        removed = int(remove_global.size)

        # 当前实例滤波后点数
        after_n = int(keep_global.size)

        # 累加删除数量
        removed_total += removed

        # 如果当前实例仍然有点，则保留更新后的 segment
        if after_n > 0:
            filtered_segments.append({
                **seg,
                "point_indices": keep_global,
                "num_points": after_n,
                "raw_num_points_before_semantic_filter": raw_n,
                "semantic_filter_removed_points": removed,
                "semantic_filter_strength": strength,
            })

        # 保存当前实例统计信息
        segment_stats.append({
            "instance_id": instance_id,
            "class_id": class_id,
            "before": raw_n,
            "after": after_n,
            "removed": removed,
            "skipped": False,
        })

    # 滤波后仍然带有语义标签的点数量
    after_labeled = int((point_class_ids >= 0).sum())

    # 返回滤波后的标签、segment 和统计结果
    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "segments": filtered_segments,
        "filter_stats": {
            "semantic_filter_strength": strength,
            "enabled": True,

            # 当前滤波参数
            "radius": radius,
            "min_neighbors": min_neighbors,
            "min_component_points": min_component_points,
            "min_component_ratio": min_component_ratio,
            "keep_largest_only": keep_largest_only,
            "boundary_only": boundary_only,
            "boundary_percentile": boundary_percentile,
            "robust_box_lower_percentile": robust_box_lower_percentile,
            "robust_box_upper_percentile": robust_box_upper_percentile,
            "robust_box_core_percentile": robust_box_core_percentile,
            "outside_hard_remove_radius": outside_hard_remove_radius,
            "outside_hard_remove_box_ratio": outside_hard_remove_box_ratio,
            "outside_neighbor_boost": outside_neighbor_boost,

            # 总体滤波统计
            "before_labeled_points": before_labeled,
            "after_labeled_points": after_labeled,
            "removed_labeled_points": int(before_labeled - after_labeled),
            "removed_ratio": float(
                (before_labeled - after_labeled) / max(before_labeled, 1)
            ),

            # 每个实例的详细统计
            "segment_stats": segment_stats,
        }
    }