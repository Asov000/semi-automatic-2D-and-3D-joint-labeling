# -*- coding: utf-8 -*-
"""点云投影模块，负责将 SUNRGBD 风格三维点投影回二维图像。"""

from typing import Optional, Tuple

import numpy as np


def project_points_to_image_with_indices(
    points3d: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int, int],
    Rtilt: Optional[np.ndarray] = None,
    points_are_after_rtilt: bool = True,
    use_matlab_pixel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将 SUNRGBD 风格的 3D 点云投影到 2D 图像平面，并返回原始点索引。

    参数：
    points3d:
        输入点云，形状为 N x 3。
        通常是 SUNRGBD 中经过 Rtilt 后的点云坐标。

    K:
        相机内参矩阵，形状为 3 x 3。
        格式一般为：
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    image_shape:
        图像尺寸，一般直接传 image.shape。
        格式为 H x W x C。

    Rtilt:
        SUNRGBD 中的倾斜校正矩阵。
        如果 points_are_after_rtilt=True，则必须传入 Rtilt，
        用于将点云从校正后的坐标系还原到原始相机坐标系。

    points_are_after_rtilt:
        表示输入点云是否已经经过 Rtilt 校正。
        True:
            输入点云已经经过 Rtilt，需要乘以 inv(Rtilt) 还原。
        False:
            输入点云已经处于可投影的原始相机坐标系。

    use_matlab_pixel:
        是否使用 MATLAB 像素坐标修正。
        SUNRGBD 原始工具箱很多坐标来自 MATLAB，
        MATLAB 像素坐标从 1 开始，而 Python / OpenCV 从 0 开始，
        所以需要减 1。

    返回：
    uv_int:
        投影到图像上的整数像素坐标，形状为 M x 2。
        每一行为 [u, v]，也就是 [x, y]。

    depth:
        每个有效投影点的深度 Z，形状为 M。

    point_indices:
        每个有效投影点在原始 points3d 中的索引，形状为 M。
        这个非常重要，可以用于从 2D mask 反查对应的 3D 点。

    uv_float:
        未取整前的浮点像素坐标，形状为 M x 2。
        可用于更精细的投影调试。
    """

    # 获取图像高度 H 和宽度 W
    H, W = image_shape[:2]

    # 将输入点云转为 float64，并保证形状为 N x 3
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    # 过滤 NaN / inf 点，避免后续矩阵运算和投影出错
    finite_mask = np.isfinite(points3d).all(axis=1)

    # 保存有限点在原始点云中的索引
    # 后续可以通过这些索引映射回原始点云
    finite_indices = np.where(finite_mask)[0]

    # 只保留有效点
    points = points3d[finite_mask]

    # 如果输入点云已经经过 Rtilt 校正，
    # 需要乘以 inv(Rtilt) 还原到原始相机坐标系再投影
    if points_are_after_rtilt:
        if Rtilt is None:
            raise ValueError("points_are_after_rtilt=True 时必须传入 Rtilt")

        # 转为 float64，保证矩阵运算稳定
        Rtilt = np.asarray(Rtilt, dtype=np.float64)

        # 将点云从 Rtilt 校正坐标系还原到原始相机坐标系
        points_before_rtilt = (np.linalg.inv(Rtilt) @ points.T).T

    # 如果输入点云本来就在原始相机坐标系，则直接使用
    else:
        points_before_rtilt = points

    # SUNRGBD read_3d_pts_general 中的坐标定义：
    # points3d = [x3, z3, -y3]
    #
    # 因此投影时需要转成相机坐标：
    # X_cam = x3
    # Y_cam = y3 = -points[:, 2]
    # Z_cam = z3
    #
    # 注意：
    # Z 是深度方向，必须大于 0 才能投影到相机前方。
    X = points_before_rtilt[:, 0]
    Z = points_before_rtilt[:, 1]
    Y = -points_before_rtilt[:, 2]

    # 只保留深度为正的点
    # Z <= 0 的点在相机后方或无效，不能投影
    valid_z = Z > 1e-6

    # 根据深度有效 mask 过滤坐标
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]

    # 同步过滤原始点索引
    # valid_indices 表示：
    # 经过 finite 过滤和深度过滤后，剩余点在原始 points3d 中的索引
    valid_indices = finite_indices[valid_z]

    # 从相机内参矩阵中取出 fx、fy、cx、cy
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 使用针孔相机模型进行 3D 到 2D 投影
    #
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy
    #
    # 其中：
    # u 是图像横坐标
    # v 是图像纵坐标
    # Z 是深度
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # 如果使用 MATLAB 坐标修正，则将 1-based 像素坐标转成 0-based
    if use_matlab_pixel:
        u = u - 1
        v = v - 1

    # 保存未取整的浮点像素坐标
    # 适合做精确投影调试
    uv_float = np.stack([u, v], axis=1)

    # 将浮点像素坐标四舍五入为整数像素坐标
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    # 过滤掉投影到图像范围外的点
    inside = (
        (u_int >= 0) & (u_int < W) &
        (v_int >= 0) & (v_int < H)
    )

    # 最终有效整数像素坐标
    uv_int = np.stack([u_int[inside], v_int[inside]], axis=1)

    # 最终有效点的深度
    depth = Z[inside]

    # 最终有效点在原始 points3d 中的索引
    point_indices = valid_indices[inside]

    # 最终有效点的浮点像素坐标
    uv_float = uv_float[inside]

    return uv_int, depth, point_indices, uv_float


def filter_visible_points_by_zbuffer(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, int, int],
    tolerance: float = 0.03
) -> np.ndarray:
    """
    使用 Z-buffer 过滤被遮挡的投影点。

    作用：
    多个 3D 点可能投影到同一个 2D 像素上。
    真实图像中只能看到最靠近相机的那个点。
    因此需要用 Z-buffer 保留每个像素位置上深度最小的点。

    参数：
    uv:
        投影后的整数像素坐标，形状为 N x 2。
        每一行为 [u, v]。

    depth:
        每个投影点对应的深度，形状为 N。

    image_shape:
        图像尺寸，一般传 image.shape。

    tolerance:
        深度容差。
        如果某个点的深度不超过该像素最小深度 + tolerance，
        就认为它仍然可见。
        这样可以避免因为点云噪声导致过度过滤。

    返回：
    visible_mask:
        布尔数组，形状为 N。
        True 表示该点在图像视角下可见；
        False 表示该点大概率被前方点遮挡。
    """

    # 获取图像高度和宽度
    H, W = image_shape[:2]

    # 转成 numpy 数组
    uv = np.asarray(uv)
    depth = np.asarray(depth)

    # 拆分像素坐标
    u = uv[:, 0]
    v = uv[:, 1]

    # 将二维像素坐标转换成一维索引
    #
    # 对于图像中像素 (u, v)，其一维索引为：
    # linear_idx = v * W + u
    #
    # 这样可以用一维数组 zbuffer 表示整张图的深度缓冲。
    linear_idx = v * W + u

    # 初始化 Z-buffer
    # 每个像素位置初始深度为无穷大
    zbuffer = np.full(H * W, np.inf, dtype=np.float64)

    # 对每个像素位置，保存投影到该像素上的最小深度
    #
    # np.minimum.at 的作用：
    # 如果多个点对应同一个 linear_idx，
    # 就把该位置更新为这些 depth 中的最小值。
    np.minimum.at(zbuffer, linear_idx, depth)

    # 查询每个点所在像素位置上的最小深度
    min_depth_at_point_pixel = zbuffer[linear_idx]

    # 判断每个点是否可见
    #
    # 如果当前点深度接近该像素的最小深度，
    # 说明它是最靠前的点，或者和最靠前点几乎在同一表面上。
    #
    # 如果当前点深度明显大于最小深度，
    # 说明它被前方点遮挡。
    visible_mask = depth <= min_depth_at_point_pixel + tolerance

    return visible_mask