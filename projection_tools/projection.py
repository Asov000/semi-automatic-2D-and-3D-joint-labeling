# -*- coding: utf-8 -*-
"""点云投影调试模块，负责把样本点云投影到图像平面。"""

import numpy as np


def project_sunrgbd_points_to_image(
    points3d,
    K,
    image_shape,
    Rtilt=None,
    points_are_after_rtilt=True,
    matlab_pixel_index=True
):
    """
    将 SUNRGBD 点云投影到 2D 图像平面。

    参数：
    points3d: 输入点云，形状为 N×3
    K: 相机内参矩阵，3×3
    image_shape: 图像尺寸，一般为 image.shape
    Rtilt: SUNRGBD 中的倾斜校正矩阵
    points_are_after_rtilt: 点云是否已经经过 Rtilt 校正
    matlab_pixel_index: 是否按照 MATLAB 的 1-based 像素坐标转成 Python 的 0-based 坐标

    返回：
    uv: 投影到图像上的像素坐标，N×2
    depth: 每个有效投影点的深度
    valid_points: 最终有效的 3D 点
    """

    # 获取图像高度和宽度
    H, W = image_shape[:2]

    # 转成 numpy 数组，并保证形状为 N×3
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    # 去除包含 NaN 或 inf 的非法点
    valid = np.isfinite(points3d).all(axis=1)
    points = points3d[valid]

    # 如果点云已经经过 Rtilt 校正，则需要乘以 Rtilt 的逆矩阵还原到原始相机坐标系
    if points_are_after_rtilt:
        if Rtilt is None:
            raise ValueError("points3d 已经经过 Rtilt 时，必须传入 Rtilt。")
        Rtilt = np.asarray(Rtilt, dtype=np.float64)
        points = (np.linalg.inv(Rtilt) @ points.T).T

    # SUNRGBD 坐标系到图像相机坐标系的转换
    # X 对应水平轴，Z 作为深度，Y 取反作为图像竖直方向
    X = points[:, 0]
    Z = points[:, 1]
    Y = -points[:, 2]

    # 去掉深度小于等于 0 的点，因为这些点无法投影到相机前方
    valid_z = Z > 1e-6
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]
    valid_points = points[valid_z]

    # 读取相机内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 使用针孔相机模型进行 3D 到 2D 投影
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # 如果原始标定来自 MATLAB，需要从 1-based 坐标转为 Python/OpenCV 常用的 0-based 坐标
    if matlab_pixel_index:
        u = u - 1
        v = v - 1

    # 将连续像素坐标四舍五入为整数像素
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # 只保留落在图像范围内的点
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 保存有效的 2D 像素坐标、深度和对应的 3D 点
    uv = np.stack([u[inside], v[inside]], axis=1)
    depth = Z[inside]
    valid_points = valid_points[inside]

    return uv, depth, valid_points


def project_points_to_image(points3d_after_rtilt, K, Rtilt, image_shape, use_matlab_pixel=True):
    """
    将已经经过 Rtilt 校正后的点云投影到图像平面。

    与上一个函数类似，但该函数额外返回 valid_mask，
    用于标记原始输入点云中哪些点成功投影到了图像内。
    """

    # 获取图像高度和宽度
    H, W = image_shape[:2]

    # 转成 numpy 数组
    points3d_after_rtilt = np.asarray(points3d_after_rtilt, dtype=np.float64)

    # 去除 NaN / inf 点
    finite_mask = np.isfinite(points3d_after_rtilt).all(axis=1)
    points = points3d_after_rtilt[finite_mask]

    # 将经过 Rtilt 的点云还原到原始相机坐标系
    points_before_rtilt = (np.linalg.inv(Rtilt) @ points.T).T

    # 坐标系转换
    X = points_before_rtilt[:, 0]
    Z = points_before_rtilt[:, 1]
    Y = -points_before_rtilt[:, 2]

    # 只保留相机前方的点
    valid_z = Z > 1e-6
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]

    # 读取相机内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 通过针孔相机模型计算像素坐标
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # MATLAB 像素坐标转 Python/OpenCV 像素坐标
    if use_matlab_pixel:
        u = u - 1
        v = v - 1

    # 四舍五入为整数像素
    u_round = np.round(u).astype(np.int32)
    v_round = np.round(v).astype(np.int32)

    # 判断投影点是否在图像范围内
    inside = (
        (u_round >= 0) & (u_round < W) &
        (v_round >= 0) & (v_round < H)
    )

    # 得到最终有效的像素坐标和深度
    uv = np.stack([u_round[inside], v_round[inside]], axis=1)
    depth = Z[inside]

    # 构造与原始输入点云长度一致的布尔 mask
    # True 表示该点最终成功投影到了图像内
    valid_mask = np.zeros(points3d_after_rtilt.shape[0], dtype=bool)

    # 找回经过 finite 过滤后的原始索引
    finite_indices = np.where(finite_mask)[0]

    # 找回深度有效点对应的原始索引
    valid_z_indices = finite_indices[valid_z]

    # 只把最终落在图像内的点标记为 True
    valid_mask[valid_z_indices[inside]] = True

    return uv, depth, valid_mask