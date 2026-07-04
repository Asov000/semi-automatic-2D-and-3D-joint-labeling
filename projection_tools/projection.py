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


    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    H, W = image_shape[:2]

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    # 去除 NaN / inf 点
    valid = np.isfinite(points3d).all(axis=1)
    points = points3d[valid]

    if points_are_after_rtilt:
        if Rtilt is None:
            raise ValueError("points3d 已经经过 Rtilt 时，必须传入 Rtilt。")
        Rtilt = np.asarray(Rtilt, dtype=np.float64)
        points = (np.linalg.inv(Rtilt) @ points.T).T

    X = points[:, 0]
    Z = points[:, 1]
    Y = -points[:, 2]

    # 去掉深度非法点
    valid_z = Z > 1e-6
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]
    valid_points = points[valid_z]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    if matlab_pixel_index:
        u = u - 1
        v = v - 1

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # 只保留落在图像范围内的点
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    uv = np.stack([u[inside], v[inside]], axis=1)
    depth = Z[inside]
    valid_points = valid_points[inside]

    return uv, depth, valid_points


def project_points_to_image(points3d_after_rtilt, K, Rtilt, image_shape, use_matlab_pixel=True):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    H, W = image_shape[:2]

    points3d_after_rtilt = np.asarray(points3d_after_rtilt, dtype=np.float64)
    finite_mask = np.isfinite(points3d_after_rtilt).all(axis=1)
    points = points3d_after_rtilt[finite_mask]
    points_before_rtilt = (np.linalg.inv(Rtilt) @ points.T).T
    X = points_before_rtilt[:, 0]
    Z = points_before_rtilt[:, 1]
    Y = -points_before_rtilt[:, 2]
    valid_z = Z > 1e-6
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    if use_matlab_pixel:
        u = u - 1
        v = v - 1

    u_round = np.round(u).astype(np.int32)
    v_round = np.round(v).astype(np.int32)

    inside = (
        (u_round >= 0) & (u_round < W) &
        (v_round >= 0) & (v_round < H)
    )

    uv = np.stack([u_round[inside], v_round[inside]], axis=1)
    depth = Z[inside]

    # 构造回原始输入点云长度的 mask，方便你后续调试
    valid_mask = np.zeros(points3d_after_rtilt.shape[0], dtype=bool)
    finite_indices = np.where(finite_mask)[0]
    valid_z_indices = finite_indices[valid_z]
    valid_mask[valid_z_indices[inside]] = True

    return uv, depth, valid_mask
