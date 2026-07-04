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
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    H, W = image_shape[:2]

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    finite_mask = np.isfinite(points3d).all(axis=1)
    finite_indices = np.where(finite_mask)[0]
    points = points3d[finite_mask]

    if points_are_after_rtilt:
        if Rtilt is None:
            raise ValueError("points_are_after_rtilt=True 时必须传入 Rtilt")

        Rtilt = np.asarray(Rtilt, dtype=np.float64)
        points_before_rtilt = (np.linalg.inv(Rtilt) @ points.T).T
    else:
        points_before_rtilt = points

    # SUNRGBD read_3d_pts_general 里的坐标定义：
    # points3d = [x3, z3, -y3]
    # 因此投影时：
    # X_cam = x3
    # Y_cam = y3 = -points[:, 2]
    # Z_cam = z3
    X = points_before_rtilt[:, 0]
    Z = points_before_rtilt[:, 1]
    Y = -points_before_rtilt[:, 2]

    valid_z = Z > 1e-6

    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]

    valid_indices = finite_indices[valid_z]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    if use_matlab_pixel:
        u = u - 1
        v = v - 1

    uv_float = np.stack([u, v], axis=1)

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    inside = (
        (u_int >= 0) & (u_int < W) &
        (v_int >= 0) & (v_int < H)
    )

    uv_int = np.stack([u_int[inside], v_int[inside]], axis=1)
    depth = Z[inside]
    point_indices = valid_indices[inside]
    uv_float = uv_float[inside]

    return uv_int, depth, point_indices, uv_float


def filter_visible_points_by_zbuffer(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, int, int],
    tolerance: float = 0.03
) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    H, W = image_shape[:2]

    uv = np.asarray(uv)
    depth = np.asarray(depth)

    u = uv[:, 0]
    v = uv[:, 1]

    linear_idx = v * W + u

    zbuffer = np.full(H * W, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, linear_idx, depth)

    min_depth_at_point_pixel = zbuffer[linear_idx]

    visible_mask = depth <= min_depth_at_point_pixel + tolerance

    return visible_mask
