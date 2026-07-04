# -*- coding: utf-8 -*-
"""投影可视化模块，负责绘制投影图、对比图和原始点云视图。"""

import cv2
import numpy as np

from tool3d_modules.io import load_mat_points


def draw_projected_points_on_image(image, uv, depth=None, point_size=1):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    vis = image.copy()

    if depth is None:
        for u, v in uv:
            cv2.circle(vis, (int(u), int(v)), point_size, (0, 0, 255), -1)
        return vis

    depth = np.asarray(depth)
    depth_norm = depth.copy()

    # 深度归一化到 0~255
    d_min = np.nanpercentile(depth_norm, 2)
    d_max = np.nanpercentile(depth_norm, 98)
    depth_norm = np.clip((depth_norm - d_min) / (d_max - d_min + 1e-6), 0, 1)
    depth_color = (depth_norm * 255).astype(np.uint8)

    # 近处和远处用不同颜色显示
    colors = cv2.applyColorMap(depth_color.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = colors.reshape(-1, 3)

    for (u, v), color in zip(uv, colors):
        color = tuple(int(c) for c in color)
        cv2.circle(vis, (int(u), int(v)), point_size, color, -1)

    return vis


def make_side_by_side(original, projected):
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if original.shape != projected.shape:
        projected = cv2.resize(projected, (original.shape[1], original.shape[0]))

    return np.concatenate([original, projected], axis=1)


def draw_projection(image, uv, depth=None, point_size=1, alpha=0.7):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    vis = image.copy()
    overlay = image.copy()

    if depth is None:
        for u, v in uv:
            cv2.circle(overlay, (int(u), int(v)), point_size, (0, 0, 255), -1)
    else:
        depth = np.asarray(depth)

        d_min = np.percentile(depth, 2)
        d_max = np.percentile(depth, 98)

        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1)

        depth_u8 = (depth_norm * 255).astype(np.uint8)
        colors = cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colors.reshape(-1, 3)

        for (u, v), color in zip(uv, colors):
            color = tuple(int(c) for c in color)
            cv2.circle(overlay, (int(u), int(v)), point_size, color, -1)

    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def create_projection_image_from_point_rgb(image_shape, uv, point_rgb):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    H, W = image_shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    rgb = np.asarray(point_rgb)

    if rgb.max() <= 1.0:
        rgb = (rgb * 255.0).clip(0, 255)

    rgb = rgb.astype(np.uint8)

    bgr = rgb[:, ::-1]

    u = uv[:, 0]
    v = uv[:, 1]

    canvas[v, u] = bgr

    return canvas


def view_raw_pointcloud(mat_path, use_rgb=True, point_size=2.0):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    points3d_rgb = load_mat_points(mat_path)

    if points3d_rgb.shape[1] < 3:
        raise ValueError(f"点云维度不对，至少需要 N x 3，当前 shape={points3d_rgb.shape}")

    points = points3d_rgb[:, 0:3].astype(np.float64)

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    print("点云文件:", mat_path)
    print("原始点数量:", points3d_rgb.shape[0])
    print("有效点数量:", points.shape[0])
    print("XYZ min:", points.min(axis=0))
    print("XYZ max:", points.max(axis=0))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if use_rgb and points3d_rgb.shape[1] >= 6:
        colors = points3d_rgb[:, 3:6].astype(np.float64)
        colors = colors[valid]

        if colors.max() > 1.0:
            colors = colors / 255.0

        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        colors = np.ones_like(points) * 0.6
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.8,
        origin=[0, 0, 0]
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Raw Point Cloud Viewer",
        width=1280,
        height=900
    )

    vis.add_geometry(pcd)
    vis.add_geometry(axis)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()
