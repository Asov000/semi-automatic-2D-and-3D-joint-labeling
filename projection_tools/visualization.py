# -*- coding: utf-8 -*-
"""投影可视化模块，负责绘制投影图、对比图和原始点云视图。"""

import cv2
import numpy as np
import open3d as o3d

from tool3d_modules.io import load_mat_points


def draw_projected_points_on_image(image, uv, depth=None, point_size=1):
    """
    将投影后的 2D 点绘制到原始图像上。

    参数：
    image: 原始图像，OpenCV 读取格式，BGR
    uv: 投影后的像素坐标，形状为 N x 2，每一行为 [u, v]
    depth: 每个点对应的深度，可选；如果提供，则根据深度上色
    point_size: 绘制点的半径

    返回：
    vis: 绘制了投影点的图像
    """

    # 复制图像，避免直接修改原图
    vis = image.copy()

    # 转换 uv 为 numpy 数组
    uv = np.asarray(uv)

    # 如果没有投影点，直接返回原图
    if uv.size == 0:
        return vis

    # 确保 uv 的形状是 N x 2
    uv = uv.reshape(-1, 2)

    # 没有传入深度时，统一用红色画点
    if depth is None:
        for u, v in uv:
            cv2.circle(
                vis,
                (int(u), int(v)),
                point_size,
                (0, 0, 255),   # OpenCV 中是 BGR，这里表示红色
                -1
            )
        return vis

    # 将 depth 转为 numpy 数组
    depth = np.asarray(depth).reshape(-1)

    # 检查 uv 和 depth 数量是否一致
    if len(depth) != len(uv):
        raise ValueError(f"uv 和 depth 数量不一致: len(uv)={len(uv)}, len(depth)={len(depth)}")

    # 只保留有效深度
    finite_mask = np.isfinite(depth)
    uv = uv[finite_mask]
    depth = depth[finite_mask]

    if len(depth) == 0:
        return vis

    # 使用 2% 和 98% 分位数做深度归一化，减少极端值影响
    d_min = np.nanpercentile(depth, 2)
    d_max = np.nanpercentile(depth, 98)

    # 防止所有深度几乎一样时除零
    if abs(d_max - d_min) < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.float64)
    else:
        depth_norm = (depth - d_min) / (d_max - d_min)
        depth_norm = np.clip(depth_norm, 0, 1)

    # 转成 0~255 的 uint8，用于 OpenCV 伪彩色映射
    depth_color = (depth_norm * 255).astype(np.uint8)

    # 使用 JET 颜色映射显示深度
    colors = cv2.applyColorMap(depth_color.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = colors.reshape(-1, 3)

    # 逐点绘制
    for (u, v), color in zip(uv, colors):
        color = tuple(int(c) for c in color)
        cv2.circle(
            vis,
            (int(u), int(v)),
            point_size,
            color,
            -1
        )

    return vis


def make_side_by_side(original, projected):
    """
    将原图和投影图横向拼接，方便对比。

    参数：
    original: 原始图像
    projected: 投影可视化图像

    返回：
    拼接后的图像
    """

    # 如果两张图大小不一致，则将 projected 缩放到 original 的大小
    if original.shape != projected.shape:
        projected = cv2.resize(
            projected,
            (original.shape[1], original.shape[0])
        )

    # axis=1 表示横向拼接
    return np.concatenate([original, projected], axis=1)


def draw_projection(image, uv, depth=None, point_size=1, alpha=0.7):
    """
    将投影点绘制到图像上，并使用透明度叠加。

    参数：
    image: 原始图像，BGR 格式
    uv: 投影后的像素坐标，N x 2
    depth: 每个点的深度；如果为 None，则统一红色绘制
    point_size: 点的半径
    alpha: 投影层透明度，越大投影点越明显

    返回：
    vis: 叠加投影点后的图像
    """

    # 原图副本
    vis = image.copy()

    # overlay 是绘制层，最后与原图按 alpha 混合
    overlay = image.copy()

    uv = np.asarray(uv)

    if uv.size == 0:
        return vis

    uv = uv.reshape(-1, 2)

    # 没有 depth 时，统一画红色点
    if depth is None:
        for u, v in uv:
            cv2.circle(
                overlay,
                (int(u), int(v)),
                point_size,
                (0, 0, 255),
                -1
            )

    else:
        depth = np.asarray(depth).reshape(-1)

        # 检查投影点数量和深度数量是否一致
        if len(depth) != len(uv):
            raise ValueError(f"uv 和 depth 数量不一致: len(uv)={len(uv)}, len(depth)={len(depth)}")

        # 去除无效深度
        finite_mask = np.isfinite(depth)
        uv = uv[finite_mask]
        depth = depth[finite_mask]

        if len(depth) == 0:
            return vis

        # 使用分位数归一化深度，避免少数异常深度影响颜色范围
        d_min = np.percentile(depth, 2)
        d_max = np.percentile(depth, 98)

        if abs(d_max - d_min) < 1e-6:
            depth_norm = np.zeros_like(depth, dtype=np.float64)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min)
            depth_norm = np.clip(depth_norm, 0, 1)

        # 转为 8 位灰度，再映射成伪彩色
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        colors = cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colors.reshape(-1, 3)

        # 绘制深度彩色点
        for (u, v), color in zip(uv, colors):
            color = tuple(int(c) for c in color)
            cv2.circle(
                overlay,
                (int(u), int(v)),
                point_size,
                color,
                -1
            )

    # 将 overlay 和原图按透明度混合
    vis = cv2.addWeighted(
        overlay,
        alpha,
        vis,
        1 - alpha,
        0
    )

    return vis


def create_projection_image_from_point_rgb(image_shape, uv, point_rgb):
    """
    根据点云自带的 RGB 颜色生成投影重建图。

    参数：
    image_shape: 原图尺寸，一般为 image.shape
    uv: 投影后的像素坐标，N x 2
    point_rgb: 每个投影点对应的 RGB 颜色，N x 3

    返回：
    canvas: 根据点云 RGB 生成的投影图像，BGR 格式
    """

    H, W = image_shape[:2]

    # 创建黑色画布
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    uv = np.asarray(uv).reshape(-1, 2)
    rgb = np.asarray(point_rgb)

    if uv.size == 0 or rgb.size == 0:
        return canvas

    rgb = rgb.reshape(-1, 3)

    # 检查 uv 和 rgb 数量是否一致
    if len(uv) != len(rgb):
        raise ValueError(f"uv 和 point_rgb 数量不一致: len(uv)={len(uv)}, len(point_rgb)={len(rgb)}")

    # 如果 RGB 是 0~1 范围，则转换到 0~255
    if np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0

    # 限制颜色范围，并转成 uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # OpenCV 图像使用 BGR，因此这里把 RGB 转成 BGR
    bgr = rgb[:, ::-1]

    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)

    # 防止 uv 越界导致数组索引报错
    inside = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    u = u[inside]
    v = v[inside]
    bgr = bgr[inside]

    # 将每个点的颜色写入对应像素
    # 注意：如果多个 3D 点投影到同一个像素，后写入的点会覆盖前面的点
    canvas[v, u] = bgr

    return canvas


def view_raw_pointcloud(mat_path, use_rgb=True, point_size=2.0):
    """
    使用 Open3D 查看原始点云。

    参数：
    mat_path: 点云 .mat 文件路径
    use_rgb: 是否使用点云自带 RGB 颜色
    point_size: Open3D 显示时的点大小

    返回：
    无返回值，直接打开 Open3D 窗口显示点云
    """

    # 读取 .mat 点云文件，一般包含 x, y, z, r, g, b
    points3d_rgb = load_mat_points(mat_path)

    # 至少需要 xyz 三列
    if points3d_rgb.shape[1] < 3:
        raise ValueError(f"点云维度不对，至少需要 N x 3，当前 shape={points3d_rgb.shape}")

    # 取前三列作为点云坐标
    points = points3d_rgb[:, 0:3].astype(np.float64)

    # 去除 NaN / inf 点
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    if points.shape[0] == 0:
        raise ValueError("点云中没有有效的 xyz 点。")

    # 打印点云基本信息，方便调试
    print("点云文件:", mat_path)
    print("原始点数量:", points3d_rgb.shape[0])
    print("有效点数量:", points.shape[0])
    print("XYZ min:", points.min(axis=0))
    print("XYZ max:", points.max(axis=0))

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点云坐标
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果点云包含 RGB 信息，则使用原始颜色
    if use_rgb and points3d_rgb.shape[1] >= 6:
        colors = points3d_rgb[:, 3:6].astype(np.float64)
        colors = colors[valid]

        # 如果颜色是 0~255，则归一化到 0~1
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0

        colors = np.clip(colors, 0.0, 1.0)

        # Open3D 使用 RGB 顺序，不需要转成 BGR
        pcd.colors = o3d.utility.Vector3dVector(colors)

    else:
        # 如果没有 RGB，则统一设置为灰色
        colors = np.ones_like(points) * 0.6
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建坐标轴，方便判断点云方向
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.8,
        origin=[0, 0, 0]
    )

    # 创建 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Raw Point Cloud Viewer",
        width=1280,
        height=900
    )

    # 加入点云和坐标轴
    vis.add_geometry(pcd)
    vis.add_geometry(axis)

    # 设置渲染参数
    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.02, 0.02, 0.02])

    # 启动窗口
    vis.run()

    # 关闭窗口并释放资源
    vis.destroy_window()