# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import scipy.io as sio
import open3d as o3d


def load_mat_points(mat_path):

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"点云文件不存在: {mat_path}")

    try:
        data = sio.loadmat(mat_path)

        if "points3d_rgb" in data:
            points3d_rgb = data["points3d_rgb"]
            return np.asarray(points3d_rgb, dtype=np.float64)

        # 自动查找 N x 6 数组
        for key, value in data.items():
            if key.startswith("__"):
                continue

            arr = np.asarray(value)

            if arr.ndim == 2 and arr.shape[1] >= 6:
                print(f"自动使用变量: {key}, shape={arr.shape}")
                return arr[:, :6].astype(np.float64)

        raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")

    except NotImplementedError:
        # v7.3 mat 文件
        with h5py.File(mat_path, "r") as f:
            if "points3d_rgb" in f:
                arr = np.array(f["points3d_rgb"])

                # h5py 读出来有时是 6 x N
                if arr.shape[0] == 6:
                    arr = arr.T

                return arr.astype(np.float64)

            for key in f.keys():
                arr = np.array(f[key])

                if arr.ndim == 2:
                    if arr.shape[0] == 6:
                        arr = arr.T

                    if arr.shape[1] >= 6:
                        print(f"自动使用变量: {key}, shape={arr.shape}")
                        return arr[:, :6].astype(np.float64)

            raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")


def view_raw_pointcloud(mat_path, use_rgb=True, point_size=2.0):

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


if __name__ == "__main__":
    MAT_PATH = r"D:\frustum-convnet\sunrgbd\mysunrgbd\training\pc\000001.mat"

    view_raw_pointcloud(
        mat_path=MAT_PATH,
        use_rgb=True,
        point_size=2.0
    )