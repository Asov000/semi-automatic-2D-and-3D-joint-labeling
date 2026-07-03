import cv2
import numpy as np


def project_sunrgbd_points_to_image(
    points3d,
    K,
    image_shape,
    Rtilt=None,
    points_are_after_rtilt=True,
    matlab_pixel_index=True
):


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


def draw_projected_points_on_image(image, uv, depth=None, point_size=1):

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
    if original.shape != projected.shape:
        projected = cv2.resize(projected, (original.shape[1], original.shape[0]))

    return np.concatenate([original, projected], axis=1)


import os
import cv2
import h5py
import numpy as np
import scipy.io as sio


def load_mat_points(mat_path):


    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"点云文件不存在: {mat_path}")

    try:
        data = sio.loadmat(mat_path)
        if "points3d_rgb" in data:
            points3d_rgb = data["points3d_rgb"]
            return np.asarray(points3d_rgb, dtype=np.float64)
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                print(f"自动使用变量: {key}, shape={arr.shape}")
                return arr[:, :6].astype(np.float64)

        raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")

    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            if "points3d_rgb" in f:
                arr = np.array(f["points3d_rgb"])

                # h5py 读出来有时是转置的
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


def load_calib(calib_path):

    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    calib = np.loadtxt(calib_path)

    if calib.shape != (2, 9):
        raise ValueError(f"calib 文件格式不对，期望 shape=(2,9)，实际是 {calib.shape}")

    Rtilt_flat = calib[0]
    K_flat = calib[1]

    Rtilt = Rtilt_flat.reshape(3, 3, order="F")
    K = K_flat.reshape(3, 3, order="F")

    return Rtilt, K


def project_points_to_image(points3d_after_rtilt, K, Rtilt, image_shape, use_matlab_pixel=True):

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


def draw_projection(image, uv, depth=None, point_size=1, alpha=0.7):

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


def test_one_sample(root_dir, image_id=1, save_result=True, show=True):

    sample_name = f"{image_id:06d}"

    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    image_path = os.path.join(root_dir, "image", sample_name + ".jpg")
    calib_path = os.path.join(root_dir, "calib", sample_name + ".txt")

    print("读取点云:", pc_path)
    print("读取图像:", image_path)
    print("读取标定:", calib_path)

    points3d_rgb = load_mat_points(pc_path)
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"图像读取失败: {image_path}")

    Rtilt, K = load_calib(calib_path)

    points3d = points3d_rgb[:, 0:3]
    point_rgb = points3d_rgb[:, 3:6]

    uv, depth, valid_mask = project_points_to_image(
        points3d_after_rtilt=points3d,
        K=K,
        Rtilt=Rtilt,
        image_shape=image.shape,
        use_matlab_pixel=True
    )

    print("原始点云数量:", points3d.shape[0])
    print("有效投影点数量:", uv.shape[0])
    print("图像尺寸:", image.shape)
    print("K =\n", K)
    print("Rtilt =\n", Rtilt)

    # 深度颜色投影图
    projected_depth_vis = draw_projection(
        image=image,
        uv=uv,
        depth=depth,
        point_size=1,
        alpha=0.75
    )

    # 用点云里保存的 RGB 重建图像
    projected_rgb = create_projection_image_from_point_rgb(
        image_shape=image.shape,
        uv=uv,
        point_rgb=point_rgb[valid_mask]
    )

    # 拼接对比
    compare = np.concatenate(
        [
            image,
            projected_depth_vis,
            projected_rgb
        ],
        axis=1
    )

    if save_result:
        save_dir = os.path.join(root_dir, "projection_test")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, sample_name + "_projection_compare.jpg")
        cv2.imwrite(save_path, compare)

        print("结果已保存:", save_path)

    if show:
        cv2.imshow("Left: original | Middle: depth projection | Right: rgb reconstruction", compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return compare, uv, depth


if __name__ == "__main__":
    ROOT_DIR = r"D:\frustum-convnet\sunrgbd\mysunrgbd\training"

    # 修改这里测试不同样本
    IMAGE_ID = 1

    test_one_sample(
        root_dir=ROOT_DIR,
        image_id=IMAGE_ID,
        save_result=True,
        show=True
    )