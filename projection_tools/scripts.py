# -*- coding: utf-8 -*-
"""单样本点云投影调试脚本。"""

import os

import cv2
import numpy as np

from tool3d_modules.io import load_calib, load_mat_points
from .projection import project_points_to_image, project_sunrgbd_points_to_image
from .visualization import create_projection_image_from_point_rgb, draw_projected_points_on_image, draw_projection, make_side_by_side


def test_one_sample(root_dir, image_id=1, save_result=True, show=True):

    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
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
