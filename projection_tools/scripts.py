# -*- coding: utf-8 -*-
"""单样本点云投影调试脚本。"""

import os

import cv2
import numpy as np

# 读取标定文件和 .mat 点云文件的工具函数
from tool3d_modules.io import load_calib, load_mat_points

# 点云投影相关函数
from projection import project_points_to_image, project_sunrgbd_points_to_image

# 可视化相关函数
from visualization import (
    create_projection_image_from_point_rgb,
    draw_projected_points_on_image,
    draw_projection,
    make_side_by_side
)


def test_one_sample(root_dir, image_id=1, save_result=True, show=True):
    """
    测试单个 SUNRGBD 样本的点云投影效果。

    参数：
    root_dir: 数据集根目录
    image_id: 样本编号，例如 1 会被格式化为 000001
    save_result: 是否保存投影对比图
    show: 是否弹窗显示投影结果

    返回：
    compare: 拼接后的对比图
    uv: 有效投影点的 2D 像素坐标
    depth: 有效投影点对应的深度
    """

    # 将 image_id 转成 6 位编号格式，例如 1 -> 000001
    sample_name = f"{image_id:06d}"

    # 构造点云、图像、标定文件路径
    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    image_path = os.path.join(root_dir, "image", sample_name + ".jpg")
    calib_path = os.path.join(root_dir, "calib", sample_name + ".txt")

    # 打印当前读取的文件路径，方便调试
    print("读取点云:", pc_path)
    print("读取图像:", image_path)
    print("读取标定:", calib_path)

    # 读取 .mat 点云数据，一般包含 xyz 和 rgb 信息
    points3d_rgb = load_mat_points(pc_path)

    # 读取 RGB 图像
    image = cv2.imread(image_path)

    # 如果图像读取失败，直接抛出错误
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {image_path}")

    # 读取标定文件，得到 Rtilt 和相机内参 K
    Rtilt, K = load_calib(calib_path)

    # 前 3 列是点云坐标 xyz
    points3d = points3d_rgb[:, 0:3]

    # 后 3 列是点云颜色 rgb
    point_rgb = points3d_rgb[:, 3:6]

    # 将 3D 点云投影到 2D 图像平面
    # uv: 投影后的像素坐标
    # depth: 对应深度
    # valid_mask: 原始点云中成功投影到图像内的点的 mask
    uv, depth, valid_mask = project_points_to_image(
        points3d_after_rtilt=points3d,
        K=K,
        Rtilt=Rtilt,
        image_shape=image.shape,
        use_matlab_pixel=True
    )

    # 打印基础调试信息
    print("原始点云数量:", points3d.shape[0])
    print("有效投影点数量:", uv.shape[0])
    print("图像尺寸:", image.shape)
    print("K =\n", K)
    print("Rtilt =\n", Rtilt)

    # 生成深度颜色投影图
    # 颜色通常根据 depth 深度值变化，用于检查投影位置是否正确
    projected_depth_vis = draw_projection(
        image=image,
        uv=uv,
        depth=depth,
        point_size=1,
        alpha=0.75
    )

    # 根据点云自带的 RGB 颜色，把点云重新投影成一张 RGB 图像
    # point_rgb[valid_mask] 保证颜色和成功投影的点一一对应
    projected_rgb = create_projection_image_from_point_rgb(
        image_shape=image.shape,
        uv=uv,
        point_rgb=point_rgb[valid_mask]
    )

    # 将原图、深度投影图、RGB 重建图横向拼接，方便对比
    compare = np.concatenate(
        [
            image,
            projected_depth_vis,
            projected_rgb
        ],
        axis=1
    )

    # 如果需要保存结果
    if save_result:
        # 创建保存目录
        save_dir = os.path.join(root_dir, "projection_test")
        os.makedirs(save_dir, exist_ok=True)

        # 构造保存路径
        save_path = os.path.join(save_dir, sample_name + "_projection_compare.jpg")

        # 保存拼接后的对比图
        cv2.imwrite(save_path, compare)

        print("结果已保存:", save_path)

    # 如果需要显示结果
    if show:
        # 显示三联图：
        # 左：原始 RGB 图像
        # 中：点云深度投影图
        # 右：点云 RGB 重建图
        cv2.imshow(
            "Left: original | Middle: depth projection | Right: rgb reconstruction",
            compare
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 返回对比图、投影像素点和深度
    return compare, uv, depth