# # -*- coding: utf-8 -*-
# """经典 SAM 点提示分割演示模块。"""
#
# from dataclasses import dataclass
# from typing import Any, Optional
#
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# # SAM 官方库中的预测器和模型注册表
# from segment_anything import SamPredictor, sam_model_registry
#
#
# @dataclass
# class ClassicSamDemoConfig:
#     """
#     经典 SAM 演示配置类。
#
#     用于统一管理：
#     1. 输入图像路径
#     2. SAM 权重路径
#     3. 模型类型
#     4. 推理设备
#     5. mask 是否随机颜色显示
#     """
#
#     # 测试图片路径
#     image_path: str = r"C:\Users\25918\Desktop\test2.jpg"
#
#     # SAM 模型权重路径
#     checkpoint: str = r"C:\Users\25918\Downloads\SAM\sam_vit_h_4b8939.pth"
#
#     # SAM 模型类型，可选 vit_b / vit_l / vit_h
#     model_type: str = "vit_h"
#
#     # 推理设备，cpu 或 cuda
#     model_device: str = "cpu"
#
#     # 是否使用随机颜色显示 mask
#     random_color: bool = False
#
#
# class SamPredict:
#     """
#     经典 SAM 点提示预测封装类。
#
#     主要负责：
#     1. 保存输入图像和提示点
#     2. 显示提示点
#     3. 加载 SAM 模型
#     4. 执行点提示分割
#     5. 可视化多个候选 mask
#     """
#
#     def __init__(
#         self,
#         image: Any,
#         checkpoint: Any,
#         model_type: str,
#         model_device: str,
#         points: np.ndarray,
#         labels: np.ndarray,
#         random_color: bool,
#     ):
#         """
#         初始化 SAM 预测类。
#
#         参数：
#         image: RGB 图像
#         checkpoint: SAM 权重路径
#         model_type: 模型类型，例如 vit_h
#         model_device: 推理设备，例如 cpu / cuda
#         points: 点提示坐标，形状为 N x 2，格式为 [x, y]
#         labels: 点提示标签，1 表示前景点，0 表示背景点
#         random_color: 是否随机颜色显示 mask
#         """
#
#         # 输入图像，注意这里应当是 RGB 格式
#         self.image = image
#
#         # SAM 权重文件路径
#         self.checkpoint = checkpoint
#
#         # SAM 模型类型
#         self.model_type = model_type
#
#         # 推理设备
#         self.model_device = model_device
#
#         # 点提示坐标，格式为 [x, y]
#         self.points = points
#
#         # 点提示标签：1 为前景点，0 为背景点
#         self.labels = labels
#
#         # matplotlib 中显示提示点的大小
#         self.marker_size = 300
#
#         # 是否随机颜色显示 mask
#         self.random_color = random_color
#
#     def show_pre(self) -> None:
#         """
#         显示原图和提示点。
#
#         作用：
#         在正式执行 SAM 分割前，先检查提示点位置是否正确。
#         """
#
#         # 显示 RGB 原图
#         plt.imshow(self.image)
#
#         # 在图像上绘制正负提示点
#         self.__show_points(self.points, self.labels)
#
#         # 关闭坐标轴
#         plt.axis("off")
#
#         # 显示图像
#         plt.show()
#
#     def __show_points(self, points: np.ndarray, labels: np.ndarray) -> None:
#         """
#         在图像上绘制提示点。
#
#         参数：
#         points: 点坐标，形状为 N x 2，格式为 [x, y]
#         labels: 点标签，1 表示前景点，0 表示背景点
#         """
#
#         # 取出前景点
#         pos_points = points[labels == 1]
#
#         # 取出背景点
#         neg_points = points[labels == 0]
#
#         # 绘制前景点，绿色星形
#         if len(pos_points) > 0:
#             plt.scatter(
#                 pos_points[:, 0],     # x 坐标
#                 pos_points[:, 1],     # y 坐标
#                 color="green",        # 前景点用绿色
#                 marker="*",           # 星形标记
#                 s=self.marker_size,    # 点大小
#                 edgecolor="white",    # 白色边缘
#                 linewidth=1.25,        # 边缘线宽
#             )
#
#         # 绘制背景点，红色星形
#         if len(neg_points) > 0:
#             plt.scatter(
#                 neg_points[:, 0],      # x 坐标
#                 neg_points[:, 1],      # y 坐标
#                 color="red",          # 背景点用红色
#                 marker="*",           # 星形标记
#                 s=self.marker_size,    # 点大小
#                 edgecolor="white",    # 白色边缘
#                 linewidth=1.25,        # 边缘线宽
#             )
#
#     def __show_mask(self, mask: np.ndarray) -> None:
#         """
#         将 SAM 输出的 mask 叠加显示在当前图像上。
#
#         参数：
#         mask:
#             SAM 输出的单个二值掩膜，形状一般为 H x W。
#         """
#
#         # 如果开启随机颜色，则每个 mask 用不同随机颜色显示
#         if self.random_color:
#             # 前 3 个值是 RGB，最后一个值是透明度 alpha
#             color = np.concatenate(
#                 [np.random.random(3), np.array([0.6])],
#                 axis=0
#             )
#
#         # 否则使用固定蓝色显示 mask
#         else:
#             # RGBA 格式，前三个是颜色，最后一个是透明度
#             color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#
#         # 获取 mask 的高度和宽度
#         h, w = mask.shape[-2:]
#
#         # 将 H x W 的 mask 转成 H x W x 1
#         # 再乘以 RGBA 颜色，得到 H x W x 4 的半透明彩色 mask
#         mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
#
#         # 显示半透明 mask
#         plt.imshow(mask_image)
#
#     def get_result(self) -> None:
#         """
#         加载 SAM 模型，执行点提示分割，并显示结果。
#
#         注意：
#         multimask_output=True 时，SAM 会输出多个候选 mask。
#         通常会输出 3 个结果，并给出每个 mask 的 score。
#         """
#
#         # 根据模型类型从注册表中创建 SAM 模型
#         model = sam_model_registry[self.model_type](
#             checkpoint=self.checkpoint
#         )
#
#         # 将模型移动到指定设备
#         model.to(device=self.model_device)
#
#         # 创建 SAM 预测器
#         predictor = SamPredictor(model)
#
#         # 设置输入图像
#         # self.image 必须是 RGB 格式
#         predictor.set_image(self.image)
#
#         # 执行点提示预测
#         masks, scores, logits = predictor.predict(
#             point_coords=self.points,        # 点坐标，格式为 [x, y]
#             point_labels=self.labels,        # 点标签，1 前景，0 背景
#             multimask_output=True,           # 输出多个候选 mask
#         )
#
#         # 打印 mask 的基本信息，方便调试
#         print("Height:", masks.shape[1])
#         print("Width:", masks.shape[2])
#         print("Mask count:", masks.shape[0])
#
#         # 遍历每一个候选 mask
#         for i, (mask, score) in enumerate(zip(masks, scores)):
#             # 显示原图
#             plt.imshow(self.image)
#
#             # 叠加当前 mask
#             self.__show_mask(mask)
#
#             # 显示提示点
#             self.__show_points(
#                 points=self.points,
#                 labels=self.labels
#             )
#
#             # 设置标题，显示当前 mask 编号和分数
#             plt.title(
#                 f"Mask_Times:{i + 1}, Mask_Scores:{score:.4f}",
#                 fontsize=18
#             )
#
#             # 关闭坐标轴
#             plt.axis("off")
#
#             # 显示当前候选结果
#             plt.show()
#
#
# def load_rgb_image(image_path: str) -> np.ndarray:
#     """
#     读取图像并转换为 RGB 格式。
#
#     参数：
#     image_path: 图像路径
#
#     返回：
#     image_rgb: RGB 格式图像
#     """
#
#     # OpenCV 默认读取 BGR 图像
#     image_bgr = cv2.imread(image_path)
#
#     # 如果读取失败，抛出异常
#     if image_bgr is None:
#         raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")
#
#     # 将 BGR 转为 RGB
#     # SAM 和 matplotlib 都更适合使用 RGB 格式
#     return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#
#
# def run_classic_sam_demo(
#     config: Optional[ClassicSamDemoConfig] = None,
#     points: Optional[np.ndarray] = None,
#     labels: Optional[np.ndarray] = None,
# ) -> None:
#     """
#     运行经典 SAM 点提示分割 demo。
#
#     参数：
#     config:
#         SAM 演示配置。
#         如果不传，则使用 ClassicSamDemoConfig 默认配置。
#
#     points:
#         点提示坐标，形状为 N x 2。
#         格式为 [x, y]，不是 [row, col]。
#
#     labels:
#         点提示标签。
#         1 表示前景点，0 表示背景点。
#
#     返回：
#     无返回值，直接显示分割结果。
#     """
#
#     # 如果没有传入配置，则使用默认配置
#     config = config or ClassicSamDemoConfig()
#
#     # 如果没有传入提示点，则使用默认点
#     if points is None:
#         points = np.array([
#             [480, 360],   # 第一个点
#             [300, 300],   # 第二个点
#         ])
#
#     # 如果没有传入点标签，则默认：
#     # 第一个点是前景点
#     # 第二个点是背景点
#     if labels is None:
#         labels = np.array([1, 0])
#
#     # 创建 SAM 预测对象
#     model_one = SamPredict(
#         # 读取图像，并转换成 RGB
#         image=load_rgb_image(config.image_path),
#
#         # SAM 权重路径
#         checkpoint=config.checkpoint,
#
#         # 模型类型
#         model_type=config.model_type,
#
#         # 推理设备
#         model_device=config.model_device,
#
#         # 点提示坐标
#         points=points,
#
#         # 点提示标签
#         labels=labels,
#
#         # 是否随机颜色显示 mask
#         random_color=config.random_color,
#     )
#
#     # 执行 SAM 点提示分割，并显示结果
#     model_one.get_result()