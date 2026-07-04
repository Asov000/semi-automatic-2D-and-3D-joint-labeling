# -*- coding: utf-8 -*-
"""SAM3 文本提示分割演示模块。"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import torch


@dataclass
class SAM3SimpleDemoConfig:
    """
    SAM3 简单演示配置类。

    用于统一管理：
    1. 输入图片路径
    2. SAM3 模型权重路径
    3. BPE 词表路径
    4. 文本提示类别
    5. 置信度阈值
    """

    # 输入图片路径
    image_path: str = r"C:\Users\25918\Desktop\test\000616.jpg"

    # SAM3 模型权重路径
    model_path: str = r"D:\sam3.pt"

    # BPE 词表文件路径
    # 注意：当前这份代码只是检查了 bpe_path 是否存在，
    # 但没有真正把它传入 predictor 使用
    bpe_path: str = r"D:\bpe_simple_vocab_16e6.txt.gz"

    # 文本提示词，也就是希望 SAM3 自动分割的目标类别
    text_prompts: List[str] = field(
        default_factory=lambda: ["chair", "sofa", "table"]
    )

    # 置信度阈值，低于该值的预测结果会被过滤
    conf: float = 0.25


def _check_file(path: str, label: str) -> None:
    """
    检查指定文件是否存在。

    参数：
    path:
        文件路径。

    label:
        文件类型说明，用于报错提示。
        例如 image、SAM3 model、BPE file。

    返回：
    无返回值。
    如果文件不存在，直接抛出 FileNotFoundError。
    """

    # 判断文件路径是否存在
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def run_sam3_simple_demo(config: Optional[SAM3SimpleDemoConfig] = None) -> None:
    """
    运行 SAM3 文本提示分割 demo。

    参数：
    config:
        SAM3SimpleDemoConfig 配置对象。
        如果不传，则使用默认配置。

    功能流程：
    1. 检查图片、模型、BPE 文件是否存在
    2. 读取图片并打印图片信息
    3. 加载 SAM3SemanticPredictor
    4. 根据文本提示词执行分割
    5. 打印 masks、boxes、conf、cls 等调试信息
    6. 显示原图和 SAM3 分割结果
    """

    # 如果没有传入配置，则使用默认配置
    config = config or SAM3SimpleDemoConfig()

    # 检查图片文件是否存在
    _check_file(config.image_path, "image")

    # 检查 SAM3 模型权重是否存在
    _check_file(config.model_path, "SAM3 model")

    # 检查 BPE 文件是否存在
    # 当前代码只检查，不传入 predictor
    _check_file(config.bpe_path, "BPE file")

    # 使用 OpenCV 读取图片
    # OpenCV 默认读取格式是 BGR
    image = cv2.imread(config.image_path)

    # 如果图片读取失败，抛出异常
    if image is None:
        raise RuntimeError(f"OpenCV cannot read image: {config.image_path}")

    # 获取图片高度和宽度
    h, w = image.shape[:2]

    # 打印图片基础信息，方便调试
    print("=" * 60)
    print("Image loaded")
    print(f"Image path: {config.image_path}")
    print(f"Image size: {w} x {h}")
    print("=" * 60)

    # 从 ultralytics 中导入 SAM3 语义分割预测器
    from ultralytics.models.sam import SAM3SemanticPredictor

    # 如果当前环境支持 CUDA，则启用 half 半精度推理
    # 半精度通常可以减少显存占用，并加快 GPU 推理
    use_half = torch.cuda.is_available()

    # 构建 SAM3 推理参数
    overrides = dict(
        conf=float(config.conf),      # 置信度阈值
        task="segment",              # 任务类型：分割
        mode="predict",              # 模式：预测
        model=config.model_path,      # SAM3 模型权重路径
        half=use_half,                # 是否使用 FP16 半精度
    )

    # 创建 SAM3 predictor
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # 打印模型加载和设备信息
    print("SAM3 model loaded")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"half: {use_half}")

    # 设置待分割图片
    # 这里传入的是图片路径
    predictor.set_image(config.image_path)
    print("Image set to SAM3")

    # 根据文本提示词执行 SAM3 分割
    # text_prompts 例如 ["chair", "sofa", "table"]
    results = predictor(
        text=config.text_prompts,
        save=False
    )

    # 打印推理完成信息
    print("=" * 60)
    print("SAM3 inference complete")
    print(f"Text prompts: {config.text_prompts}")
    print("=" * 60)

    # 如果模型没有返回结果，直接结束
    if results is None:
        print("Empty results")
        return

    # 保证 results 是列表形式，方便统一遍历
    if not isinstance(results, (list, tuple)):
        results = [results]

    # 遍历每一个 result，打印内部信息
    for result_idx, result in enumerate(results):
        print(f"\nResult {result_idx + 1}:")

        # 获取 mask 结果
        masks = getattr(result, "masks", None)

        # 获取 box 结果
        boxes = getattr(result, "boxes", None)

        # 获取类别名称映射
        names = getattr(result, "names", None)

        # 打印类别名称映射
        print(f"names: {names}")

        # 打印 mask 相关信息
        if masks is None:
            print("No masks")
        else:
            # masks.data 通常是所有 mask 的张量
            # 形状一般为 N x H x W
            mask_data = getattr(masks, "data", None)

            if mask_data is None:
                print("masks.data is empty")
            else:
                print(f"mask count: {len(mask_data)}")
                print(f"mask shape: {mask_data.shape}")

        # 打印 box 相关信息
        if boxes is None:
            print("No boxes")
        else:
            # boxes.xyxy 是检测框坐标
            # 格式一般为 [xmin, ymin, xmax, ymax]
            if hasattr(boxes, "xyxy"):
                print(f"box count: {len(boxes.xyxy)}")
                print(f"boxes.xyxy shape: {boxes.xyxy.shape}")

            # boxes.conf 是每个目标的置信度
            if hasattr(boxes, "conf"):
                print(f"conf: {boxes.conf}")

            # boxes.cls 是每个目标的类别 ID
            if hasattr(boxes, "cls"):
                print(f"cls: {boxes.cls}")

    # 创建窗口显示原始图像
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", image)

    # 遍历每一个结果并显示可视化图像
    for result_idx, result in enumerate(results):
        # result.plot() 会将 mask、box、label 等绘制到图像上
        vis_img = result.plot()

        # 设置最大显示宽度，防止图像太大超出屏幕
        max_show_width = 1200
        show_h, show_w = vis_img.shape[:2]

        # 如果图像宽度超过最大显示宽度，则按比例缩小
        if show_w > max_show_width:
            scale = max_show_width / show_w
            vis_img = cv2.resize(
                vis_img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        # 创建结果显示窗口
        window_name = f"SAM3 Result {result_idx + 1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 显示 SAM3 分割结果
        cv2.imshow(window_name, vis_img)

    # 等待用户按键关闭窗口
    print("\nPress any key to close windows")
    cv2.waitKey(0)

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()

    print("\nDemo finished")

run_sam3_simple_demo()