# -*- coding: utf-8 -*-
"""SAM3 文本提示分割演示模块。"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import torch


@dataclass
class SAM3SimpleDemoConfig:
    """SAM3 简单演示配置数据类。"""
    image_path: str = r"C:\Users\25918\Desktop\test\000616.jpg"
    model_path: str = r"D:\sam3.pt"
    bpe_path: str = r"D:\bpe_simple_vocab_16e6.txt.gz"
    text_prompts: List[str] = field(default_factory=lambda: ["chair", "sofa", "table"])
    conf: float = 0.25


def _check_file(path: str, label: str) -> None:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def run_sam3_simple_demo(config: Optional[SAM3SimpleDemoConfig] = None) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    config = config or SAM3SimpleDemoConfig()

    _check_file(config.image_path, "image")
    _check_file(config.model_path, "SAM3 model")
    _check_file(config.bpe_path, "BPE file")

    image = cv2.imread(config.image_path)
    if image is None:
        raise RuntimeError(f"OpenCV cannot read image: {config.image_path}")

    h, w = image.shape[:2]
    print("=" * 60)
    print("Image loaded")
    print(f"Image path: {config.image_path}")
    print(f"Image size: {w} x {h}")
    print("=" * 60)

    from ultralytics.models.sam import SAM3SemanticPredictor

    use_half = torch.cuda.is_available()
    overrides = dict(
        conf=float(config.conf),
        task="segment",
        mode="predict",
        model=config.model_path,
        half=use_half,
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)

    print("SAM3 model loaded")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"half: {use_half}")

    predictor.set_image(config.image_path)
    print("Image set to SAM3")

    results = predictor(text=config.text_prompts, save=False)

    print("=" * 60)
    print("SAM3 inference complete")
    print(f"Text prompts: {config.text_prompts}")
    print("=" * 60)

    if results is None:
        print("Empty results")
        return

    if not isinstance(results, (list, tuple)):
        results = [results]

    for result_idx, result in enumerate(results):
        print(f"\nResult {result_idx + 1}:")
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None)

        print(f"names: {names}")

        if masks is None:
            print("No masks")
        else:
            mask_data = getattr(masks, "data", None)
            if mask_data is None:
                print("masks.data is empty")
            else:
                print(f"mask count: {len(mask_data)}")
                print(f"mask shape: {mask_data.shape}")

        if boxes is None:
            print("No boxes")
        else:
            if hasattr(boxes, "xyxy"):
                print(f"box count: {len(boxes.xyxy)}")
                print(f"boxes.xyxy shape: {boxes.xyxy.shape}")
            if hasattr(boxes, "conf"):
                print(f"conf: {boxes.conf}")
            if hasattr(boxes, "cls"):
                print(f"cls: {boxes.cls}")

    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", image)

    for result_idx, result in enumerate(results):
        vis_img = result.plot()
        max_show_width = 1200
        show_h, show_w = vis_img.shape[:2]

        if show_w > max_show_width:
            scale = max_show_width / show_w
            vis_img = cv2.resize(
                vis_img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        window_name = f"SAM3 Result {result_idx + 1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, vis_img)

    print("\nPress any key to close windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nDemo finished")
