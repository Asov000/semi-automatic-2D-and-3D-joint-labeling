# -*- coding: utf-8 -*-
"""经典 SAM 点提示分割演示模块。"""

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from segment_anything import SamPredictor, sam_model_registry


@dataclass
class ClassicSamDemoConfig:
    """经典 SAM 演示配置数据类。"""
    image_path: str = r"C:\Users\25918\Desktop\test2.jpg"
    checkpoint: str = r"C:\Users\25918\Downloads\SAM\sam_vit_h_4b8939.pth"
    model_type: str = "vit_h"
    model_device: str = "cpu"
    random_color: bool = False


class SamPredict:
    """经典 SAM 演示预测封装类。"""
    def __init__(
        self,
        image: Any,
        checkpoint: Any,
        model_type: str,
        model_device: str,
        points: np.ndarray,
        labels: np.ndarray,
        random_color: bool,
    ):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        self.image = image
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_device = model_device
        self.points = points
        self.labels = labels
        self.marker_size = 300
        self.random_color = random_color

    def show_pre(self) -> None:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        plt.imshow(self.image)
        self.__show_points(self.points, self.labels)
        plt.axis("off")
        plt.show()

    def __show_points(self, points: np.ndarray, labels: np.ndarray) -> None:
        """执行模块内部辅助逻辑，供上层流程复用。"""
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]

        if len(pos_points) > 0:
            plt.scatter(
                pos_points[:, 0],
                pos_points[:, 1],
                color="green",
                marker="*",
                s=self.marker_size,
                edgecolor="white",
                linewidth=1.25,
            )

        if len(neg_points) > 0:
            plt.scatter(
                neg_points[:, 0],
                neg_points[:, 1],
                color="red",
                marker="*",
                s=self.marker_size,
                edgecolor="white",
                linewidth=1.25,
            )

    def __show_mask(self, mask: np.ndarray) -> None:
        """执行模块内部辅助逻辑，供上层流程复用。"""
        if self.random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
        plt.imshow(mask_image)

    def get_result(self) -> None:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        model.to(device=self.model_device)

        predictor = SamPredictor(model)
        predictor.set_image(self.image)

        masks, scores, logits = predictor.predict(
            point_coords=self.points,
            point_labels=self.labels,
            multimask_output=True,
        )

        print("Height:", masks.shape[1])
        print("Width:", masks.shape[2])
        print("Mask count:", masks.shape[0])

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)
            self.__show_mask(mask)
            self.__show_points(points=self.points, labels=self.labels)
            plt.title(f"Mask_Times:{i + 1}, Mask_Scores:{score:.4f}", fontsize=18)
            plt.axis("off")
            plt.show()


def load_rgb_image(image_path: str) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def run_classic_sam_demo(
    config: Optional[ClassicSamDemoConfig] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    config = config or ClassicSamDemoConfig()

    if points is None:
        points = np.array([[480, 360], [300, 300]])
    if labels is None:
        labels = np.array([1, 0])

    model_one = SamPredict(
        image=load_rgb_image(config.image_path),
        checkpoint=config.checkpoint,
        model_type=config.model_type,
        model_device=config.model_device,
        points=points,
        labels=labels,
        random_color=config.random_color,
    )
    model_one.get_result()
