# -*- coding: utf-8 -*-
"""SAM3 自动标注工具包导出入口。"""

from .config import SAM3AutoConfig, build_sam3_overrides
from .generator import SAM3AutoMaskGenerator
from .utils import _get_class_name, _resize_mask_to_image, _to_numpy, normalize_text_prompts

__all__ = [
    "SAM3AutoConfig",
    "build_sam3_overrides",
    "normalize_text_prompts",
    "_to_numpy",
    "_resize_mask_to_image",
    "_get_class_name",
    "SAM3AutoMaskGenerator",
]
