# -*- coding: utf-8 -*-
"""SAM 演示脚本包导出入口。"""

from .classic_sam import SamPredict, run_classic_sam_demo
from .sam3_simple import SAM3SimpleDemoConfig, run_sam3_simple_demo

__all__ = [
    "SamPredict",
    "run_classic_sam_demo",
    "SAM3SimpleDemoConfig",
    "run_sam3_simple_demo",
]
