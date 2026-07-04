# -*- coding: utf-8 -*-
"""SAM3 配置模块，负责生成自动分割推理参数。"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

try:
    from run import (
        SAM3_DEFAULT_CONF,
        SAM3_DEFAULT_DEVICE,
        SAM3_DEFAULT_HALF,
        SAM3_DEFAULT_IMGSZ,
        SAM3_DEFAULT_IOU,
        SAM3_DEFAULT_MAX_DET,
        SAM3_DEFAULT_VERBOSE,
        SAM3_MODEL_PATH,
    )
except Exception:
    SAM3_MODEL_PATH = r"D:\sam3.pt"
    SAM3_DEFAULT_CONF = 0.25
    SAM3_DEFAULT_IOU = 0.70
    SAM3_DEFAULT_IMGSZ = 1024
    SAM3_DEFAULT_MAX_DET = 100
    SAM3_DEFAULT_HALF = True
    SAM3_DEFAULT_DEVICE = None
    SAM3_DEFAULT_VERBOSE = False


@dataclass
class SAM3AutoConfig:
    """SAM3 自动推理配置数据类。"""
    model_path: str = SAM3_MODEL_PATH
    conf: float = SAM3_DEFAULT_CONF
    iou: float = SAM3_DEFAULT_IOU
    imgsz: int = SAM3_DEFAULT_IMGSZ
    max_det: int = SAM3_DEFAULT_MAX_DET
    half: bool = SAM3_DEFAULT_HALF
    device: Optional[str] = SAM3_DEFAULT_DEVICE
    verbose: bool = SAM3_DEFAULT_VERBOSE


def build_sam3_overrides(
    config: Optional[Union[SAM3AutoConfig, Dict]] = None,
    **kwargs
) -> Dict:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if config is None:
        cfg = SAM3AutoConfig()
    elif isinstance(config, SAM3AutoConfig):
        cfg = config
    elif isinstance(config, dict):
        base = SAM3AutoConfig()
        for k, v in config.items():
            if hasattr(base, k):
                setattr(base, k, v)
        cfg = base
    else:
        raise TypeError(f"不支持的 config 类型：{type(config)}")

    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg.conf = max(0.0, min(1.0, float(cfg.conf)))
    cfg.iou = max(0.0, min(1.0, float(cfg.iou)))
    cfg.imgsz = max(320, int(cfg.imgsz))
    cfg.max_det = max(1, int(cfg.max_det))
    cfg.half = bool(cfg.half)

    overrides = dict(
        conf=cfg.conf,
        iou=cfg.iou,
        task="segment",
        mode="predict",
        model=cfg.model_path,
        imgsz=cfg.imgsz,
        max_det=cfg.max_det,
        half=cfg.half,
        verbose=cfg.verbose,
    )

    if cfg.device:
        overrides["device"] = cfg.device

    return overrides
