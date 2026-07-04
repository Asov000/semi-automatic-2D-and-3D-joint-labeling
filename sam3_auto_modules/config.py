# -*- coding: utf-8 -*-
"""SAM3 配置模块，负责生成自动分割推理参数。"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

# 尝试从 run.py 中读取 SAM3 的默认配置
# 这样可以让主程序中的配置统一管理
try:
    from run import (
        SAM3_DEFAULT_CONF,      # 默认置信度阈值
        SAM3_DEFAULT_DEVICE,    # 默认推理设备，例如 "cuda:0" / "cpu"
        SAM3_DEFAULT_HALF,      # 是否使用半精度推理
        SAM3_DEFAULT_IMGSZ,     # 默认输入图像尺寸
        SAM3_DEFAULT_IOU,       # 默认 NMS 的 IoU 阈值
        SAM3_DEFAULT_MAX_DET,   # 默认最大检测数量
        SAM3_DEFAULT_VERBOSE,   # 是否输出详细推理日志
        SAM3_MODEL_PATH,        # SAM3 模型权重路径
    )

# 如果 run.py 不存在，或者导入失败，则使用下面这组备用默认参数
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
    """
    SAM3 自动推理配置数据类。

    作用：
    用一个类统一保存 SAM3 推理需要的参数，
    后续可以通过该类快速生成 overrides 配置字典。
    """

    # SAM3 模型权重路径
    model_path: str = SAM3_MODEL_PATH

    # 置信度阈值，低于该值的预测结果会被过滤
    conf: float = SAM3_DEFAULT_CONF

    # NMS 的 IoU 阈值，用于控制重叠结果的去除程度
    iou: float = SAM3_DEFAULT_IOU

    # 推理输入图像尺寸
    imgsz: int = SAM3_DEFAULT_IMGSZ

    # 最大检测 / 分割目标数量
    max_det: int = SAM3_DEFAULT_MAX_DET

    # 是否启用半精度 FP16 推理
    # 一般在 GPU 上可以加速并节省显存
    half: bool = SAM3_DEFAULT_HALF

    # 推理设备，例如 "cuda:0"、"cuda:1" 或 "cpu"
    # 为 None 时通常由框架自动选择
    device: Optional[str] = SAM3_DEFAULT_DEVICE

    # 是否输出详细日志
    verbose: bool = SAM3_DEFAULT_VERBOSE


def build_sam3_overrides(
    config: Optional[Union[SAM3AutoConfig, Dict]] = None,
    **kwargs
) -> Dict:
    """
    构建 SAM3 推理时需要传入的 overrides 参数字典。

    参数：
    config:
        可以是 None、SAM3AutoConfig 对象或 dict。
        - None：使用默认配置
        - SAM3AutoConfig：直接使用传入配置
        - dict：用字典中的字段覆盖默认配置

    **kwargs:
        额外传入的覆盖参数，优先级高于 config。

    返回：
    overrides:
        可直接传给 SAM3 / YOLO 风格模型 predict 接口的参数字典。
    """

    # 如果没有传入 config，则使用默认配置
    if config is None:
        cfg = SAM3AutoConfig()

    # 如果传入的是 SAM3AutoConfig 对象，则直接使用
    elif isinstance(config, SAM3AutoConfig):
        cfg = config

    # 如果传入的是字典，则用字典内容覆盖默认配置
    elif isinstance(config, dict):
        base = SAM3AutoConfig()

        # 只更新 SAM3AutoConfig 中存在的字段
        # 避免传入无关参数导致报错
        for k, v in config.items():
            if hasattr(base, k):
                setattr(base, k, v)

        cfg = base

    # 如果 config 类型不合法，则直接报错
    else:
        raise TypeError(f"不支持的 config 类型：{type(config)}")

    # 使用 kwargs 继续覆盖配置
    # kwargs 的优先级最高
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # 对关键参数做合法性限制，防止传入异常值
    cfg.conf = max(0.0, min(1.0, float(cfg.conf)))   # conf 限制在 0~1
    cfg.iou = max(0.0, min(1.0, float(cfg.iou)))     # iou 限制在 0~1
    cfg.imgsz = max(320, int(cfg.imgsz))             # imgsz 最小为 320
    cfg.max_det = max(1, int(cfg.max_det))           # max_det 至少为 1
    cfg.half = bool(cfg.half)                        # 转为 bool 类型

    # 构建最终传给模型推理接口的参数字典
    overrides = dict(
        conf=cfg.conf,             # 置信度阈值
        iou=cfg.iou,               # IoU 阈值
        task="segment",            # 任务类型：分割
        mode="predict",            # 模式：预测
        model=cfg.model_path,       # 模型权重路径
        imgsz=cfg.imgsz,           # 输入图像尺寸
        max_det=cfg.max_det,       # 最大检测数量
        half=cfg.half,             # 是否使用 FP16
        verbose=cfg.verbose,       # 是否输出详细日志
    )

    # 如果指定了推理设备，则加入 device 参数
    # 如果 cfg.device 为 None，则不写入，让框架自动选择设备
    if cfg.device:
        overrides["device"] = cfg.device

    # 返回最终推理配置
    return overrides