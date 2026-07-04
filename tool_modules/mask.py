# -*- coding: utf-8 -*-
"""二维 mask 后处理模块，提供二值化、连通域过滤、形态学处理和填洞。"""

from typing import Optional

import cv2
import numpy as np

from .types import ArrayLike


def to_binary_mask(mask: ArrayLike, threshold: float = 0) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    if mask is None:
        raise ValueError("mask 不能为 None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = np.max(arr, axis=2)

    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D 或单通道图像，当前 shape={arr.shape}")

    return (arr > threshold).astype(np.uint8)


def _make_odd_kernel_size(kernel_size: int) -> int:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _ellipse_kernel(kernel_size: int) -> np.ndarray:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    kernel_size = _make_odd_kernel_size(kernel_size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def remove_small_components(
    mask: ArrayLike,
    min_area: int = 64,
    keep_largest: bool = False,
) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    binary = to_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    output = np.zeros_like(binary, dtype=np.uint8)
    component_ids = list(range(1, num_labels))
    areas = [stats[i, cv2.CC_STAT_AREA] for i in component_ids]

    if keep_largest:
        largest_idx = int(np.argmax(areas))
        largest_label = component_ids[largest_idx]
        largest_area = areas[largest_idx]

        if min_area <= 0 or largest_area >= min_area:
            output[labels == largest_label] = 1
    else:
        for comp_id, area in zip(component_ids, areas):
            if area >= min_area:
                output[labels == comp_id] = 1

    return output


def fill_mask_holes(mask: ArrayLike) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    binary = to_binary_mask(mask)
    background = (binary == 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    if num_labels <= 1:
        return binary

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    output = binary.copy()

    for label_id in range(1, num_labels):
        if label_id not in border_labels:
            output[labels == label_id] = 1

    return output.astype(np.uint8)


def refine_mask(
    mask: ArrayLike,
    min_area: int = 64,
    keep_largest: bool = True,
    close_kernel_size: int = 5,
    close_iterations: int = 1,
    open_kernel_size: int = 0,
    open_iterations: int = 1,
    fill_holes: bool = True,
    max_external_expand_px: Optional[int] = 0,
) -> np.ndarray:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    original = to_binary_mask(mask)

    clean = remove_small_components(
        original,
        min_area=min_area,
        keep_largest=keep_largest,
    )

    if open_kernel_size and open_kernel_size > 1:
        open_kernel = _ellipse_kernel(open_kernel_size)
        clean = cv2.morphologyEx(
            clean,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=max(1, int(open_iterations)),
        )
        clean = to_binary_mask(clean)

    if close_kernel_size and close_kernel_size > 1:
        close_kernel = _ellipse_kernel(close_kernel_size)
        closed = cv2.morphologyEx(
            clean,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=max(1, int(close_iterations)),
        )
        closed = to_binary_mask(closed)
    else:
        closed = clean.copy()

    if fill_holes:
        filled = fill_mask_holes(closed)
    else:
        filled = closed.copy()

    if max_external_expand_px is not None and max_external_expand_px >= 0:
        if max_external_expand_px == 0:
            allowed_external = clean.astype(bool)
        else:
            px = int(max_external_expand_px)
            allow_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * px + 1, 2 * px + 1),
            )
            allowed_external = cv2.dilate(clean, allow_kernel, iterations=1).astype(bool)

        holes_to_fill = filled.astype(bool) & (~closed.astype(bool))
        refined = ((filled.astype(bool) & allowed_external) | holes_to_fill).astype(np.uint8)
    else:
        refined = filled.astype(np.uint8)

    refined = remove_small_components(
        refined,
        min_area=min_area,
        keep_largest=keep_largest,
    )

    if fill_holes:
        refined = fill_mask_holes(refined)

    return refined.astype(np.uint8)
