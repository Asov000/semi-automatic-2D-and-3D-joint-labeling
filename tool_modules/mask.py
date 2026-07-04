# -*- coding: utf-8 -*-
"""二维 mask 后处理模块，提供二值化、连通域过滤、形态学处理和填洞。"""

from typing import Optional

import cv2
import numpy as np

# ArrayLike 一般表示 numpy 数组、list、torch tensor 等可转为数组的数据类型
from .types import ArrayLike


def to_binary_mask(mask: ArrayLike, threshold: float = 0) -> np.ndarray:
    """
    将输入 mask 转换为标准二值 mask。

    参数：
    mask:
        输入 mask，可以是 0/1、0/255、灰度图、单通道图或多通道图。

    threshold:
        二值化阈值。
        大于 threshold 的位置设为 1，否则设为 0。

    返回：
    binary:
        uint8 类型的二值 mask，取值为 0 或 1。
    """

    # mask 不能为空
    if mask is None:
        raise ValueError("mask 不能为 None")

    # 转换为 numpy 数组
    arr = np.asarray(mask)

    # 如果是三维图像，需要压缩成二维 mask
    if arr.ndim == 3:
        # 情况 1：H x W x 1，直接取单通道
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

        # 情况 2：H x W x C，多通道 mask
        # 对通道维度取最大值，只要某个通道有前景，就认为该像素是前景
        else:
            arr = np.max(arr, axis=2)

    # 最终必须是二维 mask
    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D 或单通道图像，当前 shape={arr.shape}")

    # 根据阈值进行二值化，输出 0/1
    return (arr > threshold).astype(np.uint8)


def _make_odd_kernel_size(kernel_size: int) -> int:
    """
    将形态学卷积核尺寸转换为合法的奇数尺寸。

    OpenCV 形态学操作常用奇数大小的 kernel，
    例如 3、5、7。

    参数：
    kernel_size:
        输入核大小。

    返回：
    合法的奇数核大小。
    """

    # 转成整数
    kernel_size = int(kernel_size)

    # 小于等于 1 时，统一返回 1
    if kernel_size <= 1:
        return 1

    # 如果是偶数，则加 1 变成奇数
    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size


def _ellipse_kernel(kernel_size: int) -> np.ndarray:
    """
    创建椭圆形形态学卷积核。

    参数：
    kernel_size:
        卷积核尺寸。

    返回：
    OpenCV 可用的椭圆结构元素。
    """

    # 确保 kernel_size 是奇数
    kernel_size = _make_odd_kernel_size(kernel_size)

    # 创建椭圆形结构元素
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )


def remove_small_components(
    mask: ArrayLike,
    min_area: int = 64,
    keep_largest: bool = False,
) -> np.ndarray:
    """
    去除 mask 中的小连通区域。

    参数：
    mask:
        输入 mask。

    min_area:
        最小连通域面积。
        小于该面积的区域会被删除。

    keep_largest:
        是否只保留最大连通区域。
        True:
            只保留最大目标区域。
        False:
            保留所有面积大于 min_area 的区域。

    返回：
    output:
        去除小区域后的二值 mask，取值为 0 或 1。
    """

    # 转换为标准二值 mask
    binary = to_binary_mask(mask)

    # 连通域分析
    # num_labels: 连通域数量，包括背景
    # labels: 每个像素所属的连通域编号
    # stats: 每个连通域的统计信息，包括面积、bbox 等
    # connectivity=8 表示使用 8 邻域连接
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    # 如果只有背景，或者没有有效前景，直接返回原 mask
    if num_labels <= 1:
        return binary

    # 创建空白输出 mask
    output = np.zeros_like(binary, dtype=np.uint8)

    # 连通域编号从 1 开始，0 是背景
    component_ids = list(range(1, num_labels))

    # 读取每个连通域的面积
    areas = [stats[i, cv2.CC_STAT_AREA] for i in component_ids]

    # 只保留最大连通区域
    if keep_largest:
        # 找到面积最大的连通域
        largest_idx = int(np.argmax(areas))
        largest_label = component_ids[largest_idx]
        largest_area = areas[largest_idx]

        # 如果最大区域面积满足 min_area，则保留
        if min_area <= 0 or largest_area >= min_area:
            output[labels == largest_label] = 1

    # 保留所有面积大于 min_area 的连通域
    else:
        for comp_id, area in zip(component_ids, areas):
            if area >= min_area:
                output[labels == comp_id] = 1

    return output


def fill_mask_holes(mask: ArrayLike) -> np.ndarray:
    """
    填充 mask 内部空洞。

    基本思路：
    1. 将前景/背景反转，得到 background。
    2. 对 background 做连通域分析。
    3. 与图像边界相连的背景是真正外部背景。
    4. 不与边界相连的背景区域就是 mask 内部空洞。
    5. 将这些内部空洞填成前景。

    参数：
    mask:
        输入二值 mask。

    返回：
    output:
        填洞后的二值 mask，取值为 0 或 1。
    """

    # 转换为标准二值 mask
    binary = to_binary_mask(mask)

    # 反转背景区域
    # 原 mask 中 0 的位置视为背景
    background = (binary == 0).astype(np.uint8)

    # 对背景区域做连通域分析
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        background,
        connectivity=8
    )

    # 如果没有可分析的背景连通域，直接返回原 mask
    if num_labels <= 1:
        return binary

    # 记录所有与图像边界相连的背景连通域编号
    border_labels = set()

    # 上边界
    border_labels.update(np.unique(labels[0, :]).tolist())

    # 下边界
    border_labels.update(np.unique(labels[-1, :]).tolist())

    # 左边界
    border_labels.update(np.unique(labels[:, 0]).tolist())

    # 右边界
    border_labels.update(np.unique(labels[:, -1]).tolist())

    # 复制原 mask 作为输出
    output = binary.copy()

    # 遍历所有背景连通域
    # 0 是 connectedComponents 中的背景编号，这里从 1 开始即可
    for label_id in range(1, num_labels):
        # 如果某个背景连通域没有接触图像边界，
        # 说明它是目标内部的空洞，需要填成前景
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
    """
    对 mask 进行综合后处理。

    主要处理流程：
    1. 二值化 mask
    2. 去除小连通域
    3. 可选开运算，去除细小噪声
    4. 可选闭运算，连接断裂区域、平滑边界
    5. 可选填洞
    6. 限制外扩范围，避免 mask 过度膨胀
    7. 再次去除小连通域
    8. 再次填洞

    参数：
    mask:
        输入 mask。

    min_area:
        最小连通区域面积。

    keep_largest:
        是否只保留最大连通域。

    close_kernel_size:
        闭运算卷积核大小。
        闭运算 = 先膨胀再腐蚀，用于连接断裂区域和填补小缝隙。

    close_iterations:
        闭运算迭代次数。

    open_kernel_size:
        开运算卷积核大小。
        开运算 = 先腐蚀再膨胀，用于去除小噪点。
        为 0 或 1 时不执行开运算。

    open_iterations:
        开运算迭代次数。

    fill_holes:
        是否填充 mask 内部空洞。

    max_external_expand_px:
        限制 mask 最大外扩范围。
        0:
            不允许 mask 向原始 clean 区域外扩张。
        正整数:
            允许在 clean mask 外扩指定像素范围内保留新增区域。
        None:
            不限制外扩。

    返回：
    refined:
        后处理后的二值 mask，取值为 0 或 1。
    """

    # 第一步：将输入 mask 转为 0/1 二值 mask
    original = to_binary_mask(mask)

    # 第二步：去除小连通区域
    # 如果 keep_largest=True，则只保留最大连通域
    clean = remove_small_components(
        original,
        min_area=min_area,
        keep_largest=keep_largest,
    )

    # 第三步：可选开运算
    # 用于去除孤立噪声点、细小毛刺
    if open_kernel_size and open_kernel_size > 1:
        # 创建椭圆形开运算卷积核
        open_kernel = _ellipse_kernel(open_kernel_size)

        # 执行开运算：先腐蚀，再膨胀
        clean = cv2.morphologyEx(
            clean,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=max(1, int(open_iterations)),
        )

        # 再次二值化，保证输出为 0/1
        clean = to_binary_mask(clean)

    # 第四步：可选闭运算
    # 用于连接 mask 中断裂的区域，并填补较小裂缝
    if close_kernel_size and close_kernel_size > 1:
        # 创建椭圆形闭运算卷积核
        close_kernel = _ellipse_kernel(close_kernel_size)

        # 执行闭运算：先膨胀，再腐蚀
        closed = cv2.morphologyEx(
            clean,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=max(1, int(close_iterations)),
        )

        # 再次二值化
        closed = to_binary_mask(closed)

    # 如果不做闭运算，则直接使用 clean
    else:
        closed = clean.copy()

    # 第五步：可选填充内部空洞
    if fill_holes:
        filled = fill_mask_holes(closed)
    else:
        filled = closed.copy()

    # 第六步：限制 mask 外扩范围
    # 这是为了避免闭运算/填洞导致 mask 蔓延到不该覆盖的区域
    if max_external_expand_px is not None and max_external_expand_px >= 0:

        # 如果 max_external_expand_px == 0，
        # 只允许最终结果保留在 clean 原有区域内
        if max_external_expand_px == 0:
            allowed_external = clean.astype(bool)

        # 如果允许外扩，则对 clean 做一次膨胀
        # 膨胀后的区域就是允许最终 mask 存在的最大范围
        else:
            px = int(max_external_expand_px)

            # 创建允许外扩范围的结构元素
            allow_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * px + 1, 2 * px + 1),
            )

            # 对 clean 膨胀，得到允许区域
            allowed_external = cv2.dilate(
                clean,
                allow_kernel,
                iterations=1
            ).astype(bool)

        # 找出填洞新增的区域
        # filled 比 closed 多出来的部分，通常就是内部空洞被填充的区域
        holes_to_fill = filled.astype(bool) & (~closed.astype(bool))

        # 最终 refined：
        # 1. filled 中位于 allowed_external 范围内的区域可以保留
        # 2. holes_to_fill 也保留，避免内部空洞因为外扩限制被误删
        refined = (
            (filled.astype(bool) & allowed_external) |
            holes_to_fill
        ).astype(np.uint8)

    # 如果不限制外扩，则直接使用 filled
    else:
        refined = filled.astype(np.uint8)

    # 第七步：再次去除小连通域
    # 形态学操作后可能产生新的小区域，因此再过滤一次
    refined = remove_small_components(
        refined,
        min_area=min_area,
        keep_largest=keep_largest,
    )

    # 第八步：再次填洞
    # 防止连通域过滤后重新出现内部空洞
    if fill_holes:
        refined = fill_mask_holes(refined)

    # 返回 0/1 二值 mask
    return refined.astype(np.uint8)