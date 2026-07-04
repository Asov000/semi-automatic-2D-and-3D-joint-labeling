# -*- coding: utf-8 -*-
"""二维语义 mask 抑制模块，删除重叠、包含或重复的 mask。"""

from typing import Dict, List, Tuple

import numpy as np


def compute_mask_overlap(mask_a, mask_b) -> Dict:
    """
    计算两个二维 mask 之间的重叠关系。

    参数：
    mask_a:
        第一个 mask，可以是 0/1、0/255 或 bool 类型。

    mask_b:
        第二个 mask，可以是 0/1、0/255 或 bool 类型。

    返回：
    dict:
        {
            "intersection": 两个 mask 的交集面积,
            "iou": 两个 mask 的交并比 IoU,
            "overlap_min": 交集占较小 mask 面积的比例,
            "area_a": mask_a 的面积,
            "area_b": mask_b 的面积
        }

    说明：
    IoU 适合判断两个 mask 是否整体高度重合；
    overlap_min 更适合判断“小 mask 是否被大 mask 包含”。
    """

    # 将输入 mask 转成 bool 类型
    # 非零区域视为前景 True
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)

    # 计算交集面积：两个 mask 同时为 True 的像素数
    inter = int(np.logical_and(a, b).sum())

    # 分别计算两个 mask 的前景面积
    area_a = int(a.sum())
    area_b = int(b.sum())

    # 并集面积 = area_a + area_b - intersection
    union = area_a + area_b - inter

    # 较小 mask 的面积
    # max(..., 1) 是为了避免除以 0
    min_area = max(min(area_a, area_b), 1)

    return {
        # 交集面积
        "intersection": inter,

        # IoU = 交集 / 并集
        # max(union, 1) 防止 union 为 0 时除零
        "iou": float(inter / max(union, 1)),

        # 交集占较小 mask 的比例
        # 如果 overlap_min 接近 1，说明较小 mask 基本被较大 mask 包含
        "overlap_min": float(inter / min_area),

        # mask_a 面积
        "area_a": area_a,

        # mask_b 面积
        "area_b": area_b,
    }


def semantic_mask_nms(
    items: List[Dict],
    iou_threshold: float = 0.65,
    overlap_min_threshold: float = 0.80,
    suppress_same_class: bool = False,
    same_class_iou_threshold: float = 0.90,
    same_class_overlap_min_threshold: float = 0.95
) -> Tuple[List[Dict], Dict]:
    """
    对二维语义 mask 执行非极大值抑制。

    作用：
    删除高度重叠、互相包含或重复的 mask。

    参数：
    items:
        mask 结果列表。
        每个 item 通常包含：
        {
            "mask": 二值 mask,
            "score": 置信度,
            "class_name": 类别名称,
            ...
        }

    iou_threshold:
        不同类别之间的 IoU 抑制阈值。
        如果两个不同类别 mask 的 IoU 大于该值，则删除分数较低的那个。

    overlap_min_threshold:
        不同类别之间的包含关系抑制阈值。
        如果交集占较小 mask 面积的比例超过该值，则认为一个 mask 基本被另一个包含。

    suppress_same_class:
        是否抑制同类别 mask。
        False:
            同类别 mask 即使重叠，也默认不删除。
            适合同一类别可能存在多个实例的情况，例如多把 chair。
        True:
            同类别 mask 也会进行重复抑制。

    same_class_iou_threshold:
        同类别 mask 的 IoU 抑制阈值。
        一般设置得比不同类别更高，避免误删同类不同实例。

    same_class_overlap_min_threshold:
        同类别 mask 的包含关系抑制阈值。
        一般设置得更严格，比如 0.95。

    返回：
    filtered_items:
        NMS 后保留下来的 mask 结果列表。

    stats:
        NMS 统计信息，包括抑制前数量、抑制后数量、被删除项信息等。
    """

    # 如果 mask 数量小于等于 1，不需要做 NMS
    if len(items) <= 1:
        return items, {
            "enabled": True,
            "iou_threshold": float(iou_threshold),
            "overlap_min_threshold": float(overlap_min_threshold),
            "suppress_same_class": bool(suppress_same_class),
            "same_class_iou_threshold": float(same_class_iou_threshold),
            "same_class_overlap_min_threshold": float(same_class_overlap_min_threshold),
            "before": int(len(items)),
            "after": int(len(items)),
            "removed": [],
        }

    # 对所有 mask 进行排序
    # 排序优先级：
    # 1. score 越高越靠前
    # 2. 如果 score 一样，面积越大越靠前
    #
    # 这样高置信度、大面积的 mask 会优先被保留。
    order = sorted(
        range(len(items)),
        key=lambda i: (
            float(items[i].get("score", 1.0)),
            int(np.asarray(items[i]["mask"]).astype(bool).sum())
        ),
        reverse=True
    )

    # 保存最终保留的 item 索引
    kept_indices = []

    # 保存被删除的 item 信息
    removed = []

    # 按照 score / 面积排序后的顺序遍历每个 mask
    for idx in order:
        item = items[idx]

        # 当前 mask 的类别名
        class_name = str(item.get("class_name", ""))

        # 如果当前 mask 被抑制，这里会记录抑制原因
        suppress_info = None

        # 与已经保留下来的 mask 逐个比较
        for kept_idx in kept_indices:
            kept_item = items[kept_idx]

            # 已保留 mask 的类别名
            kept_class_name = str(kept_item.get("class_name", ""))

            # 判断当前 mask 和已保留 mask 是否属于同一类别
            same_class = kept_class_name == class_name

            # 如果是同类别，并且配置为不抑制同类别，
            # 则直接跳过，避免误删同一类别的多个实例。
            if same_class and not suppress_same_class:
                continue

            # 计算当前 mask 和已保留 mask 的重叠关系
            overlap = compute_mask_overlap(
                item["mask"],
                kept_item["mask"]
            )

            # 同类别 mask 使用更严格的阈值
            if same_class:
                should_suppress = (
                    overlap["iou"] >= float(same_class_iou_threshold)
                    or overlap["overlap_min"] >= float(same_class_overlap_min_threshold)
                )

            # 不同类别 mask 使用普通阈值
            else:
                should_suppress = (
                    overlap["iou"] >= float(iou_threshold)
                    or overlap["overlap_min"] >= float(overlap_min_threshold)
                )

            # 如果满足抑制条件，则记录删除信息
            if should_suppress:
                suppress_info = {
                    # 被删除 mask 的原始索引
                    "removed_index": int(idx),

                    # 被删除 mask 的类别
                    "removed_class": class_name,

                    # 被删除 mask 的置信度
                    "removed_score": float(item.get("score", 1.0)),

                    # 保留下来的 mask 的原始索引
                    "kept_index": int(kept_idx),

                    # 保留下来的 mask 的类别
                    "kept_class": kept_class_name,

                    # 保留下来的 mask 的置信度
                    "kept_score": float(kept_item.get("score", 1.0)),

                    # 两个 mask 的 IoU
                    "iou": overlap["iou"],

                    # 交集占较小 mask 的比例
                    "overlap_min": overlap["overlap_min"],

                    # 交集面积
                    "intersection": overlap["intersection"],

                    # 删除原因
                    "reason": "semantic_mask_nms_overlap",
                }

                # 当前 mask 已经确定要删除，不再继续比较其他 kept mask
                break

        # 如果没有被任何已保留 mask 抑制，则保留当前 mask
        if suppress_info is None:
            kept_indices.append(idx)

        # 否则记录删除信息
        else:
            removed.append(suppress_info)

    # 将保留索引按原始顺序排序
    # 这样 filtered_items 的顺序和输入 items 的顺序更一致
    kept_indices = sorted(kept_indices)

    # 根据保留索引取出最终 mask 结果
    filtered_items = [items[idx] for idx in kept_indices]

    # 生成 NMS 统计信息
    stats = {
        "enabled": True,
        "iou_threshold": float(iou_threshold),
        "overlap_min_threshold": float(overlap_min_threshold),
        "suppress_same_class": bool(suppress_same_class),
        "same_class_iou_threshold": float(same_class_iou_threshold),
        "same_class_overlap_min_threshold": float(same_class_overlap_min_threshold),
        "before": int(len(items)),
        "after": int(len(filtered_items)),
        "removed": removed,
    }

    return filtered_items, stats