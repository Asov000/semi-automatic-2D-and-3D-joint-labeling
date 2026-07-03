# -*- coding: utf-8 -*-

import os
import re
import json
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import scipy.io as sio


# ============================================================
# 基础工具
# ============================================================

def to_binary_mask(mask: Union[np.ndarray, List[List[int]]], threshold: float = 0) -> np.ndarray:
    """
    将输入 mask 转成 0/1 uint8 二值图。
    """
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def jsonable(obj):
    """
    将 numpy 类型转成 json 可保存类型。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    return obj


# ============================================================
# 读取 SUNRGBD 点云 / 标定
# ============================================================

def load_mat_points(mat_path: str) -> np.ndarray:

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"点云文件不存在: {mat_path}")

    try:
        data = sio.loadmat(mat_path)

        if "points3d_rgb" in data:
            return np.asarray(data["points3d_rgb"], dtype=np.float64)

        for key, value in data.items():
            if key.startswith("__"):
                continue

            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")
                return arr[:, :6].astype(np.float64)

        raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")

    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            if "points3d_rgb" in f:
                arr = np.array(f["points3d_rgb"])

                if arr.shape[0] == 6:
                    arr = arr.T

                return arr.astype(np.float64)

            for key in f.keys():
                arr = np.array(f[key])

                if arr.ndim == 2:
                    if arr.shape[0] == 6:
                        arr = arr.T

                    if arr.shape[1] >= 6:
                        print(f"[load_mat_points] 自动使用变量: {key}, shape={arr.shape}")
                        return arr[:, :6].astype(np.float64)

            raise KeyError(f"在 {mat_path} 中没有找到 points3d_rgb 或 N x 6 数组")


def load_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 calib。
    order='F'。
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    calib = np.loadtxt(calib_path)

    if calib.shape != (2, 9):
        raise ValueError(f"calib 文件格式不对，期望 shape=(2,9)，实际是 {calib.shape}")

    Rtilt = calib[0].reshape(3, 3, order="F")
    K = calib[1].reshape(3, 3, order="F")

    return Rtilt, K


# ============================================================
# 3D 点云投影回 2D 图像
# ============================================================

def project_points_to_image_with_indices(
    points3d: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int, int],
    Rtilt: Optional[np.ndarray] = None,
    points_are_after_rtilt: bool = True,
    use_matlab_pixel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将 SUNRGBD 保存的 3D 点云投影回 2D 图像。

    输入：
        points3d:
            N x 3 点云。
            如果来自 points3d_rgb[:, 0:3]，通常已经经过 Rtilt。

        K:
            3 x 3 内参矩阵。

        image_shape:
            原图 shape，例如 image.shape。

        Rtilt:
            SUNRGBD tilt 矫正矩阵。
            如果 points_are_after_rtilt=True，必须传入。

        points_are_after_rtilt:
            True:
                points3d 是保存后的点云，已经经过 Rtilt。
                需要先 inv(Rtilt) 还原到投影坐标。
            False:
                points3d 是未经过 Rtilt 的中间点云。

        use_matlab_pixel:
            True 表示投影后 u,v 减 1，用于对齐 OpenCV 像素坐标。

    返回：
        uv_int:
            M x 2，有效投影点像素坐标，[u, v]。

        depth:
            M 个投影深度。

        point_indices:
            M 个索引，表示这些投影点对应原始 points3d 中的哪一行。

        uv_float:
            M x 2，未 round 的浮点投影坐标。
    """
    H, W = image_shape[:2]

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)

    finite_mask = np.isfinite(points3d).all(axis=1)
    finite_indices = np.where(finite_mask)[0]
    points = points3d[finite_mask]

    if points_are_after_rtilt:
        if Rtilt is None:
            raise ValueError("points_are_after_rtilt=True 时必须传入 Rtilt")

        Rtilt = np.asarray(Rtilt, dtype=np.float64)
        points_before_rtilt = (np.linalg.inv(Rtilt) @ points.T).T
    else:
        points_before_rtilt = points

    # SUNRGBD read_3d_pts_general 里的坐标定义：
    # points3d = [x3, z3, -y3]
    # 因此投影时：
    # X_cam = x3
    # Y_cam = y3 = -points[:, 2]
    # Z_cam = z3
    X = points_before_rtilt[:, 0]
    Z = points_before_rtilt[:, 1]
    Y = -points_before_rtilt[:, 2]

    valid_z = Z > 1e-6

    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]

    valid_indices = finite_indices[valid_z]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    if use_matlab_pixel:
        u = u - 1
        v = v - 1

    uv_float = np.stack([u, v], axis=1)

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    inside = (
        (u_int >= 0) & (u_int < W) &
        (v_int >= 0) & (v_int < H)
    )

    uv_int = np.stack([u_int[inside], v_int[inside]], axis=1)
    depth = Z[inside]
    point_indices = valid_indices[inside]
    uv_float = uv_float[inside]

    return uv_int, depth, point_indices, uv_float


def filter_visible_points_by_zbuffer(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, int, int],
    tolerance: float = 0.03
) -> np.ndarray:
    """
    基于 z-buffer 保留每个像素附近最近的 3D 点。
    """
    H, W = image_shape[:2]

    uv = np.asarray(uv)
    depth = np.asarray(depth)

    u = uv[:, 0]
    v = uv[:, 1]

    linear_idx = v * W + u

    zbuffer = np.full(H * W, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, linear_idx, depth)

    min_depth_at_point_pixel = zbuffer[linear_idx]

    visible_mask = depth <= min_depth_at_point_pixel + tolerance

    return visible_mask


# ============================================================
# 读取 SAM 前端导出的 2D mask
# ============================================================

def find_classes_json(segmentation_dir: str) -> Optional[str]:
    for name in os.listdir(segmentation_dir):
        if name.endswith("_classes.json"):
            return os.path.join(segmentation_dir, name)

    return None


def load_detection_class_mapping(segmentation_dir: str) -> Dict[str, int]:
    """
    读取前端导出的 detection_class_to_id。
    如果没有 classes.json，则返回空 dict，后续自动生成类别 id。
    """
    json_path = find_classes_json(segmentation_dir)

    if json_path is None:
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    mapping = info.get("detection_class_to_id", {})

    if mapping is None:
        mapping = {}

    return {str(k): int(v) for k, v in mapping.items()}


def recover_class_name_from_safe_name(
    safe_name: str,
    class_to_id: Dict[str, int]
) -> str:
    """
    前端保存 binary mask 时会把空格、斜杠替换成下划线。。
    """
    if safe_name in class_to_id:
        return safe_name

    for class_name in class_to_id.keys():
        safe = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        if safe == safe_name:
            return class_name

    return safe_name


def load_binary_masks_from_segmentation_dir(
    segmentation_dir: str,
    image_name: Optional[str] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    读取前端导出的 binary_masks。
    """
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"segmentation_dir 不存在: {segmentation_dir}")

    binary_dir = os.path.join(segmentation_dir, "binary_masks")

    if not os.path.isdir(binary_dir):
        raise FileNotFoundError(f"binary_masks 目录不存在: {binary_dir}")

    class_to_id = load_detection_class_mapping(segmentation_dir)

    mask_files = [
        name for name in os.listdir(binary_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    mask_files = sorted(mask_files)

    masks = []
    next_class_id = 0 if len(class_to_id) == 0 else max(class_to_id.values()) + 1

    for default_instance_id, name in enumerate(mask_files, start=1):
        path = os.path.join(binary_dir, name)

        mask_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask_gray is None:
            print(f"[load_binary_masks] 跳过无法读取的 mask: {path}")
            continue

        mask = to_binary_mask(mask_gray)

        if int(mask.sum()) == 0:
            print(f"[load_binary_masks] 跳过空 mask: {path}")
            continue

        stem = os.path.splitext(name)[0]

        instance_id = default_instance_id
        safe_class_name = None

        if image_name is not None:
            prefix_pattern = f"{re.escape(image_name)}_"
            if re.match(prefix_pattern, stem):
                remain = stem[len(image_name) + 1:]

                # 形如 001_chair
                parts = remain.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    instance_id = int(parts[0])
                    safe_class_name = parts[1]

        if safe_class_name is None:
            # 通用匹配：xxx_001_chair
            m = re.match(r"^(.+)_([0-9]+)_(.+)$", stem)
            if m is not None:
                instance_id = int(m.group(2))
                safe_class_name = m.group(3)
            else:
                safe_class_name = f"object_{default_instance_id}"

        class_name = recover_class_name_from_safe_name(safe_class_name, class_to_id)

        if class_name not in class_to_id:
            class_to_id[class_name] = next_class_id
            next_class_id += 1

        class_id = class_to_id[class_name]

        masks.append({
            "instance_id": int(instance_id),
            "class_name": str(class_name),
            "class_id": int(class_id),
            "mask": mask.astype(np.uint8),
            "mask_path": path,
        })

    return masks, class_to_id


# ============================================================
# 2D mask -> 3D 点云类别 mask / 实例 mask
# ============================================================

def assign_3d_points_to_2d_masks(
    points3d: np.ndarray,
    K: np.ndarray,
    Rtilt: np.ndarray,
    image_shape: Tuple[int, int, int],
    masks_2d: List[Dict],
    points_are_after_rtilt: bool = True,
    use_matlab_pixel: bool = True,
    use_zbuffer: bool = True,
    zbuffer_tolerance: float = 0.03,
    background_class_id: int = -1,
    overlap_policy: str = "later"
) -> Dict:
    """
    核心函数：
    将 3D 点投影到 2D 图像，然后根据 2D mask 给 3D 点打类别标签。

    参数：
        points3d:
            N x 3 点云。
            通常是 points3d_rgb[:, 0:3]，已经经过 Rtilt。

        masks_2d:
            2D mask 列表，每个元素需要：
                {
                    "instance_id": int,
                    "class_name": str,
                    "class_id": int,
                    "mask": H x W 0/1
                }

        overlap_policy:
            "later":
                如果多个 mask 重叠，后面的 mask 覆盖前面的。
                和你前端保存 semantic_mask 的 later 逻辑一致。
            "first":
                如果多个 mask 重叠，先出现的 mask 优先。

    返回：
        result:
            {
                "point_class_ids": N,
                "point_instance_ids": N,
                "valid_projected_mask": N bool,
                "visible_projected_mask": N bool,
                "segments": [...]
                "uv": M x 2,
                "depth": M,
                "point_indices": M
            }
    """
    if overlap_policy not in ["later", "first"]:
        raise ValueError("overlap_policy 只能是 'later' 或 'first'")

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    num_points = points3d.shape[0]

    uv, depth, point_indices, uv_float = project_points_to_image_with_indices(
        points3d=points3d,
        K=K,
        image_shape=image_shape,
        Rtilt=Rtilt,
        points_are_after_rtilt=points_are_after_rtilt,
        use_matlab_pixel=use_matlab_pixel
    )

    valid_projected_mask = np.zeros(num_points, dtype=bool)
    valid_projected_mask[point_indices] = True

    if use_zbuffer:
        visible_local_mask = filter_visible_points_by_zbuffer(
            uv=uv,
            depth=depth,
            image_shape=image_shape,
            tolerance=zbuffer_tolerance
        )

        uv = uv[visible_local_mask]
        depth = depth[visible_local_mask]
        point_indices = point_indices[visible_local_mask]
        uv_float = uv_float[visible_local_mask]
    else:
        visible_local_mask = np.ones(len(point_indices), dtype=bool)

    visible_projected_mask = np.zeros(num_points, dtype=bool)
    visible_projected_mask[point_indices] = True

    point_class_ids = np.full(num_points, background_class_id, dtype=np.int32)
    point_instance_ids = np.zeros(num_points, dtype=np.int32)

    instance_meta = {}

    for item in masks_2d:
        mask = to_binary_mask(item["mask"])

        class_name = str(item["class_name"])
        class_id = int(item["class_id"])
        instance_id = int(item["instance_id"])

        u = uv[:, 0]
        v = uv[:, 1]

        hit = mask[v, u] > 0

        if overlap_policy == "first":
            hit = hit & (point_instance_ids[point_indices] == 0)

        selected_indices = point_indices[hit]

        if selected_indices.size == 0:
            continue

        point_class_ids[selected_indices] = class_id
        point_instance_ids[selected_indices] = instance_id

        instance_meta[instance_id] = {
            "instance_id": instance_id,
            "class_name": class_name,
            "class_id": class_id,
            "mask_path": item.get("mask_path", None),
        }

    segments = []

    for instance_id in sorted(instance_meta.keys()):
        indices = np.where(point_instance_ids == instance_id)[0]

        if indices.size == 0:
            continue

        meta = instance_meta[instance_id]

        segments.append({
            "instance_id": int(instance_id),
            "class_name": meta["class_name"],
            "class_id": int(meta["class_id"]),
            "point_indices": indices,
            "num_points": int(indices.size),
            "mask_path": meta.get("mask_path", None),
        })

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "valid_projected_mask": valid_projected_mask,
        "visible_projected_mask": visible_projected_mask,
        "segments": segments,
        "uv": uv,
        "uv_float": uv_float,
        "depth": depth,
        "point_indices": point_indices,
    }


# ============================================================
# 3D 语义图离群点滤波
# ============================================================

def get_semantic_filter_params(strength: int) -> Dict:
    """
    3D 语义图滤波强度参数。

    strength=0：不滤波。
    strength=1：轻度去除孤立点，尽量不破坏小物体。
    strength=2：中等强度，推荐默认。
    strength=3：强滤波，只保留每个实例的主连通区域，适合明显有飞点时使用。
    """
    strength = int(strength)

    table = {
        0: {
            "enabled": False,
            "radius": 0.00,
            "min_neighbors": 0,
            "min_component_points": 0,
            "min_component_ratio": 0.00,
            "keep_largest_only": False,
        },
        1: {
            "enabled": True,
            "radius": 0.08,
            "min_neighbors": 4,
            "min_component_points": 8,
            "min_component_ratio": 0.005,
            "keep_largest_only": False,
        },
        2: {
            "enabled": True,
            "radius": 0.10,
            "min_neighbors": 6,
            "min_component_points": 15,
            "min_component_ratio": 0.010,
            "keep_largest_only": False,
        },
        3: {
            "enabled": True,
            "radius": 0.15,
            "min_neighbors": 8,
            "min_component_points": 25,
            "min_component_ratio": 0.020,
            "keep_largest_only": True,
        },
    }

    if strength not in table:
        raise ValueError("semantic_filter_strength 只能是 0, 1, 2, 3")

    params = dict(table[strength])
    params["strength"] = strength
    return params


def _connected_components_from_neighbor_lists(neighbor_lists: List[List[int]], valid_local_mask: np.ndarray) -> List[np.ndarray]:
    """
    在局部点集上，根据半径邻接关系计算连通分量。

    neighbor_lists:
        cKDTree.query_ball_point 返回的邻接表，索引是局部索引。

    valid_local_mask:
        只有 True 的局部点参与连通域计算。
    """
    valid_local_mask = np.asarray(valid_local_mask, dtype=bool)
    n = int(valid_local_mask.shape[0])
    visited = np.zeros(n, dtype=bool)
    components = []

    valid_set = set(np.where(valid_local_mask)[0].tolist())

    for start in list(valid_set):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            for nb in neighbor_lists[cur]:
                if nb in valid_set and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        components.append(np.asarray(comp, dtype=np.int64))

    components.sort(key=lambda x: x.size, reverse=True)
    return components


def filter_semantic_outliers_3d(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    segments: List[Dict],
    strength: int = 0,
    background_class_id: int = -1,
    min_points: int = 30
) -> Dict:
    """
    对 3D 语义点云进行离群点滤波。

    核心思想：
        对每个实例单独处理，使用 3D 半径邻域判断孤立点，
        再用空间连通域去掉很小的漂浮语义块。

    返回：
        {
            "point_class_ids": filtered_class_ids,
            "point_instance_ids": filtered_instance_ids,
            "segments": filtered_segments,
            "filter_stats": {...}
        }
    """
    strength = int(strength)
    params = get_semantic_filter_params(strength)

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32).copy()
    point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32).copy()

    before_labeled = int((point_class_ids >= 0).sum())

    if not params["enabled"] or before_labeled == 0:
        return {
            "point_class_ids": point_class_ids,
            "point_instance_ids": point_instance_ids,
            "segments": segments,
            "filter_stats": {
                "semantic_filter_strength": strength,
                "enabled": False,
                "before_labeled_points": before_labeled,
                "after_labeled_points": before_labeled,
                "removed_labeled_points": 0,
                "removed_ratio": 0.0,
            }
        }

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise ImportError(
            "3D 语义图滤波需要 scipy.spatial.cKDTree。请确认 scipy 已安装。"
        ) from e

    radius = float(params["radius"])
    min_neighbors = int(params["min_neighbors"])
    min_component_points = int(params["min_component_points"])
    min_component_ratio = float(params["min_component_ratio"])
    keep_largest_only = bool(params["keep_largest_only"])

    removed_total = 0
    filtered_segments = []
    segment_stats = []

    for seg in segments:
        instance_id = int(seg["instance_id"])
        class_id = int(seg["class_id"])

        # 以实例为单位处理。这样不会把两个同类但空间分离的物体误合并。
        obj_indices = np.where(point_instance_ids == instance_id)[0]
        raw_n = int(obj_indices.size)

        if raw_n == 0:
            continue

        # 点太少时不做半径连通滤波，避免小目标被直接吃掉。
        if raw_n < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })
            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        obj_points = points3d[obj_indices]
        finite = np.isfinite(obj_points).all(axis=1)

        if finite.sum() < max(min_points, min_component_points * 2):
            filtered_segments.append({
                **seg,
                "point_indices": obj_indices,
                "num_points": raw_n,
                "semantic_filter_skipped": True,
            })
            segment_stats.append({
                "instance_id": instance_id,
                "class_id": class_id,
                "before": raw_n,
                "after": raw_n,
                "removed": 0,
                "skipped": True,
            })
            continue

        # 对非有限点直接视为离群点。
        finite_local_indices = np.where(finite)[0]
        finite_points = obj_points[finite]

        tree = cKDTree(finite_points)
        finite_neighbor_lists = tree.query_ball_point(finite_points, r=radius)
        finite_neighbor_counts = np.array([len(x) for x in finite_neighbor_lists], dtype=np.int32)

        dense_finite_mask = finite_neighbor_counts >= min_neighbors

        # 映射回 obj_points 的局部索引。
        dense_local_mask = np.zeros(raw_n, dtype=bool)
        dense_local_mask[finite_local_indices[dense_finite_mask]] = True

        # 为了复用邻接表，需要构建 raw_n 长度的邻接表，邻接索引为 obj_points 局部索引。
        neighbor_lists = [[] for _ in range(raw_n)]
        finite_to_raw = finite_local_indices
        for finite_i, raw_i in enumerate(finite_to_raw):
            neighbor_lists[int(raw_i)] = [int(finite_to_raw[j]) for j in finite_neighbor_lists[finite_i]]

        components = _connected_components_from_neighbor_lists(
            neighbor_lists=neighbor_lists,
            valid_local_mask=dense_local_mask
        )

        if len(components) == 0:
            # 极端情况下不全删，保留原实例，避免误伤整物体。
            keep_local = np.ones(raw_n, dtype=bool)
        else:
            if keep_largest_only:
                keep_components = [components[0]]
            else:
                threshold = max(min_component_points, int(round(raw_n * min_component_ratio)))
                keep_components = [comp for comp in components if comp.size >= threshold]

                # 如果阈值过严导致全无，则至少保留最大的主体。
                if len(keep_components) == 0:
                    keep_components = [components[0]]

            keep_local = np.zeros(raw_n, dtype=bool)
            for comp in keep_components:
                keep_local[comp] = True

        keep_global = obj_indices[keep_local]
        remove_global = obj_indices[~keep_local]

        if remove_global.size > 0:
            point_class_ids[remove_global] = background_class_id
            point_instance_ids[remove_global] = 0

        removed = int(remove_global.size)
        after_n = int(keep_global.size)
        removed_total += removed

        if after_n > 0:
            filtered_segments.append({
                **seg,
                "point_indices": keep_global,
                "num_points": after_n,
                "raw_num_points_before_semantic_filter": raw_n,
                "semantic_filter_removed_points": removed,
                "semantic_filter_strength": strength,
            })

        segment_stats.append({
            "instance_id": instance_id,
            "class_id": class_id,
            "before": raw_n,
            "after": after_n,
            "removed": removed,
            "skipped": False,
        })

    after_labeled = int((point_class_ids >= 0).sum())

    return {
        "point_class_ids": point_class_ids,
        "point_instance_ids": point_instance_ids,
        "segments": filtered_segments,
        "filter_stats": {
            "semantic_filter_strength": strength,
            "enabled": True,
            "radius": radius,
            "min_neighbors": min_neighbors,
            "min_component_points": min_component_points,
            "min_component_ratio": min_component_ratio,
            "keep_largest_only": keep_largest_only,
            "before_labeled_points": before_labeled,
            "after_labeled_points": after_labeled,
            "removed_labeled_points": int(before_labeled - after_labeled),
            "removed_ratio": float((before_labeled - after_labeled) / max(before_labeled, 1)),
            "segment_stats": segment_stats,
        }
    }


# ============================================================
# 3D 框计算
# ============================================================

def filter_points_by_percentile(
    points: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 10:
        return points

    low = np.percentile(points, lower_percentile, axis=0)
    high = np.percentile(points, upper_percentile, axis=0)

    keep = np.all((points >= low) & (points <= high), axis=1)

    if keep.sum() < 5:
        return points

    return points[keep]


def compute_aabb_3d(points: np.ndarray) -> Dict:
    """
    计算轴对齐 3D 框。

    返回：
        center:
            [cx, cy, cz]

        size:
            [dx, dy, dz]

        corners:
            8 个角点
    """
    points = np.asarray(points, dtype=np.float64)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    center = (min_xyz + max_xyz) / 2.0
    size = max_xyz - min_xyz

    xmin, ymin, zmin = min_xyz
    xmax, ymax, zmax = max_xyz

    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)

    return {
        "box_type": "AABB",
        "center": center,
        "size": size,
        "heading_angle": 0.0,
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "corners": corners,
    }


def compute_pca_obb_3d(
    points: np.ndarray,
    up_axis: int = 2
) -> Dict:
    """
    使用 PCA 在水平面上估计 3D 有向框。

    默认 up_axis=2：
        对 SUNRGBD / Frustum-ConvNet 常用 upright_depth 坐标：
            x: 水平左右
            y: 水平前后
            z: 竖直方向

    计算逻辑：
        1. 在水平面 x-y 上做 PCA；
        2. 第一主方向作为 box 朝向；
        3. z 方向直接取 min/max；
        4. 得到类似 SUNRGBD 的：
            center_x center_y center_z
            length width height
            heading_angle
    """
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 3:
        return compute_aabb_3d(points)

    if up_axis not in [0, 1, 2]:
        raise ValueError("up_axis 只能是 0, 1, 2")

    horizontal_axes = [i for i in range(3) if i != up_axis]

    xy = points[:, horizontal_axes]
    z = points[:, up_axis]

    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean

    cov = np.cov(xy_centered.T)

    if not np.isfinite(cov).all():
        return compute_aabb_3d(points)

    eig_vals, eig_vecs = np.linalg.eigh(cov)

    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]

    # 保证局部坐标系方向稳定
    if np.linalg.det(eig_vecs) < 0:
        eig_vecs[:, 1] *= -1

    local_xy = xy_centered @ eig_vecs

    min_local = local_xy.min(axis=0)
    max_local = local_xy.max(axis=0)

    center_local = (min_local + max_local) / 2.0
    size_local = max_local - min_local

    center_xy = xy_mean + center_local @ eig_vecs.T

    zmin = z.min()
    zmax = z.max()
    center_z = (zmin + zmax) / 2.0
    height = zmax - zmin

    center = np.zeros(3, dtype=np.float64)
    center[horizontal_axes] = center_xy
    center[up_axis] = center_z

    # 第一主方向
    main_dir = eig_vecs[:, 0]

    # heading_angle 只在默认水平轴为 x-y 时有明确意义
    heading_angle = math.atan2(main_dir[1], main_dir[0])

    # size 的语义：
    # [length, width, height]
    size = np.array([
        size_local[0],
        size_local[1],
        height
    ], dtype=np.float64)

    # 计算 8 个角点
    lx_min, ly_min = min_local
    lx_max, ly_max = max_local

    local_corners_2d = np.array([
        [lx_min, ly_min],
        [lx_max, ly_min],
        [lx_max, ly_max],
        [lx_min, ly_max],
    ], dtype=np.float64)

    world_corners_2d = xy_mean + local_corners_2d @ eig_vecs.T

    corners = []

    for zz in [zmin, zmax]:
        for p2d in world_corners_2d:
            p3d = np.zeros(3, dtype=np.float64)
            p3d[horizontal_axes] = p2d
            p3d[up_axis] = zz
            corners.append(p3d)

    corners = np.asarray(corners, dtype=np.float64)

    return {
        "box_type": "PCA_OBB",
        "center": center,
        "size": size,
        "heading_angle": float(heading_angle),
        "horizontal_axes": horizontal_axes,
        "up_axis": int(up_axis),
        "main_direction_2d": main_dir,
        "corners": corners,
    }


def build_3d_boxes_from_segments(
    points3d: np.ndarray,
    segments: List[Dict],
    box_type: str = "pca",
    min_points: int = 30,
    up_axis: int = 2,
    use_percentile_filter: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> List[Dict]:
    """
    根据每个 3D 实例点集生成 3D 标注框。

    参数：
        box_type:
            "aabb":
                轴对齐框。
            "pca":
                PCA 近似有向框。

        min_points:
            点数少于该值的实例不生成 3D 框。

        use_percentile_filter:
            是否使用分位数过滤离群点。
    """
    if box_type not in ["aabb", "pca"]:
        raise ValueError("box_type 只能是 'aabb' 或 'pca'")

    points3d = np.asarray(points3d, dtype=np.float64)

    boxes = []

    for seg in segments:
        point_indices = np.asarray(seg["point_indices"], dtype=np.int64)

        if point_indices.size < min_points:
            continue

        instance_points = points3d[point_indices]

        finite = np.isfinite(instance_points).all(axis=1)
        instance_points = instance_points[finite]

        if instance_points.shape[0] < min_points:
            continue

        raw_num_points = int(instance_points.shape[0])

        if use_percentile_filter:
            box_points = filter_points_by_percentile(
                instance_points,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile
            )
        else:
            box_points = instance_points

        if box_points.shape[0] < min_points:
            box_points = instance_points

        if box_type == "aabb":
            box = compute_aabb_3d(box_points)
        else:
            box = compute_pca_obb_3d(box_points, up_axis=up_axis)

        box.update({
            "instance_id": int(seg["instance_id"]),
            "class_name": str(seg["class_name"]),
            "class_id": int(seg["class_id"]),
            "num_points": int(box_points.shape[0]),
            "raw_num_points": raw_num_points,
            "mask_path": seg.get("mask_path", None),
        })

        boxes.append(box)

    return boxes


# ============================================================
# 保存 3D 点云 mask / 3D 框
# ============================================================

def save_point_masks(
    save_dir: str,
    sample_name: str,
    point_class_ids: np.ndarray,
    point_instance_ids: np.ndarray,
    valid_projected_mask: Optional[np.ndarray] = None,
    visible_projected_mask: Optional[np.ndarray] = None
) -> str:
    """
    保存点云类别 mask 和实例 mask。
    """
    ensure_dir(save_dir)

    save_path = os.path.join(save_dir, f"{sample_name}_point_masks.npz")

    np.savez_compressed(
        save_path,
        point_class_ids=point_class_ids.astype(np.int32),
        point_instance_ids=point_instance_ids.astype(np.int32),
        valid_projected_mask=valid_projected_mask.astype(bool) if valid_projected_mask is not None else None,
        visible_projected_mask=visible_projected_mask.astype(bool) if visible_projected_mask is not None else None,
    )

    return save_path


def save_3d_boxes_json(
    save_path: str,
    boxes: List[Dict],
    extra_info: Optional[Dict] = None
) -> None:
    """
    保存 3D 框为 JSON。
    """
    ensure_dir(os.path.dirname(save_path))

    data = {
        "boxes": boxes
    }

    if extra_info is not None:
        data["extra_info"] = extra_info

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=4)


def save_3d_boxes_txt(
    save_path: str,
    boxes: List[Dict]
) -> None:
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", encoding="utf-8") as f:
        for box in boxes:
            class_name = str(box["class_name"])
            class_id = int(box["class_id"])
            instance_id = int(box["instance_id"])

            center = np.asarray(box["center"], dtype=np.float64)
            size = np.asarray(box["size"], dtype=np.float64)
            heading = float(box.get("heading_angle", 0.0))
            num_points = int(box.get("num_points", 0))

            line = (
                f"{class_name} "
                f"{class_id:d} "
                f"{instance_id:d} "
                f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f} "
                f"{heading:.6f} "
                f"{num_points:d}"
            )

            f.write(line + "\n")


def save_labeled_pointcloud_ply(
    save_path: str,
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    class_id_to_color: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> None:
    ensure_dir(os.path.dirname(save_path))

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)

    if class_id_to_color is None:
        class_id_to_color = {}

    def _color_to_uint8(color):
        arr = np.asarray(color, dtype=np.float64).reshape(3)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)

    default_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
    ]

    colors = np.zeros((points3d.shape[0], 3), dtype=np.uint8)
    colors[:, :] = np.array([160, 160, 160], dtype=np.uint8)

    unique_class_ids = sorted([int(x) for x in np.unique(point_class_ids) if int(x) >= 0])

    for cid in unique_class_ids:
        # 兼容 {0: color} 和 {"0": color} 两种 key
        if cid in class_id_to_color:
            color = _color_to_uint8(class_id_to_color[cid])
        elif str(cid) in class_id_to_color:
            color = _color_to_uint8(class_id_to_color[str(cid)])
        else:
            # 关键修复：用 cid 取色，而不是 enumerate 后的 i
            color = np.array(default_colors[cid % len(default_colors)], dtype=np.uint8)

        colors[point_class_ids == cid] = color

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points3d.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points3d, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

def generate_3d_annotations_for_one_sample(
    root_dir: str,
    image_id: int,
    segmentation_dir: str,
    save_root: Optional[str] = None,
    box_type: str = "pca",
    min_points: int = 30,
    up_axis: int = 2,
    use_zbuffer: bool = True,
    zbuffer_tolerance: float = 0.03,
    use_percentile_filter: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    semantic_filter_strength: int = 0,
    save_ply: bool = True,
    save_point_mask: bool = True,
    save_boxes: bool = True
) -> Dict:
    """
    对单张 SUNRGBD 样本生成 3D 点云 mask 和 3D 标注框。
    """
    sample_name = f"{image_id:06d}"

    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    image_path = os.path.join(root_dir, "image", sample_name + ".jpg")
    calib_path = os.path.join(root_dir, "calib", sample_name + ".txt")

    if save_root is None:
        save_root = os.path.join(root_dir, "annotation_3d")

    save_dir = os.path.join(save_root, sample_name)
    ensure_dir(save_dir)

    print("[3DTool] 读取点云:", pc_path)
    print("[3DTool] 读取图像:", image_path)
    print("[3DTool] 读取标定:", calib_path)
    print("[3DTool] 读取2D mask:", segmentation_dir)

    points3d_rgb = load_mat_points(pc_path)
    points3d = points3d_rgb[:, 0:3]

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"图像读取失败: {image_path}")

    Rtilt, K = load_calib(calib_path)

    masks_2d, class_to_id = load_binary_masks_from_segmentation_dir(
        segmentation_dir=segmentation_dir,
        image_name=sample_name
    )

    if len(masks_2d) == 0:
        raise RuntimeError(f"没有读取到有效 2D mask: {segmentation_dir}")

    assign_result = assign_3d_points_to_2d_masks(
        points3d=points3d,
        K=K,
        Rtilt=Rtilt,
        image_shape=image.shape,
        masks_2d=masks_2d,
        points_are_after_rtilt=True,
        use_matlab_pixel=True,
        use_zbuffer=use_zbuffer,
        zbuffer_tolerance=zbuffer_tolerance,
        background_class_id=-1,
        overlap_policy="later"
    )

    semantic_filter_result = filter_semantic_outliers_3d(
        points3d=points3d,
        point_class_ids=assign_result["point_class_ids"],
        point_instance_ids=assign_result["point_instance_ids"],
        segments=assign_result["segments"],
        strength=semantic_filter_strength,
        background_class_id=-1,
        min_points=min_points
    )

    assign_result["point_class_ids"] = semantic_filter_result["point_class_ids"]
    assign_result["point_instance_ids"] = semantic_filter_result["point_instance_ids"]
    assign_result["segments"] = semantic_filter_result["segments"]
    semantic_filter_stats = semantic_filter_result["filter_stats"]

    boxes = build_3d_boxes_from_segments(
        points3d=points3d,
        segments=assign_result["segments"],
        box_type=box_type,
        min_points=min_points,
        up_axis=up_axis,
        use_percentile_filter=use_percentile_filter,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile
    )

    point_mask_path = None
    boxes_json_path = None
    boxes_txt_path = None

    if save_point_mask:
        point_mask_path = save_point_masks(
            save_dir=save_dir,
            sample_name=sample_name,
            point_class_ids=assign_result["point_class_ids"],
            point_instance_ids=assign_result["point_instance_ids"],
            valid_projected_mask=assign_result["valid_projected_mask"],
            visible_projected_mask=assign_result["visible_projected_mask"]
        )

    if save_boxes:
        boxes_json_path = os.path.join(save_dir, f"{sample_name}_3d_boxes.json")
        boxes_txt_path = os.path.join(save_dir, f"{sample_name}_3d_boxes.txt")

        extra_info = {
            "sample_name": sample_name,
            "pc_path": pc_path,
            "image_path": image_path,
            "calib_path": calib_path,
            "segmentation_dir": segmentation_dir,
            "num_points": int(points3d.shape[0]),
            "num_projected_points": int(assign_result["valid_projected_mask"].sum()),
            "num_visible_projected_points": int(assign_result["visible_projected_mask"].sum()),
            "num_labeled_points": int((assign_result["point_class_ids"] >= 0).sum()),
            "class_to_id": class_to_id,
            "box_type": box_type,
            "min_points": min_points,
            "use_zbuffer": use_zbuffer,
            "zbuffer_tolerance": zbuffer_tolerance,
            "use_percentile_filter": use_percentile_filter,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "semantic_filter_strength": int(semantic_filter_strength),
            "semantic_filter_stats": semantic_filter_stats,
        }

        save_3d_boxes_json(
            save_path=boxes_json_path,
            boxes=boxes,
            extra_info=extra_info
        )

        save_3d_boxes_txt(
            save_path=boxes_txt_path,
            boxes=boxes
        )

    ply_path = None

    if save_ply:
        ply_path = os.path.join(save_dir, f"{sample_name}_labeled_points.ply")
        save_labeled_pointcloud_ply(
            save_path=ply_path,
            points3d=points3d,
            point_class_ids=assign_result["point_class_ids"]
        )

    result = {
        "sample_name": sample_name,
        "points3d": points3d,
        "point_class_ids": assign_result["point_class_ids"],
        "point_instance_ids": assign_result["point_instance_ids"],
        "segments": assign_result["segments"],
        "boxes": boxes,
        "class_to_id": class_to_id,
        "semantic_filter_stats": semantic_filter_stats,
        "save_paths": {
            "point_masks": point_mask_path,
            "boxes_json": boxes_json_path,
            "boxes_txt": boxes_txt_path,
            "labeled_ply": ply_path,
        }
    }

    print("[3DTool] 完成")
    print("[3DTool] 3D 实例数量:", len(assign_result["segments"]))
    print("[3DTool] 3D 框数量:", len(boxes))
    print("[3DTool] 3D语义图滤波强度:", int(semantic_filter_strength))
    print("[3DTool] 滤波移除语义点数量:", semantic_filter_stats.get("removed_labeled_points", 0))
    print("[3DTool] 保存目录:", save_dir)

    return result


# ============================================================
# 批量处理
# ============================================================

def generate_3d_annotations_batch(
    root_dir: str,
    segmentation_root: str,
    image_ids: List[int],
    save_root: Optional[str] = None,
    box_type: str = "pca",
    min_points: int = 30,
    semantic_filter_strength: int = 0
) -> List[Dict]:
    results = []

    for image_id in image_ids:
        sample_name = f"{image_id:06d}"
        segmentation_dir = os.path.join(segmentation_root, f"{sample_name}_segmentation")

        if not os.path.isdir(segmentation_dir):
            print(f"[Batch] 跳过，未找到 segmentation_dir: {segmentation_dir}")
            continue

        try:
            result = generate_3d_annotations_for_one_sample(
                root_dir=root_dir,
                image_id=image_id,
                segmentation_dir=segmentation_dir,
                save_root=save_root,
                box_type=box_type,
                min_points=min_points,
                semantic_filter_strength=semantic_filter_strength
            )
            results.append(result)

        except Exception as e:
            print(f"[Batch] 处理失败: {sample_name}")
            print(f"[Batch] 错误: {e}")

    return results


# ============================================================
# Open3D 可视化：查看分类后的点云和 3D 框
# ============================================================

def get_default_color_palette() -> List[Tuple[float, float, float]]:

    colors = [
        (1.0, 0.0, 0.0),      # class_id 0, red
        (0.0, 1.0, 0.0),      # class_id 1, green
        (0.0, 0.0, 1.0),      # class_id 2, blue
        (1.0, 1.0, 0.0),      # class_id 3, yellow
        (1.0, 0.0, 1.0),      # class_id 4, magenta
        (0.0, 1.0, 1.0),      # class_id 5, cyan
        (1.0, 0.5, 0.0),      # class_id 6, orange
        (0.5, 0.0, 1.0),      # class_id 7, purple
        (0.0, 0.5, 1.0),      # class_id 8, sky blue
        (0.5, 1.0, 0.0),      # class_id 9, light green
        (1.0, 0.3, 0.3),
        (0.3, 1.0, 0.3),
        (0.3, 0.3, 1.0),
    ]
    return colors


def normalize_color_to_open3d(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    将 0~255 或 0~1 的 RGB 统一转成 Open3D 需要的 0~1 RGB。
    """
    arr = np.asarray(color, dtype=np.float64).reshape(3)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return float(arr[0]), float(arr[1]), float(arr[2])


def get_class_color_by_id(
    class_id: int,
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> Tuple[float, float, float]:
    """
    根据 class_id 获取稳定颜色。
    """
    cid = int(class_id)

    if class_id_to_color is not None:
        if cid in class_id_to_color:
            return normalize_color_to_open3d(class_id_to_color[cid])
        if str(cid) in class_id_to_color:
            return normalize_color_to_open3d(class_id_to_color[str(cid)])

    palette = get_default_color_palette()
    return palette[cid % len(palette)]


def get_instance_color_by_id(instance_id: int) -> Tuple[float, float, float]:
    """
    根据 instance_id 获取稳定实例颜色。
    instance_id 通常从 1 开始，所以这里用 instance_id - 1 对齐第一个实例为红色。
    """
    palette = get_default_color_palette()
    iid = int(instance_id)
    return palette[(iid - 1) % len(palette)]

def build_open3d_labeled_pointcloud(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: Optional[np.ndarray] = None,
    color_mode: str = "class",
    show_background: bool = True,
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
):
    """
    构建 Open3D 点云。
    """
    import open3d as o3d

    points3d = np.asarray(points3d, dtype=np.float64)
    point_class_ids = np.asarray(point_class_ids, dtype=np.int32)

    if point_instance_ids is not None:
        point_instance_ids = np.asarray(point_instance_ids, dtype=np.int32)

    if color_mode not in ["class", "instance"]:
        raise ValueError("color_mode 只能是 'class' 或 'instance'")

    if not show_background:
        keep = point_class_ids >= 0
        points_show = points3d[keep]
        class_ids_show = point_class_ids[keep]

        if point_instance_ids is not None:
            instance_ids_show = point_instance_ids[keep]
        else:
            instance_ids_show = None
    else:
        points_show = points3d
        class_ids_show = point_class_ids
        instance_ids_show = point_instance_ids

    colors = np.zeros((points_show.shape[0], 3), dtype=np.float64)
    colors[:, :] = np.array([0.55, 0.55, 0.55], dtype=np.float64)  # 未标注背景点灰色

    if color_mode == "class":
        unique_ids = sorted([int(x) for x in np.unique(class_ids_show) if int(x) >= 0])

        for cid in unique_ids:
            # 关键修复：直接用 class_id 取色，不能用 enumerate(unique_ids) 的序号取色
            color = get_class_color_by_id(cid, class_id_to_color=class_id_to_color)
            colors[class_ids_show == cid] = np.array(color, dtype=np.float64)

    else:
        if instance_ids_show is None:
            raise ValueError("color_mode='instance' 时必须传入 point_instance_ids")

        unique_ids = sorted([int(x) for x in np.unique(instance_ids_show) if int(x) > 0])

        for iid in unique_ids:
            # 实例颜色也按 instance_id 稳定取色，不再按当前样本出现顺序取色
            color = get_instance_color_by_id(iid)
            colors[instance_ids_show == iid] = np.array(color, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def build_open3d_box_lineset(
    box: Dict,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
):
    """
    根据 3D box 的 corners 构建 Open3D LineSet。

    box 需要包含：
        box["corners"]: 8 x 3

    兼容：
        AABB
        PCA_OBB
    """
    import open3d as o3d

    corners = np.asarray(box["corners"], dtype=np.float64)

    if corners.shape != (8, 3):
        raise ValueError(f"box['corners'] 必须是 8x3，当前 shape={corners.shape}")

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def build_open3d_box_center_sphere(
    box: Dict,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius: float = 0.035
):
    """
    给 3D 框中心画一个小球，方便看框中心。
    """
    import open3d as o3d

    center = np.asarray(box["center"], dtype=np.float64)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)

    return sphere


def visualize_labeled_pointcloud_and_boxes(
    points3d: np.ndarray,
    point_class_ids: np.ndarray,
    point_instance_ids: Optional[np.ndarray],
    boxes: List[Dict],
    color_mode: str = "class",
    show_background: bool = True,
    show_box_center: bool = True,
    point_size: float = 2.0,
    window_name: str = "3D labeled point cloud + boxes",
    class_id_to_color: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> None:
    """
    可视化分类后的点云和 3D 框。

    参数：
        color_mode:
            "class":
                点云按类别上色。类别颜色由 class_id 稳定决定。
            "instance":
                点云按实例上色。实例颜色由 instance_id 稳定决定。

        show_background:
            是否显示未标注背景点。

        show_box_center:
            是否显示 3D 框中心点。

        class_id_to_color:
            可选。传入 2D 前端使用的类别颜色表，保证 2D/3D 完全一致。

    本函数固定所有 3D 框为绿色。
    """
    import open3d as o3d

    geometries = []

    pcd = build_open3d_labeled_pointcloud(
        points3d=points3d,
        point_class_ids=point_class_ids,
        point_instance_ids=point_instance_ids,
        color_mode=color_mode,
        show_background=show_background,
        class_id_to_color=class_id_to_color
    )

    geometries.append(pcd)

    # 按你的要求：所有 3D 框固定为绿色，不再按类别变化。
    box_color = (0.0, 1.0, 0.0)

    for box in boxes:
        line_set = build_open3d_box_lineset(
            box=box,
            color=box_color
        )
        geometries.append(line_set)

        if show_box_center:
            center_sphere = build_open3d_box_center_sphere(
                box=box,
                color=box_color,
                radius=0.035
            )
            geometries.append(center_sphere)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6,
        origin=[0, 0, 0]
    )
    geometries.append(coord)

    print("\n========== 3D Box 信息 ==========")
    for box in boxes:
        center = np.asarray(box["center"], dtype=np.float64)
        size = np.asarray(box["size"], dtype=np.float64)

        print(
            f"class={box['class_name']} | "
            f"class_id={box['class_id']} | "
            f"instance_id={box['instance_id']} | "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
            f"size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) | "
            f"heading={float(box.get('heading_angle', 0.0)):.3f} | "
            f"points={box.get('num_points', 0)}"
        )

    print("\n========== 3D 点云颜色模式 ==========")
    print(f"color_mode={color_mode}")
    print("注意：若 color_mode='instance'，颜色表示实例，不表示类别。要和 2D 类别颜色一致，请使用 color_mode='class'。")
    print("3D 框颜色：固定绿色")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=900)

    for g in geometries:
        vis.add_geometry(g)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()

def visualize_result_from_memory(
    result: Dict,
    color_mode: str = "class",
    show_background: bool = True
) -> None:
    """
    直接可视化 generate_3d_annotations_for_one_sample() 返回的 result。

    用法：
        result = generate_3d_annotations_for_one_sample(...)
        visualize_result_from_memory(result)
    """
    visualize_labeled_pointcloud_and_boxes(
        points3d=result["points3d"],
        point_class_ids=result["point_class_ids"],
        point_instance_ids=result["point_instance_ids"],
        boxes=result["boxes"],
        color_mode=color_mode,
        show_background=show_background,
        show_box_center=True,
        point_size=2.0,
        window_name="3D annotation result"
    )


def visualize_saved_3d_annotation(
    root_dir: str,
    image_id: int,
    annotation_root: Optional[str] = None,
    color_mode: str = "class",
    show_background: bool = True
) -> None:
    """
    从已经保存的 annotation_3d 结果中读取并可视化。
    """
    sample_name = f"{image_id:06d}"

    if annotation_root is None:
        annotation_root = os.path.join(root_dir, "annotation_3d")

    sample_annotation_dir = os.path.join(annotation_root, sample_name)

    pc_path = os.path.join(root_dir, "pc", sample_name + ".mat")
    point_mask_path = os.path.join(sample_annotation_dir, f"{sample_name}_point_masks.npz")
    boxes_json_path = os.path.join(sample_annotation_dir, f"{sample_name}_3d_boxes.json")

    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"点云文件不存在: {pc_path}")

    if not os.path.exists(point_mask_path):
        raise FileNotFoundError(f"点云 mask 文件不存在: {point_mask_path}")

    if not os.path.exists(boxes_json_path):
        raise FileNotFoundError(f"3D box json 文件不存在: {boxes_json_path}")

    points3d_rgb = load_mat_points(pc_path)
    points3d = points3d_rgb[:, 0:3]

    data = np.load(point_mask_path, allow_pickle=True)

    point_class_ids = data["point_class_ids"]
    point_instance_ids = data["point_instance_ids"]

    with open(boxes_json_path, "r", encoding="utf-8") as f:
        box_data = json.load(f)

    boxes = box_data["boxes"]

    print("[Visualize] 点云:", pc_path)
    print("[Visualize] 点云 mask:", point_mask_path)
    print("[Visualize] 3D boxes:", boxes_json_path)
    print("[Visualize] 点数量:", points3d.shape[0])
    print("[Visualize] 已标注点数量:", int((point_class_ids >= 0).sum()))
    print("[Visualize] 3D 框数量:", len(boxes))

    visualize_labeled_pointcloud_and_boxes(
        points3d=points3d,
        point_class_ids=point_class_ids,
        point_instance_ids=point_instance_ids,
        boxes=boxes,
        color_mode=color_mode,
        show_background=show_background,
        show_box_center=True,
        point_size=2.0,
        window_name=f"{sample_name} labeled point cloud + boxes"
    )


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    ROOT_DIR = r"D:\frustum-convnet\sunrgbd\mysunrgbd\training"
    SEGMENTATION_ROOT = r"C:\Users\25918\Desktop\sam_output"

    IMAGE_ID = 1
    SAMPLE_NAME = f"{IMAGE_ID:06d}"

    SEGMENTATION_DIR = os.path.join(
        SEGMENTATION_ROOT,
        f"{SAMPLE_NAME}_segmentation"
    )

    # 1. 先生成 3D 点云类别 mask 和 3D 框
    result = generate_3d_annotations_for_one_sample(
        root_dir=ROOT_DIR,
        image_id=IMAGE_ID,
        segmentation_dir=SEGMENTATION_DIR,
        save_root=None,
        box_type="pca",
        min_points=30,
        up_axis=2,
        use_zbuffer=True,
        zbuffer_tolerance=0.03,
        use_percentile_filter=True,
        lower_percentile=1.0,
        upper_percentile=99.0,
        semantic_filter_strength=1,
        save_ply=True
    )

    # 2. 直接查看内存中的结果
    visualize_result_from_memory(
        result,
        color_mode="class",       # "class" 按类别上色；"instance" 按实例上色
        show_background=True      # True 显示灰色背景点；False 只显示被 mask 选中的点
    )