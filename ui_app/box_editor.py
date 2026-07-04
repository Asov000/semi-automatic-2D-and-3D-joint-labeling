# -*- coding: utf-8 -*-
"""三维框编辑模块，支持面缩放、旋转、删除和 Open3D 交互式编辑。"""

import copy
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


FACE_ITEMS = [
    ("局部 +X 面", 0, 1.0),
    ("局部 -X 面", 0, -1.0),
    ("局部 +Y 面", 1, 1.0),
    ("局部 -Y 面", 1, -1.0),
    ("局部 +Z 面", 2, 1.0),
    ("局部 -Z 面", 2, -1.0),
]

BOX_LINES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)

FACE_LINE_IDS = {
    (0, 1.0): [1, 10, 5, 9],
    (0, -1.0): [3, 11, 7, 8],
    (1, 1.0): [2, 10, 6, 11],
    (1, -1.0): [0, 9, 4, 8],
    (2, 1.0): [4, 5, 6, 7],
    (2, -1.0): [0, 1, 2, 3],
}


def _unit_vector(axis: int) -> np.ndarray:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    vec = np.zeros(3, dtype=np.float64)
    vec[int(axis)] = 1.0
    return vec


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """执行模块内部辅助逻辑，供上层流程复用。"""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8 or not np.isfinite(vec).all():
        return np.asarray(fallback, dtype=np.float64)
    return vec / norm


def get_box_axes(box: Dict) -> Tuple[List[np.ndarray], int, List[int]]:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    up_axis = int(box.get("up_axis", 2))
    if up_axis not in [0, 1, 2]:
        up_axis = 2

    horizontal_axes = box.get("horizontal_axes", [i for i in range(3) if i != up_axis])
    horizontal_axes = [int(x) for x in horizontal_axes]
    if len(horizontal_axes) != 2 or any(x not in [0, 1, 2] for x in horizontal_axes):
        horizontal_axes = [i for i in range(3) if i != up_axis]

    up_vec = _unit_vector(up_axis)

    if "main_direction_2d" in box:
        main_2d = _normalize(
            np.asarray(box["main_direction_2d"], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
        )
    else:
        heading = float(box.get("heading_angle", 0.0))
        main_2d = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)

    side_2d = np.array([-main_2d[1], main_2d[0]], dtype=np.float64)

    axis0 = np.zeros(3, dtype=np.float64)
    axis1 = np.zeros(3, dtype=np.float64)
    axis0[horizontal_axes] = main_2d
    axis1[horizontal_axes] = side_2d

    axis0 = _normalize(axis0, _unit_vector(horizontal_axes[0]))
    axis1 = _normalize(axis1, _unit_vector(horizontal_axes[1]))

    return [axis0, axis1, up_vec], up_axis, horizontal_axes


def rebuild_box(box: Dict, center: np.ndarray, size: np.ndarray, axes: List[np.ndarray], up_axis: int, horizontal_axes: List[int]) -> Dict:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    new_box = copy.deepcopy(box)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.maximum(np.asarray(size, dtype=np.float64).reshape(3), 1e-4)

    signs = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=np.float64)

    corners = []
    for sx, sy, sz in signs:
        corners.append(
            center
            + axes[0] * sx * size[0] / 2.0
            + axes[1] * sy * size[1] / 2.0
            + axes[2] * sz * size[2] / 2.0
        )
    corners = np.asarray(corners, dtype=np.float64)

    main_2d = np.asarray(axes[0][horizontal_axes], dtype=np.float64)
    main_2d = _normalize(main_2d, np.array([1.0, 0.0], dtype=np.float64))

    new_box["box_type"] = "PCA_OBB"
    new_box["center"] = center
    new_box["size"] = size
    new_box["corners"] = corners
    new_box["horizontal_axes"] = horizontal_axes
    new_box["up_axis"] = int(up_axis)
    new_box["main_direction_2d"] = main_2d
    new_box["heading_angle"] = float(math.atan2(main_2d[1], main_2d[0]))
    new_box.pop("min_xyz", None)
    new_box.pop("max_xyz", None)

    return new_box


def edit_box_face(box: Dict, face_axis: int, face_sign: float, signed_delta: float) -> Dict:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    axes, up_axis, horizontal_axes = get_box_axes(box)
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size = np.asarray(box["size"], dtype=np.float64).reshape(3)

    new_size = size.copy()
    new_size[face_axis] = size[face_axis] + float(signed_delta)
    if new_size[face_axis] <= 1e-4:
        raise ValueError("该方向尺寸已经太小，不能继续缩进")

    new_center = center + axes[face_axis] * float(face_sign) * float(signed_delta) / 2.0
    return rebuild_box(box, new_center, new_size, axes, up_axis, horizontal_axes)


def rotate_box(box: Dict, angle_degrees: float) -> Dict:
    """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
    axes, up_axis, horizontal_axes = get_box_axes(box)
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size = np.asarray(box["size"], dtype=np.float64).reshape(3)

    theta = math.radians(float(angle_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    main_2d = np.asarray(axes[0][horizontal_axes], dtype=np.float64)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    rotated_main = rot @ main_2d
    rotated_side = np.array([-rotated_main[1], rotated_main[0]], dtype=np.float64)

    new_axis0 = np.zeros(3, dtype=np.float64)
    new_axis1 = np.zeros(3, dtype=np.float64)
    new_axis0[horizontal_axes] = rotated_main
    new_axis1[horizontal_axes] = rotated_side
    new_axes = [_normalize(new_axis0, axes[0]), _normalize(new_axis1, axes[1]), axes[2]]

    return rebuild_box(box, center, size, new_axes, up_axis, horizontal_axes)


def _build_labeled_pointcloud_for_editor(result: Dict):
    """执行模块内部辅助逻辑，供上层流程复用。"""
    import open3d as o3d

    points3d = np.asarray(result["points3d"], dtype=np.float64)
    point_class_ids = np.asarray(result["point_class_ids"], dtype=np.int32)

    colors = np.zeros((points3d.shape[0], 3), dtype=np.float64)
    colors[:, :] = np.array([0.45, 0.45, 0.45], dtype=np.float64)

    palette = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
    ]

    class_id_to_color = result.get("class_id_to_color", {}) or {}

    for cid in sorted([int(x) for x in np.unique(point_class_ids) if int(x) >= 0]):
        if cid in class_id_to_color:
            color = np.asarray(class_id_to_color[cid], dtype=np.float64).reshape(3)
        elif str(cid) in class_id_to_color:
            color = np.asarray(class_id_to_color[str(cid)], dtype=np.float64).reshape(3)
        else:
            color = np.asarray(palette[cid % len(palette)], dtype=np.float64)

        if color.max() > 1.0:
            color = color / 255.0
        colors[point_class_ids == cid] = np.clip(color, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _build_editor_box_lineset(box: Dict, selected: bool = False, face_axis: int = 0, face_sign: float = 1.0):
    """执行模块内部辅助逻辑，供上层流程复用。"""
    import open3d as o3d

    corners = np.asarray(box["corners"], dtype=np.float64).reshape(8, 3)
    base_color = np.array([1.0, 0.15, 0.15] if selected else [0.0, 1.0, 0.0], dtype=np.float64)
    colors = np.tile(base_color.reshape(1, 3), (BOX_LINES.shape[0], 1))

    if selected:
        for line_id in FACE_LINE_IDS.get((int(face_axis), float(face_sign)), []):
            colors[line_id] = np.array([1.0, 1.0, 0.0], dtype=np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(BOX_LINES)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _print_interactive_editor_help():
    """执行模块内部辅助逻辑，供上层流程复用。"""
    print("\n========== Interactive 3D Box Editor ==========")
    print("N / P      : next / previous box")
    print("1..6       : select face (+X, -X, +Y, -Y, +Z, -Z)")
    print("E / W      : expand / shrink selected face")
    print("T / G      : increase / decrease edit step")
    print("D / A      : rotate selected box +angle / -angle")
    print("R / F      : increase / decrease rotation step")
    print("Delete / X : delete selected box")
    print("S          : mark edits as saved and close")
    print("Q          : close editor")
    print("Selected box is red; selected face edges are yellow.")


def interactive_edit_boxes_open3d(
    result: Dict,
    boxes: List[Dict],
    edit_step: float = 0.05,
    rotate_step_degrees: float = 5.0,
) -> Tuple[List[Dict], bool]:
    """打开 Open3D 交互窗口，对三维框进行可视化编辑。"""
    import open3d as o3d

    state = {
        "boxes": copy.deepcopy(boxes),
        "selected_idx": 0,
        "face_idx": 0,
        "edit_step": float(edit_step),
        "rotate_step": float(rotate_step_degrees),
        "saved": False,
        "box_geometries": [],
    }

    pcd = _build_labeled_pointcloud_for_editor(result)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Interactive 3D Box Editor", width=1280, height=900)
    vis.add_geometry(pcd)
    vis.add_geometry(coord)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.02, 0.02, 0.02])

    def clamp_selection():
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if len(state["boxes"]) == 0:
            state["selected_idx"] = -1
        else:
            state["selected_idx"] = max(0, min(int(state["selected_idx"]), len(state["boxes"]) - 1))

    def remove_geometry(geom):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        try:
            vis.remove_geometry(geom, reset_bounding_box=False)
        except TypeError:
            vis.remove_geometry(geom)

    def add_geometry(geom):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        try:
            vis.add_geometry(geom, reset_bounding_box=False)
        except TypeError:
            vis.add_geometry(geom)

    def redraw():
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        for geom in state["box_geometries"]:
            remove_geometry(geom)
        state["box_geometries"] = []

        clamp_selection()

        if len(state["boxes"]) == 0:
            print("[3DBoxEditor] no boxes left")
            vis.update_renderer()
            return

        label, face_axis, face_sign = FACE_ITEMS[state["face_idx"]]
        for idx, box in enumerate(state["boxes"]):
            geom = _build_editor_box_lineset(
                box,
                selected=(idx == state["selected_idx"]),
                face_axis=face_axis,
                face_sign=face_sign,
            )
            add_geometry(geom)
            state["box_geometries"].append(geom)

        selected_box = state["boxes"][state["selected_idx"]]
        print(
            "[3DBoxEditor] "
            f"box {state['selected_idx'] + 1}/{len(state['boxes'])} | "
            f"class={selected_box.get('class_name', '')} | "
            f"face={label} | step={state['edit_step']:.4f} | "
            f"rotate_step={state['rotate_step']:.2f}"
        )
        vis.update_renderer()

    def next_box(_vis):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if state["boxes"]:
            state["selected_idx"] = (state["selected_idx"] + 1) % len(state["boxes"])
            redraw()
        return False

    def prev_box(_vis):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if state["boxes"]:
            state["selected_idx"] = (state["selected_idx"] - 1) % len(state["boxes"])
            redraw()
        return False

    def select_face(face_idx: int):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        def callback(_vis):
            """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
            state["face_idx"] = int(face_idx)
            redraw()
            return False
        return callback

    def edit_selected_face(expand: bool):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        def callback(_vis):
            """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
            if not state["boxes"]:
                return False
            _label, axis, sign = FACE_ITEMS[state["face_idx"]]
            delta = state["edit_step"] if expand else -state["edit_step"]
            try:
                idx = state["selected_idx"]
                state["boxes"][idx] = edit_box_face(state["boxes"][idx], axis, sign, delta)
            except Exception as exc:
                print(f"[3DBoxEditor] edit failed: {exc}")
            redraw()
            return False
        return callback

    def rotate_selected(direction: float):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        def callback(_vis):
            """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
            if state["boxes"]:
                idx = state["selected_idx"]
                state["boxes"][idx] = rotate_box(state["boxes"][idx], direction * state["rotate_step"])
                redraw()
            return False
        return callback

    def change_step(scale: float):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        def callback(_vis):
            """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
            state["edit_step"] = max(0.001, min(10.0, state["edit_step"] * float(scale)))
            redraw()
            return False
        return callback

    def change_rotate_step(delta: float):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        def callback(_vis):
            """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
            state["rotate_step"] = max(0.1, min(90.0, state["rotate_step"] + float(delta)))
            redraw()
            return False
        return callback

    def delete_selected(_vis):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if state["boxes"] and state["selected_idx"] >= 0:
            removed = state["boxes"].pop(state["selected_idx"])
            print(f"[3DBoxEditor] deleted instance_id={removed.get('instance_id', '')}")
            clamp_selection()
            redraw()
        return False

    def save_and_close(_vis):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        state["saved"] = True
        _vis.close()
        return False

    def close_editor(_vis):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        _vis.close()
        return False

    _print_interactive_editor_help()
    redraw()

    vis.register_key_callback(ord("N"), next_box)
    vis.register_key_callback(ord("P"), prev_box)
    for key, face_idx in zip("123456", range(6)):
        vis.register_key_callback(ord(key), select_face(face_idx))
    vis.register_key_callback(ord("E"), edit_selected_face(expand=True))
    vis.register_key_callback(ord("W"), edit_selected_face(expand=False))
    vis.register_key_callback(ord("T"), change_step(1.25))
    vis.register_key_callback(ord("G"), change_step(0.8))
    vis.register_key_callback(ord("D"), rotate_selected(1.0))
    vis.register_key_callback(ord("A"), rotate_selected(-1.0))
    vis.register_key_callback(ord("R"), change_rotate_step(1.0))
    vis.register_key_callback(ord("F"), change_rotate_step(-1.0))
    vis.register_key_callback(ord("X"), delete_selected)
    vis.register_key_callback(261, delete_selected)  # Delete on GLFW.
    vis.register_key_callback(259, delete_selected)  # Backspace on GLFW.
    vis.register_key_callback(ord("S"), save_and_close)
    vis.register_key_callback(ord("Q"), close_editor)

    vis.run()
    vis.destroy_window()

    return copy.deepcopy(state["boxes"]), bool(state["saved"])


class BoxEditorDialog(QDialog):
    """三维框参数编辑对话框，用于面板式修改和预览。"""
    def __init__(self, boxes: List[Dict], parent=None, preview_callback: Optional[Callable[[List[Dict]], None]] = None):
        """执行模块内部辅助逻辑，供上层流程复用。"""
        super().__init__(parent)
        self.setWindowTitle("编辑 3D 框")
        self.resize(560, 420)
        self.boxes = copy.deepcopy(boxes)
        self.preview_callback = preview_callback

        main_layout = QVBoxLayout(self)

        main_layout.addWidget(QLabel("选择 3D 框："))
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.update_detail)
        main_layout.addWidget(self.box_list)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        main_layout.addWidget(self.detail_label)

        face_layout = QHBoxLayout()
        self.face_combo = QComboBox()
        for label, _axis, _sign in FACE_ITEMS:
            self.face_combo.addItem(label)
        self.delta_spin = QDoubleSpinBox()
        self.delta_spin.setRange(0.001, 10.0)
        self.delta_spin.setDecimals(4)
        self.delta_spin.setSingleStep(0.01)
        self.delta_spin.setValue(0.05)
        face_layout.addWidget(QLabel("面："))
        face_layout.addWidget(self.face_combo)
        face_layout.addWidget(QLabel("步长："))
        face_layout.addWidget(self.delta_spin)
        main_layout.addLayout(face_layout)

        face_btn_layout = QHBoxLayout()
        self.expand_btn = QPushButton("扩展选中面")
        self.shrink_btn = QPushButton("缩进选中面")
        self.expand_btn.clicked.connect(lambda: self.apply_face_edit(expand=True))
        self.shrink_btn.clicked.connect(lambda: self.apply_face_edit(expand=False))
        face_btn_layout.addWidget(self.expand_btn)
        face_btn_layout.addWidget(self.shrink_btn)
        main_layout.addLayout(face_btn_layout)

        rotate_layout = QHBoxLayout()
        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-180.0, 180.0)
        self.rotate_spin.setDecimals(2)
        self.rotate_spin.setSingleStep(5.0)
        self.rotate_spin.setValue(5.0)
        self.rotate_btn = QPushButton("旋转选中框")
        self.rotate_btn.clicked.connect(self.apply_rotation)
        rotate_layout.addWidget(QLabel("旋转角度："))
        rotate_layout.addWidget(self.rotate_spin)
        rotate_layout.addWidget(self.rotate_btn)
        main_layout.addLayout(rotate_layout)

        preview_layout = QHBoxLayout()
        self.preview_btn = QPushButton("预览当前编辑效果")
        self.preview_btn.clicked.connect(self.preview_current_boxes)
        preview_layout.addStretch()
        preview_layout.addWidget(self.preview_btn)
        main_layout.addLayout(preview_layout)

        action_layout = QHBoxLayout()
        self.delete_btn = QPushButton("删除选中框")
        self.ok_btn = QPushButton("应用并关闭")
        self.cancel_btn = QPushButton("取消")
        self.delete_btn.clicked.connect(self.delete_selected_box)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(self.delete_btn)
        action_layout.addStretch()
        action_layout.addWidget(self.ok_btn)
        action_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(action_layout)

        self.refresh_box_list()

    def refresh_box_list(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        current = self.box_list.currentRow()
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for idx, box in enumerate(self.boxes):
            self.box_list.addItem(self.format_box_name(idx, box))
        self.box_list.blockSignals(False)

        if self.boxes:
            self.box_list.setCurrentRow(min(max(current, 0), len(self.boxes) - 1))
        else:
            self.update_detail(-1)

    def format_box_name(self, idx: int, box: Dict) -> str:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        return (
            f"{idx + 1}. {box.get('class_name', '')} | "
            f"instance {int(box.get('instance_id', 0))} | "
            f"points {int(box.get('num_points', 0))}"
        )

    def update_detail(self, row: int):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if row < 0 or row >= len(self.boxes):
            self.detail_label.setText("当前没有可编辑的 3D 框")
            return

        box = self.boxes[row]
        center = np.asarray(box["center"], dtype=np.float64)
        size = np.asarray(box["size"], dtype=np.float64)
        self.detail_label.setText(
            f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})  "
            f"size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})  "
            f"heading={float(box.get('heading_angle', 0.0)):.4f}"
        )

    def selected_row(self) -> int:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        row = self.box_list.currentRow()
        if row < 0 or row >= len(self.boxes):
            QMessageBox.warning(self, "提示", "请先选择一个 3D 框")
            return -1
        return row

    def apply_face_edit(self, expand: bool):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        row = self.selected_row()
        if row < 0:
            return

        label, axis, sign = FACE_ITEMS[self.face_combo.currentIndex()]
        amount = float(self.delta_spin.value())
        signed_delta = amount if expand else -amount

        try:
            self.boxes[row] = edit_box_face(self.boxes[row], axis, sign, signed_delta)
        except Exception as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        self.refresh_box_list()
        self.box_list.setCurrentRow(row)

    def apply_rotation(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        row = self.selected_row()
        if row < 0:
            return

        self.boxes[row] = rotate_box(self.boxes[row], float(self.rotate_spin.value()))
        self.refresh_box_list()
        self.box_list.setCurrentRow(row)

    def delete_selected_box(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        row = self.selected_row()
        if row < 0:
            return

        del self.boxes[row]
        self.refresh_box_list()

    def preview_current_boxes(self):
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        if self.preview_callback is None:
            QMessageBox.warning(self, "提示", "当前没有可用的 3D 预览回调")
            return

        try:
            self.preview_callback(self.get_boxes())
        except Exception as exc:
            QMessageBox.warning(self, "3D 预览失败", str(exc))

    def get_boxes(self) -> List[Dict]:
        """执行该函数封装的业务逻辑，并返回调用方需要的结果。"""
        return copy.deepcopy(self.boxes)
