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
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QTimer


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
    vec = np.zeros(3, dtype=np.float64)
    vec[int(axis)] = 1.0
    return vec


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8 or not np.isfinite(vec).all():
        return np.asarray(fallback, dtype=np.float64)
    return vec / norm


def get_box_axes(box: Dict) -> Tuple[List[np.ndarray], int, List[int]]:
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


def move_box(box: Dict, offset: np.ndarray) -> Dict:
    """按照世界坐标偏移量平移三维框，并同步重建角点。"""
    axes, up_axis, horizontal_axes = get_box_axes(box)
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size = np.asarray(box["size"], dtype=np.float64).reshape(3)
    new_center = center + np.asarray(offset, dtype=np.float64).reshape(3)
    return rebuild_box(box, new_center, size, axes, up_axis, horizontal_axes)


def _build_labeled_pointcloud_for_editor(result: Dict):
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
    import open3d as o3d

    corners = np.asarray(box["corners"], dtype=np.float64).reshape(8, 3)
    base_color = np.array([1.0, 1.0, 0.0] if selected else [0.0, 1.0, 0.0], dtype=np.float64)
    colors = np.tile(base_color.reshape(1, 3), (BOX_LINES.shape[0], 1))

    if selected:
        for line_id in FACE_LINE_IDS.get((int(face_axis), float(face_sign)), []):
            colors[line_id] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(BOX_LINES)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _print_interactive_editor_help():
    print("\n========== Interactive 3D Box Editor ==========")
    print("N / P      : next / previous box")
    print("1..6       : select face (+X, -X, +Y, -Y, +Z, -Z)")
    print("E / W      : expand / shrink selected face")
    print("D / A      : rotate selected box +angle / -angle")
    print("Arrow keys : move selected box in X/Y plane")
    print("PageUp/Down: move selected box in Z axis")
    print("Delete / X : delete selected box")
    print("Save / close is handled by the Qt control panel buttons.")
    print("Selected box is yellow; selected face edges are red.")


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
        "move_step": float(edit_step),
        "saved": False,
        "open3d_closed": False,
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

    panel_widgets = {}

    def clamp_selection():
        if len(state["boxes"]) == 0:
            state["selected_idx"] = -1
        else:
            state["selected_idx"] = max(0, min(int(state["selected_idx"]), len(state["boxes"]) - 1))

    def remove_geometry(geom):
        try:
            vis.remove_geometry(geom, reset_bounding_box=False)
        except TypeError:
            vis.remove_geometry(geom)

    def add_geometry(geom):
        try:
            vis.add_geometry(geom, reset_bounding_box=False)
        except TypeError:
            vis.add_geometry(geom)

    def format_box_name(idx: int, box: Dict) -> str:
        return (
            f"{idx + 1}. {box.get('class_name', '')} | "
            f"instance {int(box.get('instance_id', 0))} | "
            f"points {int(box.get('num_points', 0))}"
        )

    def refresh_panel():
        refresh = panel_widgets.get("refresh")
        if refresh is not None:
            refresh()

    def redraw():
        for geom in state["box_geometries"]:
            remove_geometry(geom)
        state["box_geometries"] = []

        clamp_selection()

        if len(state["boxes"]) == 0:
            print("[3DBoxEditor] no boxes left")
            vis.update_renderer()
            refresh_panel()
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
        refresh_panel()

    def select_box(idx: int):
        if not state["boxes"]:
            state["selected_idx"] = -1
            redraw()
            return
        state["selected_idx"] = max(0, min(int(idx), len(state["boxes"]) - 1))
        redraw()

    def next_box():
        if state["boxes"]:
            state["selected_idx"] = (state["selected_idx"] + 1) % len(state["boxes"])
            redraw()

    def next_box_callback(_vis):
        next_box()
        return False

    def prev_box():
        if state["boxes"]:
            state["selected_idx"] = (state["selected_idx"] - 1) % len(state["boxes"])
            redraw()

    def prev_box_callback(_vis):
        prev_box()
        return False

    def select_face(face_idx: int):
        def callback(_vis):
            state["face_idx"] = int(face_idx)
            redraw()
            return False
        return callback

    def edit_selected_face(expand: bool):
        if not state["boxes"]:
            return
        _label, axis, sign = FACE_ITEMS[state["face_idx"]]
        delta = state["edit_step"] if expand else -state["edit_step"]
        try:
            idx = state["selected_idx"]
            state["boxes"][idx] = edit_box_face(state["boxes"][idx], axis, sign, delta)
        except Exception as exc:
            print(f"[3DBoxEditor] edit failed: {exc}")
        redraw()

    def edit_selected_face_callback(expand: bool):
        def callback(_vis):
            edit_selected_face(expand)
            return False
        return callback

    def rotate_selected(direction: float):
        if state["boxes"]:
            idx = state["selected_idx"]
            state["boxes"][idx] = rotate_box(state["boxes"][idx], direction * state["rotate_step"])
            redraw()

    def rotate_selected_callback(direction: float):
        def callback(_vis):
            rotate_selected(direction)
            return False
        return callback

    def move_selected(axis: int, direction: float):
        if state["boxes"]:
            offset = np.zeros(3, dtype=np.float64)
            offset[int(axis)] = float(direction) * state["move_step"]
            idx = state["selected_idx"]
            state["boxes"][idx] = move_box(state["boxes"][idx], offset)
            redraw()

    def move_selected_callback(axis: int, direction: float):
        def callback(_vis):
            move_selected(axis, direction)
            return False
        return callback

    def delete_selected():
        if state["boxes"] and state["selected_idx"] >= 0:
            removed = state["boxes"].pop(state["selected_idx"])
            print(f"[3DBoxEditor] deleted instance_id={removed.get('instance_id', '')}")
            clamp_selection()
            redraw()

    def delete_selected_callback(_vis):
        delete_selected()
        return False

    def save_and_close():
        state["saved"] = True
        dialog = panel_widgets.get("dialog")
        if dialog is not None:
            dialog.accept()

    def close_without_save():
        dialog = panel_widgets.get("dialog")
        if dialog is not None:
            dialog.reject()

    def build_control_panel():
        dialog = QDialog()
        dialog.setWindowTitle("交互式 3D 框编辑控制台")
        dialog.resize(980, 560)

        root_layout = QHBoxLayout(dialog)

        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(
            "Open3D 视窗操作\n"
            "\n"
            "鼠标拖拽：旋转 / 平移 Open3D 视角\n"
            "中间列表或 N / P：切换选中的 3D 框\n"
            "1..6：选择局部 +X、-X、+Y、-Y、+Z、-Z 面\n"
            "E / W：按右侧步长扩展 / 缩进选中面\n"
            "D / A：按右侧角度正向 / 反向旋转\n"
            "方向键：在世界 X/Y 平面移动选中框\n"
            "PageUp / PageDown：沿世界 Z 轴移动\n"
            "Delete / Backspace / X：删除选中框\n"
            "\n"
            "颜色说明\n"
            "绿色：未选中框\n"
            "黄色：当前选中框\n"
            "红色：当前选中的框面\n"
            "\n"
        )
        help_box.setMinimumWidth(260)
        root_layout.addWidget(help_box, stretch=1)

        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.addWidget(QLabel("3D 框列表："))

        box_list = QListWidget()
        middle_layout.addWidget(box_list, stretch=1)

        detail_label = QLabel("")
        detail_label.setWordWrap(True)
        middle_layout.addWidget(detail_label)
        root_layout.addWidget(middle_panel, stretch=2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        face_group = QGroupBox("面缩进 / 扩展")
        face_layout = QVBoxLayout(face_group)
        face_form = QFormLayout()
        face_combo = QComboBox()
        for label, _axis, _sign in FACE_ITEMS:
            face_combo.addItem(label)
        face_combo.setCurrentIndex(int(state["face_idx"]))
        edit_step_spin = QDoubleSpinBox()
        edit_step_spin.setRange(0.001, 10.0)
        edit_step_spin.setDecimals(4)
        edit_step_spin.setSingleStep(0.01)
        edit_step_spin.setValue(float(state["edit_step"]))
        face_form.addRow("选中面：", face_combo)
        face_form.addRow("缩放步长：", edit_step_spin)
        face_layout.addLayout(face_form)

        face_btn_layout = QHBoxLayout()
        expand_btn = QPushButton("扩展")
        shrink_btn = QPushButton("缩进")
        face_btn_layout.addWidget(expand_btn)
        face_btn_layout.addWidget(shrink_btn)
        face_layout.addLayout(face_btn_layout)
        right_layout.addWidget(face_group)

        rotate_group = QGroupBox("旋转")
        rotate_layout = QVBoxLayout(rotate_group)
        rotate_spin = QDoubleSpinBox()
        rotate_spin.setRange(0.1, 180.0)
        rotate_spin.setDecimals(2)
        rotate_spin.setSingleStep(1.0)
        rotate_spin.setValue(float(state["rotate_step"]))
        rotate_form = QFormLayout()
        rotate_form.addRow("旋转角度：", rotate_spin)
        rotate_layout.addLayout(rotate_form)

        rotate_btn_layout = QHBoxLayout()
        rotate_left_btn = QPushButton("反向旋转")
        rotate_right_btn = QPushButton("正向旋转")
        rotate_btn_layout.addWidget(rotate_left_btn)
        rotate_btn_layout.addWidget(rotate_right_btn)
        rotate_layout.addLayout(rotate_btn_layout)
        right_layout.addWidget(rotate_group)

        move_group = QGroupBox("平移")
        move_layout = QVBoxLayout(move_group)
        move_spin = QDoubleSpinBox()
        move_spin.setRange(0.001, 10.0)
        move_spin.setDecimals(4)
        move_spin.setSingleStep(0.01)
        move_spin.setValue(float(state["move_step"]))
        move_form = QFormLayout()
        move_form.addRow("移动步长：", move_spin)
        move_layout.addLayout(move_form)

        move_xy_layout = QHBoxLayout()
        move_neg_x_btn = QPushButton("-X")
        move_pos_x_btn = QPushButton("+X")
        move_neg_y_btn = QPushButton("-Y")
        move_pos_y_btn = QPushButton("+Y")
        move_xy_layout.addWidget(move_neg_x_btn)
        move_xy_layout.addWidget(move_pos_x_btn)
        move_xy_layout.addWidget(move_neg_y_btn)
        move_xy_layout.addWidget(move_pos_y_btn)
        move_layout.addLayout(move_xy_layout)

        move_z_layout = QHBoxLayout()
        move_neg_z_btn = QPushButton("-Z")
        move_pos_z_btn = QPushButton("+Z")
        move_z_layout.addWidget(move_neg_z_btn)
        move_z_layout.addWidget(move_pos_z_btn)
        move_layout.addLayout(move_z_layout)
        right_layout.addWidget(move_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        delete_btn = QPushButton("删除选中框")
        save_btn = QPushButton("保存并关闭")
        close_btn = QPushButton("关闭不保存")
        action_layout.addWidget(delete_btn)
        action_layout.addWidget(save_btn)
        action_layout.addWidget(close_btn)
        right_layout.addWidget(action_group)
        right_layout.addStretch()

        root_layout.addWidget(right_panel, stretch=1)

        def refresh():
            clamp_selection()
            face_combo.blockSignals(True)
            face_combo.setCurrentIndex(int(state["face_idx"]))
            face_combo.blockSignals(False)
            edit_step_spin.blockSignals(True)
            edit_step_spin.setValue(float(state["edit_step"]))
            edit_step_spin.blockSignals(False)
            rotate_spin.blockSignals(True)
            rotate_spin.setValue(float(state["rotate_step"]))
            rotate_spin.blockSignals(False)
            move_spin.blockSignals(True)
            move_spin.setValue(float(state["move_step"]))
            move_spin.blockSignals(False)

            box_list.blockSignals(True)
            box_list.clear()
            for idx, box in enumerate(state["boxes"]):
                box_list.addItem(format_box_name(idx, box))
            if state["selected_idx"] >= 0:
                box_list.setCurrentRow(state["selected_idx"])
            box_list.blockSignals(False)

            if state["selected_idx"] < 0:
                detail_label.setText("当前没有可编辑的 3D 框")
                return

            box = state["boxes"][state["selected_idx"]]
            center = np.asarray(box["center"], dtype=np.float64)
            size = np.asarray(box["size"], dtype=np.float64)
            face_label = FACE_ITEMS[state["face_idx"]][0]
            detail_label.setText(
                f"当前框：{state['selected_idx'] + 1}/{len(state['boxes'])}\n"
                f"类别：{box.get('class_name', '')}\n"
                f"实例：{box.get('instance_id', '')}\n"
                f"选中面：{face_label}\n"
                f"中心：({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})\n"
                f"尺寸：({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})\n"
                f"朝向角：{float(box.get('heading_angle', 0.0)):.4f}"
            )

        def on_list_row_changed(row: int):
            if row >= 0 and row != state["selected_idx"]:
                select_box(row)

        def on_face_changed(index: int):
            state["face_idx"] = int(index)
            redraw()

        def on_edit_step_changed(value: float):
            state["edit_step"] = float(value)
            refresh()

        def on_rotate_step_changed(value: float):
            state["rotate_step"] = float(value)
            refresh()

        def on_move_step_changed(value: float):
            state["move_step"] = float(value)
            refresh()

        box_list.currentRowChanged.connect(on_list_row_changed)
        face_combo.currentIndexChanged.connect(on_face_changed)
        edit_step_spin.valueChanged.connect(on_edit_step_changed)
        rotate_spin.valueChanged.connect(on_rotate_step_changed)
        move_spin.valueChanged.connect(on_move_step_changed)

        expand_btn.clicked.connect(lambda: edit_selected_face(True))
        shrink_btn.clicked.connect(lambda: edit_selected_face(False))
        rotate_left_btn.clicked.connect(lambda: rotate_selected(-1.0))
        rotate_right_btn.clicked.connect(lambda: rotate_selected(1.0))
        move_neg_x_btn.clicked.connect(lambda: move_selected(0, -1.0))
        move_pos_x_btn.clicked.connect(lambda: move_selected(0, 1.0))
        move_neg_y_btn.clicked.connect(lambda: move_selected(1, -1.0))
        move_pos_y_btn.clicked.connect(lambda: move_selected(1, 1.0))
        move_neg_z_btn.clicked.connect(lambda: move_selected(2, -1.0))
        move_pos_z_btn.clicked.connect(lambda: move_selected(2, 1.0))
        delete_btn.clicked.connect(delete_selected)
        save_btn.clicked.connect(save_and_close)
        close_btn.clicked.connect(close_without_save)

        panel_widgets["dialog"] = dialog
        panel_widgets["refresh"] = refresh
        refresh()
        return dialog

    _print_interactive_editor_help()
    dialog = build_control_panel()

    vis.register_key_callback(ord("N"), next_box_callback)
    vis.register_key_callback(ord("P"), prev_box_callback)
    for key, face_idx in zip("123456", range(6)):
        vis.register_key_callback(ord(key), select_face(face_idx))
    vis.register_key_callback(ord("E"), edit_selected_face_callback(expand=True))
    vis.register_key_callback(ord("W"), edit_selected_face_callback(expand=False))
    vis.register_key_callback(ord("D"), rotate_selected_callback(1.0))
    vis.register_key_callback(ord("A"), rotate_selected_callback(-1.0))
    vis.register_key_callback(ord("X"), delete_selected_callback)
    vis.register_key_callback(261, delete_selected_callback)  # Delete on GLFW.
    vis.register_key_callback(259, delete_selected_callback)  # Backspace on GLFW.
    vis.register_key_callback(263, move_selected_callback(0, -1.0))  # Left.
    vis.register_key_callback(262, move_selected_callback(0, 1.0))  # Right.
    vis.register_key_callback(265, move_selected_callback(1, 1.0))  # Up.
    vis.register_key_callback(264, move_selected_callback(1, -1.0))  # Down.
    vis.register_key_callback(266, move_selected_callback(2, 1.0))  # PageUp.
    vis.register_key_callback(267, move_selected_callback(2, -1.0))  # PageDown.

    redraw()

    timer = QTimer(dialog)

    def poll_open3d_window():
        try:
            alive = vis.poll_events()
            vis.update_renderer()
        except Exception:
            alive = False
        if not alive:
            state["open3d_closed"] = True
            dialog.reject()

    timer.timeout.connect(poll_open3d_window)
    timer.start(16)

    def on_dialog_finished(_result):
        timer.stop()
        if not state["open3d_closed"]:
            try:
                vis.close()
            except Exception:
                pass

    dialog.finished.connect(on_dialog_finished)
    dialog.exec_()
    vis.destroy_window()

    return copy.deepcopy(state["boxes"]), bool(state["saved"])


class BoxEditorDialog(QDialog):
    """三维框参数编辑对话框，用于面板式修改和预览。"""
    def __init__(self, boxes: List[Dict], parent=None, preview_callback: Optional[Callable[[List[Dict]], None]] = None):
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
        return (
            f"{idx + 1}. {box.get('class_name', '')} | "
            f"instance {int(box.get('instance_id', 0))} | "
            f"points {int(box.get('num_points', 0))}"
        )

    def update_detail(self, row: int):
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
        row = self.box_list.currentRow()
        if row < 0 or row >= len(self.boxes):
            QMessageBox.warning(self, "提示", "请先选择一个 3D 框")
            return -1
        return row

    def apply_face_edit(self, expand: bool):
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
        row = self.selected_row()
        if row < 0:
            return

        self.boxes[row] = rotate_box(self.boxes[row], float(self.rotate_spin.value()))
        self.refresh_box_list()
        self.box_list.setCurrentRow(row)

    def delete_selected_box(self):
        row = self.selected_row()
        if row < 0:
            return

        del self.boxes[row]
        self.refresh_box_list()

    def preview_current_boxes(self):
        if self.preview_callback is None:
            QMessageBox.warning(self, "提示", "当前没有可用的 3D 预览回调")
            return

        try:
            self.preview_callback(self.get_boxes())
        except Exception as exc:
            QMessageBox.warning(self, "3D 预览失败", str(exc))

    def get_boxes(self) -> List[Dict]:
        return copy.deepcopy(self.boxes)
