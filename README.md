# SAM 自动/半自动 二维/三维标注工具

这是一个面向 RGB-D / SUNRGBD 数据的自动/半自动标注工具。项目以 PyQt 图形界面为入口，结合经典 SAM3 点提示分割、SAM3 文本提示自动分割、二维 mask 后处理、二维检测框导出、三维点云语义赋值、三维框生成与交互编辑，形成从图片标注到 2D/3D 标注文件导出的完整流程。
注意：如果需要使用自键数据集进行标注，请修改 tool3d_modules/projection.py 文件，自行替换投影逻辑。

## 主要功能

- 图片文件夹批量浏览与逐张标注。
- 支持前景点、背景点交互式 SAM 分割。
- 支持 SAM3 文本提示自动标注，例如 `chair, table, sofa`。
- 支持类别颜色统一管理，二维和三维语义颜色保持一致。
- 支持二维分割导出：
  - `semantic_mask`
  - `instance_mask`
  - `color_mask`
  - `overlay_vis`
  - 单实例 `binary_masks`
- 支持二维检测框导出：
  - YOLO TXT
  - Pascal VOC XML
- 支持 SUNRGBD 风格点云投影到图像平面，并根据二维 mask 生成三维点云语义。
- 支持三维框生成、质量过滤、密度过滤、点数约束、NMS 和语义离群点过滤。
- 支持 Open3D 可视化点云和三维框。
- 支持交互式三维框编辑：
  - 切换选中框
  - 选择单独的局部面
  - 对选中面缩进或扩展
  - 旋转选中框
  - 删除选中框
  - 保存编辑结果并写回导出文件

## 项目结构

```text
main/
  run.py                    # 项目启动入口和全局配置
  ui_app/                   # PyQt 标注界面
    main.py                 # UI 应用入口
    annotator.py            # 主窗口状态组装
    ui_setup.py             # 界面控件布局
    interaction.py          # 点选、SAM 推理、mask 显示与提交
    sam3_auto.py            # SAM3 自动标注
    image_state.py          # 图片切换、缓存和状态恢复
    exporting.py            # 2D/3D 标注导出
    box_editor.py           # 交互式 3D 框编辑
    config.py               # UI 默认配置读取
  tool_modules/             # 二维 mask、bbox、YOLO/VOC 工具
  tool3d_modules/           # 三维投影、赋值、框生成、过滤、导出和可视化
  sam3_auto_modules/        # SAM3 自动推理封装
  projection_tools/         # 点云投影调试工具
  sam_demo/                 # SAM/SAM3 简单演示脚本
  test_tools/               # 导出结果检查脚本
```

## 环境要求

建议使用 Python 3.9 或更高版本。主要依赖包括：

- `numpy`
- `opencv-python`
- `torch`
- `PyQt5`
- `open3d`
- `scipy`
- `matplotlib`
- `segment-anything`
- `ultralytics`

如果只使用部分功能，可以按需安装。例如，不使用 Open3D 可视化时可以暂时不安装 `open3d`；不使用 SAM3 自动标注时可以暂时不安装 `ultralytics`。

## 安装示例

```bash
pip install numpy opencv-python torch PyQt5 open3d scipy matplotlib ultralytics
```

经典 SAM 还需要安装或准备 `segment-anything`，并下载对应权重，例如 `sam_vit_h_4b8939.pth`。

SAM3 自动标注需要准备对应的 `.pt` 权重文件，并在 `main/run.py` 中配置路径。

## 运行方式

在项目根目录执行：

```bash
python main/run.py
```

如果在 Windows 上使用绝对路径，也可以直接运行：

```bash
python D:\SAM\main\run.py
```

## 核心配置

大部分常用参数集中在 `main/run.py` 中。

### 类别颜色

```python
CLASS_COLOR_TABLE = {
    "wall": (255, 0, 0),
    "floor": (0, 255, 0),
    "chair": (255, 0, 255),
}
```

二维 mask、导出文件和三维点云语义会尽量复用同一套类别颜色。

### SAM3 自动标注参数

```python
SAM3_MODEL_PATH = r"D:\sam3.pt"
SAM3_DEFAULT_TEXT_PROMPTS = "chair, table, sofa"
SAM3_DEFAULT_CONF = 0.6
SAM3_DEFAULT_IOU = 1
SAM3_DEFAULT_IMGSZ = 1024
SAM3_DEFAULT_MAX_DET = 100
SAM3_DEFAULT_HALF_MODE = "自动"
```

这些参数会作为 UI 默认值，也会作为底层 SAM3 自动推理配置的默认来源。

### 二维 mask 后处理

```python
ENABLE_2D_MASK_NMS = True
MASK_NMS_IOU_THRESHOLD = 0.65
MASK_NMS_OVERLAP_MIN_THRESHOLD = 0.80
MASK_NMS_SAME_CLASS_IOU_THRESHOLD = 0.90
MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD = 0.95

REFINE_2D_MASK_MIN_AREA = 64
REFINE_2D_MASK_KEEP_LARGEST = False
ENABLE_2D_MASK_CLOSE = True
ENABLE_2D_MASK_OPEN = False
ENABLE_2D_MASK_FILL_HOLES = True
REFINE_2D_MASK_MAX_EXTERNAL_EXPAND_PX = 2
```

说明：

- `ENABLE_2D_MASK_NMS` 控制是否删除重叠或“大包小”的二维 mask。
- `ENABLE_2D_MASK_FILL_HOLES` 控制是否对二维 mask 填洞。
- `REFINE_2D_MASK_KEEP_LARGEST=False` 会保留多个有效连通块，适合椅子腿、椅背等容易断开的目标。

### 三维标注参数

```python
USE_3D_ZBUFFER = True
USE_3D_BOX_PERCENTILE_FILTER = True
BOX_3D_LOWER_PERCENTILE = 0.5
BOX_3D_UPPER_PERCENTILE = 99.5
ENABLE_3D_BOX_NMS = False
MIN_3D_BOX_INNER_POINTS = 5000
```

说明：

- `USE_3D_ZBUFFER` 控制三维点投影到二维图像时是否只保留可见点。
- `USE_3D_BOX_PERCENTILE_FILTER` 控制生成三维框前是否用分位数过滤极端点。
- `MIN_3D_BOX_INNER_POINTS` 控制每个三维框内最少语义点数量，默认 `5000`，UI 中也可调整。

## 使用流程

1. 修改 `main/run.py` 中的路径和默认参数：
   - 图片文件夹
   - 输出目录
   - SAM / SAM3 权重路径
   - SUNRGBD 根目录
   - 类别颜色和后处理参数
2. 启动程序：
   ```bash
   python main/run.py
   ```
3. 点击“打开图片文件夹”加载图片。
4. 选择或新增类别。
5. 使用以下任一方式生成 mask：
   - 在图片上左键添加前景点、右键添加背景点，使用 SAM 点提示分割。
   - 输入 SAM3 文本类别，点击“SAM3 自动标注当前图片”。
6. 在候选 mask 中选择合适结果，点击“完成该类标注”。
7. 按需继续标注其他实例或类别。
8. 选择导出格式：
   - 仅 2D 框
   - 仅 3D 框
   - 2D + 3D 框
   - 仅 2D 分割
   - 2D + 3D 分割
9. 可点击“查看当前标注点云”检查三维语义和三维框。
10. 可点击“交互式编辑 3D 框”进入 Open3D 交互编辑。
11. 点击“完成当前图片标注并保存 → 下一张”保存结果并跳转。

## 交互式 3D 框编辑

点击 UI 中的“交互式编辑 3D 框”后，会打开 Open3D 编辑窗口。

快捷键：

```text
N / P      切换下一个 / 上一个框
1..6       选择面：+X, -X, +Y, -Y, +Z, -Z
E / W      扩展 / 缩进当前选中的面
T / G      增大 / 减小编辑步长
D / A      正向 / 反向旋转当前框
R / F      增大 / 减小旋转步长
X/Delete   删除当前框
S          保存修改并关闭
Q          关闭但不保存
```

显示规则：

- 红色框表示当前选中的框。
- 黄色边表示当前选中的面。
- 绿色框表示其他框。

保存后的编辑结果会进入当前图片缓存，并在后续导出三维框时写入 `boxes_json` 和 `boxes_txt`。

## 输出目录

每张图片会在输出目录下生成一个 `{image_name}_segmentation` 文件夹。典型结构如下：

```text
sam_output/
  000001_segmentation/
    000001_semantic_mask.png
    000001_instance_mask.png
    000001_color_mask.png
    000001_overlay_vis.png
    000001_classes.json
    binary_masks/
      000001_001_chair.png
      000001_002_table.png
    labels_2d_yolo/
      000001.txt
    labels_2d_voc/
      000001.xml
  annotations_3d/
    000001/
      000001_point_masks.npz
      000001_labeled_points.ply
      000001_3d_boxes.json
      000001_3d_boxes.txt
```

## 三维数据要求

当前三维流程默认使用 SUNRGBD 风格目录：

```text
SUNRGBD_ROOT_DIR/
  image/
    000001.jpg
  pc/
    000001.mat
  calib/
    000001.txt
```

图片文件名需要是数字编号，例如 `000001.jpg`，这样程序才能找到对应的点云和标定文件。

## 重要实现说明

- 二维 mask 保存前会执行可配置后处理，包括小连通域过滤、闭运算、开运算、填洞和外扩限制。
- 二维语义 NMS 会删除同类或异类之间明显重叠的 mask，也能处理“大包小”的包含关系。
- 三维点云语义由二维 mask 投影得到，可启用 Z-buffer 只保留可见点。
- 三维框密度只统计当前实例语义点中落在框内的点，背景点不参与。
- 三维语义滤波会在点云空间中清理飞点，并在过滤后重建三维框。
- 删除低质量三维框时，可配置是否把对应实例点改回背景。

## 常见问题

### 点击完成标注后 mask 缺了一部分

检查 `REFINE_2D_MASK_KEEP_LARGEST`。如果设置为 `True`，程序只保留最大连通块，椅子腿、椅背等断开的结构可能被删除。建议保持：

```python
REFINE_2D_MASK_KEEP_LARGEST = False
```

### 二维 mask 出现“大包小”

保持 `ENABLE_2D_MASK_NMS=True`，并根据需要调低：

```python
MASK_NMS_OVERLAP_MIN_THRESHOLD
MASK_NMS_SAME_CLASS_OVERLAP_MIN_THRESHOLD
```

### 三维框太大

可以尝试：

- 开启分位数过滤：`USE_3D_BOX_PERCENTILE_FILTER=True`
- 调整 `BOX_3D_LOWER_PERCENTILE` 和 `BOX_3D_UPPER_PERCENTILE`
- 提高 `MIN_3D_BOX_INNER_POINTS`
- 开启或增强 3D 语义滤波

### Open3D 窗口没有显示或无法交互

确认已安装 `open3d`，并且当前环境支持图形窗口。远程服务器或无桌面环境可能无法打开 Open3D 可视化窗口。

## 开发与检查

对全部源码做语法检查：

```bash
python -m compileall main
```

检查二维导出 mask：

```bash
python main/test_tools/segmentation_masks.py
```

运行点云投影调试：

```bash
python main/projection_tools/scripts.py
```
