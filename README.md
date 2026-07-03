# semi-automatic-2D-and-3D-joint-labeling
# RGB-D 2D/3D Annotation Tool

这是一个面向 RGB-D 数据的半自动标注工具，主要用于实现 **2D 图像分割标注、3D 点云同步标注、2D/3D 可视化检查和标注结果保存**。项目结合 SAM 分割、相机投影矩阵和点云处理方法，支持从 RGB 图像标注结果反向映射到对应 3D 点云中，提高 RGB-D 数据集构建效率。

## 项目简介

本工具主要用于室内 RGB-D 数据标注场景,测试数据集为SUNRGB-D数据集。用户可以在 2D 图像上通过点选、框选等方式调用 SAM 生成候选掩膜，并从多个候选 mask 中选择效果最好的分割结果。随后系统会对掩膜进行后处理，并基于相机内参、外参或投影矩阵，将 2D 掩膜区域对应到 3D 点云中，实现 2D 与 3D 标注结果的同步生成。
如想运行需下载Sam的预训练权重，并于ui.py文件中进行路径替换

该工具适用于：

* RGB-D 数据集构建
* 室内场景目标标注
* 2D 图像与 3D 点云联合标注
* 三维目标检测数据预处理
* 点云语义分割数据辅助生成

## 主要功能

### 1. 2D 图像交互式标注

支持在 RGB 图像上进行目标交互式标注，可通过点击点、框选区域等方式生成目标掩膜。

### 2. SAM 多候选 Mask 选择

调用 SAM 模型后，系统可生成多个候选分割结果，例如：

* Mask 1
* Mask 2
* Mask 3

用户可以根据边界质量和目标完整性选择最合适的 mask 作为最终标注结果。

### 3. 掩膜后处理

对 SAM 输出的原始 mask 进行进一步处理，包括：

* 空洞填充
* 连通域筛选
* 离群区域去除
* 形态学平滑
* 边界优化

用于减少分割结果中的噪声和错误区域。

### 4. 2D 标注格式保存

支持将最终 mask 转换为常见 2D 标注格式，例如：

* 分割 mask
* 2D bounding box
* YOLO 格式
* VOC 格式

可用于后续训练 2D 检测或分割模型。

### 5. 3D 点云同步标注

基于 RGB-D 数据中的相机参数，将 2D 图像上的 mask 区域映射到对应的 3D 点云中，实现点云级别的语义标注。

基本流程如下：

```text
RGB 图像标注
    ↓
生成 2D mask
    ↓
读取深度图 / 点云 / calib
    ↓
根据相机投影关系建立 2D-3D 对应
    ↓
筛选 mask 内对应点云
    ↓
生成 3D 点云标注结果
```

### 6. 点云可视化检查

支持查看当前图像对应的原始点云和标注点云。显示时保留完整点云结构，仅对被标注的目标点进行上色，便于检查 2D mask 与 3D 点云之间的对应关系是否正确。

### 7. 语义图滤波强度控制

提供 3D 语义图滤波强度设置，支持多个档位：

```text
0 / 1 / 2 / 3
```

用于控制点云离群点去除强度，提高 3D 标注结果的稳定性。

## 技术流程

整体处理流程如下：

```text
加载 RGB 图像、深度图、点云和相机参数
        ↓
用户在 2D 图像上进行点选或框选
        ↓
调用 SAM 生成多个候选 mask
        ↓
用户选择最佳 mask
        ↓
对 mask 进行后处理
        ↓
保存 2D 标注结果
        ↓
利用相机投影矩阵建立 2D-3D 对应关系
        ↓
提取 mask 区域对应的 3D 点
        ↓
生成 3D 标注结果
        ↓
可视化检查标注点云
```

## 使用到的主要技术

* Python
* PyQt5
* OpenCV
* NumPy
* Open3D
* Segment Anything Model
* RGB-D 相机投影
* 点云滤波
* 连通域分析
* 形态学图像处理
* 2D/3D 标注格式转换

## 数据输入

项目通常需要以下数据：

```text
dataset/
├── image/
│   ├── rgb/
│   └── depth/
├── pointcloud/
├── calib/
└── labels/
```

其中：

* `rgb/`：RGB 图像
* `depth/`：深度图
* `pointcloud/`：点云文件
* `calib/`：相机参数文件
* `labels/`：保存生成的 2D/3D 标注结果

## 标注结果输出

输出内容包括：

```text
output/
├── masks/
├── yolo_labels/
├── voc_labels/
├── pointcloud_labels/
└── visualization/
```

其中：

* `masks/`：最终分割掩膜
* `yolo_labels/`：YOLO 格式 2D 框
* `voc_labels/`：VOC 格式 2D 框
* `pointcloud_labels/`：3D 点云标注结果
* `visualization/`：可视化检查结果

## 运行方式

安装依赖：

```bash
pip install pyqt5 opencv-python numpy open3d torch torchvision
```

运行主程序：

```bash
python main.py
```

根据实际代码文件名，也可以修改为：

```bash
python annotation_tool.py
```

## 项目特点

* 支持 2D 与 3D 标注同步生成
* 使用 SAM 降低人工分割成本
* 支持多候选 mask 选择，提高标注精度
* 支持点云标注结果可视化检查
* 可输出常见 2D 检测和分割格式
* 适合 RGB-D 数据采集、数据清洗和三维检测模型训练前处理

## 应用场景

该工具可用于构建室内 RGB-D 多模态数据集，辅助完成 2D 图像目标检测、图像分割、3D 点云语义分割和三维目标检测等任务的数据准备工作。

## 后续优化方向

* 支持更多数据集格式，例如 SUN RGB-D、KITTI、ScanNet
* 增加批量自动标注功能
* 增加 3D bounding box 自动估计
* 增加标注质量检查模块
* 支持更多点云滤波和边界优化算法
* 支持导出 MMDetection3D 格式数据

## License

This project is for research and learning purposes.
