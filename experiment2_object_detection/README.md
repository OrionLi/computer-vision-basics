### 实验二：目标检测

姓名：
学号：

本实验聚焦于使用预训练的卷积神经网络（CNN）结合 PyTorch 进行目标检测。目标是输入一张图像，检测其中的目标，并在目标周围绘制带有合适标签的边界框。

#### 目录结构
- `data/`：该目录可用于存储用于检测的样本图像。你需要自行添加图像到该目录。
- `src/`：包含实验所需的 Python 脚本。
  - `object_detector.py`：此脚本加载预训练模型（Faster R - CNN ResNet50 FPN V2），对样本图像进行检测（必要时下载一张图像），根据置信度过滤结果，在图像上绘制边界框和标签，并将结果图像保存到 `output/` 目录。它还会将检测到的目标详细信息打印到控制台。
- `output/`：该目录将存储由 `object_detector.py` 生成的带有检测到的目标及其边界框的图像。

#### 环境搭建

##### PyTorch 安装
确保你已经安装了 PyTorch 和 Torchvision。指定的安装命令如下：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
（注意：`cu128` 表示与 CUDA 12.8 兼容的版本。如果使用 CPU 或其他 CUDA 版本，请进行相应调整。目标检测脚本会尝试在可用时使用 CUDA，否则使用 CPU。）

##### 额外库
脚本使用 `Pillow`（PIL）进行图像加载和绘制边界框，使用 `requests` 下载样本图像，`numpy` 是通用依赖库。这些库通常会随 PyTorch 一起安装，或者是常见的 Python 库。
如果你希望扩展脚本或更广泛地处理图像数据，以下库可能也会有用：
- `opencv - python`：用于更高级的图像/视频处理。
- `matplotlib`：用于在 Python 环境（如 Jupyter 笔记本）中显示图像。

如果尚未安装，可通过 pip 安装 Pillow：
```bash
pip3 install Pillow requests numpy # 确保这些核心依赖项存在
# pip3 install opencv-python matplotlib # 可选扩展
```

#### 实验概述
1. **模型选择**：我们将使用 `torchvision.models.detection` 中的预训练目标检测模型，例如带有 ResNet50 骨干网络的 Faster R - CNN。这些模型在 COCO 等大型数据集上进行了训练，可以检测各种常见目标。
2. **数据（样本图像）**：你需要提供自己的样本图像供模型进行检测。将这些图像放置在 `experiment2_object_detection/data/` 目录中。
3. **检测脚本 (`object_detector.py`)**：
    - 加载预训练模型。
    - 从 `data/` 目录加载样本图像。
    - 将图像预处理为模型期望的格式。
    - 进行推理以获取目标检测结果（边界框、类别标签和置信度分数）。
    - 根据置信度阈值过滤检测结果。
    - 在图像上绘制边界框和标签。
    - 将结果图像保存到 `output/` 目录（例如，`output/detected_sample_image.jpg`）。
4. **结果与分析**：
    - 查看 `output/` 目录中保存的带有检测结果的输出图像。
    - 脚本还会将检测到的目标的名称、置信度分数和边界框坐标打印到控制台。
    - 理解模型的预测结果，包括检测到的目标类别及其位置。
    - 脚本讨论了可调整的参数，如模型选择和置信度阈值。

#### 数据标注（简要概述）
虽然我们使用的是已经“知道”目标类别并能找到它们的预训练模型，但从头开始训练目标检测模型（或微调现有模型）需要带有特定标注的数据集。对于每张图像，这些标注通常包括：
- **边界框**：坐标（例如，x_min, y_min, x_max, y_max）定义了每个感兴趣目标周围的矩形。
- **类别标签**：边界框内每个目标的类别（例如，“汽车”、“人”、“狗”）。

本实验利用了在 COCO 等数据集上预训练的模型，COCO 有 80 个常见目标类别。

#### 运行实验二

1. **准备环境**：
    - 确保已安装 PyTorch、Torchvision 和其他所需库（`requests`、`Pillow`（PIL）、`numpy`）。参考“环境搭建”部分的安装命令。
    - 如果第一次运行时满足以下条件，则需要互联网连接：
        - PyTorch 需要下载预训练模型权重。
        - 默认样本图像 (`../data/sample_image.jpg`) 不存在，需要从 `DEFAULT_IMAGE_URL` 下载。
2. **准备输入图像**：
    - 脚本首先会在 `experiment2_object_detection/data/` 目录中查找名为 `sample_image.jpg`（或 `DEFAULT_IMAGE_FILENAME` 设置的其他名称）的图像。
    - 如果该图像不存在，脚本将尝试下载 `DEFAULT_IMAGE_URL` 指定的图像（当前是齐达内的照片），并将其保存为 `DEFAULT_IMAGE_FILENAME` 到 `data/` 目录。
    - **使用你自己的图像**：
        1. 将你的图像文件（例如，`my_photo.jpg`）放入 `experiment2_object_detection/data/` 目录。
        2. 在 `src/object_detector.py` 中，将 `DEFAULT_IMAGE_FILENAME` 变量更改为你的图像名称（例如，`DEFAULT_IMAGE_FILENAME = 'my_photo.jpg'`）。
3. **运行检测脚本**：
    - 在终端中导航到 `experiment2_object_detection` 目录。
    - 执行 `object_detector.py` 脚本：
```bash
cd experiment2_object_detection
python src/object_detector.py
```
    - 脚本将执行以下操作：
        - 加载指定的图像。
        - 加载预训练的 Faster R - CNN 模型（如果是第一次运行，则下载权重）。
        - 进行目标检测。
        - 将检测到的目标详细信息（类别、分数、边界框坐标）打印到控制台。
        - 在图像上绘制这些检测结果。
        - 将标注后的图像保存到 `experiment2_object_detection/output/` 目录（例如，`output/detected_sample_image.jpg`）。

#### 输出解释

- **控制台输出**：
    - **设置信息**：PyTorch/Torchvision 版本、使用的设备（CUDA 或 CPU）。
    - **图像状态**：关于是否使用现有图像，或者下载尝试、成功或失败的消息。
    - **检测结果**：对于每个高于 `CONFIDENCE_THRESHOLD` 的检测到的目标，脚本将打印：
        - `目标: [类别名称], 分数: [置信度分数], 边界框: [x_min, y_min, x_max, y_max]`
- **可视化输出**：
    - 一个图像文件（例如，`detected_sample_image.jpg`）将保存到 `experiment2_object_detection/output/` 目录。该图像将在检测到的目标周围绘制边界框，并显示其类别标签和置信度分数。

#### 脚本可调整部分
你可以修改 `src/object_detector.py` 来进行以下实验：
- **不同模型**：
    - 更改 `MODEL_WEIGHTS` 和 `MODEL_INSTANCE` 以使用 `torchvision.models.detection` 中其他预训练的目标检测模型（例如，不同版本的 Faster R - CNN、SSD、RetinaNet）。如果新模型使用不同的类别集，或者 `MODEL_WEIGHTS.meta["categories"]` 提供了类别信息，请确保更新 `COCO_INSTANCE_CATEGORY_NAMES`。
- **置信度阈值**：
    - 修改 `CONFIDENCE_THRESHOLD`（例如，从 `0.5` 改为 `0.7` 以获得更有把握的检测结果，或改为 `0.3` 以获得更多可能不太准确的检测结果）。
- **输入图像**：
    - 将 `DEFAULT_IMAGE_FILENAME` 更改为你放置在 `data/` 目录中的图像文件名。
    - 如果你希望脚本在 `DEFAULT_IMAGE_FILENAME` 不存在时下载不同的图像，可以更改 `DEFAULT_IMAGE_URL`。

#### 故障排除和注意事项
- **互联网连接**：第一次运行时需要互联网连接，以便从 PyTorch 下载模型权重并可能下载样本图像。使用相同模型和现有图像的后续运行可能不需要互联网。
- **依赖项**：确保安装了 `Pillow`（PIL）和 `opencv - python`，因为它们在计算机视觉项目中常用于图像操作，尽管此脚本主要使用 Pillow 进行绘制。`requests` 用于下载样本图像，`numpy` 是通用依赖项。
- **检测速度**：目标检测计算量较大。在 CPU（`device: cpu`）上运行比在支持 CUDA 的 GPU（`device: cuda`）上运行要慢得多。
- **占位图像**：如果样本图像下载失败且你没有手动放置图像，则会创建一个灰色占位图像。模型可能无法在这个占位图像中检测到任何有意义的目标。