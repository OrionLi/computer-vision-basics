# 姓名: 李弢阳
# 学号: 202211621213

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont  # ImageDraw 和 Font 将在下一步使用
import os
import numpy as np
import requests  # 用于下载示例图像

# --- 配置 ---
# 模型: 使用基于ResNet50 FPN V2骨干网络的Faster R-CNN，在COCO数据集上预训练
MODEL_WEIGHTS = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
MODEL_INSTANCE = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=MODEL_WEIGHTS)
COCO_INSTANCE_CATEGORY_NAMES = MODEL_WEIGHTS.meta["categories"]

# 确保 '__background__' 在索引0处（如果不存在）
if COCO_INSTANCE_CATEGORY_NAMES[0].lower() != '__background__':
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__'] + COCO_INSTANCE_CATEGORY_NAMES

IMAGE_DIR = '../data'
OUTPUT_DIR = '../output'
DEFAULT_IMAGE_FILENAME = 'sample_image.jpg'
# 示例图像的直接链接，用于自动下载
DEFAULT_IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'
CONFIDENCE_THRESHOLD = 0.5

# 确保输出和数据目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_sample_image_if_needed(image_dir, default_filename, url):
    # 如果默认图像不在 image_dir 中，则下载示例图像。
    default_image_path = os.path.join(image_dir, default_filename)
    if not os.path.exists(default_image_path):
        print(f"默认图像 '{default_filename}' 未在 '{image_dir}' 中找到。")
        print(f"尝试从 {url} 下载...")
        try:
            response = requests.get(url, stream=True, timeout=10)  # 添加超时设置
            response.raise_for_status()
            with open(default_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"示例图像已成功下载为 '{default_filename}' 并保存到 '{image_dir}'。")
            return default_image_path
        except requests.exceptions.RequestException as e:  # 更具体的异常处理
            print(f"下载示例图像时出错: {e}")
        except Exception as e:
            print(f"下载过程中发生意外错误: {e}")

        # 如果下载失败，创建一个占位符
        print("请手动将图像放置在 '../data/' 目录中，并根据需要更新 DEFAULT_IMAGE_FILENAME，或检查网络连接。")
        try:
            placeholder_img = Image.new('RGB', (640, 480), color='lightgrey')  # 更改颜色
            draw = ImageDraw.Draw(placeholder_img)
            try:
                # 尝试加载常用字体，若失败则回退到默认字体
                font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()

            message = "示例图像下载失败或未找到图像。\n请将真实图像（例如，sample_image.jpg）替换此占位符图像并放置在 data 文件夹中。"
            # 简单的文本换行
            lines = message.split('\n')
            y_text = 10
            for line in lines:
                draw.text((10, y_text), line, fill="black", font=font)
                y_text += font.getbbox(line)[3] + 5  # 较新的Pillow版本使用 font.getbbox(line)[3]

            placeholder_img.save(default_image_path)
            print(f"名为 '{default_filename}' 的灰色占位符图像已创建并保存到 '{image_dir}'。")
            print("请将其替换为真实图像进行检测，或确保网络连接正常以进行下载。")
        except Exception as pe:
            print(f"无法创建占位符图像: {pe}")
        return default_image_path  # 即使是占位符也返回路径

    else:
        print(f"使用现有图像: {default_image_path}")
    return default_image_path

def load_model(device):
    # 加载预训练的目标检测模型。
    model = MODEL_INSTANCE
    model.eval()
    model.to(device)
    print(f"模型: Faster R-CNN ResNet50 FPN V2 已加载到 {device}，包含 {len(COCO_INSTANCE_CATEGORY_NAMES)} 个COCO类别。")
    return model

def preprocess_image(image_path):
    # 加载并预处理图像以用于模型。
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img)
        return img, img_tensor
    except FileNotFoundError:
        print(f"错误: 在 {image_path} 未找到图像文件。如果下载失败，脚本可能已创建占位符。")
        return None, None
    except Exception as e:
        print(f"加载或预处理图像 {image_path} 时出错: {e}")
        return None, None

def predict_objects(model, img_tensor, device):
    # 对图像张量进行目标检测。
    if img_tensor is None:
        return None
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])  # 模型期望输入为图像批次
    return prediction

def filter_predictions(prediction, threshold):
    # 根据置信度阈值过滤预测结果。
    if not prediction or not prediction[0]['scores'].numel():  # 检查是否存在分数
        return np.array([]), [], np.array([])  # 返回空结构

    pred_boxes = prediction[0]['boxes']
    pred_labels = prediction[0]['labels']
    pred_scores = prediction[0]['scores']

    # 按分数过滤
    keep_indices = pred_scores > threshold

    boxes = pred_boxes[keep_indices].cpu().numpy()
    labels_indices = pred_labels[keep_indices].cpu().numpy()
    scores_filtered = pred_scores[keep_indices].cpu().numpy()

    # 将标签索引映射到类别名称
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels_indices]

    return boxes, labels, scores_filtered

def get_font(size):
    """
    尝试以给定大小加载首选的TrueType字体（DejaVuSans或Arial）。
    如果首选字体未找到，则回退到Pillow的默认位图字体。
    这确保了文本始终可以在图像上渲染。

    参数:
        size (int): 所需的字体大小。

    返回:
        ImageFont: 一个Pillow ImageFont对象。
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:  # 如果未找到 DejaVuSans.ttf
        try:
            return ImageFont.truetype("arial.ttf", size)  # 尝试使用 Arial 作为回退字体
        except IOError:  # 如果 Arial.ttf 也未找到
            return ImageFont.load_default()  # 使用Pillow的内置默认字体

def draw_and_save_image(pil_img, boxes, labels, scores, output_path, threshold):
    """
    在输入图像的副本上绘制边界框、类别标签和置信度分数，并将其保存到指定的输出路径。

    参数:
        pil_img (PIL.Image.Image): 要在其上绘制的输入图像。
                                   如果应用了修改，内部会制作副本。
        boxes (np.array): 边界框数组 (x_min, y_min, x_max, y_max)。
        labels (list): 与边界框对应的类别标签列表。
        scores (np.array): 与边界框对应的置信度分数数组。
        output_path (str): 带注释的图像将保存的路径。
        threshold (float): 用于过滤的置信度阈值（用于上下文，不用于绘制）。
    """
    # 如果原始 pil_img 可能在其他地方使用，最好在副本上绘制，
    # 尽管在当前的主脚本流程中，在调用处已经完成了 pil_img.copy()。
    # 如果此函数要在不修改输入图像的情况下重用，
    # 取消注释以下行很重要:
    # draw_image = pil_img.copy()
    # draw = ImageDraw.Draw(draw_image)
    draw = ImageDraw.Draw(pil_img)  # 假设 pil_img 已经是副本或可以直接修改。

    # 这里可以添加不同类别的颜色映射。
    # 目前，所有框使用默认颜色。
    # 示例: class_colors = {"person": "blue", "car": "green"}
    class_colors = {}
    default_color = "red"  # 边界框的默认颜色
    text_color = "white"  # 标签文本的颜色
    text_background = "black"  # 文本框的背景颜色，以提高可见性

    num_detections = len(boxes)
    if num_detections == 0:
        print("没有要绘制的对象（要么未检测到，要么所有对象都低于阈值）。")
        # 即使没有检测结果也保存原始图像:
        # try:
        #     pil_img.save(output_path)
        #     print(f"原始图像（无检测结果）已保存到: {output_path}")
        # except Exception as e:
        #     print(f"保存原始图像时出错: {e}")
        return

    print(f"正在图像上绘制 {num_detections} 个框...")

    for i in range(num_detections):
        box = boxes[i].astype(np.int32)
        label = labels[i]
        score = scores[i]
        color = class_colors.get(label, default_color)

        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)

        text = f"{label}: {score:.2f}"
        font_size = 15  # 定义标签的字体大小
        font = get_font(font_size)  # 获取字体对象

        # 计算文本大小以准备背景
        try:  # 较新的Pillow（版本9.2.0+）使用 textbbox
            # (0,0) 坐标是 textbbox 的虚拟坐标，因为它仅根据文本和字体计算大小
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:  # 较旧的Pillow版本使用 textsize
            text_width, text_height = draw.textsize(text, font=font)

        # 计算文本标签的位置。
        # 优先将文本放置在边界框上方。
        # 如果文本会超出屏幕顶部，则将其放置在框下方。
        text_y_position = box[1] - text_height - 7  # 框的 y_min - 文本高度 - 填充

        # 如果计算出的Y位置超出图像顶部，则将其移动到框下方
        if text_y_position < 0:
            text_y_position = box[3] + 7  # 框的 y_max + 填充

        # 定义文本的背景矩形
        text_bg_coords = [box[0], text_y_position, box[0] + text_width + 4, text_y_position + text_height + 4]
        draw.rectangle(text_bg_coords, fill=text_background)

        # 在背景矩形上绘制文本
        draw.text((box[0] + 2, text_y_position + 2), text, fill=text_color, font=font)  # +2 用于轻微填充

    try:
        # 保存带有绘制注释的图像
        pil_img.save(output_path)
        print(f"带有检测结果的输出图像已保存到: {output_path}")
    except Exception as e:
        print(f"保存图像到 {output_path} 时出错: {e}")

if __name__ == "__main__":
    print("--- 实验2: 目标检测 ---")
    print("姓名: 李弢阳")
    print("学号: 202211621213")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if not torch.cuda.is_available():
        print("未找到CUDA。正在CPU上运行。这可能会导致目标检测速度较慢。")

    model = load_model(device)

    # --- 图像准备 ---
    # 如果默认图像未找到，则下载示例图像，否则使用现有图像。
    image_path_to_process = download_sample_image_if_needed(IMAGE_DIR, DEFAULT_IMAGE_FILENAME, DEFAULT_IMAGE_URL)

    # --- 图像预处理 ---
    print(f"\n正在加载并预处理图像: {image_path_to_process}...")
    pil_img, img_tensor = preprocess_image(image_path_to_process)

    # --- 目标检测 ---
    if pil_img and img_tensor is not None:
        print("正在进行目标检测...")
        predictions = predict_objects(model, img_tensor, device)

        # --- 过滤并显示结果 ---
        if predictions:
            print("正在过滤预测结果...")
            boxes, labels, scores = filter_predictions(predictions, CONFIDENCE_THRESHOLD)

            print(f"\n找到 {len(boxes)} 个置信度 > {CONFIDENCE_THRESHOLD} 的对象:")
            if not labels:  # 检查标签列表是否为空
                print("未找到高于置信度阈值的对象。")
            else:
                for i in range(len(boxes)):
                    print(f"- 对象: {labels[i]}, 分数: {scores[i]:.2f}, 框: {boxes[i].astype(int)}")  # 更美观的框打印

            if pil_img and labels:  # 检查 pil_img 不为 None 且标签列表不为空
                base_image_name = os.path.basename(image_path_to_process)
                # 清理基础图像名称以创建有效的输出文件名
                safe_base_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in base_image_name)
                output_filename = "detected_" + safe_base_name
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                draw_and_save_image(pil_img.copy(), boxes, labels, scores, output_path, CONFIDENCE_THRESHOLD)  # 使用 .copy() 在新图像上绘制，如果 pil_img 会被重用
            elif pil_img:  # 当 pil_img 存在但没有标签（没有高于阈值的检测结果）
                print("未检测到高于阈值的对象，默认情况下不会保存输出图像。")
                # 可选地，如果没有检测结果，保存原始图像:
                # base_image_name = os.path.basename(image_path_to_process)
                # safe_base_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in base_image_name)
                # output_filename = "no_detections_" + safe_base_name
                # output_path = os.path.join(OUTPUT_DIR, output_filename)
                # try:
                #     pil_img.save(output_path)
                #     print(f"原始图像（由于没有检测结果高于阈值）已保存到: {output_path}")
                # except Exception as e:
                #     print(f"保存原始图像时出错: {e}")
            else:
                # 如果 pil_img 为 None，早期检查应该已经捕获到，这种情况理想情况下不会发生
                print("源图像不可用，无法绘制或保存。")
        else:
            print("模型未为给定图像返回任何预测结果。")
    else:
        print(f"无法加载或处理图像: {image_path_to_process}。请确保它是有效的图像文件，或检查下载状态。")

    print("\n脚本已完成。可调整部分包括: MODEL_INSTANCE (权重), CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_FILENAME/URL。")
    print(f"要处理不同的图像，请将其放置在 '{IMAGE_DIR}' 中，并在脚本中更改 'DEFAULT_IMAGE_FILENAME'，或修改脚本以接受命令行参数。")
