# 姓名: 
# 学号: 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np # 用于示例预测
import matplotlib.pyplot as plt # 用于示例预测（可选显示）

# --- 模型定义 (从model.py复制) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_features = 64 * 8 * 8
        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 数据配置与准备 ---
BATCH_SIZE = 64
NUM_WORKERS = 2
DATA_DIR = '../data'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('../models', exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

try:
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
except Exception as e:
    print(f"下载或加载CIFAR-10数据集时出错: {e}")
    print("安装PyTorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    exit()

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- 训练配置 ---
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = '../models/simple_cnn_cifar10.pth'

# --- 展示训练后示例预测的函数 (整合自predict_examples.py) ---
def show_example_predictions_after_training(model_to_show, device_to_use, current_testloader, current_classes, num_examples=5):
    print("\n--- 展示示例预测结果 ---")
    model_to_show.eval() # 确保模型处于评估模式

    if len(current_testloader.dataset) == 0:
        print("测试加载器为空。无法获取图像进行预测。")
        return

    dataiter = iter(current_testloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("无法从测试加载器获取批次数据用于示例预测。")
        return

    images, labels = images.to(device_to_use), labels.to(device_to_use)

    with torch.no_grad():
        outputs = model_to_show(images)
        _, predicted_indices = torch.max(outputs, 1)

    print(f"\n示例预测 (测试批次中的前{num_examples}张图像):")
    for i in range(min(num_examples, images.size(0))):
        actual_label = current_classes[labels[i]]
        predicted_label = current_classes[predicted_indices[i]]
        print(f"图像 #{i+1}: 实际: {actual_label:10s} | 预测: {predicted_label:10s} {'(正确)' if actual_label == predicted_label else '(错误)'}")

    # --- 如何解释训练过程 (来自原始predict_examples.py) ---
    print("\n--- 解释训练过程 (总结) ---")
    print("训练期间，脚本会输出:")
    print("1. 每个小批量的损失: 显示模型是否在学习(损失是否下降)。")
    print("2. 每个epoch的训练准确率: 表示模型对训练数据的拟合程度。")
    print("3. 所有epoch后的测试准确率: 显示模型对未见数据的泛化能力。")
    print("   - 高训练准确率 + 低测试准确率可能表示过拟合。")
    print("   - 两者都增加并稳定是理想情况。损失值应该下降。")
    print("如需详细分析，请绘制每个epoch的损失和准确率(训练/测试)(本脚本未实现)。")

    # --- (可选) 显示带有预测结果的图像 ---
    try:
        fig = plt.figure(figsize=(15, 7))
        for i in range(min(num_examples, images.size(0))):
            ax = fig.add_subplot(1, num_examples, i + 1, xticks=[], yticks=[])
            img_to_show = images[i].cpu() / 2 + 0.5  # 反归一化
            npimg = img_to_show.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(f"Pred: {current_classes[predicted_indices[i]]}\nActual: {current_classes[labels[i]]}",
                         color=("green" if predicted_indices[i] == labels[i] else "red"))
        plt.show()
        print("\n使用Matplotlib显示了带有预测结果的示例图像。")
    except Exception as e:
        print(f"\n无法使用Matplotlib显示图像: {e}。上面显示了文本预测结果。")
        print("如果取消注释显示代码，请确保您有GUI环境。")

def train_model():
    print("开始训练过程...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if not torch.cuda.is_available():
        print("未找到CUDA。在CPU上训练。这可能会很慢。")
        print("安装带CUDA的PyTorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"训练{NUM_EPOCHS}个epoch，学习率为{LEARNING_RATE}。")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels_data = data # 将labels重命名为labels_data以避免冲突
            inputs, labels_data = inputs.to(device), labels_data.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels_data.size(0)
            correct_train += (predicted == labels_data).sum().item()
            if (i + 1) % 100 == 0:
                print(f'第[{epoch + 1}/{NUM_EPOCHS}]个epoch, 第[{i + 1}/{len(trainloader)}]步, 损失: {running_loss / 100:.4f}')
                running_loss = 0.0
        epoch_train_accuracy = 100 * correct_train / total_train
        print(f'第[{epoch + 1}/{NUM_EPOCHS}]个epoch完成。训练准确率: {epoch_train_accuracy:.2f}%')

    print('训练完成。')

    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'模型已保存到{MODEL_SAVE_PATH}')
    except Exception as e:
        print(f"保存模型时出错: {e}")

    print("\n开始在测试集上进行评估...")
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images_eval, labels_eval = data # 重命名以避免冲突
            images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
            outputs_eval = model(images_eval)
            loss_eval = criterion(outputs_eval, labels_eval)
            test_loss += loss_eval.item()
            _, predicted_eval = torch.max(outputs_eval.data, 1)
            total_test += labels_eval.size(0)
            correct_test += (predicted_eval == labels_eval).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = 100 * correct_test / total_test
    print(f'网络在{total_test}张测试图像上的准确率: {test_accuracy:.2f}%')
    print(f'测试集上的平均损失: {avg_test_loss:.4f}')

    print("\n--- 实验1: 图像识别 (合并脚本) ---")
    print("姓名: 李弢阳")
    print("学号: 202211621213")
    print("\n此脚本定义、训练、评估一个SimpleCNN模型在CIFAR-10上，并展示示例预测。")
    print("可调整部分包括网络架构、超参数、数据增强(均在此脚本中)。")
    print(f"最终测试准确率为{test_accuracy:.2f}%。")

    # 调用函数展示示例预测
    show_example_predictions_after_training(model, device, testloader, classes)

if __name__ == '__main__':
    if len(trainloader.dataset) == 0 or len(testloader.dataset) == 0:
        print("数据加载器实际上为空。检查数据集加载/下载。")
    else:
        train_model()    