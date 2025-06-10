# Name: 李弢阳
# Student ID: 202211621213

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np # For example predictions
import matplotlib.pyplot as plt # For example predictions (optional display)

# --- Model Definition (copied from model.py) ---
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

# --- Data Configuration & Preparation ---
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
    print(f"Error downloading or loading CIFAR-10 dataset: {e}")
    print("Install PyTorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    exit()

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- Training Configuration ---
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = '../models/simple_cnn_cifar10.pth'

# --- Function to show example predictions (Integrated from predict_examples.py) ---
def show_example_predictions_after_training(model_to_show, device_to_use, current_testloader, current_classes, num_examples=5):
    print("\n--- Showing Example Predictions ---")
    model_to_show.eval() # Ensure model is in evaluation mode

    if len(current_testloader.dataset) == 0:
        print("Testloader is empty. Cannot fetch images for prediction.")
        return

    dataiter = iter(current_testloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("Could not get a batch from testloader for example predictions.")
        return

    images, labels = images.to(device_to_use), labels.to(device_to_use)

    with torch.no_grad():
        outputs = model_to_show(images)
        _, predicted_indices = torch.max(outputs, 1)

    print(f"\nExample Predictions (First {num_examples} images from a test batch):")
    for i in range(min(num_examples, images.size(0))):
        actual_label = current_classes[labels[i]]
        predicted_label = current_classes[predicted_indices[i]]
        print(f"Image #{i+1}: Actual: {actual_label:10s} | Predicted: {predicted_label:10s} {'(Correct)' if actual_label == predicted_label else '(Incorrect)'}")

    # --- How to interpret training process (from original predict_examples.py) ---
    print("\n--- Interpreting Training Process (Summary) ---")
    print("During training, the script outputs:")
    print("1. Loss per mini-batch: Shows if the model is learning (loss decreasing).")
    print("2. Training accuracy per epoch: Indicates how well the model fits training data.")
    print("3. Test accuracy after all epochs: Shows generalization to unseen data.")
    print("   - High training accuracy + low test accuracy might indicate overfitting.")
    print("   - Both increasing and stabilizing is ideal. Loss values should decrease.")
    print("For detailed analysis, plot loss & accuracy (train/test) per epoch (not implemented in this script).")

    # --- (Optional) Display images with predictions ---
    # try:
    #     fig = plt.figure(figsize=(15, 7))
    #     for i in range(min(num_examples, images.size(0))):
    #         ax = fig.add_subplot(1, num_examples, i + 1, xticks=[], yticks=[])
    #         img_to_show = images[i].cpu() / 2 + 0.5  # Unnormalize
    #         npimg = img_to_show.numpy()
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #         ax.set_title(f"Pred: {current_classes[predicted_indices[i]]}\nActual: {current_classes[labels[i]]}",
    #                      color=("green" if predicted_indices[i] == labels[i] else "red"))
    #     plt.show()
    #     print("\nDisplayed example images with predictions using Matplotlib.")
    # except Exception as e:
    #     print(f"\nCould not display images with Matplotlib: {e}. Text predictions are shown above.")
    #     print("Ensure you have a GUI environment if you uncomment the display code.")

def train_model():
    print("Starting training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("CUDA not found. Training on CPU. This might be slow.")
        print("Install PyTorch with CUDA: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {NUM_EPOCHS} epochs with learning rate {LEARNING_RATE}.")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels_data = data # Renamed labels to labels_data to avoid conflict
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
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        epoch_train_accuracy = 100 * correct_train / total_train
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] completed. Training Accuracy: {epoch_train_accuracy:.2f}%')

    print('Finished Training.')

    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'Model saved to {MODEL_SAVE_PATH}')
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nStarting evaluation on the test set...")
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images_eval, labels_eval = data # Renamed to avoid conflict
            images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
            outputs_eval = model(images_eval)
            loss_eval = criterion(outputs_eval, labels_eval)
            test_loss += loss_eval.item()
            _, predicted_eval = torch.max(outputs_eval.data, 1)
            total_test += labels_eval.size(0)
            correct_test += (predicted_eval == labels_eval).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy of the network on the {total_test} test images: {test_accuracy:.2f}%')
    print(f'Average loss on the test set: {avg_test_loss:.4f}')

    print("\n--- Experiment 1: Image Recognition (Consolidated Script) ---")
    print("Name: 李弢阳")
    print("Student ID: 202211621213")
    print("\nThis script defines, trains, evaluates a SimpleCNN model on CIFAR-10, and shows example predictions.")
    print("Adjustable parts include network architecture, hyperparameters, data augmentation (all in this script).")
    print(f"The final test accuracy was {test_accuracy:.2f}%.")

    # Call the function to show example predictions
    show_example_predictions_after_training(model, device, testloader, classes)

if __name__ == '__main__':
    if len(trainloader.dataset) == 0 or len(testloader.dataset) == 0:
        print("Data loaders are effectively empty. Check dataset loading/downloading.")
    else:
        train_model()
