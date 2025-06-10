# Name: 李弢阳
# Student ID: 202211621213

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # nn.BatchNorm2d(32), # Optional: Batch Normalization
        # ReLU Activation
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # nn.BatchNorm2d(64), # Optional: Batch Normalization
        # ReLU Activation
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 3 (Optional, can be added for more complexity)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        # The input features to the first FC layer depend on the output of the last pooling layer.
        # CIFAR-10 images are 32x32.
        # After conv1 (3x32x32) -> pool1 (32x16x16)
        # After conv2 (32x16x16) -> pool2 (64x8x8)
        # If conv3 and pool3 were used: (128x4x4)
        self.fc1_input_features = 64 * 8 * 8 # Adjust if conv3 is added

        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        # self.dropout = nn.Dropout(0.5) # Optional: Dropout
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool2
        x = self.pool2(F.relu(self.conv2(x)))
        # If conv3 is added: x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the feature maps
        x = x.view(-1, self.fc1_input_features)

        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # x = self.dropout(x) # Optional: Dropout
        # FC2 (Output Layer)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    print("Model definition script executed.")
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Instantiate the model
    model = SimpleCNN(num_classes=10)
    print("\nModel Architecture:")
    print(model)

    # Perform a forward pass with the dummy input
    try:
        output = model(dummy_input)
        print(f"\nOutput shape from dummy input: {output.shape}") # Expected: [1, 10]
        print("Model seems to be defined correctly.")
    except Exception as e:
        print(f"Error during forward pass with dummy input: {e}")

    print("\nThis script defines a Simple Convolutional Neural Network (CNN) for CIFAR-10 image classification.")
    print("The model consists of convolutional layers, pooling layers, and fully connected layers.")
    print("To see a detailed summary (parameters, memory), you can use torchsummary if installed:")
    print("# from torchsummary import summary")
    print("# summary(model, (3, 32, 32)) # Assuming model is on CPU")
