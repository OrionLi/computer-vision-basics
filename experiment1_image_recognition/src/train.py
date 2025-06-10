# Name: 李弢阳
# Student ID: 202211621213

import torch
import torch.optim as optim
import torch.nn as nn
# Assuming data_loader.py and model.py are in the same directory (src)
from data_loader import trainloader, testloader, classes
from model import SimpleCNN

# --- Configuration ---
NUM_EPOCHS = 10 # Number of epochs to train for (can be adjusted)
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = '../models/simple_cnn_cifar10.pth' # Path to save the trained model
# Create model directory if it doesn't exist
import os
os.makedirs('../models', exist_ok=True)


def train_model():
    print("Starting training process...")

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("CUDA not found. Training on CPU. This might be slow.")
        print("Ensure PyTorch was installed with CUDA support if a GPU is available:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")

    # --- Initialize Model, Loss, Optimizer ---
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {NUM_EPOCHS} epochs with learning rate {LEARNING_RATE}.")

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        epoch_train_accuracy = 100 * correct_train / total_train
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] completed. Training Accuracy: {epoch_train_accuracy:.2f}%')

        # --- Optional: Validation during training (on a subset of test data or a validation set) ---
        # For simplicity, full evaluation on test set is done after all epochs.

    print('Finished Training.')

    # --- Save the trained model ---
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'Model saved to {MODEL_SAVE_PATH}')
    except Exception as e:
        print(f"Error saving model: {e}")


    # --- Evaluation on Test Set ---
    print("\nStarting evaluation on the test set...")
    model.eval() # Set model to evaluation mode
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    with torch.no_grad(): # No gradients needed for evaluation
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy of the network on the {total_test} test images: {test_accuracy:.2f}%')
    print(f'Average loss on the test set: {avg_test_loss:.4f}')

    print("\n--- Experiment 1: Image Recognition ---")
    print("Name: 李弢阳")
    print("Student ID: 202211621213")
    print("\nThis script trained a SimpleCNN model on the CIFAR-10 dataset.")
    print("Adjustable parts of the model and training process include:")
    print("- Network architecture in `model.py` (number/type of layers, activation functions).")
    print("- Hyperparameters in `train.py` (learning rate, number of epochs, batch size in `data_loader.py`).")
    print("- Data augmentation techniques in `data_loader.py`.")
    print("- Optimizer choice and its parameters (e.g., Adam, SGD with momentum).")
    print(f"The final test accuracy was {test_accuracy:.2f}%. Training progress (loss/accuracy per epoch) was printed above.")
    print("To show specific image predictions, a separate script or function would be needed to load the saved model and specific images.")

if __name__ == '__main__':
    # Check if data loaders are empty (can happen if download failed in data_loader.py)
    if len(trainloader) == 0 or len(testloader) == 0:
        print("Data loaders are empty. This might be due to an issue with dataset downloading or loading.")
        print("Please run `python src/data_loader.py` first to check for errors and ensure the dataset is available.")
        print("If CIFAR-10 data is not found in `../data/cifar-10-batches-py`, it needs to be downloaded.")
    else:
        train_model()
