# Name: 李弢阳
# Student ID: 202211621213

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Assuming data_loader.py and model.py are in the same directory (src)
from data_loader import testloader, classes, transform_test # We need transform_test for custom images if used
from model import SimpleCNN

# --- Configuration ---
MODEL_PATH = '../models/simple_cnn_cifar10.pth'
NUM_EXAMPLES = 5 # Number of examples to show

def show_example_predictions():
    print("Loading model and showing example predictions...")

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = SimpleCNN(num_classes=len(classes))
    try:
        # Load the trained model state dict
        # Add map_location to handle loading CUDA-trained model on CPU-only environment
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please ensure you have run the training script (train.py) first to generate the model.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval() # Set model to evaluation mode

    print(f"Model loaded from {MODEL_PATH}")

    # --- Get some test images ---
    if len(testloader) == 0:
        print("Testloader is empty. Cannot fetch images for prediction.")
        print("Please check data_loader.py and ensure dataset is available.")
        return

    dataiter = iter(testloader)

    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("Could not get a batch from testloader. It might be empty or an issue with data loading.")
        return

    images, labels = images.to(device), labels.to(device)

    # --- Make Predictions ---
    with torch.no_grad():
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)

    print(f"\n--- Example Predictions (First {NUM_EXAMPLES} images from a test batch) ---")
    for i in range(min(NUM_EXAMPLES, images.size(0))):
        actual_label = classes[labels[i]]
        predicted_label = classes[predicted_indices[i]]
        print(f"Image #{i+1}: Actual: {actual_label:10s} | Predicted: {predicted_label:10s} {'(Correct)' if actual_label == predicted_label else '(Incorrect)'}")

    # --- How to interpret training process ---
    print("\n--- Interpreting Training Process (from train.py output) ---")
    print("During training (running `train.py`), the script outputs:")
    print("1. Loss per mini-batch: Shows if the model is learning and if the loss is decreasing.")
    print("2. Training accuracy per epoch: Indicates how well the model is fitting the training data.")
    print("3. Test accuracy after all epochs: Shows how well the model generalizes to unseen data.")
    print("   - A high training accuracy but low test accuracy might indicate overfitting.")
    print("   - Both training and test accuracy increasing and stabilizing is ideal.")
    print("   - Loss values (both training and test) should generally decrease over time.")
    print("For visualization, you would typically log these values (loss, train_acc, test_acc per epoch) and plot them using a library like Matplotlib.")
    print("This script (`predict_examples.py`) focuses on showing concrete classification examples using the trained model.")

    # --- (Optional) Display images with predictions ---
    # This part requires a GUI environment to display images with matplotlib.
    # It might not work in all execution environments (e.g., some remote servers or containers).
    # If you run this locally and have a display, you can uncomment it.
    # try:
    #     fig = plt.figure(figsize=(15, 7))
    #     for i in range(min(NUM_EXAMPLES, images.size(0))):
    #         ax = fig.add_subplot(1, NUM_EXAMPLES, i + 1, xticks=[], yticks=[])
    #         img = images[i].cpu() / 2 + 0.5  # Unnormalize
    #         npimg = img.numpy()
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #         ax.set_title(f"Pred: {classes[predicted_indices[i]]}\nActual: {classes[labels[i]]}",
    #                      color=("green" if predicted_indices[i] == labels[i] else "red"))
    #     plt.show()
    #     print("\nDisplayed example images with predictions using Matplotlib.")
    #     print("(Close the Matplotlib window to continue.)")
    # except Exception as e:
    #     print(f"\nCould not display images with Matplotlib: {e}")
    #     print("This often happens in environments without a GUI. Text predictions are shown above.")


if __name__ == '__main__':
    # Ensure data is available for testloader to work
    if len(testloader.dataset) == 0:
         print("CIFAR-10 test dataset is not loaded. Please check `data_loader.py` and ensure data is downloaded.")
         print("You might need to run `python src/data_loader.py` once from `experiment1_image_recognition` directory.")
    else:
        show_example_predictions()
