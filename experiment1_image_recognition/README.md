# Experiment 1: Image Recognition

Student: 李弢阳
ID: 202211621213

This experiment focuses on image recognition using Convolutional Neural Networks (CNNs) with PyTorch.

## Directory Structure
- `data/`: Will store the CIFAR-10 dataset (downloaded automatically).
- `src/`: Contains the Python scripts for the experiment.
  - `data_loader.py`: Handles dataset loading, preprocessing, and augmentation for CIFAR-10.
  - `model.py`: Defines the Convolutional Neural Network (CNN) architecture.
  - `train.py`: Handles the model training, evaluation, and saving.
  - `predict_examples.py`: Loads the trained model and shows predictions for a few example images from the test set.
- `results/`: (To be created) May store training plots, saved models, etc.

## Setup and Data Preparation

### PyTorch Installation
Ensure you have PyTorch installed. The specified installation command is:
\`\`\`bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
\`\`\`
(Note: `cu128` implies a CUDA 12.8 compatible build. Adjust if using CPU or a different CUDA version.)

### `src/data_loader.py`
This script performs the following:
1.  **Defines Transformations**:
    *   For Training: Random cropping, random horizontal flipping, conversion to tensor, and normalization.
    *   For Testing: Conversion to tensor and normalization.
    *   The normalization values are standard for CIFAR-10.
2.  **Downloads/Loads CIFAR-10**:
    *   Uses `torchvision.datasets.CIFAR10` to get the training and testing sets.
    *   Data is stored in the `experiment1_image_recognition/data/` directory.
3.  **Creates DataLoaders**:
    *   `trainloader`: For the training set, with shuffling and batching.
    *   `testloader`: For the test set, with batching.
4.  **Verification**:
    *   If run directly (`python src/data_loader.py`), it prints dataset statistics and the labels of a few sample images. Image display is commented out by default to ensure compatibility in non-GUI environments.

This script is primarily intended to be imported by the main training script but can be run standalone to verify data loading.

## Running Experiment 1

1.  **Ensure Data is Ready**:
    *   The `data_loader.py` script will attempt to download CIFAR-10 automatically when `train.py` is run.
    *   You can optionally run `python src/data_loader.py` from the `experiment1_image_recognition` directory to verify data loading and see dataset statistics beforehand.
        \`\`\`bash
        cd experiment1_image_recognition
        python src/data_loader.py
        cd ..
        \`\`\`

2.  **Run the Training Script**:
    *   Navigate to the `experiment1_image_recognition` directory.
    *   Execute the `train.py` script using Python.
        \`\`\`bash
        cd experiment1_image_recognition
        python src/train.py
        \`\`\`
    *   This will:
        *   Load the data using `data_loader.py`.
        *   Initialize the CNN model from `model.py`.
        *   Train the model on the CIFAR-10 training set for a predefined number of epochs.
        *   Print training progress (loss and accuracy per epoch).
        *   Evaluate the trained model on the CIFAR-10 test set and print the final test accuracy.
        *   Save the trained model weights to `experiment1_image_recognition/models/simple_cnn_cifar10.pth`.

### Expected Output
The script will output:
- The device being used (CPU or CUDA).
- Training loss at intervals during each epoch.
- Training accuracy at the end of each epoch.
- Final test accuracy and loss after training is complete.
- A confirmation message that the model has been saved.
- A summary of adjustable parts for the experiment.

### Notes
- Training a CNN can take time, especially on a CPU. For quick testing, you can reduce `NUM_EPOCHS` in `src/train.py`.
- The PyTorch installation command `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` is for a system with a CUDA 12.8 compatible GPU. If you are using a CPU or a different CUDA version, the installation command might differ. The script will fall back to CPU if CUDA is not available.

### Viewing Example Predictions

After training the model using `python src/train.py` (which saves `simple_cnn_cifar10.pth` in the `models` directory):

1.  **Run the Prediction Script**:
    *   Navigate to the `experiment1_image_recognition` directory.
    *   Execute the `predict_examples.py` script:
        \`\`\`bash
        cd experiment1_image_recognition # if not already there
        python src/predict_examples.py
        \`\`\`
    *   This script will:
        *   Load the saved model.
        *   Fetch a few images from the CIFAR-10 test set.
        *   Print the actual and predicted class labels for these images.
        *   Provide a brief explanation on interpreting training process metrics.
    *   The script also contains commented-out code using Matplotlib to display these images with their labels. If you are running in an environment with a GUI and have Matplotlib installed, you can uncomment this section in `src/predict_examples.py` to see visual results.

### Interpreting Training Process (from `train.py`)

The `train.py` script outputs the following during and after training:
- **Training Loss (per mini-batch and epoch):** A decreasing loss indicates the model is learning.
- **Training Accuracy (per epoch):** Shows how well the model fits the data it's being trained on.
- **Test Accuracy (at the end):** Crucially, this shows how well the model generalizes to new, unseen data. A significant gap between high training accuracy and lower test accuracy suggests overfitting.

Ideally, both training and test accuracies should increase and then plateau, while the loss values should decrease. These metrics help in understanding model performance and in making decisions about hyperparameter tuning or model architecture changes (the "adjustable parts" mentioned in the experiment description). For more detailed analysis, these metrics (loss and accuracy per epoch for both training and validation/test sets) are often logged and plotted over epochs.
