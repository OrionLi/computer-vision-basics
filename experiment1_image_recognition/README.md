# Experiment 1: Image Recognition

Student: 李弢阳
ID: 202211621213

This experiment focuses on image recognition using Convolutional Neural Networks (CNNs) with PyTorch.
The goal is to define, train, and evaluate a CNN model for classifying images from the CIFAR-10 dataset.
The final script also saves the trained model and shows some example predictions.

## Directory Structure
- `data/`: Stores the CIFAR-10 dataset (downloaded automatically by the script).
- `models/`: Stores the saved trained model weights (e.g., `simple_cnn_cifar10.pth`).
- `src/`: Contains the Python script for the experiment.
  - `train.py`: A single, consolidated script that handles:
      - CIFAR-10 dataset loading, preprocessing, and augmentation (with automatic download if not present).
      - Definition of the `SimpleCNN` model architecture.
      - Training the model on the CIFAR-10 training set.
      - Evaluating the trained model on the test set.
      - Saving the trained model weights.
      - Displaying a few example predictions from the test set with their actual vs. predicted labels.
      - Providing notes on interpreting the training process.
- `README.md`: This file, providing an overview of the experiment.

## Setup

### PyTorch Installation
Ensure you have PyTorch installed. If not, a typical installation command is:
\`\`\`bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
\`\`\`
(Note: `cu128` implies a CUDA 12.8 compatible build. Adjust this command if you are using a CPU-only environment or a different CUDA version. The script will attempt to use CUDA if available, otherwise it will fall back to CPU.)

You might also need `matplotlib` if you choose to uncomment the image display part in `src/train.py`:
\`\`\`bash
pip3 install matplotlib numpy
\`\`\`
(`numpy` is generally a core dependency for PyTorch and data handling.)

## Running Experiment 1

1.  **Navigate to the Experiment Directory**:
    Open your terminal and change to the `experiment1_image_recognition` directory.
    \`\`\`bash
    cd experiment1_image_recognition
    \`\`\`

2.  **Run the Training Script**:
    Execute the `train.py` script using Python:
    \`\`\`bash
    python src/train.py
    \`\`\`

    This single command will perform all steps of the experiment:
    *   **Data Preparation**: Initializes data transformations and attempts to download/load the CIFAR-10 dataset into the `data/` directory.
    *   **Model Initialization**: Defines the `SimpleCNN` and prepares it for training on the available device (CUDA or CPU).
    *   **Training**: Trains the model for `NUM_EPOCHS` (default is 10, can be changed in the script). Training progress (loss per 100 mini-batches, accuracy per epoch) will be printed to the console.
    *   **Model Saving**: Saves the trained model's state dictionary to `models/simple_cnn_cifar10.pth`.
    *   **Evaluation**: Evaluates the model on the CIFAR-10 test set and prints the final test accuracy and average loss.
    *   **Example Predictions**: Prints a few example predictions from the test set, showing actual vs. predicted labels.
    *   **Interpretation Notes**: Provides a summary of how to interpret the training and evaluation metrics.

### Expected Output Sequence
When you run `python src/train.py`, you should expect to see:
1.  Device configuration (e.g., "Using device: cuda").
2.  Messages related to CIFAR-10 dataset download/loading.
3.  Training progress:
    *   Loss updates every 100 mini-batches for each epoch.
    *   Overall training accuracy at the end of each epoch.
4.  "Finished Training." message.
5.  Confirmation that the model has been saved (e.g., "Model saved to ../models/simple_cnn_cifar10.pth").
6.  Test set evaluation results (accuracy and average loss).
7.  A section with example predictions (e.g., "Image #1: Actual: cat | Predicted: cat (Correct)").
8.  Notes on interpreting the training process.

### Interpreting Training and Evaluation Metrics (from `train.py` output)

The `train.py` script outputs several key metrics:
- **Training Loss (per mini-batch and epoch):** A decreasing loss indicates the model is learning from the training data.
- **Training Accuracy (per epoch):** Shows how well the model is fitting the data it's being trained on.
- **Test Accuracy (at the end):** This is a crucial metric, indicating how well the model generalizes to new, unseen data. A significant gap between high training accuracy and lower test accuracy suggests potential overfitting.
- **Average Test Loss:** Similar to test accuracy, this shows the model's performance on unseen data.

Ideally, both training and test accuracies should increase and then stabilize at a high value, while the loss values (both training and test) should decrease and stabilize at a low value. These metrics help in understanding model performance and in making decisions about hyperparameter tuning (like `LEARNING_RATE`, `NUM_EPOCHS`, `BATCH_SIZE` defined in `train.py`) or model architecture changes (defined in the `SimpleCNN` class within `train.py`).

### Notes
- **Execution Time**: Training a CNN can take some time, especially if running on a CPU or for many epochs. For quick testing, you can reduce `NUM_EPOCHS` in `src/train.py`.
- **Adjustable Parameters**: Key parameters like learning rate, number of epochs, batch size, and the model architecture itself are all defined within `src/train.py` and can be modified for experimentation.
- **Matplotlib for Image Display**: The script includes commented-out code to visually display example images with their predictions using Matplotlib. If you wish to use this, ensure Matplotlib is installed and you are in an environment that supports GUI pop-ups. Uncomment the relevant section in the `show_example_predictions_after_training` function within `src/train.py`.
