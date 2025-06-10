# Experiment 2: Object Detection

Student: 李弢阳
ID: 202211621213

This experiment focuses on object detection using pre-trained Convolutional Neural Networks (CNNs) with PyTorch. The goal is to take an image, detect objects within it, and draw bounding boxes around them with appropriate labels.

## Directory Structure
- `data/`: This directory can be used to store sample images for detection. You will need to add your own images here.
- `src/`: Contains the Python scripts for the experiment.
  - `object_detector.py`: (To be created) This script will load a pre-trained model, perform detection on sample image(s), and save the results.
- `output/`: (To be created by the script) This directory will store the images with detected objects and their bounding boxes.

## Setup

### PyTorch Installation
Ensure you have PyTorch and Torchvision installed. The specified installation command is:
\`\`\`bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
\`\`\`
(Note: `cu128` implies a CUDA 12.8 compatible build. Adjust if using CPU or a different CUDA version. The object detection script will attempt to use CUDA if available, otherwise CPU.)

### Additional Libraries
You might need OpenCV for image manipulation (reading images, drawing bounding boxes, saving results) and Matplotlib for displaying images (optional, mainly for interactive sessions).
\`\`\`bash
pip3 install opencv-python matplotlib
\`\`\`

## Experiment Overview
1.  **Model Selection**: We will use a pre-trained object detection model from `torchvision.models.detection`, such as Faster R-CNN with a ResNet50 backbone. These models are trained on large datasets like COCO and can detect a variety of common objects.
2.  **Data (Sample Images)**: You will provide your own sample images for the model to perform detections on. Place these images in the `experiment2_object_detection/data/` directory.
3.  **Detection Script (`object_detector.py`)**:
    *   Load the pre-trained model.
    *   Load a sample image from the `data/` directory.
    *   Preprocess the image to the format expected by the model.
    *   Perform inference to get object detections (bounding boxes, class labels, and confidence scores).
    *   Filter detections based on a confidence threshold.
    *   Draw the bounding boxes and labels on the image.
    *   Save the resulting image to the `output/` directory.
4.  **Results and Analysis**:
    *   Examine the output images with detections.
    *   Understand the model's predictions, including the class of detected objects and their locations.
    *   The script will also discuss adjustable parameters like the choice of model and the confidence threshold.

## Data Annotation (Brief Overview)
While we are using a pre-trained model that already "knows" object classes and how to find them, training an object detection model from scratch (or fine-tuning one) requires a dataset with specific annotations. For each image, these annotations typically include:
- **Bounding Boxes**: Coordinates (e.g., x_min, y_min, x_max, y_max) defining a rectangle around each object of interest.
- **Class Labels**: The category of each object within a bounding box (e.g., 'car', 'person', 'dog').

This experiment leverages models pre-trained on datasets like COCO, which has 80 common object categories.
