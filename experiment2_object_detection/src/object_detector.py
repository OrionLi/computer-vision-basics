# Name: 李弢阳
# Student ID: 202211621213

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont # ImageDraw and Font will be used in next step
import os
import numpy as np
import requests # For downloading sample image

# --- Configuration ---
# Model: Using Faster R-CNN with ResNet50 FPN V2 backbone, pre-trained on COCO
MODEL_WEIGHTS = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
MODEL_INSTANCE = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=MODEL_WEIGHTS)
COCO_INSTANCE_CATEGORY_NAMES = MODEL_WEIGHTS.meta["categories"]

# Ensure '__background__' is at index 0 if not present
if COCO_INSTANCE_CATEGORY_NAMES[0].lower() != '__background__':
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__'] + COCO_INSTANCE_CATEGORY_NAMES

IMAGE_DIR = '../data'
OUTPUT_DIR = '../output'
DEFAULT_IMAGE_FILENAME = 'sample_image.jpg'
# A direct link to a sample image for automatic download
DEFAULT_IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'
CONFIDENCE_THRESHOLD = 0.5

# Ensure output and data directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_sample_image_if_needed(image_dir, default_filename, url):
    # Downloads a sample image if the default image doesn't exist in image_dir.
    default_image_path = os.path.join(image_dir, default_filename)
    if not os.path.exists(default_image_path):
        print(f"Default image '{default_filename}' not found in '{image_dir}'.")
        print(f"Attempting to download from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=10) # Added timeout
            response.raise_for_status()
            with open(default_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Sample image downloaded successfully as '{default_filename}' in '{image_dir}'.")
            return default_image_path
        except requests.exceptions.RequestException as e: # More specific exception
            print(f"Error downloading sample image: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")

        # If download fails, create a placeholder
        print("Please place an image manually in the '../data/' directory and update DEFAULT_IMAGE_FILENAME if needed, or check internet connection.")
        try:
            placeholder_img = Image.new('RGB', (640, 480), color = 'lightgrey') # Changed color
            draw = ImageDraw.Draw(placeholder_img)
            try:
                # Try to load a common font, fall back to default
                font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()

            message = "Sample image download failed or no image found.\nReplace this with a real image (e.g., sample_image.jpg) in the data folder."
            # Simple text wrapping
            lines = message.split('\n')
            y_text = 10
            for line in lines:
                draw.text((10, y_text), line, fill="black", font=font)
                y_text += font.getbbox(line)[3] + 5 # font.getsize(line)[1] for older Pillow

            placeholder_img.save(default_image_path)
            print(f"A grey placeholder image named '{default_filename}' has been created in '{image_dir}'.")
            print("Replace it with a real image for detection or ensure internet access for download.")
        except Exception as pe:
            print(f"Could not create a placeholder image: {pe}")
        return default_image_path # Return path even if it's just a placeholder

    else:
        print(f"Using existing image: {default_image_path}")
    return default_image_path

def load_model(device):
    # Loads the pre-trained object detection model.
    model = MODEL_INSTANCE
    model.eval()
    model.to(device)
    print(f"Model: Faster R-CNN ResNet50 FPN V2 loaded on {device} with {len(COCO_INSTANCE_CATEGORY_NAMES)} COCO classes.")
    return model

def preprocess_image(image_path):
    # Loads and preprocesses an image for the model.
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img)
        return img, img_tensor
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}. The script might have created a placeholder if download failed.")
        return None, None
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return None, None

def predict_objects(model, img_tensor, device):
    # Performs object detection on the image tensor.
    if img_tensor is None:
        return None
    with torch.no_grad():
        prediction = model([img_tensor.to(device)]) # Model expects a batch of images
    return prediction

def filter_predictions(prediction, threshold):
    # Filters predictions based on the confidence threshold.
    if not prediction or not prediction[0]['scores'].numel(): # Check if any scores exist
        return np.array([]), [], np.array([]) # Return empty structures

    pred_boxes = prediction[0]['boxes']
    pred_labels = prediction[0]['labels']
    pred_scores = prediction[0]['scores']

    # Filter by score
    keep_indices = pred_scores > threshold

    boxes = pred_boxes[keep_indices].cpu().numpy()
    labels_indices = pred_labels[keep_indices].cpu().numpy()
    scores_filtered = pred_scores[keep_indices].cpu().numpy()

    # Map label indices to class names
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels_indices]

    return boxes, labels, scores_filtered

def get_font(size):
    """
    Attempts to load a preferred TrueType font (DejaVuSans or Arial) at a given size.
    If preferred fonts are not found, it falls back to Pillow's default bitmap font.
    This ensures that text can always be rendered on the image.

    Args:
        size (int): The desired font size.

    Returns:
        ImageFont: A Pillow ImageFont object.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError: # If DejaVuSans.ttf is not found
        try:
            return ImageFont.truetype("arial.ttf", size) # Try Arial as a fallback
        except IOError: # If Arial.ttf is also not found
            return ImageFont.load_default() # Use Pillow's built-in default font

def draw_and_save_image(pil_img, boxes, labels, scores, output_path, threshold):
    """
    Draws bounding boxes, class labels, and confidence scores on a copy of the input image
    and saves it to the specified output path.

    Args:
        pil_img (PIL.Image.Image): The input image on which to draw.
                                   A copy is made internally if modifications are applied.
        boxes (np.array): Array of bounding boxes (x_min, y_min, x_max, y_max).
        labels (list): List of class labels corresponding to the boxes.
        scores (np.array): Array of confidence scores corresponding to the boxes.
        output_path (str): Path where the annotated image will be saved.
        threshold (float): The confidence threshold used for filtering (for context, not used for drawing).
    """
    # It's good practice to draw on a copy if the original pil_img might be used elsewhere,
    # though in the current main script flow, pil_img.copy() is already done at the call site.
    # If this function were to be reused where the input image shouldn't be modified,
    # uncommenting the line below would be important:
    # draw_image = pil_img.copy()
    # draw = ImageDraw.Draw(draw_image)
    draw = ImageDraw.Draw(pil_img) # Assuming pil_img is already a copy or can be modified directly.

    # Color mapping for different classes can be added here if desired.
    # For now, a default color is used for all boxes.
    # Example: class_colors = {"person": "blue", "car": "green"}
    class_colors = {}
    default_color = "red"  # Default color for bounding boxes
    text_color = "white"   # Color for the label text
    text_background = "black" # Background color for the text box for better visibility

    num_detections = len(boxes)
    if num_detections == 0:
        print("No objects to draw (either none detected or all below threshold).")
        # To save the original image even if no detections:
        # try:
        #     pil_img.save(output_path)
        #     print(f"Original image (no detections) saved to: {output_path}")
        # except Exception as e:
        #     print(f"Error saving original image: {e}")
        return

    print(f"Drawing {num_detections} boxes on the image...")

    for i in range(num_detections):
        box = boxes[i].astype(np.int32)
        label = labels[i]
        score = scores[i]
        color = class_colors.get(label, default_color)

        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)

        text = f"{label}: {score:.2f}"
        font_size = 15 # Define font size for labels
        font = get_font(font_size) # Get the font object

        # Calculate text size to prepare a background for it
        try: # Modern Pillow (version 9.2.0+) uses textbbox
            # The (0,0) coordinates are dummy for textbbox as it only calculates size based on text and font
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Older Pillow versions use textsize
            text_width, text_height = draw.textsize(text, font=font)

        # Calculate position for the text label.
        # Prefer to place text above the bounding box.
        # If text would go off-screen (above the image), place it below the box.
        text_y_position = box[1] - text_height - 7  # y_min of box - text_height - padding

        # If the calculated Y position is off the top of the image, move it below the box
        if text_y_position < 0:
            text_y_position = box[3] + 7 # y_max of box + padding

        # Define the background rectangle for the text
        text_bg_coords = [box[0], text_y_position, box[0] + text_width + 4, text_y_position + text_height + 4]
        draw.rectangle(text_bg_coords, fill=text_background)

        # Draw the text on top of the background rectangle
        draw.text((box[0] + 2, text_y_position + 2), text, fill=text_color, font=font) # +2 for minor padding

    try:
        # Save the image with drawn annotations
        pil_img.save(output_path)
        print(f"Output image with detections saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

if __name__ == "__main__":
    print("--- Experiment 2: Object Detection ---")
    print("Name: 李弢阳")
    print("Student ID: 202211621213")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("CUDA not found. Running on CPU. This might be slow for object detection.")

    model = load_model(device)

    image_path_to_process = download_sample_image_if_needed(IMAGE_DIR, DEFAULT_IMAGE_FILENAME, DEFAULT_IMAGE_URL)

    print(f"\nProcessing image: {image_path_to_process}")
    pil_img, img_tensor = preprocess_image(image_path_to_process)

    if pil_img and img_tensor is not None:
        predictions = predict_objects(model, img_tensor, device)

        if predictions:
            boxes, labels, scores = filter_predictions(predictions, CONFIDENCE_THRESHOLD)

            print(f"\nFound {len(boxes)} objects with confidence > {CONFIDENCE_THRESHOLD}:")
            if not labels: # Check if the list of labels is empty
                 print("No objects found above the confidence threshold.")
            else:
                for i in range(len(boxes)):
                    print(f"- Object: {labels[i]}, Score: {scores[i]:.2f}, Box: {boxes[i].astype(int)}") # Prettier box print

            if pil_img and labels: # Check if pil_img is not None and labels list is not empty
                base_image_name = os.path.basename(image_path_to_process)
                # Sanitize the base image name to create a valid output filename
                safe_base_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in base_image_name)
                output_filename = "detected_" + safe_base_name
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                draw_and_save_image(pil_img.copy(), boxes, labels, scores, output_path, CONFIDENCE_THRESHOLD) # Use .copy() to draw on a fresh image if pil_img is reused
            elif pil_img: # Case where pil_img exists but no labels (no detections above threshold)
                print("No objects detected above threshold to draw. Output image will not be saved by default.")
                # Optionally, save the original image if no detections:
                # base_image_name = os.path.basename(image_path_to_process)
                # safe_base_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in base_image_name)
                # output_filename = "no_detections_" + safe_base_name
                # output_path = os.path.join(OUTPUT_DIR, output_filename)
                # try:
                #     pil_img.save(output_path)
                #     print(f"Original image (as no detections were above threshold) saved to: {output_path}")
                # except Exception as e:
                #     print(f"Error saving original image: {e}")
            else:
                # This case should ideally not be reached if pil_img is None earlier checks would have caught it
                print("Source image is not available, cannot draw or save.")
        else:
            print("No predictions returned from the model for the given image.")
    else:
        print(f"Could not load or process image: {image_path_to_process}. Ensure it is a valid image file or check download status.")

    print("\nScript finished. Adjustable parts include: MODEL_INSTANCE (weights), CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_FILENAME/URL.")
    print(f"To process a different image, place it in '{IMAGE_DIR}' and change 'DEFAULT_IMAGE_FILENAME' in the script, or modify the script to take a command-line argument.")
