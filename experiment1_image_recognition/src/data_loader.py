# Name: 李弢阳
# Student ID: 202211621213

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
BATCH_SIZE = 64
NUM_WORKERS = 2 # Adjust based on your system
DATA_DIR = '../data' # Relative path to the data directory

# --- Transformations ---
# For training: include data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 mean and std
])

# For testing: only normalization
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# --- Datasets ---
try:
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
except Exception as e:
    print(f"Error downloading or loading CIFAR-10 dataset: {e}")
    print("Please check your internet connection and ensure the PyTorch environment is correctly set up.")
    print("You might need to install PyTorch with torchvision: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    exit()

# --- DataLoaders ---
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Classes ---
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- Function to show some images (for verification) ---
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    print("Data loader script executed.")
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of testing samples: {len(testset)}")
    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of testing batches: {len(testloader)}")
    print(f"Classes: {', '.join(classes)}")

    # Get some random training images
    if len(trainloader) > 0:
        try:
            dataiter = iter(trainloader)
            images, labels = next(dataiter)

            # Show images
            # imshow(torchvision.utils.make_grid(images[:4])) # Showing images requires a display environment
            print('Labels of the first 4 images in the first batch: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
            print("Image display is commented out as it may not work in all environments without a GUI.")
            print("If you run this locally with a display, uncomment 'imshow' line to see example images.")
        except Exception as e:
            print(f"Could not iterate through trainloader or display images: {e}")
            print("This might happen if the dataset download failed or was interrupted.")
    else:
        print("Trainloader is empty. Cannot display images.")

    print("\nThis script prepares the CIFAR-10 dataset for image recognition.")
    print("It includes data augmentation for the training set and normalization for both sets.")
    print("DataLoaders are created for efficient batch processing.")
