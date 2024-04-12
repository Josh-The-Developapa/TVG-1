import torch
import torchvision
from torch import nn

# Input image size of 200x200 pixels

# TumourVision Generation-1 (TVG-1) has approx 22 million parameters


class TVG1(nn.Module):
    """Class for TumourVision G1 (TVG-1), a powerful Convolutional Neural Network (CNN) designed
    to classify brain MRI scans for brain tumor detection.
    The model is built using PyTorch and leverages Torch, TorchVision, and Matplotlib
    libraries to achieve accurate and insightful results.
    """

    def __init__(self):
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 40, 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(21_160, 1_000),
            nn.ReLU(),
            nn.Linear(1_000, 740),
            nn.ReLU(),
            nn.Linear(740, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # Output size - (196x196)
        x = self.pool(x)  # Output size - (98x98)
        x = self.relu(self.conv2(x))  # Output size - (94x94)
        x = self.pool(x)  # Output size - (47x47)
        x = self.relu(self.conv3(x))  # Output size - (46x46)
        x = self.pool(x)  # Output size - (23x23)
        x = self.flatten(x)  # Output size - 21,160 input neurons
        x = self.linear_layer_stack(x)
        return x