import torch
import torchvision
from torch import nn

# Input image size of 200x200

class TVG1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=70, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=1120, out_features=740)
        )
        
    def forward(x:torch.Tensor)->torch.Tensor:
        return x