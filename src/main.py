import torch
from torch import nn, optim
import torchvision
from helper_functions import (
    LoadImageData,
    matplotlib_imshow,
    visualise_batch_images,
    blurt,
    train_model,
    test_model,
    save_model,
    load_model,
)
from model import TVG1
import torch.nn.functional as F

train_dataloader, test_dataloader = LoadImageData("../data", 20)

# Visualise the images in our Brain Tumour dataset
visualise_batch_images(69, test_dataloader, 3)

# Define our model
model = load_model(TVG1, "../models/TVG1.pt")  # Current model

# Define our criterion/ loss function
criterion = nn.CrossEntropyLoss()

# Define our optimiser
optimiser = optim.SGD(model.parameters(), lr=0.1)


# Train our model
train_model(model, criterion, optimiser, dataloader=train_dataloader, epochs=1)

# Test our model
test_model(model, criterion, test_dataloader)


# Performance report
# Time per epoch - 07m 39s
# Epochs (2) - 30.04% Accuracy | 79/263 correct predictions | 1.280 avg loss
