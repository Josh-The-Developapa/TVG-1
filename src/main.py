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
import os

train_dataloader, test_dataloader = LoadImageData("../data", 5)

# Visualise the images in our Brain Tumour dataset
visualise_batch_images(5, test_dataloader, 3)

# Define / Load our model
model = load_model(model_class=TVG1, file="../models/TVG1.pt")


# Define our criterion/ loss function
criterion = nn.CrossEntropyLoss()

# Define our optimiser
# optimiser = optim.SGD(params=model.parameters(), lr=0.001)


# Train our model
# train_model(model, criterion, optimiser, dataloader=train_dataloader, epochs=20)

# Test our model
test_model(model, criterion, test_dataloader)
