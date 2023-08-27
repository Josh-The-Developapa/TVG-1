import torch
from torch import nn, optim
import torchvision
from helper_functions import LoadImageData, matplotlib_imshow, visualise_batch_images, blurt, test_loop, train_model
from model import TVG1

train_dataloader, test_dataloader = LoadImageData('../data', 16)

# Visualise the images in our Brain Tumour dataset
visualise_batch_images(69,test_dataloader,3,False)

# # Define our model
# model = TVG1()

# # Define our criterion/ loss function
# criterion = nn.CrossEntropyLoss()

# # Define our optimiser
# optimiser = optim.Adam(model.parameters(),lr=0.01)

# test_loop(test_dataloader,model,criterion)

