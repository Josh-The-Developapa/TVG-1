from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from dataset import BrainTumourDataset
import os
import matplotlib.pyplot as plt


def blurt(accuracy, loss):
    """Function to have our system speak out the accuracy and average loss of the model"""
    os.system(
        f'say "Testing complete. Your Model has a {accuracy:.2f}% Accuracy, and an Average Loss of {loss:.2f}"'
    )


def LoadImageData(root: str, batch_size: int):
    """Function to process and load our data \n\n
    Returns the test and train dataloaders\n\n
    Each 'class' must have a subfolder inside the root, "data" folder. So data/glioma, data/notumour, data/meningioma & data/pituita
    """

    mean = 0.5
    std = 0.5

    # The transforms for our dataset
    transform = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # ImageFolder dataset containing paths and labels of images.
    dataset = datasets.ImageFolder(root)

    # Split our data into train and test data and labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset.imgs, dataset.targets, test_size=0.2, random_state=42
    )

    train_dataset = BrainTumourDataset(train_data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = BrainTumourDataset(test_data, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def matplotlib_imshow(batch: list, num_images: int, normalised: bool):
    """Function for producing an inline display
    of a set number images in a given batch"""
    classes = ("glioma", "meningioma", "notumour", "pituitary")

    # Fetch only the first (num_images)th
    batch = [(batch[0][0:num_images]), batch[1][0:num_images]]

    fig, axes = plt.subplots(1, len(batch[0]), figsize=(10, 5))
    for idx, img in enumerate(batch[0]):
        imgu = img if normalised else img * 0.5 + 0.5  # unnormalise -> img*0.5+0.5
        ax = axes[idx]
        ax.set_title(classes[int(batch[1][idx])])
        ax.imshow(imgu.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()


def visualise_batch_images(
    batch_num: int, dataloader: DataLoader, num_images: int, normalised: bool
):
    """Function that calls `matplotlib_imshow()` iteratively to produce an inline display for the batch specified\n\n
    Eliminates the need of calling matplotlib_imshow in a for loop
    """
    i = 0
    for idx, batch in enumerate(dataloader):
        if idx == (batch_num - 1):
            matplotlib_imshow(batch, num_images, normalised)
            break
