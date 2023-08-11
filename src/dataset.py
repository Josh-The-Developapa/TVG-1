import torch
from PIL import Image
from torch.utils.data import Dataset

class BrainTumourDataset(Dataset):
    """This is a custom dataset for our 200x200 MRI scans of human brains for Multi-class classification
    We'll feed this into a dataloader \n\n
    0-glioma | 1-meningioma | 2-notumour | 3-pituitary
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Converts a path in each image-label tuple to an image. i.e ('data/glioma/0.png') -> PIL Image
        imagePIL = Image.open(self.dataset[index][0]).convert('RGB')

        # Applies our transform functionality to the image. (Resize, ToTensor, Normalise)
        image = self.transform(imagePIL)

        # tensor of the label of an image according to the index. i.e ('data/glioma/0.jpg',0) -> torch.tensor(0)
        label = torch.tensor(self.dataset[index][1]).type(torch.float32)

        return image, label