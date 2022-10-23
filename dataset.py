# importing the required libraries
import os 
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset 
from glob import glob
from PIL import Image
from torchvision import transforms


# creating the dataset class
class DogsVsCatsDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = 1 if 'dog' in image_path else 0
        return image, label

# transform 
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])