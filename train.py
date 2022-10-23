import os
import torch.nn as nn 
import torch, torchvision 
from torch.utils.data import Dataset, DataLoader, Subset
from dataset import DogsVsCatsDataset, tfms 
from sklearn.model_selection import train_test_split
from glob import glob
from pathlib import Path
from model import Model
from engine import train_model



if __name__=="__main__":
    # defining the hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # path to the dataset
    path = Path('dataset/train')

    # getting the list of images
    images = glob(os.path.join(path, '*.jpg'))

    # splitting the dataset into train and validation
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # creating the dataset objects
    train_dataset = DogsVsCatsDataset(train_images, transform=tfms)
    val_dataset = DogsVsCatsDataset(val_images, transform=tfms)

    # creating subset of the dataset with 50% of the data
    train_subset = Subset(train_dataset, indices=range(0, len(train_dataset), 2))
    valid_subset = Subset(val_dataset, indices=range(0, len(val_dataset), 2))

    # creating the dataloaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=4)

    # printing the length of the dataset
    print('Length of the train dataset: ', len(train_dataset))
    print('Length of the validation dataset: ', len(val_dataset))
    print('Length of the train subset: ', len(train_subset))
    print('Length of the validation subset: ', len(valid_subset))


    # model
    model = Model()
    model.to(device)
    
    print('testing the dataloader')
    for xb, yb in train_loader:
        print(xb.shape)
        print(yb.shape)
        break

    ## loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ## training the model
    #train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)