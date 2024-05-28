import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def train_DataLoader(batch_size, n_worker = 1, shuffle = True):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5011, 0.4825, 0.4460],
                             std = [0.2677, 0.2637, 0.2828])
    ])
    train_dataset = datasets.CIFAR100(root='./data/train', train=True,
                                      download=True, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=n_worker)

    return train_dataloader

def test_DataLoader(batch_size, n_worker = 1, shuffle = True):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5011, 0.4825, 0.4460],
                             std = [0.2677, 0.2637, 0.2828])
    ])
    test_dataset = datasets.CIFAR100(root='./data/train', train=False,
                                      download=True, transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=n_worker)

    return test_dataloader