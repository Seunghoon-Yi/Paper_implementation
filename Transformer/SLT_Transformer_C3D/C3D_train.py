import os
import glob
import time

from tqdm import tqdm

import torch
from torch import nn, optim
import torch
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
from sklearn.utils import shuffle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from video_dataloader import get_category, UCF_dataset
from C3D_model import C3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datapath = './UCF_Data/UCF-101/'

############################### Hyperparameters ###############################
BATCH_SIZE   = 2
n_epoch      = 20
n_class      = 101
lr_init      = 5.e-6
lr_max       = 5.e-4
decay_param  = 0.999
clip         = 3
best_val_loss = 1.e10
###############################################################################

# video path, video label and category
total_path, total_label, Category = get_category(datapath)
print(len(total_path))

transforms_ = torch.nn.Sequential(
    transforms.RandomCrop((224, 288)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
)
# Shuffle the indices
total_path, total_label = shuffle(total_path, total_label, random_state=42)
# Index of train/test split
n_train       = len(total_path)//8*7
train_dataset = UCF_dataset(total_path[:n_train], total_label[:n_train],
                            Category, transform=transforms_)
test_dataset = UCF_dataset(total_path[n_train:], total_label[n_train:],
                            Category, transform=None)
# Get the dataloader!
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True)
val_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True)

# The Model!
Conv3D = C3D(n_class).to(device)

# Count parameters ad Initialize weights
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters = {count_params(Conv3D)}")

def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)
Conv3D.apply(initialize_weights)

# Define optimizer
optimizer = optim.Adam(Conv3D.parameters(), lr = lr_init)
criterion = nn.CrossEntropyLoss().cuda()

def epoch_time(start, end):
    elapsed = end-start
    elapsed_min = elapsed//60
    elapsed_sec = elapsed - elapsed_min*60

    return elapsed_min, elapsed_sec

def train(Currepoch, Model, iterator, optimizer, metric, clip):
    Model.train()
    epoch_loss  = 0
    n_iteration = len(iterator)
    n_warmup    = 3

    for idx, (VideoBatch, Labels) in enumerate(iterator):
        VideoBatch, Labels = VideoBatch.to(device), Labels.to(device)

        optimizer.zero_grad()
        predict_label = Model(VideoBatch)

        predict_label = predict_label.contiguous()
        labels        = Labels.contiguous()

        loss = metric(predict_label, labels)

        # Implement Warmup and Exponential decay #
        G = optimizer.param_groups[0]
        if Currepoch < n_warmup:
            G['lr'] += (lr_max-lr_init)/n_iteration/n_warmup
        else:
            G['lr'] *= decay_param

        loss.backward()
        torch.nn.utils.clip_grad_norm(Model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def eval(Currepoch, Model, iterator, metric):
    Model.eval()
    epoch_loss = 0
    for idx, (VideoBatch, Labels) in enumerate(iterator):
        VideoBatch, Labels = VideoBatch.to(device), Labels.to(device)

        optimizer.zero_grad()
        predict_label = Model(VideoBatch)

        predict_label = predict_label.contiguous()
        labels = Labels.contiguus()

        loss = metric(predict_label, labels)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

Trainloss, Valloss = [],[]
print(f"device = {device}")
for epoch in range(n_epoch):
    start = time.time()

    train_loss = train(epoch, Conv3D, train_loader, optimizer, criterion, clip)
    val_loss   = eval(epoch, Conv3D, val_loader, criterion)

    Trainloss.append(train_loss)
    Valloss.append(val_loss)

    end = time.time()
    epoch_m, epoch_s = epoch_time(start, end)

    if (val_loss<best_val_loss) and (epoch>10):
        best_val_loss = val_loss

        torch.save(Conv3D.state_dict(), f'B{BATCH_SIZE}_lr{lr_max}.pt')

    print('lr = ', optimizer.param_groups[0]['lr'])

    print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}')
    print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')

# Print and store losses #
print(Trainloss, Valloss)

