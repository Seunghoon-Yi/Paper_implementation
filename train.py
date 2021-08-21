import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import utils
import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/train',
                    help='Directory containing the dataset')
# -- : positional arg // required : make it necessary
parser.add_argument('--model', type=str, required=True,
                    help="The objective model that will be trained")
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
parser.add_argument('--epoch', type=int, default=50, help="Epochs to train")
parser.add_argument('--BS', type=int, default=64, help="Batch size of the dataset")
parser.add_argument('--gpu', action='store_true', default='False',
                    help="True when gpu is available")

def train(model_, opt, loss_fcn, dataloader):
    model_.train()

    model_dir = './results/'+model_name
    best_acc  = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc  = 0.0
        len_data   = len(dataloader.dataset)
        for i, (train_batch, GT_batch) in enumerate(dataloader):
            if args.gpu:
                train_batch, GT_batch = train_batch.cuda(), GT_batch.cuda()

            # Prediction and get loss
            pred_batch = model_(train_batch)
            loss_batch = loss_fcn(pred_batch, GT_batch)
            # Train with backprop
            opt.zero_grad()
            loss_batch.backward()
            # Update parameters
            opt.step()

            epoch_loss += loss_batch.item()
            epoch_acc  += utils.metric(pred_batch, GT_batch)

            print(f'epoch {epoch+1} | batch {i+1} | loss {loss_batch.item()}')

        #epoch_loss /= len_data
        epoch_acc  /= len_data
        print(f'epoch {epoch+1} | loss {epoch_loss} | accuracy {epoch_acc*100}')

        is_best = (epoch_acc >= best_acc)
        if is_best:
            best_acc = epoch_acc
            utils.save_checkpoints({
                "epoch" : i+1,
                "state_dict" : model_.state_dict(),
                "optim_dict" : opt.state_dict()
            }, is_best = is_best, checkpoint=model_dir)





if __name__ == '__main__':

    # Load the parameters from parser
    args = parser.parse_args()

    model_name = args.model
    lr = args.lr
    epochs = args.epoch
    batch_size = args.BS

    train_dataloader = Dataloader.train_DataLoader(args.BS)
    model_    = utils.get_network(args)
    optimizer = utils.get_optim(model_name, model_, lr_init=lr)
    loss_fcn  = nn.CrossEntropyLoss()

    train(model_, optimizer, loss_fcn, train_dataloader)
    