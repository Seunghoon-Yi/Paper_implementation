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

def test(model, dataloader):
    '''

    :param model:
    :param dataloader:
    :return:

    model.eval()
    - test dataset
    with torch.no_grad():
        blahblah...

    '''
    pass


if __name__ == '__main__' :
    pass