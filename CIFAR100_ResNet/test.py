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

#parser.add_argument('--ckpt_dir', help='Directory containing the checkpoint, ./model_name/')
parser.add_argument('--model', type=str, required=True,
                    help="The objective model that will be trained")
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
    with torch.no_grad():
        model.eval()

    for i in range(20):
        n_accurate = 0
        n_total = 0

        # Sum of the accurate samples / Total test samples*100

        for X, Y in dataloader:
            n_total += len(X)
            n_accurate += utils.metric(model(X.to(device)), Y.to(device))

        print(n_accurate / n_total * 100)


if __name__ == '__main__' :

    args = parser.parse_args()
    weight_dir = './results/'+args.model+'/'+'best.ckpt'
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model
    batch_size = args.BS

    test_dataloader = Dataloader.test_DataLoader(args.BS)
    model_          = utils.get_network(args)
    optimizer       = utils.get_optim(model_name, model_)
    loss_fcn        = nn.CrossEntropyLoss()

    utils.load_checkpoints(weight_dir, model_)

    test(model_, test_dataloader)