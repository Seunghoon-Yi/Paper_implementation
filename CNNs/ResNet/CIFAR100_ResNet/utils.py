import os
import sys
import shutil

import numpy as np
import torch
import torch.optim as optim

def get_network(args):
    '''
    :param args: args from argparse
    :return:     the pre-defined network
    '''
    if args.model == 'ResNet18':
        from model import ResNet18
        net = ResNet18()
    elif args.model == 'ResNet34':
        from model import ResNet34
        net = ResNet34()
    elif args.model == 'ResNet50':
        from model import ResNet50
        net = ResNet50()
    elif args.model == 'ResNet101':
        from model import ResNet101
        net = ResNet101()
    elif args.model == 'ResNet152':
        from model import ResNet152
        net = ResNet152()
    else:
        sys.exit("\n Error : The network name is invalid or not supported yet.")
        # with exit code 1

    if args.gpu:
        net = net.cuda()

    return net

def get_optim(model_name, model_, lr_init = 0.001):
    '''
    :param model_name: The given model name, string.
    :param model_:     net
    :param lr_init:    Initial lr
    :return:           Optimizer class.
    '''
    num = [int(i) for i in model_name if i.isdigit()]  # To store integers in model name!
    layerdepth = ''
    for k in num:
        layerdepth += str(k)
    layerdepth = int(layerdepth)

    print(layerdepth)
    if layerdepth >= 50:
        optimizer = optim.Adam(model_.parameters(), lr=lr_init)
    elif layerdepth <50:
        optimizer = optim.SGD(model_.parameters(), lr=lr_init, momentum=0.9)
    else:
        sys.exit("\n Error : The network name is invalid or not supported yet.")

    return optimizer


def metric(pred, GT):
    """
    :param pred: Prediction prob.distribution from model
    :param GT:   Ground Truth label
    :return:     metric : numbers of accurate samples
    """
    onehot_pred = pred.argmax(1, keepdim=True)
    corrects    = onehot_pred.eq(GT.view_as(onehot_pred)).sum().item()

    return corrects

def save_checkpoints(state, is_best, checkpoint):
    '''
    :param state:      (dict) state_dict of the model
    :param is_best:    (bool) True if it is the best model
    :param checkpoint: (string) directory where parameters are to be saved
    :return:           None
    '''
    filepath = os.path.join(checkpoint, 'last.ckpt')
    if not os.path.exists(checkpoint):
        print(f"Directory not exist. Making directory {checkpoint}")
        os.mkdir(checkpoint)
    else:
        pass
    torch.save(state, filepath)
    if is_best:
        shutil.copy(filepath, os.path.join(checkpoint, 'best.ckpt'))

def load_checkpoints(checkpoint, model, optimizer = None):
    '''
    :param checkpoint:  (str) path of the saved model
    :param model:       (model) given test model
    :param optimizer:   (optimizer from our checkpoint)
    :return:            weight-loaded model
    '''
    if not os.path.exists(checkpoint):
        sys.exit("\n Error : Not a valid path or checkpoint does not exists")

    CKPT = torch.load(checkpoint)
    model.load_state_dict(CKPT['state_dict'])
    if optimizer :
        optimizer.load_state_dict(CKPT['optim_dict'])

    return CKPT