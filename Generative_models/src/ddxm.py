#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests
import logging

from inspect import isfunction
from functools import partial
from datasets import load_dataset
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image
from pathlib import Path
import random
from logging.config import dictConfig

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchvision.utils import save_image
from logger import logger


#%%
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps = 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x/timesteps) + s) / (1 + s) * torch.pi*0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start