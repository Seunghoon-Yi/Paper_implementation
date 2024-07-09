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
from blocks import *