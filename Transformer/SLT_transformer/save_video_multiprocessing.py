import os
import numpy as np
import pandas as pd
import multiprocessing

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils import shuffle

import cv2
import glob


path    = 'C:/Users/PC/2021-MLVU/SLT_project/SLT_videoset/'

def save_video(folder):
    storage = 'D:/21-ML data/frames_224_192/'


    name = folder
    imgpath = path+folder

    if not os.path.exists(storage+name+'.pt'):

        images = torch.tensor([cv2.imread(file) for file in glob.glob(imgpath + '/*.png')])
        # print(images.shape)

        frames = torch.zeros((240, 224, 192, 3))
        T, H, W, C = images.shape
        print(name, images.shape)
        if images.shape[0] is not None:
            try:
                frames[:T, :, :, :] = images[:, 18:-18, 9:-9, :]
            except:
                frames[:, :, :, :] = images[:240, 18:-18, 9:-9, :]

        frames = frames.permute(0,3,1,2)
        frames = frames.type(torch.uint8)
        print(frames.shape)

        #torch.save(frames, storage+name+'.pt')


folder_list = os.listdir(path)
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(save_video, folder_list)
    pool.close()
    pool.join()