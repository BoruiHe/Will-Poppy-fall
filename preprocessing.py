import os
import glob
import yaml
import torch
import random
import pickle as pk
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch.utils.data import DataLoader

def preprocessing_vir(f_size=10, start_anchor=50, dilation=8): 
    data_path = os.path.join(os.getcwd(), 'virtual_poppy')
    if os.path.exists(os.path.join(data_path, 'videos.h5')):
        pass
    else:
        video_list = {'line': []}
        video_list['folder'] = os.listdir(os.path.join(os.getcwd(), 'virtual_poppy', 'standing')) + os.listdir(os.path.join(os.getcwd(), 'virtual_poppy', 'fall'))
        video_list['class'] = ['standing' for _ in range(int(len(video_list['folder'])/2))] + ['fall' for _ in range(int(len(video_list['folder'])/2))]
        for folder, clss in zip(list(video_list['folder']), list(video_list['class'])):
            path = os.path.join(data_path, clss, folder)
            sp_anchor = len(glob.glob(os.path.join(path, '*.png')))-1 # stop anchor is the highest index of images of this full video
            indexes = []
            shifted_indexes = []
            anchor = start_anchor
            while anchor <= sp_anchor:
                indexes.append(anchor)
                shifted_indexes.append(anchor - dilation)
                anchor += dilation

            with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
                hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
                hc_z_coordinates_shifted = np.array(hc_z_coordinates)[shifted_indexes]
                hc_z_coordinates = np.array(hc_z_coordinates)[indexes]
                
            # 1 -> fall, 0 -> standing
            label = list((np.absolute(hc_z_coordinates - hc_z_coordinates_shifted)>0.00169).astype(int))
            line = set_line(f_size, label)
            video_list['line'].append(line)

        video_list = pd.DataFrame(video_list)
        video_list.to_hdf(os.path.join(data_path, 'videos.h5'), key='videos')

def set_line(f_size, label_list):
    # this method returns a terminate frame index. For example: frames[start frame index: terminate frame index]
    i = f_size 
    while i <= len(label_list) - 1: # if index of target future frame does not exceed 
        if label_list[i] and len(label_list[(i):]) * 0.8 < sum(label_list[(i):]):
            return i
        i += 1
    return i - 1

if __name__ == '__main__':
    preprocessing_vir()
    # path = os.path.join(os.getcwd(), 'virtual_poppy', 'videos.h5')
    # video_list = pd.read_hdf(path)
    pass