import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image
import glob


def read_images(path, anchor, dilation):
    # In pybuyllet environment, camera takes 240 picture per second.
    # Read 1 image every [dilation] pictures. For example, if dilation=10, this function will read 1 image every 10 pictures.

    sp_anchor = len(glob.glob(path+'\\*.png'))-1 # stop anchor is the highest index of images of this full video

    frames, indexes = [], []
    while anchor <= sp_anchor:
        img = Image.open(path + '\\{}.png'.format(anchor))
        img.load()
        img = np.asarray(img.convert('RGB'), dtype='int32')
        frames.append(img)
        indexes.append(anchor)
        anchor += dilation

    return frames, indexes # frames, array of shape: (128, 128, 3)

def get_fls(paths, anchor, ps, dilation, f_size):
    paths = list(paths)
    
    v_fs, f_fs, v_ls, f_ls = [], [], [], []
    for path in paths:

        # read labels from log.txt
        file = open(os.path.join(path, 'log.txt'), 'r')

        labels = file.readlines()
        labels = [(int(label[1:4]), int(label[6])) for label in labels]
        frames, indexes = read_images(path, anchor, dilation)

        assert len(frames) == len(indexes), 'length of frames should equal length of lables'

        frames = np.array(frames)
        labels = np.array(labels)[indexes]

        s = 0
        while s + f_size + ps - 1 <= len(frames) - 1:

            # label = (labels[(s+f_size+ps-1)][0], labels[(s+f_size+ps-1)][1])
            if labels[(s+f_size+ps-1)][1]:
                label = (labels[(s+f_size+ps-1)][0], [1,0]) # label 1 -> [1,0] -> falling down
            else:
                label = (labels[(s+f_size+ps-1)][0], [0,1]) # lable 0 -> [0,1] -> vibration

            if 'vibration' in path:
                    v_fs.append(np.stack(frames[s:(s+f_size)]))
                    v_ls.append((label, 'V')) # lable 0 -> [0,1] -> vibration -> V
            elif 'falling down' in path:
                    f_fs.append(np.stack(frames[s:(s+f_size)]))
                    if label[1] == [1, 0]:
                        f_ls.append((label, 'F')) # label 1 -> [1,0] -> falling down -> F
                    else:
                        f_ls.append((label, 'Fv'))  # lable 1 -> [0,1] -> falling down -> Fv
            else:
                raise Exception('illegal folder name')
            
            s += 1


    return [ele for ele in zip(v_fs, v_ls)], [ele for ele in zip(f_fs, f_ls)]

class mydataset(Dataset):
    def __init__(self, path, numEclass, anchor, prediction_span, dilation, f_size, k_fold=None, save=False, key=None):
        super().__init__()
        if k_fold:
            assert k_fold > 0, "invalid k_fold"
        self.k = k_fold
        # TODO: parameters 'save' and 'key' are kept for reproducibility. Need to store CV information for reproducibility
        # if save:
        #     assert not key, 'save your dataset with a non-empty string'
        #     assert type(key) == type(''), 'save your dataset with a non-empty string'
        #     assert key != 'df', ''df' is occupied'
        self.f_size = f_size # frame size: the number of frames after an anchor (anchor is included)
        self.dilation = dilation # distance between two anchors
        self.nEc = numEclass
        self.path = path # root path, under which you should have two folders (vibration/falling down) used to store long vedioes
        self.anchor = anchor
        self.ps = prediction_span

        # Generate a DataFrame for the whole dataset if it has not been done yet
        if not 'DF for the whole dataset.h5' in os.listdir(self.path):
            
            self.f_path = os.path.join(path, 'training', 'falling down')
            self.f_folders = os.listdir(self.f_path)
            self.f_folders_path = [os.path.join(self.f_path, name) for name in self.f_folders if name != 'visualization']
            self.v_path = os.path.join(path, 'training', 'vibration')
            self.v_folders = os.listdir(self.v_path)
            self.v_folders_path = [os.path.join(self.v_path, name) for name in self.v_folders if name != 'visualization']
            self.folders_path = self.v_folders_path + self.f_folders_path
            pd.DataFrame({'paths': self.folders_path}).to_hdf(os.path.join(self.path, 'DF for the whole dataset.h5'), key='df')

        self.df = pd.read_hdf(os.path.join(self.path, 'DF for the whole dataset.h5'), key='df')

        # self.classes: Classes and their index range
        self.classes = {}
        self.paths = list(self.df['paths'])
        
        for path in self.paths:
            video_class = os.path.split(os.path.split(path)[0])[1]
            if not video_class in list(self.classes.keys()):
                start = end = self.paths.index(path)
            else:
                end += 1
            self.classes.update({video_class: (start, end)})
        
        # index_list: choose self.nEc folders/full videos from each class and collect the index of each chosen folder/full video in self.df
        assert self.nEc <= min([i[1]-i[0]+1 for i in self.classes.values()]), 'numEclass exceeds!'
        index_list =[]
        for i in self.classes.values():
            index_list += random.sample(range(i[0], i[1]+1), self.nEc)
        
        # self.chosen_paths: collect the corresponding path of each index
        self.chosen_paths = self.df.loc[index_list, :].reset_index(drop=True)

        # self.inputs: frames and labels. You may not want to use the whole dataset for your experiments.
        self.v_inputs, self.f_inputs = get_fls(self.chosen_paths['paths'], self.anchor, self.ps, dilation=self.dilation, f_size=self.f_size)
        self.inputs = self.v_inputs + self.f_inputs

    def cv_idx(self):
        
        # k-folds cross validation (k-fold CV)
        assert self.k, 'Require an input for k_fold'
        remainder = len(self) % self.k
        if remainder == 0:
            quotient = int(len(self) / self.k)
            idx, remained_idx = set(range(len(self))), set(range(int(len(self)/2)))

            folds = []
            for _ in range(self.k):
                spe_u = random.sample(remained_idx, int(quotient/2))
                spe_d = [int(len(self)/2) + spe for spe in spe_u]
                chosen_val = set(spe_u + spe_d)
                remained_idx = remained_idx.difference(chosen_val)
                folds.append((np.array(list(idx.difference(chosen_val))), np.array(list(chosen_val))))
            return folds
        
        else:
            # TODO: what if you get an unbalanced/undivisible dataset? 
            # Approach:
            # Undivisible dataset: the number of frames/short videos = self.nEc * the number of anchors. Set self.nEc and the number of anchors as parameters.
            #                      A default fix for this problem is decrease the number of anchors.
            # Unbalanced dataset: self.nEc is leq to the minimum number of full videos in each class. This result in a waste but ensure a balanced self.inputs dictionary. 
            print('Can not create balanced folds')
            
    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        data = torch.tensor(self.inputs[index][0]).float().permute(3,0,1,2) / 255 # tensor(C, f_size, H, W)
        label = torch.tensor(self.inputs[index][1][0][1]).float()
        # anchor = self.inputs[index][1][0]
        tp = self.inputs[index][1][1]

        return data, label, tp

if __name__ == '__main__':

    root_path = 'C:\\Users\\hbrch\\Desktop\\fdi_fl\\data'

    dataset = mydataset(root_path, numEclass=200, anchor=50, prediction_span=2, dilation=10, f_size=10, k_fold=7)
    folds = dataset.cv_idx()
    for fold, (train_idx, val_idx) in enumerate(folds):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=40, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=40, sampler=val_sampler)
        for num, (train_data, train_labels, t_tp) in enumerate(train_loader):
            print(num, train_data.shape, train_labels.shape)
        for num, (val_data, val_labels, v_tp) in enumerate(val_loader):
            print(num, val_data.shape, val_labels.shape)
    pass