from json.tool import main
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob


def get_frames_labels(path, label, anchor, f_size, dilation=1):

    assert anchor-f_size*dilation >= 0 and anchor <= len(glob.glob(path+'\\*.png'))-1, 'image index exceeds!'
    frames = []
    for idx in range(-f_size + 1, 1):
        img = Image.open(path + '\\{}.png'.format(anchor+idx*dilation))
        img.load()
        img = np.asarray(img.convert('RGB'), dtype='int32')
        frames.append(img)
    return (np.array(frames), label) # frames: array(11, 128, 128, 3)

def get_fls(pathNlabel, f_size, s_anchor=100, t_anchor=150):
    fls = []
    for path, label in pathNlabel:
        for anchor in range(s_anchor, t_anchor): # the number of anchors
            fl = get_frames_labels(path, label, anchor, f_size)
            fls.append((fl, anchor))

    return fls # [tuple(frames, labels)]

class toy_sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.C3d_1 = nn.Conv3d(3, 5, kernel_size=(10, 5, 5), stride=(10, 5, 5), padding=0)
        self.relu_1 = nn.ReLU()
        self.C3d_2 = nn.Conv3d(5, 2, kernel_size=(10, 5, 5), stride=(10, 5, 5), padding=0)
        self.relu_2 = nn.ReLU()
        self.C3d_3 = nn.Conv3d(2, 1, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        self.relu_3 = nn.ReLU()
        self.fc_1 = nn.Linear(in_features=25, out_features=2)

    def forward(self, input):
        out_1 = self.C3d_1(input)
        out_1_relu = self.relu_1(out_1)
        out_2 = self.C3d_2(out_1_relu)
        out_2_relu = self.relu_2(out_2)
        out_3 = self.C3d_3(out_2_relu)
        out_3_relu = self.relu_3(out_3)
        out_flat = torch.flatten(out_3_relu, start_dim=3)
        out_ln = self.fc_1(out_flat)
        return out_ln # tensor(batch, 1, 1, 2)

class toy_dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.C3d_1 = nn.Conv3d(3, 30, kernel_size=(5, 5, 5), stride=(1, 5, 5), padding=0)
        self.relu_1 = nn.ReLU()
        self.C3d_2 = nn.Conv3d(30, 60, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=0)
        self.relu_2 = nn.ReLU()
        self.C3d_3 = nn.Conv3d(60, 90, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0)
        self.relu_3 = nn.ReLU()
        self.fc_1 = nn.Linear(in_features=4500, out_features=1000)
        self.fc_2 = nn.Linear(in_features=1000, out_features=2)

    def forward(self, input):
        out_1 = self.C3d_1(input)
        out_1_relu = self.relu_1(out_1)
        out_2 = self.C3d_2(out_1_relu)
        out_2_relu = self.relu_2(out_2)
        out_3 = self.C3d_3(out_2_relu)
        out_3_relu = self.relu_3(out_3)
        out_flat = torch.flatten(out_3_relu, start_dim=1)
        out_ln1 = self.fc_1(out_flat)
        out_ln2 = self.fc_2(out_ln1)
        return out_ln2 # tensor(batch, 1, 1, 2)

if __name__ == '__main__':

    pass
