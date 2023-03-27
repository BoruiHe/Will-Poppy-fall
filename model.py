from json.tool import main
import torch
import torch.nn as nn
import math


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=0)

    elif isinstance(layer, nn.Conv3d):
        nn.init.kaiming_uniform_(layer.weight, a=0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(layer.bias, -bound, bound)

# class toy_dense_Cla(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.C3d_1 = nn.Conv3d(3, 30, kernel_size=(5, 5, 5), stride=(1, 5, 5), padding=0)
#         self.C3d_2 = nn.Conv3d(30, 60, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=0)
#         self.C3d_3 = nn.Conv3d(60, 90, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0)

#         self.relu = nn.ReLU()
#         self.dr = nn.Dropout(p=0.2)

#         self.fc_1 = nn.Linear(in_features=1620, out_features=1000)
#         self.fc_2 = nn.Linear(in_features=1000, out_features=400)
#         self.fc_3 = nn.Linear(in_features=400, out_features=2)

#         self.smx = nn.Softmax(dim=1)

#     def forward(self, input):
#         # inputs shpae: (4, 3, 10, 128, 128) ->(bs, C, f_size, H, W)

#         out_1 = self.C3d_1(input) # shpae: (4, 30, 6, 25, 25) ->(bs, C, f_size, H, W)
#         out_1_relu = self.relu(out_1)

#         out_2 = self.C3d_2(out_1_relu) # shpae: (4, 60, 4, 5, 5) ->(bs, C, f_size, H, W)
#         out_2_relu = self.relu(out_2) 

#         out_3 = self.C3d_3(out_2_relu) # shpae: (4, 90, 2, 3, 3) ->(bs, C, f_size, H, W)
#         out_3_relu = self.relu(out_3)

#         out_flat = torch.flatten(out_3_relu, start_dim=1)
#         out_dr = self.dr(out_flat)

#         out_ln1 = self.fc_1(out_dr)

#         out_ln2 = self.fc_2(out_ln1)

#         out_ln3 = self.fc_3(out_ln2)

#         return self.smx(out_ln3) # tensor(batch, 2)

class model_deep(nn.Module):
    def __init__(self, residual= False, dense= False):
        super().__init__()
        self.resi = residual
        self.dense = dense
        assert not self.resi or not self.dense, 'choose residual or dense, cannot choose both'
        self.pool_1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU())
        
        self.conv1_branch_2 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(64),
                                            nn.ReLU(),
                                            nn.Conv3d(64, 64, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(64),
                                            nn.ReLU(),
                                            nn.Conv3d(64, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128))
        
        self.conv1_branch_1 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128))
        
        self.conv2_branch_2 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(64),
                                            nn.ReLU(),
                                            nn.Conv3d(64, 64, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(64),
                                            nn.ReLU(),
                                            nn.Conv3d(64, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv3_branch_2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv3_branch_1 = nn.Sequential(nn.Conv3d(64, 256, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv4_branch_2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=(2, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 512, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.conv4_branch_1 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(2, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.relu = nn.ReLU()         
        self.pool_2 = nn.AvgPool3d((6, 8, 8), stride=(1))
        self.fc_1 = nn.Linear(in_features=512, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=1)
       

    def forward(self, input):
        # inputs shpae: (4, 3, 10, 128, 128) ->(bs, C, f_size, H, W)
        input = self.pool_1(input) # shpae: (4, 64, 10, 63, 63) ->(bs, C, f_size, H, W)
        out = self.conv1(input) # shpae: (4, 64, 9, 32, 32) ->(bs, C, f_size, H, W)
        out1_b_2 = self.conv1_branch_2(out) # shpae: (4, 128, 9, 32, 32) ->(bs, C, f_size, H, W)
        if self.resi:
            out1_b_2 += self.conv1_branch_1(out)
            out1_b_2 = self.relu(out1_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)

        out2_b_2 = self.conv2_branch_2(out1_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)
        if self.resi:
            out2_b_2 += out1_b_2
            out2_b_2 = self.relu(out2_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)

        out3_b_2 = self.conv3_branch_2(out2_b_2) # shpae: (4, 512, 7, 16, 16) ->(bs, C, f_size, H, W)
        if self.resi:
            out3_b_2 += self.conv3_branch_1(out2_b_2)
            out3_b_2 = self.relu(out3_b_2) # shpae: (4, 512, 7, 16, 16) ->(bs, C, f_size, H, W)
        
        out4_b_2 = self.conv4_branch_2(out3_b_2) # shpae: (4, 1024, 6, 8, 8) ->(bs, C, f_size, H, W)
        if self.resi:
            out4_b_2 += self.conv4_branch_1(out3_b_2)
            out4_b_2 = self.relu(out4_b_2) # shpae: (4, 1024, 6, 8, 8) ->(bs, C, f_size, H, W)

        out = self.pool_2(out4_b_2) # shpae: (4, 1024, 1, 1, 1) ->(bs, C, f_size, H, W)
        out = torch.flatten(out, start_dim=1) # shpae: (4, 1024) ->(bs, C)
        out = self.fc_1(out)
        
        return out # tensor(batch, 1)
    
class model_shallow(nn.Module):
    def __init__(self, residual= False, dense= False):
        super().__init__()
        self.resi = residual
        self.dense = dense
        assert not self.resi or not self.dense, 'choose residual or dense, cannot choose both'
        self.conv1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(5, 5, 5), stride=(1, 5, 5), padding=0),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU())
        
        self.convls1 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(64),
                                      nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(64))
        
        self.conv2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=0),
                                   nn.BatchNorm3d(128),
                                   nn.ReLU())
        
        self.convls2 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(128),
                                      nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(128))
        
        self.conv3 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                   nn.BatchNorm3d(256),
                                   nn.ReLU())
        
        self.convls3 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(256),
                                      nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(256))
        
        self.relu = nn.ReLU()
        # self.pool = nn.AvgPool3d()
        self.fc_1 = nn.Linear(in_features=4608, out_features=64)
        self.fc_2 = nn.Linear(in_features=64, out_features=16)
        self.fc_3 = nn.Linear(in_features=16, out_features=1)
       

    def forward(self, input):
        # inputs shpae: (4, 3, 10, 128, 128) ->(bs, C, f_size, H, W)
        out_1 = self.conv1(input)
        out_cs1 = self.convls1(out_1)
        if self.resi:
            out_cs1 += out_1
            out_cs1 = self.relu(out_cs1)
        
        out_2 = self.conv2(out_1)
        out_cs2 = self.convls2(out_2)
        if self.resi:
            out_cs2 += out_2
            out_cs2 = self.relu(out_cs2) 

        out_3 = self.conv3(out_2)
        out_cs3 = self.convls3(out_3)
        if self.resi:
            out_cs3 += out_3
            out_cs3 = self.relu(out_cs3)

        out = torch.flatten(out_cs3, start_dim=1)

        out_ln1 = self.fc_1(out)
        out_ln2 = self.fc_2(out_ln1)
        out_ln3 = self.fc_3(out_ln2)

        return out_ln3 # tensor(batch, 1)

if __name__ == '__main__':
    # m = nn.MaxPool2d(3, stride=2, padding=1)
    # input = torch.randn(1, 16, 33, 32)
    # output = m(input)
    model = model_shallow(True)
    input = torch.randn(1, 3, 10, 128, 128)
    output = model(input)
    pass