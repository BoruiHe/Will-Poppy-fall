from json.tool import main
import torch
import torch.nn as nn
import math
import torchvision.models


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=0)

    elif isinstance(layer, nn.Conv3d):
        nn.init.kaiming_uniform_(layer.weight, a=0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(layer.bias, -bound, bound)

class model_1(nn.Module):
    def __init__(self, f_size, height, width, latent_size):
        super().__init__()
        self.h = height
        self.w = width
        self.f_size = f_size
        self.pool_1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU())
        
        self.conv1_branch_2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv1_branch_1 = nn.Sequential(nn.Conv3d(64, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv3_branch_2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 512, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.conv3_branch_1 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.conv4_branch_2 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=(2, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(256),
                                            nn.ReLU(),
                                            nn.Conv3d(256, 256, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(256),
                                            nn.ReLU(),
                                            nn.Conv3d(256, latent_size, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(latent_size))
        
        self.conv4_branch_1 = nn.Sequential(nn.Conv3d(512, latent_size, kernel_size=(2, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(latent_size))
        self.relu = nn.ReLU()
        self.DynSize = self.dynamic_size(torch.zeros(1, 3, self.f_size, self.h, self.w))
        self.pool_2 = nn.AvgPool3d(tuple(self.DynSize), stride=(1))
        self.convt3d_1 = nn.ConvTranspose3d(latent_size, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        if self.h == 128:
            self.convt3d_2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif self.h == 96:
            self.convt3d_2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,0,1))
        self.convt3d_3 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3d_4 = nn.ConvTranspose3d(256, 64, kernel_size=(4, 3, 3), stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.convt3d_5 = nn.ConvTranspose3d(64, 64, kernel_size=(4, 3, 3), stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.convt3d_6 = nn.ConvTranspose3d(64, 3, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.convt3d_7 = nn.ConvTranspose3d(3, 3, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1))

    def forward(self, input):
        # inputs shpae: (4, 3, 10, 128, 128) ->(bs, C, f_size, H, W)
        input = self.pool_1(input) # shpae: (4, 64, 10, 63, 63) ->(bs, C, f_size, H, W)
        out = self.conv1(input) # shpae: (4, 64, 9, 32, 32) ->(bs, C, f_size, H, W)
        out1_b_2 = self.conv1_branch_2(out) # shpae: (4, 128, 9, 32, 32) ->(bs, C, f_size, H, W)

        out1_b_2 += self.conv1_branch_1(out)
        out1_b_2 = self.relu(out1_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)

        out2_b_2 = self.conv2(out1_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)

        out2_b_2 += out1_b_2
        out2_b_2 = self.relu(out2_b_2) # shpae: (4, 256, 9, 32, 32) ->(bs, C, f_size, H, W)

        out3_b_2 = self.conv3_branch_2(out2_b_2) # shpae: (4, 512, 7, 16, 16) ->(bs, C, f_size, H, W)

        out3_b_2 += self.conv3_branch_1(out2_b_2)
        out3_b_2 = self.relu(out3_b_2) # shpae: (4, 512, 7, 16, 16) ->(bs, C, f_size, H, W)
        
        out4_b_2 = self.conv4_branch_2(out3_b_2) # shpae: (4, 1024, 6, 8, 8) ->(bs, C, f_size, H, W)

        out4_b_2 += self.conv4_branch_1(out3_b_2)
        out4_b_2 = self.relu(out4_b_2) # shpae: (4, 1024, 6, 8, 8) ->(bs, C, f_size, H, W)

        out = self.pool_2(out4_b_2) # shpae: (4, 1024, 1, 1, 1) ->(bs, C, f_size, H, W)
        
        reconst_vc = self.convt3d_1(out) # tensor(batch, 512, 2, 2, 2)
        reconst_vc = self.convt3d_2(reconst_vc) # tensor(batch, 256, 4, 4/3, 4)
        reconst_vc = self.convt3d_3(reconst_vc) # tensor(batch, 256, 8, 8/6, 8)
        reconst_vc = self.convt3d_4(reconst_vc) # tensor(batch, 64, 9, 16/12, 16)
        reconst_vc = self.convt3d_5(reconst_vc) # tensor(batch, 64, 10, 32/24, 32)
        reconst_vc = self.convt3d_6(reconst_vc) # tensor(batch, 3, 10, 64/48, 64)
        reconst_vc = self.convt3d_7(reconst_vc) # tensor(batch, 3, 10, 128/96, 128)

        return reconst_vc, out 
    
    def dynamic_size(self, input):
        input = self.pool_1(input)
        out = self.conv1(input)
        out1_b_2 = self.conv1_branch_2(out) 

        out1_b_2 += self.conv1_branch_1(out)
        out1_b_2 = self.relu(out1_b_2) # (1, 256, 9, 24, 32)

        out2_b_2 = self.conv2(out1_b_2)

        out2_b_2 += out1_b_2
        out2_b_2 = self.relu(out2_b_2) # (1, 256, 9, 24, 32)

        out3_b_2 = self.conv3_branch_2(out2_b_2)

        out3_b_2 += self.conv3_branch_1(out2_b_2)
        out3_b_2 = self.relu(out3_b_2) # (1, 512, 7, 12, 16)
        
        out4_b_2 = self.conv4_branch_2(out3_b_2)

        out4_b_2 += self.conv4_branch_1(out3_b_2)
        out4_b_2 = self.relu(out4_b_2) # (1, 1024, 6, 6, 8)

        return out4_b_2.shape[2:]
    
    def latent_repre_size(self):
        return self.DynSize
    

if __name__ == '__main__':
    model = model_1(*(128, 128), 1024)
    input = torch.randn(1, 3, 128, 128)
    output = model(input)

    empty_model = torchvision.models.resnet50()
    model = torchvision.models.resnet50(pretrained=True)
    pretrained_dict = model.state_dict()
    model_dict = empty_model.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    empty_model.load_state_dict(model_dict)

    del empty_model.fc
    empty_model.add_module('fc', nn.Linear(2048, 9))

    model = model_1(*(96,128), 10, 1024)
    # input_names = ['Iris']
    # output_names = ['Iris Species Prediction', 'latent representation']
    # torch.onnx.export(model_1(128,128,10,1024), torch.randn(4, 3, 10, 128, 128), 'model.onnx', input_names=input_names, output_names=output_names)
    input = torch.randn(1, 3, 10, 96, 128)
    output = model(input)
    loss_fn = nn.MSELoss('mean')
    loss = loss_fn(input, output)
    pass