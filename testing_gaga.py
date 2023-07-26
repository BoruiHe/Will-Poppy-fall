import torch
import torch.nn as nn
import os
import math
import yaml
from model import model_1
from mydataset_gaga import m1_mydataset
from torch.utils.data import DataLoader
from utils.set_seed import set_seed
import shutil


def testing_m1(checkpoint_name):

    # Hyperparameters
    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for seed in HpParams['random seeds']:
        print(f'testing autoencoder --- Seed: {str(seed)}')
        path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed))
        set_seed(seed)
        if HpParams['dataset_name'] == 'vir_poppy':
            dataset_testing = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                    'testing', 
                                    path,
                                    scale= HpParams['scale'],
                                    anchor= HpParams['anchor'])
            
        elif HpParams['dataset_name'] == 'real_poppy':
            dataset_testing = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                    'testing', 
                                    path,
                                    )
        
        # model      
        toy_model = model_1(*dataset_testing.tell_me_HW(), HpParams['latent_size'])            
        loss_fn = nn.MSELoss(reduction='mean')

        # testing
        path_m = os.path.join(path, 'm1_best.pth')
        # load model states
        toy_model.load_state_dict(torch.load(path_m)['model_state_dict'], HpParams['latent_size'])
        toy_model.eval()
        toy_model.to(device)

        test_loader = DataLoader(dataset_testing, batch_size= HpParams['batch_size'])
        test_ave_loss = 0
        rMSE = 0

        for video_clip, _ in test_loader:
            video_clip = video_clip.to(device)
            
            test_outputs = toy_model(video_clip)[0].squeeze()
            test_loss = loss_fn(test_outputs, video_clip)

            test_ave_loss += test_loss.item()
            rMSE += math.sqrt(test_loss.item())

        test_ave_loss /= len(test_loader)
        rMSE /= len(test_loader)

        torch.save({
                'test loss': test_ave_loss,
                'test rMSE': rMSE
                }, os.path.join(path, 'm1_test.pth'))   


if __name__ == '__main__':
    testing_m1('debugging')


