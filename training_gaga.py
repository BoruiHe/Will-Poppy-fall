import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import random
import shutil
from utils.set_seed import set_seed
from mydataset_gaga import m1_mydataset
from torch.utils.data import DataLoader
from model import model_1, init_weights
from hp import hyperparameters_virtual, hyperparameters_real    


def training_m1(dataset, debugging=True):

    # Hyperparameters
    if dataset == 'vir':
        HpParams = hyperparameters_virtual
    elif dataset == 'real':
        HpParams = hyperparameters_real
    
    if debugging:
        checkpoint_name = 'debugging'
    else:
        checkpoint_name = 'gaga_{}_ps{}_{}'.format(dataset, HpParams['prediction_span'], HpParams['latent_size'])

    if os.path.exists(os.path.join('checkpoints', checkpoint_name)):
        shutil.rmtree(os.path.join('checkpoints', checkpoint_name))

    os.makedirs(os.path.join('checkpoints', checkpoint_name))

    HpParams['random seeds'] = random.sample(range(3224), k=3)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'w') as outfile:
        yaml.dump(HpParams, outfile, default_flow_style=False)
    
    for seed in HpParams['random seeds']:
        set_seed(seed)
        print('training autoencoder --- Seed: {}'.format(seed))
        path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed))
        os.makedirs(path)
        # dataset
        if HpParams['dataset_name'] == 'vir_poppy':
            dataset_training = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                'training', 
                                path,
                                scale= HpParams['scale'],
                                anchor= HpParams['anchor'])
            dataset_validation = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                'validation', 
                                path,
                                scale= HpParams['scale'],
                                anchor= HpParams['anchor'])
        elif HpParams['dataset_name'] == 'real_poppy':
            dataset_training = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                'training', 
                                path,
                                )
            dataset_validation = m1_mydataset(HpParams['dilation'], HpParams['f_size'], HpParams['prediction_span'], HpParams['dataset_name'], 
                                'validation', 
                                path,
                                )
        
        loss_fn = nn.MSELoss(reduction='mean')
        train_loss_log, val_loss_log = [], []
        train_loader = DataLoader(dataset_training, HpParams['batch_size'])
        val_loader = DataLoader(dataset_validation, HpParams['batch_size'])

        # model
        toy_model_m1 = model_1(*dataset_training.tell_me_HW(), HpParams['latent_size'])
        toy_model_m1.apply(init_weights)
        toy_model_m1.to(device)
        optimizer = optim.Adam(toy_model_m1.parameters(), lr=HpParams['lr_m1'], weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 100, 150], gamma=0.8)

        best_dic= {'val loss': float('inf')}

        for epoch in range(HpParams['epochs']):
            # training
            train_ave_loss = 0
            toy_model_m1.train()
            for video_clip, _ in train_loader:
                
                video_clip = video_clip.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                train_outputs = toy_model_m1(video_clip)[0]
                train_loss = loss_fn(train_outputs, video_clip)
                train_loss.backward()
                optimizer.step()
                
                train_ave_loss += train_loss.item()

            train_ave_loss /= len(train_loader)
            train_loss_log.append(train_ave_loss)
            scheduler.step()

            # validation
            val_ave_loss = 0
            toy_model_m1.eval()
            for video_clip, _ in val_loader:

                video_clip= video_clip.to(device)
                
                val_outputs = toy_model_m1(video_clip)[0]
                val_loss = loss_fn(val_outputs, video_clip)

                val_ave_loss += val_loss.item()

            val_ave_loss /= len(val_loader)
            val_loss_log.append(val_ave_loss)

            if val_ave_loss <= best_dic['val loss']:
                best_dic['epoch'] = epoch
                best_dic['model_state_dict'] = toy_model_m1.state_dict()
                                
            print('epoch: {}, train_ave_loss: {}, val_ave_loss: {}'.format(epoch, train_ave_loss, val_ave_loss))

        # save best checkpoints/models
        best_dic['optimizer_state_dict'] = optimizer.state_dict()
        best_dic['train loss'] = train_loss_log
        best_dic['val loss'] = val_loss_log
        torch.save(best_dic, os.path.join(path, 'm1_best.pth'))
    
    return checkpoint_name    
    
if __name__ == '__main__':
    cpn = training_m1('real', debugging=True)
    pass