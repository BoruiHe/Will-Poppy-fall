import torch
import torch.nn as nn
import torch.optim as optim
import os
from mydataset_regression import mydataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from DeepVO import DeepVO
import random
from hp import hyperparameters_regression
import yaml
import shutil


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def training():
    
    # Hyperparameters
    HpParams = hyperparameters_regression
    
    # path
    root_path = os.getcwd()
    checkpoint_name = 'ps{}_ep{}_deepvo'.format(HpParams['prediction_span'], HpParams['epochs'])
    shutil.rmtree(os.path.join('checkpoints', checkpoint_name))
    os.makedirs(os.path.join('checkpoints', checkpoint_name))
    with open(os.path.join(root_path, 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'w') as outfile:
        yaml.dump(HpParams, outfile, default_flow_style=False)

    training_p = os.path.join(root_path, 'data', 'training')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for seed in HpParams['random seeds']:
        set_seed(seed)
        print('Seed: {}'.format(seed))
        os.makedirs(os.path.join('checkpoints', checkpoint_name, str(seed)))

        dataset = mydataset(training_p, HpParams['numEclass'], HpParams['anchor_idx'], HpParams['prediction_span'], HpParams['dilation'], HpParams['f_size'], HpParams['k'])
        folds = dataset.cv_idx()
        loss_fn = nn.MSELoss(reduction='sum')

        # training & validation
        for fold, (train_idx, val_idx) in enumerate(folds):

            train_loss_log, val_loss_log = [], []
            
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, HpParams['batch_size'], sampler=train_sampler)
            val_loader = DataLoader(dataset, HpParams['batch_size'], sampler=val_sampler)

            toy_model = DeepVO(128, 128)
            toy_model.to(device)
            optimizer = optim.Adagrad(toy_model.parameters(), lr=0.001, weight_decay=0.0005)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 100, 150], gamma=0.8)

            best_dic= {'val loss': float('inf')}

            for epoch in range(HpParams['epochs']):
                # training
                train_ave_loss = 0
                toy_model.train()
                for train_data, _, train_base_labels in train_loader:
                    train_base_labels = (train_base_labels[:, 1:] - train_base_labels[:,:-1]).float()
                    train_data, train_base_labels = train_data.to(device), train_base_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    train_outputs = toy_model(train_data).squeeze()
                    train_loss = loss_fn(train_outputs, train_base_labels)
                    train_loss.backward()
                    optimizer.step()
                    
                    train_ave_loss += train_loss

                train_ave_loss = torch.sqrt(train_ave_loss).item()/len(train_sampler)
                train_loss_log.append(train_ave_loss)
                # scheduler.step()

                # validation
                val_ave_loss = 0

                toy_model.eval()

                for val_data, _, val_base_labels in val_loader:
                    val_base_labels = (val_base_labels[:, 1:] - val_base_labels[:,:-1]).float()
                    val_data, val_base_labels = val_data.to(device), val_base_labels.to(device)

                    val_outputs = toy_model(val_data).squeeze()
                    val_loss = loss_fn(val_outputs, val_base_labels)

                    val_ave_loss += val_loss

                val_ave_loss = torch.sqrt(val_ave_loss).item()/len(val_sampler)
                val_loss_log.append(val_ave_loss)

                if val_ave_loss <= best_dic['val loss']:
                    best_dic['epoch'] = epoch
                    best_dic['model_state_dict'] = toy_model.state_dict()
                    best_dic['optimizer_state_dict'] = optimizer.state_dict()
                    best_dic['train loss'] = train_ave_loss
                    best_dic['val loss'] = val_ave_loss
                            
                print('epoch: {}, train_ave_loss: {}, val_ave_loss: {}'.format(epoch, train_ave_loss, val_ave_loss))

            # save checkpoints/models
            torch.save({
                'model_state_dict': toy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train loss': train_loss_log,
                'val loss': val_loss_log,
                }, os.path.join(root_path, 'checkpoints', checkpoint_name, str(seed), 'toy_{}.pth'.format(fold)))

            # save best checkpoints/models
            torch.save(best_dic, os.path.join(root_path, 'checkpoints', checkpoint_name, str(seed), 'best_{}.pth'.format(fold)))
    
    return checkpoint_name

if __name__ == '__main__':
    checkpoint_name = training()
    pass