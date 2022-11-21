from unicodedata import name
from unittest.main import main
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from mydataset import mydataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import socket
import pickle
from toy import toy_sparse, toy_dense, get_fls
import random


def main(seed):

    # hyperparameters
    k = 4
    f_size = 10
    numEclass = 1
    epochs = 100
    batch_size = 5
    lr = 0.0001
    s_anchor, t_anchor = 100, 150
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    # path
    if socket.gethostname() == 'hbrDESKTOP':
        root_path = 'C:\\Users\\hbrch\\Desktop\\fdi'
    elif socket.gethostname() == 'MSI':
        root_path = 'C:\\Users\\Borui He\\Desktop\\falling detection images'

    dataset = mydataset(root_path, numEclass, f_size, s_anchor, t_anchor, k)
    folds = dataset.cv_idx()
    loss_fn = nn.CrossEntropyLoss()

    # training & validation
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_loss_log, train_acc_log, val_loss_log, val_acc_log, train_wrong_ip, val_wrong_ip = [], [], [], [], [], []
        
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        toy_model = toy_dense()
        toy_model.to(device)
        optimizer = optim.Adam(toy_model.parameters(), lr=lr)

        for epoch in range(epochs):
            # training
            train_running_loss, train_correct = 0, 0
            toy_model.train()
            for train_data, train_labels, train_anchor in train_loader:
                train_data, train_labels = train_data.to(device), train_labels.to(device)
                train_labels = train_labels.squeeze()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                train_outputs = toy_model(train_data).squeeze()
                train_loss = loss_fn(train_outputs, train_labels)
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()
                # Bug: when the number of fls in each fold cannot be divided by batch size, .softmax(dim=1) probably (when remainder = 1 ) cause IndexError!
                # if epoch == epochs - 1:
                #     if len(train_outputs.shape) == 1:
                #         train_wrong_ip.append((train_anchor, (train_outputs.softmax(dim=0).round() == train_labels).all(dim=0).to('cpu')))      
                #     else:
                #         train_wrong_ip.append((train_anchor, (train_outputs.softmax(dim=1).round() == train_labels).all(dim=1).to('cpu')))
                          
                if len(train_outputs.shape) == 1:
                    train_correct += (train_outputs.softmax(dim=0).round() == train_labels).all(dim=0).sum().item()
                else:
                    train_correct += (train_outputs.softmax(dim=1).round() == train_labels).all(dim=1).sum().item()
            
            train_ave_loss = train_running_loss/len(train_loader.sampler)
            train_loss_log.append(train_ave_loss)
            train_acc = train_correct / len(train_loader.sampler) * 100
            train_acc_log.append(train_acc)
            

            # validation
            val_running_loss, val_correct = 0, 0
            toy_model.eval()
            for val_data, val_labels, val_anchor in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_labels = val_labels.squeeze()

                val_outputs = toy_model(val_data).squeeze()
                val_loss = loss_fn(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                # Bug: when the number of fls in each fold cannot be divided by batch size, .softmax(dim=1) probably (when remainder = 1 ) cause IndexError!
                if len(val_outputs.shape) == 1:
                    val_correct += (val_outputs.softmax(dim=0).round() == val_labels).all(dim=0).sum().item()
                    val_wrong_ip.append((val_anchor, (val_outputs.softmax(dim=0).round() == val_labels).all(dim=0).to('cpu')))
                else:
                    val_correct += (val_outputs.softmax(dim=1).round() == val_labels).all(dim=1).sum().item()
                    val_wrong_ip.append((val_anchor, (val_outputs.softmax(dim=1).round() == val_labels).all(dim=1).to('cpu')))

            val_ave_loss = val_running_loss/len(val_loader.sampler)
            val_loss_log.append(val_ave_loss)
            val_acc = val_correct / len(val_loader.sampler) * 100
            val_acc_log.append(val_acc)
            
        
            print('epoch: {}, train_ave_loss: {}, train_acc: {}, val_ave_loss: {}, val_acc: {}'.format(epoch, train_ave_loss, train_acc, val_ave_loss, val_acc))

        # save checkpointss/models
        torch.save({
            # 'epoch': EPOCH,
            'model_state_dict': toy_model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train loss': train_loss_log,
            'train acc': train_acc_log,
            'train wrong ip': train_wrong_ip,
            'val loss': val_loss_log,
            'val acc': val_acc_log,
            'val wrong ip': val_wrong_ip
            }, os.path.join('C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection', 'checkpoints', str(seed), 'toy_{}.pth'.format(fold)))

if __name__ == '__main__':
    # random seed
    # random_seeds = np.random.choice(100, size=10, replace=False)
    # random_seeds = np.array([51, 84, 37, 42, 66, 31, 96, 77, 87, 59])
    random_seeds = np.array([51, 84])
    with open('random seed.pkl', 'wb') as f:
            pickle.dump(random_seeds, f)
    for seed in random_seeds:
        seed = seed.item()
        print('Seed: {}'.format(seed))
        os.makedirs(os.path.join('checkpoints', str(seed)))
        main(seed)
    pass