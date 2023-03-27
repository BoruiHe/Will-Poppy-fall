import torch
import torch.nn as nn
import torch.optim as optim
import os
from mydataset_classification import mydataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import toy_dense_Cla, init_weights
import random
from hp import hyperparameters_classification
import yaml


def main():
    # Hyperparameters
    HpParams = hyperparameters_classification

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # path
    root_path = 'C:\\Users\\hbrch\\Desktop\\fdi_fl'
    training_p = os.path.join(root_path, 'data', 'training')
    os.makedirs(os.path.join('checkpoints', 'p{}'.format(HpParams['prediction_span'])))
    with open(os.path.join(root_path, 'checkpoints', 'p{}'.format(HpParams['prediction_span']), 'HyperParam.yml'), 'w') as outfile:
        yaml.dump(HpParams, outfile, default_flow_style=False)

    for seed in HpParams['random seeds']:
        set_seed(seed)
        print('Seed: {}'.format(seed))
        os.makedirs(os.path.join('checkpoints', 'p{}'.format(HpParams['prediction_span']), str(seed)))

        dataset = mydataset(training_p, HpParams['numEclass'], HpParams['anchor_idx'], HpParams['prediction_span'], HpParams['dilation'], HpParams['f_size'], HpParams['k'])
        folds = dataset.cv_idx()
        loss_fn = nn.BCELoss(reduction='mean')
        # loss_fn = nn.MSELoss(reduction='mean')

        # training & validation
        for fold, (train_idx, val_idx) in enumerate(folds):
            train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
            tn_F_acc, tn_Fv_acc, tn_V_acc, val_F_acc, val_Fv_acc, val_V_acc = [], [], [], [], [], []
            
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, HpParams['batch_size'], sampler=train_sampler)
            val_loader = DataLoader(dataset, HpParams['batch_size'], sampler=val_sampler)

            toy_model = toy_dense_Cla()
            # toy_model.apply(init_weights)
            toy_model.to(device)
            optimizer = optim.Adam(toy_model.parameters(), lr=HpParams['lr'], weight_decay=0.0005)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,60], gamma=0.9)

            best_dic= {'val loss': float('inf')}

            for epoch in range(HpParams['epochs']):
                # training
                train_ave_loss, train_correct = 0, 0
                stat_dic = {'F':0, 'Fv':0, 'V':0, 'F_total':0, 'Fv_total':0, 'V_total':0,}
                toy_model.train()

                for train_data, train_labels, tn_tp in train_loader:
                    train_data, train_labels, tn_tp = train_data.to(device), train_labels.to(device), list(tn_tp)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    train_outputs = toy_model(train_data).squeeze()
                    train_loss = loss_fn(train_outputs, train_labels)
                    train_loss.backward()
                    optimizer.step()
                    
                    train_ave_loss += train_loss.item()
                    
                    # (train_outputs.round() == train_labels).sum().item() # MSELoss
                    mch = (train_outputs.round() == train_labels).all(dim=1).int()
                    for tp, tp_total in zip(['F', 'Fv', 'V'], ['F_total', 'Fv_total', 'V_total']):
                        tem_mask = []
                        for idx in range(len(tn_tp)):
                            if tn_tp[idx] == tp:
                                tem_mask.append(True)
                            else:
                                tem_mask.append(False)
                        tem_mask = torch.tensor(tem_mask)

                        masked_mch = torch.masked_select(mch.to('cpu'), tem_mask)
                        stat_dic[tp] += masked_mch.sum().item()
                        stat_dic[tp_total] += tem_mask.sum().item()

                train_correct += stat_dic['F'] + stat_dic['Fv'] + stat_dic['V']

                train_loss_log.append(train_ave_loss)
                train_acc = train_correct / len(train_loader.sampler) * 100
                train_acc_log.append(train_acc)
                if stat_dic['F_total'] == 0:
                    tn_F_acc.append(None)
                else:
                    tn_F_acc.append(stat_dic['F']/stat_dic['F_total'])
                if stat_dic['Fv_total'] == 0:
                    tn_Fv_acc.append(None)
                else:
                    tn_Fv_acc.append(stat_dic['Fv']/stat_dic['Fv_total'])
                if stat_dic['V_total'] == 0:
                    tn_V_acc.append(None)
                else:
                    tn_V_acc.append(stat_dic['V']/stat_dic['V_total'])
                scheduler.step()

                # validation
                val_ave_loss, val_correct = 0, 0
                stat_dic = {'F':0, 'Fv':0, 'V':0, 'F_total':0, 'Fv_total':0, 'V_total':0,}
                toy_model.eval()

                for val_data, val_labels, v_tp in val_loader:
                    val_data, val_labels, v_tp = val_data.to(device), val_labels.to(device), list(v_tp)

                    val_outputs = toy_model(val_data).squeeze()
                    val_loss = loss_fn(val_outputs, val_labels)

                    val_ave_loss += val_loss.item()

                    # (train_outputs.round() == train_labels).sum().item() # MSELoss
                    mch = (val_outputs.round() == val_labels).all(dim=1).int()
                    for tp, tp_total in zip(['F', 'Fv', 'V'], ['F_total', 'Fv_total', 'V_total']):
                        tem_mask = []
                        for idx in range(len(v_tp)):
                            if v_tp[idx] == tp:
                                tem_mask.append(True)
                            else:
                                tem_mask.append(False)
                        tem_mask = torch.tensor(tem_mask)

                        masked_mch = torch.masked_select(mch.to('cpu'), tem_mask)
                        stat_dic[tp] += masked_mch.sum().item()
                        stat_dic[tp_total] += tem_mask.sum().item()

                val_correct += stat_dic['F'] + stat_dic['Fv'] + stat_dic['V']

                val_loss_log.append(val_ave_loss)
                val_acc = val_correct / len(val_loader.sampler) * 100
                val_acc_log.append(val_acc)
                if stat_dic['F_total'] == 0:
                    val_F_acc.append(None)
                else:
                    val_F_acc.append(stat_dic['F']/stat_dic['F_total'])
                if stat_dic['Fv_total'] == 0:
                    val_Fv_acc.append(None)
                else:
                    val_Fv_acc.append(stat_dic['Fv']/stat_dic['Fv_total'])
                if stat_dic['V_total'] == 0:
                    val_V_acc.append(None)
                else:
                    val_V_acc.append(stat_dic['V']/stat_dic['V_total'])

                if val_ave_loss <= best_dic['val loss']:
                    best_dic['epoch'] = epoch
                    best_dic['model_state_dict'] = toy_model.state_dict()
                    best_dic['optimizer_state_dict'] = optimizer.state_dict()
                    best_dic['train acc'] = train_acc
                    best_dic['train loss'] = train_ave_loss
                    best_dic['tn_F_acc'] = tn_F_acc
                    best_dic['tn_Fv_acc'] = tn_Fv_acc
                    best_dic['tn_V_acc'] = tn_V_acc
                    best_dic['val acc'] = val_acc
                    best_dic['val loss'] = val_ave_loss
                    best_dic['val_F_acc'] = val_F_acc
                    best_dic['val_Fv_acc'] = val_Fv_acc
                    best_dic['val_V_acc'] = val_V_acc
                            
                print('epoch: {}, train_ave_loss: {}, train_acc: {}, val_ave_loss: {}, val_acc: {}'.format(epoch, train_ave_loss, train_acc, val_ave_loss, val_acc))

            # save checkpoints/models
            torch.save({
                'model_state_dict': toy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train loss': train_loss_log,
                'train acc': train_acc_log,
                'tn_F_acc': tn_F_acc,
                'tn_Fv_acc': tn_Fv_acc,
                'tn_V_acc': tn_V_acc,
                'val loss': val_loss_log,
                'val acc': val_acc_log,
                'val_F_acc': val_F_acc,
                'val_Fv_acc': val_Fv_acc,
                'val_V_acc': val_V_acc,
                }, os.path.join(root_path, 'checkpoints', 'p{}'.format(HpParams['prediction_span']), str(seed), 'toy_{}.pth'.format(fold)))

            # save best checkpoints/models
            torch.save(best_dic, os.path.join(root_path, 'checkpoints', 'p{}'.format(HpParams['prediction_span']), str(seed), 'best_{}.pth'.format(fold)))


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
    pass