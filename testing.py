import torch
import torch.nn as nn
import os
from model import toy_dense_Cla, model_shallow
from mydataset_classification import mydataset as mydataset_cla
from mydataset_regression  import mydataset as mydataset_reg
from torch.utils.data import DataLoader
from hp import hyperparameters_classification, hyperparameters_regression
import math


def testing(checkpoint_folder_name, mode):

    root_path = os.getcwd()
    testing_p = os.path.join(root_path, 'data', 'testing')
    
    # if mode == 'classificaiton':
    #     HpParams = hyperparameters_classification
    #     test_dataset = mydataset_cla(testing_p, HpParams['TnumEclass'], HpParams['anchor_idx'], HpParams['prediction_span'], HpParams['dilation'], HpParams['f_size'], None)
    #     toy_model = model_shallow(True, False)
    #     loss_fn = nn.BCELoss(reduction='mean')

    #     root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)

    #     for seed in HpParams['random seeds']:
    #         path = os.path.join(root_path, str(seed))
    #         test_loss_log, test_acc_log = [], []
    #         for model_f in ['best_0.pth', 'best_1.pth', 'best_2.pth', 'best_3.pth']:
    #             path_m = os.path.join(path, model_f)
    #             toy_model.load_state_dict(torch.load(path_m)['model_state_dict'])
    #             toy_model.eval()
    #             test_loader = DataLoader(test_dataset, batch_size= HpParams['batch_size'])
    #             test_ave_loss, test_correct = 0, 0

    #             for test_data, test_labels, test_base_labels in test_loader:
                    
    #                 test_outputs = toy_model(test_data).squeeze()
    #                 test_loss = loss_fn(test_outputs, test_labels)

    #                 test_ave_loss += test_loss.item()

    #                 test_correct += (test_outputs.round() == test_labels).all(dim=1).sum().item() # BCELoss

    #             test_loss_log.append(test_ave_loss)
    #             test_acc = test_correct / len(test_dataset) * 100
    #             test_acc_log.append(test_acc)

    #         torch.save({
    #                 'test loss': test_loss_log,
    #                 'test acc': test_acc_log,
    #                 }, os.path.join(path, 'test.pth'))
            
    if mode == 'regression':
        HpParams = hyperparameters_regression
        test_dataset = mydataset_reg(testing_p, HpParams['TnumEclass'], HpParams['anchor_idx'], HpParams['prediction_span'], HpParams['dilation'], HpParams['f_size'], None)
        toy_model = model_shallow(True, False)
        loss_fn = nn.MSELoss(reduction='sum')

        root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)

        for seed in HpParams['random seeds']:
            path = os.path.join(root_path, str(seed))
            test_loss_log = []
            for model_f in ['best_0.pth', 'best_1.pth', 'best_2.pth', 'best_3.pth']:
                path_m = os.path.join(path, model_f)
                toy_model.load_state_dict(torch.load(path_m)['model_state_dict'])
                toy_model.eval()
                test_loader = DataLoader(test_dataset, batch_size= HpParams['batch_size'])
                test_ave_loss = 0

                for test_data, test_labels, test_base_labels in test_loader:
                    test_base_labels = (test_base_labels[:, 1:] - test_base_labels[:,:-1]).float()
                    test_outputs = toy_model(test_data).squeeze()
                    test_loss = loss_fn(test_outputs, test_base_labels)

                    test_ave_loss += test_loss.item()
                
                test_ave_loss = math.sqrt(test_ave_loss)/len(test_dataset)
                test_loss_log.append(test_ave_loss)

            torch.save({
                    'test loss': test_loss_log,
                    }, os.path.join(path, 'test_b.pth'))
            
            test_loss_log = []
            for model_f in ['toy_0.pth', 'toy_1.pth', 'toy_2.pth', 'toy_3.pth']:
                path_m = os.path.join(path, model_f)
                toy_model.load_state_dict(torch.load(path_m)['model_state_dict'])
                toy_model.eval()
                test_loader = DataLoader(test_dataset, batch_size= HpParams['batch_size'])
                test_ave_loss = 0

                for test_data, test_labels, test_base_labels in test_loader:
                    test_base_labels = (test_base_labels[:, 1:] - test_base_labels[:,:-1]).float()
                    test_outputs = toy_model(test_data).squeeze()
                    test_loss = loss_fn(test_outputs, test_base_labels)

                    test_ave_loss += test_loss.item()

                test_ave_loss = math.sqrt(test_ave_loss)/len(test_dataset)
                test_loss_log.append(test_ave_loss)

            torch.save({
                    'test loss': test_loss_log,
                    }, os.path.join(path, 'test_t.pth'))
    else:
        raise Exception('wrong testing mode!')
if __name__ == '__main__':
    testing('ps2_ep200_R_Sh_diff', 'regression')


