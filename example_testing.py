import torch
import torch.nn as nn
import os
from model import toy_dense_Cla, model_shallow
from mydataset_classification import mydataset as mydataset_cla
from mydataset_regression  import mydataset as mydataset_reg
# from torch.utils.data import DataLoader
from hp import hyperparameters_classification, hyperparameters_regression
import matplotlib.pyplot as plt


def exa_testing(checkpoint_folder_name, mode):
    
    root_path = os.getcwd()
    testing_p = os.path.join(root_path, 'data', 'testing')
    
    if mode == 'regression':
        HpParams = hyperparameters_regression
        test_dataset = mydataset_reg(testing_p, 1, HpParams['anchor_idx'], HpParams['prediction_span'], HpParams['dilation'], HpParams['f_size'], None)
        toy_model = model_shallow(True, False)

        root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)

        for p in test_dataset.chosen_paths['paths']:
            if 'vibration' in p:
                vib = p[43:].split('\\')[1]
                os.makedirs(os.path.join(root_path, vib))
            else:
                fall = p[43:].split('\\')[1]
                os.makedirs(os.path.join(root_path, fall))

        for seed in HpParams['random seeds']:
            path = os.path.join(root_path, str(seed))
       
            for model_f in ['best_0.pth', 'best_1.pth', 'best_2.pth', 'best_3.pth']:
                fall_log, vib_log, v_idx, f_idx = [[], []], [[], []], [], []
                path_m = os.path.join(path, model_f)
                toy_model.load_state_dict(torch.load(path_m)['model_state_dict'])
                toy_model.eval()

                for (test_data, test_labels) in test_dataset.v_inputs:
                    test_data = torch.tensor(test_data).float().permute(3,0,1,2) / 255
                    test_data = test_data[None, :]
                    test_outputs = toy_model(test_data).squeeze()
                    vib_log[0].append(test_outputs.detach().numpy())
                    vib_log[1].append(test_labels[0])
                    v_idx.append(test_labels[2])

                plt.scatter(v_idx, vib_log[0], label='pred')
                plt.scatter(v_idx, vib_log[1], label='ground truth')
                plt.legend()
                plt.savefig(os.path.join(root_path, vib, str(seed) + '_' + model_f[:-4] + '_v'))
                plt.close()

                for (test_data, test_labels) in test_dataset.f_inputs:
                    test_data = torch.tensor(test_data).float().permute(3,0,1,2) / 255
                    test_data = test_data[None, :]
                    test_outputs = toy_model(test_data).squeeze()
                    fall_log[0].append(test_outputs.detach().numpy())
                    fall_log[1].append(test_labels[0])
                    f_idx.append(test_labels[2])
            
                plt.scatter(f_idx, fall_log[0], label='pred')
                plt.scatter(f_idx, fall_log[1], label='ground truth')
                plt.legend()
                plt.savefig(os.path.join(root_path, fall, str(seed) + '_' + model_f[:-4] + '_f'))
                plt.close()


            for model_f in ['toy_0.pth', 'toy_1.pth', 'toy_2.pth', 'toy_3.pth']:
                fall_log, vib_log, v_idx, f_idx = [[], []], [[], []], [], []
                path_m = os.path.join(path, model_f)
                toy_model.load_state_dict(torch.load(path_m)['model_state_dict'])
                toy_model.eval()

                for (test_data, test_labels) in test_dataset.v_inputs:
                    test_data = torch.tensor(test_data).float().permute(3,0,1,2) / 255
                    test_data = test_data[None, :]
                    test_outputs = toy_model(test_data).squeeze()
                    vib_log[0].append(test_outputs.detach().numpy())
                    vib_log[1].append(test_labels[0])
                    v_idx.append(test_labels[2])

                plt.scatter(v_idx, vib_log[0], label='pred')
                plt.scatter(v_idx, vib_log[1], label='ground truth')
                plt.legend()
                plt.savefig(os.path.join(root_path, vib, str(seed) + '_' + model_f[:-4] + '_v'))
                plt.close()

                for (test_data, test_labels) in test_dataset.f_inputs:
                    test_data = torch.tensor(test_data).float().permute(3,0,1,2) / 255
                    test_data = test_data[None, :]
                    test_outputs = toy_model(test_data).squeeze()
                    fall_log[0].append(test_outputs.detach().numpy())
                    fall_log[1].append(test_labels[0])
                    f_idx.append(test_labels[2])
            
                plt.scatter(f_idx, fall_log[0], label='pred')
                plt.scatter(f_idx, fall_log[1], label='ground truth')
                plt.legend()
                plt.savefig(os.path.join(root_path, fall, str(seed) + '_' + model_f[:-4] + '_f'))
                plt.close()

    else:
        raise Exception('wrong testing mode!')
if __name__ == '__main__':
    exa_testing('ps2_ep200_shallow_Zcor', 'regression')


