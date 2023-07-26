import torch
import yaml
from model import model_1
from mydataset_gaga import m2_mydataset, m2_mydataset_sup
from torch.utils.data import DataLoader
from utils.set_seed import set_seed
from sklearn.svm import LinearSVC
import numpy as np
from matplotlib import pyplot as plt
import os


def example(checkpoint_name_list):
    fs = 15
    fig, axes = plt.subplots(4, 2, figsize=(6.4,6.4))
    axes[0][0].set_title('Original', fontsize=fs, rotation=0)
    axes[0][1].set_title('Reconstructed', fontsize=fs, rotation=0)
    axes[0][0].set_ylabel('Real\nCorrect', fontsize=fs, rotation=0, ha='right')
    axes[1][0].set_ylabel('Real\nWrong', fontsize=fs, rotation=0, ha='right')
    axes[2][0].set_ylabel('Virtual\nCorrect', fontsize=fs, rotation=0, ha='right')
    axes[3][0].set_ylabel('Virtual\nWrong', fontsize=fs, rotation=0, ha='right')
    
    for checkpoint_name in checkpoint_name_list:
        correct_example, wrong_example = 1, 1 
        # Hyperparameters
        with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'r') as file:
            HpParams = yaml.safe_load(file)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sup_ps = [1,3,5,7,9]


        for seed in HpParams['random seeds']:
            if correct_example == 0 and wrong_example == 0:
                continue
            set_seed(seed)
            print('m2 Seed: {}'.format(seed))
            lsvc = LinearSVC(random_state=seed, tol=1e-5)
            
            # dataset
            dataset = m2_mydataset(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed)
            loader = DataLoader(dataset, HpParams['batch_size'])

            # load states of the best model 1 
            toy_model_m1 = model_1(*dataset.tell_me_HW(), HpParams['latent_size'])
            path_m = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed), 'm1_best.pth')
            toy_model_m1.load_state_dict(torch.load(path_m)['model_state_dict'])
            toy_model_m1.eval()
            toy_model_m1.to(device)
            x, y = [], []
            for vc, label, _ in loader:
                output = toy_model_m1(vc.to(device))[1].squeeze().detach().cpu().numpy().astype(float)
                x.append(output)
                y.append(label.numpy().astype(int)) 
            lsvc.fit(np.concatenate(x), np.concatenate(y))

            for ps in sup_ps:
                if correct_example == 0 and wrong_example == 0:
                    continue
                dataset_sup = m2_mydataset_sup(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed, ps)
                loader_sup = DataLoader(dataset_sup, 1)
                x_sup, y_sup, vc_sup, vc_recon_sup = [], [], [], []
                for vc, label, _ in loader_sup:
                    vc_sup.append(vc.squeeze().permute(1,2,3,0).numpy())
                    output = toy_model_m1(vc.to(device))
                    vc_recon_sup.append(output[0].squeeze().detach().to('cpu').permute(1,2,3,0).numpy())
                    x_sup.append(output[1].squeeze().detach().to('cpu').numpy().astype(float))
                    y_sup.append(label.numpy().astype(int))
                y_pred = lsvc.predict(x_sup)
                
                corrects = np.where((np.concatenate(y_sup) == y_pred) == True)[0]
                if len(corrects) > 0:
                    correct = corrects[0]

                    if checkpoint_name_list.index(checkpoint_name) == 0:
                      axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][0].imshow(vc_sup[correct][-1][:,:,::-1])
                    else:
                      axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][0].imshow(vc_sup[correct][-1])
                    plt.setp(axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][0].get_xticklabels(), visible=False)
                    plt.setp(axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][0].get_yticklabels(), visible=False)
                    axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][0].tick_params(axis='both', which='both', length=0)

                    if checkpoint_name_list.index(checkpoint_name) == 0:
                      axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][1].imshow(vc_recon_sup[correct][-1][:,:,::-1])
                    else:
                      axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][1].imshow(vc_recon_sup[correct][-1])
                    plt.setp(axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][1].get_xticklabels(), visible=False)
                    plt.setp(axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][1].get_yticklabels(), visible=False)
                    axes[0 + checkpoint_name_list.index(checkpoint_name) * 2][1].tick_params(axis='both', which='both', length=0)

                    correct_example -= 1
                wrongs = np.where((np.concatenate(y_sup) == y_pred) == False)[0]
                if len(wrongs) > 0:
                    wrong = wrongs[0]

                    if checkpoint_name_list.index(checkpoint_name) == 0:
                      axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][0].imshow(vc_sup[wrong][-1][:,:,::-1])
                    else:
                      axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][0].imshow(vc_sup[wrong][-1])
                    plt.setp(axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][0].get_xticklabels(), visible=False)
                    plt.setp(axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][0].get_yticklabels(), visible=False)
                    axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][0].tick_params(axis='both', which='both', length=0)

                    if checkpoint_name_list.index(checkpoint_name) == 0:
                      axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][1].imshow(vc_recon_sup[wrong][-1][:,:,::-1])
                    else:
                      axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][1].imshow(vc_recon_sup[wrong][-1])
                    plt.setp(axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][1].get_xticklabels(), visible=False)
                    plt.setp(axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][1].get_yticklabels(), visible=False)
                    axes[1 + checkpoint_name_list.index(checkpoint_name) * 2][1].tick_params(axis='both', which='both', length=0)

                    wrong_example -= 1
    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    example(['debugging'])