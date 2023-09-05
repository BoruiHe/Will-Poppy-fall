from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import os
import yaml
import torch
import pickle as pk
from model import model_1
from utils.set_seed import set_seed
from mydataset_gaga import m2_mydataset, m2_mydataset_sup
from torch.utils.data import DataLoader


def m2_sup(checkpoint_name):
    # Hyperparameters
    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    sup_ps = [1,3,5,7,9]
    summary = {}
    for seed in HpParams['random seeds']:
        set_seed(seed)
        summary[seed] = {}
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
            summary[seed][ps] = {}
            dataset_sup = m2_mydataset_sup(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed, ps)
            loader_sup = DataLoader(dataset_sup, 1)
            x_sup, y_sup = [], []
            for vc, label, _ in loader_sup:
                output = toy_model_m1(vc.to(device))[1].squeeze().detach().to('cpu').numpy().astype(float)
                x_sup.append(output)
                y_sup.append(label.numpy().astype(int))
            y_pred = lsvc.predict(x_sup)
            summary[seed][ps]['ba'] = balanced_accuracy_score(y_sup, y_pred)
            summary[seed][ps]['f1'] = f1_score(y_sup, y_pred)

    pk.dump(summary, open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'shift.pkl'), 'wb'))
    ave_ba, ave_f1 = [], []
    for ps in sup_ps:
        ave_ba.append(sum([summary[seed][ps]['ba'] for seed in HpParams['random seeds']]) / len([summary[seed][ps]['ba'] for seed in HpParams['random seeds']]))
        ave_f1.append(sum([summary[seed][ps]['f1'] for seed in HpParams['random seeds']]) / len([summary[seed][ps]['f1'] for seed in HpParams['random seeds']]))

    ax[0].bar(sup_ps, ave_ba)
    ax[0].set_ylim(bottom=0., top=1.)
    ax[0].set_xticks(sup_ps)
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('prediction span')
    ax[0].set_title('Ave. of balanced accuracy')

    ax[1].bar(sup_ps, ave_f1)
    ax[1].set_ylim(bottom=0., top=1.)
    ax[1].set_xticks(sup_ps)
    ax[1].set_ylabel('F1 score')
    ax[1].set_xlabel('prediction span')
    ax[1].set_title('Ave. of F1 score')

    # plt.show(block=True)
    fig.savefig(os.path.join(os.getcwd(), 'sup', checkpoint_name + '_shift.png'))
    plt.close()

if __name__ == '__main__':
    m2_sup('gaga_real_ps1_256')