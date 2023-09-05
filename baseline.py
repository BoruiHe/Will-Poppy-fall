import torch
import os
import yaml
import shutil
import random
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from utils.set_seed import set_seed
from sklearn.svm import LinearSVC
from mydataset_gaga import baseline_dataset, m2_mydataset, m2_mydataset_sup
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import balanced_accuracy_score, f1_score
from hp import hyperparameters_virtual, hyperparameters_real  


def testing_3rd_party_recognition(dataset_name:str):
    checkpoint_name = 'baseline_' + dataset_name
    if os.path.exists(os.path.join('checkpoints', checkpoint_name, '_recog')):
        shutil.rmtree(os.path.join('checkpoints', checkpoint_name, '_recog'))
    os.makedirs(os.path.join('checkpoints', checkpoint_name, '_recog'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet= resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)
    HpParams = {}
    HpParams['random seeds'] = random.sample(range(3224), k=3)

    for seed in HpParams['random seeds']:
        set_seed(seed)
        print('baseline Seed: {}'.format(seed))
        path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed))
        os.makedirs(path)
        # dataset
        dataset_baseline = baseline_dataset(dataset_name)
        baseline_dataloader = DataLoader(dataset_baseline, 64) # set batch size to 1 due to my poor GPU memory
        lsvc = LinearSVC(random_state=seed)
        x, y = [], []

        for frames, label in baseline_dataloader: # frames: [64, 10, 3, h, w]
            for bs_idx in range(frames.shape[0]):
                x.append(np.concatenate(resnet(frames[bs_idx,:,:,:,:].to(device)).detach().cpu().numpy()))
            y.append(label.numpy())
        
        results = cross_validate(lsvc, np.array(x), np.concatenate(y), scoring=('balanced_accuracy', 'f1'), cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed))
        pk.dump(results, open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed), 'ba.pkl'), 'wb'))
        print('baseline model on {}; seed: {}; testing balanced accuracy: {}; testing f1 score: {}.'.format(dataset_name, seed, results['test_balanced_accuracy'], results['test_f1']))

def testing_3rd_party_prediction(dataset_name:str):
    if dataset_name == 'vir_poppy':
        HpParams = hyperparameters_virtual
    elif dataset_name == 'real_poppy':
        HpParams = hyperparameters_real
    checkpoint_name = 'baseline_' + dataset_name + '_pred_ps{}_2SF'.format(HpParams['prediction_span'])
    if os.path.exists(os.path.join('checkpoints', checkpoint_name)):
        shutil.rmtree(os.path.join('checkpoints', checkpoint_name))
    os.makedirs(os.path.join('checkpoints', checkpoint_name))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet= resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)
    HpParams['random seeds'] = random.sample(range(3224), k=3)
    sup_ps = [1,3,5,7,9]
    summary = {}
    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'w') as outfile:
        yaml.dump(HpParams, outfile, default_flow_style=False)

    for seed in HpParams['random seeds']:
        set_seed(seed)
        print('baseline Seed: {}'.format(seed))
        summary[seed] = {}
        path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed))
        os.makedirs(path)
        # dataset
        dataset = m2_mydataset(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed)
        loader = DataLoader(dataset, 1)
        lsvc = LinearSVC(random_state=seed, dual=False)
        x, y = [], []
        for frames, label, _ in loader: 
            frames = frames.permute(0,2,1,3,4) # frames: [64, 10, 3, h, w]
            for bs_idx in range(frames.shape[0]):
                x.append(np.concatenate(resnet(frames[bs_idx,:,:,:,:].to(device)).detach().cpu().numpy()))
            y.append(label.numpy())
        lsvc.fit(np.array(x), np.concatenate(y))
        for ps in sup_ps:
            summary[seed][ps] = {}
            dataset_sup = m2_mydataset_sup(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed, ps)
            loader_sup = DataLoader(dataset_sup, 1)
            x_sup, y_sup = [], []
            for frames, label, _ in loader_sup: 
                frames = frames.permute(0,2,1,3,4) # frames: [64, 10, 3, h, w]
                for bs_idx in range(frames.shape[0]):
                    x_sup.append(np.concatenate(resnet(frames[bs_idx,:,:,:,:].to(device)).detach().cpu().numpy()))
                y_sup.append(label.numpy())
            y_pred = lsvc.predict(x_sup)
            summary[seed][ps]['ba'] = balanced_accuracy_score(y_sup, y_pred)
            summary[seed][ps]['f1'] = f1_score(y_sup, y_pred)
    pk.dump(summary, open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'shift.pkl'), 'wb'))
    ave_ba, ave_f1 = [], []
    for ps in sup_ps:
        ave_ba.append(sum([summary[seed][ps]['ba'] for seed in HpParams['random seeds']]) / len([summary[seed][ps]['ba'] for seed in HpParams['random seeds']]))
        ave_f1.append(sum([summary[seed][ps]['f1'] for seed in HpParams['random seeds']]) / len([summary[seed][ps]['f1'] for seed in HpParams['random seeds']]))
    fig, ax = plt.subplots(nrows=1, ncols=2) # figsize=(16, 6)
    ax[0].bar(sup_ps, ave_ba)
    ax[0].set_ylim(bottom=0., top=1.)
    ax[0].set_xticks(sup_ps)
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('prediction span')
    ax[0].set_title('Ave. of balanced accuracy')
    ax[0].spines[['right', 'top']].set_visible(False)

    ax[1].bar(sup_ps, ave_f1)
    ax[1].set_ylim(bottom=0., top=1.)
    ax[1].set_xticks(sup_ps)
    ax[1].set_ylabel('F1 score')
    ax[1].set_xlabel('prediction span')
    ax[1].set_title('Ave. of F1 score')
    ax[1].spines[['right', 'top']].set_visible(False)
    
    plt.tight_layout()
    # plt.show(block=True)
    fig.savefig(os.path.join(os.getcwd(), 'sup', checkpoint_name + '.png'.format(HpParams['prediction_span'])))
    plt.close()

if __name__ == '__main__':
    testing_3rd_party_prediction('vir_poppy')