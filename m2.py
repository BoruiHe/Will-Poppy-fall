from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate
import numpy as np
from matplotlib import pyplot as plt
import os
import yaml
import torch
import pickle as pk
from model import model_1
from utils.set_seed import set_seed
from mydataset_gaga import m2_mydataset
from torch.utils.data import DataLoader


def m2(checkpoint_name):
    # Hyperparameters
    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    for k in range(len(HpParams['random seeds'])):

        seed = HpParams['random seeds'][k]
        set_seed(seed)
        print('m2 Seed: {}'.format(seed))

        lsvc = LinearSVC(random_state=seed, tol=1e-5)

        # dataset
        if HpParams['dataset_name'] == 'vir_poppy':
            dataset = m2_mydataset(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed)
        elif HpParams['dataset_name'] == 'real_poppy':
            dataset = m2_mydataset(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), seed)

        # load states of the best model 1 
        toy_model_m1 = model_1(*dataset.tell_me_HW(), HpParams['latent_size'])
        path_m = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed), 'm1_best.pth')
        toy_model_m1.load_state_dict(torch.load(path_m)['model_state_dict'])
        toy_model_m1.eval()
        toy_model_m1.to(device)

        loader = DataLoader(dataset, 1)
        x, y = [], []
        for vc, label, idx in loader:
            output = toy_model_m1(vc.to(device))[1].squeeze().detach().to('cpu').numpy().astype(float)
            x.append(output)
            # x.append(vc.flatten().numpy())
            y.append(label.numpy().astype(int))
            
        # train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(lsvc, np.array(x), np.array(y).ravel(), train_sizes=np.linspace(0.2, 1.0, 5), scoring='f1', shuffle=True, return_times=True)
        results = cross_validate(lsvc, np.array(x), np.array(y).ravel(), scoring='f1', cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed), return_train_score=True)
        train_scores = results['train_score']
        test_scores = results['test_score']
        train_sizes = [1, 2, 3, 4, 5]
        ax[0].plot(train_sizes, train_scores, 'o-', label=seed)
        # ax[0].fill_between(
        #     train_sizes,
        #     train_scores.mean(axis=1) - train_scores.std(axis=1),
        #     train_scores.mean(axis=1) + train_scores.std(axis=1),
        #     alpha=0.3,
        # )
        ax[0].set_ylabel('F1 score')
        ax[0].set_xlabel('folds')
        ax[0].set_title('training F1 score')

        ax[1].plot(train_sizes, test_scores, 'o-', label=seed)
        # ax[1].fill_between(
        #     train_sizes,
        #     test_scores.mean(axis=1) - test_scores.std(axis=1),
        #     test_scores.mean(axis=1) + test_scores.std(axis=1),
        #     alpha=0.3,
        # )
        ax[1].set_ylabel('F1 score')
        ax[1].set_xlabel('folds')
        ax[1].set_title('testing F1 score')

        pk.dump(results, open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, str(seed), 'm2.pkl'), 'wb'))
    
    ax[0].legend()
    ax[1].legend()
    # plt.show(block=True)
    fig.savefig(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name, 'm2 F1 score.png'))
    plt.close()

if __name__ == '__main__':
    m2('debugging')