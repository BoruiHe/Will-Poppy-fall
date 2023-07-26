import os
import matplotlib.pyplot as plt
import yaml
import pickle as pk
import math
import numpy as np


def baseline_summary_pred_vir():
    root_path = os.path.join(os.getcwd(), 'checkpoints')
    
    for scenario in ['_2F', '_F_SF', '_SF_F', '_2SF']:
        print(scenario)
        highest_std_ba, highest_std_f1 = 0, 0
        for ps in [1,5,9]:
            folder_name = f'baseline_vir_poppy_pred_ps{ps}' + scenario
            with open(os.path.join(root_path, folder_name, 'shift.pkl'), 'rb') as f:
                results = pk.load(f)
            repeitions = list(results.values())
            # print(folder_name)
            summary = {}
            for tps in [1,5,9]:
                summary[tps] = {'ba':[], 'f1':[]}
                summary[tps]['ba'].extend([ele[tps]['ba'] for ele in repeitions])
                summary[tps]['ba'] = np.array(summary[tps]['ba'])
                summary[tps]['ba'] = (np.round(summary[tps]['ba'].mean(), 4), np.round(summary[tps]['ba'].std(), 4))
                summary[tps]['f1'].extend([ele[tps]['f1'] for ele in repeitions])
                summary[tps]['f1'] = np.array(summary[tps]['f1'])
                summary[tps]['f1'] = (np.round(summary[tps]['f1'].mean(), 2), np.round(summary[tps]['f1'].std(), 4))
                # print(tps, summary[tps])
                if summary[tps]['ba'][1] > highest_std_ba:
                    highest_std_ba = summary[tps]['ba'][1]
                if summary[tps]['f1'][1] > highest_std_ba:
                    highest_std_f1 = summary[tps]['f1'][1]
        print('highest_std_ba', highest_std_ba, 'highest_std_f1', highest_std_f1)

def baseline_summary_recog():
    for folder in ['baseline_vir_poppy_recog', 'baseline_real_poppy_recog']:
        path = os.path.join(os.getcwd(), 'checkpoints', folder)
        ba, f1 = [], []
        for (dirpath, dirnames, filenames) in os.walk(path):
            for file in filenames:
                with open(os.path.join(dirpath, file), 'rb') as f:
                    results = pk.load(f)
                    ba.append(results['test_balanced_accuracy'].mean())
                    f1.append(results['test_f1'].mean())
        print(folder)
        print('ba', np.round(np.array(ba).mean(), 5), np.round(np.array(ba).std(), 4))
        print('f1', np.round(np.array(f1).mean(), 5), np.round(np.array(f1).std(), 4))

def baseline_summary_pred_real():
    for folder in ['baseline_real_poppy_pred_ps1', 'baseline_real_poppy_pred_ps5', 'baseline_real_poppy_pred_ps9']:
        path = os.path.join(os.getcwd(), 'checkpoints', folder)
        ba, f1 = {}, {}
        with open(os.path.join(path, 'shift.pkl'), 'rb') as f:
            results = pk.load(f)
        
        for ps in [1,3,5,7,9]:
            ba[ps], f1[ps] = [], []
            for random_seed_dictionary in list(results.values()):
                ba[ps].append(random_seed_dictionary[ps]['ba'].mean())
                f1[ps].append(random_seed_dictionary[ps]['f1'].mean())
        print(folder)
        for ps in [1,5,9]:
            if ps <= int(folder[-1]):
                print(f'ba ps{ps}:', np.round(np.array(ba[ps]).mean(), 3), np.round(np.array(ba[ps]).std(), 4))
                print(f'f1 ps{ps}:', np.round(np.array(f1[ps]).mean(), 3), np.round(np.array(f1[ps]).std(), 4))

def plot(dataset_name):
    root_path = os.path.join(os.getcwd(), 'checkpoints')
    for ps in [1,5,9]:
        summary = {}
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        for ls in [256, 512, 1024, 2048]:
            summary[ls] = {}
            path = os.path.join(root_path, f'gaga_{dataset_name}_ps{ps}_{ls}')
            with open(os.path.join(path, 'HyperParam.yml'), 'r') as file:
                HpParams = yaml.safe_load(file)

            with open(os.path.join(path, 'shift.pkl'), 'rb') as f:
                result = pk.load(f)

                for sup_ps in [1,3,5,7,9]:
                    summary[ls][sup_ps] = {
                            'f1': [],
                            'ba': []
                        }
                    for seed in HpParams['random seeds']:
                        
                        summary[ls][sup_ps]['f1'].append(result[seed][sup_ps]['f1'])
                        summary[ls][sup_ps]['ba'].append(result[seed][sup_ps]['ba'])
            
                    summary[ls][sup_ps]['f1'] = sum(summary[ls][sup_ps]['f1']) / len(summary[ls][sup_ps]['f1'])
                    summary[ls][sup_ps]['ba'] = sum(summary[ls][sup_ps]['ba']) / len(summary[ls][sup_ps]['ba'])

        offset = 0.6
        for ls in [256, 512, 1024, 2048]:
            ax[0].bar([ps - offset for ps in [1,3,5,7,9]], [summary[ls][sup_ps]['f1'] for sup_ps in [1,3,5,7,9]], width=0.4, label= str(ls), alpha=0.8)
            ax[1].bar([ps - offset for ps in [1,3,5,7,9]], [summary[ls][sup_ps]['ba'] for sup_ps in [1,3,5,7,9]], width=0.4, label= str(ls), alpha=0.8)
            offset -= 0.4

        ax[0].set_ylabel('f1 score')
        ax[0].set_xlabel('prediction span')
        ax[0].set_xticks([1,3,5,7,9])
        ax[0].set_ylim(bottom=0., top=1.2)
        ax[0].set_title('testing f1 score')
        ax[0].spines[['right', 'top']].set_visible(False)
        ax[0].legend(loc=2, title='Latent size')

        ax[1].set_ylabel('accuracy')
        ax[1].set_xlabel('prediction span')
        ax[1].set_xticks([1,3,5,7,9])
        ax[1].set_ylim(bottom=0., top=1.2)
        ax[1].set_title('testing balanced accuracy')
        ax[1].spines[['right', 'top']].set_visible(False)
        ax[1].legend(loc=2, title='Latent size')

        # plt.show(block=True)
        fig.savefig(os.path.join(os.getcwd(), 'plots', '{}_Poppy_{}_shift.png'.format(dataset_name, ps)))

def ours_summary(dataset_name):
    root_path = os.path.join(os.getcwd(), 'checkpoints')
    highest_std_ba, highest_std_f1 = 0, 0
    for ps in [1,5,9]:
        summary = {}
        for ls in [256, 512, 1024, 2048]:
            summary[ls] = {}
            path = os.path.join(root_path, f'gaga_{dataset_name}_ps{ps}_{ls}')
            with open(os.path.join(path, 'HyperParam.yml'), 'r') as file:
                HpParams = yaml.safe_load(file)

            with open(os.path.join(path, 'shift.pkl'), 'rb') as f:
                result = pk.load(f)

                for sup_ps in [1,5,9]:
                    summary[ls][sup_ps] = {
                            'f1': [],
                            'ba': []
                        }
                    for seed in HpParams['random seeds']:
                        
                        summary[ls][sup_ps]['f1'].append(result[seed][sup_ps]['f1'])
                        summary[ls][sup_ps]['ba'].append(result[seed][sup_ps]['ba'])
            
                    summary[ls][sup_ps]['f1'] = np.round(np.array(summary[ls][sup_ps]['f1']).mean(), 2), np.round(np.array(summary[ls][sup_ps]['f1']).std(), 8)

                    summary[ls][sup_ps]['ba'] = np.round(np.array(summary[ls][sup_ps]['ba']).mean(), 4), np.round(np.array(summary[ls][sup_ps]['ba']).std(), 8)

                    if sup_ps <= ps:
                        if summary[ls][sup_ps]['f1'][1] > highest_std_f1:
                            highest_std_f1 = summary[ls][sup_ps]['f1'][1]
                        if summary[ls][sup_ps]['ba'][1] > highest_std_ba:
                            highest_std_ba = summary[ls][sup_ps]['ba'][1]
                        print(f'{ls, ps, sup_ps}', summary[ls][sup_ps])
    print('highest_std_ba', highest_std_ba, 'highest_std_f1', highest_std_f1)

def scoreplot(dataset_name):
    root_path = os.path.join(os.getcwd(), 'checkpoints')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 16))
    ax[0].tick_params(axis='both', which='both', labelsize=20)
    ax[1].tick_params(axis='both', which='both', labelsize=20)
    for ps in [1,9]:
        summary = {}
        for ls in [256, 512, 1024, 2048]:
            summary[ls] = {}
            path = os.path.join(root_path, f'gaga_{dataset_name}_ps{ps}_{ls}')
            with open(os.path.join(path, 'HyperParam.yml'), 'r') as file:
                HpParams = yaml.safe_load(file)

            with open(os.path.join(path, 'shift.pkl'), 'rb') as f:
                result = pk.load(f)

                for sup_ps in [1,3,5,7,9]:
                    summary[ls][sup_ps] = {
                            'f1': [],
                            'ba': []
                        }
                    for seed in HpParams['random seeds']:
                        
                        summary[ls][sup_ps]['f1'].append(result[seed][sup_ps]['f1'])
                        summary[ls][sup_ps]['ba'].append(result[seed][sup_ps]['ba'])
            
                    summary[ls][sup_ps]['f1'] = sum(summary[ls][sup_ps]['f1']) / len(summary[ls][sup_ps]['f1'])
                    summary[ls][sup_ps]['ba'] = sum(summary[ls][sup_ps]['ba']) / len(summary[ls][sup_ps]['ba'])

        offset = 0.6
        for ls in [256, 512, 1024, 2048]:
            ax[[1,9].index(ps)].bar([ps - offset for ps in [1,3,5,7,9]], [summary[ls][sup_ps]['ba'] for sup_ps in [1,3,5,7,9]], width=0.4, label= str(ls), alpha=0.8)
            offset -= 0.4

        fs=25

        ax[0].set_ylabel('accuracy', fontsize=fs)
        ax[0].set_xticks([1,3,5,7,9], fontsize=fs)
        ax[0].set_ylim(bottom=0., top=1.)
        # ax[0].set_title('training predicition span: 1', fontsize=fs)
        ax[0].spines[['right', 'top']].set_visible(False)
        ax[0].legend(loc=1, title='Latent size', fontsize=fs, title_fontsize=fs)

        ax[1].set_ylabel('accuracy', fontsize=fs)
        ax[1].set_xlabel('testing prediction span', fontsize=fs)
        ax[1].set_xticks([1,3,5,7,9], fontsize=fs)
        ax[1].set_ylim(bottom=0., top=1.)
        # ax[1].set_title('training predicition span: 9', fontsize=fs)
        ax[1].spines[['right', 'top']].set_visible(False)
        ax[1].legend(loc=1, title='Latent size', fontsize=fs, title_fontsize=fs)

    plt.tight_layout()
    plt.show(block=True)
    # fig.savefig(os.path.join('comp.pdf'))

if __name__ == '__main__':
    # scoreplot('real')
    # summary('vir')
    # ours_summary('real')
    baseline_summary_pred_real()
    # ours_summary('real')
