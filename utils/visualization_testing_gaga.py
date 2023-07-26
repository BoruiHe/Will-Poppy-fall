import os
import torch
import matplotlib.pyplot as plt
import yaml


def vis_testing_m1(checkpoint_folder_name):

    with open(os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)
    
    fig1, axs1 = plt.subplot_mosaic([['loss'], ['rMSE']], 
                                    tight_layout= True, 
                                    figsize=(12,9))

    loss_sum, rMSE_sum = 0, 0

    for seed in HpParams['random seeds']:

        path = os.path.join(root_path, str(seed))
        loss = torch.load(os.path.join(path, 'm1_test.pth'))['test loss']
        rMSE = torch.load(os.path.join(path, 'm1_test.pth'))['test rMSE']
        loss_sum += loss
        rMSE_sum += rMSE

        axs1['loss'].scatter(HpParams['random seeds'].index(seed), loss, label=str(seed))
        axs1['rMSE'].scatter(HpParams['random seeds'].index(seed), rMSE, label=str(seed))

    # fig1.suptitle('last models (bottom row) and best models (top row), labels are random seeds')
        
    axs1['loss'].set_title('Ave. testing loss: {}'.format(loss_sum/len(HpParams['random seeds'])))
    axs1['loss'].set_xlabel('model')
    axs1['loss'].set_ylabel('loss')
    axs1['loss'].legend()
    axs1['loss'].tick_params(bottom=False, labelbottom=False)

    axs1['rMSE'].set_title('Ave. testing rMSE: {}'.format(rMSE_sum/len(HpParams['random seeds'])))
    axs1['rMSE'].set_xlabel('model')
    axs1['rMSE'].set_ylabel('rMSE')
    axs1['rMSE'].legend()
    axs1['rMSE'].tick_params(bottom=False, labelbottom=False)
    
    # plt.show()
    fig1.savefig(os.path.join(root_path, f'm1_testing_curve.png'))
    plt.close()

if __name__ == '__main__':
    vis_testing_m1('debugging')
