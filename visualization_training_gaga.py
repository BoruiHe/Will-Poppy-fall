import os
import torch
import matplotlib.pyplot as plt
import yaml
import pickle as pk


def vis_training_m1(checkpoint_folder_name):
    
    rp = os.getcwd()
    with open(os.path.join(rp, 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    x = range(0, HpParams['epochs'])


    fig1, axs1 = plt.subplot_mosaic([['Training'], ['Validation']], tight_layout= True, figsize=(12,9))
    
    for seed in HpParams['random seeds']:

        path = os.path.join(rp, 'checkpoints', checkpoint_folder_name, str(seed))

        checkpoint_path = os.path.join(path, 'm1_best.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        axs1['Training'].plot(x, checkpoint['train loss'], label=str(seed))
        axs1['Validation'].plot(x, checkpoint['val loss'], label=str(seed))

    axs1['Training'].set_title('Training loss')
    axs1['Training'].set_xlabel('epoch')
    axs1['Training'].set_ylabel('loss')
    axs1['Training'].legend()
    axs1['Validation'].set_title('Validation loss')
    axs1['Validation'].set_xlabel('epoch')
    axs1['Validation'].set_ylabel('loss')
    axs1['Validation'].legend()

    save_path = os.path.join(rp, 'checkpoints', checkpoint_folder_name)
    fig1.savefig(os.path.join(save_path, 'm1_learning_curve.png'))
        
    plt.close()

def vis_training_m2(checkpoint_folder_name):
    
    rp = os.getcwd()
    with open(os.path.join(rp, 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    x = range(0, HpParams['epochs'])

    fig1, axs1 = plt.subplot_mosaic([['Training'], ['Validation']], tight_layout= True, figsize=(12,9))
    fig2, axs2 = plt.subplot_mosaic([['Training'], ['Validation']], tight_layout= True, figsize=(12,9))

    for seed in HpParams['random seeds']:

        path = os.path.join(rp, 'checkpoints', checkpoint_folder_name, str(seed))

        # checkpoint_path = os.path.join(path, 'm1_last.pth')
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # axs1['Training'].plot(x, checkpoint['train loss'], label=str(seed))
        # axs1['Validation'].plot(x, checkpoint['val loss'], label=str(seed))

        checkpoint_path = os.path.join(path, 'm2.pkl')
        checkpoint = pk.load(open(checkpoint_path, 'rb'))
        
        axs2['Training'].scatter(HpParams['random seeds'].index(seed), checkpoint['train acc'], label=str(seed))
        axs2['Validation'].scatter(HpParams['random seeds'].index(seed), checkpoint['val acc'], label=str(seed))

    # axs1['Training'].set_title('Training loss')
    # axs1['Training'].set_xlabel('epoch')
    # axs1['Training'].set_ylabel('loss')
    # axs1['Training'].legend()
    # axs1['Validation'].set_title('Validation loss')
    # axs1['Validation'].set_xlabel('epoch')
    # axs1['Validation'].set_ylabel('loss')
    # axs1['Validation'].legend()

    save_path = os.path.join(rp, 'checkpoints', checkpoint_folder_name)
    # fig1.savefig(os.path.join(save_path, f'train&val loss m2.png'))

    axs2['Training'].set_title('Training accuracy')
    axs2['Training'].set_xlabel('random seeds')
    axs2['Training'].set_ylabel('accuracy')
    axs2['Training'].legend()
    axs2['Validation'].set_title('Validation accuracy')
    axs2['Validation'].set_xlabel('random seeds')
    axs2['Validation'].set_ylabel('accuracy')
    axs2['Validation'].legend()
    fig2.savefig(os.path.join(save_path, 'train&val acc m2.png'))
        
    plt.close()


if __name__ == '__main__':
    vis_training_m1('debugging')