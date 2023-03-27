import os
import torch
import matplotlib.pyplot as plt
import yaml


def vis_tr(checkpoint_folder_name):
    
    rp = os.getcwd()
    with open(os.path.join(rp, 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    fs = HpParams['random seeds']
    x = range(0, HpParams['epochs'])

    fig1, axs1 = plt.subplot_mosaic([[str(seed) for seed in fs], ['emp1', 'emp2', 'emp3']], tight_layout= True, figsize=(12,9))
    fig1.canvas.manager.set_window_title('Training Loss')

    # fig2, axs2 = plt.subplot_mosaic([[str(seed) for seed in fs], ['emp1', 'emp2', 'emp3']], tight_layout= True, figsize=(12,9))
    # fig2.canvas.manager.set_window_title('Training Acc')

    fig3, axs3 = plt.subplot_mosaic([[str(seed) for seed in fs], ['emp1', 'emp2', 'emp3']], tight_layout= True, figsize=(12,9))
    fig3.canvas.manager.set_window_title('Validation Loss')

    # fig4, axs4 = plt.subplot_mosaic([[str(seed) for seed in fs], ['emp1', 'emp2', 'emp3']], tight_layout= True, figsize=(12,9))
    # fig4.canvas.manager.set_window_title('Validation Acc')

    # fig5, axs5 = plt.subplot_mosaic([[str(seed) + '_F' for seed in fs], [str(seed) + '_Fv' for seed in fs], [str(seed) + '_V' for seed in fs]], tight_layout= True, figsize=(12,9))
    # fig5.canvas.manager.set_window_title('Categorical tn Acc')

    # fig6, axs6 = plt.subplot_mosaic([[str(seed) + '_F' for seed in fs], [str(seed) + '_Fv' for seed in fs], [str(seed) + '_V' for seed in fs]], tight_layout= True, figsize=(12,9))
    # fig6.canvas.manager.set_window_title('Categorical val Acc')


    for cp in fs:
        cp = str(cp)
        path = os.path.join(rp, 'checkpoints', checkpoint_folder_name, cp)
        
        train_loss_log, val_loss_log = [], []
        # train_acc_log, val_acc_log = [], []
        # tn_F_acc, tn_Fv_acc, tn_V_acc, val_F_acc, val_Fv_acc, val_V_acc = [], [], [], [], [], []
        for model in ['toy_0.pth', 'toy_1.pth', 'toy_2.pth', 'toy_3.pth']:
            
            checkpoint_path = os.path.join(path, model)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            train_loss_log.append(checkpoint['train loss'])
            # train_acc_log.append(checkpoint['train acc'])
            val_loss_log.append(checkpoint['val loss'])
            # val_acc_log.append(checkpoint['val acc'])
            # tn_F_acc.append(checkpoint['tn_F_acc'])
            # tn_Fv_acc.append(checkpoint['tn_Fv_acc'])
            # tn_V_acc.append(checkpoint['tn_V_acc'])
            # val_F_acc.append(checkpoint['val_F_acc'])
            # val_Fv_acc.append(checkpoint['val_Fv_acc'])
            # val_V_acc.append(checkpoint['val_V_acc'])

        
        axs1[cp].set_title(cp)
        axs1[cp].set_xlabel('epoch')
        axs1[cp].plot(x, train_loss_log[0], color='r', label='fold 1')
        axs1[cp].plot(x, train_loss_log[1], color='b', label='fold 2')
        axs1[cp].plot(x, train_loss_log[2], color='g', label='fold 3')
        axs1[cp].plot(x, train_loss_log[3], color='y', label='fold 4')

        # axs2[cp].set_title(cp)
        # axs2[cp].set_xlabel('epoch')
        # axs2[cp].plot(x, train_acc_log[0], color='r', label='fold 1')
        # axs2[cp].plot(x, train_acc_log[1], color='b', label='fold 2')
        # axs2[cp].plot(x, train_acc_log[2], color='g', label='fold 3')
        # axs2[cp].plot(x, train_acc_log[3], color='y', label='fold 4')

        axs3[cp].set_title(cp)
        axs3[cp].set_xlabel('epoch')
        axs3[cp].plot(x, val_loss_log[0], color='r', label='fold 1')
        axs3[cp].plot(x, val_loss_log[1], color='b', label='fold 2')
        axs3[cp].plot(x, val_loss_log[2], color='g', label='fold 3')
        axs3[cp].plot(x, val_loss_log[3], color='y', label='fold 4')

        # axs4[cp].set_title(cp)
        # axs4[cp].set_xlabel('epoch')
        # axs4[cp].plot(x, val_acc_log[0], color='r', label='fold 1')
        # axs4[cp].plot(x, val_acc_log[1], color='b', label='fold 2')
        # axs4[cp].plot(x, val_acc_log[2], color='g', label='fold 3')
        # axs4[cp].plot(x, val_acc_log[3], color='y', label='fold 4')

        # axs5[cp + '_F'].set_title(cp + '_F')
        # axs5[cp + '_F'].set_xlabel('epoch')
        # axs5[cp + '_F'].plot(x, tn_F_acc[0], color='r', label='fold 1')
        # axs5[cp + '_F'].plot(x, tn_F_acc[1], color='b', label='fold 2')
        # axs5[cp + '_F'].plot(x, tn_F_acc[2], color='g', label='fold 3')
        # axs5[cp + '_F'].plot(x, tn_F_acc[3], color='y', label='fold 4')
        # axs5[cp + '_Fv'].set_title(cp+'_Fv')
        # axs5[cp + '_Fv'].set_xlabel('epoch')
        # axs5[cp + '_Fv'].plot(x, tn_Fv_acc[0], color='r', label='fold 1')
        # axs5[cp + '_Fv'].plot(x, tn_Fv_acc[1], color='b', label='fold 2')
        # axs5[cp + '_Fv'].plot(x, tn_Fv_acc[2], color='g', label='fold 3')
        # axs5[cp + '_Fv'].plot(x, tn_Fv_acc[3], color='y', label='fold 4')
        # axs5[cp + '_V'].set_title(cp+'_V')
        # axs5[cp + '_V'].set_xlabel('epoch')
        # axs5[cp + '_V'].plot(x, tn_V_acc[0], color='r', label='fold 1')
        # axs5[cp + '_V'].plot(x, tn_V_acc[1], color='b', label='fold 2')
        # axs5[cp + '_V'].plot(x, tn_V_acc[2], color='g', label='fold 3')
        # axs5[cp + '_V'].plot(x, tn_V_acc[3], color='y', label='fold 4')

        # axs6[cp + '_F'].set_title(cp + '_F')
        # axs6[cp + '_F'].set_xlabel('epoch')
        # axs6[cp + '_F'].plot(x, val_F_acc[0], color='r', label='fold 1')
        # axs6[cp + '_F'].plot(x, val_F_acc[1], color='b', label='fold 2')
        # axs6[cp + '_F'].plot(x, val_F_acc[2], color='g', label='fold 3')
        # axs6[cp + '_F'].plot(x, val_F_acc[3], color='y', label='fold 4')
        # axs6[cp + '_Fv'].set_title(cp+'_Fv')
        # axs6[cp + '_Fv'].set_xlabel('epoch')
        # axs6[cp + '_Fv'].plot(x, val_Fv_acc[0], color='r', label='fold 1')
        # axs6[cp + '_Fv'].plot(x, val_Fv_acc[1], color='b', label='fold 2')
        # axs6[cp + '_Fv'].plot(x, val_Fv_acc[2], color='g', label='fold 3')
        # axs6[cp + '_Fv'].plot(x, val_Fv_acc[3], color='y', label='fold 4')
        # axs6[cp + '_V'].set_title(cp+'_V')
        # axs6[cp + '_V'].set_xlabel('epoch')
        # axs6[cp + '_V'].plot(x, val_V_acc[0], color='r', label='fold 1')
        # axs6[cp + '_V'].plot(x, val_V_acc[1], color='b', label='fold 2')
        # axs6[cp + '_V'].plot(x, val_V_acc[2], color='g', label='fold 3')
        # axs6[cp + '_V'].plot(x, val_V_acc[3], color='y', label='fold 4')

        axs1[cp].legend()
        # axs2[cp].legend()
        axs3[cp].legend()
        # axs4[cp].legend()
        # axs5[cp + '_F'].legend()
        # axs5[cp + '_Fv'].legend()
        # axs5[cp + '_V'].legend()
        # axs6[cp + '_F'].legend()
        # axs6[cp + '_Fv'].legend()
        # axs6[cp + '_V'].legend()
        
    # plt.show()

    save_path = os.path.join(rp, 'checkpoints', checkpoint_folder_name)
    fig1.savefig(os.path.join(save_path, 'train loss.png'))
    # fig2.savefig(save_path + '\\train acc.png')
    fig3.savefig(os.path.join(save_path, 'val loss.png'))
    # fig4.savefig(save_path + '\\val acc.png')
    # fig5.savefig(save_path + '\\Categorical training acc.png')
    # fig6.savefig(save_path + '\\Categorical validation acc.png')


if __name__ == '__main__':
    vis_tr('ps2_ep200_R_Sh_diff')