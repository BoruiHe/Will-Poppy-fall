import os
import torch
import matplotlib.pyplot as plt
import yaml


def vis_testing(checkpoint_folder_name, mode):
    rp = os.getcwd()

    if mode == 'classification':
        with open(os.path.join(rp, 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
            HpParams = yaml.safe_load(file)
        
        root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)

        for seed in HpParams['random seeds']:

            path = os.path.join(root_path, str(seed))
            
            testb = torch.load(os.path.join(path, 'test_b.pth')), torch.load(os.path.join(path, 'test_t.pth'))
            testb_loss_log = testb['test loss']
            # testt_loss_log = testt['test acc']
            
            fig, axs = plt.subplots(1,2, tight_layout= True)
            fig.suptitle('testing results')
            
            axs[0].set_title('loss {}'.format(sum(testt_loss_log)/len(testt_loss_log)))
            axs[0].set_xlabel('last models')
            axs[0].set_ylabel('loss')
            for xx, y in zip(range(1, len(testt_loss_log)+1), testt_loss_log):
                axs[0].plot(xx, y, 'ro')
            

            axs[1].set_title('loss {}'.format(sum(testb_loss_log)/len(testb_loss_log)))
            axs[1].set_xlabel('best models')
            axs[1].set_ylabel('loss')
            for xx, y in zip(range(1, len(testb_loss_log)+1), testb_loss_log):
                axs[1].plot(xx, y, 'ro')
            
            
            fig.canvas.manager.set_window_title(str(seed))
            # fig.canvas.manager.window.state('zoomed')

            # plt.show()
            fig.savefig(os.path.join(root_path, 'testing results {}.png'.format(seed)))
            plt.close()

    elif mode == 'regression':
        with open(os.path.join(rp, 'checkpoints', checkpoint_folder_name, 'HyperParam.yml'), 'r') as file:
            HpParams = yaml.safe_load(file)
        
        root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_folder_name)

        for seed in HpParams['random seeds']:

            path = os.path.join(root_path, str(seed))
            
            testb, testt = torch.load(os.path.join(path, 'test_b.pth')), torch.load(os.path.join(path, 'test_t.pth'))
            testb_loss_log = testb['test loss']
            testt_loss_log = testt['test loss']
            
            fig, axs = plt.subplots(1,2, tight_layout= True)
            fig.suptitle('testing results')
            
            axs[0].set_title('loss {}'.format(sum(testt_loss_log)/len(testt_loss_log)))
            axs[0].set_xlabel('last models')
            axs[0].set_ylabel('loss')
            for xx, y in zip(range(1, len(testt_loss_log)+1), testt_loss_log):
                axs[0].plot(xx, y, 'ro')
            

            axs[1].set_title('loss {}'.format(sum(testb_loss_log)/len(testb_loss_log)))
            axs[1].set_xlabel('best models')
            axs[1].set_ylabel('loss')
            for xx, y in zip(range(1, len(testb_loss_log)+1), testb_loss_log):
                axs[1].plot(xx, y, 'ro')
            
            
            fig.canvas.manager.set_window_title(str(seed))

            # plt.show()
            fig.savefig(os.path.join(root_path, 'testing results {}.png'.format(seed)))
            plt.close()
    
    else:
        raise Exception('wrong testing mode!')

if __name__ == '__main__':
    vis_testing('ps2_ep200_R_Sh_diff', 'regression')