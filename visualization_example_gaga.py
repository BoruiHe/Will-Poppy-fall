import matplotlib.pyplot as plt
import os
import yaml
import pickle as pk
import pandas as pd
import glob
from PIL import Image
import numpy as np

def vis_example_m1(checkpoint_name):
    root_path = os.path.join(os.getcwd(), 'checkpoints', checkpoint_name)
    with open(os.path.join(root_path, 'HyperParam.yml'), 'r') as file:
        HpParams = yaml.safe_load(file)
    
    # virtual poppy dataset
    if HpParams['dataset_name'] == 'vir_poppy':
        for seed in HpParams['random seeds']:
            path = os.path.join(root_path, 'example_testing_m1', str(seed))
            video_list = pd.read_hdf(os.path.join(root_path, str(seed), 'videos.h5'))

            for example in os.listdir(path):
                recon_vc, vc_start_idx = pk.load(open(os.path.join(path, example), 'rb'))
                clss = video_list[video_list['folder'] == example[:-4]]['class'].item()
                ori_path = os.path.join(os.getcwd(), 'virtual_poppy', clss, example[:-4])
                sp_anchor = len(glob.glob(os.path.join(ori_path, '*.png')))-1 # stop anchor is the highest index of images of this full video
                frames = []
                anchor = HpParams['anchor']
                while anchor <= sp_anchor:
                    img = Image.open(os.path.join(ori_path, '{}.png'.format(anchor)))
                    img.load()
                    img = np.asarray(img.convert('RGB'), dtype='uint8')
                    frames.append(img)
                    anchor += HpParams['dilation']

                recon_vc = np.squeeze(recon_vc.transpose(0,2,3,4,1))
                for i in range(recon_vc.shape[0]):
                    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,9), layout='tight')
                    fig.suptitle(f'{example}, {clss},  frame: {vc_start_idx + i}/{vc_start_idx + recon_vc.shape[0]}') # VC: {idx+1}/{len(target)},
                    ax1.imshow(frames[vc_start_idx + 1])
                    ax1.set_title('Original')
                    ax1.axis('off')
                    ax2.imshow(recon_vc[i][:, :, ::-1])
                    ax2.set_title('Reconstructed')
                    ax2.axis('off')
                    plt.show(block=True)
                    plt.close()
    # real poppy dataset
    elif HpParams['dataset_name'] == 'real_poppy':
        for seed in HpParams['random seeds']:
            path = os.path.join(root_path, 'example_testing_m1', str(seed))

            for example in os.listdir(path):
                recon_vc, vc_start_idx = pk.load(open(os.path.join(path, example), 'rb'))
                ori_path = os.path.join(os.getcwd(), 'real_poppy', example)

                temp = []
                with open(ori_path, 'rb') as f:
                    frames = pk.load(f, encoding='latin1')
                for k in range(len(frames)):
                    temp.append(np.concatenate(np.stack(frames[k])))
                temp = np.concatenate(temp)

                with open(os.path.join(os.getcwd(), 'real_poppy', 'results_' + example[7:]), 'rb') as f:
                    (_, _, bufs) = pk.load(f, encoding='latin1') # motor_names, results, bufs
                
                elapsed_rep = []
                for k in range(5):
                    _, _, elapsed = zip(*bufs[k]) # (flag, buffers, elapsed)
                    # accumulate elapsed time over multiple waypoints
                    for i in range(1, len(elapsed)):
                        elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                    elapsed = np.concatenate(elapsed)
                    elapsed_rep.append(elapsed)
                for i in range(1, len(elapsed_rep)):
                    elapsed_rep[i][:] = elapsed_rep[i] + elapsed_rep[i-1][-1]
                elapsed_rep = np.concatenate(elapsed_rep)
                interval = elapsed_rep[-1]/elapsed_rep.shape[0]
                idx = [interval * j * HpParams['dilation'] for j in [vc_start_idx + i for i in range(10)]]
                upper_bound_idx = np.searchsorted(elapsed_rep, idx)
                lower_bound_idx = upper_bound_idx -1
                recon_vc = recon_vc.squeeze().transpose(1,2,3,0)
                for i in range(recon_vc.shape[0]):
                    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,9), layout='tight')
                    fig.suptitle(f'{example}, frame: {vc_start_idx + i}/{vc_start_idx + recon_vc.shape[0]}') # VC: {idx+1}/{len(last)}, 
                    ax1.imshow(temp[lower_bound_idx[i]][:, :, ::-1])
                    ax1.set_title('Original: lower')
                    ax1.axis('off')
                    ax2.imshow(temp[upper_bound_idx[i]][:, :, ::-1])
                    ax2.set_title('Original: higher')
                    ax2.axis('off')
                    ax3.imshow(recon_vc[i][:, :, ::-1])
                    ax3.set_title('Reconstructed')
                    ax3.axis('off')
                    plt.show(block=True)
                    plt.close()

if __name__ == '__main__':
    vis_example_m1('debugging')