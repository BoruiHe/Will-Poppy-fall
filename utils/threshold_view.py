from matplotlib import pyplot as plt
import os
import pickle as pk
import numpy as np
import math
import torchvision
import torch
from torchvision import transforms as t


def view_virtual():
    indexes = [50 + i * 8 for i in range(math.ceil(250/8))]
    n_bins = len(indexes) - 1
    fall, standing = {}, {}
    fs = 20
    for i in range(n_bins):
        fall[str(i)] = []
        standing[str(i)] = []

    # fall
    path = os.path.join(os.getcwd(), 'virtual_poppy', 'fall')
    folders = os.listdir(path)
    for folder in folders:
        with open(os.path.join(path, folder, 'hc_z.pkl'), 'rb') as f:
            hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
            hc_z_coordinates = np.array(hc_z_coordinates)[indexes]

        delta_z = (np.absolute(hc_z_coordinates[1:] - hc_z_coordinates[:-1]))
        for i in range(len(indexes)-1):
            fall[str(i)].append(delta_z[i])

    # standing
    maximum = 0
    path = os.path.join(os.getcwd(), 'virtual_poppy', 'standing')
    folders = os.listdir(path)
    for folder in folders:
        with open(os.path.join(path, folder, 'hc_z.pkl'), 'rb') as f:
            hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
            hc_z_coordinates = np.array(hc_z_coordinates)[indexes]

        delta_z = (np.absolute(hc_z_coordinates[1:] - hc_z_coordinates[:-1]))
        if delta_z.max() > maximum:
            maximum = delta_z.max()
        for i in range(len(indexes)-1):
            standing[str(i)].append(delta_z[i])
    print(maximum)

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    plt.tight_layout()
    axs[0].plot(range(31), np.array(list(fall.values())).mean(axis=1), 'o-', label='fall')
    axs[0].fill_between(
        range(31),
        np.array(list(fall.values())).mean(axis=1) - np.array(list(fall.values())).std(axis=1),
        np.array(list(fall.values())).mean(axis=1) + np.array(list(fall.values())).std(axis=1),
        alpha=0.3,
    )
    axs[0].plot(range(31), np.array(list(standing.values())).mean(axis=1), 'o-', label='standing')
    axs[0].fill_between(
        range(31),
        np.array(list(standing.values())).mean(axis=1) - np.array(list(standing.values())).std(axis=1),
        np.array(list(standing.values())).mean(axis=1) + np.array(list(standing.values())).std(axis=1),
        alpha=0.3,
    )
    axs[0].axhline(y= maximum, color = 'r', linestyle = 'dashed', label='maximum')
    axs[0].set_xticklabels([1, 5, 10, 15, 20, 25, 30], fontsize=fs)
    axs[0].set_yticklabels([0.00, 0.02, 0.04, 0.06, 0.08, 0.10], fontsize=fs)
    axs[0].set_xlabel('timestep', fontsize=fs)
    axs[0].set_ylabel(r'$\Delta_{z} = \|z_{i} - z_{i+1}\|$', fontsize=fs)
    axs[0].legend(fontsize=fs)
    axs[0].set_ylim([0., 0.1])
    axs[0].spines[['right', 'top']].set_visible(False)

    axs[1].plot(range(31), np.array(list(fall.values())).mean(axis=1), 'o-', label='fall')
    axs[1].fill_between(
        range(31),
        np.array(list(fall.values())).mean(axis=1) - np.array(list(fall.values())).std(axis=1),
        np.array(list(fall.values())).mean(axis=1) + np.array(list(fall.values())).std(axis=1),
        alpha=0.3,
    )
    axs[1].plot(range(31), np.array(list(standing.values())).mean(axis=1), 'o-', label='standing')
    axs[1].fill_between(
        range(31),
        np.array(list(standing.values())).mean(axis=1) - np.array(list(standing.values())).std(axis=1),
        np.array(list(standing.values())).mean(axis=1) + np.array(list(standing.values())).std(axis=1),
        alpha=0.3,
    )
    axs[1].axhline(y= maximum, color = 'r', linestyle = 'dashed', label='maximum')
    axs[1].set_xticklabels([1, 5, 10, 15, 20, 25, 30], fontsize=fs)
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1].set_yticklabels([0.000, 0.004, 0.008, 0.012, 0.016, 0.002], fontsize=fs)
    axs[1].set_xlabel('timestep', fontsize=fs)
    axs[1].set_ylabel(r'$\Delta_{z} = \|z_{i} - z_{i+1}\|$', fontsize=fs)
    axs[1].legend(fontsize=fs)
    axs[1].set_ylim([0., 0.0020])
    axs[1].spines[['right', 'top']].set_visible(False)

    plt.show(block=True)
    plt.close()

def view_3rd():
    fall_path = [os.path.join(os.getcwd(), '3rd_party_dataset', env, 'falls') for env in ['Indoor', 'Outdoor']]
    nonfall_path = [os.path.join(os.getcwd(), '3rd_party_dataset', env, 'nonfalls') for env in ['Indoor', 'Outdoor']]
    fall, nonfall = {}, {}
    for i in range(10):
        fall[i] = []
        nonfall[i] = []
        
    maximum = 0 # 186.8786 and 187.0916
    for nfp in nonfall_path:
        for path, _, files in os.walk(nfp):
            for file in files:
                file_path = os.path.join(path, file)
                frames, _, _ = torchvision.io.read_video(file_path, output_format='TCHW')
                indexes = np.linspace(0, frames.shape[0], 10, endpoint=False)
                pixel_difference = (t.Resize(size=(96,128))(frames[indexes + 1]) - t.Resize(size=(96,128))(frames[indexes])) / 255
                for i in range(pixel_difference.shape[0]):
                    norm = torch.norm(pixel_difference[i])
                    nonfall[i].append(norm)
                    if norm > maximum:
                        maximum = norm

    for fp in fall_path:
        for path, _, files in os.walk(fp):
            for file in files:
                file_path = os.path.join(path, file)
                frames, _, _ = torchvision.io.read_video(file_path, output_format='TCHW')
                indexes = np.linspace(0, frames.shape[0], 10, endpoint=False)
                pixel_difference = (t.Resize(size=(96,128))(frames[indexes + 1]) - t.Resize(size=(96,128))(frames[indexes])) / 255

                for i in range(pixel_difference.shape[0]):
                    fall[i].append(torch.norm(pixel_difference[i]))

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.plot(range(10), np.array(list(fall.values())).mean(axis=1), 'o-', label='fall')
    axs.fill_between(
        range(10),
        np.array(list(fall.values())).mean(axis=1) - np.array(list(fall.values())).std(axis=1),
        np.array(list(fall.values())).mean(axis=1) + np.array(list(fall.values())).std(axis=1),
        alpha=0.3,
    )
    axs.plot(range(10), np.array(list(nonfall.values())).mean(axis=1), 'o-', label='nonfall')
    axs.fill_between(
        range(10),
        np.array(list(nonfall.values())).mean(axis=1) - np.array(list(nonfall.values())).std(axis=1),
        np.array(list(nonfall.values())).mean(axis=1) + np.array(list(nonfall.values())).std(axis=1),
        alpha=0.3,
    )
    axs.axhline(y= maximum, color = 'r', linestyle = 'dashed', label='maximum')
    axs.set_xlabel('timestep')
    axs.set_ylabel(r'$\sqrt{\Sigma(z_{i+1} - z_{i})^2}$')
    axs.legend(prop={'size': 10})
    plt.show(block=True)
    plt.close()
    print(maximum)
                
if __name__ == '__main__':
    view_virtual()
