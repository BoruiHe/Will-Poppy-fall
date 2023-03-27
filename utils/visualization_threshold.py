import matplotlib.pyplot as plt
import pickle
import os
import numpy as np



# path

root_path = 'C:\\Users\\hbrch\\Desktop\\fdi_fl\\data'

def boxplot(cls_name, p):
    path = p + '\\' + cls_name
    fl = os.listdir(path)

    if 'visualization' in fl:
        fl.remove('visualization')
    else:
        os.makedirs(path + '\\' + 'visualization')
    
    ary = []
    for fd in fl:
        with open(path + '\\' + fd + '\\hc_z.pkl', 'rb') as f:
            x = pickle.load(f)
        x = [c[2] for c in x]
        x = np.array(x[50:])
        ary.append(x)
    ary = np.array(ary)
    print(ary.shape)

    # Each column is one timestep in each long video, (50, 300); each row refers to one long video; data is the z coordiantes of head camera.
    states = np.array([[ary[:,i].mean() - 3*ary[:,i].std(), ary[:,i].mean(), ary[:,i].mean() + 3*ary[:,i].std()] for i in range(ary.shape[1])])

    x_axis = np.array(range(50, 300))
    # plt.scatter(x_axis, states[:, 0], color='red', label='min', s=2)
    plt.scatter(x_axis, states[:, 0], color='purple', label='m-3std', s=2)
    plt.scatter(x_axis, states[:, 1], color='black', label='mean', s=2)
    plt.scatter(x_axis, states[:, 2], color='blue', label='m+3std', s=2)
    # plt.scatter(x_axis, states[:, 4], color='purple', label='max', s=2)
    plt.legend()
    plt.title(cls_name + ': head camera, z coordinate')
    plt.savefig(p + '\\' + cls_name + '\\visualization\\head camera, z coordinate')
    # plt.show()
    plt.close()

def z_axis(cls_name, p):
    path = p + '\\' + cls_name
    fl = os.listdir(path)

    if 'visualization' in fl:
        fl.remove('visualization')
    else:
        os.makedirs(path + '\\visualization')
        
    for fd in fl:
        with open(path + '\\' + fd + '\\hc_z.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(path + '\\' + fd + '\\base_z.pkl', 'rb') as f:
            y = pickle.load(f)
        x = np.array([c[2] for c in x])
        y = [c[2] for c in y]

        x_axis = np.array(range(300))
        plt.plot(x_axis, x, label= 'head camera')
        plt.plot(x_axis, y, label= 'base')
        plt.legend()
        plt.title(cls_name + ': z coordinates')
        plt.get_current_fig_manager().canvas.set_window_title(fd)
        plt.savefig(p + '\\' + cls_name + '\\' + 'visualization' + '\\' + fd + '.png')
        # plt.show()
        plt.close()

if __name__ == '__main__':
    z_axis('vibration', root_path)
    boxplot('vibration', root_path)
    # z_axis('falling down', root_path)
    # boxplot('falling down', root_path)
    pass
