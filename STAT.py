import pickle
import os
import numpy as np


def analysis(cls_name):
    path = os.getcwd() + '\\data\\training\\' + cls_name
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
        x = np.array(x)
        ary.append(x)
    ary = np.array(ary)
    print(ary.shape)
    ave = np.average(ary, axis=0)
    std = np.std(ary, axis=0)
    # print('0-50:')
    # print(ave[:50])
    # print('50-150:')
    # print(ave[50:150])
    # print('150-300:')
    # print(ave[150:])

    return ave, std

v_ave, v_std = analysis('vibration')
f_ave, f_std = analysis('falling down')

if not 'statistics' in os.getcwd():
    path = os.getcwd() + '\\' + 'statistics'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '\\v_ave.pkl', 'wb') as pf:
        pickle.dump((v_ave, v_std), pf)
    with open(path + '\\f_ave.pkl', 'wb') as pf:
        pickle.dump((f_ave, f_std), pf)
