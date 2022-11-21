import os
import matplotlib.pyplot as plt
import pickle
import torch


with open('random seed.pkl', 'rb') as f:
    random_seeds = pickle.load(f)

checkpoint = []
for seed in random_seeds:
    train_wip, val_wip = [], []
    for model in ['toy_0.pth', 'toy_1.pth', 'toy_2.pth', 'toy_3.pth']:
        checkpoint = torch.load(os.path.join('checkpoints', str(seed), model), map_location=torch.device('cpu'))
        val_wip += checkpoint['val wrong ip']
    vrec = []
    stat = {}
    for p in val_wip[-20:]:
        vrec += list(zip(p[0].tolist(), p[1].tolist()))
    for rec in vrec:
        # if anchor is new
        if not rec[0] in list(stat.keys()):
            # if it is wrong prediction
            if not rec[1]:
                stat[rec[0]] = [1, 1]
            # if it is correct prediction
            else:
                stat[rec[0]] = [0, 1]
        else:
            # if it is wrong prediction
            if not rec[1]:
                stat[rec[0]][0] += 1               
            stat[rec[0]][1] += 1
    for k in list(stat.keys()):
        stat[k] = stat[k][0] / stat[k][1]
    tup = [(k, stat[k]) for k in stat]
    tip = sorted(tup, key=lambda x: x[0])
    x, y = [], []
    for r in tip:
        x.append(r[0])
        y.append(r[1])
    fig = plt.figure()
    fig.suptitle('{}: wrong prediction rate'.format(seed), fontsize=16)
    plt.plot(x,y)
    plt.xlabel('anchor index')
    plt.ylabel('wrong prediction rate')
    plt.show()
    fig.savefig('C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\checkpoints\\{}_wpr.png'.format(seed))
    plt.close('all')
