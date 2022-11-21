import os
import matplotlib.pyplot as plt
import pickle
import torch


with open('random seed.pkl', 'rb') as f:
    random_seeds = pickle.load(f)
    
# fig1, axs1 = plt.subplot_mosaic([[str(seed) for seed in random_seeds[:5]], [str(seed) for seed in random_seeds[5:]]], figsize=(18, 10))
# fig2, axs2 = plt.subplot_mosaic([[str(seed) for seed in random_seeds[:5]], [str(seed) for seed in random_seeds[5:]]], figsize=(18, 10))
# fig3, axs3 = plt.subplot_mosaic([[str(seed) for seed in random_seeds[:5]], [str(seed) for seed in random_seeds[5:]]], figsize=(18, 10))
# fig4, axs4 = plt.subplot_mosaic([[str(seed) for seed in random_seeds[:5]], [str(seed) for seed in random_seeds[5:]]], figsize=(18, 10))
fig1, axs1 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
fig2, axs2 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
fig3, axs3 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
fig4, axs4 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
fig5, axs5 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
fig6, axs6 = plt.subplot_mosaic([[str(seed) for seed in random_seeds], [str(seed) for seed in random_seeds]], figsize=(18, 10))
x = range(100)
checkpoint = []
for seed in random_seeds:
    train_acc_log, train_loss_log, val_loss_log, val_acc_log, train_wip, val_wip = [], [], [], [], [], []
    for model in ['toy_0.pth', 'toy_1.pth', 'toy_2.pth', 'toy_3.pth']:
        checkpoint = torch.load(os.path.join('checkpoints', str(seed), model), map_location=torch.device('cpu'))
        train_loss_log.append(checkpoint['train loss'])
        train_acc_log.append(checkpoint['train acc'])
        train_wip.append(checkpoint['train wrong ip'])
        val_loss_log.append(checkpoint['val loss'])
        val_acc_log.append(checkpoint['val acc'])
        val_wip.append(checkpoint['val wrong ip'])

    axs1[str(seed)].set_title('training loss')
    axs1[str(seed)].set_xlabel('epoch')
    axs1[str(seed)].plot(x, train_loss_log[0], color='red', label='fold 1')
    axs1[str(seed)].plot(x, train_loss_log[1], color='blue', label='fold 2')
    axs1[str(seed)].plot(x, train_loss_log[2], color='green', label='fold 3')
    axs1[str(seed)].plot(x, train_loss_log[3], color='yellow', label='fold 4')

    axs2[str(seed)].set_title('training accuracy')
    axs2[str(seed)].set_xlabel('epoch')
    axs2[str(seed)].plot(x, train_acc_log[0], color='red', label='fold 1')
    axs2[str(seed)].plot(x, train_acc_log[1], color='blue', label='fold 2')
    axs2[str(seed)].plot(x, train_acc_log[2], color='green', label='fold 3')
    axs2[str(seed)].plot(x, train_acc_log[3], color='yellow', label='fold 4')

    axs3[str(seed)].set_title('validation loss')
    axs3[str(seed)].set_xlabel('epoch')
    axs3[str(seed)].plot(x, val_loss_log[0], color='red', label='fold 1')
    axs3[str(seed)].plot(x, val_loss_log[1], color='blue', label='fold 2')
    axs3[str(seed)].plot(x, val_loss_log[2], color='green', label='fold 3')
    axs3[str(seed)].plot(x, val_loss_log[3], color='yellow', label='fold 4')

    axs4[str(seed)].set_title('validation accuracy')
    axs4[str(seed)].set_xlabel('epoch')
    axs4[str(seed)].plot(x, val_acc_log[0], color='red', label='fold 1')
    axs4[str(seed)].plot(x, val_acc_log[1], color='blue', label='fold 2')
    axs4[str(seed)].plot(x, val_acc_log[2], color='green', label='fold 3')
    axs4[str(seed)].plot(x, val_acc_log[3], color='yellow', label='fold 4')

    axs1[str(seed)].legend()
    axs2[str(seed)].legend()
    axs3[str(seed)].legend()
    axs4[str(seed)].legend()

    axs1[str(seed)].set_title(str(seed))
    axs2[str(seed)].set_title(str(seed))
    axs3[str(seed)].set_title(str(seed))
    axs4[str(seed)].set_title(str(seed))

fig1.suptitle('train loss', fontsize=16)
fig2.suptitle('train acc', fontsize=16)
fig3.suptitle('val loss', fontsize=16)
fig4.suptitle('val acc', fontsize=16)
fig1.savefig('C:\\Users\\hbrch\\OneDrive - Syracuse University\\falling detection\\checkpoints\\train loss.png')
fig2.savefig('C:\\Users\\hbrch\\OneDrive - Syracuse University\\falling detection\\checkpoints\\train acc.png')
fig3.savefig('C:\\Users\\hbrch\\OneDrive - Syracuse University\\falling detection\\checkpoints\\val loss.png')
fig4.savefig('C:\\Users\\hbrch\\OneDrive - Syracuse University\\falling detection\\checkpoints\\val acc.png')
plt.close('all')
