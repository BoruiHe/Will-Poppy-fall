import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mydataset import mydataset
from torch.utils.data import DataLoader
from toy import toy_sparse, get_fls
import socket


model = toy_sparse()

checkpoint = torch.load('checkpoints\\toy_1.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

if socket.gethostname() == 'hbrDESKTOP':
    root_path = 'C:\\Users\\hbrch\\Desktop\\falling detection images'
elif socket.gethostname() == 'MSI':
    root_path = 'C:\\Users\\Borui He\\Desktop\\fdi'

dataset = mydataset(root_path)
loader = DataLoader(dataset, batch_size=1)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_running_loss, val_correct = 0, 0

for paths, val_labels in loader:

    val_data = get_fls(paths)
    val_data, val_labels = val_data.to(device), val_labels.to(device)
    val_outputs = model(val_data).squeeze()

    val_loss = loss_fn(val_outputs, val_labels.squeeze())
    val_running_loss += val_loss.item()
    if len(val_outputs.shape) == 1:
        val_correct += (val_outputs.softmax(dim=0).round() == val_labels.squeeze()).all().sum().item()
    else:
        val_correct += (val_outputs.softmax(dim=1).round() == val_labels).all(dim=1).sum().item()

print('test_ave_loss: {}, test_acc: {}'.format(val_running_loss/len(dataset), val_correct/len(dataset)*100))
pass