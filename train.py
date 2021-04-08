# coding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import random

from dataset import LOBDataset
from deeplob import DeepLOB

'''data'''
dataset_train = LOBDataset(split='train')
dataloader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=False)
# dataset_test = LOBDataset(split='test')
# dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

'''model'''
model = DeepLOB()
model = nn.DataParallel(model)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(reduction='mean')

epochs = 200
'''epoch'''
for epoch in range(epochs):
    print('train epoch', epoch)
    '''batch'''
    for i, (lob, label) in enumerate(dataloader_train):
        lob, label = lob.cuda(), label.cuda()
        '''forward'''
        pred = model(lob)

        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 30 == 0:
            print(loss)


