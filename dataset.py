# coding=utf8
import numpy as np
import torch
from torch.utils.data import Dataset
# from keras.utils import np_utils

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape  # [254750, 40]
    df = np.array(X)
    dY = np.array(Y)
    # 每100次订单薄数据拼接起来作为输入
    dataX = np.zeros((N - T + 1, T, D))  # (254651, 100, 40)
    dataY = dY[T - 1:N]  # (254651, 5)
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

class LOBDataset(Dataset):
    def __init__(self, split):
        data_path = '/p300/FinData/F12010'
        if split == 'train':
            print('loading train data...')
            dec_train = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')

            # 20档买卖深度，价格与量
            train_lob = prepare_x(dec_train)
            # print(train_lob.shape)  # (254750, 40)
            train_label = get_label(dec_train)
            # print(train_label.shape)  # (254750, 5)

            trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=100)
            # print(trainX_CNN.shape, trainY_CNN.shape)  # (254651, 100, 40, 1) (254651, 5)
            trainY_CNN = trainY_CNN[:, 3] - 1
            # print(trainY_CNN.shape)  # (254651,)
            # print(trainY_CNN.max(), trainY_CNN.min())  # 2, 0
            # trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)
            # print(trainY_CNN.shape)  # (254651, 3)
            self.lob, self.label = torch.from_numpy(trainX_CNN), torch.from_numpy(trainY_CNN).long()
            # print(trainX_CNN.size(), trainY_CNN.size())  # torch.Size([254651, 100, 40, 1]) torch.Size([254651, 3])
            self.lob = self.lob.permute(0, 3, 1, 2).float()  # torch.Size([254651, 1, 100, 40])
        elif split == 'test':
            print('loading test data...')
            dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
            dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
            dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
            dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

            test_lob = prepare_x(dec_test)  # (139587, 40)
            test_label = get_label(dec_test)  # (139587, 5)

            testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=100)
            testY_CNN = testY_CNN[:, 3] - 1
            # testY_CNN = np_utils.to_categorical(testY_CNN, 3)
            self.lob, self.label = torch.from_numpy(testX_CNN), torch.from_numpy(testY_CNN).long()
            self.lob = self.lob.permute(0, 3, 1, 2).float()

    def __getitem__(self, index):
        lob = self.lob[index, :, :, :]  # [1, 100, 40]
        label = self.label[index]
        return lob, label

    def __len__(self):
        return self.lob.size(0)
