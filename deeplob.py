# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLOB(nn.Module):
    def __init__(self):
        super(DeepLOB, self).__init__()
        self.lrelu = nn.LeakyReLU(0.01)
        # build the convolutional block
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2))
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2))
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10))
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv3_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))

        # build the inception module
        self.conv_incep_1_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))
        self.conv_incep_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1))

        self.conv_incep_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))
        self.conv_incep_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1))

        self.pool_incep = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))
        self.conv_incep_3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))

        # build the last LSTM layer
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1)

        # build the output layer
        self.fc = nn.Linear(64, 3)

    def forward(self, x):  #
        # conv part
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))
        x = self.conv1_3(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))
        x = self.conv2_3(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))
        x = self.conv3_3(x)
        x = self.lrelu(x)
        x = F.pad(x, pad=(0, 0, 3, 0))

        # inception part
        inception1 = self.conv_incep_1_1(x)
        inception1 = self.lrelu(inception1)
        inception1 = self.conv_incep_1_2(inception1)
        inception1 = self.lrelu(inception1)
        inception1 = F.pad(inception1, pad=(0, 0, 2, 0))

        inception2 = self.conv_incep_2_1(x)
        inception2 = self.lrelu(inception2)
        inception2 = self.conv_incep_2_2(inception2)
        inception2 = self.lrelu(inception2)
        inception2 = F.pad(inception2, pad=(0, 0, 4, 0))

        inception3 = self.pool_incep(x)
        inception3 = self.conv_incep_3_1(inception3)
        inception3 = self.lrelu(inception3)
        inception3 = F.pad(inception3, pad=(0, 0, 2, 0))

        inception = torch.cat((inception1, inception2, inception3), dim=1)
        # inception = inception.view(inception.size(0), inception.size(1), -1)
        inception = inception.squeeze()
        inception = inception.permute(2, 0, 1)

        # lstm
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(inception)
        output = output[-1, :, :]

        # fc
        output = self.fc(output)
        return output