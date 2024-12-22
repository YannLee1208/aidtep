import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchinfo import summary


class DilResNet(nn.Module):
    def __init__(self,R_dic):
        super(DilResNet, self).__init__()

        in_dim = R_dic['in_dim']
        feature_dim = R_dic['feature_dim']
        self.num_layers = R_dic['num_layers']
        self.y_norm = R_dic['y_norm'] if 'y_norm' in R_dic else None

        # Encoder CNN
        self.encoder_conv = nn.Conv2d(in_channels=in_dim, out_channels=feature_dim, kernel_size=3, padding=1)

        # Processor
        self.processor = nn.ModuleList([])
        for i in range(self.num_layers):
            self.processor.append(dCNN(feature_dim))

        # Decoder CNN
        self.decoder_conv = nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        # Process with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.processor[i](x)
            x = x + identity
            x = F.relu(x)

        # Decode
        x = self.decoder_conv(x)
        x = x.permute(0, 2, 3, 1)

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)

        return torch.squeeze(x)


class dCNN(nn.Module):
    def __init__(self, feature_dim=48):
        super(dCNN, self).__init__()

        # Dilation rates
        dilation_rates = [1, 2, 4, 8, 4, 2, 1]

        # 7 dilated CNN layers
        self.conv1 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[0],
                               padding=dilation_rates[0])
        self.conv2 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[1],
                               padding=dilation_rates[1])
        self.conv3 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[2],
                               padding=dilation_rates[2])
        self.conv4 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[3],
                               padding=dilation_rates[3])
        self.conv5 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[4],
                               padding=dilation_rates[4])
        self.conv6 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[5],
                               padding=dilation_rates[5])
        self.conv7 = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, dilation=dilation_rates[6],
                               padding=dilation_rates[6])

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))

        x = F.relu(self.conv7(x))
        return x

