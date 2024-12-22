import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchinfo import summary

# class CNN(nn.Module):
#     def __init__(self, R_dic):
#         super(CNN, self).__init__()


#         in_dim = R_dic['in_dim']
#         self.feature_dim = R_dic['feature_dim']
#         self.num_layers = R_dic['num_layers']
#         self.y_norm = R_dic['y_norm'] if 'y_norm' in R_dic else None


#         self.encoder = nn.Conv2d(in_channels=1, out_channels=self.feature_dim, kernel_size=3, padding=1)
        
#         self.process = nn.ModuleList([])
#         for i in range(self.num_layers):
#             self.process.append(nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1))

#         self.decoder = nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1)

#         if 'y_norm' in R_dic:
#             self.y_norm = R_dic['y_norm']
#         else:
#             self.y_norm = None

#     def forward(self, x):


#         # x.shape = [b, 1, res_input,res_input]
#         x = self.encoder(x)

#         for i in range(self.num_layers):
#             x0 = x
#             x = self.process[i](x)
#             x = x0 + F.gelu(x)
#         # x.shape = [b,feature,res_out,res_out]

#         x = self.decoder(x)
#         # x.shape = [b,res_out,res_out,1]

#         if self.y_norm is not None:
#             x = torch.squeeze(x)
#             x = self.y_norm.decode(x)

#         return torch.squeeze(x)

#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)


class CNN(nn.Module):
    def __init__(self,R_dic):
        super(CNN, self).__init__()

        in_dim = R_dic['in_dim']
        feature_dim = R_dic['feature_dim']
        self.num_layers = R_dic['num_layers']
        self.y_norm = R_dic['y_norm'] if 'y_norm' in R_dic else None

        # Encoder CNN
        self.encoder_conv = nn.Conv2d(in_channels=in_dim, out_channels=feature_dim, kernel_size=7, padding=3)

        # Processor
        self.processor = nn.ModuleList([])
        for i in range(self.num_layers):
            self.processor.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=7, padding=3))

        # Decoder CNN
        self.decoder_conv = nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=7, padding=3)


    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        # Process with residual connections
        for i in range(self.num_layers):
            x = self.processor[i](x) 
            x = F.relu(x)

        # Decode
        x = self.decoder_conv(x)
        x = x.permute(0, 2, 3, 1)

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)

        return torch.squeeze(x)

