from torch import nn
import torch

from aidtep.ml.models import ModelRegistry


class VCNN(nn.Module, ModelRegistry):

    @classmethod
    def name(cls):
        return 'VCNN'

    def __init__(self, ):
        super(VCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(48, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x_final = self.final_conv(x)
        return x_final


if __name__ == '__main__':
    model = VCNN()
    # 打模模型大小和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total trainable parameters: {total_params}')

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    print(f'Total size of parameters and buffers: {total_size} bytes')
