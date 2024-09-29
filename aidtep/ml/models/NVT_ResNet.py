from torch import nn

from aidtep.ml.models import ModelRegistry


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_sample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.down_sample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class NVT_ResNet(nn.Module, ModelRegistry):

    @classmethod
    def name(cls):
        return 'NVT-ResNet'

    def __init__(self, block=BasicBlock, layers=None):
        super(NVT_ResNet, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 256, layers[1])
        self.layer3 = self.make_layer(block, 128, layers[2])
        self.layer4 = self.make_layer(block, 64, layers[3])

        self.dconv1 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.dconv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.final_conv(x)

        return x


if __name__ == '__main__':
    model = NVT_ResNet()
    # 打模模型大小和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total trainable parameters: {total_params}')

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    print(f'Total size of parameters and buffers: {total_size} bytes')
