import logging

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from aidtep.ml.criterien.L2 import L2Loss
from aidtep.ml.processor.trainer import Trainer
from aidtep.ml.models.NVT_ResNet import NVT_ResNet
from aidtep.ml.models.torch_model import PyTorchModel
from aidtep.ml.data.dataloader import create_dataloaders
from aidtep.utils.initialize import initialize


def main_process(epoch: int = 20, lr: float = 0.005, device=None):
    nvt_model = NVT_ResNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nvt_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    observation_path = "../../data/processed/IAEA/interpolation_voronoi_float16_2_mean_1_0.01_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85]_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85].npy"
    output_path = "../../data/processed/IAEA/phione.npy"
    train_ratio = 0.85
    val_ratio = 0.1
    batch_size = 64
    seed = 42
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = create_dataloaders(observation_path, output_path, train_ratio, val_ratio,
                                                               batch_size, seed)
    model = PyTorchModel(nvt_model, criterion, optimizer, scheduler, device)
    model.add_criteria("L2", L2Loss())

    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, epoch)


if __name__ == '__main__':
    initialize(log_level=logging.DEBUG)
    main_process()
    # import torch
    # print(torch.cuda.is_available())
