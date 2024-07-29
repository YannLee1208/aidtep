from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from loguru import logger

from aidtep.utils.file import check_file_exist


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.data_size = len(x_data)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x_data[idx]).unsqueeze(0).float()
        y = torch.from_numpy(self.y_data[idx]).unsqueeze(0).float()
        return x, y


def create_dataloaders(x_data_path: str, y_data_path: str, train_ratio: float, val_ratio: float, batch_size: int,
                       seed: int = 0) -> tuple[
    DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """
    Create torch Dataloader from ndarray, split by train_ratio and val_ratio
    :param x_data_path: str, path of model input data, end with .npy
    :param y_data_path: str, path of model output data, end with .npy
    :param train_ratio: float, ratio of training data
    :param val_ratio: float, ratio of validation data
    :param batch_size: int, batch size
    :param seed: int, random seed
    :return: tuple[DataLoader, DataLoader, DataLoader], train_loader, val_loader, test_loader
    """
    if not check_file_exist(x_data_path) or not check_file_exist(y_data_path):
        raise FileNotFoundError("Data file not found")
    if not x_data_path.endswith(".npy") or not y_data_path.endswith(".npy"):
        raise ValueError("Data file must be .npy file")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    dataset = CustomDataset(np.load(x_data_path), np.load(y_data_path))

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # set random seed
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.debug(f"Batch Size: {batch_size}")
    logger.debug(f"Train Loader len: {len(train_loader)}, Train Size: {train_size}")
    logger.debug(f"Val Loader len: {len(val_loader)}, Val Size: {val_size}")
    logger.debug(f"Test Loader len: {len(test_loader)}, Test Size: {test_size}")

    return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    observation_path = "../../data/processed/IAEA/obs_float16_2_mean_0_0_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85]_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85].npy"
    output_path = "../../data/processed/IAEA/phione.npy"
    train_ratio = 1
    val_ratio = 0.0
    batch_size = 32
    seed = 42

    train_loader, val_loader, test_loader = create_dataloaders(observation_path, output_path
                                                               , train_ratio,val_ratio, batch_size, seed)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))