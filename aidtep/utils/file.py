from pathlib import Path as pth

import numpy as np


def check_file_exist(file_path: str) -> bool:
    if not file_path:
        return False
    file_path = pth(file_path)
    if not file_path.exists():
        return False
    return True


def make_parent_dir(file_path: str):
    """
    Make parent directory of file
    :param file_path: str
    :return:
    """
    if file_path:
        file_path = pth(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("file_path is empty")


def save_ndarray(file_path: str, data: np.ndarray):
    """
    save numpy array to file
    :param file_path: file path
    :param data: numpy array
    """
    file_path = pth(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, data)
