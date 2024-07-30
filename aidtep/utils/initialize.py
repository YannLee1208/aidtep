import logging
import random
import torch
import numpy as np

from aidtep.utils.logger import init_logger


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def initialize(random_seed:int, log_dir:str, log_level=logging.INFO):
    set_random_seed(random_seed)
    init_logger(log_dir, log_level)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
