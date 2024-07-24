import logging

from aidtep.utils.logger import init_logger
from aidtep.utils.config import init_config


def initialize(config_path:str,   log_level=logging.INFO):
    init_logger(log_level)
    init_config(config_path)