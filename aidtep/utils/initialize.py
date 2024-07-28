import logging

from aidtep.utils.logger import init_logger


def initialize( log_level=logging.INFO):
    init_logger(log_level)
