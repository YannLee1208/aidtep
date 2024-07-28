import os
import sys
import logging

from loguru import logger


def init_logger( log_level=logging.INFO):
    log_dir = "../../logs"
    logger.remove()
    logger.add(sys.stdout, format="{time} {level} {message}", level=log_level)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(f"{log_dir}/file_{{time}}.log", rotation="1 day", level=log_level)
    logger.info("Logger initialized")
