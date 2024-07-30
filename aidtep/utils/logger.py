import os
import sys
import logging
from loguru import logger


def init_logger(log_dir: str, log_level=logging.INFO):
    logger.remove()
    logger.add(sys.stdout, format="<green>{time}</green> <level>{level}</level> <cyan>{message}</cyan>",
               level=log_level, colorize=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(f"{log_dir}/file_{{time}}.log", rotation="1 day", level=log_level, format="{time} {level} {message}")
    logger.info("Logger initialized")

