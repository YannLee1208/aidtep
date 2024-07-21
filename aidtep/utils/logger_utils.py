from loguru import logger

def init_logger():
    # Configure the logger
    logger.remove()  # Remove the default logger
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="DEBUG")
    logger.add("file_{time}.log", rotation="1 day", level="DEBUG")