from .config import settings
import sys
import uvicorn
import logging


def get_logger(name):
    logger = logging.getLogger(name)
    LOG_LEVEL = settings.log_level
    if not LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logger.critical(f"Invalid log level {LOG_LEVEL}. Setting to INFO")
        LOG_LEVEL = "INFO"

    level = getattr(logging, LOG_LEVEL)
    logger.setLevel(level)
    logger.debug(f"Setting log level to {LOG_LEVEL}")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:   %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
