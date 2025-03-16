import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    logger = logging.getLogger("Ranking System Pipeline")
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler("ranking_system.log", maxBytes=5_000_000, backupCount=3)
    handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
