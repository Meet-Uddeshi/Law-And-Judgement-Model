import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    os.makedirs("Output/logs", exist_ok=True)
    log_path = os.path.join("Output", "logs", f"{datetime.now().date()}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s: %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
