# -*- coding: utf-8 -*-
"""
utils/logger.py — Logging professionnel centralisé RoboCare
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

_INITIALIZED = False


def setup_logging(log_file: str = "robocare.log", level: int = logging.INFO) -> None:
    """Configure le système de logging global (à appeler une seule fois depuis main.py)."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)-28s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler fichier rotatif (max 5 Mo × 3 fichiers)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé. Appeler setup_logging() d'abord."""
    return logging.getLogger(name)