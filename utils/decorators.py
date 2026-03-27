# -*- coding: utf-8 -*-
"""
utils/decorators.py — Décorateurs utilitaires RoboCare
"""

import time
import functools
from utils.logger import get_logger

logger = get_logger("robocare.decorators")


def retry(max_attempts: int = 3, delay: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Décorateur de retry avec backoff exponentiel.

    Usage :
        @retry(max_attempts=3, delay=5)
        def send_notification(...): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "Échec définitif [%s] après %d tentatives : %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    wait = delay * (2 ** (attempt - 1))   # backoff exponentiel
                    logger.warning(
                        "Tentative %d/%d échouée [%s] : %s — retry dans %.1fs",
                        attempt, max_attempts, func.__name__, exc, wait,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator